#include "hydrogen_bond.h"

#include "bond_order.h"  // for find_bond_index

static __global__ void Calculate_HB_Kernel(
    int atom_numbers, const VECTOR* crd, const int* atom_type,
    const int* is_hydrogen, const REAXFF_HB_Info* hb_info,
    const REAXFF_HB_Entry* hb_entries, int atom_type_numbers, const float* bo_s,
    const float* bo_pi, const float* bo_pi2, float* d_dE_dBO_s,
    float* d_dE_dBO_pi, float* d_dE_dBO_pi2, const LTMatrix3 cell,
    const LTMatrix3 rcell, const ATOM_GROUP* nl, float* atom_energy,
    VECTOR* frc, LTMatrix3* atom_virial, float* d_energy_hb_sum,
    const int* bond_count, const int* bond_offset, const int* bond_nbr,
    const int* bond_idx_arr)
{
    SIMPLE_DEVICE_FOR(h, atom_numbers)
    {
        if (is_hydrogen[h])
        {
            int type_h = atom_type[h];
            VECTOR rh = crd[h];
            ATOM_GROUP nl_h = nl[h];

            double en_hb = 0.0;

            int bc_h = bond_count[h];
            int bo_h = bond_offset[h];
            for (int pd = 0; pd < bc_h; pd++)
            {
                int b_dh = bond_idx_arr[bo_h + pd];
                int d = bond_nbr[bo_h + pd];
                int type_d = atom_type[d];
                float bo_dh_val = bo_s[b_dh] + bo_pi[b_dh] + bo_pi2[b_dh];
                if (bo_dh_val < 0.01f) continue;

                VECTOR rd = crd[d];
                VECTOR ddh = Get_Periodic_Displacement(rd, rh, cell, rcell);
                float r_dh = norm3df(ddh.x, ddh.y, ddh.z);

                for (int pa = 0; pa < nl_h.atom_numbers; pa++)
                {
                    int a = nl_h.atom_serial[pa];
                    if (a == d) continue;
                    int type_a = atom_type[a];

                    int hb_idx = ((type_d * atom_type_numbers + type_h) *
                                      atom_type_numbers +
                                  type_a);
                    REAXFF_HB_Info info = hb_info[hb_idx];
                    if (info.entry_count == 0) continue;

                    VECTOR ra = crd[a];
                    VECTOR dah = Get_Periodic_Displacement(ra, rh, cell, rcell);
                    float r_ah = norm3df(dah.x, dah.y, dah.z);
                    if (r_ah > 7.5f) continue;

                    float cos_theta =
                        (ddh.x * dah.x + ddh.y * dah.y + ddh.z * dah.z) /
                        (r_dh * r_ah);
                    if (cos_theta > 1.0f) cos_theta = 1.0f;
                    if (cos_theta < -1.0f) cos_theta = -1.0f;
                    float theta = acosf(cos_theta);

                    float sin_half_theta = sinf(theta * 0.5f);
                    float sin_p4 = sin_half_theta * sin_half_theta *
                                   sin_half_theta * sin_half_theta;

                    for (int e = 0; e < info.entry_count; e++)
                    {
                        const REAXFF_HB_Entry* param =
                            &hb_entries[info.start_idx + e];

                        SADfloat<1> s_bo_dh(bo_dh_val, 0);
                        SADfloat<1> s_f_hb =
                            1.0f - expf(-(float)param->p_hb2 * s_bo_dh);

                        float exp_hb3 =
                            expf(-(float)param->p_hb3 *
                                 ((float)param->r0_hb / r_ah +
                                  r_ah / (float)param->r0_hb - 2.0f));

                        SADfloat<1> s_en_total =
                            (float)param->p_hb1 * s_f_hb * exp_hb3 * sin_p4;

                        atomicAdd(&d_dE_dBO_s[b_dh], s_en_total.dval[0]);
                        atomicAdd(&d_dE_dBO_pi[b_dh], s_en_total.dval[0]);
                        atomicAdd(&d_dE_dBO_pi2[b_dh], s_en_total.dval[0]);

                        float dE_dr_ah = (float)param->p_hb1 * s_f_hb.val *
                                         sin_p4 * exp_hb3 *
                                         (-(float)param->p_hb3) *
                                         (-(float)param->r0_hb / (r_ah * r_ah) +
                                          1.0f / (float)param->r0_hb);

                        float f_ah = -dE_dr_ah;
                        VECTOR f_a_rad = {f_ah * dah.x / r_ah,
                                          f_ah * dah.y / r_ah,
                                          f_ah * dah.z / r_ah};
                        atomicAdd(&frc[a].x, f_a_rad.x);
                        atomicAdd(&frc[a].y, f_a_rad.y);
                        atomicAdd(&frc[a].z, f_a_rad.z);
                        atomicAdd(&frc[h].x, -f_a_rad.x);
                        atomicAdd(&frc[h].y, -f_a_rad.y);
                        atomicAdd(&frc[h].z, -f_a_rad.z);

                        float dE_dsinp4 =
                            (float)param->p_hb1 * s_f_hb.val * exp_hb3;
                        float dsinp4_dtheta = 4.0f * sin_half_theta *
                                              sin_half_theta * sin_half_theta *
                                              cosf(theta * 0.5f) * 0.5f;
                        float dE_dtheta = dE_dsinp4 * dsinp4_dtheta;

                        float sin_theta = sinf(theta);
                        if (sin_theta < 1e-5f) sin_theta = 1e-5f;
                        float dE_dcos = dE_dtheta / (-sin_theta);

                        float inv_rdh = 1.0f / r_dh;
                        float inv_rah = 1.0f / r_ah;

                        VECTOR fd, fa, fh;
                        fd.x =
                            dE_dcos * (dah.x * inv_rdh * inv_rah -
                                       ddh.x * cos_theta * inv_rdh * inv_rdh);
                        fd.y =
                            dE_dcos * (dah.y * inv_rdh * inv_rah -
                                       ddh.y * cos_theta * inv_rdh * inv_rdh);
                        fd.z =
                            dE_dcos * (dah.z * inv_rdh * inv_rah -
                                       ddh.z * cos_theta * inv_rdh * inv_rdh);

                        fa.x =
                            dE_dcos * (ddh.x * inv_rdh * inv_rah -
                                       dah.x * cos_theta * inv_rah * inv_rah);
                        fa.y =
                            dE_dcos * (ddh.y * inv_rdh * inv_rah -
                                       dah.y * cos_theta * inv_rah * inv_rah);
                        fa.z =
                            dE_dcos * (ddh.z * inv_rdh * inv_rah -
                                       dah.z * cos_theta * inv_rah * inv_rah);

                        fh.x = -(fd.x + fa.x);
                        fh.y = -(fd.y + fa.y);
                        fh.z = -(fd.z + fa.z);

                        atomicAdd(&frc[d].x, -fd.x);
                        atomicAdd(&frc[d].y, -fd.y);
                        atomicAdd(&frc[d].z, -fd.z);
                        atomicAdd(&frc[a].x, -fa.x);
                        atomicAdd(&frc[a].y, -fa.y);
                        atomicAdd(&frc[a].z, -fa.z);
                        atomicAdd(&frc[h].x, -fh.x);
                        atomicAdd(&frc[h].y, -fh.y);
                        atomicAdd(&frc[h].z, -fh.z);
                        if (atom_virial)
                        {
                            VECTOR f_d = {-fd.x, -fd.y, -fd.z};
                            VECTOR f_a = {-fa.x, -fa.y, -fa.z};
                            LTMatrix3 v =
                                Get_Virial_From_Force_Dis(f_a_rad, dah) +
                                Get_Virial_From_Force_Dis(f_d, ddh) +
                                Get_Virial_From_Force_Dis(f_a, dah);
                            atomicAdd(atom_virial + h, v);
                        }

                        en_hb += s_en_total.val;
                    }
                }
            }
            atomicAdd(d_energy_hb_sum, (float)en_hb);
            if (atom_energy) atomicAdd(&atom_energy[h], (float)en_hb);
        }
    }
}

void REAXFF_HYDROGEN_BOND::Initial(CONTROLLER* controller, int atom_numbers,
                                   const char* module_name)
{
    if (module_name == NULL) module_name = "REAXFF";
    this->atom_numbers = atom_numbers;
    if (!controller->Command_Exist(module_name, "in_file")) return;

    const char* parameter_in_file = controller->Command(module_name, "in_file");
    const char* type_in_file = controller->Command(module_name, "type_in_file");
    if (parameter_in_file == NULL || type_in_file == NULL)
    {
        controller->printf(
            "REAXFF_HYDROGEN_BOND IS NOT INITIALIZED (missing input "
            "files)\n\n");
        return;
    }
    controller->Step_Print_Initial("REAXFF_HB", "%14.7e");

    FILE* fp;
    Open_File_Safely(&fp, parameter_in_file, "r");
    char line[1024];
    auto throw_bad_format = [&](const char* file_name, const char* reason)
    {
        char error_msg[1024];
        sprintf(error_msg, "Reason:\n\t%s in file %s\n", reason, file_name);
        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                       "REAXFF_HYDROGEN_BOND::Initial",
                                       error_msg);
    };
    auto read_line_or_throw =
        [&](FILE* file, const char* file_name, const char* stage)
    {
        if (fgets(line, 1024, file) == NULL)
        {
            char reason[512];
            sprintf(reason, "failed to read %s", stage);
            throw_bad_format(file_name, reason);
        }
    };

    read_line_or_throw(fp, parameter_in_file, "parameter header line 1");
    read_line_or_throw(fp, parameter_in_file, "general parameter count line");
    int n_gp = 0;
    if (sscanf(line, "%d", &n_gp) != 1 || n_gp < 0)
    {
        throw_bad_format(parameter_in_file,
                         "failed to parse number of general parameters");
    }
    for (int i = 0; i < n_gp; i++)
    {
        read_line_or_throw(fp, parameter_in_file, "general parameter block");
    }

    read_line_or_throw(fp, parameter_in_file, "atom type count line");
    int n_atom_types = 0;
    if (sscanf(line, "%d", &n_atom_types) != 1 || n_atom_types <= 0)
    {
        throw_bad_format(parameter_in_file,
                         "failed to parse number of atom types");
    }
    this->atom_type_numbers = n_atom_types;
    read_line_or_throw(fp, parameter_in_file, "atom type header line 1");
    read_line_or_throw(fp, parameter_in_file, "atom type header line 2");
    read_line_or_throw(fp, parameter_in_file, "atom type header line 3");

    std::map<std::string, int> type_map;
    for (int i = 0; i < n_atom_types; i++)
    {
        read_line_or_throw(fp, parameter_in_file, "atom type block line 1");
        char name[16];
        if (sscanf(line, "%15s", name) != 1)
        {
            char reason[512];
            sprintf(reason,
                    "failed to parse atom type block line 1 at index %d",
                    i + 1);
            throw_bad_format(parameter_in_file, reason);
        }
        type_map[name] = i;
        read_line_or_throw(fp, parameter_in_file, "atom type block line 2");
        read_line_or_throw(fp, parameter_in_file, "atom type block line 3");
        read_line_or_throw(fp, parameter_in_file, "atom type block line 4");
    }

    read_line_or_throw(fp, parameter_in_file, "bond parameter count line");
    int n_bond = 0;
    if (sscanf(line, "%d", &n_bond) != 1 || n_bond < 0)
    {
        throw_bad_format(parameter_in_file,
                         "failed to parse number of bond parameters");
    }
    read_line_or_throw(fp, parameter_in_file, "bond parameter header line");
    for (int i = 0; i < n_bond * 2; i++)
    {
        read_line_or_throw(fp, parameter_in_file, "bond parameter block");
    }

    read_line_or_throw(fp, parameter_in_file, "off-diagonal count line");
    int n_off = 0;
    if (sscanf(line, "%d", &n_off) != 1 || n_off < 0)
    {
        throw_bad_format(parameter_in_file,
                         "failed to parse number of off-diagonal parameters");
    }
    for (int i = 0; i < n_off; i++)
    {
        read_line_or_throw(fp, parameter_in_file,
                           "off-diagonal parameter entry");
    }

    read_line_or_throw(fp, parameter_in_file, "angle parameter count line");
    int n_thb = 0;
    if (sscanf(line, "%d", &n_thb) != 1 || n_thb < 0)
    {
        throw_bad_format(parameter_in_file,
                         "failed to parse number of angle parameters");
    }
    for (int i = 0; i < n_thb; i++)
    {
        read_line_or_throw(fp, parameter_in_file, "angle parameter entry");
    }

    read_line_or_throw(fp, parameter_in_file, "torsion parameter count line");
    int n_tor = 0;
    if (sscanf(line, "%d", &n_tor) != 1 || n_tor < 0)
    {
        throw_bad_format(parameter_in_file,
                         "failed to parse number of torsion parameters");
    }
    for (int i = 0; i < n_tor; i++)
    {
        read_line_or_throw(fp, parameter_in_file, "torsion parameter entry");
    }

    read_line_or_throw(fp, parameter_in_file, "hydrogen bond count line");
    int n_hb = 0;
    if (sscanf(line, "%d", &n_hb) != 1 || n_hb < 0)
    {
        throw_bad_format(parameter_in_file,
                         "failed to parse number of hydrogen bond parameters");
    }
    std::vector<REAXFF_HB_Entry> all_entries;
    std::map<int, std::vector<int>> triplet_to_entries;
    for (int i = 0; i < n_hb; i++)
    {
        read_line_or_throw(fp, parameter_in_file,
                           "hydrogen bond parameter entry");
        int t1, t2, t3;
        float r0, p1, p2, p3;
        int read_cnt = sscanf(line, "%d %d %d %f %f %f %f", &t1, &t2, &t3, &r0,
                              &p1, &p2, &p3);
        if (read_cnt != 7)
        {
            char reason[512];
            sprintf(reason,
                    "failed to parse hydrogen bond parameter entry at index %d",
                    i + 1);
            throw_bad_format(parameter_in_file, reason);
        }
        int idx1 = t1 - 1;
        int idx2 = t2 - 1;
        int idx3 = t3 - 1;
        if (idx1 < 0 || idx1 >= n_atom_types || idx2 < 0 ||
            idx2 >= n_atom_types || idx3 < 0 || idx3 >= n_atom_types)
        {
            char reason[512];
            sprintf(reason,
                    "hydrogen bond atom type index out of range at index %d",
                    i + 1);
            throw_bad_format(parameter_in_file, reason);
        }
        REAXFF_HB_Entry entry = {r0, p1, p2, p3};
        int entry_idx = all_entries.size();
        all_entries.push_back(entry);
        int tri_idx = (idx1 * n_atom_types + idx2) * n_atom_types + idx3;
        triplet_to_entries[tri_idx].push_back(entry_idx);
    }
    fclose(fp);

    Malloc_Safely((void**)&h_hb_info, sizeof(REAXFF_HB_Info) * n_atom_types *
                                          n_atom_types * n_atom_types);
    memset(h_hb_info, 0,
           sizeof(REAXFF_HB_Info) * n_atom_types * n_atom_types * n_atom_types);
    std::vector<REAXFF_HB_Entry> sorted_entries;
    for (int i = 0; i < n_atom_types * n_atom_types * n_atom_types; i++)
    {
        if (triplet_to_entries.count(i))
        {
            h_hb_info[i].start_idx = sorted_entries.size();
            h_hb_info[i].entry_count = triplet_to_entries[i].size();
            for (int idx : triplet_to_entries[i])
                sorted_entries.push_back(all_entries[idx]);
        }
    }
    Malloc_Safely((void**)&h_hb_entries,
                  sizeof(REAXFF_HB_Entry) * sorted_entries.size());
    for (size_t i = 0; i < sorted_entries.size(); i++)
        h_hb_entries[i] = sorted_entries[i];

    Open_File_Safely(&fp, type_in_file, "r");
    int check_atom_numbers = 0;
    read_line_or_throw(fp, type_in_file, "atom number line");
    if (sscanf(line, "%d", &check_atom_numbers) != 1)
    {
        throw_bad_format(type_in_file, "failed to parse atom numbers");
    }
    if (check_atom_numbers != atom_numbers)
    {
        char reason[512];
        sprintf(reason, "atom numbers (%d) does not match system (%d)",
                check_atom_numbers, atom_numbers);
        throw_bad_format(type_in_file, reason);
    }
    Malloc_Safely((void**)&h_atom_type, sizeof(int) * atom_numbers);
    Malloc_Safely((void**)&h_is_hydrogen, sizeof(int) * atom_numbers);
    for (int i = 0; i < atom_numbers; i++)
    {
        read_line_or_throw(fp, type_in_file, "atom type entry line");
        char name[16];
        if (sscanf(line, "%15s", name) != 1)
        {
            char reason[512];
            sprintf(reason, "failed to parse atom type at index %d", i + 1);
            throw_bad_format(type_in_file, reason);
        }
        auto iter = type_map.find(std::string(name));
        if (iter == type_map.end())
        {
            char reason[512];
            sprintf(reason, "atom type %s not found in parameter file %s", name,
                    parameter_in_file);
            throw_bad_format(type_in_file, reason);
        }
        h_atom_type[i] = iter->second;
        h_is_hydrogen[i] = (std::string(name) == "H");
    }
    fclose(fp);

    Device_Malloc_And_Copy_Safely((void**)&d_atom_type, h_atom_type,
                                  sizeof(int) * atom_numbers);
    Device_Malloc_And_Copy_Safely((void**)&d_is_hydrogen, h_is_hydrogen,
                                  sizeof(int) * atom_numbers);
    Device_Malloc_And_Copy_Safely(
        (void**)&d_hb_info, h_hb_info,
        sizeof(REAXFF_HB_Info) * n_atom_types * n_atom_types * n_atom_types);
    Device_Malloc_And_Copy_Safely(
        (void**)&d_hb_entries, h_hb_entries,
        sizeof(REAXFF_HB_Entry) * sorted_entries.size());
    Device_Malloc_Safely((void**)&d_energy_hb_sum, sizeof(float));
    is_initialized = 1;
}

void REAXFF_HYDROGEN_BOND::Calculate_HB_Energy_And_Force(
    int atom_numbers, const VECTOR* crd, VECTOR* frc, const LTMatrix3 cell,
    const LTMatrix3 rcell, const ATOM_GROUP* nl, REAXFF_BOND_ORDER* bo_module,
    const int need_atom_energy, float* atom_energy, const int need_virial,
    LTMatrix3* atom_virial)
{
    if (!is_initialized) return;
    dim3 blockSize(32);
    dim3 gridSize((atom_numbers + blockSize.x - 1) / blockSize.x);
    deviceMemset(d_energy_hb_sum, 0, sizeof(float));

    Launch_Device_Kernel(
        Calculate_HB_Kernel, gridSize, blockSize, 0, NULL, atom_numbers, crd,
        d_atom_type, d_is_hydrogen, d_hb_info, d_hb_entries, atom_type_numbers,
        bo_module->d_corrected_bo_s, bo_module->d_corrected_bo_pi,
        bo_module->d_corrected_bo_pi2, d_dE_dBO_s, d_dE_dBO_pi, d_dE_dBO_pi2,
        cell, rcell, nl, atom_energy, frc, need_virial ? atom_virial : NULL,
        d_energy_hb_sum, bo_module->d_bond_count, bo_module->d_bond_offset,
        bo_module->d_bond_nbr, bo_module->d_bond_idx);
}

void REAXFF_HYDROGEN_BOND::Step_Print(CONTROLLER* controller)
{
    if (!is_initialized) return;
    deviceMemcpy(&h_energy_hb, d_energy_hb_sum, sizeof(float),
                 deviceMemcpyDeviceToHost);
    controller->Step_Print("REAXFF_HB", h_energy_hb, true);
}

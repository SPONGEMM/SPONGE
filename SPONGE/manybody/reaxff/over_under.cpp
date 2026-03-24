#include "over_under.h"

static __global__ void Calculate_Delta_Kernel(
    int atom_numbers, const int* atom_type, const float* total_corrected_bo,
    const float* valency, const float* valency_e, const float* valency_boc,
    const float* valency_val, const float* mass, float p_lp1, float* d_Delta,
    float* d_Delta_boc, float* d_Delta_val, float* d_Delta_lp, float* d_nlp,
    float* d_vlpex, float* d_Delta_lp_temp, float* d_dDelta_lp)
{
    SIMPLE_DEVICE_FOR(i, atom_numbers)
    {
        int type = atom_type[i];
        float total_bo = total_corrected_bo[i];

        float val = valency[type];
        float val_e = valency_e[type];
        float val_boc = valency_boc[type];
        float val_val = valency_val[type];
        float m = mass[type];

        d_Delta[i] = total_bo - val;
        d_Delta_boc[i] = total_bo - val_boc;
        d_Delta_val[i] = total_bo - val_val;

        float Delta_e = total_bo - val_e;

        float vlpex = Delta_e - 2.0f * (int)(Delta_e / 2.0f);
        d_vlpex[i] = vlpex;

        float explp1 = expf(-p_lp1 * (2.0f + vlpex) * (2.0f + vlpex));
        float nlp = explp1 - (int)(Delta_e / 2.0f);
        d_nlp[i] = nlp;

        float nlp_opt = 0.5f * (val_e - val);
        d_Delta_lp[i] = nlp_opt - nlp;

        float Clp = 2.0f * p_lp1 * explp1 * (2.0f + vlpex);
        d_dDelta_lp[i] = Clp;

        if (m > 21.0f)
        {
            d_Delta_lp_temp[i] = 0.0f;
        }
        else
        {
            d_Delta_lp_temp[i] = d_Delta_lp[i];
        }
    }
}

static __global__ void Calculate_Energy_Force_Prep_Kernel(
    int atom_numbers, const int* atom_type, const float* mass,
    const float* Delta, const float* Delta_lp, const float* Delta_lp_temp,
    const float* dDelta_lp, const float* bo_s, const float* bo_pi,
    const float* bo_pi2, const float* p_ovun1, const float* De_s,
    int atom_type_numbers, const float* p_lp2, const float* valency,
    float p_ovun3, float p_ovun4, float p_ovun6, float p_ovun7, float p_ovun8,
    const float* p_ovun2, const float* p_ovun5, float* d_dE_dBO_s,
    float* d_dE_dBO_pi, float* d_dE_dBO_pi2, float* CdDelta, float* atom_energy,
    float* d_energy_ovun_sum, float* d_energy_elp_sum, const int* bond_count,
    const int* bond_offset, const int* bond_nbr, const int* bond_idx_arr)
{
    SIMPLE_DEVICE_FOR(i, atom_numbers)
    {
        int type_i = atom_type[i];
        float m = mass[type_i];
        float dfvl = (m > 21.0f) ? 0.0f : 1.0f;
        float val = valency[type_i];

        float s_ovun1 = 0.0f;
        float s_ovun2 = 0.0f;

        int start = bond_offset[i];
        int end = start + bond_count[i];
        for (int kk = start; kk < end; kk++)
        {
            int j = bond_nbr[kk];
            int b = bond_idx_arr[kk];
            float b_s = bo_s[b];
            float b_pi = bo_pi[b];
            float b_pi2 = bo_pi2[b];
            float bo_total = b_s + b_pi + b_pi2;

            if (bo_total < 1e-10f) continue;

            int type_j = atom_type[j];
            int pair_idx = type_i * atom_type_numbers + type_j;

            s_ovun1 += p_ovun1[pair_idx] * De_s[pair_idx] * bo_total;

            float dfvl_j = (mass[type_j] > 21.0f) ? 0.0f : 1.0f;
            s_ovun2 += (Delta[j] - dfvl_j * Delta_lp_temp[j]) * (b_pi + b_pi2);
        }

        float en_ovun = 0.0f;
        float en_lp = 0.0f;
        float cdd = 0.0f;

        float delta_lp_i = Delta_lp[i];
        float p_lp2_val = p_lp2[type_i];
        float expvd2 = expf(-75.0f * delta_lp_i);
        float inv_expvd2 = 1.0f / (1.0f + expvd2);

        en_lp = p_lp2_val * delta_lp_i * inv_expvd2;
        float dElp = p_lp2_val * inv_expvd2 + 75.0f * p_lp2_val * delta_lp_i *
                                                  expvd2 * inv_expvd2 *
                                                  inv_expvd2;
        cdd += dElp * dDelta_lp[i];

        float exp_ovun1 = p_ovun3 * expf(p_ovun4 * s_ovun2);
        float inv_exp_ovun1 = 1.0f / (1.0f + exp_ovun1);
        float Delta_lpcorr =
            Delta[i] - (dfvl * Delta_lp_temp[i]) * inv_exp_ovun1;

        float p_ovun2_val = p_ovun2[type_i];
        float exp_ovun2 = expf(p_ovun2_val * Delta_lpcorr);
        float inv_exp_ovun2 = 1.0f / (1.0f + exp_ovun2);

        float DlpVi = 1.0f / (Delta_lpcorr + val + 1e-8f);
        float CEover1 = Delta_lpcorr * DlpVi * inv_exp_ovun2;

        float e_ov = s_ovun1 * CEover1;
        en_ovun += e_ov;

        float CEover2 =
            s_ovun1 * DlpVi * inv_exp_ovun2 *
            (1.0f -
             Delta_lpcorr * (DlpVi + p_ovun2_val * exp_ovun2 * inv_exp_ovun2));
        float CEover3 = CEover2 * (1.0f - dfvl * dDelta_lp[i] * inv_exp_ovun1);
        float CEover4 = CEover2 * (dfvl * Delta_lp_temp[i]) * p_ovun4 *
                        exp_ovun1 * inv_exp_ovun1 * inv_exp_ovun1;

        cdd += CEover3;

        float p_ovun5_val = p_ovun5[type_i];
        float exp_ovun2n = 1.0f / exp_ovun2;
        float exp_ovun6 = expf(p_ovun6 * Delta_lpcorr);
        float exp_ovun8 = p_ovun7 * expf(p_ovun8 * s_ovun2);
        float inv_exp_ovun2n = 1.0f / (1.0f + exp_ovun2n);
        float inv_exp_ovun8 = 1.0f / (1.0f + exp_ovun8);

        float e_un =
            -p_ovun5_val * (1.0f - exp_ovun6) * inv_exp_ovun2n * inv_exp_ovun8;
        en_ovun += e_un;

        float CEunder1 = inv_exp_ovun2n *
                         (p_ovun5_val * p_ovun6 * exp_ovun6 * inv_exp_ovun8 +
                          p_ovun2_val * e_un * exp_ovun2n);
        float CEunder2 = -e_un * p_ovun8 * exp_ovun8 * inv_exp_ovun8;
        float CEunder3 =
            CEunder1 * (1.0f - dfvl * dDelta_lp[i] * inv_exp_ovun1);

        cdd += CEunder3;

        atomicAdd(&CdDelta[i], cdd);

        float CEunder4 = CEunder1 * (dfvl * Delta_lp_temp[i]) * p_ovun4 *
                             exp_ovun1 * inv_exp_ovun1 * inv_exp_ovun1 +
                         CEunder2;
        float CE_sum_4 = CEover4 + CEunder4;

        for (int kk = start; kk < end; kk++)
        {
            int j = bond_nbr[kk];
            int b = bond_idx_arr[kk];
            float b_s = bo_s[b];
            float b_pi = bo_pi[b];
            float b_pi2 = bo_pi2[b];
            float bo_total = b_s + b_pi + b_pi2;

            if (bo_total < 1e-10f) continue;

            int type_j = atom_type[j];
            int pair_idx = type_i * atom_type_numbers + type_j;

            float de_dbo_s = CEover1 * p_ovun1[pair_idx] * De_s[pair_idx];
            float de_dbo_pi =
                de_dbo_s + CE_sum_4 * (Delta[j] - dfvl * Delta_lp_temp[j]);
            float de_dbo_pi2 = de_dbo_pi;

            atomicAdd(&d_dE_dBO_s[b], de_dbo_s);
            atomicAdd(&d_dE_dBO_pi[b], de_dbo_pi);
            atomicAdd(&d_dE_dBO_pi2[b], de_dbo_pi2);

            float dfvl_j = (mass[type_j] > 21.0f) ? 0.0f : 1.0f;
            float term =
                CE_sum_4 * (1.0f - dfvl_j * dDelta_lp[j]) * (b_pi + b_pi2);
            atomicAdd(&CdDelta[j], term);
        }

        if (atom_energy) atomicAdd(&atom_energy[i], en_ovun + en_lp);
        atomicAdd(d_energy_ovun_sum, en_ovun);
        atomicAdd(d_energy_elp_sum, en_lp);
    }
}

void REAXFF_OVER_UNDER::Initial(CONTROLLER* controller, int atom_numbers,
                                const char* module_name)
{
    if (module_name == NULL) module_name = "REAXFF";
    this->atom_numbers = atom_numbers;

    if (!controller->Command_Exist(module_name, "in_file")) return;

    controller->printf("START INITIALIZING REAXFF OVER/UNDER COORD\n");

    const char* parameter_in_file = controller->Command(module_name, "in_file");
    const char* type_in_file = controller->Command(module_name, "type_in_file");
    if (parameter_in_file == NULL || type_in_file == NULL)
    {
        controller->printf(
            "REAXFF_OVER_UNDER IS NOT INITIALIZED (missing input files)\n\n");
        return;
    }
    controller->Step_Print_Initial("REAXFF_ELP", "%14.7e");
    controller->Step_Print_Initial("REAXFF_OVUN", "%14.7e");

    FILE* fp;
    Open_File_Safely(&fp, parameter_in_file, "r");
    char line[1024];
    auto throw_bad_format = [&](const char* file_name, const char* reason)
    {
        char error_msg[1024];
        sprintf(error_msg, "Reason:\n\t%s in file %s\n", reason, file_name);
        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                       "REAXFF_OVER_UNDER::Initial", error_msg);
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
    if (sscanf(line, "%d", &n_gp) != 1 || n_gp <= 0)
    {
        throw_bad_format(parameter_in_file,
                         "failed to parse number of general parameters");
    }
    if (n_gp <= 32)
    {
        throw_bad_format(parameter_in_file,
                         "general parameter count is too small for over/under");
    }
    float* gp = new float[n_gp];
    for (int i = 0; i < n_gp; i++)
    {
        read_line_or_throw(fp, parameter_in_file, "general parameter block");
        if (sscanf(line, "%f", &gp[i]) != 1)
        {
            char reason[512];
            sprintf(reason, "failed to parse general parameter at index %d",
                    i + 1);
            throw_bad_format(parameter_in_file, reason);
        }
    }
    p_lp1 = gp[15];
    p_lp3 = gp[5];
    p_ovun3 = gp[32];
    p_ovun4 = gp[31];
    p_ovun6 = gp[6];
    p_ovun7 = gp[8];
    p_ovun8 = gp[9];
    delete[] gp;

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

    Malloc_Safely((void**)&h_valency, sizeof(float) * n_atom_types);
    Malloc_Safely((void**)&h_valency_e, sizeof(float) * n_atom_types);
    Malloc_Safely((void**)&h_valency_boc, sizeof(float) * n_atom_types);
    Malloc_Safely((void**)&h_valency_val, sizeof(float) * n_atom_types);
    Malloc_Safely((void**)&h_mass, sizeof(float) * n_atom_types);
    Malloc_Safely((void**)&h_p_lp2, sizeof(float) * n_atom_types);
    Malloc_Safely((void**)&h_p_ovun2, sizeof(float) * n_atom_types);
    Malloc_Safely((void**)&h_p_ovun5, sizeof(float) * n_atom_types);

    std::map<std::string, int> type_map;
    for (int i = 0; i < n_atom_types; i++)
    {
        read_line_or_throw(fp, parameter_in_file, "atom type block line 1");
        char name[16];
        if (sscanf(line, "%15s %*f %f %f %*f %*f %*f %*f %f", name,
                   &h_valency[i], &h_mass[i], &h_valency_e[i]) != 4)
        {
            char reason[512];
            sprintf(reason,
                    "failed to parse atom type block line 1 at index %d",
                    i + 1);
            throw_bad_format(parameter_in_file, reason);
        }
        type_map[name] = i;

        read_line_or_throw(fp, parameter_in_file, "atom type block line 2");
        if (sscanf(line, "%*f %*f %f %f %*f %*f %*f %*f", &h_valency_boc[i],
                   &h_p_ovun5[i]) != 2)
        {
            char reason[512];
            sprintf(reason,
                    "failed to parse atom type block line 2 at index %d",
                    i + 1);
            throw_bad_format(parameter_in_file, reason);
        }

        read_line_or_throw(fp, parameter_in_file, "atom type block line 3");
        if (sscanf(line, "%*f %f", &h_p_lp2[i]) != 1)
        {
            char reason[512];
            sprintf(reason,
                    "failed to parse atom type block line 3 at index %d",
                    i + 1);
            throw_bad_format(parameter_in_file, reason);
        }

        read_line_or_throw(fp, parameter_in_file, "atom type block line 4");
        if (sscanf(line, "%f %*f %*f %f %*f", &h_p_ovun2[i],
                   &h_valency_val[i]) != 2)
        {
            char reason[512];
            sprintf(reason,
                    "failed to parse atom type block line 4 at index %d",
                    i + 1);
            throw_bad_format(parameter_in_file, reason);
        }
    }

    read_line_or_throw(fp, parameter_in_file, "bond parameter count line");
    int n_bond_params = 0;
    if (sscanf(line, "%d", &n_bond_params) != 1 || n_bond_params < 0)
    {
        throw_bad_format(parameter_in_file,
                         "failed to parse number of bond parameters");
    }
    Malloc_Safely((void**)&h_p_ovun1,
                  sizeof(float) * n_atom_types * n_atom_types);
    Malloc_Safely((void**)&h_De_s, sizeof(float) * n_atom_types * n_atom_types);
    memset(h_p_ovun1, 0, sizeof(float) * n_atom_types * n_atom_types);
    memset(h_De_s, 0, sizeof(float) * n_atom_types * n_atom_types);
    read_line_or_throw(fp, parameter_in_file, "bond parameter header line");
    for (int i = 0; i < n_bond_params; i++)
    {
        read_line_or_throw(fp, parameter_in_file,
                           "bond parameter block line 1");
        int t1, t2;
        float de_s, povun1;
        if (sscanf(line, "%d %d %f %*f %*f %*f %*f %*f %*f %f", &t1, &t2, &de_s,
                   &povun1) != 4)
        {
            char reason[512];
            sprintf(reason,
                    "failed to parse bond parameter block line 1 at index %d",
                    i + 1);
            throw_bad_format(parameter_in_file, reason);
        }
        int idx1 = t1 - 1;
        int idx2 = t2 - 1;
        if (idx1 >= 0 && idx1 < n_atom_types && idx2 >= 0 &&
            idx2 < n_atom_types)
        {
            h_De_s[idx1 * n_atom_types + idx2] =
                h_De_s[idx2 * n_atom_types + idx1] = de_s;
            h_p_ovun1[idx1 * n_atom_types + idx2] =
                h_p_ovun1[idx2 * n_atom_types + idx1] = povun1;
        }
        else
        {
            char reason[512];
            sprintf(reason,
                    "bond type index out of range at bond parameter index %d",
                    i + 1);
            throw_bad_format(parameter_in_file, reason);
        }
        read_line_or_throw(fp, parameter_in_file,
                           "bond parameter block line 2");
    }
    fclose(fp);

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
    }
    fclose(fp);

    Device_Malloc_And_Copy_Safely((void**)&d_atom_type, h_atom_type,
                                  sizeof(int) * atom_numbers);
    Device_Malloc_And_Copy_Safely((void**)&d_p_lp2, h_p_lp2,
                                  sizeof(float) * n_atom_types);
    Device_Malloc_And_Copy_Safely((void**)&d_p_ovun2, h_p_ovun2,
                                  sizeof(float) * n_atom_types);
    Device_Malloc_And_Copy_Safely((void**)&d_p_ovun5, h_p_ovun5,
                                  sizeof(float) * n_atom_types);
    Device_Malloc_And_Copy_Safely((void**)&d_valency, h_valency,
                                  sizeof(float) * n_atom_types);
    Device_Malloc_And_Copy_Safely((void**)&d_valency_e, h_valency_e,
                                  sizeof(float) * n_atom_types);
    Device_Malloc_And_Copy_Safely((void**)&d_valency_boc, h_valency_boc,
                                  sizeof(float) * n_atom_types);
    Device_Malloc_And_Copy_Safely((void**)&d_valency_val, h_valency_val,
                                  sizeof(float) * n_atom_types);
    Device_Malloc_And_Copy_Safely((void**)&d_mass, h_mass,
                                  sizeof(float) * n_atom_types);
    Device_Malloc_And_Copy_Safely((void**)&d_p_ovun1, h_p_ovun1,
                                  sizeof(float) * n_atom_types * n_atom_types);
    Device_Malloc_And_Copy_Safely((void**)&d_De_s, h_De_s,
                                  sizeof(float) * n_atom_types * n_atom_types);

    Device_Malloc_Safely((void**)&d_Delta, sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&d_Delta_boc, sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&d_Delta_val, sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&d_Delta_lp, sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&d_nlp, sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&d_vlpex, sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&d_Delta_lp_temp,
                         sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&d_dDelta_lp, sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&d_CdDelta, sizeof(float) * atom_numbers);

    Device_Malloc_Safely((void**)&d_energy_elp_sum, sizeof(float));
    Device_Malloc_Safely((void**)&d_energy_ovun_sum, sizeof(float));
    Device_Malloc_Safely((void**)&d_energy_atom, sizeof(float) * atom_numbers);

    is_initialized = 1;
    controller->printf("END INITIALIZING REAXFF OVER/UNDER COORD\n\n");
}

void REAXFF_OVER_UNDER::Calculate_Over_Under_Energy_And_Force(
    int atom_numbers, const VECTOR* crd, VECTOR* frc, const LTMatrix3 cell,
    const LTMatrix3 rcell, REAXFF_BOND_ORDER* bo_module,
    const int need_atom_energy, float* atom_energy, const int need_virial,
    LTMatrix3* atom_virial)
{
    if (!is_initialized) return;

    dim3 blockSize(128);
    dim3 gridSize((atom_numbers + blockSize.x - 1) / blockSize.x);

    Launch_Device_Kernel(Calculate_Delta_Kernel, gridSize, blockSize, 0, NULL,
                         atom_numbers, d_atom_type,
                         bo_module->d_total_corrected_bond_order, d_valency,
                         d_valency_e, d_valency_boc, d_valency_val, d_mass,
                         p_lp1, d_Delta, d_Delta_boc, d_Delta_val, d_Delta_lp,
                         d_nlp, d_vlpex, d_Delta_lp_temp, d_dDelta_lp);

    deviceMemset(d_energy_elp_sum, 0, sizeof(float));
    deviceMemset(d_energy_ovun_sum, 0, sizeof(float));

    Launch_Device_Kernel(
        Calculate_Energy_Force_Prep_Kernel, gridSize, blockSize, 0, NULL,
        atom_numbers, d_atom_type, d_mass, d_Delta, d_Delta_lp, d_Delta_lp_temp,
        d_dDelta_lp, bo_module->d_corrected_bo_s, bo_module->d_corrected_bo_pi,
        bo_module->d_corrected_bo_pi2, d_p_ovun1, d_De_s, atom_type_numbers,
        d_p_lp2, d_valency, p_ovun3, p_ovun4, p_ovun6, p_ovun7, p_ovun8,
        d_p_ovun2, d_p_ovun5, d_dE_dBO_s, d_dE_dBO_pi, d_dE_dBO_pi2, d_CdDelta,
        need_atom_energy ? atom_energy : NULL, d_energy_ovun_sum,
        d_energy_elp_sum, bo_module->d_bond_count, bo_module->d_bond_offset,
        bo_module->d_bond_nbr, bo_module->d_bond_idx);
}

void REAXFF_OVER_UNDER::Step_Print(CONTROLLER* controller)
{
    if (!is_initialized) return;
    deviceMemcpy(&h_energy_ovun, d_energy_ovun_sum, sizeof(float),
                 deviceMemcpyDeviceToHost);
    controller->Step_Print("REAXFF_OVUN", h_energy_ovun, true);
}

void REAXFF_OVER_UNDER::Step_Print_ELP(CONTROLLER* controller)
{
    if (!is_initialized) return;
    deviceMemcpy(&h_energy_lp, d_energy_elp_sum, sizeof(float),
                 deviceMemcpyDeviceToHost);
    controller->Step_Print("REAXFF_ELP", h_energy_lp, true);
}

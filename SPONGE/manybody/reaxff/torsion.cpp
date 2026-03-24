#include "torsion.h"

#include "bond_order.h"  // for find_bond_index

static __global__ void Calculate_Torsion_Kernel(
    int atom_numbers, const VECTOR* crd, const int* atom_type,
    const float p_tor2, const float p_tor3, const float p_tor4,
    const float p_cot2, const float thb_cut, const float* Delta_boc,
    const REAXFF_TORSION_Info* torsion_info,
    const REAXFF_TORSION_Entry* torsion_entries, int atom_type_numbers,
    const float* bo_s, const float* bo_pi, const float* bo_pi2,
    float* d_dE_dBO_s, float* d_dE_dBO_pi, float* d_dE_dBO_pi2, float* CdDelta,
    const LTMatrix3 cell, const LTMatrix3 rcell, float* atom_energy,
    VECTOR* frc, LTMatrix3* atom_virial, float* d_energy_tor_sum,
    float* d_energy_cot_sum, const int* bond_count, const int* bond_offset,
    const int* bond_nbr, const int* bond_idx_arr)
{
    SIMPLE_DEVICE_FOR(j, atom_numbers)
    {
        int type_j = atom_type[j];
        if (type_j >= 0)
        {
            VECTOR rj = crd[j];
            float delta_j_val = Delta_boc[j];

            double en_tor = 0.0;
            double en_cot = 0.0;

            int bc_j = bond_count[j];
            int bo_j = bond_offset[j];

            for (int pk = 0; pk < bc_j; pk++)
            {
                int b_jk = bond_idx_arr[bo_j + pk];
                int k = bond_nbr[bo_j + pk];
                if (j >= k) continue;

                int type_k = atom_type[k];
                float bo_jk_val = bo_s[b_jk] + bo_pi[b_jk] + bo_pi2[b_jk];
                if (bo_jk_val <= thb_cut) continue;
                float bo_jk_pi_val = bo_pi[b_jk];

                VECTOR rk = crd[k];
                VECTOR djk = Get_Periodic_Displacement(rj, rk, cell, rcell);
                float r_jk = norm3df(djk.x, djk.y, djk.z);
                float delta_k_val = Delta_boc[k];

                int bc_k = bond_count[k];
                int bo_k = bond_offset[k];

                for (int pi = 0; pi < bc_j; pi++)
                {
                    int b_ij = bond_idx_arr[bo_j + pi];
                    int i = bond_nbr[bo_j + pi];
                    if (i == k) continue;
                    int type_i = atom_type[i];
                    float bo_ij_val = bo_s[b_ij] + bo_pi[b_ij] + bo_pi2[b_ij];
                    if (bo_ij_val <= thb_cut) continue;

                    VECTOR ri = crd[i];
                    VECTOR dji = Get_Periodic_Displacement(rj, ri, cell, rcell);
                    float r_ij = norm3df(dji.x, dji.y, dji.z);

                    for (int pl = 0; pl < bc_k; pl++)
                    {
                        int b_kl = bond_idx_arr[bo_k + pl];
                        int l = bond_nbr[bo_k + pl];
                        if (l == j || l == i) continue;
                        int type_l = atom_type[l];
                        float bo_kl_val =
                            bo_s[b_kl] + bo_pi[b_kl] + bo_pi2[b_kl];
                        if (bo_kl_val <= thb_cut) continue;
                        if (bo_ij_val * bo_jk_val * bo_kl_val <= thb_cut)
                            continue;

                        VECTOR rl = crd[l];
                        VECTOR dkl =
                            Get_Periodic_Displacement(rk, rl, cell, rcell);
                        float r_kl = norm3df(dkl.x, dkl.y, dkl.z);

                        float cos_ijk =
                            (dji.x * djk.x + dji.y * djk.y + dji.z * djk.z) /
                            (r_ij * r_jk);
                        if (cos_ijk > 1.0f) cos_ijk = 1.0f;
                        if (cos_ijk < -1.0f) cos_ijk = -1.0f;
                        float sin_ijk = sinf(acosf(cos_ijk));
                        if (sin_ijk < 1e-5f) sin_ijk = 1e-5f;

                        float cos_jkl = ((-djk.x) * dkl.x + (-djk.y) * dkl.y +
                                         (-djk.z) * dkl.z) /
                                        (r_jk * r_kl);
                        if (cos_jkl > 1.0f) cos_jkl = 1.0f;
                        if (cos_jkl < -1.0f) cos_jkl = -1.0f;
                        float sin_jkl = sinf(acosf(cos_jkl));
                        if (sin_jkl < 1e-5f) sin_jkl = 1e-5f;

                        float unnorm_cos_phi =
                            -(dji.x * djk.x + dji.y * djk.y + dji.z * djk.z) *
                                (djk.x * dkl.x + djk.y * dkl.y +
                                 djk.z * dkl.z) +
                            (r_jk * r_jk) *
                                (dji.x * dkl.x + dji.y * dkl.y + dji.z * dkl.z);
                        VECTOR cross_jk_kl = {
                            (float)(djk.y * dkl.z - djk.z * dkl.y),
                            (float)(djk.z * dkl.x - djk.x * dkl.z),
                            (float)(djk.x * dkl.y - djk.y * dkl.x)};
                        float unnorm_sin_phi = -r_jk * (dji.x * cross_jk_kl.x +
                                                        dji.y * cross_jk_kl.y +
                                                        dji.z * cross_jk_kl.z);
                        float phi = atan2f(unnorm_sin_phi, unnorm_cos_phi);
                        float cos_phi = cosf(phi);

                        int quartet_idx =
                            (((type_i * atom_type_numbers + type_j) *
                                  atom_type_numbers +
                              type_k) *
                                 atom_type_numbers +
                             type_l);
                        REAXFF_TORSION_Info info = torsion_info[quartet_idx];

                        SADvector<9> s_dji;
                        s_dji.x = SADfloat<9>(dji.x, 0);
                        s_dji.y = SADfloat<9>(dji.y, 1);
                        s_dji.z = SADfloat<9>(dji.z, 2);
                        SADvector<9> s_djk;
                        s_djk.x = SADfloat<9>(djk.x, 3);
                        s_djk.y = SADfloat<9>(djk.y, 4);
                        s_djk.z = SADfloat<9>(djk.z, 5);
                        SADvector<9> s_dkl;
                        s_dkl.x = SADfloat<9>(dkl.x, 6);
                        s_dkl.y = SADfloat<9>(dkl.y, 7);
                        s_dkl.z = SADfloat<9>(dkl.z, 8);

                        SADfloat<9> s_r_ij = norm3df(s_dji.x, s_dji.y, s_dji.z);
                        SADfloat<9> s_r_jk = norm3df(s_djk.x, s_djk.y, s_djk.z);
                        SADfloat<9> s_r_kl = norm3df(s_dkl.x, s_dkl.y, s_dkl.z);

                        SADfloat<9> s_cos_ijk =
                            (s_dji * s_djk) / (s_r_ij * s_r_jk);
                        if (s_cos_ijk.val > 1.0f) s_cos_ijk = SADfloat<9>(1.0f);
                        if (s_cos_ijk.val < -1.0f)
                            s_cos_ijk = SADfloat<9>(-1.0f);
                        SADfloat<9> s_sin_ijk = sinf(acosf(s_cos_ijk));
                        if (s_sin_ijk.val < 1e-5f)
                            s_sin_ijk = SADfloat<9>(1e-5f);

                        SADfloat<9> s_cos_jkl =
                            (((-1.0f) * s_djk) * s_dkl) / (s_r_jk * s_r_kl);
                        if (s_cos_jkl.val > 1.0f) s_cos_jkl = SADfloat<9>(1.0f);
                        if (s_cos_jkl.val < -1.0f)
                            s_cos_jkl = SADfloat<9>(-1.0f);
                        SADfloat<9> s_sin_jkl = sinf(acosf(s_cos_jkl));
                        if (s_sin_jkl.val < 1e-5f)
                            s_sin_jkl = SADfloat<9>(1e-5f);

                        SADfloat<9> s_unnorm_cos_phi =
                            -(s_dji * s_djk) * (s_djk * s_dkl) +
                            (s_r_jk * s_r_jk) * (s_dji * s_dkl);
                        SADvector<9> s_cross_jk_kl = s_djk ^ s_dkl;
                        SADfloat<9> s_unnorm_sin_phi =
                            -s_r_jk * (s_dji * s_cross_jk_kl);
                        SADfloat<9> s_phi =
                            atan2f(s_unnorm_sin_phi, s_unnorm_cos_phi);
                        SADfloat<9> s_cos_phi = cosf(s_phi);

                        for (int e = 0; e < info.entry_count; e++)
                        {
                            const REAXFF_TORSION_Entry* param =
                                &torsion_entries[info.start_idx + e];

                            SADfloat<6> s_bo_ij(bo_ij_val, 0);
                            SADfloat<6> s_bo_jk(bo_jk_val, 1);
                            SADfloat<6> s_bo_kl(bo_kl_val, 2);
                            SADfloat<6> s_delta_j(delta_j_val, 3);
                            SADfloat<6> s_delta_k(delta_k_val, 4);
                            SADfloat<6> s_bo_jk_pi(bo_jk_pi_val, 5);

                            SADfloat<6> s_boa_ij = s_bo_ij - thb_cut;
                            SADfloat<6> s_boa_jk = s_bo_jk - thb_cut;
                            SADfloat<6> s_boa_kl = s_bo_kl - thb_cut;

                            SADfloat<6> s_fn10 =
                                (1.0f - expf(-p_tor2 * s_boa_ij)) *
                                (1.0f - expf(-p_tor2 * s_boa_jk)) *
                                (1.0f - expf(-p_tor2 * s_boa_kl));
                            SADfloat<6> s_exp_tor3 =
                                expf(-p_tor3 * (s_delta_j + s_delta_k));
                            SADfloat<6> s_exp_tor4 =
                                expf(p_tor4 * (s_delta_j + s_delta_k));
                            SADfloat<6> s_f11_DjDk =
                                (2.0f + s_exp_tor3) /
                                (1.0f + s_exp_tor3 + s_exp_tor4);

                            SADfloat<6> s_tor_diff =
                                2.0f - s_bo_jk_pi - s_f11_DjDk;
                            SADfloat<6> s_exp_tor1 =
                                expf(param->p_tor1 * s_tor_diff * s_tor_diff);
                            SADfloat<6> s_CV =
                                0.5f * (param->V1 * (1.0f + cos_phi) +
                                        param->V2 * s_exp_tor1 *
                                            (1.0f - cosf(2.0f * phi)) +
                                        param->V3 * (1.0f + cosf(3.0f * phi)));

                            SADfloat<6> s_en_tor =
                                s_fn10 * sin_ijk * sin_jkl * s_CV;

                            SADfloat<6> s_en_cot(0.0f);
                            if (fabsf(param->p_cot1) > 0.0001f)
                            {
                                SADfloat<6> s_cot_diff_ij = s_boa_ij - 1.5f;
                                SADfloat<6> s_cot_diff_jk = s_boa_jk - 1.5f;
                                SADfloat<6> s_cot_diff_kl = s_boa_kl - 1.5f;
                                SADfloat<6> s_fn12 =
                                    expf(-p_cot2 * s_cot_diff_ij *
                                         s_cot_diff_ij) *
                                    expf(-p_cot2 * s_cot_diff_jk *
                                         s_cot_diff_jk) *
                                    expf(-p_cot2 * s_cot_diff_kl *
                                         s_cot_diff_kl);
                                s_en_cot = (float)param->p_cot1 * s_fn12 *
                                           (1.0f + (cos_phi * cos_phi - 1.0f) *
                                                       sin_ijk * sin_jkl);
                            }

                            SADfloat<6> s_en_total = s_en_tor + s_en_cot;

                            atomicAdd(&d_dE_dBO_s[b_ij], s_en_total.dval[0]);
                            atomicAdd(&d_dE_dBO_pi[b_ij], s_en_total.dval[0]);
                            atomicAdd(&d_dE_dBO_pi2[b_ij], s_en_total.dval[0]);

                            atomicAdd(&d_dE_dBO_s[b_jk], s_en_total.dval[1]);
                            atomicAdd(&d_dE_dBO_pi[b_jk],
                                      s_en_total.dval[1] + s_en_total.dval[5]);
                            atomicAdd(&d_dE_dBO_pi2[b_jk], s_en_total.dval[1]);

                            atomicAdd(&d_dE_dBO_s[b_kl], s_en_total.dval[2]);
                            atomicAdd(&d_dE_dBO_pi[b_kl], s_en_total.dval[2]);
                            atomicAdd(&d_dE_dBO_pi2[b_kl], s_en_total.dval[2]);

                            atomicAdd(&CdDelta[j], s_en_total.dval[3]);
                            atomicAdd(&CdDelta[k], s_en_total.dval[4]);

                            float boa_ij_val = bo_ij_val - thb_cut;
                            float boa_jk_val = bo_jk_val - thb_cut;
                            float boa_kl_val = bo_kl_val - thb_cut;
                            float fn10_val =
                                (1.0f - expf(-p_tor2 * boa_ij_val)) *
                                (1.0f - expf(-p_tor2 * boa_jk_val)) *
                                (1.0f - expf(-p_tor2 * boa_kl_val));
                            float exp_tor3_val =
                                expf(-p_tor3 * (delta_j_val + delta_k_val));
                            float exp_tor4_val =
                                expf(p_tor4 * (delta_j_val + delta_k_val));
                            float f11_DjDk_val =
                                (2.0f + exp_tor3_val) /
                                (1.0f + exp_tor3_val + exp_tor4_val);
                            float tor_diff_val =
                                2.0f - bo_jk_pi_val - f11_DjDk_val;
                            float exp_tor1_val = expf(
                                param->p_tor1 * tor_diff_val * tor_diff_val);

                            SADfloat<9> s_cv_dir =
                                0.5f *
                                (param->V1 * (1.0f + s_cos_phi) +
                                 param->V2 * exp_tor1_val *
                                     (1.0f - cosf(2.0f * s_phi)) +
                                 param->V3 * (1.0f + cosf(3.0f * s_phi)));
                            SADfloat<9> s_en_tor_dir =
                                fn10_val * s_sin_ijk * s_sin_jkl * s_cv_dir;

                            SADfloat<9> s_en_cot_dir(0.0f);
                            if (fabsf(param->p_cot1) > 0.0001f)
                            {
                                float cot_diff_ij_val = boa_ij_val - 1.5f;
                                float cot_diff_jk_val = boa_jk_val - 1.5f;
                                float cot_diff_kl_val = boa_kl_val - 1.5f;
                                float fn12_val =
                                    expf(-p_cot2 * cot_diff_ij_val *
                                         cot_diff_ij_val) *
                                    expf(-p_cot2 * cot_diff_jk_val *
                                         cot_diff_jk_val) *
                                    expf(-p_cot2 * cot_diff_kl_val *
                                         cot_diff_kl_val);
                                s_en_cot_dir =
                                    param->p_cot1 * fn12_val *
                                    (1.0f + (s_cos_phi * s_cos_phi - 1.0f) *
                                                s_sin_ijk * s_sin_jkl);
                            }
                            SADfloat<9> s_en_direct =
                                s_en_tor_dir + s_en_cot_dir;

                            VECTOR dE_ddji = {s_en_direct.dval[0],
                                              s_en_direct.dval[1],
                                              s_en_direct.dval[2]};
                            VECTOR dE_ddjk = {s_en_direct.dval[3],
                                              s_en_direct.dval[4],
                                              s_en_direct.dval[5]};
                            VECTOR dE_ddkl = {s_en_direct.dval[6],
                                              s_en_direct.dval[7],
                                              s_en_direct.dval[8]};

                            VECTOR fi = dE_ddji;
                            VECTOR fj = {-(dE_ddji.x + dE_ddjk.x),
                                         -(dE_ddji.y + dE_ddjk.y),
                                         -(dE_ddji.z + dE_ddjk.z)};
                            VECTOR fk = {dE_ddjk.x - dE_ddkl.x,
                                         dE_ddjk.y - dE_ddkl.y,
                                         dE_ddjk.z - dE_ddkl.z};
                            VECTOR fl = dE_ddkl;

                            atomicAdd(&frc[i].x, fi.x);
                            atomicAdd(&frc[i].y, fi.y);
                            atomicAdd(&frc[i].z, fi.z);
                            atomicAdd(&frc[j].x, fj.x);
                            atomicAdd(&frc[j].y, fj.y);
                            atomicAdd(&frc[j].z, fj.z);
                            atomicAdd(&frc[k].x, fk.x);
                            atomicAdd(&frc[k].y, fk.y);
                            atomicAdd(&frc[k].z, fk.z);
                            atomicAdd(&frc[l].x, fl.x);
                            atomicAdd(&frc[l].y, fl.y);
                            atomicAdd(&frc[l].z, fl.z);
                            if (atom_virial)
                            {
                                VECTOR dr_ji = {-dji.x, -dji.y, -dji.z};
                                VECTOR dr_jk = {-djk.x, -djk.y, -djk.z};
                                VECTOR dr_jl = {dr_jk.x - dkl.x,
                                                dr_jk.y - dkl.y,
                                                dr_jk.z - dkl.z};
                                LTMatrix3 v =
                                    Get_Virial_From_Force_Dis(fi, dr_ji) +
                                    Get_Virial_From_Force_Dis(fk, dr_jk) +
                                    Get_Virial_From_Force_Dis(fl, dr_jl);
                                atomicAdd(atom_virial + j, v);
                            }

                            en_tor += s_en_tor.val;
                            en_cot += s_en_cot.val;
                        }
                    }
                }
            }
            atomicAdd(d_energy_tor_sum, (float)en_tor);
            atomicAdd(d_energy_cot_sum, (float)en_cot);
            if (atom_energy)
                atomicAdd(&atom_energy[j], (float)(en_tor + en_cot));
        }
    }
}

void REAXFF_TORSION::Initial(CONTROLLER* controller, int atom_numbers,
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
            "REAXFF_TORSION IS NOT INITIALIZED (missing input files)\n\n");
        return;
    }
    controller->Step_Print_Initial("REAXFF_TOR", "%14.7e");
    controller->Step_Print_Initial("REAXFF_CONJ", "%14.7e");

    FILE* fp;
    Open_File_Safely(&fp, parameter_in_file, "r");
    char line[1024];
    auto throw_bad_format = [&](const char* file_name, const char* reason)
    {
        char error_msg[1024];
        sprintf(error_msg, "Reason:\n\t%s in file %s\n", reason, file_name);
        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                       "REAXFF_TORSION::Initial", error_msg);
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
    if (n_gp <= 29)
    {
        throw_bad_format(parameter_in_file,
                         "general parameter count is too small for torsion");
    }
    std::vector<float> gp(n_gp);
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
    p_tor2 = gp[23];
    p_tor3 = gp[24];
    p_tor4 = gp[25];
    p_cot2 = gp[27];

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

    read_line_or_throw(fp, parameter_in_file, "three-body count line");
    int n_thb = 0;
    if (sscanf(line, "%d", &n_thb) != 1 || n_thb < 0)
    {
        throw_bad_format(parameter_in_file,
                         "failed to parse number of three-body parameters");
    }
    for (int i = 0; i < n_thb; i++)
    {
        read_line_or_throw(fp, parameter_in_file, "three-body parameter entry");
    }

    read_line_or_throw(fp, parameter_in_file, "torsion count line");
    int n_tor = 0;
    if (sscanf(line, "%d", &n_tor) != 1 || n_tor < 0)
    {
        throw_bad_format(parameter_in_file,
                         "failed to parse number of torsion parameters");
    }
    struct TmpTorEntry
    {
        int t1, t2, t3, t4;
        REAXFF_TORSION_Entry entry;
    };
    std::vector<TmpTorEntry> tmp_entries;
    for (int i = 0; i < n_tor; i++)
    {
        read_line_or_throw(fp, parameter_in_file, "torsion parameter entry");
        int t1, t2, t3, t4;
        float v1, v2, v3, p1, p2, cot;
        int read_cnt = sscanf(line, "%d %d %d %d %f %f %f %f %f %f", &t1, &t2,
                              &t3, &t4, &v1, &v2, &v3, &p1, &p2, &cot);
        if (read_cnt < 9)
        {
            char reason[512];
            sprintf(reason,
                    "failed to parse torsion parameter entry at index %d",
                    i + 1);
            throw_bad_format(parameter_in_file, reason);
        }
        TmpTorEntry te;
        te.t1 = t1;
        te.t2 = t2;
        te.t3 = t3;
        te.t4 = t4;
        te.entry.p_tor1 = p1;
        te.entry.V1 = v1;
        te.entry.V2 = v2;
        te.entry.V3 = v3;
        te.entry.p_tor2 = 0.0f;
        te.entry.p_cot1 = p2;
        tmp_entries.push_back(te);
    }
    fclose(fp);

    int n4 = n_atom_types * n_atom_types * n_atom_types * n_atom_types;
    std::vector<int> tor_flag(n4, 0);
    std::vector<REAXFF_TORSION_Entry> all_entries;
    std::map<int, std::vector<int>> quartet_to_entries;

    for (const auto& te : tmp_entries)
    {
        if (te.t1 > 0 && te.t4 > 0)
        {
            int t1 = te.t1, t2 = te.t2, t3 = te.t3, t4 = te.t4;
            if (t1 < 1 || t1 > n_atom_types || t2 < 1 || t2 > n_atom_types ||
                t3 < 1 || t3 > n_atom_types || t4 < 1 || t4 > n_atom_types)
            {
                throw_bad_format(
                    parameter_in_file,
                    "torsion atom type index out of range for explicit entry");
            }
            int q_idx = (((t1 - 1) * n_atom_types + (t2 - 1)) * n_atom_types +
                         (t3 - 1)) *
                            n_atom_types +
                        (t4 - 1);
            int q_idx_rev =
                (((t4 - 1) * n_atom_types + (t3 - 1)) * n_atom_types +
                 (t2 - 1)) *
                    n_atom_types +
                (t1 - 1);
            int entry_idx = all_entries.size();
            all_entries.push_back(te.entry);
            quartet_to_entries[q_idx] = {entry_idx};
            tor_flag[q_idx] = 1;
            if (q_idx != q_idx_rev)
            {
                quartet_to_entries[q_idx_rev] = {entry_idx};
                tor_flag[q_idx_rev] = 1;
            }
        }
    }

    for (const auto& te : tmp_entries)
    {
        if (te.t1 == 0 && te.t4 == 0)
        {
            int t2 = te.t2, t3 = te.t3;
            if (t2 < 1 || t2 > n_atom_types || t3 < 1 || t3 > n_atom_types)
            {
                throw_bad_format(
                    parameter_in_file,
                    "torsion atom type index out of range for wildcard entry");
            }
            int entry_idx = all_entries.size();
            all_entries.push_back(te.entry);
            for (int p = 0; p < n_atom_types; p++)
            {
                for (int o = 0; o < n_atom_types; o++)
                {
                    int q_idx = (((p)*n_atom_types + (t2 - 1)) * n_atom_types +
                                 (t3 - 1)) *
                                    n_atom_types +
                                (o);
                    if (tor_flag[q_idx] == 0)
                        quartet_to_entries[q_idx] = {entry_idx};
                    int q_idx_rev =
                        (((o)*n_atom_types + (t3 - 1)) * n_atom_types +
                         (t2 - 1)) *
                            n_atom_types +
                        (p);
                    if (q_idx_rev != q_idx && tor_flag[q_idx_rev] == 0)
                        quartet_to_entries[q_idx_rev] = {entry_idx};
                }
            }
        }
    }

    Malloc_Safely((void**)&h_torsion_info, sizeof(REAXFF_TORSION_Info) * n4);
    memset(h_torsion_info, 0, sizeof(REAXFF_TORSION_Info) * n4);
    std::vector<REAXFF_TORSION_Entry> sorted_entries;
    for (int i = 0; i < n4; i++)
    {
        if (quartet_to_entries.count(i))
        {
            h_torsion_info[i].start_idx = sorted_entries.size();
            h_torsion_info[i].entry_count = quartet_to_entries[i].size();
            for (int idx : quartet_to_entries[i])
                sorted_entries.push_back(all_entries[idx]);
        }
    }
    Malloc_Safely((void**)&h_torsion_entries,
                  sizeof(REAXFF_TORSION_Entry) * sorted_entries.size());
    for (size_t i = 0; i < sorted_entries.size(); i++)
        h_torsion_entries[i] = sorted_entries[i];

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
    Device_Malloc_And_Copy_Safely((void**)&d_torsion_info, h_torsion_info,
                                  sizeof(REAXFF_TORSION_Info) * n4);
    Device_Malloc_And_Copy_Safely(
        (void**)&d_torsion_entries, h_torsion_entries,
        sizeof(REAXFF_TORSION_Entry) * sorted_entries.size());
    Device_Malloc_Safely((void**)&d_energy_tor_sum, sizeof(float));
    Device_Malloc_Safely((void**)&d_energy_cot_sum, sizeof(float));
    // thb_cutoff means the three-/four-body screening cutoff used by torsion
    // and conjugation terms. It is a control parameter, not the bond-order
    // cutoff in the force-field file.
    this->thb_cut = 0.001f;
    if (controller->Command_Exist(module_name, "thb_cutoff"))
    {
        this->thb_cut = atof(controller->Command(module_name, "thb_cutoff"));
    }
    is_initialized = 1;
}

void REAXFF_TORSION::Calculate_Torsion_Energy_And_Force(
    int atom_numbers, const VECTOR* crd, VECTOR* frc, const LTMatrix3 cell,
    const LTMatrix3 rcell, const ATOM_GROUP* nl, REAXFF_BOND_ORDER* bo_module,
    const float* Delta_boc, const int need_atom_energy, float* atom_energy,
    const int need_virial, LTMatrix3* atom_virial)
{
    if (!is_initialized) return;
    dim3 blockSize(32);
    dim3 gridSize((atom_numbers + blockSize.x - 1) / blockSize.x);
    deviceMemset(d_energy_tor_sum, 0, sizeof(float));
    deviceMemset(d_energy_cot_sum, 0, sizeof(float));

    Launch_Device_Kernel(
        Calculate_Torsion_Kernel, gridSize, blockSize, 0, NULL, atom_numbers,
        crd, d_atom_type, p_tor2, p_tor3, p_tor4, p_cot2, thb_cut, Delta_boc,
        d_torsion_info, d_torsion_entries, atom_type_numbers,
        bo_module->d_corrected_bo_s, bo_module->d_corrected_bo_pi,
        bo_module->d_corrected_bo_pi2, d_dE_dBO_s, d_dE_dBO_pi, d_dE_dBO_pi2,
        d_CdDelta, cell, rcell, atom_energy, frc,
        need_virial ? atom_virial : NULL, d_energy_tor_sum, d_energy_cot_sum,
        bo_module->d_bond_count, bo_module->d_bond_offset,
        bo_module->d_bond_nbr, bo_module->d_bond_idx);
}

void REAXFF_TORSION::Step_Print(CONTROLLER* controller)
{
    if (!is_initialized) return;
    deviceMemcpy(&h_energy_tor, d_energy_tor_sum, sizeof(float),
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(&h_energy_cot, d_energy_cot_sum, sizeof(float),
                 deviceMemcpyDeviceToHost);
    controller->Step_Print("REAXFF_TOR", h_energy_tor, true);
    controller->Step_Print("REAXFF_CONJ", h_energy_cot, true);
}

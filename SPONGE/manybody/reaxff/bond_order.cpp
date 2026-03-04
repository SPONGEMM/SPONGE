#include "bond_order.h"

static __global__ void Calculate_Uncorrected_Bond_Orders_Kernel(
    int atom_numbers, const VECTOR* crd, const LTMatrix3 cell,
    const LTMatrix3 rcell, float cutoff, const int* atom_type, const float* r_s,
    const float* r_p, const float* r_pp, const float* bo_1, const float* bo_2,
    const float* bo_3, const float* bo_4, const float* bo_5, const float* bo_6,
    const float* ro_pi, const float* ro_pi2, const int atom_type_numbers,
    float bo_cut, float* total_bond_order, int* pair_i, int* pair_j,
    float* distances, int max_pairs, int* num_pairs)
{
    SIMPLE_DEVICE_FOR(i, atom_numbers)
    {
        int type_i = atom_type[i];
        if (type_i >= 0 && type_i < atom_type_numbers)
        {
            VECTOR ri = crd[i];

            for (int j = i + 1; j < atom_numbers; j++)
            {
                int type_j = atom_type[j];
                if (type_j < 0 || type_j >= atom_type_numbers) continue;

                VECTOR rj = crd[j];
                VECTOR drij = Get_Periodic_Displacement(ri, rj, cell, rcell);
                float r2 = drij.x * drij.x + drij.y * drij.y + drij.z * drij.z;

                if (r2 < cutoff * cutoff && r2 > 0.0001f)
                {
                    float r = sqrtf(r2);

                    int idx = type_i * atom_type_numbers + type_j;

                    float ros = r_s[idx];
                    float bo_s = 0.0f;
                    if (ros > 0.0f)
                    {
                        float C12 = bo_1[idx] * powf(r / ros, bo_2[idx]);
                        bo_s = (1.0f + bo_cut) * expf(C12);
                    }

                    float bo_p = 0.0f;
                    if (ro_pi[type_i] > 0.0f && ro_pi[type_j] > 0.0f)
                    {
                        float rop = r_p[idx];
                        if (rop > 0.0f)
                        {
                            float C34 = bo_3[idx] * powf(r / rop, bo_4[idx]);
                            bo_p = expf(C34);
                        }
                    }

                    float bo_p2 = 0.0f;
                    if (ro_pi2[type_i] > 0.0f && ro_pi2[type_j] > 0.0f)
                    {
                        float rop2 = r_pp[idx];
                        if (rop2 > 0.0f)
                        {
                            float C56 = bo_5[idx] * powf(r / rop2, bo_6[idx]);
                            bo_p2 = expf(C56);
                        }
                    }

                    float total_bo = bo_s + bo_p + bo_p2;

                    if (total_bo >= bo_cut)
                    {
                        bo_s -= bo_cut;
                        if (bo_s < 0.0f) bo_s = 0.0f;
                        total_bo -= bo_cut;

                        atomicAdd(&total_bond_order[i], total_bo);
                        atomicAdd(&total_bond_order[j], total_bo);

                        int pos = atomicAdd(num_pairs, 1);
                        if (pos < max_pairs)
                        {
                            pair_i[pos] = i;
                            pair_j[pos] = j;
                            distances[pos] = r;
                        }
                    }
                }
            }
        }
    }
}

static __global__ void Apply_Bond_Order_Corrections_Kernel(
    int num_pairs, int* pair_i, int* pair_j, float* distances,
    const VECTOR* crd, const LTMatrix3 cell, const LTMatrix3 rcell,
    const int* atom_type, const float* r_s, const float* r_p, const float* r_pp,
    const float* bo_1, const float* bo_2, const float* bo_3, const float* bo_4,
    const float* bo_5, const float* bo_6, const float* ro_pi,
    const float* ro_pi2, const float* valency, const float* valency_val,
    const float* ovc, const float* v13cor, const float* p_boc3,
    const float* p_boc4, const float* p_boc5, const int atom_type_numbers,
    const int atom_numbers, float gp_boc1, float gp_boc2, float bo_cut,
    const float* total_bond_order, float* corrected_bo, float* corrected_bo_s,
    float* corrected_bo_pi, float* corrected_bo_pi2, float* dbo_s_dr,
    float* dbo_pi_dr, float* dbo_pi2_dr, float* dbo_s_dDelta_i,
    float* dbo_pi_dDelta_i, float* dbo_pi2_dDelta_i, float* dbo_s_dDelta_j,
    float* dbo_pi_dDelta_j, float* dbo_pi2_dDelta_j, float* dbo_raw_total_dr)
{
    SIMPLE_DEVICE_FOR(idx, num_pairs)
    {
        int i = pair_i[idx];
        int j = pair_j[idx];
        float r = distances[idx];

        int type_i = atom_type[i];
        int type_j = atom_type[j];

        if (type_i >= 0 && type_i < atom_type_numbers && type_j >= 0 &&
            type_j < atom_type_numbers)
        {
            int pair_idx = type_i * atom_type_numbers + type_j;

            float ros = r_s[pair_idx];
            float bo_s_raw_val = 0.0f, dbo_s_raw_dr = 0.0f;
            if (ros > 0.0f)
            {
                float ratio = r / ros;
                float pow_ratio = powf(ratio, bo_2[pair_idx]);
                bo_s_raw_val =
                    (1.0f + bo_cut) * expf(bo_1[pair_idx] * pow_ratio);
                dbo_s_raw_dr = bo_s_raw_val * bo_1[pair_idx] * bo_2[pair_idx] *
                               powf(ratio, bo_2[pair_idx] - 1.0f) *
                               (1.0f / ros);
            }

            float bo_p_val = 0.0f, dbo_p_raw_dr = 0.0f;
            if (ro_pi[type_i] > 0.0f && ro_pi[type_j] > 0.0f)
            {
                float rop = r_p[pair_idx];
                if (rop > 0.0f)
                {
                    float ratio = r / rop;
                    float pow_ratio = powf(ratio, bo_4[pair_idx]);
                    bo_p_val = expf(bo_3[pair_idx] * pow_ratio);
                    dbo_p_raw_dr = bo_p_val * bo_3[pair_idx] * bo_4[pair_idx] *
                                   powf(ratio, bo_4[pair_idx] - 1.0f) *
                                   (1.0f / rop);
                }
            }

            float bo_p2_val = 0.0f, dbo_p2_raw_dr = 0.0f;
            if (ro_pi2[type_i] > 0.0f && ro_pi2[type_j] > 0.0f)
            {
                float rop2 = r_pp[pair_idx];
                if (rop2 > 0.0f)
                {
                    float ratio = r / rop2;
                    float pow_ratio = powf(ratio, bo_6[pair_idx]);
                    bo_p2_val = expf(bo_5[pair_idx] * pow_ratio);
                    dbo_p2_raw_dr =
                        bo_p2_val * bo_5[pair_idx] * bo_6[pair_idx] *
                        powf(ratio, bo_6[pair_idx] - 1.0f) * (1.0f / rop2);
                }
            }

            float total_bo_raw = bo_s_raw_val + bo_p_val + bo_p2_val;
            float dbo_raw_total_dr_val =
                dbo_s_raw_dr + dbo_p_raw_dr + dbo_p2_raw_dr;

            int dense_idx = i * atom_numbers + j;
            int dense_idx_ji = j * atom_numbers + i;

            if (total_bo_raw >= bo_cut)
            {
                SADfloat<5> bo_s_raw(bo_s_raw_val, 0);
                SADfloat<5> bo_p(bo_p_val, 1);
                SADfloat<5> bo_p2(bo_p2_val, 2);
                SADfloat<5> Delta_i(total_bond_order[i], 3);
                SADfloat<5> Delta_j(total_bond_order[j], 4);

                SADfloat<5> total_bo_orig = (bo_s_raw + bo_p + bo_p2) - bo_cut;
                SADfloat<5> bo_s = bo_s_raw - bo_cut;
                if (bo_s.val < 0) bo_s = SADfloat<5>(0.0f);

                float ovc_val = ovc[pair_idx];
                float v13cor_val = v13cor[pair_idx];

                SADfloat<5> f1(1.0f);
                if (ovc_val >= 0.001f)
                {
                    SADfloat<5> Deltap_i = Delta_i - valency[type_i];
                    SADfloat<5> Deltap_j = Delta_j - valency[type_j];

                    SADfloat<5> exp_p1i = expf(-gp_boc1 * Deltap_i);
                    SADfloat<5> exp_p1j = expf(-gp_boc1 * Deltap_j);
                    SADfloat<5> f2 = exp_p1i + exp_p1j;

                    SADfloat<5> f3 =
                        -1.0f / gp_boc2 *
                        (Log_Sum_Exp(-gp_boc2 * Deltap_i, -gp_boc2 * Deltap_j) -
                         0.6931471805599453f);

                    float val_i = valency[type_i];
                    float val_j = valency[type_j];

                    f1 = 0.5f * ((val_i + f2) / (val_i + f2 + f3) +
                                 (val_j + f2) / (val_j + f2 + f3));
                }

                SADfloat<5> f4(1.0f), f5(1.0f);
                if (v13cor_val >= 0.001f)
                {
                    SADfloat<5> Deltap_boc_i = Delta_i - valency_val[type_i];
                    SADfloat<5> Deltap_boc_j = Delta_j - valency_val[type_j];

                    float p_boc3_val = p_boc3[pair_idx];
                    float p_boc4_val = p_boc4[pair_idx];
                    float p_boc5_val = p_boc5[pair_idx];

                    SADfloat<5> exp_f4 =
                        expf(-(p_boc4_val * total_bo_orig * total_bo_orig -
                               Deltap_boc_i) *
                                 p_boc3_val +
                             p_boc5_val);
                    SADfloat<5> exp_f5 =
                        expf(-(p_boc4_val * total_bo_orig * total_bo_orig -
                               Deltap_boc_j) *
                                 p_boc3_val +
                             p_boc5_val);

                    f4 = 1.0f / (1.0f + exp_f4);
                    f5 = 1.0f / (1.0f + exp_f5);
                }

                SADfloat<5> A0 = f1 * f4 * f5;

                SADfloat<5> s_corrected_bo_pi = bo_p * A0 * f1;
                SADfloat<5> s_corrected_bo_pi2 = bo_p2 * A0 * f1;
                SADfloat<5> s_corrected_bo_s =
                    total_bo_orig * A0 -
                    (s_corrected_bo_pi + s_corrected_bo_pi2);
                if (s_corrected_bo_s.val < 0)
                    s_corrected_bo_s = SADfloat<5>(0.0f);

                corrected_bo_s[dense_idx] = s_corrected_bo_s.val;
                corrected_bo_pi[dense_idx] = s_corrected_bo_pi.val;
                corrected_bo_pi2[dense_idx] = s_corrected_bo_pi2.val;

                corrected_bo_s[dense_idx_ji] = s_corrected_bo_s.val;
                corrected_bo_pi[dense_idx_ji] = s_corrected_bo_pi.val;
                corrected_bo_pi2[dense_idx_ji] = s_corrected_bo_pi2.val;

                dbo_s_dr[dense_idx] = dbo_s_dr[dense_idx_ji] =
                    s_corrected_bo_s.dval[0] * dbo_s_raw_dr +
                    s_corrected_bo_s.dval[1] * dbo_p_raw_dr +
                    s_corrected_bo_s.dval[2] * dbo_p2_raw_dr;

                dbo_pi_dr[dense_idx] = dbo_pi_dr[dense_idx_ji] =
                    s_corrected_bo_pi.dval[0] * dbo_s_raw_dr +
                    s_corrected_bo_pi.dval[1] * dbo_p_raw_dr +
                    s_corrected_bo_pi.dval[2] * dbo_p2_raw_dr;

                dbo_pi2_dr[dense_idx] = dbo_pi2_dr[dense_idx_ji] =
                    s_corrected_bo_pi2.dval[0] * dbo_s_raw_dr +
                    s_corrected_bo_pi2.dval[1] * dbo_p_raw_dr +
                    s_corrected_bo_pi2.dval[2] * dbo_p2_raw_dr;

                dbo_s_dDelta_i[dense_idx] = s_corrected_bo_s.dval[3];
                dbo_pi_dDelta_i[dense_idx] = s_corrected_bo_pi.dval[3];
                dbo_pi2_dDelta_i[dense_idx] = s_corrected_bo_pi2.dval[3];

                dbo_s_dDelta_j[dense_idx] = s_corrected_bo_s.dval[4];
                dbo_pi_dDelta_j[dense_idx] = s_corrected_bo_pi.dval[4];
                dbo_pi2_dDelta_j[dense_idx] = s_corrected_bo_pi2.dval[4];

                dbo_s_dDelta_i[dense_idx_ji] = s_corrected_bo_s.dval[4];
                dbo_pi_dDelta_i[dense_idx_ji] = s_corrected_bo_pi.dval[4];
                dbo_pi2_dDelta_i[dense_idx_ji] = s_corrected_bo_pi2.dval[4];

                dbo_s_dDelta_j[dense_idx_ji] = s_corrected_bo_s.dval[3];
                dbo_pi_dDelta_j[dense_idx_ji] = s_corrected_bo_pi.dval[3];
                dbo_pi2_dDelta_j[dense_idx_ji] = s_corrected_bo_pi2.dval[3];

                dbo_raw_total_dr[dense_idx] = dbo_raw_total_dr[dense_idx_ji] =
                    dbo_raw_total_dr_val;

                corrected_bo[idx] = s_corrected_bo_s.val +
                                    s_corrected_bo_pi.val +
                                    s_corrected_bo_pi2.val;
            }
            else
            {
                corrected_bo[idx] = 0.0f;
                corrected_bo_s[dense_idx] = corrected_bo_s[dense_idx_ji] = 0.0f;
                corrected_bo_pi[dense_idx] = corrected_bo_pi[dense_idx_ji] =
                    0.0f;
                corrected_bo_pi2[dense_idx] = corrected_bo_pi2[dense_idx_ji] =
                    0.0f;

                dbo_s_dr[dense_idx] = dbo_s_dr[dense_idx_ji] = 0.0f;
                dbo_pi_dr[dense_idx] = dbo_pi_dr[dense_idx_ji] = 0.0f;
                dbo_pi2_dr[dense_idx] = dbo_pi2_dr[dense_idx_ji] = 0.0f;

                dbo_s_dDelta_i[dense_idx] = dbo_s_dDelta_i[dense_idx_ji] = 0.0f;
                dbo_pi_dDelta_i[dense_idx] = dbo_pi_dDelta_i[dense_idx_ji] =
                    0.0f;
                dbo_pi2_dDelta_i[dense_idx] = dbo_pi2_dDelta_i[dense_idx_ji] =
                    0.0f;

                dbo_s_dDelta_j[dense_idx] = dbo_s_dDelta_j[dense_idx_ji] = 0.0f;
                dbo_pi_dDelta_j[dense_idx] = dbo_pi_dDelta_j[dense_idx_ji] =
                    0.0f;
                dbo_pi2_dDelta_j[dense_idx] = dbo_pi2_dDelta_j[dense_idx_ji] =
                    0.0f;

                dbo_raw_total_dr[dense_idx] = dbo_raw_total_dr[dense_idx_ji] =
                    0.0f;
            }
        }
    }
}

static __global__ void Reduce_Total_Corrected_Bond_Order_Kernel(
    int atom_numbers, const float* bo_s, const float* bo_pi,
    const float* bo_pi2, float* total_bo)
{
    SIMPLE_DEVICE_FOR(i, atom_numbers)
    {
        float sum = 0.0f;
        for (int j = 0; j < atom_numbers; j++)
        {
            if (i == j) continue;
            int idx = i * atom_numbers + j;
            sum += bo_s[idx] + bo_pi[idx] + bo_pi2[idx];
        }
        total_bo[i] = sum;
    }
}

void REAXFF_BOND_ORDER::Initial(CONTROLLER* controller, int atom_numbers,
                                const char* parameter_in_file,
                                const char* type_in_file, const float cutoff,
                                float* cutoff_full)
{
    if (parameter_in_file == NULL || type_in_file == NULL)
    {
        return;
    }

    this->atom_numbers = atom_numbers;
    controller->printf("START INITIALIZING REAXFF_BOND_ORDER\n");

    FILE* fp_p;
    Open_File_Safely(&fp_p, parameter_in_file, "r");
    char line[1024];
    auto throw_bad_format = [&](const char* file_name, const char* reason)
    {
        char error_msg[1024];
        sprintf(error_msg, "Reason:\n\t%s in file %s\n", reason, file_name);
        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                       "REAXFF_BOND_ORDER::Initial", error_msg);
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

    read_line_or_throw(fp_p, parameter_in_file, "parameter header line 1");
    read_line_or_throw(fp_p, parameter_in_file, "general parameter count line");
    int n_gen_params = 0;
    if (sscanf(line, "%d", &n_gen_params) != 1 || n_gen_params < 2)
    {
        throw_bad_format(parameter_in_file,
                         "failed to parse number of general parameters");
    }

    std::vector<float> gen_params(n_gen_params, 0.0f);
    for (int i = 0; i < n_gen_params; i++)
    {
        read_line_or_throw(fp_p, parameter_in_file, "general parameter block");
        if (sscanf(line, "%f", &gen_params[i]) != 1)
        {
            char reason[512];
            sprintf(reason, "failed to parse general parameter at index %d",
                    i + 1);
            throw_bad_format(parameter_in_file, reason);
        }
    }
    gp_boc1 = gen_params[0];
    gp_boc2 = gen_params[1];
    if (n_gen_params > 29) gp_bo_cut = 0.01f * gen_params[29];
    controller->printf("Debug: gp_boc1 = %f, gp_boc2 = %f, gp_bo_cut = %f\n",
                       gp_boc1, gp_boc2, gp_bo_cut);

    read_line_or_throw(fp_p, parameter_in_file, "atom type count line");
    int n_atom_types = 0;
    if (sscanf(line, "%d", &n_atom_types) != 1 || n_atom_types <= 0)
    {
        throw_bad_format(parameter_in_file,
                         "failed to parse number of atom types");
    }
    this->atom_type_numbers = n_atom_types;
    controller->printf("Debug: n_atom_types = %d\n", n_atom_types);

    read_line_or_throw(fp_p, parameter_in_file, "atom type header line 1");
    read_line_or_throw(fp_p, parameter_in_file, "atom type header line 2");
    read_line_or_throw(fp_p, parameter_in_file, "atom type header line 3");

    std::map<std::string, int> type_map;
    Malloc_Safely((void**)&h_ro_sigma, sizeof(float) * n_atom_types);
    Malloc_Safely((void**)&h_ro_pi, sizeof(float) * n_atom_types);
    Malloc_Safely((void**)&h_ro_pi2, sizeof(float) * n_atom_types);
    Malloc_Safely((void**)&h_valency, sizeof(float) * n_atom_types);
    Malloc_Safely((void**)&h_valency_val, sizeof(float) * n_atom_types);
    Malloc_Safely((void**)&h_b_o_131, sizeof(float) * n_atom_types);
    Malloc_Safely((void**)&h_b_o_132, sizeof(float) * n_atom_types);
    Malloc_Safely((void**)&h_b_o_133, sizeof(float) * n_atom_types);

    for (int i = 0; i < n_atom_types; i++)
    {
        read_line_or_throw(fp_p, parameter_in_file,
                           "atom type parameter line 1");
        char element_name[16];
        float ro_sigma, valency, mass, r_vdw, epsilon, gamma, ro_pi, valency_e;
        if (sscanf(line, "%s %f %f %f %f %f %f %f %f", element_name, &ro_sigma,
                   &valency, &mass, &r_vdw, &epsilon, &gamma, &ro_pi,
                   &valency_e) != 9)
        {
            char reason[512];
            sprintf(reason,
                    "failed to parse atom type block line 1 for type index %d",
                    i + 1);
            throw_bad_format(parameter_in_file, reason);
        }
        type_map[std::string(element_name)] = i;
        h_ro_sigma[i] = ro_sigma;
        h_ro_pi[i] = ro_pi;
        h_valency[i] = valency;

        read_line_or_throw(fp_p, parameter_in_file,
                           "atom type parameter line 2");

        read_line_or_throw(fp_p, parameter_in_file,
                           "atom type parameter line 3");
        float ro_pi_pi, p_lp2, heat_inc, boc4_i, boc3_i, boc5_i;
        if (sscanf(line, "%f %f %f %f %f %f", &ro_pi_pi, &p_lp2, &heat_inc,
                   &boc4_i, &boc3_i, &boc5_i) != 6)
        {
            char reason[512];
            sprintf(reason,
                    "failed to parse atom type block line 3 for type index %d",
                    i + 1);
            throw_bad_format(parameter_in_file, reason);
        }
        h_ro_pi2[i] = ro_pi_pi;
        h_b_o_131[i] = boc4_i;
        h_b_o_132[i] = boc3_i;
        h_b_o_133[i] = boc5_i;

        read_line_or_throw(fp_p, parameter_in_file,
                           "atom type parameter line 4");
        float p_ovun2, p_val3, unused, valency_val, p_val5;
        if (sscanf(line, "%f %f %f %f %f", &p_ovun2, &p_val3, &unused,
                   &valency_val, &p_val5) != 5)
        {
            char reason[512];
            sprintf(reason,
                    "failed to parse atom type block line 4 for type index %d",
                    i + 1);
            throw_bad_format(parameter_in_file, reason);
        }
        h_valency_val[i] = valency_val;
    }
    controller->printf("Debug: Atom params read.\n");

    read_line_or_throw(fp_p, parameter_in_file, "bond parameter count line");
    int n_bond_params = 0;
    if (sscanf(line, "%d", &n_bond_params) != 1 || n_bond_params < 0)
    {
        throw_bad_format(parameter_in_file,
                         "failed to parse number of bond parameters");
    }
    controller->printf("Debug: n_bond_params = %d\n", n_bond_params);

    Malloc_Safely((void**)&h_bo_1, sizeof(float) * n_atom_types * n_atom_types);
    Malloc_Safely((void**)&h_bo_2, sizeof(float) * n_atom_types * n_atom_types);
    Malloc_Safely((void**)&h_bo_3, sizeof(float) * n_atom_types * n_atom_types);
    Malloc_Safely((void**)&h_bo_4, sizeof(float) * n_atom_types * n_atom_types);
    Malloc_Safely((void**)&h_bo_5, sizeof(float) * n_atom_types * n_atom_types);
    Malloc_Safely((void**)&h_bo_6, sizeof(float) * n_atom_types * n_atom_types);
    Malloc_Safely((void**)&h_ovc, sizeof(float) * n_atom_types * n_atom_types);
    Malloc_Safely((void**)&h_v13cor,
                  sizeof(float) * n_atom_types * n_atom_types);
    Malloc_Safely((void**)&h_p_boc3,
                  sizeof(float) * n_atom_types * n_atom_types);
    Malloc_Safely((void**)&h_p_boc4,
                  sizeof(float) * n_atom_types * n_atom_types);
    Malloc_Safely((void**)&h_p_boc5,
                  sizeof(float) * n_atom_types * n_atom_types);
    Malloc_Safely((void**)&h_r_s, sizeof(float) * n_atom_types * n_atom_types);
    Malloc_Safely((void**)&h_r_p, sizeof(float) * n_atom_types * n_atom_types);
    Malloc_Safely((void**)&h_r_pp, sizeof(float) * n_atom_types * n_atom_types);

    for (int i = 0; i < n_atom_types * n_atom_types; i++)
    {
        h_bo_1[i] = 0.0f;
        h_bo_2[i] = 0.0f;
        h_bo_3[i] = 0.0f;
        h_bo_4[i] = 0.0f;
        h_bo_5[i] = 0.0f;
        h_bo_6[i] = 0.0f;
        h_ovc[i] = 0.0f;
        h_v13cor[i] = 0.0f;
        h_p_boc3[i] = 0.0f;
        h_p_boc4[i] = 0.0f;
        h_p_boc5[i] = 0.0f;
        h_r_s[i] = 0.0f;
        h_r_p[i] = 0.0f;
        h_r_pp[i] = 0.0f;
    }

    for (int i = 0; i < n_atom_types; i++)
    {
        for (int j = 0; j < n_atom_types; j++)
        {
            int idx = i * n_atom_types + j;
            h_p_boc3[idx] = sqrtf(h_b_o_132[i] * h_b_o_132[j]);
            h_p_boc4[idx] = sqrtf(h_b_o_131[i] * h_b_o_131[j]);
            h_p_boc5[idx] = sqrtf(h_b_o_133[i] * h_b_o_133[j]);
            h_r_s[idx] = 0.5f * (h_ro_sigma[i] + h_ro_sigma[j]);
            h_r_p[idx] = 0.5f * (h_ro_pi[i] + h_ro_pi[j]);
            h_r_pp[idx] = 0.5f * (h_ro_pi2[i] + h_ro_pi2[j]);
        }
    }

    read_line_or_throw(fp_p, parameter_in_file, "bond parameter header line");

    for (int i = 0; i < n_bond_params; i++)
    {
        read_line_or_throw(fp_p, parameter_in_file, "bond parameter line 1");
        int t1, t2;
        float De_s, De_p, De_pp, p_be1, p_bo5_val, v13cor_val, p_bo6_val,
            p_ovun1;
        if (sscanf(line, "%d %d %f %f %f %f %f %f %f %f", &t1, &t2, &De_s,
                   &De_p, &De_pp, &p_be1, &p_bo5_val, &v13cor_val, &p_bo6_val,
                   &p_ovun1) != 10)
        {
            char reason[512];
            sprintf(reason, "failed to parse bond parameter line 1 at index %d",
                    i + 1);
            throw_bad_format(parameter_in_file, reason);
        }

        int idx1 = t1 - 1;
        int idx2 = t2 - 1;

        if (idx1 < 0 || idx1 >= n_atom_types || idx2 < 0 ||
            idx2 >= n_atom_types)
        {
            char reason[512];
            sprintf(reason,
                    "invalid bond type indices %d %d (max atom type %d)", t1,
                    t2, n_atom_types);
            throw_bad_format(parameter_in_file, reason);
        }

        read_line_or_throw(fp_p, parameter_in_file, "bond parameter line 2");
        float p_be2, p_bo3_val, p_bo4_val, unused1, p_bo1_val, p_bo2_val,
            ovc_val;
        if (sscanf(line, "%f %f %f %f %f %f %f", &p_be2, &p_bo3_val, &p_bo4_val,
                   &unused1, &p_bo1_val, &p_bo2_val, &ovc_val) != 7)
        {
            char reason[512];
            sprintf(reason, "failed to parse bond parameter line 2 at index %d",
                    i + 1);
            throw_bad_format(parameter_in_file, reason);
        }

        h_bo_1[idx1 * n_atom_types + idx2] =
            h_bo_1[idx2 * n_atom_types + idx1] = p_bo1_val;
        h_bo_2[idx1 * n_atom_types + idx2] =
            h_bo_2[idx2 * n_atom_types + idx1] = p_bo2_val;
        h_bo_3[idx1 * n_atom_types + idx2] =
            h_bo_3[idx2 * n_atom_types + idx1] = p_bo3_val;
        h_bo_4[idx1 * n_atom_types + idx2] =
            h_bo_4[idx2 * n_atom_types + idx1] = p_bo4_val;
        h_bo_5[idx1 * n_atom_types + idx2] =
            h_bo_5[idx2 * n_atom_types + idx1] = p_bo5_val;
        h_bo_6[idx1 * n_atom_types + idx2] =
            h_bo_6[idx2 * n_atom_types + idx1] = p_bo6_val;

        h_ovc[idx1 * n_atom_types + idx2] = h_ovc[idx2 * n_atom_types + idx1] =
            ovc_val;
        h_v13cor[idx1 * n_atom_types + idx2] =
            h_v13cor[idx2 * n_atom_types + idx1] = v13cor_val;
    }

    if (fgets(line, 1024, fp_p) != NULL)
    {
        int n_off = 0;
        if (sscanf(line, "%d", &n_off) != 1 || n_off < 0)
        {
            throw_bad_format(parameter_in_file,
                             "failed to parse number of off-diagonal terms");
        }
        for (int off = 0; off < n_off; off++)
        {
            read_line_or_throw(fp_p, parameter_in_file,
                               "off-diagonal parameter line");
            int t1, t2;
            float dij = 0.0f, rvdw = 0.0f, alfa = 0.0f;
            float ro_sigma_od = -1.0f, ro_pi_od = -1.0f, ro_pipi_od = -1.0f;
            int read_cnt =
                sscanf(line, "%d %d %f %f %f %f %f %f", &t1, &t2, &dij, &rvdw,
                       &alfa, &ro_sigma_od, &ro_pi_od, &ro_pipi_od);
            if (read_cnt < 5)
            {
                char reason[512];
                sprintf(reason,
                        "failed to parse off-diagonal parameters at index %d",
                        off + 1);
                throw_bad_format(parameter_in_file, reason);
            }
            int idx1 = t1 - 1;
            int idx2 = t2 - 1;
            if (idx1 < 0 || idx1 >= n_atom_types || idx2 < 0 ||
                idx2 >= n_atom_types)
            {
                char reason[512];
                sprintf(reason,
                        "invalid off-diagonal type indices %d %d at index %d",
                        t1, t2, off + 1);
                throw_bad_format(parameter_in_file, reason);
            }
            int pair_idx = idx1 * n_atom_types + idx2;
            if (ro_sigma_od > 0.0f)
            {
                h_r_s[pair_idx] = h_r_s[idx2 * n_atom_types + idx1] =
                    ro_sigma_od;
            }
            if (ro_pi_od > 0.0f)
            {
                h_r_p[pair_idx] = h_r_p[idx2 * n_atom_types + idx1] = ro_pi_od;
            }
            if (ro_pipi_od > 0.0f)
            {
                h_r_pp[pair_idx] = h_r_pp[idx2 * n_atom_types + idx1] =
                    ro_pipi_od;
            }
        }
    }
    fclose(fp_p);

    FILE* fp_t;
    Open_File_Safely(&fp_t, type_in_file, "r");
    int check_atom_numbers = 0;
    read_line_or_throw(fp_t, type_in_file, "atom number line");
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
        char type_name[16];
        read_line_or_throw(fp_t, type_in_file, "atom type entry line");
        if (sscanf(line, "%s", type_name) != 1)
        {
            char reason[512];
            sprintf(reason, "failed to parse atom type at index %d", i + 1);
            throw_bad_format(type_in_file, reason);
        }
        std::string type_str(type_name);
        if (type_map.find(type_str) != type_map.end())
        {
            h_atom_type[i] = type_map[type_str];
        }
        else
        {
            char reason[512];
            sprintf(reason, "atom type %s not found in parameter file %s",
                    type_name, parameter_in_file);
            throw_bad_format(type_in_file, reason);
        }
    }
    fclose(fp_t);

    Device_Malloc_And_Copy_Safely((void**)&d_ro_sigma, h_ro_sigma,
                                  sizeof(float) * n_atom_types);
    Device_Malloc_And_Copy_Safely((void**)&d_ro_pi, h_ro_pi,
                                  sizeof(float) * n_atom_types);
    Device_Malloc_And_Copy_Safely((void**)&d_ro_pi2, h_ro_pi2,
                                  sizeof(float) * n_atom_types);
    Device_Malloc_And_Copy_Safely((void**)&d_bo_1, h_bo_1,
                                  sizeof(float) * n_atom_types * n_atom_types);
    Device_Malloc_And_Copy_Safely((void**)&d_bo_2, h_bo_2,
                                  sizeof(float) * n_atom_types * n_atom_types);
    Device_Malloc_And_Copy_Safely((void**)&d_bo_3, h_bo_3,
                                  sizeof(float) * n_atom_types * n_atom_types);
    Device_Malloc_And_Copy_Safely((void**)&d_bo_4, h_bo_4,
                                  sizeof(float) * n_atom_types * n_atom_types);
    Device_Malloc_And_Copy_Safely((void**)&d_bo_5, h_bo_5,
                                  sizeof(float) * n_atom_types * n_atom_types);
    Device_Malloc_And_Copy_Safely((void**)&d_bo_6, h_bo_6,
                                  sizeof(float) * n_atom_types * n_atom_types);
    Device_Malloc_And_Copy_Safely((void**)&d_r_s, h_r_s,
                                  sizeof(float) * n_atom_types * n_atom_types);
    Device_Malloc_And_Copy_Safely((void**)&d_r_p, h_r_p,
                                  sizeof(float) * n_atom_types * n_atom_types);
    Device_Malloc_And_Copy_Safely((void**)&d_r_pp, h_r_pp,
                                  sizeof(float) * n_atom_types * n_atom_types);
    Device_Malloc_And_Copy_Safely((void**)&d_valency, h_valency,
                                  sizeof(float) * n_atom_types);
    Device_Malloc_And_Copy_Safely((void**)&d_valency_val, h_valency_val,
                                  sizeof(float) * n_atom_types);
    Device_Malloc_And_Copy_Safely((void**)&d_ovc, h_ovc,
                                  sizeof(float) * n_atom_types * n_atom_types);
    Device_Malloc_And_Copy_Safely((void**)&d_v13cor, h_v13cor,
                                  sizeof(float) * n_atom_types * n_atom_types);
    Device_Malloc_And_Copy_Safely((void**)&d_p_boc3, h_p_boc3,
                                  sizeof(float) * n_atom_types * n_atom_types);
    Device_Malloc_And_Copy_Safely((void**)&d_p_boc4, h_p_boc4,
                                  sizeof(float) * n_atom_types * n_atom_types);
    Device_Malloc_And_Copy_Safely((void**)&d_p_boc5, h_p_boc5,
                                  sizeof(float) * n_atom_types * n_atom_types);
    Device_Malloc_And_Copy_Safely((void**)&d_atom_type, h_atom_type,
                                  sizeof(int) * atom_numbers);

    Device_Malloc_Safely((void**)&d_bo_s,
                         sizeof(float) * atom_numbers * atom_numbers);
    Device_Malloc_Safely((void**)&d_bo_p,
                         sizeof(float) * atom_numbers * atom_numbers);
    Device_Malloc_Safely((void**)&d_bo_p2,
                         sizeof(float) * atom_numbers * atom_numbers);
    Device_Malloc_Safely((void**)&d_total_bo_raw,
                         sizeof(float) * atom_numbers * atom_numbers);
    Device_Malloc_Safely((void**)&d_total_bond_order,
                         sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&d_total_corrected_bond_order,
                         sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&d_corrected_bo,
                         sizeof(float) * atom_numbers * atom_numbers);
    Device_Malloc_Safely((void**)&d_corrected_bo_s,
                         sizeof(float) * atom_numbers * atom_numbers);
    Device_Malloc_Safely((void**)&d_corrected_bo_pi,
                         sizeof(float) * atom_numbers * atom_numbers);
    Device_Malloc_Safely((void**)&d_corrected_bo_pi2,
                         sizeof(float) * atom_numbers * atom_numbers);

    Device_Malloc_Safely((void**)&d_dE_dBO_s,
                         sizeof(float) * atom_numbers * atom_numbers);
    Device_Malloc_Safely((void**)&d_dE_dBO_pi,
                         sizeof(float) * atom_numbers * atom_numbers);
    Device_Malloc_Safely((void**)&d_dE_dBO_pi2,
                         sizeof(float) * atom_numbers * atom_numbers);

    Device_Malloc_Safely((void**)&d_dbo_s_dr,
                         sizeof(float) * atom_numbers * atom_numbers);
    Device_Malloc_Safely((void**)&d_dbo_pi_dr,
                         sizeof(float) * atom_numbers * atom_numbers);
    Device_Malloc_Safely((void**)&d_dbo_pi2_dr,
                         sizeof(float) * atom_numbers * atom_numbers);
    Device_Malloc_Safely((void**)&d_dbo_s_dDelta_i,
                         sizeof(float) * atom_numbers * atom_numbers);
    Device_Malloc_Safely((void**)&d_dbo_pi_dDelta_i,
                         sizeof(float) * atom_numbers * atom_numbers);
    Device_Malloc_Safely((void**)&d_dbo_pi2_dDelta_i,
                         sizeof(float) * atom_numbers * atom_numbers);
    Device_Malloc_Safely((void**)&d_dbo_s_dDelta_j,
                         sizeof(float) * atom_numbers * atom_numbers);
    Device_Malloc_Safely((void**)&d_dbo_pi_dDelta_j,
                         sizeof(float) * atom_numbers * atom_numbers);
    Device_Malloc_Safely((void**)&d_dbo_pi2_dDelta_j,
                         sizeof(float) * atom_numbers * atom_numbers);
    Device_Malloc_Safely((void**)&d_dbo_raw_total_dr,
                         sizeof(float) * atom_numbers * atom_numbers);
    Device_Malloc_Safely((void**)&d_CdDelta_prime,
                         sizeof(float) * atom_numbers);

    int max_pairs = atom_numbers * (atom_numbers - 1) / 2;
    Device_Malloc_Safely((void**)&d_pair_i, sizeof(int) * max_pairs);
    Device_Malloc_Safely((void**)&d_pair_j, sizeof(int) * max_pairs);
    Device_Malloc_Safely((void**)&d_pair_distances, sizeof(float) * max_pairs);
    Device_Malloc_Safely((void**)&d_num_pairs_ptr, sizeof(int));

    is_initialized = 1;
    controller->printf("END INITIALIZING REAXFF_BOND_ORDER\n\n");
}

void REAXFF_BOND_ORDER::Calculate_Uncorrected_Bond_Orders_GPU(
    int atom_numbers, const VECTOR* d_crd, const LTMatrix3 cell,
    const LTMatrix3 rcell, float cutoff, int* d_pair_i, int* d_pair_j,
    float* d_distances, int* d_num_pairs_ptr)
{
    if (!is_initialized) return;

    dim3 blockSize = {CONTROLLER::device_max_thread};
    dim3 gridSize = {(atom_numbers + blockSize.x - 1) / blockSize.x};

    deviceMemset(d_total_bond_order, 0, sizeof(float) * atom_numbers);

    int h_num_pairs = 0;
    deviceMemcpy(d_num_pairs_ptr, &h_num_pairs, sizeof(int),
                 deviceMemcpyHostToDevice);

    Launch_Device_Kernel(
        Calculate_Uncorrected_Bond_Orders_Kernel, gridSize, blockSize, 0, NULL,
        atom_numbers, d_crd, cell, rcell, cutoff, d_atom_type, d_r_s, d_r_p,
        d_r_pp, d_bo_1, d_bo_2, d_bo_3, d_bo_4, d_bo_5, d_bo_6, d_ro_pi,
        d_ro_pi2, atom_type_numbers, gp_bo_cut, d_total_bond_order, d_pair_i,
        d_pair_j, d_distances, atom_numbers * atom_numbers, d_num_pairs_ptr);
}

void REAXFF_BOND_ORDER::Calculate_Corrected_Bond_Orders_GPU(
    int atom_numbers, const VECTOR* d_crd, const LTMatrix3 cell,
    const LTMatrix3 rcell, float cutoff, int num_pairs, int* d_pair_i,
    int* d_pair_j, float* d_distances)
{
    if (!is_initialized) return;

    if (num_pairs <= 0) return;

    dim3 blockSize = {CONTROLLER::device_max_thread};
    dim3 gridSize = {(num_pairs + blockSize.x - 1) / blockSize.x};

    Launch_Device_Kernel(
        Apply_Bond_Order_Corrections_Kernel, gridSize, blockSize, 0, NULL,
        num_pairs, d_pair_i, d_pair_j, d_distances, d_crd, cell, rcell,
        d_atom_type, d_r_s, d_r_p, d_r_pp, d_bo_1, d_bo_2, d_bo_3, d_bo_4,
        d_bo_5, d_bo_6, d_ro_pi, d_ro_pi2, d_valency, d_valency_val, d_ovc,
        d_v13cor, d_p_boc3, d_p_boc4, d_p_boc5, atom_type_numbers, atom_numbers,
        gp_boc1, gp_boc2, gp_bo_cut, d_total_bond_order, d_corrected_bo,
        d_corrected_bo_s, d_corrected_bo_pi, d_corrected_bo_pi2, d_dbo_s_dr,
        d_dbo_pi_dr, d_dbo_pi2_dr, d_dbo_s_dDelta_i, d_dbo_pi_dDelta_i,
        d_dbo_pi2_dDelta_i, d_dbo_s_dDelta_j, d_dbo_pi_dDelta_j,
        d_dbo_pi2_dDelta_j, d_dbo_raw_total_dr);
}

void REAXFF_BOND_ORDER::Calculate_Corrected_Bond_Order(
    int atom_numbers, const VECTOR* d_crd, const LTMatrix3 cell,
    const LTMatrix3 rcell, const ATOM_GROUP* fnl_d_nl, float cutoff)
{
    if (!is_initialized) return;

    if (h_atom_type == NULL)
    {
        printf(
            "ERROR: REAXFF_BOND_ORDER::Calculate_Corrected_Bond_Order - "
            "h_atom_type is NULL\n");
        return;
    }

    FILE* fp = fopen("reaxff_bond_order.txt", "w");
    if (!fp)
    {
        printf("ERROR: Failed to open reaxff_bond_order.txt\n");
        return;
    }

    deviceMemset(d_corrected_bo, 0,
                 sizeof(float) * atom_numbers * atom_numbers);
    deviceMemset(d_corrected_bo_s, 0,
                 sizeof(float) * atom_numbers * atom_numbers);
    deviceMemset(d_corrected_bo_pi, 0,
                 sizeof(float) * atom_numbers * atom_numbers);
    deviceMemset(d_corrected_bo_pi2, 0,
                 sizeof(float) * atom_numbers * atom_numbers);

    Calculate_Uncorrected_Bond_Orders_GPU(atom_numbers, d_crd, cell, rcell,
                                          cutoff, d_pair_i, d_pair_j,
                                          d_pair_distances, d_num_pairs_ptr);

    deviceMemcpy(&h_num_pairs, d_num_pairs_ptr, sizeof(int),
                 deviceMemcpyDeviceToHost);
    int num_pairs = h_num_pairs;

    if (num_pairs > 0)
    {
        Calculate_Corrected_Bond_Orders_GPU(atom_numbers, d_crd, cell, rcell,
                                            cutoff, num_pairs, d_pair_i,
                                            d_pair_j, d_pair_distances);

        dim3 blockSize = {CONTROLLER::device_max_thread};
        dim3 gridSize = {(atom_numbers + blockSize.x - 1) / blockSize.x};
        deviceMemset(d_total_corrected_bond_order, 0,
                     sizeof(float) * atom_numbers);
        Launch_Device_Kernel(Reduce_Total_Corrected_Bond_Order_Kernel, gridSize,
                             blockSize, 0, NULL, atom_numbers, d_corrected_bo_s,
                             d_corrected_bo_pi, d_corrected_bo_pi2,
                             d_total_corrected_bond_order);

        int* h_pair_i = (int*)malloc(sizeof(int) * num_pairs);
        int* h_pair_j = (int*)malloc(sizeof(int) * num_pairs);
        float* h_corrected_bo = (float*)malloc(sizeof(float) * num_pairs);

        deviceMemcpy(h_pair_i, d_pair_i, sizeof(int) * num_pairs,
                     deviceMemcpyDeviceToHost);
        deviceMemcpy(h_pair_j, d_pair_j, sizeof(int) * num_pairs,
                     deviceMemcpyDeviceToHost);
        deviceMemcpy(h_corrected_bo, d_corrected_bo, sizeof(float) * num_pairs,
                     deviceMemcpyDeviceToHost);

        for (int idx = 0; idx < num_pairs; idx++)
        {
            int i = h_pair_i[idx];
            int j = h_pair_j[idx];
            float corrected_bond_order = h_corrected_bo[idx];

            if (corrected_bond_order > 1e-10f)
            {
                fprintf(fp, "%d %d %f\n", i + 1, j + 1, corrected_bond_order);
            }
        }

        free(h_pair_i);
        free(h_pair_j);
        free(h_corrected_bo);
    }

    fclose(fp);
}

static __global__ void Calculate_CdDelta_Prime_Kernel(
    int num_pairs, const int* pair_i, const int* pair_j, const int atom_numbers,
    const float* dE_dBO_s, const float* dE_dBO_pi, const float* dE_dBO_pi2,
    const float* CdDelta, const float* dbo_s_dDelta_i,
    const float* dbo_pi_dDelta_i, const float* dbo_pi2_dDelta_i,
    const float* dbo_s_dDelta_j, const float* dbo_pi_dDelta_j,
    const float* dbo_pi2_dDelta_j, float* CdDelta_prime)
{
    SIMPLE_DEVICE_FOR(idx, num_pairs)
    {
        int i = pair_i[idx];
        int j = pair_j[idx];
        int dense_idx = i * atom_numbers + j;
        int dense_idx_ji = j * atom_numbers + i;

        float de_dbo_s_total = dE_dBO_s[dense_idx] + dE_dBO_s[dense_idx_ji];
        float de_dbo_pi_total = dE_dBO_pi[dense_idx] + dE_dBO_pi[dense_idx_ji];
        float de_dbo_pi2_total =
            dE_dBO_pi2[dense_idx] + dE_dBO_pi2[dense_idx_ji];

        float term_i = (de_dbo_s_total + CdDelta[i] + CdDelta[j]) *
                           dbo_s_dDelta_i[dense_idx] +
                       (de_dbo_pi_total + CdDelta[i] + CdDelta[j]) *
                           dbo_pi_dDelta_i[dense_idx] +
                       (de_dbo_pi2_total + CdDelta[i] + CdDelta[j]) *
                           dbo_pi2_dDelta_i[dense_idx];
        atomicAdd(&CdDelta_prime[i], term_i);

        float term_j = (de_dbo_s_total + CdDelta[j] + CdDelta[i]) *
                           dbo_s_dDelta_j[dense_idx] +
                       (de_dbo_pi_total + CdDelta[j] + CdDelta[i]) *
                           dbo_pi_dDelta_j[dense_idx] +
                       (de_dbo_pi2_total + CdDelta[j] + CdDelta[i]) *
                           dbo_pi2_dDelta_j[dense_idx];
        atomicAdd(&CdDelta_prime[j], term_j);
    }
}

static __global__ void REAXFF_Force_Projection_Kernel(
    int num_pairs, const int* pair_i, const int* pair_j, const float* distances,
    const VECTOR* crd, const LTMatrix3 cell, const LTMatrix3 rcell,
    const int atom_numbers, const float* dE_dBO_s, const float* dE_dBO_pi,
    const float* dE_dBO_pi2, const float* CdDelta, const float* dbo_s_dr,
    const float* dbo_pi_dr, const float* dbo_pi2_dr,
    const float* dbo_raw_total_dr, const float* CdDelta_prime, VECTOR* frc,
    LTMatrix3* atom_virial)
{
    SIMPLE_DEVICE_FOR(idx, num_pairs)
    {
        int i = pair_i[idx];
        int j = pair_j[idx];
        float r_val = distances[idx];
        if (r_val >= 0.0001f)
        {
            int dense_idx = i * atom_numbers + j;
            int dense_idx_ji = j * atom_numbers + i;

            float de_dbo_s_total = dE_dBO_s[dense_idx] + dE_dBO_s[dense_idx_ji];
            float de_dbo_pi_total =
                dE_dBO_pi[dense_idx] + dE_dBO_pi[dense_idx_ji];
            float de_dbo_pi2_total =
                dE_dBO_pi2[dense_idx] + dE_dBO_pi2[dense_idx_ji];

            float eff_cdd = CdDelta[i] + CdDelta[j];

            float de_dr = (de_dbo_s_total + eff_cdd) * dbo_s_dr[dense_idx] +
                          (de_dbo_pi_total + eff_cdd) * dbo_pi_dr[dense_idx] +
                          (de_dbo_pi2_total + eff_cdd) * dbo_pi2_dr[dense_idx];

            de_dr += (CdDelta_prime[i] + CdDelta_prime[j]) *
                     dbo_raw_total_dr[dense_idx];

            float force_mag = -de_dr;

            VECTOR ri = crd[i];
            VECTOR rj = crd[j];
            VECTOR drij = Get_Periodic_Displacement(ri, rj, cell, rcell);

            float fx = force_mag * drij.x / r_val;
            float fy = force_mag * drij.y / r_val;
            float fz = force_mag * drij.z / r_val;

            atomicAdd(&frc[i].x, fx);
            atomicAdd(&frc[i].y, fy);
            atomicAdd(&frc[i].z, fz);
            atomicAdd(&frc[j].x, -fx);
            atomicAdd(&frc[j].y, -fy);
            atomicAdd(&frc[j].z, -fz);

            if (atom_virial)
            {
                VECTOR fij = {fx, fy, fz};
                atomicAdd(atom_virial + i,
                          Get_Virial_From_Force_Dis(fij, drij));
            }
        }
    }
}

void REAXFF_BOND_ORDER::Calculate_Forces(int atom_numbers, const VECTOR* d_crd,
                                         VECTOR* d_frc, const LTMatrix3 cell,
                                         const LTMatrix3 rcell, float cutoff,
                                         float* d_CdDelta, int need_virial,
                                         LTMatrix3* atom_virial)
{
    if (!is_initialized || h_num_pairs <= 0) return;

    dim3 blockSize = {CONTROLLER::device_max_thread};
    dim3 gridSize = {(h_num_pairs + blockSize.x - 1) / blockSize.x};

    Launch_Device_Kernel(
        Calculate_CdDelta_Prime_Kernel, gridSize, blockSize, 0, NULL,
        h_num_pairs, d_pair_i, d_pair_j, atom_numbers, d_dE_dBO_s, d_dE_dBO_pi,
        d_dE_dBO_pi2, d_CdDelta, d_dbo_s_dDelta_i, d_dbo_pi_dDelta_i,
        d_dbo_pi2_dDelta_i, d_dbo_s_dDelta_j, d_dbo_pi_dDelta_j,
        d_dbo_pi2_dDelta_j, d_CdDelta_prime);

    Launch_Device_Kernel(
        REAXFF_Force_Projection_Kernel, gridSize, blockSize, 0, NULL,
        h_num_pairs, d_pair_i, d_pair_j, d_pair_distances, d_crd, cell, rcell,
        atom_numbers, d_dE_dBO_s, d_dE_dBO_pi, d_dE_dBO_pi2, d_CdDelta,
        d_dbo_s_dr, d_dbo_pi_dr, d_dbo_pi2_dr, d_dbo_raw_total_dr,
        d_CdDelta_prime, d_frc, need_virial ? atom_virial : NULL);
}

void REAXFF_BOND_ORDER::Clear_Derivatives(int atom_numbers, float* d_CdDelta)
{
    if (!is_initialized) return;
    deviceMemset(d_dE_dBO_s, 0, sizeof(float) * atom_numbers * atom_numbers);
    deviceMemset(d_dE_dBO_pi, 0, sizeof(float) * atom_numbers * atom_numbers);
    deviceMemset(d_dE_dBO_pi2, 0, sizeof(float) * atom_numbers * atom_numbers);
    if (d_CdDelta)
    {
        deviceMemset(d_CdDelta, 0, sizeof(float) * atom_numbers);
    }
    deviceMemset(d_CdDelta_prime, 0, sizeof(float) * atom_numbers);
}

void REAXFF_BOND_ORDER::Calculate_Bond_Order(
    int atom_numbers, const VECTOR* d_crd, const LTMatrix3 cell,
    const LTMatrix3 rcell, const ATOM_GROUP* fnl_d_nl, float cutoff)
{
    Calculate_Corrected_Bond_Order(atom_numbers, d_crd, cell, rcell, fnl_d_nl,
                                   cutoff);
}

#include "valence_angle.h"

#include "bond_order.h"  // for find_bond_index

static __global__ void Calculate_Valence_Angle_Kernel(
    int atom_numbers, const VECTOR* crd, const int* atom_type,
    const float* Delta_boc, const float* Delta, const float* Delta_val,
    const float* p_val3, const float* p_val5,
    const REAXFF_VALENCE_ANGLE_PARAMS p, const REAXFF_THBP_Info* thbp_info,
    const REAXFF_THBP_Entry* thbp_entries, int atom_type_numbers,
    const float* bo_s, const float* bo_pi, const float* bo_pi2,
    const float* total_bo, const float* nlp, const float* vlpex,
    const float* dDelta_lp, const LTMatrix3 cell, const LTMatrix3 rcell,
    float* d_dE_dBO_s, float* d_dE_dBO_pi, float* d_dE_dBO_pi2, float* CdDelta,
    float* atom_energy, VECTOR* frc, LTMatrix3* atom_virial,
    float* d_energy_ang_sum, float* d_energy_pen_sum, float* d_energy_coa_sum,
    const int* bond_count, const int* bond_offset, const int* bond_nbr,
    const int* bond_idx_arr)
{
    SIMPLE_DEVICE_FOR(j, atom_numbers)
    {
        int type_j = atom_type[j];
        if (type_j >= 0)
        {
            VECTOR rj = crd[j];

            float p_val3_j = p_val3[type_j];
            float p_val5_j = p_val5[type_j];
            float delta_boc_j_val = Delta_boc[j];
            float delta_j_val = Delta[j];
            float delta_val_j_val = Delta_val[j];

            float SBOp = 0, prod_SBO = 1.0f;
            int bc_j = bond_count[j];
            int bo_j = bond_offset[j];
            for (int t = 0; t < bc_j; t++)
            {
                int b_jt = bond_idx_arr[bo_j + t];
                float bo_jt = bo_s[b_jt] + bo_pi[b_jt] + bo_pi2[b_jt];

                SBOp += (bo_pi[b_jt] + bo_pi2[b_jt]);
                float bo_jt_sq = bo_jt * bo_jt;
                float bo_jt_p4 = bo_jt_sq * bo_jt_sq;
                prod_SBO *= expf(-bo_jt_p4 * bo_jt_p4);
            }

            bool has_lp_corr = (vlpex[j] < 0.0f);
            float vlpadj = has_lp_corr ? nlp[j] : 0.0f;
            float SBO = SBOp + (1.0f - prod_SBO) *
                                   (-delta_boc_j_val - p.p_val8 * vlpadj);
            float dSBO1 =
                -8.0f * prod_SBO * (delta_boc_j_val + p.p_val8 * vlpadj);
            float dSBO2 = has_lp_corr ? (prod_SBO - 1.0f) *
                                            (1.0f - p.p_val8 * dDelta_lp[j])
                                      : (prod_SBO - 1.0f);
            float SBO2;
            float CSBO2;
            if (SBO <= 0.0f)
            {
                SBO2 = 0.0f;
                CSBO2 = 0.0f;
            }
            else if (SBO <= 1.0f)
            {
                SBO2 = powf(SBO, p.p_val9);
                CSBO2 = p.p_val9 * powf(SBO, p.p_val9 - 1.0f);
            }
            else if (SBO < 2.0f)
            {
                SBO2 = 2.0f - powf(2.0f - SBO, p.p_val9);
                CSBO2 = p.p_val9 * powf(2.0f - SBO, p.p_val9 - 1.0f);
            }
            else
            {
                SBO2 = 2.0f;
                CSBO2 = 0.0f;
            }

            for (int bi = 0; bi < bc_j; bi++)
            {
                int b_ij = bond_idx_arr[bo_j + bi];
                int i = bond_nbr[bo_j + bi];
                int type_i = atom_type[i];
                float bo_ij_val = bo_s[b_ij] + bo_pi[b_ij] + bo_pi2[b_ij];
                float boa_ij_val = bo_ij_val - p.thb_cut;

                if (boa_ij_val <= 0) continue;

                VECTOR ri = crd[i];
                VECTOR dji = Get_Periodic_Displacement(rj, ri, cell, rcell);
                float r_ij = norm3df(dji.x, dji.y, dji.z);

                for (int bk = bi + 1; bk < bc_j; bk++)
                {
                    int b_kj = bond_idx_arr[bo_j + bk];
                    int k = bond_nbr[bo_j + bk];
                    int type_k = atom_type[k];
                    float bo_jk_val = bo_s[b_kj] + bo_pi[b_kj] + bo_pi2[b_kj];
                    float boa_jk_val = bo_jk_val - p.thb_cut;

                    if (boa_jk_val <= 0) continue;
                    if (bo_ij_val * bo_jk_val <= p.thb_cutsq) continue;

                    VECTOR rk = crd[k];
                    VECTOR djk = Get_Periodic_Displacement(rj, rk, cell, rcell);
                    float r_jk = norm3df(djk.x, djk.y, djk.z);

                    float cos_theta =
                        (dji.x * djk.x + dji.y * djk.y + dji.z * djk.z) /
                        (r_ij * r_jk);
                    if (cos_theta > 1.0f) cos_theta = 1.0f;
                    if (cos_theta < -1.0f) cos_theta = -1.0f;
                    float theta = acosf(cos_theta);

                    int tri_info_idx = (type_i * atom_type_numbers + type_j) *
                                           atom_type_numbers +
                                       type_k;
                    REAXFF_THBP_Info info = thbp_info[tri_info_idx];

                    for (int e = 0; e < info.entry_count; e++)
                    {
                        const REAXFF_THBP_Entry* param =
                            &thbp_entries[info.start_idx + e];
                        if (fabsf(param->p_val1) < 0.0001f) continue;

                        SADfloat<3> s_bo_ij(bo_ij_val, 0);
                        SADfloat<3> s_bo_jk(bo_jk_val, 1);
                        SADfloat<3> s_delta_j(delta_boc_j_val, 2);
                        SADfloat<3> s_delta_pen =
                            s_delta_j + (delta_j_val - delta_boc_j_val);

                        SADfloat<3> s_boa_ij = s_bo_ij - p.thb_cut;
                        SADfloat<3> s_boa_jk = s_bo_jk - p.thb_cut;

                        SADfloat<3> exp3ij = expf(
                            -p_val3_j * powf(s_boa_ij, (float)param->p_val4));
                        SADfloat<3> f7_ij = 1.0f - exp3ij;
                        SADfloat<3> exp3jk = expf(
                            -p_val3_j * powf(s_boa_jk, (float)param->p_val4));
                        SADfloat<3> f7_jk = 1.0f - exp3jk;

                        SADfloat<3> expval6 = expf(p.p_val6 * s_delta_j);
                        SADfloat<3> expval7 = expf(-param->p_val7 * s_delta_j);
                        SADfloat<3> trm8 = 1.0f + expval6 + expval7;
                        SADfloat<3> f8_Dj =
                            p_val5_j -
                            ((p_val5_j - 1.0f) * (2.0f + expval6) / trm8);

                        float theta_0 =
                            (180.0f -
                             param->theta_00 *
                                 (1.0f - expf(-p.p_val10 * (2.0f - SBO2)))) *
                            (CONSTANT_Pi / 180.0f);
                        float sin_theta = sinf(theta);
                        if (sin_theta < 1e-5f) sin_theta = 1e-5f;

                        float expval2theta =
                            expf(-param->p_val2 * (theta_0 - theta) *
                                 (theta_0 - theta));
                        float val12theta =
                            (param->p_val1 >= 0)
                                ? param->p_val1 * (1.0f - expval2theta)
                                : param->p_val1 * (-expval2theta);

                        SADfloat<3> s_en_ang =
                            f7_ij * f7_jk * f8_Dj * val12theta;

                        SADfloat<3> s_pen_diff_ij = s_boa_ij - 2.0f;
                        SADfloat<3> s_pen_diff_jk = s_boa_jk - 2.0f;
                        SADfloat<3> exp_pen2ij =
                            expf(-p.p_pen2 * s_pen_diff_ij * s_pen_diff_ij);
                        SADfloat<3> exp_pen2jk =
                            expf(-p.p_pen2 * s_pen_diff_jk * s_pen_diff_jk);

                        SADfloat<3> exp_pen3 = expf(-p.p_pen3 * s_delta_pen);
                        SADfloat<3> exp_pen4 = expf(p.p_pen4 * s_delta_pen);
                        SADfloat<3> trm_pen34 = 1.0f + exp_pen3 + exp_pen4;
                        SADfloat<3> f9_Dj = (2.0f + exp_pen3) / trm_pen34;
                        SADfloat<3> s_en_pen =
                            param->p_pen1 * f9_Dj * exp_pen2ij * exp_pen2jk;

                        SADfloat<3> exp_coa2 =
                            expf(p.p_coa2 * (s_delta_j + delta_val_j_val -
                                             delta_boc_j_val));
                        SADfloat<3> s_coa_diff_i = total_bo[i] - bo_ij_val;
                        SADfloat<3> s_coa_diff_k = total_bo[k] - bo_jk_val;
                        SADfloat<3> s_coa_diff_ij = s_boa_ij - 1.5f + p.thb_cut;
                        SADfloat<3> s_coa_diff_jk = s_boa_jk - 1.5f + p.thb_cut;

                        SADfloat<3> s_en_coa =
                            param->p_coa1 / (1.0f + exp_coa2) *
                            expf(-p.p_coa3 * s_coa_diff_i * s_coa_diff_i) *
                            expf(-p.p_coa3 * s_coa_diff_k * s_coa_diff_k) *
                            expf(-p.p_coa4 * s_coa_diff_ij * s_coa_diff_ij) *
                            expf(-p.p_coa4 * s_coa_diff_jk * s_coa_diff_jk);

                        SADfloat<3> s_en_total = s_en_ang + s_en_pen + s_en_coa;

                        atomicAdd(&d_dE_dBO_s[b_ij], s_en_total.dval[0]);
                        atomicAdd(&d_dE_dBO_pi[b_ij], s_en_total.dval[0]);
                        atomicAdd(&d_dE_dBO_pi2[b_ij], s_en_total.dval[0]);

                        atomicAdd(&d_dE_dBO_s[b_kj], s_en_total.dval[1]);
                        atomicAdd(&d_dE_dBO_pi[b_kj], s_en_total.dval[1]);
                        atomicAdd(&d_dE_dBO_pi2[b_kj], s_en_total.dval[1]);

                        atomicAdd(&CdDelta[j], s_en_total.dval[2]);

                        float dE_dtheta = -f7_ij.val * f7_jk.val * f8_Dj.val *
                                          param->p_val1 * expval2theta * 2.0f *
                                          param->p_val2 * (theta_0 - theta);
                        float Ctheta_0 = p.p_val10 * (CONSTANT_Pi / 180.0f) *
                                         param->theta_00 *
                                         expf(-p.p_val10 * (2.0f - SBO2));
                        float CEval5 = -dE_dtheta * Ctheta_0 * CSBO2;
                        float CEval6 = CEval5 * dSBO1;
                        float CEval7 = CEval5 * dSBO2;
                        atomicAdd(&CdDelta[j], CEval7);

                        for (int pt = 0; pt < bc_j; pt++)
                        {
                            int b_jt = bond_idx_arr[bo_j + pt];
                            float bo_jt =
                                bo_s[b_jt] + bo_pi[b_jt] + bo_pi2[b_jt];
                            if (bo_jt <= 0.0f) continue;
                            float bo_jt_2 = bo_jt * bo_jt;
                            float bo_jt_4 = bo_jt_2 * bo_jt_2;
                            float bo_jt_7 = bo_jt_4 * bo_jt_2 * bo_jt;
                            float dE_dbo_total = CEval6 * bo_jt_7;

                            atomicAdd(&d_dE_dBO_s[b_jt], dE_dbo_total);
                            atomicAdd(&d_dE_dBO_pi[b_jt],
                                      dE_dbo_total + CEval5);
                            atomicAdd(&d_dE_dBO_pi2[b_jt],
                                      dE_dbo_total + CEval5);
                        }

                        float dcos_dtheta = -sin_theta;
                        float dE_dcos = dE_dtheta / dcos_dtheta;

                        float inv_rij = 1.0f / r_ij;
                        float inv_rjk = 1.0f / r_jk;

                        VECTOR fi, fk, fj;
                        fi.x =
                            dE_dcos * (djk.x * inv_rij * inv_rjk -
                                       dji.x * cos_theta * inv_rij * inv_rij);
                        fi.y =
                            dE_dcos * (djk.y * inv_rij * inv_rjk -
                                       dji.y * cos_theta * inv_rij * inv_rij);
                        fi.z =
                            dE_dcos * (djk.z * inv_rij * inv_rjk -
                                       dji.z * cos_theta * inv_rij * inv_rij);

                        fk.x =
                            dE_dcos * (dji.x * inv_rij * inv_rjk -
                                       djk.x * cos_theta * inv_rjk * inv_rjk);
                        fk.y =
                            dE_dcos * (dji.y * inv_rij * inv_rjk -
                                       djk.y * cos_theta * inv_rjk * inv_rjk);
                        fk.z =
                            dE_dcos * (dji.z * inv_rij * inv_rjk -
                                       djk.z * cos_theta * inv_rjk * inv_rjk);

                        fj.x = -(fi.x + fk.x);
                        fj.y = -(fi.y + fk.y);
                        fj.z = -(fi.z + fk.z);

                        atomicAdd(&frc[i].x, fi.x);
                        atomicAdd(&frc[i].y, fi.y);
                        atomicAdd(&frc[i].z, fi.z);
                        atomicAdd(&frc[j].x, fj.x);
                        atomicAdd(&frc[j].y, fj.y);
                        atomicAdd(&frc[j].z, fj.z);
                        atomicAdd(&frc[k].x, fk.x);
                        atomicAdd(&frc[k].y, fk.y);
                        atomicAdd(&frc[k].z, fk.z);
                        if (atom_virial)
                        {
                            LTMatrix3 v = {0, 0, 0, 0, 0, 0};
                            v = v - Get_Virial_From_Force_Dis(fi, dji) -
                                Get_Virial_From_Force_Dis(fk, djk);
                            atomicAdd(atom_virial + j, v);
                        }

                        if (atom_energy)
                        {
                            float en_total = s_en_total.val;
                            atomicAdd(&atom_energy[j], en_total);
                            atomicAdd(d_energy_ang_sum, s_en_ang.val);
                            atomicAdd(d_energy_pen_sum, s_en_pen.val);
                            atomicAdd(d_energy_coa_sum, s_en_coa.val);
                        }
                    }
                }
            }
        }
    }
}

void REAXFF_VALENCE_ANGLE::Initial(CONTROLLER* controller, int atom_numbers,
                                   const char* module_name)
{
    if (module_name == NULL) module_name = "REAXFF";
    this->atom_numbers = atom_numbers;

    if (!controller->Command_Exist(module_name, "in_file")) return;

    controller->printf("START INITIALIZING REAXFF VALENCE ANGLE\n");

    const char* parameter_in_file = controller->Command(module_name, "in_file");
    const char* type_in_file = controller->Command(module_name, "type_in_file");
    if (parameter_in_file == NULL || type_in_file == NULL)
    {
        controller->printf(
            "REAXFF_VALENCE_ANGLE IS NOT INITIALIZED (missing input "
            "files)\n\n");
        return;
    }
    controller->Step_Print_Initial("REAXFF_ANG", "%14.7e");
    controller->Step_Print_Initial("REAXFF_PEN", "%14.7e");
    controller->Step_Print_Initial("REAXFF_COA", "%14.7e");

    FILE* fp;
    Open_File_Safely(&fp, parameter_in_file, "r");
    char line[1024];
    auto throw_bad_format = [&](const char* file_name, const char* reason)
    {
        char error_msg[1024];
        sprintf(error_msg, "Reason:\n\t%s in file %s\n", reason, file_name);
        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                       "REAXFF_VALENCE_ANGLE::Initial",
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
    if (sscanf(line, "%d", &n_gp) != 1 || n_gp <= 0)
    {
        throw_bad_format(parameter_in_file,
                         "failed to parse number of general parameters");
    }
    if (n_gp <= 38)
    {
        throw_bad_format(
            parameter_in_file,
            "general parameter count is too small for valence angle");
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
    params.p_coa2 = gp[2];
    params.p_val6 = gp[14];
    params.p_val9 = gp[16];
    params.p_val10 = gp[17];
    params.p_pen2 = gp[19];
    params.p_pen3 = gp[20];
    params.p_pen4 = gp[21];
    params.p_coa4 = gp[30];
    params.p_val8 = gp[33];
    params.p_coa3 = gp[38];
    // thb_cutoff means the three-body screening cutoff used by angle/torsion
    // style terms. It is a control parameter, not the bond-order cutoff in
    // the force-field file.
    params.thb_cut = 0.001f;
    if (controller->Command_Exist(module_name, "thb_cutoff"))
    {
        params.thb_cut = atof(controller->Command(module_name, "thb_cutoff"));
    }
    params.thb_cutsq = params.thb_cut * params.thb_cut;

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

    Malloc_Safely((void**)&h_p_val3, sizeof(float) * n_atom_types);
    Malloc_Safely((void**)&h_p_val5, sizeof(float) * n_atom_types);
    Malloc_Safely((void**)&h_mass, sizeof(float) * n_atom_types);
    Malloc_Safely((void**)&h_valency_boc, sizeof(float) * n_atom_types);

    std::map<std::string, int> type_map;
    for (int i = 0; i < n_atom_types; i++)
    {
        read_line_or_throw(fp, parameter_in_file, "atom type block line 1");
        char name[16];
        float m;
        if (sscanf(line, "%15s %*f %*f %f %*f %*f %*f %*f %*f", name, &m) != 2)
        {
            char reason[512];
            sprintf(reason,
                    "failed to parse atom type block line 1 at index %d",
                    i + 1);
            throw_bad_format(parameter_in_file, reason);
        }
        h_mass[i] = m;
        type_map[name] = i;
        read_line_or_throw(fp, parameter_in_file, "atom type block line 2");
        if (sscanf(line, "%*f %*f %f", &h_valency_boc[i]) != 1)
        {
            char reason[512];
            sprintf(reason,
                    "failed to parse atom type block line 2 at index %d",
                    i + 1);
            throw_bad_format(parameter_in_file, reason);
        }
        read_line_or_throw(fp, parameter_in_file, "atom type block line 3");
        read_line_or_throw(fp, parameter_in_file, "atom type block line 4");
        if (sscanf(line, "%*f %f %*f %*f %f", &h_p_val3[i], &h_p_val5[i]) != 2)
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
    read_line_or_throw(fp, parameter_in_file, "bond parameter header line");
    for (int i = 0; i < n_bond_params; i++)
    {
        read_line_or_throw(fp, parameter_in_file,
                           "bond parameter block line 1");
        read_line_or_throw(fp, parameter_in_file,
                           "bond parameter block line 2");
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

    read_line_or_throw(fp, parameter_in_file,
                       "three-body parameter count line");
    int n_thb = 0;
    if (sscanf(line, "%d", &n_thb) != 1 || n_thb < 0)
    {
        throw_bad_format(parameter_in_file,
                         "failed to parse number of three-body parameters");
    }

    std::vector<REAXFF_THBP_Entry> all_entries;
    std::map<int, std::vector<int>> triplet_to_entries;

    for (int i = 0; i < n_thb; i++)
    {
        read_line_or_throw(fp, parameter_in_file, "three-body parameter entry");
        int t1, t2, t3;
        float th0, pv1, pv2, pcoa1, pv7, ppen1, pv4;
        if (sscanf(line, "%d %d %d %f %f %f %f %f %f %f", &t1, &t2, &t3, &th0,
                   &pv1, &pv2, &pcoa1, &pv7, &ppen1, &pv4) != 10)
        {
            char reason[512];
            sprintf(reason,
                    "failed to parse three-body parameter entry at index %d",
                    i + 1);
            throw_bad_format(parameter_in_file, reason);
        }
        int idx1 = t1 - 1;
        int idx2 = t2 - 1;
        int idx3 = t3 - 1;
        if (idx1 >= 0 && idx1 < n_atom_types && idx2 >= 0 &&
            idx2 < n_atom_types && idx3 >= 0 && idx3 < n_atom_types)
        {
            REAXFF_THBP_Entry entry;
            entry.theta_00 = th0;
            entry.p_val1 = pv1;
            entry.p_val2 = pv2;
            entry.p_coa1 = pcoa1;
            entry.p_val7 = pv7;
            entry.p_pen1 = ppen1;
            entry.p_val4 = pv4;

            int entry_idx = all_entries.size();
            all_entries.push_back(entry);

            int tri_idx = (idx1 * n_atom_types + idx2) * n_atom_types + idx3;
            triplet_to_entries[tri_idx].push_back(entry_idx);

            if (idx1 != idx3)
            {
                int tri_idx_rev =
                    (idx3 * n_atom_types + idx2) * n_atom_types + idx1;
                triplet_to_entries[tri_idx_rev].push_back(entry_idx);
            }
        }
    }
    fclose(fp);

    Malloc_Safely(
        (void**)&h_thbp_info,
        sizeof(REAXFF_THBP_Info) * n_atom_types * n_atom_types * n_atom_types);
    memset(
        h_thbp_info, 0,
        sizeof(REAXFF_THBP_Info) * n_atom_types * n_atom_types * n_atom_types);

    std::vector<REAXFF_THBP_Entry> sorted_entries;
    for (int i = 0; i < n_atom_types * n_atom_types * n_atom_types; i++)
    {
        if (triplet_to_entries.count(i))
        {
            h_thbp_info[i].start_idx = sorted_entries.size();
            h_thbp_info[i].entry_count = triplet_to_entries[i].size();
            for (int entry_idx : triplet_to_entries[i])
            {
                sorted_entries.push_back(all_entries[entry_idx]);
            }
        }
    }

    Malloc_Safely((void**)&h_thbp_entries,
                  sizeof(REAXFF_THBP_Entry) * sorted_entries.size());
    for (size_t i = 0; i < sorted_entries.size(); i++)
        h_thbp_entries[i] = sorted_entries[i];

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
    Device_Malloc_And_Copy_Safely((void**)&d_p_val3, h_p_val3,
                                  sizeof(float) * n_atom_types);
    Device_Malloc_And_Copy_Safely((void**)&d_p_val5, h_p_val5,
                                  sizeof(float) * n_atom_types);
    Device_Malloc_And_Copy_Safely((void**)&d_mass, h_mass,
                                  sizeof(float) * n_atom_types);
    Device_Malloc_And_Copy_Safely((void**)&d_valency_boc, h_valency_boc,
                                  sizeof(float) * n_atom_types);
    Device_Malloc_And_Copy_Safely(
        (void**)&d_thbp_info, h_thbp_info,
        sizeof(REAXFF_THBP_Info) * n_atom_types * n_atom_types * n_atom_types);
    Device_Malloc_And_Copy_Safely(
        (void**)&d_thbp_entries, h_thbp_entries,
        sizeof(REAXFF_THBP_Entry) * sorted_entries.size());

    Device_Malloc_Safely((void**)&d_energy_ang_sum, sizeof(float));
    Device_Malloc_Safely((void**)&d_energy_pen_sum, sizeof(float));
    Device_Malloc_Safely((void**)&d_energy_coa_sum, sizeof(float));

    is_initialized = 1;
    controller->printf("END INITIALIZING REAXFF VALENCE ANGLE\n\n");
}

void REAXFF_VALENCE_ANGLE::Calculate_Valence_Angle_Energy_And_Force(
    int atom_numbers, const VECTOR* crd, VECTOR* frc, const LTMatrix3 cell,
    const LTMatrix3 rcell, const ATOM_GROUP* nl, REAXFF_BOND_ORDER* bo_module,
    const float* Delta, const float* Delta_boc, const float* Delta_val,
    const float* nlp, const float* vlpex, const float* dDelta_lp,
    float* CdDelta, const int need_atom_energy, float* atom_energy,
    const int need_virial, LTMatrix3* atom_virial)
{
    if (!is_initialized) return;

    dim3 blockSize(32);
    dim3 gridSize((atom_numbers + blockSize.x - 1) / blockSize.x);

    deviceMemset(d_energy_ang_sum, 0, sizeof(float));
    deviceMemset(d_energy_pen_sum, 0, sizeof(float));
    deviceMemset(d_energy_coa_sum, 0, sizeof(float));

    Launch_Device_Kernel(
        Calculate_Valence_Angle_Kernel, gridSize, blockSize, 0, NULL,
        atom_numbers, crd, d_atom_type, Delta_boc, Delta, Delta_val, d_p_val3,
        d_p_val5, params, d_thbp_info, d_thbp_entries, atom_type_numbers,
        bo_module->d_corrected_bo_s, bo_module->d_corrected_bo_pi,
        bo_module->d_corrected_bo_pi2, bo_module->d_total_corrected_bond_order,
        nlp, vlpex, dDelta_lp, cell, rcell, d_dE_dBO_s, d_dE_dBO_pi,
        d_dE_dBO_pi2, CdDelta, need_atom_energy ? atom_energy : NULL, frc,
        need_virial ? atom_virial : NULL, d_energy_ang_sum, d_energy_pen_sum,
        d_energy_coa_sum, bo_module->d_bond_count, bo_module->d_bond_offset,
        bo_module->d_bond_nbr, bo_module->d_bond_idx);

#ifdef USE_GPU
    deviceError_t err = deviceGetLastError();
    if (err != 0)
    {
        printf("Device Error in Calculate_Valence_Angle_Kernel: %s\n",
               deviceGetErrorString(err));
    }
#endif
}

void REAXFF_VALENCE_ANGLE::Step_Print(CONTROLLER* controller)
{
    if (!is_initialized) return;
    deviceMemcpy(&h_energy_ang, d_energy_ang_sum, sizeof(float),
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(&h_energy_pen, d_energy_pen_sum, sizeof(float),
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(&h_energy_coa, d_energy_coa_sum, sizeof(float),
                 deviceMemcpyDeviceToHost);
    controller->Step_Print("REAXFF_ANG", h_energy_ang, true);
    controller->Step_Print("REAXFF_PEN", h_energy_pen, true);
    controller->Step_Print("REAXFF_COA", h_energy_coa, true);
}

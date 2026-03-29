#include "sap.h"

#include "../quantum_chemistry.h"

// ====================== SAP 拟合参数 (sap_helfem_large) ======================
// 来源: Psi4 / Basis Set Exchange, S. Lehtola 用 HelFEM 计算
// 原子势展开为: V(r) = -Z_eff(r)/r = -(Z + Σ c_k erf(√α_k r))/r
// 每个 SAP_TERM 存储一组 (alpha, coeff)
// =============================================================================

struct SAP_TERM
{
    float alpha;
    float coeff;
};

struct SAP_ATOM_DATA
{
    int n_terms;
    SAP_TERM terms[16];
};

// clang-format off
static const SAP_ATOM_DATA SAP_DATA[] = {
    // Z=0 dummy
    {0, {}},
    // Z=1  H (4 terms)
    {4, {
        {3.214199700e-01f, -1.509766448e+00f},
        {1.639897806e-01f,  1.524537908e+00f},
        {8.366825543e-02f, -5.962131922e-01f},
        {4.268788542e-02f, -4.185582672e-01f},
    }},
    // Z=2  He (5 terms)
    {5, {
        {1.234766957e+00f, -2.692215072e+00f},
        {6.299831413e-01f,  3.577647758e+00f},
        {3.214199700e-01f, -3.585641910e+00f},
        {1.639897806e-01f,  2.055414703e+00f},
        {8.366825543e-02f, -1.355205480e+00f},
    }},
    // Z=3  Li (9 terms)
    {9, {
        {2.420143236e+00f, -3.852087393e+00f},
        {1.234766957e+00f,  7.028106110e+00f},
        {6.299831413e-01f, -9.186191175e+00f},
        {3.214199700e-01f,  7.556086713e+00f},
        {1.639897806e-01f, -5.862783425e+00f},
        {8.366825543e-02f,  3.144340503e+00f},
        {4.268788542e-02f, -1.834061209e+00f},
        {2.177953338e-02f,  7.878101198e-01f},
        {1.111200683e-02f, -7.812202443e-01f},
    }},
    // Z=4  Be (9 terms)
    {9, {
        {4.743480742e+00f, -3.747794313e+00f},
        {2.420143236e+00f,  7.400723658e+00f},
        {1.234766957e+00f, -1.126269935e+01f},
        {6.299831413e-01f,  1.028838749e+01f},
        {3.214199700e-01f, -8.457585408e+00f},
        {1.639897806e-01f,  5.218894454e+00f},
        {8.366825543e-02f, -3.766492406e+00f},
        {4.268788542e-02f,  2.010666947e+00f},
        {2.177953338e-02f, -1.684101068e+00f},
    }},
    // Z=5  B (11 terms)
    {11, {
        {9.297222254e+00f, -3.210170651e+00f},
        {4.743480742e+00f,  6.274732119e+00f},
        {2.420143236e+00f, -9.910984211e+00f},
        {1.234766957e+00f,  9.050180592e+00f},
        {6.299831413e-01f, -7.905614916e+00f},
        {3.214199700e-01f,  5.143829342e+00f},
        {1.639897806e-01f, -5.306160429e+00f},
        {8.366825543e-02f,  3.536343834e+00f},
        {4.268788542e-02f, -3.396693199e+00f},
        {2.177953338e-02f,  1.724570476e+00f},
        {1.111200683e-02f, -1.000032957e+00f},
    }},
    // Z=6  C (9 terms)
    {9, {
        {9.297222254e+00f, -4.171886390e+00f},
        {4.743480742e+00f,  6.945998759e+00f},
        {2.420143236e+00f, -8.611687746e+00f},
        {1.234766957e+00f,  7.178058194e+00f},
        {6.299831413e-01f, -8.253920026e+00f},
        {3.214199700e-01f,  3.953077308e+00f},
        {1.639897806e-01f, -3.633981649e+00f},
        {8.366825543e-02f,  1.856334429e+00f},
        {4.268788542e-02f, -1.261992879e+00f},
    }},
    // Z=7  N (9 terms)
    {9, {
        {9.297222254e+00f, -5.081318620e+00f},
        {4.743480742e+00f,  8.213310291e+00f},
        {2.420143236e+00f, -1.011875403e+01f},
        {1.234766957e+00f,  7.506884912e+00f},
        {6.299831413e-01f, -8.398524643e+00f},
        {3.214199700e-01f,  3.934491196e+00f},
        {1.639897806e-01f, -2.897407144e+00f},
        {8.366825543e-02f,  8.940734185e-01f},
        {4.268788542e-02f, -1.052755376e+00f},
    }},
    // Z=8  O (11 terms)
    {11, {
        {1.822255562e+01f, -3.710759698e+00f},
        {9.297222254e+00f,  5.973193470e+00f},
        {4.743480742e+00f, -8.779265645e+00f},
        {2.420143236e+00f,  8.253574506e+00f},
        {1.234766957e+00f, -1.038518857e+01f},
        {6.299831413e-01f,  6.169044227e+00f},
        {3.214199700e-01f, -7.334251392e+00f},
        {1.639897806e-01f,  5.393036064e+00f},
        {8.366825543e-02f, -4.956605911e+00f},
        {4.268788542e-02f,  2.680316414e+00f},
        {2.177953338e-02f, -1.303093471e+00f},
    }},
    // Z=9  F (10 terms)
    {10, {
        {1.822255562e+01f, -4.480332919e+00f},
        {9.297222254e+00f,  6.513433163e+00f},
        {4.743480742e+00f, -7.521000012e+00f},
        {2.420143236e+00f,  4.125841245e+00f},
        {1.234766957e+00f, -5.779926691e+00f},
        {6.299831413e-01f,  7.449690639e-02f},
        {3.214199700e-01f, -3.972697871e-01f},
        {1.639897806e-01f, -1.113489110e+00f},
        {8.366825543e-02f,  4.317019303e-01f},
        {4.268788542e-02f, -8.534547254e-01f},
    }},
    // Z=10 Ne (9 terms)
    {9, {
        {1.822255562e+01f, -5.958028604e+00f},
        {9.297222254e+00f,  1.011627001e+01f},
        {4.743480742e+00f, -1.270734070e+01f},
        {2.420143236e+00f,  8.699803143e+00f},
        {1.234766957e+00f, -1.169101850e+01f},
        {6.299831413e-01f,  6.158487736e+00f},
        {3.214199700e-01f, -5.499603264e+00f},
        {1.639897806e-01f,  2.783781030e+00f},
        {8.366825543e-02f, -1.902350849e+00f},
    }},
    // Z=11 Na (12 terms)
    {12, {
        {1.822255562e+01f, -7.056086550e+00f},
        {9.297222254e+00f,  1.212093663e+01f},
        {4.743480742e+00f, -1.448740462e+01f},
        {2.420143236e+00f,  6.636541098e+00f},
        {1.234766957e+00f, -6.903178811e+00f},
        {6.299831413e-01f,  1.898600546e+00f},
        {3.214199700e-01f, -2.168022868e+00f},
        {1.639897806e-01f,  3.194724799e-02f},
        {8.366825543e-02f, -1.175181806e-01f},
        {4.268788542e-02f, -7.548534333e-01f},
        {2.177953338e-02f,  7.158732078e-01f},
        {1.111200683e-02f, -9.168342651e-01f},
    }},
    // Z=12 Mg (12 terms)
    {12, {
        {3.571620901e+01f, -4.034598541e+00f},
        {1.822255562e+01f,  4.803185159e+00f},
        {9.297222254e+00f, -4.010052213e+00f},
        {4.743480742e+00f, -2.366611176e+00f},
        {2.420143236e+00f,  8.404480061e-02f},
        {1.234766957e+00f, -3.974227232e+00f},
        {6.299831413e-01f,  1.552244047e+00f},
        {3.214199700e-01f, -4.967518430e+00f},
        {1.639897806e-01f,  5.516111564e+00f},
        {8.366825543e-02f, -6.272272959e+00f},
        {4.268788542e-02f,  4.613997280e+00f},
        {2.177953338e-02f, -2.944302301e+00f},
    }},
    // Z=13 Al (14 terms)
    {14, {
        {3.571620901e+01f, -5.251578959e+00f},
        {1.822255562e+01f,  8.010281155e+00f},
        {9.297222254e+00f, -8.963330198e+00f},
        {4.743480742e+00f,  1.240931190e+00f},
        {2.420143236e+00f, -1.474410769e+00f},
        {1.234766957e+00f, -3.547210662e+00f},
        {6.299831413e-01f,  3.357770781e-01f},
        {3.214199700e-01f, -1.015387812e+00f},
        {1.639897806e-01f, -1.306685496e-02f},
        {8.366825543e-02f, -1.546703856e+00f},
        {4.268788542e-02f,  5.877102623e-01f},
        {2.177953338e-02f, -1.344342009e+00f},
        {1.111200683e-02f,  5.722753595e-01f},
        {5.669391238e-03f, -5.909439251e-01f},
    }},
    // Z=14 Si (12 terms)
    {12, {
        {3.571620901e+01f, -6.463826394e+00f},
        {1.822255562e+01f,  1.144490883e+01f},
        {9.297222254e+00f, -1.549272349e+01f},
        {4.743480742e+00f,  9.045648331e+00f},
        {2.420143236e+00f, -9.690376613e+00f},
        {1.234766957e+00f,  3.303393574e+00f},
        {6.299831413e-01f, -1.505396563e+00f},
        {3.214199700e-01f, -2.552601512e+00f},
        {1.639897806e-01f, -7.042284475e-01f},
        {8.366825543e-02f, -8.260441654e-01f},
        {4.268788542e-02f,  2.540161869e-01f},
        {2.177953338e-02f, -8.127697353e-01f},
    }},
    // Z=15 P (12 terms)
    {12, {
        {3.571620901e+01f, -7.499457306e+00f},
        {1.822255562e+01f,  1.413607958e+01f},
        {9.297222254e+00f, -2.083107788e+01f},
        {4.743480742e+00f,  1.545938830e+01f},
        {2.420143236e+00f, -1.516434673e+01f},
        {1.234766957e+00f,  6.871837386e+00f},
        {6.299831413e-01f, -4.421751029e+00f},
        {3.214199700e-01f, -9.332405743e-01f},
        {1.639897806e-01f, -1.126289570e+00f},
        {8.366825543e-02f, -3.279153040e-01f},
        {4.268788542e-02f, -5.639590452e-01f},
        {2.177953338e-02f, -5.992678117e-01f},
    }},
    // Z=16 S (13 terms)
    {13, {
        {7.000376966e+01f, -4.413991410e+00f},
        {3.571620901e+01f,  6.987646126e+00f},
        {1.822255562e+01f, -9.305896443e+00f},
        {9.297222254e+00f,  3.837245954e+00f},
        {4.743480742e+00f, -5.327800851e+00f},
        {2.420143236e+00f, -1.290586609e-01f},
        {1.234766957e+00f, -2.994288265e+00f},
        {6.299831413e-01f,  2.485891972e+00f},
        {3.214199700e-01f, -7.884977126e+00f},
        {1.639897806e-01f,  5.040648057e+00f},
        {8.366825543e-02f, -5.392239935e+00f},
        {4.268788542e-02f,  2.845631622e+00f},
        {2.177953338e-02f, -1.748811041e+00f},
    }},
    // Z=17 Cl (12 terms)
    {12, {
        {7.000376966e+01f, -5.034726162e+00f},
        {3.571620901e+01f,  8.283061239e+00f},
        {1.822255562e+01f, -1.124528832e+01f},
        {9.297222254e+00f,  4.646057634e+00f},
        {4.743480742e+00f, -3.899564247e+00f},
        {2.420143236e+00f, -5.076906891e+00f},
        {1.234766957e+00f,  6.393424695e+00f},
        {6.299831413e-01f, -1.081530331e+01f},
        {3.214199700e-01f,  5.247693471e+00f},
        {1.639897806e-01f, -6.755512726e+00f},
        {8.366825543e-02f,  3.893888689e+00f},
        {4.268788542e-02f, -2.636824076e+00f},
    }},
    // Z=18 Ar (12 terms)
    {12, {
        {7.000376966e+01f, -5.860103640e+00f},
        {3.571620901e+01f,  1.059347901e+01f},
        {1.822255562e+01f, -1.611130313e+01f},
        {9.297222254e+00f,  1.159485676e+01f},
        {4.743480742e+00f, -1.271069027e+01f},
        {2.420143236e+00f,  4.629048535e+00f},
        {1.234766957e+00f, -2.322904477e+00f},
        {6.299831413e-01f, -5.233347484e+00f},
        {3.214199700e-01f,  2.881619632e-01f},
        {1.639897806e-01f, -2.222161542e+00f},
        {8.366825543e-02f,  8.208771994e-01f},
        {4.268788542e-02f, -1.465912932e+00f},
    }},
};
// clang-format on

static const int SAP_MAX_Z = (int)(sizeof(SAP_DATA) / sizeof(SAP_DATA[0])) - 1;

// ====================== V_SAP 积分核函数 ======================
// 基于 arXiv:2603.16989 的方法：对核吸引积分的 Boys 函数做修正
//
// 标准核吸引: F_m(T), T = g * R_PC²
// SAP 修正:   F_m(T) → F_m(T) - Σ_k c̃_k (α_k/(g+α_k))^(m+1/2)
// F_m(T·α_k/(g+α_k)) 其中 c̃_k = c_k / Z_C
//
// 前因子 (-Z * 2π/g) 和 R tensor 递推完全不变
// ==============================================================

#include "../integrals/one_e.hpp"

// 对 Boys 函数值施加 SAP 修正
static __device__ void apply_sap_correction(double* F_vals, int L_tot, float T,
                                            float g, const SAP_ATOM_DATA& sap,
                                            int Z)
{
    if (Z == 0) return;
    const float inv_Z = 1.0f / (float)Z;

    for (int k = 0; k < sap.n_terms; k++)
    {
        float alpha_k = sap.terms[k].alpha;
        float c_tilde = sap.terms[k].coeff * inv_Z;
        float ratio = alpha_k / (g + alpha_k);
        float T_mod = T * ratio;

        double F_mod[ONEE_MD_BASE];
        compute_boys_double(F_mod, T_mod, L_tot);

        double r_pow = sqrtf(ratio);  // ratio^(1/2) for m=0
        for (int m = 0; m <= L_tot; m++)
        {
            F_vals[m] += (double)c_tilde * r_pow * F_mod[m];
            r_pow *= (double)ratio;  // ratio^(m+1/2)
        }
    }
}

static __global__ void SAP_Kernel(const int n_tasks, const QC_ONE_E_TASK* tasks,
                                  const VECTOR* centers, const int* l_list,
                                  const float* exps, const float* coeffs,
                                  const int* shell_offsets,
                                  const int* shell_sizes, const int* ao_offsets,
                                  const int* atm, const float* env, int natm,
                                  const int* d_Z, float* out_V_SAP,
                                  int nao_total)
{
    SIMPLE_DEVICE_FOR(task_id, n_tasks)
    {
        QC_ONE_E_TASK sh_idx = tasks[task_id];
        int i_sh = sh_idx.x;
        int j_sh = sh_idx.y;

        int li = l_list[i_sh], lj = l_list[j_sh];
        int ni = (li + 1) * (li + 2) / 2, nj = (lj + 1) * (lj + 2) / 2;
        int off_i = ao_offsets[i_sh], off_j = ao_offsets[j_sh];
        const VECTOR A = centers[i_sh];
        const VECTOR B = centers[j_sh];
        float Ax = A.x, Ay = A.y, Az = A.z;
        float Bx = B.x, By = B.y, Bz = B.z;
        float dist_sq = (Ax - Bx) * (Ax - Bx) + (Ay - By) * (Ay - By) +
                        (Az - Bz) * (Az - Bz);

        for (int idx_i = 0; idx_i < ni; idx_i++)
        {
            for (int idx_j = 0; idx_j < nj; idx_j++)
            {
                int lx_i, ly_i, lz_i, lx_j, ly_j, lz_j;
                QC_Get_Lxyz_Device(li, idx_i, lx_i, ly_i, lz_i);
                QC_Get_Lxyz_Device(lj, idx_j, lx_j, ly_j, lz_j);
                float total_V = 0.0f;

                for (int pi = 0; pi < shell_sizes[i_sh]; pi++)
                {
                    float ei = exps[shell_offsets[i_sh] + pi];
                    float ci = coeffs[shell_offsets[i_sh] + pi];
                    for (int pj = 0; pj < shell_sizes[j_sh]; pj++)
                    {
                        float ej = exps[shell_offsets[j_sh] + pj];
                        float cj = coeffs[shell_offsets[j_sh] + pj];
                        float g = ei + ej;
                        float Kab = expf(-ei * ej / g * dist_sq);
                        float Px = (ei * Ax + ej * Bx) / g;
                        float Py = (ei * Ay + ej * By) / g;
                        float Pz = (ei * Az + ej * Bz) / g;
                        float E_x[5][5][9], E_y[5][5][9], E_z[5][5][9];
                        compute_md_coeffs(E_x, li, lj, Px - Ax, Px - Bx,
                                          0.5f / g);
                        compute_md_coeffs(E_y, li, lj, Py - Ay, Py - By,
                                          0.5f / g);
                        compute_md_coeffs(E_z, li, lj, Pz - Az, Pz - Bz,
                                          0.5f / g);
                        int L_tot = li + lj;

                        for (int iat = 0; iat < natm; iat++)
                        {
                            int Z = d_Z[iat];
                            if (Z <= 0 || Z > SAP_MAX_Z) continue;

                            int ptr_coord = atm[iat * 6 + 1];
                            float Cx = env[ptr_coord];
                            float Cy = env[ptr_coord + 1];
                            float Cz = env[ptr_coord + 2];
                            float PC2 = (Px - Cx) * (Px - Cx) +
                                        (Py - Cy) * (Py - Cy) +
                                        (Pz - Cz) * (Pz - Cz);
                            float PC[3] = {Px - Cx, Py - Cy, Pz - Cz};
                            float T = g * PC2;

                            double F_vals[ONEE_MD_BASE];
                            compute_boys_double(F_vals, T, L_tot);
                            apply_sap_correction(F_vals, L_tot, T, (float)g,
                                                 SAP_DATA[Z], Z);

                            float R_vals[ONEE_MD_BASE * ONEE_MD_BASE *
                                         ONEE_MD_BASE * ONEE_MD_BASE];
                            compute_r_tensor_1e(R_vals, F_vals, (float)g, PC,
                                                L_tot);

                            double v_sum = 0.0;
                            for (int tx = 0; tx <= lx_i + lx_j; tx++)
                            {
                                float ex = E_x[lx_i][lx_j][tx];
                                if (ex == 0.0f) continue;
                                for (int ty = 0; ty <= ly_i + ly_j; ty++)
                                {
                                    float ey = E_y[ly_i][ly_j][ty];
                                    if (ey == 0.0f) continue;
                                    for (int tz = 0; tz <= lz_i + lz_j; tz++)
                                    {
                                        float ez = E_z[lz_i][lz_j][tz];
                                        if (ez == 0.0f) continue;
                                        v_sum += (double)ex * (double)ey *
                                                 (double)ez *
                                                 (double)R_vals[ONEE_MD_IDX(
                                                     tx, ty, tz, 0)];
                                    }
                                }
                            }
                            total_V += ci * cj * Kab * -(float)Z *
                                       (2.0f * CONSTANT_Pi / g) * (float)v_sum;
                        }
                    }
                }
                int idx = (int)(off_i + idx_i) * nao_total + (off_j + idx_j);
                atomicAdd(&out_V_SAP[idx], total_V);
            }
        }
    }
}

void QC_Compute_V_SAP(const QC_MOLECULE& mol, const QC_INTEGRAL_TASKS& task_ctx,
                      float* d_V_SAP)
{
    const int nao_c = mol.nao_cart;
    deviceMemset(d_V_SAP, 0, sizeof(float) * nao_c * nao_c);

    const int chunk_size = ONE_E_BATCH_SIZE;
    for (int i = 0; i < task_ctx.topo.n_1e_tasks; i += chunk_size)
    {
        int current_chunk = std::min(chunk_size, task_ctx.topo.n_1e_tasks - i);
        QC_ONE_E_TASK* task_ptr = task_ctx.buffers.d_1e_tasks + i;
        Launch_Device_Kernel(
            SAP_Kernel, (current_chunk + 63) / 64, 64, 0, 0, current_chunk,
            task_ptr, mol.d_centers, mol.d_l_list, mol.d_exps, mol.d_coeffs,
            mol.d_shell_offsets, mol.d_shell_sizes, mol.d_ao_offsets, mol.d_atm,
            mol.d_env, mol.natm, mol.d_Z, d_V_SAP, nao_c);
    }
}

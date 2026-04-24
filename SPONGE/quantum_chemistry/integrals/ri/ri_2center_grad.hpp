#pragma once

// 二中心 Coulomb 积分导数 d(P|Q)/dR 的内核
// 使用 McMurchie-Davidson 方案，与 ri_2center.hpp 对应
// 导数公式: d(P|Q)/dA_{P,x} = 2*alpha_P * ((P+1_x)|Q) - l_{P,x} * ((P-1_x)|Q)
// 平移不变性: d(P|Q)/dA_Q = -d(P|Q)/dA_P

#include "../one_e.hpp"

// 扩展维度的 E 系数和 R 张量，用于梯度计算
// 最大角动量 l=4 (g函数)，导数需要 l+1=5，E 数组需 [6][6][11]
// R 张量 base 需要 ONEE_MD_BASE+2 = 11 来容纳 L_tot+1
#define RI_GRAD_E_DIM1 6
#define RI_GRAD_E_DIM2 6
#define RI_GRAD_E_DIM3 11
#define RI_GRAD_R_BASE 11
#define RI_GRAD_R_IDX(t, u, v, n)                                             \
    ((((t) * RI_GRAD_R_BASE + (u)) * RI_GRAD_R_BASE + (v)) * RI_GRAD_R_BASE + \
     (n))

// 扩展版 compute_md_coeffs，支持更大数组维度
static __device__ void compute_md_coeffs_grad(
    float E[RI_GRAD_E_DIM1][RI_GRAD_E_DIM2][RI_GRAD_E_DIM3], int la_max,
    int lb_max, float PA, float PB, float one_over_2p)
{
    for (int i = 0; i < RI_GRAD_E_DIM1; i++)
        for (int j = 0; j < RI_GRAD_E_DIM2; j++)
            for (int n = 0; n < RI_GRAD_E_DIM3; n++) E[i][j][n] = 0.0f;
    E[0][0][0] = 1.0f;
    for (int la = 0; la <= la_max; la++)
    {
        for (int lb = 0; lb <= lb_max; lb++)
        {
            if (la == 0 && lb == 0) continue;
            if (la > 0)
            {
                int la_p = la - 1;
                for (int n = 0; n <= la + lb; n++)
                {
                    float val = PA * E[la_p][lb][n];
                    if (n > 0) val += one_over_2p * E[la_p][lb][n - 1];
                    if ((n + 1) <= la_p + lb)
                        val += (float)(n + 1) * E[la_p][lb][n + 1];
                    E[la][lb][n] = val;
                }
            }
            else
            {
                int lb_p = lb - 1;
                for (int n = 0; n <= la + lb; n++)
                {
                    float val = PB * E[la][lb_p][n];
                    if (n > 0) val += one_over_2p * E[la][lb_p][n - 1];
                    if ((n + 1) <= la + lb_p)
                        val += (float)(n + 1) * E[la][lb_p][n + 1];
                    E[la][lb][n] = val;
                }
            }
        }
    }
}

// 扩展版 Boys 函数 (double 精度)
static __device__ void compute_boys_double_grad(double* F, float t, int max_m)
{
    const double td = (double)t;
    if (td < 1e-15)
    {
        for (int m = 0; m <= max_m; m++) F[m] = 1.0 / (2.0 * m + 1.0);
        return;
    }
    const double exp_t = exp(-td);
    const double st = sqrt(td);
    const double f0 = 0.5 * 1.7724538509055159 * erf(st) / st;
    if (td <= 30.0)
    {
        double work[64];
        const int m_top = max_m + 25;
        work[m_top] = 0.0;
        for (int m = m_top - 1; m >= 0; m--)
            work[m] = (2.0 * td * work[m + 1] + exp_t) / (2.0 * m + 1.0);
        const double scale = f0 / work[0];
        for (int m = 0; m <= max_m; m++) F[m] = work[m] * scale;
    }
    else
    {
        F[0] = f0;
        double prev = f0;
        for (int m = 0; m < max_m; m++)
        {
            double next = ((2.0 * m + 1.0) * prev - exp_t) / (2.0 * td);
            F[m + 1] = next;
            prev = next;
        }
    }
}

// 扩展版 R 张量（动态 base，避免固定 11⁴ 的浪费）
static __device__ void compute_r_tensor_grad(float* R, double* F, float alpha,
                                             float PC[3], int L_tot)
{
    const int base = L_tot + 2;  // 实际需要的最小 base
    const int total_size = base * base * base * base;
    for (int i = 0; i < total_size; i++) R[i] = 0.0f;

// 使用动态 base 的索引宏
#define R_IDX(t, u, v, n) ((((t) * base + (u)) * base + (v)) * base + (n))

    double m2a = -2.0 * (double)alpha;
    double fac = 1.0;
    for (int n = 0; n <= L_tot; n++)
    {
        R[R_IDX(0, 0, 0, n)] = (float)(fac * F[n]);
        fac *= m2a;
    }

    for (int N = 1; N <= L_tot; N++)
    {
        for (int t = 0; t <= N; t++)
        {
            for (int u = 0; u <= N - t; u++)
            {
                int v = N - t - u;
                int max_n = L_tot - N;
                for (int n = 0; n <= max_n; n++)
                {
                    double val = 0.0;
                    if (t > 0)
                    {
                        val = (double)PC[0] * R[R_IDX(t - 1, u, v, n + 1)];
                        if (t > 1)
                            val +=
                                (double)(t - 1) * R[R_IDX(t - 2, u, v, n + 1)];
                    }
                    else if (u > 0)
                    {
                        val = (double)PC[1] * R[R_IDX(t, u - 1, v, n + 1)];
                        if (u > 1)
                            val +=
                                (double)(u - 1) * R[R_IDX(t, u - 2, v, n + 1)];
                    }
                    else if (v > 0)
                    {
                        val = (double)PC[2] * R[R_IDX(t, u, v - 1, n + 1)];
                        if (v > 1)
                            val +=
                                (double)(v - 1) * R[R_IDX(t, u, v - 2, n + 1)];
                    }
                    R[R_IDX(t, u, v, n)] = (float)val;
                }
            }
        }
    }
#undef R_IDX
}

// 二中心积分导数内核
// 对每个辅助 shell P_sh 并行，内部循环 Q_sh <= P_sh
// workspace: 预分配缓冲 [n_workers × ws_stride] doubles
static __global__ void QC_RI_2Center_Grad_Kernel(
    const int naux_bas, const VECTOR* aux_centers, const int* aux_l_list,
    const float* aux_exps, const float* aux_coeffs,
    const int* aux_shell_offsets, const int* aux_shell_sizes,
    const int* aux_ao_offsets_cart, const int* aux_ao_offsets_sph,
    const float* aux_norms, const float* U_aux, int naux_cart, int naux_sph,
    const double* D2_eff, const int* shell_atom_aux, double* workspace,
    int ws_stride, int n_workers, double* grad)
{
    SIMPLE_DEVICE_FOR(P_sh, naux_bas)
    {
        const int worker_id = P_sh % n_workers;
        double* d_cart = workspace + (size_t)worker_id * ws_stride;

        for (int Q_sh = 0; Q_sh <= P_sh; Q_sh++)
        {
            const int lP = aux_l_list[P_sh], lQ = aux_l_list[Q_sh];
            const int nP_cart = (lP + 1) * (lP + 2) / 2;
            const int nQ_cart = (lQ + 1) * (lQ + 2) / 2;
            const int offP_cart = aux_ao_offsets_cart[P_sh];
            const int offQ_cart = aux_ao_offsets_cart[Q_sh];
            const int offP_sph = aux_ao_offsets_sph[P_sh];
            const int offQ_sph = aux_ao_offsets_sph[Q_sh];
            const int nP_sph = 2 * lP + 1;
            const int nQ_sph = 2 * lQ + 1;

            const VECTOR A = aux_centers[P_sh];
            const VECTOR B = aux_centers[Q_sh];
            const float Ax = A.x, Ay = A.y, Az = A.z;
            const float Bx = B.x, By = B.y, Bz = B.z;
            const float dist_sq = (Ax - Bx) * (Ax - Bx) +
                                  (Ay - By) * (Ay - By) + (Az - Bz) * (Az - Bz);

            const int atom_P = shell_atom_aux[P_sh];
            const int atom_Q = shell_atom_aux[Q_sh];

            // 清零 workspace 用于本 shell pair
            const int d_cart_size = nP_cart * nQ_cart * 3;
            for (int k = 0; k < d_cart_size; k++) d_cart[k] = 0.0;

            for (int idxP = 0; idxP < nP_cart; idxP++)
            {
                int lxP, lyP, lzP;
                QC_Get_Lxyz_Device(lP, idxP, lxP, lyP, lzP);

                for (int idxQ = 0; idxQ < nQ_cart; idxQ++)
                {
                    int lxQ, lyQ, lzQ;
                    QC_Get_Lxyz_Device(lQ, idxQ, lxQ, lyQ, lzQ);

                    double dA[3] = {0.0, 0.0, 0.0};

                    for (int pi = 0; pi < aux_shell_sizes[P_sh]; pi++)
                    {
                        const float eP = aux_exps[aux_shell_offsets[P_sh] + pi];
                        const float cP =
                            aux_coeffs[aux_shell_offsets[P_sh] + pi];

                        for (int pj = 0; pj < aux_shell_sizes[Q_sh]; pj++)
                        {
                            const float eQ =
                                aux_exps[aux_shell_offsets[Q_sh] + pj];
                            const float cQ =
                                aux_coeffs[aux_shell_offsets[Q_sh] + pj];

                            float E_Px[RI_GRAD_E_DIM1][RI_GRAD_E_DIM2]
                                      [RI_GRAD_E_DIM3];
                            float E_Py[RI_GRAD_E_DIM1][RI_GRAD_E_DIM2]
                                      [RI_GRAD_E_DIM3];
                            float E_Pz[RI_GRAD_E_DIM1][RI_GRAD_E_DIM2]
                                      [RI_GRAD_E_DIM3];
                            compute_md_coeffs_grad(E_Px, lxP + 1, 0, 0.0f, 0.0f,
                                                   0.5f / eP);
                            compute_md_coeffs_grad(E_Py, lyP + 1, 0, 0.0f, 0.0f,
                                                   0.5f / eP);
                            compute_md_coeffs_grad(E_Pz, lzP + 1, 0, 0.0f, 0.0f,
                                                   0.5f / eP);

                            float E_Qx[RI_GRAD_E_DIM1][RI_GRAD_E_DIM2]
                                      [RI_GRAD_E_DIM3];
                            float E_Qy[RI_GRAD_E_DIM1][RI_GRAD_E_DIM2]
                                      [RI_GRAD_E_DIM3];
                            float E_Qz[RI_GRAD_E_DIM1][RI_GRAD_E_DIM2]
                                      [RI_GRAD_E_DIM3];
                            compute_md_coeffs_grad(E_Qx, lxQ, 0, 0.0f, 0.0f,
                                                   0.5f / eQ);
                            compute_md_coeffs_grad(E_Qy, lyQ, 0, 0.0f, 0.0f,
                                                   0.5f / eQ);
                            compute_md_coeffs_grad(E_Qz, lzQ, 0, 0.0f, 0.0f,
                                                   0.5f / eQ);

                            const float alpha_pq = eP * eQ / (eP + eQ);
                            const float T_val = alpha_pq * dist_sq;
                            const int L_tot = lP + lQ;
                            const int R_base = L_tot + 3;

                            double F_vals[RI_GRAD_R_BASE];
                            compute_boys_double_grad(F_vals, T_val, L_tot + 1);
                            float AB[3] = {Ax - Bx, Ay - By, Az - Bz};
                            float R_vals[RI_GRAD_R_BASE * RI_GRAD_R_BASE *
                                         RI_GRAD_R_BASE * RI_GRAD_R_BASE];
                            compute_r_tensor_grad(R_vals, F_vals, alpha_pq, AB,
                                                  L_tot + 1);

                            const double prefactor =
                                (double)cP * (double)cQ *
                                (2.0 * CONSTANT_Pi * CONSTANT_Pi *
                                 sqrt(CONSTANT_Pi)) /
                                ((double)eP * (double)eQ *
                                 sqrt((double)(eP + eQ)));

                            auto contract_2c = [&](int axP, int ayP,
                                                   int azP) -> double
                            {
                                if (axP < 0 || ayP < 0 || azP < 0) return 0.0;
                                double v_sum = 0.0;
                                for (int t = 0; t <= axP; t++)
                                {
                                    double ePx = (double)E_Px[axP][0][t];
                                    if (ePx == 0.0) continue;
                                    for (int u = 0; u <= ayP; u++)
                                    {
                                        double ePy = (double)E_Py[ayP][0][u];
                                        if (ePy == 0.0) continue;
                                        for (int v = 0; v <= azP; v++)
                                        {
                                            double ePz =
                                                (double)E_Pz[azP][0][v];
                                            if (ePz == 0.0) continue;
                                            for (int tt = 0; tt <= lxQ; tt++)
                                            {
                                                double eQx =
                                                    (double)E_Qx[lxQ][0][tt];
                                                if (eQx == 0.0) continue;
                                                for (int uu = 0; uu <= lyQ;
                                                     uu++)
                                                {
                                                    double eQy = (double)
                                                        E_Qy[lyQ][0][uu];
                                                    if (eQy == 0.0) continue;
                                                    for (int vv = 0; vv <= lzQ;
                                                         vv++)
                                                    {
                                                        double eQz = (double)
                                                            E_Qz[lzQ][0][vv];
                                                        if (eQz == 0.0)
                                                            continue;
                                                        double sign =
                                                            ((tt + uu + vv) & 1)
                                                                ? -1.0
                                                                : 1.0;
                                                        v_sum +=
                                                            ePx * ePy * ePz *
                                                            eQx * eQy * eQz *
                                                            sign *
                                                            (double)R_vals
                                                                [(((t + tt) *
                                                                       R_base +
                                                                   (u + uu)) *
                                                                      R_base +
                                                                  (v + vv)) *
                                                                 R_base];
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                return v_sum;
                            };

                            double dx = 2.0 * (double)eP *
                                        contract_2c(lxP + 1, lyP, lzP);
                            if (lxP > 0)
                                dx -= (double)lxP *
                                      contract_2c(lxP - 1, lyP, lzP);
                            double dy = 2.0 * (double)eP *
                                        contract_2c(lxP, lyP + 1, lzP);
                            if (lyP > 0)
                                dy -= (double)lyP *
                                      contract_2c(lxP, lyP - 1, lzP);
                            double dz = 2.0 * (double)eP *
                                        contract_2c(lxP, lyP, lzP + 1);
                            if (lzP > 0)
                                dz -= (double)lzP *
                                      contract_2c(lxP, lyP, lzP - 1);

                            dA[0] += prefactor * dx;
                            dA[1] += prefactor * dy;
                            dA[2] += prefactor * dz;
                        }
                    }

                    const int cart_idx = (idxP * nQ_cart + idxQ) * 3;
                    d_cart[cart_idx + 0] = dA[0];
                    d_cart[cart_idx + 1] = dA[1];
                    d_cart[cart_idx + 2] = dA[2];
                }
            }

            // Cart2Sph + 归一化 + D2_eff 收缩 → atomicAdd 到 grad
            for (int ps = 0; ps < nP_sph; ps++)
            {
                const int P_sph = offP_sph + ps;
                const double normP = (double)aux_norms[P_sph];

                for (int qs = 0; qs < nQ_sph; qs++)
                {
                    const int Q_sph = offQ_sph + qs;
                    const double normQ = (double)aux_norms[Q_sph];

                    double d_sph[3] = {0.0, 0.0, 0.0};
                    for (int pc = 0; pc < nP_cart; pc++)
                    {
                        double u_p =
                            (double)U_aux[(offP_cart + pc) * naux_sph + P_sph];
                        if (u_p == 0.0) continue;
                        for (int qc = 0; qc < nQ_cart; qc++)
                        {
                            double u_q = (double)
                                U_aux[(offQ_cart + qc) * naux_sph + Q_sph];
                            if (u_q == 0.0) continue;
                            double w = u_p * u_q;
                            const int cidx = (pc * nQ_cart + qc) * 3;
                            d_sph[0] += w * d_cart[cidx + 0];
                            d_sph[1] += w * d_cart[cidx + 1];
                            d_sph[2] += w * d_cart[cidx + 2];
                        }
                    }

                    double norm_pq = normP * normQ;
                    d_sph[0] *= norm_pq;
                    d_sph[1] *= norm_pq;
                    d_sph[2] *= norm_pq;

                    double dens = D2_eff[P_sph * naux_sph + Q_sph];
                    if (P_sh != Q_sh) dens += D2_eff[Q_sph * naux_sph + P_sph];

                    for (int d = 0; d < 3; d++)
                    {
                        double contrib = dens * d_sph[d];
                        atomicAdd(&grad[atom_P * 3 + d], contrib);
                        atomicAdd(&grad[atom_Q * 3 + d], -contrib);
                    }
                }
            }
        }
    }
}

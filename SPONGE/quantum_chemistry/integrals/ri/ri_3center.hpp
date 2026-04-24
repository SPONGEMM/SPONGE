#pragma once

// 三中心 Coulomb 积分 (P|μν) = ∫∫ P(r1) 1/r12 μ(r2)ν(r2) dr1 dr2
// 使用 McMurchie-Davidson 方案
// 优化: primitive 循环在外，Cartesian 分量循环在内，
//       E 系数和 Boys/R 张量只计算一次后查表收缩

#include "../one_e.hpp"

// 三中心积分任务
struct QC_RI_3C_TASK
{
    int P_sh, mu_sh, nu_sh;
};

// 缓存一个 bra primitive 的 E 系数（所有角动量分量）
struct QC_RI_Bra_Prim_Cache
{
    float E_x[6][6][11];  // E^{lP}_x for all lxP
    float E_y[6][6][11];
    float E_z[6][6][11];
    double coeff;  // cP * prefactor 部分
    float eP;
};

// 三中心 Coulomb 积分内核
static __global__ void QC_RI_3Center_Kernel(
    const int n_tasks, const QC_RI_3C_TASK* tasks, const VECTOR* aux_centers,
    const int* aux_l_list, const float* aux_exps, const float* aux_coeffs,
    const int* aux_shell_offsets, const int* aux_shell_sizes,
    const int* aux_ao_offsets, const VECTOR* orb_centers, const int* orb_l_list,
    const float* orb_exps, const float* orb_coeffs,
    const int* orb_shell_offsets, const int* orb_shell_sizes,
    const int* orb_ao_offsets, int naux, int out_mu_dim, int out_nu_dim,
    int mu_offset_base, int nu_offset_base, bool fill_symmetric,
    double* out_eri3c)
{
    SIMPLE_DEVICE_FOR(task_id, n_tasks)
    {
        const QC_RI_3C_TASK& task = tasks[task_id];
        const int P_sh = task.P_sh;
        const int mu_sh = task.mu_sh;
        const int nu_sh = task.nu_sh;

        const int lP = aux_l_list[P_sh];
        const int lmu = orb_l_list[mu_sh];
        const int lnu = orb_l_list[nu_sh];

        const int nP = (lP + 1) * (lP + 2) / 2;
        const int nmu = (lmu + 1) * (lmu + 2) / 2;
        const int nnu = (lnu + 1) * (lnu + 2) / 2;

        const int offP = aux_ao_offsets[P_sh];
        const int offmu = orb_ao_offsets[mu_sh];
        const int offnu = orb_ao_offsets[nu_sh];

        const VECTOR A = aux_centers[P_sh];
        const VECTOR B = orb_centers[mu_sh];
        const VECTOR C = orb_centers[nu_sh];

        // 累积缓冲: 所有笛卡尔分量
        double buf[15 * 15 * 15];  // max nP×nmu×nnu for l=4
        const int n_cart = nP * nmu * nnu;
        for (int i = 0; i < n_cart; i++) buf[i] = 0.0;

        // primitive 循环 (外层)
        for (int pP = 0; pP < aux_shell_sizes[P_sh]; pP++)
        {
            const float eP = aux_exps[aux_shell_offsets[P_sh] + pP];
            const float cP = aux_coeffs[aux_shell_offsets[P_sh] + pP];

            // Bra E 系数: 对所有 lxP ∈ [0, lP]，一次算完
            float E_Px[6][6][11], E_Py[6][6][11], E_Pz[6][6][11];
            compute_md_coeffs(E_Px, lP, 0, 0.0f, 0.0f, 0.5f / eP);
            compute_md_coeffs(E_Py, lP, 0, 0.0f, 0.0f, 0.5f / eP);
            compute_md_coeffs(E_Pz, lP, 0, 0.0f, 0.0f, 0.5f / eP);

            for (int p_mu = 0; p_mu < orb_shell_sizes[mu_sh]; p_mu++)
            {
                const float e_mu = orb_exps[orb_shell_offsets[mu_sh] + p_mu];
                const float c_mu = orb_coeffs[orb_shell_offsets[mu_sh] + p_mu];

                for (int p_nu = 0; p_nu < orb_shell_sizes[nu_sh]; p_nu++)
                {
                    const float e_nu =
                        orb_exps[orb_shell_offsets[nu_sh] + p_nu];
                    const float c_nu =
                        orb_coeffs[orb_shell_offsets[nu_sh] + p_nu];

                    const float g_ket = e_mu + e_nu;
                    const float BC2 = (B.x - C.x) * (B.x - C.x) +
                                      (B.y - C.y) * (B.y - C.y) +
                                      (B.z - C.z) * (B.z - C.z);
                    const float K_ket = expf(-e_mu * e_nu / g_ket * BC2);

                    // 原始函数筛选
                    if (fabsf(cP * c_mu * c_nu * K_ket) < 1e-15f) continue;

                    const float Qx = (e_mu * B.x + e_nu * C.x) / g_ket;
                    const float Qy = (e_mu * B.y + e_nu * C.y) / g_ket;
                    const float Qz = (e_mu * B.z + e_nu * C.z) / g_ket;

                    // Ket E 系数: 对所有 (lx_mu, lx_nu) 分量一次算完
                    float E_Kx[6][6][11], E_Ky[6][6][11], E_Kz[6][6][11];
                    compute_md_coeffs(E_Kx, lmu, lnu, Qx - B.x, Qx - C.x,
                                      0.5f / g_ket);
                    compute_md_coeffs(E_Ky, lmu, lnu, Qy - B.y, Qy - C.y,
                                      0.5f / g_ket);
                    compute_md_coeffs(E_Kz, lmu, lnu, Qz - B.z, Qz - C.z,
                                      0.5f / g_ket);

                    // Boys 函数 + R 张量: 只算一次
                    const float alpha_pq = eP * g_ket / (eP + g_ket);
                    const float AQ2 = (A.x - Qx) * (A.x - Qx) +
                                      (A.y - Qy) * (A.y - Qy) +
                                      (A.z - Qz) * (A.z - Qz);
                    const float T_val = alpha_pq * AQ2;
                    const int L_tot = lP + lmu + lnu;

                    double F_vals[ONEE_MD_BASE];
                    float R_vals[ONEE_MD_BASE * ONEE_MD_BASE * ONEE_MD_BASE *
                                 ONEE_MD_BASE];
                    compute_boys_double(F_vals, T_val, L_tot);
                    float AQ[3] = {A.x - Qx, A.y - Qy, A.z - Qz};
                    compute_r_tensor_1e(R_vals, F_vals, alpha_pq, AQ, L_tot);

                    const double prefactor =
                        (double)cP * (double)c_mu * (double)c_nu *
                        (double)K_ket *
                        (2.0 * CONSTANT_Pi * CONSTANT_Pi * sqrt(CONSTANT_Pi)) /
                        ((double)eP * (double)g_ket *
                         sqrt((double)(eP + g_ket)));

                    // Cartesian 分量循环 (内层): 只做查表 + 乘法
                    for (int idxP = 0; idxP < nP; idxP++)
                    {
                        int lxP, lyP, lzP;
                        QC_Get_Lxyz_Device(lP, idxP, lxP, lyP, lzP);

                        for (int idx_mu = 0; idx_mu < nmu; idx_mu++)
                        {
                            int lx_mu, ly_mu, lz_mu;
                            QC_Get_Lxyz_Device(lmu, idx_mu, lx_mu, ly_mu,
                                               lz_mu);

                            for (int idx_nu = 0; idx_nu < nnu; idx_nu++)
                            {
                                int lx_nu, ly_nu, lz_nu;
                                QC_Get_Lxyz_Device(lnu, idx_nu, lx_nu, ly_nu,
                                                   lz_nu);

                                double v_sum = 0.0;
                                for (int t = 0; t <= lxP; t++)
                                {
                                    double ePx = (double)E_Px[lxP][0][t];
                                    if (ePx == 0.0) continue;
                                    for (int u = 0; u <= lyP; u++)
                                    {
                                        double ePy = (double)E_Py[lyP][0][u];
                                        if (ePy == 0.0) continue;
                                        for (int v = 0; v <= lzP; v++)
                                        {
                                            double ePz =
                                                (double)E_Pz[lzP][0][v];
                                            if (ePz == 0.0) continue;

                                            for (int tt = 0;
                                                 tt <= lx_mu + lx_nu; tt++)
                                            {
                                                double eKx = (double)
                                                    E_Kx[lx_mu][lx_nu][tt];
                                                if (eKx == 0.0) continue;
                                                for (int uu = 0;
                                                     uu <= ly_mu + ly_nu; uu++)
                                                {
                                                    double eKy = (double)
                                                        E_Ky[ly_mu][ly_nu][uu];
                                                    if (eKy == 0.0) continue;
                                                    for (int vv = 0;
                                                         vv <= lz_mu + lz_nu;
                                                         vv++)
                                                    {
                                                        double eKz = (double)
                                                            E_Kz[lz_mu][lz_nu]
                                                                [vv];
                                                        if (eKz == 0.0)
                                                            continue;
                                                        double sign =
                                                            ((tt + uu + vv) & 1)
                                                                ? -1.0
                                                                : 1.0;
                                                        v_sum +=
                                                            ePx * ePy * ePz *
                                                            eKx * eKy * eKz *
                                                            sign *
                                                            (double)R_vals
                                                                [ONEE_MD_IDX(
                                                                    t + tt,
                                                                    u + uu,
                                                                    v + vv, 0)];
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                buf[idxP * nmu * nnu + idx_mu * nnu + idx_nu] +=
                                    prefactor * v_sum;
                            }
                        }
                    }
                }
            }
        }

        // 写入输出
        for (int idxP = 0; idxP < nP; idxP++)
        {
            const int P_idx = offP + idxP;
            for (int idx_mu = 0; idx_mu < nmu; idx_mu++)
            {
                const int mu_idx = offmu + idx_mu;
                const int mu_local = mu_idx - mu_offset_base;
                for (int idx_nu = 0; idx_nu < nnu; idx_nu++)
                {
                    const int nu_idx = offnu + idx_nu;
                    const int nu_local = nu_idx - nu_offset_base;
                    const long long idx3c =
                        (long long)P_idx * out_mu_dim * out_nu_dim +
                        (long long)mu_local * out_nu_dim + nu_local;
                    out_eri3c[idx3c] =
                        buf[idxP * nmu * nnu + idx_mu * nnu + idx_nu];

                    if (fill_symmetric && mu_sh != nu_sh)
                    {
                        const int nu_local_sym = nu_idx - mu_offset_base;
                        const int mu_local_sym = mu_idx - nu_offset_base;
                        const long long idx3c_sym =
                            (long long)P_idx * out_mu_dim * out_nu_dim +
                            (long long)nu_local_sym * out_nu_dim + mu_local_sym;
                        out_eri3c[idx3c_sym] =
                            buf[idxP * nmu * nnu + idx_mu * nnu + idx_nu];
                    }
                }
            }
        }
    }
}

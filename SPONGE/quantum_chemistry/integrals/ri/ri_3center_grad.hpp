#pragma once

// 三中心 Coulomb 积分导数 d(P|μν)/dR 的 CPU 内核
// 使用 McMurchie-Davidson 方案，与 ri_3center.hpp 对应
// 三个中心: A_P (辅助), A_mu (轨道), A_nu (轨道)
// 导数:
//   d/dA_P: bra E 系数在 lP+1
//   d/dA_mu: ket E 系数在 (l_mu+1, l_nu)
//   d/dA_nu = -(d/dA_P + d/dA_mu) (平移不变性)

#include "ri_2center_grad.hpp"  // 复用 grad E/R/Boys 工具函数

#ifndef USE_GPU

// 三中心积分导数 CPU 内核
// 对每个 (P_sh, mu_sh, nu_sh) 任务，计算 d(P|μν)/dA_P 和 d(P|μν)/dA_mu
// 与 D3_eff 收缩后累加到 grad[natm * 3]
static inline void QC_RI_3Center_Grad_CPU(
    const int naux_bas, const int norb_bas,
    // 辅助基参数 (host)
    const VECTOR* aux_centers, const int* aux_l_list, const float* aux_exps,
    const float* aux_coeffs, const int* aux_shell_offsets,
    const int* aux_shell_sizes, const int* aux_ao_offsets_cart,
    const int* aux_ao_offsets_sph,
    // 轨道基参数 (host)
    const VECTOR* orb_centers, const int* orb_l_list, const float* orb_exps,
    const float* orb_coeffs, const int* orb_shell_offsets,
    const int* orb_shell_sizes, const int* orb_ao_offsets_cart,
    const int* orb_ao_offsets_sph, bool is_spherical,
    // 归一化与 cart2sph
    const float* aux_norms, const float* orb_norms,
    const float* U_aux, const float* U_orb,
    int naux_cart, int naux_sph, int nao_cart, int nao_sph,
    // 有效密度 [naux_sph × nao_sph × nao_sph]
    const double* D3_eff,
    // 壳层到原子映射
    const int* shell_atom_aux,  // [naux_bas]
    const int* shell_atom_orb,  // [norb_bas]
    // 输出: 梯度累加器 [natm * 3]
    double* grad)
{
    const int nao = nao_sph;

    for (int P_sh = 0; P_sh < naux_bas; P_sh++)
    {
        const int lP = aux_l_list[P_sh];
        const int nP_cart = (lP + 1) * (lP + 2) / 2;
        const int offP_cart = aux_ao_offsets_cart[P_sh];
        const int offP_sph = aux_ao_offsets_sph[P_sh];
        const int nP_sph = 2 * lP + 1;
        const VECTOR A = aux_centers[P_sh];
        const int atom_P = shell_atom_aux[P_sh];

        for (int mu_sh = 0; mu_sh < norb_bas; mu_sh++)
        {
            const int lmu = orb_l_list[mu_sh];
            const int nmu_cart = (lmu + 1) * (lmu + 2) / 2;
            const int offmu_cart = orb_ao_offsets_cart[mu_sh];
            const int offmu_sph = is_spherical ? orb_ao_offsets_sph[mu_sh]
                                               : orb_ao_offsets_cart[mu_sh];
            const int nmu_sph = is_spherical ? (2 * lmu + 1) : nmu_cart;
            const VECTOR B = orb_centers[mu_sh];
            const int atom_mu = shell_atom_orb[mu_sh];

            for (int nu_sh = 0; nu_sh <= mu_sh; nu_sh++)
            {
                const int lnu = orb_l_list[nu_sh];
                const int nnu_cart = (lnu + 1) * (lnu + 2) / 2;
                const int offnu_cart = orb_ao_offsets_cart[nu_sh];
                const int offnu_sph = is_spherical
                                          ? orb_ao_offsets_sph[nu_sh]
                                          : orb_ao_offsets_cart[nu_sh];
                const int nnu_sph = is_spherical ? (2 * lnu + 1) : nnu_cart;
                const VECTOR C = orb_centers[nu_sh];
                const int atom_nu = shell_atom_orb[nu_sh];

                // 笛卡尔导数缓冲:
                // d(P_c|mu_c,nu_c)/dA_P [nP_cart × nmu_cart × nnu_cart × 3]
                // d(P_c|mu_c,nu_c)/dA_mu [同上 × 3]
                const size_t n_cart_total =
                    (size_t)nP_cart * nmu_cart * nnu_cart;
                std::vector<double> d_cart_P(n_cart_total * 3, 0.0);
                std::vector<double> d_cart_mu(n_cart_total * 3, 0.0);

                for (int idxP = 0; idxP < nP_cart; idxP++)
                {
                    int lxP, lyP, lzP;
                    QC_Get_Lxyz_Inline(lP, idxP, lxP, lyP, lzP);

                    for (int idx_mu = 0; idx_mu < nmu_cart; idx_mu++)
                    {
                        int lx_mu, ly_mu, lz_mu;
                        QC_Get_Lxyz_Inline(lmu, idx_mu, lx_mu, ly_mu, lz_mu);

                        for (int idx_nu = 0; idx_nu < nnu_cart; idx_nu++)
                        {
                            int lx_nu, ly_nu, lz_nu;
                            QC_Get_Lxyz_Inline(lnu, idx_nu, lx_nu, ly_nu,
                                               lz_nu);

                            double dAP[3] = {0.0, 0.0, 0.0};
                            double dAmu[3] = {0.0, 0.0, 0.0};

                            // 辅助基原始函数循环
                            for (int pP = 0; pP < aux_shell_sizes[P_sh]; pP++)
                            {
                                const float eP =
                                    aux_exps[aux_shell_offsets[P_sh] + pP];
                                const float cP =
                                    aux_coeffs[aux_shell_offsets[P_sh] + pP];

                                // Bra E 系数: 算到 lP+1 以支持 d/dA_P
                                float E_Px[RI_GRAD_E_DIM1][RI_GRAD_E_DIM2]
                                          [RI_GRAD_E_DIM3];
                                float E_Py[RI_GRAD_E_DIM1][RI_GRAD_E_DIM2]
                                          [RI_GRAD_E_DIM3];
                                float E_Pz[RI_GRAD_E_DIM1][RI_GRAD_E_DIM2]
                                          [RI_GRAD_E_DIM3];
                                compute_md_coeffs_grad(E_Px, lxP + 1, 0, 0.0f,
                                                       0.0f, 0.5f / eP);
                                compute_md_coeffs_grad(E_Py, lyP + 1, 0, 0.0f,
                                                       0.0f, 0.5f / eP);
                                compute_md_coeffs_grad(E_Pz, lzP + 1, 0, 0.0f,
                                                       0.0f, 0.5f / eP);

                                // μν 原始函数循环
                                for (int p_mu = 0;
                                     p_mu < orb_shell_sizes[mu_sh]; p_mu++)
                                {
                                    const float e_mu =
                                        orb_exps[orb_shell_offsets[mu_sh] +
                                                 p_mu];
                                    const float c_mu =
                                        orb_coeffs[orb_shell_offsets[mu_sh] +
                                                   p_mu];

                                    for (int p_nu = 0;
                                         p_nu < orb_shell_sizes[nu_sh]; p_nu++)
                                    {
                                        const float e_nu =
                                            orb_exps[orb_shell_offsets[nu_sh] +
                                                     p_nu];
                                        const float c_nu =
                                            orb_coeffs
                                                [orb_shell_offsets[nu_sh] +
                                                 p_nu];

                                        const float g_ket = e_mu + e_nu;
                                        const float BC2 =
                                            (B.x - C.x) * (B.x - C.x) +
                                            (B.y - C.y) * (B.y - C.y) +
                                            (B.z - C.z) * (B.z - C.z);
                                        const float K_ket =
                                            expf(-e_mu * e_nu / g_ket * BC2);

                                        const float Qx =
                                            (e_mu * B.x + e_nu * C.x) / g_ket;
                                        const float Qy =
                                            (e_mu * B.y + e_nu * C.y) / g_ket;
                                        const float Qz =
                                            (e_mu * B.z + e_nu * C.z) / g_ket;

                                        // Ket E 系数: 算到 (l_mu+1, l_nu+1) 以支持 d/dA_mu 和 d/dA_nu
                                        float E_Kx[RI_GRAD_E_DIM1]
                                                  [RI_GRAD_E_DIM2]
                                                  [RI_GRAD_E_DIM3];
                                        float E_Ky[RI_GRAD_E_DIM1]
                                                  [RI_GRAD_E_DIM2]
                                                  [RI_GRAD_E_DIM3];
                                        float E_Kz[RI_GRAD_E_DIM1]
                                                  [RI_GRAD_E_DIM2]
                                                  [RI_GRAD_E_DIM3];
                                        compute_md_coeffs_grad(
                                            E_Kx, lx_mu + 1, lx_nu + 1,
                                            Qx - B.x, Qx - C.x,
                                            0.5f / g_ket);
                                        compute_md_coeffs_grad(
                                            E_Ky, ly_mu + 1, ly_nu + 1,
                                            Qy - B.y, Qy - C.y,
                                            0.5f / g_ket);
                                        compute_md_coeffs_grad(
                                            E_Kz, lz_mu + 1, lz_nu + 1,
                                            Qz - B.z, Qz - C.z,
                                            0.5f / g_ket);

                                        const float alpha_pq =
                                            eP * g_ket / (eP + g_ket);
                                        const float AQ2 =
                                            (A.x - Qx) * (A.x - Qx) +
                                            (A.y - Qy) * (A.y - Qy) +
                                            (A.z - Qz) * (A.z - Qz);
                                        const float T_val = alpha_pq * AQ2;

                                        const int L_tot = lP + lmu + lnu;

                                        double F_vals[RI_GRAD_R_BASE];
                                        compute_boys_double_host(
                                            F_vals, T_val, L_tot + 1);
                                        float AQ[3] = {A.x - Qx, A.y - Qy,
                                                       A.z - Qz};
                                        float R_vals
                                            [RI_GRAD_R_BASE * RI_GRAD_R_BASE *
                                             RI_GRAD_R_BASE * RI_GRAD_R_BASE];
                                        compute_r_tensor_host(
                                            R_vals, F_vals, alpha_pq, AQ,
                                            L_tot + 1);

                                        const double prefactor =
                                            (double)cP * (double)c_mu *
                                            (double)c_nu * (double)K_ket *
                                            (2.0 * M_PI * M_PI * sqrt(M_PI)) /
                                            ((double)eP * (double)g_ket *
                                             sqrt((double)(eP + g_ket)));

                                        // 收缩函数: 给定 bra 和 ket 角动量，求 E*E*R 乘积
                                        auto contract_3c =
                                            [&](int axP, int ayP, int azP,
                                                int ax_mu, int ay_mu, int az_mu,
                                                int ax_nu, int ay_nu,
                                                int az_nu) -> double
                                        {
                                            if (axP < 0 || ayP < 0 ||
                                                azP < 0 || ax_mu < 0 ||
                                                ay_mu < 0 || az_mu < 0 ||
                                                ax_nu < 0 || ay_nu < 0 ||
                                                az_nu < 0)
                                                return 0.0;
                                            double v_sum = 0.0;
                                            for (int t = 0; t <= axP; t++)
                                            {
                                                double ePx =
                                                    (double)E_Px[axP][0][t];
                                                if (ePx == 0.0) continue;
                                                for (int u = 0; u <= ayP; u++)
                                                {
                                                    double ePy =
                                                        (double)
                                                            E_Py[ayP][0][u];
                                                    if (ePy == 0.0) continue;
                                                    for (int v = 0; v <= azP;
                                                         v++)
                                                    {
                                                        double ePz =
                                                            (double)E_Pz[azP]
                                                                        [0][v];
                                                        if (ePz == 0.0)
                                                            continue;
                                                        for (int tt = 0;
                                                             tt <=
                                                             ax_mu + ax_nu;
                                                             tt++)
                                                        {
                                                            double eKx =
                                                                (double)E_Kx
                                                                    [ax_mu]
                                                                    [ax_nu]
                                                                    [tt];
                                                            if (eKx == 0.0)
                                                                continue;
                                                            for (int uu = 0;
                                                                 uu <=
                                                                 ay_mu + ay_nu;
                                                                 uu++)
                                                            {
                                                                double eKy =
                                                                    (double)
                                                                        E_Ky
                                                                            [ay_mu]
                                                                            [ay_nu]
                                                                            [uu];
                                                                if (eKy == 0.0)
                                                                    continue;
                                                                for (int vv =
                                                                         0;
                                                                     vv <=
                                                                     az_mu +
                                                                         az_nu;
                                                                     vv++)
                                                                {
                                                                    double
                                                                        eKz =
                                                                            (double)
                                                                                E_Kz
                                                                                    [az_mu]
                                                                                    [az_nu]
                                                                                    [vv];
                                                                    if (eKz ==
                                                                        0.0)
                                                                        continue;
                                                                    double
                                                                        sign =
                                                                            ((tt +
                                                                              uu +
                                                                              vv) &
                                                                             1)
                                                                                ? -1.0
                                                                                : 1.0;
                                                                    v_sum +=
                                                                        ePx *
                                                                        ePy *
                                                                        ePz *
                                                                        eKx *
                                                                        eKy *
                                                                        eKz *
                                                                        sign *
                                                                        (double)
                                                                            R_vals
                                                                                [RI_GRAD_R_IDX(
                                                                                    t +
                                                                                        tt,
                                                                                    u +
                                                                                        uu,
                                                                                    v +
                                                                                        vv,
                                                                                    0)];
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                            return v_sum;
                                        };

                                        // d(P|μν)/dA_P: 对 bra P 的 E 系数求导
                                        {
                                            double dx =
                                                2.0 * (double)eP *
                                                contract_3c(
                                                    lxP + 1, lyP, lzP, lx_mu,
                                                    ly_mu, lz_mu, lx_nu, ly_nu,
                                                    lz_nu);
                                            if (lxP > 0)
                                                dx -= (double)lxP *
                                                      contract_3c(
                                                          lxP - 1, lyP, lzP,
                                                          lx_mu, ly_mu, lz_mu,
                                                          lx_nu, ly_nu, lz_nu);
                                            double dy =
                                                2.0 * (double)eP *
                                                contract_3c(
                                                    lxP, lyP + 1, lzP, lx_mu,
                                                    ly_mu, lz_mu, lx_nu, ly_nu,
                                                    lz_nu);
                                            if (lyP > 0)
                                                dy -= (double)lyP *
                                                      contract_3c(
                                                          lxP, lyP - 1, lzP,
                                                          lx_mu, ly_mu, lz_mu,
                                                          lx_nu, ly_nu, lz_nu);
                                            double dz =
                                                2.0 * (double)eP *
                                                contract_3c(
                                                    lxP, lyP, lzP + 1, lx_mu,
                                                    ly_mu, lz_mu, lx_nu, ly_nu,
                                                    lz_nu);
                                            if (lzP > 0)
                                                dz -= (double)lzP *
                                                      contract_3c(
                                                          lxP, lyP, lzP - 1,
                                                          lx_mu, ly_mu, lz_mu,
                                                          lx_nu, ly_nu, lz_nu);
                                            dAP[0] += prefactor * dx;
                                            dAP[1] += prefactor * dy;
                                            dAP[2] += prefactor * dz;
                                        }

                                        // d(P|μν)/dA_mu: 对 ket 中 μ 的 E 系数求导
                                        {
                                            double dx =
                                                2.0 * (double)e_mu *
                                                contract_3c(
                                                    lxP, lyP, lzP, lx_mu + 1,
                                                    ly_mu, lz_mu, lx_nu, ly_nu,
                                                    lz_nu);
                                            if (lx_mu > 0)
                                                dx -=
                                                    (double)lx_mu *
                                                    contract_3c(
                                                        lxP, lyP, lzP,
                                                        lx_mu - 1, ly_mu,
                                                        lz_mu, lx_nu, ly_nu,
                                                        lz_nu);
                                            double dy =
                                                2.0 * (double)e_mu *
                                                contract_3c(
                                                    lxP, lyP, lzP, lx_mu,
                                                    ly_mu + 1, lz_mu, lx_nu,
                                                    ly_nu, lz_nu);
                                            if (ly_mu > 0)
                                                dy -=
                                                    (double)ly_mu *
                                                    contract_3c(
                                                        lxP, lyP, lzP, lx_mu,
                                                        ly_mu - 1, lz_mu,
                                                        lx_nu, ly_nu, lz_nu);
                                            double dz =
                                                2.0 * (double)e_mu *
                                                contract_3c(
                                                    lxP, lyP, lzP, lx_mu,
                                                    ly_mu, lz_mu + 1, lx_nu,
                                                    ly_nu, lz_nu);
                                            if (lz_mu > 0)
                                                dz -=
                                                    (double)lz_mu *
                                                    contract_3c(
                                                        lxP, lyP, lzP, lx_mu,
                                                        ly_mu, lz_mu - 1,
                                                        lx_nu, ly_nu, lz_nu);
                                            dAmu[0] += prefactor * dx;
                                            dAmu[1] += prefactor * dy;
                                            dAmu[2] += prefactor * dz;
                                        }
                                    }
                                }
                            }

                            const size_t cidx =
                                ((size_t)idxP * nmu_cart * nnu_cart +
                                 (size_t)idx_mu * nnu_cart + idx_nu);
                            d_cart_P[cidx * 3 + 0] = dAP[0];
                            d_cart_P[cidx * 3 + 1] = dAP[1];
                            d_cart_P[cidx * 3 + 2] = dAP[2];
                            d_cart_mu[cidx * 3 + 0] = dAmu[0];
                            d_cart_mu[cidx * 3 + 1] = dAmu[1];
                            d_cart_mu[cidx * 3 + 2] = dAmu[2];
                        }
                    }
                }

                // Cart2Sph 变换 + 归一化 + 与 D3_eff 收缩
                for (int ps = 0; ps < nP_sph; ps++)
                {
                    const int P_sph = offP_sph + ps;
                    const double normP_val = (double)aux_norms[P_sph];

                    for (int ms = 0; ms < nmu_sph; ms++)
                    {
                        const int mu_sph = offmu_sph + ms;
                        const double normMu = (double)orb_norms[mu_sph];

                        for (int ns = 0; ns < nnu_sph; ns++)
                        {
                            const int nu_sph = offnu_sph + ns;
                            const double normNu = (double)orb_norms[nu_sph];

                            // Cart2sph 变换
                            double d_sph_P[3] = {0.0, 0.0, 0.0};
                            double d_sph_mu[3] = {0.0, 0.0, 0.0};

                            for (int pc = 0; pc < nP_cart; pc++)
                            {
                                double u_p = (double)U_aux[(offP_cart + pc) *
                                                               naux_sph +
                                                           P_sph];
                                if (u_p == 0.0) continue;

                                for (int mc = 0; mc < nmu_cart; mc++)
                                {
                                    double u_m = is_spherical
                                        ? (double)U_orb[(offmu_cart + mc) *
                                                            nao_sph +
                                                        mu_sph]
                                        : (mc == ms ? 1.0 : 0.0);
                                    if (u_m == 0.0) continue;

                                    for (int nc = 0; nc < nnu_cart; nc++)
                                    {
                                        double u_n = is_spherical
                                            ? (double)U_orb[(offnu_cart + nc) *
                                                                nao_sph +
                                                            nu_sph]
                                            : (nc == ns ? 1.0 : 0.0);
                                        if (u_n == 0.0) continue;

                                        double w = u_p * u_m * u_n;
                                        const size_t cidx =
                                            ((size_t)pc * nmu_cart * nnu_cart +
                                             (size_t)mc * nnu_cart + nc);
                                        d_sph_P[0] +=
                                            w * d_cart_P[cidx * 3 + 0];
                                        d_sph_P[1] +=
                                            w * d_cart_P[cidx * 3 + 1];
                                        d_sph_P[2] +=
                                            w * d_cart_P[cidx * 3 + 2];
                                        d_sph_mu[0] +=
                                            w * d_cart_mu[cidx * 3 + 0];
                                        d_sph_mu[1] +=
                                            w * d_cart_mu[cidx * 3 + 1];
                                        d_sph_mu[2] +=
                                            w * d_cart_mu[cidx * 3 + 2];
                                    }
                                }
                            }

                            // 乘归一化
                            double norm_all = normP_val * normMu * normNu;
                            for (int d = 0; d < 3; d++)
                            {
                                d_sph_P[d] *= norm_all;
                                d_sph_mu[d] *= norm_all;
                            }

                            // 与 D3_eff 收缩
                            // D3_eff[P, mu, nu] 索引
                            double dens =
                                D3_eff[(long long)P_sph * nao * nao +
                                       (long long)mu_sph * nao + nu_sph];
                            // 对称 shell pair: 还需加上 D3_eff[P, nu, mu]
                            if (mu_sh != nu_sh)
                                dens +=
                                    D3_eff[(long long)P_sph * nao * nao +
                                           (long long)nu_sph * nao + mu_sph];

                            // d/dA_nu = -(d/dA_P + d/dA_mu) (平移不变性)
                            for (int d = 0; d < 3; d++)
                            {
                                double g_P = dens * d_sph_P[d];
                                double g_mu = dens * d_sph_mu[d];
                                double g_nu = -(g_P + g_mu);

                                grad[atom_P * 3 + d] += g_P;
                                grad[atom_mu * 3 + d] += g_mu;
                                grad[atom_nu * 3 + d] += g_nu;
                            }
                        }
                    }
                }
            }
        }
    }
}

#endif // USE_GPU

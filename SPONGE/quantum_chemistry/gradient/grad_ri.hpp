#pragma once

// RI (Density Fitting) 解析梯度驱动
//
// RI-J 梯度:
//   dE_J/dR = Σ_{P,μν} g_P D_μν d(P|μν)/dR
//           - (1/2) Σ_{PQ} g_P g_Q d(P|Q)/dR
//
// RI-K 梯度:
//   3c 部分: dE_K/dR|_{3c} = -exx Σ_{Q,μν} D3_K[Q,μν] d(Q|μν)/dR
//     D3_K[Q,μ,λ] = Σ_{P,ν,i} M^{-1/2}[QP] B_occ[P,ν,i] C[λ,i] D[μ,ν]
//   2c 部分: dE_K/dR|_{2c} = Σ_{PQ} D2_K[P,Q] d(P|Q)/dR
//     D2_K = U (F ⊙ (U^T Z_K U)) U^T   (Daleckii-Kreĭn 矩阵函数导数)
//     Z_K[P,Q] = -Σ_{m,n,l,i} (Q|ml) C[l,i] B_occ[P,n,i] D[m,n]
//     F[k,l] = { -1/2 λ_k^{-3/2}                    if k=l
//              { (λ_k^{-1/2} - λ_l^{-1/2})/(λ_k-λ_l) if k≠l

#include "../integrals/ri/ri_2center_grad.hpp"
#include "../integrals/ri/ri_3center_grad.hpp"

#include <cmath>
#include <cstring>
#include <vector>

#ifndef USE_GPU

// RI 梯度构建主函数 (stored 模式)
static inline void QC_Build_RI_Gradient_Stored(
    // 分子与基组信息
    const int natm, const int nao, const int nao_cart,
    const int naux, const int naux_cart,
    const int naux_bas, const int norb_bas,
    const bool is_spherical,
    // 辅助基 host 数据
    const VECTOR* aux_centers, const int* aux_l_list,
    const float* aux_exps, const float* aux_coeffs,
    const int* aux_shell_offsets, const int* aux_shell_sizes,
    const int* aux_ao_offsets_cart, const int* aux_ao_offsets_sph,
    const float* aux_norms,
    const float* U_aux,
    // 轨道基 host 数据
    const VECTOR* orb_centers, const int* orb_l_list,
    const float* orb_exps, const float* orb_coeffs,
    const int* orb_shell_offsets, const int* orb_shell_sizes,
    const int* orb_ao_offsets_cart, const int* orb_ao_offsets_sph,
    const float* orb_norms,
    const float* U_orb,
    // SCF 数据 (host)
    const double* metric_inv_sqrt,   // [naux × naux]
    const double* eri3c,             // [naux × nao × nao]
    const double* g_vec,             // [naux] RI-J 拟合系数
    const float* P_density,          // [nao × nao] 密度矩阵 (Coulomb)
    const float* B_occ,              // [naux * nao × nocc] 列优先
    const float* C_occ,              // [nao × nao] MO 系数 (行优先: C[ao*nao+mo])
    const int nocc,
    const float exx_fraction,
    // 特征分解数据 (用于 D2_K 的矩阵函数导数)
    const double* eigval,            // [naux] 特征值
    const double* eigvec,            // [naux × naux] 列优先特征向量
    const int naux_eff,              // 有效特征值个数
    // 壳层到原子映射
    const int* shell_atom_aux,       // [naux_bas]
    const int* shell_atom_orb,       // [norb_bas]
    // 输出
    double* grad)
{
    const long long nao2 = (long long)nao * nao;
    const long long naux2 = (long long)naux * naux;

    // ---- 1. 构建 D3_eff (三中心有效密度) ----
    std::vector<double> D3_eff((size_t)naux * nao2, 0.0);

    // D3_J[P, μ, ν] = g_P * D_μν
    for (int P = 0; P < naux; P++)
    {
        const double gP = g_vec[P];
        for (int mu = 0; mu < nao; mu++)
            for (int nu = 0; nu < nao; nu++)
                D3_eff[(long long)P * nao2 + (long long)mu * nao + nu] =
                    gP * (double)P_density[mu * nao + nu];
    }

    // D3_K[Q,μ,λ] = Σ_{P,ν,i} M^{-1/2}[QP] B_occ[P,ν,i] C[λ,i] D[μ,ν]
    if (exx_fraction != 0.0f && nocc > 0 && B_occ != nullptr)
    {
        const int M_dim = naux * nao;  // B_occ 的 leading dimension

        // Step 1: X[P,ν,λ] = Σ_i B_occ[P,ν,i] * C[λ,i]
        // Step 2: Y[P,μ,λ] = Σ_ν D[μ,ν] * X[P,ν,λ]
        // Step 3: D3_K[Q,μ,λ] = Σ_P M^{-1/2}[QP] * Y[P,μ,λ]
        std::vector<double> Y((size_t)naux * nao2, 0.0);
        for (int P = 0; P < naux; P++)
        {
            // X_P[ν,λ] = Σ_i B_occ[P,ν,i] * C[λ,i]
            for (int nu = 0; nu < nao; nu++)
            {
                for (int i = 0; i < nocc; i++)
                {
                    // B_occ 列优先: B_occ[(P*nao+nu) + M_dim*i]
                    double bval =
                        (double)B_occ[(size_t)(P * nao + nu) + (size_t)M_dim * i];
                    if (bval == 0.0) continue;
                    for (int lam = 0; lam < nao; lam++)
                    {
                        // C 行优先: C[lam, i] = C_occ[lam*nao + i]
                        double c = (double)C_occ[lam * nao + i];
                        // 累加到 X_P[nu, lam]，但直接与 D 收缩到 Y
                        // Y[P,mu,lam] += D[mu,nu] * X_P[nu,lam]
                        for (int mu = 0; mu < nao; mu++)
                        {
                            Y[(long long)P * nao2 + (long long)mu * nao + lam] +=
                                (double)P_density[mu * nao + nu] * bval * c;
                        }
                    }
                }
            }
        }

        // D3_K[Q,μ,λ] = Σ_P M^{-1/2}[QP] * Y[P,μ,λ]
        // D3_eff -= exx * D3_K  (不需要对称化，因为 kernel 会遍历全 μν)
        const double neg_exx = -(double)exx_fraction;
        for (int Q = 0; Q < naux; Q++)
        {
            for (int P = 0; P < naux; P++)
            {
                double s = metric_inv_sqrt[(size_t)Q * naux + P];
                if (s == 0.0) continue;
                double w = neg_exx * s;
                for (long long mn = 0; mn < nao2; mn++)
                    D3_eff[(long long)Q * nao2 + mn] +=
                        w * Y[(long long)P * nao2 + mn];
            }
        }
    }

    // ---- 2. 构建 D2_eff (二中心有效密度) ----
    std::vector<double> D2_eff((size_t)naux2, 0.0);

    // D2_J[P,Q] = -(1/2) g_P * g_Q
    for (int P = 0; P < naux; P++)
        for (int Q = 0; Q < naux; Q++)
            D2_eff[(size_t)P * naux + Q] = -0.5 * g_vec[P] * g_vec[Q];

    // D2_K: Daleckii-Kreĭn 矩阵函数导数
    // Z_K[P',Q'] = -Σ_{m,n,l,i} (Q'|ml) C[l,i] B_occ[P',n,i] D[m,n]
    // D2_K = U (F ⊙ (U^T Z_K U)) U^T
    if (exx_fraction != 0.0f && nocc > 0 && B_occ != nullptr &&
        eigval != nullptr && eigvec != nullptr)
    {
        const int M_dim = naux * nao;
        const int n_skip = naux - naux_eff;

        // Step 1: 计算 Z_K[P,Q]
        std::vector<double> Z_K((size_t)naux2, 0.0);
        for (int Pp = 0; Pp < naux; Pp++)
        {
            for (int Qp = 0; Qp < naux; Qp++)
            {
                double z = 0.0;
                // Z_K[P',Q'] = -Σ_{m,l,i} [Σ_n (Q'|ml) D[m,n]] [B_occ[P',n,i] C[l,i]]
                // 但这样嵌套太深，先 precompute eri3c_D[Q,l,n] = Σ_m (Q|ml) D[mn]
                // ... 太慢了，naux^2 * nao^3。用更聪明的方式。
                for (int m = 0; m < nao; m++)
                {
                    for (int l = 0; l < nao; l++)
                    {
                        double ql = eri3c[(long long)Qp * nao2 +
                                          (long long)m * nao + l];
                        if (ql == 0.0) continue;
                        for (int i = 0; i < nocc; i++)
                        {
                            double c = (double)C_occ[l * nao + i];
                            for (int n = 0; n < nao; n++)
                            {
                                z -= ql * c *
                                     (double)B_occ[(size_t)(Pp * nao + n) +
                                                   (size_t)M_dim * i] *
                                     (double)P_density[m * nao + n];
                            }
                        }
                    }
                }
                Z_K[(size_t)Pp * naux + Qp] = z;
            }
        }

        // Step 2: U^T Z_K U
        std::vector<double> UZU((size_t)naux2, 0.0);
        // tmp = Z_K @ U
        std::vector<double> tmp((size_t)naux2, 0.0);
        for (int i = 0; i < naux; i++)
            for (int j = 0; j < naux; j++)
                for (int k = 0; k < naux; k++)
                    tmp[(size_t)i * naux + j] +=
                        Z_K[(size_t)i * naux + k] *
                        eigvec[(size_t)k + (size_t)j * naux];
        // UZU = U^T @ tmp
        for (int i = 0; i < naux; i++)
            for (int j = 0; j < naux; j++)
                for (int k = 0; k < naux; k++)
                    UZU[(size_t)i * naux + j] +=
                        eigvec[(size_t)k + (size_t)i * naux] *
                        tmp[(size_t)k * naux + j];

        // Step 3: F ⊙ UZU (Hadamard 积)
        for (int k = 0; k < naux; k++)
        {
            for (int l = 0; l < naux; l++)
            {
                double f;
                if (k == l)
                    f = -0.5 * pow(eigval[k], -1.5);
                else
                    f = (pow(eigval[k], -0.5) - pow(eigval[l], -0.5)) /
                        (eigval[k] - eigval[l]);
                UZU[(size_t)k * naux + l] *= f;
            }
        }

        // Step 4: D2_K = U @ (F⊙UZU) @ U^T
        // tmp2 = (F⊙UZU) @ U^T
        std::fill(tmp.begin(), tmp.end(), 0.0);
        for (int i = 0; i < naux; i++)
            for (int j = 0; j < naux; j++)
                for (int k = 0; k < naux; k++)
                    tmp[(size_t)i * naux + j] +=
                        UZU[(size_t)i * naux + k] *
                        eigvec[(size_t)j + (size_t)k * naux];
        // D2_K = U @ tmp2
        for (int i = 0; i < naux; i++)
            for (int j = 0; j < naux; j++)
                for (int k = 0; k < naux; k++)
                    D2_eff[(size_t)i * naux + j] +=
                        eigvec[(size_t)i + (size_t)k * naux] *
                        tmp[(size_t)k * naux + j];
    }

    // ---- 3. 调用二中心导数内核 ----
    QC_RI_2Center_Grad_CPU(
        naux_bas, aux_centers, aux_l_list, aux_exps, aux_coeffs,
        aux_shell_offsets, aux_shell_sizes, aux_ao_offsets_cart,
        aux_ao_offsets_sph, aux_norms, U_aux, naux_cart, naux,
        D2_eff.data(), shell_atom_aux, grad);

    // ---- 4. 调用三中心导数内核 ----
    QC_RI_3Center_Grad_CPU(
        naux_bas, norb_bas, aux_centers, aux_l_list, aux_exps, aux_coeffs,
        aux_shell_offsets, aux_shell_sizes, aux_ao_offsets_cart,
        aux_ao_offsets_sph, orb_centers, orb_l_list, orb_exps, orb_coeffs,
        orb_shell_offsets, orb_shell_sizes, orb_ao_offsets_cart,
        orb_ao_offsets_sph, is_spherical, aux_norms, orb_norms, U_aux, U_orb,
        naux_cart, naux, nao_cart, nao, D3_eff.data(), shell_atom_aux,
        shell_atom_orb, grad);
}

#endif // USE_GPU

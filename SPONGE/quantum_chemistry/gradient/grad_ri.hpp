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

// ---- 共用辅助函数 ----

// 构建 D3_eff = D3_J - exx * D3_K (三中心有效密度)
static inline void QC_Build_D3_eff(
    const int nao, const int naux,
    const double* g_vec,
    const float* P_density,
    const double* metric_inv_sqrt,
    const float* B_occ,    // [M × nocc] 列优先, M = naux*nao (或 NULL)
    const float* C_occ,    // [nao × nao] 行优先 (或 NULL)
    const int nocc,
    const float exx_fraction,
    std::vector<double>& D3_eff)
{
    const long long nao2 = (long long)nao * nao;

    D3_eff.assign((size_t)naux * nao2, 0.0);

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
        const int M_dim = naux * nao;

        std::vector<double> Y((size_t)naux * nao2, 0.0);
        for (int P = 0; P < naux; P++)
        {
            for (int nu = 0; nu < nao; nu++)
            {
                for (int i = 0; i < nocc; i++)
                {
                    double bval =
                        (double)B_occ[(size_t)(P * nao + nu) + (size_t)M_dim * i];
                    if (bval == 0.0) continue;
                    for (int lam = 0; lam < nao; lam++)
                    {
                        double c = (double)C_occ[lam * nao + i];
                        for (int mu = 0; mu < nao; mu++)
                        {
                            Y[(long long)P * nao2 + (long long)mu * nao + lam] +=
                                (double)P_density[mu * nao + nu] * bval * c;
                        }
                    }
                }
            }
        }

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
}

// D2_J 初始化: D2_eff[P,Q] = -0.5 * g_P * g_Q
static inline void QC_Init_D2_J(
    const int naux, const double* g_vec,
    std::vector<double>& D2_eff)
{
    const size_t naux2 = (size_t)naux * naux;
    D2_eff.assign(naux2, 0.0);
    for (int P = 0; P < naux; P++)
        for (int Q = 0; Q < naux; Q++)
            D2_eff[(size_t)P * naux + Q] = -0.5 * g_vec[P] * g_vec[Q];
}

// D2_K 的 Daleckii-Kreĭn 矩阵函数导数: D2_K = U (F ⊙ (U^T Z_K U)) U^T
// 结果累加到 D2_eff。
static inline void QC_Build_D2K_DaleckiiKrein(
    const int naux,
    const double* Z_K,
    const double* eigval,
    const double* eigvec,
    std::vector<double>& D2_eff)
{
    const size_t naux2 = (size_t)naux * naux;

    // U^T Z_K U
    std::vector<double> UZU(naux2, 0.0);
    std::vector<double> tmp(naux2, 0.0);
    // tmp = Z_K @ U
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

    // F ⊙ UZU (Hadamard 积)
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

    // D2_K = U @ (F⊙UZU) @ U^T
    std::fill(tmp.begin(), tmp.end(), 0.0);
    for (int i = 0; i < naux; i++)
        for (int j = 0; j < naux; j++)
            for (int k = 0; k < naux; k++)
                tmp[(size_t)i * naux + j] +=
                    UZU[(size_t)i * naux + k] *
                    eigvec[(size_t)j + (size_t)k * naux];
    for (int i = 0; i < naux; i++)
        for (int j = 0; j < naux; j++)
            for (int k = 0; k < naux; k++)
                D2_eff[(size_t)i * naux + j] +=
                    eigvec[(size_t)i + (size_t)k * naux] *
                    tmp[(size_t)k * naux + j];
}

// 上传 D2/D3 有效密度到 device，分配 workspace，启动 2c/3c 梯度内核。
// 基组指针均为 device 指针（已在 SCF 初始化时分配）。
// 仅 h_U_aux/h_U_orb 和 D2/D3 需要临时上传。
static inline void QC_Launch_RI_Grad_Kernels(
    const int naux_bas, const int norb_bas, const bool is_spherical,
    const int naux_cart, const int naux, const int nao_cart, const int nao,
    const int max_aux_cart, const int max_orb_cart,
    // 辅助基 (device 指针)
    const VECTOR* d_aux_centers, const int* d_aux_l_list,
    const float* d_aux_exps, const float* d_aux_coeffs,
    const int* d_aux_shell_offsets, const int* d_aux_shell_sizes,
    const int* d_aux_ao_offsets_cart, const int* d_aux_ao_offsets_sph,
    const float* d_aux_norms,
    // 轨道基 (device 指针)
    const VECTOR* d_orb_centers, const int* d_orb_l_list,
    const float* d_orb_exps, const float* d_orb_coeffs,
    const int* d_orb_shell_offsets, const int* d_orb_shell_sizes,
    const int* d_orb_ao_offsets_cart, const int* d_orb_ao_offsets_sph,
    const float* d_orb_norms,
    // cart2sph 矩阵 (host, 需临时上传)
    const float* h_U_aux, const float* h_U_orb,
    // shell-atom 映射 (device 指针)
    const int* d_shell_atom_aux, const int* d_shell_atom_orb,
    // 有效密度 (host, 需临时上传)
    const std::vector<double>& D2_eff, const std::vector<double>& D3_eff,
    // 输出 (device 指针)
    double* d_grad)
{
    const int threads = 64;

    // 上传 cart2sph 矩阵（host only, 无 device 副本）
    float* d_U_aux = NULL;
    {
        const size_t n = (size_t)naux_cart * naux;
        if (n > 0 && h_U_aux != NULL)
        {
            Device_Malloc_Safely((void**)&d_U_aux, sizeof(float) * n);
            deviceMemcpy(d_U_aux, h_U_aux, sizeof(float) * n,
                         deviceMemcpyHostToDevice);
        }
    }
    float* d_U_orb = NULL;
    {
        const size_t n = (h_U_orb != NULL) ? (size_t)nao_cart * nao : 0;
        if (n > 0)
        {
            Device_Malloc_Safely((void**)&d_U_orb, sizeof(float) * n);
            deviceMemcpy(d_U_orb, h_U_orb, sizeof(float) * n,
                         deviceMemcpyHostToDevice);
        }
    }

    // ---- 2c 梯度内核 ----
    double* d_D2 = NULL;
    Device_Malloc_Safely((void**)&d_D2, sizeof(double) * D2_eff.size());
    deviceMemcpy(d_D2, D2_eff.data(), sizeof(double) * D2_eff.size(),
                 deviceMemcpyHostToDevice);

    const int ws_2c = max_aux_cart * max_aux_cart * 3;
    const int grid_2c = (naux_bas + threads - 1) / threads;
    double* d_ws_2c = NULL;
    Device_Malloc_Safely((void**)&d_ws_2c,
                         sizeof(double) * (size_t)naux_bas * ws_2c);

    Launch_Device_Kernel(
        QC_RI_2Center_Grad_Kernel, grid_2c, threads, 0, 0,
        naux_bas, d_aux_centers, d_aux_l_list, d_aux_exps, d_aux_coeffs,
        d_aux_shell_offsets, d_aux_shell_sizes, d_aux_ao_offsets_cart,
        d_aux_ao_offsets_sph, d_aux_norms, d_U_aux, naux_cart, naux,
        d_D2, d_shell_atom_aux, d_ws_2c, ws_2c, naux_bas, d_grad);

    deviceFree(d_ws_2c);
    deviceFree(d_D2);

    // ---- 3c 梯度内核 ----
    double* d_D3 = NULL;
    Device_Malloc_Safely((void**)&d_D3, sizeof(double) * D3_eff.size());
    deviceMemcpy(d_D3, D3_eff.data(), sizeof(double) * D3_eff.size(),
                 deviceMemcpyHostToDevice);

    const int half_ws_3c = max_aux_cart * max_orb_cart * max_orb_cart * 3;
    const int ws_3c = half_ws_3c * 2;
    const int grid_3c = (naux_bas + threads - 1) / threads;
    double* d_ws_3c = NULL;
    Device_Malloc_Safely((void**)&d_ws_3c,
                         sizeof(double) * (size_t)naux_bas * ws_3c);

    Launch_Device_Kernel(
        QC_RI_3Center_Grad_Kernel, grid_3c, threads, 0, 0,
        naux_bas, norb_bas, d_aux_centers, d_aux_l_list, d_aux_exps,
        d_aux_coeffs, d_aux_shell_offsets, d_aux_shell_sizes,
        d_aux_ao_offsets_cart, d_aux_ao_offsets_sph,
        d_orb_centers, d_orb_l_list, d_orb_exps, d_orb_coeffs,
        d_orb_shell_offsets, d_orb_shell_sizes, d_orb_ao_offsets_cart,
        d_orb_ao_offsets_sph, (int)is_spherical, d_aux_norms, d_orb_norms,
        d_U_aux, d_U_orb, naux_cart, naux, nao_cart, nao,
        d_D3, d_shell_atom_aux, d_shell_atom_orb,
        d_ws_3c, ws_3c, naux_bas, d_grad);

    deviceFree(d_ws_3c);
    deviceFree(d_D3);
    deviceFree(d_U_aux);
    deviceFree(d_U_orb);
}

// D2_eff 构建辅助: D2_J + D2_K (从 eri3c 计算 Z_K 再做 Daleckii-Kreĭn)
static inline void QC_Build_D2_eff_Stored(
    const int nao, const int naux,
    const double* g_vec, const double* eri3c,
    const float* P_density, const float* B_occ, const float* C_occ,
    const int nocc, const float exx_fraction,
    const double* eigval, const double* eigvec,
    std::vector<double>& D2_eff)
{
    const long long nao2 = (long long)nao * nao;
    const size_t naux2 = (size_t)naux * naux;

    QC_Init_D2_J(naux, g_vec, D2_eff);

    if (exx_fraction != 0.0f && nocc > 0 && B_occ != nullptr &&
        eigval != nullptr && eigvec != nullptr)
    {
        const int M_dim = naux * nao;
        std::vector<double> Z_K(naux2, 0.0);
        for (int Pp = 0; Pp < naux; Pp++)
        {
            for (int Qp = 0; Qp < naux; Qp++)
            {
                double z = 0.0;
                for (int m = 0; m < nao; m++)
                    for (int l = 0; l < nao; l++)
                    {
                        double ql = eri3c[(long long)Qp * nao2 +
                                          (long long)m * nao + l];
                        if (ql == 0.0) continue;
                        for (int i = 0; i < nocc; i++)
                        {
                            double c = (double)C_occ[l * nao + i];
                            for (int n = 0; n < nao; n++)
                                z -= ql * c *
                                     (double)B_occ[(size_t)(Pp * nao + n) +
                                                   (size_t)M_dim * i] *
                                     (double)P_density[m * nao + n];
                        }
                    }
                Z_K[(size_t)Pp * naux + Qp] = z;
            }
        }
        QC_Build_D2K_DaleckiiKrein(naux, Z_K.data(), eigval, eigvec, D2_eff);
    }
}

// D2_eff 构建辅助: D2_J + D2_K (从预累积 Z_K)
static inline void QC_Build_D2_eff_FromZK(
    const int naux,
    const double* g_vec,
    const float* B_occ, const int nocc, const float exx_fraction,
    const double* Z_K, const double* eigval, const double* eigvec,
    std::vector<double>& D2_eff)
{
    QC_Init_D2_J(naux, g_vec, D2_eff);

    if (exx_fraction != 0.0f && nocc > 0 && B_occ != nullptr &&
        Z_K != nullptr && eigval != nullptr && eigvec != nullptr)
        QC_Build_D2K_DaleckiiKrein(naux, Z_K, eigval, eigvec, D2_eff);
}

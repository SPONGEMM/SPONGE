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

#include <cmath>
#include <cstring>
#include <vector>

#include "../integrals/ri/ri_2center_grad.hpp"
#include "../integrals/ri/ri_3center_grad.hpp"

// cblas.h 已通过 device_backend/cpu_api.h 引入，此处仅做特性检测
#if !defined(USE_GPU) && (defined(USE_MKL) || defined(USE_OPENBLAS))
#define RI_GRAD_HAS_BLAS 1
#else
#define RI_GRAD_HAS_BLAS 0
#endif

// 共用辅助函数

// 构建 D3_eff = D3_J - exx * D3_K (三中心有效密度)
static inline void QC_Build_D3_eff(
    const int nao, const int naux, const double* g_vec, const float* P_density,
    const double* metric_inv_sqrt,
    const float* B_occ,  // [M × nocc] 列优先, M = naux*nao (或 NULL)
    const float* C_occ,  // [nao × nao] 行优先 (或 NULL)
    const int nocc, const float exx_fraction, std::vector<double>& D3_eff)
{
    const long long nao2 = (long long)nao * nao;

    D3_eff.resize((size_t)naux * nao2);

    // D3_J[P, μν] = g_P * D_μν — 外积，P 在外层连续写入
    {
        // 先转换 D 为 double
        std::vector<double> D_d(nao2);
        for (int i = 0; i < nao2; i++) D_d[i] = (double)P_density[i];
        for (int P = 0; P < naux; P++)
        {
            const double gP = g_vec[P];
            double* dst = D3_eff.data() + (long long)P * nao2;
            for (long long mn = 0; mn < nao2; mn++) dst[mn] = gP * D_d[mn];
        }
    }

    // D3_K[Q,μ,λ] = Σ_{P,ν,i} M^{-1/2}[QP] B_occ[P,ν,i] C[λ,i] D[μ,ν]
    // 分解:
    //   X_P[ν,λ] = Σ_i B_occ[P,ν,i] × C[λ,i]   (per-P matmul)
    //   Y_P[μ,λ] = Σ_ν D[μ,ν] × X_P[ν,λ]       (per-P matmul)
    //   D3_K[Q,ml] = -exx × Σ_P M^{-1/2}[Q,P] × Y[P,ml]  (matmul)
    if (exx_fraction != 0.0f && nocc > 0 && B_occ != nullptr)
    {
        const int M_dim = naux * nao;
        const double neg_exx = -(double)exx_fraction;
        const double one_d = 1.0;
        const double zero_d = 0.0;

#if RI_GRAD_HAS_BLAS
        // BLAS 路径: 使用 cblas_dgemm 加速矩阵乘法

        // 准备 double 精度缓冲
        std::vector<double> D_d(nao2);
        for (long long i = 0; i < nao2; i++) D_d[i] = (double)P_density[i];

        // C_occ 前 nocc 列的 double 版本，行优先 [nao, nocc]
        std::vector<double> C_d((size_t)nao * nocc);
        for (int lam = 0; lam < nao; lam++)
            for (int i = 0; i < nocc; i++)
                C_d[lam * nocc + i] = (double)C_occ[lam * nao + i];

        // B_P_d: 行优先 [nao, nocc] 连续缓冲
        std::vector<double> B_P_d((size_t)nao * nocc);
        std::vector<double> X_P((size_t)nao * nao);
        std::vector<double> Y((size_t)naux * nao2, 0.0);

        for (int P = 0; P < naux; P++)
        {
            // 拷贝 B_P → 行优先 [nao, nocc]: B_P_d[ν*nocc+i]
            for (int i = 0; i < nocc; i++)
                for (int nu = 0; nu < nao; nu++)
                    B_P_d[(size_t)nu * nocc + i] = (double)
                        B_occ[(size_t)(P * nao + nu) + (size_t)M_dim * i];

            // X_P[ν,λ] = Σ_i B_P[ν,i] × C[λ,i]
            // 行优先: X_P(nao×nao) = B_P_d(nao×nocc) @ C_d^T(nocc×nao)
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, nao, nao, nocc,
                        1.0, B_P_d.data(), nocc, C_d.data(), nocc, 0.0,
                        X_P.data(), nao);

            // Y_P[μ,λ] = Σ_ν D[μ,ν] × X_P[ν,λ]
            // 行优先: Y_P(nao×nao) = D(nao×nao) @ X_P(nao×nao)
            double* Y_P = Y.data() + (long long)P * nao2;
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nao, nao,
                        nao, 1.0, D_d.data(), nao, X_P.data(), nao, 0.0, Y_P,
                        nao);
        }

        // Step 3: D3_eff += neg_exx × M^{-1/2} @ Y
        // 行优先: D3_eff[naux, nao²] += neg_exx * M[naux,naux] @ Y[naux,nao²]
        // 列优先视角: D3^T[nao²,naux] += neg_exx * Y^T[nao²,naux] *
        // M^T[naux,naux] M^{-1/2} 对称 → M^T = M
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, naux, (int)nao2,
                    naux, neg_exx, metric_inv_sqrt, naux, Y.data(), (int)nao2,
                    one_d, D3_eff.data(), (int)nao2);

#else
        // 回退: 标量循环 (无 BLAS 可用)
        std::vector<double> Y((size_t)naux * nao2, 0.0);
        std::vector<double> X_P((size_t)nao * nao, 0.0);
        for (int P = 0; P < naux; P++)
        {
            std::fill(X_P.begin(), X_P.end(), 0.0);
            for (int nu = 0; nu < nao; nu++)
                for (int i = 0; i < nocc; i++)
                {
                    double bval = (double)
                        B_occ[(size_t)(P * nao + nu) + (size_t)M_dim * i];
                    if (bval == 0.0) continue;
                    for (int lam = 0; lam < nao; lam++)
                        X_P[nu * nao + lam] +=
                            bval * (double)C_occ[lam * nao + i];
                }
            double* Y_P = Y.data() + (long long)P * nao2;
            for (int mu = 0; mu < nao; mu++)
                for (int nu = 0; nu < nao; nu++)
                {
                    double d = (double)P_density[mu * nao + nu];
                    if (d == 0.0) continue;
                    for (int lam = 0; lam < nao; lam++)
                        Y_P[mu * nao + lam] += d * X_P[nu * nao + lam];
                }
        }
        for (int Q = 0; Q < naux; Q++)
            for (int P = 0; P < naux; P++)
            {
                double w = neg_exx * metric_inv_sqrt[(size_t)Q * naux + P];
                if (w == 0.0) continue;
                for (long long mn = 0; mn < nao2; mn++)
                    D3_eff[(long long)Q * nao2 + mn] +=
                        w * Y[(long long)P * nao2 + mn];
            }
#endif
    }
}

// D2_J 初始化: D2_eff[P,Q] = -0.5 * g_P * g_Q
static inline void QC_Init_D2_J(const int naux, const double* g_vec,
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
static inline void QC_Build_D2K_DaleckiiKrein(const int naux, const double* Z_K,
                                              const double* eigval,
                                              const double* eigvec,
                                              std::vector<double>& D2_eff)
{
    const size_t naux2 = (size_t)naux * naux;

    std::vector<double> UZU(naux2, 0.0);
    std::vector<double> tmp(naux2, 0.0);

#if RI_GRAD_HAS_BLAS
    // BLAS 路径: U^T Z_K U → D2_K = U (F ⊙ UZU) U^T
    // Z_K 行优先 [naux,naux], eigvec 列优先 [naux,naux]

    // tmp = Z_K @ U: Z_K 行优先, U 列优先
    // 行优先 Z_K × 列优先 U: 用 cblas_dgemm(RowMajor, N, N, ...)
    // Z_K[i,k] * U_col[k,j] = Z_K[i,k] * eigvec[k + j*naux]
    // cblas_dgemm(CblasRowMajor, N, N, naux, naux, naux, 1, Z_K, naux,
    // eigvec(列→行等价T), naux, 0, tmp, naux) eigvec 列优先 [naux,naux]
    // 用行优先读 = eigvec^T 所以 Z_K @ U = Z_K(行) @ eigvec^T(行)^T → RowMajor,
    // N, N 不对 正确: tmp[i,j] = Σ_k Z_K[i,k] * U[k,j] U[k,j] = eigvec[k +
    // j*naux] (col-major) = eigvec_rowmajor^T → tmp = Z_K @ U, 其中 U 列优先
    // CblasRowMajor: C = A × B, A 行优先 [M,K], B 行优先 [K,N]
    // 但 U 是列优先, 行优先读就是 U^T, 需要转置标记
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, naux, naux, naux, 1.0,
                Z_K, naux, eigvec, naux, 0.0, tmp.data(), naux);
    // 上面: RowMajor, N, T → tmp[i,j] = Σ_k Z_K[i,k] * eigvec_row[j,k]
    //   = Σ_k Z_K[i,k] * eigvec[j*naux+k] ← 不对，应该是 eigvec[k+j*naux]
    // eigvec 列优先: eigvec[k + j*naux]
    // CblasRowMajor 把指针读为行优先: B_row[j,k] = eigvec[j*naux + k]
    // CblasTrans → B^T[k,j] = B_row[j,k] = eigvec[j*naux + k]
    // tmp[i,j] = Σ_k Z_K[i,k] * B^T[k,j] = Σ_k Z_K[i,k] * eigvec[j*naux+k]
    // 但我们要 Σ_k Z_K[i,k] * eigvec[k + j*naux]
    // eigvec[j*naux+k] = eigvec[k + j*naux] ← 只有 naux*j+k vs k+j*naux →
    // 是同一个东西！ ✓ 正确

    // UZU = U^T @ tmp: UZU[i,j] = Σ_k U[k,i] * tmp[k,j]
    //   = Σ_k eigvec[k + i*naux] * tmp[k,j]
    // RowMajor: U_row(行优先读 eigvec) = eigvec[i*naux+k], 即 U^T[i,k]
    // UZU = U_row @ tmp (RowMajor, NoTrans, NoTrans)
    // UZU[i,j] = Σ_k U_row[i,k] * tmp[k,j] = Σ_k eigvec[i*naux+k] * tmp[k,j]
    //   = Σ_k eigvec[k + i*naux] * tmp[k,j] ✓
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, naux, naux, naux,
                1.0, eigvec, naux, tmp.data(), naux, 0.0, UZU.data(), naux);

#else
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
#endif

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

#if RI_GRAD_HAS_BLAS
    // D2_K = U @ (F⊙UZU) @ U^T
    // tmp = (F⊙UZU) @ U^T: tmp[i,j] = Σ_k UZU[i,k] * U[j,k]^T
    //   = Σ_k UZU[i,k] * eigvec[j + k*naux]
    // RowMajor 读 eigvec: eigvec_row[k,j] = eigvec[k*naux+j]
    // tmp = UZU @ eigvec_row^T → RowMajor, N, T
    // tmp[i,j] = Σ_k UZU[i,k] * eigvec_row[j,k] = Σ_k UZU[i,k] *
    // eigvec[j*naux+k]
    //   = Σ_k UZU[i,k] * eigvec[k*naux+j]? No.
    // eigvec_row[j,k] = eigvec[j*naux+k] (RowMajor 解释)
    // 我们要 eigvec[j + k*naux] ← 不等于 eigvec[j*naux+k]
    // 用 NoTrans: tmp = UZU @ eigvec_row(NoTrans)
    // tmp[i,j] = Σ_k UZU[i,k] * eigvec_row[k,j] = Σ_k UZU[i,k] *
    // eigvec[k*naux+j]
    //   = Σ_k UZU[i,k] * eigvec[j + k*naux] ✓ (因为 k*naux+j = j + k*naux)
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, naux, naux, naux,
                1.0, UZU.data(), naux, eigvec, naux, 0.0, tmp.data(), naux);

    // D2_eff += U @ tmp: D2_eff[i,j] += Σ_k U[i,k] * tmp[k,j]
    //   = Σ_k eigvec[i + k*naux] * tmp[k,j]
    //   = Σ_k eigvec_row[k,i]^T * tmp[k,j]
    // D2_eff += eigvec_row^T @ tmp → RowMajor, Trans, NoTrans
    // 但 eigvec_row^T[i,k] = eigvec_row[k,i] = eigvec[k*naux+i] = eigvec[i +
    // k*naux] ✓
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, naux, naux, naux, 1.0,
                eigvec, naux, tmp.data(), naux, 1.0, D2_eff.data(), naux);
#else
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
#endif
}

// 上传 D2/D3 有效密度到 device，分配 workspace，启动 2c/3c 梯度内核。
// 基组指针均为 device 指针（已在 SCF 初始化时分配）。
// 仅 h_U_aux/h_U_orb 和 D2/D3 需要临时上传。
static inline void QC_Launch_RI_Grad_Kernels(
    const QC_MOLECULE& mol, const QC_RI_WORKSPACE& ri, const float* d_orb_norms,
    const QC_GRAD_WORKSPACE& grad_ws, int max_aux_cart, int max_orb_cart,
    const std::vector<double>& D2_eff, const std::vector<double>& D3_eff)
{
    // 局部别名: 保持函数体不变
    const int naux_bas = ri.naux_bas;
    const int norb_bas = mol.nbas;
    const bool is_spherical = mol.is_spherical;
    const int naux_cart = ri.naux_cart;
    const int naux = ri.naux;
    const int nao_cart = mol.nao_cart;
    const int nao = mol.nao;
    const VECTOR* d_aux_centers = ri.d_aux_centers;
    const int* d_aux_l_list = ri.d_aux_l_list;
    const float* d_aux_exps = ri.d_aux_exps;
    const float* d_aux_coeffs = ri.d_aux_coeffs;
    const int* d_aux_shell_offsets = ri.d_aux_shell_offsets;
    const int* d_aux_shell_sizes = ri.d_aux_shell_sizes;
    const int* d_aux_ao_offsets_cart = ri.d_aux_ao_offsets;
    const int* d_aux_ao_offsets_sph = ri.d_aux_ao_offsets_sph;
    const float* d_aux_norms = ri.d_aux_norms;
    const VECTOR* d_orb_centers = mol.d_centers;
    const int* d_orb_l_list = mol.d_l_list;
    const float* d_orb_exps = mol.d_exps;
    const float* d_orb_coeffs = mol.d_coeffs;
    const int* d_orb_shell_offsets = mol.d_shell_offsets;
    const int* d_orb_shell_sizes = mol.d_shell_sizes;
    const int* d_orb_ao_offsets_cart = mol.d_ao_offsets;
    const int* d_orb_ao_offsets_sph = mol.d_ao_offsets_sph;
    const float* h_U_aux = ri.h_U_aux.data();
    const float* h_U_orb = is_spherical ? ri.h_U_orb.data() : nullptr;
    const int* d_shell_atom_aux = grad_ws.d_shell_atom_aux;
    const int* d_shell_atom_orb = grad_ws.d_shell_atom;
    double* d_grad = grad_ws.d_grad;
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

    // 2c 梯度内核
    double* d_D2 = NULL;
    Device_Malloc_Safely((void**)&d_D2, sizeof(double) * D2_eff.size());
    deviceMemcpy(d_D2, D2_eff.data(), sizeof(double) * D2_eff.size(),
                 deviceMemcpyHostToDevice);

    const int ws_2c = max_aux_cart * max_aux_cart * 3;
    const int grid_2c = (naux_bas + threads - 1) / threads;
    double* d_ws_2c = NULL;
    Device_Malloc_Safely((void**)&d_ws_2c,
                         sizeof(double) * (size_t)naux_bas * ws_2c);

    Launch_Device_Kernel(QC_RI_2Center_Grad_Kernel, grid_2c, threads, 0, 0,
                         naux_bas, d_aux_centers, d_aux_l_list, d_aux_exps,
                         d_aux_coeffs, d_aux_shell_offsets, d_aux_shell_sizes,
                         d_aux_ao_offsets_cart, d_aux_ao_offsets_sph,
                         d_aux_norms, d_U_aux, naux_cart, naux, d_D2,
                         d_shell_atom_aux, d_ws_2c, ws_2c, naux_bas, d_grad);

    deviceFree(d_ws_2c);
    deviceFree(d_D2);

    // 3c 梯度内核
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
        QC_RI_3Center_Grad_Kernel, grid_3c, threads, 0, 0, naux_bas, norb_bas,
        d_aux_centers, d_aux_l_list, d_aux_exps, d_aux_coeffs,
        d_aux_shell_offsets, d_aux_shell_sizes, d_aux_ao_offsets_cart,
        d_aux_ao_offsets_sph, d_orb_centers, d_orb_l_list, d_orb_exps,
        d_orb_coeffs, d_orb_shell_offsets, d_orb_shell_sizes,
        d_orb_ao_offsets_cart, d_orb_ao_offsets_sph, (int)is_spherical,
        d_aux_norms, d_orb_norms, d_U_aux, d_U_orb, naux_cart, naux, nao_cart,
        nao, d_D3, d_shell_atom_aux, d_shell_atom_orb, d_ws_3c, ws_3c, naux_bas,
        d_grad);

    deviceFree(d_ws_3c);
    deviceFree(d_D3);
    deviceFree(d_U_aux);
    deviceFree(d_U_orb);
}

// D2_eff 构建辅助: D2_J + D2_K (从 eri3c 计算 Z_K 再做 Daleckii-Kreĭn)
static inline void QC_Build_D2_eff_Stored(
    const int nao, const int naux, const double* g_vec, const double* eri3c,
    const float* P_density, const float* B_occ, const float* C_occ,
    const int nocc, const float exx_fraction, const double* eigval,
    const double* eigvec, std::vector<double>& D2_eff)
{
    const long long nao2 = (long long)nao * nao;
    const size_t naux2 = (size_t)naux * naux;

    QC_Init_D2_J(naux, g_vec, D2_eff);

    if (exx_fraction != 0.0f && nocc > 0 && B_occ != nullptr &&
        eigval != nullptr && eigvec != nullptr)
    {
        const int M_dim = naux * nao;

        // Z_K[P,Q] = -Σ_{m,l} eri3c[Q,m,l] × V[P,m,l]
        //   V[P,m,l] = Σ_{n,i} B_occ[P,n,i] × C[l,i] × D[m,n]
        // 分解为矩阵乘法链:
        //   R[P,i,m] = Σ_n B_occ[P,n,i] × D[m,n]
        //   V[P,m,l] = Σ_i R[P,i,m] × C[l,i] = Σ_i D[m,n]·B_occ[P,n,i]·C[l,i]
        //   Z_K = -V_flat @ eri3c^T   (将 P,ml 展开)
        //
        // 实际用 V[naux, nao²] 和 eri3c[naux, nao²]:
        //   Z_K[P,Q] = -Σ_{ml} V[P, m*nao+l] × eri3c[Q, m*nao+l]

#if RI_GRAD_HAS_BLAS
        // BLAS 路径: V_P = D @ B_P @ C^T, Z_K = -V @ eri3c^T
        std::vector<double> D_d(nao2);
        for (long long i = 0; i < nao2; i++) D_d[i] = (double)P_density[i];

        std::vector<double> C_d((size_t)nao * nocc);
        for (int lam = 0; lam < nao; lam++)
            for (int i = 0; i < nocc; i++)
                C_d[lam * nocc + i] = (double)C_occ[lam * nao + i];

        std::vector<double> V(naux * nao2, 0.0);
        std::vector<double> B_P_d((size_t)nao * nocc);
        std::vector<double> X_P((size_t)nao * nao);
        for (int P = 0; P < naux; P++)
        {
            for (int i = 0; i < nocc; i++)
                for (int n = 0; n < nao; n++)
                    B_P_d[(size_t)n * nocc + i] = (double)
                        B_occ[(size_t)(P * nao + n) + (size_t)M_dim * i];

            // X_P[n,l] = Σ_i B_P[n,i] × C[l,i]
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, nao, nao, nocc,
                        1.0, B_P_d.data(), nocc, C_d.data(), nocc, 0.0,
                        X_P.data(), nao);

            // V_P[m,l] = Σ_n D[m,n] × X_P[n,l]
            double* V_P = V.data() + (long long)P * nao2;
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nao, nao,
                        nao, 1.0, D_d.data(), nao, X_P.data(), nao, 0.0, V_P,
                        nao);
        }

        // Z_K = -V @ eri3c^T: [naux,nao²] @ [nao²,naux] → [naux,naux]
        std::vector<double> Z_K(naux2, 0.0);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, naux, naux,
                    (int)nao2, -1.0, V.data(), (int)nao2, eri3c, (int)nao2, 0.0,
                    Z_K.data(), naux);
#else
        // 标量回退
        std::vector<double> V(naux * nao2, 0.0);
        for (int P = 0; P < naux; P++)
        {
            for (int i = 0; i < nocc; i++)
            {
                for (int m = 0; m < nao; m++)
                {
                    double r = 0.0;
                    for (int n = 0; n < nao; n++)
                        r += (double)B_occ[(size_t)(P * nao + n) +
                                           (size_t)M_dim * i] *
                             (double)P_density[m * nao + n];
                    for (int l = 0; l < nao; l++)
                        V[(long long)P * nao2 + m * nao + l] +=
                            r * (double)C_occ[l * nao + i];
                }
            }
        }

        std::vector<double> Z_K(naux2, 0.0);
        for (int P = 0; P < naux; P++)
            for (int Q = 0; Q < naux; Q++)
            {
                double z = 0.0;
                for (long long ml = 0; ml < nao2; ml++)
                    z += V[(long long)P * nao2 + ml] *
                         eri3c[(long long)Q * nao2 + ml];
                Z_K[(size_t)P * naux + Q] = -z;
            }
#endif

        QC_Build_D2K_DaleckiiKrein(naux, Z_K.data(), eigval, eigvec, D2_eff);
    }
}

// D2_eff 构建辅助: D2_J + D2_K (从预累积 Z_K)
static inline void QC_Build_D2_eff_FromZK(
    const int naux, const double* g_vec, const float* B_occ, const int nocc,
    const float exx_fraction, const double* Z_K, const double* eigval,
    const double* eigvec, std::vector<double>& D2_eff)
{
    QC_Init_D2_J(naux, g_vec, D2_eff);

    if (exx_fraction != 0.0f && nocc > 0 && B_occ != nullptr &&
        Z_K != nullptr && eigval != nullptr && eigvec != nullptr)
        QC_Build_D2K_DaleckiiKrein(naux, Z_K, eigval, eigvec, D2_eff);
}

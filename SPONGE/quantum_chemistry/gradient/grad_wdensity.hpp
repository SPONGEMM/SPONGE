#pragma once

// ====================== 能量加权密度矩阵 W ======================
// W_μν = occ_factor × Σ_{i∈occ} ε_i × C_μi × C_νi
//
// 用于 Pulay 力项: F_A = -Tr[W · dS/dR_A]
//
// 利用 BLAS:
//   1. D_μi = ε_i × C_μi  (对角缩放)
//   2. W = occ_factor × D @ C^T  (矩阵乘法)
// ==============================================================

// 对占据轨道系数乘以特征值: D[i*nao + mu] = epsilon[i] * C[mu*nao + i]
// 注意 C 是行主序 [mu, i] (nao × nao_padded)，D 也是 [mu, i]
static __global__ void QC_Scale_MO_By_Eigenvalue_Kernel(const int nao,
                                                        const int n_occ,
                                                        const float* C,
                                                        const float* epsilon,
                                                        float* D)
{
    const int total = nao * n_occ;
    SIMPLE_DEVICE_FOR(idx, total)
    {
        const int mu = idx / n_occ;
        const int i = idx % n_occ;
        D[mu * nao + i] = C[mu * nao + i] * epsilon[i];
    }
}

static void QC_Build_Energy_Weighted_Density(BLAS_HANDLE blas_handle, int nao,
                                             int n_occ, float occ_factor,
                                             const float* d_C,
                                             const float* d_epsilon,
                                             float* d_W, float* d_D_tmp)
{
    if (n_occ == 0) return;
    const int threads = 256;
    const int total = nao * n_occ;

    // D_μi = ε_i × C_μi
    Launch_Device_Kernel(QC_Scale_MO_By_Eigenvalue_Kernel,
                         (total + threads - 1) / threads, threads, 0, 0, nao,
                         n_occ, d_C, d_epsilon, d_D_tmp);

    // W = occ_factor × D @ C^T (行主序)
    // 与 QC_Build_Density_Blas 相同的 sgemm 模式:
    // P = occ × C_row @ C_row^T 用 sgemm(T, N, nao, nao, n_occ, C, nao, C, nao)
    // W = occ × D_row @ C_row^T 用 sgemm(T, N, nao, nao, n_occ, C, nao, D, nao)
    const float alpha = occ_factor;
    const float beta = 0.0f;
    deviceBlasSgemm(blas_handle, DEVICE_BLAS_OP_T, DEVICE_BLAS_OP_N, nao, nao,
                    n_occ, &alpha, d_C, nao, d_D_tmp, nao, &beta, d_W, nao);
}

#pragma once

// Metric (P|Q) 分解：
// 1. Cholesky: (P|Q) = L L^T  → 用于 RI-J 三角求解
// 2. 特征分解: (P|Q)^{-1/2}  → 用于 RI-K B 张量构建

#include "../../structure/matrix.h"

// 构建 (P|Q)^{-1/2} via 特征分解
// 输入: d_metric[naux × naux] (double, 对称)
// 输出: d_inv_sqrt[naux × naux] (double)
// 返回: naux_eff (去除线性依赖后的有效维度)
static int QC_RI_Build_Metric_InvSqrt(SOLVER_HANDLE solver_handle,
                                      BLAS_HANDLE blas_handle, int naux,
                                      const double* d_metric,
                                      double* d_inv_sqrt,
                                      double lindep_thresh = 1e-10)
{
    const int naux2 = naux * naux;

    // 拷贝 metric 到 inv_sqrt 作为工作区 (特征分解会覆盖)
    deviceMemcpy(d_inv_sqrt, d_metric, sizeof(double) * naux2,
                 deviceMemcpyDeviceToDevice);

    // 特征值和 workspace
    double* d_eigval = NULL;
    Device_Malloc_Safely((void**)&d_eigval, sizeof(double) * naux);

    double* d_work = NULL;
    int lwork = 0;
    int stat = QC_Diagonalize_Double_Workspace_Size(
        solver_handle, naux, d_inv_sqrt, d_eigval, &d_work, &lwork);
    if (stat != 0)
    {
        printf("    [QC-RI] WARNING: eigensolver workspace failed: %d\n", stat);
        if (d_eigval) deviceFree(d_eigval);
        return 0;
    }

    int info = 0;
    QC_Diagonalize_Double(solver_handle, naux, d_inv_sqrt, d_eigval, d_work,
                          lwork, &info);
    if (info != 0)
    {
        printf("    [QC-RI] WARNING: eigendecomposition failed: info=%d\n",
               info);
        if (d_work) deviceFree(d_work);
        if (d_eigval) deviceFree(d_eigval);
        return 0;
    }

    // d_inv_sqrt 现在包含特征向量 (列优先)
    // d_eigval 包含特征值 (升序)

    // 拷贝特征值到 host 检查线性依赖
    std::vector<double> h_eigval(naux);
    deviceMemcpy(h_eigval.data(), d_eigval, sizeof(double) * naux,
                 deviceMemcpyDeviceToHost);

    // 找到最小有效特征值的索引
    int n_skip = 0;
    for (int i = 0; i < naux; i++)
    {
        if (h_eigval[i] < lindep_thresh)
            n_skip++;
        else
            break;
    }
    const int naux_eff = naux - n_skip;

    if (n_skip > 0)
    {
        printf(
            "    [QC-RI] Removed %d linearly dependent aux functions "
            "(threshold=%.1e)\n",
            n_skip, lindep_thresh);
    }

    // 构建 (P|Q)^{-1/2} = U * diag(1/√λ) * U^T
    // 其中 U 是有效特征向量, λ 是有效特征值
    // 先在 host 上构建 diag(1/√λ) 缩放后的特征向量，再做矩阵乘

    // 暂存缩放后的特征向量: V[i,k] = U[i,k] / λ_k^{1/4}
    double* d_V = NULL;
    Device_Malloc_Safely((void**)&d_V, sizeof(double) * naux * naux_eff);

    // 拷贝有效特征向量并缩放
    // d_inv_sqrt 是列优先: U[i, k] = d_inv_sqrt[i + k*naux]
    // 有效部分从列 n_skip 开始
    std::vector<double> h_eigvec(naux * naux);
    deviceMemcpy(h_eigvec.data(), d_inv_sqrt, sizeof(double) * naux2,
                 deviceMemcpyDeviceToHost);

    std::vector<double> h_metric_head(2, 0.0);
    deviceMemcpy(h_metric_head.data(), d_metric, sizeof(double) * 2,
                 deviceMemcpyDeviceToHost);
    double recon00 = 0.0, recon01 = 0.0;
    double ortho00 = 0.0, ortho01 = 0.0;
    double alt_recon00 = 0.0, alt_recon01 = 0.0;
    for (int k = 0; k < naux; k++)
    {
        const double u0k = h_eigvec[0 + k * naux];
        const double u1k = h_eigvec[1 + k * naux];
        const double uk0 = h_eigvec[k + 0 * naux];
        const double uk1 = h_eigvec[k + 1 * naux];
        recon00 += u0k * h_eigval[k] * u0k;
        recon01 += u0k * h_eigval[k] * u1k;
        alt_recon00 += uk0 * h_eigval[k] * uk0;
        alt_recon01 += uk0 * h_eigval[k] * uk1;
        ortho00 += u0k * u0k;
        ortho01 += u0k * u1k;
    }
    printf(
        "    [QC-RI] metric recon m00=%.12e src00=%.12e m01=%.12e "
        "src01=%.12e\n",
        recon00, h_metric_head[0], recon01, h_metric_head[1]);
    printf("    [QC-RI] metric recon_alt m00=%.12e m01=%.12e\n", alt_recon00,
           alt_recon01);
    printf("    [QC-RI] eigvec ortho col0·col0=%.12e col0·col1=%.12e\n",
           ortho00, ortho01);

    std::vector<double> h_V(naux * naux_eff);
    for (int k = 0; k < naux_eff; k++)
    {
        double scale = pow(h_eigval[k + n_skip], -0.25);
        for (int i = 0; i < naux; i++)
        {
            h_V[i + k * naux] = h_eigvec[i + (k + n_skip) * naux] * scale;
        }
    }
    double host_inv00 = 0.0, host_inv01 = 0.0;
    double alt_inv00 = 0.0, alt_inv01 = 0.0;
    for (int k = 0; k < naux_eff; k++)
    {
        const double scale = pow(h_eigval[k + n_skip], -0.25);
        const double u0k = h_eigvec[0 + (k + n_skip) * naux];
        const double u1k = h_eigvec[1 + (k + n_skip) * naux];
        const double uk0 = h_eigvec[(k + n_skip) + 0 * naux];
        const double uk1 = h_eigvec[(k + n_skip) + 1 * naux];
        host_inv00 += u0k * scale * u0k;
        host_inv01 += u0k * scale * u1k;
        alt_inv00 += uk0 * scale * uk0;
        alt_inv01 += uk0 * scale * uk1;
    }
    printf(
        "    [QC-RI] host inv_sqrt cur[0,0]=%.12e cur[1,0]=%.12e "
        "alt[0,0]=%.12e alt[1,0]=%.12e\n",
        host_inv00, host_inv01, alt_inv00, alt_inv01);
    deviceMemcpy(d_V, h_V.data(), sizeof(double) * naux * naux_eff,
                 deviceMemcpyHostToDevice);

    // inv_sqrt = V * V^T (col-major DGEMM)
    // C[naux,naux] = V[naux,naux_eff] * V^T[naux_eff,naux]
    const double one = 1.0, zero = 0.0;
    deviceBlasDgemm(blas_handle, DEVICE_BLAS_OP_N, DEVICE_BLAS_OP_T, naux, naux,
                    naux_eff, &one, d_V, naux, d_V, naux, &zero, d_inv_sqrt,
                    naux);

    std::vector<double> h_inv_sqrt_dump(2, 0.0);
    deviceMemcpy(h_inv_sqrt_dump.data(), d_inv_sqrt, sizeof(double) * 2,
                 deviceMemcpyDeviceToHost);
    printf("    [QC-RI] metric eigval[0]=%.12e eigval[first_valid]=%.12e\n",
           h_eigval[0], h_eigval[n_skip]);
    printf("    [QC-RI] metric inv_sqrt[0]=%.12e inv_sqrt[1]=%.12e\n",
           h_inv_sqrt_dump[0], h_inv_sqrt_dump[1]);
    fflush(stdout);

    if (d_V) deviceFree(d_V);
    if (d_work) deviceFree(d_work);
    if (d_eigval) deviceFree(d_eigval);

    return naux_eff;
}

// 构建 (P|Q)^{-1} via 特征分解
// 输入: d_metric[naux × naux] (double, 对称)
// 输出: d_inv[naux × naux] (double)
static void QC_RI_Build_Metric_Inv(SOLVER_HANDLE solver_handle,
                                   BLAS_HANDLE blas_handle, int naux,
                                   const double* d_metric, double* d_inv,
                                   int naux_eff, double lindep_thresh = 1e-10)
{
    const int naux2 = naux * naux;

    // 特征分解 (复用与 inv_sqrt 相同的步骤)
    double* d_eigvec = NULL;
    Device_Malloc_Safely((void**)&d_eigvec, sizeof(double) * naux2);
    deviceMemcpy(d_eigvec, d_metric, sizeof(double) * naux2,
                 deviceMemcpyDeviceToDevice);

    double* d_eigval = NULL;
    Device_Malloc_Safely((void**)&d_eigval, sizeof(double) * naux);

    double* d_work = NULL;
    int lwork = 0;
    QC_Diagonalize_Double_Workspace_Size(solver_handle, naux, d_eigvec,
                                         d_eigval, &d_work, &lwork);
    int info = 0;
    QC_Diagonalize_Double(solver_handle, naux, d_eigvec, d_eigval, d_work,
                          lwork, &info);

    std::vector<double> h_eigval(naux);
    deviceMemcpy(h_eigval.data(), d_eigval, sizeof(double) * naux,
                 deviceMemcpyDeviceToHost);

    std::vector<double> h_eigvec(naux2);
    deviceMemcpy(h_eigvec.data(), d_eigvec, sizeof(double) * naux2,
                 deviceMemcpyDeviceToHost);

    int n_skip = naux - naux_eff;

    // V[i,k] = U[i,k] / λ_k
    std::vector<double> h_V(naux * naux_eff);
    for (int k = 0; k < naux_eff; k++)
    {
        double scale = 1.0 / h_eigval[k + n_skip];
        for (int i = 0; i < naux; i++)
            h_V[i + k * naux] = h_eigvec[i + (k + n_skip) * naux] * scale;
    }

    double* d_V = NULL;
    Device_Malloc_Safely((void**)&d_V, sizeof(double) * naux * naux_eff);
    deviceMemcpy(d_V, h_V.data(), sizeof(double) * naux * naux_eff,
                 deviceMemcpyHostToDevice);

    // 需要原始有效特征向量 U_eff
    double* d_U = NULL;
    Device_Malloc_Safely((void**)&d_U, sizeof(double) * naux * naux_eff);
    std::vector<double> h_U(naux * naux_eff);
    for (int k = 0; k < naux_eff; k++)
        for (int i = 0; i < naux; i++)
            h_U[i + k * naux] = h_eigvec[i + (k + n_skip) * naux];
    deviceMemcpy(d_U, h_U.data(), sizeof(double) * naux * naux_eff,
                 deviceMemcpyHostToDevice);

    // inv = V * U^T
    const double one = 1.0, zero = 0.0;
    deviceBlasDgemm(blas_handle, DEVICE_BLAS_OP_N, DEVICE_BLAS_OP_T, naux, naux,
                    naux_eff, &one, d_V, naux, d_U, naux, &zero, d_inv, naux);

    if (d_V) deviceFree(d_V);
    if (d_U) deviceFree(d_U);
    if (d_eigvec) deviceFree(d_eigvec);
    if (d_eigval) deviceFree(d_eigval);
    if (d_work) deviceFree(d_work);
}

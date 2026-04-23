#pragma once

// Metric (P|Q) 分解（特征分解）：
// (P|Q)^{-1/2}  → 用于 RI-K B 张量构建
// (P|Q)^{-1}    → 用于 RI-J 拟合系数求解

#include "../../structure/matrix.h"

// 构建 (P|Q)^{-1/2} via 特征分解
// 输入: d_metric[naux × naux] (double, 对称)
// 输出: d_inv_sqrt[naux × naux] (double)
// 返回: naux_eff (去除线性依赖后的有效维度)
static int QC_RI_Build_Metric_InvSqrt(SOLVER_HANDLE solver_handle,
                                      BLAS_HANDLE blas_handle, int naux,
                                      const double* d_metric,
                                      double* d_inv_sqrt,
                                      std::vector<double>* out_eigval = nullptr,
                                      std::vector<double>* out_eigvec = nullptr,
                                      double lindep_thresh = 1e-10)
{
    const int naux2 = naux * naux;

    deviceMemcpy(d_inv_sqrt, d_metric, sizeof(double) * naux2,
                 deviceMemcpyDeviceToDevice);

    double* d_eigval = NULL;
    Device_Malloc_Safely((void**)&d_eigval, sizeof(double) * naux);

    double* d_work = NULL;
    int lwork = 0;
    int stat = QC_Diagonalize_Double_Workspace_Size(
        solver_handle, naux, d_inv_sqrt, d_eigval, &d_work, &lwork);
    if (stat != 0)
    {
        if (d_eigval) deviceFree(d_eigval);
        return 0;
    }

    int info = 0;
    QC_Diagonalize_Double(solver_handle, naux, d_inv_sqrt, d_eigval, d_work,
                          lwork, &info);
    if (info != 0)
    {
        if (d_work) deviceFree(d_work);
        if (d_eigval) deviceFree(d_eigval);
        return 0;
    }

    std::vector<double> h_eigval(naux);
    deviceMemcpy(h_eigval.data(), d_eigval, sizeof(double) * naux,
                 deviceMemcpyDeviceToHost);

    // 特征值升序排列，跳过低于阈值的（线性依赖）
    int n_skip = 0;
    for (int i = 0; i < naux; i++)
    {
        if (h_eigval[i] < lindep_thresh)
            n_skip++;
        else
            break;
    }
    const int naux_eff = naux - n_skip;

    // 构建 (P|Q)^{-1/2} = V * V^T，其中 V[i,k] = U[i,k] * λ_k^{-1/4}
    double* d_V = NULL;
    Device_Malloc_Safely((void**)&d_V, sizeof(double) * naux * naux_eff);

    std::vector<double> h_eigvec(naux * naux);
    deviceMemcpy(h_eigvec.data(), d_inv_sqrt, sizeof(double) * naux2,
                 deviceMemcpyDeviceToHost);

    std::vector<double> h_V(naux * naux_eff);
    for (int k = 0; k < naux_eff; k++)
    {
        double scale = pow(h_eigval[k + n_skip], -0.25);
        for (int i = 0; i < naux; i++)
            h_V[i + k * naux] = h_eigvec[i + (k + n_skip) * naux] * scale;
    }
    deviceMemcpy(d_V, h_V.data(), sizeof(double) * naux * naux_eff,
                 deviceMemcpyHostToDevice);

    // inv_sqrt = V * V^T
    const double one = 1.0, zero = 0.0;
    deviceBlasDgemm(blas_handle, DEVICE_BLAS_OP_N, DEVICE_BLAS_OP_T, naux, naux,
                    naux_eff, &one, d_V, naux, d_V, naux, &zero, d_inv_sqrt,
                    naux);

    // 保存特征值/向量供梯度使用 (Daleckii-Kreĭn 公式)
    if (out_eigval) *out_eigval = h_eigval;
    if (out_eigvec) *out_eigvec = h_eigvec;

    if (d_V) deviceFree(d_V);
    if (d_work) deviceFree(d_work);
    if (d_eigval) deviceFree(d_eigval);

    return naux_eff;
}

// 构建 (P|Q)^{-1} via 特征分解
// V[i,k] = U[i,k] / λ_k, inv = V * U^T
static void QC_RI_Build_Metric_Inv(SOLVER_HANDLE solver_handle,
                                   BLAS_HANDLE blas_handle, int naux,
                                   const double* d_metric, double* d_inv,
                                   int naux_eff, double lindep_thresh = 1e-10)
{
    const int naux2 = naux * naux;

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

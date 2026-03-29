#pragma once

// ========================== DIIS 误差构造 ==========================
// 在设备侧用双精度构造对易子误差 e = F P S - S P F
// ================================================================
static void QC_Build_DIIS_Error_Double(BLAS_HANDLE blas_handle, int nao,
                                       const double* d_F, const float* d_P,
                                       const float* d_S, double* d_err,
                                       double* d_tmp1, double* d_tmp2,
                                       double* d_tmp3)
{
    const int nao2 = nao * nao;

    // 将 P 和 S 提升到双精度工作区
    QC_Float_To_Double(nao2, d_P, d_tmp1);  // d_tmp1 = dP
    QC_Float_To_Double(nao2, d_S, d_tmp2);  // d_tmp2 = dS

    // d_tmp3 = F * P
    QC_Dgemm_NN(blas_handle, nao, nao, nao, d_F, nao, d_tmp1, nao, d_tmp3, nao);
    // d_err = FP * S
    QC_Dgemm_NN(blas_handle, nao, nao, nao, d_tmp3, nao, d_tmp2, nao, d_err,
                nao);
    // d_tmp3 = S * P
    QC_Dgemm_NN(blas_handle, nao, nao, nao, d_tmp2, nao, d_tmp1, nao, d_tmp3,
                nao);
    // d_tmp1 = SP * F
    QC_Dgemm_NN(blas_handle, nao, nao, nao, d_tmp3, nao, d_F, nao, d_tmp1, nao);
    // d_err = FPS - SPF
    QC_Double_Sub(nao2, d_err, d_tmp1, d_err);
}

// ========================= CDIIS 历史压入 =========================
// 维护环形历史缓冲，将最新 Fock 和误差写入历史
// ================================================================
static void QC_DIIS_History_Push_Double(int nao2, int diis_space,
                                        int& hist_count, int& hist_head,
                                        double** d_f_hist, double** d_e_hist,
                                        const double* d_f_new,
                                        const double* d_e_new)
{
    if (diis_space <= 0) return;
    const int bytes = sizeof(double) * nao2;
    int write_idx = 0;
    if (hist_count < diis_space)
    {
        write_idx = (hist_head + hist_count) % diis_space;
        hist_count++;
    }
    else
    {
        write_idx = hist_head;
        hist_head = (hist_head + 1) % diis_space;
        hist_count = diis_space;
    }
    deviceMemcpy(d_f_hist[write_idx], d_f_new, bytes,
                 deviceMemcpyDeviceToDevice);
    deviceMemcpy(d_e_hist[write_idx], d_e_new, bytes,
                 deviceMemcpyDeviceToDevice);
}

// ========================= CDIIS 外推求解 =========================
// 在主机侧构造并求解 CDIIS 线性系统，再在设备侧线性组合历史 Fock
// ================================================================
static bool QC_DIIS_Extrapolate_Double(int nao, int diis_space, int hist_count,
                                       int hist_head, double** d_f_hist,
                                       double** d_e_hist, double reg,
                                       double* d_f_out, double* d_accum)
{
    if (hist_count < 2 || diis_space <= 0) return false;
    const int m = std::min(hist_count, diis_space);
    if (m < 2) return false;
    const int n = m + 1;
    const int nao2 = nao * nao;
    auto hist_idx = [&](int logical_idx) -> int
    { return (hist_head + logical_idx) % diis_space; };

    // 在主机构造 B 矩阵和右端项
    std::vector<double> h_B(n * n, 0.0);
    std::vector<double> h_rhs(n, 0.0);
    h_rhs[m] = -1.0;
    for (int i = 0; i < m; i++)
    {
        h_B[i * n + m] = -1.0;
        h_B[m * n + i] = -1.0;
    }

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j <= i; j++)
        {
            deviceMemset(d_accum, 0, sizeof(double));
            QC_Double_Dot(nao2, d_e_hist[hist_idx(i)], d_e_hist[hist_idx(j)],
                          d_accum);
            double v;
            deviceMemcpy(&v, d_accum, sizeof(double), deviceMemcpyDeviceToHost);
            if (i == j) v += reg;
            h_B[i * n + j] = v;
            h_B[j * n + i] = v;
        }
    }

    // 在主机侧求解小规模 CDIIS 线性系统
    {
        std::vector<double> H(n * n);
        for (int i = 0; i < n * n; i++) H[i] = h_B[i];
        std::vector<double> w(n);

#if defined(USE_MKL) || defined(USE_OPENBLAS)
        int lwork_q = -1;
        double wq;
        LAPACKE_dsyev_work(LAPACK_COL_MAJOR, 'V', 'U', n, H.data(), n, w.data(),
                           &wq, lwork_q);
        int lwork_h = (int)wq;
        std::vector<double> work_h(lwork_h);
        int info = LAPACKE_dsyev_work(LAPACK_COL_MAJOR, 'V', 'U', n, H.data(),
                                      n, w.data(), work_h.data(), lwork_h);
        if (info != 0) return false;
#else
        // 无 LAPACK 时退化为主机侧 LU 消元
        std::vector<double> A_lu(n * n);
        for (int i = 0; i < n * n; i++) A_lu[i] = h_B[i];
        // 带部分主元的高斯消元
        for (int k = 0; k < n; k++)
        {
            int pivot = k;
            double max_abs = fabs(A_lu[k * n + k]);
            for (int i = k + 1; i < n; i++)
            {
                double v = fabs(A_lu[i * n + k]);
                if (v > max_abs)
                {
                    max_abs = v;
                    pivot = i;
                }
            }
            if (max_abs < 1e-18) return false;
            if (pivot != k)
            {
                for (int j = k; j < n; j++)
                    std::swap(A_lu[k * n + j], A_lu[pivot * n + j]);
                std::swap(h_rhs[k], h_rhs[pivot]);
            }
            for (int i = k + 1; i < n; i++)
            {
                double factor = A_lu[i * n + k] / A_lu[k * n + k];
                for (int j = k + 1; j < n; j++)
                    A_lu[i * n + j] -= factor * A_lu[k * n + j];
                h_rhs[i] -= factor * h_rhs[k];
            }
        }
        for (int i = n - 1; i >= 0; i--)
        {
            double sum = h_rhs[i];
            for (int j = i + 1; j < n; j++) sum -= A_lu[i * n + j] * h_rhs[j];
            h_rhs[i] = sum / A_lu[i * n + i];
        }
        goto do_extrapolate;
#endif

        // H 中现为列主序本征向量，按特征分解结果恢复 CDIIS 系数
        std::vector<double> c(n, 0.0);
        for (int k = 0; k < n; k++)
        {
            if (fabs(w[k]) < 1e-14) continue;
            double vg = 0.0;
            for (int i = 0; i < n; i++) vg += H[k * n + i] * h_rhs[i];
            double coeff = vg / w[k];
            for (int i = 0; i < n; i++) c[i] += coeff * H[k * n + i];
        }
        for (int i = 0; i < n; i++) h_rhs[i] = c[i];
    }

#if !defined(USE_MKL) && !defined(USE_OPENBLAS)
do_extrapolate:
#endif
    // 在设备侧执行 F_out = sum_i c_i * F_hist[i]
    deviceMemset(d_f_out, 0, sizeof(double) * nao2);
    for (int i = 0; i < m; i++)
    {
        double c = h_rhs[i];
        QC_Double_Axpy(nao2, c, d_f_hist[hist_idx(i)], d_f_out);
    }
    return true;
}

// ========================== SCF 中应用 DIIS ==========================
// 在每轮 SCF 中更新 alpha/beta 历史，并执行 CDIIS 外推
// ================================================================
void QUANTUM_CHEMISTRY::Apply_DIIS(int iter)
{
    if (!scf_ws.runtime.use_diis || (iter + 1) < scf_ws.runtime.diis_start_iter)
        return;

    double* dF = scf_ws.alpha.d_F_double;
    if (!dF) return;
    const int nao2 = (int)mol.nao2;

    // 在设备侧构造 alpha 通道 DIIS 误差
    QC_Build_DIIS_Error_Double(
        blas_handle, mol.nao, dF, scf_ws.alpha.d_P, scf_ws.core.d_S,
        scf_ws.diis.d_diis_err, scf_ws.ortho.d_dwork_nao2_2,
        scf_ws.ortho.d_dwork_nao2_3, scf_ws.ortho.d_dwork_nao2_4);

    // 将 alpha Fock 与误差压入 CDIIS 历史
    QC_DIIS_History_Push_Double(
        nao2, scf_ws.runtime.diis_space, scf_ws.diis.diis_hist_count,
        scf_ws.diis.diis_hist_head, scf_ws.diis.d_diis_f_hist.data(),
        scf_ws.diis.d_diis_e_hist.data(), dF, scf_ws.diis.d_diis_err);

    if (scf_ws.diis.diis_hist_count >= 2)
    {
        bool extrapolated = QC_DIIS_Extrapolate_Double(
            mol.nao, scf_ws.runtime.diis_space, scf_ws.diis.diis_hist_count,
            scf_ws.diis.diis_hist_head, scf_ws.diis.d_diis_f_hist.data(),
            scf_ws.diis.d_diis_e_hist.data(), scf_ws.runtime.diis_reg, dF,
            scf_ws.diis.d_diis_accum);
        if (extrapolated)
        {
            QC_Double_To_Float(nao2, dF, scf_ws.alpha.d_F);

        }
    }

    if (!scf_ws.runtime.unrestricted) return;

    // beta 通道 CDIIS
    double* dFb = scf_ws.beta.d_F_double;
    if (!dFb) return;
    QC_Build_DIIS_Error_Double(
        blas_handle, mol.nao, dFb, scf_ws.beta.d_P, scf_ws.core.d_S,
        scf_ws.diis.d_diis_err, scf_ws.ortho.d_dwork_nao2_2,
        scf_ws.ortho.d_dwork_nao2_3, scf_ws.ortho.d_dwork_nao2_4);
    QC_DIIS_History_Push_Double(
        nao2, scf_ws.runtime.diis_space, scf_ws.diis.diis_hist_count_b,
        scf_ws.diis.diis_hist_head_b, scf_ws.diis.d_diis_f_hist_b.data(),
        scf_ws.diis.d_diis_e_hist_b.data(), dFb, scf_ws.diis.d_diis_err);
    if (scf_ws.diis.diis_hist_count_b >= 2)
    {
        if (QC_DIIS_Extrapolate_Double(
                mol.nao, scf_ws.runtime.diis_space,
                scf_ws.diis.diis_hist_count_b, scf_ws.diis.diis_hist_head_b,
                scf_ws.diis.d_diis_f_hist_b.data(),
                scf_ws.diis.d_diis_e_hist_b.data(), scf_ws.runtime.diis_reg,
                dFb, scf_ws.diis.d_diis_accum))
        {
            QC_Double_To_Float(nao2, dFb, scf_ws.beta.d_F);

        }
    }
}

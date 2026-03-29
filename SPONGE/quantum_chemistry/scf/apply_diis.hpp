#pragma once

// ========================== DIIS 误差构造 ==========================
static void QC_Build_DIIS_Error_Double(BLAS_HANDLE blas_handle, int nao,
                                       const double* d_F, const float* d_P,
                                       const float* d_S, double* d_err,
                                       double* d_tmp1, double* d_tmp2,
                                       double* d_tmp3)
{
    const int nao2 = nao * nao;
    QC_Float_To_Double(nao2, d_P, d_tmp1);
    QC_Float_To_Double(nao2, d_S, d_tmp2);
    QC_Dgemm_NN(blas_handle, nao, nao, nao, d_F, nao, d_tmp1, nao, d_tmp3, nao);
    QC_Dgemm_NN(blas_handle, nao, nao, nao, d_tmp3, nao, d_tmp2, nao, d_err,
                nao);
    QC_Dgemm_NN(blas_handle, nao, nao, nao, d_tmp2, nao, d_tmp1, nao, d_tmp3,
                nao);
    QC_Dgemm_NN(blas_handle, nao, nao, nao, d_tmp3, nao, d_F, nao, d_tmp1, nao);
    QC_Double_Sub(nao2, d_err, d_tmp1, d_err);
}

// ========================= 历史压入 =========================
static void QC_DIIS_History_Push(int nao2, int diis_space, int& hist_count,
                                 int& hist_head, double** d_f_hist,
                                 double** d_e_hist, double** d_d_hist,
                                 const double* d_f_new, const double* d_e_new,
                                 const float* d_P, double* energy_hist,
                                 double energy)
{
    if (diis_space <= 0) return;
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
    const int bytes_d = sizeof(double) * nao2;
    deviceMemcpy(d_f_hist[write_idx], d_f_new, bytes_d,
                 deviceMemcpyDeviceToDevice);
    deviceMemcpy(d_e_hist[write_idx], d_e_new, bytes_d,
                 deviceMemcpyDeviceToDevice);
    QC_Float_To_Double(nao2, d_P, d_d_hist[write_idx]);
    energy_hist[write_idx] = energy;
}

// ========================= Simplex QP 求解器 =========================
// 求解: min 0.5 * c^T H c + g^T c, s.t. sum(c) = 1, c >= 0
// 使用投影梯度法，适合 DIIS space 大小（6-8 维）的小规模问题
// ====================================================================
static void QC_Solve_Simplex_QP(int m, const std::vector<double>& H,
                                const std::vector<double>& g,
                                std::vector<double>& c)
{
    c.assign(m, 1.0 / m);
    const int max_iter = 500;

    for (int step = 0; step < max_iter; step++)
    {
        // 计算梯度: grad = H*c + g
        std::vector<double> grad(m);
        for (int i = 0; i < m; i++)
        {
            grad[i] = g[i];
            for (int j = 0; j < m; j++) grad[i] += H[i * m + j] * c[j];
        }
        // 投影到 sum=0 的切空间: grad -= mean(grad)
        double mean = 0.0;
        for (int i = 0; i < m; i++) mean += grad[i];
        mean /= m;
        for (int i = 0; i < m; i++) grad[i] -= mean;

        // 计算步长: Armijo-like，确保 c >= 0
        double max_step = 1e10;
        for (int i = 0; i < m; i++)
        {
            if (grad[i] > 1e-15 && c[i] > 0)
                max_step = std::min(max_step, c[i] / grad[i]);
        }
        // Hessian 步长估计
        double gHg = 0.0;
        for (int i = 0; i < m; i++)
            for (int j = 0; j < m; j++) gHg += grad[i] * H[i * m + j] * grad[j];
        double gnorm2 = 0.0;
        for (int i = 0; i < m; i++) gnorm2 += grad[i] * grad[i];
        if (gnorm2 < 1e-20) break;

        double alpha = (gHg > 1e-15) ? gnorm2 / gHg : 1.0;
        alpha = std::min(alpha, max_step * 0.99);
        alpha = std::max(alpha, 1e-10);

        // 更新并投影到 simplex
        for (int i = 0; i < m; i++)
        {
            c[i] -= alpha * grad[i];
            if (c[i] < 0.0) c[i] = 0.0;
        }
        // 归一化
        double s = 0.0;
        for (int i = 0; i < m; i++) s += c[i];
        if (s < 1e-15)
        {
            c.assign(m, 1.0 / m);
            continue;
        }
        for (int i = 0; i < m; i++) c[i] /= s;
    }
}

// ========================= EDIIS 外推 =========================
// E^EDIIS(c) = Σ c_i E_i - 0.5 Σ_ij c_i c_j Tr((F_i-F_j)(D_i-D_j))
// ================================================================
static bool QC_EDIIS_Extrapolate(int nao2, int diis_space, int hist_count,
                                 int hist_head, double** d_f_hist,
                                 double** d_d_hist, double* energy_hist,
                                 double* d_accum, std::vector<double>& c_out)
{
    const int m = std::min(hist_count, diis_space);
    if (m < 2) return false;
    auto idx = [&](int i) { return (hist_head + i) % diis_space; };

    // 构造 Tr(F_i * D_j) 矩阵
    std::vector<double> FD(m * m);
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < m; j++)
        {
            deviceMemset(d_accum, 0, sizeof(double));
            QC_Double_Dot(nao2, d_f_hist[idx(i)], d_d_hist[idx(j)], d_accum);
            deviceMemcpy(&FD[i * m + j], d_accum, sizeof(double),
                         deviceMemcpyDeviceToHost);
        }
    }

    // H_ij = Tr((F_i-F_j)(D_i-D_j)) = FD[i,i] + FD[j,j] - FD[i,j] - FD[j,i]
    // 目标: min Σ c_i E_i - 0.5 Σ_ij c_i c_j H_ij = min g^T c + 0.5 c^T (-H) c
    std::vector<double> H(m * m), g(m);
    for (int i = 0; i < m; i++)
    {
        g[i] = energy_hist[idx(i)];
        for (int j = 0; j < m; j++)
        {
            double hij =
                FD[i * m + i] + FD[j * m + j] - FD[i * m + j] - FD[j * m + i];
            H[i * m + j] = -hij;  // QP 的 H 矩阵取负号
        }
    }

    QC_Solve_Simplex_QP(m, H, g, c_out);
    return true;
}

// ========================= ADIIS 外推 =========================
// E^ADIIS(c) = E_n + 2 Σ c_i Tr((D_i-D_n) F_n)
//            + Σ_ij c_i c_j Tr((D_i-D_n)(F_j-F_n))
// ================================================================
static bool QC_ADIIS_Extrapolate(int nao2, int diis_space, int hist_count,
                                 int hist_head, double** d_f_hist,
                                 double** d_d_hist, double* energy_hist,
                                 double* d_accum, std::vector<double>& c_out)
{
    const int m = std::min(hist_count, diis_space);
    if (m < 2) return false;
    auto idx = [&](int i) { return (hist_head + i) % diis_space; };

    const int n = m - 1;  // 最新的历史点
    int idx_n = idx(n);

    // Tr(D_i * F_j) 矩阵
    std::vector<double> DF(m * m);
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < m; j++)
        {
            deviceMemset(d_accum, 0, sizeof(double));
            QC_Double_Dot(nao2, d_d_hist[idx(i)], d_f_hist[idx(j)], d_accum);
            deviceMemcpy(&DF[i * m + j], d_accum, sizeof(double),
                         deviceMemcpyDeviceToHost);
        }
    }

    // g_i = 2 * Tr((D_i - D_n) * F_n) = 2 * (DF[i,n] - DF[n,n])
    // H_ij = Tr((D_i - D_n)(F_j - F_n))
    //      = DF[i,j] - DF[i,n] - DF[n,j] + DF[n,n]
    std::vector<double> H(m * m), g(m);
    double dn_fn = DF[n * m + n];
    for (int i = 0; i < m; i++)
    {
        g[i] = 2.0 * (DF[i * m + n] - dn_fn);
        for (int j = 0; j < m; j++)
        {
            H[i * m + j] =
                DF[i * m + j] - DF[i * m + n] - DF[n * m + j] + dn_fn;
        }
    }

    QC_Solve_Simplex_QP(m, H, g, c_out);
    return true;
}

// ========================= CDIIS 外推 =========================
static bool QC_CDIIS_Extrapolate(int nao, int diis_space, int hist_count,
                                 int hist_head, double** d_f_hist,
                                 double** d_e_hist, double reg, double* d_f_out,
                                 double* d_accum)
{
    if (hist_count < 2 || diis_space <= 0) return false;
    const int m = std::min(hist_count, diis_space);
    if (m < 2) return false;
    const int n = m + 1;
    const int nao2 = nao * nao;
    auto hist_idx = [&](int logical_idx) -> int
    { return (hist_head + logical_idx) % diis_space; };

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
        std::vector<double> A_lu(n * n);
        for (int i = 0; i < n * n; i++) A_lu[i] = h_B[i];
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
    deviceMemset(d_f_out, 0, sizeof(double) * nao2);
    for (int i = 0; i < m; i++)
    {
        double c = h_rhs[i];
        QC_Double_Axpy(nao2, c, d_f_hist[hist_idx(i)], d_f_out);
    }
    return true;
}

// ========================= MESA + CDIIS 外推 =========================
// MESA: 同时算 EDIIS 和 ADIIS，选密度变化更小的
// 切换: 误差范数大时用 MESA，小时用 CDIIS
// ===================================================================
static bool QC_MESA_Or_CDIIS_Extrapolate(int nao, int diis_space,
                                         int hist_count, int hist_head,
                                         double** d_f_hist, double** d_e_hist,
                                         double** d_d_hist, double* energy_hist,
                                         double reg, double enorm,
                                         double mesa_to_cdiis_threshold,
                                         double* d_f_out, double* d_accum)
{
    const int m = std::min(hist_count, diis_space);
    if (m < 2) return false;
    const int nao2 = nao * nao;
    auto idx = [&](int i) { return (hist_head + i) % diis_space; };

    if (enorm <= mesa_to_cdiis_threshold)
    {
        // 近收敛: 用 CDIIS
        return QC_CDIIS_Extrapolate(nao, diis_space, hist_count, hist_head,
                                    d_f_hist, d_e_hist, reg, d_f_out, d_accum);
    }

    // 远离收敛: MESA (EDIIS vs ADIIS)
    std::vector<double> c_ediis, c_adiis;
    bool ok_e =
        QC_EDIIS_Extrapolate(nao2, diis_space, hist_count, hist_head, d_f_hist,
                             d_d_hist, energy_hist, d_accum, c_ediis);
    bool ok_a =
        QC_ADIIS_Extrapolate(nao2, diis_space, hist_count, hist_head, d_f_hist,
                             d_d_hist, energy_hist, d_accum, c_adiis);
    if (!ok_e && !ok_a) return false;

    // 选密度变化更小的: ||P_new - P_current||_F
    // P_new = Σ c_i P_i, P_current = P_{最新}
    // ||P_new - P_n||^2 = Σ_ij (c_i - δ_{i,n})(c_j - δ_{j,n}) Tr(P_i P_j)
    // 简化: 直接比较系数的分散度（最新项系数越大越好）
    const std::vector<double>& c_best = [&]() -> const std::vector<double>&
    {
        if (!ok_e) return c_adiis;
        if (!ok_a) return c_ediis;
        // 比较: 最新项（index m-1）的权重越大 → 变化越小
        return (c_ediis[m - 1] >= c_adiis[m - 1]) ? c_ediis : c_adiis;
    }();

    // 线性组合 Fock
    deviceMemset(d_f_out, 0, sizeof(double) * nao2);
    for (int i = 0; i < m; i++)
    {
        QC_Double_Axpy(nao2, c_best[i], d_f_hist[idx(i)], d_f_out);
    }
    return true;
}

// ========================== SCF 中应用 DIIS ==========================
// MESA 算法: 远离收敛用 EDIIS/ADIIS，近收敛用 CDIIS
//
// 参考:
//   S. Lehtola, "OpenOrbitalOptimizer", arXiv:2503.23034 (2025).
//   X. Hu, W. Yang, J. Chem. Phys. 132, 054109 (2010). (ADIIS)
//   K. N. Kudin et al., J. Chem. Phys. 116, 8255 (2002). (EDIIS)
// ================================================================
void QUANTUM_CHEMISTRY::Apply_DIIS(int iter)
{
    if (!scf_ws.runtime.use_diis || (iter + 1) < scf_ws.runtime.diis_start_iter)
        return;

    double* dF = scf_ws.alpha.d_F_double;
    if (!dF) return;
    const int nao2 = (int)mol.nao2;

    // 获取当前 SCF 能量
    double current_energy = 0.0;
    deviceMemcpy(&current_energy, scf_ws.core.d_scf_energy, sizeof(double),
                 deviceMemcpyDeviceToHost);

    // 构造 DIIS 误差并计算范数
    QC_Build_DIIS_Error_Double(
        blas_handle, mol.nao, dF, scf_ws.alpha.d_P, scf_ws.core.d_S,
        scf_ws.diis.d_diis_err, scf_ws.ortho.d_dwork_nao2_2,
        scf_ws.ortho.d_dwork_nao2_3, scf_ws.ortho.d_dwork_nao2_4);

    deviceMemset(scf_ws.diis.d_diis_accum, 0, sizeof(double));
    QC_Double_Dot(nao2, scf_ws.diis.d_diis_err, scf_ws.diis.d_diis_err,
                  scf_ws.diis.d_diis_accum);
    double enorm_sq;
    deviceMemcpy(&enorm_sq, scf_ws.diis.d_diis_accum, sizeof(double),
                 deviceMemcpyDeviceToHost);
    double enorm = sqrt(enorm_sq / nao2);

    // 压入历史（Fock + 误差 + 密度 + 能量）
    QC_DIIS_History_Push(
        nao2, scf_ws.runtime.diis_space, scf_ws.diis.diis_hist_count,
        scf_ws.diis.diis_hist_head, scf_ws.diis.d_diis_f_hist.data(),
        scf_ws.diis.d_diis_e_hist.data(), scf_ws.diis.d_diis_d_hist.data(), dF,
        scf_ws.diis.d_diis_err, scf_ws.alpha.d_P,
        scf_ws.diis.energy_hist.data(), current_energy);

    if (scf_ws.diis.diis_hist_count >= 2)
    {
        bool ok = QC_MESA_Or_CDIIS_Extrapolate(
            mol.nao, scf_ws.runtime.diis_space, scf_ws.diis.diis_hist_count,
            scf_ws.diis.diis_hist_head, scf_ws.diis.d_diis_f_hist.data(),
            scf_ws.diis.d_diis_e_hist.data(), scf_ws.diis.d_diis_d_hist.data(),
            scf_ws.diis.energy_hist.data(), scf_ws.runtime.diis_reg, enorm,
            scf_ws.diis.mesa_to_cdiis_threshold, dF, scf_ws.diis.d_diis_accum);
        if (ok) QC_Double_To_Float(nao2, dF, scf_ws.alpha.d_F);
    }

    if (!scf_ws.runtime.unrestricted) return;

    // beta 通道: 只用 CDIIS（EDIIS/ADIIS 基于总能量，只需对 alpha 做 MESA）
    double* dFb = scf_ws.beta.d_F_double;
    if (!dFb) return;
    QC_Build_DIIS_Error_Double(
        blas_handle, mol.nao, dFb, scf_ws.beta.d_P, scf_ws.core.d_S,
        scf_ws.diis.d_diis_err, scf_ws.ortho.d_dwork_nao2_2,
        scf_ws.ortho.d_dwork_nao2_3, scf_ws.ortho.d_dwork_nao2_4);
    QC_DIIS_History_Push(
        nao2, scf_ws.runtime.diis_space, scf_ws.diis.diis_hist_count_b,
        scf_ws.diis.diis_hist_head_b, scf_ws.diis.d_diis_f_hist_b.data(),
        scf_ws.diis.d_diis_e_hist_b.data(), scf_ws.diis.d_diis_d_hist_b.data(),
        dFb, scf_ws.diis.d_diis_err, scf_ws.beta.d_P,
        scf_ws.diis.energy_hist.data(), current_energy);
    if (scf_ws.diis.diis_hist_count_b >= 2)
    {
        if (QC_CDIIS_Extrapolate(
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

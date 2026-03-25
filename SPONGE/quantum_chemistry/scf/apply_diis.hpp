#pragma once

// Compute DIIS error e = FPS - SPF in double precision (device-side)
static void QC_Build_DIIS_Error_Double(BLAS_HANDLE blas_handle, int nao,
                                       const double* d_F, const float* d_P,
                                       const float* d_S, double* d_err,
                                       double* d_tmp1, double* d_tmp2,
                                       double* d_tmp3)
{
    const int nao2 = nao * nao;

    // Promote P and S to double
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

    // Build B matrix and rhs on host
    std::vector<double> h_B(n * n, 0.0);
    std::vector<double> h_rhs(n, 0.0);
    h_rhs[m] = -1.0;
    for (int i = 0; i < m; i++)
    {
        h_B[i * n + m] = -1.0;
        h_B[m * n + i] = -1.0;
    }

    // Compute dot products on device
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

    // Solve using eigenvalue decomposition on host (PySCF approach)
    {
        std::vector<double> H(n * n);
        for (int i = 0; i < n * n; i++) H[i] = h_B[i];
        std::vector<double> w(n);

        // Simple Jacobi eigenvalue solver for small matrices (n <= 7)
        // Eigendecompose symmetric H
        // Use power iteration / Jacobi sweeps for small n
        // For robustness, use the same approach as before with host LAPACKE
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
        // Fallback: LU solve
        std::vector<double> A_lu(n * n);
        for (int i = 0; i < n * n; i++) A_lu[i] = h_B[i];
        // Gaussian elimination with partial pivoting
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

        // H now contains eigenvectors (col-major)
        // c = V * (1/w) * (V^T * g), filtering |w| < 1e-14
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
    // Extrapolate on device: F_out = sum_i c[i] * F_hist[i]
    deviceMemset(d_f_out, 0, sizeof(double) * nao2);
    for (int i = 0; i < m; i++)
    {
        double c = h_rhs[i];
        QC_Double_Axpy(nao2, c, d_f_hist[hist_idx(i)], d_f_out);
    }
    return true;
}

// ADIIS: minimize energy estimate over convex combination of stored Fock
// matrices Ref: JCP 132, 054109 (2010)
static bool QC_ADIIS_Extrapolate(BLAS_HANDLE blas_handle, int nao,
                                 int diis_space, int adiis_count,
                                 int adiis_head, double** d_f_hist,
                                 double** d_d_hist, double* d_f_out,
                                 double* d_accum)
{
    if (adiis_count < 2) return false;
    const int m = std::min(adiis_count, diis_space);
    if (m < 2) return false;
    const int nao2 = nao * nao;
    auto hist_idx = [&](int i) { return (adiis_head + i) % diis_space; };

    // df[i,j] = Tr(D_i * F_j) -- compute on device
    std::vector<double> df(m * m);
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < m; j++)
        {
            deviceMemset(d_accum, 0, sizeof(double));
            QC_Double_Dot(nao2, d_d_hist[hist_idx(i)], d_f_hist[hist_idx(j)],
                          d_accum);
            deviceMemcpy(&df[i * m + j], d_accum, sizeof(double),
                         deviceMemcpyDeviceToHost);
        }
    }

    // Build ADIIS quadratic form (small, on host)
    std::vector<double> dd_fn(m);
    double dn_fn = df[(m - 1) * m + (m - 1)];
    for (int i = 0; i < m; i++)
    {
        dd_fn[i] = df[i * m + (m - 1)] - dn_fn;
    }
    std::vector<double> df_adj(m * m);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++)
            df_adj[i * m + j] = df[i * m + j] - df[i * m + (m - 1)] -
                                df[(m - 1) * m + j] + dn_fn;

    // Minimize cost(c) = 2*sum(c_i*dd_fn_i) + sum(c_i*df_adj_ij*c_j)
    // Parametrize: c_i = x_i^2 / sum(x_j^2)
    std::vector<double> x(m, 1.0);
    for (int step = 0; step < 300; step++)
    {
        double x2sum = 0;
        for (int i = 0; i < m; i++) x2sum += x[i] * x[i];
        std::vector<double> c(m);
        for (int i = 0; i < m; i++) c[i] = x[i] * x[i] / x2sum;

        std::vector<double> gc(m);
        for (int k = 0; k < m; k++)
        {
            gc[k] = 2.0 * dd_fn[k];
            for (int j = 0; j < m; j++)
                gc[k] += (df_adj[k * m + j] + df_adj[j * m + k]) * c[j];
        }

        std::vector<double> gx(m, 0.0);
        for (int nn = 0; nn < m; nn++)
        {
            for (int k = 0; k < m; k++)
            {
                double dc = 2.0 * x[nn] *
                            ((k == nn ? x2sum : 0.0) - x[k] * x[k]) /
                            (x2sum * x2sum);
                gx[nn] += gc[k] * dc;
            }
        }

        double gnorm = 0;
        for (int i = 0; i < m; i++) gnorm += gx[i] * gx[i];
        if (gnorm < 1e-20) break;
        double lr = 0.1;
        for (int i = 0; i < m; i++) x[i] -= lr * gx[i];
    }

    // Final coefficients
    double x2sum = 0;
    for (int i = 0; i < m; i++) x2sum += x[i] * x[i];
    std::vector<double> c(m);
    for (int i = 0; i < m; i++) c[i] = x[i] * x[i] / x2sum;

    // Extrapolate on device: F = sum(c_i * F_i)
    deviceMemset(d_f_out, 0, sizeof(double) * nao2);
    for (int i = 0; i < m; i++)
    {
        double ci = c[i];
        QC_Double_Axpy(nao2, ci, d_f_hist[hist_idx(i)], d_f_out);
    }
    return true;
}

void QUANTUM_CHEMISTRY::Apply_DIIS(int iter)
{
    if (!scf_ws.use_diis || (iter + 1) < scf_ws.diis_start_iter) return;

    double* dF = scf_ws.d_F_double;
    if (!dF) return;
    const int nao2 = (int)mol.nao2;

    // Compute DIIS error and its norm (device-side)
    QC_Build_DIIS_Error_Double(blas_handle, mol.nao, dF, scf_ws.d_P, scf_ws.d_S,
                               scf_ws.d_diis_err, scf_ws.d_dwork_nao2_2,
                               scf_ws.d_dwork_nao2_3, scf_ws.d_dwork_nao2_4);

    // Compute error norm on device
    deviceMemset(scf_ws.d_diis_accum, 0, sizeof(double));
    QC_Double_Dot(nao2, scf_ws.d_diis_err, scf_ws.d_diis_err,
                  scf_ws.d_diis_accum);
    double enorm_sq;
    deviceMemcpy(&enorm_sq, scf_ws.d_diis_accum, sizeof(double),
                 deviceMemcpyDeviceToHost);
    double enorm = sqrt(enorm_sq);

    // Push F and error to CDIIS history
    QC_DIIS_History_Push_Double(
        nao2, scf_ws.diis_space, scf_ws.diis_hist_count, scf_ws.diis_hist_head,
        scf_ws.d_diis_f_hist.data(), scf_ws.d_diis_e_hist.data(), dF,
        scf_ws.d_diis_err);

    // Push D (density) to ADIIS history
    {
        int& ac = scf_ws.adiis_count;
        int& ah = scf_ws.adiis_head;
        int ws = scf_ws.diis_space;
        int write_idx = (ac < ws) ? ((ah + ac) % ws) : ah;
        if (ac < ws)
            ac++;
        else
            ah = (ah + 1) % ws;
        // Store density as double on device
        QC_Float_To_Double_Copy(nao2, scf_ws.d_P,
                                scf_ws.d_adiis_d_hist[write_idx]);
    }

    bool extrapolated = false;
    if (scf_ws.diis_hist_count >= 2)
    {
        // Switch: ADIIS when error is large, CDIIS when small
        if (enorm > scf_ws.adiis_to_cdiis_threshold)
        {
            // ADIIS
            extrapolated = QC_ADIIS_Extrapolate(
                blas_handle, mol.nao, scf_ws.diis_space, scf_ws.adiis_count,
                scf_ws.adiis_head, scf_ws.d_diis_f_hist.data(),
                scf_ws.d_adiis_d_hist.data(), dF, scf_ws.d_diis_accum);
        }
        else
        {
            // CDIIS
            extrapolated = QC_DIIS_Extrapolate_Double(
                mol.nao, scf_ws.diis_space, scf_ws.diis_hist_count,
                scf_ws.diis_hist_head, scf_ws.d_diis_f_hist.data(),
                scf_ws.d_diis_e_hist.data(), scf_ws.diis_reg, dF,
                scf_ws.d_diis_accum);
        }
        if (extrapolated)
        {
            // Copy extrapolated double Fock back to float
            QC_Double_To_Float(nao2, dF, scf_ws.d_F);
        }
    }

    if (!scf_ws.unrestricted) return;

    // Beta spin
    double* dFb = scf_ws.d_F_b_double;
    if (!dFb) return;
    QC_Build_DIIS_Error_Double(
        blas_handle, mol.nao, dFb, scf_ws.d_P_b, scf_ws.d_S, scf_ws.d_diis_err,
        scf_ws.d_dwork_nao2_2, scf_ws.d_dwork_nao2_3, scf_ws.d_dwork_nao2_4);
    QC_DIIS_History_Push_Double(
        nao2, scf_ws.diis_space, scf_ws.diis_hist_count_b,
        scf_ws.diis_hist_head_b, scf_ws.d_diis_f_hist_b.data(),
        scf_ws.d_diis_e_hist_b.data(), dFb, scf_ws.d_diis_err);
    if (scf_ws.diis_hist_count_b >= 2)
    {
        if (QC_DIIS_Extrapolate_Double(
                mol.nao, scf_ws.diis_space, scf_ws.diis_hist_count_b,
                scf_ws.diis_hist_head_b, scf_ws.d_diis_f_hist_b.data(),
                scf_ws.d_diis_e_hist_b.data(), scf_ws.diis_reg, dFb,
                scf_ws.d_diis_accum))
        {
            QC_Double_To_Float(nao2, dFb, scf_ws.d_F_b);
        }
    }
}

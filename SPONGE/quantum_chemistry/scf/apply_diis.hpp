#pragma once

static __global__ void QC_DIIS_Init_System_Kernel(const int n, const int m,
                                                  double* B, double* rhs)
{
    for (int i = 0; i < n; i++)
    {
        rhs[i] = 0.0;
        for (int j = 0; j < n; j++) B[i * n + j] = 0.0;
    }
    rhs[m] = -1.0;
    for (int i = 0; i < m; i++)
    {
        B[i * n + m] = -1.0;
        B[m * n + i] = -1.0;
    }
    B[m * n + m] = 0.0;
}

static __global__ void QC_DIIS_Set_B_From_Accum_Kernel(const int n, const int i,
                                                       const int j,
                                                       const double reg,
                                                       const double* d_accum,
                                                       double* B)
{
    const double v = d_accum[0] + ((i == j) ? reg : 0.0);
    B[i * n + j] = v;
    B[j * n + i] = v;
}

static __global__ void QC_DIIS_Solve_Linear_System_Kernel(const int n,
                                                          double* A, double* b,
                                                          int* info)
{
    info[0] = 0;
    for (int k = 0; k < n; k++)
    {
        int pivot = k;
        double max_abs = fabs(A[k * n + k]);
        for (int i = k + 1; i < n; i++)
        {
            const double v = fabs(A[i * n + k]);
            if (v > max_abs)
            {
                max_abs = v;
                pivot = i;
            }
        }
        if (max_abs < 1e-18)
        {
            info[0] = k + 1;
            return;
        }
        if (pivot != k)
        {
            for (int j = k; j < n; j++)
            {
                const double tmp = A[k * n + j];
                A[k * n + j] = A[pivot * n + j];
                A[pivot * n + j] = tmp;
            }
            const double tb = b[k];
            b[k] = b[pivot];
            b[pivot] = tb;
        }
        const double diag = A[k * n + k];
        for (int i = k + 1; i < n; i++)
        {
            const double factor = A[i * n + k] / diag;
            A[i * n + k] = 0.0;
            for (int j = k + 1; j < n; j++)
                A[i * n + j] -= factor * A[k * n + j];
            b[i] -= factor * b[k];
        }
    }
    for (int i = n - 1; i >= 0; i--)
    {
        double sum = b[i];
        for (int j = i + 1; j < n; j++) sum -= A[i * n + j] * b[j];
        const double diag = A[i * n + i];
        if (fabs(diag) < 1e-18)
        {
            info[0] = i + 1;
            return;
        }
        b[i] = sum / diag;
    }
}

// Compute DIIS error e = FPS - SPF in double precision
static void QC_Build_DIIS_Error_Double(int nao, const double* d_F,
                                       const float* d_P, const float* d_S,
                                       double* d_err)
{
    const int nao2 = nao * nao;
    std::vector<double> dP(nao2), dS(nao2);
    std::vector<double> t1(nao2), t2(nao2), t3(nao2);
    for (int i = 0; i < nao2; i++)
    {
        dP[i] = (double)d_P[i];
        dS[i] = (double)d_S[i];
    }
    const double one = 1.0, zero = 0.0;
    // t1 = F * P
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                nao, nao, nao, one, d_F, nao, dP.data(), nao,
                zero, t1.data(), nao);
    // t2 = FP * S
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                nao, nao, nao, one, t1.data(), nao, dS.data(), nao,
                zero, t2.data(), nao);
    // t1 = S * P
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                nao, nao, nao, one, dS.data(), nao, dP.data(), nao,
                zero, t1.data(), nao);
    // t3 = SP * F
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                nao, nao, nao, one, t1.data(), nao, d_F, nao,
                zero, t3.data(), nao);
    // err = FPS - SPF
    for (int i = 0; i < nao2; i++)
        d_err[i] = t2[i] - t3[i];
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
    memcpy(d_f_hist[write_idx], d_f_new, bytes);
    memcpy(d_e_hist[write_idx], d_e_new, bytes);
}

// Dot product of two double arrays
static double QC_Double_Dot(int n, const double* a, const double* b)
{
    double sum = 0.0;
    for (int i = 0; i < n; i++)
        sum += a[i] * b[i];
    return sum;
}

static bool QC_DIIS_Extrapolate_Double(int nao, int diis_space, int hist_count,
                                       int hist_head, double** d_f_hist,
                                       double** d_e_hist, double reg,
                                       double* d_f_out, double* d_B,
                                       double* d_rhs, int* d_info)
{
    if (hist_count < 2 || diis_space <= 0) return false;
    const int m = std::min(hist_count, diis_space);
    if (m < 2) return false;
    const int n = m + 1;
    const int nao2 = nao * nao;
    auto hist_idx = [&](int logical_idx) -> int
    { return (hist_head + logical_idx) % diis_space; };

    // Build B matrix and rhs
    for (int i = 0; i < n; i++)
    {
        d_rhs[i] = 0.0;
        for (int j = 0; j < n; j++) d_B[i * n + j] = 0.0;
    }
    d_rhs[m] = -1.0;
    for (int i = 0; i < m; i++)
    {
        d_B[i * n + m] = -1.0;
        d_B[m * n + i] = -1.0;
    }
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j <= i; j++)
        {
            double v = QC_Double_Dot(nao2, d_e_hist[hist_idx(i)],
                                     d_e_hist[hist_idx(j)]);
            if (i == j) v += reg;
            d_B[i * n + j] = v;
            d_B[j * n + i] = v;
        }
    }

    // Solve using eigenvalue decomposition (PySCF approach)
    // This handles ill-conditioned B matrices by filtering small eigenvalues
    d_info[0] = 0;
    {
        // dsyev workspace query and solve (B is symmetric)
        // B is stored row-major; dsyev expects col-major
        // For symmetric matrix, row-major = col-major (transpose of symmetric = itself)
        std::vector<double> H(n * n);
        memcpy(H.data(), d_B, sizeof(double) * n * n);
        std::vector<double> w(n);
        int lwork_q = -1;
        double wq;
        LAPACKE_dsyev_work(LAPACK_COL_MAJOR, 'V', 'U', n,
                           H.data(), n, w.data(), &wq, lwork_q);
        int lwork = (int)wq;
        std::vector<double> work(lwork);
        int info = LAPACKE_dsyev_work(LAPACK_COL_MAJOR, 'V', 'U', n,
                                      H.data(), n, w.data(),
                                      work.data(), lwork);
        if (info != 0) { d_info[0] = info; return false; }

        // H now contains eigenvectors (col-major)
        // c = V * (1/w) * (V^T * g), filtering |w| < 1e-14
        // g = d_rhs = [0, 0, ..., -1]
        std::vector<double> c(n, 0.0);
        for (int k = 0; k < n; k++)
        {
            if (fabs(w[k]) < 1e-14) continue;
            // v_k^T * g
            double vg = 0.0;
            for (int i = 0; i < n; i++)
                vg += H[k * n + i] * d_rhs[i];  // col-major: H[i, k] = H[k*n+i]
            double coeff = vg / w[k];
            for (int i = 0; i < n; i++)
                c[i] += coeff * H[k * n + i];
        }
        memcpy(d_rhs, c.data(), sizeof(double) * n);
    }

    // Extrapolate: F_out = sum_i c[i] * F_hist[i]
    memset(d_f_out, 0, sizeof(double) * nao2);
    for (int i = 0; i < m; i++)
    {
        double c = d_rhs[i];
        const double* fh = d_f_hist[hist_idx(i)];
        for (int idx = 0; idx < nao2; idx++)
            d_f_out[idx] += c * fh[idx];
    }
    return true;
}

static void QC_DIIS_Reset(int& hist_count, int& hist_head)
{
    hist_count = 0;
    hist_head = 0;
}

// ADIIS: minimize energy estimate over convex combination of stored Fock matrices
// Ref: JCP 132, 054109 (2010)
static bool QC_ADIIS_Extrapolate(int nao, int diis_space, int adiis_count,
                                  int adiis_head, double** d_f_hist,
                                  double** d_d_hist, double* d_f_out)
{
    if (adiis_count < 2) return false;
    const int m = std::min(adiis_count, diis_space);
    if (m < 2) return false;
    const int nao2 = nao * nao;
    auto hist_idx = [&](int i) { return (adiis_head + i) % diis_space; };
    const int newest = hist_idx(m - 1);

    // df[i,j] = Tr(D_i * F_j)
    std::vector<double> df(m * m);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++)
            df[i * m + j] = QC_Double_Dot(nao2, d_d_hist[hist_idx(i)],
                                           d_f_hist[hist_idx(j)]);

    // Build ADIIS quadratic form
    std::vector<double> dd_fn(m), dn_f(m);
    double dn_fn = df[( m-1) * m + (m-1)];
    for (int i = 0; i < m; i++)
    {
        dd_fn[i] = df[i * m + (m-1)] - dn_fn;
        dn_f[i] = df[(m-1) * m + i];
    }
    // df_adj[i,j] = df[i,j] - df[i,newest] - df[newest,j] + df[newest,newest]
    std::vector<double> df_adj(m * m);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++)
            df_adj[i * m + j] = df[i * m + j] - df[i * m + (m-1)]
                                 - df[(m-1) * m + j] + dn_fn;

    // Minimize cost(c) = 2*sum(c_i*dd_fn_i) + sum(c_i*df_adj_ij*c_j)
    // with c_i >= 0, sum(c_i) = 1
    // Parametrize: c_i = x_i^2 / sum(x_j^2)
    // Use simple gradient descent
    std::vector<double> x(m, 1.0);

    for (int step = 0; step < 300; step++)
    {
        double x2sum = 0;
        for (int i = 0; i < m; i++) x2sum += x[i] * x[i];
        std::vector<double> c(m);
        for (int i = 0; i < m; i++) c[i] = x[i] * x[i] / x2sum;

        // Gradient of cost w.r.t. c
        std::vector<double> gc(m);
        for (int k = 0; k < m; k++)
        {
            gc[k] = 2.0 * dd_fn[k];
            for (int j = 0; j < m; j++)
                gc[k] += (df_adj[k * m + j] + df_adj[j * m + k]) * c[j];
        }

        // Chain rule: dc/dx
        // dc_k/dx_n = (2*x_n*delta_kn*x2sum - x_k^2*2*x_n) / x2sum^2
        std::vector<double> gx(m, 0.0);
        for (int n = 0; n < m; n++)
        {
            for (int k = 0; k < m; k++)
            {
                double dc = 2.0 * x[n] * ((k == n ? x2sum : 0.0) - x[k] * x[k])
                            / (x2sum * x2sum);
                gx[n] += gc[k] * dc;
            }
        }

        // Line search: step size
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

    // Extrapolate: F = sum(c_i * F_i)
    memset(d_f_out, 0, sizeof(double) * nao2);
    for (int i = 0; i < m; i++)
    {
        double ci = c[i];
        const double* fh = d_f_hist[hist_idx(i)];
        for (int idx = 0; idx < nao2; idx++)
            d_f_out[idx] += ci * fh[idx];
    }
    return true;
}

void QUANTUM_CHEMISTRY::Apply_DIIS(int iter)
{
    if (!scf_ws.use_diis || (iter + 1) < scf_ws.diis_start_iter) return;

#ifndef USE_GPU
    double* dF = scf_ws.d_F_double;
    if (!dF) return;
    const int nao2 = (int)mol.nao2;

    // Compute DIIS error and its norm
    QC_Build_DIIS_Error_Double(mol.nao, dF, scf_ws.d_P, scf_ws.d_S,
                               scf_ws.d_diis_err);
    double enorm = 0;
    for (int i = 0; i < nao2; i++)
        enorm += scf_ws.d_diis_err[i] * scf_ws.d_diis_err[i];
    enorm = sqrt(enorm);

    // Push F and error to CDIIS history
    QC_DIIS_History_Push_Double(
        nao2, scf_ws.diis_space, scf_ws.diis_hist_count,
        scf_ws.diis_hist_head, scf_ws.d_diis_f_hist.data(),
        scf_ws.d_diis_e_hist.data(), dF, scf_ws.d_diis_err);

    // Push D (density) to ADIIS history (same ring buffer indexing as F)
    {
        int& ac = scf_ws.adiis_count;
        int& ah = scf_ws.adiis_head;
        int ws = scf_ws.diis_space;
        int write_idx = (ac < ws) ? ((ah + ac) % ws) : ah;
        if (ac < ws) ac++;
        else ah = (ah + 1) % ws;
        // Store density as double
        for (int i = 0; i < nao2; i++)
            scf_ws.d_adiis_d_hist[write_idx][i] = (double)scf_ws.d_P[i];
    }

    bool extrapolated = false;
    if (scf_ws.diis_hist_count >= 2)
    {
        // Switch: ADIIS when error is large, CDIIS when small
        if (enorm > scf_ws.adiis_to_cdiis_threshold)
        {
            // ADIIS
            extrapolated = QC_ADIIS_Extrapolate(
                mol.nao, scf_ws.diis_space, scf_ws.adiis_count,
                scf_ws.adiis_head, scf_ws.d_diis_f_hist.data(),
                scf_ws.d_adiis_d_hist.data(), dF);
        }
        else
        {
            // CDIIS
            extrapolated = QC_DIIS_Extrapolate_Double(
                mol.nao, scf_ws.diis_space, scf_ws.diis_hist_count,
                scf_ws.diis_hist_head, scf_ws.d_diis_f_hist.data(),
                scf_ws.d_diis_e_hist.data(), scf_ws.diis_reg, dF,
                scf_ws.d_diis_B, scf_ws.d_diis_rhs, scf_ws.d_diis_info);
        }
        if (extrapolated)
        {
            for (int i = 0; i < nao2; i++)
                scf_ws.d_F[i] = (float)dF[i];
        }
    }

    if (!scf_ws.unrestricted) return;
    // Beta spin: simplified (CDIIS only for now)
    double* dFb = scf_ws.d_F_b_double;
    if (!dFb) return;
    QC_Build_DIIS_Error_Double(mol.nao, dFb, scf_ws.d_P_b, scf_ws.d_S,
                               scf_ws.d_diis_err);
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
                scf_ws.d_diis_B, scf_ws.d_diis_rhs, scf_ws.d_diis_info))
        {
            for (int i = 0; i < nao2; i++)
                scf_ws.d_F_b[i] = (float)dFb[i];
        }
    }
#endif
}

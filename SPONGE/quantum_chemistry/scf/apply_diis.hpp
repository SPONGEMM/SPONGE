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

// Compute DIIS error e = FPS - SPF in double precision (all pointers are HOST)
static void QC_Build_DIIS_Error_Double(int nao, const double* h_F,
                                       const double* h_P, const double* h_S,
                                       double* h_err)
{
    const int nao2 = nao * nao;
    std::vector<double> t1(nao2), t2(nao2), t3(nao2);
    const double one = 1.0, zero = 0.0;
    // t1 = F * P
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                nao, nao, nao, one, h_F, nao, h_P, nao,
                zero, t1.data(), nao);
    // t2 = FP * S
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                nao, nao, nao, one, t1.data(), nao, h_S, nao,
                zero, t2.data(), nao);
    // t1 = S * P
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                nao, nao, nao, one, h_S, nao, h_P, nao,
                zero, t1.data(), nao);
    // t3 = SP * F
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                nao, nao, nao, one, t1.data(), nao, h_F, nao,
                zero, t3.data(), nao);
    // err = FPS - SPF
    for (int i = 0; i < nao2; i++)
        h_err[i] = t2[i] - t3[i];
}

// Dot product of two double arrays (HOST pointers)
static double QC_Double_Dot(int n, const double* a, const double* b)
{
    double sum = 0.0;
    for (int i = 0; i < n; i++)
        sum += a[i] * b[i];
    return sum;
}

static bool QC_DIIS_Extrapolate_Double(int nao, int diis_space, int hist_count,
                                       int hist_head, double** h_f_hist,
                                       double** h_e_hist, double reg,
                                       double* h_f_out, double* h_B,
                                       double* h_rhs, int* h_info)
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
        h_rhs[i] = 0.0;
        for (int j = 0; j < n; j++) h_B[i * n + j] = 0.0;
    }
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
            double v = QC_Double_Dot(nao2, h_e_hist[hist_idx(i)],
                                     h_e_hist[hist_idx(j)]);
            if (i == j) v += reg;
            h_B[i * n + j] = v;
            h_B[j * n + i] = v;
        }
    }

    // Solve using eigenvalue decomposition (PySCF approach)
    h_info[0] = 0;
    {
        std::vector<double> H(n * n);
        memcpy(H.data(), h_B, sizeof(double) * n * n);
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
        if (info != 0) { h_info[0] = info; return false; }

        std::vector<double> c(n, 0.0);
        for (int k = 0; k < n; k++)
        {
            if (fabs(w[k]) < 1e-14) continue;
            double vg = 0.0;
            for (int i = 0; i < n; i++)
                vg += H[k * n + i] * h_rhs[i];
            double coeff = vg / w[k];
            for (int i = 0; i < n; i++)
                c[i] += coeff * H[k * n + i];
        }
        memcpy(h_rhs, c.data(), sizeof(double) * n);
    }

    // Extrapolate: F_out = sum_i c[i] * F_hist[i]
    memset(h_f_out, 0, sizeof(double) * nao2);
    for (int i = 0; i < m; i++)
    {
        double c = h_rhs[i];
        const double* fh = h_f_hist[hist_idx(i)];
        for (int idx = 0; idx < nao2; idx++)
            h_f_out[idx] += c * fh[idx];
    }
    return true;
}

static void QC_DIIS_Reset(int& hist_count, int& hist_head)
{
    hist_count = 0;
    hist_head = 0;
}

// ADIIS: minimize energy estimate over convex combination of stored Fock matrices
// All pointers are HOST memory
static bool QC_ADIIS_Extrapolate(int nao, int diis_space, int adiis_count,
                                  int adiis_head, double** h_f_hist,
                                  double** h_d_hist, double* h_f_out)
{
    if (adiis_count < 2) return false;
    const int m = std::min(adiis_count, diis_space);
    if (m < 2) return false;
    const int nao2 = nao * nao;
    auto hist_idx = [&](int i) { return (adiis_head + i) % diis_space; };

    // df[i,j] = Tr(D_i * F_j)
    std::vector<double> df(m * m);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++)
            df[i * m + j] = QC_Double_Dot(nao2, h_d_hist[hist_idx(i)],
                                           h_f_hist[hist_idx(j)]);

    // Build ADIIS quadratic form
    std::vector<double> dd_fn(m), dn_f(m);
    double dn_fn = df[(m-1) * m + (m-1)];
    for (int i = 0; i < m; i++)
    {
        dd_fn[i] = df[i * m + (m-1)] - dn_fn;
        dn_f[i] = df[(m-1) * m + i];
    }
    std::vector<double> df_adj(m * m);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++)
            df_adj[i * m + j] = df[i * m + j] - df[i * m + (m-1)]
                                 - df[(m-1) * m + j] + dn_fn;

    // Minimize cost(c) = 2*sum(c_i*dd_fn_i) + sum(c_i*df_adj_ij*c_j)
    // with c_i >= 0, sum(c_i) = 1
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
        for (int n = 0; n < m; n++)
        {
            for (int k = 0; k < m; k++)
            {
                double dc = 2.0 * x[n] * ((k == n ? x2sum : 0.0) - x[k] * x[k])
                            / (x2sum * x2sum);
                gx[n] += gc[k] * dc;
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

    // Extrapolate: F = sum(c_i * F_i)
    memset(h_f_out, 0, sizeof(double) * nao2);
    for (int i = 0; i < m; i++)
    {
        double ci = c[i];
        const double* fh = h_f_hist[hist_idx(i)];
        for (int idx = 0; idx < nao2; idx++)
            h_f_out[idx] += ci * fh[idx];
    }
    return true;
}

void QUANTUM_CHEMISTRY::Apply_DIIS(int iter)
{
    if (!scf_ws.use_diis || (iter + 1) < scf_ws.diis_start_iter) return;

    const int nao2 = (int)mol.nao2;
    const int diis_space = scf_ws.diis_space;

    // Copy F_double, P, S from device to host
    std::vector<double> h_F(nao2);
    if (scf_ws.d_F_double)
    {
        deviceMemcpy(h_F.data(), scf_ws.d_F_double, sizeof(double) * nao2,
                     deviceMemcpyDeviceToHost);
    }
    else
    {
        // No double Fock available, promote float
        std::vector<float> h_Ff(nao2);
        deviceMemcpy(h_Ff.data(), scf_ws.d_F, sizeof(float) * nao2,
                     deviceMemcpyDeviceToHost);
        for (int i = 0; i < nao2; i++) h_F[i] = (double)h_Ff[i];
    }

    std::vector<float> h_Pf(nao2), h_Sf(nao2);
    deviceMemcpy(h_Pf.data(), scf_ws.d_P, sizeof(float) * nao2,
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(h_Sf.data(), scf_ws.d_S, sizeof(float) * nao2,
                 deviceMemcpyDeviceToHost);
    std::vector<double> h_P(nao2), h_S(nao2);
    for (int i = 0; i < nao2; i++)
    {
        h_P[i] = (double)h_Pf[i];
        h_S[i] = (double)h_Sf[i];
    }

    // Copy DIIS history from device to host
    // Each history slot is nao2 doubles on device
    std::vector<std::vector<double>> h_f_hist(diis_space, std::vector<double>(nao2));
    std::vector<std::vector<double>> h_e_hist(diis_space, std::vector<double>(nao2));
    std::vector<std::vector<double>> h_d_hist(diis_space, std::vector<double>(nao2));
    std::vector<double*> h_f_hist_ptrs(diis_space);
    std::vector<double*> h_e_hist_ptrs(diis_space);
    std::vector<double*> h_d_hist_ptrs(diis_space);
    for (int i = 0; i < diis_space; i++)
    {
        deviceMemcpy(h_f_hist[i].data(), scf_ws.d_diis_f_hist[i],
                     sizeof(double) * nao2, deviceMemcpyDeviceToHost);
        deviceMemcpy(h_e_hist[i].data(), scf_ws.d_diis_e_hist[i],
                     sizeof(double) * nao2, deviceMemcpyDeviceToHost);
        deviceMemcpy(h_d_hist[i].data(), scf_ws.d_adiis_d_hist[i],
                     sizeof(double) * nao2, deviceMemcpyDeviceToHost);
        h_f_hist_ptrs[i] = h_f_hist[i].data();
        h_e_hist_ptrs[i] = h_e_hist[i].data();
        h_d_hist_ptrs[i] = h_d_hist[i].data();
    }

    // DIIS working buffers on host
    const int bn = diis_space + 1;
    std::vector<double> h_B(bn * bn, 0.0);
    std::vector<double> h_rhs(bn, 0.0);
    int h_info = 0;

    // Compute DIIS error
    std::vector<double> h_err(nao2);
    QC_Build_DIIS_Error_Double(mol.nao, h_F.data(), h_P.data(), h_S.data(),
                               h_err.data());
    double enorm = 0;
    for (int i = 0; i < nao2; i++)
        enorm += h_err[i] * h_err[i];
    enorm = sqrt(enorm);

    // Push F and error to CDIIS history (on host copies)
    {
        int write_idx = 0;
        if (scf_ws.diis_hist_count < diis_space)
        {
            write_idx = (scf_ws.diis_hist_head + scf_ws.diis_hist_count)
                        % diis_space;
            scf_ws.diis_hist_count++;
        }
        else
        {
            write_idx = scf_ws.diis_hist_head;
            scf_ws.diis_hist_head = (scf_ws.diis_hist_head + 1) % diis_space;
            scf_ws.diis_hist_count = diis_space;
        }
        memcpy(h_f_hist_ptrs[write_idx], h_F.data(), sizeof(double) * nao2);
        memcpy(h_e_hist_ptrs[write_idx], h_err.data(), sizeof(double) * nao2);
    }

    // Push density to ADIIS history
    {
        int& ac = scf_ws.adiis_count;
        int& ah = scf_ws.adiis_head;
        int ws = diis_space;
        int write_idx = (ac < ws) ? ((ah + ac) % ws) : ah;
        if (ac < ws) ac++;
        else ah = (ah + 1) % ws;
        memcpy(h_d_hist_ptrs[write_idx], h_P.data(), sizeof(double) * nao2);
    }

    bool extrapolated = false;
    if (scf_ws.diis_hist_count >= 2)
    {
        if (enorm > scf_ws.adiis_to_cdiis_threshold)
        {
            // ADIIS
            extrapolated = QC_ADIIS_Extrapolate(
                mol.nao, diis_space, scf_ws.adiis_count,
                scf_ws.adiis_head, h_f_hist_ptrs.data(),
                h_d_hist_ptrs.data(), h_F.data());
        }
        else
        {
            // CDIIS
            extrapolated = QC_DIIS_Extrapolate_Double(
                mol.nao, diis_space, scf_ws.diis_hist_count,
                scf_ws.diis_hist_head, h_f_hist_ptrs.data(),
                h_e_hist_ptrs.data(), scf_ws.diis_reg, h_F.data(),
                h_B.data(), h_rhs.data(), &h_info);
        }
        if (extrapolated)
        {
            // Copy extrapolated F back to device (both float and double)
            if (scf_ws.d_F_double)
            {
                deviceMemcpy(scf_ws.d_F_double, h_F.data(),
                             sizeof(double) * nao2, deviceMemcpyHostToDevice);
            }
            std::vector<float> h_Ff(nao2);
            for (int i = 0; i < nao2; i++) h_Ff[i] = (float)h_F[i];
            deviceMemcpy(scf_ws.d_F, h_Ff.data(), sizeof(float) * nao2,
                         deviceMemcpyHostToDevice);
        }
    }

    // Write back all DIIS history to device
    for (int i = 0; i < diis_space; i++)
    {
        deviceMemcpy(scf_ws.d_diis_f_hist[i], h_f_hist_ptrs[i],
                     sizeof(double) * nao2, deviceMemcpyHostToDevice);
        deviceMemcpy(scf_ws.d_diis_e_hist[i], h_e_hist_ptrs[i],
                     sizeof(double) * nao2, deviceMemcpyHostToDevice);
        deviceMemcpy(scf_ws.d_adiis_d_hist[i], h_d_hist_ptrs[i],
                     sizeof(double) * nao2, deviceMemcpyHostToDevice);
    }

    if (!scf_ws.unrestricted) return;

    // Beta spin
    std::vector<double> h_Fb(nao2);
    if (scf_ws.d_F_b_double)
    {
        deviceMemcpy(h_Fb.data(), scf_ws.d_F_b_double, sizeof(double) * nao2,
                     deviceMemcpyDeviceToHost);
    }
    else
    {
        std::vector<float> h_Fbf(nao2);
        deviceMemcpy(h_Fbf.data(), scf_ws.d_F_b, sizeof(float) * nao2,
                     deviceMemcpyDeviceToHost);
        for (int i = 0; i < nao2; i++) h_Fb[i] = (double)h_Fbf[i];
    }

    std::vector<float> h_Pb_f(nao2);
    deviceMemcpy(h_Pb_f.data(), scf_ws.d_P_b, sizeof(float) * nao2,
                 deviceMemcpyDeviceToHost);
    std::vector<double> h_Pb(nao2);
    for (int i = 0; i < nao2; i++) h_Pb[i] = (double)h_Pb_f[i];

    // Copy beta DIIS history from device
    std::vector<std::vector<double>> h_f_hist_b(diis_space, std::vector<double>(nao2));
    std::vector<std::vector<double>> h_e_hist_b(diis_space, std::vector<double>(nao2));
    std::vector<double*> h_f_hist_b_ptrs(diis_space);
    std::vector<double*> h_e_hist_b_ptrs(diis_space);
    for (int i = 0; i < diis_space; i++)
    {
        deviceMemcpy(h_f_hist_b[i].data(), scf_ws.d_diis_f_hist_b[i],
                     sizeof(double) * nao2, deviceMemcpyDeviceToHost);
        deviceMemcpy(h_e_hist_b[i].data(), scf_ws.d_diis_e_hist_b[i],
                     sizeof(double) * nao2, deviceMemcpyDeviceToHost);
        h_f_hist_b_ptrs[i] = h_f_hist_b[i].data();
        h_e_hist_b_ptrs[i] = h_e_hist_b[i].data();
    }

    std::vector<double> h_err_b(nao2);
    QC_Build_DIIS_Error_Double(mol.nao, h_Fb.data(), h_Pb.data(), h_S.data(),
                               h_err_b.data());

    // Push beta F and error to history
    {
        int write_idx = 0;
        if (scf_ws.diis_hist_count_b < diis_space)
        {
            write_idx = (scf_ws.diis_hist_head_b + scf_ws.diis_hist_count_b)
                        % diis_space;
            scf_ws.diis_hist_count_b++;
        }
        else
        {
            write_idx = scf_ws.diis_hist_head_b;
            scf_ws.diis_hist_head_b = (scf_ws.diis_hist_head_b + 1) % diis_space;
            scf_ws.diis_hist_count_b = diis_space;
        }
        memcpy(h_f_hist_b_ptrs[write_idx], h_Fb.data(), sizeof(double) * nao2);
        memcpy(h_e_hist_b_ptrs[write_idx], h_err_b.data(), sizeof(double) * nao2);
    }

    if (scf_ws.diis_hist_count_b >= 2)
    {
        if (QC_DIIS_Extrapolate_Double(
                mol.nao, diis_space, scf_ws.diis_hist_count_b,
                scf_ws.diis_hist_head_b, h_f_hist_b_ptrs.data(),
                h_e_hist_b_ptrs.data(), scf_ws.diis_reg, h_Fb.data(),
                h_B.data(), h_rhs.data(), &h_info))
        {
            if (scf_ws.d_F_b_double)
            {
                deviceMemcpy(scf_ws.d_F_b_double, h_Fb.data(),
                             sizeof(double) * nao2, deviceMemcpyHostToDevice);
            }
            std::vector<float> h_Fbf(nao2);
            for (int i = 0; i < nao2; i++) h_Fbf[i] = (float)h_Fb[i];
            deviceMemcpy(scf_ws.d_F_b, h_Fbf.data(), sizeof(float) * nao2,
                         deviceMemcpyHostToDevice);
        }
    }

    // Write back beta DIIS history to device
    for (int i = 0; i < diis_space; i++)
    {
        deviceMemcpy(scf_ws.d_diis_f_hist_b[i], h_f_hist_b_ptrs[i],
                     sizeof(double) * nao2, deviceMemcpyHostToDevice);
        deviceMemcpy(scf_ws.d_diis_e_hist_b[i], h_e_hist_b_ptrs[i],
                     sizeof(double) * nao2, deviceMemcpyHostToDevice);
    }
}

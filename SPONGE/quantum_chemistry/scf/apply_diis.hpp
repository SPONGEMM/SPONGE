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

static void QC_Build_DIIS_Error(BLAS_HANDLE blas_handle, int nao,
                                const float* d_F, const float* d_P,
                                const float* d_S, float* d_err, float* d_w1,
                                float* d_w2, float* d_w3, float* d_w4)
{
    const int nao2 = (int)nao * nao;
#ifndef USE_GPU
    // CPU path: compute DIIS error e = FPS - SPF in double precision
    // to avoid float32 accumulation errors in matmul for large basis sets
    std::vector<double> dF(nao2), dP(nao2), dS(nao2);
    std::vector<double> t1(nao2), t2(nao2), t3(nao2);
    for (int i = 0; i < nao2; i++)
    {
        dF[i] = (double)d_F[i];
        dP[i] = (double)d_P[i];
        dS[i] = (double)d_S[i];
    }
    // t1 = F * P
    for (int i = 0; i < nao; i++)
        for (int j = 0; j < nao; j++)
        {
            double s = 0.0;
            for (int k = 0; k < nao; k++)
                s += dF[i * nao + k] * dP[k * nao + j];
            t1[i * nao + j] = s;
        }
    // t2 = FP * S
    for (int i = 0; i < nao; i++)
        for (int j = 0; j < nao; j++)
        {
            double s = 0.0;
            for (int k = 0; k < nao; k++)
                s += t1[i * nao + k] * dS[k * nao + j];
            t2[i * nao + j] = s;
        }
    // t1 = S * P
    for (int i = 0; i < nao; i++)
        for (int j = 0; j < nao; j++)
        {
            double s = 0.0;
            for (int k = 0; k < nao; k++)
                s += dS[i * nao + k] * dP[k * nao + j];
            t1[i * nao + j] = s;
        }
    // t3 = SP * F
    for (int i = 0; i < nao; i++)
        for (int j = 0; j < nao; j++)
        {
            double s = 0.0;
            for (int k = 0; k < nao; k++)
                s += t1[i * nao + k] * dF[k * nao + j];
            t3[i * nao + j] = s;
        }
    // err = FPS - SPF
    for (int i = 0; i < nao2; i++)
        d_err[i] = (float)(t2[i] - t3[i]);
#else
    const int threads = 256;
    QC_MatMul_RowRow_Blas(blas_handle, nao, nao, nao, d_F, d_P, d_w1);
    QC_MatMul_RowRow_Blas(blas_handle, nao, nao, nao, d_w1, d_S, d_w2);
    QC_MatMul_RowRow_Blas(blas_handle, nao, nao, nao, d_S, d_P, d_w3);
    QC_MatMul_RowRow_Blas(blas_handle, nao, nao, nao, d_w3, d_F, d_w4);
    Launch_Device_Kernel(QC_Sub_Matrix_Kernel,
                         ((int)nao2 + threads - 1) / threads, threads, 0, 0,
                         (int)nao2, d_w2, d_w4, d_err);
#endif
}

static void QC_DIIS_History_Push_Device(int nao2, int diis_space,
                                        int& hist_count, int& hist_head,
                                        float** d_f_hist, float** d_e_hist,
                                        const float* d_f_new,
                                        const float* d_e_new)
{
    if (diis_space <= 0) return;
    const int bytes = sizeof(float) * (int)nao2;
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

static bool QC_DIIS_Extrapolate_Device(int nao, int diis_space, int hist_count,
                                       int hist_head, float** d_f_hist,
                                       float** d_e_hist, double reg,
                                       float* d_f_out, double* d_accum,
                                       double* d_B, double* d_rhs, int* d_info)
{
    if (hist_count < 2 || diis_space <= 0) return false;
    const int m = std::min(hist_count, diis_space);
    if (m < 2) return false;
    const int n = m + 1;
    const int nao2 = nao * nao;
    const int threads = 256;
    const int reduction_blocks = (nao2 + threads - 1) / threads;
    auto hist_idx = [&](int logical_idx) -> int
    { return (hist_head + logical_idx) % diis_space; };

    Launch_Device_Kernel(QC_DIIS_Init_System_Kernel, 1, 1, 0, 0, n, m, d_B,
                         d_rhs);
    for (int i = 0; i < m; i++)
    {
        const int i_idx = hist_idx(i);
        for (int j = 0; j <= i; j++)
        {
            const int j_idx = hist_idx(j);
            deviceMemset(d_accum, 0, sizeof(double));
            Launch_Device_Kernel(QC_Mat_Dot_Accumulate_Kernel, reduction_blocks,
                                 threads, 0, 0, nao2, d_e_hist[i_idx],
                                 d_e_hist[j_idx], d_accum);
            Launch_Device_Kernel(QC_DIIS_Set_B_From_Accum_Kernel, 1, 1, 0, 0, n,
                                 i, j, reg, d_accum, d_B);
        }
    }
    Launch_Device_Kernel(QC_DIIS_Solve_Linear_System_Kernel, 1, 1, 0, 0, n, d_B,
                         d_rhs, d_info);

    deviceMemset(d_f_out, 0, sizeof(float) * (int)nao2);
    for (int i = 0; i < m; i++)
    {
        const int i_idx = hist_idx(i);
        Launch_Device_Kernel(QC_Add_Scaled_Matrix_From_Double_Kernel,
                             ((nao2 + threads - 1) / threads), threads, 0, 0,
                             nao2, i, d_rhs, d_f_hist[i_idx], d_f_out);
    }
    return true;
}

static void QC_DIIS_Reset(int& hist_count, int& hist_head)
{
    hist_count = 0;
    hist_head = 0;
}

void QUANTUM_CHEMISTRY::Apply_DIIS(int iter)
{
    if (!scf_ws.use_diis || (iter + 1) < scf_ws.diis_start_iter) return;

    // After a DIIS reset, skip DIIS for a few iterations to let
    // plain density mixing stabilize before DIIS re-engages
    if (scf_ws.diis_cooldown > 0)
    {
        scf_ws.diis_cooldown--;
        return;
    }

    // Check if DIIS is stagnating: energy not improving
    // If so, reset DIIS subspace and enter cooldown
    {
        double h_energy = 0.0;
        deviceMemcpy(&h_energy, scf_ws.d_scf_energy, sizeof(double),
                     deviceMemcpyDeviceToHost);
        if (scf_ws.diis_hist_count >= 2)
        {
            const double improvement = scf_ws.diis_best_energy - h_energy;
            if (improvement > 1e-6)
            {
                scf_ws.diis_best_energy = h_energy;
                scf_ws.diis_stagnant_count = 0;
            }
            else
            {
                scf_ws.diis_stagnant_count++;
                if (scf_ws.diis_stagnant_count >= 3)
                {
                    QC_DIIS_Reset(scf_ws.diis_hist_count,
                                  scf_ws.diis_hist_head);
                    if (scf_ws.unrestricted)
                        QC_DIIS_Reset(scf_ws.diis_hist_count_b,
                                      scf_ws.diis_hist_head_b);
                    scf_ws.diis_stagnant_count = 0;
                    scf_ws.diis_best_energy = h_energy;
                    scf_ws.diis_cooldown = 3;
                    return;
                }
            }
        }
        else
        {
            scf_ws.diis_best_energy = h_energy;
            scf_ws.diis_stagnant_count = 0;
        }
    }

    QC_Build_DIIS_Error(blas_handle, mol.nao, scf_ws.d_F, scf_ws.d_P,
                        scf_ws.d_S, scf_ws.d_diis_err, scf_ws.d_diis_w1,
                        scf_ws.d_diis_w2, scf_ws.d_diis_w3, scf_ws.d_diis_w4);
    QC_DIIS_History_Push_Device(
        (int)mol.nao2, scf_ws.diis_space, scf_ws.diis_hist_count,
        scf_ws.diis_hist_head, scf_ws.d_diis_f_hist.data(),
        scf_ws.d_diis_e_hist.data(), scf_ws.d_F, scf_ws.d_diis_err);
    if (scf_ws.diis_hist_count >= 2)
    {
        QC_DIIS_Extrapolate_Device(
            mol.nao, scf_ws.diis_space, scf_ws.diis_hist_count,
            scf_ws.diis_hist_head, scf_ws.d_diis_f_hist.data(),
            scf_ws.d_diis_e_hist.data(), scf_ws.diis_reg, scf_ws.d_F,
            scf_ws.d_diis_accum, scf_ws.d_diis_B, scf_ws.d_diis_rhs,
            scf_ws.d_diis_info);
    }

    if (!scf_ws.unrestricted) return;

    QC_Build_DIIS_Error(blas_handle, mol.nao, scf_ws.d_F_b, scf_ws.d_P_b,
                        scf_ws.d_S, scf_ws.d_diis_err, scf_ws.d_diis_w1,
                        scf_ws.d_diis_w2, scf_ws.d_diis_w3, scf_ws.d_diis_w4);
    QC_DIIS_History_Push_Device(
        (int)mol.nao2, scf_ws.diis_space, scf_ws.diis_hist_count_b,
        scf_ws.diis_hist_head_b, scf_ws.d_diis_f_hist_b.data(),
        scf_ws.d_diis_e_hist_b.data(), scf_ws.d_F_b, scf_ws.d_diis_err);
    if (scf_ws.diis_hist_count_b >= 2)
    {
        QC_DIIS_Extrapolate_Device(
            mol.nao, scf_ws.diis_space, scf_ws.diis_hist_count_b,
            scf_ws.diis_hist_head_b, scf_ws.d_diis_f_hist_b.data(),
            scf_ws.d_diis_e_hist_b.data(), scf_ws.diis_reg, scf_ws.d_F_b,
            scf_ws.d_diis_accum, scf_ws.d_diis_B, scf_ws.d_diis_rhs,
            scf_ws.d_diis_info);
    }
}

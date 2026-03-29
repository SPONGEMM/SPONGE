#include "quantum_chemistry.h"
#include "structure/matrix.h"

// ====================== 单精度 BLAS/Solver 包装 ======================

int QC_Diagonalize_Workspace_Size(SOLVER_HANDLE solver_handle, int n,
                                  float* mat, float* w, float** work_ptr,
                                  void** iwork_ptr, int* lwork, int* liwork)
{
#ifdef USE_GPU
    if (work_ptr == NULL || iwork_ptr == NULL || lwork == NULL ||
        liwork == NULL || mat == NULL || w == NULL)
        return -1;

    *liwork = 0;
    int stat = (int)deviceSolverSsyevdBufferSize(
        solver_handle, DEVICE_EIG_MODE_VECTOR, DEVICE_FILL_MODE_UPPER, n, mat,
        n, w, lwork);
    if (stat != 0 || *lwork <= 0) return (stat != 0) ? stat : -2;

    if (*work_ptr != NULL)
    {
        deviceFree(*work_ptr);
        *work_ptr = NULL;
    }
    Device_Malloc_Safely((void**)work_ptr, sizeof(float) * (int)(*lwork));
    deviceMemset(*work_ptr, 0, sizeof(float) * (int)(*lwork));

    if (*iwork_ptr != NULL)
    {
        deviceFree(*iwork_ptr);
        *iwork_ptr = NULL;
    }
    return 0;
#elif defined(USE_MKL) || defined(USE_OPENBLAS)
    (void)solver_handle;
    if (work_ptr == NULL || iwork_ptr == NULL || lwork == NULL ||
        liwork == NULL || mat == NULL || w == NULL)
        return -1;
    if (n <= 0)
    {
        *lwork = 0;
        *liwork = 0;
        if (*work_ptr != NULL)
        {
            deviceFree(*work_ptr);
            *work_ptr = NULL;
        }
        if (*iwork_ptr != NULL)
        {
            deviceFree(*iwork_ptr);
            *iwork_ptr = NULL;
        }
        return 0;
    }

    float work_opt = 0.0f;
    lapack_int iwork_opt = 0;
    lapack_int info = LAPACKE_ssyevd_work(
        LAPACK_COL_MAJOR, 'V', 'U', (lapack_int)n, mat, (lapack_int)n, w,
        &work_opt, (lapack_int)-1, &iwork_opt, (lapack_int)-1);
    if (info != 0) return (int)info;
    *lwork = (int)(work_opt + 0.5f);
    if (*lwork < 1) *lwork = 1;
    *liwork = (int)iwork_opt;
    if (*liwork < 0) *liwork = 0;

    if (*work_ptr != NULL)
    {
        deviceFree(*work_ptr);
        *work_ptr = NULL;
    }
    Device_Malloc_Safely((void**)work_ptr, sizeof(float) * (int)(*lwork));
    deviceMemset(*work_ptr, 0, sizeof(float) * (int)(*lwork));

    if (*iwork_ptr != NULL)
    {
        deviceFree(*iwork_ptr);
        *iwork_ptr = NULL;
    }
    if (*liwork > 0)
    {
        Device_Malloc_Safely(iwork_ptr, sizeof(lapack_int) * (int)(*liwork));
        deviceMemset(*iwork_ptr, 0, sizeof(lapack_int) * (int)(*liwork));
    }

    return 0;
#else
    (void)solver_handle;
    (void)n;
    (void)mat;
    (void)w;
    (void)work_ptr;
    (void)iwork_ptr;
    (void)lwork;
    (void)liwork;
    return -1;
#endif
}

void QC_Diagonalize(SOLVER_HANDLE solver_handle, int n, float* mat, float* w,
                    float* work, int lwork, void* iwork, int liwork, int* info)
{
#ifdef USE_GPU
    deviceSolverSsyevd(solver_handle, DEVICE_EIG_MODE_VECTOR,
                       DEVICE_FILL_MODE_UPPER, n, mat, n, w, work, lwork, info);
#elif defined(USE_MKL) || defined(USE_OPENBLAS)
    *info = (int)LAPACKE_ssyevd_work(
        LAPACK_COL_MAJOR, 'V', 'U', (lapack_int)n, mat, (lapack_int)n, w, work,
        (lapack_int)lwork, (lapack_int*)iwork, (lapack_int)liwork);
#else
    *info = -1;
#endif
}

void QC_MatMul_RowRow_Blas(BLAS_HANDLE blas_handle, int m, int n, int kdim,
                           const float* A_row, const float* B_row, float* C_row)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;
    deviceBlasSgemm(blas_handle, DEVICE_BLAS_OP_N, DEVICE_BLAS_OP_N, n, m, kdim,
                    &alpha, B_row, n, A_row, kdim, &beta, C_row, n);
}

void QC_MatMul_RowCol_Blas(BLAS_HANDLE blas_handle, int m, int n, int kdim,
                           const float* A_row, const float* B_col, float* C_row)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;
    deviceBlasSgemm(blas_handle, DEVICE_BLAS_OP_T, DEVICE_BLAS_OP_N, n, m, kdim,
                    &alpha, B_col, kdim, A_row, kdim, &beta, C_row, n);
}

void QC_Build_Density_Blas(BLAS_HANDLE blas_handle, int nao, int n_occ,
                           float density_factor, const float* C_row,
                           float* P_new_row)
{
    const int nao2 = (int)nao * (int)nao;
    deviceMemset(P_new_row, 0, sizeof(float) * nao2);
    if (n_occ <= 0 || density_factor == 0.0f) return;

    const float alpha = density_factor;
    const float beta = 0.0f;
    deviceBlasSgemm(blas_handle, DEVICE_BLAS_OP_T, DEVICE_BLAS_OP_N, nao, nao,
                    n_occ, &alpha, C_row, nao, C_row, nao, &beta, P_new_row,
                    nao);
}

// ====================== 双精度 BLAS/Solver 包装 ======================

int QC_Diagonalize_Double_Workspace_Size(SOLVER_HANDLE solver_handle, int n,
                                         double* mat, double* w,
                                         double** work_ptr, int* lwork)
{
    if (n <= 0)
    {
        *lwork = 0;
        return 0;
    }
    int stat = (int)deviceSolverDsyevdBufferSize(
        solver_handle, DEVICE_EIG_MODE_VECTOR, DEVICE_FILL_MODE_UPPER, n, mat,
        n, w, lwork);
    if (stat != 0 || *lwork <= 0) return (stat != 0) ? stat : -2;
    if (*work_ptr)
    {
        deviceFree(*work_ptr);
        *work_ptr = NULL;
    }
    Device_Malloc_Safely((void**)work_ptr, sizeof(double) * (*lwork));
    deviceMemset(*work_ptr, 0, sizeof(double) * (*lwork));
    return 0;
}

void QC_Diagonalize_Double(SOLVER_HANDLE solver_handle, int n, double* mat,
                           double* w, double* work, int lwork, int* info)
{
    deviceSolverDsyevd(solver_handle, DEVICE_EIG_MODE_VECTOR,
                       DEVICE_FILL_MODE_UPPER, n, mat, n, w, work, lwork, info);
}

void QC_Dgemm_NN(BLAS_HANDLE handle, int m, int n, int k, const double* A,
                 int lda, const double* B, int ldb, double* C, int ldc)
{
    const double one = 1.0, zero = 0.0;
    deviceBlasDgemm(handle, DEVICE_BLAS_OP_N, DEVICE_BLAS_OP_N, n, m, k, &one,
                    B, ldb, A, lda, &zero, C, ldc);
}

void QC_Dgemm_TN(BLAS_HANDLE handle, int m, int n, int k, const double* A,
                 int lda, const double* B, int ldb, double* C, int ldc)
{
    const double one = 1.0, zero = 0.0;
    deviceBlasDgemm(handle, DEVICE_BLAS_OP_N, DEVICE_BLAS_OP_T, n, m, k, &one,
                    B, ldb, A, lda, &zero, C, ldc);
}

void QC_Dgemm_NT(BLAS_HANDLE handle, int m, int n, int k, const double* A,
                 int lda, const double* B, int ldb, double* C, int ldc)
{
    const double one = 1.0, zero = 0.0;
    deviceBlasDgemm(handle, DEVICE_BLAS_OP_T, DEVICE_BLAS_OP_N, n, m, k, &one,
                    B, ldb, A, lda, &zero, C, ldc);
}

// ====================== 常用通用矩阵函数包装 ======================

static __global__ void QC_Add_Matrix_Kernel(const int n, const float* A,
                                            const float* B, float* C)
{
    SIMPLE_DEVICE_FOR(idx, n) { C[idx] = A[idx] + B[idx]; }
}

void QC_Add_Matrix(int n, const float* A, const float* B, float* C)
{
    const int threads = 256;
    Launch_Device_Kernel(QC_Add_Matrix_Kernel, (n + threads - 1) / threads,
                         threads, 0, 0, n, A, B, C);
}

static __global__ void QC_Sub_Matrix_Kernel(const int n, const float* A,
                                            const float* B, float* C)
{
    SIMPLE_DEVICE_FOR(idx, n) { C[idx] = A[idx] - B[idx]; }
}

void QC_Sub_Matrix(int n, const float* A, const float* B, float* C)
{
    const int threads = 256;
    Launch_Device_Kernel(QC_Sub_Matrix_Kernel, (n + threads - 1) / threads,
                         threads, 0, 0, n, A, B, C);
}

static __global__ void QC_Scale_Matrix_By_Norms_Kernel(const int nao,
                                                       const float* norms,
                                                       float* M)
{
    const int total = nao * nao;
    SIMPLE_DEVICE_FOR(idx, total)
    {
        int i = idx / nao;
        int j = idx - i * nao;
        M[idx] *= norms[i] * norms[j];
    }
}

void QC_Scale_Matrix_By_Norms(int nao, const float* norms, float* M)
{
    const int threads = 256;
    const int total = nao * nao;
    Launch_Device_Kernel(QC_Scale_Matrix_By_Norms_Kernel,
                         (total + threads - 1) / threads, threads, 0, 0, nao,
                         norms, M);
}

static __global__ void QC_Float_To_Double_Kernel(const int n, const float* src,
                                                 double* dst)
{
    SIMPLE_DEVICE_FOR(i, n) { dst[i] = (double)src[i]; }
}

void QC_Float_To_Double(int n, const float* src, double* dst)
{
    const int threads = 256;
    Launch_Device_Kernel(QC_Float_To_Double_Kernel, (n + threads - 1) / threads,
                         threads, 0, 0, n, src, dst);
}

static __global__ void QC_Double_To_Float_Kernel(const int n, const double* src,
                                                 float* dst)
{
    SIMPLE_DEVICE_FOR(i, n) { dst[i] = (float)src[i]; }
}

void QC_Double_To_Float(int n, const double* src, float* dst)
{
    const int threads = 256;
    Launch_Device_Kernel(QC_Double_To_Float_Kernel, (n + threads - 1) / threads,
                         threads, 0, 0, n, src, dst);
}

static __global__ void QC_Float_To_Double_Copy_Kernel(const int n,
                                                      const float* src,
                                                      double* dst)
{
    SIMPLE_DEVICE_FOR(i, n) { dst[i] = (double)src[i]; }
}

void QC_Float_To_Double_Copy(int n, const float* src, double* dst)
{
    const int threads = 256;
    Launch_Device_Kernel(QC_Float_To_Double_Copy_Kernel,
                         (n + threads - 1) / threads, threads, 0, 0, n, src,
                         dst);
}

static __global__ void QC_Double_Dot_Kernel(const int n, const double* A,
                                            const double* B, double* out_sum)
{
    SIMPLE_DEVICE_FOR(i, n) { atomicAdd(out_sum, A[i] * B[i]); }
}

void QC_Double_Dot(int n, const double* A, const double* B, double* out_sum)
{
    const int threads = 256;
    Launch_Device_Kernel(QC_Double_Dot_Kernel, (n + threads - 1) / threads,
                         threads, 0, 0, n, A, B, out_sum);
}

static __global__ void QC_Double_Axpy_Kernel(const int n, const double coeff,
                                             const double* src, double* dst)
{
    SIMPLE_DEVICE_FOR(i, n) { dst[i] += coeff * src[i]; }
}

void QC_Double_Axpy(int n, double coeff, const double* src, double* dst)
{
    const int threads = 256;
    Launch_Device_Kernel(QC_Double_Axpy_Kernel, (n + threads - 1) / threads,
                         threads, 0, 0, n, coeff, src, dst);
}

static __global__ void QC_Double_Sub_Kernel(const int n, const double* A,
                                            const double* B, double* dst)
{
    SIMPLE_DEVICE_FOR(i, n) { dst[i] = A[i] - B[i]; }
}

void QC_Double_Sub(int n, const double* A, const double* B, double* dst)
{
    const int threads = 256;
    Launch_Device_Kernel(QC_Double_Sub_Kernel, (n + threads - 1) / threads,
                         threads, 0, 0, n, A, B, dst);
}

// ====================== 常用SCF矩阵函数包装 ======================

static __global__ void QC_Elec_Energy_Accumulate_Kernel(const int nao2,
                                                        const float* P,
                                                        const float* H_core,
                                                        const float* F,
                                                        double* out_sum)
{
    SIMPLE_DEVICE_FOR(idx, nao2)
    {
        atomicAdd(out_sum, 0.5 * (double)P[idx] *
                               ((double)H_core[idx] + (double)F[idx]));
    }
}

void QC_Elec_Energy_Accumulate(int nao2, const float* P, const float* H_core,
                               const float* F, double* out_sum)
{
    const int threads = 256;
    Launch_Device_Kernel(QC_Elec_Energy_Accumulate_Kernel,
                         (nao2 + threads - 1) / threads, threads, 0, 0, nao2, P,
                         H_core, F, out_sum);
}

static __global__ void QC_Mat_Dot_Accumulate_Kernel(const int nao2,
                                                    const float* A,
                                                    const float* B,
                                                    double* out_sum)
{
    SIMPLE_DEVICE_FOR(idx, nao2)
    {
        atomicAdd(out_sum, (double)A[idx] * (double)B[idx]);
    }
}

void QC_Mat_Dot_Accumulate(int nao2, const float* A, const float* B,
                           double* out_sum)
{
    const int threads = 256;
    Launch_Device_Kernel(QC_Mat_Dot_Accumulate_Kernel,
                         (nao2 + threads - 1) / threads, threads, 0, 0, nao2, A,
                         B, out_sum);
}

static __global__ void QC_Level_Shift_Kernel(const int n, const double ls,
                                             const double* dS,
                                             const double* dSPS, double* dF)
{
    SIMPLE_DEVICE_FOR(i, n) { dF[i] += ls * (dS[i] - 0.5 * dSPS[i]); }
}

void QC_Level_Shift(int n, double ls, const double* dS, const double* dSPS,
                    double* dF)
{
    const int threads = 256;
    Launch_Device_Kernel(QC_Level_Shift_Kernel, (n + threads - 1) / threads,
                         threads, 0, 0, n, ls, dS, dSPS, dF);
}

static __global__ void QC_Build_X_Canonical_Kernel(
    const int nao, const int nao_eff, const double* eigvec_col,
    const double* eigval, const double lindep_thresh, double* X_row)
{
    SIMPLE_DEVICE_FOR(i, nao)
    {
        int col = 0;
        for (int k = 0; k < nao; k++)
        {
            if (eigval[k] < lindep_thresh) continue;
            X_row[i * nao + col] = eigvec_col[i + k * nao] / sqrt(eigval[k]);
            col++;
        }
    }
}

void QC_Build_X_Canonical(int nao, int nao_eff, const double* eigvec_col,
                          const double* eigval, double lindep_thresh,
                          double* X_row)
{
    const int threads = 256;
    Launch_Device_Kernel(QC_Build_X_Canonical_Kernel,
                         (nao + threads - 1) / threads, threads, 0, 0, nao,
                         nao_eff, eigvec_col, eigval, lindep_thresh, X_row);
}

static __global__ void QC_Rect_Double_To_Padded_Float_Kernel(const int nao,
                                                             const int ne,
                                                             const double* src,
                                                             float* dst)
{
    SIMPLE_DEVICE_FOR(idx, nao * nao)
    {
        int i = idx / nao;
        int j = idx % nao;
        dst[idx] = (j < ne) ? (float)src[i * ne + j] : 0.0f;
    }
}

void QC_Rect_Double_To_Padded_Float(int nao, int ne, const double* src,
                                    float* dst)
{
    const int nao2 = nao * nao;
    const int threads = 256;
    Launch_Device_Kernel(QC_Rect_Double_To_Padded_Float_Kernel,
                         (nao2 + threads - 1) / threads, threads, 0, 0, nao, ne,
                         src, dst);
}

// ====================== 笛卡尔基转球谐基 ======================

// 用于将归一化笛卡尔基映射到归一化实球谐基
// l=2（d 轨道）
static const float CART2SPH_MAT_D[6][5] = {
    {0.00000000f, 0.00000000f, -0.31539157f, 0.00000000f, 0.54627422f},
    {1.09254843f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f},
    {0.00000000f, 0.00000000f, 0.00000000f, 1.09254843f, 0.00000000f},
    {0.00000000f, 0.00000000f, -0.31539157f, 0.00000000f, -0.54627422f},
    {0.00000000f, 1.09254843f, 0.00000000f, 0.00000000f, 0.00000000f},
    {0.00000000f, 0.00000000f, 0.63078313f, 0.00000000f, 0.00000000f},
};

// l=3（f 轨道）
static const float CART2SPH_MAT_F[10][7] = {
    {0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, -0.45704580f,
     0.00000000f, 0.59004359f},
    {1.77013077f, 0.00000000f, -0.45704580f, 0.00000000f, 0.00000000f,
     0.00000000f, 0.00000000f},
    {0.00000000f, 0.00000000f, 0.00000000f, -1.11952900f, 0.00000000f,
     1.44530572f, 0.00000000f},
    {0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, -0.45704580f,
     0.00000000f, -1.77013077f},
    {0.00000000f, 2.89061144f, 0.00000000f, 0.00000000f, 0.00000000f,
     0.00000000f, 0.00000000f},
    {0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 1.82818320f,
     0.00000000f, 0.00000000f},
    {-0.59004359f, 0.00000000f, -0.45704580f, 0.00000000f, 0.00000000f,
     0.00000000f, 0.00000000f},
    {0.00000000f, 0.00000000f, 0.00000000f, -1.11952900f, 0.00000000f,
     -1.44530572f, 0.00000000f},
    {0.00000000f, 0.00000000f, 1.82818320f, 0.00000000f, 0.00000000f,
     0.00000000f, 0.00000000f},
    {0.00000000f, 0.00000000f, 0.00000000f, 0.74635267f, 0.00000000f,
     0.00000000f, 0.00000000f},
};

// l=4（g 轨道）
static const float CART2SPH_MAT_G[15][9] = {
    {0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.31735664f,
     0.00000000f, -0.47308735f, 0.00000000f, 0.62583574f},
    {2.50334294f, 0.00000000f, -0.94617470f, 0.00000000f, 0.00000000f,
     0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f},
    {0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f,
     -2.00713963f, 0.00000000f, 1.77013077f, 0.00000000f},
    {0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.63471328f,
     0.00000000f, 0.00000000f, 0.00000000f, -3.75501441f},
    {0.00000000f, 5.31039231f, 0.00000000f, -2.00713963f, 0.00000000f,
     0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f},
    {0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, -2.53885313f,
     0.00000000f, 2.83852409f, 0.00000000f, 0.00000000f},
    {-2.50334294f, 0.00000000f, -0.94617470f, 0.00000000f, 0.00000000f,
     0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f},
    {0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f,
     -2.00713963f, 0.00000000f, -5.31039231f, 0.00000000f},
    {0.00000000f, 0.00000000f, 5.67704817f, 0.00000000f, 0.00000000f,
     0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f},
    {0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f,
     2.67618617f, 0.00000000f, 0.00000000f, 0.00000000f},
    {0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.31735664f,
     0.00000000f, 0.47308735f, 0.00000000f, 0.62583574f},
    {0.00000000f, -1.77013077f, 0.00000000f, -2.00713963f, 0.00000000f,
     0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f},
    {0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, -2.53885313f,
     0.00000000f, -2.83852409f, 0.00000000f, 0.00000000f},
    {0.00000000f, 0.00000000f, 0.00000000f, 2.67618617f, 0.00000000f,
     0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f},
    {0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.84628438f,
     0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f},
};

static __global__ void QC_Cart2Sph_MatMul_UT_RowRow_Kernel(
    const int m, const int n, const int kdim, const float* U_row_k_m,
    const float* B_row_k_n, float* C_row_m_n)
{
    SIMPLE_DEVICE_FOR(idx, m * n)
    {
        int i = idx / n;
        int j = idx - i * n;
        double sum = 0.0;
        for (int k = 0; k < kdim; k++)
        {
            sum += (double)U_row_k_m[k * m + i] * (double)B_row_k_n[k * n + j];
        }
        C_row_m_n[i * n + j] = (float)sum;
    }
}

void QUANTUM_CHEMISTRY::Build_Cart2Sph_Matrix()
{
    int nao_c = mol.nao_cart;
    int nao_s = mol.nao_sph;
    std::vector<float> cart2sph_mat(nao_c * nao_s, 0.0f);

    int offset_c = 0;
    int offset_s = 0;

    for (int k = 0; k < mol.h_l_list.size(); k++)
    {
        int l = mol.h_l_list[k];
        int dim_c = (l + 1) * (l + 2) / 2;
        int dim_s = 2 * l + 1;

        switch (l)
        {
            case 0:  // s 轨道
                cart2sph_mat[offset_c * nao_s + offset_s] = 0.28209479f;
                break;
            case 1:  // p 轨道
                cart2sph_mat[(offset_c + 0) * nao_s + (offset_s + 0)] =
                    0.48860251f;
                cart2sph_mat[(offset_c + 1) * nao_s + (offset_s + 1)] =
                    0.48860251f;
                cart2sph_mat[(offset_c + 2) * nao_s + (offset_s + 2)] =
                    0.48860251f;
                break;
            case 2:  // d 轨道
                for (int i = 0; i < 6; i++)
                    for (int j = 0; j < 5; j++)
                        cart2sph_mat[(offset_c + i) * nao_s + (offset_s + j)] =
                            CART2SPH_MAT_D[i][j];
                break;
            case 3:  // f 轨道
                for (int i = 0; i < 10; i++)
                    for (int j = 0; j < 7; j++)
                        cart2sph_mat[(offset_c + i) * nao_s + (offset_s + j)] =
                            CART2SPH_MAT_F[i][j];
                break;
            case 4:  // g 轨道
                for (int i = 0; i < 15; i++)
                    for (int j = 0; j < 9; j++)
                        cart2sph_mat[(offset_c + i) * nao_s + (offset_s + j)] =
                            CART2SPH_MAT_G[i][j];
                break;
        }
        offset_c += dim_c;
        offset_s += dim_s;
    }
    if (cart2sph_mat.empty())
    {
        cart2sph.d_cart2sph_mat = nullptr;
    }
    else
    {
        Device_Malloc_Safely((void**)&cart2sph.d_cart2sph_mat,
                             sizeof(float) * cart2sph_mat.size());
        deviceMemcpy(cart2sph.d_cart2sph_mat, cart2sph_mat.data(),
                     sizeof(float) * cart2sph_mat.size(),
                     deviceMemcpyHostToDevice);
    }
}

void QUANTUM_CHEMISTRY::Cart2Sph_Single_Matrix(float* d_cart, float* d_sph)
{
    if (!mol.is_spherical) return;
    const int nao_c = mol.nao_cart;
    const int nao_s = mol.nao_sph;
    const int threads = 256;
    const int total = nao_s * nao_s;
    QC_MatMul_RowRow_Blas(blas_handle, nao_c, nao_s, nao_c, d_cart,
                          cart2sph.d_cart2sph_mat, cart2sph.d_cart2sph_1e_tmp);
    Launch_Device_Kernel(QC_Cart2Sph_MatMul_UT_RowRow_Kernel,
                         (total + threads - 1) / threads, threads, 0, 0, nao_s,
                         nao_s, nao_c, cart2sph.d_cart2sph_mat,
                         cart2sph.d_cart2sph_1e_tmp, d_sph);
}

void QUANTUM_CHEMISTRY::Cart2Sph_OneE_Integrals()
{
    if (!mol.is_spherical) return;
    Cart2Sph_Single_Matrix(cart2sph.d_S_cart, scf_ws.core.d_S);
    Cart2Sph_Single_Matrix(cart2sph.d_T_cart, scf_ws.core.d_T);
    Cart2Sph_Single_Matrix(cart2sph.d_V_cart, scf_ws.core.d_V);
}

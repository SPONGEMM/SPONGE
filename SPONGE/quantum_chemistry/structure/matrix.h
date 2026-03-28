#ifndef QC_STRUCTURE_MATRIX_H
#define QC_STRUCTURE_MATRIX_H

#include "../../common.h"

// ====================== Float BLAS/Solver wrappers ======================

int QC_Diagonalize_Workspace_Size(SOLVER_HANDLE solver_handle, int n,
                                  float* mat, float* w, float** work_ptr,
                                  void** iwork_ptr, int* lwork, int* liwork);

void QC_Diagonalize(SOLVER_HANDLE solver_handle, int n, float* mat, float* w,
                    float* work, int lwork, void* iwork, int liwork, int* info);

void QC_MatMul_RowRow_Blas(BLAS_HANDLE blas_handle, int m, int n, int kdim,
                           const float* A_row, const float* B_row,
                           float* C_row);

void QC_MatMul_RowCol_Blas(BLAS_HANDLE blas_handle, int m, int n, int kdim,
                           const float* A_row, const float* B_col,
                           float* C_row);

void QC_Build_Density_Blas(BLAS_HANDLE blas_handle, int nao, int n_occ,
                           float density_factor, const float* C_row,
                           float* P_new_row);

// ====================== Double BLAS/Solver wrappers ======================

int QC_Diagonalize_Double_Workspace_Size(SOLVER_HANDLE solver_handle, int n,
                                         double* mat, double* w,
                                         double** work_ptr, int* lwork);

void QC_Diagonalize_Double(SOLVER_HANDLE solver_handle, int n, double* mat,
                           double* w, double* work, int lwork, int* info);

void QC_Dgemm_NN(BLAS_HANDLE handle, int m, int n, int k, const double* A,
                 int lda, const double* B, int ldb, double* C, int ldc);

void QC_Dgemm_TN(BLAS_HANDLE handle, int m, int n, int k, const double* A,
                 int lda, const double* B, int ldb, double* C, int ldc);

void QC_Dgemm_NT(BLAS_HANDLE handle, int m, int n, int k, const double* A,
                 int lda, const double* B, int ldb, double* C, int ldc);

// ====================== Common matrix utility wrappers ======================

void QC_Add_Matrix(int n, const float* A, const float* B, float* C);
void QC_Sub_Matrix(int n, const float* A, const float* B, float* C);
void QC_Scale_Matrix_By_Norms(int nao, const float* norms, float* M);

void QC_Float_To_Double(int n, const float* src, double* dst);
void QC_Double_To_Float(int n, const double* src, float* dst);
void QC_Float_To_Double_Copy(int n, const float* src, double* dst);

void QC_Level_Shift(int n, double ls, const double* dS, const double* dSPS,
                    double* dF);

void QC_Build_X_Canonical(int nao, int nao_eff, const double* eigvec_col,
                          const double* eigval, double lindep_thresh,
                          double* X_row);

void QC_Rect_Double_To_Padded_Float(int nao, int ne, const double* src,
                                    float* dst);

void QC_Double_Dot(int n, const double* A, const double* B, double* out_sum);
void QC_Double_Axpy(int n, double coeff, const double* src, double* dst);
void QC_Double_Sub(int n, const double* A, const double* B, double* dst);

// ====================== Common SCF matrix utility wrappers
// ======================

void QC_Elec_Energy_Accumulate(int nao2, const float* P, const float* H_core,
                               const float* F, double* out_sum);

void QC_Mat_Dot_Accumulate(int nao2, const float* A, const float* B,
                           double* out_sum);

void QC_Level_Shift(int n, double ls, const double* dS, const double* dSPS,
                    double* dF);

void QC_Build_X_Canonical(int nao, int nao_eff, const double* eigvec_col,
                          const double* eigval, double lindep_thresh,
                          double* X_row);

void QC_Rect_Double_To_Padded_Float(int nao, int ne, const double* src,
                                    float* dst);

#endif

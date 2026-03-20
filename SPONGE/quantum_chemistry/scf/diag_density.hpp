#pragma once

void QUANTUM_CHEMISTRY::Diagonalize_And_Build_Density()
{
    const int nao = mol.nao;
    const int nao2 = mol.nao2;

#ifndef USE_GPU
    // CPU path: use dgemm for Fp = X^T * F * X to avoid float32 sgemm
    // accumulation errors that corrupt eigenvalues for large basis sets.
    // F and X are float, but matmul is done in double.
    {
        // Promote F to double; X is already double
        std::vector<double> dF(nao2), dTmp(nao2), dFp(nao2);
        for (int i = 0; i < nao2; i++)
            dF[i] = (double)scf_ws.d_F[i];
        const double* dX = scf_ws.d_X;
        // SPONGE stores matrices in row-major order.
        // cblas_dgemm with CblasRowMajor handles this directly.
        // Tmp = F * X  (nao x nao) * (nao x nao)
        const double one = 1.0, zero = 0.0;
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    nao, nao, nao, one, dF.data(), nao, dX, nao,
                    zero, dTmp.data(), nao);
        // Fp = X^T * Tmp  (nao x nao)^T * (nao x nao)
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    nao, nao, nao, one, dX, nao, dTmp.data(), nao,
                    zero, dFp.data(), nao);
        // Diagonalize Fp in double using dsyevd
        // dFp is row-major; LAPACK COL_MAJOR + 'U' reads upper triangle
        // For symmetric matrix: row-major upper = col-major lower
        std::vector<double> dW(nao);
        {
            int lw = -1, liw = -1;
            double wq; lapack_int iwq;
            LAPACKE_dsyevd_work(LAPACK_COL_MAJOR, 'V', 'L', (lapack_int)nao,
                                dFp.data(), (lapack_int)nao, dW.data(),
                                &wq, lw, &iwq, liw);
            lw = (int)wq; liw = iwq;
            std::vector<double> dwork(lw);
            std::vector<lapack_int> diwork(liw);
            LAPACKE_dsyevd_work(LAPACK_COL_MAJOR, 'V', 'L', (lapack_int)nao,
                                dFp.data(), (lapack_int)nao, dW.data(),
                                dwork.data(), (lapack_int)lw,
                                diwork.data(), (lapack_int)liw);
        }
        // dFp now holds eigenvectors in col-major layout
        // Store eigenvalues to float for other uses
        for (int i = 0; i < nao; i++)
            scf_ws.d_W[i] = (float)dW[i];

        // C = X * eigvec (both double, X row-major, eigvec col-major)
        // col-major eigvec viewed as row-major = transposed
        std::vector<double> dC(nao2);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    nao, nao, nao, 1.0, dX, nao, dFp.data(), nao,
                    0.0, dC.data(), nao);
        for (int i = 0; i < nao2; i++)
            scf_ws.d_C[i] = (float)dC[i];
    }

    // P_new = occ_factor * C_occ * C_occ^T (use sgemm, C is row-major float)
    QC_Build_Density_Blas(blas_handle, nao, scf_ws.n_alpha,
                          scf_ws.occ_factor, scf_ws.d_C, scf_ws.d_P_new);

    if (!scf_ws.unrestricted) return;

    // Beta spin: same pipeline
    {
        std::vector<double> dF(nao2), dTmp(nao2), dFp(nao2);
        for (int i = 0; i < nao2; i++)
            dF[i] = (double)scf_ws.d_F_b[i];
        const double* dX = scf_ws.d_X;
        const double one = 1.0, zero = 0.0;
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    nao, nao, nao, one, dF.data(), nao, dX, nao,
                    zero, dTmp.data(), nao);
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    nao, nao, nao, one, dX, nao, dTmp.data(), nao,
                    zero, dFp.data(), nao);
        std::vector<double> dW(nao);
        {
            int lw = -1, liw = -1;
            double wq; lapack_int iwq;
            LAPACKE_dsyevd_work(LAPACK_COL_MAJOR, 'V', 'L', (lapack_int)nao,
                                dFp.data(), (lapack_int)nao, dW.data(),
                                &wq, lw, &iwq, liw);
            lw = (int)wq; liw = iwq;
            std::vector<double> dwork(lw);
            std::vector<lapack_int> diwork(liw);
            LAPACKE_dsyevd_work(LAPACK_COL_MAJOR, 'V', 'L', (lapack_int)nao,
                                dFp.data(), (lapack_int)nao, dW.data(),
                                dwork.data(), (lapack_int)lw,
                                diwork.data(), (lapack_int)liw);
        }
        for (int i = 0; i < nao; i++)
            scf_ws.d_W[i] = (float)dW[i];
        std::vector<double> dC(nao2);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    nao, nao, nao, 1.0, dX, nao, dFp.data(), nao,
                    0.0, dC.data(), nao);
        for (int i = 0; i < nao2; i++)
            scf_ws.d_C_b[i] = (float)dC[i];
    }
    QC_Build_Density_Blas(blas_handle, nao, scf_ws.n_beta, 1.0f,
                          scf_ws.d_C_b, scf_ws.d_P_b_new);

#else
    // GPU path: keep original sgemm
    QC_MatMul_RowRow_Blas(blas_handle, mol.nao, mol.nao, mol.nao, scf_ws.d_F,
                          scf_ws.d_X, scf_ws.d_Tmp);
    QC_MatMul_RowRow_Blas(blas_handle, mol.nao, mol.nao, mol.nao, scf_ws.d_X,
                          scf_ws.d_Tmp, scf_ws.d_Fp);

    deviceMemcpy(scf_ws.d_Work, scf_ws.d_Fp, sizeof(float) * mol.nao2,
                 deviceMemcpyDeviceToDevice);
    QC_Diagonalize(solver_handle, mol.nao, scf_ws.d_Work, scf_ws.d_W,
                   scf_ws.d_solver_work, scf_ws.lwork, scf_ws.d_solver_iwork,
                   scf_ws.liwork, scf_ws.d_info);
    deviceMemcpy(scf_ws.d_Fp, scf_ws.d_Work, sizeof(float) * mol.nao2,
                 deviceMemcpyDeviceToDevice);

    QC_MatMul_RowCol_Blas(blas_handle, mol.nao, mol.nao, mol.nao, scf_ws.d_X,
                          scf_ws.d_Fp, scf_ws.d_C);
    QC_Build_Density_Blas(blas_handle, mol.nao, scf_ws.n_alpha,
                          scf_ws.occ_factor, scf_ws.d_C, scf_ws.d_P_new);

    if (!scf_ws.unrestricted) return;

    QC_MatMul_RowRow_Blas(blas_handle, mol.nao, mol.nao, mol.nao, scf_ws.d_F_b,
                          scf_ws.d_X, scf_ws.d_Tmp);
    QC_MatMul_RowRow_Blas(blas_handle, mol.nao, mol.nao, mol.nao, scf_ws.d_X,
                          scf_ws.d_Tmp, scf_ws.d_Fp_b);

    deviceMemcpy(scf_ws.d_Work, scf_ws.d_Fp_b, sizeof(float) * mol.nao2,
                 deviceMemcpyDeviceToDevice);
    QC_Diagonalize(solver_handle, mol.nao, scf_ws.d_Work, scf_ws.d_W,
                   scf_ws.d_solver_work, scf_ws.lwork, scf_ws.d_solver_iwork,
                   scf_ws.liwork, scf_ws.d_info);
    deviceMemcpy(scf_ws.d_Fp_b, scf_ws.d_Work, sizeof(float) * mol.nao2,
                 deviceMemcpyDeviceToDevice);

    QC_MatMul_RowCol_Blas(blas_handle, mol.nao, mol.nao, mol.nao, scf_ws.d_X,
                          scf_ws.d_Fp_b, scf_ws.d_C_b);
    QC_Build_Density_Blas(blas_handle, mol.nao, scf_ws.n_beta, 1.0f,
                          scf_ws.d_C_b, scf_ws.d_P_b_new);
#endif
}

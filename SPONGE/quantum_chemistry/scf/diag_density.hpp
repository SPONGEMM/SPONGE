#pragma once

void QUANTUM_CHEMISTRY::Diagonalize_And_Build_Density()
{
    const int nao = mol.nao;
    const int nao2 = mol.nao2;

#ifndef USE_GPU
    // CPU path: canonical orthogonalization with rectangular X (nao × nao_eff)
    // X is stored row-major in d_X with stride nao, columns 0..nao_eff-1
    {
        const int ne = scf_ws.nao_eff > 0 ? scf_ws.nao_eff : nao;
        const double* dX = scf_ws.d_X;  // nao × ne, stride nao

        // Use d_F_double (DIIS extrapolates in double)
        std::vector<double> dF(nao2);
        if (scf_ws.d_F_double)
            for (int i = 0; i < nao2; i++) dF[i] = scf_ws.d_F_double[i];
        else
            for (int i = 0; i < nao2; i++) dF[i] = (double)scf_ws.d_F[i];

        // Level shift: F += shift * (S - 0.5 * SPS)
        const double ls = scf_ws.level_shift;
        if (ls > 0.0)
        {
            std::vector<double> dS(nao2), dP(nao2), dSP(nao2), dSPS(nao2);
            for (int i = 0; i < nao2; i++)
            {
                dS[i] = (double)scf_ws.d_S[i];
                dP[i] = (double)scf_ws.d_P[i];
            }
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        nao, nao, nao, 1.0, dS.data(), nao, dP.data(), nao,
                        0.0, dSP.data(), nao);
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        nao, nao, nao, 1.0, dSP.data(), nao, dS.data(), nao,
                        0.0, dSPS.data(), nao);
            for (int i = 0; i < nao2; i++)
                dF[i] += ls * (dS[i] - 0.5 * dSPS[i]);
        }

        // Tmp = F * X: (nao×nao) @ (nao×ne) → nao×ne, stored with stride ne
        std::vector<double> dTmp(nao * ne);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    nao, ne, nao, 1.0, dF.data(), nao, dX, nao,
                    0.0, dTmp.data(), ne);
        // Fp = X^T * Tmp: (ne×nao) @ (nao×ne) → ne×ne
        std::vector<double> dFp(ne * ne);
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    ne, ne, nao, 1.0, dX, nao, dTmp.data(), ne,
                    0.0, dFp.data(), ne);

        // Diagonalize Fp (ne × ne) using dsyevd
        std::vector<double> dW(ne);
        {
            int lw = -1, liw = -1;
            double wq; lapack_int iwq;
            LAPACKE_dsyevd_work(LAPACK_COL_MAJOR, 'V', 'L', (lapack_int)ne,
                                dFp.data(), (lapack_int)ne, dW.data(),
                                &wq, lw, &iwq, liw);
            lw = (int)wq; liw = iwq;
            std::vector<double> dwork(lw);
            std::vector<lapack_int> diwork(liw);
            LAPACKE_dsyevd_work(LAPACK_COL_MAJOR, 'V', 'L', (lapack_int)ne,
                                dFp.data(), (lapack_int)ne, dW.data(),
                                dwork.data(), (lapack_int)lw,
                                diwork.data(), (lapack_int)liw);
        }
        // dFp now holds eigenvectors in col-major (ne × ne)
        for (int i = 0; i < nao && i < ne; i++)
            scf_ws.d_W[i] = (float)dW[i];

        // C = X * eigvec: (nao×ne) @ (ne×ne) → nao×ne
        // eigvec in col-major → viewed as row-major = transposed
        // C_row = X_row @ eigvec_colmaj^T
        std::vector<double> dC(nao * ne);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    nao, ne, ne, 1.0, dX, nao, dFp.data(), ne,
                    0.0, dC.data(), ne);

        // Copy to d_C (nao × nao, float, row-major with stride nao)
        // Only first ne columns are meaningful
        memset(scf_ws.d_C, 0, sizeof(float) * nao2);
        for (int i = 0; i < nao; i++)
            for (int j = 0; j < ne; j++)
                scf_ws.d_C[i * nao + j] = (float)dC[i * ne + j];
    }

    // P_new = occ_factor * C_occ * C_occ^T
    QC_Build_Density_Blas(blas_handle, nao, scf_ws.n_alpha,
                          scf_ws.occ_factor, scf_ws.d_C, scf_ws.d_P_new);

    if (!scf_ws.unrestricted) return;

    // Beta spin: same pipeline
    {
        const int ne = scf_ws.nao_eff > 0 ? scf_ws.nao_eff : nao;
        const double* dX = scf_ws.d_X;
        std::vector<double> dF(nao2);
        if (scf_ws.d_F_b_double)
            for (int i = 0; i < nao2; i++) dF[i] = scf_ws.d_F_b_double[i];
        else
            for (int i = 0; i < nao2; i++) dF[i] = (double)scf_ws.d_F_b[i];

        std::vector<double> dTmp(nao * ne);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    nao, ne, nao, 1.0, dF.data(), nao, dX, nao,
                    0.0, dTmp.data(), ne);
        std::vector<double> dFp(ne * ne);
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    ne, ne, nao, 1.0, dX, nao, dTmp.data(), ne,
                    0.0, dFp.data(), ne);
        std::vector<double> dW(ne);
        {
            int lw = -1, liw = -1;
            double wq; lapack_int iwq;
            LAPACKE_dsyevd_work(LAPACK_COL_MAJOR, 'V', 'L', (lapack_int)ne,
                                dFp.data(), (lapack_int)ne, dW.data(),
                                &wq, lw, &iwq, liw);
            lw = (int)wq; liw = iwq;
            std::vector<double> dwork(lw);
            std::vector<lapack_int> diwork(liw);
            LAPACKE_dsyevd_work(LAPACK_COL_MAJOR, 'V', 'L', (lapack_int)ne,
                                dFp.data(), (lapack_int)ne, dW.data(),
                                dwork.data(), (lapack_int)lw,
                                diwork.data(), (lapack_int)liw);
        }
        for (int i = 0; i < nao && i < ne; i++)
            scf_ws.d_W[i] = (float)dW[i];
        std::vector<double> dC(nao * ne);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    nao, ne, ne, 1.0, dX, nao, dFp.data(), ne,
                    0.0, dC.data(), ne);
        memset(scf_ws.d_C_b, 0, sizeof(float) * nao2);
        for (int i = 0; i < nao; i++)
            for (int j = 0; j < ne; j++)
                scf_ws.d_C_b[i * nao + j] = (float)dC[i * ne + j];
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

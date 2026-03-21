#pragma once

void QUANTUM_CHEMISTRY::Diagonalize_And_Build_Density()
{
    const int nao = mol.nao;
    const int nao2 = mol.nao2;

    // Unified CPU/GPU path: copy device data to host, compute on host,
    // copy results back. The diag step is O(nao^3) on small matrices
    // and is not the performance bottleneck (build_fock is).
    {
        const int ne = scf_ws.nao_eff > 0 ? scf_ws.nao_eff : nao;

        // Copy X from device to host
        std::vector<double> h_X(nao2);
        deviceMemcpy(h_X.data(), scf_ws.d_X, sizeof(double) * nao2,
                     deviceMemcpyDeviceToHost);

        // Get double Fock: prefer d_F_double, fall back to promoting d_F
        std::vector<double> dF(nao2);
        if (scf_ws.d_F_double)
        {
            deviceMemcpy(dF.data(), scf_ws.d_F_double, sizeof(double) * nao2,
                         deviceMemcpyDeviceToHost);
        }
        else
        {
            std::vector<float> h_F(nao2);
            deviceMemcpy(h_F.data(), scf_ws.d_F, sizeof(float) * nao2,
                         deviceMemcpyDeviceToHost);
            for (int i = 0; i < nao2; i++) dF[i] = (double)h_F[i];
        }

        // Level shift: F += shift * (S - 0.5 * SPS)
        const double ls = scf_ws.level_shift;
        if (ls > 0.0)
        {
            std::vector<float> h_S_f(nao2), h_P_f(nao2);
            deviceMemcpy(h_S_f.data(), scf_ws.d_S, sizeof(float) * nao2,
                         deviceMemcpyDeviceToHost);
            deviceMemcpy(h_P_f.data(), scf_ws.d_P, sizeof(float) * nao2,
                         deviceMemcpyDeviceToHost);
            std::vector<double> dS(nao2), dP(nao2), dSP(nao2), dSPS(nao2);
            for (int i = 0; i < nao2; i++)
            {
                dS[i] = (double)h_S_f[i];
                dP[i] = (double)h_P_f[i];
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

        // Tmp = F * X: (nao x nao) @ (nao x ne) -> nao x ne
        std::vector<double> dTmp(nao * ne);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    nao, ne, nao, 1.0, dF.data(), nao, h_X.data(), nao,
                    0.0, dTmp.data(), ne);
        // Fp = X^T * Tmp: (ne x nao) @ (nao x ne) -> ne x ne
        std::vector<double> dFp(ne * ne);
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    ne, ne, nao, 1.0, h_X.data(), nao, dTmp.data(), ne,
                    0.0, dFp.data(), ne);

        // Diagonalize Fp (ne x ne) using dsyevd on host
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
        // dFp now holds eigenvectors in col-major (ne x ne)

        // Store eigenvalues to device
        std::vector<float> h_W(nao, 0.0f);
        for (int i = 0; i < ne; i++) h_W[i] = (float)dW[i];
        deviceMemcpy(scf_ws.d_W, h_W.data(), sizeof(float) * nao,
                     deviceMemcpyHostToDevice);

        // C = X * eigvec: (nao x ne) @ (ne x ne) -> nao x ne
        // eigvec in col-major -> viewed as row-major = transposed
        std::vector<double> dC(nao * ne);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    nao, ne, ne, 1.0, h_X.data(), nao, dFp.data(), ne,
                    0.0, dC.data(), ne);

        // Copy to d_C (nao x nao, float, row-major with stride nao)
        std::vector<float> h_C(nao2, 0.0f);
        for (int i = 0; i < nao; i++)
            for (int j = 0; j < ne; j++)
                h_C[i * nao + j] = (float)dC[i * ne + j];
        deviceMemcpy(scf_ws.d_C, h_C.data(), sizeof(float) * nao2,
                     deviceMemcpyHostToDevice);
    }

    // P_new = occ_factor * C_occ * C_occ^T (on device, uses device BLAS)
    QC_Build_Density_Blas(blas_handle, nao, scf_ws.n_alpha,
                          scf_ws.occ_factor, scf_ws.d_C, scf_ws.d_P_new);

    if (!scf_ws.unrestricted) return;

    // Beta spin: same pipeline
    {
        const int ne = scf_ws.nao_eff > 0 ? scf_ws.nao_eff : nao;

        // Copy X from device to host
        std::vector<double> h_X(nao2);
        deviceMemcpy(h_X.data(), scf_ws.d_X, sizeof(double) * nao2,
                     deviceMemcpyDeviceToHost);

        // Get double Fock for beta
        std::vector<double> dF(nao2);
        if (scf_ws.d_F_b_double)
        {
            deviceMemcpy(dF.data(), scf_ws.d_F_b_double, sizeof(double) * nao2,
                         deviceMemcpyDeviceToHost);
        }
        else
        {
            std::vector<float> h_Fb(nao2);
            deviceMemcpy(h_Fb.data(), scf_ws.d_F_b, sizeof(float) * nao2,
                         deviceMemcpyDeviceToHost);
            for (int i = 0; i < nao2; i++) dF[i] = (double)h_Fb[i];
        }

        std::vector<double> dTmp(nao * ne);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    nao, ne, nao, 1.0, dF.data(), nao, h_X.data(), nao,
                    0.0, dTmp.data(), ne);
        std::vector<double> dFp(ne * ne);
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    ne, ne, nao, 1.0, h_X.data(), nao, dTmp.data(), ne,
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

        std::vector<float> h_W(nao, 0.0f);
        for (int i = 0; i < ne; i++) h_W[i] = (float)dW[i];
        deviceMemcpy(scf_ws.d_W, h_W.data(), sizeof(float) * nao,
                     deviceMemcpyHostToDevice);

        std::vector<double> dC(nao * ne);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    nao, ne, ne, 1.0, h_X.data(), nao, dFp.data(), ne,
                    0.0, dC.data(), ne);

        std::vector<float> h_Cb(nao2, 0.0f);
        for (int i = 0; i < nao; i++)
            for (int j = 0; j < ne; j++)
                h_Cb[i * nao + j] = (float)dC[i * ne + j];
        deviceMemcpy(scf_ws.d_C_b, h_Cb.data(), sizeof(float) * nao2,
                     deviceMemcpyHostToDevice);
    }
    QC_Build_Density_Blas(blas_handle, nao, scf_ws.n_beta, 1.0f,
                          scf_ws.d_C_b, scf_ws.d_P_b_new);
}

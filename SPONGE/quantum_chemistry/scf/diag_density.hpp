#pragma once

void QUANTUM_CHEMISTRY::Diagonalize_And_Build_Density()
{
    const int nao = mol.nao;
    const int nao2 = mol.nao2;
    const int ne = scf_ws.nao_eff > 0 ? scf_ws.nao_eff : nao;

    // --- Alpha spin ---
    // dF = double Fock (from d_F_double or promoted d_F)
    double* dF = scf_ws.d_dwork_nao2_1;
    if (scf_ws.d_F_double)
        deviceMemcpy(dF, scf_ws.d_F_double, sizeof(double) * nao2,
                     deviceMemcpyDeviceToDevice);
    else
        QC_Float_To_Double(nao2, scf_ws.d_F, dF);

    // Level shift: dF += ls * (S - 0.5 * SPS)
    const double ls = scf_ws.level_shift;
    if (ls > 0.0)
    {
        double* dS = scf_ws.d_dwork_nao2_2;
        double* dP = scf_ws.d_dwork_nao2_3;
        double* dSP = scf_ws.d_dwork_nao2_4;  // reuse as temp
        QC_Float_To_Double(nao2, scf_ws.d_S, dS);
        QC_Float_To_Double(nao2, scf_ws.d_P, dP);
        // SP = S * P
        QC_Dgemm_NN(blas_handle, nao, nao, nao, dS, nao, dP, nao, dSP, nao);
        // SPS = SP * S (reuse dP as output)
        QC_Dgemm_NN(blas_handle, nao, nao, nao, dSP, nao, dS, nao, dP, nao);
        // dF += ls * (dS - 0.5 * dP)  where dP now holds SPS
        QC_Level_Shift(nao2, ls, dS, dP, dF);
    }

    // Tmp = F * X: (nao x nao) @ (nao x ne, stride nao) -> nao x ne, stride ne
    double* dTmp = scf_ws.d_dwork_nao2_2;  // nao*ne <= nao2
    QC_Dgemm_NN(blas_handle, nao, ne, nao, dF, nao, scf_ws.d_X, nao, dTmp, ne);

    // Fp = X^T * Tmp: (ne x nao, X stride=nao) @ (nao x ne, Tmp stride=ne) -> ne x ne
    double* dFp = scf_ws.d_dwork_nao2_3;  // ne*ne <= nao2
    QC_Dgemm_TN(blas_handle, ne, ne, nao, scf_ws.d_X, nao, dTmp, ne, dFp, ne);

    // Diag Fp (ne x ne)
    double* dW = scf_ws.d_dW_double;
    int info = 0;
    QC_Diagonalize_Double(solver_handle, ne, dFp, dW,
                          scf_ws.d_solver_work_double, scf_ws.lwork_double,
                          &info);

    // Store eigenvalues
    QC_Double_To_Float(ne, dW, scf_ws.d_W);

    // C = X * eigvec: (nao x ne, stride nao) @ (ne x ne col-major)
    // eigvec is in dFp, col-major. Viewed as row-major it's transposed.
    double* dC = dTmp;  // reuse, nao*ne
    QC_Dgemm_NT(blas_handle, nao, ne, ne, scf_ws.d_X, nao, dFp, ne, dC, ne);

    // Copy to float d_C (nao x nao, padded)
    QC_Rect_Double_To_Padded_Float(nao, ne, dC, scf_ws.d_C);

    // P_new = occ * C_occ * C_occ^T
    QC_Build_Density_Blas(blas_handle, nao, scf_ws.n_alpha,
                          scf_ws.occ_factor, scf_ws.d_C, scf_ws.d_P_new);

    if (!scf_ws.unrestricted) return;

    // --- Beta spin (same pipeline) ---
    if (scf_ws.d_F_b_double)
        deviceMemcpy(dF, scf_ws.d_F_b_double, sizeof(double) * nao2,
                     deviceMemcpyDeviceToDevice);
    else
        QC_Float_To_Double(nao2, scf_ws.d_F_b, dF);

    QC_Dgemm_NN(blas_handle, nao, ne, nao, dF, nao, scf_ws.d_X, nao, dTmp, ne);
    QC_Dgemm_TN(blas_handle, ne, ne, nao, scf_ws.d_X, nao, dTmp, ne, dFp, ne);
    QC_Diagonalize_Double(solver_handle, ne, dFp, dW,
                          scf_ws.d_solver_work_double, scf_ws.lwork_double,
                          &info);
    QC_Double_To_Float(ne, dW, scf_ws.d_W);
    QC_Dgemm_NT(blas_handle, nao, ne, ne, scf_ws.d_X, nao, dFp, ne, dC, ne);
    QC_Rect_Double_To_Padded_Float(nao, ne, dC, scf_ws.d_C_b);
    QC_Build_Density_Blas(blas_handle, nao, scf_ws.n_beta, 1.0f,
                          scf_ws.d_C_b, scf_ws.d_P_b_new);
}

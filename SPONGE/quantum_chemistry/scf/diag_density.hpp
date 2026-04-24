#pragma once

void QUANTUM_CHEMISTRY::Diagonalize_And_Build_Density()
{
    const int nao = mol.nao;
    const int nao2 = mol.nao2;
    const int ne = scf_ws.ortho.nao_eff > 0 ? scf_ws.ortho.nao_eff : nao;

    // alpha 通道
    // 优先使用双精度 Fock；若当前只有浮点 Fock，则先提升到双精度
    double* dF = scf_ws.ortho.d_dwork_nao2_1;
    if (scf_ws.alpha.d_F_double)
        deviceMemcpy(dF, scf_ws.alpha.d_F_double, sizeof(double) * nao2,
                     deviceMemcpyDeviceToDevice);
    else
        QC_Float_To_Double(nao2, scf_ws.alpha.d_F, dF);

    // 执行 level shift: F <- F + ls * (S - 0.5 * S P S)
    const double ls = scf_ws.runtime.level_shift;
    if (ls > 0.0)
    {
        double* dS = scf_ws.ortho.d_dwork_nao2_2;
        double* dP = scf_ws.ortho.d_dwork_nao2_3;
        double* dSP = scf_ws.ortho.d_dwork_nao2_4;  // 复用为临时缓冲
        QC_Float_To_Double(nao2, scf_ws.core.d_S, dS);
        QC_Float_To_Double(nao2, scf_ws.alpha.d_P, dP);
        // SP = S * P
        QC_Dgemm_NN(blas_handle, nao, nao, nao, dS, nao, dP, nao, dSP, nao);
        // SPS = SP * S，复用 dP 作为输出
        QC_Dgemm_NN(blas_handle, nao, nao, nao, dSP, nao, dS, nao, dP, nao);
        // 此时 dP 中保存的是 SPS
        QC_Level_Shift(nao2, ls, dS, dP, dF);
    }

    // Tmp = F * X
    double* dTmp = scf_ws.ortho.d_dwork_nao2_2;  // nao*ne <= nao2
    QC_Dgemm_NN(blas_handle, nao, ne, nao, dF, nao, scf_ws.ortho.d_X, nao, dTmp,
                ne);

    // Fp = X^T * F * X
    double* dFp = scf_ws.ortho.d_dwork_nao2_3;  // ne*ne <= nao2
    QC_Dgemm_TN(blas_handle, ne, ne, nao, scf_ws.ortho.d_X, nao, dTmp, ne, dFp,
                ne);

    // 对正交化表象下的 Fp 做本征分解
    double* dW = scf_ws.ortho.d_dW_double;
    int info = 0;
    QC_Diagonalize_Double(solver_handle, ne, dFp, dW,
                          scf_ws.ortho.d_solver_work_double,
                          scf_ws.ortho.lwork_double, &info);

    // 保存分子轨道本征值
    QC_Double_To_Float(ne, dW, scf_ws.ortho.d_W);

    // C = X * eigvec，dFp 中保存的是列主序本征向量
    double* dC = dTmp;  // reuse, nao*ne
    QC_Dgemm_NT(blas_handle, nao, ne, ne, scf_ws.ortho.d_X, nao, dFp, ne, dC,
                ne);

    // 将紧致双精度轨道系数写回到补零后的浮点缓冲
    QC_Rect_Double_To_Padded_Float(nao, ne, dC, scf_ws.alpha.d_C);

    // 用占据轨道构造新的密度矩阵
    QC_Build_Density_Blas(blas_handle, nao, scf_ws.runtime.n_alpha,
                          scf_ws.runtime.occ_factor, scf_ws.alpha.d_C,
                          scf_ws.alpha.d_P_new);

    if (!scf_ws.runtime.unrestricted) return;

    // UHF: 保存 alpha 特征值（beta 对角化会覆盖 d_W）
    if (scf_ws.ortho.d_W_alpha)
        deviceMemcpy(scf_ws.ortho.d_W_alpha, scf_ws.ortho.d_W,
                     sizeof(float) * ne, deviceMemcpyDeviceToDevice);

    // beta 通道
    // 非限制体系下复用同一流程处理 beta Fock / 密度
    if (scf_ws.beta.d_F_double)
        deviceMemcpy(dF, scf_ws.beta.d_F_double, sizeof(double) * nao2,
                     deviceMemcpyDeviceToDevice);
    else
        QC_Float_To_Double(nao2, scf_ws.beta.d_F, dF);

    QC_Dgemm_NN(blas_handle, nao, ne, nao, dF, nao, scf_ws.ortho.d_X, nao, dTmp,
                ne);
    QC_Dgemm_TN(blas_handle, ne, ne, nao, scf_ws.ortho.d_X, nao, dTmp, ne, dFp,
                ne);
    QC_Diagonalize_Double(solver_handle, ne, dFp, dW,
                          scf_ws.ortho.d_solver_work_double,
                          scf_ws.ortho.lwork_double, &info);
    QC_Double_To_Float(ne, dW, scf_ws.ortho.d_W);
    QC_Dgemm_NT(blas_handle, nao, ne, ne, scf_ws.ortho.d_X, nao, dFp, ne, dC,
                ne);
    QC_Rect_Double_To_Padded_Float(nao, ne, dC, scf_ws.beta.d_C);
    QC_Build_Density_Blas(blas_handle, nao, scf_ws.runtime.n_beta, 1.0f,
                          scf_ws.beta.d_C, scf_ws.beta.d_P_new);
}

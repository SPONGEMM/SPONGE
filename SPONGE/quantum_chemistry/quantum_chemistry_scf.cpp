// clang-format off
#include "integrals/one_e.hpp"
#include "integrals/eri/common/eri_rys.hpp"
#include "quantum_chemistry.h"
#include "scf/accumulate_energy.hpp"
#include "scf/apply_diis.hpp"
#include "scf/build_fock.hpp"
#include "scf/diag_density.hpp"
#include "scf/mix_converge.hpp"
#include "scf/pre_scf.hpp"
#include "scf/workspace.hpp"
#include "structure/matrix.h"
// clang-format on

void QUANTUM_CHEMISTRY::Solve_SCF(const VECTOR* crd, const VECTOR box_length,
                                  bool need_energy, int md_step)
{
    if (!is_initialized) return;

    Update_Coordinates_From_MD(crd, box_length);
    if (dft.enable_dft) Update_DFT_Grid();

    Reset_SCF_State();

    // 解析计算 norms（不依赖 1e 积分的 S 矩阵）
    Compute_Analytical_Norms();
    Compute_OneE_Integrals();
    Compute_ECP_Matrix();
    if (need_energy) Compute_Nuclear_Repulsion(box_length);
    Prepare_Integrals();
    Build_Shell_Pair_Bounds();
    if (scf_ws.ri.enabled) RI_Precompute();
    Build_Overlap_X();

    if (need_initial_guess)
    {
        Build_Initial_Guess();
        need_initial_guess = false;
    }

    // SCF 收敛策略: HF 和 DFT 使用不同的启动策略
    //
    // HF: DIIS 从 iter 2 开始，level shift 0.25（默认）
    //
    // DFT: 分三个阶段
    //   Phase 1 (iter 0 ~ warmup-1): 纯对角化 + 大 level shift，禁用 DIIS
    //     解决 SAP→DFT Fock 的不连续性
    //   Phase 2 (warmup ~ stable): MESA (EDIIS/ADIIS)，shift 衰减
    //     等待历史点稳定
    //   Phase 3 (stable ~): CDIIS，shift 关闭
    //     超线性收敛
    const int dft_warmup = dft.enable_dft ? 3 : 0;
    const double dft_warmup_ls = 1.5;
    double dft_ls = dft_warmup_ls;
    int stable_count = 0;

    for (int iter = 0; iter < scf_ws.runtime.max_scf_iter; ++iter)
    {
        Build_Fock(iter);
        Accumulate_SCF_Energy(iter);

        // 缓存 DIIS 前的 Fock 供梯度使用（避免梯度中重建 Fock）
        if (need_gradient && scf_ws.alpha.d_F_for_grad)
        {
            deviceMemcpy(scf_ws.alpha.d_F_for_grad, scf_ws.alpha.d_F_double,
                         sizeof(double) * mol.nao2, deviceMemcpyDeviceToDevice);
            if (scf_ws.runtime.unrestricted && scf_ws.beta.d_F_for_grad)
                deviceMemcpy(scf_ws.beta.d_F_for_grad, scf_ws.beta.d_F_double,
                             sizeof(double) * mol.nao2,
                             deviceMemcpyDeviceToDevice);
        }

        if (dft.enable_dft && iter < dft_warmup)
        {
            scf_ws.runtime.level_shift = dft_warmup_ls;
        }
        else
        {
            Apply_DIIS(iter);

            if (dft.enable_dft)
            {
                double h_delta_e = 0.0;
                if (iter > 0)
                    deviceMemcpy(&h_delta_e, scf_ws.runtime.d_delta_e,
                                 sizeof(double), deviceMemcpyDeviceToHost);
                if (h_delta_e < 0.0)
                    stable_count++;
                else
                    stable_count = 0;

                if (stable_count >= 2)
                    dft_ls *= 0.8;
                else
                    dft_ls = fmin(dft_ls * 1.2, dft_warmup_ls);

                scf_ws.runtime.level_shift = fmax(dft_ls, 0.0);
            }
            else
            {
                scf_ws.runtime.level_shift = 0.25;
            }
        }

        Diagonalize_And_Build_Density();
        if (Check_Convergence(iter, md_step)) break;
    }
}

void QUANTUM_CHEMISTRY::Compute_Spin_Square()
{
    const int nao = mol.nao;
    const int nao2 = mol.nao2;

    // <S²> = s(s+1) + N_beta - Tr(P_alpha · S · P_beta · S)
    // 使用 ortho 的 double workspace 作为临时缓冲，避免污染 Fock 矩阵
    double* d_tmp1 = scf_ws.ortho.d_dwork_nao2_1;
    double* d_tmp2 = scf_ws.ortho.d_dwork_nao2_2;
    double* d_tmp3 = scf_ws.ortho.d_dwork_nao2_3;

    // 提升到 double: dPa = P_alpha, dS = S
    QC_Float_To_Double(nao2, scf_ws.alpha.d_P, d_tmp1);
    QC_Float_To_Double(nao2, scf_ws.core.d_S, d_tmp2);

    // d_tmp3 = P_alpha * S
    QC_Dgemm_NN(blas_handle, nao, nao, nao, d_tmp1, nao, d_tmp2, nao, d_tmp3,
                nao);

    // d_tmp1 = P_beta (提升)
    QC_Float_To_Double(nao2, scf_ws.beta.d_P, d_tmp1);

    // d_tmp1 = (P_alpha * S) * P_beta -> 复用: d_tmp4 借用 d_dwork_nao2_4
    double* d_tmp4 = scf_ws.ortho.d_dwork_nao2_4;
    QC_Dgemm_NN(blas_handle, nao, nao, nao, d_tmp3, nao, d_tmp1, nao, d_tmp4,
                nao);

    // Tr(P_alpha * S * P_beta * S) = Σ_ij (P_alpha·S·P_beta)_ij * S_ij
    double trace = 0.0;
    double* d_accum = scf_ws.diis.d_diis_accum;
    if (d_accum == NULL)
    {
        Device_Malloc_Safely((void**)&d_accum, sizeof(double));
    }
    deviceMemset(d_accum, 0, sizeof(double));
    QC_Double_Dot(nao2, d_tmp4, d_tmp2, d_accum);
    deviceMemcpy(&trace, d_accum, sizeof(double), deviceMemcpyDeviceToHost);
    if (scf_ws.diis.d_diis_accum == NULL)
    {
        deviceFree(d_accum);
    }

    double s = 0.5 * (scf_ws.runtime.n_alpha - scf_ws.runtime.n_beta);
    scf_ws.runtime.spin_square_exact = s * (s + 1.0);
    scf_ws.runtime.spin_square =
        scf_ws.runtime.spin_square_exact + scf_ws.runtime.n_beta - trace;
}

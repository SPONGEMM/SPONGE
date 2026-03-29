#include "integrals/one_e.hpp"
#include "quantum_chemistry.h"
#include "scf/accumulate_energy.hpp"
#include "scf/apply_diis.hpp"
#include "scf/build_fock.hpp"
#include "scf/diag_density.hpp"
#include "scf/mix_converge.hpp"
#include "scf/pre_scf.hpp"
#include "scf/workspace.hpp"
#include "structure/matrix.h"

void QUANTUM_CHEMISTRY::Solve_SCF(const VECTOR* crd, const VECTOR box_length,
                                  bool need_energy, int md_step)
{
    if (!is_initialized) return;

    Update_Coordinates_From_MD(crd, box_length);
    if (dft.enable_dft) Update_DFT_Grid();

    Reset_SCF_State();
    Compute_OneE_Integrals();
    if (need_energy) Compute_Nuclear_Repulsion(box_length);
    Prepare_Integrals();
    Build_Overlap_X();

    if (need_initial_guess)
    {
        Build_Initial_Guess();
        need_initial_guess = false;
    }

    for (int iter = 0; iter < scf_ws.runtime.max_scf_iter; ++iter)
    {
        Build_Fock(iter);
        Accumulate_SCF_Energy(iter);
        Apply_DIIS(iter);
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
    deviceMemset(scf_ws.diis.d_diis_accum, 0, sizeof(double));
    QC_Double_Dot(nao2, d_tmp4, d_tmp2, scf_ws.diis.d_diis_accum);
    deviceMemcpy(&trace, scf_ws.diis.d_diis_accum, sizeof(double),
                 deviceMemcpyDeviceToHost);

    double s = 0.5 * (scf_ws.runtime.n_alpha - scf_ws.runtime.n_beta);
    scf_ws.runtime.spin_square_exact = s * (s + 1.0);
    scf_ws.runtime.spin_square =
        scf_ws.runtime.spin_square_exact + scf_ws.runtime.n_beta - trace;
}

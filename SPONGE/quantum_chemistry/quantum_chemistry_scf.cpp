#include "quantum_chemistry.h"

#include "integrals/eri.hpp"
#include "integrals/one_e.hpp"
#include "structure/matrix.h"

#include "scf/accumulate_energy.hpp"
#include "scf/apply_diis.hpp"
#include "scf/build_fock.hpp"
#include "scf/diag_density.hpp"
#include "scf/mix_converge.hpp"
#include "scf/pre_scf.hpp"
#include "scf/workspace.hpp"

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

    for (int iter = 0; iter < scf_ws.max_scf_iter; ++iter)
    {
        Build_Fock();
        Accumulate_SCF_Energy(iter);
        Apply_DIIS(iter);
        Diagonalize_And_Build_Density();
        if (Mix_And_Check_Convergence(iter, md_step)) break;
    }
}

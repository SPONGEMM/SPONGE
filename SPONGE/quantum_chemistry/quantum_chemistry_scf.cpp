#include "integrals/eri.hpp"
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

    const double t_pre_scf_begin = omp_get_wtime();
    Update_Coordinates_From_MD(crd, box_length);
    if (dft.enable_dft) Update_DFT_Grid();

    Reset_SCF_State();
    Compute_OneE_Integrals();
    if (need_energy) Compute_Nuclear_Repulsion(box_length);
    Prepare_Integrals();
    Build_Overlap_X();
    scf_ws.last_pre_scf_s = omp_get_wtime() - t_pre_scf_begin;

    if (scf_ws.bench_fock_only)
    {
        // Bootstrap a non-zero density so the Fock-only benchmark reflects the
        // direct-SCF workload instead of the trivial zero-density first pass.
        Build_Fock(0);
#ifdef USE_GPU
        deviceStreamSynchronize(0);
#endif
        Accumulate_SCF_Energy(0);
#ifdef USE_GPU
        deviceStreamSynchronize(0);
#endif
        Apply_DIIS(0);
#ifdef USE_GPU
        deviceStreamSynchronize(0);
#endif
        Diagonalize_And_Build_Density();
#ifdef USE_GPU
        deviceStreamSynchronize(0);
#endif
        (void)Mix_And_Check_Convergence(0, md_step);
#ifdef USE_GPU
        deviceStreamSynchronize(0);
#endif

        const int bench_repeats = std::max(1, scf_ws.bench_fock_repeats);
        const double t_bench_begin = omp_get_wtime();
        int active_sum = 0;
        for (int rep = 0; rep < bench_repeats; rep++)
        {
            Build_Fock(2);
#ifdef USE_GPU
            deviceStreamSynchronize(0);
#endif
            active_sum += scf_ws.last_active_eri_tasks;
        }
        scf_ws.last_fock_bench_total_s = omp_get_wtime() - t_bench_begin;
        if (CONTROLLER::MPI_rank == 0)
        {
            printf("QC Fock benchmark | repeats=%d | total=%.6fs | avg=%.6fs | "
                   "avg_active_quartets=%d\n",
                   bench_repeats, scf_ws.last_fock_bench_total_s,
                   scf_ws.last_fock_bench_total_s / bench_repeats,
                   active_sum / bench_repeats);
        }
        return;
    }

    for (int iter = 0; iter < scf_ws.max_scf_iter; ++iter)
    {
        double t_stage = omp_get_wtime();
        Build_Fock(iter);
 #ifdef USE_GPU
        deviceStreamSynchronize(0);
 #endif
        scf_ws.last_build_fock_s = omp_get_wtime() - t_stage;
        t_stage = omp_get_wtime();
        Accumulate_SCF_Energy(iter);
 #ifdef USE_GPU
        deviceStreamSynchronize(0);
 #endif
        scf_ws.last_accumulate_energy_s = omp_get_wtime() - t_stage;
        t_stage = omp_get_wtime();
        Apply_DIIS(iter);
 #ifdef USE_GPU
        deviceStreamSynchronize(0);
 #endif
        scf_ws.last_apply_diis_s = omp_get_wtime() - t_stage;
        t_stage = omp_get_wtime();
        Diagonalize_And_Build_Density();
 #ifdef USE_GPU
        deviceStreamSynchronize(0);
 #endif
        scf_ws.last_diag_density_s = omp_get_wtime() - t_stage;
        t_stage = omp_get_wtime();
        if (Mix_And_Check_Convergence(iter, md_step)) break;
        scf_ws.last_mix_converge_s = omp_get_wtime() - t_stage;
    }
}

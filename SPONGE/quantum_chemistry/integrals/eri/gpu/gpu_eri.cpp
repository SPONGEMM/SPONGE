// Shared ERI backend glue for all GPU paths.

// clang-format off
// Include order matters: quantum_chemistry.h provides macros/types needed by
// ERI GPU headers.
#include "../../../quantum_chemistry.h"
#include "../common/eri_kernel_utils.hpp"
#include "../../../../common.h"
#include "launch.hpp"
#include "gpu_eri.hpp"
#include "screen.hpp"
// clang-format on

void QC_Launch_Screen(
    int n_total, const QC_INTEGRAL_TASKS::ScreenCombo* combos,
    const int* combo_prefix, int n_combos, const int* sorted_pair_ids,
    const QC_ONE_E_TASK* shell_pairs, const float* shell_pair_bounds,
    const float* pair_density_coul, const float* pair_density_exx_a,
    const float* pair_density_exx_b, float shell_screen_tol, float exx_scale_a,
    float exx_scale_b, QC_ERI_TASK* output_tasks, int* output_counts)
{
    const int threads = 256;
    Launch_Device_Kernel(
        QC_Screen_All_Combos_Kernel, (n_total + threads - 1) / threads, threads,
        0, 0, n_total, combos, combo_prefix, n_combos, sorted_pair_ids,
        shell_pairs, shell_pair_bounds, pair_density_coul, pair_density_exx_a,
        pair_density_exx_b, shell_screen_tol, exx_scale_a, exx_scale_b,
        output_tasks, output_counts);
}

void QC_Build_Fock_Direct_GPU(
    const QC_INTEGRAL_TASKS& task_ctx, const int* atm, const int* bas,
    const float* env, const int* ao_offsets_cart, const int* ao_offsets_sph,
    const float* norms, const float* shell_pair_bounds,
    const float* pair_density_coul, const float* pair_density_exx_a,
    const float* pair_density_exx_b, const float shell_screen_tol,
    const float* P_coul, const float* P_exx_a, const float* P_exx_b,
    const float exx_scale_a, const float exx_scale_b, const int nao,
    const int nao_sph, const int is_spherical, const float* cart2sph_mat,
    float* F_a, float* F_b, float* global_hr_pool, const float prim_screen_tol)
{
    deviceMemset(task_ctx.d_screen_counts, 0, sizeof(int) * task_ctx.n_combos);

    int* d_combo_prefix = NULL;
    Device_Malloc_And_Copy_Safely((void**)&d_combo_prefix,
                                  (void*)task_ctx.combo_prefix,
                                  sizeof(int) * (task_ctx.n_combos + 1));

    QC_Launch_Screen(task_ctx.total_quartets, task_ctx.d_combos, d_combo_prefix,
                     task_ctx.n_combos, task_ctx.d_sorted_pair_ids,
                     task_ctx.d_shell_pairs, task_ctx.d_shell_pair_bounds,
                     pair_density_coul, pair_density_exx_a, pair_density_exx_b,
                     shell_screen_tol, exx_scale_a, exx_scale_b,
                     task_ctx.d_screened_tasks, task_ctx.d_screen_counts);

    deviceFree(d_combo_prefix);

    int h_counts[QC_INTEGRAL_TASKS::MAX_COMBOS] = {};
    deviceMemcpy(h_counts, task_ctx.d_screen_counts,
                 sizeof(int) * task_ctx.n_combos, deviceMemcpyDeviceToHost);

    using LaunchFunc = void (*)(ERI_KERNEL_PARAMS);
    auto launch_eri = [&](const int combo_index, LaunchFunc func)
    {
        const int n = h_counts[combo_index];
        if (n == 0) return;
        func(n,
             task_ctx.d_screened_tasks +
                 task_ctx.h_combos[combo_index].output_offset,
             atm, bas, env, ao_offsets_cart, ao_offsets_sph, norms,
             shell_pair_bounds, pair_density_coul, pair_density_exx_a,
             pair_density_exx_b, shell_screen_tol, P_coul, P_exx_a, P_exx_b,
             exx_scale_a, exx_scale_b, nao, nao_sph, is_spherical,
             cart2sph_mat, F_a, F_b, global_hr_pool, task_ctx.eri_hr_base,
             task_ctx.eri_hr_size, task_ctx.eri_shell_buf_size,
             prim_screen_tol);
    };

    for (int combo_index = 0; combo_index < task_ctx.n_combos; combo_index++)
    {
        if (h_counts[combo_index] == 0) continue;
        const auto& combo = task_ctx.h_combos[combo_index];
        const int lkey =
            combo.l0 * 1000 + combo.l1 * 100 + combo.l2 * 10 + combo.l3;
        switch (lkey)
        {
            case 0: launch_eri(combo_index, QC_Launch_ssss); break;
            case 1000: launch_eri(combo_index, QC_Launch_psss); break;
            case 100: launch_eri(combo_index, QC_Launch_spss); break;
            case 10: launch_eri(combo_index, QC_Launch_ssps); break;
            case 1: launch_eri(combo_index, QC_Launch_sssp); break;
            case 1100: launch_eri(combo_index, QC_Launch_ppss); break;
            case 1010: launch_eri(combo_index, QC_Launch_psps); break;
            case 1001: launch_eri(combo_index, QC_Launch_pssp); break;
            case 110: launch_eri(combo_index, QC_Launch_spps); break;
            case 101: launch_eri(combo_index, QC_Launch_spsp); break;
            case 11: launch_eri(combo_index, QC_Launch_sspp); break;
            case 111: launch_eri(combo_index, QC_Launch_sppp); break;
            case 1011: launch_eri(combo_index, QC_Launch_pspp); break;
            case 1101: launch_eri(combo_index, QC_Launch_ppsp); break;
            case 1110: launch_eri(combo_index, QC_Launch_ppps); break;
            case 1111: launch_eri(combo_index, QC_Launch_pppp); break;
            default:
            {
                const int l_max =
                    std::max({combo.l0, combo.l1, combo.l2, combo.l3});
                const int l_sum = combo.l0 + combo.l1 + combo.l2 + combo.l3;
                if (l_max <= 2)
                {
                    switch (l_sum)
                    {
                        case 2: launch_eri(combo_index, QC_Launch_D_L2); break;
                        case 3: launch_eri(combo_index, QC_Launch_D_L3); break;
                        case 4: launch_eri(combo_index, QC_Launch_D_L4); break;
                        case 5: launch_eri(combo_index, QC_Launch_D_L5); break;
                        case 6: launch_eri(combo_index, QC_Launch_D_L6); break;
                        case 7: launch_eri(combo_index, QC_Launch_D_L7); break;
                        case 8: launch_eri(combo_index, QC_Launch_D_L8); break;
                    }
                }
                else
                {
                    switch (l_sum)
                    {
                        case 3:
                            launch_eri(combo_index, QC_Launch_Rys_L3);
                            break;
                        case 4:
                            launch_eri(combo_index, QC_Launch_Rys_L4);
                            break;
                        case 5:
                            launch_eri(combo_index, QC_Launch_Rys_L5);
                            break;
                        case 6:
                            launch_eri(combo_index, QC_Launch_Rys_L6);
                            break;
                        case 7:
                            launch_eri(combo_index, QC_Launch_Rys_L7);
                            break;
                        case 8:
                            launch_eri(combo_index, QC_Launch_Rys_L8);
                            break;
                        case 9:
                            launch_eri(combo_index, QC_Launch_Rys_L9);
                            break;
                        case 10:
                            launch_eri(combo_index, QC_Launch_Rys_L10);
                            break;
                        case 11:
                            launch_eri(combo_index, QC_Launch_Rys_L11);
                            break;
                        case 12:
                            launch_eri(combo_index, QC_Launch_Rys_L12);
                            break;
                        case 13:
                            launch_eri(combo_index, QC_Launch_Rys_L13);
                            break;
                        case 14:
                            launch_eri(combo_index, QC_Launch_Rys_L14);
                            break;
                        case 15:
                            launch_eri(combo_index, QC_Launch_Rys_L15);
                            break;
                        case 16:
                            launch_eri(combo_index, QC_Launch_Rys_L16);
                            break;
                    }
                }
                break;
            }
        }
    }
}

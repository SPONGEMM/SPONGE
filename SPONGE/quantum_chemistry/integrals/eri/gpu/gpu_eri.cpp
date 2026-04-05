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

// Reduce N Fock copies into F: F[i] += sum of copies[c][i]
static __global__ void QC_Reduce_Fock_Copies_Kernel(
    const int nao2, const int n_copies, const float* __restrict__ copies,
    float* __restrict__ F)
{
    SIMPLE_DEVICE_FOR(i, nao2)
    {
        float sum = 0.0f;
        for (int c = 0; c < n_copies; c++) sum += copies[(size_t)c * nao2 + i];
        F[i] += sum;
    }
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
    const int nao2 = nao * nao;
    deviceMemset(task_ctx.buffers.d_screen_counts, 0,
                 sizeof(int) * task_ctx.topo.n_combos);

    // 持久化 combo_prefix 缓冲
    static int* s_d_combo_prefix = NULL;
    static int s_combo_prefix_size = 0;
    const int needed = task_ctx.topo.n_combos + 1;
    if (!s_d_combo_prefix || s_combo_prefix_size < needed)
    {
        if (s_d_combo_prefix) deviceFree(s_d_combo_prefix);
        Device_Malloc_Safely((void**)&s_d_combo_prefix, sizeof(int) * needed);
        s_combo_prefix_size = needed;
    }
    deviceMemcpy(s_d_combo_prefix, (void*)task_ctx.topo.combo_prefix,
                 sizeof(int) * needed, deviceMemcpyHostToDevice);

    QC_Launch_Screen(
        task_ctx.topo.total_quartets, task_ctx.buffers.d_combos,
        s_d_combo_prefix, task_ctx.topo.n_combos,
        task_ctx.buffers.d_sorted_pair_ids, task_ctx.buffers.d_shell_pairs,
        task_ctx.buffers.d_shell_pair_bounds, pair_density_coul,
        pair_density_exx_a, pair_density_exx_b, shell_screen_tol, exx_scale_a,
        exx_scale_b, task_ctx.buffers.d_screened_tasks,
        task_ctx.buffers.d_screen_counts);

    // 持久化 multi-copy Fock 缓冲，减少 atomicAdd 竞争
    const int N_FOCK_COPIES = 64;
    static float* s_d_F_copies = NULL;
    static float* s_d_F_b_copies = NULL;
    static int s_fock_copies_nao2 = 0;
    const int copies_needed = N_FOCK_COPIES * nao2;
    if (!s_d_F_copies || s_fock_copies_nao2 < nao2)
    {
        if (s_d_F_copies) deviceFree(s_d_F_copies);
        if (s_d_F_b_copies) deviceFree(s_d_F_b_copies);
        Device_Malloc_Safely((void**)&s_d_F_copies,
                             sizeof(float) * copies_needed);
        Device_Malloc_Safely((void**)&s_d_F_b_copies,
                             sizeof(float) * copies_needed);
        s_fock_copies_nao2 = nao2;
    }
    deviceMemset(s_d_F_copies, 0, sizeof(float) * copies_needed);
    float* d_F_b_mc = (F_b != NULL) ? s_d_F_b_copies : (float*)NULL;
    if (d_F_b_mc) deviceMemset(d_F_b_mc, 0, sizeof(float) * copies_needed);

    int h_counts[QC_INTEGRAL_TASKS::MAX_COMBOS] = {};
    deviceMemcpy(h_counts, task_ctx.buffers.d_screen_counts,
                 sizeof(int) * task_ctx.topo.n_combos,
                 deviceMemcpyDeviceToHost);

#ifdef GPU_ARCH_NAME
    // Per-L_sum streams for concurrent kernel execution.
    // Heavy kernels (L8-L12) have few blocks and would leave SMs idle
    // when launched sequentially; streams allow them to overlap.
    static deviceStream_t s_streams[17] = {};  // 0=SP, 2..16=Rys L_sum
    static bool s_streams_init = false;
    if (!s_streams_init)
    {
        for (int i = 0; i < 17; i++) deviceStreamCreate(&s_streams[i]);
        s_streams_init = true;
    }
#endif

    using LaunchFunc = void (*)(ERI_KERNEL_PARAMS);
    auto launch_eri =
        [&](const int combo_index, LaunchFunc func, int stream_idx)
    {
        const int n = h_counts[combo_index];
        if (n == 0) return;
#ifdef GPU_ARCH_NAME
        g_eri_stream = s_streams[stream_idx];
#endif
        func(n,
             task_ctx.buffers.d_screened_tasks +
                 task_ctx.topo.h_combos[combo_index].output_offset,
             atm, bas, env, ao_offsets_cart, ao_offsets_sph, norms,
             shell_pair_bounds, pair_density_coul, pair_density_exx_a,
             pair_density_exx_b, shell_screen_tol, P_coul, P_exx_a, P_exx_b,
             exx_scale_a, exx_scale_b, nao, nao_sph, is_spherical, cart2sph_mat,
             s_d_F_copies, d_F_b_mc, global_hr_pool,
             task_ctx.params.eri_hr_base, task_ctx.params.eri_hr_size,
             task_ctx.params.eri_shell_buf_size, prim_screen_tol,
             N_FOCK_COPIES);
    };

    for (int combo_index = 0; combo_index < task_ctx.topo.n_combos;
         combo_index++)
    {
        if (h_counts[combo_index] == 0) continue;
        const auto& combo = task_ctx.topo.h_combos[combo_index];
        const int lkey =
            combo.l0 * 1000 + combo.l1 * 100 + combo.l2 * 10 + combo.l3;
        switch (lkey)
        {
            case 0:
                launch_eri(combo_index, QC_Launch_ssss, 0);
                break;
            case 1000:
                launch_eri(combo_index, QC_Launch_psss, 0);
                break;
            case 100:
                launch_eri(combo_index, QC_Launch_spss, 0);
                break;
            case 10:
                launch_eri(combo_index, QC_Launch_ssps, 0);
                break;
            case 1:
                launch_eri(combo_index, QC_Launch_sssp, 0);
                break;
            case 1100:
                launch_eri(combo_index, QC_Launch_ppss, 0);
                break;
            case 1010:
                launch_eri(combo_index, QC_Launch_psps, 0);
                break;
            case 1001:
                launch_eri(combo_index, QC_Launch_pssp, 0);
                break;
            case 110:
                launch_eri(combo_index, QC_Launch_spps, 0);
                break;
            case 101:
                launch_eri(combo_index, QC_Launch_spsp, 0);
                break;
            case 11:
                launch_eri(combo_index, QC_Launch_sspp, 0);
                break;
            case 111:
                launch_eri(combo_index, QC_Launch_sppp, 0);
                break;
            case 1011:
                launch_eri(combo_index, QC_Launch_pspp, 0);
                break;
            case 1101:
                launch_eri(combo_index, QC_Launch_ppsp, 0);
                break;
            case 1110:
                launch_eri(combo_index, QC_Launch_ppps, 0);
                break;
            case 1111:
                launch_eri(combo_index, QC_Launch_pppp, 0);
                break;
            default:
            {
                const int l_sum = combo.l0 + combo.l1 + combo.l2 + combo.l3;
                {
                    switch (l_sum)
                    {
                        case 2:
                            launch_eri(combo_index, QC_Launch_Rys_L2, 2);
                            break;
                        case 3:
                            launch_eri(combo_index, QC_Launch_Rys_L3, 3);
                            break;
                        case 4:
                            launch_eri(combo_index, QC_Launch_Rys_L4, 4);
                            break;
                        case 5:
                            launch_eri(combo_index, QC_Launch_Rys_L5, 5);
                            break;
                        case 6:
                            launch_eri(combo_index, QC_Launch_Rys_L6, 6);
                            break;
                        case 7:
                            launch_eri(combo_index, QC_Launch_Rys_L7, 7);
                            break;
                        case 8:
                            launch_eri(combo_index, QC_Launch_Rys_L8, 8);
                            break;
                        case 9:
                            launch_eri(combo_index, QC_Launch_Rys_L9, 9);
                            break;
                        case 10:
                            launch_eri(combo_index, QC_Launch_Rys_L10, 10);
                            break;
                        case 11:
                            launch_eri(combo_index, QC_Launch_Rys_L11, 11);
                            break;
                        case 12:
                            launch_eri(combo_index, QC_Launch_Rys_L12, 12);
                            break;
                        case 13:
                            launch_eri(combo_index, QC_Launch_Rys_L13, 13);
                            break;
                        case 14:
                            launch_eri(combo_index, QC_Launch_Rys_L14, 14);
                            break;
                        case 15:
                            launch_eri(combo_index, QC_Launch_Rys_L15, 15);
                            break;
                        case 16:
                            launch_eri(combo_index, QC_Launch_Rys_L16, 16);
                            break;
                    }
                }
                break;
            }
        }
    }
#ifdef GPU_ARCH_NAME
    // Synchronize all ERI streams before reduce
    for (int i = 0; i < 17; i++) deviceStreamSynchronize(s_streams[i]);
    g_eri_stream = 0;  // reset to default
#endif
    // Reduce multi-copy Fock buffers back to F_a / F_b
    const int threads = 256;
    Launch_Device_Kernel(QC_Reduce_Fock_Copies_Kernel,
                         (nao2 + threads - 1) / threads, threads, 0, 0, nao2,
                         N_FOCK_COPIES, s_d_F_copies, F_a);
    if (F_b != NULL)
        Launch_Device_Kernel(QC_Reduce_Fock_Copies_Kernel,
                             (nao2 + threads - 1) / threads, threads, 0, 0,
                             nao2, N_FOCK_COPIES, d_F_b_mc, F_b);
}

#pragma once

// Single-launch GPU/CPU screening kernel with LaneGroup compaction.
// Covers ALL pair-type combinations in one kernel launch.
// Each thread finds its combo via linear search in prefix sums,
// generates the quartet on-the-fly, screens, and compacts to per-combo output.

__global__ void QC_Screen_All_Combos_Kernel(
    const int n_total,
    const QC_INTEGRAL_TASKS::ScreenCombo* __restrict__ combos,
    const int* __restrict__ combo_prefix, const int n_combos,
    const int* __restrict__ sorted_pair_ids,
    const QC_ONE_E_TASK* __restrict__ shell_pairs,
    const float* __restrict__ shell_pair_bounds,
    const float* __restrict__ pair_density_coul,
    const float* __restrict__ pair_density_exx_a,
    const float* __restrict__ pair_density_exx_b, const float shell_screen_tol,
    const float exx_scale_a, const float exx_scale_b,
    QC_ERI_TASK* __restrict__ output_tasks, int* __restrict__ output_counts)
{
    SIMPLE_DEVICE_FOR(global_idx, n_total)
    {
        // --- Find which combo this thread belongs to (linear search, n_combos
        // <= ~10) ---
        int ci = 0;
        while (ci < n_combos - 1 && global_idx >= combo_prefix[ci + 1]) ci++;
        const auto& combo = combos[ci];
        const int local_idx = global_idx - combo_prefix[ci];

        // --- On-the-fly: local_idx → (pair_ij, pair_kl) ---
        int local_ij, local_kl;
        if (combo.same_type)
        {
            local_ij =
                (int)floor((sqrt(8.0 * (double)local_idx + 1.0) - 1.0) * 0.5);
            local_kl = local_idx - local_ij * (local_ij + 1) / 2;
            if (local_ij * (local_ij + 1) / 2 + local_kl != local_idx)
            {
                local_ij++;
                local_kl = local_idx - local_ij * (local_ij + 1) / 2;
            }
        }
        else
        {
            local_ij = local_idx / combo.n_B;
            local_kl = local_idx % combo.n_B;
        }
        const int pair_ij = sorted_pair_ids[combo.pair_base_A + local_ij];
        const int pair_kl = sorted_pair_ids[combo.pair_base_B + local_kl];

        const QC_ONE_E_TASK pij = shell_pairs[pair_ij];
        const QC_ONE_E_TASK pkl = shell_pairs[pair_kl];

        // --- Screening ---
        const int ij = QC_Shell_Pair_Index(pij.x, pij.y);
        const int kl = QC_Shell_Pair_Index(pkl.x, pkl.y);
        const int ik = QC_Shell_Pair_Index(pij.x, pkl.x);
        const int il = QC_Shell_Pair_Index(pij.x, pkl.y);
        const int jk = QC_Shell_Pair_Index(pij.y, pkl.x);
        const int jl = QC_Shell_Pair_Index(pij.y, pkl.y);

        const float sb = shell_pair_bounds[ij] * shell_pair_bounds[kl];
        float screen = sb * fmaxf(pair_density_coul[ij], pair_density_coul[kl]);
        if (exx_scale_a != 0.0f)
            screen = fmaxf(
                screen,
                sb * exx_scale_a *
                    QC_Max4(pair_density_exx_a[ik], pair_density_exx_a[il],
                            pair_density_exx_a[jk], pair_density_exx_a[jl]));
        if (pair_density_exx_b != NULL && exx_scale_b != 0.0f)
            screen = fmaxf(
                screen,
                sb * exx_scale_b *
                    QC_Max4(pair_density_exx_b[ik], pair_density_exx_b[il],
                            pair_density_exx_b[jk], pair_density_exx_b[jl]));

        const bool pass = (screen >= shell_screen_tol);

        // --- Per-thread atomicAdd compaction to per-combo output region ---
        // (Can't use warp-level LaneGroup here because threads in the same warp
        //  may belong to different combos with different output regions.)
        if (pass)
        {
            const int slot = atomicAdd(&output_counts[ci], 1);
            output_tasks[combo.output_offset + slot] = {pij.x, pij.y, pkl.x,
                                                        pkl.y};
        }
    }
}

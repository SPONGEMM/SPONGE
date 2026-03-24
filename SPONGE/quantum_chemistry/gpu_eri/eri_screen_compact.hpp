// GPU/CPU screening kernel with LaneGroup-based compaction.
// Generates quartets on-the-fly from pair-type blocks, screens them,
// and writes surviving quartets to a compact output buffer.
//
// Expected macros before include:
//   SCREEN_KERNEL_NAME
//   SAME_PAIR_TYPE  - 0 or 1

static __global__ void SCREEN_KERNEL_NAME(
    const int n_quartets,
    const int pair_id_base_A, const int n_A,
    const int pair_id_base_B, const int n_B,
    const int* __restrict__ sorted_pair_ids,
    const QC_ONE_E_TASK* __restrict__ shell_pairs,
    const float* __restrict__ shell_pair_bounds,
    const float* __restrict__ pair_density_coul,
    const float* __restrict__ pair_density_exx_a,
    const float* __restrict__ pair_density_exx_b,
    const float shell_screen_tol,
    const float exx_scale_a, const float exx_scale_b,
    QC_ERI_TASK* __restrict__ output_tasks,
    int* __restrict__ output_count)
{
    SIMPLE_DEVICE_FOR(idx, n_quartets)
    {
        // --- On-the-fly: flat_idx → (pair_ij, pair_kl) ---
        int local_ij, local_kl;
#if SAME_PAIR_TYPE
        local_ij = (int)floor((sqrt(8.0 * (double)idx + 1.0) - 1.0) * 0.5);
        local_kl = idx - local_ij * (local_ij + 1) / 2;
        if (local_ij * (local_ij + 1) / 2 + local_kl != idx)
        {
            local_ij++;
            local_kl = idx - local_ij * (local_ij + 1) / 2;
        }
#else
        local_ij = idx / n_B;
        local_kl = idx % n_B;
#endif
        const int pair_ij = sorted_pair_ids[pair_id_base_A + local_ij];
        const int pair_kl = sorted_pair_ids[pair_id_base_B + local_kl];

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
            screen = fmaxf(screen,
                sb * exx_scale_a *
                QC_Max4(pair_density_exx_a[ik], pair_density_exx_a[il],
                        pair_density_exx_a[jk], pair_density_exx_a[jl]));
        if (pair_density_exx_b != NULL && exx_scale_b != 0.0f)
            screen = fmaxf(screen,
                sb * exx_scale_b *
                QC_Max4(pair_density_exx_b[ik], pair_density_exx_b[il],
                        pair_density_exx_b[jk], pair_density_exx_b[jl]));

        const bool pass = (screen >= shell_screen_tol);

        // --- LaneGroup compaction ---
        LaneMask mask = LaneGroup::Ballot(pass);
        if (LaneGroup::Any(mask))
        {
            const int count = LaneGroup::Count(mask);
            int base_slot = 0;
            const int leader = LaneGroup::First_Lane(mask);
            if (LaneGroup::Lane_Id() == leader)
                base_slot = atomicAdd(output_count, count);
            base_slot = LaneGroup::Broadcast(base_slot, leader);

            if (pass)
            {
                const int rank = LaneGroup::Prefix_Count(mask);
                output_tasks[base_slot + rank] = {pij.x, pij.y, pkl.x, pkl.y};
            }
        }
    }
}

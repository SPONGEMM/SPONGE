// On-the-fly quartet index computation from flat index within a pair-type block.
//
// For pair types A and B with counts nA and nB:
//   If A == B (same type): triangular block, n_quartets = nA*(nA+1)/2
//   If A > B (cross type, A comes later): rectangular block, n_quartets = nA*nB
//
// Expected macros:
//   KERNEL_NAME
//   SAME_PAIR_TYPE  - 0 or 1 (rectangular vs triangular)
//
// Expected runtime params:
//   pair_id_base_A, pair_id_base_B - offsets into d_sorted_pair_ids
//   n_A, n_B - counts for each type

static __global__ void KERNEL_NAME(
    const int n_quartets,
    const int pair_id_base_A, const int n_A,
    const int pair_id_base_B, const int n_B,
    const int* __restrict__ sorted_pair_ids,
    const QC_ONE_E_TASK* __restrict__ shell_pairs,
    const int* __restrict__ atm, const int* __restrict__ bas,
    const float* __restrict__ env,
    const int* __restrict__ ao_offsets_cart,
    const int* __restrict__ ao_offsets_sph,
    const float* __restrict__ norms,
    const float* __restrict__ shell_pair_bounds,
    const float* __restrict__ pair_density_coul,
    const float* __restrict__ pair_density_exx_a,
    const float* __restrict__ pair_density_exx_b,
    const float shell_screen_tol,
    const float* __restrict__ P_coul,
    const float* __restrict__ P_exx_a,
    const float* __restrict__ P_exx_b,
    const float exx_scale_a, const float exx_scale_b,
    const int nao, const int nao_sph, const int is_spherical,
    const float* __restrict__ cart2sph_mat,
    float* __restrict__ F_a, float* __restrict__ F_b,
    float prim_screen_tol)
{
    SIMPLE_DEVICE_FOR(idx, n_quartets)
    {
#ifdef GPU_ARCH_NAME
        float* F_a_accum = F_a;
        float* F_b_accum = F_b;
#else
        const int tid = omp_get_thread_num();
        const int nao2 = nao * nao;
        float* F_a_accum = F_a + (size_t)tid * (size_t)nao2;
        float* F_b_accum =
            (F_b != NULL) ? (F_b + (size_t)tid * (size_t)nao2) : NULL;
#endif

        // --- On-the-fly: flat_idx → (pair_ij, pair_kl) ---
        int local_ij, local_kl;
#if SAME_PAIR_TYPE
        // Triangular: idx maps to (local_ij, local_kl) with local_ij >= local_kl
        local_ij = (int)floor((sqrt(8.0 * (double)idx + 1.0) - 1.0) * 0.5);
        local_kl = idx - local_ij * (local_ij + 1) / 2;
        // Fix potential floating-point rounding
        if (local_ij * (local_ij + 1) / 2 + local_kl != idx)
        {
            local_ij++;
            local_kl = idx - local_ij * (local_ij + 1) / 2;
        }
#else
        // Rectangular: pair_ij from type A (higher offset), pair_kl from type B
        local_ij = idx / n_B;
        local_kl = idx % n_B;
#endif
        const int pair_ij = sorted_pair_ids[pair_id_base_A + local_ij];
        const int pair_kl = sorted_pair_ids[pair_id_base_B + local_kl];

        const QC_ONE_E_TASK pij = shell_pairs[pair_ij];
        const QC_ONE_E_TASK pkl = shell_pairs[pair_kl];

        // --- Screening ---
        const int ij_pair = QC_Shell_Pair_Index(pij.x, pij.y);
        const int kl_pair = QC_Shell_Pair_Index(pkl.x, pkl.y);
        const int ik_pair = QC_Shell_Pair_Index(pij.x, pkl.x);
        const int il_pair = QC_Shell_Pair_Index(pij.x, pkl.y);
        const int jk_pair = QC_Shell_Pair_Index(pij.y, pkl.x);
        const int jl_pair = QC_Shell_Pair_Index(pij.y, pkl.y);

        const float shell_bound =
            shell_pair_bounds[ij_pair] * shell_pair_bounds[kl_pair];
        const float coul_screen =
            shell_bound *
            fmaxf(pair_density_coul[ij_pair], pair_density_coul[kl_pair]);
        const float exx_screen_a =
            exx_scale_a == 0.0f
                ? 0.0f
                : shell_bound * exx_scale_a *
                      QC_Max4(pair_density_exx_a[ik_pair],
                              pair_density_exx_a[il_pair],
                              pair_density_exx_a[jk_pair],
                              pair_density_exx_a[jl_pair]);
        float exx_screen_b = 0.0f;
        if (F_b != NULL && pair_density_exx_b != NULL && exx_scale_b != 0.0f)
        {
            exx_screen_b =
                shell_bound * exx_scale_b *
                QC_Max4(pair_density_exx_b[ik_pair],
                        pair_density_exx_b[il_pair],
                        pair_density_exx_b[jk_pair],
                        pair_density_exx_b[jl_pair]);
        }
        if (fmaxf(coul_screen, fmaxf(exx_screen_a, exx_screen_b)) <
            shell_screen_tol)
        {
            // Screened out — skip
        }
        else
        {
        // --- The rest is identical to existing ERI kernels ---
        // Shell indices
        const int sh[4] = {pij.x, pij.y, pkl.x, pkl.y};

        // TODO: Insert ERI computation here.
        // For now, delegate to QC_Compute_Shell_Quartet_ERI_Buffer via
        // the generic kernel path. This will be replaced with specialized
        // register-only code per pair-type combination.

        // Placeholder: this is where the specialized ERI code goes.
        // The (l0, l1, l2, l3) are known at compile time from the
        // pair types of the kernel instance.

        } // end screening
    }
}

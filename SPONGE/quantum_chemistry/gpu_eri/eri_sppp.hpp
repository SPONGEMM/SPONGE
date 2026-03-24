#pragma once

// Specialized Fock kernel for (sp|pp)-type shell quartets (one s shell, three p shells).
// Uses generic QC_Compute_Shell_Quartet_ERI_Buffer with scratch buffers.

static __global__ void QC_Fock_sppp_Kernel(
    const int n_tasks, const QC_ERI_TASK* __restrict__ tasks,
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
    float* __restrict__ global_hr_pool, int hr_base, int hr_size,
    int shell_buf_size, float prim_screen_tol)
{
    SIMPLE_DEVICE_FOR(task_id, n_tasks)
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
        const QC_ERI_TASK t = tasks[task_id];
        const int ij_pair = QC_Shell_Pair_Index(t.x, t.y);
        const int kl_pair = QC_Shell_Pair_Index(t.z, t.w);
        const int ik_pair = QC_Shell_Pair_Index(t.x, t.z);
        const int il_pair = QC_Shell_Pair_Index(t.x, t.w);
        const int jk_pair = QC_Shell_Pair_Index(t.y, t.z);
        const int jl_pair = QC_Shell_Pair_Index(t.y, t.w);

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
        if (fmaxf(coul_screen, fmaxf(exx_screen_a, exx_screen_b)) >=
            shell_screen_tol)
        {
#ifdef GPU_ARCH_NAME
            const int scratch_id = task_id;
#else
            const int scratch_id = omp_get_thread_num();
#endif
            float* task_pool = global_hr_pool +
                               (int)scratch_id * (hr_size + 2 * shell_buf_size);
            float* HR = task_pool;
            float* shell_eri = task_pool + hr_size;
            float* shell_tmp = shell_eri + shell_buf_size;
            int sh[4] = {t.x, t.y, t.z, t.w};
            int dims_eff[4];
            int off_eff[4];
            if (QC_Compute_Shell_Quartet_ERI_Buffer(
                    sh, atm, bas, env, ao_offsets_cart, ao_offsets_sph, norms,
                    is_spherical, cart2sph_mat, nao_sph, HR, shell_eri,
                    shell_tmp, hr_base, shell_buf_size, prim_screen_tol,
                    dims_eff, off_eff))
            {
                const bool jk_same_bra = (t.x == t.y);
                const bool jk_same_ket = (t.z == t.w);
                const bool jk_same_braket = (t.x == t.z && t.y == t.w);
                for (int i = 0; i < dims_eff[0]; i++)
                {
                    const int p = off_eff[0] + i;
                    for (int j = 0; j < dims_eff[1]; j++)
                    {
                        const int q = off_eff[1] + j;
                        if (jk_same_bra && q > p) continue;
                        for (int k = 0; k < dims_eff[2]; k++)
                        {
                            const int r = off_eff[2] + k;
                            for (int l_idx = 0; l_idx < dims_eff[3]; l_idx++)
                            {
                                const int s = off_eff[3] + l_idx;
                                if (jk_same_ket && s > r) continue;

                                if (jk_same_braket)
                                {
                                    const int pq_pair = QC_AO_Pair_Index(p, q);
                                    const int rs_pair = QC_AO_Pair_Index(r, s);
                                    if (rs_pair > pq_pair) continue;
                                }

                                const float val =
                                    shell_eri[QC_Shell_Buffer_Index(
                                        i, j, k, l_idx, dims_eff[1],
                                        dims_eff[2], dims_eff[3])];
                                if (val == 0.0f) continue;
                                if (!jk_same_bra && !jk_same_ket &&
                                    !jk_same_braket)
                                {
                                    QC_Accumulate_Fock_General_Quartet(
                                        p, q, r, s, val, nao, P_coul, P_exx_a,
                                        P_exx_b, exx_scale_a, exx_scale_b,
                                        F_a_accum, F_b_accum);
                                }
                                else
                                {
                                    QC_Accumulate_Fock_Unique_Quartet(
                                        p, q, r, s, val, nao, P_coul, P_exx_a,
                                        P_exx_b, exx_scale_a, exx_scale_b,
                                        F_a_accum, F_b_accum);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

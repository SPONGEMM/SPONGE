#pragma once

// Register-only (ss|ss) Fock-build kernel.
// L_sum=0: only needs Boys F0(T). Single ERI value per task.
// No scratch buffers — everything in registers.

__global__ void QC_Fock_ssss_Kernel(
    const int n_tasks, const QC_ERI_TASK* __restrict__ tasks,
    const int* __restrict__ atm, const int* __restrict__ bas,
    const float* __restrict__ env, const int* __restrict__ ao_offsets_cart,
    const int* __restrict__ ao_offsets_sph, const float* __restrict__ norms,
    const float* __restrict__ shell_pair_bounds,
    const float* __restrict__ pair_density_coul,
    const float* __restrict__ pair_density_exx_a,
    const float* __restrict__ pair_density_exx_b, const float shell_screen_tol,
    const float* __restrict__ P_coul, const float* __restrict__ P_exx_a,
    const float* __restrict__ P_exx_b, const float exx_scale_a,
    const float exx_scale_b, const int nao, const int nao_sph,
    const int is_spherical, const float* __restrict__ cart2sph_mat,
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
        const QC_ERI_TASK tk = tasks[task_id];
        const int ij_pair = QC_Shell_Pair_Index(tk.x, tk.y);
        const int kl_pair = QC_Shell_Pair_Index(tk.z, tk.w);
        const int ik_pair = QC_Shell_Pair_Index(tk.x, tk.z);
        const int il_pair = QC_Shell_Pair_Index(tk.x, tk.w);
        const int jk_pair = QC_Shell_Pair_Index(tk.y, tk.z);
        const int jl_pair = QC_Shell_Pair_Index(tk.y, tk.w);

        const float shell_bound =
            shell_pair_bounds[ij_pair] * shell_pair_bounds[kl_pair];
        const float coul_screen =
            shell_bound *
            fmaxf(pair_density_coul[ij_pair], pair_density_coul[kl_pair]);
        const float exx_screen_a =
            exx_scale_a == 0.0f ? 0.0f
                                : shell_bound * exx_scale_a *
                                      QC_Max4(pair_density_exx_a[ik_pair],
                                              pair_density_exx_a[il_pair],
                                              pair_density_exx_a[jk_pair],
                                              pair_density_exx_a[jl_pair]);
        float exx_screen_b = 0.0f;
        if (F_b != NULL && pair_density_exx_b != NULL && exx_scale_b != 0.0f)
        {
            exx_screen_b = shell_bound * exx_scale_b *
                           QC_Max4(pair_density_exx_b[ik_pair],
                                   pair_density_exx_b[il_pair],
                                   pair_density_exx_b[jk_pair],
                                   pair_density_exx_b[jl_pair]);
        }
        if (fmaxf(coul_screen, fmaxf(exx_screen_a, exx_screen_b)) >=
            shell_screen_tol)
        {
            // ---- Read shell data ----
            const int sh[4] = {tk.x, tk.y, tk.z, tk.w};
            int np[4], p_exp_off[4], p_cof_off[4];
            float R[4][3];
            int off[4];

            for (int i = 0; i < 4; i++)
            {
                const int si8 = sh[i] * 8;
                np[i] = bas[si8 + 2];
                p_exp_off[i] = bas[si8 + 5];
                p_cof_off[i] = bas[si8 + 6];
                const int ptr_R = atm[bas[si8 + 0] * 6 + 1];
                R[i][0] = env[ptr_R + 0];
                R[i][1] = env[ptr_R + 1];
                R[i][2] = env[ptr_R + 2];
                off[i] = is_spherical ? ao_offsets_sph[sh[i]]
                                      : ao_offsets_cart[sh[i]];
            }

            const float rab2 = (R[0][0] - R[1][0]) * (R[0][0] - R[1][0]) +
                               (R[0][1] - R[1][1]) * (R[0][1] - R[1][1]) +
                               (R[0][2] - R[1][2]) * (R[0][2] - R[1][2]);
            const float rcd2 = (R[2][0] - R[3][0]) * (R[2][0] - R[3][0]) +
                               (R[2][1] - R[3][1]) * (R[2][1] - R[3][1]) +
                               (R[2][2] - R[3][2]) * (R[2][2] - R[3][2]);

            // ---- Primitive contraction: single ERI = sum n_abcd * F0(T) ----
            float eri = 0.0f;

            for (int ip = 0; ip < np[0]; ip++)
            {
                const float ai = env[p_exp_off[0] + ip];
                const float ci = env[p_cof_off[0] + ip];
                for (int jp = 0; jp < np[1]; jp++)
                {
                    const float aj = env[p_exp_off[1] + jp];
                    const float p_val = ai + aj;
                    const float inv_p = 1.0f / p_val;
                    const float kab = expf(-(ai * aj * inv_p) * rab2);
                    const float n_ab = ci * env[p_cof_off[1] + jp] * kab;
                    if (fabsf(n_ab) < prim_screen_tol) continue;

                    const float Px = (ai * R[0][0] + aj * R[1][0]) * inv_p;
                    const float Py = (ai * R[0][1] + aj * R[1][1]) * inv_p;
                    const float Pz = (ai * R[0][2] + aj * R[1][2]) * inv_p;

                    for (int kp = 0; kp < np[2]; kp++)
                    {
                        const float ak = env[p_exp_off[2] + kp];
                        const float ck = env[p_cof_off[2] + kp];
                        for (int lp = 0; lp < np[3]; lp++)
                        {
                            const float al = env[p_exp_off[3] + lp];
                            const float q_val = ak + al;
                            const float inv_q = 1.0f / q_val;
                            const float kcd = expf(-(ak * al * inv_q) * rcd2);

                            const float pref =
                                2.0f * PI_25 /
                                (p_val * q_val * sqrtf(p_val + q_val));
                            const float n_abcd =
                                n_ab * ck * env[p_cof_off[3] + lp] * kcd * pref;
                            if (fabsf(n_abcd) < prim_screen_tol) continue;

                            const float Qx =
                                (ak * R[2][0] + al * R[3][0]) * inv_q;
                            const float Qy =
                                (ak * R[2][1] + al * R[3][1]) * inv_q;
                            const float Qz =
                                (ak * R[2][2] + al * R[3][2]) * inv_q;
                            const float PQx = Px - Qx, PQy = Py - Qy,
                                        PQz = Pz - Qz;
                            const float alpha = p_val * q_val / (p_val + q_val);
                            const float T =
                                alpha * (PQx * PQx + PQy * PQy + PQz * PQz);

                            // Boys F0 only
                            const double td = (double)T;
                            double F0_d;
                            if (td < 1e-7)
                                F0_d =
                                    1.0 - td * (1.0 / 3.0 -
                                                td * (1.0 / 10.0 - td / 42.0));
                            else
                            {
                                const double st = sqrt(td);
                                F0_d = 0.5 * 1.7724538509055159 * erf(st) / st;
                            }

                            eri += n_abcd * (float)F0_d;
                        }
                    }
                }
            }

            // ---- Apply cart2sph (scalar for s shells) + norms ----
            float c2s = 1.0f;
            if (is_spherical)
            {
                for (int i = 0; i < 4; i++)
                    c2s *=
                        cart2sph_mat[ao_offsets_cart[sh[i]] * nao_sph + off[i]];
            }
            const float norm_all =
                norms[off[0]] * norms[off[1]] * norms[off[2]] * norms[off[3]];
            const float val = eri * norm_all * c2s;
            if (val != 0.0f)
            {
                const int p = off[0], q = off[1], r = off[2], s = off[3];
                const bool jk_same_bra = (tk.x == tk.y);
                const bool jk_same_ket = (tk.z == tk.w);
                const bool jk_same_braket = (tk.x == tk.z && tk.y == tk.w);

                if (!jk_same_bra && !jk_same_ket && !jk_same_braket)
                    QC_Accumulate_Fock_General_Quartet(
                        p, q, r, s, val, nao, P_coul, P_exx_a, P_exx_b,
                        exx_scale_a, exx_scale_b, F_a_accum, F_b_accum);
                else
                    QC_Accumulate_Fock_Unique_Quartet(
                        p, q, r, s, val, nao, P_coul, P_exx_a, P_exx_b,
                        exx_scale_a, exx_scale_b, F_a_accum, F_b_accum);
            }
        }  // end screening
    }
}

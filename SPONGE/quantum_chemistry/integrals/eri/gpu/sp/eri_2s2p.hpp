// Register-only kernel for 2s+2p shell quartets (L_sum=2).
// Included multiple times with different P_POS0, P_POS1, KERNEL_NAME.
//
// Expected macros:
//   P_POS0     - position of 1st p shell (0-3), P_POS0 < P_POS1
//   P_POS1     - position of 2nd p shell (0-3)
//   KERNEL_NAME
//
// No #pragma once.

#define _2S2P_IS_P(pos) ((pos) == P_POS0 || (pos) == P_POS1)

__global__ void KERNEL_NAME(
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
    // Compile-time constants from macros
    const int l[4] = {_2S2P_IS_P(0), _2S2P_IS_P(1), _2S2P_IS_P(2),
                      _2S2P_IS_P(3)};
    const int dim_c[4] = {_2S2P_IS_P(0) ? 3 : 1, _2S2P_IS_P(1) ? 3 : 1,
                          _2S2P_IS_P(2) ? 3 : 1, _2S2P_IS_P(3) ? 3 : 1};
    const int dim_e[4] = {dim_c[0], dim_c[1], dim_c[2],
                          dim_c[3]};  // p: cart==sph==3
    const int n_cart = dim_c[0] * dim_c[1] * dim_c[2] * dim_c[3];  // 9

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

        // ---- Screening ----
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
            exx_screen_b = shell_bound * exx_scale_b *
                           QC_Max4(pair_density_exx_b[ik_pair],
                                   pair_density_exx_b[il_pair],
                                   pair_density_exx_b[jk_pair],
                                   pair_density_exx_b[jl_pair]);
        if (fmaxf(coul_screen, fmaxf(exx_screen_a, exx_screen_b)) >=
            shell_screen_tol)
        {
            const int sh[4] = {tk.x, tk.y, tk.z, tk.w};
            int np[4], p_exp_off[4], p_cof_off[4];
            float RC[4][3];
            int off[4];
            for (int i = 0; i < 4; i++)
            {
                const int si8 = sh[i] * 8;
                np[i] = bas[si8 + 2];
                p_exp_off[i] = bas[si8 + 5];
                p_cof_off[i] = bas[si8 + 6];
                const int ptr_R = atm[bas[si8] * 6 + 1];
                RC[i][0] = env[ptr_R];
                RC[i][1] = env[ptr_R + 1];
                RC[i][2] = env[ptr_R + 2];
                off[i] = is_spherical ? ao_offsets_sph[sh[i]]
                                      : ao_offsets_cart[sh[i]];
            }
            const float rab2 = (RC[0][0] - RC[1][0]) * (RC[0][0] - RC[1][0]) +
                               (RC[0][1] - RC[1][1]) * (RC[0][1] - RC[1][1]) +
                               (RC[0][2] - RC[1][2]) * (RC[0][2] - RC[1][2]);
            const float rcd2 = (RC[2][0] - RC[3][0]) * (RC[2][0] - RC[3][0]) +
                               (RC[2][1] - RC[3][1]) * (RC[2][1] - RC[3][1]) +
                               (RC[2][2] - RC[3][2]) * (RC[2][2] - RC[3][2]);

            float eri_cart[9];  // max 3*3 for 2s2p
            for (int i = 0; i < n_cart; i++) eri_cart[i] = 0.0f;

            for (int ip = 0; ip < np[0]; ip++)
            {
                const float ai = env[p_exp_off[0] + ip],
                            ci = env[p_cof_off[0] + ip];
                for (int jp = 0; jp < np[1]; jp++)
                {
                    const float aj = env[p_exp_off[1] + jp];
                    const float p_val = ai + aj, inv_p = 1.0f / p_val;
                    const float kab = expf(-(ai * aj * inv_p) * rab2);
                    const float n_ab = ci * env[p_cof_off[1] + jp] * kab;
                    if (fabsf(n_ab) < prim_screen_tol) continue;
                    const float Px = (ai * RC[0][0] + aj * RC[1][0]) * inv_p;
                    const float Py = (ai * RC[0][1] + aj * RC[1][1]) * inv_p;
                    const float Pz = (ai * RC[0][2] + aj * RC[1][2]) * inv_p;
                    const float PA[3] = {Px - RC[0][0], Py - RC[0][1],
                                         Pz - RC[0][2]};
                    const float PB[3] = {Px - RC[1][0], Py - RC[1][1],
                                         Pz - RC[1][2]};
                    const float inv2p = 0.5f * inv_p;
                    for (int kp = 0; kp < np[2]; kp++)
                    {
                        const float ak = env[p_exp_off[2] + kp],
                                    ck = env[p_cof_off[2] + kp];
                        for (int lp = 0; lp < np[3]; lp++)
                        {
                            const float al = env[p_exp_off[3] + lp];
                            const float q_val = ak + al, inv_q = 1.0f / q_val;
                            const float kcd = expf(-(ak * al * inv_q) * rcd2);
                            const float pref =
                                2.0f * PI_25 /
                                (p_val * q_val * sqrtf(p_val + q_val));
                            const float n_abcd =
                                n_ab * ck * env[p_cof_off[3] + lp] * kcd * pref;
                            if (fabsf(n_abcd) < prim_screen_tol) continue;
                            const float Qx =
                                (ak * RC[2][0] + al * RC[3][0]) * inv_q;
                            const float Qy =
                                (ak * RC[2][1] + al * RC[3][1]) * inv_q;
                            const float Qz =
                                (ak * RC[2][2] + al * RC[3][2]) * inv_q;
                            const float QCv[3] = {Qx - RC[2][0], Qy - RC[2][1],
                                                  Qz - RC[2][2]};
                            const float QD[3] = {Qx - RC[3][0], Qy - RC[3][1],
                                                 Qz - RC[3][2]};
                            const float inv2q = 0.5f * inv_q;
                            const float PQ[3] = {Px - Qx, Py - Qy, Pz - Qz};
                            const float alpha = p_val * q_val / (p_val + q_val);
                            const float T =
                                alpha *
                                (PQ[0] * PQ[0] + PQ[1] * PQ[1] + PQ[2] * PQ[2]);

                            double F_boys[3];
                            eri_boys(F_boys, T, 2);
                            float R0[10], Rw[15];
                            eri_build_R0(R0, Rw, F_boys, alpha, PQ, 2);

                            int idx = 0;
                            for (int c0 = 0; c0 < dim_c[0]; c0++)
                                for (int c1 = 0; c1 < dim_c[1]; c1++)
                                    for (int c2 = 0; c2 < dim_c[2]; c2++)
                                        for (int c3 = 0; c3 < dim_c[3]; c3++)
                                        {
                                            const int cv[4] = {c0, c1, c2, c3};
                                            eri_cart[idx++] +=
                                                n_abcd *
                                                eri_contract(l, cv, PA, PB, QCv,
                                                             QD, inv2p, inv2q,
                                                             R0);
                                        }
                        }
                    }
                }
            }

            // ---- Cart2sph for p shells + s-shell scalar ----
            if (is_spherical)
            {
                for (int si = 0; si < 4; si++)
                {
                    if (l[si] == 0)
                    {
                        const float c2s =
                            cart2sph_mat[ao_offsets_cart[sh[si]] * nao_sph +
                                         off[si]];
                        for (int i = 0; i < n_cart; i++) eri_cart[i] *= c2s;
                        continue;
                    }
                    const int oc = ao_offsets_cart[sh[si]], os = off[si];
                    int stride = 1;
                    int cur_dim[4] = {dim_c[0], dim_c[1], dim_c[2], dim_c[3]};
                    for (int j = si + 1; j < 4; j++) stride *= cur_dim[j];
                    const int n_outer = n_cart / (3 * stride);
                    float tmp[9];
                    for (int i = 0; i < n_cart; i++) tmp[i] = eri_cart[i];
                    for (int o = 0; o < n_outer; o++)
                        for (int s = 0; s < 3; s++)
                            for (int inn = 0; inn < stride; inn++)
                            {
                                double sum = 0.0;
                                for (int cc = 0; cc < 3; cc++)
                                    sum +=
                                        (double)
                                            cart2sph_mat[(oc + cc) * nao_sph +
                                                         (os + s)] *
                                        (double)tmp[o * 3 * stride +
                                                    cc * stride + inn];
                                eri_cart[o * 3 * stride + s * stride + inn] =
                                    (float)sum;
                            }
                }
            }

            // ---- Norms ----
            {
                int idx = 0;
                for (int c0 = 0; c0 < dim_e[0]; c0++)
                    for (int c1 = 0; c1 < dim_e[1]; c1++)
                        for (int c2 = 0; c2 < dim_e[2]; c2++)
                            for (int c3 = 0; c3 < dim_e[3]; c3++)
                                eri_cart[idx++] *=
                                    norms[off[0] + c0] * norms[off[1] + c1] *
                                    norms[off[2] + c2] * norms[off[3] + c3];
            }

            // ---- Fock accumulation with dedup ----
            const bool jk_same_bra = (tk.x == tk.y);
            const bool jk_same_ket = (tk.z == tk.w);
            const bool jk_same_braket = (tk.x == tk.z && tk.y == tk.w);
            {
                int idx = 0;
                for (int c0 = 0; c0 < dim_e[0]; c0++)
                {
                    const int p = off[0] + c0;
                    for (int c1 = 0; c1 < dim_e[1]; c1++)
                    {
                        const int q = off[1] + c1;
                        if (jk_same_bra && q > p)
                        {
                            idx += dim_e[2] * dim_e[3];
                            continue;
                        }
                        for (int c2 = 0; c2 < dim_e[2]; c2++)
                        {
                            const int r = off[2] + c2;
                            for (int c3 = 0; c3 < dim_e[3]; c3++)
                            {
                                const int s = off[3] + c3;
                                const float val = eri_cart[idx++];
                                if (jk_same_ket && s > r) continue;
                                if (jk_same_braket)
                                {
                                    if (QC_AO_Pair_Index(r, s) >
                                        QC_AO_Pair_Index(p, q))
                                        continue;
                                }
                                if (val == 0.0f) continue;
                                if (!jk_same_bra && !jk_same_ket &&
                                    !jk_same_braket)
                                    QC_Accumulate_Fock_General_Quartet(
                                        p, q, r, s, val, nao, P_coul, P_exx_a,
                                        P_exx_b, exx_scale_a, exx_scale_b,
                                        F_a_accum, F_b_accum);
                                else
                                    QC_Accumulate_Fock_Unique_Quartet(
                                        p, q, r, s, val, nao, P_coul, P_exx_a,
                                        P_exx_b, exx_scale_a, exx_scale_b,
                                        F_a_accum, F_b_accum);
                            }
                        }
                    }
                }
            }
        }  // screening
    }
}

#undef _2S2P_IS_P

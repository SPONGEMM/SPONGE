// Register-only Fock kernel for d-containing quartets at a fixed L_sum.
// Included multiple times with different ERI_LSUM and KERNEL_NAME.
//
// Expected macros:
//   ERI_LSUM     - total angular momentum sum (compile-time constant)
//   KERNEL_NAME  - name of the generated kernel
//   ERI_MAX_CART - max product of Cartesian dims for this L_sum
//   ERI_F_SIZE   - ERI_LSUM + 1
//   ERI_R0_SIZE  - eri_T_count(ERI_LSUM)
//   ERI_RW_SIZE  - sum of eri_T_count(ERI_LSUM - n) for n=0..ERI_LSUM
//
// No #pragma once — designed for multiple inclusion.

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
            int l[4], np[4], p_exp_off[4], p_cof_off[4];
            float RC[4][3];
            int off[4], dim_cart[4], dim_eff[4];
            for (int i = 0; i < 4; i++)
            {
                const int si8 = sh[i] * 8;
                l[i] = bas[si8 + 1];
                np[i] = bas[si8 + 2];
                p_exp_off[i] = bas[si8 + 5];
                p_cof_off[i] = bas[si8 + 6];
                dim_cart[i] = (l[i] + 1) * (l[i] + 2) / 2;
                dim_eff[i] = QC_Shell_Dim(l[i], is_spherical);
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

            const int n_cart =
                dim_cart[0] * dim_cart[1] * dim_cart[2] * dim_cart[3];
            const int n_eff = dim_eff[0] * dim_eff[1] * dim_eff[2] * dim_eff[3];
            float eri_cart[ERI_MAX_CART];  // compile-time sized
            for (int i = 0; i < n_cart; i++) eri_cart[i] = 0.0f;

            // ---- Primitive loop ----
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

                            // Compile-time sized arrays
                            double F_boys[ERI_F_SIZE];
                            eri_boys(F_boys, T, ERI_LSUM);
                            float R0[ERI_R0_SIZE], Rw[ERI_RW_SIZE];
                            eri_build_R0(R0, Rw, F_boys, alpha, PQ, ERI_LSUM);

                            int idx = 0;
                            for (int c0 = 0; c0 < dim_cart[0]; c0++)
                                for (int c1 = 0; c1 < dim_cart[1]; c1++)
                                    for (int c2 = 0; c2 < dim_cart[2]; c2++)
                                        for (int c3 = 0; c3 < dim_cart[3]; c3++)
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

            // ---- Cart2sph ----
            float eri_out[ERI_MAX_CART];
            if (is_spherical)
            {
                int cur_dim[4] = {dim_cart[0], dim_cart[1], dim_cart[2],
                                  dim_cart[3]};
                float buf_a[ERI_MAX_CART], buf_b[ERI_MAX_CART];
                int n_cur = n_cart;
                for (int i = 0; i < n_cur; i++) buf_a[i] = eri_cart[i];
                for (int si = 0; si < 4; si++)
                {
                    const int nc = dim_cart[si], ns = dim_eff[si];
                    if (nc == ns && l[si] == 0)
                    {
                        float c2s =
                            cart2sph_mat[ao_offsets_cart[sh[si]] * nao_sph +
                                         off[si]];
                        for (int i = 0; i < n_cur; i++) buf_a[i] *= c2s;
                        continue;
                    }
                    const int oc = ao_offsets_cart[sh[si]], os = off[si];
                    int stride_si = 1;
                    for (int j = si + 1; j < 4; j++) stride_si *= cur_dim[j];
                    const int n_outer = n_cur / (nc * stride_si);
                    const int n_new = n_cur / nc * ns;
                    for (int o = 0; o < n_outer; o++)
                        for (int s = 0; s < ns; s++)
                            for (int inn = 0; inn < stride_si; inn++)
                            {
                                double sum = 0.0;
                                for (int cc = 0; cc < nc; cc++)
                                    sum +=
                                        (double)
                                            cart2sph_mat[(oc + cc) * nao_sph +
                                                         (os + s)] *
                                        (double)buf_a[o * nc * stride_si +
                                                      cc * stride_si + inn];
                                buf_b[o * ns * stride_si + s * stride_si +
                                      inn] = (float)sum;
                            }
                    for (int i = 0; i < n_new; i++) buf_a[i] = buf_b[i];
                    cur_dim[si] = ns;
                    n_cur = n_new;
                }
                for (int i = 0; i < n_eff; i++) eri_out[i] = buf_a[i];
            }
            else
                for (int i = 0; i < n_cart; i++) eri_out[i] = eri_cart[i];

            // ---- Norms ----
            {
                int idx = 0;
                for (int c0 = 0; c0 < dim_eff[0]; c0++)
                    for (int c1 = 0; c1 < dim_eff[1]; c1++)
                        for (int c2 = 0; c2 < dim_eff[2]; c2++)
                            for (int c3 = 0; c3 < dim_eff[3]; c3++)
                                eri_out[idx++] *=
                                    norms[off[0] + c0] * norms[off[1] + c1] *
                                    norms[off[2] + c2] * norms[off[3] + c3];
            }

            // ---- Fock with dedup ----
            const bool jk_same_bra = (tk.x == tk.y);
            const bool jk_same_ket = (tk.z == tk.w);
            const bool jk_same_braket = (tk.x == tk.z && tk.y == tk.w);
            {
                int idx = 0;
                for (int c0 = 0; c0 < dim_eff[0]; c0++)
                {
                    const int p = off[0] + c0;
                    for (int c1 = 0; c1 < dim_eff[1]; c1++)
                    {
                        const int q = off[1] + c1;
                        if (jk_same_bra && q > p)
                        {
                            idx += dim_eff[2] * dim_eff[3];
                            continue;
                        }
                        for (int c2 = 0; c2 < dim_eff[2]; c2++)
                        {
                            const int r = off[2] + c2;
                            for (int c3 = 0; c3 < dim_eff[3]; c3++)
                            {
                                const int s = off[3] + c3;
                                const float val = eri_out[idx++];
                                if (jk_same_ket && s > r) continue;
                                if (jk_same_braket)
                                    if (QC_AO_Pair_Index(r, s) >
                                        QC_AO_Pair_Index(p, q))
                                        continue;
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

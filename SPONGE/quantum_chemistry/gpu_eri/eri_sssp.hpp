#pragma once

// Register-only specialized kernel for (ss|sp)-type shell quartets.
// Exactly one of the four shells is p (l=1), the other three are s (l=0).
//
// McMurchie-Davidson for L_sum=1 reduces to:
//   R^0_{000} = F_0(T)
//   R^0_{100} = PQ_x * (-2α) * F_1(T),  and similarly for y,z
//
// For each Cartesian p-component d ∈ {x,y,z}:
//   ERI_d = Σ_prims  n_abcd * (shift_d * F0 + rcoeff * PQ_d * m2a_F1)
//
// where shift = PA/PB/QC/QD depending on p-shell position,
//       rcoeff = +1/(2p) for bra positions, -1/(2q) for ket positions.
//
// No scratch buffers needed — everything stays in registers.

// Lightweight Boys F0,F1 for L_sum=1.  No arrays — pure register computation.
// Returns F0 and (-2*alpha)*F1 directly for the R-tensor contraction.
static __device__ __forceinline__ void sssp_boys_01(
    float& out_F0, float& out_m2a_F1, float t_arg, float alpha)
{
    const double td = (double)t_arg;
    double F0_d, F1_d;
    if (td < 1e-7)
    {
        // Taylor: F_n(T) = 1/(2n+1) - T/(2n+3) + T^2/(2*(2n+5)) - ...
        F0_d = 1.0 - td * (1.0 / 3.0 - td * (1.0 / 10.0 - td / 42.0));
        F1_d = (1.0 / 3.0) - td * (1.0 / 5.0 - td * (1.0 / 14.0 - td / 54.0));
    }
    else
    {
        const double exp_t = exp(-td);
        const double st = sqrt(td);
        F0_d = 0.5 * 1.7724538509055159 * erf(st) / st;
        // Upward recursion: F1 = ((2*0+1)*F0 - exp(-T)) / (2T)
        F1_d = (F0_d - exp_t) / (2.0 * td);
    }
    out_F0 = (float)F0_d;
    out_m2a_F1 = (float)(-2.0 * (double)alpha * F1_d);
}

static __global__ void QC_Fock_sssp_Kernel(
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
        // ---- Read shell data ----
        const int sh[4] = {tk.x, tk.y, tk.z, tk.w};
        int p_pos = -1;
        int np[4], p_exp_off[4], p_cof_off[4];
        float R[4][3];
        int off[4];

        for (int i = 0; i < 4; i++)
        {
            const int si = sh[i];
            const int si8 = si * 8;
            const int li = bas[si8 + 1];
            if (li == 1) p_pos = i;
            np[i] = bas[si8 + 2];
            p_exp_off[i] = bas[si8 + 5];
            p_cof_off[i] = bas[si8 + 6];
            const int ptr_R = atm[bas[si8 + 0] * 6 + 1];
            R[i][0] = env[ptr_R + 0];
            R[i][1] = env[ptr_R + 1];
            R[i][2] = env[ptr_R + 2];
            off[i] = is_spherical ? ao_offsets_sph[si] : ao_offsets_cart[si];
        }

        const float rab2 = (R[0][0] - R[1][0]) * (R[0][0] - R[1][0]) +
                            (R[0][1] - R[1][1]) * (R[0][1] - R[1][1]) +
                            (R[0][2] - R[1][2]) * (R[0][2] - R[1][2]);
        const float rcd2 = (R[2][0] - R[3][0]) * (R[2][0] - R[3][0]) +
                            (R[2][1] - R[3][1]) * (R[2][1] - R[3][1]) +
                            (R[2][2] - R[3][2]) * (R[2][2] - R[3][2]);

        // ---- Primitive contraction: accumulate 3 Cartesian ERI values ----
        float eri_x = 0.0f, eri_y = 0.0f, eri_z = 0.0f;

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
                            2.0f * PI_25 / (p_val * q_val * sqrtf(p_val + q_val));
                        const float n_abcd =
                            n_ab * ck * env[p_cof_off[3] + lp] * kcd * pref;
                        if (fabsf(n_abcd) < prim_screen_tol) continue;

                        const float Qx = (ak * R[2][0] + al * R[3][0]) * inv_q;
                        const float Qy = (ak * R[2][1] + al * R[3][1]) * inv_q;
                        const float Qz = (ak * R[2][2] + al * R[3][2]) * inv_q;

                        const float PQx = Px - Qx;
                        const float PQy = Py - Qy;
                        const float PQz = Pz - Qz;
                        const float alpha =
                            p_val * q_val / (p_val + q_val);
                        const float T =
                            alpha * (PQx * PQx + PQy * PQy + PQz * PQz);

                        // Boys F0(T) and (-2α)F1(T) — register-only, no arrays
                        float F0, m2a_F1;
                        sssp_boys_01(F0, m2a_F1, T, alpha);

                        // Determine shift vector and recurrence coefficient
                        // Bra positions (0,1): shift = PA or PB, rcoeff = +1/(2p)
                        // Ket positions (2,3): shift = QC or QD, rcoeff = -1/(2q)
                        float sx, sy, sz, rcoeff;
                        if (p_pos == 0)
                        {
                            sx = Px - R[0][0];
                            sy = Py - R[0][1];
                            sz = Pz - R[0][2];
                            rcoeff = 0.5f * inv_p;
                        }
                        else if (p_pos == 1)
                        {
                            sx = Px - R[1][0];
                            sy = Py - R[1][1];
                            sz = Pz - R[1][2];
                            rcoeff = 0.5f * inv_p;
                        }
                        else if (p_pos == 2)
                        {
                            sx = Qx - R[2][0];
                            sy = Qy - R[2][1];
                            sz = Qz - R[2][2];
                            rcoeff = -0.5f * inv_q;
                        }
                        else // p_pos == 3
                        {
                            sx = Qx - R[3][0];
                            sy = Qy - R[3][1];
                            sz = Qz - R[3][2];
                            rcoeff = -0.5f * inv_q;
                        }

                        // ERI_d = n_abcd * (shift_d * F0 + rcoeff * PQ_d * m2a_F1)
                        eri_x += n_abcd * (sx * F0 + rcoeff * PQx * m2a_F1);
                        eri_y += n_abcd * (sy * F0 + rcoeff * PQy * m2a_F1);
                        eri_z += n_abcd * (sz * F0 + rcoeff * PQz * m2a_F1);
                    }
                }
            }
        }

        // ---- Cart→Sph transform for the p shell (if spherical) ----
        // For l=1: dims_cart = dims_sph = 3. s shells are trivially identity.
        float eri_final[3];
        if (is_spherical)
        {
            const int oc = ao_offsets_cart[sh[p_pos]];
            const int os = off[p_pos]; // = ao_offsets_sph[sh[p_pos]]
            const float eri_cart[3] = {eri_x, eri_y, eri_z};
            for (int s = 0; s < 3; s++)
            {
                double sum = 0.0;
                for (int c = 0; c < 3; c++)
                    sum += (double)cart2sph_mat[(oc + c) * nao_sph + (os + s)] *
                           (double)eri_cart[c];
                eri_final[s] = (float)sum;
            }
        }
        else
        {
            eri_final[0] = eri_x;
            eri_final[1] = eri_y;
            eri_final[2] = eri_z;
        }

        // ---- Apply norms and accumulate into Fock matrix ----
        // s-shell norms: single value each
        float norm_s = 1.0f;
        for (int i = 0; i < 4; i++)
            if (i != p_pos) norm_s *= norms[off[i]];

        const bool jk_same_bra = (tk.x == tk.y);
        const bool jk_same_ket = (tk.z == tk.w);
        const bool jk_same_braket = (tk.x == tk.z && tk.y == tk.w);
        const int p_off = off[p_pos];

        for (int d = 0; d < 3; d++)
        {
            const float val = eri_final[d] * norm_s * norms[p_off + d];
            if (val == 0.0f) continue;

            // Build AO indices: s shells use off[i], p shell uses off[p_pos]+d
            const int ao_p = (p_pos == 0) ? (off[0] + d) : off[0];
            const int ao_q = (p_pos == 1) ? (off[1] + d) : off[1];
            const int ao_r = (p_pos == 2) ? (off[2] + d) : off[2];
            const int ao_s = (p_pos == 3) ? (off[3] + d) : off[3];

            if (!jk_same_bra && !jk_same_ket && !jk_same_braket)
            {
                QC_Accumulate_Fock_General_Quartet(
                    ao_p, ao_q, ao_r, ao_s, val, nao, P_coul, P_exx_a,
                    P_exx_b, exx_scale_a, exx_scale_b, F_a_accum, F_b_accum);
            }
            else
            {
                QC_Accumulate_Fock_Unique_Quartet(
                    ao_p, ao_q, ao_r, ao_s, val, nao, P_coul, P_exx_a,
                    P_exx_b, exx_scale_a, exx_scale_b, F_a_accum, F_b_accum);
            }
        }
        } // end screening
    }
}

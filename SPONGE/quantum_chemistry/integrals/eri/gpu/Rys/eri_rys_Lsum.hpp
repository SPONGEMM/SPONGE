// Rys quadrature Fock kernel (per-L_sum, compile-time sized).
// Expected macros: KERNEL_NAME, ERI_NRYS, ERI_MAX_G, ERI_MAX_CART
// No #pragma once — designed for multiple inclusion.
//
// Uses factored HRR: precomputes Ix_full[ax0][ax1][ax2][ax3] for each axis
// before the Cartesian assembly loop, avoiding redundant bra-HRR recomputation.

// Max factored HRR array size per axis: (l_max+1)^4 = 5^4 = 625 for g-shells
#define ERI_MAX_IX 625

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
            // ---- Shell data ----
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
            const int ij_am = l[0] + l[1];
            const int kl_am = l[2] + l[3];
            const int g_stride = kl_am + 1;

            const float rab2 = (RC[0][0] - RC[1][0]) * (RC[0][0] - RC[1][0]) +
                               (RC[0][1] - RC[1][1]) * (RC[0][1] - RC[1][1]) +
                               (RC[0][2] - RC[1][2]) * (RC[0][2] - RC[1][2]);
            const float rcd2 = (RC[2][0] - RC[3][0]) * (RC[2][0] - RC[3][0]) +
                               (RC[2][1] - RC[3][1]) * (RC[2][1] - RC[3][1]) +
                               (RC[2][2] - RC[3][2]) * (RC[2][2] - RC[3][2]);
            const float AB[3] = {RC[0][0] - RC[1][0], RC[0][1] - RC[1][1],
                                 RC[0][2] - RC[1][2]};
            const float CD[3] = {RC[2][0] - RC[3][0], RC[2][1] - RC[3][1],
                                 RC[2][2] - RC[3][2]};

            const int n_cart =
                dim_cart[0] * dim_cart[1] * dim_cart[2] * dim_cart[3];
            float eri_cart[ERI_MAX_CART];
            for (int i = 0; i < n_cart; i++) eri_cart[i] = 0.0f;

            // Strides for Ix_full indexing: Ix[ax0][ax1][ax2][ax3]
            // Each axis index ranges 0..l[i], so stride for axis i = product of
            // (l[j]+1) for j>i
            const int ix_d3 = 1;
            const int ix_d2 = (l[3] + 1);
            const int ix_d1 = (l[2] + 1) * ix_d2;
            const int ix_d0 = (l[1] + 1) * ix_d1;

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
                            const float QC_v[3] = {Qx - RC[2][0], Qy - RC[2][1],
                                                   Qz - RC[2][2]};
                            const float PQ[3] = {Px - Qx, Py - Qy, Pz - Qz};
                            const float rho = p_val * q_val / (p_val + q_val);
                            const float T =
                                rho *
                                (PQ[0] * PQ[0] + PQ[1] * PQ[1] + PQ[2] * PQ[2]);

                            // Rys roots and weights
                            double rys_r[ERI_NRYS], rys_w[ERI_NRYS];
                            rys_roots_weights(ERI_NRYS, (double)T, rys_r,
                                              rys_w);

                            // For each Rys root: VRR → factored HRR →
                            // accumulate
                            for (int ir = 0; ir < ERI_NRYS; ir++)
                            {
                                const float u = (float)rys_r[ir];
                                const float w = (float)rys_w[ir];
                                const float factor = u / (p_val + q_val);
                                const float B00 = 0.5f * factor;
                                const float B10 =
                                    0.5f / p_val * (1.0f - q_val * factor);
                                const float B01 =
                                    0.5f / q_val * (1.0f - p_val * factor);

                                // VRR for each axis
                                float Gx[ERI_MAX_G], Gy[ERI_MAX_G],
                                    Gz[ERI_MAX_G];
                                const float Cx_bra[3] = {
                                    PA[0] - factor * q_val * PQ[0],
                                    PA[1] - factor * q_val * PQ[1],
                                    PA[2] - factor * q_val * PQ[2]};
                                const float Cx_ket[3] = {
                                    QC_v[0] + factor * p_val * PQ[0],
                                    QC_v[1] + factor * p_val * PQ[1],
                                    QC_v[2] + factor * p_val * PQ[2]};

                                rys_vrr_2d(Gx, ij_am, kl_am, g_stride,
                                           Cx_bra[0], Cx_ket[0], B00, B10, B01);
                                rys_vrr_2d(Gy, ij_am, kl_am, g_stride,
                                           Cx_bra[1], Cx_ket[1], B00, B10, B01);
                                rys_vrr_2d(Gz, ij_am, kl_am, g_stride,
                                           Cx_bra[2], Cx_ket[2], B00, B10, B01);

                                // ---- Factored HRR: precompute
                                // Ix_full[ax0][ax1][ax2][ax3] ---- Step 1: Bra
                                // HRR for all (ax, bx) pairs on each axis Step
                                // 2: Ket HRR to get full factored integrals
                                float Ix_full[ERI_MAX_IX], Iy_full[ERI_MAX_IX],
                                    Iz_full[ERI_MAX_IX];

                                // X-axis
                                for (int ax0 = 0; ax0 <= l[0]; ax0++)
                                    for (int ax1 = 0; ax1 <= l[1]; ax1++)
                                    {
                                        // Bra HRR: G[i][j] for i=0..ij_am,
                                        // j=0..kl_am → H[j] for j=0..kl_am
                                        float h_bra[9];  // kl_am max = 8
                                        for (int j = 0; j <= kl_am; j++)
                                        {
                                            float col[9];  // ij_am max = 8
                                            for (int i = 0; i <= ij_am; i++)
                                                col[i] = Gx[i * g_stride + j];
                                            h_bra[j] = rys_hrr_1d(col, ax0, ax1,
                                                                  AB[0]);
                                        }
                                        // Ket HRR: H[j] for j=0..kl_am →
                                        // I[ax2][ax3]
                                        for (int ax2 = 0; ax2 <= l[2]; ax2++)
                                            for (int ax3 = 0; ax3 <= l[3];
                                                 ax3++)
                                                Ix_full[ax0 * ix_d0 +
                                                        ax1 * ix_d1 +
                                                        ax2 * ix_d2 + ax3] =
                                                    rys_hrr_1d(h_bra, ax2, ax3,
                                                               CD[0]);
                                    }

                                // Y-axis
                                for (int ay0 = 0; ay0 <= l[0]; ay0++)
                                    for (int ay1 = 0; ay1 <= l[1]; ay1++)
                                    {
                                        float h_bra[9];
                                        for (int j = 0; j <= kl_am; j++)
                                        {
                                            float col[9];
                                            for (int i = 0; i <= ij_am; i++)
                                                col[i] = Gy[i * g_stride + j];
                                            h_bra[j] = rys_hrr_1d(col, ay0, ay1,
                                                                  AB[1]);
                                        }
                                        for (int ay2 = 0; ay2 <= l[2]; ay2++)
                                            for (int ay3 = 0; ay3 <= l[3];
                                                 ay3++)
                                                Iy_full[ay0 * ix_d0 +
                                                        ay1 * ix_d1 +
                                                        ay2 * ix_d2 + ay3] =
                                                    rys_hrr_1d(h_bra, ay2, ay3,
                                                               CD[1]);
                                    }

                                // Z-axis
                                for (int az0 = 0; az0 <= l[0]; az0++)
                                    for (int az1 = 0; az1 <= l[1]; az1++)
                                    {
                                        float h_bra[9];
                                        for (int j = 0; j <= kl_am; j++)
                                        {
                                            float col[9];
                                            for (int i = 0; i <= ij_am; i++)
                                                col[i] = Gz[i * g_stride + j];
                                            h_bra[j] = rys_hrr_1d(col, az0, az1,
                                                                  AB[2]);
                                        }
                                        for (int az2 = 0; az2 <= l[2]; az2++)
                                            for (int az3 = 0; az3 <= l[3];
                                                 az3++)
                                                Iz_full[az0 * ix_d0 +
                                                        az1 * ix_d1 +
                                                        az2 * ix_d2 + az3] =
                                                    rys_hrr_1d(h_bra, az2, az3,
                                                               CD[2]);
                                    }

                                // ---- Accumulate into eri_cart using factored
                                // Ix*Iy*Iz ----
                                const float wn = n_abcd * w;
                                int idx = 0;
                                for (int c0 = 0; c0 < dim_cart[0]; c0++)
                                {
                                    int ax0, ay0, az0;
                                    QC_Get_Lxyz_Device(l[0], c0, ax0, ay0, az0);
                                    for (int c1 = 0; c1 < dim_cart[1]; c1++)
                                    {
                                        int ax1, ay1, az1;
                                        QC_Get_Lxyz_Device(l[1], c1, ax1, ay1,
                                                           az1);
                                        for (int c2 = 0; c2 < dim_cart[2]; c2++)
                                        {
                                            int ax2, ay2, az2;
                                            QC_Get_Lxyz_Device(l[2], c2, ax2,
                                                               ay2, az2);
                                            for (int c3 = 0; c3 < dim_cart[3];
                                                 c3++)
                                            {
                                                int ax3, ay3, az3;
                                                QC_Get_Lxyz_Device(
                                                    l[3], c3, ax3, ay3, az3);

                                                const int ix_idx =
                                                    ax0 * ix_d0 + ax1 * ix_d1 +
                                                    ax2 * ix_d2 + ax3;
                                                const int iy_idx =
                                                    ay0 * ix_d0 + ay1 * ix_d1 +
                                                    ay2 * ix_d2 + ay3;
                                                const int iz_idx =
                                                    az0 * ix_d0 + az1 * ix_d1 +
                                                    az2 * ix_d2 + az3;

                                                eri_cart[idx] +=
                                                    wn * Ix_full[ix_idx] *
                                                    Iy_full[iy_idx] *
                                                    Iz_full[iz_idx];
                                                idx++;
                                            }
                                        }
                                    }
                                }
                            }  // end Rys roots
                        }
                    }
                }
            }  // end primitives

            // ---- Cart2sph (same as before) ----
            const int n_eff = dim_eff[0] * dim_eff[1] * dim_eff[2] * dim_eff[3];
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

            // ---- Norms + Fock ----
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

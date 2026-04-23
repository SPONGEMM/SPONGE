#pragma once

#include "direct_fock_terms.hpp"

#ifndef USE_GPU
static inline int QC_Count_Active_Partners_By_Bound(
    const std::vector<int>& sorted_pair_ids, const float* shell_pair_bounds,
    const float threshold)
{
    int low = 0;
    int high = (int)sorted_pair_ids.size();
    while (low < high)
    {
        const int mid = low + (high - low) / 2;
        if (shell_pair_bounds[sorted_pair_ids[mid]] >= threshold)
            low = mid + 1;
        else
            high = mid;
    }
    return low;
}

static inline int QC_Count_Active_Partners_By_Activity(
    const std::vector<int>& sorted_pair_ids, const float* pair_activity,
    const float threshold)
{
    int low = 0;
    int high = (int)sorted_pair_ids.size();
    while (low < high)
    {
        const int mid = low + (high - low) / 2;
        if (pair_activity[sorted_pair_ids[mid]] >= threshold)
            low = mid + 1;
        else
            high = mid;
    }
    return low;
}

static inline float QC_Exact_Quartet_Screen_CPU(
    const QC_INTEGRAL_TASKS& task_ctx, const int pair_ij, const int pair_kl,
    const float* shell_pair_bounds, const float* pair_density_coul,
    const float* pair_density_exx_a, const float* pair_density_exx_b,
    const float exx_scale_a, const float exx_scale_b)
{
    const QC_ONE_E_TASK& ij = task_ctx.topo.h_shell_pairs[pair_ij];
    const QC_ONE_E_TASK& kl = task_ctx.topo.h_shell_pairs[pair_kl];
    const int ik_pair = QC_Shell_Pair_Index(ij.x, kl.x);
    const int il_pair = QC_Shell_Pair_Index(ij.x, kl.y);
    const int jk_pair = QC_Shell_Pair_Index(ij.y, kl.x);
    const int jl_pair = QC_Shell_Pair_Index(ij.y, kl.y);

    const float shell_bound =
        shell_pair_bounds[pair_ij] * shell_pair_bounds[pair_kl];
    const float coul_screen = shell_bound * fmaxf(pair_density_coul[pair_ij],
                                                  pair_density_coul[pair_kl]);
    const float exx_screen_a = exx_scale_a == 0.0f
                                   ? 0.0f
                                   : shell_bound * exx_scale_a *
                                         QC_Max4(pair_density_exx_a[ik_pair],
                                                 pair_density_exx_a[il_pair],
                                                 pair_density_exx_a[jk_pair],
                                                 pair_density_exx_a[jl_pair]);
    float exx_screen_b = 0.0f;
    if (pair_density_exx_b != NULL && exx_scale_b != 0.0f)
    {
        exx_screen_b =
            shell_bound * exx_scale_b *
            QC_Max4(pair_density_exx_b[ik_pair], pair_density_exx_b[il_pair],
                    pair_density_exx_b[jk_pair], pair_density_exx_b[jl_pair]);
    }
    return fmaxf(coul_screen, fmaxf(exx_screen_a, exx_screen_b));
}

struct QC_Shell_Pair_Meta_CPU
{
    int sh[2];
    int l[2];
    int np[2];
    int p_exp[2];
    int p_cof[2];
    int dims_cart[2];
    int dims_sph[2];
    int dims_eff[2];
    int off_cart[2];
    int off_eff[2];
    float R[2][3];
    float pair_dist2;
    int comp_x[2][MAX_CART_SHELL];
    int comp_y[2][MAX_CART_SHELL];
    int comp_z[2][MAX_CART_SHELL];
};

struct QC_Bra_Prim_Cache_CPU
{
    float P[3];
    float inv_p;
    float n_ab;
    float E_bra[3][6][6][11];
};

struct QC_Cart_Pair_Geom_CPU
{
    unsigned short c0x, c0y, c0z;
    unsigned short c1x, c1y, c1z;
    unsigned short sumx, sumy, sumz;
    unsigned short shell_offset;
};

struct QC_Generic_Pair_View_CPU
{
    const QC_Angular_Term_CPU* terms;
    int term_count;
    int shell_offset;
};

static inline int QC_Build_Angular_Terms_CPU(
    const float* ex_row, const float* ey_row, const float* ez_row,
    const int max_x, const int max_y, const int max_z, const int apply_phase,
    const int hr_stride_x, const int hr_stride_y, const int hr_stride_z,
    QC_Angular_Term_CPU* terms)
{
    int term_count = 0;
    for (int x = 0; x <= max_x; x++)
    {
        const float ex = ex_row[x];
        if (ex == 0.0f) continue;
        for (int y = 0; y <= max_y; y++)
        {
            const float ey = ey_row[y];
            if (ey == 0.0f) continue;
            const float exy = ex * ey;
            for (int z = 0; z <= max_z; z++)
            {
                const float ez = ez_row[z];
                if (ez == 0.0f) continue;
                float coeff = exy * ez;
                if (apply_phase && ((x + y + z) & 1)) coeff = -coeff;
                terms[term_count].hr_offset =
                    (unsigned short)(x * hr_stride_x + y * hr_stride_y +
                                     z * hr_stride_z);
                terms[term_count].coeff = coeff;
                term_count++;
            }
        }
    }
    return term_count;
}

static inline void QC_Init_Shell_Pair_Meta_CPU(
    const QC_ONE_E_TASK& pair, const int* atm, const int* bas, const float* env,
    const int* ao_offsets_cart, const int* ao_offsets_sph,
    const int is_spherical, QC_Shell_Pair_Meta_CPU& meta)
{
    meta.sh[0] = pair.x;
    meta.sh[1] = pair.y;
    for (int s = 0; s < 2; s++)
    {
        const int sh = meta.sh[s];
        meta.l[s] = bas[sh * 8 + 1];
        meta.np[s] = bas[sh * 8 + 2];
        meta.p_exp[s] = bas[sh * 8 + 5];
        meta.p_cof[s] = bas[sh * 8 + 6];
        meta.dims_cart[s] = (meta.l[s] + 1) * (meta.l[s] + 2) / 2;
        meta.dims_sph[s] = 2 * meta.l[s] + 1;
        meta.dims_eff[s] = QC_Shell_Dim(meta.l[s], is_spherical);
        meta.off_cart[s] = ao_offsets_cart[sh];
        meta.off_eff[s] = is_spherical ? ao_offsets_sph[sh] : meta.off_cart[s];
        const int ptr_R = atm[bas[sh * 8 + 0] * 6 + 1];
        meta.R[s][0] = env[ptr_R + 0];
        meta.R[s][1] = env[ptr_R + 1];
        meta.R[s][2] = env[ptr_R + 2];
        for (int c = 0; c < meta.dims_cart[s]; c++)
        {
            QC_Get_Lxyz_Host(meta.l[s], c, meta.comp_x[s][c], meta.comp_y[s][c],
                             meta.comp_z[s][c]);
        }
    }
    meta.pair_dist2 =
        (meta.R[0][0] - meta.R[1][0]) * (meta.R[0][0] - meta.R[1][0]) +
        (meta.R[0][1] - meta.R[1][1]) * (meta.R[0][1] - meta.R[1][1]) +
        (meta.R[0][2] - meta.R[1][2]) * (meta.R[0][2] - meta.R[1][2]);
}

static inline void QC_Build_Bra_Prim_Cache_CPU(
    const QC_Shell_Pair_Meta_CPU& bra, const float* env,
    const float prim_screen_tol, std::vector<QC_Bra_Prim_Cache_CPU>& prims)
{
    prims.clear();
    prims.reserve((size_t)bra.np[0] * (size_t)bra.np[1]);
    for (int ip = 0; ip < bra.np[0]; ip++)
    {
        for (int jp = 0; jp < bra.np[1]; jp++)
        {
            const float ai = env[bra.p_exp[0] + ip];
            const float aj = env[bra.p_exp[1] + jp];
            const float p = ai + aj;
            const float inv_p = 1.0f / p;
            const float kab = expf(-(ai * aj * inv_p) * bra.pair_dist2);
            const float n_ab =
                env[bra.p_cof[0] + ip] * env[bra.p_cof[1] + jp] * kab;
            if (fabsf(n_ab) < prim_screen_tol) continue;

            QC_Bra_Prim_Cache_CPU prim = {};
            prim.P[0] = (ai * bra.R[0][0] + aj * bra.R[1][0]) * inv_p;
            prim.P[1] = (ai * bra.R[0][1] + aj * bra.R[1][1]) * inv_p;
            prim.P[2] = (ai * bra.R[0][2] + aj * bra.R[1][2]) * inv_p;
            prim.inv_p = inv_p;
            prim.n_ab = n_ab;
            for (int d = 0; d < 3; d++)
            {
                compute_md_coeffs(prim.E_bra[d], bra.l[0], bra.l[1],
                                  prim.P[d] - bra.R[0][d],
                                  prim.P[d] - bra.R[1][d], 0.5f * inv_p);
            }
            prims.push_back(prim);
        }
    }
}

static inline void QC_Cart2Sph_Step_CPU(const float* C, const int nc,
                                        const int ns, const int leading,
                                        const int tail, const float* src,
                                        float* dst)
{
    for (int lead = 0; lead < leading; lead++)
    {
        const float* src_blk = src + lead * nc * tail;
        float* dst_blk = dst + lead * ns * tail;
        memset(dst_blk, 0, (size_t)ns * tail * sizeof(float));
        for (int a = 0; a < nc; a++)
        {
            const float* src_row = src_blk + a * tail;
            for (int p = 0; p < ns; p++)
            {
                const float c = C[a * ns + p];
                if (c == 0.0f) continue;
                float* dst_row = dst_blk + p * tail;
                for (int idx = 0; idx < tail; idx++)
                    dst_row[idx] += c * src_row[idx];
            }
        }
    }
}

static inline void QC_Cart2Sph_Shell_ERI_CPU(
    const float* U, const int nao_s, const int* off_cart, const int* off_sph,
    const int* dims_cart, const int* dims_sph, float* buf0, float* buf1)
{
    float C[4][MAX_CART_SHELL * MAX_CART_SHELL];
    for (int s = 0; s < 4; s++)
        for (int i = 0; i < dims_cart[s]; i++)
            for (int j = 0; j < dims_sph[s]; j++)
                C[s][i * dims_sph[s] + j] =
                    U[(off_cart[s] + i) * nao_s + (off_sph[s] + j)];

    QC_Cart2Sph_Step_CPU(C[0], dims_cart[0], dims_sph[0], 1,
                         dims_cart[1] * dims_cart[2] * dims_cart[3], buf0,
                         buf1);
    QC_Cart2Sph_Step_CPU(C[1], dims_cart[1], dims_sph[1], dims_sph[0],
                         dims_cart[2] * dims_cart[3], buf1, buf0);
    QC_Cart2Sph_Step_CPU(C[2], dims_cart[2], dims_sph[2],
                         dims_sph[0] * dims_sph[1], dims_cart[3], buf0, buf1);
    QC_Cart2Sph_Step_CPU(C[3], dims_cart[3], dims_sph[3],
                         dims_sph[0] * dims_sph[1] * dims_sph[2], 1, buf1,
                         buf0);
}

static inline bool QC_Compute_Shell_Quartet_ERI_Buffer_CPU_BraCached(
    const QC_Shell_Pair_Meta_CPU& bra, const QC_Shell_Pair_Meta_CPU& ket,
    const float* env, const float* norms, const int is_spherical,
    const float* cart2sph_mat, const int nao_sph,
    const std::vector<QC_Bra_Prim_Cache_CPU>& bra_prims, float* HR,
    float* shell_eri, float* shell_tmp, QC_Angular_Term_CPU* bra_terms_buf,
    QC_Angular_Term_CPU* ket_terms_buf, int hr_base, int shell_buf_size,
    float prim_screen_tol, int* dims_eff, int* off_eff)
{
    const int dims_cart[4] = {bra.dims_cart[0], bra.dims_cart[1],
                              ket.dims_cart[0], ket.dims_cart[1]};
    const int dims_sph[4] = {bra.dims_sph[0], bra.dims_sph[1], ket.dims_sph[0],
                             ket.dims_sph[1]};
    const int off_cart[4] = {bra.off_cart[0], bra.off_cart[1], ket.off_cart[0],
                             ket.off_cart[1]};
    const int l[4] = {bra.l[0], bra.l[1], ket.l[0], ket.l[1]};
    dims_eff[0] = bra.dims_eff[0];
    dims_eff[1] = bra.dims_eff[1];
    dims_eff[2] = ket.dims_eff[0];
    dims_eff[3] = ket.dims_eff[1];
    off_eff[0] = bra.off_eff[0];
    off_eff[1] = bra.off_eff[1];
    off_eff[2] = ket.off_eff[0];
    off_eff[3] = ket.off_eff[1];

    const int shell_size =
        dims_cart[0] * dims_cart[1] * dims_cart[2] * dims_cart[3];
    if (shell_size > shell_buf_size) return false;
    for (int i = 0; i < shell_size; i++) shell_eri[i] = 0.0f;
    if (bra_prims.empty()) return true;

    const int shell_stride_k = dims_cart[3];
    const int shell_stride_j = dims_cart[2] * dims_cart[3];
    const int shell_stride_i = dims_cart[1] * dims_cart[2] * dims_cart[3];
    const int hr_stride_z = hr_base;
    const int hr_stride_y = hr_base * hr_base;
    const int hr_stride_x = hr_base * hr_base * hr_base;

    const int bra_pair_count = bra.dims_cart[0] * bra.dims_cart[1];
    const int ket_pair_count = ket.dims_cart[0] * ket.dims_cart[1];

    QC_Cart_Pair_Geom_CPU bra_geom[QC_MAX_CART_PAIR_COUNT_CPU];
    QC_Cart_Pair_Geom_CPU ket_geom[QC_MAX_CART_PAIR_COUNT_CPU];
    for (int i = 0; i < bra.dims_cart[0]; i++)
        for (int j = 0; j < bra.dims_cart[1]; j++)
        {
            const int ij = i * bra.dims_cart[1] + j;
            QC_Cart_Pair_Geom_CPU& g = bra_geom[ij];
            g.c0x = bra.comp_x[0][i];
            g.c0y = bra.comp_y[0][i];
            g.c0z = bra.comp_z[0][i];
            g.c1x = bra.comp_x[1][j];
            g.c1y = bra.comp_y[1][j];
            g.c1z = bra.comp_z[1][j];
            g.sumx = g.c0x + g.c1x;
            g.sumy = g.c0y + g.c1y;
            g.sumz = g.c0z + g.c1z;
            g.shell_offset =
                (unsigned short)(i * shell_stride_i + j * shell_stride_j);
        }
    for (int k = 0; k < ket.dims_cart[0]; k++)
        for (int l_idx = 0; l_idx < ket.dims_cart[1]; l_idx++)
        {
            const int kl = k * ket.dims_cart[1] + l_idx;
            QC_Cart_Pair_Geom_CPU& g = ket_geom[kl];
            g.c0x = ket.comp_x[0][k];
            g.c0y = ket.comp_y[0][k];
            g.c0z = ket.comp_z[0][k];
            g.c1x = ket.comp_x[1][l_idx];
            g.c1y = ket.comp_y[1][l_idx];
            g.c1z = ket.comp_z[1][l_idx];
            g.sumx = g.c0x + g.c1x;
            g.sumy = g.c0y + g.c1y;
            g.sumz = g.c0z + g.c1z;
            g.shell_offset = (unsigned short)(k * shell_stride_k + l_idx);
        }

    const bool low_l_fast_path =
        (bra.l[0] <= 1 && bra.l[1] <= 1 && ket.l[0] <= 1 && ket.l[1] <= 1);

    int bra_term_counts[QC_MAX_CART_PAIR_COUNT_CPU];
    int ket_term_counts[QC_MAX_CART_PAIR_COUNT_CPU];

    float E_ket[3][6][6][11];
    for (const QC_Bra_Prim_Cache_CPU& prim : bra_prims)
    {
        // Build bra term lists (once per bra prim, reused across ket prims)
        if (!low_l_fast_path)
        {
            for (int ij = 0; ij < bra_pair_count; ij++)
            {
                const QC_Cart_Pair_Geom_CPU& g = bra_geom[ij];
                bra_term_counts[ij] = QC_Build_Angular_Terms_CPU(
                    prim.E_bra[0][g.c0x][g.c1x], prim.E_bra[1][g.c0y][g.c1y],
                    prim.E_bra[2][g.c0z][g.c1z], g.sumx, g.sumy, g.sumz, 0,
                    hr_stride_x, hr_stride_y, hr_stride_z,
                    bra_terms_buf + (size_t)ij * QC_MAX_PAIR_TERM_COUNT_CPU);
            }
        }

        const float p = 1.0f / prim.inv_p;
        for (int kp = 0; kp < ket.np[0]; kp++)
        {
            for (int lp = 0; lp < ket.np[1]; lp++)
            {
                const float ak = env[ket.p_exp[0] + kp];
                const float al = env[ket.p_exp[1] + lp];
                const float q = ak + al;
                const float inv_q = 1.0f / q;
                const float kcd = expf(-(ak * al * inv_q) * ket.pair_dist2);
                const float pref = 2.0f * PI_25 / (p * q * sqrtf(p + q));
                const float n_abcd = prim.n_ab * env[ket.p_cof[0] + kp] *
                                     env[ket.p_cof[1] + lp] * kcd * pref;
                if (fabsf(n_abcd) < prim_screen_tol) continue;

                float Q[3] = {(ak * ket.R[0][0] + al * ket.R[1][0]) * inv_q,
                              (ak * ket.R[0][1] + al * ket.R[1][1]) * inv_q,
                              (ak * ket.R[0][2] + al * ket.R[1][2]) * inv_q};
                const float alpha = p * q / (p + q);
                float PQ_val[3] = {prim.P[0] - Q[0], prim.P[1] - Q[1],
                                   prim.P[2] - Q[2]};
                const int L_sum = l[0] + l[1] + l[2] + l[3];
                float t_arg =
                    alpha * (PQ_val[0] * PQ_val[0] + PQ_val[1] * PQ_val[1] +
                             PQ_val[2] * PQ_val[2]);
                compute_hr_tensor(HR, alpha, PQ_val, L_sum, hr_base, t_arg);

                for (int d = 0; d < 3; d++)
                    compute_md_coeffs(E_ket[d], ket.l[0], ket.l[1],
                                      Q[d] - ket.R[0][d], Q[d] - ket.R[1][d],
                                      0.5f * inv_q);

                if (low_l_fast_path)
                {
                    // Low-L path: all l <= 1, at most 9 pairs per side
                    // Build term lists on stack (max 27 terms each)
                    QC_Angular_Term_CPU ll_bra[9][27];
                    QC_Angular_Term_CPU ll_ket[9][27];
                    int ll_bra_tc[9], ll_ket_tc[9];
                    for (int ij = 0; ij < bra_pair_count; ij++)
                    {
                        const QC_Cart_Pair_Geom_CPU& g = bra_geom[ij];
                        ll_bra_tc[ij] = QC_Build_Angular_Terms_CPU(
                            prim.E_bra[0][g.c0x][g.c1x],
                            prim.E_bra[1][g.c0y][g.c1y],
                            prim.E_bra[2][g.c0z][g.c1z], g.sumx, g.sumy, g.sumz,
                            0, hr_stride_x, hr_stride_y, hr_stride_z,
                            ll_bra[ij]);
                    }
                    for (int kl = 0; kl < ket_pair_count; kl++)
                    {
                        const QC_Cart_Pair_Geom_CPU& g = ket_geom[kl];
                        ll_ket_tc[kl] = QC_Build_Angular_Terms_CPU(
                            E_ket[0][g.c0x][g.c1x], E_ket[1][g.c0y][g.c1y],
                            E_ket[2][g.c0z][g.c1z], g.sumx, g.sumy, g.sumz, 1,
                            hr_stride_x, hr_stride_y, hr_stride_z, ll_ket[kl]);
                    }
                    for (int ij = 0; ij < bra_pair_count; ij++)
                    {
                        const int btc = ll_bra_tc[ij];
                        if (btc == 0) continue;
                        const QC_Angular_Term_CPU* bt = ll_bra[ij];
                        float* eri_ij = shell_eri + bra_geom[ij].shell_offset;
                        for (int kl = 0; kl < ket_pair_count; kl++)
                        {
                            const int ktc = ll_ket_tc[kl];
                            if (ktc == 0) continue;
                            const QC_Angular_Term_CPU* kt = ll_ket[kl];
                            float val = 0.0f;
                            for (int b = 0; b < btc; b++)
                            {
                                const float bc = bt[b].coeff;
                                const float* hr_b = HR + (int)bt[b].hr_offset;
                                for (int ki = 0; ki < ktc; ki++)
                                    val += bc * kt[ki].coeff *
                                           hr_b[(int)kt[ki].hr_offset];
                            }
                            const int ko = ket_geom[kl].shell_offset;
                            eri_ij[ko] += val * n_abcd;
                        }
                    }
                }
                else
                {
                    // Generic path: build ket terms, accumulate with 4x unroll
                    for (int kl = 0; kl < ket_pair_count; kl++)
                    {
                        const QC_Cart_Pair_Geom_CPU& g = ket_geom[kl];
                        ket_term_counts[kl] = QC_Build_Angular_Terms_CPU(
                            E_ket[0][g.c0x][g.c1x], E_ket[1][g.c0y][g.c1y],
                            E_ket[2][g.c0z][g.c1z], g.sumx, g.sumy, g.sumz, 1,
                            hr_stride_x, hr_stride_y, hr_stride_z,
                            ket_terms_buf +
                                (size_t)kl * QC_MAX_PAIR_TERM_COUNT_CPU);
                    }

                    for (int ij = 0; ij < bra_pair_count; ij++)
                    {
                        const int btc = bra_term_counts[ij];
                        if (btc == 0) continue;
                        const QC_Angular_Term_CPU* bt =
                            bra_terms_buf +
                            (size_t)ij * QC_MAX_PAIR_TERM_COUNT_CPU;
                        float* eri_ij = shell_eri + bra_geom[ij].shell_offset;
                        for (int kl = 0; kl < ket_pair_count; kl++)
                        {
                            const int ktc = ket_term_counts[kl];
                            if (ktc == 0) continue;
                            const QC_Angular_Term_CPU* kt =
                                ket_terms_buf +
                                (size_t)kl * QC_MAX_PAIR_TERM_COUNT_CPU;
                            // Put smaller count in outer loop
                            const QC_Angular_Term_CPU* outer = bt;
                            const QC_Angular_Term_CPU* inner = kt;
                            int otc = btc, itc = ktc;
                            if (otc > itc)
                            {
                                outer = kt;
                                inner = bt;
                                otc = ktc;
                                itc = btc;
                            }
                            const QC_Angular_Term_CPU* inner_end = inner + itc;
                            const QC_Angular_Term_CPU* inner_u4 =
                                inner + (itc & ~3);
                            float val = 0.0f;
                            for (int oi = 0; oi < otc; oi++)
                            {
                                const float oc = outer[oi].coeff;
                                const float* hr_o =
                                    HR + (int)outer[oi].hr_offset;
                                const QC_Angular_Term_CPU* ip = inner;
                                for (; ip < inner_u4; ip += 4)
                                {
                                    val += oc * ip[0].coeff *
                                           hr_o[(int)ip[0].hr_offset];
                                    val += oc * ip[1].coeff *
                                           hr_o[(int)ip[1].hr_offset];
                                    val += oc * ip[2].coeff *
                                           hr_o[(int)ip[2].hr_offset];
                                    val += oc * ip[3].coeff *
                                           hr_o[(int)ip[3].hr_offset];
                                }
                                for (; ip < inner_end; ip++)
                                    val += oc * ip->coeff *
                                           hr_o[(int)ip->hr_offset];
                            }
                            const int ko = ket_geom[kl].shell_offset;
                            eri_ij[ko] += val * n_abcd;
                        }
                    }
                }
            }
        }
    }

    if (is_spherical)
    {
        QC_Cart2Sph_Shell_ERI_CPU(cart2sph_mat, nao_sph, off_cart, off_eff,
                                  dims_cart, dims_sph, shell_eri, shell_tmp);
    }

    for (int i = 0; i < dims_eff[0]; i++)
    {
        const float ni = norms[off_eff[0] + i];
        for (int j = 0; j < dims_eff[1]; j++)
        {
            const float nj = norms[off_eff[1] + j];
            for (int k = 0; k < dims_eff[2]; k++)
            {
                const float nk = norms[off_eff[2] + k];
                for (int l_idx = 0; l_idx < dims_eff[3]; l_idx++)
                {
                    const float nl = norms[off_eff[3] + l_idx];
                    const int idx = QC_Shell_Buffer_Index(
                        i, j, k, l_idx, dims_eff[1], dims_eff[2], dims_eff[3]);
                    shell_eri[idx] *= ni * nj * nk * nl;
                }
            }
        }
    }
    return true;
}

// Rys quadrature ERI for high-L quartets.
// Replaces McMurchie-Davidson HR-tensor path at L_sum >= 6.
// Uses VRR + HRR per Rys root: I arrays ~625 floats vs HR tensor ~83 521
// floats.

static inline void QC_Fock_VRR_2D(float* G, int ij_max, int kl_max,
                                  int g_stride, float Cx_bra, float Cx_ket,
                                  float B00, float B10, float B01)
{
    G[0] = 1.0f;
    for (int i = 0; i < ij_max; i++)
    {
        float val = Cx_bra * G[i * g_stride];
        if (i > 0) val += (float)i * B10 * G[(i - 1) * g_stride];
        G[(i + 1) * g_stride] = val;
    }
    for (int j = 0; j < kl_max; j++)
        for (int i = 0; i <= ij_max; i++)
        {
            float val = Cx_ket * G[i * g_stride + j];
            if (j > 0) val += (float)j * B01 * G[i * g_stride + (j - 1)];
            if (i > 0) val += (float)i * B00 * G[(i - 1) * g_stride + j];
            G[i * g_stride + (j + 1)] = val;
        }
}

// HRR: distribute total AM to individual shells for ERI.
// Input: G[0..ij_am, 0..kl_am] from VRR (stride g_stride)
// Output: I[a0, a1, a2, a3] for a0 in 0..l[0], a1 in 0..l[1], etc.
static inline void QC_Fock_HRR_Batch(const float* G, int ij_am, int kl_am,
                                     int g_stride, const int* l, float AB_d,
                                     float CD_d, float* __restrict__ I_full,
                                     int d0, int d1, int d2)
{
    const int l0 = l[0], l1 = l[1], l2 = l[2], l3 = l[3];
    const int ij_ext = ij_am + 1;
    const int kl_ext = kl_am + 1;

    const int h_a1_stride = kl_ext;
    const int h_a0_stride = (l1 + 1) * h_a1_stride;
    float h_all[10 * 10 * 10];  // max (l0+1)*(l1+1)*(kl_am+1) = 5*5*9=225

    for (int j = 0; j < kl_ext; j++)
    {
        float work[2][10];
        for (int i = 0; i < ij_ext; i++) work[0][i] = G[i * g_stride + j];

        for (int a0 = 0; a0 <= l0; a0++)
            h_all[a0 * h_a0_stride + 0 * h_a1_stride + j] = work[0][a0];

        int cur = 0;
        for (int b = 0; b < l1; b++)
        {
            int nxt = 1 - cur;
            int n_curr = ij_am - b - 1;
            for (int a = 0; a <= n_curr; a++)
                work[nxt][a] = work[cur][a + 1] + AB_d * work[cur][a];
            int a0_max = std::min(l0, n_curr);
            for (int a0 = 0; a0 <= a0_max; a0++)
                h_all[a0 * h_a0_stride + (b + 1) * h_a1_stride + j] =
                    work[nxt][a0];
            cur = nxt;
        }
    }

    for (int a0 = 0; a0 <= l0; a0++)
        for (int a1 = 0; a1 <= l1; a1++)
        {
            if (a0 + a1 > ij_am) continue;
            const float* h_bra = &h_all[a0 * h_a0_stride + a1 * h_a1_stride];

            for (int a2 = 0; a2 <= l2; a2++)
                I_full[a0 * d0 + a1 * d1 + a2 * d2 + 0] = h_bra[a2];

            if (l3 > 0)
            {
                float work[2][10];
                for (int i = 0; i < kl_ext; i++) work[0][i] = h_bra[i];

                int cur = 0;
                for (int d = 0; d < l3; d++)
                {
                    int nxt = 1 - cur;
                    int n_curr = kl_am - d - 1;
                    for (int a = 0; a <= n_curr; a++)
                        work[nxt][a] = work[cur][a + 1] + CD_d * work[cur][a];
                    int a2_max = std::min(l2, n_curr);
                    for (int a2 = 0; a2 <= a2_max; a2++)
                        I_full[a0 * d0 + a1 * d1 + a2 * d2 + (d + 1)] =
                            work[nxt][a2];
                    cur = nxt;
                }
            }
        }
}

static inline bool QC_Compute_Shell_Quartet_ERI_Rys_CPU(
    const QC_Shell_Pair_Meta_CPU& bra, const QC_Shell_Pair_Meta_CPU& ket,
    const float* env, const float* norms, const int is_spherical,
    const float* cart2sph_mat, const int nao_sph,
    const std::vector<QC_Bra_Prim_Cache_CPU>& bra_prims, float* shell_eri,
    float* shell_tmp, int shell_buf_size, float prim_screen_tol, int* dims_eff,
    int* off_eff)
{
    const int dims_cart[4] = {bra.dims_cart[0], bra.dims_cart[1],
                              ket.dims_cart[0], ket.dims_cart[1]};
    const int dims_sph[4] = {bra.dims_sph[0], bra.dims_sph[1], ket.dims_sph[0],
                             ket.dims_sph[1]};
    const int off_cart[4] = {bra.off_cart[0], bra.off_cart[1], ket.off_cart[0],
                             ket.off_cart[1]};
    const int l[4] = {bra.l[0], bra.l[1], ket.l[0], ket.l[1]};
    dims_eff[0] = bra.dims_eff[0];
    dims_eff[1] = bra.dims_eff[1];
    dims_eff[2] = ket.dims_eff[0];
    dims_eff[3] = ket.dims_eff[1];
    off_eff[0] = bra.off_eff[0];
    off_eff[1] = bra.off_eff[1];
    off_eff[2] = ket.off_eff[0];
    off_eff[3] = ket.off_eff[1];

    const int shell_size =
        dims_cart[0] * dims_cart[1] * dims_cart[2] * dims_cart[3];
    if (shell_size > shell_buf_size) return false;
    memset(shell_eri, 0, (size_t)shell_size * sizeof(float));
    if (bra_prims.empty()) return true;

    const int shell_stride_k = dims_cart[3];
    const int shell_stride_j = dims_cart[2] * dims_cart[3];
    const int shell_stride_i = dims_cart[1] * dims_cart[2] * dims_cart[3];

    const int ij_am = l[0] + l[1], kl_am = l[2] + l[3];
    const int L_sum = ij_am + kl_am;
    const int nrys = (L_sum + 2) / 2;
    const int g_stride = kl_am + 1;

    // I array strides: I[a0][a1][a2][a3]
    const int ix_d2 = l[3] + 1;
    const int ix_d1 = (l[2] + 1) * ix_d2;
    const int ix_d0 = (l[1] + 1) * ix_d1;

    const float AB[3] = {bra.R[0][0] - bra.R[1][0], bra.R[0][1] - bra.R[1][1],
                         bra.R[0][2] - bra.R[1][2]};
    const float CD[3] = {ket.R[0][0] - ket.R[1][0], ket.R[0][1] - ket.R[1][1],
                         ket.R[0][2] - ket.R[1][2]};

    // Working buffers (stack-allocated)
    static constexpr int MAX_NRYS_FOCK = 12;
    static constexpr int MAX_I_SIZE_FOCK = 625;  // (l+1)^4 = 5^4 for g shells
    float all_Ix[MAX_NRYS_FOCK * MAX_I_SIZE_FOCK];
    float all_Iy[MAX_NRYS_FOCK * MAX_I_SIZE_FOCK];
    float all_Iz[MAX_NRYS_FOCK * MAX_I_SIZE_FOCK];
    float Gx[100], Gy[100], Gz[100];  // max (ij_am+1)*(kl_am+1) = 9*9
    const int I_size = (l[0] + 1) * ix_d0;

    for (const QC_Bra_Prim_Cache_CPU& bp : bra_prims)
    {
        const float p = 1.0f / bp.inv_p;
        const float PA[3] = {bp.P[0] - bra.R[0][0], bp.P[1] - bra.R[0][1],
                             bp.P[2] - bra.R[0][2]};

        for (int kp = 0; kp < ket.np[0]; kp++)
        {
            const float ak = env[ket.p_exp[0] + kp];
            for (int lp = 0; lp < ket.np[1]; lp++)
            {
                const float al = env[ket.p_exp[1] + lp];
                const float q = ak + al;
                const float inv_q = 1.0f / q;
                const float kcd = expf(-(ak * al * inv_q) * ket.pair_dist2);
                const float pref = 2.0f * PI_25 / (p * q * sqrtf(p + q));
                const float n_abcd = bp.n_ab * env[ket.p_cof[0] + kp] *
                                     env[ket.p_cof[1] + lp] * kcd * pref;
                if (fabsf(n_abcd) < prim_screen_tol) continue;

                float Q[3], QCv[3], PQ[3];
                for (int d = 0; d < 3; d++)
                {
                    Q[d] = (ak * ket.R[0][d] + al * ket.R[1][d]) * inv_q;
                    QCv[d] = Q[d] - ket.R[0][d];
                    PQ[d] = bp.P[d] - Q[d];
                }
                const float rho = p * q / (p + q);
                const float T =
                    rho * (PQ[0] * PQ[0] + PQ[1] * PQ[1] + PQ[2] * PQ[2]);

                double rys_r[MAX_NRYS_FOCK], rys_w[MAX_NRYS_FOCK];
                rys_roots_weights(nrys, (double)T, rys_r, rys_w);

                // Phase 1: Compute all roots' I arrays
                float wn_all[MAX_NRYS_FOCK];
                for (int ir = 0; ir < nrys; ir++)
                {
                    const float u = (float)rys_r[ir];
                    wn_all[ir] = n_abcd * (float)rys_w[ir];

                    const float factor = u / (p + q);
                    const float B00 = 0.5f * factor;
                    const float B10 = 0.5f / p * (1.0f - q * factor);
                    const float B01 = 0.5f / q * (1.0f - p * factor);

                    const float Cx_bra[3] = {PA[0] - factor * q * PQ[0],
                                             PA[1] - factor * q * PQ[1],
                                             PA[2] - factor * q * PQ[2]};
                    const float Cx_ket[3] = {QCv[0] + factor * p * PQ[0],
                                             QCv[1] + factor * p * PQ[1],
                                             QCv[2] + factor * p * PQ[2]};

                    QC_Fock_VRR_2D(Gx, ij_am, kl_am, g_stride, Cx_bra[0],
                                   Cx_ket[0], B00, B10, B01);
                    QC_Fock_VRR_2D(Gy, ij_am, kl_am, g_stride, Cx_bra[1],
                                   Cx_ket[1], B00, B10, B01);
                    QC_Fock_VRR_2D(Gz, ij_am, kl_am, g_stride, Cx_bra[2],
                                   Cx_ket[2], B00, B10, B01);

                    QC_Fock_HRR_Batch(Gx, ij_am, kl_am, g_stride, l, AB[0],
                                      CD[0], &all_Ix[ir * I_size], ix_d0, ix_d1,
                                      ix_d2);
                    QC_Fock_HRR_Batch(Gy, ij_am, kl_am, g_stride, l, AB[1],
                                      CD[1], &all_Iy[ir * I_size], ix_d0, ix_d1,
                                      ix_d2);
                    QC_Fock_HRR_Batch(Gz, ij_am, kl_am, g_stride, l, AB[2],
                                      CD[2], &all_Iz[ir * I_size], ix_d0, ix_d1,
                                      ix_d2);
                }

                // Phase 2: Assemble shell_eri (one pass over output)
                for (int ci = 0; ci < dims_cart[0]; ci++)
                {
                    const int ixi = bra.comp_x[0][ci];
                    const int iyi = bra.comp_y[0][ci];
                    const int izi = bra.comp_z[0][ci];
                    for (int cj = 0; cj < dims_cart[1]; cj++)
                    {
                        const int ixj = bra.comp_x[1][cj];
                        const int iyj = bra.comp_y[1][cj];
                        const int izj = bra.comp_z[1][cj];
                        float* eri_ij = shell_eri + ci * shell_stride_i +
                                        cj * shell_stride_j;
                        const int bx = ixi * ix_d0 + ixj * ix_d1;
                        const int by = iyi * ix_d0 + iyj * ix_d1;
                        const int bz = izi * ix_d0 + izj * ix_d1;
                        for (int ck = 0; ck < dims_cart[2]; ck++)
                        {
                            const int ixk = ket.comp_x[0][ck];
                            const int iyk = ket.comp_y[0][ck];
                            const int izk = ket.comp_z[0][ck];
                            const int kx = bx + ixk * ix_d2;
                            const int ky = by + iyk * ix_d2;
                            const int kz = bz + izk * ix_d2;
                            for (int cl = 0; cl < dims_cart[3]; cl++)
                            {
                                const int ixl = ket.comp_x[1][cl];
                                const int iyl = ket.comp_y[1][cl];
                                const int izl = ket.comp_z[1][cl];
                                const int ax = kx + ixl;
                                const int ay = ky + iyl;
                                const int az = kz + izl;
                                float val = 0.0f;
                                for (int ir = 0; ir < nrys; ir++)
                                {
                                    const int off = ir * I_size;
                                    val += wn_all[ir] * all_Ix[off + ax] *
                                           all_Iy[off + ay] * all_Iz[off + az];
                                }
                                eri_ij[ck * shell_stride_k + cl] += val;
                            }
                        }
                    }
                }
            }
        }
    }

    if (is_spherical)
    {
        QC_Cart2Sph_Shell_ERI_CPU(cart2sph_mat, nao_sph, off_cart, off_eff,
                                  dims_cart, dims_sph, shell_eri, shell_tmp);
    }

    for (int i = 0; i < dims_eff[0]; i++)
    {
        const float ni = norms[off_eff[0] + i];
        for (int j = 0; j < dims_eff[1]; j++)
        {
            const float nj = norms[off_eff[1] + j];
            for (int k = 0; k < dims_eff[2]; k++)
            {
                const float nk = norms[off_eff[2] + k];
                for (int l_idx = 0; l_idx < dims_eff[3]; l_idx++)
                {
                    const float nl = norms[off_eff[3] + l_idx];
                    const int idx = QC_Shell_Buffer_Index(
                        i, j, k, l_idx, dims_eff[1], dims_eff[2], dims_eff[3]);
                    shell_eri[idx] *= ni * nj * nk * nl;
                }
            }
        }
    }
    return true;
}

// Direct Fock contraction (Rys, no shell_eri buffer).
// For non-unique quartets: pre-contracts density → 2D partial Fock → 2D
// Cart2Sph. Eliminates 4D Cart2Sph (~50x cheaper) and the 4D Fock accumulation
// loop entirely.

// Build Cartesian density: D_cart[ci,cj] = sum_{si,sj}
// C[ci,si]*C[cj,sj]*P[oi+si,oj+sj]*norm[oi+si]*norm[oj+sj]
static inline void QC_Build_Cart_Density_2D(
    const float* P, int nao, const float* norms, const float* C2S, int nao_sph,
    int off_cart_0, int off_cart_1, int off_eff_0, int off_eff_1, int nc0,
    int nc1, int ns0, int ns1, const int* l, int is_spherical, float* D_cart,
    bool symmetrize = true)
{
    if (!is_spherical)
    {
        for (int ci = 0; ci < nc0; ci++)
            for (int cj = 0; cj < nc1; cj++)
            {
                const int p = off_eff_0 + ci, q = off_eff_1 + cj;
                const float Pval = symmetrize
                                       ? (P[p * nao + q] + P[q * nao + p])
                                       : P[p * nao + q];
                D_cart[ci * nc1 + cj] = Pval * norms[p] * norms[q];
            }
        return;
    }
    for (int ci = 0; ci < nc0; ci++)
    {
        for (int cj = 0; cj < nc1; cj++)
        {
            double sum = 0.0;
            for (int si = 0; si < ns0; si++)
            {
                const float c_i =
                    C2S[(off_cart_0 + ci) * nao_sph + (off_eff_0 + si)];
                if (c_i == 0.0f) continue;
                const int p = off_eff_0 + si;
                const double cn_i = (double)c_i * (double)norms[p];
                for (int sj = 0; sj < ns1; sj++)
                {
                    const float c_j =
                        C2S[(off_cart_1 + cj) * nao_sph + (off_eff_1 + sj)];
                    if (c_j == 0.0f) continue;
                    const int q = off_eff_1 + sj;
                    const double Pval =
                        symmetrize ? (double)(P[p * nao + q] + P[q * nao + p])
                                   : (double)P[p * nao + q];
                    sum += cn_i * (double)c_j * (double)norms[q] * Pval;
                }
            }
            D_cart[ci * nc1 + cj] = (float)sum;
        }
    }
}

// Apply 2D Cart2Sph to partial Fock and accumulate into thread-local F (double)
static inline void QC_Cart2Sph_Fock_2D(const float* R_cart, int nc0, int nc1,
                                       int ns0, int ns1, const float* C2S,
                                       int nao_sph, int off_cart_0,
                                       int off_cart_1, int off_eff_0,
                                       int off_eff_1, const float* norms,
                                       double scale, double* F, int nao)
{
    for (int si = 0; si < ns0; si++)
    {
        const int p = off_eff_0 + si;
        const double n_i = (double)norms[p] * scale;
        for (int sj = 0; sj < ns1; sj++)
        {
            const int q = off_eff_1 + sj;
            const double n_ij = n_i * (double)norms[q];
            double val = 0.0;
            for (int ci = 0; ci < nc0; ci++)
            {
                const float c_i =
                    C2S[(off_cart_0 + ci) * nao_sph + (off_eff_0 + si)];
                if (c_i == 0.0f) continue;
                for (int cj = 0; cj < nc1; cj++)
                {
                    const float c_j =
                        C2S[(off_cart_1 + cj) * nao_sph + (off_eff_1 + sj)];
                    if (c_j == 0.0f) continue;
                    val += (double)c_i * (double)c_j *
                           (double)R_cart[ci * nc1 + cj];
                }
            }
            val *= n_ij;
            F[p * nao + q] += val;
            F[q * nao + p] += val;
        }
    }
}

// Non-spherical version: just apply norms and accumulate
static inline void QC_Accum_Fock_2D_NoSph(const float* R_cart, int nc0, int nc1,
                                          int off0, int off1,
                                          const float* norms, double scale,
                                          double* F, int nao)
{
    for (int ci = 0; ci < nc0; ci++)
    {
        const int p = off0 + ci;
        const double n_i = (double)norms[p] * scale;
        for (int cj = 0; cj < nc1; cj++)
        {
            const int q = off1 + cj;
            const double val =
                n_i * (double)norms[q] * (double)R_cart[ci * nc1 + cj];
            F[p * nao + q] += val;
            F[q * nao + p] += val;
        }
    }
}

static inline void QC_Direct_Fock_Rys_CPU(
    const QC_Shell_Pair_Meta_CPU& bra, const QC_Shell_Pair_Meta_CPU& ket,
    const float* env, const std::vector<QC_Bra_Prim_Cache_CPU>& bra_prims,
    const float* norms, const int is_spherical, const float* C2S,
    const int nao_sph, const float* P_coul, const float* P_exx_a,
    const float exx_scale_a, const int nao, float prim_screen_tol, double* F_a,
    float* eri_cart_buf)
{
    const int l[4] = {bra.l[0], bra.l[1], ket.l[0], ket.l[1]};
    const int ni = bra.dims_cart[0], nj = bra.dims_cart[1];
    const int nk = ket.dims_cart[0], nl = ket.dims_cart[1];
    const int nijkl = ni * nj * nk * nl;

    const int ij_am = l[0] + l[1], kl_am = l[2] + l[3];
    const int L_sum = ij_am + kl_am;
    const int nrys = (L_sum + 2) / 2;
    const int g_stride = kl_am + 1;

    const int ix_d2 = l[3] + 1;
    const int ix_d1 = (l[2] + 1) * ix_d2;
    const int ix_d0 = (l[1] + 1) * ix_d1;
    const int I_size = (l[0] + 1) * ix_d0;

    const float AB[3] = {bra.R[0][0] - bra.R[1][0], bra.R[0][1] - bra.R[1][1],
                         bra.R[0][2] - bra.R[1][2]};
    const float CD[3] = {ket.R[0][0] - ket.R[1][0], ket.R[0][1] - ket.R[1][1],
                         ket.R[0][2] - ket.R[1][2]};

    // Phase 1: Rys assembly → Cart ERI buffer (no Cart2Sph)
    const int shell_stride_k = nl;
    const int shell_stride_j = nk * nl;
    const int shell_stride_i = nj * nk * nl;
    memset(eri_cart_buf, 0, (size_t)nijkl * sizeof(float));

    static constexpr int MAX_NRYS_DF = 12;
    static constexpr int MAX_I_SIZE_DF = 625;
    float all_Ix[MAX_NRYS_DF * MAX_I_SIZE_DF];
    float all_Iy[MAX_NRYS_DF * MAX_I_SIZE_DF];
    float all_Iz[MAX_NRYS_DF * MAX_I_SIZE_DF];
    float Gx[100], Gy[100], Gz[100];

    for (const QC_Bra_Prim_Cache_CPU& bp : bra_prims)
    {
        const float p = 1.0f / bp.inv_p;
        const float PA[3] = {bp.P[0] - bra.R[0][0], bp.P[1] - bra.R[0][1],
                             bp.P[2] - bra.R[0][2]};
        for (int kp = 0; kp < ket.np[0]; kp++)
        {
            const float ak = env[ket.p_exp[0] + kp];
            for (int lp = 0; lp < ket.np[1]; lp++)
            {
                const float al = env[ket.p_exp[1] + lp];
                const float q = ak + al;
                const float inv_q = 1.0f / q;
                const float kcd = expf(-(ak * al * inv_q) * ket.pair_dist2);
                const float pref = 2.0f * PI_25 / (p * q * sqrtf(p + q));
                const float n_abcd = bp.n_ab * env[ket.p_cof[0] + kp] *
                                     env[ket.p_cof[1] + lp] * kcd * pref;
                if (fabsf(n_abcd) < prim_screen_tol) continue;

                float Q[3], QCv[3], PQ[3];
                for (int d = 0; d < 3; d++)
                {
                    Q[d] = (ak * ket.R[0][d] + al * ket.R[1][d]) * inv_q;
                    QCv[d] = Q[d] - ket.R[0][d];
                    PQ[d] = bp.P[d] - Q[d];
                }
                const float rho = p * q / (p + q);
                const float T =
                    rho * (PQ[0] * PQ[0] + PQ[1] * PQ[1] + PQ[2] * PQ[2]);

                double rys_r[MAX_NRYS_DF], rys_w[MAX_NRYS_DF];
                rys_roots_weights(nrys, (double)T, rys_r, rys_w);

                float wn_all[MAX_NRYS_DF];
                for (int ir = 0; ir < nrys; ir++)
                {
                    const float u = (float)rys_r[ir];
                    wn_all[ir] = n_abcd * (float)rys_w[ir];
                    const float factor = u / (p + q);
                    const float B00 = 0.5f * factor;
                    const float B10 = 0.5f / p * (1.0f - q * factor);
                    const float B01 = 0.5f / q * (1.0f - p * factor);
                    const float Cx_bra[3] = {PA[0] - factor * q * PQ[0],
                                             PA[1] - factor * q * PQ[1],
                                             PA[2] - factor * q * PQ[2]};
                    const float Cx_ket[3] = {QCv[0] + factor * p * PQ[0],
                                             QCv[1] + factor * p * PQ[1],
                                             QCv[2] + factor * p * PQ[2]};
                    QC_Fock_VRR_2D(Gx, ij_am, kl_am, g_stride, Cx_bra[0],
                                   Cx_ket[0], B00, B10, B01);
                    QC_Fock_VRR_2D(Gy, ij_am, kl_am, g_stride, Cx_bra[1],
                                   Cx_ket[1], B00, B10, B01);
                    QC_Fock_VRR_2D(Gz, ij_am, kl_am, g_stride, Cx_bra[2],
                                   Cx_ket[2], B00, B10, B01);
                    QC_Fock_HRR_Batch(Gx, ij_am, kl_am, g_stride, l, AB[0],
                                      CD[0], &all_Ix[ir * I_size], ix_d0, ix_d1,
                                      ix_d2);
                    QC_Fock_HRR_Batch(Gy, ij_am, kl_am, g_stride, l, AB[1],
                                      CD[1], &all_Iy[ir * I_size], ix_d0, ix_d1,
                                      ix_d2);
                    QC_Fock_HRR_Batch(Gz, ij_am, kl_am, g_stride, l, AB[2],
                                      CD[2], &all_Iz[ir * I_size], ix_d0, ix_d1,
                                      ix_d2);
                }

                // Assembly: accumulate into Cart ERI buffer
                for (int ci = 0; ci < ni; ci++)
                {
                    const int ixi = bra.comp_x[0][ci], iyi = bra.comp_y[0][ci],
                              izi = bra.comp_z[0][ci];
                    for (int cj = 0; cj < nj; cj++)
                    {
                        const int ixj = bra.comp_x[1][cj],
                                  iyj = bra.comp_y[1][cj],
                                  izj = bra.comp_z[1][cj];
                        float* eri_ij = eri_cart_buf + ci * shell_stride_i +
                                        cj * shell_stride_j;
                        const int bx = ixi * ix_d0 + ixj * ix_d1;
                        const int by = iyi * ix_d0 + iyj * ix_d1;
                        const int bz = izi * ix_d0 + izj * ix_d1;
                        for (int ck = 0; ck < nk; ck++)
                        {
                            const int ixk = ket.comp_x[0][ck],
                                      iyk = ket.comp_y[0][ck],
                                      izk = ket.comp_z[0][ck];
                            const int kx = bx + ixk * ix_d2;
                            const int ky = by + iyk * ix_d2;
                            const int kz = bz + izk * ix_d2;
                            for (int cl = 0; cl < nl; cl++)
                            {
                                const int ixl = ket.comp_x[1][cl],
                                          iyl = ket.comp_y[1][cl],
                                          izl = ket.comp_z[1][cl];
                                float val = 0.0f;
                                for (int ir = 0; ir < nrys; ir++)
                                {
                                    const int off = ir * I_size;
                                    val += wn_all[ir] * all_Ix[off + kx + ixl] *
                                           all_Iy[off + ky + iyl] *
                                           all_Iz[off + kz + izl];
                                }
                                eri_ij[ck * shell_stride_k + cl] += val;
                            }
                        }
                    }
                }
            }
        }
    }

    // Phase 2: Pre-compute Cartesian density matrices
    float D_J_kl[MAX_CART_SHELL * MAX_CART_SHELL];
    float D_J_ij[MAX_CART_SHELL * MAX_CART_SHELL];

    QC_Build_Cart_Density_2D(P_coul, nao, norms, C2S, nao_sph, bra.off_cart[0],
                             bra.off_cart[1], bra.off_eff[0], bra.off_eff[1],
                             ni, nj, bra.dims_eff[0], bra.dims_eff[1], l,
                             is_spherical, D_J_ij);
    QC_Build_Cart_Density_2D(P_coul, nao, norms, C2S, nao_sph, ket.off_cart[0],
                             ket.off_cart[1], ket.off_eff[0], ket.off_eff[1],
                             nk, nl, ket.dims_eff[0], ket.dims_eff[1], l + 2,
                             is_spherical, D_J_kl);

    float D_K_jl[MAX_CART_SHELL * MAX_CART_SHELL];
    float D_K_jk[MAX_CART_SHELL * MAX_CART_SHELL];
    float D_K_il[MAX_CART_SHELL * MAX_CART_SHELL];
    float D_K_ik[MAX_CART_SHELL * MAX_CART_SHELL];
    const bool need_exx = (exx_scale_a != 0.0f);
    if (need_exx)
    {
        QC_Build_Cart_Density_2D(
            P_exx_a, nao, norms, C2S, nao_sph, bra.off_cart[1], ket.off_cart[1],
            bra.off_eff[1], ket.off_eff[1], nj, nl, bra.dims_eff[1],
            ket.dims_eff[1], l + 1, is_spherical, D_K_jl, false);
        QC_Build_Cart_Density_2D(
            P_exx_a, nao, norms, C2S, nao_sph, bra.off_cart[1], ket.off_cart[0],
            bra.off_eff[1], ket.off_eff[0], nj, nk, bra.dims_eff[1],
            ket.dims_eff[0], l + 1, is_spherical, D_K_jk, false);
        QC_Build_Cart_Density_2D(
            P_exx_a, nao, norms, C2S, nao_sph, bra.off_cart[0], ket.off_cart[1],
            bra.off_eff[0], ket.off_eff[1], ni, nl, bra.dims_eff[0],
            ket.dims_eff[1], l, is_spherical, D_K_il, false);
        QC_Build_Cart_Density_2D(
            P_exx_a, nao, norms, C2S, nao_sph, bra.off_cart[0], ket.off_cart[0],
            bra.off_eff[0], ket.off_eff[0], ni, nk, bra.dims_eff[0],
            ket.dims_eff[0], l, is_spherical, D_K_ik, false);
    }

    // Phase 3: Contract Cart ERI with Cart density → Cart partial Fock
    double R_J_bra[MAX_CART_SHELL * MAX_CART_SHELL] = {};
    double R_J_ket[MAX_CART_SHELL * MAX_CART_SHELL] = {};
    double R_K_ik[MAX_CART_SHELL * MAX_CART_SHELL] = {};
    double R_K_il[MAX_CART_SHELL * MAX_CART_SHELL] = {};
    double R_K_jk[MAX_CART_SHELL * MAX_CART_SHELL] = {};
    double R_K_jl[MAX_CART_SHELL * MAX_CART_SHELL] = {};

    for (int ci = 0; ci < ni; ci++)
    {
        for (int cj = 0; cj < nj; cj++)
        {
            const float* eri_ij =
                eri_cart_buf + ci * shell_stride_i + cj * shell_stride_j;
            const float d_ij = D_J_ij[ci * nj + cj];
            double r_j_bra = 0.0;

            for (int ck = 0; ck < nk; ck++)
            {
                for (int cl = 0; cl < nl; cl++)
                {
                    const float val = eri_ij[ck * nl + cl];
                    if (val == 0.0f) continue;
                    const double v = (double)val;

                    r_j_bra += v * (double)D_J_kl[ck * nl + cl];
                    R_J_ket[ck * nl + cl] += v * (double)d_ij;

                    if (need_exx)
                    {
                        const double nv = -(double)exx_scale_a * v;
                        R_K_ik[ci * nk + ck] +=
                            nv * (double)D_K_jl[cj * nl + cl];
                        R_K_il[ci * nl + cl] +=
                            nv * (double)D_K_jk[cj * nk + ck];
                        R_K_jk[cj * nk + ck] +=
                            nv * (double)D_K_il[ci * nl + cl];
                        R_K_jl[cj * nl + cl] +=
                            nv * (double)D_K_ik[ci * nk + ck];
                    }
                }
            }
            R_J_bra[ci * nj + cj] += r_j_bra;
        }
    }

    // Phase 4: 2D Cart2Sph + accumulate into F
    if (is_spherical)
    {
        float R_f[MAX_CART_SHELL * MAX_CART_SHELL];
        auto accum_2d = [&](const double* R, int nc0, int nc1, int oc0, int oc1,
                            int oe0, int oe1, int ns0, int ns1)
        {
            for (int i = 0; i < nc0 * nc1; i++) R_f[i] = (float)R[i];
            QC_Cart2Sph_Fock_2D(R_f, nc0, nc1, ns0, ns1, C2S, nao_sph, oc0, oc1,
                                oe0, oe1, norms, 1.0, F_a, nao);
        };

        accum_2d(R_J_bra, ni, nj, bra.off_cart[0], bra.off_cart[1],
                 bra.off_eff[0], bra.off_eff[1], bra.dims_eff[0],
                 bra.dims_eff[1]);
        accum_2d(R_J_ket, nk, nl, ket.off_cart[0], ket.off_cart[1],
                 ket.off_eff[0], ket.off_eff[1], ket.dims_eff[0],
                 ket.dims_eff[1]);
        if (need_exx)
        {
            accum_2d(R_K_ik, ni, nk, bra.off_cart[0], ket.off_cart[0],
                     bra.off_eff[0], ket.off_eff[0], bra.dims_eff[0],
                     ket.dims_eff[0]);
            accum_2d(R_K_il, ni, nl, bra.off_cart[0], ket.off_cart[1],
                     bra.off_eff[0], ket.off_eff[1], bra.dims_eff[0],
                     ket.dims_eff[1]);
            accum_2d(R_K_jk, nj, nk, bra.off_cart[1], ket.off_cart[0],
                     bra.off_eff[1], ket.off_eff[0], bra.dims_eff[1],
                     ket.dims_eff[0]);
            accum_2d(R_K_jl, nj, nl, bra.off_cart[1], ket.off_cart[1],
                     bra.off_eff[1], ket.off_eff[1], bra.dims_eff[1],
                     ket.dims_eff[1]);
        }
    }
    else
    {
        auto accum_nosph =
            [&](const double* R, int nc0, int nc1, int off0, int off1)
        {
            float R_f[MAX_CART_SHELL * MAX_CART_SHELL];
            for (int i = 0; i < nc0 * nc1; i++) R_f[i] = (float)R[i];
            QC_Accum_Fock_2D_NoSph(R_f, nc0, nc1, off0, off1, norms, 1.0, F_a,
                                   nao);
        };
        accum_nosph(R_J_bra, ni, nj, bra.off_eff[0], bra.off_eff[1]);
        accum_nosph(R_J_ket, nk, nl, ket.off_eff[0], ket.off_eff[1]);
        if (need_exx)
        {
            accum_nosph(R_K_ik, ni, nk, bra.off_eff[0], ket.off_eff[0]);
            accum_nosph(R_K_il, ni, nl, bra.off_eff[0], ket.off_eff[1]);
            accum_nosph(R_K_jk, nj, nk, bra.off_eff[1], ket.off_eff[0]);
            accum_nosph(R_K_jl, nj, nl, bra.off_eff[1], ket.off_eff[1]);
        }
    }
}

static inline void QC_Accumulate_Fock_Unique_Quartet_Double(
    const int p, const int q, const int r, const int s, const float value,
    const int nao, const float* P_coul, const float* P_exx_a,
    const float* P_exx_b, const float exx_scale_a, const float exx_scale_b,
    double* F_a, double* F_b)
{
    const int j_terms[8][4] = {{p, q, r, s}, {q, p, r, s}, {p, q, s, r},
                               {q, p, s, r}, {r, s, p, q}, {s, r, p, q},
                               {r, s, q, p}, {s, r, q, p}};
    for (int n = 0; n < 8; n++)
    {
        bool duplicate = false;
        for (int prev = 0; prev < n; prev++)
            if (QC_Same_Ordered_Fock_Term(j_terms[n], j_terms[prev]))
            {
                duplicate = true;
                break;
            }
        if (duplicate) continue;
        const int i = j_terms[n][0], j = j_terms[n][1];
        const int k = j_terms[n][2], l = j_terms[n][3];
        const double j_val = (double)P_coul[k * nao + l] * (double)value;
        F_a[i * nao + j] += j_val;
        if (F_b != NULL) F_b[i * nao + j] += j_val;
    }
    const int k_terms[8][4] = {{p, r, q, s}, {p, s, q, r}, {q, r, p, s},
                               {q, s, p, r}, {r, p, s, q}, {r, q, s, p},
                               {s, p, r, q}, {s, q, r, p}};
    for (int n = 0; n < 8; n++)
    {
        bool duplicate = false;
        for (int prev = 0; prev < n; prev++)
            if (QC_Same_Ordered_Fock_Term(k_terms[n], k_terms[prev]))
            {
                duplicate = true;
                break;
            }
        if (duplicate) continue;
        const int i = k_terms[n][0], j = k_terms[n][1];
        const int k = k_terms[n][2], l = k_terms[n][3];
        if (exx_scale_a != 0.0f)
        {
            const double exx_a = -(double)exx_scale_a *
                                 (double)P_exx_a[k * nao + l] * (double)value;
            F_a[i * nao + j] += exx_a;
        }
        if (F_b != NULL && P_exx_b != NULL && exx_scale_b != 0.0f)
        {
            const double exx_b = -(double)exx_scale_b *
                                 (double)P_exx_b[k * nao + l] * (double)value;
            F_b[i * nao + j] += exx_b;
        }
    }
}

static inline void QC_Build_Fock_Direct_CPU(
    const QC_INTEGRAL_TASKS& task_ctx, const int nbas, const int* atm,
    const int* bas, const float* env, const int* ao_offsets_cart,
    const int* ao_offsets_sph, const float* norms,
    const float* shell_pair_bounds, const float* pair_density_coul,
    const float* pair_density_exx_a, const float* pair_density_exx_b,
    const float shell_screen_tol, const float* P_coul, const float* P_exx_a,
    const float* P_exx_b, const float exx_scale_a, const float exx_scale_b,
    const int nao, const int nao_sph, const int is_spherical,
    const float* cart2sph_mat, double* F_a, double* F_b, float* global_hr_pool,
    QC_Angular_Term_CPU* global_bra_terms,
    QC_Angular_Term_CPU* global_ket_terms, int hr_base, int hr_size,
    int shell_buf_size, float prim_screen_tol, const int fock_thread_count)
{
    const int n_pairs = task_ctx.topo.n_shell_pairs;
    if (n_pairs <= 0) return;

    std::vector<QC_Shell_Pair_Meta_CPU> pair_meta((size_t)n_pairs);
    for (int pair_id = 0; pair_id < n_pairs; pair_id++)
    {
        QC_Init_Shell_Pair_Meta_CPU(task_ctx.topo.h_shell_pairs[pair_id], atm,
                                    bas, env, ao_offsets_cart, ao_offsets_sph,
                                    is_spherical, pair_meta[(size_t)pair_id]);
    }

    std::vector<float> shell_max_exx_a((size_t)nbas, 0.0f);
    std::vector<float> shell_max_exx_b((size_t)nbas, 0.0f);
    for (int pair_id = 0; pair_id < n_pairs; pair_id++)
    {
        const QC_ONE_E_TASK& pair = task_ctx.topo.h_shell_pairs[pair_id];
        const float exx_a = pair_density_exx_a[pair_id];
        shell_max_exx_a[(size_t)pair.x] =
            fmaxf(shell_max_exx_a[(size_t)pair.x], exx_a);
        shell_max_exx_a[(size_t)pair.y] =
            fmaxf(shell_max_exx_a[(size_t)pair.y], exx_a);
        if (pair_density_exx_b != NULL)
        {
            const float exx_b = pair_density_exx_b[pair_id];
            shell_max_exx_b[(size_t)pair.x] =
                fmaxf(shell_max_exx_b[(size_t)pair.x], exx_b);
            shell_max_exx_b[(size_t)pair.y] =
                fmaxf(shell_max_exx_b[(size_t)pair.y], exx_b);
        }
    }

    std::vector<float> anchor_activity((size_t)n_pairs, 0.0f);
    std::vector<int> sorted_pair_ids((size_t)n_pairs, 0);
    for (int pair_id = 0; pair_id < n_pairs; pair_id++)
    {
        const QC_ONE_E_TASK& pair = task_ctx.topo.h_shell_pairs[pair_id];
        const float exx_anchor_a =
            exx_scale_a == 0.0f
                ? 0.0f
                : exx_scale_a * fmaxf(shell_max_exx_a[(size_t)pair.x],
                                      shell_max_exx_a[(size_t)pair.y]);
        const float exx_anchor_b =
            (pair_density_exx_b == NULL || exx_scale_b == 0.0f)
                ? 0.0f
                : exx_scale_b * fmaxf(shell_max_exx_b[(size_t)pair.x],
                                      shell_max_exx_b[(size_t)pair.y]);
        anchor_activity[(size_t)pair_id] =
            shell_pair_bounds[pair_id] * QC_Max4(pair_density_coul[pair_id],
                                                 exx_anchor_a, exx_anchor_b,
                                                 0.0f);
        sorted_pair_ids[(size_t)pair_id] = pair_id;
    }
    std::sort(sorted_pair_ids.begin(), sorted_pair_ids.end(),
              [shell_pair_bounds](const int lhs, const int rhs)
              { return shell_pair_bounds[lhs] > shell_pair_bounds[rhs]; });
    std::vector<int> sorted_activity_ids = sorted_pair_ids;
    std::sort(sorted_activity_ids.begin(), sorted_activity_ids.end(),
              [&anchor_activity](const int lhs, const int rhs)
              {
                  return anchor_activity[(size_t)lhs] >
                         anchor_activity[(size_t)rhs];
              });

    const float max_bound = shell_pair_bounds[sorted_pair_ids.front()];
    const float max_activity =
        anchor_activity[(size_t)sorted_activity_ids.front()];
    const int nao2 = nao * nao;

#pragma omp parallel num_threads(fock_thread_count)
    {
        const int tid = omp_get_thread_num();
        double* F_a_accum = F_a + (size_t)tid * (size_t)nao2;
        double* F_b_accum =
            (F_b != NULL) ? (F_b + (size_t)tid * (size_t)nao2) : NULL;
        float* task_pool = global_hr_pool +
                           (size_t)tid * (size_t)(hr_size + 2 * shell_buf_size);
        float* HR = task_pool;
        float* shell_eri = task_pool + hr_size;
        float* shell_tmp = shell_eri + shell_buf_size;
        QC_Angular_Term_CPU* bra_terms_buf =
            global_bra_terms + (size_t)tid * QC_MAX_CART_PAIR_COUNT_CPU *
                                   QC_MAX_PAIR_TERM_COUNT_CPU;
        QC_Angular_Term_CPU* ket_terms_buf =
            global_ket_terms + (size_t)tid * QC_MAX_CART_PAIR_COUNT_CPU *
                                   QC_MAX_PAIR_TERM_COUNT_CPU;
        std::vector<int> partner_marks((size_t)n_pairs, -1);
        std::vector<int> candidate_partners;
        candidate_partners.reserve(256);
        std::vector<QC_Bra_Prim_Cache_CPU> bra_prims;

#pragma omp for schedule(dynamic)
        for (int pair_ij = 0; pair_ij < n_pairs; pair_ij++)
        {
            const QC_Shell_Pair_Meta_CPU& bra_meta = pair_meta[(size_t)pair_ij];
            const float activity_ij = anchor_activity[(size_t)pair_ij];
            if (fmaxf(activity_ij * max_bound,
                      shell_pair_bounds[pair_ij] * max_activity) <
                shell_screen_tol)
                continue;

            candidate_partners.clear();
            const int stamp = pair_ij;

            if (activity_ij > 0.0f)
            {
                const float bound_threshold = shell_screen_tol / activity_ij;
                const int bound_count = QC_Count_Active_Partners_By_Bound(
                    sorted_pair_ids, shell_pair_bounds, bound_threshold);
                for (int rank = 0; rank < bound_count; rank++)
                {
                    const int pair_kl = sorted_pair_ids[(size_t)rank];
                    if (pair_kl > pair_ij ||
                        partner_marks[(size_t)pair_kl] == stamp)
                        continue;
                    partner_marks[(size_t)pair_kl] = stamp;
                    candidate_partners.push_back(pair_kl);
                }
            }

            const float activity_threshold =
                shell_screen_tol / shell_pair_bounds[pair_ij];
            const int activity_count = QC_Count_Active_Partners_By_Activity(
                sorted_activity_ids, anchor_activity.data(),
                activity_threshold);
            for (int rank = 0; rank < activity_count; rank++)
            {
                const int pair_kl = sorted_activity_ids[(size_t)rank];
                if (pair_kl > pair_ij ||
                    partner_marks[(size_t)pair_kl] == stamp)
                    continue;
                partner_marks[(size_t)pair_kl] = stamp;
                candidate_partners.push_back(pair_kl);
            }
            QC_Build_Bra_Prim_Cache_CPU(bra_meta, env, prim_screen_tol,
                                        bra_prims);
            if (bra_prims.empty()) continue;

            for (const int pair_kl : candidate_partners)
            {
                const float exact_screen = QC_Exact_Quartet_Screen_CPU(
                    task_ctx, pair_ij, pair_kl, shell_pair_bounds,
                    pair_density_coul, pair_density_exx_a, pair_density_exx_b,
                    exx_scale_a, exx_scale_b);
                if (exact_screen < shell_screen_tol) continue;

                const QC_ONE_E_TASK& ij = task_ctx.topo.h_shell_pairs[pair_ij];
                const QC_ONE_E_TASK& kl = task_ctx.topo.h_shell_pairs[pair_kl];
                const QC_Shell_Pair_Meta_CPU& ket_meta =
                    pair_meta[(size_t)pair_kl];
                const int L_sum_q = bra_meta.l[0] + bra_meta.l[1] +
                                    ket_meta.l[0] + ket_meta.l[1];
                const bool use_rys = (L_sum_q >= 4);

                const bool jk_same_bra = (ij.x == ij.y);
                const bool jk_same_ket = (kl.x == kl.y);
                const bool jk_same_braket = (ij.x == kl.x && ij.y == kl.y);

                // Compute shell ERI buffer then accumulate into Fock
                int dims_eff[4];
                int off_eff[4];
                const bool eri_ok =
                    use_rys ? QC_Compute_Shell_Quartet_ERI_Rys_CPU(
                                  bra_meta, ket_meta, env, norms, is_spherical,
                                  cart2sph_mat, nao_sph, bra_prims, shell_eri,
                                  shell_tmp, shell_buf_size, prim_screen_tol,
                                  dims_eff, off_eff)
                            : QC_Compute_Shell_Quartet_ERI_Buffer_CPU_BraCached(
                                  bra_meta, ket_meta, env, norms, is_spherical,
                                  cart2sph_mat, nao_sph, bra_prims, HR,
                                  shell_eri, shell_tmp, bra_terms_buf,
                                  ket_terms_buf, hr_base, shell_buf_size,
                                  prim_screen_tol, dims_eff, off_eff);
                if (!eri_ok) continue;

                if (!jk_same_bra && !jk_same_ket && !jk_same_braket)
                {
                    for (int i = 0; i < dims_eff[0]; i++)
                    {
                        const int p = off_eff[0] + i;
                        const int pn = p * nao;
                        for (int j = 0; j < dims_eff[1]; j++)
                        {
                            const int q = off_eff[1] + j;
                            const int qn = q * nao;
                            const float Ppq_sym =
                                P_coul[pn + q] + P_coul[qn + p];
                            for (int k = 0; k < dims_eff[2]; k++)
                            {
                                const int r = off_eff[2] + k;
                                const int rn = r * nao;
                                for (int l_idx = 0; l_idx < dims_eff[3];
                                     l_idx++)
                                {
                                    const int s = off_eff[3] + l_idx;
                                    const float val =
                                        shell_eri[QC_Shell_Buffer_Index(
                                            i, j, k, l_idx, dims_eff[1],
                                            dims_eff[2], dims_eff[3])];
                                    if (val == 0.0f) continue;
                                    const int sn = s * nao;
                                    const float j_pq =
                                        (P_coul[rn + s] + P_coul[sn + r]) * val;
                                    F_a_accum[pn + q] += j_pq;
                                    F_a_accum[qn + p] += j_pq;
                                    const float j_rs = Ppq_sym * val;
                                    F_a_accum[rn + s] += j_rs;
                                    F_a_accum[sn + r] += j_rs;
                                    if (F_b_accum != NULL)
                                    {
                                        F_b_accum[pn + q] += j_pq;
                                        F_b_accum[qn + p] += j_pq;
                                        F_b_accum[rn + s] += j_rs;
                                        F_b_accum[sn + r] += j_rs;
                                    }
                                    if (exx_scale_a != 0.0f)
                                    {
                                        const float nsv = -exx_scale_a * val;
                                        const float k1 = nsv * P_exx_a[qn + s];
                                        const float k2 = nsv * P_exx_a[qn + r];
                                        const float k3 = nsv * P_exx_a[pn + s];
                                        const float k4 = nsv * P_exx_a[pn + r];
                                        F_a_accum[pn + r] += k1;
                                        F_a_accum[rn + p] += k1;
                                        F_a_accum[pn + s] += k2;
                                        F_a_accum[sn + p] += k2;
                                        F_a_accum[qn + r] += k3;
                                        F_a_accum[rn + q] += k3;
                                        F_a_accum[qn + s] += k4;
                                        F_a_accum[sn + q] += k4;
                                    }
                                    if (F_b_accum != NULL && P_exx_b != NULL &&
                                        exx_scale_b != 0.0f)
                                    {
                                        const float nsv = -exx_scale_b * val;
                                        const float k1 = nsv * P_exx_b[qn + s];
                                        const float k2 = nsv * P_exx_b[qn + r];
                                        const float k3 = nsv * P_exx_b[pn + s];
                                        const float k4 = nsv * P_exx_b[pn + r];
                                        F_b_accum[pn + r] += k1;
                                        F_b_accum[rn + p] += k1;
                                        F_b_accum[pn + s] += k2;
                                        F_b_accum[sn + p] += k2;
                                        F_b_accum[qn + r] += k3;
                                        F_b_accum[rn + q] += k3;
                                        F_b_accum[qn + s] += k4;
                                        F_b_accum[sn + q] += k4;
                                    }
                                }
                            }
                        }
                    }
                }
                else
                {
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
                                for (int l_idx = 0; l_idx < dims_eff[3];
                                     l_idx++)
                                {
                                    const int s = off_eff[3] + l_idx;
                                    if (jk_same_ket && s > r) continue;
                                    if (jk_same_braket)
                                    {
                                        const int pq_pair =
                                            QC_AO_Pair_Index(p, q);
                                        const int rs_pair =
                                            QC_AO_Pair_Index(r, s);
                                        if (rs_pair > pq_pair) continue;
                                    }
                                    const float val =
                                        shell_eri[QC_Shell_Buffer_Index(
                                            i, j, k, l_idx, dims_eff[1],
                                            dims_eff[2], dims_eff[3])];
                                    if (val == 0.0f) continue;
                                    QC_Accumulate_Fock_Unique_Quartet_Double(
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

static __global__ void QC_Reduce_Thread_Fock_Kernel(const int total,
                                                    const int n_threads,
                                                    const double* F_thread,
                                                    float* F_out,
                                                    double* F_out_double)
{
    SIMPLE_DEVICE_FOR(idx, total)
    {
        double sum = (double)F_out[idx];
        for (int tid = 0; tid < n_threads; tid++)
            sum += F_thread[(size_t)tid * (size_t)total + (size_t)idx];
        F_out[idx] = (float)sum;
        if (F_out_double != NULL) F_out_double[idx] = sum;
    }
}

// Incremental Fock reduce: accumulate thread ΔF_eri into F_eri_accum,
// then build final Fock = H_core(+Vxc) + F_eri_accum
static __global__ void QC_Reduce_Thread_Fock_Incremental_Kernel(
    const int total, const int n_threads, const double* F_thread, float* F_out,
    double* F_out_double, double* F_eri_accum)
{
    SIMPLE_DEVICE_FOR(idx, total)
    {
        double delta = 0.0;
        for (int tid = 0; tid < n_threads; tid++)
            delta += F_thread[(size_t)tid * (size_t)total + (size_t)idx];
        F_eri_accum[idx] += delta;
        double sum = (double)F_out[idx] + F_eri_accum[idx];
        F_out[idx] = (float)sum;
        if (F_out_double != NULL) F_out_double[idx] = sum;
    }
}
#endif

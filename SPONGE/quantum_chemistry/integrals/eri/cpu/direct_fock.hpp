#pragma once

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
    const QC_ONE_E_TASK& ij = task_ctx.h_shell_pairs[pair_ij];
    const QC_ONE_E_TASK& kl = task_ctx.h_shell_pairs[pair_kl];
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
    float E_bra[3][5][5][9];
};

struct QC_Angular_Term_CPU
{
    unsigned short hr_offset;
    float coeff;
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

constexpr int QC_MAX_PAIR_TERM_COUNT_CPU = 9 * 9 * 9;
constexpr int QC_MAX_CART_PAIR_COUNT_CPU = MAX_CART_SHELL * MAX_CART_SHELL;

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
    float* shell_eri, float* shell_tmp, int hr_base, int shell_buf_size,
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

    QC_Angular_Term_CPU
        bra_terms_buf[QC_MAX_CART_PAIR_COUNT_CPU * QC_MAX_PAIR_TERM_COUNT_CPU];
    int bra_term_counts[QC_MAX_CART_PAIR_COUNT_CPU];
    QC_Angular_Term_CPU
        ket_terms_buf[QC_MAX_CART_PAIR_COUNT_CPU * QC_MAX_PAIR_TERM_COUNT_CPU];
    int ket_term_counts[QC_MAX_CART_PAIR_COUNT_CPU];

    float E_ket[3][5][5][9];
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
    int hr_base, int hr_size, int shell_buf_size, float prim_screen_tol,
    const int fock_thread_count)
{
    const int n_pairs = task_ctx.n_shell_pairs;
    if (n_pairs <= 0) return;

    std::vector<QC_Shell_Pair_Meta_CPU> pair_meta((size_t)n_pairs);
    for (int pair_id = 0; pair_id < n_pairs; pair_id++)
    {
        QC_Init_Shell_Pair_Meta_CPU(task_ctx.h_shell_pairs[pair_id], atm, bas,
                                    env, ao_offsets_cart, ao_offsets_sph,
                                    is_spherical, pair_meta[(size_t)pair_id]);
    }

    std::vector<float> shell_max_exx_a((size_t)nbas, 0.0f);
    std::vector<float> shell_max_exx_b((size_t)nbas, 0.0f);
    for (int pair_id = 0; pair_id < n_pairs; pair_id++)
    {
        const QC_ONE_E_TASK& pair = task_ctx.h_shell_pairs[pair_id];
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
        const QC_ONE_E_TASK& pair = task_ctx.h_shell_pairs[pair_id];
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

                const QC_ONE_E_TASK& ij = task_ctx.h_shell_pairs[pair_ij];
                const QC_ONE_E_TASK& kl = task_ctx.h_shell_pairs[pair_kl];
                const QC_Shell_Pair_Meta_CPU& ket_meta =
                    pair_meta[(size_t)pair_kl];
                int dims_eff[4];
                int off_eff[4];
                const bool eri_ok =
                    QC_Compute_Shell_Quartet_ERI_Buffer_CPU_BraCached(
                        bra_meta, ket_meta, env, norms, is_spherical,
                        cart2sph_mat, nao_sph, bra_prims, HR, shell_eri,
                        shell_tmp, hr_base, shell_buf_size, prim_screen_tol,
                        dims_eff, off_eff);
                if (!eri_ok) continue;

                const bool jk_same_bra = (ij.x == ij.y);
                const bool jk_same_ket = (kl.x == kl.y);
                const bool jk_same_braket = (ij.x == kl.x && ij.y == kl.y);
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
#endif

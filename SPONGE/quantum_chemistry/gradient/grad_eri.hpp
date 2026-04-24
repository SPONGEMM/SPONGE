#pragma once

#include <cstdlib>
#include <cstring>
#include <vector>

// 双电子积分梯度 (Rys quadrature)
// dE_2e/dR_A = Σ_{pqrs} Γ_eff(pqrs) × d(pq|rs)/dR_A
// Γ_eff = 4·P_pq·P_rs − exx·(P_pr·P_qs + P_ps·P_qr)
// d(pq|rs)/dA_x = 2αi·((p+1)q|rs) − p_x·((p−1)q|rs)
// 使用 Rys quadrature + VRR + 因式分解 HRR 计算积分及其导数。
// 优化: 预计算 Cartesian 有效密度 (gamma_cart)，在组装循环中直接收缩，
// 消除 d_buf 中间数组和 Cart2Sph 变换。
// 依赖: 此文件需要在 scf/build_fock.hpp 之后 include，
// 且需要 eri_rys.hpp 提供 rys_roots_weights。

// 有效密度幅值低于此阈值时，该 task 对梯度无贡献，跳过
static constexpr float QC_GRAD_GAMMA_CUTOFF = 1e-15f;

#ifndef USE_GPU

// Rys VRR for gradient (supports extended angular momentum l+1)
static inline void QC_Grad_VRR_2D(float* __restrict__ G, int ij_max, int kl_max,
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

// Optimized batched factored HRR
static inline void QC_Grad_Factored_HRR_Batch(
    const float* __restrict__ G, int ij_am, int kl_am, int g_stride,
    const int* __restrict__ l, float AB_d, float CD_d,
    float* __restrict__ I_full, int d0, int d1, int d2)
{
    const int l0_up = l[0] + 1, l1_up = l[1] + 1;
    const int l2_up = l[2] + 1, l3_max = l[3];
    const int ij_ext = ij_am + 1;
    const int kl_ext = kl_am + 1;

    const int h_a1_stride = kl_ext + 1;
    const int h_a0_stride = (l1_up + 1) * h_a1_stride;
    float h_all[12 * 12 * 12];

    for (int j = 0; j <= kl_ext; j++)
    {
        float work[2][12];
        for (int i = 0; i <= ij_ext; i++) work[0][i] = G[i * g_stride + j];

        for (int a0 = 0; a0 <= l0_up; a0++)
            h_all[a0 * h_a0_stride + 0 * h_a1_stride + j] = work[0][a0];

        int cur = 0;
        for (int b = 0; b < l1_up; b++)
        {
            int nxt = 1 - cur;
            int n_curr = ij_ext - b - 1;
            for (int a = 0; a <= n_curr; a++)
                work[nxt][a] = work[cur][a + 1] + AB_d * work[cur][a];
            int a0_max = std::min(l0_up, n_curr);
            for (int a0 = 0; a0 <= a0_max; a0++)
                h_all[a0 * h_a0_stride + (b + 1) * h_a1_stride + j] =
                    work[nxt][a0];
            cur = nxt;
        }
    }

    for (int a0 = 0; a0 <= l0_up; a0++)
        for (int a1 = 0; a1 <= l1_up; a1++)
        {
            if (a0 + a1 > ij_ext) continue;
            const float* h_bra = &h_all[a0 * h_a0_stride + a1 * h_a1_stride];

            for (int a2 = 0; a2 <= l2_up; a2++)
                I_full[a0 * d0 + a1 * d1 + a2 * d2 + 0] = h_bra[a2];

            if (l3_max > 0)
            {
                float work[2][12];
                for (int i = 0; i <= kl_ext; i++) work[0][i] = h_bra[i];

                int cur = 0;
                for (int d = 0; d < l3_max; d++)
                {
                    int nxt = 1 - cur;
                    int n_curr = kl_ext - d - 1;
                    for (int a = 0; a <= n_curr; a++)
                        work[nxt][a] = work[cur][a + 1] + CD_d * work[cur][a];
                    int a2_max = std::min(l2_up, n_curr);
                    for (int a2 = 0; a2 <= a2_max; a2++)
                        I_full[a0 * d0 + a1 * d1 + a2 * d2 + (d + 1)] =
                            work[nxt][a2];
                    cur = nxt;
                }
            }
        }
}

// Sph2Cart step: expands spherical → Cartesian along one index
// dst[lead, cart, tail] = sum_sph C[cart * ns + sph] * src[lead, sph, tail]
static inline void QC_Sph2Cart_Step_CPU(const float* C, int nc, int ns,
                                        int leading, int tail, const float* src,
                                        float* dst)
{
    for (int lead = 0; lead < leading; lead++)
    {
        const float* src_blk = src + lead * ns * tail;
        float* dst_blk = dst + lead * nc * tail;
        memset(dst_blk, 0, (size_t)nc * tail * sizeof(float));
        for (int p = 0; p < ns; p++)
        {
            const float* src_row = src_blk + p * tail;
            for (int a = 0; a < nc; a++)
            {
                const float c = C[a * ns + p];
                if (c == 0.0f) continue;
                float* dst_row = dst_blk + a * tail;
                for (int idx = 0; idx < tail; idx++)
                    dst_row[idx] += c * src_row[idx];
            }
        }
    }
}

// 4-index Sph→Cart transform for effective density
// Input: gamma_sph in buf0 [ns0 × ns1 × ns2 × ns3]
// Output: gamma_cart in buf0 [nc0 × nc1 × nc2 × nc3]
static inline void QC_Sph2Cart_Density_CPU(
    const float* U, int nao_s, const int* off_cart, const int* off_sph,
    const int* dims_cart, const int* dims_sph, float* buf0, float* buf1)
{
    float C[4][MAX_CART_SHELL * MAX_CART_SHELL];
    for (int s = 0; s < 4; s++)
        for (int i = 0; i < dims_cart[s]; i++)
            for (int j = 0; j < dims_sph[s]; j++)
                C[s][i * dims_sph[s] + j] =
                    U[(off_cart[s] + i) * nao_s + (off_sph[s] + j)];

    // Transform each index from sph → cart sequentially
    // buf0: [ns0, ns1, ns2, ns3] → buf1: [nc0, ns1, ns2, ns3]
    QC_Sph2Cart_Step_CPU(C[0], dims_cart[0], dims_sph[0], 1,
                         dims_sph[1] * dims_sph[2] * dims_sph[3], buf0, buf1);
    // buf1 → buf0: [nc0, nc1, ns2, ns3]
    QC_Sph2Cart_Step_CPU(C[1], dims_cart[1], dims_sph[1], dims_cart[0],
                         dims_sph[2] * dims_sph[3], buf1, buf0);
    // buf0 → buf1: [nc0, nc1, nc2, ns3]
    QC_Sph2Cart_Step_CPU(C[2], dims_cart[2], dims_sph[2],
                         dims_cart[0] * dims_cart[1], dims_sph[3], buf0, buf1);
    // buf1 → buf0: [nc0, nc1, nc2, nc3]
    QC_Sph2Cart_Step_CPU(C[3], dims_cart[3], dims_sph[3],
                         dims_cart[0] * dims_cart[1] * dims_cart[2], 1, buf1,
                         buf0);
}

static inline void QC_Build_ERI_Gradient_CPU(
    const QC_INTEGRAL_TASKS& task_ctx, const QC_MOLECULE& mol,
    const QC_CARTESIAN_TO_SPHERICAL& cart2sph, const QC_SCF_WORKSPACE& scf_ws,
    float exx_fraction, float shell_screen_tol, float prim_screen_tol,
    const int* shell_atom, double* grad)
{
    // 局部别名: 保持函数体不变
    const bool unrestricted = scf_ws.runtime.unrestricted;
    const int nbas = mol.nbas;
    const int* atm = mol.d_atm;
    const int* bas = mol.d_bas;
    const float* env = mol.d_env;
    const int* ao_offsets_cart = mol.d_ao_offsets;
    const int* ao_offsets_sph = mol.d_ao_offsets_sph;
    const float* norms = scf_ws.ortho.d_norms;
    const float* shell_pair_bounds = task_ctx.buffers.d_shell_pair_bounds;
    const float* pair_density_coul = scf_ws.direct.d_pair_density_coul;
    const float* pair_density_exx_a = scf_ws.direct.d_pair_density_exx;
    const float* pair_density_exx_b = unrestricted
                                          ? scf_ws.direct.d_pair_density_exx_b
                                          : (const float*)nullptr;
    const float* P_coul = scf_ws.direct.d_P_coul;
    const float* P_exx_a = scf_ws.alpha.d_P;
    const float* P_exx_b =
        unrestricted ? scf_ws.beta.d_P : (const float*)nullptr;
    const float exx_scale_a =
        unrestricted ? exx_fraction : (0.5f * exx_fraction);
    const float exx_scale_b = unrestricted ? exx_fraction : 0.0f;
    const int nao = mol.nao;
    const int nao_sph = mol.nao_sph;
    const int is_spherical = mol.is_spherical;
    const float* cart2sph_mat = cart2sph.d_cart2sph_mat;
    const int hr_base = task_ctx.params.eri_hr_base;
    const int hr_size = task_ctx.params.eri_hr_size;
    const int shell_buf_size = task_ctx.params.eri_shell_buf_size;
    const int thread_count = scf_ws.direct.fock_thread_count;
    int natm_max = 0;
    for (int i = 0; i < nbas; i++)
        natm_max = std::max(natm_max, shell_atom[i] + 1);

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
    for (int pair_id = 0; pair_id < n_pairs; pair_id++)
    {
        const QC_ONE_E_TASK& pair = task_ctx.topo.h_shell_pairs[pair_id];
        const float exx_a = pair_density_exx_a[pair_id];
        shell_max_exx_a[(size_t)pair.x] =
            fmaxf(shell_max_exx_a[(size_t)pair.x], exx_a);
        shell_max_exx_a[(size_t)pair.y] =
            fmaxf(shell_max_exx_a[(size_t)pair.y], exx_a);
    }
    std::vector<float> anchor_activity((size_t)n_pairs, 0.0f);
    std::vector<int> sorted_pair_ids((size_t)n_pairs);
    for (int pair_id = 0; pair_id < n_pairs; pair_id++)
    {
        const QC_ONE_E_TASK& pair = task_ctx.topo.h_shell_pairs[pair_id];
        const float exx_anchor_a =
            exx_scale_a == 0.0f
                ? 0.0f
                : exx_scale_a * fmaxf(shell_max_exx_a[(size_t)pair.x],
                                      shell_max_exx_a[(size_t)pair.y]);
        anchor_activity[(size_t)pair_id] =
            shell_pair_bounds[pair_id] *
            QC_Max4(pair_density_coul[pair_id], exx_anchor_a, 0.0f, 0.0f);
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

    // Max I array size per axis for supported angular momentum (up to g=4)
    // I_size = (l0+2)(l1+2)(l2+2)(l3+1), max = 6*6*6*5 = 1080
    static constexpr int MAX_I_SIZE = 1100;
    // Max Rys roots: nrys = (L_sum+3)/2, for g-shells L_sum_max=16 → nrys=9
    static constexpr int MAX_NRYS = 12;

#pragma omp parallel num_threads(thread_count)
    {
        std::vector<double> grad_local((size_t)natm_max * 3, 0.0);
        std::vector<int> partner_marks((size_t)n_pairs, -1);
        std::vector<int> candidate_partners;
        candidate_partners.reserve(256);
        // Buffers for density Sph2Cart transform
        std::vector<float> gamma_buf0(MAX_SHELL_ERI);
        std::vector<float> gamma_buf1(MAX_SHELL_ERI);
        // Thread-local I array storage for factored contraction (high L)
        std::vector<float> all_Ix_buf(MAX_NRYS * MAX_I_SIZE);
        std::vector<float> all_Iy_buf(MAX_NRYS * MAX_I_SIZE);
        std::vector<float> all_Iz_buf(MAX_NRYS * MAX_I_SIZE);

#pragma omp for schedule(dynamic)
        for (int pair_ij = 0; pair_ij < n_pairs; pair_ij++)
        {
            const QC_Shell_Pair_Meta_CPU& bra = pair_meta[(size_t)pair_ij];
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

            // Pre-screen and cache bra primitives
            struct BraPrim
            {
                float ai, aj, p, inv_p, n_ab;
                float P[3], PA[3];
            };
            std::vector<BraPrim> bra_prims;
            bra_prims.reserve((size_t)bra.np[0] * (size_t)bra.np[1]);
            for (int ip = 0; ip < bra.np[0]; ip++)
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
                    BraPrim bp;
                    bp.ai = ai;
                    bp.aj = aj;
                    bp.p = p;
                    bp.inv_p = inv_p;
                    bp.n_ab = n_ab;
                    for (int d = 0; d < 3; d++)
                    {
                        bp.P[d] = (ai * bra.R[0][d] + aj * bra.R[1][d]) * inv_p;
                        bp.PA[d] = bp.P[d] - bra.R[0][d];
                    }
                    bra_prims.push_back(bp);
                }
            if (bra_prims.empty()) continue;

            const QC_ONE_E_TASK& ij = task_ctx.topo.h_shell_pairs[pair_ij];
            const int atom_A = shell_atom[ij.x];
            const int atom_B = shell_atom[ij.y];

            for (const int pair_kl : candidate_partners)
            {
                const float exact_screen = QC_Exact_Quartet_Screen_CPU(
                    task_ctx, pair_ij, pair_kl, shell_pair_bounds,
                    pair_density_coul, pair_density_exx_a, pair_density_exx_b,
                    exx_scale_a, exx_scale_b);
                if (exact_screen < shell_screen_tol) continue;

                const QC_ONE_E_TASK& kl = task_ctx.topo.h_shell_pairs[pair_kl];
                const QC_Shell_Pair_Meta_CPU& ket = pair_meta[(size_t)pair_kl];
                const int atom_C = shell_atom[kl.x];
                const int atom_D = shell_atom[kl.y];

                const int l[4] = {bra.l[0], bra.l[1], ket.l[0], ket.l[1]};
                const int ij_am = l[0] + l[1];
                const int kl_am = l[2] + l[3];
                const int L_sum = ij_am + kl_am;

                const bool jk_same_bra = (ij.x == ij.y);
                const bool jk_same_ket = (kl.x == kl.y);
                const bool jk_same_braket = (ij.x == kl.x && ij.y == kl.y);

                const int ni_cart = bra.dims_cart[0],
                          nj_cart = bra.dims_cart[1];
                const int nk_cart = ket.dims_cart[0],
                          nl_cart = ket.dims_cart[1];
                const int shell_size_cart =
                    ni_cart * nj_cart * nk_cart * nl_cart;

                const int ni = bra.dims_eff[0], nj = bra.dims_eff[1];
                const int nk = ket.dims_eff[0], nl = ket.dims_eff[1];

                // Step 1: compute gamma_sph (Cartesian effective density),
                // with symmetry and norms applied.
                const int sph_size = ni * nj * nk * nl;
                memset(gamma_buf0.data(), 0, (size_t)sph_size * sizeof(float));

                for (int ci = 0; ci < ni; ci++)
                {
                    const int p = bra.off_eff[0] + ci;
                    for (int cj = 0; cj < nj; cj++)
                    {
                        const int q_idx = bra.off_eff[1] + cj;
                        if (jk_same_bra && q_idx > p) continue;
                        const double nij =
                            (double)norms[p] * (double)norms[q_idx];
                        for (int ck = 0; ck < nk; ck++)
                        {
                            const int r = ket.off_eff[0] + ck;
                            const double nijr = nij * (double)norms[r];
                            for (int cl = 0; cl < nl; cl++)
                            {
                                const int s = ket.off_eff[1] + cl;
                                if (jk_same_ket && s > r) continue;
                                if (jk_same_braket)
                                {
                                    const int pq = p * nao + q_idx;
                                    const int rs = r * nao + s;
                                    if (rs > pq) continue;
                                }

                                double sym = nijr * (double)norms[s];
                                if (jk_same_bra && p == q_idx) sym *= 0.5;
                                if (jk_same_ket && r == s) sym *= 0.5;
                                if (jk_same_braket && p == r && q_idx == s)
                                    sym *= 0.5;

                                double gamma = sym * 4.0 *
                                               (double)P_coul[p * nao + q_idx] *
                                               (double)P_coul[r * nao + s];
                                if (exx_scale_a != 0.0f)
                                    gamma -=
                                        sym * 2.0 * (double)exx_scale_a *
                                        ((double)P_exx_a[p * nao + r] *
                                             (double)P_exx_a[q_idx * nao + s] +
                                         (double)P_exx_a[p * nao + s] *
                                             (double)P_exx_a[q_idx * nao + r]);
                                if (exx_scale_b != 0.0f && P_exx_b != nullptr)
                                    gamma -=
                                        sym * 2.0 * (double)exx_scale_b *
                                        ((double)P_exx_b[p * nao + r] *
                                             (double)P_exx_b[q_idx * nao + s] +
                                         (double)P_exx_b[p * nao + s] *
                                             (double)P_exx_b[q_idx * nao + r]);

                                const int sph_idx =
                                    ((ci * nj + cj) * nk + ck) * nl + cl;
                                gamma_buf0[(size_t)sph_idx] = (float)gamma;
                            }
                        }
                    }
                }

                // Step 2: transform gamma from spherical to Cartesian basis
                float* gamma_cart;
                if (is_spherical)
                {
                    const int dims_cart_arr[4] = {ni_cart, nj_cart, nk_cart,
                                                  nl_cart};
                    const int dims_sph_arr[4] = {
                        bra.dims_sph[0], bra.dims_sph[1], ket.dims_sph[0],
                        ket.dims_sph[1]};
                    const int off_cart_arr[4] = {
                        bra.off_cart[0], bra.off_cart[1], ket.off_cart[0],
                        ket.off_cart[1]};
                    const int off_sph_arr[4] = {bra.off_eff[0], bra.off_eff[1],
                                                ket.off_eff[0], ket.off_eff[1]};
                    QC_Sph2Cart_Density_CPU(cart2sph_mat, nao_sph, off_cart_arr,
                                            off_sph_arr, dims_cart_arr,
                                            dims_sph_arr, gamma_buf0.data(),
                                            gamma_buf1.data());
                    gamma_cart = gamma_buf0.data();
                }
                else
                {
                    gamma_cart = gamma_buf0.data();
                }

                float max_gamma = 0.0f;
                for (int i = 0; i < shell_size_cart; i++)
                    max_gamma = fmaxf(max_gamma, fabsf(gamma_cart[i]));
                if (max_gamma < QC_GRAD_GAMMA_CUTOFF) continue;

                const float AB[3] = {bra.R[0][0] - bra.R[1][0],
                                     bra.R[0][1] - bra.R[1][1],
                                     bra.R[0][2] - bra.R[1][2]};
                const float CD[3] = {ket.R[0][0] - ket.R[1][0],
                                     ket.R[0][1] - ket.R[1][1],
                                     ket.R[0][2] - ket.R[1][2]};

                const int nrys = (L_sum + 3) / 2;
                const int g_stride = kl_am + 2;
                const int ix_d2 = (l[3] + 1);
                const int ix_d1 = (l[2] + 2) * ix_d2;
                const int ix_d0 = (l[1] + 2) * ix_d1;
                const int I_size = (l[0] + 2) * ix_d0;

                // Gradient accumulators for this quartet (double precision)
                double g_A[3] = {0.0, 0.0, 0.0};
                double g_B[3] = {0.0, 0.0, 0.0};
                double g_C[3] = {0.0, 0.0, 0.0};

                // Threshold: use factored contraction for high angular momentum
                const bool use_factored = (L_sum >= 6);

                // ==== Primitive loop (Rys quadrature) ====
                for (const auto& bp : bra_prims)
                {
                    const float two_ai = 2.0f * bp.ai;
                    const float two_aj = 2.0f * bp.aj;

                    for (int kp = 0; kp < ket.np[0]; kp++)
                    {
                        const float ak = env[ket.p_exp[0] + kp];
                        const float two_ak = 2.0f * ak;
                        for (int lp = 0; lp < ket.np[1]; lp++)
                        {
                            const float al = env[ket.p_exp[1] + lp];
                            const float q_val = ak + al;
                            const float inv_q = 1.0f / q_val;
                            const float kcd =
                                expf(-(ak * al * inv_q) * ket.pair_dist2);
                            const float pref =
                                2.0f * PI_25 /
                                (bp.p * q_val * sqrtf(bp.p + q_val));
                            const float n_abcd =
                                bp.n_ab * env[ket.p_cof[0] + kp] *
                                env[ket.p_cof[1] + lp] * kcd * pref;
                            if (fabsf(n_abcd) < prim_screen_tol) continue;

                            float Q[3], QCv[3], PQ[3];
                            for (int d = 0; d < 3; d++)
                            {
                                Q[d] = (ak * ket.R[0][d] + al * ket.R[1][d]) *
                                       inv_q;
                                QCv[d] = Q[d] - ket.R[0][d];
                                PQ[d] = bp.P[d] - Q[d];
                            }
                            const float rho = bp.p * q_val / (bp.p + q_val);
                            const float T =
                                rho *
                                (PQ[0] * PQ[0] + PQ[1] * PQ[1] + PQ[2] * PQ[2]);

                            double rys_r[MAX_NRYS], rys_w[MAX_NRYS];
                            rys_roots_weights(nrys, (double)T, rys_r, rys_w);

                            if (use_factored)
                            {
                                // === HIGH-L PATH: Factored contraction ===
                                // Store all roots' I arrays, then contract
                                // gamma with two axes at once for each
                                // derivative axis.
                                double all_wn[MAX_NRYS];
                                float* aIx = all_Ix_buf.data();
                                float* aIy = all_Iy_buf.data();
                                float* aIz = all_Iz_buf.data();
                                for (int ir = 0; ir < nrys; ir++)
                                {
                                    const float u = (float)rys_r[ir];
                                    const float w = (float)rys_w[ir];
                                    all_wn[ir] = (double)n_abcd * (double)w;
                                    const float factor = u / (bp.p + q_val);
                                    const float B00 = 0.5f * factor;
                                    const float B10 =
                                        0.5f / bp.p * (1.0f - q_val * factor);
                                    const float B01 =
                                        0.5f / q_val * (1.0f - bp.p * factor);

                                    float Gx[120], Gy[120], Gz[120];
                                    const float Cx_bra[3] = {
                                        bp.PA[0] - factor * q_val * PQ[0],
                                        bp.PA[1] - factor * q_val * PQ[1],
                                        bp.PA[2] - factor * q_val * PQ[2]};
                                    const float Cx_ket[3] = {
                                        QCv[0] + factor * bp.p * PQ[0],
                                        QCv[1] + factor * bp.p * PQ[1],
                                        QCv[2] + factor * bp.p * PQ[2]};

                                    QC_Grad_VRR_2D(Gx, ij_am + 1, kl_am + 1,
                                                   g_stride, Cx_bra[0],
                                                   Cx_ket[0], B00, B10, B01);
                                    QC_Grad_VRR_2D(Gy, ij_am + 1, kl_am + 1,
                                                   g_stride, Cx_bra[1],
                                                   Cx_ket[1], B00, B10, B01);
                                    QC_Grad_VRR_2D(Gz, ij_am + 1, kl_am + 1,
                                                   g_stride, Cx_bra[2],
                                                   Cx_ket[2], B00, B10, B01);

                                    QC_Grad_Factored_HRR_Batch(
                                        Gx, ij_am, kl_am, g_stride, l, AB[0],
                                        CD[0], &aIx[ir * I_size], ix_d0, ix_d1,
                                        ix_d2);
                                    QC_Grad_Factored_HRR_Batch(
                                        Gy, ij_am, kl_am, g_stride, l, AB[1],
                                        CD[1], &aIy[ir * I_size], ix_d0, ix_d1,
                                        ix_d2);
                                    QC_Grad_Factored_HRR_Batch(
                                        Gz, ij_am, kl_am, g_stride, l, AB[2],
                                        CD[2], &aIz[ir * I_size], ix_d0, ix_d1,
                                        ix_d2);
                                }

                                const float* all_I[3] = {aIx, aIy, aIz};
                                for (int d_ax = 0; d_ax < 3; d_ax++)
                                {
                                    const int a1 = (d_ax + 1) % 3;
                                    const int a2 = (d_ax + 2) % 3;
                                    const float* Id = all_I[d_ax];
                                    const float* Ic1 = all_I[a1];
                                    const float* Ic2 = all_I[a2];

                                    for (int i0d = 0; i0d <= l[0]; i0d++)
                                        for (int i1d = 0; i1d <= l[1]; i1d++)
                                            for (int i2d = 0; i2d <= l[2];
                                                 i2d++)
                                                for (int i3d = 0; i3d <= l[3];
                                                     i3d++)
                                                {
                                                    const int bd = i0d * ix_d0 +
                                                                   i1d * ix_d1 +
                                                                   i2d * ix_d2 +
                                                                   i3d;
                                                    double sum[MAX_NRYS] = {};

                                                    for (int i0c = 0;
                                                         i0c <= l[0] - i0d;
                                                         i0c++)
                                                    {
                                                        const int i0t =
                                                            l[0] - i0d - i0c;
                                                        int i0x, i0y;
                                                        if (d_ax == 0)
                                                        {
                                                            i0x = i0d;
                                                            i0y = i0c;
                                                        }
                                                        else if (d_ax == 1)
                                                        {
                                                            i0x = i0t;
                                                            i0y = i0d;
                                                        }
                                                        else
                                                        {
                                                            i0x = i0c;
                                                            i0y = i0t;
                                                        }
                                                        const int r0 =
                                                            l[0] - i0x;
                                                        const int c0 =
                                                            r0 * (r0 + 1) / 2 +
                                                            (r0 - i0y);

                                                        for (int i1c = 0;
                                                             i1c <= l[1] - i1d;
                                                             i1c++)
                                                        {
                                                            const int i1t =
                                                                l[1] - i1d -
                                                                i1c;
                                                            int i1x, i1y;
                                                            if (d_ax == 0)
                                                            {
                                                                i1x = i1d;
                                                                i1y = i1c;
                                                            }
                                                            else if (d_ax == 1)
                                                            {
                                                                i1x = i1t;
                                                                i1y = i1d;
                                                            }
                                                            else
                                                            {
                                                                i1x = i1c;
                                                                i1y = i1t;
                                                            }
                                                            const int r1 =
                                                                l[1] - i1x;
                                                            const int c1 =
                                                                r1 * (r1 + 1) /
                                                                    2 +
                                                                (r1 - i1y);

                                                            for (int i2c = 0;
                                                                 i2c <=
                                                                 l[2] - i2d;
                                                                 i2c++)
                                                            {
                                                                const int i2t =
                                                                    l[2] - i2d -
                                                                    i2c;
                                                                int i2x, i2y;
                                                                if (d_ax == 0)
                                                                {
                                                                    i2x = i2d;
                                                                    i2y = i2c;
                                                                }
                                                                else if (d_ax ==
                                                                         1)
                                                                {
                                                                    i2x = i2t;
                                                                    i2y = i2d;
                                                                }
                                                                else
                                                                {
                                                                    i2x = i2c;
                                                                    i2y = i2t;
                                                                }
                                                                const int r2 =
                                                                    l[2] - i2x;
                                                                const int c2 =
                                                                    r2 *
                                                                        (r2 +
                                                                         1) /
                                                                        2 +
                                                                    (r2 - i2y);

                                                                const int bc1_base =
                                                                    i0c *
                                                                        ix_d0 +
                                                                    i1c *
                                                                        ix_d1 +
                                                                    i2c * ix_d2;
                                                                const int bc2_base =
                                                                    i0t *
                                                                        ix_d0 +
                                                                    i1t *
                                                                        ix_d1 +
                                                                    i2t * ix_d2;
                                                                const int idx_base =
                                                                    ((c0 *
                                                                          nj_cart +
                                                                      c1) *
                                                                         nk_cart +
                                                                     c2) *
                                                                    nl_cart;

                                                                for (int i3c =
                                                                         0;
                                                                     i3c <=
                                                                     l[3] - i3d;
                                                                     i3c++)
                                                                {
                                                                    const int
                                                                        i3t =
                                                                            l[3] -
                                                                            i3d -
                                                                            i3c;
                                                                    int i3x,
                                                                        i3y;
                                                                    if (d_ax ==
                                                                        0)
                                                                    {
                                                                        i3x =
                                                                            i3d;
                                                                        i3y =
                                                                            i3c;
                                                                    }
                                                                    else if (
                                                                        d_ax ==
                                                                        1)
                                                                    {
                                                                        i3x =
                                                                            i3t;
                                                                        i3y =
                                                                            i3d;
                                                                    }
                                                                    else
                                                                    {
                                                                        i3x =
                                                                            i3c;
                                                                        i3y =
                                                                            i3t;
                                                                    }
                                                                    const int r3 =
                                                                        l[3] -
                                                                        i3x;
                                                                    const int c3 =
                                                                        r3 *
                                                                            (r3 +
                                                                             1) /
                                                                            2 +
                                                                        (r3 -
                                                                         i3y);

                                                                    const float g =
                                                                        gamma_cart
                                                                            [idx_base +
                                                                             c3];
                                                                    if (g ==
                                                                        0.0f)
                                                                        continue;

                                                                    const int bc1 =
                                                                        bc1_base +
                                                                        i3c;
                                                                    const int bc2 =
                                                                        bc2_base +
                                                                        i3t;

                                                                    for (
                                                                        int ir =
                                                                            0;
                                                                        ir <
                                                                        nrys;
                                                                        ir++)
                                                                        sum[ir] +=
                                                                            (double)
                                                                                g *
                                                                            all_wn
                                                                                [ir] *
                                                                            (double)Ic1
                                                                                [ir *
                                                                                     I_size +
                                                                                 bc1] *
                                                                            (double)Ic2
                                                                                [ir *
                                                                                     I_size +
                                                                                 bc2];
                                                                }
                                                            }
                                                        }
                                                    }

                                                    for (int ir = 0; ir < nrys;
                                                         ir++)
                                                    {
                                                        if (sum[ir] == 0.0)
                                                            continue;
                                                        double dA =
                                                            (double)two_ai *
                                                            (double)
                                                                Id[ir * I_size +
                                                                   bd + ix_d0];
                                                        if (i0d > 0)
                                                            dA -=
                                                                (double)i0d *
                                                                (double)Id
                                                                    [ir *
                                                                         I_size +
                                                                     bd -
                                                                     ix_d0];
                                                        g_A[d_ax] +=
                                                            sum[ir] * dA;

                                                        double dB =
                                                            (double)two_aj *
                                                            (double)
                                                                Id[ir * I_size +
                                                                   bd + ix_d1];
                                                        if (i1d > 0)
                                                            dB -=
                                                                (double)i1d *
                                                                (double)Id
                                                                    [ir *
                                                                         I_size +
                                                                     bd -
                                                                     ix_d1];
                                                        g_B[d_ax] +=
                                                            sum[ir] * dB;

                                                        double dC =
                                                            (double)two_ak *
                                                            (double)
                                                                Id[ir * I_size +
                                                                   bd + ix_d2];
                                                        if (i2d > 0)
                                                            dC -=
                                                                (double)i2d *
                                                                (double)Id
                                                                    [ir *
                                                                         I_size +
                                                                     bd -
                                                                     ix_d2];
                                                        g_C[d_ax] +=
                                                            sum[ir] * dC;
                                                    }
                                                }
                                }
                            }  // end factored path
                            else
                            {
                                // === LOW-L PATH: Direct per-root assembly ===
                                for (int ir = 0; ir < nrys; ir++)
                                {
                                    const float u = (float)rys_r[ir];
                                    const float w = (float)rys_w[ir];
                                    const float factor = u / (bp.p + q_val);
                                    const float B00 = 0.5f * factor;
                                    const float B10 =
                                        0.5f / bp.p * (1.0f - q_val * factor);
                                    const float B01 =
                                        0.5f / q_val * (1.0f - bp.p * factor);

                                    float Gx[120], Gy[120], Gz[120];
                                    const float Cx_bra[3] = {
                                        bp.PA[0] - factor * q_val * PQ[0],
                                        bp.PA[1] - factor * q_val * PQ[1],
                                        bp.PA[2] - factor * q_val * PQ[2]};
                                    const float Cx_ket[3] = {
                                        QCv[0] + factor * bp.p * PQ[0],
                                        QCv[1] + factor * bp.p * PQ[1],
                                        QCv[2] + factor * bp.p * PQ[2]};

                                    QC_Grad_VRR_2D(Gx, ij_am + 1, kl_am + 1,
                                                   g_stride, Cx_bra[0],
                                                   Cx_ket[0], B00, B10, B01);
                                    QC_Grad_VRR_2D(Gy, ij_am + 1, kl_am + 1,
                                                   g_stride, Cx_bra[1],
                                                   Cx_ket[1], B00, B10, B01);
                                    QC_Grad_VRR_2D(Gz, ij_am + 1, kl_am + 1,
                                                   g_stride, Cx_bra[2],
                                                   Cx_ket[2], B00, B10, B01);

                                    float Ix[500], Iy[500], Iz[500];
                                    QC_Grad_Factored_HRR_Batch(
                                        Gx, ij_am, kl_am, g_stride, l, AB[0],
                                        CD[0], Ix, ix_d0, ix_d1, ix_d2);
                                    QC_Grad_Factored_HRR_Batch(
                                        Gy, ij_am, kl_am, g_stride, l, AB[1],
                                        CD[1], Iy, ix_d0, ix_d1, ix_d2);
                                    QC_Grad_Factored_HRR_Batch(
                                        Gz, ij_am, kl_am, g_stride, l, AB[2],
                                        CD[2], Iz, ix_d0, ix_d1, ix_d2);

                                    const float wn = n_abcd * w;
                                    int idx = 0;
                                    for (int c0 = 0; c0 < ni_cart; c0++)
                                    {
                                        const int i0x = bra.comp_x[0][c0];
                                        const int i0y = bra.comp_y[0][c0];
                                        const int i0z = bra.comp_z[0][c0];
                                        for (int c1 = 0; c1 < nj_cart; c1++)
                                        {
                                            const int i1x = bra.comp_x[1][c1];
                                            const int i1y = bra.comp_y[1][c1];
                                            const int i1z = bra.comp_z[1][c1];
                                            for (int c2 = 0; c2 < nk_cart; c2++)
                                            {
                                                const int i2x =
                                                    ket.comp_x[0][c2];
                                                const int i2y =
                                                    ket.comp_y[0][c2];
                                                const int i2z =
                                                    ket.comp_z[0][c2];
                                                const int bx_base =
                                                    i0x * ix_d0 + i1x * ix_d1 +
                                                    i2x * ix_d2;
                                                const int by_base =
                                                    i0y * ix_d0 + i1y * ix_d1 +
                                                    i2y * ix_d2;
                                                const int bz_base =
                                                    i0z * ix_d0 + i1z * ix_d1 +
                                                    i2z * ix_d2;

                                                for (int c3 = 0; c3 < nl_cart;
                                                     c3++, idx++)
                                                {
                                                    const float g =
                                                        gamma_cart[idx];
                                                    if (g == 0.0f) continue;

                                                    const int i3x =
                                                        ket.comp_x[1][c3];
                                                    const int i3y =
                                                        ket.comp_y[1][c3];
                                                    const int i3z =
                                                        ket.comp_z[1][c3];
                                                    const int bx =
                                                        bx_base + i3x;
                                                    const int by =
                                                        by_base + i3y;
                                                    const int bz =
                                                        bz_base + i3z;

                                                    const double gwn =
                                                        (double)g * (double)wn;

                                                    const double yz =
                                                        (double)Iy[by] *
                                                        (double)Iz[bz];
                                                    float cAx =
                                                        two_ai * Ix[bx + ix_d0];
                                                    if (i0x > 0)
                                                        cAx -= (float)i0x *
                                                               Ix[bx - ix_d0];
                                                    float cBx =
                                                        two_aj * Ix[bx + ix_d1];
                                                    if (i1x > 0)
                                                        cBx -= (float)i1x *
                                                               Ix[bx - ix_d1];
                                                    float cCx =
                                                        two_ak * Ix[bx + ix_d2];
                                                    if (i2x > 0)
                                                        cCx -= (float)i2x *
                                                               Ix[bx - ix_d2];
                                                    const double fxyz =
                                                        gwn * yz;
                                                    g_A[0] +=
                                                        fxyz * (double)cAx;
                                                    g_B[0] +=
                                                        fxyz * (double)cBx;
                                                    g_C[0] +=
                                                        fxyz * (double)cCx;

                                                    const double xz =
                                                        (double)Ix[bx] *
                                                        (double)Iz[bz];
                                                    float cAy =
                                                        two_ai * Iy[by + ix_d0];
                                                    if (i0y > 0)
                                                        cAy -= (float)i0y *
                                                               Iy[by - ix_d0];
                                                    float cBy =
                                                        two_aj * Iy[by + ix_d1];
                                                    if (i1y > 0)
                                                        cBy -= (float)i1y *
                                                               Iy[by - ix_d1];
                                                    float cCy =
                                                        two_ak * Iy[by + ix_d2];
                                                    if (i2y > 0)
                                                        cCy -= (float)i2y *
                                                               Iy[by - ix_d2];
                                                    const double fxyz_y =
                                                        gwn * xz;
                                                    g_A[1] +=
                                                        fxyz_y * (double)cAy;
                                                    g_B[1] +=
                                                        fxyz_y * (double)cBy;
                                                    g_C[1] +=
                                                        fxyz_y * (double)cCy;

                                                    const double xy =
                                                        (double)Ix[bx] *
                                                        (double)Iy[by];
                                                    float cAz =
                                                        two_ai * Iz[bz + ix_d0];
                                                    if (i0z > 0)
                                                        cAz -= (float)i0z *
                                                               Iz[bz - ix_d0];
                                                    float cBz =
                                                        two_aj * Iz[bz + ix_d1];
                                                    if (i1z > 0)
                                                        cBz -= (float)i1z *
                                                               Iz[bz - ix_d1];
                                                    float cCz =
                                                        two_ak * Iz[bz + ix_d2];
                                                    if (i2z > 0)
                                                        cCz -= (float)i2z *
                                                               Iz[bz - ix_d2];
                                                    const double fxyz_z =
                                                        gwn * xy;
                                                    g_A[2] +=
                                                        fxyz_z * (double)cAz;
                                                    g_B[2] +=
                                                        fxyz_z * (double)cBz;
                                                    g_C[2] +=
                                                        fxyz_z * (double)cCz;
                                                }
                                            }
                                        }
                                    }
                                }  // end Rys roots (direct path)
                            }  // end direct path
                        }
                    }
                }  // end primitives

                // Write to grad_local with translational invariance
                for (int d = 0; d < 3; d++)
                {
                    grad_local[(size_t)atom_A * 3 + d] += g_A[d];
                    grad_local[(size_t)atom_B * 3 + d] += g_B[d];
                    grad_local[(size_t)atom_C * 3 + d] += g_C[d];
                    grad_local[(size_t)atom_D * 3 + d] -=
                        (g_A[d] + g_B[d] + g_C[d]);
                }
            }
        }

#pragma omp critical
        {
            for (size_t i = 0; i < grad_local.size(); i++)
                grad[i] += grad_local[i];
        }
    }
}

#else  // USE_GPU — GPU ERI gradient kernel

// GPU ERI Gradient Kernel
// One thread per screened shell quartet. Computes effective density gamma,
// transforms to Cartesian basis (if spherical), then runs Rys quadrature
// with extended VRR + HRR to produce derivative integrals and accumulates
// atomic gradients.
//
// Scratch sizes are chosen to cover up to g-shells (l_max=4):
// G: (ij_am+2)*(kl_am+2) <= 10*10 = 100
// I: (l0+2)*(l1+2)*(l2+2)*(l3+1) <= 6*6*6*5 = 1080

static __device__ void grad_factored_hrr_batch(
    const float* __restrict__ G, int ij_am, int kl_am, int g_stride,
    const int* __restrict__ l, float AB_d, float CD_d,
    float* __restrict__ I_full, int d0, int d1, int d2)
{
    const int l0_up = l[0] + 1, l1_up = l[1] + 1;
    const int l2_up = l[2] + 1, l3_max = l[3];
    const int ij_ext = ij_am + 1;
    const int kl_ext = kl_am + 1;

    const int h_a1_stride = kl_ext + 1;
    const int h_a0_stride = (l1_up + 1) * h_a1_stride;
    float h_all[12 * 12 * 12];

    for (int j = 0; j <= kl_ext; j++)
    {
        float work[2][12];
        for (int i = 0; i <= ij_ext; i++) work[0][i] = G[i * g_stride + j];

        for (int a0 = 0; a0 <= l0_up; a0++)
            h_all[a0 * h_a0_stride + j] = work[0][a0];

        int cur = 0;
        for (int b = 0; b < l1_up; b++)
        {
            const int nxt = 1 - cur;
            const int n_curr = ij_ext - b - 1;
            for (int a = 0; a <= n_curr; a++)
                work[nxt][a] = work[cur][a + 1] + AB_d * work[cur][a];
            const int a0_max = l0_up < n_curr ? l0_up : n_curr;
            for (int a0 = 0; a0 <= a0_max; a0++)
                h_all[a0 * h_a0_stride + (b + 1) * h_a1_stride + j] =
                    work[nxt][a0];
            cur = nxt;
        }
    }

    for (int a0 = 0; a0 <= l0_up; a0++)
        for (int a1 = 0; a1 <= l1_up; a1++)
        {
            if (a0 + a1 > ij_ext) continue;
            const float* h_bra = &h_all[a0 * h_a0_stride + a1 * h_a1_stride];

            for (int a2 = 0; a2 <= l2_up; a2++)
                I_full[a0 * d0 + a1 * d1 + a2 * d2] = h_bra[a2];

            if (l3_max > 0)
            {
                float work[2][12];
                for (int i = 0; i <= kl_ext; i++) work[0][i] = h_bra[i];

                int cur = 0;
                for (int d = 0; d < l3_max; d++)
                {
                    const int nxt = 1 - cur;
                    const int n_curr = kl_ext - d - 1;
                    for (int a = 0; a <= n_curr; a++)
                        work[nxt][a] = work[cur][a + 1] + CD_d * work[cur][a];
                    const int a2_max = l2_up < n_curr ? l2_up : n_curr;
                    for (int a2 = 0; a2 <= a2_max; a2++)
                        I_full[a0 * d0 + a1 * d1 + a2 * d2 + (d + 1)] =
                            work[nxt][a2];
                    cur = nxt;
                }
            }
        }
}

// Sph→Cart device helper: one axis transform, in-place capable via temp copy.
// dst[lead, cart, tail] = Σ_sph C[cart * ns + sph] * src[lead, sph, tail]
static __device__ void grad_sph2cart_step(
    const float* __restrict__ cart2sph_mat, int nao_sph, int off_cart,
    int off_sph, int nc, int ns, int leading, int tail,
    const float* __restrict__ src, float* __restrict__ dst)
{
    for (int lead = 0; lead < leading; lead++)
    {
        const float* src_blk = src + lead * ns * tail;
        float* dst_blk = dst + lead * nc * tail;
        for (int a = 0; a < nc * tail; a++) dst_blk[a] = 0.0f;
        for (int p = 0; p < ns; p++)
        {
            const float* src_row = src_blk + p * tail;
            for (int a = 0; a < nc; a++)
            {
                const float c =
                    cart2sph_mat[(off_cart + a) * nao_sph + (off_sph + p)];
                if (c == 0.0f) continue;
                float* dst_row = dst_blk + a * tail;
                for (int idx = 0; idx < tail; idx++)
                    dst_row[idx] += c * src_row[idx];
            }
        }
    }
}

__global__ void QC_ERI_Grad_Kernel(
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
    float* __restrict__ global_gamma_pool, const int gamma_buf_size,
    const int* __restrict__ shell_atom, double* __restrict__ grad_copies,
    const int n_grad_copies, const int natm, const float prim_screen_tol)
{
    const int worker_id = blockDim.x * blockIdx.x + threadIdx.x;
    const int worker_stride = blockDim.x * gridDim.x;
    for (int task_id = worker_id; task_id < n_tasks; task_id += worker_stride)
    {
        double* grad_local =
            grad_copies +
            (size_t)(blockIdx.x % n_grad_copies) * (size_t)(natm * 3);

        const QC_ERI_TASK tk = tasks[task_id];

        // Screening (same as Fock kernel)
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
        if (pair_density_exx_b != NULL && exx_scale_b != 0.0f)
            exx_screen_b = shell_bound * exx_scale_b *
                           QC_Max4(pair_density_exx_b[ik_pair],
                                   pair_density_exx_b[il_pair],
                                   pair_density_exx_b[jk_pair],
                                   pair_density_exx_b[jl_pair]);

        if (fmaxf(coul_screen, fmaxf(exx_screen_a, exx_screen_b)) >=
            shell_screen_tol)
        {
            // Shell data
            const int sh[4] = {tk.x, tk.y, tk.z, tk.w};
            int l[4], np[4], p_exp_off[4], p_cof_off[4];
            float RC[4][3];
            int off_cart_sh[4], off_eff[4], dim_cart[4], dim_eff[4];
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
                off_cart_sh[i] = ao_offsets_cart[sh[i]];
                off_eff[i] = is_spherical ? ao_offsets_sph[sh[i]]
                                          : ao_offsets_cart[sh[i]];
            }

            const int atom_A = shell_atom[tk.x];
            const int atom_B = shell_atom[tk.y];
            const int atom_C = shell_atom[tk.z];
            const int atom_D = shell_atom[tk.w];

            const bool jk_same_bra = (tk.x == tk.y);
            const bool jk_same_ket = (tk.z == tk.w);
            const bool jk_same_braket = (tk.x == tk.z && tk.y == tk.w);

            const int ni = dim_eff[0], nj = dim_eff[1];
            const int nk = dim_eff[2], nl = dim_eff[3];
            const int ni_cart = dim_cart[0], nj_cart = dim_cart[1];
            const int nk_cart = dim_cart[2], nl_cart = dim_cart[3];
            const int shell_size_cart = ni_cart * nj_cart * nk_cart * nl_cart;
            float* gamma_buf0 =
                global_gamma_pool +
                (size_t)worker_id * (size_t)(2 * gamma_buf_size);
            float* gamma_buf1 = gamma_buf0 + gamma_buf_size;

            // Compute gamma in effective (sph or cart) basis
            const int sph_size = ni * nj * nk * nl;
            for (int i = 0; i < sph_size; i++) gamma_buf0[i] = 0.0f;

            for (int ci = 0; ci < ni; ci++)
            {
                const int p = off_eff[0] + ci;
                for (int cj = 0; cj < nj; cj++)
                {
                    const int q = off_eff[1] + cj;
                    if (jk_same_bra && q > p) continue;
                    const double nij = (double)norms[p] * (double)norms[q];
                    for (int ck = 0; ck < nk; ck++)
                    {
                        const int r = off_eff[2] + ck;
                        const double nijr = nij * (double)norms[r];
                        for (int cl = 0; cl < nl; cl++)
                        {
                            const int s = off_eff[3] + cl;
                            if (jk_same_ket && s > r) continue;
                            if (jk_same_braket)
                            {
                                const int pq = p * nao + q;
                                const int rs = r * nao + s;
                                if (rs > pq) continue;
                            }

                            double sym = nijr * (double)norms[s];
                            if (jk_same_bra && p == q) sym *= 0.5;
                            if (jk_same_ket && r == s) sym *= 0.5;
                            if (jk_same_braket && p == r && q == s) sym *= 0.5;

                            double gamma = sym * 4.0 *
                                           (double)P_coul[p * nao + q] *
                                           (double)P_coul[r * nao + s];
                            if (exx_scale_a != 0.0f)
                                gamma -= sym * 2.0 * (double)exx_scale_a *
                                         ((double)P_exx_a[p * nao + r] *
                                              (double)P_exx_a[q * nao + s] +
                                          (double)P_exx_a[p * nao + s] *
                                              (double)P_exx_a[q * nao + r]);
                            if (exx_scale_b != 0.0f && P_exx_b != NULL)
                                gamma -= sym * 2.0 * (double)exx_scale_b *
                                         ((double)P_exx_b[p * nao + r] *
                                              (double)P_exx_b[q * nao + s] +
                                          (double)P_exx_b[p * nao + s] *
                                              (double)P_exx_b[q * nao + r]);

                            const int sph_idx =
                                ((ci * nj + cj) * nk + ck) * nl + cl;
                            gamma_buf0[sph_idx] = (float)gamma;
                        }
                    }
                }
            }

            // Sph→Cart transform of gamma
            float* gamma_cart;
            if (is_spherical)
            {
                // Step 0→1: buf0[ns0,ns1,ns2,ns3] → buf1[nc0,ns1,ns2,ns3]
                grad_sph2cart_step(cart2sph_mat, nao_sph, off_cart_sh[0],
                                   off_eff[0], ni_cart, ni, 1, nj * nk * nl,
                                   gamma_buf0, gamma_buf1);
                // Step 1→0: buf1 → buf0[nc0,nc1,ns2,ns3]
                grad_sph2cart_step(cart2sph_mat, nao_sph, off_cart_sh[1],
                                   off_eff[1], nj_cart, nj, ni_cart, nk * nl,
                                   gamma_buf1, gamma_buf0);
                // Step 0→1: buf0 → buf1[nc0,nc1,nc2,ns3]
                grad_sph2cart_step(cart2sph_mat, nao_sph, off_cart_sh[2],
                                   off_eff[2], nk_cart, nk, ni_cart * nj_cart,
                                   nl, gamma_buf0, gamma_buf1);
                // Step 1→0: buf1 → buf0[nc0,nc1,nc2,nc3]
                grad_sph2cart_step(
                    cart2sph_mat, nao_sph, off_cart_sh[3], off_eff[3], nl_cart,
                    nl, ni_cart * nj_cart * nk_cart, 1, gamma_buf1, gamma_buf0);
                gamma_cart = gamma_buf0;
            }
            else
            {
                gamma_cart = gamma_buf0;
            }

            float max_gamma = 0.0f;
            for (int i = 0; i < shell_size_cart; i++)
                max_gamma = fmaxf(max_gamma, fabsf(gamma_cart[i]));
            if (max_gamma < QC_GRAD_GAMMA_CUTOFF) continue;

            {
                const int ij_am = l[0] + l[1];
                const int kl_am = l[2] + l[3];
                const int nrys = (ij_am + kl_am + 3) / 2;

                // Extended strides for I arrays:
                // I[0..l0+1][0..l1+1][0..l2+1][0..l3]
                const int ix_d2 = (l[3] + 1);
                const int ix_d1 = (l[2] + 2) * ix_d2;
                const int ix_d0 = (l[1] + 2) * ix_d1;

                const float rab2 =
                    (RC[0][0] - RC[1][0]) * (RC[0][0] - RC[1][0]) +
                    (RC[0][1] - RC[1][1]) * (RC[0][1] - RC[1][1]) +
                    (RC[0][2] - RC[1][2]) * (RC[0][2] - RC[1][2]);
                const float rcd2 =
                    (RC[2][0] - RC[3][0]) * (RC[2][0] - RC[3][0]) +
                    (RC[2][1] - RC[3][1]) * (RC[2][1] - RC[3][1]) +
                    (RC[2][2] - RC[3][2]) * (RC[2][2] - RC[3][2]);
                const float AB[3] = {RC[0][0] - RC[1][0], RC[0][1] - RC[1][1],
                                     RC[0][2] - RC[1][2]};
                const float CD[3] = {RC[2][0] - RC[3][0], RC[2][1] - RC[3][1],
                                     RC[2][2] - RC[3][2]};

                // g_stride for extended VRR: (kl_am+2) columns
                const int g_stride = kl_am + 2;

                double g_A[3] = {0.0, 0.0, 0.0};
                double g_B[3] = {0.0, 0.0, 0.0};
                double g_C[3] = {0.0, 0.0, 0.0};

                // Primitive loop
                for (int ip = 0; ip < np[0]; ip++)
                {
                    const float ai = env[p_exp_off[0] + ip];
                    const float ci_v = env[p_cof_off[0] + ip];
                    const float two_ai = 2.0f * ai;
                    for (int jp = 0; jp < np[1]; jp++)
                    {
                        const float aj = env[p_exp_off[1] + jp];
                        const float p_val = ai + aj;
                        const float inv_p = 1.0f / p_val;
                        const float kab = expf(-(ai * aj * inv_p) * rab2);
                        const float n_ab = ci_v * env[p_cof_off[1] + jp] * kab;
                        if (fabsf(n_ab) < prim_screen_tol) continue;
                        const float two_aj = 2.0f * aj;
                        const float Px =
                            (ai * RC[0][0] + aj * RC[1][0]) * inv_p;
                        const float Py =
                            (ai * RC[0][1] + aj * RC[1][1]) * inv_p;
                        const float Pz =
                            (ai * RC[0][2] + aj * RC[1][2]) * inv_p;
                        const float PA[3] = {Px - RC[0][0], Py - RC[0][1],
                                             Pz - RC[0][2]};

                        for (int kp = 0; kp < np[2]; kp++)
                        {
                            const float ak = env[p_exp_off[2] + kp];
                            const float ck = env[p_cof_off[2] + kp];
                            const float two_ak = 2.0f * ak;
                            for (int lp = 0; lp < np[3]; lp++)
                            {
                                const float al = env[p_exp_off[3] + lp];
                                const float q_val = ak + al;
                                const float inv_q = 1.0f / q_val;
                                const float kcd =
                                    expf(-(ak * al * inv_q) * rcd2);
                                const float pref =
                                    2.0f * PI_25 /
                                    (p_val * q_val * sqrtf(p_val + q_val));
                                const float n_abcd = n_ab * ck *
                                                     env[p_cof_off[3] + lp] *
                                                     kcd * pref;
                                if (fabsf(n_abcd) < prim_screen_tol) continue;

                                const float Qx =
                                    (ak * RC[2][0] + al * RC[3][0]) * inv_q;
                                const float Qy =
                                    (ak * RC[2][1] + al * RC[3][1]) * inv_q;
                                const float Qz =
                                    (ak * RC[2][2] + al * RC[3][2]) * inv_q;
                                const float QCv[3] = {Qx - RC[2][0],
                                                      Qy - RC[2][1],
                                                      Qz - RC[2][2]};
                                const float PQ[3] = {Px - Qx, Py - Qy, Pz - Qz};
                                const float rho =
                                    p_val * q_val / (p_val + q_val);
                                const float T =
                                    rho * (PQ[0] * PQ[0] + PQ[1] * PQ[1] +
                                           PQ[2] * PQ[2]);

                                double rys_r[12], rys_w[12];
                                rys_roots_weights(nrys, (double)T, rys_r,
                                                  rys_w);

                                // Per-root: VRR(extended) → HRR(extended) →
                                // contract with gamma
                                for (int ir = 0; ir < nrys; ir++)
                                {
                                    const float u = (float)rys_r[ir];
                                    const float w = (float)rys_w[ir];
                                    const float factor = u / (p_val + q_val);
                                    const float B00 = 0.5f * factor;
                                    const float B10 =
                                        0.5f / p_val * (1.0f - q_val * factor);
                                    const float B01 =
                                        0.5f / q_val * (1.0f - p_val * factor);

                                    const float Cx_bra[3] = {
                                        PA[0] - factor * q_val * PQ[0],
                                        PA[1] - factor * q_val * PQ[1],
                                        PA[2] - factor * q_val * PQ[2]};
                                    const float Cx_ket[3] = {
                                        QCv[0] + factor * p_val * PQ[0],
                                        QCv[1] + factor * p_val * PQ[1],
                                        QCv[2] + factor * p_val * PQ[2]};

                                    // Extended VRR: up to (ij_am+1, kl_am+1)
                                    float Gx[120], Gy[120], Gz[120];
                                    rys_vrr_2d(Gx, ij_am + 1, kl_am + 1,
                                               g_stride, Cx_bra[0], Cx_ket[0],
                                               B00, B10, B01);
                                    rys_vrr_2d(Gy, ij_am + 1, kl_am + 1,
                                               g_stride, Cx_bra[1], Cx_ket[1],
                                               B00, B10, B01);
                                    rys_vrr_2d(Gz, ij_am + 1, kl_am + 1,
                                               g_stride, Cx_bra[2], Cx_ket[2],
                                               B00, B10, B01);

                                    float Ix[1100], Iy[1100], Iz[1100];
                                    grad_factored_hrr_batch(
                                        Gx, ij_am, kl_am, g_stride, l, AB[0],
                                        CD[0], Ix, ix_d0, ix_d1, ix_d2);
                                    grad_factored_hrr_batch(
                                        Gy, ij_am, kl_am, g_stride, l, AB[1],
                                        CD[1], Iy, ix_d0, ix_d1, ix_d2);
                                    grad_factored_hrr_batch(
                                        Gz, ij_am, kl_am, g_stride, l, AB[2],
                                        CD[2], Iz, ix_d0, ix_d1, ix_d2);

                                    // Contract derivatives with gamma
                                    const float wn = n_abcd * w;
                                    int idx = 0;
                                    for (int c0 = 0; c0 < ni_cart; c0++)
                                    {
                                        int i0x, i0y, i0z;
                                        QC_Get_Lxyz_Device(l[0], c0, i0x, i0y,
                                                           i0z);
                                        for (int c1 = 0; c1 < nj_cart; c1++)
                                        {
                                            int i1x, i1y, i1z;
                                            QC_Get_Lxyz_Device(l[1], c1, i1x,
                                                               i1y, i1z);
                                            for (int c2 = 0; c2 < nk_cart; c2++)
                                            {
                                                int i2x, i2y, i2z;
                                                QC_Get_Lxyz_Device(
                                                    l[2], c2, i2x, i2y, i2z);
                                                const int bx_base =
                                                    i0x * ix_d0 + i1x * ix_d1 +
                                                    i2x * ix_d2;
                                                const int by_base =
                                                    i0y * ix_d0 + i1y * ix_d1 +
                                                    i2y * ix_d2;
                                                const int bz_base =
                                                    i0z * ix_d0 + i1z * ix_d1 +
                                                    i2z * ix_d2;

                                                for (int c3 = 0; c3 < nl_cart;
                                                     c3++, idx++)
                                                {
                                                    const float g =
                                                        gamma_cart[idx];
                                                    if (g == 0.0f) continue;

                                                    int i3x, i3y, i3z;
                                                    QC_Get_Lxyz_Device(l[3], c3,
                                                                       i3x, i3y,
                                                                       i3z);
                                                    const int bx =
                                                        bx_base + i3x;
                                                    const int by =
                                                        by_base + i3y;
                                                    const int bz =
                                                        bz_base + i3z;

                                                    const double gwn =
                                                        (double)g * (double)wn;

                                                    // X derivatives
                                                    const double yz =
                                                        (double)Iy[by] *
                                                        (double)Iz[bz];
                                                    float cAx =
                                                        two_ai * Ix[bx + ix_d0];
                                                    if (i0x > 0)
                                                        cAx -= (float)i0x *
                                                               Ix[bx - ix_d0];
                                                    float cBx =
                                                        two_aj * Ix[bx + ix_d1];
                                                    if (i1x > 0)
                                                        cBx -= (float)i1x *
                                                               Ix[bx - ix_d1];
                                                    float cCx =
                                                        two_ak * Ix[bx + ix_d2];
                                                    if (i2x > 0)
                                                        cCx -= (float)i2x *
                                                               Ix[bx - ix_d2];
                                                    const double fxyz =
                                                        gwn * yz;
                                                    g_A[0] +=
                                                        fxyz * (double)cAx;
                                                    g_B[0] +=
                                                        fxyz * (double)cBx;
                                                    g_C[0] +=
                                                        fxyz * (double)cCx;

                                                    // Y derivatives
                                                    const double xz =
                                                        (double)Ix[bx] *
                                                        (double)Iz[bz];
                                                    float cAy =
                                                        two_ai * Iy[by + ix_d0];
                                                    if (i0y > 0)
                                                        cAy -= (float)i0y *
                                                               Iy[by - ix_d0];
                                                    float cBy =
                                                        two_aj * Iy[by + ix_d1];
                                                    if (i1y > 0)
                                                        cBy -= (float)i1y *
                                                               Iy[by - ix_d1];
                                                    float cCy =
                                                        two_ak * Iy[by + ix_d2];
                                                    if (i2y > 0)
                                                        cCy -= (float)i2y *
                                                               Iy[by - ix_d2];
                                                    const double fxyz_y =
                                                        gwn * xz;
                                                    g_A[1] +=
                                                        fxyz_y * (double)cAy;
                                                    g_B[1] +=
                                                        fxyz_y * (double)cBy;
                                                    g_C[1] +=
                                                        fxyz_y * (double)cCy;

                                                    // Z derivatives
                                                    const double xy =
                                                        (double)Ix[bx] *
                                                        (double)Iy[by];
                                                    float cAz =
                                                        two_ai * Iz[bz + ix_d0];
                                                    if (i0z > 0)
                                                        cAz -= (float)i0z *
                                                               Iz[bz - ix_d0];
                                                    float cBz =
                                                        two_aj * Iz[bz + ix_d1];
                                                    if (i1z > 0)
                                                        cBz -= (float)i1z *
                                                               Iz[bz - ix_d1];
                                                    float cCz =
                                                        two_ak * Iz[bz + ix_d2];
                                                    if (i2z > 0)
                                                        cCz -= (float)i2z *
                                                               Iz[bz - ix_d2];
                                                    const double fxyz_z =
                                                        gwn * xy;
                                                    g_A[2] +=
                                                        fxyz_z * (double)cAz;
                                                    g_B[2] +=
                                                        fxyz_z * (double)cBz;
                                                    g_C[2] +=
                                                        fxyz_z * (double)cCz;
                                                }
                                            }
                                        }
                                    }
                                }  // end Rys roots
                            }
                        }
                    }
                }  // end primitives

                // Translational invariance: g_D = -(g_A + g_B + g_C)
                for (int d = 0; d < 3; d++)
                {
                    atomicAdd(&grad_local[atom_A * 3 + d], g_A[d]);
                    atomicAdd(&grad_local[atom_B * 3 + d], g_B[d]);
                    atomicAdd(&grad_local[atom_C * 3 + d], g_C[d]);
                    atomicAdd(&grad_local[atom_D * 3 + d],
                              -(g_A[d] + g_B[d] + g_C[d]));
                }
            }
        }  // screening
    }
}

// Reduce N gradient copies into grad: grad[i] += sum of copies[c][i]
static __global__ void QC_Reduce_Grad_Copies_Kernel(
    const int n, const int n_copies, const double* __restrict__ copies,
    double* __restrict__ grad)
{
    SIMPLE_DEVICE_FOR(i, n)
    {
        double sum = 0.0;
        for (int c = 0; c < n_copies; c++) sum += copies[(size_t)c * n + i];
        grad[i] += sum;
    }
}

#endif  // USE_GPU

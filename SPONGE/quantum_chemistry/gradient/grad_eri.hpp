#pragma once

#include <cstdio>
#include <cstdlib>
#include <vector>

// 依赖: 此文件需要在 scf/build_fock.hpp 之后 include
// (build_fock.hpp 带入 direct_fock.hpp 和完整 ERI 基础设施)

// ====================== 双电子积分梯度 ======================
// dE_2e/dR_A = Σ_{pqrs} Γ_eff(pqrs) × d(pq|rs)/dR_A
//
// Γ_eff = 4·P_pq·P_rs − exx·(P_pr·P_qs + P_ps·P_qr)
// (此闭合公式适用于所有对称情况，包括退化 shell quartet)
//
// d(pq|rs)/dA_x 使用 E 系数翻译递推:
//   dE^{ab}_t/dA_x = 2αi·E^{(a+1)b}_t − a_x·E^{(a-1)b}_t
// 需要: E_bra at (l0+1, l1+1), E_ket at (l2+1, l3+1), HR at L_sum+1
// ==============================================================

#ifndef USE_GPU

// 扩展的 Bra Prim Cache: E 系数算到 (l0+1, l1+1) 以支持 d/dA 和 d/dB
struct QC_Bra_Prim_Cache_Grad_CPU
{
    float ai, aj;          // 原始指数 (用于 dE/dA, dE/dB)
    float P[3];
    float AB[3];
    float inv_p;
    float n_ab;
    float E_bra[3][5][5][9]; // E at (l0+1, l1+1)
};

static inline void QC_Build_Bra_Prim_Cache_Grad_CPU(
    const QC_Shell_Pair_Meta_CPU& bra, const float* env,
    const float prim_screen_tol,
    std::vector<QC_Bra_Prim_Cache_Grad_CPU>& prims)
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

            QC_Bra_Prim_Cache_Grad_CPU prim = {};
            prim.ai = ai;
            prim.aj = aj;
            prim.AB[0] = bra.R[0][0] - bra.R[1][0];
            prim.AB[1] = bra.R[0][1] - bra.R[1][1];
            prim.AB[2] = bra.R[0][2] - bra.R[1][2];
            prim.P[0] = (ai * bra.R[0][0] + aj * bra.R[1][0]) * inv_p;
            prim.P[1] = (ai * bra.R[0][1] + aj * bra.R[1][1]) * inv_p;
            prim.P[2] = (ai * bra.R[0][2] + aj * bra.R[1][2]) * inv_p;
            prim.inv_p = inv_p;
            prim.n_ab = n_ab;
            // E at (l0+1, l1+1) — 支持 dE/dA (需要 l0+1) 和 dE/dB (需要 l1+1)
            const int la_up = std::min(bra.l[0] + 1, 4);
            const int lb_up = std::min(bra.l[1] + 1, 4);
            for (int d = 0; d < 3; d++)
            {
                compute_md_coeffs(prim.E_bra[d], la_up, lb_up,
                                  prim.P[d] - bra.R[0][d],
                                  prim.P[d] - bra.R[1][d], 0.5f * inv_p);
            }
            prims.push_back(prim);
        }
    }
}

// 计算一个 AO quartet 的 9 个导数分量 (d/dA_xyz, d/dB_xyz, d/dC_xyz)
// 直接对 E 系数求和，不做 angular term 预计算
static inline void QC_Compute_AO_Quartet_Deriv(
    const QC_Bra_Prim_Cache_Grad_CPU& bra_prim,
    const float E_ket[3][5][5][9], const float ak,
    const int ix, const int jx, const int iy, const int jy,
    const int iz, const int jz,
    const int kx, const int lx, const int ky, const int ly,
    const int kz, const int lz,
    const float* HR, const int hr_base, const float n_abcd,
    float d_A[3], float d_B[3], float d_C[3])
{
    const float ai = bra_prim.ai;
    const float aj = bra_prim.aj;
    const auto& E = bra_prim.E_bra;

    d_A[0] = d_A[1] = d_A[2] = 0.0f;
    d_B[0] = d_B[1] = d_B[2] = 0.0f;
    d_C[0] = d_C[1] = d_C[2] = 0.0f;

    const int hr_stride_z = hr_base;
    const int hr_stride_y = hr_base * hr_base;
    const int hr_stride_x = hr_base * hr_base * hr_base;

    // Contract E_bra * E_ket * (-1)^{ket} * R for a given set of angular momenta.
    // The standard MD derivative formula d(ab|cd)/dA_x = 2·ai·(a+1,b|cd) - a·(a-1,b|cd)
    // is COMPLETE: E^{(a+1,b)}_t at higher Hermite index t naturally picks up
    // the R-tensor chain rule through PQ, so NO explicit HR shift is needed.
    auto contract_quartet = [&](const int aix, const int ajx, const int aiy,
                                const int ajy, const int aiz, const int ajz,
                                const int akx, const int alx, const int aky,
                                const int aly, const int akz, const int alz)
    {
        if (aix < 0 || ajx < 0 || aiy < 0 || ajy < 0 || aiz < 0 || ajz < 0 ||
            akx < 0 || alx < 0 || aky < 0 || aly < 0 || akz < 0 || alz < 0)
            return 0.0f;
        if (aix >= 5 || ajx >= 5 || aiy >= 5 || ajy >= 5 || aiz >= 5 ||
            ajz >= 5 || akx >= 5 || alx >= 5 || aky >= 5 || aly >= 5 ||
            akz >= 5 || alz >= 5)
            return 0.0f;

        const float* ex_bra = E[0][aix][ajx];
        const float* ey_bra = E[1][aiy][ajy];
        const float* ez_bra = E[2][aiz][ajz];
        const float* ex_ket = E_ket[0][akx][alx];
        const float* ey_ket = E_ket[1][aky][aly];
        const float* ez_ket = E_ket[2][akz][alz];

        const int bra_max_x = aix + ajx;
        const int bra_max_y = aiy + ajy;
        const int bra_max_z = aiz + ajz;
        const int ket_max_x = akx + alx;
        const int ket_max_y = aky + aly;
        const int ket_max_z = akz + alz;

        double val = 0.0;
        for (int mx = 0; mx <= bra_max_x; mx++)
        {
            for (int my = 0; my <= bra_max_y; my++)
            {
                for (int mz = 0; mz <= bra_max_z; mz++)
                {
                    const float e_bra =
                        ex_bra[mx] * ey_bra[my] * ez_bra[mz];
                    if (fabsf(e_bra) < 1e-30f) continue;
                    for (int nx = 0; nx <= ket_max_x; nx++)
                    {
                        for (int ny = 0; ny <= ket_max_y; ny++)
                        {
                            for (int nz = 0; nz <= ket_max_z; nz++)
                            {
                                const float e_ket =
                                    ex_ket[nx] * ey_ket[ny] * ez_ket[nz];
                                if (fabsf(e_ket) < 1e-30f) continue;
                                const float phase =
                                    ((nx + ny + nz) & 1) ? -1.0f : 1.0f;
                                const int hr_idx =
                                    (mx + nx) * hr_stride_x +
                                    (my + ny) * hr_stride_y +
                                    (mz + nz) * hr_stride_z;
                                val += (double)e_bra * (double)e_ket *
                                       (double)(HR[hr_idx] * phase);
                            }
                        }
                    }
                }
            }
        }
        return (float)val;
    };

    // d(ab|cd)/dA_x = 2·ai·(a_x+1,b|cd) - a_x·(a_x-1,b|cd)
    d_A[0] = 2.0f * ai * contract_quartet(ix + 1, jx, iy, jy, iz, jz, kx, lx,
                                           ky, ly, kz, lz);
    if (ix > 0)
        d_A[0] -= (float)ix * contract_quartet(ix - 1, jx, iy, jy, iz, jz, kx,
                                               lx, ky, ly, kz, lz);
    d_A[1] = 2.0f * ai * contract_quartet(ix, jx, iy + 1, jy, iz, jz, kx, lx,
                                           ky, ly, kz, lz);
    if (iy > 0)
        d_A[1] -= (float)iy * contract_quartet(ix, jx, iy - 1, jy, iz, jz, kx,
                                               lx, ky, ly, kz, lz);
    d_A[2] = 2.0f * ai * contract_quartet(ix, jx, iy, jy, iz + 1, jz, kx, lx,
                                           ky, ly, kz, lz);
    if (iz > 0)
        d_A[2] -= (float)iz * contract_quartet(ix, jx, iy, jy, iz - 1, jz, kx,
                                               lx, ky, ly, kz, lz);

    // d(ab|cd)/dB_x = 2·aj·(a,b_x+1|cd) - b_x·(a,b_x-1|cd)
    d_B[0] = 2.0f * aj * contract_quartet(ix, jx + 1, iy, jy, iz, jz, kx, lx,
                                           ky, ly, kz, lz);
    if (jx > 0)
        d_B[0] -= (float)jx * contract_quartet(ix, jx - 1, iy, jy, iz, jz, kx,
                                               lx, ky, ly, kz, lz);
    d_B[1] = 2.0f * aj * contract_quartet(ix, jx, iy, jy + 1, iz, jz, kx, lx,
                                           ky, ly, kz, lz);
    if (jy > 0)
        d_B[1] -= (float)jy * contract_quartet(ix, jx, iy, jy - 1, iz, jz, kx,
                                               lx, ky, ly, kz, lz);
    d_B[2] = 2.0f * aj * contract_quartet(ix, jx, iy, jy, iz, jz + 1, kx, lx,
                                           ky, ly, kz, lz);
    if (jz > 0)
        d_B[2] -= (float)jz * contract_quartet(ix, jx, iy, jy, iz, jz - 1, kx,
                                               lx, ky, ly, kz, lz);

    // d(ab|cd)/dC_x = 2·ak·(ab|c_x+1,d) - c_x·(ab|c_x-1,d)
    d_C[0] = 2.0f * ak * contract_quartet(ix, jx, iy, jy, iz, jz, kx + 1, lx,
                                           ky, ly, kz, lz);
    if (kx > 0)
        d_C[0] -= (float)kx * contract_quartet(ix, jx, iy, jy, iz, jz, kx - 1,
                                               lx, ky, ly, kz, lz);
    d_C[1] = 2.0f * ak * contract_quartet(ix, jx, iy, jy, iz, jz, kx, lx,
                                           ky + 1, ly, kz, lz);
    if (ky > 0)
        d_C[1] -= (float)ky * contract_quartet(ix, jx, iy, jy, iz, jz, kx, lx,
                                               ky - 1, ly, kz, lz);
    d_C[2] = 2.0f * ak * contract_quartet(ix, jx, iy, jy, iz, jz, kx, lx,
                                           ky, ly, kz + 1, lz);
    if (kz > 0)
        d_C[2] -= (float)kz * contract_quartet(ix, jx, iy, jy, iz, jz, kx, lx,
                                               ky, ly, kz - 1, lz);

    // 乘以归一化前因子
    for (int d = 0; d < 3; d++)
    {
        d_A[d] *= n_abcd;
        d_B[d] *= n_abcd;
        d_C[d] *= n_abcd;
    }
}

static inline void QC_Build_ERI_Gradient_CPU(
    const QC_INTEGRAL_TASKS& task_ctx, const int nbas, const int* atm,
    const int* bas, const float* env, const int* ao_offsets_cart,
    const int* ao_offsets_sph, const float* norms,
    const float* shell_pair_bounds, const float* pair_density_coul,
    const float* pair_density_exx_a, const float* pair_density_exx_b,
    const float shell_screen_tol, const float* P_coul,
    const float* P_exx_a, const float* P_exx_b,
    const float exx_scale_a, const float exx_scale_b, const int nao,
    const int nao_sph, const int is_spherical, const float* cart2sph_mat,
    const int* shell_atom, double* grad, int hr_base, int hr_size,
    int shell_buf_size, float prim_screen_tol, const int thread_count)
{
    const bool debug_small_eri_grad =
        (nao <= 2 && std::getenv("SPONGE_DEBUG_ERI_GRAD") != nullptr);
    int debug_natm = 0;
    if (debug_small_eri_grad)
    {
        for (int i = 0; i < nbas; i++)
            if (shell_atom[i] + 1 > debug_natm) debug_natm = shell_atom[i] + 1;
    }
    std::vector<double> debug_grad_before;
    if (debug_small_eri_grad)
        debug_grad_before.assign(grad, grad + (size_t)debug_natm * 3);
    const int n_pairs = task_ctx.topo.n_shell_pairs;
    if (n_pairs <= 0) return;

    // 预计算 shell pair metadata (同 Fock build)
    std::vector<QC_Shell_Pair_Meta_CPU> pair_meta((size_t)n_pairs);
    for (int pair_id = 0; pair_id < n_pairs; pair_id++)
    {
        QC_Init_Shell_Pair_Meta_CPU(task_ctx.topo.h_shell_pairs[pair_id], atm,
                                    bas, env, ao_offsets_cart, ao_offsets_sph,
                                    is_spherical, pair_meta[(size_t)pair_id]);
    }

    // 预计算 anchor activity (同 Fock build)
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

    const int natm = (int)(std::count_if(
        shell_atom, shell_atom + nbas,
        [](int a) { return a >= 0; }));
    // 获取最大原子索引
    int natm_max = 0;
    for (int i = 0; i < nbas; i++)
        natm_max = std::max(natm_max, shell_atom[i] + 1);

#pragma omp parallel num_threads(thread_count)
    {
        // 线程私有梯度累加器
        std::vector<double> grad_local((size_t)natm_max * 3, 0.0);
        // Gradient needs HR at order L_sum+1, one higher than the ERI itself.
        // hr_base was sized for L_sum, so we use grad_hr_base = hr_base + 1.
        const int grad_hr_base = hr_base + 1;
        const int grad_hr_size =
            grad_hr_base * grad_hr_base * grad_hr_base * grad_hr_base;
        float* HR = (float*)malloc(sizeof(float) * (size_t)grad_hr_size);
        std::vector<int> partner_marks((size_t)n_pairs, -1);
        std::vector<int> candidate_partners;
        candidate_partners.reserve(256);
        std::vector<QC_Bra_Prim_Cache_Grad_CPU> bra_prims;

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

            QC_Build_Bra_Prim_Cache_Grad_CPU(bra, env, prim_screen_tol,
                                              bra_prims);
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

                const QC_ONE_E_TASK& kl =
                    task_ctx.topo.h_shell_pairs[pair_kl];
                const QC_Shell_Pair_Meta_CPU& ket =
                    pair_meta[(size_t)pair_kl];
                const int atom_C = shell_atom[kl.x];
                const int atom_D = shell_atom[kl.y];

                const int l[4] = {bra.l[0], bra.l[1], ket.l[0], ket.l[1]};
                const int L_sum = l[0] + l[1] + l[2] + l[3];

                // 壳层退化判断
                const bool jk_same_bra = (ij.x == ij.y);
                const bool jk_same_ket = (kl.x == kl.y);
                const bool jk_same_braket =
                    (ij.x == kl.x && ij.y == kl.y);

                // 导数积分壳层缓冲 [ni*nj*nk*nl][3] for A, B, C
                const int ni = bra.dims_eff[0], nj = bra.dims_eff[1];
                const int nk = ket.dims_eff[0], nl = ket.dims_eff[1];
                const int shell_size = ni * nj * nk * nl;

                // 在栈上分配导数缓冲 (3 centers × 3 directions = 9 buffers)
                std::vector<float> d_buf_A(shell_size * 3, 0.0f);
                std::vector<float> d_buf_B(shell_size * 3, 0.0f);
                std::vector<float> d_buf_C(shell_size * 3, 0.0f);

                const float p0 = 1.0f / bra_prims[0].inv_p; // rough estimate
                float E_ket[3][5][5][9];
                const int lc_up = std::min(ket.l[0] + 1, 4);
                const int ld_up = std::min(ket.l[1] + 1, 4);

                // 对所有原始函数组合求和
                for (const auto& bra_prim : bra_prims)
                {
                    const float p = 1.0f / bra_prim.inv_p;
                    for (int kp = 0; kp < ket.np[0]; kp++)
                    {
                        for (int lp = 0; lp < ket.np[1]; lp++)
                        {
                            const float ak = env[ket.p_exp[0] + kp];
                            const float al = env[ket.p_exp[1] + lp];
                            const float q = ak + al;
                            const float inv_q = 1.0f / q;
                            const float kcd =
                                expf(-(ak * al * inv_q) * ket.pair_dist2);
                            const float pref =
                                2.0f * PI_25 / (p * q * sqrtf(p + q));
                            const float n_abcd =
                                bra_prim.n_ab * env[ket.p_cof[0] + kp] *
                                env[ket.p_cof[1] + lp] * kcd * pref;
                            if (fabsf(n_abcd) < prim_screen_tol) continue;

                            float Q[3] = {
                                (ak * ket.R[0][0] + al * ket.R[1][0]) * inv_q,
                                (ak * ket.R[0][1] + al * ket.R[1][1]) * inv_q,
                                (ak * ket.R[0][2] + al * ket.R[1][2]) * inv_q};
                            const float CD[3] = {
                                ket.R[0][0] - ket.R[1][0],
                                ket.R[0][1] - ket.R[1][1],
                                ket.R[0][2] - ket.R[1][2]};
                            const float alpha = p * q / (p + q);
                            float PQ[3] = {bra_prim.P[0] - Q[0],
                                           bra_prim.P[1] - Q[1],
                                           bra_prim.P[2] - Q[2]};
                            float t_arg = alpha * (PQ[0] * PQ[0] +
                                                   PQ[1] * PQ[1] +
                                                   PQ[2] * PQ[2]);

                            // HR at L_sum+1 (一阶更高)
                            // 必须清零: 导数循环可能读到 L_sum+1 以外的索引
                            // (E系数为0所以乘积为0, 但未初始化的HR可能含NaN)
                            memset(HR, 0, sizeof(float) * (size_t)grad_hr_size);
                            compute_hr_tensor(HR, alpha, PQ, L_sum + 1,
                                              grad_hr_base, t_arg);

                            // E_ket at (l2+1, l3+1)
                            for (int d = 0; d < 3; d++)
                                compute_md_coeffs(E_ket[d], lc_up, ld_up,
                                                  Q[d] - ket.R[0][d],
                                                  Q[d] - ket.R[1][d],
                                                  0.5f * inv_q);

                            // 对每个 AO 组合计算导数
                            for (int ci = 0; ci < ni; ci++)
                            {
                                const int ix = bra.comp_x[0][ci];
                                const int iy = bra.comp_y[0][ci];
                                const int iz = bra.comp_z[0][ci];
                                for (int cj = 0; cj < nj; cj++)
                                {
                                    const int jx = bra.comp_x[1][cj];
                                    const int jy = bra.comp_y[1][cj];
                                    const int jz = bra.comp_z[1][cj];
                                    for (int ck = 0; ck < nk; ck++)
                                    {
                                        const int kx2 = ket.comp_x[0][ck];
                                        const int ky2 = ket.comp_y[0][ck];
                                        const int kz2 = ket.comp_z[0][ck];
                                        for (int cl = 0; cl < nl; cl++)
                                        {
                                            const int lx2 =
                                                ket.comp_x[1][cl];
                                            const int ly2 =
                                                ket.comp_y[1][cl];
                                            const int lz2 =
                                                ket.comp_z[1][cl];

                                            float dA[3], dB[3], dC[3];
                                            QC_Compute_AO_Quartet_Deriv(
                                                bra_prim, E_ket, ak, ix,
                                                jx, iy, jy, iz, jz, kx2, lx2,
                                                ky2, ly2, kz2, lz2, HR,
                                                grad_hr_base, n_abcd, dA, dB, dC);

                                            const int idx =
                                                ((ci * nj + cj) * nk + ck) *
                                                    nl +
                                                cl;
                                            for (int d = 0; d < 3; d++)
                                            {
                                                d_buf_A[idx * 3 + d] += dA[d];
                                                d_buf_B[idx * 3 + d] += dB[d];
                                                d_buf_C[idx * 3 + d] += dC[d];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // 应用归一化因子 (同 ERI buffer)
                for (int ci = 0; ci < ni; ci++)
                {
                    const float norm_i = norms[bra.off_eff[0] + ci];
                    for (int cj = 0; cj < nj; cj++)
                    {
                        const float nij = norm_i * norms[bra.off_eff[1] + cj];
                        for (int ck = 0; ck < nk; ck++)
                        {
                            const float nijk =
                                nij * norms[ket.off_eff[0] + ck];
                            for (int cl = 0; cl < nl; cl++)
                            {
                                const float nijkl =
                                    nijk * norms[ket.off_eff[1] + cl];
                                const int idx =
                                    ((ci * nj + cj) * nk + ck) * nl + cl;
                                for (int d = 0; d < 3; d++)
                                {
                                    d_buf_A[idx * 3 + d] *= nijkl;
                                    d_buf_B[idx * 3 + d] *= nijkl;
                                    d_buf_C[idx * 3 + d] *= nijkl;
                                }
                            }
                        }
                    }
                }

                // 与密度矩阵收缩，累加到梯度
                for (int ci = 0; ci < ni; ci++)
                {
                    const int p = bra.off_eff[0] + ci;
                    const int pn = p * nao;
                    for (int cj = 0; cj < nj; cj++)
                    {
                        const int q_idx = bra.off_eff[1] + cj;
                        const int qn = q_idx * nao;
                        if (jk_same_bra && q_idx > p) continue;
                        for (int ck = 0; ck < nk; ck++)
                        {
                            const int r = ket.off_eff[0] + ck;
                            const int rn = r * nao;
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

                                const int idx =
                                    ((ci * nj + cj) * nk + ck) * nl + cl;
                                const float* dA = &d_buf_A[idx * 3];
                                const float* dB = &d_buf_B[idx * 3];
                                const float* dC = &d_buf_C[idx * 3];

                                const int ao_idx[4] = {p, q_idx, r, s};
                                const int atom_idx[4] = {atom_A, atom_B,
                                                         atom_C, atom_D};
                                const double d_slot[4][3] = {
                                    {(double)dA[0], (double)dA[1],
                                     (double)dA[2]},
                                    {(double)dB[0], (double)dB[1],
                                     (double)dB[2]},
                                    {(double)dC[0], (double)dC[1],
                                     (double)dC[2]},
                                    {-(double)dA[0] - (double)dB[0] -
                                         (double)dC[0],
                                     -(double)dA[1] - (double)dB[1] -
                                         (double)dC[1],
                                     -(double)dA[2] - (double)dB[2] -
                                         (double)dC[2]}};

                                double gamma_j = 0.0;
                                double gamma_k = 0.0;
                                double g_atom_debug[4][3] = {};

                                const int jt_slot[8][4] = {
                                    {0, 1, 2, 3}, {1, 0, 2, 3},
                                    {0, 1, 3, 2}, {1, 0, 3, 2},
                                    {2, 3, 0, 1}, {3, 2, 0, 1},
                                    {2, 3, 1, 0}, {3, 2, 1, 0}};
                                for (int n = 0; n < 8; n++)
                                {
                                    const int i0 = ao_idx[jt_slot[n][0]];
                                    const int i1 = ao_idx[jt_slot[n][1]];
                                    const int i2 = ao_idx[jt_slot[n][2]];
                                    const int i3 = ao_idx[jt_slot[n][3]];
                                    bool dup = false;
                                    for (int pv = 0; pv < n; pv++)
                                    {
                                        const int p0 = ao_idx[jt_slot[pv][0]];
                                        const int p1 = ao_idx[jt_slot[pv][1]];
                                        const int p2 = ao_idx[jt_slot[pv][2]];
                                        const int p3 = ao_idx[jt_slot[pv][3]];
                                        if (i0 == p0 && i1 == p1 && i2 == p2 &&
                                            i3 == p3)
                                        {
                                            dup = true;
                                            break;
                                        }
                                    }
                                    if (dup) continue;

                                    const double weight =
                                        0.5 * (double)P_coul[i0 * nao + i1] *
                                        (double)P_coul[i2 * nao + i3];
                                    gamma_j += weight;
                                    for (int slot = 0; slot < 4; slot++)
                                    {
                                        const int src = jt_slot[n][slot];
                                        const int atom = atom_idx[src];
                                        for (int d = 0; d < 3; d++)
                                        {
                                            const double contrib =
                                                weight * d_slot[src][d];
                                            grad_local[atom * 3 + d] += contrib;
                                            g_atom_debug[src][d] += contrib;
                                        }
                                    }
                                }

                                if (exx_scale_a != 0.0f)
                                {
                                    const int kt_slot[8][4] = {
                                        {0, 2, 1, 3}, {0, 3, 1, 2},
                                        {1, 2, 0, 3}, {1, 3, 0, 2},
                                        {2, 0, 3, 1}, {2, 1, 3, 0},
                                        {3, 0, 2, 1}, {3, 1, 2, 0}};
                                    for (int n = 0; n < 8; n++)
                                    {
                                        const int i0 = ao_idx[kt_slot[n][0]];
                                        const int i1 = ao_idx[kt_slot[n][1]];
                                        const int i2 = ao_idx[kt_slot[n][2]];
                                        const int i3 = ao_idx[kt_slot[n][3]];
                                        bool dup = false;
                                        for (int pv = 0; pv < n; pv++)
                                        {
                                            const int p0 =
                                                ao_idx[kt_slot[pv][0]];
                                            const int p1 =
                                                ao_idx[kt_slot[pv][1]];
                                            const int p2 =
                                                ao_idx[kt_slot[pv][2]];
                                            const int p3 =
                                                ao_idx[kt_slot[pv][3]];
                                            if (i0 == p0 && i1 == p1 &&
                                                i2 == p2 && i3 == p3)
                                            {
                                                dup = true;
                                                break;
                                            }
                                        }
                                        if (dup) continue;

                                        const double weight =
                                            -0.5 * (double)exx_scale_a *
                                            (double)P_exx_a[i0 * nao + i1] *
                                            (double)P_exx_a[i2 * nao + i3];
                                        gamma_k += weight;
                                        for (int slot = 0; slot < 4; slot++)
                                        {
                                            const int src = kt_slot[n][slot];
                                            const int atom = atom_idx[src];
                                            for (int d = 0; d < 3; d++)
                                            {
                                                const double contrib =
                                                    weight * d_slot[src][d];
                                                grad_local[atom * 3 + d] +=
                                                    contrib;
                                                g_atom_debug[src][d] +=
                                                    contrib;
                                            }
                                        }
                                    }
                                }

                                if (debug_small_eri_grad)
                                {
                                    std::fprintf(
                                        stderr,
                                        "ERI_GRAD pqrs=(%d,%d|%d,%d) atoms=(%d,%d,%d,%d) "
                                        "gamma=(J=% .12e,K=% .12e,T=% .12e)\n"
                                        "  dA=(% .12e,% .12e,% .12e)\n"
                                        "  dB=(% .12e,% .12e,% .12e)\n"
                                        "  dC=(% .12e,% .12e,% .12e)\n"
                                        "  dD=(% .12e,% .12e,% .12e)\n"
                                        "  gA=(% .12e,% .12e,% .12e)\n"
                                        "  gB=(% .12e,% .12e,% .12e)\n"
                                        "  gC=(% .12e,% .12e,% .12e)\n"
                                        "  gD=(% .12e,% .12e,% .12e)\n",
                                        p, q_idx, r, s, atom_A, atom_B, atom_C,
                                        atom_D, gamma_j, gamma_k,
                                        gamma_j + gamma_k, d_slot[0][0],
                                        d_slot[0][1], d_slot[0][2], d_slot[1][0],
                                        d_slot[1][1], d_slot[1][2], d_slot[2][0],
                                        d_slot[2][1], d_slot[2][2], d_slot[3][0],
                                        d_slot[3][1], d_slot[3][2],
                                        g_atom_debug[0][0], g_atom_debug[0][1],
                                        g_atom_debug[0][2], g_atom_debug[1][0],
                                        g_atom_debug[1][1], g_atom_debug[1][2],
                                        g_atom_debug[2][0], g_atom_debug[2][1],
                                        g_atom_debug[2][2], g_atom_debug[3][0],
                                        g_atom_debug[3][1], g_atom_debug[3][2]);
                                }
                            }
                        }
                    }
                }
            }
        }

        // 归约线程私有梯度
#pragma omp critical
        {
            for (size_t i = 0; i < grad_local.size(); i++)
                grad[i] += grad_local[i];
        }
        free(HR);
    }

    if (debug_small_eri_grad)
    {
        std::fprintf(stderr, "ERI_GRAD_TOTAL\n");
        for (int ia = 0; ia < debug_natm; ia++)
        {
            const double gx = grad[ia * 3 + 0] - debug_grad_before[(size_t)ia * 3 + 0];
            const double gy = grad[ia * 3 + 1] - debug_grad_before[(size_t)ia * 3 + 1];
            const double gz = grad[ia * 3 + 2] - debug_grad_before[(size_t)ia * 3 + 2];
            std::fprintf(stderr, "  atom %d : (% .12e,% .12e,% .12e)\n",
                         ia, gx, gy, gz);
        }
    }
}

#endif // USE_GPU

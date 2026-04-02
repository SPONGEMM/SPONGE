#pragma once

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

// =================== Optimized ERI gradient via R-tensor factorization =============
// Instead of 18 full 6-nested contract_quartet calls per AO quartet,
// precompute half-contracted intermediates:
//   R_ket[m] = sum_n E_ket(n) * (-1)^|n| * HR(m+n)  — reused across all (ci,cj)
//   R_bra[n] = sum_m E_bra(m) * HR(m+n)              — reused across all (ck,cl)
// Then each derivative is just a 3-nested contraction (~8x faster for p-p quartets).

// Precompute R_ket: half-contract HR with ket E-coefficients
static inline void QC_Precompute_R_ket(
    const float E_ket[3][5][5][9],
    int kx, int lx, int ky, int ly, int kz, int lz,
    const float* HR, int hr_base,
    int mx_max, int my_max, int mz_max,
    float* __restrict R, int r_yz, int r_z)
{
    const int ket_mx = kx + lx, ket_my = ky + ly, ket_mz = kz + lz;
    const int sx = hr_base * hr_base * hr_base;
    const int sy = hr_base * hr_base;
    const float* ekx = E_ket[0][kx][lx];
    const float* eky = E_ket[1][ky][ly];
    const float* ekz = E_ket[2][kz][lz];

    for (int mx = 0; mx <= mx_max; mx++)
        for (int my = 0; my <= my_max; my++)
            for (int mz = 0; mz <= mz_max; mz++)
            {
                double s = 0.0;
                for (int nx = 0; nx <= ket_mx; nx++)
                    for (int ny = 0; ny <= ket_my; ny++)
                    {
                        float exy = ekx[nx] * eky[ny];
                        if (fabsf(exy) < 1e-30f) continue;
                        for (int nz = 0; nz <= ket_mz; nz++)
                        {
                            float e3 = exy * ekz[nz];
                            if (fabsf(e3) < 1e-30f) continue;
                            float ph = ((nx + ny + nz) & 1) ? -1.0f : 1.0f;
                            s += (double)(e3 * ph *
                                          HR[(mx + nx) * sx + (my + ny) * sy +
                                             (mz + nz) * hr_base]);
                        }
                    }
                R[mx * r_yz + my * r_z + mz] = (float)s;
            }
}

// Precompute R_bra: half-contract HR with bra E-coefficients (no ket phase)
static inline void QC_Precompute_R_bra(
    const float E_bra[3][5][5][9],
    int ix, int jx, int iy, int jy, int iz, int jz,
    const float* HR, int hr_base,
    int nx_max, int ny_max, int nz_max,
    float* __restrict R, int r_yz, int r_z)
{
    const int bra_mx = ix + jx, bra_my = iy + jy, bra_mz = iz + jz;
    const int sx = hr_base * hr_base * hr_base;
    const int sy = hr_base * hr_base;
    const float* ebx = E_bra[0][ix][jx];
    const float* eby = E_bra[1][iy][jy];
    const float* ebz = E_bra[2][iz][jz];

    for (int nx = 0; nx <= nx_max; nx++)
        for (int ny = 0; ny <= ny_max; ny++)
            for (int nz = 0; nz <= nz_max; nz++)
            {
                double s = 0.0;
                for (int mx = 0; mx <= bra_mx; mx++)
                    for (int my = 0; my <= bra_my; my++)
                    {
                        float exy = ebx[mx] * eby[my];
                        if (fabsf(exy) < 1e-30f) continue;
                        for (int mz = 0; mz <= bra_mz; mz++)
                            s += (double)(exy * ebz[mz] *
                                          HR[(mx + nx) * sx + (my + ny) * sy +
                                             (mz + nz) * hr_base]);
                    }
                R[nx * r_yz + ny * r_z + nz] = (float)s;
            }
}

// Contract bra E-coeff with R_ket (no phase — already in R_ket)
static inline float QC_Contract_NoPhase(
    const float* ex, int mx, const float* ey, int my,
    const float* ez, int mz,
    const float* R, int r_yz, int r_z)
{
    double v = 0.0;
    for (int x = 0; x <= mx; x++)
        for (int y = 0; y <= my; y++)
            for (int z = 0; z <= mz; z++)
                v += (double)(ex[x] * ey[y] * ez[z]) *
                     (double)R[x * r_yz + y * r_z + z];
    return (float)v;
}

// Contract ket E-coeff with R_bra (includes (-1)^|n| phase)
static inline float QC_Contract_WithPhase(
    const float* ex, int mx, const float* ey, int my,
    const float* ez, int mz,
    const float* R, int r_yz, int r_z)
{
    double v = 0.0;
    for (int x = 0; x <= mx; x++)
        for (int y = 0; y <= my; y++)
            for (int z = 0; z <= mz; z++)
            {
                float ph = ((x + y + z) & 1) ? -1.0f : 1.0f;
                v += (double)(ex[x] * ey[y] * ez[z] * ph) *
                     (double)R[x * r_yz + y * r_z + z];
            }
    return (float)v;
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

#pragma omp parallel num_threads(thread_count)
    {
        std::vector<double> grad_local((size_t)natm_max * 3, 0.0);
        const int grad_hr_base = hr_base + 1;
        const int grad_hr_size =
            grad_hr_base * grad_hr_base * grad_hr_base * grad_hr_base;
        float* HR = (float*)malloc(sizeof(float) * (size_t)grad_hr_size);
        std::vector<int> partner_marks((size_t)n_pairs, -1);
        std::vector<int> candidate_partners;
        candidate_partners.reserve(256);
        std::vector<QC_Bra_Prim_Cache_Grad_CPU> bra_prims;
        std::vector<float> d_buf_A_cart, d_buf_B_cart, d_buf_C_cart;
        std::vector<float> d_buf_A_sph, d_buf_B_sph, d_buf_C_sph;
        std::vector<float> sph_buf0, sph_buf1;
        std::vector<float> R_ket_buf, R_bra_buf;

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

                const bool jk_same_bra = (ij.x == ij.y);
                const bool jk_same_ket = (kl.x == kl.y);
                const bool jk_same_braket =
                    (ij.x == kl.x && ij.y == kl.y);

                const int ni_cart = bra.dims_cart[0], nj_cart = bra.dims_cart[1];
                const int nk_cart = ket.dims_cart[0], nl_cart = ket.dims_cart[1];
                const int shell_size_cart =
                    ni_cart * nj_cart * nk_cart * nl_cart;

                const int ni = bra.dims_eff[0], nj = bra.dims_eff[1];
                const int nk = ket.dims_eff[0], nl = ket.dims_eff[1];
                const int shell_size_eff = ni * nj * nk * nl;

                d_buf_A_cart.assign((size_t)shell_size_cart * 3, 0.0f);
                d_buf_B_cart.assign((size_t)shell_size_cart * 3, 0.0f);
                d_buf_C_cart.assign((size_t)shell_size_cart * 3, 0.0f);

                float E_ket[3][5][5][9];
                const int lc_up = std::min(ket.l[0] + 1, 4);
                const int ld_up = std::min(ket.l[1] + 1, 4);

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

                            // R-tensor factorization: precompute half-contracted
                            // intermediates to reduce 18×O(N^6) to O(N^6)+18×O(N^3)
                            const int bra_ext = l[0] + l[1] + 1;
                            const int ket_ext = l[2] + l[3] + 1;
                            const int rk_dim = bra_ext + 1;
                            const int rk_yz = rk_dim * rk_dim;
                            const int rk_elem = rk_dim * rk_dim * rk_dim;
                            const int rb_dim = ket_ext + 1;
                            const int rb_yz = rb_dim * rb_dim;
                            const int rb_elem = rb_dim * rb_dim * rb_dim;
                            const int nkl = nk_cart * nl_cart;

                            R_ket_buf.resize((size_t)nkl * rk_elem);
                            R_bra_buf.resize((size_t)rb_elem);

                            // Step 1: precompute R_ket for all (ck,cl)
                            for (int ck = 0; ck < nk_cart; ck++)
                                for (int cl = 0; cl < nl_cart; cl++)
                                    QC_Precompute_R_ket(
                                        E_ket,
                                        ket.comp_x[0][ck], ket.comp_x[1][cl],
                                        ket.comp_y[0][ck], ket.comp_y[1][cl],
                                        ket.comp_z[0][ck], ket.comp_z[1][cl],
                                        HR, grad_hr_base,
                                        bra_ext, bra_ext, bra_ext,
                                        &R_ket_buf[(ck * nl_cart + cl) * rk_elem],
                                        rk_yz, rk_dim);

                            const float ai = bra_prim.ai;
                            const float aj = bra_prim.aj;
                            const auto& E = bra_prim.E_bra;

                            // Step 2: compute derivatives with precomputed R
                            for (int ci = 0; ci < ni_cart; ci++)
                            {
                                const int ix = bra.comp_x[0][ci];
                                const int iy = bra.comp_y[0][ci];
                                const int iz = bra.comp_z[0][ci];
                                for (int cj = 0; cj < nj_cart; cj++)
                                {
                                    const int jx = bra.comp_x[1][cj];
                                    const int jy = bra.comp_y[1][cj];
                                    const int jz = bra.comp_z[1][cj];

                                    // Precompute R_bra for this (ci,cj)
                                    QC_Precompute_R_bra(
                                        E, ix, jx, iy, jy, iz, jz,
                                        HR, grad_hr_base,
                                        ket_ext, ket_ext, ket_ext,
                                        R_bra_buf.data(), rb_yz, rb_dim);

                                    // Base bra E-coefficients for this (ci,cj)
                                    const float* ex_b = E[0][ix][jx];
                                    const float* ey_b = E[1][iy][jy];
                                    const float* ez_b = E[2][iz][jz];
                                    const int mx_b = ix + jx;
                                    const int my_b = iy + jy;
                                    const int mz_b = iz + jz;

                                    // 1D dot product helpers
                                    auto dot1d = [](const float* e, int n,
                                                    const float* r) {
                                        double v = 0.0;
                                        for (int t = 0; t <= n; t++)
                                            v += (double)e[t] * (double)r[t];
                                        return (float)v;
                                    };
                                    auto dot1d_ph = [](const float* e, int n,
                                                       const float* r) {
                                        double v = 0.0;
                                        for (int t = 0; t <= n; t++)
                                        {
                                            float ph = (t & 1) ? -1.0f : 1.0f;
                                            v += (double)(e[t] * ph) * (double)r[t];
                                        }
                                        return (float)v;
                                    };

                                    for (int ck = 0; ck < nk_cart; ck++)
                                    {
                                        const int kx2 = ket.comp_x[0][ck];
                                        const int ky2 = ket.comp_y[0][ck];
                                        const int kz2 = ket.comp_z[0][ck];
                                        for (int cl = 0; cl < nl_cart; cl++)
                                        {
                                            const int lx2 = ket.comp_x[1][cl];
                                            const int ly2 = ket.comp_y[1][cl];
                                            const int lz2 = ket.comp_z[1][cl];

                                            const float* rk = &R_ket_buf[
                                                (ck * nl_cart + cl) * rk_elem];
                                            const float* rb = R_bra_buf.data();
                                            float dA[3] = {}, dB[3] = {}, dC[3] = {};

                                            // Partial contractions of R_ket:
                                            // reduce 3D→1D by fixing the derivative
                                            // dimension and summing the other two
                                            float pc_yz[10], pc_xz[10], pc_xy[10];
                                            for (int mx = 0; mx <= bra_ext; mx++)
                                            {
                                                double s = 0.0;
                                                for (int my = 0; my <= my_b; my++)
                                                    for (int mz = 0; mz <= mz_b; mz++)
                                                        s += (double)(ey_b[my] * ez_b[mz]) *
                                                             (double)rk[mx*rk_yz + my*rk_dim + mz];
                                                pc_yz[mx] = (float)s;
                                            }
                                            for (int my = 0; my <= bra_ext; my++)
                                            {
                                                double s = 0.0;
                                                for (int mx = 0; mx <= mx_b; mx++)
                                                    for (int mz = 0; mz <= mz_b; mz++)
                                                        s += (double)(ex_b[mx] * ez_b[mz]) *
                                                             (double)rk[mx*rk_yz + my*rk_dim + mz];
                                                pc_xz[my] = (float)s;
                                            }
                                            for (int mz = 0; mz <= bra_ext; mz++)
                                            {
                                                double s = 0.0;
                                                for (int mx = 0; mx <= mx_b; mx++)
                                                    for (int my = 0; my <= my_b; my++)
                                                        s += (double)(ex_b[mx] * ey_b[my]) *
                                                             (double)rk[mx*rk_yz + my*rk_dim + mz];
                                                pc_xy[mz] = (float)s;
                                            }

                                            // dA: 1D dot with shifted bra E-coeff
                                            if (ix + 1 < 5)
                                                dA[0] = 2.0f*ai * dot1d(E[0][ix+1][jx], ix+1+jx, pc_yz);
                                            if (ix > 0)
                                                dA[0] -= (float)ix * dot1d(E[0][ix-1][jx], ix-1+jx, pc_yz);
                                            if (iy + 1 < 5)
                                                dA[1] = 2.0f*ai * dot1d(E[1][iy+1][jy], iy+1+jy, pc_xz);
                                            if (iy > 0)
                                                dA[1] -= (float)iy * dot1d(E[1][iy-1][jy], iy-1+jy, pc_xz);
                                            if (iz + 1 < 5)
                                                dA[2] = 2.0f*ai * dot1d(E[2][iz+1][jz], iz+1+jz, pc_xy);
                                            if (iz > 0)
                                                dA[2] -= (float)iz * dot1d(E[2][iz-1][jz], iz-1+jz, pc_xy);

                                            // dB: 1D dot with shifted bra E-coeff
                                            if (jx + 1 < 5)
                                                dB[0] = 2.0f*aj * dot1d(E[0][ix][jx+1], ix+jx+1, pc_yz);
                                            if (jx > 0)
                                                dB[0] -= (float)jx * dot1d(E[0][ix][jx-1], ix+jx-1, pc_yz);
                                            if (jy + 1 < 5)
                                                dB[1] = 2.0f*aj * dot1d(E[1][iy][jy+1], iy+jy+1, pc_xz);
                                            if (jy > 0)
                                                dB[1] -= (float)jy * dot1d(E[1][iy][jy-1], iy+jy-1, pc_xz);
                                            if (jz + 1 < 5)
                                                dB[2] = 2.0f*aj * dot1d(E[2][iz][jz+1], iz+jz+1, pc_xy);
                                            if (jz > 0)
                                                dB[2] -= (float)jz * dot1d(E[2][iz][jz-1], iz+jz-1, pc_xy);

                                            // Partial contractions of R_bra for dC
                                            const float* ekx_b = E_ket[0][kx2][lx2];
                                            const float* eky_b = E_ket[1][ky2][ly2];
                                            const float* ekz_b = E_ket[2][kz2][lz2];
                                            const int nkx_b = kx2+lx2, nky_b = ky2+ly2, nkz_b = kz2+lz2;

                                            float qc_yz[10], qc_xz[10], qc_xy[10];
                                            for (int nx = 0; nx <= ket_ext; nx++)
                                            {
                                                double s = 0.0;
                                                for (int ny = 0; ny <= nky_b; ny++)
                                                    for (int nz = 0; nz <= nkz_b; nz++)
                                                    {
                                                        float ph = ((ny+nz) & 1) ? -1.0f : 1.0f;
                                                        s += (double)(eky_b[ny]*ekz_b[nz]*ph) *
                                                             (double)rb[nx*rb_yz + ny*rb_dim + nz];
                                                    }
                                                qc_yz[nx] = (float)s;
                                            }
                                            for (int ny = 0; ny <= ket_ext; ny++)
                                            {
                                                double s = 0.0;
                                                for (int nx = 0; nx <= nkx_b; nx++)
                                                    for (int nz = 0; nz <= nkz_b; nz++)
                                                    {
                                                        float ph = ((nx+nz) & 1) ? -1.0f : 1.0f;
                                                        s += (double)(ekx_b[nx]*ekz_b[nz]*ph) *
                                                             (double)rb[nx*rb_yz + ny*rb_dim + nz];
                                                    }
                                                qc_xz[ny] = (float)s;
                                            }
                                            for (int nz = 0; nz <= ket_ext; nz++)
                                            {
                                                double s = 0.0;
                                                for (int nx = 0; nx <= nkx_b; nx++)
                                                    for (int ny = 0; ny <= nky_b; ny++)
                                                    {
                                                        float ph = ((nx+ny) & 1) ? -1.0f : 1.0f;
                                                        s += (double)(ekx_b[nx]*eky_b[ny]*ph) *
                                                             (double)rb[nx*rb_yz + ny*rb_dim + nz];
                                                    }
                                                qc_xy[nz] = (float)s;
                                            }

                                            // dC: 1D dot with shifted ket E-coeff
                                            if (kx2 + 1 < 5)
                                                dC[0] = 2.0f*ak * dot1d_ph(E_ket[0][kx2+1][lx2], kx2+1+lx2, qc_yz);
                                            if (kx2 > 0)
                                                dC[0] -= (float)kx2 * dot1d_ph(E_ket[0][kx2-1][lx2], kx2-1+lx2, qc_yz);
                                            if (ky2 + 1 < 5)
                                                dC[1] = 2.0f*ak * dot1d_ph(E_ket[1][ky2+1][ly2], ky2+1+ly2, qc_xz);
                                            if (ky2 > 0)
                                                dC[1] -= (float)ky2 * dot1d_ph(E_ket[1][ky2-1][ly2], ky2-1+ly2, qc_xz);
                                            if (kz2 + 1 < 5)
                                                dC[2] = 2.0f*ak * dot1d_ph(E_ket[2][kz2+1][lz2], kz2+1+lz2, qc_xy);
                                            if (kz2 > 0)
                                                dC[2] -= (float)kz2 * dot1d_ph(E_ket[2][kz2-1][lz2], kz2-1+lz2, qc_xy);

                                            for (int d = 0; d < 3; d++)
                                            {
                                                dA[d] *= n_abcd;
                                                dB[d] *= n_abcd;
                                                dC[d] *= n_abcd;
                                            }

                                            const int idx =
                                                ((ci * nj_cart + cj) * nk_cart +
                                                 ck) *
                                                    nl_cart +
                                                cl;
                                            for (int d = 0; d < 3; d++)
                                            {
                                                d_buf_A_cart[idx * 3 + d] +=
                                                    dA[d];
                                                d_buf_B_cart[idx * 3 + d] +=
                                                    dB[d];
                                                d_buf_C_cart[idx * 3 + d] +=
                                                    dC[d];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                float* d_buf_A_use = d_buf_A_cart.data();
                float* d_buf_B_use = d_buf_B_cart.data();
                float* d_buf_C_use = d_buf_C_cart.data();

                if (is_spherical)
                {
                    d_buf_A_sph.assign((size_t)shell_size_eff * 3, 0.0f);
                    d_buf_B_sph.assign((size_t)shell_size_eff * 3, 0.0f);
                    d_buf_C_sph.assign((size_t)shell_size_eff * 3, 0.0f);

                    const int dims_cart[4] = {bra.dims_cart[0], bra.dims_cart[1],
                                              ket.dims_cart[0], ket.dims_cart[1]};
                    const int dims_sph[4] = {bra.dims_sph[0], bra.dims_sph[1],
                                             ket.dims_sph[0], ket.dims_sph[1]};
                    const int off_cart[4] = {bra.off_cart[0], bra.off_cart[1],
                                             ket.off_cart[0], ket.off_cart[1]};
                    const int off_sph[4] = {bra.off_eff[0], bra.off_eff[1],
                                            ket.off_eff[0], ket.off_eff[1]};

                    sph_buf0.assign((size_t)shell_size_cart, 0.0f);
                    sph_buf1.assign((size_t)shell_size_cart, 0.0f);
                    auto transform_deriv = [&](const std::vector<float>& src3,
                                               std::vector<float>& dst3)
                    {
                        for (int d = 0; d < 3; d++)
                        {
                            for (int idx = 0; idx < shell_size_cart; idx++)
                                sph_buf0[(size_t)idx] = src3[(size_t)idx * 3 + d];
                            std::fill(sph_buf1.begin(), sph_buf1.end(), 0.0f);
                            QC_Cart2Sph_Shell_ERI_CPU(
                                cart2sph_mat, nao_sph, off_cart, off_sph,
                                dims_cart, dims_sph, sph_buf0.data(), sph_buf1.data());
                            for (int idx = 0; idx < shell_size_eff; idx++)
                                dst3[(size_t)idx * 3 + d] = sph_buf0[(size_t)idx];
                        }
                    };

                    transform_deriv(d_buf_A_cart, d_buf_A_sph);
                    transform_deriv(d_buf_B_cart, d_buf_B_sph);
                    transform_deriv(d_buf_C_cart, d_buf_C_sph);

                    d_buf_A_use = d_buf_A_sph.data();
                    d_buf_B_use = d_buf_B_sph.data();
                    d_buf_C_use = d_buf_C_sph.data();
                }

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
                                    d_buf_A_use[idx * 3 + d] *= nijkl;
                                    d_buf_B_use[idx * 3 + d] *= nijkl;
                                    d_buf_C_use[idx * 3 + d] *= nijkl;
                                }
                            }
                        }
                    }
                }

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
                                const float* dA = &d_buf_A_use[idx * 3];
                                const float* dB = &d_buf_B_use[idx * 3];
                                const float* dC = &d_buf_C_use[idx * 3];

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

                                auto accumulate_8perm = [&](
                                    const int perm_slot[8][4],
                                    auto weight_fn,
                                    double& gamma_acc)
                                {
                                    for (int n = 0; n < 8; n++)
                                    {
                                        const int i0 = ao_idx[perm_slot[n][0]];
                                        const int i1 = ao_idx[perm_slot[n][1]];
                                        const int i2 = ao_idx[perm_slot[n][2]];
                                        const int i3 = ao_idx[perm_slot[n][3]];
                                        bool dup = false;
                                        for (int pv = 0; pv < n; pv++)
                                        {
                                            if (i0 == ao_idx[perm_slot[pv][0]] &&
                                                i1 == ao_idx[perm_slot[pv][1]] &&
                                                i2 == ao_idx[perm_slot[pv][2]] &&
                                                i3 == ao_idx[perm_slot[pv][3]])
                                            {
                                                dup = true;
                                                break;
                                            }
                                        }
                                        if (dup) continue;

                                        const double weight =
                                            weight_fn(i0, i1, i2, i3);
                                        gamma_acc += weight;
                                        for (int slot = 0; slot < 4; slot++)
                                        {
                                            const int src = perm_slot[n][slot];
                                            const int atom = atom_idx[src];
                                            for (int d = 0; d < 3; d++)
                                            {
                                                const double contrib =
                                                    weight * d_slot[src][d];
                                                grad_local[atom * 3 + d] +=
                                                    contrib;
                                            }
                                        }
                                    }
                                };

                                // Coulomb (J) permutations: (pq|rs) symmetry
                                const int jt_slot[8][4] = {
                                    {0, 1, 2, 3}, {1, 0, 2, 3},
                                    {0, 1, 3, 2}, {1, 0, 3, 2},
                                    {2, 3, 0, 1}, {3, 2, 0, 1},
                                    {2, 3, 1, 0}, {3, 2, 1, 0}};
                                accumulate_8perm(
                                    jt_slot,
                                    [&](int i0, int i1, int i2, int i3)
                                    {
                                        return 0.5 *
                                               (double)P_coul[i0 * nao + i1] *
                                               (double)P_coul[i2 * nao + i3];
                                    },
                                    gamma_j);

                                // Exchange (K) permutations: (pr|qs) symmetry
                                const int kt_slot[8][4] = {
                                    {0, 2, 1, 3}, {0, 3, 1, 2},
                                    {1, 2, 0, 3}, {1, 3, 0, 2},
                                    {2, 0, 3, 1}, {2, 1, 3, 0},
                                    {3, 0, 2, 1}, {3, 1, 2, 0}};

                                if (exx_scale_a != 0.0f)
                                {
                                    accumulate_8perm(
                                        kt_slot,
                                        [&](int i0, int i1, int i2, int i3)
                                        {
                                            return -0.5 *
                                                   (double)exx_scale_a *
                                                   (double)P_exx_a[i0 * nao + i1] *
                                                   (double)P_exx_a[i2 * nao + i3];
                                        },
                                        gamma_k);
                                }

                                if (exx_scale_b != 0.0f && P_exx_b != nullptr)
                                {
                                    accumulate_8perm(
                                        kt_slot,
                                        [&](int i0, int i1, int i2, int i3)
                                        {
                                            return -0.5 *
                                                   (double)exx_scale_b *
                                                   (double)P_exx_b[i0 * nao + i1] *
                                                   (double)P_exx_b[i2 * nao + i3];
                                        },
                                        gamma_k);
                                }

                            }
                        }
                    }
                }
            }
        }

#pragma omp critical
        {
            for (size_t i = 0; i < grad_local.size(); i++)
                grad[i] += grad_local[i];
        }
        free(HR);
    }

}

#endif // USE_GPU

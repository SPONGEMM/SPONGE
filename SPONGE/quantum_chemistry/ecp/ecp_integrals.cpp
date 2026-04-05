// ECP 积分求值 (Type-1 local + Type-2 semi-local)
//
// Type-1 (local, n_k=2): 三中心 Gaussian overlap — 无需 Boys 函数
// Type-2 (semi-local): factored angular projection at ECP center
//
// 当前实现: 仅支持 n_k=2 项 (覆盖 def2-ECP / LANL2DZ 全部项)
// 后续扩展: n_k=0,1 项 (需要 Boys 函数)

// clang-format off
#include "../quantum_chemistry.h"
#include "../integrals/one_e.hpp"
#include "ecp_integrals.h"
// clang-format on

// ==================== Angular projection coefficients ====================
// c_{abc,lm} = ∫ Y_lm(Ω) × (x/r)^a (y/r)^b (z/r)^c dΩ
// where Y_lm are ORTHONORMAL real spherical harmonics (∫ Y_lm Y_l'm' dΩ = δ)
//
// 索引: c_table[l][cart_idx][m_idx]
// Cartesian 分量按降序排列 (与 QC_Get_Lxyz_Device 一致)
// m_idx = m + l (m 从 -l 到 +l)
//
// 使用正交归一 Y_lm 后, 投影公式不需要额外的 4π/(2l+1) 因子。

// l=0: s → Y_00 = 1/√(4π)
static __device__ const float c_l0[1][1] = {
    {3.54490770f}  // s: c = √(4π)
};

// l=1: p → Y_{1m}
// Cart: x(1,0,0), y(0,1,0), z(0,0,1)
// Sph: m=-1(y), m=0(z), m=+1(x)
static __device__ const float c_l1[3][3] = {
    {0.0f, 0.0f, 2.04665342f},   // x → Y_{1,+1}
    {2.04665342f, 0.0f, 0.0f},   // y → Y_{1,-1}
    {0.0f, 2.04665342f, 0.0f}    // z → Y_{1,0}
};

// l=2: d → Y_{2m}
// Cart: xx, xy, xz, yy, yz, zz
// Sph: m=-2,-1,0,+1,+2
static __device__ const float c_l2[6][5] = {
    {0.0f, 0.0f, -0.52844364f, 0.0f,  0.91529123f},   // xx
    {0.91529123f, 0.0f,  0.0f, 0.0f,  0.0f},           // xy
    {0.0f, 0.0f,  0.0f, 0.91529123f,  0.0f},           // xz
    {0.0f, 0.0f, -0.52844364f, 0.0f, -0.91529123f},    // yy
    {0.0f, 0.91529123f,  0.0f, 0.0f,  0.0f},           // yz
    {0.0f, 0.0f,  1.05688728f, 0.0f,  0.0f}            // zz
};

// l=3: f → Y_{3m}
// Cart: xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz
// Sph: m=-3,-2,-1,0,+1,+2,+3
static __device__ const float c_l3[10][7] = {
    {0.0f, 0.0f, 0.0f, 0.0f, -0.32819468f, 0.0f, 0.42369751f},       // xxx
    {0.42369751f, 0.0f, -0.10939823f, 0.0f, 0.0f, 0.0f, 0.0f},       // xxy
    {0.0f, 0.0f, 0.0f, -0.26796983f, 0.0f, 0.34594757f, 0.0f},       // xxz
    {0.0f, 0.0f, 0.0f, 0.0f, -0.10939823f, 0.0f, -0.42369751f},      // xyy
    {0.0f, 0.34594757f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},               // xyz
    {0.0f, 0.0f, 0.0f, 0.0f, 0.43759291f, 0.0f, 0.0f},               // xzz
    {-0.42369751f, 0.0f, -0.32819468f, 0.0f, 0.0f, 0.0f, 0.0f},      // yyy
    {0.0f, 0.0f, 0.0f, -0.26796983f, 0.0f, -0.34594757f, 0.0f},      // yyz
    {0.0f, 0.0f, 0.43759291f, 0.0f, 0.0f, 0.0f, 0.0f},               // yzz
    {0.0f, 0.0f, 0.0f, 0.53593967f, 0.0f, 0.0f, 0.0f}                // zzz
};

// ==================== ECP 径向积分 ====================
// R_l(η) = ∫₀^∞ r^{2l+2} exp(-η r²) dr = (2l+1)!! √π / (2^{l+2} η^{l+3/2})

// Double-factorial: double_factorial(n) returns (2n-1)!!
static __device__ float double_factorial(int n)
{
    if (n <= 0) return 1.0f;
    float result = 1.0f;
    for (int k = 1; k <= n; k++) result *= (2.0f * k - 1.0f);
    return result;
}

// ==================== Type-1 Local 积分 (n_k=2) ====================
// ⟨μ|d_k exp(-ζ_k r_C²)|ν⟩ — 三中心 Gaussian overlap, 无投影算子
static __device__ float ecp_local_n2(
    float ei, float ej, float Ax, float Ay, float Az, float Bx, float By,
    float Bz, float Cx, float Cy, float Cz, float dist_sq_AB, float zeta,
    int lx_i, int ly_i, int lz_i, int lx_j, int ly_j, int lz_j)
{
    const float g = ei + ej;
    const float Kab = expf(-ei * ej / g * dist_sq_AB);
    if (Kab < 1e-15f) return 0.0f;

    const float Px = (ei * Ax + ej * Bx) / g;
    const float Py = (ei * Ay + ej * By) / g;
    const float Pz = (ei * Az + ej * Bz) / g;

    const float eta = g + zeta;
    const float PC2 = (Px - Cx) * (Px - Cx) + (Py - Cy) * (Py - Cy) +
                      (Pz - Cz) * (Pz - Cz);
    const float Kpc = expf(-g * zeta / eta * PC2);

    const float Qx = (g * Px + zeta * Cx) / eta;
    const float Qy = (g * Py + zeta * Cy) / eta;
    const float Qz = (g * Pz + zeta * Cz) / eta;

    float E_x[5][5][9], E_y[5][5][9], E_z[5][5][9];
    compute_md_coeffs(E_x, lx_i, lx_j, Qx - Ax, Qx - Bx, 0.5f / eta);
    compute_md_coeffs(E_y, ly_i, ly_j, Qy - Ay, Qy - By, 0.5f / eta);
    compute_md_coeffs(E_z, lz_i, lz_j, Qz - Az, Qz - Bz, 0.5f / eta);

    const float ox = E_x[lx_i][lx_j][0];
    const float oy = E_y[ly_i][ly_j][0];
    const float oz = E_z[lz_i][lz_j][0];

    const float pi_over_eta_32 =
        sqrtf(CONSTANT_Pi / eta) * (CONSTANT_Pi / eta);

    return Kab * Kpc * ox * oy * oz * pi_over_eta_32;
}

// ==================== Type-2 Semi-local 积分 (n_k=2) ====================
//
// ⟨μ|ΔV_l P_l|ν⟩ 使用 factored angular projection:
// P_l = (2l+1)/(4π) Σ_m |C_lm⟩⟨C_lm|
//
// 将基函数在 ECP 中心 C 展开 (二项式 + Taylor), 取 leading-order 分量:
//   φ_μ(C + rΩ) ≈ K_A exp(-α r²) Σ_{a+b+c=l} f_a^x f_b^y f_c^z r^l Ω^a Ω^b Ω^c
//
// 角投影 ∫ C_lm φ_μ dΩ ≈ K_A exp(-αr²) r^l × [4π/(2l+1)] × B̃_m^μ
// 其中 B̃_m = Σ_{abc:a+b+c=l} Ω_{abc,lm} f_a f_b f_c
//
// 最终: V = [4π/(2l+1)] × K_A K_B × [Σ_m B̃_m^μ B̃_m^ν] × R_l(η)

static __device__ float get_c_lm(int l, int cart_idx, int m_idx)
{
    if (l == 0) return c_l0[cart_idx][m_idx];
    if (l == 1) return c_l1[cart_idx][m_idx];
    if (l == 2) return c_l2[cart_idx][m_idx];
    if (l == 3) return c_l3[cart_idx][m_idx];
    return 0.0f;
}

static __device__ float ecp_semilocal_n2(
    float ei, float ej, float Ax, float Ay, float Az, float Bx, float By,
    float Bz, float Cx, float Cy, float Cz, float dist_sq_AB, float zeta,
    int lx_i, int ly_i, int lz_i, int lx_j, int ly_j, int lz_j, int ecp_l)
{
    // 基函数中心到 ECP 中心的位移 d = C - A (或 C - B)
    const float dAx = Cx - Ax, dAy = Cy - Ay, dAz = Cz - Az;
    const float dBx = Cx - Bx, dBy = Cy - By, dBz = Cz - Bz;
    const float dA2 = dAx * dAx + dAy * dAy + dAz * dAz;
    const float dB2 = dBx * dBx + dBy * dBy + dBz * dBz;

    // 偏心 Gaussian 衰减: K_A = exp(-α|A-C|²)
    const float KA = expf(-ei * dA2);
    const float KB = expf(-ej * dB2);
    if (KA * KB < 1e-15f) return 0.0f;

    const float eta = ei + ej + zeta;

    // 二项式系数 C(n,k), n ≤ 4
    static __device__ const float binom[5][5] = {
        {1, 0, 0, 0, 0}, {1, 1, 0, 0, 0}, {1, 2, 1, 0, 0},
        {1, 3, 3, 1, 0}, {1, 4, 6, 4, 1}};
    // 1/k!, k ≤ 4
    static __device__ const float inv_fact[5] = {1.0f, 1.0f, 0.5f,
                                                  1.0f / 6.0f, 1.0f / 24.0f};

    // f_p 系数: 基函数在 ECP 中心 C 展开后 (r-C)^p 方向的权重
    // 数组大小 4 支持 ecp_l ≤ 3 (覆盖 def2-ECP/LANL2DZ 所有通道)
    float fi_x[4], fi_y[4], fi_z[4];
    float fj_x[4], fj_y[4], fj_z[4];

    // 预计算 d^n 和 (-2αd)^n 的幂次
    float dA_px[5] = {1}, dA_py[5] = {1}, dA_pz[5] = {1};
    float dB_px[5] = {1}, dB_py[5] = {1}, dB_pz[5] = {1};
    float tA_x[5] = {1}, tA_y[5] = {1}, tA_z[5] = {1};
    float tB_x[5] = {1}, tB_y[5] = {1}, tB_z[5] = {1};
    const float mAx = -2.0f * ei * dAx, mAy = -2.0f * ei * dAy,
                mAz = -2.0f * ei * dAz;
    const float mBx = -2.0f * ej * dBx, mBy = -2.0f * ej * dBy,
                mBz = -2.0f * ej * dBz;
    for (int n = 1; n <= 4; n++)
    {
        dA_px[n] = dA_px[n - 1] * dAx;
        dA_py[n] = dA_py[n - 1] * dAy;
        dA_pz[n] = dA_pz[n - 1] * dAz;
        dB_px[n] = dB_px[n - 1] * dBx;
        dB_py[n] = dB_py[n - 1] * dBy;
        dB_pz[n] = dB_pz[n - 1] * dBz;
        tA_x[n] = tA_x[n - 1] * mAx;
        tA_y[n] = tA_y[n - 1] * mAy;
        tA_z[n] = tA_z[n - 1] * mAz;
        tB_x[n] = tB_x[n - 1] * mBx;
        tB_y[n] = tB_y[n - 1] * mBy;
        tB_z[n] = tB_z[n - 1] * mBz;
    }

    // 计算 f_p for μ (basis i)
    for (int p = 0; p <= ecp_l; p++)
    {
        float sx = 0, sy = 0, sz = 0;
        int kx = lx_i < p ? lx_i : p;
        int ky = ly_i < p ? ly_i : p;
        int kz = lz_i < p ? lz_i : p;
        for (int k = 0; k <= kx; k++)
            sx += binom[lx_i][k] * dA_px[lx_i - k] * tA_x[p - k] *
                  inv_fact[p - k];
        for (int k = 0; k <= ky; k++)
            sy += binom[ly_i][k] * dA_py[ly_i - k] * tA_y[p - k] *
                  inv_fact[p - k];
        for (int k = 0; k <= kz; k++)
            sz += binom[lz_i][k] * dA_pz[lz_i - k] * tA_z[p - k] *
                  inv_fact[p - k];
        fi_x[p] = sx;
        fi_y[p] = sy;
        fi_z[p] = sz;
    }

    // 计算 f_p for ν (basis j)
    for (int p = 0; p <= ecp_l; p++)
    {
        float sx = 0, sy = 0, sz = 0;
        int kx = lx_j < p ? lx_j : p;
        int ky = ly_j < p ? ly_j : p;
        int kz = lz_j < p ? lz_j : p;
        for (int k = 0; k <= kx; k++)
            sx += binom[lx_j][k] * dB_px[lx_j - k] * tB_x[p - k] *
                  inv_fact[p - k];
        for (int k = 0; k <= ky; k++)
            sy += binom[ly_j][k] * dB_py[ly_j - k] * tB_y[p - k] *
                  inv_fact[p - k];
        for (int k = 0; k <= kz; k++)
            sz += binom[lz_j][k] * dB_pz[lz_j - k] * tB_z[p - k] *
                  inv_fact[p - k];
        fj_x[p] = sx;
        fj_y[p] = sy;
        fj_z[p] = sz;
    }

    // B̃_m = Σ_{abc:a+b+c=l} Ω_{abc,lm} × f_a^x × f_b^y × f_c^z
    // 数组大小 7 = 2*3+1, 支持 ecp_l ≤ 3
    const int n_cart = (ecp_l + 1) * (ecp_l + 2) / 2;
    const int n_sph = 2 * ecp_l + 1;
    float Bi[7] = {}, Bj[7] = {};

    for (int ic = 0; ic < n_cart; ic++)
    {
        int a, b, c;
        QC_Get_Lxyz_Device(ecp_l, ic, a, b, c);
        float fi_abc = fi_x[a] * fi_y[b] * fi_z[c];
        float fj_abc = fj_x[a] * fj_y[b] * fj_z[c];
        for (int m = 0; m < n_sph; m++)
        {
            float clm = get_c_lm(ecp_l, ic, m);
            Bi[m] += clm * fi_abc;
            Bj[m] += clm * fj_abc;
        }
    }

    // 角动量求和: Σ_m B̃_m^μ × B̃_m^ν
    float angular_sum = 0.0f;
    for (int m = 0; m < n_sph; m++)
        angular_sum += Bi[m] * Bj[m];

    // 径向积分: R_l(η) = (2l+1)!! √π / (2^{l+2} η^{l+3/2})
    float R_l = double_factorial(ecp_l + 1) * sqrtf(CONSTANT_Pi);
    float eta_pow = eta * sqrtf(eta);  // η^{3/2}
    for (int i = 0; i < ecp_l; i++) eta_pow *= eta;
    float two_pow = (float)(1 << (ecp_l + 2));  // 2^{l+2}
    R_l /= (two_pow * eta_pow);

    // K_A × K_B × angular_sum × R_l (no extra factor with orthonormal Y_lm)
    return KA * KB * angular_sum * R_l;
}

// ==================== ECP 主 Kernel ====================
static __global__ void ECP_Kernel(
    const int n_tasks, const QC_ONE_E_TASK* tasks, const VECTOR* centers,
    const int* l_list, const float* exps_arr, const float* coeffs_arr,
    const int* shell_offsets, const int* shell_sizes, const int* ao_offsets,
    // ECP 参数
    const VECTOR* atom_coords, int natm, const int* ecp_l_max,
    const int* ecp_atom_channel_range, const int* ecp_channel_l,
    const int* ecp_channel_offsets, const int* ecp_channel_sizes,
    const float* ecp_d, const float* ecp_zeta, const int* ecp_n_arr,
    // 输出
    float* out_V_ECP, int nao_total)
{
    SIMPLE_DEVICE_FOR(task_id, n_tasks)
    {
        QC_ONE_E_TASK sh_idx = tasks[task_id];
        const int i_sh = sh_idx.x;
        const int j_sh = sh_idx.y;

        const int li = l_list[i_sh], lj = l_list[j_sh];
        const int ni = (li + 1) * (li + 2) / 2;
        const int nj = (lj + 1) * (lj + 2) / 2;
        const int off_i = ao_offsets[i_sh], off_j = ao_offsets[j_sh];
        const VECTOR A = centers[i_sh], B = centers[j_sh];
        const float Ax = A.x, Ay = A.y, Az = A.z;
        const float Bx = B.x, By = B.y, Bz = B.z;
        const float dist_sq = (Ax - Bx) * (Ax - Bx) + (Ay - By) * (Ay - By) +
                              (Az - Bz) * (Az - Bz);

        for (int idx_i = 0; idx_i < ni; idx_i++)
        {
            for (int idx_j = 0; idx_j < nj; idx_j++)
            {
                int lx_i, ly_i, lz_i, lx_j, ly_j, lz_j;
                QC_Get_Lxyz_Device(li, idx_i, lx_i, ly_i, lz_i);
                QC_Get_Lxyz_Device(lj, idx_j, lx_j, ly_j, lz_j);

                float total_ecp = 0.0f;

                // 遍历 ECP 原子
                for (int iat = 0; iat < natm; iat++)
                {
                    if (ecp_l_max[iat] < 0) continue;  // 该原子无 ECP

                    const float Cx = atom_coords[iat].x;
                    const float Cy = atom_coords[iat].y;
                    const float Cz = atom_coords[iat].z;
                    const int l_max = ecp_l_max[iat];
                    const int ch_start = ecp_atom_channel_range[iat];
                    const int ch_end = ecp_atom_channel_range[iat + 1];

                    // 找到 local channel (l_max 或 l<0)
                    int local_ch = -1;
                    for (int ich = ch_start; ich < ch_end; ich++)
                    {
                        int cl = ecp_channel_l[ich];
                        if (cl < 0 || cl == l_max)
                        {
                            local_ch = ich;
                            break;
                        }
                    }

                    for (int pi = 0; pi < shell_sizes[i_sh]; pi++)
                    {
                        const float ei = exps_arr[shell_offsets[i_sh] + pi];
                        const float ci = coeffs_arr[shell_offsets[i_sh] + pi];

                        for (int pj = 0; pj < shell_sizes[j_sh]; pj++)
                        {
                            const float ej =
                                exps_arr[shell_offsets[j_sh] + pj];
                            const float cj =
                                coeffs_arr[shell_offsets[j_sh] + pj];
                            const float cc = ci * cj;

                            // 1. Local 贡献: ⟨μ|U_L|ν⟩ (无投影)
                            if (local_ch >= 0)
                            {
                                const int t_off =
                                    ecp_channel_offsets[local_ch];
                                const int t_cnt =
                                    ecp_channel_sizes[local_ch];
                                for (int it = 0; it < t_cnt; it++)
                                {
                                    const float dk = ecp_d[t_off + it];
                                    const float zk = ecp_zeta[t_off + it];
                                    const int nk = ecp_n_arr[t_off + it];
                                    if (nk != 2) continue;

                                    float val = ecp_local_n2(
                                        ei, ej, Ax, Ay, Az, Bx, By, Bz, Cx,
                                        Cy, Cz, dist_sq, zk, lx_i, ly_i,
                                        lz_i, lx_j, ly_j, lz_j);
                                    total_ecp += cc * dk * val;
                                }
                            }

                            // 2. Semi-local: ⟨μ|ΔU_l P_l|ν⟩
                            //    通道数据已存储 ΔU_l = U_l - U_L
                            for (int ich = ch_start; ich < ch_end; ich++)
                            {
                                const int ch_l = ecp_channel_l[ich];
                                if (ch_l < 0 || ch_l == l_max)
                                    continue;  // skip local

                                const int t_off = ecp_channel_offsets[ich];
                                const int t_cnt = ecp_channel_sizes[ich];
                                for (int it = 0; it < t_cnt; it++)
                                {
                                    const float dk = ecp_d[t_off + it];
                                    const float zk = ecp_zeta[t_off + it];
                                    const int nk = ecp_n_arr[t_off + it];
                                    if (nk != 2) continue;

                                    float val = ecp_semilocal_n2(
                                        ei, ej, Ax, Ay, Az, Bx, By, Bz, Cx,
                                        Cy, Cz, dist_sq, zk, lx_i, ly_i,
                                        lz_i, lx_j, ly_j, lz_j, ch_l);
                                    total_ecp += cc * dk * val;
                                }
                            }
                        }
                    }
                }

                // 1e task 列表包含全部 (i,j) 和 (j,i)，无需手动对称化
                const int idx = (off_i + idx_i) * nao_total + (off_j + idx_j);
                atomicAdd(&out_V_ECP[idx], total_ecp);
            }
        }
    }
}

// ==================== ECP 积分驱动 ====================
void QC_Compute_V_ECP(const QC_MOLECULE& mol,
                      const QC_INTEGRAL_TASKS& task_ctx, float* d_V_ECP)
{
    if (!mol.has_ecp || mol.ecp_total_terms == 0) return;

    const int n_total = task_ctx.topo.n_1e_tasks;
    const int chunk_size = ONE_E_BATCH_SIZE;

    for (int i = 0; i < n_total; i += chunk_size)
    {
        int current_chunk = std::min(chunk_size, n_total - i);
        const QC_ONE_E_TASK* task_ptr = task_ctx.buffers.d_1e_tasks + i;
        Launch_Device_Kernel(
            ECP_Kernel, (current_chunk + 63) / 64, 64, 0, 0, current_chunk,
            task_ptr, mol.d_centers, mol.d_l_list, mol.d_exps, mol.d_coeffs,
            mol.d_shell_offsets, mol.d_shell_sizes, mol.d_ao_offsets,
            mol.d_atom_coords, mol.natm, mol.d_ecp_l_max,
            mol.d_ecp_atom_channel_range, mol.d_ecp_l,
            mol.d_ecp_channel_offsets, mol.d_ecp_channel_sizes, mol.d_ecp_d,
            mol.d_ecp_zeta, mol.d_ecp_n, d_V_ECP, mol.nao_cart);
    }
}

// ==================== ECP 梯度 Kernel ====================
// 使用角动量平移求导: d/dA_x V = 2α V(lx_i+1) - lx_i V(lx_i-1)
// ECP 中心导数由平移不变性得: d/dC = -(d/dA + d/dB)
// 遵循 V_Grad_Kernel 约定:
//   bra 导数 × 2.0 × P → atom_i (因子 2 包含 ket 贡献)
//   ECP 中心导数 × 1.0 × P → ecp atom

static __device__ float ecp_integral_for_term(
    float ei, float ej, float Ax, float Ay, float Az, float Bx, float By,
    float Bz, float Cx, float Cy, float Cz, float dist_sq_AB, float zeta,
    int lx_i, int ly_i, int lz_i, int lx_j, int ly_j, int lz_j, int ch_l,
    bool is_local)
{
    if (lx_i < 0 || ly_i < 0 || lz_i < 0) return 0.0f;
    if (lx_j < 0 || ly_j < 0 || lz_j < 0) return 0.0f;
    if (is_local)
        return ecp_local_n2(ei, ej, Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz,
                            dist_sq_AB, zeta, lx_i, ly_i, lz_i, lx_j, ly_j,
                            lz_j);
    else
        return ecp_semilocal_n2(ei, ej, Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz,
                                dist_sq_AB, zeta, lx_i, ly_i, lz_i, lx_j,
                                ly_j, lz_j, ch_l);
}

static __global__ void ECP_Grad_Kernel(
    const int n_tasks, const QC_ONE_E_TASK* tasks, const VECTOR* centers,
    const int* l_list, const float* exps_arr, const float* coeffs_arr,
    const int* shell_offsets, const int* shell_sizes, const int* ao_offsets,
    // ECP 参数
    const VECTOR* atom_coords, int natm, const int* ecp_l_max,
    const int* ecp_atom_channel_range, const int* ecp_channel_l,
    const int* ecp_channel_offsets, const int* ecp_channel_sizes,
    const float* ecp_d, const float* ecp_zeta, const int* ecp_n_arr,
    // 密度 (已含 norm 权重)
    int nao_total, const int* shell_atom, const float* P_weighted,
    // 输出
    double* grad)
{
    SIMPLE_DEVICE_FOR(task_id, n_tasks)
    {
        QC_ONE_E_TASK sh_idx = tasks[task_id];
        const int i_sh = sh_idx.x;
        const int j_sh = sh_idx.y;

        const int li = l_list[i_sh], lj = l_list[j_sh];
        const int ni = (li + 1) * (li + 2) / 2;
        const int nj = (lj + 1) * (lj + 2) / 2;
        const int off_i = ao_offsets[i_sh], off_j = ao_offsets[j_sh];
        const int atom_i = shell_atom[i_sh];
        const VECTOR A = centers[i_sh], B = centers[j_sh];
        const float Ax = A.x, Ay = A.y, Az = A.z;
        const float Bx = B.x, By = B.y, Bz = B.z;
        const float dist_sq = (Ax - Bx) * (Ax - Bx) + (Ay - By) * (Ay - By) +
                              (Az - Bz) * (Az - Bz);

        for (int idx_i = 0; idx_i < ni; idx_i++)
        {
            for (int idx_j = 0; idx_j < nj; idx_j++)
            {
                int lx_i, ly_i, lz_i, lx_j, ly_j, lz_j;
                QC_Get_Lxyz_Device(li, idx_i, lx_i, ly_i, lz_i);
                QC_Get_Lxyz_Device(lj, idx_j, lx_j, ly_j, lz_j);

                const int mu = off_i + idx_i;
                const int nu = off_j + idx_j;
                const float p_val = P_weighted[mu * nao_total + nu];

                // 遍历 ECP 原子
                for (int iat = 0; iat < natm; iat++)
                {
                    if (ecp_l_max[iat] < 0) continue;

                    const float Cx = atom_coords[iat].x;
                    const float Cy = atom_coords[iat].y;
                    const float Cz = atom_coords[iat].z;
                    const int l_max = ecp_l_max[iat];
                    const int ch_start = ecp_atom_channel_range[iat];
                    const int ch_end = ecp_atom_channel_range[iat + 1];

                    int local_ch = -1;
                    for (int ich = ch_start; ich < ch_end; ich++)
                    {
                        int cl = ecp_channel_l[ich];
                        if (cl < 0 || cl == l_max)
                        {
                            local_ch = ich;
                            break;
                        }
                    }

                    // 对每个 primitive pair 累积梯度
                    for (int pi = 0; pi < shell_sizes[i_sh]; pi++)
                    {
                        const float ei = exps_arr[shell_offsets[i_sh] + pi];
                        const float ci = coeffs_arr[shell_offsets[i_sh] + pi];
                        for (int pj = 0; pj < shell_sizes[j_sh]; pj++)
                        {
                            const float ej =
                                exps_arr[shell_offsets[j_sh] + pj];
                            const float cj =
                                coeffs_arr[shell_offsets[j_sh] + pj];
                            const float cc = ci * cj;

                            // 收集所有 ECP 通道的梯度贡献
                            double dV_dAx = 0, dV_dAy = 0, dV_dAz = 0;
                            double dV_dBx = 0, dV_dBy = 0, dV_dBz = 0;

                            // Lambda: 对单个 ECP 项计算 bra+ket ���数
                            auto accumulate_grad =
                                [&](float dk, float zk, int ch_l,
                                    bool is_local)
                            {
                                double cdk = (double)(cc * dk);

                                // d/dA_x = 2α V(lx+1) - lx V(lx-1)
                                float vp = ecp_integral_for_term(
                                    ei, ej, Ax, Ay, Az, Bx, By, Bz, Cx,
                                    Cy, Cz, dist_sq, zk, lx_i + 1, ly_i,
                                    lz_i, lx_j, ly_j, lz_j, ch_l,
                                    is_local);
                                float vm = ecp_integral_for_term(
                                    ei, ej, Ax, Ay, Az, Bx, By, Bz, Cx,
                                    Cy, Cz, dist_sq, zk, lx_i - 1, ly_i,
                                    lz_i, lx_j, ly_j, lz_j, ch_l,
                                    is_local);
                                dV_dAx += cdk * (2.0 * (double)ei *
                                                     (double)vp -
                                                 (double)lx_i * (double)vm);

                                // d/dA_y
                                vp = ecp_integral_for_term(
                                    ei, ej, Ax, Ay, Az, Bx, By, Bz, Cx,
                                    Cy, Cz, dist_sq, zk, lx_i, ly_i + 1,
                                    lz_i, lx_j, ly_j, lz_j, ch_l,
                                    is_local);
                                vm = ecp_integral_for_term(
                                    ei, ej, Ax, Ay, Az, Bx, By, Bz, Cx,
                                    Cy, Cz, dist_sq, zk, lx_i, ly_i - 1,
                                    lz_i, lx_j, ly_j, lz_j, ch_l,
                                    is_local);
                                dV_dAy += cdk * (2.0 * (double)ei *
                                                     (double)vp -
                                                 (double)ly_i * (double)vm);

                                // d/dA_z
                                vp = ecp_integral_for_term(
                                    ei, ej, Ax, Ay, Az, Bx, By, Bz, Cx,
                                    Cy, Cz, dist_sq, zk, lx_i, ly_i,
                                    lz_i + 1, lx_j, ly_j, lz_j, ch_l,
                                    is_local);
                                vm = ecp_integral_for_term(
                                    ei, ej, Ax, Ay, Az, Bx, By, Bz, Cx,
                                    Cy, Cz, dist_sq, zk, lx_i, ly_i,
                                    lz_i - 1, lx_j, ly_j, lz_j, ch_l,
                                    is_local);
                                dV_dAz += cdk * (2.0 * (double)ei *
                                                     (double)vp -
                                                 (double)lz_i * (double)vm);

                                // d/dB_x = 2β V(lx_j+1) - lx_j V(lx_j-1)
                                vp = ecp_integral_for_term(
                                    ei, ej, Ax, Ay, Az, Bx, By, Bz, Cx,
                                    Cy, Cz, dist_sq, zk, lx_i, ly_i,
                                    lz_i, lx_j + 1, ly_j, lz_j, ch_l,
                                    is_local);
                                vm = ecp_integral_for_term(
                                    ei, ej, Ax, Ay, Az, Bx, By, Bz, Cx,
                                    Cy, Cz, dist_sq, zk, lx_i, ly_i,
                                    lz_i, lx_j - 1, ly_j, lz_j, ch_l,
                                    is_local);
                                dV_dBx += cdk * (2.0 * (double)ej *
                                                     (double)vp -
                                                 (double)lx_j * (double)vm);

                                // d/dB_y
                                vp = ecp_integral_for_term(
                                    ei, ej, Ax, Ay, Az, Bx, By, Bz, Cx,
                                    Cy, Cz, dist_sq, zk, lx_i, ly_i,
                                    lz_i, lx_j, ly_j + 1, lz_j, ch_l,
                                    is_local);
                                vm = ecp_integral_for_term(
                                    ei, ej, Ax, Ay, Az, Bx, By, Bz, Cx,
                                    Cy, Cz, dist_sq, zk, lx_i, ly_i,
                                    lz_i, lx_j, ly_j - 1, lz_j, ch_l,
                                    is_local);
                                dV_dBy += cdk * (2.0 * (double)ej *
                                                     (double)vp -
                                                 (double)ly_j * (double)vm);

                                // d/dB_z
                                vp = ecp_integral_for_term(
                                    ei, ej, Ax, Ay, Az, Bx, By, Bz, Cx,
                                    Cy, Cz, dist_sq, zk, lx_i, ly_i,
                                    lz_i, lx_j, ly_j, lz_j + 1, ch_l,
                                    is_local);
                                vm = ecp_integral_for_term(
                                    ei, ej, Ax, Ay, Az, Bx, By, Bz, Cx,
                                    Cy, Cz, dist_sq, zk, lx_i, ly_i,
                                    lz_i, lx_j, ly_j, lz_j - 1, ch_l,
                                    is_local);
                                dV_dBz += cdk * (2.0 * (double)ej *
                                                     (double)vp -
                                                 (double)lz_j * (double)vm);
                            };

                            // Local 通道
                            if (local_ch >= 0)
                            {
                                const int t_off =
                                    ecp_channel_offsets[local_ch];
                                const int t_cnt =
                                    ecp_channel_sizes[local_ch];
                                for (int it = 0; it < t_cnt; it++)
                                {
                                    if (ecp_n_arr[t_off + it] != 2) continue;
                                    accumulate_grad(ecp_d[t_off + it],
                                                    ecp_zeta[t_off + it], -1,
                                                    true);
                                }
                            }

                            // Semi-local 通道
                            for (int ich = ch_start; ich < ch_end; ich++)
                            {
                                const int ch_l = ecp_channel_l[ich];
                                if (ch_l < 0 || ch_l == l_max) continue;
                                const int t_off = ecp_channel_offsets[ich];
                                const int t_cnt = ecp_channel_sizes[ich];
                                for (int it = 0; it < t_cnt; it++)
                                {
                                    if (ecp_n_arr[t_off + it] != 2) continue;
                                    accumulate_grad(ecp_d[t_off + it],
                                                    ecp_zeta[t_off + it],
                                                    ch_l, false);
                                }
                            }

                            // 累加到梯度: bra × 2.0, ECP center × 1.0
                            double dp = (double)p_val;
                            atomicAdd(&grad[atom_i * 3 + 0],
                                      2.0 * dp * dV_dAx);
                            atomicAdd(&grad[atom_i * 3 + 1],
                                      2.0 * dp * dV_dAy);
                            atomicAdd(&grad[atom_i * 3 + 2],
                                      2.0 * dp * dV_dAz);
                            // d/dC = -(d/dA + d/dB) (平移不变性)
                            atomicAdd(&grad[iat * 3 + 0],
                                      -dp * (dV_dAx + dV_dBx));
                            atomicAdd(&grad[iat * 3 + 1],
                                      -dp * (dV_dAy + dV_dBy));
                            atomicAdd(&grad[iat * 3 + 2],
                                      -dp * (dV_dAz + dV_dBz));
                        }
                    }
                }
            }
        }
    }
}

// ==================== 球谐→笛卡尔密度变换 ====================
void QC_Sph2Cart_Density_Host(int ns, int nc,
                              const std::vector<float>& h_norms,
                              const std::vector<float>& h_C,
                              const std::vector<float>& h_M_sph,
                              std::vector<float>& h_M_cart)
{
    // NMN[i,j] = norms[i] * M[i,j] * norms[j]
    std::vector<float> NMN(ns * ns);
    for (int i = 0; i < ns; i++)
        for (int j = 0; j < ns; j++)
            NMN[i * ns + j] = h_norms[i] * h_M_sph[i * ns + j] * h_norms[j];

    // temp[a,k] = Σ_i C[i,a] * NMN[i,k]   (a ∈ cart, i,k ∈ sph)
    std::vector<float> temp(nc * ns, 0.0f);
    for (int a = 0; a < nc; a++)
        for (int k = 0; k < ns; k++)
        {
            double s = 0;
            for (int i = 0; i < ns; i++)
                s += (double)h_C[i * nc + a] * (double)NMN[i * ns + k];
            temp[a * ns + k] = (float)s;
        }

    // M_cart[a,b] = Σ_k temp[a,k] * C[k,b]
    h_M_cart.assign(nc * nc, 0.0f);
    for (int a = 0; a < nc; a++)
        for (int b = 0; b < nc; b++)
        {
            double s = 0;
            for (int k = 0; k < ns; k++)
                s += (double)temp[a * ns + k] * (double)h_C[k * nc + b];
            h_M_cart[a * nc + b] = (float)s;
        }
}

// ==================== ECP 梯度驱动 ====================
void QC_Compute_ECP_Gradient(const QC_MOLECULE& mol,
                             const QC_INTEGRAL_TASKS& task_ctx,
                             const int* d_shell_atom,
                             const float* d_P_cart_eff, double* d_grad)
{
    if (!mol.has_ecp || mol.ecp_total_terms == 0) return;

    const int nao_cart = mol.nao_cart;
    const int n_total = task_ctx.topo.n_1e_tasks;
    const int chunk_size = ONE_E_BATCH_SIZE;

    for (int i = 0; i < n_total; i += chunk_size)
    {
        int current_chunk = std::min(chunk_size, n_total - i);
        const QC_ONE_E_TASK* task_ptr = task_ctx.buffers.d_1e_tasks + i;
        Launch_Device_Kernel(
            ECP_Grad_Kernel, (current_chunk + 63) / 64, 64, 0, 0,
            current_chunk, task_ptr, mol.d_centers, mol.d_l_list, mol.d_exps,
            mol.d_coeffs, mol.d_shell_offsets, mol.d_shell_sizes,
            mol.d_ao_offsets,
            mol.d_atom_coords, mol.natm, mol.d_ecp_l_max,
            mol.d_ecp_atom_channel_range, mol.d_ecp_l,
            mol.d_ecp_channel_offsets, mol.d_ecp_channel_sizes, mol.d_ecp_d,
            mol.d_ecp_zeta, mol.d_ecp_n,
            nao_cart, d_shell_atom, d_P_cart_eff,
            d_grad);
    }
}

#pragma once

#include "../integrals/one_e.hpp"

// 单电子积分导数
// 计算 dS/dR, dT/dR, dV/dR 对原子梯度的贡献:
//   grad_A -= Tr[W · dS/dR_A]      (Pulay)
//   grad_A += Tr[P · dT/dR_A]      (动能)
//   grad_A += Tr[P · dV/dR_A]      (核吸引)
// 全矩阵遍历只计算 bra 侧导数，乘以 2 补偿 ket 侧
// (等价于 PySCF 的 h1ao + h1ao.T 对称化)
//
// McMurchie-Davidson 导数公式:
//   dS(a,b)/dA_x = 2αi·S(a+1,b) - a·S(a-1,b)  [重叠]
//   dE^{ab}_t/dA_x = 2αi·E^{(a+1)b}_t - a·E^{(a-1)b}_t  [E系数]
//   dR_{tuv,0}/dC_x = -R_{(t+1)uv,0}  [R-tensor, 核中心]
//
// 分离式 1e 梯度核:
//   S/T 梯度: 无原子循环，每线程仅需 overlap 数组 (res_x/y/z)
//   V 梯度: 按 (shell_pair × atom) 并行化，提升 GPU 利用率

// V 梯度用 R-tensor: 支持到 L_max=9 (g+g+1)
#define GRAD_R_BASE 10
#define GRAD_R_IDX(t, u, v, n) \
    ((((t) * GRAD_R_BASE + (u)) * GRAD_R_BASE + (v)) * GRAD_R_BASE + (n))

static __device__ void compute_r_tensor_1e_grad(float* R, double* F,
                                                float alpha, float PC[3],
                                                int L_tot)
{
    const int total = GRAD_R_BASE * GRAD_R_BASE * GRAD_R_BASE * GRAD_R_BASE;
    for (int i = 0; i < total; i++) R[i] = 0.0f;

    double m2a = -2.0 * (double)alpha;
    double fac = 1.0;
    for (int n = 0; n <= L_tot; n++)
    {
        R[GRAD_R_IDX(0, 0, 0, n)] = (float)(fac * F[n]);
        fac *= m2a;
    }

    for (int N = 1; N <= L_tot; N++)
    {
        for (int t = 0; t <= N; t++)
            for (int u = 0; u <= N - t; u++)
            {
                int v = N - t - u;
                int max_n = L_tot - N;
                for (int n = 0; n <= max_n; n++)
                {
                    double val = 0.0;
                    if (t > 0)
                    {
                        val = (double)PC[0] * R[GRAD_R_IDX(t - 1, u, v, n + 1)];
                        if (t > 1)
                            val += (double)(t - 1) *
                                   R[GRAD_R_IDX(t - 2, u, v, n + 1)];
                    }
                    else if (u > 0)
                    {
                        val = (double)PC[1] * R[GRAD_R_IDX(t, u - 1, v, n + 1)];
                        if (u > 1)
                            val += (double)(u - 1) *
                                   R[GRAD_R_IDX(t, u - 2, v, n + 1)];
                    }
                    else if (v > 0)
                    {
                        val = (double)PC[2] * R[GRAD_R_IDX(t, u, v - 1, n + 1)];
                        if (v > 1)
                            val += (double)(v - 1) *
                                   R[GRAD_R_IDX(t, u, v - 2, n + 1)];
                    }
                    R[GRAD_R_IDX(t, u, v, n)] = (float)val;
                }
            }
    }
}

// S/T 梯度核: 只计算 Pulay (dS) 和动能 (dT) 导数
// 无 R-tensor，无原子循环，每线程内存 ~1KB
static __global__ void OneE_ST_Grad_Kernel(
    const int n_tasks, const QC_ONE_E_TASK* tasks, const VECTOR* centers,
    const int* l_list, const float* exps, const float* coeffs,
    const int* shell_offsets, const int* shell_sizes, const int* ao_offsets,
    int nao_total, const int* shell_atom, const float* P, const float* W,
    const float* norms, double* grad)
{
    SIMPLE_DEVICE_FOR(task_id, n_tasks)
    {
        QC_ONE_E_TASK sh_idx = tasks[task_id];
        int i_sh = sh_idx.x;
        int j_sh = sh_idx.y;

        int li = l_list[i_sh], lj = l_list[j_sh];
        int ni = (li + 1) * (li + 2) / 2, nj = (lj + 1) * (lj + 2) / 2;
        int off_i = ao_offsets[i_sh], off_j = ao_offsets[j_sh];
        int atom_i = shell_atom[i_sh];
        const VECTOR A = centers[i_sh];
        const VECTOR B = centers[j_sh];
        float Ax = A.x, Ay = A.y, Az = A.z;
        float Bx = B.x, By = B.y, Bz = B.z;
        float dist_sq = (Ax - Bx) * (Ax - Bx) + (Ay - By) * (Ay - By) +
                        (Az - Bz) * (Az - Bz);

        for (int idx_i = 0; idx_i < ni; idx_i++)
        {
            for (int idx_j = 0; idx_j < nj; idx_j++)
            {
                int lx_i, ly_i, lz_i, lx_j, ly_j, lz_j;
                QC_Get_Lxyz_Device(li, idx_i, lx_i, ly_i, lz_i);
                QC_Get_Lxyz_Device(lj, idx_j, lx_j, ly_j, lz_j);

                int mu = off_i + idx_i;
                int nu = off_j + idx_j;
                float norm_mu_nu = norms[mu] * norms[nu];
                float p_val = P[mu * nao_total + nu] * norm_mu_nu;
                float w_val = W[mu * nao_total + nu] * norm_mu_nu;
                for (int pi = 0; pi < shell_sizes[i_sh]; pi++)
                {
                    float ei = exps[shell_offsets[i_sh] + pi];
                    float ci = coeffs[shell_offsets[i_sh] + pi];
                    for (int pj = 0; pj < shell_sizes[j_sh]; pj++)
                    {
                        float ej = exps[shell_offsets[j_sh] + pj];
                        float cj = coeffs[shell_offsets[j_sh] + pj];
                        float g = ei + ej;
                        float Kab = expf(-ei * ej / g * dist_sq);
                        float cc = ci * cj * Kab;
                        if (fabsf(cc) < 1e-20f) continue;

                        float Px = (ei * Ax + ej * Bx) / g;
                        float Py = (ei * Ay + ej * By) / g;
                        float Pz = (ei * Az + ej * Bz) / g;

                        float res_x[7][7], res_y[7][7], res_z[7][7];
                        get_overlap1d_arr(lx_i + 2, lx_j + 1, Px - Ax, Px - Bx,
                                          g, res_x);
                        get_overlap1d_arr(ly_i + 2, ly_j + 1, Py - Ay, Py - By,
                                          g, res_y);
                        get_overlap1d_arr(lz_i + 2, lz_j + 1, Pz - Az, Pz - Bz,
                                          g, res_z);

                        float sx = res_x[lx_i][lx_j];
                        float sy = res_y[ly_i][ly_j];
                        float sz = res_z[lz_i][lz_j];

                        // dS/dA
                        float dsx = 2.0f * ei * res_x[lx_i + 1][lx_j];
                        if (lx_i > 0)
                            dsx -= (float)lx_i * res_x[lx_i - 1][lx_j];
                        float dsy = 2.0f * ei * res_y[ly_i + 1][ly_j];
                        if (ly_i > 0)
                            dsy -= (float)ly_i * res_y[ly_i - 1][ly_j];
                        float dsz = 2.0f * ei * res_z[lz_i + 1][lz_j];
                        if (lz_i > 0)
                            dsz -= (float)lz_i * res_z[lz_i - 1][lz_j];

                        atomicAdd(&grad[atom_i * 3 + 0],
                                  -2.0 * (double)w_val *
                                      (double)(cc * dsx * sy * sz));
                        atomicAdd(&grad[atom_i * 3 + 1],
                                  -2.0 * (double)w_val *
                                      (double)(cc * sx * dsy * sz));
                        atomicAdd(&grad[atom_i * 3 + 2],
                                  -2.0 * (double)w_val *
                                      (double)(cc * sx * sy * dsz));

                        // dT/dA
                        auto kin1d = [&](float res[7][7], int la, int lb,
                                         float ai, float bj) -> float
                        {
                            float t = 2.0f * ai * bj * res[la + 1][lb + 1];
                            if (lb > 0)
                                t -= ai * (float)lb * res[la + 1][lb - 1];
                            if (la > 0)
                                t -= bj * (float)la * res[la - 1][lb + 1];
                            if (la > 0 && lb > 0)
                                t += 0.5f * (float)la * (float)lb *
                                     res[la - 1][lb - 1];
                            return t;
                        };
                        float tx = kin1d(res_x, lx_i, lx_j, ei, ej);
                        float ty = kin1d(res_y, ly_i, ly_j, ei, ej);
                        float tz = kin1d(res_z, lz_i, lz_j, ei, ej);
                        float dtx =
                            2.0f * ei * kin1d(res_x, lx_i + 1, lx_j, ei, ej);
                        if (lx_i > 0)
                            dtx -= (float)lx_i *
                                   kin1d(res_x, lx_i - 1, lx_j, ei, ej);
                        float dty =
                            2.0f * ei * kin1d(res_y, ly_i + 1, ly_j, ei, ej);
                        if (ly_i > 0)
                            dty -= (float)ly_i *
                                   kin1d(res_y, ly_i - 1, ly_j, ei, ej);
                        float dtz =
                            2.0f * ei * kin1d(res_z, lz_i + 1, lz_j, ei, ej);
                        if (lz_i > 0)
                            dtz -= (float)lz_i *
                                   kin1d(res_z, lz_i - 1, lz_j, ei, ej);

                        atomicAdd(
                            &grad[atom_i * 3 + 0],
                            2.0 * (double)p_val *
                                (double)(cc * (dtx * sy * sz + dsx * ty * sz +
                                               dsx * sy * tz)));
                        atomicAdd(
                            &grad[atom_i * 3 + 1],
                            2.0 * (double)p_val *
                                (double)(cc * (tx * dsy * sz + sx * dty * sz +
                                               sx * dsy * tz)));
                        atomicAdd(
                            &grad[atom_i * 3 + 2],
                            2.0 * (double)p_val *
                                (double)(cc * (tx * sy * dsz + sx * ty * dsz +
                                               sx * sy * dtz)));
                    }
                }
            }
        }
    }
}

// V 梯度核: 按 (shell_pair × atom) 并行化
// 每线程 R_vals = GRAD_R_BASE^4 = 10000 floats (~40KB)
static __global__ void OneE_V_Grad_Kernel(
    const int n_tasks, const QC_ONE_E_TASK* tasks, const VECTOR* centers,
    const int* l_list, const float* exps, const float* coeffs,
    const int* shell_offsets, const int* shell_sizes, const int* ao_offsets,
    const int* atm, const float* env, int natm, int nao_total,
    const int* shell_atom, const float* P, const float* norms, double* grad)
{
    const int total_work = n_tasks * natm;
    SIMPLE_DEVICE_FOR(flat_id, total_work)
    {
        const int task_id = flat_id / natm;
        const int iat = flat_id % natm;

        QC_ONE_E_TASK sh_idx = tasks[task_id];
        int i_sh = sh_idx.x;
        int j_sh = sh_idx.y;

        int li = l_list[i_sh], lj = l_list[j_sh];
        int ni = (li + 1) * (li + 2) / 2, nj = (lj + 1) * (lj + 2) / 2;
        int off_i = ao_offsets[i_sh], off_j = ao_offsets[j_sh];
        int atom_i = shell_atom[i_sh];
        const VECTOR A = centers[i_sh];
        const VECTOR B = centers[j_sh];
        float Ax = A.x, Ay = A.y, Az = A.z;
        float Bx = B.x, By = B.y, Bz = B.z;
        float dist_sq = (Ax - Bx) * (Ax - Bx) + (Ay - By) * (Ay - By) +
                        (Az - Bz) * (Az - Bz);

        int ptr_coord = atm[iat * 6 + 1];
        float Cx = env[ptr_coord];
        float Cy = env[ptr_coord + 1];
        float Cz = env[ptr_coord + 2];
        float Z_C = (float)atm[iat * 6];
        int L_tot = li + lj;

        for (int idx_i = 0; idx_i < ni; idx_i++)
        {
            for (int idx_j = 0; idx_j < nj; idx_j++)
            {
                int lx_i, ly_i, lz_i, lx_j, ly_j, lz_j;
                QC_Get_Lxyz_Device(li, idx_i, lx_i, ly_i, lz_i);
                QC_Get_Lxyz_Device(lj, idx_j, lx_j, ly_j, lz_j);

                int mu = off_i + idx_i;
                int nu = off_j + idx_j;
                float norm_mu_nu = norms[mu] * norms[nu];
                float p_val = P[mu * nao_total + nu] * norm_mu_nu;

                for (int pi = 0; pi < shell_sizes[i_sh]; pi++)
                {
                    float ei = exps[shell_offsets[i_sh] + pi];
                    float ci = coeffs[shell_offsets[i_sh] + pi];
                    for (int pj = 0; pj < shell_sizes[j_sh]; pj++)
                    {
                        float ej = exps[shell_offsets[j_sh] + pj];
                        float cj = coeffs[shell_offsets[j_sh] + pj];
                        float g = ei + ej;
                        float Kab = expf(-ei * ej / g * dist_sq);
                        float cc = ci * cj * Kab;
                        if (fabsf(cc) < 1e-20f) continue;

                        float Px = (ei * Ax + ej * Bx) / g;
                        float Py = (ei * Ay + ej * By) / g;
                        float Pz = (ei * Az + ej * Bz) / g;
                        float one2p = 0.5f / g;

                        // E-coefficients
                        float Ex0[6][6][11], Ey0[6][6][11], Ez0[6][6][11];
                        compute_md_coeffs(Ex0, li, lj, Px - Ax, Px - Bx, one2p);
                        compute_md_coeffs(Ey0, li, lj, Py - Ay, Py - By, one2p);
                        compute_md_coeffs(Ez0, li, lj, Pz - Az, Pz - Bz, one2p);
                        float Ex1[6][6][11], Ey1[6][6][11], Ez1[6][6][11];
                        if (lx_i + 1 < 6)
                            compute_md_coeffs(Ex1, lx_i + 1, lx_j, Px - Ax,
                                              Px - Bx, one2p);
                        if (ly_i + 1 < 6)
                            compute_md_coeffs(Ey1, ly_i + 1, ly_j, Py - Ay,
                                              Py - By, one2p);
                        if (lz_i + 1 < 6)
                            compute_md_coeffs(Ez1, lz_i + 1, lz_j, Pz - Az,
                                              Pz - Bz, one2p);

                        float PC2 = (Px - Cx) * (Px - Cx) +
                                    (Py - Cy) * (Py - Cy) +
                                    (Pz - Cz) * (Pz - Cz);
                        float PC[3] = {Px - Cx, Py - Cy, Pz - Cz};

                        double F_vals[GRAD_R_BASE];
                        float R_vals[GRAD_R_BASE * GRAD_R_BASE * GRAD_R_BASE *
                                     GRAD_R_BASE];
                        compute_boys_double(F_vals, g * PC2, L_tot + 1);
                        compute_r_tensor_1e_grad(R_vals, F_vals, g, PC,
                                                 L_tot + 1);

                        float prefac = cc * (-Z_C) * (2.0f * CONSTANT_Pi / g);

                        // AO 中心 A 导数
                        double dv_dAx = 0.0, dv_dAy = 0.0, dv_dAz = 0.0;
                        const int tmax_x = lx_i + lx_j;
                        const int tmax_y = ly_i + ly_j;
                        const int tmax_z = lz_i + lz_j;
                        float dEx[11], dEy[11], dEz[11];
                        for (int t = 0; t <= tmax_x + 1; t++)
                        {
                            float d = 0.0f;
                            if (t <= (lx_i + 1) + lx_j && (lx_i + 1) < 6)
                                d += 2.0f * ei * Ex1[lx_i + 1][lx_j][t];
                            if (lx_i > 0 && t <= (lx_i - 1) + lx_j)
                                d -= (float)lx_i * Ex0[lx_i - 1][lx_j][t];
                            dEx[t] = d;
                        }
                        for (int u = 0; u <= tmax_y + 1; u++)
                        {
                            float d = 0.0f;
                            if (u <= (ly_i + 1) + ly_j && (ly_i + 1) < 6)
                                d += 2.0f * ei * Ey1[ly_i + 1][ly_j][u];
                            if (ly_i > 0 && u <= (ly_i - 1) + ly_j)
                                d -= (float)ly_i * Ey0[ly_i - 1][ly_j][u];
                            dEy[u] = d;
                        }
                        for (int v = 0; v <= tmax_z + 1; v++)
                        {
                            float d = 0.0f;
                            if (v <= (lz_i + 1) + lz_j && (lz_i + 1) < 6)
                                d += 2.0f * ei * Ez1[lz_i + 1][lz_j][v];
                            if (lz_i > 0 && v <= (lz_i - 1) + lz_j)
                                d -= (float)lz_i * Ez0[lz_i - 1][lz_j][v];
                            dEz[v] = d;
                        }

                        for (int t = 0; t <= tmax_x + 1; t++)
                        {
                            const float ex =
                                (t <= tmax_x) ? Ex0[lx_i][lx_j][t] : 0.0f;
                            const float dex = dEx[t];
                            if (fabsf(ex) < 1e-30f && fabsf(dex) < 1e-30f)
                                continue;
                            for (int u = 0; u <= tmax_y + 1; u++)
                            {
                                const float ey =
                                    (u <= tmax_y) ? Ey0[ly_i][ly_j][u] : 0.0f;
                                const float dey = dEy[u];
                                const bool in_base =
                                    (t <= tmax_x && u <= tmax_y);
                                if (fabsf(ey) < 1e-30f && fabsf(dey) < 1e-30f)
                                    continue;
                                for (int v = 0; v <= tmax_z + 1; v++)
                                {
                                    const float ez = (v <= tmax_z)
                                                         ? Ez0[lz_i][lz_j][v]
                                                         : 0.0f;
                                    const float dez = dEz[v];
                                    const float r0 =
                                        R_vals[GRAD_R_IDX(t, u, v, 0)];
                                    if (fabsf(r0) < 1e-30f) continue;
                                    const double dr = (double)r0;
                                    if (u <= tmax_y && v <= tmax_z &&
                                        fabsf(dex) > 1e-30f)
                                        dv_dAx += (double)dex * (double)ey *
                                                  (double)ez * dr;
                                    if (t <= tmax_x && v <= tmax_z &&
                                        fabsf(dey) > 1e-30f)
                                        dv_dAy += (double)ex * (double)dey *
                                                  (double)ez * dr;
                                    if (in_base && fabsf(dez) > 1e-30f)
                                        dv_dAz += (double)ex * (double)ey *
                                                  (double)dez * dr;
                                }
                            }
                        }

                        dv_dAx *= (double)prefac;
                        dv_dAy *= (double)prefac;
                        dv_dAz *= (double)prefac;

                        // 核中心 C 导数
                        double dv_dCx = 0.0, dv_dCy = 0.0, dv_dCz = 0.0;
                        for (int t = 0; t <= tmax_x; t++)
                        {
                            float ex = Ex0[lx_i][lx_j][t];
                            if (fabsf(ex) < 1e-30f) continue;
                            for (int u = 0; u <= tmax_y; u++)
                            {
                                float ey = Ey0[ly_i][ly_j][u];
                                if (fabsf(ey) < 1e-30f) continue;
                                double exy = (double)ex * (double)ey;
                                for (int v = 0; v <= tmax_z; v++)
                                {
                                    float ez = Ez0[lz_i][lz_j][v];
                                    if (fabsf(ez) < 1e-30f) continue;
                                    double eee = exy * (double)ez;
                                    dv_dCx -=
                                        eee *
                                        (double)
                                            R_vals[GRAD_R_IDX(t + 1, u, v, 0)];
                                    dv_dCy -=
                                        eee *
                                        (double)
                                            R_vals[GRAD_R_IDX(t, u + 1, v, 0)];
                                    dv_dCz -=
                                        eee *
                                        (double)
                                            R_vals[GRAD_R_IDX(t, u, v + 1, 0)];
                                }
                            }
                        }
                        dv_dCx *= (double)prefac;
                        dv_dCy *= (double)prefac;
                        dv_dCz *= (double)prefac;

                        atomicAdd(&grad[atom_i * 3 + 0],
                                  2.0 * (double)p_val * dv_dAx);
                        atomicAdd(&grad[atom_i * 3 + 1],
                                  2.0 * (double)p_val * dv_dAy);
                        atomicAdd(&grad[atom_i * 3 + 2],
                                  2.0 * (double)p_val * dv_dAz);
                        atomicAdd(&grad[iat * 3 + 0], (double)p_val * dv_dCx);
                        atomicAdd(&grad[iat * 3 + 1], (double)p_val * dv_dCy);
                        atomicAdd(&grad[iat * 3 + 2], (double)p_val * dv_dCz);
                    }
                }
            }
        }
    }
}

// Legacy combined kernel (kept for CPU path reference)
static __global__ void OneE_Grad_Kernel(
    const int n_tasks, const QC_ONE_E_TASK* tasks, const VECTOR* centers,
    const int* l_list, const float* exps, const float* coeffs,
    const int* shell_offsets, const int* shell_sizes, const int* ao_offsets,
    const int* atm, const float* env, int natm, int nao_total,
    const int* shell_atom,  // [nbas] 壳层到原子映射
    const float* P,         // [nao * nao] 密度矩阵 (归一化后)
    const float* W,         // [nao * nao] 能量加权密度矩阵
    const float* norms,     // [nao] 归一化因子
    double* grad)           // [natm * 3] 原子梯度累加器
{
    SIMPLE_DEVICE_FOR(task_id, n_tasks)
    {
        QC_ONE_E_TASK sh_idx = tasks[task_id];
        int i_sh = sh_idx.x;
        int j_sh = sh_idx.y;

        int li = l_list[i_sh], lj = l_list[j_sh];
        int ni = (li + 1) * (li + 2) / 2, nj = (lj + 1) * (lj + 2) / 2;
        int off_i = ao_offsets[i_sh], off_j = ao_offsets[j_sh];
        int atom_i = shell_atom[i_sh];
        const VECTOR A = centers[i_sh];
        const VECTOR B = centers[j_sh];
        float Ax = A.x, Ay = A.y, Az = A.z;
        float Bx = B.x, By = B.y, Bz = B.z;
        float dist_sq = (Ax - Bx) * (Ax - Bx) + (Ay - By) * (Ay - By) +
                        (Az - Bz) * (Az - Bz);

        for (int idx_i = 0; idx_i < ni; idx_i++)
        {
            for (int idx_j = 0; idx_j < nj; idx_j++)
            {
                int lx_i, ly_i, lz_i, lx_j, ly_j, lz_j;
                QC_Get_Lxyz_Device(li, idx_i, lx_i, ly_i, lz_i);
                QC_Get_Lxyz_Device(lj, idx_j, lx_j, ly_j, lz_j);

                int mu = off_i + idx_i;
                int nu = off_j + idx_j;
                float norm_mu_nu = norms[mu] * norms[nu];
                float p_val = P[mu * nao_total + nu] * norm_mu_nu;
                float w_val = W[mu * nao_total + nu] * norm_mu_nu;
                for (int pi = 0; pi < shell_sizes[i_sh]; pi++)
                {
                    float ei = exps[shell_offsets[i_sh] + pi];
                    float ci = coeffs[shell_offsets[i_sh] + pi];
                    for (int pj = 0; pj < shell_sizes[j_sh]; pj++)
                    {
                        float ej = exps[shell_offsets[j_sh] + pj];
                        float cj = coeffs[shell_offsets[j_sh] + pj];
                        float g = ei + ej;
                        float Kab = expf(-ei * ej / g * dist_sq);
                        float cc = ci * cj * Kab;
                        if (fabsf(cc) < 1e-20f) continue;

                        float Px = (ei * Ax + ej * Bx) / g;
                        float Py = (ei * Ay + ej * By) / g;
                        float Pz = (ei * Az + ej * Bz) / g;
                        float one2p = 0.5f / g;

                        // 重叠: bra 侧需要到 l+2 阶
                        // (T 的 AO 导数 kin1d(res, la+1, ...) 访问
                        // res[la+2][lb+1])
                        float res_x[7][7], res_y[7][7], res_z[7][7];
                        get_overlap1d_arr(lx_i + 2, lx_j + 1, Px - Ax, Px - Bx,
                                          g, res_x);
                        get_overlap1d_arr(ly_i + 2, ly_j + 1, Py - Ay, Py - By,
                                          g, res_y);
                        get_overlap1d_arr(lz_i + 2, lz_j + 1, Pz - Az, Pz - Bz,
                                          g, res_z);

                        float sx = res_x[lx_i][lx_j];
                        float sy = res_y[ly_i][ly_j];
                        float sz = res_z[lz_i][lz_j];

                        // === dS/dA_x ===
                        float dsx_dAx = 2.0f * ei * res_x[lx_i + 1][lx_j];
                        if (lx_i > 0)
                            dsx_dAx -= (float)lx_i * res_x[lx_i - 1][lx_j];
                        float dsy_dAy = 2.0f * ei * res_y[ly_i + 1][ly_j];
                        if (ly_i > 0)
                            dsy_dAy -= (float)ly_i * res_y[ly_i - 1][ly_j];
                        float dsz_dAz = 2.0f * ei * res_z[lz_i + 1][lz_j];
                        if (lz_i > 0)
                            dsz_dAz -= (float)lz_i * res_z[lz_i - 1][lz_j];

                        float ds_dAx = cc * dsx_dAx * sy * sz;
                        float ds_dAy = cc * sx * dsy_dAy * sz;
                        float ds_dAz = cc * sx * sy * dsz_dAz;

                        // Pulay: grad -= Tr[W · dS/dR_A] (bra ×2)
                        atomicAdd(&grad[atom_i * 3 + 0],
                                  -2.0 * (double)w_val * (double)ds_dAx);
                        atomicAdd(&grad[atom_i * 3 + 1],
                                  -2.0 * (double)w_val * (double)ds_dAy);
                        atomicAdd(&grad[atom_i * 3 + 2],
                                  -2.0 * (double)w_val * (double)ds_dAz);

                        // === dT/dA ===
                        auto kin1d = [&](float res[7][7], int la, int lb,
                                         float ai, float bj) -> float
                        {
                            float t = 2.0f * ai * bj * res[la + 1][lb + 1];
                            if (lb > 0)
                                t -= ai * (float)lb * res[la + 1][lb - 1];
                            if (la > 0)
                                t -= bj * (float)la * res[la - 1][lb + 1];
                            if (la > 0 && lb > 0)
                                t += 0.5f * (float)la * (float)lb *
                                     res[la - 1][lb - 1];
                            return t;
                        };

                        float tx = kin1d(res_x, lx_i, lx_j, ei, ej);
                        float ty = kin1d(res_y, ly_i, ly_j, ei, ej);
                        float tz = kin1d(res_z, lz_i, lz_j, ei, ej);

                        float dtx_dAx =
                            2.0f * ei * kin1d(res_x, lx_i + 1, lx_j, ei, ej);
                        if (lx_i > 0)
                            dtx_dAx -= (float)lx_i *
                                       kin1d(res_x, lx_i - 1, lx_j, ei, ej);
                        float dty_dAy =
                            2.0f * ei * kin1d(res_y, ly_i + 1, ly_j, ei, ej);
                        if (ly_i > 0)
                            dty_dAy -= (float)ly_i *
                                       kin1d(res_y, ly_i - 1, ly_j, ei, ej);
                        float dtz_dAz =
                            2.0f * ei * kin1d(res_z, lz_i + 1, lz_j, ei, ej);
                        if (lz_i > 0)
                            dtz_dAz -= (float)lz_i *
                                       kin1d(res_z, lz_i - 1, lz_j, ei, ej);

                        // T = Tx*Sy*Sz + Sx*Ty*Sz + Sx*Sy*Tz
                        float dt_dAx =
                            cc * (dtx_dAx * sy * sz + dsx_dAx * ty * sz +
                                  dsx_dAx * sy * tz);
                        float dt_dAy =
                            cc * (tx * dsy_dAy * sz + sx * dty_dAy * sz +
                                  sx * dsy_dAy * tz);
                        float dt_dAz =
                            cc * (tx * sy * dsz_dAz + sx * ty * dsz_dAz +
                                  sx * sy * dtz_dAz);

                        // grad += Tr[P · dT/dR_A] (bra ×2)
                        atomicAdd(&grad[atom_i * 3 + 0],
                                  2.0 * (double)p_val * (double)dt_dAx);
                        atomicAdd(&grad[atom_i * 3 + 1],
                                  2.0 * (double)p_val * (double)dt_dAy);
                        atomicAdd(&grad[atom_i * 3 + 2],
                                  2.0 * (double)p_val * (double)dt_dAz);

                        // === dV/dR_A ===
                        float Ex0[6][6][11], Ey0[6][6][11], Ez0[6][6][11];
                        compute_md_coeffs(Ex0, li, lj, Px - Ax, Px - Bx, one2p);
                        compute_md_coeffs(Ey0, li, lj, Py - Ay, Py - By, one2p);
                        compute_md_coeffs(Ez0, li, lj, Pz - Az, Pz - Bz, one2p);
                        float Ex1[6][6][11], Ey1[6][6][11], Ez1[6][6][11];
                        if (lx_i + 1 < 6)
                        {
                            compute_md_coeffs(Ex1, lx_i + 1, lx_j, Px - Ax,
                                              Px - Bx, one2p);
                        }
                        if (ly_i + 1 < 6)
                        {
                            compute_md_coeffs(Ey1, ly_i + 1, ly_j, Py - Ay,
                                              Py - By, one2p);
                        }
                        if (lz_i + 1 < 6)
                        {
                            compute_md_coeffs(Ez1, lz_i + 1, lz_j, Pz - Az,
                                              Pz - Bz, one2p);
                        }

                        for (int iat = 0; iat < natm; iat++)
                        {
                            int ptr_coord = atm[iat * 6 + 1];
                            float Cx = env[ptr_coord];
                            float Cy = env[ptr_coord + 1];
                            float Cz = env[ptr_coord + 2];
                            float PC2 = (Px - Cx) * (Px - Cx) +
                                        (Py - Cy) * (Py - Cy) +
                                        (Pz - Cz) * (Pz - Cz);
                            float PC[3] = {Px - Cx, Py - Cy, Pz - Cz};
                            int L_tot = li + lj;
                            float Z_C = (float)atm[iat * 6];

                            double F_vals[ONEE_MD_BASE];
                            float R_vals[ONEE_MD_BASE * ONEE_MD_BASE *
                                         ONEE_MD_BASE * ONEE_MD_BASE];
                            compute_boys_double(F_vals, g * PC2, L_tot + 1);
                            compute_r_tensor_1e(R_vals, F_vals, g, PC,
                                                L_tot + 1);

                            float prefac =
                                cc * (-Z_C) * (2.0f * CONSTANT_Pi / g);

                            // AO 中心 A 导数 — 融合三个方向到一次 (t,u,v) 遍历
                            double dv_dAx = 0.0, dv_dAy = 0.0, dv_dAz = 0.0;

                            // 预计算 dE/dA 一维数组 (避免循环内重复计算)
                            const int tmax_x = lx_i + lx_j;
                            const int tmax_y = ly_i + ly_j;
                            const int tmax_z = lz_i + lz_j;
                            float dEx[11], dEy[11], dEz[11];
                            for (int t = 0; t <= tmax_x + 1; t++)
                            {
                                float d = 0.0f;
                                if (t <= (lx_i + 1) + lx_j && (lx_i + 1) < 6)
                                    d += 2.0f * ei * Ex1[lx_i + 1][lx_j][t];
                                if (lx_i > 0 && t <= (lx_i - 1) + lx_j)
                                    d -= (float)lx_i * Ex0[lx_i - 1][lx_j][t];
                                dEx[t] = d;
                            }
                            for (int u = 0; u <= tmax_y + 1; u++)
                            {
                                float d = 0.0f;
                                if (u <= (ly_i + 1) + ly_j && (ly_i + 1) < 6)
                                    d += 2.0f * ei * Ey1[ly_i + 1][ly_j][u];
                                if (ly_i > 0 && u <= (ly_i - 1) + ly_j)
                                    d -= (float)ly_i * Ey0[ly_i - 1][ly_j][u];
                                dEy[u] = d;
                            }
                            for (int v = 0; v <= tmax_z + 1; v++)
                            {
                                float d = 0.0f;
                                if (v <= (lz_i + 1) + lz_j && (lz_i + 1) < 6)
                                    d += 2.0f * ei * Ez1[lz_i + 1][lz_j][v];
                                if (lz_i > 0 && v <= (lz_i - 1) + lz_j)
                                    d -= (float)lz_i * Ez0[lz_i - 1][lz_j][v];
                                dEz[v] = d;
                            }

                            // 单次融合遍历: 基础范围 [0..tmax_x] × [0..tmax_y]
                            // × [0..tmax_z] dv_dAx 额外需要 t = tmax_x+1;
                            // dv_dAy 额外需要 u = tmax_y+1; dv_dAz 额外需要 v =
                            // tmax_z+1
                            for (int t = 0; t <= tmax_x + 1; t++)
                            {
                                const float ex =
                                    (t <= tmax_x) ? Ex0[lx_i][lx_j][t] : 0.0f;
                                const float dex = dEx[t];
                                if (fabsf(ex) < 1e-30f && fabsf(dex) < 1e-30f)
                                    continue;
                                for (int u = 0; u <= tmax_y + 1; u++)
                                {
                                    const float ey = (u <= tmax_y)
                                                         ? Ey0[ly_i][ly_j][u]
                                                         : 0.0f;
                                    const float dey = dEy[u];
                                    // 在基础范围外只有对应导数分量非零
                                    const bool in_base =
                                        (t <= tmax_x && u <= tmax_y);
                                    if (fabsf(ey) < 1e-30f &&
                                        fabsf(dey) < 1e-30f)
                                        continue;
                                    for (int v = 0; v <= tmax_z + 1; v++)
                                    {
                                        const float ez =
                                            (v <= tmax_z) ? Ez0[lz_i][lz_j][v]
                                                          : 0.0f;
                                        const float dez = dEz[v];
                                        const float r0 =
                                            R_vals[ONEE_MD_IDX(t, u, v, 0)];
                                        if (fabsf(r0) < 1e-30f) continue;

                                        const double dr = (double)r0;
                                        // dV/dAx: dEx * ey * ez (需要
                                        // u<=tmax_y, v<=tmax_z)
                                        if (u <= tmax_y && v <= tmax_z &&
                                            fabsf(dex) > 1e-30f)
                                            dv_dAx += (double)dex * (double)ey *
                                                      (double)ez * dr;
                                        // dV/dAy: ex * dEy * ez (需要
                                        // t<=tmax_x, v<=tmax_z)
                                        if (t <= tmax_x && v <= tmax_z &&
                                            fabsf(dey) > 1e-30f)
                                            dv_dAy += (double)ex * (double)dey *
                                                      (double)ez * dr;
                                        // dV/dAz: ex * ey * dEz (需要
                                        // t<=tmax_x, u<=tmax_y)
                                        if (in_base && fabsf(dez) > 1e-30f)
                                            dv_dAz += (double)ex * (double)ey *
                                                      (double)dez * dr;
                                    }
                                }
                            }

                            dv_dAx *= (double)prefac;
                            dv_dAy *= (double)prefac;
                            dv_dAz *= (double)prefac;

                            // 核中心 C 导数: dR/dC = -R_{t+1,u,v}, etc.
                            double dv_dCx = 0.0, dv_dCy = 0.0, dv_dCz = 0.0;
                            for (int t = 0; t <= tmax_x; t++)
                            {
                                float ex = Ex0[lx_i][lx_j][t];
                                if (fabsf(ex) < 1e-30f) continue;
                                for (int u = 0; u <= tmax_y; u++)
                                {
                                    float ey = Ey0[ly_i][ly_j][u];
                                    if (fabsf(ey) < 1e-30f) continue;
                                    double exy = (double)ex * (double)ey;
                                    for (int v = 0; v <= tmax_z; v++)
                                    {
                                        float ez = Ez0[lz_i][lz_j][v];
                                        if (fabsf(ez) < 1e-30f) continue;
                                        double eee = exy * (double)ez;
                                        dv_dCx -=
                                            eee * (double)R_vals[ONEE_MD_IDX(
                                                      t + 1, u, v, 0)];
                                        dv_dCy -=
                                            eee * (double)R_vals[ONEE_MD_IDX(
                                                      t, u + 1, v, 0)];
                                        dv_dCz -=
                                            eee * (double)R_vals[ONEE_MD_IDX(
                                                      t, u, v + 1, 0)];
                                    }
                                }
                            }
                            dv_dCx *= (double)prefac;
                            dv_dCy *= (double)prefac;
                            dv_dCz *= (double)prefac;

                            // V AO 中心导数 (bra ×2)
                            atomicAdd(&grad[atom_i * 3 + 0],
                                      2.0 * (double)p_val * dv_dAx);
                            atomicAdd(&grad[atom_i * 3 + 1],
                                      2.0 * (double)p_val * dv_dAy);
                            atomicAdd(&grad[atom_i * 3 + 2],
                                      2.0 * (double)p_val * dv_dAz);
                            // V 核中心导数 (全矩阵求和已正确)
                            atomicAdd(&grad[iat * 3 + 0],
                                      (double)p_val * dv_dCx);
                            atomicAdd(&grad[iat * 3 + 1],
                                      (double)p_val * dv_dCy);
                            atomicAdd(&grad[iat * 3 + 2],
                                      (double)p_val * dv_dCz);
                        }
                    }
                }
            }
        }
    }
}

#ifndef USE_GPU
static inline void QC_Cart2Sph_Step_OneE_CPU(const float* C, const int nc,
                                             const int ns, const int leading,
                                             const int tail, const float* src,
                                             float* dst)
{
    for (int lead = 0; lead < leading; lead++)
    {
        const float* src_blk = src + (size_t)lead * nc * tail;
        float* dst_blk = dst + (size_t)lead * ns * tail;
        memset(dst_blk, 0, (size_t)ns * tail * sizeof(float));
        for (int a = 0; a < nc; a++)
        {
            const float* src_row = src_blk + (size_t)a * tail;
            for (int p = 0; p < ns; p++)
            {
                const float c = C[a * ns + p];
                if (c == 0.0f) continue;
                float* dst_row = dst_blk + (size_t)p * tail;
                for (int idx = 0; idx < tail; idx++)
                    dst_row[idx] += c * src_row[idx];
            }
        }
    }
}

static inline void QC_Cart2Sph_Shell_OneE_CPU(
    const float* U, const int nao_sph, const int off_i_cart,
    const int off_j_cart, const int off_i_sph, const int off_j_sph,
    const int ni_cart, const int nj_cart, const int ni_sph, const int nj_sph,
    float* buf0, float* buf1)
{
    float Ci[MAX_CART_SHELL * MAX_CART_SHELL];
    float Cj[MAX_CART_SHELL * MAX_CART_SHELL];
    for (int a = 0; a < ni_cart; a++)
        for (int p = 0; p < ni_sph; p++)
            Ci[a * ni_sph + p] =
                U[(off_i_cart + a) * nao_sph + (off_i_sph + p)];
    for (int b = 0; b < nj_cart; b++)
        for (int q = 0; q < nj_sph; q++)
            Cj[b * nj_sph + q] =
                U[(off_j_cart + b) * nao_sph + (off_j_sph + q)];

    QC_Cart2Sph_Step_OneE_CPU(Ci, ni_cart, ni_sph, 1, nj_cart, buf0, buf1);
    QC_Cart2Sph_Step_OneE_CPU(Cj, nj_cart, nj_sph, ni_sph, 1, buf1, buf0);
}

static inline void QC_Cart2Sph_Shell_OneE_Block3_CPU(
    const float* U, const int nao_sph, const int off_i_cart,
    const int off_j_cart, const int off_i_sph, const int off_j_sph,
    const int ni_cart, const int nj_cart, const int ni_sph, const int nj_sph,
    const std::vector<float>& src3, std::vector<float>& dst3,
    std::vector<float>& buf0, std::vector<float>& buf1)
{
    dst3.assign((size_t)ni_sph * nj_sph * 3, 0.0f);
    for (int d = 0; d < 3; d++)
    {
        for (int idx = 0; idx < ni_cart * nj_cart; idx++)
            buf0[(size_t)idx] = src3[(size_t)idx * 3 + d];
        QC_Cart2Sph_Shell_OneE_CPU(U, nao_sph, off_i_cart, off_j_cart,
                                   off_i_sph, off_j_sph, ni_cart, nj_cart,
                                   ni_sph, nj_sph, buf0.data(), buf1.data());
        for (int idx = 0; idx < ni_sph * nj_sph; idx++)
            dst3[(size_t)idx * 3 + d] = buf0[(size_t)idx];
    }
}

static inline void QC_Build_OneE_Gradient_Spherical_CPU(
    const std::vector<QC_ONE_E_TASK>& tasks, const std::vector<VECTOR>& centers,
    const std::vector<int>& l_list, const std::vector<float>& exps,
    const std::vector<float>& coeffs, const std::vector<int>& shell_offsets,
    const std::vector<int>& shell_sizes,
    const std::vector<int>& ao_offsets_cart,
    const std::vector<int>& ao_offsets_sph, const std::vector<int>& atm,
    const std::vector<float>& env, const std::vector<int>& shell_atom,
    const float* P, const float* W, const float* norms,
    const float* cart2sph_mat, const int natm, const int nao_sph, double* grad)
{
    std::vector<float> dS_cart, dT_cart, dV_A_cart, dV_C_cart;
    std::vector<float> sph_buf0, sph_buf1;
    std::vector<float> dS_sph, dT_sph, dV_A_sph, dV_C_sph_one;
    std::vector<float> src3;

    for (const QC_ONE_E_TASK& sh_idx : tasks)
    {
        const int i_sh = sh_idx.x;
        const int j_sh = sh_idx.y;
        const int li = l_list[(size_t)i_sh];
        const int lj = l_list[(size_t)j_sh];
        const int ni_cart = (li + 1) * (li + 2) / 2;
        const int nj_cart = (lj + 1) * (lj + 2) / 2;
        const int ni_sph = 2 * li + 1;
        const int nj_sph = 2 * lj + 1;
        const int off_i_cart = ao_offsets_cart[(size_t)i_sh];
        const int off_j_cart = ao_offsets_cart[(size_t)j_sh];
        const int off_i_sph = ao_offsets_sph[(size_t)i_sh];
        const int off_j_sph = ao_offsets_sph[(size_t)j_sh];
        const int atom_i = shell_atom[(size_t)i_sh];
        const VECTOR A = centers[(size_t)i_sh];
        const VECTOR B = centers[(size_t)j_sh];
        const float Ax = A.x, Ay = A.y, Az = A.z;
        const float Bx = B.x, By = B.y, Bz = B.z;
        const float dist_sq = (Ax - Bx) * (Ax - Bx) + (Ay - By) * (Ay - By) +
                              (Az - Bz) * (Az - Bz);

        const int shell_size_cart = ni_cart * nj_cart;
        dS_cart.assign((size_t)shell_size_cart * 3, 0.0f);
        dT_cart.assign((size_t)shell_size_cart * 3, 0.0f);
        dV_A_cart.assign((size_t)shell_size_cart * 3, 0.0f);
        dV_C_cart.assign((size_t)natm * shell_size_cart * 3, 0.0f);

        for (int idx_i = 0; idx_i < ni_cart; idx_i++)
        {
            for (int idx_j = 0; idx_j < nj_cart; idx_j++)
            {
                int lx_i, ly_i, lz_i, lx_j, ly_j, lz_j;
                QC_Get_Lxyz_Host(li, idx_i, lx_i, ly_i, lz_i);
                QC_Get_Lxyz_Host(lj, idx_j, lx_j, ly_j, lz_j);
                const int idx = idx_i * nj_cart + idx_j;

                for (int pi = 0; pi < shell_sizes[(size_t)i_sh]; pi++)
                {
                    const float ei =
                        exps[(size_t)shell_offsets[(size_t)i_sh] + pi];
                    const float ci =
                        coeffs[(size_t)shell_offsets[(size_t)i_sh] + pi];
                    for (int pj = 0; pj < shell_sizes[(size_t)j_sh]; pj++)
                    {
                        const float ej =
                            exps[(size_t)shell_offsets[(size_t)j_sh] + pj];
                        const float cj =
                            coeffs[(size_t)shell_offsets[(size_t)j_sh] + pj];
                        const float g = ei + ej;
                        const float Kab = expf(-ei * ej / g * dist_sq);
                        const float cc = ci * cj * Kab;
                        if (fabsf(cc) < 1e-20f) continue;

                        const float Px = (ei * Ax + ej * Bx) / g;
                        const float Py = (ei * Ay + ej * By) / g;
                        const float Pz = (ei * Az + ej * Bz) / g;
                        const float one2p = 0.5f / g;

                        float res_x[7][7], res_y[7][7], res_z[7][7];
                        get_overlap1d_arr(lx_i + 2, lx_j + 1, Px - Ax, Px - Bx,
                                          g, res_x);
                        get_overlap1d_arr(ly_i + 2, ly_j + 1, Py - Ay, Py - By,
                                          g, res_y);
                        get_overlap1d_arr(lz_i + 2, lz_j + 1, Pz - Az, Pz - Bz,
                                          g, res_z);

                        const float sx = res_x[lx_i][lx_j];
                        const float sy = res_y[ly_i][ly_j];
                        const float sz = res_z[lz_i][lz_j];

                        float dsx_dAx = 2.0f * ei * res_x[lx_i + 1][lx_j];
                        if (lx_i > 0)
                            dsx_dAx -= (float)lx_i * res_x[lx_i - 1][lx_j];
                        float dsy_dAy = 2.0f * ei * res_y[ly_i + 1][ly_j];
                        if (ly_i > 0)
                            dsy_dAy -= (float)ly_i * res_y[ly_i - 1][ly_j];
                        float dsz_dAz = 2.0f * ei * res_z[lz_i + 1][lz_j];
                        if (lz_i > 0)
                            dsz_dAz -= (float)lz_i * res_z[lz_i - 1][lz_j];

                        dS_cart[(size_t)idx * 3 + 0] += cc * dsx_dAx * sy * sz;
                        dS_cart[(size_t)idx * 3 + 1] += cc * sx * dsy_dAy * sz;
                        dS_cart[(size_t)idx * 3 + 2] += cc * sx * sy * dsz_dAz;

                        auto kin1d = [&](float res[7][7], int la, int lb,
                                         float ai, float bj) -> float
                        {
                            float t = 2.0f * ai * bj * res[la + 1][lb + 1];
                            if (lb > 0)
                                t -= ai * (float)lb * res[la + 1][lb - 1];
                            if (la > 0)
                                t -= bj * (float)la * res[la - 1][lb + 1];
                            if (la > 0 && lb > 0)
                                t += 0.5f * (float)la * (float)lb *
                                     res[la - 1][lb - 1];
                            return t;
                        };

                        const float tx = kin1d(res_x, lx_i, lx_j, ei, ej);
                        const float ty = kin1d(res_y, ly_i, ly_j, ei, ej);
                        const float tz = kin1d(res_z, lz_i, lz_j, ei, ej);

                        float dtx_dAx =
                            2.0f * ei * kin1d(res_x, lx_i + 1, lx_j, ei, ej);
                        if (lx_i > 0)
                            dtx_dAx -= (float)lx_i *
                                       kin1d(res_x, lx_i - 1, lx_j, ei, ej);
                        float dty_dAy =
                            2.0f * ei * kin1d(res_y, ly_i + 1, ly_j, ei, ej);
                        if (ly_i > 0)
                            dty_dAy -= (float)ly_i *
                                       kin1d(res_y, ly_i - 1, ly_j, ei, ej);
                        float dtz_dAz =
                            2.0f * ei * kin1d(res_z, lz_i + 1, lz_j, ei, ej);
                        if (lz_i > 0)
                            dtz_dAz -= (float)lz_i *
                                       kin1d(res_z, lz_i - 1, lz_j, ei, ej);

                        dT_cart[(size_t)idx * 3 + 0] +=
                            cc * (dtx_dAx * sy * sz + dsx_dAx * ty * sz +
                                  dsx_dAx * sy * tz);
                        dT_cart[(size_t)idx * 3 + 1] +=
                            cc * (tx * dsy_dAy * sz + sx * dty_dAy * sz +
                                  sx * dsy_dAy * tz);
                        dT_cart[(size_t)idx * 3 + 2] +=
                            cc * (tx * sy * dsz_dAz + sx * ty * dsz_dAz +
                                  sx * sy * dtz_dAz);

                        float Ex0[6][6][11], Ey0[6][6][11], Ez0[6][6][11];
                        compute_md_coeffs(Ex0, li, lj, Px - Ax, Px - Bx, one2p);
                        compute_md_coeffs(Ey0, li, lj, Py - Ay, Py - By, one2p);
                        compute_md_coeffs(Ez0, li, lj, Pz - Az, Pz - Bz, one2p);
                        float Ex1[6][6][11], Ey1[6][6][11], Ez1[6][6][11];
                        if (lx_i + 1 < 6)
                            compute_md_coeffs(Ex1, lx_i + 1, lx_j, Px - Ax,
                                              Px - Bx, one2p);
                        if (ly_i + 1 < 6)
                            compute_md_coeffs(Ey1, ly_i + 1, ly_j, Py - Ay,
                                              Py - By, one2p);
                        if (lz_i + 1 < 6)
                            compute_md_coeffs(Ez1, lz_i + 1, lz_j, Pz - Az,
                                              Pz - Bz, one2p);

                        for (int iat = 0; iat < natm; iat++)
                        {
                            const int ptr_coord = atm[(size_t)iat * 6 + 1];
                            const float Cx = env[(size_t)ptr_coord];
                            const float Cy = env[(size_t)ptr_coord + 1];
                            const float Cz = env[(size_t)ptr_coord + 2];
                            const float PC2 = (Px - Cx) * (Px - Cx) +
                                              (Py - Cy) * (Py - Cy) +
                                              (Pz - Cz) * (Pz - Cz);
                            float PC[3] = {Px - Cx, Py - Cy, Pz - Cz};
                            const int L_tot = li + lj;
                            const float Z_C = (float)atm[(size_t)iat * 6];

                            double F_vals[ONEE_MD_BASE];
                            float R_vals[ONEE_MD_BASE * ONEE_MD_BASE *
                                         ONEE_MD_BASE * ONEE_MD_BASE];
                            compute_boys_double(F_vals, g * PC2, L_tot + 1);
                            compute_r_tensor_1e(R_vals, F_vals, g, PC,
                                                L_tot + 1);

                            const float prefac =
                                cc * (-Z_C) * (2.0f * CONSTANT_Pi / g);

                            double dv_dAx = 0.0, dv_dAy = 0.0, dv_dAz = 0.0;
                            for (int t = 0; t <= lx_i + lx_j + 1; t++)
                            {
                                float dex = 0.0f;
                                if (t <= (lx_i + 1) + lx_j && (lx_i + 1) < 6)
                                    dex += 2.0f * ei * Ex1[lx_i + 1][lx_j][t];
                                if (lx_i > 0 && t <= (lx_i - 1) + lx_j)
                                    dex -= (float)lx_i * Ex0[lx_i - 1][lx_j][t];

                                for (int u = 0; u <= ly_i + ly_j; u++)
                                {
                                    const float ey = Ey0[ly_i][ly_j][u];
                                    if (fabsf(ey) < 1e-30f &&
                                        fabsf(dex) < 1e-30f)
                                        continue;
                                    for (int v = 0; v <= lz_i + lz_j; v++)
                                    {
                                        const float ez = Ez0[lz_i][lz_j][v];
                                        const float r0 =
                                            R_vals[ONEE_MD_IDX(t, u, v, 0)];
                                        if (fabsf(dex) > 1e-30f)
                                            dv_dAx += (double)dex * (double)ey *
                                                      (double)ez * (double)r0;
                                    }
                                }
                            }
                            for (int t = 0; t <= lx_i + lx_j; t++)
                            {
                                const float ex = Ex0[lx_i][lx_j][t];
                                if (fabsf(ex) < 1e-30f) continue;
                                for (int u = 0; u <= ly_i + ly_j + 1; u++)
                                {
                                    float dey = 0.0f;
                                    if (u <= (ly_i + 1) + ly_j &&
                                        (ly_i + 1) < 6)
                                        dey +=
                                            2.0f * ei * Ey1[ly_i + 1][ly_j][u];
                                    if (ly_i > 0 && u <= (ly_i - 1) + ly_j)
                                        dey -= (float)ly_i *
                                               Ey0[ly_i - 1][ly_j][u];
                                    if (fabsf(dey) < 1e-30f) continue;
                                    for (int v = 0; v <= lz_i + lz_j; v++)
                                    {
                                        const float ez = Ez0[lz_i][lz_j][v];
                                        const float r0 =
                                            R_vals[ONEE_MD_IDX(t, u, v, 0)];
                                        dv_dAy += (double)ex * (double)dey *
                                                  (double)ez * (double)r0;
                                    }
                                }
                            }
                            for (int t = 0; t <= lx_i + lx_j; t++)
                            {
                                const float ex = Ex0[lx_i][lx_j][t];
                                if (fabsf(ex) < 1e-30f) continue;
                                for (int u = 0; u <= ly_i + ly_j; u++)
                                {
                                    const float ey = Ey0[ly_i][ly_j][u];
                                    if (fabsf(ey) < 1e-30f) continue;
                                    for (int v = 0; v <= lz_i + lz_j + 1; v++)
                                    {
                                        float dez = 0.0f;
                                        if (v <= (lz_i + 1) + lz_j &&
                                            (lz_i + 1) < 6)
                                            dez += 2.0f * ei *
                                                   Ez1[lz_i + 1][lz_j][v];
                                        if (lz_i > 0 && v <= (lz_i - 1) + lz_j)
                                            dez -= (float)lz_i *
                                                   Ez0[lz_i - 1][lz_j][v];
                                        const float r0 =
                                            R_vals[ONEE_MD_IDX(t, u, v, 0)];
                                        dv_dAz += (double)ex * (double)ey *
                                                  (double)dez * (double)r0;
                                    }
                                }
                            }

                            dV_A_cart[(size_t)idx * 3 + 0] +=
                                (float)(dv_dAx * (double)prefac);
                            dV_A_cart[(size_t)idx * 3 + 1] +=
                                (float)(dv_dAy * (double)prefac);
                            dV_A_cart[(size_t)idx * 3 + 2] +=
                                (float)(dv_dAz * (double)prefac);

                            double dv_dCx = 0.0, dv_dCy = 0.0, dv_dCz = 0.0;
                            for (int t = 0; t <= lx_i + lx_j; t++)
                            {
                                const float ex = Ex0[lx_i][lx_j][t];
                                if (fabsf(ex) < 1e-30f) continue;
                                for (int u = 0; u <= ly_i + ly_j; u++)
                                {
                                    const float ey = Ey0[ly_i][ly_j][u];
                                    if (fabsf(ey) < 1e-30f) continue;
                                    for (int v = 0; v <= lz_i + lz_j; v++)
                                    {
                                        const float ez = Ez0[lz_i][lz_j][v];
                                        if (fabsf(ez) < 1e-30f) continue;
                                        const double eee = (double)ex *
                                                           (double)ey *
                                                           (double)ez;
                                        dv_dCx -=
                                            eee * (double)R_vals[ONEE_MD_IDX(
                                                      t + 1, u, v, 0)];
                                        dv_dCy -=
                                            eee * (double)R_vals[ONEE_MD_IDX(
                                                      t, u + 1, v, 0)];
                                        dv_dCz -=
                                            eee * (double)R_vals[ONEE_MD_IDX(
                                                      t, u, v + 1, 0)];
                                    }
                                }
                            }
                            dV_C_cart[((size_t)iat * shell_size_cart + idx) *
                                          3 +
                                      0] += (float)(dv_dCx * (double)prefac);
                            dV_C_cart[((size_t)iat * shell_size_cart + idx) *
                                          3 +
                                      1] += (float)(dv_dCy * (double)prefac);
                            dV_C_cart[((size_t)iat * shell_size_cart + idx) *
                                          3 +
                                      2] += (float)(dv_dCz * (double)prefac);
                        }
                    }
                }
            }
        }

        sph_buf0.assign((size_t)ni_cart * nj_cart, 0.0f);
        sph_buf1.assign((size_t)std::max(ni_sph * nj_cart, ni_cart * nj_sph),
                        0.0f);
        QC_Cart2Sph_Shell_OneE_Block3_CPU(cart2sph_mat, nao_sph, off_i_cart,
                                          off_j_cart, off_i_sph, off_j_sph,
                                          ni_cart, nj_cart, ni_sph, nj_sph,
                                          dS_cart, dS_sph, sph_buf0, sph_buf1);
        QC_Cart2Sph_Shell_OneE_Block3_CPU(cart2sph_mat, nao_sph, off_i_cart,
                                          off_j_cart, off_i_sph, off_j_sph,
                                          ni_cart, nj_cart, ni_sph, nj_sph,
                                          dT_cart, dT_sph, sph_buf0, sph_buf1);
        QC_Cart2Sph_Shell_OneE_Block3_CPU(
            cart2sph_mat, nao_sph, off_i_cart, off_j_cart, off_i_sph, off_j_sph,
            ni_cart, nj_cart, ni_sph, nj_sph, dV_A_cart, dV_A_sph, sph_buf0,
            sph_buf1);

        for (int ci = 0; ci < ni_sph; ci++)
        {
            const int p = off_i_sph + ci;
            const float norm_p = norms[p];
            const int pn = p * nao_sph;
            for (int cj = 0; cj < nj_sph; cj++)
            {
                const int q = off_j_sph + cj;
                const float scale = norm_p * norms[q];
                const int idx = ci * nj_sph + cj;
                const float p_val = P[pn + q];
                const float w_val = W[pn + q];
                for (int d = 0; d < 3; d++)
                {
                    const double ds =
                        (double)dS_sph[(size_t)idx * 3 + d] * (double)scale;
                    const double dt =
                        (double)dT_sph[(size_t)idx * 3 + d] * (double)scale;
                    const double dvA =
                        (double)dV_A_sph[(size_t)idx * 3 + d] * (double)scale;
                    grad[atom_i * 3 + d] += -2.0 * (double)w_val * ds +
                                            2.0 * (double)p_val * (dt + dvA);
                }
            }
        }

        for (int iat = 0; iat < natm; iat++)
        {
            const float* src_iat =
                dV_C_cart.data() + (size_t)iat * shell_size_cart * 3;
            src3.assign(src_iat, src_iat + (size_t)shell_size_cart * 3);
            QC_Cart2Sph_Shell_OneE_Block3_CPU(
                cart2sph_mat, nao_sph, off_i_cart, off_j_cart, off_i_sph,
                off_j_sph, ni_cart, nj_cart, ni_sph, nj_sph, src3, dV_C_sph_one,
                sph_buf0, sph_buf1);
            for (int ci = 0; ci < ni_sph; ci++)
            {
                const int p = off_i_sph + ci;
                const float norm_p = norms[p];
                const int pn = p * nao_sph;
                for (int cj = 0; cj < nj_sph; cj++)
                {
                    const int q = off_j_sph + cj;
                    const double scale = (double)norm_p * (double)norms[q];
                    const float p_val = P[pn + q];
                    const int idx = ci * nj_sph + cj;
                    for (int d = 0; d < 3; d++)
                    {
                        grad[iat * 3 + d] +=
                            (double)p_val *
                            ((double)dV_C_sph_one[(size_t)idx * 3 + d] * scale);
                    }
                }
            }
        }
    }
}
#endif

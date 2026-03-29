#pragma once

#include "../integrals/one_e.hpp"

// ====================== 单电子积分导数 ======================
// 计算 dS/dR, dT/dR, dV/dR 对原子梯度的贡献:
//   grad_A -= Tr[W · dS/dR_A]      (Pulay)
//   grad_A += Tr[P · dT/dR_A]      (动能)
//   grad_A += Tr[P · dV/dR_A]      (核吸引)
//
// 全矩阵遍历只计算 bra 侧导数，乘以 2 补偿 ket 侧
// (等价于 PySCF 的 h1ao + h1ao.T 对称化)
//
// McMurchie-Davidson 导数公式:
//   dS(a,b)/dA_x = 2αi·S(a+1,b) - a·S(a-1,b)  [重叠]
//   dE^{ab}_t/dA_x = 2αi·E^{(a+1)b}_t - a·E^{(a-1)b}_t  [E系数]
//   dR_{tuv,0}/dC_x = -R_{(t+1)uv,0}  [R-tensor, 核中心]
// ==============================================================

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
                        // (T 的 AO 导数 kin1d(res, la+1, ...) 访问 res[la+2][lb+1])
                        float res_x[6][6], res_y[6][6], res_z[6][6];
                        get_overlap1d_arr(lx_i + 2, lx_j + 1, Px - Ax,
                                          Px - Bx, g, res_x);
                        get_overlap1d_arr(ly_i + 2, ly_j + 1, Py - Ay,
                                          Py - By, g, res_y);
                        get_overlap1d_arr(lz_i + 2, lz_j + 1, Pz - Az,
                                          Pz - Bz, g, res_z);

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
                        auto kin1d = [&](float res[6][6], int la, int lb,
                                        float ai, float bj) -> float
                        {
                            float t = 2.0f * ai * bj * res[la + 1][lb + 1];
                            if (lb > 0) t -= ai * (float)lb * res[la + 1][lb - 1];
                            if (la > 0) t -= bj * (float)la * res[la - 1][lb + 1];
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
                            dtx_dAx -=
                                (float)lx_i *
                                kin1d(res_x, lx_i - 1, lx_j, ei, ej);
                        float dty_dAy =
                            2.0f * ei * kin1d(res_y, ly_i + 1, ly_j, ei, ej);
                        if (ly_i > 0)
                            dty_dAy -=
                                (float)ly_i *
                                kin1d(res_y, ly_i - 1, ly_j, ei, ej);
                        float dtz_dAz =
                            2.0f * ei * kin1d(res_z, lz_i + 1, lz_j, ei, ej);
                        if (lz_i > 0)
                            dtz_dAz -=
                                (float)lz_i *
                                kin1d(res_z, lz_i - 1, lz_j, ei, ej);

                        // T = Tx*Sy*Sz + Sx*Ty*Sz + Sx*Sy*Tz
                        float dt_dAx = cc * (dtx_dAx * sy * sz +
                                             dsx_dAx * ty * sz +
                                             dsx_dAx * sy * tz);
                        float dt_dAy = cc * (tx * dsy_dAy * sz +
                                             sx * dty_dAy * sz +
                                             sx * dsy_dAy * tz);
                        float dt_dAz = cc * (tx * sy * dsz_dAz +
                                             sx * ty * dsz_dAz +
                                             sx * sy * dtz_dAz);

                        // grad += Tr[P · dT/dR_A] (bra ×2)
                        atomicAdd(&grad[atom_i * 3 + 0],
                                  2.0 * (double)p_val * (double)dt_dAx);
                        atomicAdd(&grad[atom_i * 3 + 1],
                                  2.0 * (double)p_val * (double)dt_dAy);
                        atomicAdd(&grad[atom_i * 3 + 2],
                                  2.0 * (double)p_val * (double)dt_dAz);

                        // === dV/dR_A ===
                        float Ex0[5][5][9], Ey0[5][5][9], Ez0[5][5][9];
                        compute_md_coeffs(Ex0, li, lj, Px - Ax, Px - Bx,
                                          one2p);
                        compute_md_coeffs(Ey0, li, lj, Py - Ay, Py - By,
                                          one2p);
                        compute_md_coeffs(Ez0, li, lj, Pz - Az, Pz - Bz,
                                          one2p);
                        float Ex1[5][5][9], Ey1[5][5][9], Ez1[5][5][9];
                        if (lx_i + 1 < 5)
                        {
                            compute_md_coeffs(Ex1, lx_i + 1, lx_j, Px - Ax,
                                              Px - Bx, one2p);
                        }
                        if (ly_i + 1 < 5)
                        {
                            compute_md_coeffs(Ey1, ly_i + 1, ly_j, Py - Ay,
                                              Py - By, one2p);
                        }
                        if (lz_i + 1 < 5)
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

                            float prefac = cc * (-Z_C) * (2.0f * CONSTANT_Pi / g);

                            // AO 中心 A 导数
                            double dv_dAx = 0.0, dv_dAy = 0.0, dv_dAz = 0.0;

                            for (int t = 0; t <= lx_i + lx_j + 1; t++)
                            {
                                float dex = 0.0f;
                                if (t <= (lx_i + 1) + lx_j && (lx_i + 1) < 5)
                                    dex += 2.0f * ei *
                                           Ex1[lx_i + 1][lx_j][t];
                                if (lx_i > 0 && t <= (lx_i - 1) + lx_j)
                                    dex -= (float)lx_i *
                                           Ex0[lx_i - 1][lx_j][t];

                                for (int u = 0; u <= ly_i + ly_j; u++)
                                {
                                    float ey = Ey0[ly_i][ly_j][u];
                                    if (fabsf(ey) < 1e-30f && fabsf(dex) < 1e-30f)
                                        continue;
                                    for (int v = 0; v <= lz_i + lz_j; v++)
                                    {
                                        float ez = Ez0[lz_i][lz_j][v];
                                        float r0 = R_vals[ONEE_MD_IDX(t, u, v, 0)];

                                        if (fabsf(dex) > 1e-30f)
                                            dv_dAx += (double)dex * (double)ey *
                                                      (double)ez * (double)r0;
                                    }
                                }
                            }
                            for (int t = 0; t <= lx_i + lx_j; t++)
                            {
                                float ex = Ex0[lx_i][lx_j][t];
                                if (fabsf(ex) < 1e-30f) continue;
                                for (int u = 0; u <= ly_i + ly_j + 1; u++)
                                {
                                    float dey = 0.0f;
                                    if (u <= (ly_i + 1) + ly_j && (ly_i + 1) < 5)
                                        dey += 2.0f * ei *
                                               Ey1[ly_i + 1][ly_j][u];
                                    if (ly_i > 0 && u <= (ly_i - 1) + ly_j)
                                        dey -= (float)ly_i *
                                               Ey0[ly_i - 1][ly_j][u];
                                    if (fabsf(dey) < 1e-30f) continue;
                                    for (int v = 0; v <= lz_i + lz_j; v++)
                                    {
                                        float ez = Ez0[lz_i][lz_j][v];
                                        float r0 = R_vals[ONEE_MD_IDX(t, u, v, 0)];
                                        dv_dAy += (double)ex * (double)dey *
                                                  (double)ez * (double)r0;
                                    }
                                }
                            }
                            for (int t = 0; t <= lx_i + lx_j; t++)
                            {
                                float ex = Ex0[lx_i][lx_j][t];
                                if (fabsf(ex) < 1e-30f) continue;
                                for (int u = 0; u <= ly_i + ly_j; u++)
                                {
                                    float ey = Ey0[ly_i][ly_j][u];
                                    if (fabsf(ey) < 1e-30f) continue;
                                    for (int v = 0; v <= lz_i + lz_j + 1; v++)
                                    {
                                        float dez = 0.0f;
                                        if (v <= (lz_i + 1) + lz_j &&
                                            (lz_i + 1) < 5)
                                            dez += 2.0f * ei *
                                                   Ez1[lz_i + 1][lz_j][v];
                                        if (lz_i > 0 && v <= (lz_i - 1) + lz_j)
                                            dez -= (float)lz_i *
                                                   Ez0[lz_i - 1][lz_j][v];
                                        float r0 = R_vals[ONEE_MD_IDX(t, u, v, 0)];
                                        dv_dAz += (double)ex * (double)ey *
                                                  (double)dez * (double)r0;
                                    }
                                }
                            }

                            dv_dAx *= (double)prefac;
                            dv_dAy *= (double)prefac;
                            dv_dAz *= (double)prefac;

                            // 核中心 C 导数
                            double dv_dCx = 0.0, dv_dCy = 0.0, dv_dCz = 0.0;
                            for (int t = 0; t <= lx_i + lx_j; t++)
                            {
                                float ex = Ex0[lx_i][lx_j][t];
                                if (fabsf(ex) < 1e-30f) continue;
                                for (int u = 0; u <= ly_i + ly_j; u++)
                                {
                                    float ey = Ey0[ly_i][ly_j][u];
                                    if (fabsf(ey) < 1e-30f) continue;
                                    for (int v = 0; v <= lz_i + lz_j; v++)
                                    {
                                        float ez = Ez0[lz_i][lz_j][v];
                                        if (fabsf(ez) < 1e-30f) continue;
                                        double eee = (double)ex * (double)ey *
                                                     (double)ez;
                                        dv_dCx -= eee * (double)R_vals
                                                            [ONEE_MD_IDX(
                                                                t + 1, u, v, 0)];
                                        dv_dCy -= eee * (double)R_vals
                                                            [ONEE_MD_IDX(
                                                                t, u + 1, v, 0)];
                                        dv_dCz -= eee * (double)R_vals
                                                            [ONEE_MD_IDX(
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

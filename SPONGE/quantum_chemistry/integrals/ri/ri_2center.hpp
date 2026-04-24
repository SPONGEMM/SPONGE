#pragma once

// 二中心 Coulomb 积分 (P|Q) = ∫∫ P(r1) 1/r12 Q(r2) dr1 dr2
// 使用 McMurchie-Davidson 方案，复用 one_e.hpp 中的 Boys 函数和 R 张量

#include "../one_e.hpp"

// 二中心 Coulomb 积分内核
// 每个线程处理一个辅助 shell 对 (P_sh, Q_sh)
static __global__ void QC_RI_2Center_Kernel(
    const int n_tasks, const QC_ONE_E_TASK* tasks, const VECTOR* centers,
    const int* l_list, const float* exps, const float* coeffs,
    const int* shell_offsets, const int* shell_sizes, const int* ao_offsets,
    int naux, double* out_metric)
{
    SIMPLE_DEVICE_FOR(task_id, n_tasks)
    {
        const QC_ONE_E_TASK sh = tasks[task_id];
        const int P_sh = sh.x;
        const int Q_sh = sh.y;

        const int lP = l_list[P_sh], lQ = l_list[Q_sh];
        const int nP = (lP + 1) * (lP + 2) / 2;
        const int nQ = (lQ + 1) * (lQ + 2) / 2;
        const int offP = ao_offsets[P_sh];
        const int offQ = ao_offsets[Q_sh];

        const VECTOR A = centers[P_sh];
        const VECTOR B = centers[Q_sh];
        const float Ax = A.x, Ay = A.y, Az = A.z;
        const float Bx = B.x, By = B.y, Bz = B.z;
        const float dist_sq = (Ax - Bx) * (Ax - Bx) + (Ay - By) * (Ay - By) +
                              (Az - Bz) * (Az - Bz);

        for (int idxP = 0; idxP < nP; idxP++)
        {
            for (int idxQ = 0; idxQ < nQ; idxQ++)
            {
                int lxP, lyP, lzP, lxQ, lyQ, lzQ;
                QC_Get_Lxyz_Device(lP, idxP, lxP, lyP, lzP);
                QC_Get_Lxyz_Device(lQ, idxQ, lxQ, lyQ, lzQ);

                double total = 0.0;

                for (int pi = 0; pi < shell_sizes[P_sh]; pi++)
                {
                    const float eP = exps[shell_offsets[P_sh] + pi];
                    const float cP = coeffs[shell_offsets[P_sh] + pi];

                    for (int pj = 0; pj < shell_sizes[Q_sh]; pj++)
                    {
                        const float eQ = exps[shell_offsets[Q_sh] + pj];
                        const float cQ = coeffs[shell_offsets[Q_sh] + pj];

                        // McMurchie-Davidson: each single primitive expands
                        // in Hermite basis with E-coefficients at its own
                        // center (PA=0), then R-tensor couples bra/ket.
                        //
                        // (P|Q) = 2π^{5/2}/(p*q*√(p+q)) *
                        //   Σ E^P_{tuv} E^Q_{τμν} (-1)^{τ+μ+ν}
                        //     R_{t+τ,u+μ,v+ν}(α_PQ, A-B)
                        // where α_PQ = p*q/(p+q)

                        float E_Px[6][6][11], E_Py[6][6][11], E_Pz[6][6][11];
                        compute_md_coeffs(E_Px, lxP, 0, 0.0f, 0.0f, 0.5f / eP);
                        compute_md_coeffs(E_Py, lyP, 0, 0.0f, 0.0f, 0.5f / eP);
                        compute_md_coeffs(E_Pz, lzP, 0, 0.0f, 0.0f, 0.5f / eP);

                        float E_Qx[6][6][11], E_Qy[6][6][11], E_Qz[6][6][11];
                        compute_md_coeffs(E_Qx, lxQ, 0, 0.0f, 0.0f, 0.5f / eQ);
                        compute_md_coeffs(E_Qy, lyQ, 0, 0.0f, 0.0f, 0.5f / eQ);
                        compute_md_coeffs(E_Qz, lzQ, 0, 0.0f, 0.0f, 0.5f / eQ);

                        const float alpha_pq = eP * eQ / (eP + eQ);
                        const float T_val = alpha_pq * dist_sq;
                        const int L_tot = lP + lQ;
                        double F_vals[ONEE_MD_BASE];
                        float R_vals[ONEE_MD_BASE * ONEE_MD_BASE *
                                     ONEE_MD_BASE * ONEE_MD_BASE];
                        compute_boys_double(F_vals, T_val, L_tot);
                        float AB[3] = {Ax - Bx, Ay - By, Az - Bz};
                        compute_r_tensor_1e(R_vals, F_vals, alpha_pq, AB,
                                            L_tot);

                        const double prefactor =
                            (double)cP * (double)cQ *
                            (2.0 * CONSTANT_Pi * CONSTANT_Pi *
                             sqrt(CONSTANT_Pi)) /
                            ((double)eP * (double)eQ * sqrt((double)(eP + eQ)));

                        double v_sum = 0.0;
                        for (int t = 0; t <= lxP; t++)
                        {
                            double ePx = (double)E_Px[lxP][0][t];
                            if (ePx == 0.0) continue;
                            for (int u = 0; u <= lyP; u++)
                            {
                                double ePy = (double)E_Py[lyP][0][u];
                                if (ePy == 0.0) continue;
                                for (int v = 0; v <= lzP; v++)
                                {
                                    double ePz = (double)E_Pz[lzP][0][v];
                                    if (ePz == 0.0) continue;
                                    for (int tt = 0; tt <= lxQ; tt++)
                                    {
                                        double eQx = (double)E_Qx[lxQ][0][tt];
                                        if (eQx == 0.0) continue;
                                        for (int uu = 0; uu <= lyQ; uu++)
                                        {
                                            double eQy =
                                                (double)E_Qy[lyQ][0][uu];
                                            if (eQy == 0.0) continue;
                                            for (int vv = 0; vv <= lzQ; vv++)
                                            {
                                                double eQz =
                                                    (double)E_Qz[lzQ][0][vv];
                                                if (eQz == 0.0) continue;
                                                double sign =
                                                    ((tt + uu + vv) & 1) ? -1.0
                                                                         : 1.0;
                                                v_sum +=
                                                    ePx * ePy * ePz * eQx *
                                                    eQy * eQz * sign *
                                                    (double)R_vals[ONEE_MD_IDX(
                                                        t + tt, u + uu, v + vv,
                                                        0)];
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        total += prefactor * v_sum;
                    }
                }

                const int P_idx = offP + idxP;
                const int Q_idx = offQ + idxQ;
                out_metric[P_idx * naux + Q_idx] = total;
                if (P_sh != Q_sh) out_metric[Q_idx * naux + P_idx] = total;
            }
        }
    }
}

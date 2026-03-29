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

                        const float g = eP + eQ;
                        const float Kab = expf(-eP * eQ / g * dist_sq);

                        // Gaussian product center
                        const float Px = (eP * Ax + eQ * Bx) / g;
                        const float Py = (eP * Ay + eQ * By) / g;
                        const float Pz = (eP * Az + eQ * Bz) / g;

                        // McMurchie-Davidson E-coefficients for (P|Q)
                        // For 2-center Coulomb, we treat it as a 1-center
                        // problem after forming the Gaussian product:
                        // (P|Q) = 2π/g * Σ E^P_t E^P_u E^P_v R_{tuv}(0,g,P)
                        // where R is the auxiliary Hermite Coulomb integral
                        // (自身中心, PC=0)

                        // Actually for 2-center Coulomb:
                        // (a|b) = Σ_prim ca*cb * Kab * 2π/(a+b) *
                        //         Σ_{tuv} E^ab_t * E^ab_u * E^ab_v *
                        //         R_{tuv}^0(0, a+b, 0)
                        // where R_{tuv}^0 at zero distance simplifies

                        // More precisely, for 2-center ERI (a|b):
                        // This is ∫∫ a(r1) 1/r12 b(r2) dr1 dr2
                        // = (2π^{5/2}) / (p*q * sqrt(p+q)) * Σ E * R
                        // But for 2-center (not 4-center), we have:
                        // primitive pair P=(a,b) at center P:
                        // (a|b) = Kab * (2π/g) * F_0(0) for s-type
                        //       = Kab * (2π/g) * Σ E_t E_u E_v R_{tuv}(0)

                        // For 2c integral: T = 0 (self-repulsion of product)
                        // R_{tuv}^n(0) = δ_{t0}δ_{u0}δ_{v0} * F_n(0) * (-2g)^n
                        // But this is wrong - 2c Coulomb is NOT self-repulsion

                        // Correct: (P|Q) = ∫∫ P(r1) |r1-r2|^{-1} Q(r2) dr1 dr2
                        // For primitives a*exp(-α|r-A|²) and b*exp(-β|r-B|²):
                        // = ab * (2π^{5/2}) / (αβ√(α+β)) * exp(-αβ/(α+β)|A-B|²)
                        //   * Σ_{tuv} E^x_t E^y_u E^z_v * (-1)^{t+u+v} *
                        //     F_{t+u+v}(0)
                        // Wait, this isn't right either. Let me think again.

                        // The correct formula for (a|b) 2-center Coulomb integral:
                        // Use Fourier transform: 1/r12 = (2/√π) ∫ exp(-t²r12²) dt
                        // After integration over r1 and r2:
                        //
                        // (P|Q) = Kab * 2π^{5/2} / (α * β * √(α+β))
                        //       * ... not the standard MD
                        //
                        // Actually, the proper approach is to recognize that
                        // a 2-center Coulomb integral is just a special case
                        // of the 4-center ERI (Ps|Qs) where s has exponent=0.
                        // But that's problematic.
                        //
                        // Alternative: treat as nuclear attraction but between
                        // two shells. The key insight is:
                        //
                        // (P|Q) = ∫ P(r1) V_Q(r1) dr1
                        // where V_Q(r1) = ∫ Q(r2)/|r1-r2| dr2
                        //               = (2π/β) Σ_{tuv} E^Q_{tuv} R_{tuv}(β,PQ)
                        //
                        // Then (P|Q) = Σ E^P * Σ E^Q * R
                        //
                        // This is essentially the nuclear attraction integral
                        // but with Q acting as the "nucleus" (distributed charge).

                        // For the McMurchie-Davidson approach:
                        // (a|b) = K_ab * (2π^{5/2})/(p*q*√(p+q)) *
                        //   Σ_{t,u,v} E^P_{t} E^P_{u} E^P_{v} *
                        //   Σ_{τ,μ,ν} E^Q_{τ} E^Q_{μ} E^Q_{ν} *
                        //   R_{t+τ,u+μ,v+ν}(α_PQ, PQ)
                        //
                        // where α_PQ = p*q/(p+q), PQ = P-Q (product centers)
                        // and K_ab = exp(-p*q/(p+q) * |P-Q|²) ... wait no.
                        //
                        // Let me use the correct 4-center formulation with
                        // two dummy s-shells:
                        // (P 1s | Q 1s) where 1s has exp=0 at same center.
                        // This is just: (PQ|) for the product.
                        //
                        // Better: use the Obara-Saika / MD approach directly.
                        // For 2-electron 2-center:
                        // bra: shell P, primitive (cP, eP, center A)
                        // ket: shell Q, primitive (cQ, eQ, center B)
                        // The bra product center is just A (single function),
                        // ket product center is just B.
                        // Then:
                        //   p = eP, q = eQ, alpha = p*q/(p+q)
                        //   W = (p*P + q*Q)/(p+q) (weighted center)
                        //   T = alpha * |P-Q|²
                        //
                        // The integral becomes:
                        // (P|Q) = cP*cQ * N_P * N_Q * 2π^{5/2}/(p*q*√(p+q))
                        //       * Σ E^P_t(p,A) * E^Q_τ(q,B) * R_{t+τ}(alpha,W-?)
                        //
                        // OK, I'm overcomplicating this. Let me use the simple
                        // Hermite expansion directly.

                        // For a single primitive pair:
                        // The bra is a single Gaussian P at center A with exp p
                        // The ket is a single Gaussian Q at center B with exp q
                        // They don't form products within bra/ket (no shell pair)
                        // Instead, the Coulomb integral is:
                        //
                        // (P|Q) = 2π^{5/2} / (p * q * √(p+q)) * exp(-α|AB|²)
                        //       * Σ_{tuv,τμν} E^P_{t,u,v} * E^Q_{τ,μ,ν}
                        //         * R_{t+τ, u+μ, v+ν}(α, W-P_center)
                        //
                        // But for single primitives, E^P has PA=0, PB=0
                        // (center = A itself), so E^P_{lx,0} = δ(all indices)
                        // No wait, P and Q are each single Gaussians, so
                        // the E-coefficients are identity-like.

                        // Actually for a SINGLE Gaussian a*exp(-α|r-A|²) * x^lx:
                        // E_t coefficients are: E_t(lx, 0) where only one
                        // center is involved (PA = 0, one_over_2p = 1/(2α))
                        // So E_t is nonzero only for t having same parity as lx
                        // and t ≤ lx.

                        // Let me just use the simple approach: form product
                        // within each center and use R-tensor between products.

                        // E-coefficients for P shell at center A:
                        // These expand |lx,ly,lz⟩ in Hermite basis
                        // For a single center: E^x_t(lx, 0, one_over_2p=1/(2eP))
                        float E_Px[5][5][9], E_Py[5][5][9], E_Pz[5][5][9];
                        compute_md_coeffs(E_Px, lxP, 0, 0.0f, 0.0f,
                                          0.5f / eP);
                        compute_md_coeffs(E_Py, lyP, 0, 0.0f, 0.0f,
                                          0.5f / eP);
                        compute_md_coeffs(E_Pz, lzP, 0, 0.0f, 0.0f,
                                          0.5f / eP);

                        // E-coefficients for Q shell at center B:
                        float E_Qx[5][5][9], E_Qy[5][5][9], E_Qz[5][5][9];
                        compute_md_coeffs(E_Qx, lxQ, 0, 0.0f, 0.0f,
                                          0.5f / eQ);
                        compute_md_coeffs(E_Qy, lyQ, 0, 0.0f, 0.0f,
                                          0.5f / eQ);
                        compute_md_coeffs(E_Qz, lzQ, 0, 0.0f, 0.0f,
                                          0.5f / eQ);

                        // R-tensor between bra center A and ket center B
                        const float alpha_pq = eP * eQ / (eP + eQ);
                        // Weighted center
                        const float Wx = (eP * Ax + eQ * Bx) / (eP + eQ);
                        const float Wy = (eP * Ay + eQ * By) / (eP + eQ);
                        const float Wz = (eP * Az + eQ * Bz) / (eP + eQ);

                        // For the R-tensor, we need PC = W - A for bra-side
                        // and PC = W - B for ket-side. But in the Coulomb
                        // integral, the R-tensor uses the weighted center
                        // relative to... Actually the R-tensor argument is:
                        // R_{N}(alpha_PQ, A-B) where alpha_PQ = p*q/(p+q)
                        //
                        // The Hermite Coulomb integral R_{tuv}:
                        // R_{tuv}^n(p, PC) using Boys function F_n(p*PC²)
                        //
                        // For our case: p = p+q (total), and the separation
                        // is PQ = A - B (between the two Hermite centers)
                        // But we're using unnormalized Hermite Gaussians,
                        // so the R-tensor takes alpha = p+q and PC = A-B...
                        // No. Let me follow the standard MD formulation.

                        // Standard McMurchie-Davidson for 2-center Coulomb:
                        //
                        // [a|b] = Σ_{prim} c_a c_b * Norm *
                        //   (2π^{5/2}) / (p * q * √(p+q)) *
                        //   Σ_{tuv} E^a_t E^a_u E^a_v *
                        //   Σ_{τμν} E^b_τ E^b_μ E^b_ν *
                        //   R_{t+τ, u+μ, v+ν}^0(α, A-B)
                        //
                        // where α = p*q/(p+q), R uses Boys F_m(α*|AB|²)

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
                            ((double)eP * (double)eQ *
                             sqrt((double)(eP + eQ)));

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
                                        double eQx =
                                            (double)E_Qx[lxQ][0][tt];
                                        if (eQx == 0.0) continue;
                                        for (int uu = 0; uu <= lyQ; uu++)
                                        {
                                            double eQy =
                                                (double)E_Qy[lyQ][0][uu];
                                            if (eQy == 0.0) continue;
                                            for (int vv = 0; vv <= lzQ; vv++)
                                            {
                                                double eQz =
                                                    (double)
                                                        E_Qz[lzQ][0][vv];
                                                if (eQz == 0.0) continue;
                                                // (-1)^{tt+uu+vv} for ket
                                                // Hermite integrals
                                                double sign =
                                                    ((tt + uu + vv) & 1)
                                                        ? -1.0
                                                        : 1.0;
                                                v_sum +=
                                                    ePx * ePy * ePz * eQx *
                                                    eQy * eQz * sign *
                                                    (double)R_vals
                                                        [ONEE_MD_IDX(
                                                            t + tt, u + uu,
                                                            v + vv, 0)];
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
                if (P_sh != Q_sh)
                    out_metric[Q_idx * naux + P_idx] = total;
            }
        }
    }
}

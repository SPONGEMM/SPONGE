#pragma once

// Common utilities for register-only s/p ERI kernels.
// Boys function (max_m <= 4), compact R tensor, E-coefficient helpers.

// ---- Inline Boys function for small max_m (≤4) ----
// Computes F[0..max_m] in double precision.
static __device__ __forceinline__ void sp_boys(
    double* F, float T, int max_m)
{
    const double td = (double)T;
    if (td < 1e-7)
    {
        // Taylor: F_n(T) = 1/(2n+1) - T/(2n+3) + T^2/(2(2n+5)) - ...
        for (int m = 0; m <= max_m; m++)
        {
            const double a = 2.0 * m + 1.0;
            F[m] = 1.0 / a -
                   td * (1.0 / (a + 2.0) -
                         td * (0.5 / (a + 4.0) - td / (6.0 * (a + 6.0))));
        }
        return;
    }
    const double exp_t = exp(-td);
    const double st = sqrt(td);
    F[0] = 0.5 * 1.7724538509055159 * erf(st) / st;
    for (int m = 0; m < max_m; m++)
        F[m + 1] = ((2.0 * m + 1.0) * F[m] - exp_t) / (2.0 * td);
}

// ---- Compact R^0 tensor index ----
// Maps (t,u,v) with t+u+v <= L to a flat index in [0, (L+1)(L+2)(L+3)/6).
// Ordering: by level N=t+u+v, then decreasing t, then decreasing u.
static __device__ __forceinline__ int sp_R0_idx(int t, int u, int v)
{
    const int N = t + u + v;
    return N * (N + 1) * (N + 2) / 6 + (N - t) * (N - t + 1) / 2 + (N - t - u);
}

// Total number of (t,u,v) with t+u+v <= M
static __device__ __forceinline__ int sp_T_count(int M)
{
    return (M + 1) * (M + 2) * (M + 3) / 6;
}

// Combined index for R^n_{tuv} in the recurrence workspace.
// Elements grouped by n: offset_for_n + flat_within_n.
static __device__ __forceinline__ int sp_Rn_idx(int t, int u, int v, int n, int L)
{
    int offset = 0;
    for (int i = 0; i < n; i++) offset += sp_T_count(L - i);
    return offset + sp_R0_idx(t, u, v);
}

// ---- Build R^0 tensor in registers ----
// Input:  Boys values F[0..L], alpha, PQ[3]
// Output: R0[0..T_count(L)-1] indexed by sp_R0_idx(t,u,v)
// Workspace: Rw[0..total-1] where total = sum_{n=0}^{L} T_count(L-n)
//   For L=2: 15, L=3: 35, L=4: 70
static __device__ void sp_build_R0(
    float* R0, float* Rw, const double* F, float alpha, const float* PQ, int L)
{
    // Seed: R^n_{000} = (-2α)^n * F_n
    double m2a = -2.0 * (double)alpha;
    double fac = 1.0;
    for (int n = 0; n <= L; n++)
    {
        Rw[sp_Rn_idx(0, 0, 0, n, L)] = (float)(fac * F[n]);
        fac *= m2a;
    }

    // Recurrence: build level N from level N-1
    for (int N = 1; N <= L; N++)
    {
        for (int t = N; t >= 0; t--)
        {
            for (int u = N - t; u >= 0; u--)
            {
                const int v = N - t - u;
                for (int n = 0; n <= L - N; n++)
                {
                    float val = 0.0f;
                    if (t > 0)
                    {
                        val = PQ[0] * Rw[sp_Rn_idx(t - 1, u, v, n + 1, L)];
                        if (t > 1)
                            val += (float)(t - 1) *
                                   Rw[sp_Rn_idx(t - 2, u, v, n + 1, L)];
                    }
                    else if (u > 0)
                    {
                        val = PQ[1] * Rw[sp_Rn_idx(t, u - 1, v, n + 1, L)];
                        if (u > 1)
                            val += (float)(u - 1) *
                                   Rw[sp_Rn_idx(t, u - 2, v, n + 1, L)];
                    }
                    else // v > 0
                    {
                        val = PQ[2] * Rw[sp_Rn_idx(t, u, v - 1, n + 1, L)];
                        if (v > 1)
                            val += (float)(v - 1) *
                                   Rw[sp_Rn_idx(t, u, v - 2, n + 1, L)];
                    }
                    Rw[sp_Rn_idx(t, u, v, n, L)] = val;
                }
            }
        }
    }

    // Copy R^0 slice to output
    const int n0 = sp_T_count(L);
    for (int i = 0; i < n0; i++) R0[i] = Rw[i];
}

// ---- McMurchie-Davidson E-coefficient for one axis ----
// Pair (la, lb) with shift_a = PA_d or QC_d, shift_b = PB_d or QD_d, inv2x = 1/(2p) or 1/(2q).
// Writes e[0..la+lb] and returns la+lb+1 (number of terms).
static __device__ __forceinline__ int sp_E_coeff(
    float* e, int la, int lb, float shift_a, float shift_b, float inv2x)
{
    if (la == 0 && lb == 0)
    {
        e[0] = 1.0f;
        return 1;
    }
    if (la == 1 && lb == 0)
    {
        e[0] = shift_a;
        e[1] = inv2x;
        return 2;
    }
    if (la == 0 && lb == 1)
    {
        e[0] = shift_b;
        e[1] = inv2x;
        return 2;
    }
    // la == 1, lb == 1
    e[0] = shift_a * shift_b + inv2x;
    e[1] = (shift_a + shift_b) * inv2x;
    e[2] = inv2x * inv2x;
    return 3;
}

// ---- Contract E-coefficients with R^0 tensor for one Cartesian component ----
// l[4]: angular momenta, c[4]: Cartesian component indices (0=x,1=y,2=z for p; 0 for s)
// PA[3], PB[3], QC[3], QD[3]: shift vectors
// inv2p, inv2q: 1/(2p), 1/(2q)
// R0: precomputed R^0 tensor
static __device__ __forceinline__ float sp_contract_eri(
    const int* l, const int* c,
    const float* PA, const float* PB, const float* QC, const float* QD,
    float inv2p, float inv2q, const float* R0)
{
    // Build E-coefficients per axis
    float E_bra[3][3]; // E_bra[axis][hermite_idx], max 3 terms
    float E_ket[3][3];
    int n_bra[3], n_ket[3]; // number of terms per axis

    for (int d = 0; d < 3; d++)
    {
        const int la_d = (l[0] == 1 && c[0] == d) ? 1 : 0;
        const int lb_d = (l[1] == 1 && c[1] == d) ? 1 : 0;
        const int lc_d = (l[2] == 1 && c[2] == d) ? 1 : 0;
        const int ld_d = (l[3] == 1 && c[3] == d) ? 1 : 0;
        n_bra[d] = sp_E_coeff(E_bra[d], la_d, lb_d, PA[d], PB[d], inv2p);
        n_ket[d] = sp_E_coeff(E_ket[d], lc_d, ld_d, QC[d], QD[d], inv2q);
    }

    // 6-level contraction
    float eri = 0.0f;
    for (int mx = 0; mx < n_bra[0]; mx++)
        for (int my = 0; my < n_bra[1]; my++)
            for (int mz = 0; mz < n_bra[2]; mz++)
            {
                const float e_bra = E_bra[0][mx] * E_bra[1][my] * E_bra[2][mz];
                if (e_bra == 0.0f) continue;
                for (int nx = 0; nx < n_ket[0]; nx++)
                    for (int ny = 0; ny < n_ket[1]; ny++)
                        for (int nz = 0; nz < n_ket[2]; nz++)
                        {
                            const float e_ket =
                                E_ket[0][nx] * E_ket[1][ny] * E_ket[2][nz];
                            if (e_ket == 0.0f) continue;
                            const float sign =
                                ((nx + ny + nz) % 2 == 0) ? 1.0f : -1.0f;
                            eri += e_bra * e_ket * sign *
                                   R0[sp_R0_idx(mx + nx, my + ny, mz + nz)];
                        }
            }
    return eri;
}

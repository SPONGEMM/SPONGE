#pragma once

// Common utilities for register-only ERI kernels (s/p/d shells).
// Boys function, compact R tensor, E-coefficient helpers, contraction.

// ---- Get angular momentum on axis d for shell with l and Cartesian component c ----
static __device__ __forceinline__ int eri_get_l_axis(int l, int c, int axis)
{
    if (l == 0) return 0;
    if (l == 1) return (c == axis) ? 1 : 0;
    // l >= 2: use device constant lookup tables
    const int offset = QC_Comp_Offset(l);
    if (axis == 0) return QC_COMP_LX_DEVICE[offset + c];
    if (axis == 1) return QC_COMP_LY_DEVICE[offset + c];
    return QC_COMP_LZ_DEVICE[offset + c];
}

// ---- Boys function for max_m up to ~8 ----
// Uses upward recursion for max_m <= 4 (fast, register-only).
// Uses downward recursion for max_m > 4 (stable, needs small work array).
static __device__ __forceinline__ void eri_boys(
    double* F, float T, int max_m)
{
    const double td = (double)T;
    if (td < 1e-15)
    {
        for (int m = 0; m <= max_m; m++) F[m] = 1.0 / (2.0 * m + 1.0);
        return;
    }
    const double exp_t = exp(-td);
    const double st = sqrt(td);
    const double f0 = 0.5 * 1.7724538509055159 * erf(st) / st;

    if (td > 30.0)
    {
        // Upward recursion — stable for large T (exp(-T) is tiny, no cancellation)
        F[0] = f0;
        for (int m = 0; m < max_m; m++)
            F[m + 1] = ((2.0 * m + 1.0) * F[m] - exp_t) / (2.0 * td);
        return;
    }

    // Downward recursion (Miller's algorithm) — stable for T <= 30
    // For large T, exp(-T) underflows and work[] becomes all-zero → scale = inf → NaN.
    const int m_top = max_m + 25;
    double work[48]; // max_m <= 16 → m_top <= 41
    work[m_top] = 0.0;
    for (int m = m_top - 1; m >= 0; m--)
        work[m] = (2.0 * td * work[m + 1] + exp_t) / (2.0 * m + 1.0);
    const double scale = f0 / work[0];
    for (int m = 0; m <= max_m; m++) F[m] = work[m] * scale;
}

// ---- Compact R^0 tensor index ----
static __device__ __forceinline__ int eri_R0_idx(int t, int u, int v)
{
    const int N = t + u + v;
    return N * (N + 1) * (N + 2) / 6 + (N - t) * (N - t + 1) / 2 + (N - t - u);
}

static __device__ __forceinline__ int eri_T_count(int M)
{
    return (M + 1) * (M + 2) * (M + 3) / 6;
}

static __device__ __forceinline__ int eri_Rn_idx(int t, int u, int v, int n, int L)
{
    int offset = 0;
    for (int i = 0; i < n; i++) offset += eri_T_count(L - i);
    return offset + eri_R0_idx(t, u, v);
}

// ---- Build R^0 tensor in registers ----
static __device__ void eri_build_R0(
    float* R0, float* Rw, const double* F, float alpha, const float* PQ, int L)
{
    double m2a = -2.0 * (double)alpha;
    double fac = 1.0;
    for (int n = 0; n <= L; n++)
    {
        Rw[eri_Rn_idx(0, 0, 0, n, L)] = (float)(fac * F[n]);
        fac *= m2a;
    }
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
                        val = PQ[0] * Rw[eri_Rn_idx(t - 1, u, v, n + 1, L)];
                        if (t > 1)
                            val += (float)(t - 1) *
                                   Rw[eri_Rn_idx(t - 2, u, v, n + 1, L)];
                    }
                    else if (u > 0)
                    {
                        val = PQ[1] * Rw[eri_Rn_idx(t, u - 1, v, n + 1, L)];
                        if (u > 1)
                            val += (float)(u - 1) *
                                   Rw[eri_Rn_idx(t, u - 2, v, n + 1, L)];
                    }
                    else
                    {
                        val = PQ[2] * Rw[eri_Rn_idx(t, u, v - 1, n + 1, L)];
                        if (v > 1)
                            val += (float)(v - 1) *
                                   Rw[eri_Rn_idx(t, u, v - 2, n + 1, L)];
                    }
                    Rw[eri_Rn_idx(t, u, v, n, L)] = val;
                }
            }
        }
    }
    const int n0 = eri_T_count(L);
    for (int i = 0; i < n0; i++) R0[i] = Rw[i];
}

// ---- McMurchie-Davidson E-coefficient for one axis ----
// Supports la, lb up to 2 (d shells). Returns number of terms.
static __device__ __forceinline__ int eri_E_coeff(
    float* e, int la, int lb, float shift_a, float shift_b, float inv2x)
{
    // Fast paths for common cases
    if (la == 0 && lb == 0) { e[0] = 1.0f; return 1; }
    if (la == 1 && lb == 0) { e[0] = shift_a; e[1] = inv2x; return 2; }
    if (la == 0 && lb == 1) { e[0] = shift_b; e[1] = inv2x; return 2; }
    if (la == 1 && lb == 1)
    {
        e[0] = shift_a * shift_b + inv2x;
        e[1] = (shift_a + shift_b) * inv2x;
        e[2] = inv2x * inv2x;
        return 3;
    }
    if (la == 2 && lb == 0)
    {
        e[0] = shift_a * shift_a + inv2x;
        e[1] = 2.0f * shift_a * inv2x;
        e[2] = inv2x * inv2x;
        return 3;
    }
    if (la == 0 && lb == 2)
    {
        e[0] = shift_b * shift_b + inv2x;
        e[1] = 2.0f * shift_b * inv2x;
        e[2] = inv2x * inv2x;
        return 3;
    }
    // General recurrence for la+lb > 2 (e.g., (2,1), (1,2), (2,2))
    // Build E^{la,0} first, then step up lb times
    float prev[5];
    int n_prev;
    if (la == 0) { prev[0] = 1.0f; n_prev = 1; }
    else if (la == 1) { prev[0] = shift_a; prev[1] = inv2x; n_prev = 2; }
    else { prev[0] = shift_a*shift_a+inv2x; prev[1] = 2.0f*shift_a*inv2x; prev[2] = inv2x*inv2x; n_prev = 3; }
    for (int step = 0; step < lb; step++)
    {
        float next[5];
        int n_next = n_prev + 1;
        for (int t = 0; t < n_next; t++)
        {
            float val = 0.0f;
            if (t > 0) val += inv2x * prev[t - 1];
            if (t < n_prev) val += shift_b * prev[t];
            if (t + 1 < n_prev) val += (float)(t + 1) * prev[t + 1];
            next[t] = val;
        }
        for (int t = 0; t < n_next; t++) prev[t] = next[t];
        n_prev = n_next;
    }
    for (int t = 0; t < n_prev; t++) e[t] = prev[t];
    return n_prev;
}

// ---- Contract E-coefficients with R^0 tensor ----
// General version: supports any l values via eri_get_l_axis lookup.
// For l<=1, eri_get_l_axis inlines to (c==d)?1:0, same perf as before.
static __device__ __forceinline__ float eri_contract(
    const int* l, const int* c,
    const float* PA, const float* PB, const float* QC, const float* QD,
    float inv2p, float inv2q, const float* R0)
{
    float E_bra[3][5]; // max 5 terms per axis (la+lb up to 4)
    float E_ket[3][5];
    int n_bra[3], n_ket[3];

    for (int d = 0; d < 3; d++)
    {
        const int la_d = eri_get_l_axis(l[0], c[0], d);
        const int lb_d = eri_get_l_axis(l[1], c[1], d);
        const int lc_d = eri_get_l_axis(l[2], c[2], d);
        const int ld_d = eri_get_l_axis(l[3], c[3], d);
        n_bra[d] = eri_E_coeff(E_bra[d], la_d, lb_d, PA[d], PB[d], inv2p);
        n_ket[d] = eri_E_coeff(E_ket[d], lc_d, ld_d, QC[d], QD[d], inv2q);
    }

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
                                   R0[eri_R0_idx(mx + nx, my + ny, mz + nz)];
                        }
            }
    return eri;
}

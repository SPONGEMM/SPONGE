#pragma once

// Rys quadrature engine: roots/weights via Chebyshev interpolation (from
// gpu4pyscf), VRR for 2D integrals, HRR for angular momentum distribution.
//
// Coefficient data in rys_roots_dat.cu, declarations in rys_roots.cuh.

#include "rys_data.hpp"

#define RYS_SQRTPIE4 0.8862269254527580136
#define RYS_PIE4 0.7853981633974483096

// ---- Rys roots and weights via Chebyshev interpolation (double precision)
// ---- Output: roots[nrys] = t_i^2, weights[nrys] Satisfies: sum_i w_i *
// root_i^k = F_k(T)
static __device__ void rys_roots_weights(int nrys, double T, double* roots,
                                         double* weights)
{
    if (T < 3.e-7)
    {
        int off = nrys * (nrys - 1) / 2;
        for (int i = 0; i < nrys; i++)
        {
            roots[i] = RYS_SMALLX_R0[off + i] + RYS_SMALLX_R1[off + i] * T;
            weights[i] = RYS_SMALLX_W0[off + i] + RYS_SMALLX_W1[off + i] * T;
        }
        return;
    }

    if (T > 35 + nrys * 5)
    {
        int off = nrys * (nrys - 1) / 2;
        double t = sqrt(RYS_PIE4 / T);
        for (int i = 0; i < nrys; i++)
        {
            roots[i] = RYS_LARGEX_R[off + i] / T;
            weights[i] = RYS_LARGEX_W[off + i] * t;
        }
        return;
    }

    if (nrys == 1)
    {
        double tt = sqrt(T);
        double fmt0 = RYS_SQRTPIE4 / tt * erf(tt);
        weights[0] = fmt0;
        double e = exp(-T);
        double b = 0.5 / T;
        double fmt1 = b * (fmt0 - e);
        roots[0] = fmt1 / fmt0;
        return;
    }

    // Chebyshev interpolation (Clenshaw evaluation)
    const double* datax =
        RYS_RW_DATA + RYS_DEGREE1 * RYS_INTERVALS * nrys * (nrys - 1);
    int it = (int)(T * 0.4);
    double u = (T - it * 2.5) * 0.8 - 1.0;
    double u2 = u * 2.0;

    for (int i = 0; i < nrys; i++)
    {
        // Root
        {
            const double* c = datax + (2 * i) * RYS_DEGREE1 * RYS_INTERVALS;
            double c0 = c[it + RYS_DEGREE * RYS_INTERVALS];
            double c1 = c[it + (RYS_DEGREE - 1) * RYS_INTERVALS];
            double c2, c3;
            for (int n = RYS_DEGREE - 2; n > 0; n -= 2)
            {
                c2 = c[it + n * RYS_INTERVALS] - c1;
                c3 = c0 + c1 * u2;
                c1 = c2 + c3 * u2;
                c0 = c[it + (n - 1) * RYS_INTERVALS] - c3;
            }
            // RYS_DEGREE=13 is odd, so we end with:
            roots[i] = c0 + c1 * u;
        }
        // Weight
        {
            const double* c = datax + (2 * i + 1) * RYS_DEGREE1 * RYS_INTERVALS;
            double c0 = c[it + RYS_DEGREE * RYS_INTERVALS];
            double c1 = c[it + (RYS_DEGREE - 1) * RYS_INTERVALS];
            double c2, c3;
            for (int n = RYS_DEGREE - 2; n > 0; n -= 2)
            {
                c2 = c[it + n * RYS_INTERVALS] - c1;
                c3 = c0 + c1 * u2;
                c1 = c2 + c3 * u2;
                c0 = c[it + (n - 1) * RYS_INTERVALS] - c3;
            }
            weights[i] = c0 + c1 * u;
        }
    }
}

// VRR: build G[i][j] for one axis at one Rys root
// G[0][0] = 1
// G[i+1][j] = Cx_bra * G[i][j] + i * B10 * G[i-1][j] + j * B00 * G[i][j-1]
// G[i][j+1] = Cx_ket * G[i][j] + j * B01 * G[i][j-1] + i * B00 * G[i-1][j]
static __device__ void rys_vrr_2d(float* G, int ij_max, int kl_max,
                                  int g_stride, float Cx_bra, float Cx_ket,
                                  float B00, float B10, float B01)
{
    G[0] = 1.0f;

    // Vertical on bra: G[i+1][0]
    for (int i = 0; i < ij_max; i++)
        G[(i + 1) * g_stride] =
            Cx_bra * G[i * g_stride] +
            (float)i * B10 * (i > 0 ? G[(i - 1) * g_stride] : 0.0f);

    // Transfer to ket: G[i][j+1]
    for (int j = 0; j < kl_max; j++)
        for (int i = 0; i <= ij_max; i++)
        {
            float val = Cx_ket * G[i * g_stride + j];
            if (j > 0) val += (float)j * B01 * G[i * g_stride + (j - 1)];
            if (i > 0) val += (float)i * B00 * G[(i - 1) * g_stride + j];
            G[i * g_stride + (j + 1)] = val;
        }
}

// HRR: distribute total AM (a+b) to individual shells (a, b)
// I(a, b) from G[0..a+b] using: I(a, b) = I(a+1, b-1) + AB * I(a, b-1)
static __device__ float rys_hrr_1d(const float* g_col, int la, int lb, float AB)
{
    if (lb == 0) return g_col[la];

    float tmp[9];  // max la+lb = 4 for d shells
    const int n = la + lb;
    for (int i = 0; i <= n; i++) tmp[i] = g_col[i];

    for (int b = 0; b < lb; b++)
    {
        const int n_curr = n - b - 1;
        for (int i = 0; i <= n_curr; i++) tmp[i] = tmp[i + 1] + AB * tmp[i];
    }
    return tmp[la];
}

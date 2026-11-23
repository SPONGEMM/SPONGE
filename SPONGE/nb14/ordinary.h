#pragma once
#include <cmath>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define SPONGE_ORDINARY_HD __host__ __device__
#else
#define SPONGE_ORDINARY_HD
#endif

// A union-excluded pair has its reciprocal pair term removed by PME.
// Restore weighted full Coulomb inside the cutoff, and only the removed
// reciprocal term outside. LJ A/B are already weighted derivative coefficients.
template <typename Real>
SPONGE_ORDINARY_HD inline void NB14_Ordinary_Restoration(
    Real r2, Real a, Real b, Real scaled_charge, Real cutoff, Real beta,
    Real* lj_energy, Real* cf_energy, Real* force_over_r)
{
    *lj_energy = *cf_energy = *force_over_r = Real(0);
    const bool inside = r2 < cutoff * cutoff;
    if (inside && (a != Real(0) || b != Real(0)))
    {
        const Real inverse_r2 = Real(1) / r2;
        const Real inverse_r6 = inverse_r2 * inverse_r2 * inverse_r2;
        *lj_energy = (a * inverse_r6 / Real(12) - b / Real(6)) * inverse_r6;
        *force_over_r = (b - a * inverse_r6) * inverse_r6 * inverse_r2;
    }
    if (scaled_charge != Real(0))
    {
        const Real r = sqrt(r2);
        const Real inverse_r = Real(1) / r;
        if (inside)
        {
            *cf_energy = scaled_charge * inverse_r;
            *force_over_r -= scaled_charge * inverse_r / r2;
        }
        else
        {
            const Real x = beta * r;
            const Real screen = erf(x);
            *cf_energy = scaled_charge * screen * inverse_r;
            *force_over_r += scaled_charge *
                             (Real(1.1283791670955125739) * beta * exp(-x * x) -
                              screen * inverse_r) /
                             r2;
        }
    }
}

#undef SPONGE_ORDINARY_HD

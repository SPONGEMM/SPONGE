#pragma once

// A=12*C12 and B=6*C6 already include the window's LJ weight.
// delta has units of length^6. No Coulomb softening is performed here.
#if defined(__CUDACC__) || defined(__HIPCC__)
#define SPONGE_NB14_HD __host__ __device__
#else
#define SPONGE_NB14_HD
#endif
template <typename Real>
SPONGE_NB14_HD inline void NB14_Softcore_LJ(Real r2, Real A, Real B, Real delta,
                                            Real* energy, Real* force_over_r)
{
    *energy = Real(0);
    *force_over_r = Real(0);
    if (A == Real(0) && B == Real(0)) return;
    const Real r4 = r2 * r2;
    const Real inv_s = Real(1) / (r4 * r2 + delta);
    *energy = (A / Real(12) * inv_s - B / Real(6)) * inv_s;
    // dU/dr / r, multiplying displacement r_j-r_i gives force on i.
    *force_over_r = r4 * inv_s * inv_s * (B - A * inv_s);
}

#undef SPONGE_NB14_HD

#pragma once

#include <cmath>

// Frozen-window ordinary-pair mathematics. The caller owns validated M/A/B
// membership, live charges and exclusions; this file owns no buffers or state.
// A/B coefficients use SPONGE's derivative convention (12*C12, 6*C6).
#if defined(__CUDACC__) || defined(__HIPCC__)
#define SPONGE_STERICS_HD __host__ __device__
#else
#define SPONGE_STERICS_HD
#endif

template <typename Real>
struct IndependentStericsResult
{
    Real lj_energy = 0;
    Real direct_energy = 0;
    Real force_over_r = 0;
};

// Membership codes are 0=M/environment, 1=A-only, 2=B-only. Type indices
// and zero LJ coefficients are not membership information.
template <typename Real>
SPONGE_STERICS_HD inline Real Independent_Sterics_C6(int branch_i, int branch_j,
                                                     Real c6_a, Real c6_b,
                                                     Real core_mix,
                                                     Real weight_a,
                                                     Real weight_b)
{
    int branch = branch_i | branch_j;
    if (branch == 3) return Real(0);
    if (branch == 1) return weight_a * c6_a;
    if (branch == 2) return weight_b * c6_b;
    return (Real(1) - core_mix) * c6_a + core_mix * c6_b;
}

template <typename Real>
SPONGE_STERICS_HD inline void Independent_Sterics_LJ_Term(
    Real r2, Real a, Real b, Real weight, Real delta,
    IndependentStericsResult<Real>* result)
{
    if (weight == Real(0) || (a == Real(0) && b == Real(0))) return;
    Real r4 = r2 * r2;
    Real inverse = Real(1) / (r4 * r2 + delta);
    result->lj_energy +=
        weight * (a * inverse / Real(12) - b / Real(6)) * inverse;
    result->force_over_r += weight * r4 * inverse * inverse * (b - a * inverse);
}

template <typename Real>
SPONGE_STERICS_HD inline IndependentStericsResult<Real>
Evaluate_Independent_Sterics(Real r2, int branch_i, int branch_j, Real aa,
                             Real ab, Real ba, Real bb, Real core_mix,
                             Real weight_a, Real weight_b, Real alpha,
                             Real power, Real fallback_sigma6,
                             Real minimum_sigma6, Real charge_product,
                             Real pme_beta)
{
    IndependentStericsResult<Real> result;
    int branch = branch_i | branch_j;
    if (branch == 3) return result;
    if (branch == 0)
    {
        Independent_Sterics_LJ_Term(r2, aa, ab, Real(1) - core_mix, Real(0),
                                    &result);
        Independent_Sterics_LJ_Term(r2, ba, bb, core_mix, Real(0), &result);
    }
    else
    {
        Real weight = branch == 1 ? weight_a : weight_b;
        Real a = branch == 1 ? aa : ba;
        Real b = branch == 1 ? ab : bb;
        Real sigma6 = (a == Real(0) || b == Real(0))
                          ? fallback_sigma6
                          : fmax(Real(0.5) * a / b, minimum_sigma6);
        Real delta = alpha * pow(Real(1) - weight, power) * sigma6;
        Independent_Sterics_LJ_Term(r2, a, b, weight, delta, &result);
    }
    // Exactly one direct electrostatic interaction from the already frozen
    // current charges. Independent steric weights must not multiply it again.
    if (charge_product != Real(0))
    {
        Real r = sqrt(r2);
        Real beta_r = pme_beta * r;
        Real screened = erfc(beta_r);
        result.direct_energy = charge_product * screened / r;
        result.force_over_r -= charge_product *
                               (screened + Real(1.1283791670955125739) *
                                               beta_r * exp(-beta_r * beta_r)) /
                               (r2 * r);
    }
    return result;
}

#undef SPONGE_STERICS_HD

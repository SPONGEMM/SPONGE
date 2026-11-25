#pragma once

#include <cfloat>

#include "precision.h"

// A relative residual alone is unattainable when two stored Real velocities
// are nearly equal.  Bound the subtraction and three-component dot-product
// roundoff by the scale of the operands that are actually represented.
template <class Real>
static __device__ __host__ __forceinline__ Real
Velocity_Constraint_Residual_Tolerance(
    const Real displacement_squared, const ConstraintVector<Real> velocity_i,
    const ConstraintVector<Real> velocity_j,
    const ConstraintVector<Real> velocity_difference,
    const Real relative_tolerance)
{
    const Real displacement_norm = ConstraintSqrt(displacement_squared);
    const Real relative_scale =
        displacement_norm *
        ConstraintSqrt(
            fmax(velocity_difference * velocity_difference, Real(1.0e-12)));
    const Real velocity_scale = fmax(ConstraintSqrt(velocity_i * velocity_i),
                                     ConstraintSqrt(velocity_j * velocity_j));
    const Real roundoff_floor = Real(8.0) * ConstraintEpsilon<Real>() *
                                displacement_norm * velocity_scale;
    return fmax(relative_tolerance * relative_scale, roundoff_floor);
}

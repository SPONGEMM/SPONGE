#pragma once

#include <cfloat>

#include "../common.h"

// A relative residual alone is unattainable when two stored float velocities
// are nearly equal.  Bound the subtraction and three-component dot-product
// roundoff by the scale of the operands that are actually represented.
static __device__ __host__ __forceinline__ float
Velocity_Constraint_Residual_Tolerance(const float displacement_squared,
                                       const VECTOR velocity_i,
                                       const VECTOR velocity_j,
                                       const VECTOR velocity_difference,
                                       const float relative_tolerance)
{
    const float displacement_norm = sqrtf(displacement_squared);
    const float relative_scale =
        displacement_norm *
        sqrtf(fmaxf(velocity_difference * velocity_difference, 1.0e-12f));
    const float velocity_scale =
        fmaxf(sqrtf(velocity_i * velocity_i), sqrtf(velocity_j * velocity_j));
    const float roundoff_floor =
        8.0f * FLT_EPSILON * displacement_norm * velocity_scale;
    return fmaxf(relative_tolerance * relative_scale, roundoff_floor);
}

#pragma once

#include <vector>

#include "../MD_core/dynamics_types.h"

namespace SpongeDynamics
{
// Coordinates must be a continuous, unwrapped group. All arguments use one
// consistent unit system. Returns mass-weighted minimum-norm atomic forces
// with the requested total force and torque about the center of mass.
// A torque outside the range of a singular inertia tensor is an input error.
std::vector<Vec3> Distribute_Group_Load(const std::vector<Vec3>& positions,
                                        const std::vector<double>& masses,
                                        const Vec3& force, const Vec3& torque);
}  // namespace SpongeDynamics

#pragma once
#include <array>
#include <vector>

#include "../MD_core/dynamics_types.h"
namespace SpongeDynamics
{
using Stress6 =
    std::array<double, 6>;  // xx, yy, zz, xy, xz, yz (physical components)
struct EquilibriumStress
{
    Stress6 pressure{};  // kcal/mol/A^3
    Vec3 com_velocity{};
    double kinetic_energy = 0;
};
// Center the incoming leapfrog velocities on the current force time, remove
// only COM translation, then add the already physical potential virial.
EquilibriumStress Centered_Stress(const std::vector<double>& mass,
                                  const std::vector<Vec3>& half_velocity,
                                  const std::vector<Vec3>& force,
                                  const Stress6& potential_virial,
                                  double dt_internal, double volume);
}  // namespace SpongeDynamics

#include "stress_math.h"

#include <stdexcept>

#include "../utils/float_classification.hpp"
namespace SpongeDynamics
{
EquilibriumStress Centered_Stress(const std::vector<double>& mass,
                                  const std::vector<Vec3>& half_velocity,
                                  const std::vector<Vec3>& force,
                                  const Stress6& potential_virial,
                                  double dt_internal, double volume)
{
    if (mass.size() < 2 || half_velocity.size() != mass.size() ||
        force.size() != mass.size() || !SpongeFloat::Is_Finite(dt_internal) ||
        dt_internal < 0 || !SpongeFloat::Is_Finite(volume) || volume <= 0)
        throw std::runtime_error(
            "invalid equilibrium stress dimensions/time/volume");
    EquilibriumStress out;
    double total_mass = 0;
    std::vector<Vec3> velocity(mass.size());
    for (std::size_t i = 0; i < mass.size(); ++i)
    {
        if (!SpongeFloat::Is_Finite(mass[i]) || mass[i] <= 0)
            throw std::runtime_error("GK requires positive finite masses");
        total_mass += mass[i];
        for (int j = 0; j < 3; ++j)
        {
            if (!SpongeFloat::Is_Finite(half_velocity[i][j]) ||
                !SpongeFloat::Is_Finite(force[i][j]))
                throw std::runtime_error(
                    "nonfinite velocity/force in equilibrium stress");
            velocity[i][j] =
                half_velocity[i][j] + 0.5 * dt_internal * force[i][j] / mass[i];
            out.com_velocity[j] += mass[i] * velocity[i][j];
        }
    }
    for (double& v : out.com_velocity) v /= total_mass;
    out.pressure = potential_virial;
    for (std::size_t i = 0; i < mass.size(); ++i)
    {
        const double x = velocity[i][0] - out.com_velocity[0],
                     y = velocity[i][1] - out.com_velocity[1],
                     z = velocity[i][2] - out.com_velocity[2];
        out.pressure[0] += mass[i] * x * x;
        out.pressure[1] += mass[i] * y * y;
        out.pressure[2] += mass[i] * z * z;
        out.pressure[3] += mass[i] * x * y;
        out.pressure[4] += mass[i] * x * z;
        out.pressure[5] += mass[i] * y * z;
        out.kinetic_energy += 0.5 * mass[i] * (x * x + y * y + z * z);
    }
    for (double& p : out.pressure)
    {
        p /= volume;
        if (!SpongeFloat::Is_Finite(p))
            throw std::runtime_error("nonfinite equilibrium stress");
    }
    return out;
}
}  // namespace SpongeDynamics

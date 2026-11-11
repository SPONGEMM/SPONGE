#pragma once
#include <array>
#include <vector>

#include "../../MD_core/dynamics_types.h"
namespace SpongeDynamics
{
struct CosineFit
{
    Vec3 mean_velocity{0, 0, 0};
    double mean_cosine = 0, variance_cosine = 0, amplitude = 0;
    double fourier = 0, centered_fourier = 0, thermal_energy = 0,
           total_energy = 0;
    double mass = 0;
};
// Joint mass-weighted projection on three translations and one cosine mode.
// A rank-deficient mode is an error rather than an arbitrary temperature DOF.
CosineFit Fit_Cosine(const std::vector<double>& mass,
                     const std::vector<Vec3>& positions,
                     const std::vector<Vec3>& velocities, int flow,
                     int gradient, double wave_number);
}  // namespace SpongeDynamics

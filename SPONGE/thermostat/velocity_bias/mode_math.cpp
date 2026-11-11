#include "mode_math.h"

#include <cmath>
#include <stdexcept>

#include "../../utils/float_classification.hpp"
namespace SpongeDynamics
{
CosineFit Fit_Cosine(const std::vector<double>& mass,
                     const std::vector<Vec3>& positions,
                     const std::vector<Vec3>& velocities, int flow,
                     int gradient, double wave_number)
{
    if (mass.size() < 2 || positions.size() != mass.size() ||
        velocities.size() != mass.size() || flow < 0 || flow > 2 ||
        gradient < 0 || gradient > 2 || flow == gradient ||
        !SpongeFloat::Is_Finite(wave_number) || wave_number <= 0)
        throw std::runtime_error(
            "invalid cosine projection dimensions or wave number");
    CosineFit fit;
    std::vector<double> basis(mass.size());
    for (std::size_t i = 0; i < mass.size(); ++i)
    {
        if (!SpongeFloat::Is_Finite(mass[i]) || mass[i] <= 0)
            throw std::runtime_error(
                "cosine flow requires positive finite masses");
        for (int j = 0; j < 3; ++j)
            if (!SpongeFloat::Is_Finite(positions[i][j]) ||
                !SpongeFloat::Is_Finite(velocities[i][j]))
                throw std::runtime_error(
                    "nonfinite cosine flow coordinates or velocities");
        basis[i] = std::cos(wave_number * positions[i][gradient]);
        fit.mass += mass[i];
        fit.mean_cosine += mass[i] * basis[i];
        fit.fourier += 2 * mass[i] * velocities[i][flow] * basis[i];
        for (int j = 0; j < 3; ++j)
        {
            fit.mean_velocity[j] += mass[i] * velocities[i][j];
            fit.total_energy +=
                0.5 * mass[i] * velocities[i][j] * velocities[i][j];
        }
    }
    fit.mean_cosine /= fit.mass;
    fit.fourier /= fit.mass;
    for (double& value : fit.mean_velocity) value /= fit.mass;
    double covariance = 0;
    for (std::size_t i = 0; i < mass.size(); ++i)
    {
        const double c = basis[i] - fit.mean_cosine;
        fit.variance_cosine += mass[i] * c * c;
        covariance +=
            mass[i] * (velocities[i][flow] - fit.mean_velocity[flow]) * c;
    }
    fit.variance_cosine /= fit.mass;
    covariance /= fit.mass;
    if (!SpongeFloat::Is_Finite(fit.variance_cosine) ||
        fit.variance_cosine <= 1e-12)
        throw std::runtime_error(
            "cosine flow mode is rank deficient; sample distinct phases");
    fit.amplitude = covariance / fit.variance_cosine;
    fit.centered_fourier = 2 * covariance;
    for (std::size_t i = 0; i < mass.size(); ++i)
        for (int j = 0; j < 3; ++j)
        {
            const double u =
                fit.mean_velocity[j] +
                (j == flow ? fit.amplitude * (basis[i] - fit.mean_cosine) : 0);
            const double residual = velocities[i][j] - u;
            fit.thermal_energy += 0.5 * mass[i] * residual * residual;
        }
    if (!SpongeFloat::Is_Finite(fit.thermal_energy) ||
        !SpongeFloat::Is_Finite(fit.amplitude))
        throw std::runtime_error("cosine projection overflow");
    return fit;
}
}  // namespace SpongeDynamics

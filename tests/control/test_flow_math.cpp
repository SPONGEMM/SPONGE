#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>

#include "thermostat/velocity_bias/mode_math.h"
using namespace SpongeDynamics;
void Check(bool ok, const char* reason)
{
    if (!ok)
    {
        std::cerr << reason << '\n';
        std::exit(1);
    }
}
int main()
{
    const double k = 2 * 3.14159265358979323846 / 13;
    std::vector<double> mass{1, 3, 2, 5, 7, 11};
    std::vector<Vec3> r{{0, 0, 0}, {0, 0, 1}, {0, 0, 3},
                        {0, 0, 5}, {0, 0, 8}, {0, 0, 11}};
    std::vector<Vec3> v;
    for (const auto& x : r) v.push_back({2 + 4 * std::cos(k * x[2]), -3, 7});
    auto f = Fit_Cosine(mass, r, v, 0, 2, k);
    Check(std::abs(f.amplitude - 4) < 1e-12,
          "WLS does not recover pure mode at irregular phases/masses");
    Check(f.thermal_energy < 1e-25, "pure streaming motion counted as heat");
    Check(std::abs(f.centered_fourier - 2 * f.variance_cosine * f.amplitude) <
              1e-12,
          "Fourier normalization");
    for (auto& x : r) x[2] += 13;
    auto periodic = Fit_Cosine(mass, r, v, 0, 2, k);
    Check(std::abs(periodic.amplitude - f.amplitude) < 1e-12,
          "periodic phase invariance");
    // Draw independent Maxwell velocities and project. Mean thermal energy
    // must have 3N-4 DOF; fluctuations must remain, not be rescaled away.
    std::mt19937_64 rng(20261027);
    std::normal_distribution<double> normal;
    double sum = 0, square = 0;
    const int samples = 20000, dof = 3 * mass.size() - 4;
    for (int t = 0; t < samples; ++t)
    {
        for (std::size_t i = 0; i < v.size(); ++i)
            for (int j = 0; j < 3; ++j)
                v[i][j] = normal(rng) / std::sqrt(mass[i]);
        auto draw = Fit_Cosine(mass, r, v, 0, 2, k);
        sum += 2 * draw.thermal_energy;
        square += 4 * draw.thermal_energy * draw.thermal_energy;
        double residual[4] = {};
        for (std::size_t i = 0; i < v.size(); ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                v[i][j] -= draw.mean_velocity[j] +
                           (j == 0 ? draw.amplitude * (std::cos(k * r[i][2]) -
                                                       draw.mean_cosine)
                                   : 0);
                residual[j] += mass[i] * v[i][j];
            }
            residual[3] += mass[i] * v[i][0] * std::cos(k * r[i][2]);
        }
        for (double value : residual)
            Check(std::abs(value) < 1e-12,
                  "residual is not mass-orthogonal to collective modes");
    }
    const double mean = sum / samples,
                 variance = square / samples - mean * mean;
    Check(std::abs(mean - dof) < 0.1, "projected Maxwell mean DOF");
    Check(std::abs(variance - 2 * dof) < 0.8,
          "projected Maxwell chi-square fluctuations");
    for (auto& x : r) x[2] = 0;
    bool rejected = false;
    try
    {
        Fit_Cosine(mass, r, v, 0, 2, k);
    }
    catch (const std::exception&)
    {
        rejected = true;
    }
    Check(rejected, "rank deficient mode accepted");
}

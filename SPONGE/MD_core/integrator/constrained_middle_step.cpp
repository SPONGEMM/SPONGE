#include "constrained_middle_step.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "../../utils/float_classification.hpp"

namespace SpongeDynamics
{
ConstrainedFlowStep Advance_Constrained_Flow(
    const ConstraintStressPlan& plan, const std::vector<Vec3>& position,
    const std::vector<Vec3>& staggered_velocity, const std::vector<Vec3>& force,
    const std::vector<Vec3>& normal, const Vec3& box, double dt, int flow,
    int gradient, double wave_number, double damping, double kBT,
    double input_length_tolerance)
{
    if (!SpongeFloat::Is_Finite(dt) || dt <= 0 || flow < 0 || flow > 2 ||
        gradient < 0 || gradient > 2 || flow == gradient ||
        !SpongeFloat::Is_Finite(wave_number) || wave_number <= 0 ||
        force.size() != position.size() ||
        staggered_velocity.size() != position.size())
        throw std::runtime_error("invalid constrained flow step");
    const auto frame = plan.Prepare(position, box, input_length_tolerance);
    const auto& mass = frame.Mass();
    const auto tangent = frame.Project(staggered_velocity);
    double residual2 = 0, norm2 = 0;
    for (std::size_t i = 0; i < tangent.size(); ++i)
        for (int j = 0; j < 3; ++j)
        {
            const double residual = tangent[i][j] - staggered_velocity[i][j];
            residual2 += mass[i] * residual * residual;
            norm2 +=
                mass[i] * staggered_velocity[i][j] * staggered_velocity[i][j];
        }
    if (!SpongeFloat::Is_Finite(norm2) ||
        residual2 > 1e-20 * std::max(1., norm2))
        throw std::runtime_error(
            "constrained flow step requires tangent staggered velocity");
    auto kick = staggered_velocity;
    for (std::size_t i = 0; i < kick.size(); ++i)
        for (int j = 0; j < 3; ++j)
        {
            if (!SpongeFloat::Is_Finite(force[i][j]) ||
                !SpongeFloat::Is_Finite(kick[i][j]))
                throw std::runtime_error(
                    "nonfinite constrained flow force/state");
            kick[i][j] += dt * force[i][j] / mass[i];
        }
    ConstrainedFlowStep out;
    kick = frame.Project(kick, &out.mechanical_constraint_impulse_virial);
    const auto first =
        plan.Drift(position, kick, .5 * dt, box, input_length_tolerance);
    std::vector<double> cosine(position.size());
    for (std::size_t i = 0; i < cosine.size(); ++i)
        cosine[i] = std::cos(wave_number * first.position[i][gradient]);
    ConstrainedFlowFrame midpoint(plan.Prepare(first.position, box, 2e-12),
                                  cosine, flow);
    const auto ou = midpoint.Apply_OU(first.velocity, normal, damping, kBT);
    const auto second =
        plan.Drift(first.position, ou.velocity, .5 * dt, box, 2e-12);
    out.position = second.position;
    out.staggered_velocity = second.velocity;
    out.heat = ou.heat;
    out.drift_kinetic_change = first.kinetic_change + second.kinetic_change;
    for (int j = 0; j < 6; ++j)
        out.mechanical_constraint_impulse_virial[j] +=
            first.impulse_virial[j] + second.impulse_virial[j];
    return out;
}
}  // namespace SpongeDynamics

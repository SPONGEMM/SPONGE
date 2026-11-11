#include "rattle_step.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "../../utils/float_classification.hpp"

namespace SpongeDynamics
{
ConstrainedNVEStep Advance_Constrained_NVE(
    const ConstraintStressPlan& plan, const std::vector<Vec3>& position,
    const std::vector<Vec3>& staggered_velocity, const std::vector<Vec3>& force,
    const Vec3& box, double dt, double input_length_tolerance)
{
    if (!SpongeFloat::Is_Finite(dt) || dt <= 0 ||
        force.size() != position.size() ||
        staggered_velocity.size() != position.size())
        throw std::runtime_error("invalid constrained NVE step");
    const auto frame = plan.Prepare(position, box, input_length_tolerance);
    const auto tangent = frame.Project(staggered_velocity);
    const auto& mass = frame.Mass();
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
            "constrained NVE requires tangent staggered velocity");

    auto kick = staggered_velocity;
    for (std::size_t i = 0; i < kick.size(); ++i)
        for (int j = 0; j < 3; ++j)
        {
            if (!SpongeFloat::Is_Finite(force[i][j]))
                throw std::runtime_error("nonfinite constrained NVE force");
            kick[i][j] += dt * force[i][j] / mass[i];
        }
    ConstrainedNVEStep out;
    kick = frame.Project(kick, &out.mechanical_constraint_impulse_virial);
    auto drift = plan.Drift(position, kick, dt, box, input_length_tolerance);
    out.position = std::move(drift.position);
    out.staggered_velocity = std::move(drift.velocity);
    out.drift_kinetic_change = drift.kinetic_change;
    for (int j = 0; j < 6; ++j)
    {
        out.mechanical_constraint_impulse_virial[j] += drift.impulse_virial[j];
        if (!SpongeFloat::Is_Finite(
                out.mechanical_constraint_impulse_virial[j]))
            throw std::runtime_error("constrained NVE impulse overflow");
    }
    return out;
}
}  // namespace SpongeDynamics

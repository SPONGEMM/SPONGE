#pragma once
#include "../../constrain/tangent_frame.h"

namespace SpongeDynamics
{
struct ConstrainedNVEStep
{
    std::vector<Vec3> position, staggered_velocity;
    double drift_kinetic_change = 0;
    // Mechanical impulse over the staggered B-A interval, not force-time
    // stress. A physical B/2-A-B/2 interval additionally replaces the initial
    // half-B impulse with the final half-B impulse at the new force.
    Stress6 mechanical_constraint_impulse_virial{};
};

// One mechanical B-A step, with one full-dt RATTLE drift. The stored velocity
// is tangent and u_n = s_n + (dt/2) Pc(q_n) M^-1 F_n. Reconstructing the next
// physical velocity requires a NEW force at the returned position. No force
// evaluation, thermostat, flow mode, RNG, clock or persistent state is owned
// here. This core alone does not enable a new runtime integrator.
ConstrainedNVEStep Advance_Constrained_NVE(
    const ConstraintStressPlan& plan, const std::vector<Vec3>& position,
    const std::vector<Vec3>& staggered_velocity, const std::vector<Vec3>& force,
    const Vec3& box, double dt, double input_length_tolerance);
}  // namespace SpongeDynamics

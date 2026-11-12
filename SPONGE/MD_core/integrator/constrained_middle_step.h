#pragma once
#include "../../constrain/thermal_frame.h"

namespace SpongeDynamics
{
struct ConstrainedFlowStep
{
    std::vector<Vec3> position, staggered_velocity;
    double heat = 0, drift_kinetic_change = 0;
    // Sum over the staggered B-A/2-O-A/2 interval. For the physical BAOAB
    // interval, subtract the initial half-B constraint impulse virial and add
    // the final half-B one. Neither is a force-time stress estimator.
    Stress6 mechanical_constraint_impulse_virial{};
};
// One staggered B-A/2-O-A/2 step. Physical force-time velocity is
// Pc(q)[staggered_velocity + dt*force/(2m)]. Adjacent half B kicks are merged
// at the same q; the function does not evaluate a force or advance a clock.
// Uses one force-free RATTLE step per A half. OU impulse is a thermostat
// contribution and is deliberately separate from the mechanical virial.
ConstrainedFlowStep Advance_Constrained_Flow(
    const ConstraintStressPlan& plan, const std::vector<Vec3>& position,
    const std::vector<Vec3>& staggered_velocity, const std::vector<Vec3>& force,
    const std::vector<Vec3>& normal, const Vec3& box, double dt, int flow,
    int gradient, double wave_number, double damping, double kBT,
    double input_length_tolerance);
}  // namespace SpongeDynamics

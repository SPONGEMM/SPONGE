#pragma once
#include "../../constrain/tangent_frame.h"

namespace SpongeDynamics
{
struct ConstrainedFlowOU
{
    std::vector<Vec3> velocity;
    double heat = 0;  // K_after - K_before, same internal energy units as kBT
};
// Fixed-coordinate OU building block; not a constrained time integrator.
// Keeps global translation and the tangent-projected flow mode. It must be
// rebuilt after coordinates change. Physical Fourier measurements remain raw
// atom-velocity observables, not coefficients of this thermostat basis.
class ConstrainedFlowFrame
{
   public:
    ConstrainedFlowFrame(ConstraintTangentFrame constraints,
                         const std::vector<double>& cosine, int flow_axis);
    int Thermal_Degrees_Of_Freedom() const;
    std::vector<Vec3> Preserved(const std::vector<Vec3>& velocity) const;
    std::vector<Vec3> Thermal(const std::vector<Vec3>& velocity) const;
    // The caller owns RNG/state. normal contains independent N(0,1) Cartesian
    // draws; damping=exp(-gamma*dt). Rejects non-tangent input rather than
    // hiding a missing constraint substep. No coordinates or forces are
    // changed.
    ConstrainedFlowOU Apply_OU(const std::vector<Vec3>& velocity,
                               const std::vector<Vec3>& normal, double damping,
                               double kBT) const;

   private:
    ConstraintTangentFrame constraints_;
    std::vector<std::vector<Vec3>> basis_;
    void Validate(const std::vector<Vec3>& velocity) const;
    double Inner(const std::vector<Vec3>& a, const std::vector<Vec3>& b) const;
};

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

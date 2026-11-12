#pragma once
#include "tangent_frame.h"

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

}  // namespace SpongeDynamics

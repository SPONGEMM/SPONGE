#pragma once
#include "../observables/stress_math.h"

namespace SpongeDynamics
{
struct DistanceConstraint
{
    int a, b;
    double length;
};
struct ConstrainedStress
{
    std::vector<Vec3> velocity, constraint_force;
    Stress6 constraint_virial{};
    EquilibriumStress equilibrium;
    int thermal_dof = 0;
    double max_relative_length_error = 0;
    double max_tangent_residual = 0;       // velocity units
    double max_acceleration_residual = 0;  // acceleration units
};
// Fixed-geometry mass-orthogonal tangent projector. A frame is valid only at
// the coordinates used by Prepare; callers must rebuild after any drift.
class ConstraintTangentFrame
{
   public:
    std::vector<Vec3> Project(const std::vector<Vec3>& velocity,
                              Stress6* impulse_virial = nullptr) const;
    const std::vector<double>& Mass() const { return mass_; }
    int Degrees_Of_Freedom() const
    {
        return int(3 * mass_.size() - pairs_.size());
    }

   private:
    friend class ConstraintStressPlan;
    ConstraintTangentFrame() = default;
    std::vector<double> mass_, length_;
    std::vector<DistanceConstraint> pairs_;
    std::vector<Vec3> unit_;
    std::vector<std::vector<int>> components_;
    std::vector<std::vector<double>> factors_;
    double max_relative_length_error_ = 0;
};
struct ConstrainedDrift
{
    std::vector<Vec3> position, velocity;
    // Integral of constraint virial over this drift, NOT force-time stress.
    Stress6 impulse_virial{};
    double kinetic_change = 0;
    int iterations = 0;
};
// Force-time observer, NOT a discrete SHAKE impulse estimator.
// All arrays share stable atom indices and one consistent internal unit system.
// The host-only dense solve is per connected constraint component (<=256 rows).
class ConstraintStressPlan
{
   public:
    ConstraintStressPlan(std::vector<double> mass,
                         std::vector<DistanceConstraint> pairs);
    ConstraintTangentFrame Prepare(const std::vector<Vec3>& position,
                                   const Vec3& orthogonal_box,
                                   double length_tolerance) const;
    // One force-free RATTLE step. Solves position constraints along the OLD
    // normals, then projects velocity at the NEW geometry. Signed nonzero dt
    // permits reversibility checks. Throws on a failed/local-branch solve.
    ConstrainedDrift Drift(const std::vector<Vec3>& position,
                           const std::vector<Vec3>& tangent_velocity, double dt,
                           const Vec3& orthogonal_box,
                           double input_length_tolerance) const;
    ConstrainedStress Evaluate(const std::vector<Vec3>& position,
                               const std::vector<Vec3>& half_velocity,
                               const std::vector<Vec3>& force,
                               const Stress6& potential_virial,
                               double dt_internal, const Vec3& orthogonal_box,
                               double length_tolerance) const;

   private:
    std::vector<double> mass_;
    std::vector<DistanceConstraint> pairs_;
    std::vector<std::vector<int>> components_;
};
}  // namespace SpongeDynamics

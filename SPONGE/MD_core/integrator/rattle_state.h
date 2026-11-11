#pragma once
#include <memory>

#include "../../common.h"
#include "../../control.h"
#include "../../utils/h5md/h5_structural_state.hpp"
#include "rattle_step.h"

namespace SpongeDynamics
{
// Owns canonical global-atom double state. No flow modes or thermostat state.
class ConstrainedNVEState
{
   public:
    void Initial(CONTROLLER* controller);
    bool Selected() const { return selected_; }
    bool Active() const { return plan_ != nullptr; }
    void Configure(int count, const float* mass,
                   std::vector<DistanceConstraint> pairs, VECTOR box,
                   VECTOR angles, double dt, double dt_ps,
                   VECTOR* global_coordinates, VECTOR* global_velocity,
                   double initial_thermal_energy = 0);
    void Observe(int count, const int* ids, const VECTOR* coordinates,
                 const VECTOR* velocity, const VECTOR* force, const float* mass,
                 const LTMatrix3* virial, double dt);
    void Advance(int count, VECTOR* coordinates, VECTOR* velocity);
    void Restore(const SpongeH5MD::RestartDynamicState& state);
    void Export(SpongeH5MD::RestartDynamicState* state) const;
    const std::vector<Vec3>* Positions() const
    {
        return Active() ? &q_ : nullptr;
    }
    const std::vector<Vec3>* Velocities() const
    {
        return Active() ? &s_ : nullptr;
    }
    // Continuous coordinates in the same owner order as coupling force arrays.
    const std::vector<Vec3>* Work_Positions() const
    {
        return Active() ? &local_positions_ : nullptr;
    }
    const ConstrainedStress& Observation() const { return observation_; }
    const std::string& Identity() const { return identity_; }

   private:
    void Publish(VECTOR* coordinates, VECTOR* velocity, bool global);
    CONTROLLER* controller_ = nullptr;
    bool selected_ = false, ready_ = false, restored_ = false;
    int count_ = 0;
    double dt_ = 0, drift_energy_ = 0;
    Vec3 box_{};
    std::string identity_, restored_identity_;
    std::map<std::string, std::vector<double>> restored_values_;
    std::unique_ptr<ConstraintStressPlan> plan_;
    std::vector<double> mass_;
    std::vector<Vec3> q_, s_, force_, local_positions_;
    std::vector<int> ids_;
    std::vector<VECTOR> host_q_, host_s_, host_force_;
    std::vector<float> host_mass_;
    std::vector<LTMatrix3> host_virial_;
    ConstrainedStress observation_;
};
}  // namespace SpongeDynamics

#pragma once
#include <memory>

#include "../../common.h"
#include "../../control.h"
#include "../../thermostat/velocity_bias/mode_math.h"
#include "../../thermostat/velocity_bias/spatial_mode.h"
#include "../../utils/h5md/h5_structural_state.hpp"
#include "../physical_state.h"
#include "constrained_middle_step.h"
namespace SpongeDynamics
{
class ConstrainedMiddle
{
   public:
    void Initial(CONTROLLER*, int, const SpatialMode&);
    bool enabled() const { return parameters_.enabled; }
    bool Constrained() const { return constraint_plan_ != nullptr; }
    double Constrained_Time_Step() const { return constrained_dt_; }
    void Configure_Constraints(const float*, std::vector<DistanceConstraint>,
                               VECTOR, double, VECTOR*, VECTOR*);
    void Restore_Constrained_State(const SpongeH5MD::RestartDynamicState&);
    void Export_Constrained_State(SpongeH5MD::RestartDynamicState*) const;
    void Observe_Constrained(int, const int*, const VECTOR*, const VECTOR*,
                             const VECTOR*, const float*, const LTMatrix3*,
                             double);
    double Advance_Constraints(int, VECTOR*, VECTOR*, const VECTOR*, double,
                               double);
    const std::vector<Vec3>* Work_Positions() const
    {
        return Constrained() ? &local_positions_ : nullptr;
    }
    const ConstrainedStress& Constrained_Observation() const
    {
        return constrained_observation_;
    }
    const std::string& Constraint_Identity() const
    {
        return constraint_identity_;
    }
    const std::vector<Vec3>* Positions() const
    {
        return Constrained() ? &constrained_positions_ : nullptr;
    }
    const std::vector<Vec3>* Velocities() const
    {
        return Constrained() ? &constrained_velocities_ : nullptr;
    }
    PhysicalStateView View() const
    {
        return {&constrained_positions_,
                &constrained_observation_.velocity,
                &constrained_masses_,
                &constrained_observation_,
                &sample_,
                constrained_dt_,
                drift_energy_,
                thermal_dof_,
                constrained_ready_};
    }
    void Clear() { *this = ConstrainedMiddle{}; }

   private:
    void Load_Host(int, const VECTOR*, const VECTOR*, const float*);
    void Publish_Constrained(VECTOR*, VECTOR*, bool);
    CONTROLLER* controller_ = nullptr;
    SpatialMode parameters_;
    int count_ = 0, thermal_dof_ = 0;
    CosineFit sample_;
    std::vector<VECTOR> h_crd_, h_vel_, h_force_;
    std::vector<float> h_mass_;
    std::vector<Vec3> positions_, velocities_;
    std::vector<double> masses_;
    std::unique_ptr<ConstraintStressPlan> constraint_plan_;
    std::vector<Vec3> constrained_positions_, constrained_velocities_,
        constrained_forces_, local_positions_;
    std::vector<double> constrained_masses_;
    std::vector<int> constrained_ids_;
    std::vector<LTMatrix3> constrained_virial_;
    ConstrainedStress constrained_observation_;
    Vec3 constrained_box_{};
    double constrained_dt_ = 0, drift_energy_ = 0;
    bool constrained_ready_ = false, constrained_restart_ = false;
    std::string constraint_identity_, restart_constraint_identity_;
    std::map<std::string, std::vector<double>> restart_constraint_values_;
};
}  // namespace SpongeDynamics

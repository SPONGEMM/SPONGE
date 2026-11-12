#pragma once
#include <memory>

#include "../../common.h"
#include "../../constrain/thermal_frame.h"
#include "../../control.h"
#include "mode_math.h"
#include "spatial_mode.h"
namespace SpongeDynamics
{
class VelocityBias
{
   public:
    void Initial(CONTROLLER*, int, const float*, VECTOR, VECTOR, bool,
                 const std::string&);
    bool enabled() const { return parameters_.enabled; }
    const SpatialMode& Parameters() const { return parameters_; }
    const ThermalState& State() const { return state_; }
    const std::string& Configuration() const { return parameters_.identity; }
    const VECTOR* Retained_Velocity() const { return retained_; }
    void Configure_Constraints(const float*, std::vector<DistanceConstraint>,
                               VECTOR, const std::string&);
    bool Constrained() const { return constraint_plan_ != nullptr; }
    const ConstraintStressPlan* Constraint_Plan() const
    {
        return constraint_plan_.get();
    }
    const std::string& Constraint_Identity() const
    {
        return constraint_identity_;
    }
    void Expect_Constraint_Identity(const std::string& identity)
    {
        expected_constraint_identity_ = identity;
    }
    int Thermal_Degrees_Of_Freedom() const { return thermal_dof_; }
    ThermalState Prepare_State(int, const VECTOR*, const VECTOR*, const float*);
    void Remove_Bias(int, const VECTOR*, VECTOR*, const float*);
    void Restore_Bias(int, const VECTOR*, VECTOR*, const float*);
    void Sample(int, const VECTOR*, const VECTOR*, const float*);
    void Sample_Constrained(double, int);
    void Step_Print();
    void Clear();

   private:
    void Load_Host(int, const VECTOR*, const VECTOR*, const float*);
    CosineFit Fit(int, const VECTOR*, const VECTOR*, const float*);
    void Shift(int, const VECTOR*, VECTOR*, const CosineFit&, double);
    CONTROLLER* controller_ = nullptr;
    SpatialMode parameters_;
    ThermalState state_;
    double reported_temperature_ = 0;
    CosineFit sample_, bias_, thermal_fit_;
    int count_ = 0;
    bool bias_removed_ = false;
    VECTOR* retained_ = nullptr;
    std::vector<VECTOR> h_crd_, h_vel_, h_retained_;
    std::vector<float> h_mass_;
    std::vector<Vec3> positions_, velocities_;
    std::vector<double> masses_;
    std::unique_ptr<ConstraintStressPlan> constraint_plan_;
    Vec3 constraint_box_{};
    std::string constraint_identity_, expected_constraint_identity_;
    int thermal_dof_ = 0;
    std::vector<Vec3> constraint_thermal_;
};
}  // namespace SpongeDynamics

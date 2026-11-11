#pragma once
#include <memory>

#include "../MD_core/energy_ledger.h"
#include "../constrain/tangent_frame.h"
#include "sampling.h"
#include "stress_math.h"
namespace SpongeDynamics
{
class StressSampler
{
   public:
    void Initial(CONTROLLER*, int, const SamplingPlan&, double);
    bool enabled() const { return enabled_; }
    std::map<std::string, std::string> Export_Restart() const;
    void Restore_Restart(const std::map<std::string, std::string>&, double);
    void Configure_Constraint_Stress(int count, const float* mass,
                                     std::vector<DistanceConstraint> pairs,
                                     VECTOR box, VECTOR angles,
                                     double dt_internal,
                                     const std::string& solver_identity);
    bool Constrained() const { return constraint_plan_ != nullptr; }
    std::map<std::string, std::string> Constraint_Metadata() const;
    std::map<std::string, std::string> Constraint_Units() const;
    std::map<std::string, double> Sample_Stress(
        int count, const VECTOR* velocity, const VECTOR* force,
        const float* mass, const LTMatrix3* virial, double dt_internal,
        double volume, const VECTOR* coordinates, const int* local_to_global,
        VECTOR box, const std::vector<Vec3>* canonical_positions = nullptr,
        const std::vector<Vec3>* canonical_velocities = nullptr);

    void Clear() { *this = StressSampler{}; }

   private:
    CONTROLLER* controller_ = nullptr;
    bool enabled_ = false;
    std::vector<VECTOR> stress_vel_, stress_force_;
    std::vector<float> stress_mass_;
    std::vector<LTMatrix3> stress_virial_;
    std::vector<double> stress_masses_;
    std::vector<Vec3> stress_velocities_, stress_forces_;
    std::unique_ptr<ConstraintStressPlan> constraint_plan_;
    std::vector<VECTOR> stress_crd_;
    std::vector<Vec3> stress_positions_;
    std::vector<int> stress_ids_;
    std::vector<double> constraint_masses_;
    Vec3 constraint_box_{};
    int constraint_count_ = 0;
    double constraint_dt_ = 0, stress_dt_ps_ = 0;
    double constraint_length_tolerance_ = 1e-4;
    bool constraint_canonical_double_ = false;
    std::string constraint_identity_, constraint_solver_;
    bool deferred_restart_ = false;
    std::map<std::string, std::string> restart_;
};
}  // namespace SpongeDynamics

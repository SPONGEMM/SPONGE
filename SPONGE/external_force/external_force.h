#pragma once
#include "../common.h"
#include "../control.h"
#include "drive_config.h"
namespace SpongeDynamics
{
class ExternalForce
{
   public:
    bool enabled() const { return !drives_.empty(); }
    const std::string& Configuration() const { return configuration_; }
    const VECTOR* External_Force() const { return external_force_; }
    void Initial(CONTROLLER*, int, const float*, const float*, bool, VECTOR,
                 VECTOR);
    void Evaluate(int, const int*, VECTOR*, const VECTOR*, double);
    const VECTOR* Prepare_Force(int, const int*, const VECTOR*, double);
    void Clear();

   private:
    CONTROLLER* controller_ = nullptr;
    int atom_count_ = 0;
    bool has_groups_ = false;
    VECTOR box_{};
    std::string configuration_;
    std::vector<Drive> drives_;
    std::vector<VECTOR*> base_forces_;
    std::vector<double> max_base_;
    VECTOR* external_force_ = nullptr;
    std::vector<double> masses_;
    std::vector<int> ids_;
    std::vector<VECTOR> coordinates_;
    std::vector<Vec3> global_coordinates_;
};
}  // namespace SpongeDynamics

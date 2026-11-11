#pragma once
#include "../MD_core/physical_state.h"
#include "../momentum_exchange/exchange_math.h"
#include "sampling.h"
namespace SpongeDynamics
{
class ProfileSampler
{
   public:
    void Initial(CONTROLLER*, int, const float*, VECTOR, const SamplingPlan&,
                 bool);
    bool enabled() const { return parameters_.enabled; }
    std::map<std::string, double> Sample(int, const VECTOR*, const VECTOR*,
                                         const VECTOR*, const float*, double,
                                         PhysicalStateView = {});
    std::map<std::string, std::string> Units() const
    {
        return parameters_.cosine ? Cosine_Units() : Profile_Units();
    }
    void Clear() { *this = ProfileSampler{}; }

   private:
    bool Constrained() const { return constrained_; }
    std::map<std::string, std::string> Profile_Units() const,
        Cosine_Units() const;
    std::map<std::string, double> Profile(int, const VECTOR*, const VECTOR*,
                                          const VECTOR*, const float*, double);
    std::map<std::string, double> Cosine_Profile(int, const VECTOR*,
                                                 const VECTOR*, const VECTOR*,
                                                 const float*, double);
    void Load_Host(int, const VECTOR*, const VECTOR*, const float*);
    CONTROLLER* controller_ = nullptr;
    SpatialMode parameters_;
    int count_ = 0, thermal_dof_ = 0;
    double density_ = 0;
    bool constrained_ = false;
    PhysicalStateView view_;
    std::vector<VECTOR> h_crd_, h_vel_, h_force_;
    std::vector<float> h_mass_;
    std::vector<Vec3> positions_, velocities_;
    std::vector<double> masses_;
};
}  // namespace SpongeDynamics

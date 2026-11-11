#pragma once
#include "../common.h"
#include "../control.h"
#include "../thermostat/velocity_bias/spatial_mode.h"
namespace SpongeDynamics
{
struct SamplingPlan
{
    bool stress = false, profile = false, centered = true;
    int interval = 1;
    std::string output = "observables.h5";
    SpatialMode mode;
    bool Enabled() const { return stress || profile; }
    bool Due(int step) const { return Enabled() && step % interval == 0; }
    void Initial(CONTROLLER*, VECTOR, VECTOR, bool);
};
}  // namespace SpongeDynamics

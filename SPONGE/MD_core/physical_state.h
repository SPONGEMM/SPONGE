#pragma once
#include "../constrain/tangent_frame.h"
#include "../thermostat/velocity_bias/mode_math.h"
namespace SpongeDynamics
{
struct PhysicalStateView
{
    const std::vector<Vec3>* position = nullptr;
    const std::vector<Vec3>* velocity = nullptr;
    const std::vector<double>* mass = nullptr;
    const ConstrainedStress* observation = nullptr;
    const CosineFit* mode_fit = nullptr;
    double dt = 0, drift_energy = 0;
    int thermal_dof = 0;
    bool ready = false;
    bool active() const { return position != nullptr; }
};
}  // namespace SpongeDynamics

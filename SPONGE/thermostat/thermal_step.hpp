#pragma once
#include "../MD_core/energy_accounting.h"
#include "velocity_bias/velocity_bias.h"
namespace SpongeDynamics
{
template <class Thermostat>
void Scale_Thermal(Thermostat& thermostat, VelocityBias& bias,
                   EnergyAccounting* energy, int n, const VECTOR* q, VECTOR* v,
                   const float* mass)
{
    if (energy) energy->Begin_Thermal(n, v);
    bias.Remove_Bias(n, q, v, mass);
    const auto& state = bias.State();
    thermostat.Record_Temperature(float(state.temperature),
                                  state.degrees_of_freedom);
    thermostat.Scale_Velocity(n, v);
    bias.Restore_Bias(n, q, v, mass);
    if (energy) energy->End_Thermal(n, v, mass);
}
}  // namespace SpongeDynamics

#pragma once
#include <string>
namespace SpongeDynamics
{
struct SpatialMode
{
    bool enabled = false;
    bool cosine = true;
    int flow = 0, gradient = 2, bins = 20;
    double wave_number = 0, length = 0;
    std::string identity;
};
struct ThermalState
{
    double kinetic_energy = 0, temperature = 0;
    int degrees_of_freedom = 0;
    const char* time_layer = "thermostat_substep";
};
}  // namespace SpongeDynamics

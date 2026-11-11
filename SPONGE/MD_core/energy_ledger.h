#pragma once

#include <cstdint>
#include <map>
#include <string>

namespace SpongeDynamics
{
// Serializable numerical state, independent of backend and file format.
struct EnergyLedger
{
    std::string configuration;
    double dt = 0, time = 0, work = 0, correction = 0;
    // Optional heat from the actual thermal/friction velocity update, positive
    // into the particles. It shares the work ledger's completed-step clock.
    bool heat_enabled = false;
    std::string heat_method = "thermal_substep_v1";
    double heat = 0, heat_correction = 0;
    std::int64_t steps = 0;
    void Initialize(const std::string& config, double dt_ps, double time_ps,
                    bool measure_heat = false);
    void Complete(double increment, double end_time, double heat_increment = 0);
    std::map<std::string, std::string> Export() const;
    void Restore(const std::map<std::string, std::string>& state,
                 double current_time);
};
std::string Precise_Number(double value);
bool Same_Time(double a, double b);
}  // namespace SpongeDynamics

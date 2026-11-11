#include "energy_ledger.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>

#include "../utils/float_classification.hpp"

namespace SpongeDynamics
{
namespace
{
double Number(const std::string& text)
{
    std::istringstream input(text);
    input.imbue(std::locale::classic());
    double value;
    if (!(input >> value) || !SpongeFloat::Is_Finite(value))
        throw std::runtime_error("coupling restart contains an invalid number");
    input >> std::ws;
    if (!input.eof())
        throw std::runtime_error("coupling restart number has trailing data");
    return value;
}
// Explicit rounding points preserve compensation under fast-math.
void Add(double increment, double& sum, double& correction)
{
    volatile double adjusted = increment - correction;
    volatile double next = sum + adjusted;
    volatile double difference = next - sum;
    correction = difference - adjusted;
    sum = next;
    if (!SpongeFloat::Is_Finite(sum) || !SpongeFloat::Is_Finite(correction))
        throw std::runtime_error("coupling energy ledger overflow");
}
}  // namespace

std::string Precise_Number(double value)
{
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(std::numeric_limits<double>::max_digits10)
        << value;
    return out.str();
}
bool Same_Time(double a, double b)
{
    return SpongeFloat::Is_Finite(a) && SpongeFloat::Is_Finite(b) &&
           std::abs(a - b) <= 32 * std::numeric_limits<double>::epsilon() *
                                  std::max({1.0, std::abs(a), std::abs(b)});
}
void EnergyLedger::Initialize(const std::string& config, double dt_ps,
                              double time_ps, bool measure_heat)
{
    if (!SpongeFloat::Is_Finite(dt_ps) || dt_ps < 0 ||
        !SpongeFloat::Is_Finite(time_ps))
        throw std::runtime_error(
            "coupling requires finite time and nonnegative dt");
    configuration = config;
    dt = dt_ps;
    time = time_ps;
    work = correction = heat = heat_correction = 0;
    heat_enabled = measure_heat;
    heat_method = "thermal_substep_v1";
    steps = 0;
}
void EnergyLedger::Complete(double increment, double end_time,
                            double heat_increment)
{
    if (!SpongeFloat::Is_Finite(increment) ||
        !SpongeFloat::Is_Finite(heat_increment) ||
        (!heat_enabled && heat_increment != 0) ||
        !Same_Time(time + dt, end_time) ||
        steps == std::numeric_limits<std::int64_t>::max())
        throw std::runtime_error(
            "coupling energy: invalid increment or completed-step time");
    double next_work = work, next_correction = correction;
    double next_heat = heat, next_heat_correction = heat_correction;
    Add(increment, next_work, next_correction);
    if (heat_enabled) Add(heat_increment, next_heat, next_heat_correction);
    work = next_work;
    correction = next_correction;
    heat = next_heat;
    heat_correction = next_heat_correction;
    time = end_time;
    ++steps;
}
std::map<std::string, std::string> EnergyLedger::Export() const
{
    std::map<std::string, std::string> state{
        {"method", "trapezoid_path_work_v1"},
        {"configuration", configuration},
        {"dt", Precise_Number(dt)},
        {"time", Precise_Number(time)},
        {"work", Precise_Number(work)},
        {"correction", Precise_Number(correction)},
        {"steps", std::to_string(steps)}};
    if (heat_enabled)
    {
        state["heat_method"] = heat_method;
        state["heat"] = Precise_Number(heat);
        state["heat_correction"] = Precise_Number(heat_correction);
    }
    return state;
}
void EnergyLedger::Restore(const std::map<std::string, std::string>& state,
                           double current_time)
{
    if (heat_enabled && (!state.count("heat_method") || !state.count("heat") ||
                         !state.count("heat_correction")))
        throw std::runtime_error(
            "coupling restart lacks projected OU heat history; use structural "
            "load to start a new experiment");
    if (state.size() != (heat_enabled ? 10u : 7u) || !state.count("method") ||
        !state.count("configuration") || !state.count("dt") ||
        !state.count("time") || !state.count("work") ||
        !state.count("correction") || !state.count("steps"))
        throw std::runtime_error(
            "coupling restart ledger is incomplete or has unknown fields");
    if (state.at("method") != "trapezoid_path_work_v1" ||
        state.at("configuration") != configuration ||
        Number(state.at("dt")) != dt)
        throw std::runtime_error(
            "coupling restart configuration, dt or work method mismatch; use "
            "structural load to start a new experiment");
    if (heat_enabled && state.at("heat_method") != heat_method)
        throw std::runtime_error("coupling restart heat method mismatch");
    const double restored_heat = heat_enabled ? Number(state.at("heat")) : 0;
    const double restored_heat_correction =
        heat_enabled ? Number(state.at("heat_correction")) : 0;
    const double restored_time = Number(state.at("time"));
    if (!Same_Time(restored_time, current_time))
        throw std::runtime_error(
            "coupling restart time does not match integrator time");
    const double restored_work = Number(state.at("work"));
    const double restored_correction = Number(state.at("correction"));
    std::size_t consumed = 0;
    const auto restored_steps = std::stoll(state.at("steps"), &consumed);
    if (consumed != state.at("steps").size() || restored_steps < 0)
        throw std::runtime_error("coupling restart step count is invalid");
    time = restored_time;
    work = restored_work;
    correction = restored_correction;
    steps = restored_steps;
    heat = restored_heat;
    heat_correction = restored_heat_correction;
}
}  // namespace SpongeDynamics

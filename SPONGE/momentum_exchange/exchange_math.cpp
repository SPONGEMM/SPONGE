#include "exchange_math.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <locale>
#include <set>
#include <sstream>
#include <stdexcept>

#include "../MD_core/energy_ledger.h"
#include "../utils/float_classification.hpp"

namespace SpongeDynamics
{
int Periodic_Bin(double position, double length, int bins)
{
    if (!SpongeFloat::Is_Finite(position) || !SpongeFloat::Is_Finite(length) ||
        length <= 0 || bins < 1)
        throw std::runtime_error("invalid periodic profile geometry");
    double fraction = position / length;
    if (!SpongeFloat::Is_Finite(fraction) || std::abs(fraction) > 1e12)
        throw std::runtime_error(
            "profile coordinate exceeds reliable wrapping range");
    fraction -= std::floor(fraction);
    return std::min(bins - 1, static_cast<int>(fraction * bins));
}
MomentumExchange Select_Exchange(const std::vector<double>& mass,
                                 const std::vector<double>& position,
                                 const std::vector<double>& velocity,
                                 const std::vector<int>& ids, double length,
                                 int bins)
{
    if (mass.size() != position.size() || mass.size() != velocity.size() ||
        mass.size() != ids.size() || bins < 4 || bins % 2)
        throw std::runtime_error("invalid RNEMD arrays or bin count");
    MomentumExchange result;
    bool have_low = false, have_middle = false;
    std::set<int> seen;
    for (std::size_t i = 0; i < mass.size(); ++i)
    {
        if (!SpongeFloat::Is_Finite(mass[i]) || mass[i] <= 0 ||
            !SpongeFloat::Is_Finite(velocity[i]) || ids[i] < 0 ||
            !seen.insert(ids[i]).second)
            throw std::runtime_error(
                "invalid RNEMD mass, velocity or owner ID");
        const int bin = Periodic_Bin(position[i], length, bins);
        if (bin == 0)
        {
            have_low = true;
            if (velocity[i] >= 0 &&
                (result.low < 0 || velocity[i] > velocity[result.low] ||
                 (velocity[i] == velocity[result.low] &&
                  ids[i] < ids[result.low])))
                result.low = static_cast<int>(i);
        }
        else if (bin == bins / 2)
        {
            have_middle = true;
            if (velocity[i] <= 0 &&
                (result.middle < 0 || velocity[i] < velocity[result.middle] ||
                 (velocity[i] == velocity[result.middle] &&
                  ids[i] < ids[result.middle])))
                result.middle = static_cast<int>(i);
        }
    }
    if (!have_low || !have_middle)
    {
        result.status = ExchangeStatus::EmptySlab;
        return result;
    }
    if (result.low < 0 || result.middle < 0) return result;
    const int a = result.low, b = result.middle;
    if (velocity[a] <= velocity[b]) return result;
    const double com =
        (mass[a] * velocity[a] + mass[b] * velocity[b]) / (mass[a] + mass[b]);
    result.low_velocity = 2 * com - velocity[a];
    result.middle_velocity = 2 * com - velocity[b];
    result.momentum = mass[b] * (result.middle_velocity - velocity[b]);
    if (!SpongeFloat::Is_Finite(result.momentum) ||
        !SpongeFloat::Is_Finite(result.low_velocity) ||
        !SpongeFloat::Is_Finite(result.middle_velocity))
        throw std::runtime_error("RNEMD exchange overflow");
    result.status = ExchangeStatus::Swapped;
    return result;
}
namespace
{
double Number(const std::string& text)
{
    std::istringstream in(text);
    in.imbue(std::locale::classic());
    double value;
    if (!(in >> value) || !SpongeFloat::Is_Finite(value))
        throw std::runtime_error("invalid RNEMD restart number");
    in >> std::ws;
    if (!in.eof())
        throw std::runtime_error("invalid RNEMD restart number suffix");
    return value;
}
std::int64_t Integer(const std::string& text, bool negative = false)
{
    if (negative && text == "-1") return -1;
    if (text.empty() ||
        text.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("invalid RNEMD restart counter");
    return std::stoll(text);
}
}  // namespace
void ExchangeLedger::Initialize(const std::string& config, double dt_ps,
                                double time_ps)
{
    if (!SpongeFloat::Is_Finite(dt_ps) || dt_ps <= 0 ||
        !SpongeFloat::Is_Finite(time_ps))
        throw std::runtime_error("RNEMD requires finite time and positive dt");
    *this = {};
    configuration = config;
    dt = dt_ps;
    time = time_ps;
}
void ExchangeLedger::Record(const MomentumExchange& event, double time_ps,
                            int step)
{
    if (!Same_Time(time, time_ps) || step <= last_exchange_step || step <= 0 ||
        attempts == std::numeric_limits<std::int64_t>::max())
        throw std::runtime_error("RNEMD exchange time/order invalid");
    ++attempts;
    last_exchange_step = step;
    if (event.status == ExchangeStatus::EmptySlab)
    {
        ++empty;
        return;
    }
    if (event.status == ExchangeStatus::NoCandidate)
    {
        ++no_candidate;
        return;
    }
    if (!SpongeFloat::Is_Finite(event.momentum) || event.momentum < 0 ||
        !SpongeFloat::Is_Finite(event.momentum_error) ||
        !SpongeFloat::Is_Finite(event.energy_error))
        throw std::runtime_error("invalid RNEMD exchange increment");
    ++swaps;
    volatile double adjusted = event.momentum - correction;
    volatile double next = momentum + adjusted;
    volatile double difference = next - momentum;
    correction = difference - adjusted;
    momentum = next;
    momentum_error += event.momentum_error;
    energy_error += event.energy_error;
    if (!SpongeFloat::Is_Finite(momentum) ||
        !SpongeFloat::Is_Finite(momentum_error) ||
        !SpongeFloat::Is_Finite(energy_error))
        throw std::runtime_error("RNEMD ledger overflow");
}
void ExchangeLedger::Complete(double end_time)
{
    if (!Same_Time(time + dt, end_time) ||
        steps == std::numeric_limits<std::int64_t>::max())
        throw std::runtime_error("invalid RNEMD completed-step time");
    time = end_time;
    ++steps;
}
std::map<std::string, std::string> ExchangeLedger::Export() const
{
    return {{"method", "centered_rnemd_v1"},
            {"configuration", configuration},
            {"dt", Precise_Number(dt)},
            {"time", Precise_Number(time)},
            {"momentum", Precise_Number(momentum)},
            {"correction", Precise_Number(correction)},
            {"momentum_error", Precise_Number(momentum_error)},
            {"energy_error", Precise_Number(energy_error)},
            {"steps", std::to_string(steps)},
            {"attempts", std::to_string(attempts)},
            {"swaps", std::to_string(swaps)},
            {"empty", std::to_string(empty)},
            {"no_candidate", std::to_string(no_candidate)},
            {"last_exchange_step", std::to_string(last_exchange_step)}};
}
void ExchangeLedger::Restore(const std::map<std::string, std::string>& state,
                             double current_time)
{
    const auto expected = Export();
    if (state.size() != expected.size())
        throw std::runtime_error("invalid RNEMD restart field count");
    for (const auto& field : expected)
        if (!state.count(field.first))
            throw std::runtime_error("missing RNEMD restart field");
    if (state.at("method") != "centered_rnemd_v1" ||
        state.at("configuration") != configuration ||
        Number(state.at("dt")) != dt ||
        !Same_Time(Number(state.at("time")), current_time))
        throw std::runtime_error("RNEMD restart configuration/time mismatch");
    ExchangeLedger restored = *this;
    restored.time = Number(state.at("time"));
    restored.momentum = Number(state.at("momentum"));
    restored.correction = Number(state.at("correction"));
    restored.momentum_error = Number(state.at("momentum_error"));
    restored.energy_error = Number(state.at("energy_error"));
    restored.steps = Integer(state.at("steps"));
    restored.attempts = Integer(state.at("attempts"));
    restored.swaps = Integer(state.at("swaps"));
    restored.empty = Integer(state.at("empty"));
    restored.no_candidate = Integer(state.at("no_candidate"));
    restored.last_exchange_step = Integer(state.at("last_exchange_step"), true);
    if (restored.momentum < 0 || restored.swaps > restored.attempts ||
        restored.empty > restored.attempts - restored.swaps ||
        restored.no_candidate !=
            restored.attempts - restored.swaps - restored.empty ||
        ((restored.attempts == 0) != (restored.last_exchange_step == -1)))
        throw std::runtime_error("inconsistent RNEMD restart counters");
    *this = restored;
}
}  // namespace SpongeDynamics

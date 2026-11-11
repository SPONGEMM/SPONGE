#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "../MD_core/dynamics_types.h"

namespace SpongeDynamics
{
struct ExchangeParameters
{
    bool enabled = false;
    int flow = 0, gradient = 2, bins = 20, exchange_interval = 0;
    double length = 0, area = 0;  // Angstrom and Angstrom^2
    std::string identity;
};
enum class ExchangeStatus
{
    EmptySlab,
    NoCandidate,
    Swapped
};
struct MomentumExchange
{
    ExchangeStatus status = ExchangeStatus::NoCandidate;
    int low = -1, middle = -1;                     // local owner indices
    double low_velocity = 0, middle_velocity = 0;  // centered internal units
    double momentum = 0, momentum_error = 0, energy_error = 0;
};
int Periodic_Bin(double position, double length, int bins);
MomentumExchange Select_Exchange(const std::vector<double>& mass,
                                 const std::vector<double>& position,
                                 const std::vector<double>& velocity,
                                 const std::vector<int>& ids, double length,
                                 int bins);

// Pure numerical state; serialized through the existing coupling restart map.
struct ExchangeLedger
{
    std::string configuration;
    double dt = 0, time = 0, momentum = 0, correction = 0;
    double momentum_error = 0, energy_error = 0;
    std::int64_t steps = 0, attempts = 0, swaps = 0, empty = 0,
                 no_candidate = 0;
    std::int64_t last_exchange_step = -1;
    void Initialize(const std::string& config, double dt_ps, double time_ps);
    void Record(const MomentumExchange& event, double time_ps, int step);
    void Complete(double end_time);
    std::map<std::string, std::string> Export() const;
    void Restore(const std::map<std::string, std::string>& state,
                 double current_time);
};
}  // namespace SpongeDynamics

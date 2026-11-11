#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

#include "momentum_exchange/exchange_math.h"

void Check(bool ok, const char* message)
{
    if (!ok)
    {
        std::cerr << message << "\n";
        std::exit(1);
    }
}
template <class F>
void Reject(F function)
{
    bool rejected = false;
    try
    {
        function();
    }
    catch (const std::exception&)
    {
        rejected = true;
    }
    Check(rejected, "invalid RNEMD input accepted");
}
int main()
{
    using namespace SpongeDynamics;
    Check(Periodic_Bin(-1, 32, 8) == 7 && Periodic_Bin(32, 32, 8) == 0,
          "periodic wrapping");
    Reject([]
           { Periodic_Bin(std::numeric_limits<double>::quiet_NaN(), 32, 8); });
    Reject([] { Periodic_Bin(1, 0, 8); });
    std::mt19937 generator(20261027);
    std::uniform_real_distribution<double> mass(.1, 100), speed(0, 10);
    for (int i = 0; i < 1000; ++i)
    {
        double a = mass(generator), b = mass(generator), u = speed(generator),
               v = -speed(generator);
        auto event = Select_Exchange({a, b}, {1, 17}, {u, v}, {0, 1}, 32, 8);
        Check(event.status == ExchangeStatus::Swapped, "valid pair missing");
        double p0 = a * u + b * v,
               p1 = a * event.low_velocity + b * event.middle_velocity;
        double e0 = a * u * u + b * v * v,
               e1 = a * event.low_velocity * event.low_velocity +
                    b * event.middle_velocity * event.middle_velocity;
        Check(std::abs(p1 - p0) < 1e-10 && std::abs(e1 - e0) < 1e-9,
              "mass COM reflection not conservative");
        Check(event.momentum > 0, "momentum sign");
    }
    auto ties = Select_Exchange({1, 1, 1, 1}, {1, 2, 17, 18}, {2, 2, -3, -3},
                                {8, 2, 9, 3}, 32, 8);
    Check(ties.low == 1 && ties.middle == 3, "global ID tie breaker");
    Check(Select_Exchange({1, 1}, {1, 5}, {2, -3}, {0, 1}, 32, 8).status ==
              ExchangeStatus::EmptySlab,
          "empty slab");
    Check(Select_Exchange({1, 1}, {1, 17}, {-2, 3}, {0, 1}, 32, 8).status ==
              ExchangeStatus::NoCandidate,
          "wrong sign candidates");
    Check(Select_Exchange({1, 1}, {1, 17}, {0, 0}, {0, 1}, 32, 8).status ==
              ExchangeStatus::NoCandidate,
          "zero impulse");
    Reject([] { Select_Exchange({1, 1}, {1, 17}, {2, -3}, {0, 0}, 32, 8); });
    Reject([] { Select_Exchange({1, -1}, {1, 17}, {2, -3}, {0, 1}, 32, 8); });
    ExchangeLedger full, resumed;
    full.Initialize("config", .000123456789, 0);
    MomentumExchange event;
    event.status = ExchangeStatus::Swapped;
    event.momentum = 1e16;
    full.Complete(full.dt);
    full.Record(event, full.time, 1);
    event.momentum = 1;
    full.Complete(2 * full.dt);
    full.Record(event, full.time, 2);
    const auto state = full.Export();
    resumed.Initialize("config", full.dt, 0);
    resumed.Restore(state, full.time);
    for (int i = 3; i <= 10; ++i)
    {
        full.Complete(i * full.dt);
        resumed.Complete(i * full.dt);
        full.Record(event, full.time, i);
        resumed.Record(event, resumed.time, i);
    }
    Check(full.Export() == resumed.Export(), "restart changed ledger");
    Check(full.momentum == 1e16 + 8,
          "compensated accumulation under fast math");
    for (const auto& field : state)
    {
        auto broken = state;
        broken.erase(field.first);
        Reject([&] { resumed.Restore(broken, 2 * full.dt); });
    }
    for (const auto& field :
         std::map<std::string, std::string>{{"method", "bad"},
                                            {"configuration", "changed"},
                                            {"dt", "1"},
                                            {"time", "1"},
                                            {"momentum", "nan"},
                                            {"correction", "inf"},
                                            {"steps", "-1"},
                                            {"swaps", "99"},
                                            {"empty", "1"},
                                            {"last_exchange_step", "-1"}})
    {
        auto broken = state;
        broken[field.first] = field.second;
        Reject([&] { resumed.Restore(broken, 2 * full.dt); });
    }
    Reject([&] { full.Record(event, full.time, 10); });
}

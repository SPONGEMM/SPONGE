#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "MD_core/energy_ledger.h"

void Check(bool ok, const char* message)
{
    if (!ok)
    {
        std::cerr << message << '\n';
        std::exit(1);
    }
}
int main()
{
    SpongeDynamics::EnergyLedger full, resumed;
    full.Initialize("force protocol", 0.000123456789, 0);
    full.Complete(1e16, full.dt);
    full.Complete(1, 2 * full.dt);
    auto state = full.Export();
    resumed.Initialize("force protocol", full.dt, 0);
    resumed.Restore(state, 2 * full.dt);
    for (int i = 3; i <= 10; ++i)
    {
        full.Complete(1, i * full.dt);
        resumed.Complete(1, i * full.dt);
    }
    Check(full.Export() == resumed.Export(),
          "restart lost compensated sum or precise time");
    Check(full.work == 1e16 + 8,
          "compensated summation failed under fast-math");
    for (const auto& key : {"method", "configuration", "dt", "time", "work",
                            "correction", "steps"})
    {
        auto broken = state;
        broken.erase(key);
        bool rejected = false;
        try
        {
            resumed.Restore(broken, 2 * full.dt);
        }
        catch (const std::exception&)
        {
            rejected = true;
        }
        Check(rejected, "incomplete ledger accepted");
    }
    for (const auto& item :
         std::map<std::string, std::string>{{"method", "v2"},
                                            {"configuration", "changed"},
                                            {"dt", "1"},
                                            {"time", "1"},
                                            {"work", "nan"},
                                            {"correction", "inf"},
                                            {"steps", "-1"}})
    {
        auto broken = state;
        broken[item.first] = item.second;
        bool rejected = false;
        try
        {
            resumed.Restore(broken, 2 * full.dt);
        }
        catch (const std::exception&)
        {
            rejected = true;
        }
        Check(rejected, "invalid ledger accepted");
    }
    auto broken = state;
    broken["new_field"] = "0";
    bool rejected = false;
    try
    {
        resumed.Restore(broken, 2 * full.dt);
    }
    catch (const std::exception&)
    {
        rejected = true;
    }
    Check(rejected, "unknown ledger field accepted");
    rejected = false;
    try
    {
        full.Complete(std::numeric_limits<double>::quiet_NaN(),
                      full.time + full.dt);
    }
    catch (const std::exception&)
    {
        rejected = true;
    }
    Check(rejected, "nonfinite work accepted");

    SpongeDynamics::EnergyLedger hot, hot_resumed;
    hot.Initialize("projected Middle", 0.001, 0, true);
    hot.Complete(2, .001, -1e16);
    hot.Complete(3, .002, -1);
    hot_resumed.Initialize("projected Middle", .001, 0, true);
    hot_resumed.Restore(hot.Export(), .002);
    for (int i = 3; i <= 10; ++i)
    {
        hot.Complete(1, i * .001, -1);
        hot_resumed.Complete(1, i * .001, -1);
    }
    Check(hot.Export() == hot_resumed.Export(),
          "heat restart lost compensation");
    Check(hot.heat == -1e16 - 8, "heat compensation failed");
    auto reject_heat = [&](const std::map<std::string, std::string>& payload)
    {
        bool failed = false;
        const auto previous = hot_resumed.Export();
        try
        {
            hot_resumed.Restore(payload, hot.time);
        }
        catch (const std::exception&)
        {
            failed = true;
        }
        Check(failed, "invalid heat payload accepted");
        Check(previous == hot_resumed.Export(),
              "failed restore mutated ledger");
    };
    for (const char* key : {"heat_method", "heat", "heat_correction"})
    {
        auto payload = hot.Export();
        payload.erase(key);
        reject_heat(payload);
        payload = hot.Export();
        payload[key] = "nan";
        reject_heat(payload);
    }
    auto legacy = hot.Export();
    for (const char* key : {"heat_method", "heat", "heat_correction"})
        legacy.erase(key);
    reject_heat(legacy);
    auto extra = hot.Export();
    extra["unknown"] = "0";
    reject_heat(extra);
    const auto previous = hot.Export();
    rejected = false;
    try
    {
        hot.Complete(1, hot.time + hot.dt,
                     std::numeric_limits<double>::quiet_NaN());
    }
    catch (const std::exception&)
    {
        rejected = true;
    }
    Check(rejected && hot.Export() == previous,
          "invalid heat changed completed ledger");
    rejected = false;
    try
    {
        full.Restore(hot.Export(), hot.time);
    }
    catch (const std::exception&)
    {
        rejected = true;
    }
    Check(rejected, "NVE accepted a heat ledger");
    rejected = false;
    try
    {
        full.Complete(0, full.time + full.dt, 1);
    }
    catch (const std::exception&)
    {
        rejected = true;
    }
    Check(rejected, "NVE accepted a heat event");
}

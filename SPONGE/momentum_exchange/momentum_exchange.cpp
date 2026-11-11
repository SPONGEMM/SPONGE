#include "momentum_exchange.h"

#include <cmath>
#include <limits>
#include <stdexcept>

#include "../MD_core/energy_ledger.h"
#include "../io/component_state.h"
#include "../utils/control/component_options.hpp"
namespace SpongeDynamics
{
void MomentumExchangeController::Initial(CONTROLLER* c, int n,
                                         const float* mass, VECTOR box,
                                         VECTOR angles, bool periodic,
                                         const std::string& integrator,
                                         double dt, double time)
{
    Clear();
    controller_ = c;
    try
    {
        Component_Keys(c, "momentum_exchange",
                       {"interval", "bins", "flow", "gradient"});
        if (!c->Command_Exist("momentum_exchange_interval"))
        {
            for (const auto& kv : c->commands)
                if (kv.first.rfind("momentum_exchange_", 0) == 0)
                    throw std::runtime_error(
                        "momentum_exchange.interval is required");
            return;
        }
        if (!SpongeFloat::Is_Finite(dt) || dt <= 0)
            throw std::runtime_error("exchange requires positive dt");
        Orthogonal_Box(box, angles, periodic);
        if (n < 2)
            throw std::runtime_error("exchange requires at least two atoms");
        parameters_.exchange_interval =
            Option_Integer(c, "momentum_exchange_interval", 0);
        parameters_.bins =
            Option_Integer(c, "momentum_exchange_bins", 20, 4, 10000);
        if (parameters_.bins % 2)
            throw std::runtime_error("exchange bins must be even");
        parameters_.flow = Option_Axis(c, "momentum_exchange_flow", 0);
        parameters_.gradient = Option_Axis(c, "momentum_exchange_gradient", 2);
        if (parameters_.flow == parameters_.gradient)
            throw std::runtime_error("exchange flow and gradient must differ");
        parameters_.length = parameters_.gradient == 0   ? box.x
                             : parameters_.gradient == 1 ? box.y
                                                         : box.z;
        parameters_.area = double(box.x) * box.y * box.z / parameters_.length;
        parameters_.identity =
            "atomic_centered_exchange_v2;flow=" +
            std::to_string(parameters_.flow) +
            ";gradient=" + std::to_string(parameters_.gradient) +
            ";bins=" + std::to_string(parameters_.bins) +
            ";interval=" + std::to_string(parameters_.exchange_interval) +
            ";box=" + Precise_Number(box.x) + "," + Precise_Number(box.y) +
            "," + Precise_Number(box.z) + ";" + integrator;
        atom_count_ = n;
        masses_.assign(mass, mass + n);
        for (double m : masses_)
            if (!SpongeFloat::Is_Finite(m) || m <= 0)
                throw std::runtime_error(
                    "exchange requires positive finite masses");
        ids_.resize(n);
        coordinates_.resize(n);
        exchange_velocity_.resize(n);
        exchange_force_.resize(n);
        exchange_mass_.resize(n);
        exchange_position_.resize(n);
        exchange_centered_.resize(n);
        ledger_.Initialize(parameters_.identity, dt, time);
        parameters_.enabled = true;
    }
    catch (const std::exception& e)
    {
        c->Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                              "MomentumExchange::Initial", e.what());
    }
}
void MomentumExchangeController::Begin_Step(double t)
{
    if (!enabled()) return;
    if (pending_ || !Same_Time(t, ledger_.time))
        throw std::runtime_error("exchange begin time mismatch");
    pending_ = true;
}
void MomentumExchangeController::Complete_Step(double t)
{
    if (!enabled()) return;
    if (!pending_)
        throw std::runtime_error("exchange completion without begin");
    ledger_.Complete(t);
    pending_ = false;
}
void MomentumExchangeController::Record(const MomentumExchange& e, double t,
                                        int step)
{
    if (!pending_ || !Due(step))
        throw std::runtime_error("exchange record outside scheduled stage");
    ledger_.Record(e, t, step);
}
std::map<std::string, std::string> MomentumExchangeController::Export_Restart()
    const
{
    if (!enabled()) return {};
    if (pending_) throw std::runtime_error("unfinished exchange step");
    return State_Prefix(ledger_.Export(), "exchange_");
}
void MomentumExchangeController::Restore_Restart(
    const std::map<std::string, std::string>& all, double t)
{
    auto s = State_Select(all, "exchange_");
    if (!enabled() && s.empty()) return;
    if (!enabled() || s.empty())
        throw std::runtime_error(
            "exchange restart configuration missing or removed");
    ledger_.Restore(s, t);
}
MomentumExchange MomentumExchangeController::Apply(
    int count, const int* global_ids, const VECTOR* coordinates,
    VECTOR* velocity, const VECTOR* force, double dt)
{
    try
    {
        if (!parameters_.enabled || count != atom_count_ || !global_ids ||
            !coordinates || !velocity || !force ||
            !SpongeFloat::Is_Finite(dt) || dt <= 0)
            throw std::runtime_error("invalid RNEMD exchange arrays/time step");
        deviceMemcpy(ids_.data(), global_ids, count * sizeof(int),
                     deviceMemcpyDeviceToHost);
        deviceMemcpy(coordinates_.data(), coordinates, count * sizeof(VECTOR),
                     deviceMemcpyDeviceToHost);
        deviceMemcpy(exchange_velocity_.data(), velocity,
                     count * sizeof(VECTOR), deviceMemcpyDeviceToHost);
        deviceMemcpy(exchange_force_.data(), force, count * sizeof(VECTOR),
                     deviceMemcpyDeviceToHost);
        auto component = [](const VECTOR& v, int axis)
        { return axis == 0 ? v.x : (axis == 1 ? v.y : v.z); };
        for (int i = 0; i < count; ++i)
        {
            if (ids_[i] < 0 || ids_[i] >= atom_count_)
                throw std::runtime_error("invalid RNEMD owner ID");
            exchange_mass_[i] = masses_[ids_[i]];
            exchange_position_[i] =
                component(coordinates_[i], parameters_.gradient);
            exchange_centered_[i] =
                component(exchange_velocity_[i], parameters_.flow) +
                .5 * dt * component(exchange_force_[i], parameters_.flow) /
                    exchange_mass_[i];
        }
        auto event = Select_Exchange(exchange_mass_, exchange_position_,
                                     exchange_centered_, ids_,
                                     parameters_.length, parameters_.bins);
        if (event.status != ExchangeStatus::Swapped) return event;
        double impulses[2] = {};
        int n = 0;
        for (int i : {event.low, event.middle})
        {
            const double centered =
                i == event.low ? event.low_velocity : event.middle_velocity;
            const double half =
                component(exchange_velocity_[i], parameters_.flow) +
                (centered - exchange_centered_[i]);
            if (!SpongeFloat::Is_Finite(half) ||
                std::abs(half) > std::numeric_limits<float>::max())
                throw std::runtime_error(
                    "RNEMD velocity exceeds engine precision");
            const float written = static_cast<float>(half);
            const double increment =
                static_cast<double>(written) -
                component(exchange_velocity_[i], parameters_.flow);
            impulses[n++] = exchange_mass_[i] * increment;
            event.energy_error += .5 * exchange_mass_[i] * increment *
                                  (2 * exchange_centered_[i] + increment);
            if (parameters_.flow == 0)
                exchange_velocity_[i].x = written;
            else if (parameters_.flow == 1)
                exchange_velocity_[i].y = written;
            else
                exchange_velocity_[i].z = written;
        }
        // Do not apply an exchange whose direction is lost to float rounding.
        // Only the host scratch has changed; the next call gathers it again.
        if (impulses[0] >= 0 || impulses[1] <= 0)
        {
            event.status = ExchangeStatus::NoCandidate;
            return event;
        }
        event.momentum = impulses[1];
        event.momentum_error = impulses[0] + impulses[1];
        for (int i : {event.low, event.middle})
            deviceMemcpy(velocity + i, &exchange_velocity_[i], sizeof(VECTOR),
                         deviceMemcpyHostToDevice);
        return event;
    }
    catch (const std::exception& error)
    {
        controller_->Throw_SPONGE_Error(spongeErrorSimulationBreakDown,
                                        "MomentumExchangeController::Apply",
                                        error.what());
    }
    return {};
}

std::map<std::string, std::string> MomentumExchangeController::Units() const
{
    return {{"momentum", "Da nm/ps"},
            {"momentum_error", "Da nm/ps"},
            {"energy_error", "kcal/mol"},
            {"attempts", "1"},
            {"swaps", "1"},
            {"empty", "1"},
            {"no_candidate", "1"},
            {"area", "nm^2"},
            {"length", "nm"}};
}
std::map<std::string, double> MomentumExchangeController::Values() const
{
    const double conversion = CONSTANT_TIME_CONVERTION / 10.0;
    return {{"momentum", ledger_.momentum * conversion},
            {"momentum_error", ledger_.momentum_error * conversion},
            {"energy_error", ledger_.energy_error},
            {"attempts", static_cast<double>(ledger_.attempts)},
            {"swaps", static_cast<double>(ledger_.swaps)},
            {"empty", static_cast<double>(ledger_.empty)},
            {"no_candidate", static_cast<double>(ledger_.no_candidate)},
            {"area", parameters_.area / 100},
            {"length", parameters_.length / 10}};
}

}  // namespace SpongeDynamics

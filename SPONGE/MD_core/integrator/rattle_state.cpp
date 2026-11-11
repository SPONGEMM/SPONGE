#include "rattle_state.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <tuple>

#include "../../utils/float_classification.hpp"
#include "../energy_ledger.h"

namespace SpongeDynamics
{
namespace
{
float Native(double x, double box)
{
    if (!SpongeFloat::Is_Finite(x))
        throw std::runtime_error("nonfinite RATTLE position");
    double wrapped = std::fmod(x, box);
    if (wrapped < 0) wrapped += box;
    const float result = float(wrapped);
    return result >= box ? 0.f : result;
}
std::vector<double> Flatten(const std::vector<Vec3>& v)
{
    std::vector<double> out;
    for (const auto& row : v) out.insert(out.end(), row.begin(), row.end());
    return out;
}
std::vector<Vec3> Unflatten(const std::vector<double>& v, int n)
{
    if (v.size() != 3 * std::size_t(n))
        throw std::runtime_error("RATTLE restart dimensions");
    std::vector<Vec3> out(n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < 3; ++j)
        {
            if (!SpongeFloat::Is_Finite(v[3 * i + j]))
                throw std::runtime_error("nonfinite RATTLE restart");
            out[i][j] = v[3 * i + j];
        }
    return out;
}
}  // namespace

void ConstrainedNVEState::Initial(CONTROLLER* controller)
{
    controller_ = controller;
    selected_ = controller->Command_Choice("constrain_mode", "RATTLE", true);
}

void ConstrainedNVEState::Restore(const SpongeH5MD::RestartDynamicState& state)
{
    try
    {
        const auto config =
            state.integrator_state_text.find("constrained_nve_configuration");
        const bool present =
            config != state.integrator_state_text.end() ||
            state.integrator_double_states.count("constrained_nve_position") ||
            state.integrator_double_states.count("constrained_nve_velocity") ||
            state.integrator_double_states.count(
                "constrained_nve_drift_energy");
        if (!selected_ && !present) return;
        if (!selected_ || !present ||
            config == state.integrator_state_text.end() ||
            state.integrator_double_states.size() != 3 ||
            !state.integrator_double_states.count("constrained_nve_position") ||
            !state.integrator_double_states.count("constrained_nve_velocity") ||
            !state.integrator_double_states.count(
                "constrained_nve_drift_energy") ||
            !state.integrator_state_text.count("mode") ||
            state.integrator_state_text.at("mode") != "nve" ||
            !state.integrator_state_text.count("step") ||
            !state.integrator_state_text.count("time"))
            throw std::runtime_error(
                "dynamic restart requires matching RATTLE NVE state and clock; "
                "use structural load for a new experiment");
        restored_ = true;
        restored_identity_ = config->second;
        restored_values_ = state.integrator_double_states;
    }
    catch (const std::exception& e)
    {
        controller_->Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                                        "ConstrainedNVEState::Restore",
                                        e.what());
    }
}

void ConstrainedNVEState::Export(SpongeH5MD::RestartDynamicState* state) const
{
    if (!Active()) return;
    if (!state || ready_)
        throw std::runtime_error(
            "cannot checkpoint unfinished RATTLE NVE step");
    state->integrator_state_text["constrained_nve_configuration"] = identity_;
    state->integrator_double_states["constrained_nve_position"] = Flatten(q_);
    state->integrator_double_states["constrained_nve_velocity"] = Flatten(s_);
    state->integrator_double_states["constrained_nve_drift_energy"] = {
        drift_energy_};
}

void ConstrainedNVEState::Configure(int count, const float* mass,
                                    std::vector<DistanceConstraint> pairs,
                                    VECTOR box, VECTOR angles, double dt,
                                    double dt_ps, VECTOR* coordinates,
                                    VECTOR* velocity,
                                    double initial_thermal_energy)
{
    try
    {
        if (!selected_ || Active() || count < 2 || !mass || !coordinates ||
            !velocity || pairs.empty() ||
            !SpongeFloat::Is_Finite(initial_thermal_energy) ||
            initial_thermal_energy < 0 || !SpongeFloat::Is_Finite(box.x) ||
            box.x <= 0 || !SpongeFloat::Is_Finite(box.y) || box.y <= 0 ||
            !SpongeFloat::Is_Finite(box.z) || box.z <= 0 ||
            !SpongeFloat::Is_Finite(dt) || dt <= 0 ||
            !SpongeFloat::Is_Finite(dt_ps) || dt_ps <= 0 ||
            !SpongeFloat::Is_Finite(angles.x) ||
            !SpongeFloat::Is_Finite(angles.y) ||
            !SpongeFloat::Is_Finite(angles.z) ||
            std::abs(angles.x - 90) > 1e-4 || std::abs(angles.y - 90) > 1e-4 ||
            std::abs(angles.z - 90) > 1e-4)
            throw std::runtime_error(
                "RATTLE NVE requires active distances, positive dt and "
                "orthogonal box");
        count_ = count;
        dt_ = dt;
        box_ = {box.x, box.y, box.z};
        mass_.assign(mass, mass + count);
        for (auto& pair : pairs)
            if (pair.b < pair.a) std::swap(pair.a, pair.b);
        std::sort(pairs.begin(), pairs.end(),
                  [](const auto& a, const auto& b)
                  {
                      return std::tie(a.a, a.b, a.length) <
                             std::tie(b.a, b.b, b.length);
                  });
        identity_ =
            "rattle_nve_double_v3;BA_full_drift;staggered;tol=1e-12;max_sweeps="
            "256;raw_float_input_v1;output_preserves_native_v1;"
            "energy_virial_every_step_v1;"
            "native_periodic_double_v1;dt=" +
            Precise_Number(dt) + ";dt_ps=" + Precise_Number(dt_ps);
        for (double x : box_) identity_ += ";box=" + Precise_Number(x);
        for (double m : mass_) identity_ += ";m=" + Precise_Number(m);
        for (const auto& p : pairs)
            identity_ += ";c=" + std::to_string(p.a) + "," +
                         std::to_string(p.b) + "," + Precise_Number(p.length);
        plan_.reset(new ConstraintStressPlan(mass_, std::move(pairs)));
        q_.resize(count);
        s_.resize(count);
        force_.resize(count);
        local_positions_.resize(count);
        ids_.resize(count);
        host_q_.resize(count);
        host_s_.resize(count);
        host_force_.resize(count);
        host_mass_.resize(count);
        host_virial_.resize(count);
        deviceMemcpy(host_q_.data(), coordinates, count * sizeof(VECTOR),
                     deviceMemcpyDeviceToHost);
        deviceMemcpy(host_s_.data(), velocity, count * sizeof(VECTOR),
                     deviceMemcpyDeviceToHost);
        for (int i = 0; i < count; ++i)
        {
            q_[i] = {host_q_[i].x, host_q_[i].y, host_q_[i].z};
            s_[i] = {host_s_[i].x, host_s_[i].y, host_s_[i].z};
        }
        if (restored_)
        {
            if (initial_thermal_energy != 0)
                throw std::runtime_error(
                    "cannot regenerate RATTLE dynamic restart velocities");
            if (restored_identity_ != identity_)
                throw std::runtime_error(
                    "RATTLE NVE restart configuration mismatch");
            auto q = Unflatten(restored_values_.at("constrained_nve_position"),
                               count);
            auto s = Unflatten(restored_values_.at("constrained_nve_velocity"),
                               count);
            const auto& energy =
                restored_values_.at("constrained_nve_drift_energy");
            if (energy.size() != 1 || !SpongeFloat::Is_Finite(energy[0]))
                throw std::runtime_error("RATTLE restart drift energy");
            for (int i = 0; i < count; ++i)
                for (int j = 0; j < 3; ++j)
                {
                    double diff = Native(q[i][j], box_[j]) - q_[i][j];
                    diff -= box_[j] * std::round(diff / box_[j]);
                    if (std::abs(diff) > 2e-6 * box_[j] ||
                        float(s[i][j]) != s_[i][j])
                        throw std::runtime_error(
                            "RATTLE restart double/structural state mismatch");
                }
            q_ = std::move(q);
            s_ = std::move(s);
            drift_energy_ = energy[0];
        }
        else
        {
            q_ = plan_->Drift(q_, std::vector<Vec3>(count), 1., box_, 1e-4)
                     .position;
            const auto frame = plan_->Prepare(q_, box_, 2e-12);
            s_ = frame.Project(s_);
            if (initial_thermal_energy > 0)
            {
                // Preserve initial_velocity's fixed-energy initialization,
                // but normalize AFTER the actual double tangent projection.
                Vec3 com{};
                double total_mass = 0, kinetic = 0;
                for (int i = 0; i < count; ++i)
                {
                    total_mass += mass_[i];
                    for (int j = 0; j < 3; ++j) com[j] += mass_[i] * s_[i][j];
                }
                for (double& x : com) x /= total_mass;
                for (int i = 0; i < count; ++i)
                    for (int j = 0; j < 3; ++j)
                    {
                        s_[i][j] -= com[j];
                        kinetic += .5 * mass_[i] * s_[i][j] * s_[i][j];
                    }
                if (!SpongeFloat::Is_Finite(kinetic) || kinetic <= 0)
                    throw std::runtime_error(
                        "RATTLE generated velocity has no thermal kinetic "
                        "energy");
                const double target = .5 * (frame.Degrees_Of_Freedom() - 3) *
                                      initial_thermal_energy;
                const double scale = std::sqrt(target / kinetic);
                for (auto& row : s_)
                    for (double& x : row) x *= scale;
            }
        }
        // Validate the restored state without changing its bits.
        const auto projected = plan_->Prepare(q_, box_, 2e-12).Project(s_);
        double residual = 0, norm = 0;
        for (int i = 0; i < count; ++i)
            for (int j = 0; j < 3; ++j)
            {
                residual += mass_[i] * std::pow(projected[i][j] - s_[i][j], 2);
                norm += mass_[i] * s_[i][j] * s_[i][j];
            }
        if (!SpongeFloat::Is_Finite(norm) ||
            residual > 1e-20 * std::max(1., norm))
            throw std::runtime_error("RATTLE restart velocity is not tangent");
        Publish(coordinates, velocity, true);
        restored_values_.clear();
        restored_identity_.clear();
        controller_->printf(
            "    RATTLE NVE: canonical double q/s, native float forces; no "
            "thermostat\n");
    }
    catch (const std::exception& e)
    {
        controller_->Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                                        "ConstrainedNVEState::Configure",
                                        e.what());
    }
}

void ConstrainedNVEState::Publish(VECTOR* coordinates, VECTOR* velocity,
                                  bool global)
{
    for (int i = 0; i < count_; ++i)
    {
        const int id = global ? i : ids_[i];
        const auto& q = q_[id];
        const auto& s = s_[id];
        local_positions_[i] = q;
        for (double x : s)
            if (!SpongeFloat::Is_Finite(float(x)))
                throw std::runtime_error("RATTLE native velocity overflow");
        host_q_[i] = {Native(q[0], box_[0]), Native(q[1], box_[1]),
                      Native(q[2], box_[2])};
        host_s_[i] = {float(s[0]), float(s[1]), float(s[2])};
    }
    deviceMemcpy(coordinates, host_q_.data(), count_ * sizeof(VECTOR),
                 deviceMemcpyHostToDevice);
    deviceMemcpy(velocity, host_s_.data(), count_ * sizeof(VECTOR),
                 deviceMemcpyHostToDevice);
}

void ConstrainedNVEState::Observe(int count, const int* ids,
                                  const VECTOR* coordinates,
                                  const VECTOR* velocity, const VECTOR* force,
                                  const float* mass, const LTMatrix3* virial,
                                  double dt)
{
    if (!Active()) return;
    try
    {
        if (ready_ || count != count_ || dt != dt_ || !ids || !coordinates ||
            !velocity || !force || !mass || !virial)
            throw std::runtime_error("invalid RATTLE NVE force-time state");
        deviceMemcpy(ids_.data(), ids, count * sizeof(int),
                     deviceMemcpyDeviceToHost);
        deviceMemcpy(host_q_.data(), coordinates, count * sizeof(VECTOR),
                     deviceMemcpyDeviceToHost);
        deviceMemcpy(host_s_.data(), velocity, count * sizeof(VECTOR),
                     deviceMemcpyDeviceToHost);
        deviceMemcpy(host_force_.data(), force, count * sizeof(VECTOR),
                     deviceMemcpyDeviceToHost);
        deviceMemcpy(host_mass_.data(), mass, count * sizeof(float),
                     deviceMemcpyDeviceToHost);
        deviceMemcpy(host_virial_.data(), virial, count * sizeof(LTMatrix3),
                     deviceMemcpyDeviceToHost);
        std::vector<bool> seen(count, false);
        Stress6 potential{};
        for (int i = 0; i < count; ++i)
        {
            const int id = ids_[i];
            if (id < 0 || id >= count || seen[id] || host_mass_[i] != mass_[id])
                throw std::runtime_error(
                    "RATTLE owner permutation/mass mismatch");
            seen[id] = true;
            const Vec3 native_q{host_q_[i].x, host_q_[i].y, host_q_[i].z},
                native_s{host_s_[i].x, host_s_[i].y, host_s_[i].z};
            for (int j = 0; j < 3; ++j)
            {
                double diff = native_q[j] - Native(q_[id][j], box_[j]);
                diff -= box_[j] * std::round(diff / box_[j]);
                if (!SpongeFloat::Is_Finite(diff) ||
                    std::abs(diff) > 2e-6 * box_[j] ||
                    native_s[j] != float(s_[id][j]))
                    throw std::runtime_error(
                        "native state changed outside RATTLE NVE integrator");
            }
            force_[id] = {host_force_[i].x, host_force_[i].y, host_force_[i].z};
            local_positions_[i] = q_[id];
            const auto& w = host_virial_[i];
            potential[0] += w.a11;
            potential[1] += w.a22;
            potential[2] += w.a33;
            potential[3] += .5 * w.a21;
            potential[4] += .5 * w.a31;
            potential[5] += .5 * w.a32;
        }
        observation_ =
            plan_->Evaluate(q_, s_, force_, potential, dt_, box_, 2e-12);
        ready_ = true;
    }
    catch (const std::exception& e)
    {
        controller_->Throw_SPONGE_Error(spongeErrorSimulationBreakDown,
                                        "ConstrainedNVEState::Observe",
                                        e.what());
    }
}

void ConstrainedNVEState::Advance(int count, VECTOR* coordinates,
                                  VECTOR* velocity)
{
    try
    {
        if (!Active() || !ready_ || count != count_ || !coordinates ||
            !velocity)
            throw std::runtime_error("RATTLE advance without force-time state");
        auto step =
            Advance_Constrained_NVE(*plan_, q_, s_, force_, box_, dt_, 2e-12);
        q_ = std::move(step.position);
        s_ = std::move(step.staggered_velocity);
        drift_energy_ += step.drift_kinetic_change;
        if (!SpongeFloat::Is_Finite(drift_energy_))
            throw std::runtime_error("RATTLE drift energy overflow");
        Publish(coordinates, velocity, false);
        ready_ = false;
    }
    catch (const std::exception& e)
    {
        controller_->Throw_SPONGE_Error(spongeErrorSimulationBreakDown,
                                        "ConstrainedNVEState::Advance",
                                        e.what());
    }
}
}  // namespace SpongeDynamics

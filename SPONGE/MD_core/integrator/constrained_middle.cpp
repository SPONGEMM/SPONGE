#include "constrained_middle.h"

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
float Native_Periodic_Position(double x, double box)
{
    if (!SpongeFloat::Is_Finite(x))
        throw std::runtime_error("nonfinite constrained native position");
    // Reduce in double before casting: unwrapped integration/work coordinates
    // can span many boxes, but force-coordinate precision must remain
    // box-sized.
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
        throw std::runtime_error("constrained restart state size mismatch");
    std::vector<Vec3> out(n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < 3; ++j)
        {
            if (!SpongeFloat::Is_Finite(v[3 * i + j]))
                throw std::runtime_error("nonfinite constrained restart state");
            out[i][j] = v[3 * i + j];
        }
    return out;
}
}  // namespace

void ConstrainedMiddle::Initial(CONTROLLER* c, int count,
                                const SpatialMode& mode)
{
    Clear();
    controller_ = c;
    count_ = count;
    parameters_ = mode;
    if (!mode.enabled) return;
    h_crd_.resize(count);
    h_vel_.resize(count);
    h_force_.resize(count);
    h_mass_.resize(count);
    positions_.resize(count);
    velocities_.resize(count);
    masses_.resize(count);
}
void ConstrainedMiddle::Load_Host(int count, const VECTOR* crd,
                                  const VECTOR* vel, const float* mass)
{
    if (count != count_ || !crd || !vel || !mass)
        throw std::runtime_error("invalid flow owner arrays");
    deviceMemcpy(h_crd_.data(), crd, count * sizeof(VECTOR),
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(h_vel_.data(), vel, count * sizeof(VECTOR),
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(h_mass_.data(), mass, count * sizeof(float),
                 deviceMemcpyDeviceToHost);
    for (int i = 0; i < count; ++i)
    {
        positions_[i] = {h_crd_[i].x, h_crd_[i].y, h_crd_[i].z};
        velocities_[i] = {h_vel_[i].x, h_vel_[i].y, h_vel_[i].z};
        masses_[i] = h_mass_[i];
    }
}

void ConstrainedMiddle::Restore_Constrained_State(
    const SpongeH5MD::RestartDynamicState& state)
{
    try
    {
        const auto config =
            state.integrator_state_text.find("constrained_flow_configuration");
        const bool present =
            config != state.integrator_state_text.end() ||
            state.integrator_double_states.count("constrained_flow_position") ||
            state.integrator_double_states.count("constrained_flow_velocity") ||
            state.integrator_double_states.count(
                "constrained_flow_drift_energy");
        const bool expected =
            enabled() && controller_->Command_Exist("constrain_mode");
        if (!expected && !present) return;
        if (!expected || !present ||
            config == state.integrator_state_text.end() ||
            state.integrator_double_states.size() != 3 ||
            !state.integrator_double_states.count(
                "constrained_flow_position") ||
            !state.integrator_double_states.count(
                "constrained_flow_velocity") ||
            !state.integrator_double_states.count(
                "constrained_flow_drift_energy"))
            throw std::runtime_error(
                "dynamic restart requires matching constrained flow state; use "
                "structural load for a new experiment");
        constrained_restart_ = true;
        restart_constraint_identity_ = config->second;
        restart_constraint_values_ = state.integrator_double_states;
    }
    catch (const std::exception& e)
    {
        controller_->Throw_SPONGE_Error(
            spongeErrorValueErrorCommand,
            "ConstrainedMiddle::Restore_Constrained_State", e.what());
    }
}

void ConstrainedMiddle::Export_Constrained_State(
    SpongeH5MD::RestartDynamicState* state) const
{
    if (!Constrained()) return;
    if (!state || constrained_ready_)
        throw std::runtime_error(
            "cannot checkpoint unfinished constrained step");
    state->integrator_state_text["constrained_flow_configuration"] =
        constraint_identity_;
    state->integrator_double_states["constrained_flow_position"] =
        Flatten(constrained_positions_);
    state->integrator_double_states["constrained_flow_velocity"] =
        Flatten(constrained_velocities_);
    state->integrator_double_states["constrained_flow_drift_energy"] = {
        drift_energy_};
}

void ConstrainedMiddle::Configure_Constraints(
    const float* mass, std::vector<DistanceConstraint> pairs, VECTOR box,
    double dt, VECTOR* global_crd, VECTOR* global_vel)
{
    try
    {
        if (!enabled() || Constrained() || pairs.empty() || !mass ||
            !global_crd || !global_vel || !SpongeFloat::Is_Finite(dt) ||
            dt <= 0)
            throw std::runtime_error(
                "invalid constrained cosine initialization");
        constrained_box_ = {box.x, box.y, box.z};
        constrained_dt_ = dt;
        constrained_masses_.assign(mass, mass + count_);
        for (auto& p : pairs)
            if (p.b < p.a) std::swap(p.a, p.b);
        std::sort(pairs.begin(), pairs.end(),
                  [](const auto& a, const auto& b)
                  {
                      return std::tie(a.a, a.b, a.length) <
                             std::tie(b.a, b.b, b.length);
                  });
        constraint_identity_ =
            "rattle_flow_double_v2;tol=1e-12;max_sweeps=256;global_philox;"
            "staggered;raw_float_input_v1;output_preserves_native_v1;"
            "energy_virial_every_step_v1;native_periodic_double_v1;" +
            parameters_.identity + ";dt=" + Precise_Number(dt);
        for (double x : constrained_box_)
            constraint_identity_ += ";box=" + Precise_Number(x);
        for (double m : constrained_masses_)
            constraint_identity_ += ";m=" + Precise_Number(m);
        for (const auto& p : pairs)
            constraint_identity_ += ";c=" + std::to_string(p.a) + "," +
                                    std::to_string(p.b) + "," +
                                    Precise_Number(p.length);
        constraint_plan_.reset(
            new ConstraintStressPlan(constrained_masses_, std::move(pairs)));
        deviceMemcpy(h_crd_.data(), global_crd, count_ * sizeof(VECTOR),
                     deviceMemcpyDeviceToHost);
        deviceMemcpy(h_vel_.data(), global_vel, count_ * sizeof(VECTOR),
                     deviceMemcpyDeviceToHost);
        constrained_positions_.resize(count_);
        constrained_velocities_.resize(count_);
        for (int i = 0; i < count_; ++i)
        {
            constrained_positions_[i] = {h_crd_[i].x, h_crd_[i].y, h_crd_[i].z};
            constrained_velocities_[i] = {h_vel_[i].x, h_vel_[i].y,
                                          h_vel_[i].z};
        }
        if (constrained_restart_)
        {
            if (restart_constraint_identity_ != constraint_identity_)
                throw std::runtime_error(
                    "constrained flow restart configuration mismatch");
            auto q = Unflatten(
                restart_constraint_values_.at("constrained_flow_position"),
                count_);
            auto s = Unflatten(
                restart_constraint_values_.at("constrained_flow_velocity"),
                count_);
            const auto& energy =
                restart_constraint_values_.at("constrained_flow_drift_energy");
            if (energy.size() != 1 || !SpongeFloat::Is_Finite(energy[0]))
                throw std::runtime_error(
                    "invalid constrained restart drift energy");
            for (int i = 0; i < count_; ++i)
                for (int j = 0; j < 3; ++j)
                {
                    double diff =
                        Native_Periodic_Position(q[i][j], constrained_box_[j]) -
                        constrained_positions_[i][j];
                    diff -= constrained_box_[j] *
                            std::round(diff / constrained_box_[j]);
                    if (std::abs(diff) > 2e-6 * constrained_box_[j] ||
                        float(s[i][j]) != constrained_velocities_[i][j])
                        throw std::runtime_error(
                            "constrained restart double/structural state "
                            "mismatch");
                }
            constrained_positions_ = std::move(q);
            constrained_velocities_ = std::move(s);
            drift_energy_ = energy[0];
        }
        else
        {
            // Input float coordinates need one initial position projection.
            // Its auxiliary drift velocity is discarded; input velocity keeps
            // the established incoming-staggered convention and is projected.
            constrained_positions_ =
                constraint_plan_
                    ->Drift(constrained_positions_, std::vector<Vec3>(count_),
                            1., constrained_box_, 1e-4)
                    .position;
            constrained_velocities_ =
                constraint_plan_
                    ->Prepare(constrained_positions_, constrained_box_, 2e-12)
                    .Project(constrained_velocities_);
        }
        std::vector<double> cosine(count_);
        for (int i = 0; i < count_; ++i)
            cosine[i] =
                std::cos(parameters_.wave_number *
                         constrained_positions_[i][parameters_.gradient]);
        ConstrainedFlowFrame frame(
            constraint_plan_->Prepare(constrained_positions_, constrained_box_,
                                      2e-12),
            cosine, parameters_.flow);
        frame.Apply_OU(constrained_velocities_, std::vector<Vec3>(count_), 1.,
                       0.);
        thermal_dof_ = frame.Thermal_Degrees_Of_Freedom();
        constrained_forces_.resize(count_);
        constrained_ids_.resize(count_);
        constrained_virial_.resize(count_);
        local_positions_.resize(count_);
        Publish_Constrained(global_crd, global_vel, true);
        restart_constraint_values_.clear();
        restart_constraint_identity_.clear();
        controller_->printf(
            "    Constrained cosine: host double RATTLE state; float force "
            "coordinates; canonical Philox; thermal DOF=%d\n",
            thermal_dof_);
    }
    catch (const std::exception& e)
    {
        controller_->Throw_SPONGE_Error(
            spongeErrorValueErrorCommand,
            "ConstrainedMiddle::Configure_Constraints", e.what());
    }
}

void ConstrainedMiddle::Publish_Constrained(VECTOR* crd, VECTOR* vel,
                                            bool global)
{
    for (int i = 0; i < count_; ++i)
    {
        const int id = global ? i : constrained_ids_[i];
        const auto& q = constrained_positions_[id];
        const auto& s = constrained_velocities_[id];
        h_crd_[i] = {Native_Periodic_Position(q[0], constrained_box_[0]),
                     Native_Periodic_Position(q[1], constrained_box_[1]),
                     Native_Periodic_Position(q[2], constrained_box_[2])};
        h_vel_[i] = {float(s[0]), float(s[1]), float(s[2])};
        for (int j = 0; j < 3; ++j)
            if (!SpongeFloat::Is_Finite(float(q[j])) ||
                !SpongeFloat::Is_Finite(float(s[j])))
                throw std::runtime_error("constrained native state overflow");
        if (!global) local_positions_[i] = q;
    }
    deviceMemcpy(crd, h_crd_.data(), count_ * sizeof(VECTOR),
                 deviceMemcpyHostToDevice);
    deviceMemcpy(vel, h_vel_.data(), count_ * sizeof(VECTOR),
                 deviceMemcpyHostToDevice);
}

void ConstrainedMiddle::Observe_Constrained(
    int count, const int* ids, const VECTOR* crd, const VECTOR* vel,
    const VECTOR* force, const float* mass, const LTMatrix3* virial, double dt)
{
    if (!Constrained()) return;
    try
    {
        if (constrained_ready_ || count != count_ || !ids || !virial ||
            !force || dt != constrained_dt_)
            throw std::runtime_error(
                "invalid constrained force-time observation");
        Load_Host(count, crd, vel, mass);
        deviceMemcpy(constrained_ids_.data(), ids, count * sizeof(int),
                     deviceMemcpyDeviceToHost);
        deviceMemcpy(h_force_.data(), force, count * sizeof(VECTOR),
                     deviceMemcpyDeviceToHost);
        deviceMemcpy(constrained_virial_.data(), virial,
                     count * sizeof(LTMatrix3), deviceMemcpyDeviceToHost);
        std::vector<bool> seen(count, false);
        Stress6 potential{};
        for (int i = 0; i < count; ++i)
        {
            const int id = constrained_ids_[i];
            if (id < 0 || id >= count || seen[id] ||
                h_mass_[i] != constrained_masses_[id])
                throw std::runtime_error(
                    "constrained owner permutation/mass mismatch");
            seen[id] = true;
            for (int j = 0; j < 3; ++j)
            {
                double diff =
                    positions_[i][j] -
                    Native_Periodic_Position(constrained_positions_[id][j],
                                             constrained_box_[j]);
                diff -= constrained_box_[j] *
                        std::round(diff / constrained_box_[j]);
                if (std::abs(diff) > 2e-6 * constrained_box_[j] ||
                    velocities_[i][j] != float(constrained_velocities_[id][j]))
                    throw std::runtime_error(
                        "native state changed outside constrained integrator");
            }
            constrained_forces_[id] = {h_force_[i].x, h_force_[i].y,
                                       h_force_[i].z};
            local_positions_[i] = constrained_positions_[id];
            const auto& w = constrained_virial_[i];
            potential[0] += w.a11;
            potential[1] += w.a22;
            potential[2] += w.a33;
            potential[3] += .5 * w.a21;
            potential[4] += .5 * w.a31;
            potential[5] += .5 * w.a32;
        }
        constrained_observation_ = constraint_plan_->Evaluate(
            constrained_positions_, constrained_velocities_,
            constrained_forces_, potential, dt, constrained_box_, 2e-12);
        sample_ =
            Fit_Cosine(constrained_masses_, constrained_positions_,
                       constrained_observation_.velocity, parameters_.flow,
                       parameters_.gradient, parameters_.wave_number);
        std::vector<double> cosine(count);
        for (int i = 0; i < count; ++i)
            cosine[i] =
                std::cos(parameters_.wave_number *
                         constrained_positions_[i][parameters_.gradient]);
        ConstrainedFlowFrame frame(
            constraint_plan_->Prepare(constrained_positions_, constrained_box_,
                                      2e-12),
            cosine, parameters_.flow);
        auto thermal = frame.Thermal(constrained_observation_.velocity);
        sample_.thermal_energy = 0;
        for (int i = 0; i < count; ++i)
            for (double x : thermal[i])
                sample_.thermal_energy += .5 * constrained_masses_[i] * x * x;
        constrained_ready_ = true;
    }
    catch (const std::exception& e)
    {
        controller_->Throw_SPONGE_Error(
            spongeErrorSimulationBreakDown,
            "ConstrainedMiddle::Observe_Constrained", e.what());
    }
}

double ConstrainedMiddle::Advance_Constraints(int count, VECTOR* crd,
                                              VECTOR* vel,
                                              const VECTOR* canonical_noise,
                                              double damping, double kBT)
{
    try
    {
        if (!Constrained() || !constrained_ready_ || count != count_ ||
            !canonical_noise)
            throw std::runtime_error(
                "constrained advance without force-time state");
        std::vector<VECTOR> noise(count);
        std::vector<Vec3> normal(count);
        deviceMemcpy(noise.data(), canonical_noise, count * sizeof(VECTOR),
                     deviceMemcpyDeviceToHost);
        for (int i = 0; i < count; ++i)
            normal[i] = {noise[i].x, noise[i].y, noise[i].z};
        auto step = Advance_Constrained_Flow(
            *constraint_plan_, constrained_positions_, constrained_velocities_,
            constrained_forces_, normal, constrained_box_, constrained_dt_,
            parameters_.flow, parameters_.gradient, parameters_.wave_number,
            damping, kBT, 2e-12);
        constrained_positions_ = std::move(step.position);
        constrained_velocities_ = std::move(step.staggered_velocity);
        drift_energy_ += step.drift_kinetic_change;
        if (!SpongeFloat::Is_Finite(drift_energy_))
            throw std::runtime_error("constrained drift energy overflow");
        Publish_Constrained(crd, vel, false);
        constrained_ready_ = false;
        return step.heat;
    }
    catch (const std::exception& e)
    {
        controller_->Throw_SPONGE_Error(
            spongeErrorSimulationBreakDown,
            "ConstrainedMiddle::Advance_Constraints", e.what());
    }
    return 0;
}
}  // namespace SpongeDynamics

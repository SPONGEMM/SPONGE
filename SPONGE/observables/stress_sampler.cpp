#include "stress_sampler.h"

#include <cmath>
#include <stdexcept>

#include "../utils/float_classification.hpp"
namespace SpongeDynamics
{
void StressSampler::Initial(CONTROLLER* c, int n, const SamplingPlan& p,
                            double dt)
{
    Clear();
    controller_ = c;
    enabled_ = p.stress;
    stress_dt_ps_ = dt;
    if (!enabled_) return;
    if (n < 2)
        throw std::runtime_error("stress requires positive COM-free DOF");
    stress_vel_.resize(n);
    stress_force_.resize(n);
    stress_mass_.resize(n);
    stress_virial_.resize(n);
    stress_masses_.resize(n);
    stress_velocities_.resize(n);
    stress_forces_.resize(n);
}
std::map<std::string, std::string> StressSampler::Export_Restart() const
{
    return constraint_plan_
               ? std::map<std::string, std::string>{{"constraint_configuration",
                                                     constraint_identity_}}
               : std::map<std::string, std::string>{};
}
void StressSampler::Restore_Restart(const std::map<std::string, std::string>& s,
                                    double)
{
    auto it = s.find("constraint_configuration");
    if (!enabled_ && it != s.end())
        throw std::runtime_error(
            "constraint force policy removed from restart");
    if (enabled_)
    {
        deferred_restart_ = true;
        restart_.clear();
        if (it != s.end()) restart_[it->first] = it->second;
    }
}
void StressSampler::Configure_Constraint_Stress(
    int count, const float* mass, std::vector<DistanceConstraint> pairs,
    VECTOR box, VECTOR angles, double dt_internal,
    const std::string& solver_identity)
{
    if (!enabled_) return;
    try
    {
        if (controller_->Command_Exist("constrain_mode"))
        {
            if (!controller_->Command_Choice("constrain_mode", "SETTLE") &&
                !controller_->Command_Choice("constrain_mode", "SHAKE") &&
                !controller_->Command_Choice("constrain_mode", "RATTLE"))
                throw std::runtime_error(
                    "constrained stress supports SETTLE/SHAKE/RATTLE only");
            if (pairs.empty())
                throw std::runtime_error(
                    "constrained stress requires active distance constraints");
            if (!SpongeFloat::Is_Finite(dt_internal) || dt_internal <= 0 ||
                !SpongeFloat::Is_Finite(angles.x) ||
                !SpongeFloat::Is_Finite(angles.y) ||
                !SpongeFloat::Is_Finite(angles.z) ||
                std::abs(angles.x - 90) > 1e-4 ||
                std::abs(angles.y - 90) > 1e-4 ||
                std::abs(angles.z - 90) > 1e-4)
                throw std::runtime_error(
                    "constrained stress requires positive dt and an orthogonal "
                    "fixed box");
            constraint_box_ = {box.x, box.y, box.z};
            for (double x : constraint_box_)
                if (!SpongeFloat::Is_Finite(x) || x <= 0)
                    throw std::runtime_error("invalid constrained stress box");
            constraint_dt_ = dt_internal;
            constraint_canonical_double_ =
                controller_->Command_Choice("constrain_mode", "RATTLE");
            constraint_length_tolerance_ =
                constraint_canonical_double_ ? 2e-12 : 1e-4;
            constraint_masses_.assign(mass, mass + count);
            constraint_count_ = pairs.size();
            constraint_solver_ = solver_identity;
            // Canonical effective values (including complete pair order), not
            // file names. The checkpoint already hashes this text state.
            constraint_identity_ =
                "instantaneous_tangent_v1|" + solver_identity +
                "|dt_ps=" + Precise_Number(stress_dt_ps_) +
                "|dt=" + Precise_Number(dt_internal) +
                "|tol=" + Precise_Number(constraint_length_tolerance_);
            for (double x : constraint_box_)
                constraint_identity_ += "|box=" + Precise_Number(x);
            for (double m : constraint_masses_)
                constraint_identity_ += "|m=" + Precise_Number(m);
            for (auto pair : pairs)
                constraint_identity_ += "|pair=" + std::to_string(pair.a) +
                                        "," + std::to_string(pair.b) + "," +
                                        Precise_Number(pair.length);
            constraint_plan_.reset(
                new ConstraintStressPlan(constraint_masses_, std::move(pairs)));
            stress_crd_.resize(count);
            stress_positions_.resize(count);
            stress_ids_.resize(count);
            controller_->printf(
                "        constrained stress: %d distances, %d thermal DOF; "
                "instantaneous observer, host path\n",
                constraint_count_, 3 * count - constraint_count_ - 3);
        }
        if (deferred_restart_ && restart_ != Export_Restart())
            throw std::runtime_error(
                "constrained stress restart configuration mismatch or missing "
                "identity; "
                "use structural load for a new experiment");
        restart_.clear();
        deferred_restart_ = false;
    }
    catch (const std::exception& e)
    {
        controller_->Throw_SPONGE_Error(
            spongeErrorValueErrorCommand,
            "StressSampler::Configure_Constraint_Stress", e.what());
    }
}
std::map<std::string, std::string> StressSampler::Constraint_Metadata() const
{
    if (!constraint_plan_) return {};
    return {{"constraint_observer", "instantaneous_tangent_v1"},
            {"constraint_count", std::to_string(constraint_count_)},
            {"thermal_dof", std::to_string(3 * constraint_masses_.size() -
                                           constraint_count_ - 3)},
            {"constraint_solver", constraint_solver_},
            {"constraint_length_tolerance",
             Precise_Number(constraint_length_tolerance_)}};
}
std::map<std::string, std::string> StressSampler::Constraint_Units() const
{
    if (!constraint_plan_) return {};
    std::map<std::string, std::string> units{
        {"constraint_length_residual", "1"},
        {"constraint_tangent_residual", "nm/ps"},
        {"constraint_acceleration_residual", "nm/ps^2"}};
    for (const char* suffix : {"xx", "yy", "zz", "xy", "xz", "yz"})
        units[std::string("constraint_w") + suffix] = "kcal/mol";
    return units;
}
std::map<std::string, double> StressSampler::Sample_Stress(
    int count, const VECTOR* velocity, const VECTOR* force, const float* mass,
    const LTMatrix3* virial, double dt_internal, double volume,
    const VECTOR* coordinates, const int* local_to_global, VECTOR box,
    const std::vector<Vec3>* canonical_positions,
    const std::vector<Vec3>* canonical_velocities)
{
    try
    {
        if (!enabled_ || !velocity || !force || !mass || !virial ||
            count != static_cast<int>(stress_mass_.size()))
            throw std::runtime_error("invalid stress owner count");
        const bool canonical = canonical_positions != nullptr;
        if ((constraint_canonical_double_ && !canonical) ||
            canonical != (canonical_velocities != nullptr) ||
            (canonical && (canonical_positions->size() != std::size_t(count) ||
                           canonical_velocities->size() != std::size_t(count))))
            throw std::runtime_error(
                "constrained stress canonical state missing or unexpected");
        deviceMemcpy(stress_vel_.data(), velocity, count * sizeof(VECTOR),
                     deviceMemcpyDeviceToHost);
        deviceMemcpy(stress_force_.data(), force, count * sizeof(VECTOR),
                     deviceMemcpyDeviceToHost);
        deviceMemcpy(stress_mass_.data(), mass, count * sizeof(float),
                     deviceMemcpyDeviceToHost);
        deviceMemcpy(stress_virial_.data(), virial, count * sizeof(LTMatrix3),
                     deviceMemcpyDeviceToHost);
        if (constraint_plan_)
        {
            if (!coordinates || !local_to_global ||
                dt_internal != constraint_dt_ ||
                Vec3{box.x, box.y, box.z} != constraint_box_ ||
                std::abs(volume / (constraint_box_[0] * constraint_box_[1] *
                                   constraint_box_[2]) -
                         1) > 1e-6)
                throw std::runtime_error(
                    "constrained stress geometry/dt changed");
            deviceMemcpy(stress_crd_.data(), coordinates,
                         count * sizeof(VECTOR), deviceMemcpyDeviceToHost);
            deviceMemcpy(stress_ids_.data(), local_to_global,
                         count * sizeof(int), deviceMemcpyDeviceToHost);
            std::vector<bool> seen(count, false);
            for (int i = 0; i < count; ++i)
            {
                const int id = stress_ids_[i];
                if (id < 0 || id >= count || seen[id])
                    throw std::runtime_error(
                        "invalid constrained stress owner permutation");
                seen[id] = true;
                if (stress_mass_[i] != constraint_masses_[id])
                    throw std::runtime_error(
                        "constrained stress owner mass mismatch");
                stress_positions_[id] = {stress_crd_[i].x, stress_crd_[i].y,
                                         stress_crd_[i].z};
            }
        }
        Stress6 potential{};
        for (int i = 0; i < count; ++i)
        {
            const int id = constraint_plan_ ? stress_ids_[i] : i;
            stress_masses_[id] = stress_mass_[i];
            stress_velocities_[id] = {stress_vel_[i].x, stress_vel_[i].y,
                                      stress_vel_[i].z};
            stress_forces_[id] = {stress_force_[i].x, stress_force_[i].y,
                                  stress_force_[i].z};
            const auto& w = stress_virial_[i];
            // Engine LTMatrix3 off diagonals store the symmetric sum.
            potential[0] += w.a11;
            potential[1] += w.a22;
            potential[2] += w.a33;
            potential[3] += 0.5 * w.a21;
            potential[4] += 0.5 * w.a31;
            potential[5] += 0.5 * w.a32;
        }
        EquilibriumStress state;
        int dof = 3 * count - 3;
        std::map<std::string, double> diagnostics;
        if (constraint_plan_)
        {
            if (canonical)
            {
                stress_positions_ = *canonical_positions;
                stress_velocities_ = *canonical_velocities;
            }
            const auto observed = constraint_plan_->Evaluate(
                stress_positions_, stress_velocities_, stress_forces_,
                potential, dt_internal, constraint_box_,
                constraint_length_tolerance_);
            state = observed.equilibrium;
            dof = observed.thermal_dof;
            diagnostics["constraint_length_residual"] =
                observed.max_relative_length_error;
            diagnostics["constraint_tangent_residual"] =
                observed.max_tangent_residual * CONSTANT_TIME_CONVERTION / 10;
            diagnostics["constraint_acceleration_residual"] =
                observed.max_acceleration_residual * CONSTANT_TIME_CONVERTION *
                CONSTANT_TIME_CONVERTION / 10;
            const char* suffix[6] = {"xx", "yy", "zz", "xy", "xz", "yz"};
            for (int j = 0; j < 6; ++j)
                diagnostics[std::string("constraint_w") + suffix[j]] =
                    observed.constraint_virial[j];
        }
        else
            state =
                Centered_Stress(stress_masses_, stress_velocities_,
                                stress_forces_, potential, dt_internal, volume);
        constexpr double pressure_to_pa = 4184.0 / 6.02214076e23 / 1e-30;
        std::map<std::string, double> values{
            {"temperature", 2 * state.kinetic_energy / (dof * CONSTANT_kB)},
            {"volume", volume / 1000.0}};
        values.insert(diagnostics.begin(), diagnostics.end());
        const char* axes[6] = {"pxx", "pyy", "pzz", "pxy", "pxz", "pyz"};
        for (int j = 0; j < 6; ++j)
            values[axes[j]] = state.pressure[j] * pressure_to_pa;
        for (int j = 0; j < 3; ++j)
            values[std::string("com_v") + "xyz"[j]] =
                state.com_velocity[j] * CONSTANT_TIME_CONVERTION / 10.0;
        return values;
    }
    catch (const std::exception& e)
    {
        controller_->Throw_SPONGE_Error(spongeErrorSimulationBreakDown,
                                        "StressSampler::Sample_Stress",
                                        e.what());
    }
    return {};
}

}  // namespace SpongeDynamics

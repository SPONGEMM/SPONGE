#include "velocity_bias.h"

#include <cmath>
#include <stdexcept>

#include "../../MD_core/energy_ledger.h"
#include "../../utils/control/component_options.hpp"
namespace SpongeDynamics
{
namespace
{
static __global__ void Shift_Bias(int count, const VECTOR* crd, VECTOR* vel,
                                  int flow, int gradient, double k,
                                  double mean_c, double amplitude, VECTOR mean,
                                  double sign)
{
    SIMPLE_DEVICE_FOR(i, count)
    {
        const double coordinate =
            gradient == 0 ? crd[i].x : (gradient == 1 ? crd[i].y : crd[i].z);
        const double mode = amplitude * (cos(k * coordinate) - mean_c);
        vel[i].x +=
            static_cast<float>(sign * (mean.x + (flow == 0 ? mode : 0)));
        vel[i].y +=
            static_cast<float>(sign * (mean.y + (flow == 1 ? mode : 0)));
        vel[i].z +=
            static_cast<float>(sign * (mean.z + (flow == 2 ? mode : 0)));
    }
}
}  // namespace
void VelocityBias::Initial(CONTROLLER* c, int count, const float* mass,
                           VECTOR box, VECTOR angles, bool periodic,
                           const std::string& identity)
{
    Clear();
    controller_ = c;
    try
    {
        Component_Keys(c, "velocity_bias",
                       {"mode", "flow", "gradient", "harmonic"});
        std::string kind = c->Command_Exist("velocity_bias_mode")
                               ? c->Command("velocity_bias_mode")
                               : "none";
        if (kind == "none") return;
        if (kind != "cosine" && kind != "com")
            throw std::runtime_error(
                "velocity_bias.mode must be none/com/cosine");
        if (count < 2)
            throw std::runtime_error(
                "velocity bias requires positive thermal DOF");
        parameters_.cosine = kind == "cosine";
        parameters_.flow = Option_Axis(c, "velocity_bias_flow", 0);
        parameters_.gradient = Option_Axis(c, "velocity_bias_gradient", 2);
        if (parameters_.cosine)
        {
            Orthogonal_Box(box, angles, periodic);
            if (parameters_.flow == parameters_.gradient)
                throw std::runtime_error("bias flow and gradient must differ");
            parameters_.length = parameters_.gradient == 0   ? box.x
                                 : parameters_.gradient == 1 ? box.y
                                                             : box.z;
            parameters_.wave_number =
                2 * 3.14159265358979323846 *
                Option_Integer(c, "velocity_bias_harmonic", 1, 1, 1000) /
                parameters_.length;
        }
        for (int i = 0; i < count; ++i)
            if (!SpongeFloat::Is_Finite(mass[i]) || mass[i] <= 0)
                throw std::runtime_error(
                    "velocity bias requires positive finite masses");
        parameters_.identity =
            "velocity_bias_v1;mode=" + kind +
            ";flow=" + std::to_string(parameters_.flow) +
            ";gradient=" + std::to_string(parameters_.gradient) +
            ";k=" + Precise_Number(parameters_.wave_number) + ";" + identity;
        count_ = count;
        thermal_dof_ = 3 * count - (parameters_.cosine ? 4 : 3);
        parameters_.enabled = true;
        h_crd_.resize(count);
        h_vel_.resize(count);
        h_mass_.resize(count);
        h_retained_.resize(count);
        positions_.resize(count);
        velocities_.resize(count);
        masses_.resize(count);
        Device_Malloc_Safely((void**)&retained_, count * sizeof(VECTOR));
        c->Step_Print_Initial("thermal_temperature", "%.12e");
        c->Step_Print_Initial("bias_amplitude", "%.12e");
        for (const char* axis : {"x", "y", "z"})
            c->Step_Print_Initial((std::string("bias_com_") + axis).c_str(),
                                  "%.12e");
    }
    catch (const std::exception& e)
    {
        c->Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                              "VelocityBias::Initial", e.what());
    }
}
void VelocityBias::Configure_Constraints(const float* mass,
                                         std::vector<DistanceConstraint> pairs,
                                         VECTOR box, const std::string& solver)
{
    if (!enabled() || !parameters_.cosine || !mass || pairs.empty())
        throw std::runtime_error(
            "constraint-aware bias requires a cosine mode and distance "
            "topology");
    std::vector<double> masses(count_);
    constraint_identity_ = "native_nhc_constraint_bias_v1;" + solver;
    for (double x : {double(box.x), double(box.y), double(box.z)})
        constraint_identity_ += ";box=" + Precise_Number(x);
    for (int i = 0; i < count_; ++i)
    {
        masses[i] = mass[i];
        constraint_identity_ += ";m=" + Precise_Number(masses[i]);
    }
    for (const auto& p : pairs)
        constraint_identity_ += ";p=" + std::to_string(p.a) + "," +
                                std::to_string(p.b) + "," +
                                Precise_Number(p.length);
    thermal_dof_ = 3 * count_ - int(pairs.size()) - 4;
    if (thermal_dof_ <= 0)
        throw std::runtime_error("nonpositive constrained thermal DOF");
    constraint_plan_.reset(
        new ConstraintStressPlan(std::move(masses), std::move(pairs)));
    constraint_box_ = {box.x, box.y, box.z};
    if (!expected_constraint_identity_.empty() &&
        expected_constraint_identity_ != constraint_identity_)
        throw std::runtime_error(
            "NHC constraint/thermostat restart configuration mismatch; use "
            "structural load");
}
void VelocityBias::Sample_Constrained(double energy, int dof)
{
    reported_temperature_ = 2 * energy / (dof * CONSTANT_kB);
}
void VelocityBias::Step_Print()
{
    if (enabled())
    {
        controller_->Step_Print("thermal_temperature", reported_temperature_);
        controller_->Step_Print(
            "bias_amplitude",
            sample_.amplitude * CONSTANT_TIME_CONVERTION / 10.);
        for (int i = 0; i < 3; ++i)
            controller_->Step_Print(
                (std::string("bias_com_") + "xyz"[i]).c_str(),
                sample_.mean_velocity[i] * CONSTANT_TIME_CONVERTION / 10.);
    }
}
void VelocityBias::Clear()
{
    if (retained_) deviceFree(retained_);
    retained_ = nullptr;
    parameters_ = {};
    state_ = {};
    sample_ = {};
    bias_ = {};
    count_ = 0;
    bias_removed_ = false;
    h_crd_.clear();
    h_vel_.clear();
    h_mass_.clear();
    h_retained_.clear();
    positions_.clear();
    velocities_.clear();
    masses_.clear();
    constraint_plan_.reset();
    constraint_identity_.clear();
    expected_constraint_identity_.clear();
    constraint_thermal_.clear();
    thermal_dof_ = 0;
}
void VelocityBias::Load_Host(int count, const VECTOR* crd, const VECTOR* vel,
                             const float* mass)
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
CosineFit VelocityBias::Fit(int count, const VECTOR* crd, const VECTOR* vel,
                            const float* mass)
{
    Load_Host(count, crd, vel, mass);
    if (parameters_.cosine)
    {
        auto fit =
            Fit_Cosine(masses_, positions_, velocities_, parameters_.flow,
                       parameters_.gradient, parameters_.wave_number);
        if (Constrained())
        {
            std::vector<double> cosine(count);
            for (int i = 0; i < count; ++i)
                cosine[i] = std::cos(parameters_.wave_number *
                                     positions_[i][parameters_.gradient]);
            ConstrainedFlowFrame frame(
                constraint_plan_->Prepare(positions_, constraint_box_, 1e-4),
                cosine, parameters_.flow);
            constraint_thermal_ = frame.Thermal(velocities_);
            thermal_dof_ = frame.Thermal_Degrees_Of_Freedom();
            fit.thermal_energy = 0;
            for (int i = 0; i < count; ++i)
                for (double v : constraint_thermal_[i])
                    fit.thermal_energy += .5 * masses_[i] * v * v;
        }
        return fit;
    }
    CosineFit fit;
    for (int i = 0; i < count; ++i)
    {
        if (!SpongeFloat::Is_Finite(masses_[i]) || masses_[i] <= 0)
            throw std::runtime_error("bias requires positive masses");
        fit.mass += masses_[i];
        for (int j = 0; j < 3; ++j)
        {
            if (!SpongeFloat::Is_Finite(velocities_[i][j]))
                throw std::runtime_error("nonfinite thermal velocity");
            fit.mean_velocity[j] += masses_[i] * velocities_[i][j];
            fit.total_energy +=
                .5 * masses_[i] * velocities_[i][j] * velocities_[i][j];
        }
    }
    for (double& x : fit.mean_velocity) x /= fit.mass;
    for (int i = 0; i < count; ++i)
        for (int j = 0; j < 3; ++j)
        {
            double v = velocities_[i][j] - fit.mean_velocity[j];
            fit.thermal_energy += .5 * masses_[i] * v * v;
        }
    return fit;
}
ThermalState VelocityBias::Prepare_State(int count, const VECTOR* crd,
                                         const VECTOR* vel, const float* mass)
{
    thermal_fit_ = Fit(count, crd, vel, mass);
    state_ = {thermal_fit_.thermal_energy,
              2 * thermal_fit_.thermal_energy / (thermal_dof_ * CONSTANT_kB),
              thermal_dof_, "thermostat_substep"};
    for (int i = 0; i < count; ++i)
    {
        if (Constrained())
        {
            // Only the tangent thermal part receives friction. Keep both the
            // collective mode and the stored SHAKE velocity's normal part;
            // the existing constraint integrator owns the latter correction.
            h_retained_[i] = {
                float(velocities_[i][0] - constraint_thermal_[i][0]),
                float(velocities_[i][1] - constraint_thermal_[i][1]),
                float(velocities_[i][2] - constraint_thermal_[i][2])};
            continue;
        }
        double mode = parameters_.cosine
                          ? thermal_fit_.amplitude *
                                (std::cos(parameters_.wave_number *
                                          positions_[i][parameters_.gradient]) -
                                 thermal_fit_.mean_cosine)
                          : 0;
        h_retained_[i] = {float(thermal_fit_.mean_velocity[0] +
                                (parameters_.flow == 0 ? mode : 0)),
                          float(thermal_fit_.mean_velocity[1] +
                                (parameters_.flow == 1 ? mode : 0)),
                          float(thermal_fit_.mean_velocity[2] +
                                (parameters_.flow == 2 ? mode : 0))};
    }
    deviceMemcpy(retained_, h_retained_.data(), count * sizeof(VECTOR),
                 deviceMemcpyHostToDevice);
    return state_;
}
void VelocityBias::Shift(int count, const VECTOR* crd, VECTOR* vel,
                         const CosineFit& fit, double sign)
{
    VECTOR mean{static_cast<float>(fit.mean_velocity[0]),
                static_cast<float>(fit.mean_velocity[1]),
                static_cast<float>(fit.mean_velocity[2])};
    Launch_Device_Kernel(Shift_Bias, (count + 127) / 128, 128, 0, nullptr,
                         count, crd, vel, parameters_.flow,
                         parameters_.gradient, parameters_.wave_number,
                         fit.mean_cosine, fit.amplitude, mean, sign);
}
void VelocityBias::Sample(int count, const VECTOR* crd, const VECTOR* vel,
                          const float* mass)
{
    if (!enabled()) return;
    try
    {
        sample_ = Fit(count, crd, vel, mass);
        reported_temperature_ =
            2 * sample_.thermal_energy / (thermal_dof_ * CONSTANT_kB);
    }
    catch (const std::exception& e)
    {
        controller_->Throw_SPONGE_Error(spongeErrorSimulationBreakDown,
                                        "VelocityBias::Sample", e.what());
    }
}
void VelocityBias::Remove_Bias(int count, const VECTOR* crd, VECTOR* vel,
                               const float* mass)
{
    try
    {
        if (bias_removed_)
            throw std::runtime_error("flow bias already removed");
        Prepare_State(count, crd, vel, mass);
        bias_ = thermal_fit_;
        Shift(count, crd, vel, bias_, -1);
        bias_removed_ = true;
    }
    catch (const std::exception& e)
    {
        controller_->Throw_SPONGE_Error(spongeErrorSimulationBreakDown,
                                        "VelocityBias::Remove_Bias", e.what());
    }
}
void VelocityBias::Restore_Bias(int count, const VECTOR* crd, VECTOR* vel,
                                const float* mass)
{
    try
    {
        if (!bias_removed_)
            throw std::runtime_error("flow bias restore without remove");
        const auto noise_bias = Fit(count, crd, vel, mass);
        Shift(count, crd, vel, noise_bias, -1);
        Shift(count, crd, vel, bias_, 1);
        bias_removed_ = false;
    }
    catch (const std::exception& e)
    {
        controller_->Throw_SPONGE_Error(spongeErrorSimulationBreakDown,
                                        "VelocityBias::Restore_Bias", e.what());
    }
}

}  // namespace SpongeDynamics

#include "profile_sampler.h"

#include <cmath>
#include <stdexcept>

#include "../constrain/thermal_frame.h"
#include "../utils/float_classification.hpp"
namespace SpongeDynamics
{
void ProfileSampler::Initial(CONTROLLER* c, int count, const float* mass,
                             VECTOR box, const SamplingPlan& p,
                             bool constrained,
                             const ConstraintStressPlan* constraint_plan)
{
    Clear();
    controller_ = c;
    parameters_ = p.mode;
    count_ = count;
    constrained_ = constrained;
    constraint_plan_ = constraint_plan;
    constraint_box_ = {box.x, box.y, box.z};
    if (!enabled()) return;
    if (count < 2)
        throw std::runtime_error("profile requires positive thermal DOF");
    double total = 0;
    for (int i = 0; i < count; ++i)
    {
        if (!SpongeFloat::Is_Finite(mass[i]) || mass[i] <= 0)
            throw std::runtime_error("profile requires positive masses");
        total += mass[i];
    }
    density_ = total / (double(box.x) * box.y * box.z) * 1.66053906660;
    thermal_dof_ = 3 * count - 4;
    h_crd_.resize(count);
    h_vel_.resize(count);
    h_force_.resize(count);
    h_mass_.resize(count);
    positions_.resize(count);
    velocities_.resize(count);
    masses_.resize(count);
}
std::map<std::string, double> ProfileSampler::Sample(int n, const VECTOR* q,
                                                     const VECTOR* v,
                                                     const VECTOR* f,
                                                     const float* m, double dt,
                                                     PhysicalStateView view)
{
    view_ = view;
    if (constrained_)
    {
        if (!view.active())
            throw std::runtime_error("canonical profile view missing");
        thermal_dof_ = view.thermal_dof;
    }
    return parameters_.cosine ? Cosine_Profile(n, q, v, f, m, dt)
                              : Profile(n, q, v, f, m, dt);
}
std::map<std::string, std::string> ProfileSampler::Profile_Units() const
{
    std::map<std::string, std::string> units{{"kinetic_temperature", "K"},
                                             {"thermal_temperature", "K"}};
    for (int i = 0; i < parameters_.bins; ++i)
    {
        const auto prefix = "bin_" + std::to_string(i) + "_";
        units[prefix + "count"] = "1";
        units[prefix + "mass"] = "Da";
        units[prefix + "momentum"] = "Da nm/ps";
    }
    return units;
}
std::map<std::string, double> ProfileSampler::Profile(
    int count, const VECTOR* crd, const VECTOR* vel, const VECTOR* force,
    const float* mass, double dt)
{
    try
    {
        if (!parameters_.enabled || count != count_ || !crd || !vel || !force ||
            !mass || !SpongeFloat::Is_Finite(dt) || dt < 0)
            throw std::runtime_error("invalid RNEMD profile arrays/time step");
        deviceMemcpy(h_crd_.data(), crd, count * sizeof(VECTOR),
                     deviceMemcpyDeviceToHost);
        deviceMemcpy(h_vel_.data(), vel, count * sizeof(VECTOR),
                     deviceMemcpyDeviceToHost);
        deviceMemcpy(h_force_.data(), force, count * sizeof(VECTOR),
                     deviceMemcpyDeviceToHost);
        deviceMemcpy(h_mass_.data(), mass, count * sizeof(float),
                     deviceMemcpyDeviceToHost);
        std::vector<double> counts(parameters_.bins), masses(parameters_.bins),
            momenta(parameters_.bins);
        double total_mass = 0;
        Vec3 total_momentum{};
        std::vector<Vec3> centered_velocity(count);
        std::vector<int> bins(count);
        for (int i = 0; i < count; ++i)
        {
            const double m = h_mass_[i];
            if (!SpongeFloat::Is_Finite(m) || m <= 0)
                throw std::runtime_error("invalid RNEMD profile mass");
            const double pos[3] = {h_crd_[i].x, h_crd_[i].y, h_crd_[i].z};
            const double v[3] = {h_vel_[i].x, h_vel_[i].y, h_vel_[i].z};
            const double f[3] = {h_force_[i].x, h_force_[i].y, h_force_[i].z};
            const int bin = Periodic_Bin(pos[parameters_.gradient],
                                         parameters_.length, parameters_.bins);
            bins[i] = bin;
            ++counts[bin];
            masses[bin] += m;
            total_mass += m;
            for (int j = 0; j < 3; ++j)
            {
                const double centered = v[j] + .5 * dt * f[j] / m;
                if (!SpongeFloat::Is_Finite(centered))
                    throw std::runtime_error(
                        "nonfinite RNEMD profile velocity");
                total_momentum[j] += m * centered;
                centered_velocity[i][j] = centered;
                if (j == parameters_.flow) momenta[bin] += m * centered;
            }
        }
        // Sum residual squares instead of subtracting large bulk energies.
        double kinetic = 0, thermal = 0;
        int occupied = 0;
        for (int i = 0; i < count; ++i)
            for (int j = 0; j < 3; ++j)
            {
                const double residual =
                    centered_velocity[i][j] - total_momentum[j] / total_mass;
                kinetic += h_mass_[i] * residual * residual;
                const double bias = j == parameters_.flow
                                        ? momenta[bins[i]] / masses[bins[i]]
                                        : total_momentum[j] / total_mass;
                const double thermal_residual = centered_velocity[i][j] - bias;
                thermal += h_mass_[i] * thermal_residual * thermal_residual;
            }
        std::map<std::string, double> values;
        for (int i = 0; i < parameters_.bins; ++i)
        {
            if (masses[i] > 0) ++occupied;
            const auto prefix = "bin_" + std::to_string(i) + "_";
            values[prefix + "count"] = counts[i];
            values[prefix + "mass"] = masses[i];
            values[prefix + "momentum"] =
                momenta[i] * CONSTANT_TIME_CONVERTION / 10;
        }
        values["kinetic_temperature"] =
            kinetic / ((3 * count - 3) * CONSTANT_kB);
        values["thermal_temperature"] =
            thermal / ((3 * count - occupied - 2) * CONSTANT_kB);
        return values;
    }
    catch (const std::exception& error)
    {
        controller_->Throw_SPONGE_Error(spongeErrorSimulationBreakDown,
                                        "ProfileSampler::Profile",
                                        error.what());
    }
    return {};
}
void ProfileSampler::Load_Host(int count, const VECTOR* crd, const VECTOR* vel,
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
std::map<std::string, std::string> ProfileSampler::Cosine_Units() const
{
    std::map<std::string, std::string> units{
        {"velocity_fourier", "nm/ps"}, {"velocity_com_fourier", "nm/ps"},
        {"velocity_fit", "nm/ps"},     {"mean_cosine", "1"},
        {"variance_cosine", "1"},      {"thermal_temperature", "K"},
        {"com_vx", "nm/ps"},           {"com_vy", "nm/ps"},
        {"com_vz", "nm/ps"},           {"density", "g/cm^3"},
        {"wave_number", "nm^-1"},      {"length", "nm"}};
    for (int i = 0; i < parameters_.bins; ++i)
    {
        const auto prefix = "bin_" + std::to_string(i) + "_";
        units[prefix + "count"] = "1";
        units[prefix + "mass"] = "Da";
        units[prefix + "momentum"] = "Da nm/ps";
        units[prefix + "cosine_mass"] = "Da";
        units[prefix + "sine_mass"] = "Da";
    }
    if (Constrained())
    {
        for (const char* name : {"pxx", "pyy", "pzz", "pxy", "pxz", "pyz"})
            units[name] = "Pa";
        units["constraint_drift_energy"] = "kcal/mol";
        units["constraint_length_residual"] = "1";
        units["constraint_tangent_residual"] = "nm/ps";
        units["kinetic_temperature"] = "K";
    }
    return units;
}
std::map<std::string, double> ProfileSampler::Cosine_Profile(
    int count, const VECTOR* crd, const VECTOR* vel, const VECTOR* force,
    const float* mass, double dt)
{
    try
    {
        if (!enabled() || !force || !SpongeFloat::Is_Finite(dt) || dt < 0)
            throw std::runtime_error("invalid cosine profile state/time step");
        CosineFit fit;
        if (Constrained())
        {
            if (!view_.ready || dt != view_.dt || count != count_)
                throw std::runtime_error(
                    "constrained profile without force-time observation");
            positions_ = *view_.position;
            velocities_ = *view_.velocity;
            masses_ = *view_.mass;
            fit = *view_.mode_fit;
        }
        else
        {
            Load_Host(count, crd, vel, mass);
            deviceMemcpy(h_force_.data(), force, count * sizeof(VECTOR),
                         deviceMemcpyDeviceToHost);
            // At q_n the stored velocity s_n is the incoming staggered
            // variable. u_n = s_n + h F(q_n)/(2m) conjugates kick-drift (or
            // kick-AOA) to velocity Verlet (or B/2-A/2-O-A/2-B/2). Read only:
            // no extra OU.
            for (int i = 0; i < count; ++i)
            {
                if (!SpongeFloat::Is_Finite(masses_[i]) || masses_[i] <= 0)
                    throw std::runtime_error("invalid cosine profile mass");
                const double f[3] = {h_force_[i].x, h_force_[i].y,
                                     h_force_[i].z};
                for (int j = 0; j < 3; ++j)
                    velocities_[i][j] += .5 * dt * f[j] / masses_[i];
            }
            fit = Fit_Cosine(masses_, positions_, velocities_, parameters_.flow,
                             parameters_.gradient, parameters_.wave_number);
            if (constraint_plan_)
            {
                if (dt != 0)
                    throw std::runtime_error(
                        "native NHC constraint profile requires stored "
                        "velocity");
                std::vector<double> cosine(count);
                for (int i = 0; i < count; ++i)
                    cosine[i] = std::cos(parameters_.wave_number *
                                         positions_[i][parameters_.gradient]);
                ConstrainedFlowFrame frame(
                    constraint_plan_->Prepare(positions_, constraint_box_,
                                              1e-4),
                    cosine, parameters_.flow);
                const auto thermal = frame.Thermal(velocities_);
                thermal_dof_ = frame.Thermal_Degrees_Of_Freedom();
                fit.thermal_energy = 0;
                for (int i = 0; i < count; ++i)
                    for (double v : thermal[i])
                        fit.thermal_energy += .5 * masses_[i] * v * v;
            }
        }
        const double conversion = CONSTANT_TIME_CONVERTION / 10.0;
        std::map<std::string, double> values{
            {"velocity_fourier", fit.fourier * conversion},
            {"velocity_com_fourier", fit.centered_fourier * conversion},
            {"velocity_fit", fit.amplitude * conversion},
            {"mean_cosine", fit.mean_cosine},
            {"variance_cosine", fit.variance_cosine},
            {"thermal_temperature",
             2 * fit.thermal_energy / (thermal_dof_ * CONSTANT_kB)},
            {"com_vx", fit.mean_velocity[0] * conversion},
            {"com_vy", fit.mean_velocity[1] * conversion},
            {"com_vz", fit.mean_velocity[2] * conversion},
            {"density", density_},
            {"wave_number", parameters_.wave_number * 10},
            {"length", parameters_.length / 10}};
        std::vector<double> counts(parameters_.bins), masses(parameters_.bins),
            momenta(parameters_.bins), cosines(parameters_.bins),
            sines(parameters_.bins);
        for (int i = 0; i < count; ++i)
        {
            const double z = positions_[i][parameters_.gradient];
            const int bin =
                Periodic_Bin(z, parameters_.length, parameters_.bins);
            ++counts[bin];
            masses[bin] += masses_[i];
            momenta[bin] += masses_[i] * velocities_[i][parameters_.flow];
            cosines[bin] += masses_[i] * std::cos(parameters_.wave_number * z);
            sines[bin] += masses_[i] * std::sin(parameters_.wave_number * z);
        }
        for (int i = 0; i < parameters_.bins; ++i)
        {
            const auto prefix = "bin_" + std::to_string(i) + "_";
            values[prefix + "count"] = counts[i];
            values[prefix + "mass"] = masses[i];
            values[prefix + "momentum"] = momenta[i] * conversion;
            values[prefix + "cosine_mass"] = cosines[i];
            values[prefix + "sine_mass"] = sines[i];
        }
        if (Constrained())
        {
            constexpr double pressure_to_pa = 4184. / 6.02214076e23 / 1e-30;
            const char* names[6] = {"pxx", "pyy", "pzz", "pxy", "pxz", "pyz"};
            for (int j = 0; j < 6; ++j)
                values[names[j]] =
                    view_.observation->equilibrium.pressure[j] * pressure_to_pa;
            values["constraint_drift_energy"] = view_.drift_energy;
            values["constraint_length_residual"] =
                view_.observation->max_relative_length_error;
            values["constraint_tangent_residual"] =
                view_.observation->max_tangent_residual * conversion;
            values["kinetic_temperature"] =
                2 * view_.observation->equilibrium.kinetic_energy /
                (view_.observation->thermal_dof * CONSTANT_kB);
        }
        return values;
    }
    catch (const std::exception& error)
    {
        controller_->Throw_SPONGE_Error(spongeErrorSimulationBreakDown,
                                        "ProfileSampler::Cosine_Profile",
                                        error.what());
    }
    return {};
}

}  // namespace SpongeDynamics

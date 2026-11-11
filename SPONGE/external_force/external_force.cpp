#include "external_force.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "../utils/control/component_options.hpp"
#include "../utils/float_classification.hpp"
#include "group_load.h"

namespace SpongeDynamics
{
namespace
{
// 1 pN * 1 Angstrom * N_A / (4184 J/kcal).
constexpr double pN_to_internal = 6.02214076e23 * 1e-22 / 4184.0;

static __global__ void Add_Drive(const int count, const int* global_ids,
                                 const VECTOR* base_force, const float scale,
                                 VECTOR* external, const VECTOR* crd, int axis,
                                 double k)
{
    SIMPLE_DEVICE_FOR(i, count)
    {
        const double x = axis == 0 ? crd[i].x : axis == 1 ? crd[i].y : crd[i].z;
        const float profile = axis < 0 ? 1.f : float(cos(k * x));
        external[i] =
            external[i] + (scale * profile) * base_force[global_ids[i]];
    }
}
static __global__ void Merge_Drives(const int count, const VECTOR* external,
                                    VECTOR* force)
{
    SIMPLE_DEVICE_FOR(i, count) { force[i] = force[i] + external[i]; }
}
}  // namespace

void ExternalForce::Initial(CONTROLLER* controller, int atom_count,
                            const float* mass, const float* scaled_charge,
                            bool periodic, VECTOR box, VECTOR angles)
{
    Clear();
    controller_ = controller;
    if (!controller->Command_Exist("external_force")) return;
    try
    {
        // Decode the effective controller value, including command-line
        // overrides; the existing TOML parser preserves structured arrays.
        if (controller->Command_Exist("external_force"))
            drives_ = Parse_Drives(std::string("external_force = ") +
                                       controller->Command("external_force"),
                                   "external_force");
        Validate_Selections(drives_, atom_count);
        atom_count_ = atom_count;
        configuration_ =
            "external_force_v2;atoms=" + std::to_string(atom_count) + ";" +
            controller->Command("external_force");
        box_ = box;
        masses_.assign(mass, mass + atom_count);
        for (const auto& drive : drives_)
        {
            if (drive.profile.enabled)
            {
                Orthogonal_Box(box, angles, periodic);
                if (drive.kind == DriveKind::GroupLoad)
                    throw std::runtime_error(
                        "group_load does not accept a spatial profile");
            }
            if (drive.kind == DriveKind::GroupLoad)
            {
                if (periodic)
                    throw std::runtime_error(
                        "group_load currently requires pbc=false; continuous "
                        "periodic group coordinates are not implemented");
                has_groups_ = true;
            }
            std::vector<char> selected(atom_count, drive.atoms.empty());
            for (int id : drive.atoms) selected[id] = 1;
            std::vector<VECTOR> base(atom_count, VECTOR{0, 0, 0});
            double maximum = 0;
            for (int i = 0; i < atom_count; ++i)
            {
                if (!selected[i]) continue;
                if (!SpongeFloat::Is_Finite(mass[i]) || mass[i] <= 0)
                    throw std::runtime_error(
                        "selected atoms must have positive finite mass in this "
                        "development slice");
                if (drive.kind == DriveKind::GroupLoad) continue;
                double factor = pN_to_internal;
                if (drive.kind == DriveKind::Acceleration)
                    factor = mass[i] * 10.0 /
                             (static_cast<double>(CONSTANT_TIME_CONVERTION) *
                              CONSTANT_TIME_CONVERTION);
                else if (drive.kind == DriveKind::ElectricField)
                    factor =
                        (scaled_charge[i] /
                         static_cast<double>(CONSTANT_SPONGE_CHARGE_SCALE)) *
                        CONSTANT_EV_TO_KCAL_MOL / 10.0;
                double converted[3];
                for (int k = 0; k < 3; ++k)
                {
                    converted[k] = factor * drive.value[k];
                    if (!SpongeFloat::Is_Finite(converted[k]) ||
                        std::abs(converted[k]) >
                            std::numeric_limits<float>::max())
                        throw std::runtime_error(
                            "coupling force exceeds engine precision range");
                    maximum = std::max(maximum, std::abs(converted[k]));
                }
                base[i] = {static_cast<float>(converted[0]),
                           static_cast<float>(converted[1]),
                           static_cast<float>(converted[2])};
            }
            VECTOR* device_base = nullptr;
            // The combined allocation/copy helper aliases host memory on CPU.
            // These buffers must own their storage after the temporary dies.
            Device_Malloc_Safely(reinterpret_cast<void**>(&device_base),
                                 atom_count * sizeof(VECTOR));
            deviceMemcpy(device_base, base.data(), atom_count * sizeof(VECTOR),
                         deviceMemcpyHostToDevice);
            base_forces_.push_back(device_base);
            max_base_.push_back(maximum);
        }
        if (has_groups_)
        {
            ids_.resize(atom_count);
            coordinates_.resize(atom_count);
            global_coordinates_.resize(atom_count);
        }
        if (enabled())
            Device_Malloc_Safely(reinterpret_cast<void**>(&external_force_),
                                 atom_count * sizeof(VECTOR));
        controller->printf(
            "    EXTERNAL FORCE: %zu drives; path work is separate from "
            "internal potential/virial\n",
            drives_.size());
    }
    catch (const std::exception& error)
    {
        const std::string reason =
            std::string("Reason:\n\t") + error.what() + "\n";
        controller->Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                                       "ExternalForce::Initial",
                                       reason.c_str());
    }
}

void ExternalForce::Evaluate(int owner_count, const int* global_ids,
                             VECTOR* force, const VECTOR* coordinates,
                             double time_ps)
{
    if (!enabled()) return;
    Prepare_Force(owner_count, global_ids, coordinates, time_ps);
    if (owner_count)
        Launch_Device_Kernel(Merge_Drives, (owner_count + 127) / 128, 128, 0,
                             nullptr, owner_count, external_force_, force);
}

const VECTOR* ExternalForce::Prepare_Force(int owner_count,
                                           const int* global_ids,
                                           const VECTOR* coordinates,
                                           double time_ps)
{
    if (!enabled()) return nullptr;
    try
    {
        if (owner_count < 0 || owner_count > atom_count_)
            throw std::runtime_error("invalid coupling owner count");
        if (!owner_count) return external_force_;
        if (has_groups_)
        {
            if (owner_count != atom_count_)
                throw std::runtime_error(
                    "group_load requires a single rank owning all atoms");
            deviceMemcpy(ids_.data(), global_ids, owner_count * sizeof(int),
                         deviceMemcpyDeviceToHost);
            deviceMemcpy(coordinates_.data(), coordinates,
                         owner_count * sizeof(VECTOR),
                         deviceMemcpyDeviceToHost);
            for (int i = 0; i < owner_count; ++i)
                global_coordinates_[ids_[i]] = {
                    coordinates_[i].x, coordinates_[i].y, coordinates_[i].z};
        }
        std::vector<float> scales;
        double sum_bound = 0;
        for (std::size_t i = 0; i < drives_.size(); ++i)
        {
            const double scale = drives_[i].schedule.Scale(time_ps);
            const auto& drive = drives_[i];
            if (drive.kind == DriveKind::GroupLoad)
            {
                std::vector<Vec3> positions;
                std::vector<double> masses;
                for (int id : drive.atoms)
                {
                    positions.push_back(global_coordinates_[id]);
                    masses.push_back(masses_[id]);
                }
                Vec3 force, torque;
                for (int k = 0; k < 3; ++k)
                {
                    force[k] = drive.value[k] * pN_to_internal;
                    torque[k] = drive.torque[k] * pN_to_internal * 10.0;
                }
                const auto loads =
                    Distribute_Group_Load(positions, masses, force, torque);
                std::vector<VECTOR> base(atom_count_, VECTOR{0, 0, 0});
                max_base_[i] = 0;
                for (std::size_t j = 0; j < loads.size(); ++j)
                {
                    for (double component : loads[j])
                    {
                        if (std::abs(component) >
                            std::numeric_limits<float>::max())
                            throw std::runtime_error(
                                "group_load force exceeds engine precision "
                                "range");
                        max_base_[i] =
                            std::max(max_base_[i], std::abs(component));
                    }
                    base[drive.atoms[j]] = {static_cast<float>(loads[j][0]),
                                            static_cast<float>(loads[j][1]),
                                            static_cast<float>(loads[j][2])};
                }
                deviceMemcpy(base_forces_[i], base.data(),
                             atom_count_ * sizeof(VECTOR),
                             deviceMemcpyHostToDevice);
            }
            sum_bound += std::abs(scale) * max_base_[i];
            if (!SpongeFloat::Is_Finite(scale) ||
                !SpongeFloat::Is_Finite(sum_bound) ||
                std::abs(scale) > std::numeric_limits<float>::max() ||
                sum_bound > std::numeric_limits<float>::max())
                throw std::runtime_error(
                    "scheduled coupling force exceeds engine precision range");
            scales.push_back(static_cast<float>(scale));
        }
        deviceMemset(external_force_, 0, owner_count * sizeof(VECTOR));
        for (std::size_t i = 0; i < drives_.size(); ++i)
            Launch_Device_Kernel(
                Add_Drive, (owner_count + 127) / 128, 128, 0, nullptr,
                owner_count, global_ids, base_forces_[i], scales[i],
                external_force_, coordinates,
                drives_[i].profile.enabled ? drives_[i].profile.axis : -1,
                drives_[i].profile.enabled
                    ? 2 * 3.14159265358979323846 * drives_[i].profile.mode /
                          (drives_[i].profile.axis == 0   ? box_.x
                           : drives_[i].profile.axis == 1 ? box_.y
                                                          : box_.z)
                    : 0);
    }
    catch (const std::exception& error)
    {
        const std::string reason =
            std::string("Reason:\n\t") + error.what() + "\n";
        controller_->Throw_SPONGE_Error(spongeErrorSimulationBreakDown,
                                        "ExternalForce::Evaluate",
                                        reason.c_str());
    }
    return external_force_;
}

void ExternalForce::Clear()
{
    for (auto* ptr : base_forces_) deviceFree(ptr);
    if (external_force_) deviceFree(external_force_);
    external_force_ = nullptr;
    base_forces_.clear();
    max_base_.clear();
    drives_.clear();
    has_groups_ = false;
    masses_.clear();
    ids_.clear();
    coordinates_.clear();
    global_coordinates_.clear();
    atom_count_ = 0;
    configuration_.clear();
}
}  // namespace SpongeDynamics

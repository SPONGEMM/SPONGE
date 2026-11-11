#include "energy_accounting.h"

#include <algorithm>
#include <stdexcept>

#include "../io/component_state.h"
#include "../utils/float_classification.hpp"
namespace SpongeDynamics
{
namespace
{
__host__ __device__ double Atom_Work(const VECTOR a, const VECTOR b,
                                     const VECTOR fa, const VECTOR fb)
{
    return 0.5 * ((static_cast<double>(b.x) - a.x) *
                      (static_cast<double>(fa.x) + fb.x) +
                  (static_cast<double>(b.y) - a.y) *
                      (static_cast<double>(fa.y) + fb.y) +
                  (static_cast<double>(b.z) - a.z) *
                      (static_cast<double>(fa.z) + fb.z));
}
__host__ __device__ double Atom_Heat(VECTOR before, VECTOR after, float mass)
{
    // Difference of squares avoids cancellation between large total energies.
    return 0.5 * static_cast<double>(mass) *
           ((static_cast<double>(after.x) - before.x) *
                (static_cast<double>(after.x) + before.x) +
            (static_cast<double>(after.y) - before.y) *
                (static_cast<double>(after.y) + before.y) +
            (static_cast<double>(after.z) - before.z) *
                (static_cast<double>(after.z) + before.z));
}
#ifdef USE_GPU
static __global__ void Work_Partials(int n, const VECTOR* a, const VECTOR* b,
                                     const VECTOR* fa, const VECTOR* fb,
                                     double* out)
{
    __shared__ double sums[128];
    double sum = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x)
        sum += Atom_Work(a[i], b[i], fa[i], fb[i]);
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = 64; stride; stride /= 2)
    {
        if (threadIdx.x < stride)
            sums[threadIdx.x] += sums[threadIdx.x + stride];
        __syncthreads();
    }
    if (!threadIdx.x) out[blockIdx.x] = sums[0];
}
static __global__ void Heat_Partials(int n, const VECTOR* a, const VECTOR* b,
                                     const float* mass, double* out)
{
    __shared__ double sums[128];
    double sum = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x)
        sum += Atom_Heat(a[i], b[i], mass[i]);
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = 64; stride; stride /= 2)
    {
        if (threadIdx.x < stride)
            sums[threadIdx.x] += sums[threadIdx.x + stride];
        __syncthreads();
    }
    if (!threadIdx.x) out[blockIdx.x] = sums[0];
}
#endif
}  // namespace
void EnergyAccounting::Initial(CONTROLLER* controller, int atom_count,
                               bool enabled, const std::string& configuration,
                               double dt_ps, double time_ps, bool measure_heat)
{
    Clear();
    controller_ = controller;
    enabled_ = enabled || measure_heat;
    try
    {
        if (!enabled_) return;
        const std::string work_configuration =
            configuration +
            (controller_->Command_Choice("constrain_mode", "RATTLE", true)
                 ? ";rattle_path_work_unwrapped_owner_double_v1"
                 : "");
        ledger_.Initialize(work_configuration, dt_ps, time_ps, measure_heat);
        if (measure_heat) ledger_.heat_method = "thermal_substep_v1";
        if (measure_heat)
        {
            Device_Malloc_Safely(reinterpret_cast<void**>(&previous_velocity_),
                                 atom_count * sizeof(VECTOR));
            controller_->Step_Print_Initial("thermostat_heat", "%.16e");
        }
        atom_count_ = atom_count;
        Device_Malloc_Safely((void**)&zero_external_,
                             atom_count * sizeof(VECTOR));
        deviceMemset(zero_external_, 0, atom_count * sizeof(VECTOR));
        Device_Malloc_Safely(reinterpret_cast<void**>(&previous_coordinates_),
                             atom_count * sizeof(VECTOR));
        Device_Malloc_Safely(reinterpret_cast<void**>(&previous_force_),
                             atom_count * sizeof(VECTOR));
        const int blocks = std::min(256, (atom_count + 127) / 128);
        host_work_.resize(blocks);
        Device_Malloc_Safely(reinterpret_cast<void**>(&partial_work_),
                             blocks * sizeof(double));
        controller_->Step_Print_Initial("coupling_work", "%.16e");
    }
    catch (const std::exception& e)
    {
        controller_->Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                                        "EnergyAccounting::Initial", e.what());
    }
}
void EnergyAccounting::Begin_Step(int count, const VECTOR* coordinates,
                                  const VECTOR* external, double time_ps,
                                  const std::vector<Vec3>* precise_coordinates)
{
    if (!enabled_) return;
    if (!external) external = zero_external_;
    if (pending_ || count != atom_count_ || !coordinates || !external ||
        !Same_Time(time_ps, ledger_.time))
        controller_->Throw_SPONGE_Error(
            spongeErrorSimulationBreakDown, "EnergyAccounting::Begin_Step",
            "coupling work step order/count/time is invalid");
    // Ordinary observables carry the force-evaluation time t_n. Checkpoints
    // instead serialize the completed ledger at t_(n+1).
    reported_work_ = ledger_.work;
    reported_heat_ = ledger_.heat;
    ou_pending_ = ou_recorded_ = false;
    heat_increment_ = 0;
    previous_precise_coordinates_.clear();
    if (precise_coordinates)
    {
        if (precise_coordinates->size() != std::size_t(count))
            controller_->Throw_SPONGE_Error(spongeErrorSimulationBreakDown,
                                            "EnergyAccounting::Begin_Step",
                                            "invalid precise work coordinates");
        previous_precise_coordinates_ = *precise_coordinates;
        previous_host_external_.resize(count);
        current_host_external_.resize(count);
        deviceMemcpy(previous_host_external_.data(), external,
                     count * sizeof(VECTOR), deviceMemcpyDeviceToHost);
    }
    deviceMemcpy(previous_coordinates_, coordinates, count * sizeof(VECTOR),
                 deviceMemcpyDeviceToDevice);
    deviceMemcpy(previous_force_, external, count * sizeof(VECTOR),
                 deviceMemcpyDeviceToDevice);
    pending_ = true;
}
void EnergyAccounting::Complete_Step(
    int count, const VECTOR* coordinates, const VECTOR* external,
    double end_time, const std::vector<Vec3>* precise_coordinates)
{
    if (!enabled_) return;
    if (!external) external = zero_external_;
    try
    {
        if (!pending_ || count != atom_count_ || !coordinates || !external)
            throw std::runtime_error(
                "coupling work completion without a matching begin");
        if (ledger_.heat_enabled && (!ou_recorded_ || ou_pending_))
            throw std::runtime_error(
                "coupling step lacks one completed thermal heat event");
        // Main_Iteration preserves owner order in the supported single-rank
        // path. Output wrapping happens later, so these are actual step
        // displacements, not minimum-image displacements.
        double increment = 0;
        if (precise_coordinates)
        {
            if (precise_coordinates->size() != std::size_t(count) ||
                previous_precise_coordinates_.size() != std::size_t(count))
                throw std::runtime_error(
                    "precise work completion without matching begin");
            deviceMemcpy(current_host_external_.data(), external,
                         count * sizeof(VECTOR), deviceMemcpyDeviceToHost);
            for (int i = 0; i < count; ++i)
            {
                const auto& a = previous_host_external_[i];
                const auto& b = current_host_external_[i];
                const double f[3] = {double(a.x) + b.x, double(a.y) + b.y,
                                     double(a.z) + b.z};
                for (int j = 0; j < 3; ++j)
                    increment += .5 * f[j] *
                                 ((*precise_coordinates)[i][j] -
                                  previous_precise_coordinates_[i][j]);
            }
        }
        else
        {
            if (!previous_precise_coordinates_.empty())
                throw std::runtime_error(
                    "precise work state lost at completion");
#ifdef USE_GPU
            Launch_Device_Kernel(Work_Partials, host_work_.size(), 128, 0,
                                 nullptr, count, previous_coordinates_,
                                 coordinates, previous_force_, external,
                                 partial_work_);
            deviceMemcpy(host_work_.data(), partial_work_,
                         host_work_.size() * sizeof(double),
                         deviceMemcpyDeviceToHost);
            for (double part : host_work_) increment += part;
#else
            for (int i = 0; i < count; ++i)
                increment += Atom_Work(previous_coordinates_[i], coordinates[i],
                                       previous_force_[i], external[i]);
#endif
        }
        ledger_.Complete(increment, end_time, heat_increment_);
        pending_ = false;
    }
    catch (const std::exception& e)
    {
        controller_->Throw_SPONGE_Error(spongeErrorSimulationBreakDown,
                                        "EnergyAccounting::Complete_Step",
                                        e.what());
    }
}
void EnergyAccounting::Begin_Thermal(int count, const VECTOR* velocity)
{
    if (!ledger_.heat_enabled || !pending_ || ou_pending_ || ou_recorded_ ||
        count != atom_count_ || !velocity)
        controller_->Throw_SPONGE_Error(
            spongeErrorSimulationBreakDown, "EnergyAccounting::Begin_Thermal",
            "invalid thermal heat event order/count");
    deviceMemcpy(previous_velocity_, velocity, count * sizeof(VECTOR),
                 deviceMemcpyDeviceToDevice);
    ou_pending_ = true;
}
void EnergyAccounting::End_Thermal(int count, const VECTOR* velocity,
                                   const float* mass)
{
    if (!ledger_.heat_enabled || !pending_ || !ou_pending_ || ou_recorded_ ||
        count != atom_count_ || !velocity || !mass)
        controller_->Throw_SPONGE_Error(
            spongeErrorSimulationBreakDown, "EnergyAccounting::End_Thermal",
            "invalid thermal heat event completion");
    double increment = 0;
#ifdef USE_GPU
    Launch_Device_Kernel(Heat_Partials, host_work_.size(), 128, 0, nullptr,
                         count, previous_velocity_, velocity, mass,
                         partial_work_);
    deviceMemcpy(host_work_.data(), partial_work_,
                 host_work_.size() * sizeof(double), deviceMemcpyDeviceToHost);
    for (double part : host_work_) increment += part;
#else
    for (int i = 0; i < count; ++i)
        increment += Atom_Heat(previous_velocity_[i], velocity[i], mass[i]);
#endif
    heat_increment_ = increment;
    ou_pending_ = false;
    ou_recorded_ = true;
}
void EnergyAccounting::Step_Print()
{
    if (enabled_) controller_->Step_Print("coupling_work", reported_work_);
    if (ledger_.heat_enabled)
        controller_->Step_Print("thermostat_heat", reported_heat_);
}
void EnergyAccounting::Record_Host_Thermal(double heat)
{
    if (!ledger_.heat_enabled || !pending_ || ou_pending_ || ou_recorded_ ||
        !SpongeFloat::Is_Finite(heat))
        controller_->Throw_SPONGE_Error(spongeErrorSimulationBreakDown,
                                        "EnergyAccounting::Record_Host_Thermal",
                                        "invalid host thermal event");
    heat_increment_ = heat;
    ou_recorded_ = true;
}
std::map<std::string, std::string> EnergyAccounting::Export_Restart() const
{
    if (!enabled_) return {};
    if (pending_)
        throw std::runtime_error("cannot checkpoint unfinished energy step");
    return State_Prefix(ledger_.Export(), "energy_");
}
void EnergyAccounting::Restore_Restart(
    const std::map<std::string, std::string>& all, double time)
{
    try
    {
        auto state = State_Select(all, "energy_");
        if (!enabled_ && state.empty()) return;
        if (!enabled_ || state.empty())
            throw std::runtime_error(
                "dynamic restart energy configuration missing or removed; use "
                "structural load");
        ledger_.Restore(state, time);
        reported_work_ = ledger_.work;
        reported_heat_ = ledger_.heat;
    }
    catch (const std::exception& e)
    {
        controller_->Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                                        "EnergyAccounting::Restore", e.what());
    }
}
static __global__ void Kicked_Velocity(int n, const VECTOR* v, const VECTOR* f,
                                       const float* m, double dt, VECTOR* out)
{
    SIMPLE_DEVICE_FOR(i, n) { out[i] = v[i] + float(dt / m[i]) * f[i]; }
}
void EnergyAccounting::Begin_Kicked_Thermal(int n, const VECTOR* v,
                                            const VECTOR* f, const float* m,
                                            double dt)
{
    if (!ledger_.heat_enabled || !pending_ || ou_pending_ || ou_recorded_ ||
        n != atom_count_)
        throw std::runtime_error("invalid friction heat stage");
    Launch_Device_Kernel(Kicked_Velocity, (n + 127) / 128, 128, 0, nullptr, n,
                         v, f, m, dt, previous_velocity_);
    ou_pending_ = true;
}
void EnergyAccounting::Clear()
{
    for (auto* p : {previous_coordinates_, previous_force_, previous_velocity_,
                    zero_external_})
        if (p) deviceFree(p);
    if (partial_work_) deviceFree(partial_work_);
    previous_coordinates_ = previous_force_ = previous_velocity_ =
        zero_external_ = nullptr;
    partial_work_ = nullptr;
    enabled_ = pending_ = ou_pending_ = ou_recorded_ = false;
    atom_count_ = 0;
    ledger_ = {};
    host_work_.clear();
    heat_increment_ = reported_heat_ = reported_work_ = 0;
    previous_precise_coordinates_.clear();
    previous_host_external_.clear();
    current_host_external_.clear();
}

}  // namespace SpongeDynamics

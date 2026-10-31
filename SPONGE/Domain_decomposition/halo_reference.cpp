#include "halo_reference.h"

#include <stdexcept>

#include "../control.h"
#include "../utils/float_classification.hpp"

namespace
{
constexpr unsigned int kThreads = 128;

unsigned int Blocks(int count)
{
    return static_cast<unsigned int>(count) / kThreads +
           (static_cast<unsigned int>(count) % kThreads != 0);
}

void Check_Input(int count, const int* ids, const VECTOR* crd,
                 const Boundary& boundary)
{
    if (count < 0 || (count > 0 && (ids == nullptr || crd == nullptr)))
        throw std::invalid_argument("invalid halo reference atom inputs");
    const auto& cell = boundary.cell;
    const float values[] = {cell.a11, cell.a21, cell.a22,
                            cell.a31, cell.a32, cell.a33};
    for (float value : values)
        if (!SpongeFloat::Is_Finite(value))
            throw std::invalid_argument("nonfinite halo reference cell");
}

bool Same_Cell(const Boundary& a, const Boundary& b)
{
    return a.policy == b.policy && a.cell.a11 == b.cell.a11 &&
           a.cell.a21 == b.cell.a21 && a.cell.a22 == b.cell.a22 &&
           a.cell.a31 == b.cell.a31 && a.cell.a32 == b.cell.a32 &&
           a.cell.a33 == b.cell.a33;
}

void Check_Launch(const char* kernel)
{
#ifdef USE_GPU
    const auto error = deviceGetLastError();
    if (error != 0)
        throw std::runtime_error(std::string("halo reference ") + kernel +
                                 ": " + deviceGetErrorString(error));
#else
    (void)kernel;
#endif
}

__global__ void Reset_Status_Device(int* status, int refresh)
{
    status[0] = refresh;
    status[1] = 0;
}

__global__ void Record_Reference_Device(int count, const int* ids,
                                        const VECTOR* crd, int* reference_ids,
                                        VECTOR* reference_crd, int* status)
{
    SIMPLE_DEVICE_FOR(i, count)
    {
        const VECTOR value = crd[i];
        if (!SpongeFloat::Is_Finite(value.x) ||
            !SpongeFloat::Is_Finite(value.y) ||
            !SpongeFloat::Is_Finite(value.z))
            atomicExch(status + 1, 1);
        reference_ids[i] = ids[i];
        reference_crd[i] = value;
    }
}

__global__ void Check_Motion_Device(int count, const int* ids,
                                    const VECTOR* crd, const int* reference_ids,
                                    const VECTOR* reference_crd, bool compare,
                                    double limit_squared, int* status)
{
    SIMPLE_DEVICE_FOR(i, count)
    {
        const VECTOR value = crd[i];
        if (!SpongeFloat::Is_Finite(value.x) ||
            !SpongeFloat::Is_Finite(value.y) ||
            !SpongeFloat::Is_Finite(value.z))
        {
            atomicExch(status + 1, 1);
        }
        else if (compare)
        {
            const double dx = static_cast<double>(value.x) - reference_crd[i].x;
            const double dy = static_cast<double>(value.y) - reference_crd[i].y;
            const double dz = static_cast<double>(value.z) - reference_crd[i].z;
            if (ids[i] != reference_ids[i] ||
                dx * dx + dy * dy + dz * dz >= limit_squared)
                atomicExch(status, 1);
        }
    }
}

bool Read_Status(int* device_status, deviceStream_t stream)
{
    int status[2] = {};
#ifdef USE_GPU
    auto error = deviceMemcpyAsync(status, device_status, sizeof(status),
                                   deviceMemcpyDeviceToHost, stream);
    if (error == 0) error = deviceStreamSynchronize(stream);
    if (error != 0)
        throw std::runtime_error(std::string("halo reference readback: ") +
                                 deviceGetErrorString(error));
#else
    deviceMemcpyAsync(status, device_status, sizeof(status),
                      deviceMemcpyDeviceToHost, stream);
#endif
    if (status[1])
        throw std::runtime_error(
            "halo reference received nonfinite coordinates");
    return status[0] != 0;
}
}  // namespace

DD_HALO_REFERENCE::~DD_HALO_REFERENCE() { Clear(); }

void DD_HALO_REFERENCE::Ensure_Capacity(int count)
{
    if (status_ == nullptr &&
        !Device_Malloc_Safely(reinterpret_cast<void**>(&status_),
                              2 * sizeof(int)))
        throw std::bad_alloc();
    if (count <= capacity_) return;
    if (static_cast<size_t>(count) >
        std::numeric_limits<size_t>::max() / sizeof(VECTOR))
        throw std::overflow_error("halo reference allocation size overflow");
    VECTOR* new_crd = nullptr;
    int* new_ids = nullptr;
    if (!Device_Malloc_Safely(reinterpret_cast<void**>(&new_crd),
                              sizeof(VECTOR) * static_cast<size_t>(count)))
        throw std::bad_alloc();
    if (!Device_Malloc_Safely(reinterpret_cast<void**>(&new_ids),
                              sizeof(int) * static_cast<size_t>(count)))
    {
        Free_Single_Device_Pointer(reinterpret_cast<void**>(&new_crd));
        throw std::bad_alloc();
    }
    Free_Single_Device_Pointer(reinterpret_cast<void**>(&reference_crd_));
    Free_Single_Device_Pointer(reinterpret_cast<void**>(&reference_ids_));
    reference_crd_ = new_crd;
    reference_ids_ = new_ids;
    capacity_ = count;
}

void DD_HALO_REFERENCE::Record(int owned_count, const int* atom_local,
                               const VECTOR* crd, const Boundary& boundary,
                               deviceStream_t stream)
{
    valid_ = false;
    Check_Input(owned_count, atom_local, crd, boundary);
    if (owned_count > 0)
    {
        Ensure_Capacity(owned_count);
        Launch_Device_Kernel(Reset_Status_Device, 1, 1, 0, stream, status_, 0);
        Check_Launch("Reset_Status_Device");
        Launch_Device_Kernel(Record_Reference_Device, Blocks(owned_count),
                             kThreads, 0, stream, owned_count, atom_local, crd,
                             reference_ids_, reference_crd_, status_);
        Check_Launch("Record_Reference_Device");
        Read_Status(status_, stream);
    }
    owned_count_ = owned_count;
    boundary_ = boundary;
    valid_ = true;
}

bool DD_HALO_REFERENCE::Needs_Refresh(int owned_count, const int* atom_local,
                                      const VECTOR* crd,
                                      const Boundary& boundary,
                                      double displacement_limit,
                                      deviceStream_t stream)
{
    Check_Input(owned_count, atom_local, crd, boundary);
    if (!SpongeFloat::Is_Finite(displacement_limit) || displacement_limit < 0)
        throw std::invalid_argument("invalid halo displacement limit");
    const bool compare =
        valid_ && owned_count == owned_count_ && Same_Cell(boundary, boundary_);
    const bool refresh = !compare || displacement_limit == 0;
    if (owned_count == 0) return refresh;
    Ensure_Capacity(0);
    Launch_Device_Kernel(Reset_Status_Device, 1, 1, 0, stream, status_,
                         static_cast<int>(refresh));
    Check_Launch("Reset_Status_Device");
    Launch_Device_Kernel(Check_Motion_Device, Blocks(owned_count), kThreads, 0,
                         stream, owned_count, atom_local, crd, reference_ids_,
                         reference_crd_, compare,
                         displacement_limit * displacement_limit, status_);
    Check_Launch("Check_Motion_Device");
    return Read_Status(status_, stream);
}

void DD_HALO_REFERENCE::Clear()
{
    Free_Single_Device_Pointer(reinterpret_cast<void**>(&reference_crd_));
    Free_Single_Device_Pointer(reinterpret_cast<void**>(&reference_ids_));
    Free_Single_Device_Pointer(reinterpret_cast<void**>(&status_));
    capacity_ = owned_count_ = 0;
    valid_ = false;
    boundary_ = {};
}

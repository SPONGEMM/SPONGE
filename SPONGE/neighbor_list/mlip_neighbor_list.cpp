#include "mlip_neighbor_list.h"

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include "../utils/float_classification.hpp"
#include "../utils/mlip_geometry.hpp"

#ifndef USE_CPU
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#endif

namespace
{
constexpr int kInitialBucketCapacity = 2;

inline void Require_Allocation(bool allocated, const char* what)
{
    if (!allocated) throw std::bad_alloc();
    (void)what;
}

inline int Grow_Capacity(int capacity, int required)
{
    int next = capacity > 0 ? capacity : 1;
    while (next < required)
    {
        if (next > std::numeric_limits<int>::max() / 2)
        {
            next = required;
            break;
        }
        next *= 2;
    }
    return next;
}

inline bool Same_Orthogonal_Cell(const Boundary& left, const Boundary& right)
{
    return left.cell.a11 == right.cell.a11 && left.cell.a22 == right.cell.a22 &&
           left.cell.a33 == right.cell.a33 && left.cell.a21 == right.cell.a21 &&
           left.cell.a31 == right.cell.a31 && left.cell.a32 == right.cell.a32;
}

inline void Validate_Boundary(const Boundary& boundary)
{
    if (boundary.policy != BoundaryPolicy::Periodic)
        throw std::invalid_argument(
            "MLIP neighbor graph supports periodic orthogonal cells only");
    if (boundary.cell.a21 != 0.0f || boundary.cell.a31 != 0.0f ||
        boundary.cell.a32 != 0.0f)
        throw std::invalid_argument(
            "MLIP neighbor graph does not support triclinic cells");
    if (!std::isfinite(boundary.cell.a11) ||
        !std::isfinite(boundary.cell.a22) ||
        !std::isfinite(boundary.cell.a33) || boundary.cell.a11 <= 0.0f ||
        boundary.cell.a22 <= 0.0f || boundary.cell.a33 <= 0.0f)
        throw std::invalid_argument(
            "MLIP neighbor graph requires finite positive cell lengths");
}

__device__ __forceinline__ bool Wrap_Axis(float raw, float length,
                                          double* wrapped, int* error)
{
    if (!SpongeFloat::Is_Finite(raw))
    {
        atomicExch(error, 1);
        return false;
    }
    *wrapped = MlipGeometry::Canonical(raw, length);
    return true;
}

__device__ __forceinline__ float Round_Down_To_Float(double coordinate)
{
    float rounded = static_cast<float>(coordinate);
    if (static_cast<double>(rounded) > coordinate)
        rounded = nextafterf(rounded, -FLT_MAX);
    return rounded;
}

__device__ __forceinline__ float Round_Up_To_Float(double coordinate)
{
    float rounded = static_cast<float>(coordinate);
    if (static_cast<double>(rounded) < coordinate)
        rounded = nextafterf(rounded, FLT_MAX);
    return rounded;
}

__device__ __forceinline__ int Clamp_Bin(float coordinate, float grid_length,
                                         int dimension)
{
    // Match the MLIP-only double-coordinate bucket mapping. Interval
    // endpoints are rounded outward; the mapping is monotonic.
    int bin = MlipGeometry::Bin(coordinate, grid_length, dimension);
    if (bin < 0) bin = 0;
    if (bin >= dimension) bin = dimension - 1;
    return bin;
}

__device__ __forceinline__ bool Get_Image_Shift(
    int64_t image, float target, float source, double target_canonical,
    double source_canonical, float length, int* q, int* error)
{
    // Recover the relative wrap directly. Subtracting huge absolute wrap
    // counts loses integer units even when the actual image fits int32.
    const double relative = nearbyint(
        ((double(target) - source) - (target_canonical - source_canonical)) /
        double(length));
    const double value = relative + double(image);
    if (!SpongeFloat::Is_Finite(value) || value < INT_MIN || value > INT_MAX)
    {
        atomicExch(error, 1);
        return false;
    }
    *q = static_cast<int>(value);
    return true;
}

struct Count_Emit
{
    uint64_t count;
    int* error;

    __device__ __forceinline__ void operator()(int, int, int, int)
    {
        if (count == ~uint64_t(0))
            atomicExch(error, 1);
        else
            ++count;
    }
};

struct Fill_Emit
{
    uint64_t offset;
    uint64_t count;
    int total_edges;
    int* source_local_index;
    int* image_shift;
    int* error;

    __device__ __forceinline__ void operator()(int source, int qx, int qy,
                                               int qz)
    {
        const uint64_t edge = offset + count;
        if (edge >= static_cast<uint64_t>(total_edges))
        {
            atomicExch(error, 1);
        }
        else
        {
            source_local_index[edge] = source;
            image_shift[3 * edge] = qx;
            image_shift[3 * edge + 1] = qy;
            image_shift[3 * edge + 2] = qz;
        }
        ++count;
    }
};

template <typename Emit>
__device__ __forceinline__ void Emit_If_Neighbor(
    int target, int source, int64_t mx, int64_t my, int64_t mz,
    double target_canonical_x, double target_canonical_y,
    double target_canonical_z, const int* atom_local, const VECTOR* crd,
    const VECTOR target_raw, const Boundary boundary, double radius_squared,
    int* error, Emit* emit)
{
    const VECTOR source_raw = crd[source];
    double sx, sy, sz;
    if (!Wrap_Axis(source_raw.x, boundary.cell.a11, &sx, error) ||
        !Wrap_Axis(source_raw.y, boundary.cell.a22, &sy, error) ||
        !Wrap_Axis(source_raw.z, boundary.cell.a33, &sz, error))
        return;
    int qx, qy, qz;
    if (!Get_Image_Shift(mx, target_raw.x, source_raw.x, target_canonical_x, sx,
                         boundary.cell.a11, &qx, error) ||
        !Get_Image_Shift(my, target_raw.y, source_raw.y, target_canonical_y, sy,
                         boundary.cell.a22, &qy, error) ||
        !Get_Image_Shift(mz, target_raw.z, source_raw.z, target_canonical_z, sz,
                         boundary.cell.a33, &qz, error))
        return;
    if (atom_local[source] == atom_local[target] && qx == 0 && qy == 0 &&
        qz == 0)
        return;
    const double dx = static_cast<double>(target_raw.x) - source_raw.x -
                      static_cast<double>(qx) * boundary.cell.a11;
    const double dy = static_cast<double>(target_raw.y) - source_raw.y -
                      static_cast<double>(qy) * boundary.cell.a22;
    const double dz = static_cast<double>(target_raw.z) - source_raw.z -
                      static_cast<double>(qz) * boundary.cell.a33;
    if (dx * dx + dy * dy + dz * dz <= radius_squared)
        (*emit)(source, qx, qy, qz);
}

template <typename Emit>
__device__ __forceinline__ void Visit_Target_Candidates(
    int target, const int* atom_local, const VECTOR* crd,
    const Boundary boundary, float radius, float grid_length, int nx, int ny,
    int nz, const int* grid_atom_numbers, const int* grid_atoms,
    int owned_bucket_capacity, const int* grid_ghost_numbers,
    const int* grid_ghosts, int ghost_bucket_capacity, int* error, Emit* emit)
{
    const float lx = boundary.cell.a11;
    const float ly = boundary.cell.a22;
    const float lz = boundary.cell.a33;
    const VECTOR target_raw = crd[target];
    double tx, ty, tz;
    if (!Wrap_Axis(target_raw.x, lx, &tx, error) ||
        !Wrap_Axis(target_raw.y, ly, &ty, error) ||
        !Wrap_Axis(target_raw.z, lz, &tz, error))
        return;

    const double r = radius;
    // Enclose the exact canonical coordinate before constructing search
    // intervals. The same double canonicalization populates MLIP-only buckets.
    const double txlo = MlipGeometry::Lower(nextafter(tx, -HUGE_VAL));
    const double txhi = MlipGeometry::Upper(nextafter(tx, HUGE_VAL));
    const double tylo = MlipGeometry::Lower(nextafter(ty, -HUGE_VAL));
    const double tyhi = MlipGeometry::Upper(nextafter(ty, HUGE_VAL));
    const double tzlo = MlipGeometry::Lower(nextafter(tz, -HUGE_VAL));
    const double tzhi = MlipGeometry::Upper(nextafter(tz, HUGE_VAL));
    const double mx_min_d = floor(nextafter((txlo - r) / lx, -HUGE_VAL));
    const double mx_max_d = floor(nextafter((txhi + r) / lx, HUGE_VAL));
    const double my_min_d = floor(nextafter((tylo - r) / ly, -HUGE_VAL));
    const double my_max_d = floor(nextafter((tyhi + r) / ly, HUGE_VAL));
    const double mz_min_d = floor(nextafter((tzlo - r) / lz, -HUGE_VAL));
    const double mz_max_d = floor(nextafter((tzhi + r) / lz, HUGE_VAL));
    const double image_limit = 9.0e18;
    if (mx_min_d <= -image_limit || mx_max_d >= image_limit ||
        my_min_d <= -image_limit || my_max_d >= image_limit ||
        mz_min_d <= -image_limit || mz_max_d >= image_limit ||
        mx_max_d - mx_min_d > INT_MAX || my_max_d - my_min_d > INT_MAX ||
        mz_max_d - mz_min_d > INT_MAX)
    {
        atomicExch(error, 1);
        return;
    }
    const int64_t mx_min = static_cast<int64_t>(mx_min_d);
    const int64_t mx_max = static_cast<int64_t>(mx_max_d);
    const int64_t my_min = static_cast<int64_t>(my_min_d);
    const int64_t my_max = static_cast<int64_t>(my_max_d);
    const int64_t mz_min = static_cast<int64_t>(mz_min_d);
    const int64_t mz_max = static_cast<int64_t>(mz_max_d);
    if (mx_max < mx_min || my_max < my_min || mz_max < mz_min)
    {
        atomicExch(error, 1);
        return;
    }

    const double radius_squared = r * r;
    for (int64_t mx = mx_min; mx <= mx_max; ++mx)
    {
        double xlo =
            nextafter(txlo - r - static_cast<double>(mx) * lx, -HUGE_VAL);
        double xhi =
            nextafter(txhi + r - static_cast<double>(mx) * lx, HUGE_VAL);
        if (xlo < 0.0) xlo = 0.0;
        if (xhi > lx) xhi = lx;
        if (xlo > xhi) continue;
        const int bx0 = Clamp_Bin(Round_Down_To_Float(xlo), grid_length, nx);
        const int bx1 = Clamp_Bin(Round_Up_To_Float(xhi), grid_length, nx);
        for (int64_t my = my_min; my <= my_max; ++my)
        {
            double ylo =
                nextafter(tylo - r - static_cast<double>(my) * ly, -HUGE_VAL);
            double yhi =
                nextafter(tyhi + r - static_cast<double>(my) * ly, HUGE_VAL);
            if (ylo < 0.0) ylo = 0.0;
            if (yhi > ly) yhi = ly;
            if (ylo > yhi) continue;
            const int by0 =
                Clamp_Bin(Round_Down_To_Float(ylo), grid_length, ny);
            const int by1 = Clamp_Bin(Round_Up_To_Float(yhi), grid_length, ny);
            for (int64_t mz = mz_min; mz <= mz_max; ++mz)
            {
                double zlo = nextafter(tzlo - r - static_cast<double>(mz) * lz,
                                       -HUGE_VAL);
                double zhi = nextafter(tzhi + r - static_cast<double>(mz) * lz,
                                       HUGE_VAL);
                if (zlo < 0.0) zlo = 0.0;
                if (zhi > lz) zhi = lz;
                if (zlo > zhi) continue;
                const int bz0 =
                    Clamp_Bin(Round_Down_To_Float(zlo), grid_length, nz);
                const int bz1 =
                    Clamp_Bin(Round_Up_To_Float(zhi), grid_length, nz);
                for (int bx = bx0; bx <= bx1; ++bx)
                {
                    for (int by = by0; by <= by1; ++by)
                    {
                        for (int bz = bz0; bz <= bz1; ++bz)
                        {
                            const int grid = (bx * ny + by) * nz + bz;
                            const int owned_in_bin = grid_atom_numbers[grid];
                            const int ghost_in_bin = grid_ghost_numbers[grid];
                            for (int entry = 0; entry < owned_in_bin; ++entry)
                            {
                                const int source =
                                    grid_atoms[static_cast<size_t>(grid) *
                                                   owned_bucket_capacity +
                                               entry];
                                Emit_If_Neighbor(target, source, mx, my, mz, tx,
                                                 ty, tz, atom_local, crd,
                                                 target_raw, boundary,
                                                 radius_squared, error, emit);
                            }
                            for (int entry = 0; entry < ghost_in_bin; ++entry)
                            {
                                const int source =
                                    grid_ghosts[static_cast<size_t>(grid) *
                                                    ghost_bucket_capacity +
                                                entry];
                                Emit_If_Neighbor(target, source, mx, my, mz, tx,
                                                 ty, tz, atom_local, crd,
                                                 target_raw, boundary,
                                                 radius_squared, error, emit);
                            }
                        }
                    }
                }
            }
        }
    }
}

__global__ void Validate_Coordinates_Device(int atom_count, const VECTOR* crd,
                                            int* error)
{
    SIMPLE_DEVICE_FOR(i, atom_count)
    {
        if (!SpongeFloat::Is_Finite(crd[i].x) ||
            !SpongeFloat::Is_Finite(crd[i].y) ||
            !SpongeFloat::Is_Finite(crd[i].z))
            atomicExch(error, 1);
    }
}

__global__ void Check_Rebuild_Device(int atom_count, const int* atom_local,
                                     const int* reference_atom_local,
                                     const VECTOR* crd,
                                     const VECTOR* reference_crd,
                                     double threshold_squared, int* needed)
{
    SIMPLE_DEVICE_FOR(i, atom_count)
    {
        if (!SpongeFloat::Is_Finite(crd[i].x) ||
            !SpongeFloat::Is_Finite(crd[i].y) ||
            !SpongeFloat::Is_Finite(crd[i].z))
        {
            // Route malformed updates through Build, whose coordinate
            // validation reports an error before replacing the published CSR.
            atomicExch(needed, 1);
        }
        else if (atom_local[i] != reference_atom_local[i])
        {
            atomicExch(needed, 1);
        }
        else
        {
            const double dx =
                static_cast<double>(crd[i].x) - reference_crd[i].x;
            const double dy =
                static_cast<double>(crd[i].y) - reference_crd[i].y;
            const double dz =
                static_cast<double>(crd[i].z) - reference_crd[i].z;
            if (dx * dx + dy * dy + dz * dz > threshold_squared)
                atomicExch(needed, 1);
        }
    }
}

__global__ void Count_Row_Entries_Device(
    int owned_count, const int* atom_local, const VECTOR* crd,
    const Boundary boundary, float radius, float grid_length, int nx, int ny,
    int nz, const int* grid_atom_numbers, const int* grid_atoms,
    int owned_bucket_capacity, const int* grid_ghost_numbers,
    const int* grid_ghosts, int ghost_bucket_capacity, uint64_t* row_count,
    int* error)
{
    SIMPLE_DEVICE_FOR(target, owned_count)
    {
        Count_Emit emit{0, error};
        Visit_Target_Candidates(target, atom_local, crd, boundary, radius,
                                grid_length, nx, ny, nz, grid_atom_numbers,
                                grid_atoms, owned_bucket_capacity,
                                grid_ghost_numbers, grid_ghosts,
                                ghost_bucket_capacity, error, &emit);
        row_count[target] = emit.count;
    }
}

__global__ void Convert_Row_Offsets_Device(int rows, const uint64_t* offsets,
                                           int total_edges, int* row_ptr)
{
    SIMPLE_DEVICE_FOR(i, rows + 1)
    {
        row_ptr[i] = i == rows ? total_edges : static_cast<int>(offsets[i]);
    }
}

__global__ void Fill_Row_Entries_Device(
    int owned_count, const int* atom_local, const VECTOR* crd,
    const Boundary boundary, float radius, float grid_length, int nx, int ny,
    int nz, const int* grid_atom_numbers, const int* grid_atoms,
    int owned_bucket_capacity, const int* grid_ghost_numbers,
    const int* grid_ghosts, int ghost_bucket_capacity,
    const uint64_t* row_count, const uint64_t* row_offset, int total_edges,
    int* source_local_index, int* image_shift, int* error)
{
    SIMPLE_DEVICE_FOR(target, owned_count)
    {
        Fill_Emit emit{row_offset[target], 0,           total_edges,
                       source_local_index, image_shift, error};
        Visit_Target_Candidates(target, atom_local, crd, boundary, radius,
                                grid_length, nx, ny, nz, grid_atom_numbers,
                                grid_atoms, owned_bucket_capacity,
                                grid_ghost_numbers, grid_ghosts,
                                ghost_bucket_capacity, error, &emit);
        if (emit.count != row_count[target]) atomicExch(error, 1);
    }
}

inline unsigned int Kernel_Threads()
{
    // Geometry traversal uses more registers than simple coordinate kernels.
    // The hardware thread limit alone is not a valid launch budget.
    constexpr unsigned int thread_budget = 128;
    return CONTROLLER::device_max_thread > 0
               ? std::min(CONTROLLER::device_max_thread, thread_budget)
               : thread_budget;
}

inline unsigned int Kernel_Blocks(int count)
{
    const unsigned int threads = Kernel_Threads();
    return count > 0 ? static_cast<unsigned int>(count) / threads +
                           (static_cast<unsigned int>(count) % threads != 0)
                     : 0u;
}

inline void Check_Kernel_Launch(const char* kernel)
{
#ifdef USE_GPU
    const auto status = deviceGetLastError();
    if (status != 0)
        throw std::runtime_error(std::string("MLIP ") + kernel + ": " +
                                 deviceGetErrorString(status));
#else
    (void)kernel;
#endif
}

inline void Copy_Device_Error(int* d_error, const char* message)
{
    int error = 0;
    deviceMemcpy(&error, d_error, sizeof(error), deviceMemcpyDeviceToHost);
    if (error != 0) throw std::overflow_error(message);
}
}  // namespace

MLIP_NEIGHBOR_LIST::~MLIP_NEIGHBOR_LIST() { Clear(); }

void MLIP_NEIGHBOR_LIST::Initial(CONTROLLER* controller, int max_local_atoms,
                                 double cutoff, double skin)
{
    (void)controller;
    if (max_local_atoms < 0)
        throw std::invalid_argument("maximum local atom count is negative");
    if (!std::isfinite(cutoff) || cutoff <= 0.0f || !std::isfinite(skin) ||
        skin < 0.0f || !std::isfinite(cutoff + skin))
        throw std::invalid_argument(
            "MLIP cutoff must be finite and positive and skin finite and "
            "nonnegative");
    Clear();
    skin_ = skin;
    graph_radius_ = MlipGeometry::Radius(cutoff, skin);
    grid_length_ = graph_radius_ * 0.5f;
    if (!std::isfinite(grid_length_) || grid_length_ <= 0.0f)
        throw std::invalid_argument("MLIP graph radius is not representable");
    Require_Allocation(
        Device_Malloc_Safely((void**)&d_owned_overflow_, sizeof(int)),
        "owned overflow");
    Require_Allocation(
        Device_Malloc_Safely((void**)&d_ghost_overflow_, sizeof(int)),
        "ghost overflow");
    Require_Allocation(
        Device_Malloc_Safely((void**)&d_build_error_, sizeof(int)),
        "build error");
    Require_Allocation(
        Device_Malloc_Safely((void**)&d_rebuild_needed_, sizeof(int)),
        "rebuild flag");
    Ensure_Reference_Capacity(max_local_atoms);
    initialized_ = true;
}

bool MLIP_NEIGHBOR_LIST::Update(const int* atom_local, int owned_count,
                                int ghost_count, const VECTOR* crd,
                                const Boundary& boundary, uint64_t domain_epoch,
                                bool force_rebuild)
{
    if (!initialized_)
        throw std::logic_error("MLIP neighbor graph has not been initialized");
    if (owned_count < 0 || ghost_count < 0 ||
        owned_count == std::numeric_limits<int>::max() ||
        owned_count > std::numeric_limits<int>::max() - ghost_count)
        throw std::invalid_argument("MLIP local atom counts are invalid");
    const int local_count = owned_count + ghost_count;
    if (local_count != 0 && (atom_local == nullptr || crd == nullptr))
        throw std::invalid_argument(
            "MLIP local IDs and coordinates are required for nonempty input");
    Validate_Boundary(boundary);

    bool rebuild = force_rebuild || !has_graph_ ||
                   owned_count != owned_count_ || ghost_count != ghost_count_ ||
                   domain_epoch != domain_epoch_ ||
                   !Same_Orthogonal_Cell(boundary, boundary_);
    if (skin_ == 0.0f) rebuild = true;
    if (local_count > reference_capacity_)
    {
        Ensure_Reference_Capacity(local_count);
        rebuild = true;
    }
    if (!rebuild && local_count > 0)
    {
        deviceMemset(d_rebuild_needed_, 0, sizeof(int));
        const double half_skin = MlipGeometry::Motion_Limit(skin_);
        Launch_Device_Kernel(Check_Rebuild_Device, Kernel_Blocks(local_count),
                             Kernel_Threads(), 0, NULL, local_count, atom_local,
                             reference_atom_local_, crd, reference_crd_,
                             half_skin * half_skin, d_rebuild_needed_);
        Check_Kernel_Launch("Check_Rebuild_Device");
        int needed = 0;
        deviceMemcpy(&needed, d_rebuild_needed_, sizeof(needed),
                     deviceMemcpyDeviceToHost);
        rebuild = needed != 0;
    }
    if (!rebuild) return false;
    Build(atom_local, owned_count, ghost_count, crd, boundary, domain_epoch);
    return true;
}

void MLIP_NEIGHBOR_LIST::Build(const int* atom_local, int owned_count,
                               int ghost_count, const VECTOR* crd,
                               const Boundary& boundary, uint64_t domain_epoch)
{
    const int local_count = owned_count + ghost_count;
    const float lengths[3] = {boundary.cell.a11, boundary.cell.a22,
                              boundary.cell.a33};
    int dimensions[3];
    for (int axis = 0; axis < 3; ++axis)
    {
        const double quotient = lengths[axis] / grid_length_;
        if (!std::isfinite(quotient) ||
            quotient > std::numeric_limits<int>::max())
            throw std::overflow_error(
                "MLIP bucket dimension exceeds int range");
        dimensions[axis] = std::max(1, static_cast<int>(floor(quotient)));
    }
    const bool grid_changed =
        dimensions[0] != nx_ || dimensions[1] != ny_ || dimensions[2] != nz_;
    if (grid_changed)
    {
        buckets_.Clear();
        nx_ = dimensions[0];
        ny_ = dimensions[1];
        nz_ = dimensions[2];
        buckets_.Initial_Buckets(nx_, ny_, nz_, kInitialBucketCapacity,
                                 kInitialBucketCapacity);
    }

    Ensure_Row_Capacity(owned_count);
    Ensure_Reference_Capacity(local_count);
    if (local_count > 0)
    {
        deviceMemset(d_build_error_, 0, sizeof(int));
        Launch_Device_Kernel(Validate_Coordinates_Device,
                             Kernel_Blocks(local_count), Kernel_Threads(), 0,
                             NULL, local_count, crd, d_build_error_);
        Check_Kernel_Launch("Validate_Coordinates_Device");
        Copy_Device_Error(d_build_error_,
                          "MLIP graph received a non-finite coordinate");
    }

    for (;;)
    {
        buckets_.Populate_Buckets(owned_count, ghost_count, crd, boundary,
                                  grid_length_, d_owned_overflow_,
                                  d_ghost_overflow_, true);
        Check_Kernel_Launch("Populate_Buckets");
        int owned_overflow = 0, ghost_overflow = 0;
        deviceMemcpy(&owned_overflow, d_owned_overflow_, sizeof(int),
                     deviceMemcpyDeviceToHost);
        deviceMemcpy(&ghost_overflow, d_ghost_overflow_, sizeof(int),
                     deviceMemcpyDeviceToHost);
        if (owned_overflow == 0 && ghost_overflow == 0) break;
        int owned_capacity = buckets_.owned_bucket_capacity;
        int ghost_capacity = buckets_.ghost_bucket_capacity;
        if (owned_overflow != 0)
        {
            if (owned_capacity >= std::max(1, owned_count))
                throw std::overflow_error(
                    "owned MLIP bucket overflow persisted");
            const int owned_limit = std::max(1, owned_count);
            const int owned_target =
                owned_capacity > std::numeric_limits<int>::max() / 2
                    ? owned_limit
                    : std::min(owned_limit, owned_capacity * 2);
            owned_capacity = Grow_Capacity(owned_capacity, owned_target);
        }
        if (ghost_overflow != 0)
        {
            if (ghost_capacity >= std::max(1, ghost_count))
                throw std::overflow_error(
                    "ghost MLIP bucket overflow persisted");
            const int ghost_limit = std::max(1, ghost_count);
            const int ghost_target =
                ghost_capacity > std::numeric_limits<int>::max() / 2
                    ? ghost_limit
                    : std::min(ghost_limit, ghost_capacity * 2);
            ghost_capacity = Grow_Capacity(ghost_capacity, ghost_target);
        }
        buckets_.Clear();
        buckets_.Initial_Buckets(nx_, ny_, nz_, owned_capacity, ghost_capacity);
    }

    deviceMemset(d_build_error_, 0, sizeof(int));
    if (owned_count > 0)
    {
        Launch_Device_Kernel(
            Count_Row_Entries_Device, Kernel_Blocks(owned_count),
            Kernel_Threads(), 0, NULL, owned_count, atom_local, crd, boundary,
            graph_radius_, grid_length_, nx_, ny_, nz_,
            buckets_.d_grid_atom_numbers, buckets_.d_grid_atoms,
            buckets_.owned_bucket_capacity, buckets_.d_grid_ghost_numbers,
            buckets_.d_grid_ghosts, buckets_.ghost_bucket_capacity, row_count_,
            d_build_error_);
        Check_Kernel_Launch("Count_Row_Entries_Device");
    }
    Copy_Device_Error(
        d_build_error_,
        "MLIP graph image or per-row count exceeds supported range");

    uint64_t total_edges = 0;
#ifdef USE_CPU
    if (owned_count > 0)
    {
        std::vector<uint64_t> host_counts(owned_count);
        std::vector<uint64_t> host_offsets(owned_count);
        deviceMemcpy(host_counts.data(), row_count_,
                     sizeof(uint64_t) * owned_count, deviceMemcpyDeviceToHost);
        for (int row = 0; row < owned_count; ++row)
        {
            host_offsets[row] = total_edges;
            if (host_counts[row] >
                std::numeric_limits<uint64_t>::max() - total_edges)
                throw std::overflow_error(
                    "MLIP CSR edge count exceeds uint64 range");
            total_edges += host_counts[row];
        }
        if (total_edges >
            static_cast<uint64_t>(std::numeric_limits<int>::max()))
            throw std::overflow_error("MLIP CSR edge count exceeds int32 ABI");
        deviceMemcpy(row_offset_, host_offsets.data(),
                     sizeof(uint64_t) * owned_count, deviceMemcpyHostToDevice);
    }
#else
    if (owned_count > 0)
    {
        thrust::device_ptr<uint64_t> counts(row_count_);
        thrust::device_ptr<uint64_t> offsets(row_offset_);
        const uint64_t largest_row =
            thrust::reduce(counts, counts + owned_count, uint64_t(0),
                           thrust::maximum<uint64_t>());
        if (largest_row >
            static_cast<uint64_t>(std::numeric_limits<int>::max()))
            throw std::overflow_error("MLIP CSR row exceeds int32 ABI");
        thrust::exclusive_scan(counts, counts + owned_count, offsets);
        total_edges = thrust::reduce(counts, counts + owned_count, uint64_t(0),
                                     thrust::plus<uint64_t>());
        if (total_edges >
            static_cast<uint64_t>(std::numeric_limits<int>::max()))
            throw std::overflow_error("MLIP CSR edge count exceeds int32 ABI");
    }
#endif

    const int edge_count = static_cast<int>(total_edges);
    Ensure_Edge_Capacity(edge_count);
    Launch_Device_Kernel(Convert_Row_Offsets_Device,
                         Kernel_Blocks(owned_count + 1), Kernel_Threads(), 0,
                         NULL, owned_count, row_offset_, edge_count,
                         staging_row_ptr_);
    Check_Kernel_Launch("Convert_Row_Offsets_Device");
    if (edge_count > 0)
    {
        deviceMemset(d_build_error_, 0, sizeof(int));
        Launch_Device_Kernel(
            Fill_Row_Entries_Device, Kernel_Blocks(owned_count),
            Kernel_Threads(), 0, NULL, owned_count, atom_local, crd, boundary,
            graph_radius_, grid_length_, nx_, ny_, nz_,
            buckets_.d_grid_atom_numbers, buckets_.d_grid_atoms,
            buckets_.owned_bucket_capacity, buckets_.d_grid_ghost_numbers,
            buckets_.d_grid_ghosts, buckets_.ghost_bucket_capacity, row_count_,
            row_offset_, edge_count, staging_source_local_index_,
            staging_image_shift_, d_build_error_);
        Check_Kernel_Launch("Fill_Row_Entries_Device");
        Copy_Device_Error(d_build_error_,
                          "MLIP graph count/fill traversal mismatch");
    }

    if (neighbor_epoch_ == std::numeric_limits<uint64_t>::max())
        throw std::overflow_error("MLIP neighbor epoch exhausted uint64 range");
    if (local_count > 0)
    {
        deviceMemcpy(reference_crd_, crd, sizeof(VECTOR) * local_count,
                     deviceMemcpyDeviceToDevice);
        deviceMemcpy(reference_atom_local_, atom_local,
                     sizeof(int) * local_count, deviceMemcpyDeviceToDevice);
    }

    std::swap(row_ptr_, staging_row_ptr_);
    std::swap(source_local_index_, staging_source_local_index_);
    std::swap(image_shift_, staging_image_shift_);
    std::swap(row_capacity_, staging_row_capacity_);
    std::swap(edge_capacity_, staging_edge_capacity_);
    owned_count_ = owned_count;
    ghost_count_ = ghost_count;
    edge_count_ = edge_count;
    domain_epoch_ = domain_epoch;
    boundary_ = boundary;
    has_graph_ = true;
    ++neighbor_epoch_;
}

void MLIP_NEIGHBOR_LIST::Ensure_Reference_Capacity(int local_count)
{
    if (local_count <= reference_capacity_) return;
    const int capacity =
        Grow_Capacity(reference_capacity_, std::max(1, local_count));
    Free_Single_Device_Pointer((void**)&reference_crd_);
    Free_Single_Device_Pointer((void**)&reference_atom_local_);
    Require_Allocation(Device_Malloc_Safely((void**)&reference_crd_,
                                            sizeof(VECTOR) * capacity),
                       "reference coordinates");
    Require_Allocation(Device_Malloc_Safely((void**)&reference_atom_local_,
                                            sizeof(int) * capacity),
                       "reference atom IDs");
    reference_capacity_ = capacity;
}

void MLIP_NEIGHBOR_LIST::Ensure_Row_Capacity(int owned_count)
{
    const int required = std::max(1, owned_count + 1);
    if (required > staging_row_capacity_)
    {
        const int capacity = Grow_Capacity(staging_row_capacity_, required);
        Free_Single_Device_Pointer((void**)&staging_row_ptr_);
        Require_Allocation(Device_Malloc_Safely((void**)&staging_row_ptr_,
                                                sizeof(int) * capacity),
                           "staging row pointer");
        staging_row_capacity_ = capacity;
    }
    const int row_required = std::max(1, owned_count);
    if (row_required > row_scratch_capacity_)
    {
        const int capacity = Grow_Capacity(row_scratch_capacity_, row_required);
        Free_Single_Device_Pointer((void**)&row_count_);
        Free_Single_Device_Pointer((void**)&row_offset_);
        Require_Allocation(Device_Malloc_Safely((void**)&row_count_,
                                                sizeof(uint64_t) * capacity),
                           "row count scratch");
        Require_Allocation(Device_Malloc_Safely((void**)&row_offset_,
                                                sizeof(uint64_t) * capacity),
                           "row offset scratch");
        row_scratch_capacity_ = capacity;
    }
}

void MLIP_NEIGHBOR_LIST::Ensure_Edge_Capacity(int edge_count)
{
    const int required = std::max(1, edge_count);
    if (required <= staging_edge_capacity_) return;
    const int capacity = Grow_Capacity(staging_edge_capacity_, required);
    Free_Single_Device_Pointer((void**)&staging_source_local_index_);
    Free_Single_Device_Pointer((void**)&staging_image_shift_);
    Require_Allocation(
        Device_Malloc_Safely((void**)&staging_source_local_index_,
                             sizeof(int) * capacity),
        "staging source indices");
    Require_Allocation(Device_Malloc_Safely((void**)&staging_image_shift_,
                                            sizeof(int) * 3 * capacity),
                       "staging image shifts");
    staging_edge_capacity_ = capacity;
}

void MLIP_NEIGHBOR_LIST::Clear()
{
    buckets_.Clear();
    Free_Single_Device_Pointer((void**)&row_ptr_);
    Free_Single_Device_Pointer((void**)&source_local_index_);
    Free_Single_Device_Pointer((void**)&image_shift_);
    Free_Single_Device_Pointer((void**)&staging_row_ptr_);
    Free_Single_Device_Pointer((void**)&staging_source_local_index_);
    Free_Single_Device_Pointer((void**)&staging_image_shift_);
    Free_Single_Device_Pointer((void**)&row_count_);
    Free_Single_Device_Pointer((void**)&row_offset_);
    Free_Single_Device_Pointer((void**)&reference_crd_);
    Free_Single_Device_Pointer((void**)&reference_atom_local_);
    Free_Single_Device_Pointer((void**)&d_owned_overflow_);
    Free_Single_Device_Pointer((void**)&d_ghost_overflow_);
    Free_Single_Device_Pointer((void**)&d_build_error_);
    Free_Single_Device_Pointer((void**)&d_rebuild_needed_);
    initialized_ = false;
    has_graph_ = false;
    skin_ = graph_radius_ = grid_length_ = 0.0f;
    owned_count_ = ghost_count_ = edge_count_ = 0;
    row_capacity_ = edge_capacity_ = staging_row_capacity_ =
        staging_edge_capacity_ = reference_capacity_ = row_scratch_capacity_ =
            0;
    nx_ = ny_ = nz_ = 0;
    neighbor_epoch_ = domain_epoch_ = 0;
    boundary_ = Boundary{};
}

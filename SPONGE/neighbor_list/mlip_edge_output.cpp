#include "mlip_edge_output.h"

#include <stdexcept>

#include "../control.h"
#include "../utils/float_classification.hpp"

namespace
{
constexpr int kThreads = 128;
__host__ __device__ float Rounded_Multiply(float left, float right)
{
#if defined(__CUDA_ARCH__)
    return __fmul_rn(left, right);
#else
    volatile float result = left * right;
    return result;
#endif
}
__device__ float Rounded_Subtract(float left, float right)
{
#if defined(__CUDA_ARCH__)
    return __fsub_rn(left, right);
#else
    volatile float result = left - right;
    return result;
#endif
}
template <class T>
void Reserve(T*& pointer, int& capacity, int required)
{
    required = std::max(1, required);
    if (capacity >= required) return;
    T* next = nullptr;
    if (!Device_Malloc_Safely(reinterpret_cast<void**>(&next),
                              sizeof(T) * static_cast<size_t>(required)))
        throw std::bad_alloc();
    Free_Single_Device_Pointer(reinterpret_cast<void**>(&pointer));
    pointer = next;
    capacity = required;
}
void Check_Launch(const char* kernel)
{
#ifdef USE_GPU
    const auto status = deviceGetLastError();
    if (status != 0)
        throw std::runtime_error(std::string(kernel) + ": " +
                                 deviceGetErrorString(status));
#else
    (void)kernel;
#endif
}
__global__ void Cache_Target_Rows(int owned, const int32_t* row_ptr,
                                  int* targets)
{
    SIMPLE_DEVICE_FOR(target, owned)
    {
        for (int edge = row_ptr[target]; edge < row_ptr[target + 1]; ++edge)
            targets[edge] = target;
    }
}
__device__ bool Scatter_Edge(int selected, int edge_count, int local_count,
                             const int64_t* ids, const float* gradients,
                             const int* targets, const int32_t* sources,
                             const int32_t* shifts, const VECTOR* coordinates,
                             LTMatrix3 model_cell, float inverse_length,
                             bool virial, float* forces, float moment[6],
                             int* error)
{
    const int64_t edge = ids[selected];
    if (edge < 0 || edge >= edge_count)
    {
        atomicExch(error, 1);
        return false;
    }
    const int target = targets[edge], source = sources[edge];
    if (source < 0 || source >= local_count || target < 0 ||
        target >= local_count)
    {
        atomicExch(error, 1);
        return false;
    }
    const size_t offset = static_cast<size_t>(selected) * 3;
    const float gx = gradients[offset], gy = gradients[offset + 1],
                gz = gradients[offset + 2];
    if (!SpongeFloat::Is_Finite(gx) || !SpongeFloat::Is_Finite(gy) ||
        !SpongeFloat::Is_Finite(gz))
    {
        atomicExch(error, 1);
        return false;
    }
    // A periodic self-image contributes stress, but has exactly zero net force.
    if (source != target)
    {
        atomicAdd(forces + static_cast<size_t>(source) * 3, gx);
        atomicAdd(forces + static_cast<size_t>(source) * 3 + 1, gy);
        atomicAdd(forces + static_cast<size_t>(source) * 3 + 2, gz);
        atomicAdd(forces + static_cast<size_t>(target) * 3, -gx);
        atomicAdd(forces + static_cast<size_t>(target) * 3 + 1, -gy);
        atomicAdd(forces + static_cast<size_t>(target) * 3 + 2, -gz);
    }
    if (virial)
    {
        const VECTOR rt = coordinates[target], rs = coordinates[source];
        // Torch CUDA tensor/scalar division multiplies a host FP32 reciprocal.
        // Preserve the rounded normalization and subtraction under fast math.
        const float tx = Rounded_Multiply(rt.x, inverse_length),
                    ty = Rounded_Multiply(rt.y, inverse_length),
                    tz = Rounded_Multiply(rt.z, inverse_length);
        const float sx = Rounded_Multiply(rs.x, inverse_length),
                    sy = Rounded_Multiply(rs.y, inverse_length),
                    sz = Rounded_Multiply(rs.z, inverse_length);
        const size_t shift_offset = static_cast<size_t>(edge) * 3;
        const float qx = shifts[shift_offset], qy = shifts[shift_offset + 1],
                    qz = shifts[shift_offset + 2];
        const float shift_x = Rounded_Multiply(qx, model_cell.a11) +
                              Rounded_Multiply(qy, model_cell.a21) +
                              Rounded_Multiply(qz, model_cell.a31);
        const float shift_y = Rounded_Multiply(qy, model_cell.a22) +
                              Rounded_Multiply(qz, model_cell.a32);
        const float shift_z = Rounded_Multiply(qz, model_cell.a33);
        const float vx = Rounded_Subtract(Rounded_Subtract(tx, sx), shift_x);
        const float vy = Rounded_Subtract(Rounded_Subtract(ty, sy), shift_y);
        const float vz = Rounded_Subtract(Rounded_Subtract(tz, sz), shift_z);
        moment[0] = -vx * gx;
        moment[1] = -0.5f * (vx * gy + vy * gx);
        moment[2] = -vy * gy;
        moment[3] = -0.5f * (vx * gz + vz * gx);
        moment[4] = -0.5f * (vy * gz + vz * gy);
        moment[5] = -vz * gz;
    }
    return true;
}
__global__ void Assemble_Edges(int selected_count, int edge_count,
                               int local_count, const int64_t* ids,
                               const float* gradients, const int* targets,
                               const int32_t* sources, const int32_t* shifts,
                               const VECTOR* coordinates, LTMatrix3 model_cell,
                               float inverse_length, bool virial, float* forces,
                               float* partials, int* error)
{
#ifdef USE_GPU
    const int selected = static_cast<int>(
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x);
    float moment[6] = {};
    if (selected < selected_count)
        Scatter_Edge(selected, edge_count, local_count, ids, gradients, targets,
                     sources, shifts, coordinates, model_cell, inverse_length,
                     virial, forces, moment, error);
    if (!virial) return;
    __shared__ float sums[6 * kThreads];
    for (int c = 0; c < 6; ++c) sums[c * kThreads + threadIdx.x] = moment[c];
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride /= 2)
    {
        if (threadIdx.x < stride)
            for (int c = 0; c < 6; ++c)
                sums[c * kThreads + threadIdx.x] +=
                    sums[c * kThreads + threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        for (int c = 0; c < 6; ++c)
            partials[static_cast<size_t>(blockIdx.x) * 6 + c] =
                sums[c * kThreads];
#else
    // CPU contract tests use the identical per-edge arithmetic without GPU
    // reduction.
    if (virial)
        for (int c = 0; c < 6; ++c) partials[c] = 0;
    for (int selected = 0; selected < selected_count; ++selected)
    {
        float moment[6] = {};
        Scatter_Edge(selected, edge_count, local_count, ids, gradients, targets,
                     sources, shifts, coordinates, model_cell, inverse_length,
                     virial, forces, moment, error);
        if (virial)
            for (int c = 0; c < 6; ++c) partials[c] += moment[c];
    }
#endif
}
__global__ void Finish_Virial(int partial_count, const float* partials,
                              float* result)
{
    float compact[6] = {};
#ifdef USE_GPU
    for (int block = threadIdx.x; block < partial_count; block += blockDim.x)
        for (int c = 0; c < 6; ++c)
            compact[c] += partials[static_cast<size_t>(block) * 6 + c];
    __shared__ float sums[6 * kThreads];
    for (int c = 0; c < 6; ++c) sums[c * kThreads + threadIdx.x] = compact[c];
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride /= 2)
    {
        if (threadIdx.x < stride)
            for (int c = 0; c < 6; ++c)
                sums[c * kThreads + threadIdx.x] +=
                    sums[c * kThreads + threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x != 0) return;
    for (int c = 0; c < 6; ++c) compact[c] = sums[c * kThreads];
#else
    for (int block = 0; block < partial_count; ++block)
        for (int c = 0; c < 6; ++c)
            compact[c] += partials[static_cast<size_t>(block) * 6 + c];
#endif
    result[0] = compact[0];
    result[1] = result[3] = compact[1];
    result[4] = compact[2];
    result[2] = result[6] = compact[3];
    result[5] = result[7] = compact[4];
    result[8] = compact[5];
}
}  // namespace

void MLIP_EDGE_OUTPUT::Assemble(const SPONGE_MLIP_NEIGHBOR_VIEW& graph,
                                int ghosts, const VECTOR* coordinates,
                                const LTMatrix3& cell, int selected_count,
                                const int64_t* ids, const float* gradients,
                                bool virial, double length)
{
    if (graph.neighbor_epoch == 0 || graph.owned_count < 0 || ghosts < 0 ||
        graph.edge_count < 0 || graph.owned_count > INT_MAX - ghosts ||
        graph.row_ptr == nullptr ||
        (graph.edge_count > 0 && (graph.source_local_index == nullptr ||
                                  graph.image_shift == nullptr)) ||
        (graph.owned_count == 0 && graph.edge_count != 0) ||
        selected_count < 0 ||
        (selected_count > 0 && (ids == nullptr || gradients == nullptr)) ||
        !SpongeFloat::Is_Finite(length) || length <= 0)
        throw std::invalid_argument("invalid MLIP edge output metadata");
    const float model_length = static_cast<float>(length);
    if (!SpongeFloat::Is_Finite(model_length) || model_length <= 0)
        throw std::invalid_argument(
            "MLIP model length conversion is not representable in FP32");
    const float inverse_length = 1.0f / model_length;
    if (!SpongeFloat::Is_Finite(inverse_length))
        throw std::invalid_argument(
            "MLIP model length reciprocal is not representable in FP32");
    const int local_count = graph.owned_count + ghosts;
    if (local_count > INT_MAX / 3)
        throw std::overflow_error(
            "MLIP force workspace exceeds native int range");
    if (local_count > 0 && coordinates == nullptr)
        throw std::invalid_argument("MLIP edge output coordinates missing");
    Reserve(local_forces_, local_capacity_, local_count * 3);
    if (error_ == nullptr &&
        !Device_Malloc_Safely(reinterpret_cast<void**>(&error_), sizeof(int)))
        throw std::bad_alloc();
    if (virial_9_ == nullptr &&
        !Device_Malloc_Safely(reinterpret_cast<void**>(&virial_9_),
                              9 * sizeof(float)))
        throw std::bad_alloc();
    deviceMemset(local_forces_, 0,
                 static_cast<size_t>(local_count) * 3 * sizeof(float));
    deviceMemset(error_, 0, sizeof(int));
    if (cached_neighbor_epoch_ != graph.neighbor_epoch)
    {
        Reserve(target_rows_, target_capacity_, graph.edge_count);
        if (graph.owned_count > 0)
        {
            Launch_Device_Kernel(Cache_Target_Rows,
                                 graph.owned_count / kThreads +
                                     (graph.owned_count % kThreads != 0),
                                 kThreads, 0, NULL, graph.owned_count,
                                 graph.row_ptr, target_rows_);
            Check_Launch("Cache_Target_Rows");
        }
        cached_neighbor_epoch_ = graph.neighbor_epoch;
    }
    owned_count_ = graph.owned_count;
    LTMatrix3 model_cell(Rounded_Multiply(cell.a11, inverse_length),
                         Rounded_Multiply(cell.a21, inverse_length),
                         Rounded_Multiply(cell.a22, inverse_length),
                         Rounded_Multiply(cell.a31, inverse_length),
                         Rounded_Multiply(cell.a32, inverse_length),
                         Rounded_Multiply(cell.a33, inverse_length));
    int blocks = selected_count / kThreads + (selected_count % kThreads != 0);
#ifndef USE_GPU
    blocks = selected_count > 0 ? 1 : 0;
#endif
    if (virial) Reserve(virial_partials_, partial_capacity_, blocks * 6);
    if (selected_count > 0)
    {
        Launch_Device_Kernel(Assemble_Edges, blocks, kThreads, 0, NULL,
                             selected_count, graph.edge_count, local_count, ids,
                             gradients, target_rows_, graph.source_local_index,
                             graph.image_shift, coordinates, model_cell,
                             inverse_length, virial, local_forces_,
                             virial_partials_, error_);
        Check_Launch("Assemble_Edges");
    }
    if (virial)
    {
        Launch_Device_Kernel(Finish_Virial, 1, kThreads, 0, NULL, blocks,
                             virial_partials_, virial_9_);
        Check_Launch("Finish_Virial");
    }
    int error = 0;
    deviceMemcpy(&error, error_, sizeof(int), deviceMemcpyDeviceToHost);
    if (error)
        throw std::invalid_argument(
            "invalid selected MLIP candidate ID or gradient");
}

void MLIP_EDGE_OUTPUT::Copy_Outputs(float* forces, float* virial,
                                    bool need_virial) const
{
    if ((owned_count_ > 0 && forces == nullptr) ||
        (need_virial && virial == nullptr))
        throw std::invalid_argument("MLIP edge output destination missing");
    if (owned_count_ > 0)
        deviceMemcpy(forces, local_forces_,
                     static_cast<size_t>(owned_count_) * 3 * sizeof(float),
                     deviceMemcpyDeviceToDevice);
    if (need_virial)
        deviceMemcpy(virial, virial_9_, 9 * sizeof(float),
                     deviceMemcpyDeviceToDevice);
}

void MLIP_EDGE_OUTPUT::Clear()
{
    Free_Single_Device_Pointer(reinterpret_cast<void**>(&target_rows_));
    Free_Single_Device_Pointer(reinterpret_cast<void**>(&local_forces_));
    Free_Single_Device_Pointer(reinterpret_cast<void**>(&virial_partials_));
    Free_Single_Device_Pointer(reinterpret_cast<void**>(&virial_9_));
    Free_Single_Device_Pointer(reinterpret_cast<void**>(&error_));
    target_capacity_ = local_capacity_ = partial_capacity_ = owned_count_ = 0;
    cached_neighbor_epoch_ = 0;
}

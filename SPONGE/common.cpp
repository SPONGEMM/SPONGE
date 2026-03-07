#include "common.h"

#ifdef USE_CPU
float rnorm3df(float a, float b, float c)
{
    return 1.0f / sqrtf(a * a + b * b + c * c);
}

float norm3df(float a, float b, float c)
{
    return sqrtf(a * a + b * b + c * c);
}

float erfcxf(float x) { return expf(x * x) * erfcf(x); }

float atomicAdd(float* x, float y)
{
    float x0;
#ifdef _WIN32
#pragma omp critical(sponge_atomic_add_float)
#else
#pragma omp atomic capture
#endif
    {
        x0 = *x;
        *x += y;
    }
    return x0;
}
double atomicAdd(double* x, double y)
{
    double x0;
#ifdef _WIN32
#pragma omp critical(sponge_atomic_add_double)
#else
#pragma omp atomic capture
#endif
    {
        x0 = *x;
        *x += y;
    }
    return x0;
}

int atomicAdd(int* x, int y)
{
    int x0;
#ifdef _WIN32
#pragma omp critical(sponge_atomic_add_int)
#else
#pragma omp atomic capture
#endif
    {
        x0 = *x;
        *x += y;
    }
    return x0;
}

int atomicExch(int* address, int val)
{
    int old;
#ifdef _WIN32
#pragma omp critical(sponge_atomic_exch_int)
#else
#pragma omp atomic capture
#endif
    {
        old = *address;
        *address = val;
    }
    return old;
}

void deviceMemcpy(void* to, const void* from, size_t size,
                  deviceMemcpyKind kind)
{
    if (to != from)
    {
        memcpy(to, from, size);
    }
}

void deviceMemcpyAsync(void* to, const void* from, size_t size,
                       deviceMemcpyKind kind, deviceStream_t stream)
{
    if (to != from)
    {
        memcpy(to, from, size);
    }
}

void deviceMemset(void* to, int val, size_t size)
{
    Reset_List((int*)to, val, size / sizeof(int));
}

void deviceFree(void* ptr) { free(ptr); }

#endif

void Free_Single_Device_Pointer(void** device_ptr)
{
#ifndef CPU_ARCH_NAME
    deviceFree(device_ptr[0]);
#else
    free(device_ptr[0]);
#endif
    device_ptr[0] = NULL;
}

void Free_Host_And_Device_Pointer(void** host_ptr, void** device_ptr)
{
    if (host_ptr != NULL)
    {
        free(host_ptr[0]);
        host_ptr[0] = NULL;
    }
#ifndef CPU_ARCH_NAME
    deviceFree(device_ptr[0]);
#endif
    device_ptr[0] = NULL;
}

static __global__ void Reset_List_Device(const int element_numbers, int* list,
                                         const int replace_element)
{
    SIMPLE_DEVICE_FOR(i, element_numbers) { list[i] = replace_element; }
}

static __global__ void Reset_List_Device(const int element_numbers, float* list,
                                         const float replace_element)
{
    SIMPLE_DEVICE_FOR(i, element_numbers) { list[i] = replace_element; }
}

void Reset_List(int* list, const int replace_element, const int element_numbers,
                const int threads)
{
    Launch_Device_Kernel(Reset_List_Device,
                         (element_numbers + threads - 1) / threads, threads, 0,
                         NULL, element_numbers, list, replace_element);
}

void Reset_List(float* list, const float replace_element,
                const int element_numbers, const int threads)
{
    Launch_Device_Kernel(Reset_List_Device,
                         (element_numbers + threads - 1) / threads, threads, 0,
                         NULL, element_numbers, list, replace_element);
}

static __global__ void Scale_List_Device(const int element_numbers, float* list,
                                         float scaler)
{
    SIMPLE_DEVICE_FOR(i, element_numbers) { list[i] = list[i] * scaler; }
}

void Scale_List(float* list, const float scaler, const int element_numbers,
                int threads)
{
    Launch_Device_Kernel(Scale_List_Device,
                         (element_numbers + threads - 1) / threads, threads, 0,
                         NULL, element_numbers, list, scaler);
}
static __global__ void Sum_Of_List_Device(const int start, const int end,
                                          const int* list, int* sum)
{
#ifdef GPU_ARCH_NAME
    if (threadIdx.x == 0)
    {
        sum[0] = 0;
    }
    __syncthreads();
#else
    sum[0] = 0;
#endif
    int lin = 0;
#ifdef GPU_ARCH_NAME
    for (int i = threadIdx.x + start; i < end; i = i + blockDim.x)
#else
#pragma omp parallel for reduction(+ : lin)
    for (int i = start; i < end; i += 1)
#endif
    {
        lin = lin + list[i];
    }
    atomicAdd(sum, lin);
}

static __global__ void Sum_Of_List_Device(const int start, const int end,
                                          const float* list, float* sum)
{
#ifdef GPU_ARCH_NAME
    if (threadIdx.x == 0)
    {
        sum[0] = 0;
    }
    __syncthreads();
#else
    sum[0] = 0;
#endif
    float lin = 0;
#ifdef GPU_ARCH_NAME
    for (int i = threadIdx.x + start; i < end; i = i + blockDim.x)
#else
#pragma omp parallel for reduction(+ : lin)
    for (int i = start; i < end; i += 1)
#endif
    {
        lin = lin + list[i];
    }
    atomicAdd(sum, lin);
}
static __global__ void Sum_Of_List_Device(const int start, const int end,
                                          const VECTOR* list, VECTOR* sum)
{
#ifdef GPU_ARCH_NAME
    if (threadIdx.x == 0)
    {
        sum[0] = {0, 0, 0};
    }
    __syncthreads();
#else
    sum[0] = {0, 0, 0};
#endif
    VECTOR lin = {0., 0., 0.};
#ifdef GPU_ARCH_NAME
    for (int i = threadIdx.x + start; i < end; i = i + blockDim.x)
    {
        lin = lin + list[i];
    }
#else
    float lin_x = 0.0f, lin_y = 0.0f, lin_z = 0.0f;
#pragma omp parallel for reduction(+ : lin_x, lin_y, lin_z)
    for (int i = start; i < end; i += 1)
    {
        lin_x += list[i].x;
        lin_y += list[i].y;
        lin_z += list[i].z;
    }
    lin = {lin_x, lin_y, lin_z};
#endif
    atomicAdd(sum, lin);
}

static __global__ void Sum_Of_List_Device(const int start, const int end,
                                          const LTMatrix3* list, LTMatrix3* sum)
{
#ifdef GPU_ARCH_NAME
    if (threadIdx.x == 0)
    {
        sum[0] = {0, 0, 0, 0, 0, 0};
    }
    __syncthreads();
#else
    sum[0] = {0, 0, 0, 0, 0, 0};
#endif
    LTMatrix3 lin = {0, 0, 0, 0, 0, 0};
#ifdef GPU_ARCH_NAME
    for (int i = threadIdx.x + start; i < end; i = i + blockDim.x)
    {
        lin = lin + list[i];
    }
#else
    float a11 = 0.0f, a21 = 0.0f, a22 = 0.0f;
    float a31 = 0.0f, a32 = 0.0f, a33 = 0.0f;
#pragma omp parallel for reduction(+ : a11, a21, a22, a31, a32, a33)
    for (int i = start; i < end; i += 1)
    {
        a11 += list[i].a11;
        a21 += list[i].a21;
        a22 += list[i].a22;
        a31 += list[i].a31;
        a32 += list[i].a32;
        a33 += list[i].a33;
    }
    lin = {a11, a21, a22, a31, a32, a33};
#endif
    atomicAdd(sum, lin);
}

void Sum_Of_List(const int* list, int* sum, const int end, const int start,
                 int threads)
{
    Launch_Device_Kernel(Sum_Of_List_Device, 1, threads, 0, NULL, start, end,
                         list, sum);
}

// 使用双精度 warp shuffle 归约的 float 求和，接口仍为 float
#ifdef GPU_ARCH_NAME
static __global__ void Sum_Of_List_Float_Block(const int start, const int end,
                                               const float* list,
                                               double* block_sums)
{
    extern __shared__ double warp_sums[];
    double partial = 0.0;
    int global_thread = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = global_thread + start; i < end; i += stride)
    {
        partial += static_cast<double>(list[i]);
    }

    device_mask_t lane_mask = deviceActiveMask();
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    {
        partial += deviceShflDown(lane_mask, partial, offset);
    }

    int lane = threadIdx.x & (warpSize - 1);
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0)
    {
        warp_sums[warp_id] = partial;
    }
    __syncthreads();

    const int warp_numbers = (blockDim.x + warpSize - 1) / warpSize;
    if (warp_id == 0)
    {
        double block_sum = (lane < warp_numbers) ? warp_sums[lane] : 0.0;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        {
            block_sum += deviceShflDown(FULL_MASK, block_sum, offset);
        }
        if (lane == 0)
        {
            block_sums[blockIdx.x] = block_sum;
        }
    }
}

static __global__ void Sum_Of_List_Float_Final(const double* block_sums,
                                               const int block_count,
                                               float* sum)
{
    double partial = 0.0;
    int idx = threadIdx.x;
    for (int i = idx; i < block_count; i += blockDim.x)
    {
        partial += block_sums[i];
    }
    device_mask_t lane_mask = deviceActiveMask();
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    {
        partial += deviceShflDown(lane_mask, partial, offset);
    }

    int lane = threadIdx.x & (warpSize - 1);
    int warp_id = threadIdx.x / warpSize;
    __shared__ double warp_buf[32];  // 支持最多 32 个 warp（blockDim <= 1024）
    if (lane == 0)
    {
        warp_buf[warp_id] = partial;
    }
    __syncthreads();

    int warp_numbers = (blockDim.x + warpSize - 1) / warpSize;
    partial = (lane < warp_numbers) ? warp_buf[lane] : 0.0;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    {
        partial += deviceShflDown(FULL_MASK, partial, offset);
    }
    if (lane == 0)
    {
        sum[0] = static_cast<float>(partial);
    }
}
#endif

void Sum_Of_List(const float* list, float* sum, const int end, const int start,
                 int threads)
{
#ifdef GPU_ARCH_NAME
    const int WARP_SIZE = 32;
    if (threads < WARP_SIZE) threads = WARP_SIZE;
    if (threads > 1024) threads = 1024;
    threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

    int grid = (end - start + threads - 1) / threads;
    if (grid < 1) grid = 1;
    int warp_numbers = (threads + WARP_SIZE - 1) / WARP_SIZE;
    size_t shared_memory_bytes = sizeof(double) * warp_numbers;

    double* block_sums = nullptr;
    deviceMalloc((void**)&block_sums, sizeof(double) * grid);

    Launch_Device_Kernel(Sum_Of_List_Float_Block, grid, threads,
                         shared_memory_bytes, NULL, start, end, list,
                         block_sums);

    int final_threads = (grid < 256) ? grid : 256;
    if (final_threads < WARP_SIZE) final_threads = WARP_SIZE;
    final_threads = ((final_threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    if (final_threads > 256) final_threads = 256;
    Launch_Device_Kernel(Sum_Of_List_Float_Final, 1, final_threads, 0, NULL,
                         block_sums, grid, sum);

    deviceFree(block_sums);
#else
    double s = 0.0;
#pragma omp parallel for reduction(+ : s)
    for (int i = start; i < end; i += 1)
    {
        s += list[i];
    }
    sum[0] = static_cast<float>(s);
#endif
}

void Sum_Of_List(const VECTOR* list, VECTOR* sum, const int end,
                 const int start, int threads)
{
    Launch_Device_Kernel(Sum_Of_List_Device, 1, threads, 0, NULL, start, end,
                         list, sum);
}

void Sum_Of_List(const LTMatrix3* list, LTMatrix3* sum, const int end,
                 const int start, int threads)
{
    Launch_Device_Kernel(Sum_Of_List_Device, 1, threads, 0, NULL, start, end,
                         list, sum);
}

__global__ void Setup_Rand_Normal_Kernel(const int float4_numbers,
                                         Philox4_32_10_t* rand_state,
                                         const int seed)
{
    SIMPLE_DEVICE_FOR(id, float4_numbers)
    {
        device_rand_init(seed, id, 0, rand_state + id);
    }
}
__global__ void Rand_Normal(const int float4_numbers,
                            Philox4_32_10_t* rand_state, float4* rand_float4)
{
    SIMPLE_DEVICE_FOR(i, float4_numbers)
    {
        device_get_4_normal_distributed_random_numbers(rand_float4, rand_state,
                                                       i);
    }
}

static __global__ void Device_Debug_Print_Device(const float* x)
{
    printf("DEBUG: %e\n", x[0]);
}

static __global__ void Device_Debug_Print_Device(const VECTOR* x)
{
    printf("DEBUG: %e %e %e\n", x[0].x, x[0].y, x[0].z);
}

static __global__ void Device_Debug_Print_Device(const int* x)
{
    printf("DEBUG: %d\n", x[0]);
}

static __global__ void Device_Debug_Print_Device(const LTMatrix3* x)
{
    printf(
        "DEBUG:trace = "
        "%.5e\n%.5e\t%.5e\t%.5e\n%.5e\t%.5e\t%.5e\n%.5e\t%.5e\t%.5e\n",
        x[0].a11 + x[0].a22 + x[0].a33, x[0].a11, 0.0f, 0.0f, x[0].a21,
        x[0].a22, 0.0f, x[0].a31, x[0].a32, x[0].a33);
}

void Device_Debug_Print(const int* x)
{
    Launch_Device_Kernel(Device_Debug_Print_Device, 1, 1, 0, NULL, x);
}

void Device_Debug_Print(const float* x)
{
    Launch_Device_Kernel(Device_Debug_Print_Device, 1, 1, 0, NULL, x);
}

void Device_Debug_Print(const VECTOR* x)
{
    Launch_Device_Kernel(Device_Debug_Print_Device, 1, 1, 0, NULL, x);
}
void Device_Debug_Print(const LTMatrix3* x)
{
    Launch_Device_Kernel(Device_Debug_Print_Device, 1, 1, 0, NULL, x);
}

int Check_2357_Factor(int number)
{
    int tempn;
    while (number > 0)
    {
        if (number == 1) return 1;
        tempn = number / 2;
        if (tempn * 2 != number) break;
        number = tempn;
    }

    while (number > 0)
    {
        if (number == 1) return 1;
        tempn = number / 3;
        if (tempn * 3 != number) break;
        number = tempn;
    }

    while (number > 0)
    {
        if (number == 1) return 1;
        tempn = number / 5;
        if (tempn * 5 != number) break;
        number = tempn;
    }

    while (number > 0)
    {
        if (number == 1) return 1;
        tempn = number / 7;
        if (tempn * 7 != number) break;
        number = tempn;
    }

    return 0;
}

#ifdef GPU_ARCH_NAME
static __global__ void upSweep(int* d_out, int* d_in, int n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = 1;

    for (int d = n >> 1; d > 0; d >>= 1)
    {
        __syncthreads();
        if (index < d)
        {
            int ai = offset * (2 * index + 1) - 1;
            int bi = offset * (2 * index + 2) - 1;
            d_in[bi] += d_in[ai];
        }
        offset *= 2;
    }

    if (index == 0)
    {
        d_out[n - 1] = d_in[n - 1];
        d_in[n - 1] = 0;
    }
}

static __global__ void downSweep(int* d_out, int* d_in, int n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = n;
    for (int d = 1; d < n; d *= 2)
    {
        offset >>= 1;
        __syncthreads();
        if (index < d)
        {
            int ai = offset * (2 * index + 1) - 1;
            int bi = offset * (2 * index + 2) - 1;

            int temp = d_in[ai];
            d_in[ai] = d_in[bi];
            d_in[bi] += temp;
        }
    }

    __syncthreads();

    if (index < n)
    {
        d_out[index] = d_in[index];
    }
}

Prefix_Sum::Prefix_Sum(int size)
{
    padded_size = 1;
    while (padded_size < size)
    {
        padded_size *= 2;
    }
    deviceMalloc((void**)&in, padded_size * sizeof(int));
    deviceMemset(in, 0, padded_size * sizeof(int));
    deviceMalloc((void**)&temp, padded_size * sizeof(int));
    deviceMemset(temp, 0, padded_size * sizeof(int));
    deviceMalloc((void**)&out, (padded_size + 1) * sizeof(int));
    deviceMemset(out, 0, (padded_size + 1) * sizeof(int));
    blockSize = std::min(padded_size, 1024);
    gridSize = (padded_size + blockSize - 1) / blockSize;
}

void Prefix_Sum::Scan()
{
    deviceMemcpy(temp, in, sizeof(int) * padded_size,
                 deviceMemcpyDeviceToDevice);
    Launch_Device_Kernel(upSweep, gridSize, blockSize, 0, NULL, out, temp,
                         padded_size);
    Launch_Device_Kernel(downSweep, gridSize, blockSize, 0, NULL, out, temp,
                         padded_size);
}
#endif

int Get_Fft_Patameter(float length)
{
    int tempi = (int)ceil(length + 3) >> 2 << 2;

    if (tempi >= 60 && tempi <= 68)
        tempi = 64;
    else if (tempi >= 120 && tempi <= 136)
        tempi = 128;
    else if (tempi >= 240 && tempi <= 272)
        tempi = 256;
    else if (tempi >= 480 && tempi <= 544)
        tempi = 512;
    else if (tempi >= 960 && tempi <= 1088)
        tempi = 1024;

    while (1)
    {
        if (Check_2357_Factor(tempi)) return tempi;
        tempi += 4;
    }
}

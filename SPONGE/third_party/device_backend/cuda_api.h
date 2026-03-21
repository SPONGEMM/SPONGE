#ifndef BASIC_BACKEND_H
#define BASIC_BACKEND_H
#define GPU_ARCH_NAME "CUDA"

#include <cuda.h>
#include <cuda_runtime.h>

#include "curand_kernel.h"
#include "device_launch_parameters.h"
#include "nvrtc.h"

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)
static __device__ __forceinline__ double atomicAdd(double* address, double val)
{
    auto* address_as_ull = reinterpret_cast<unsigned long long int*>(address);
    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed;

    do
    {
        assumed = old;
        old = atomicCAS(
            address_as_ull, assumed,
            __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

#define Philox4_32_10_t curandStatePhilox4_32_10_t
#define device_rand_init curand_init
#define device_get_4_normal_distributed_random_numbers(rand_float4,   \
                                                       rand_state, i) \
    rand_float4[i] = curand_normal4(rand_state + i)

#define DEVICE_INIT_SUCCESS CUDA_SUCCESS
#define DEVICE_MALLOC_SUCCESS cudaSuccess

#define deviceInit cuInit
#define deviceGetDeviceCount cudaGetDeviceCount
#define deviceProp cudaDeviceProp
#define getDeviceProperties cudaGetDeviceProperties
#define setWorkingDevice cudaSetDevice

#define deviceMalloc cudaMalloc
#define deviceMemcpy cudaMemcpy
#define deviceMemcpyAsync cudaMemcpyAsync
#define deviceMemcpyKind cudaMemcpyKind
#define deviceMemcpyHostToHost cudaMemcpyHostToHost
#define deviceMemcpyHostToDevice cudaMemcpyHostToDevice
#define deviceMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define deviceMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define deviceMemcpyDefault cudaMemcpyDefault
#define deviceMemset(PTR, VAL, SIZE) cudaMemsetAsync(PTR, VAL, SIZE, NULL)
#define deviceFree cudaFree
#define deviceMalloc cudaMalloc

#define deviceError_t cudaError_t
#define deviceGetErrorName cudaGetErrorName
#define deviceGetErrorString cudaGetErrorString
#define deviceErrorLaunchOutOfResources cudaErrorLaunchOutOfResources
#define deviceErrorInvalidValue cudaErrorInvalidValue
#define deviceErrorInvalidConfiguration cudaErrorInvalidConfiguration
#define deviceGetLastError cudaGetLastError

#define deviceStream_t cudaStream_t
#define deviceStreamCreate cudaStreamCreate
#define deviceStreamDestroy cudaStreamDestroy
#define deviceStreamSynchronize cudaStreamSynchronize

#define Launch_Device_Kernel(kernel, grid, block, sm_memory, stream, ...) \
    kernel<<<grid, block, sm_memory, stream>>>(__VA_ARGS__)

#define FULL_MASK 0xffffffff

using device_mask_t = unsigned int;

static __device__ __forceinline__ device_mask_t deviceActiveMask()
{
    return __activemask();
}

template <typename T>
static __device__ __forceinline__ T deviceShflDown(device_mask_t mask, T value,
                                                   unsigned int delta,
                                                   int width = warpSize)
{
    return __shfl_down_sync(mask, value, delta, width);
}

template <typename T>
static __device__ __forceinline__ T deviceShfl(device_mask_t mask, T value,
                                               int src_lane,
                                               int width = warpSize)
{
    return __shfl_sync(mask, value, src_lane, width);
}

static __device__ __forceinline__ device_mask_t deviceBallot(device_mask_t mask,
                                                             int predicate)
{
    return __ballot_sync(mask, predicate);
}

static __device__ __forceinline__ int devicePopCount(device_mask_t mask)
{
    return __popc(mask);
}

static __device__ __forceinline__ int deviceFindFirstSet(device_mask_t mask)
{
    return __ffs(mask);
}

static __device__ __forceinline__ device_mask_t deviceLowerLaneMask(int lane)
{
    return lane <= 0 ? 0U : (device_mask_t{1} << lane) - 1U;
}

static __device__ __forceinline__ void deviceSyncWarp(
    device_mask_t mask = FULL_MASK)
{
    __syncwarp(mask);
}

#define hostDeviceSynchronize cudaDeviceSynchronize

#define DEVICE_JIT_COMPILER_NAME "NVRTC"
#define DEVICE_JIT_CODE_NAME "PTX"
#define deviceCompilerResult_t nvrtcResult
#define DEVICE_COMPILER_SUCCESS NVRTC_SUCCESS
#define deviceJitProgram_t nvrtcProgram
#define deviceJitCreateProgram nvrtcCreateProgram
#define deviceJitCompileProgram nvrtcCompileProgram
#define deviceJitDestroyProgram nvrtcDestroyProgram
#define deviceCompilerGetErrorString nvrtcGetErrorString
#define deviceJitGetProgramLogSize nvrtcGetProgramLogSize
#define deviceJitGetProgramLog nvrtcGetProgramLog
#define deviceJitGetCodeSize nvrtcGetPTXSize
#define deviceJitGetCode nvrtcGetPTX

#define deviceModule_t CUmodule
#define deviceFunction_t CUfunction
#define deviceModuleResult_t CUresult
#define DEVICE_MODULE_SUCCESS CUDA_SUCCESS
static inline deviceModuleResult_t deviceModuleLoad(deviceModule_t* module,
                                                    const void* image)
{
    return cuModuleLoadDataEx(module, image, 0, nullptr, nullptr);
}
#define deviceModuleGetFunction cuModuleGetFunction
static inline const char* deviceModuleGetErrorName(deviceModuleResult_t result)
{
    const char* name = nullptr;
    cuGetErrorName(result, &name);
    return name != nullptr ? name : "CUDA_ERROR_UNKNOWN";
}
static inline const char* deviceModuleGetErrorString(
    deviceModuleResult_t result)
{
    const char* reason = nullptr;
    cuGetErrorString(result, &reason);
    return reason != nullptr ? reason : "unknown CUDA driver error";
}
static inline deviceModuleResult_t deviceModuleLaunchKernel(
    deviceFunction_t function, unsigned int grid_x, unsigned int grid_y,
    unsigned int grid_z, unsigned int block_x, unsigned int block_y,
    unsigned int block_z, unsigned int shared_memory_size,
    deviceStream_t stream, void** args)
{
    return cuLaunchKernel(function, grid_x, grid_y, grid_z, block_x, block_y,
                          block_z, shared_memory_size, stream, args, nullptr);
}
#endif  // BASIC_BACKEND_H

#ifndef FFT_BACKEND_H
#define FFT_BACKEND_H

#include <cufftw.h>
#define FFT_LIBRARY_NAME "cuFFT"
#define FFT_COMPLEX cufftComplex
#define REAL(c) c.x
#define IMAGINARY(c) c.y
#define FFT_HANDLE cufftHandle
#define FFT_SUCCESS CUFFT_SUCCESS
#define FFT_RESULT cufftResult

#define FFT_TYPE cufftType
#define FFT_R2C CUFFT_R2C
#define FFT_C2R CUFFT_C2R
#define FFT_SIZE_t int

#define deviceFFTPlanMany cufftPlanMany
#define deviceFFTExecR2C cufftExecR2C
#define deviceFFTExecC2R cufftExecC2R
#define deviceFFTDestroy cufftDestroy

#endif  // FFT_BACKEND_H

#ifndef BLAS_BACKEND_H
#define BLAS_BACKEND_H

#include <cublas_v2.h>
#define BLAS_LIBRARY_NAME "cuBLAS"
#define BLAS_HANDLE cublasHandle_t
#define BLAS_SUCCESS CUBLAS_STATUS_SUCCESS

#define DEVICE_BLAS_OP_N CUBLAS_OP_N
#define DEVICE_BLAS_OP_T CUBLAS_OP_T
#define DEVICE_BLAS_OP_C CUBLAS_OP_C

#define deviceBlasCreate cublasCreate
#define deviceBlasDestroy cublasDestroy
#define deviceBlasSgeam cublasSgeam
#define deviceBlasSgemm cublasSgemm
#define deviceBlasDgemm cublasDgemm

#endif  // BLAS_BACKEND_H

#ifndef SOLVER_BACKEND_H
#define SOLVER_BACKEND_H

#include <cusolverDn.h>
#define SOLVER_LIBRARY_NAME "cuSolver"
#define SOLVER_HANDLE cusolverDnHandle_t
#define SOLVER_SUCCESS CUSOLVER_STATUS_SUCCESS

#define DEVICE_FILL_MODE_UPPER CUBLAS_FILL_MODE_UPPER
#define DEVICE_EIG_MODE_VECTOR CUSOLVER_EIG_MODE_VECTOR

#define deviceSolverCreate cusolverDnCreate
#define deviceSolverDestroy cusolverDnDestroy
#define deviceSolverSsyevdBufferSize cusolverDnSsyevd_bufferSize
#define deviceSolverSsyevd cusolverDnSsyevd
#define deviceSolverDsyevdBufferSize cusolverDnDsyevd_bufferSize
#define deviceSolverDsyevd cusolverDnDsyevd

#endif  // SOLVER_BACKEND_H

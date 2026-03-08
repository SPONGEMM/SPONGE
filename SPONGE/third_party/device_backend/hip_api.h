#ifndef BASIC_BACKEND_H
#define BASIC_BACKEND_H
#define GPU_ARCH_NAME "HIP"

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <hiprand/hiprand_kernel.h>

#define Philox4_32_10_t hiprandStatePhilox4_32_10_t
#define device_rand_init hiprand_init
#define device_get_4_normal_distributed_random_numbers(rand_float4,   \
                                                       rand_state, i) \
    rand_float4[i] = hiprand_normal4(rand_state + i)

#define DEVICE_INIT_SUCCESS hipSuccess
#define DEVICE_MALLOC_SUCCESS hipSuccess

#define deviceInit hipInit
#define deviceGetDeviceCount hipGetDeviceCount
#define deviceProp hipDeviceProp_t
#define getDeviceProperties hipGetDeviceProperties
#define setWorkingDevice hipSetDevice

#define deviceMalloc hipMalloc
#define deviceMemcpy hipMemcpy
#define deviceMemcpyAsync hipMemcpyAsync
#define deviceMemcpyKind hipMemcpyKind
#define deviceMemcpyHostToHost hipMemcpyHostToHost
#define deviceMemcpyHostToDevice hipMemcpyHostToDevice
#define deviceMemcpyDeviceToHost hipMemcpyDeviceToHost
#define deviceMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define deviceMemcpyDefault hipMemcpyDefault
#define deviceMemset(PTR, VAL, SIZE) hipMemsetAsync(PTR, VAL, SIZE, nullptr)
#define deviceFree hipFree

#define deviceError_t hipError_t
#define deviceGetErrorName hipGetErrorName
#define deviceGetErrorString hipGetErrorString
#define deviceErrorLaunchOutOfResources hipErrorLaunchOutOfResources
#define deviceErrorInvalidValue hipErrorInvalidValue
#define deviceErrorInvalidConfiguration hipErrorInvalidConfiguration
#define deviceGetLastError hipGetLastError

#define deviceStream_t hipStream_t
#define deviceStreamCreate hipStreamCreate
#define deviceStreamDestroy hipStreamDestroy
#define deviceStreamSynchronize hipStreamSynchronize

#define Launch_Device_Kernel(kernel, grid, block, sm_memory, stream, ...) \
    kernel<<<grid, block, sm_memory, stream>>>(__VA_ARGS__)

#define FULL_MASK 0xffffffffffffffffull

using device_mask_t = unsigned long long;

static __device__ __forceinline__ device_mask_t deviceActiveMask()
{
    return __ballot(1);
}

template <typename T>
static __device__ __forceinline__ T deviceShflDown(device_mask_t mask, T value,
                                                   unsigned int delta,
                                                   int width = warpSize)
{
    (void)mask;
    return __shfl_down(value, delta, width);
}

template <typename T>
static __device__ __forceinline__ T deviceShfl(device_mask_t mask, T value,
                                               int src_lane,
                                               int width = warpSize)
{
    (void)mask;
    return __shfl(value, src_lane, width);
}

static __device__ __forceinline__ device_mask_t deviceBallot(device_mask_t mask,
                                                             int predicate)
{
    return __ballot(predicate) & mask;
}

static __device__ __forceinline__ int devicePopCount(device_mask_t mask)
{
    return __popcll(mask);
}

static __device__ __forceinline__ int deviceFindFirstSet(device_mask_t mask)
{
    return __builtin_ffsll(mask);
}

static __device__ __forceinline__ device_mask_t deviceLowerLaneMask(int lane)
{
    if (lane <= 0)
    {
        return 0;
    }
    if (lane >= static_cast<int>(sizeof(device_mask_t) * 8))
    {
        return ~device_mask_t{0};
    }
    return (device_mask_t{1} << lane) - 1;
}

static __device__ __forceinline__ void deviceSyncWarp(
    device_mask_t mask = FULL_MASK)
{
    (void)mask;
    __builtin_amdgcn_wave_barrier();
    return;
}

#define hostDeviceSynchronize hipDeviceSynchronize

#define DEVICE_JIT_COMPILER_NAME "HIPRTC"
#define DEVICE_JIT_CODE_NAME "code object"
#define deviceCompilerResult_t hiprtcResult
#define DEVICE_COMPILER_SUCCESS HIPRTC_SUCCESS
#define deviceJitProgram_t hiprtcProgram
#define deviceJitCreateProgram hiprtcCreateProgram
#define deviceJitCompileProgram hiprtcCompileProgram
#define deviceJitDestroyProgram hiprtcDestroyProgram
#define deviceCompilerGetErrorString hiprtcGetErrorString
#define deviceJitGetProgramLogSize hiprtcGetProgramLogSize
#define deviceJitGetProgramLog hiprtcGetProgramLog
#define deviceJitGetCodeSize hiprtcGetCodeSize
#define deviceJitGetCode hiprtcGetCode

#define deviceModule_t hipModule_t
#define deviceFunction_t hipFunction_t
#define deviceModuleResult_t hipError_t
#define DEVICE_MODULE_SUCCESS hipSuccess
static inline deviceModuleResult_t deviceModuleLoad(deviceModule_t* module,
                                                    const void* image)
{
    return hipModuleLoadData(module, image);
}
#define deviceModuleGetFunction hipModuleGetFunction
static inline const char* deviceModuleGetErrorName(deviceModuleResult_t result)
{
    return hipGetErrorName(result);
}
static inline const char* deviceModuleGetErrorString(
    deviceModuleResult_t result)
{
    return hipGetErrorString(result);
}
static inline deviceModuleResult_t deviceModuleLaunchKernel(
    deviceFunction_t function, unsigned int grid_x, unsigned int grid_y,
    unsigned int grid_z, unsigned int block_x, unsigned int block_y,
    unsigned int block_z, unsigned int shared_memory_size,
    deviceStream_t stream, void** args)
{
    return hipModuleLaunchKernel(function, grid_x, grid_y, grid_z, block_x,
                                 block_y, block_z, shared_memory_size, stream,
                                 args, nullptr);
}
#endif  // BASIC_BACKEND_H

#ifndef FFT_BACKEND_H
#define FFT_BACKEND_H

#include <hipfft/hipfft.h>
#define FFT_LIBRARY_NAME "hipFFT"
#define FFT_COMPLEX hipfftComplex
#define REAL(c) c.x
#define IMAGINARY(c) c.y
#define FFT_HANDLE hipfftHandle
#define FFT_SUCCESS HIPFFT_SUCCESS
#define FFT_RESULT hipfftResult

#define FFT_TYPE hipfftType
#define FFT_R2C HIPFFT_R2C
#define FFT_C2R HIPFFT_C2R
#define FFT_SIZE_t int

#define deviceFFTPlanMany hipfftPlanMany
#define deviceFFTExecR2C hipfftExecR2C
#define deviceFFTExecC2R hipfftExecC2R
#define deviceFFTDestroy hipfftDestroy

#endif  // FFT_BACKEND_H

#ifndef BLAS_BACKEND_H
#define BLAS_BACKEND_H

#include <hipblas/hipblas.h>
#define BLAS_LIBRARY_NAME "hipBLAS"
#define BLAS_HANDLE hipblasHandle_t
#define BLAS_SUCCESS HIPBLAS_STATUS_SUCCESS

#define DEVICE_BLAS_OP_N HIPBLAS_OP_N
#define DEVICE_BLAS_OP_T HIPBLAS_OP_T
#define DEVICE_BLAS_OP_C HIPBLAS_OP_C

#define deviceBlasCreate hipblasCreate
#define deviceBlasDestroy hipblasDestroy
#define deviceBlasSgeam hipblasSgeam
#define deviceBlasSgemm hipblasSgemm

#endif  // BLAS_BACKEND_H

#ifndef SOLVER_BACKEND_H
#define SOLVER_BACKEND_H

#include <hipsolver/hipsolver.h>
#define SOLVER_LIBRARY_NAME "hipSOLVER"
#define SOLVER_HANDLE hipsolverHandle_t
#define SOLVER_SUCCESS HIPSOLVER_STATUS_SUCCESS

#define DEVICE_FILL_MODE_UPPER HIPSOLVER_FILL_MODE_UPPER
#define DEVICE_EIG_MODE_VECTOR HIPSOLVER_EIG_MODE_VECTOR

#define deviceSolverCreate hipsolverCreate
#define deviceSolverDestroy hipsolverDestroy
#define deviceSolverSsyevdBufferSize hipsolverSsyevd_bufferSize
#define deviceSolverSsyevd hipsolverSsyevd

#endif  // SOLVER_BACKEND_H

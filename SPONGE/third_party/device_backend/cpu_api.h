#ifndef BASIC_BACKEND_H
#define BASIC_BACKEND_H
#define CPU_ARCH_NAME "cpu"

#include "../philox.hpp"
#define Philox4_32_10_t Philox4x32_10
#define device_rand_init(seed, id, offset, state_ptr) \
    (state_ptr)[0] = Philox4_32_10_t(seed, id, offset)
#define device_get_4_normal_distributed_random_numbers(rand_float4,   \
                                                       rand_state, i) \
    rand_state[i].normal((float*)rand_float4, i)
#define __device__
#define __host__
#define __global__
#define __forceinline__ inline
#define __launch_bounds__(THREAD)

#define deviceStream_t int
#define deviceStreamCreate(stream)
#define deviceStreamDestroy(stream)
#define deviceStreamSynchronize(stream)

struct float4
{
    float x, y, z, w;
};
float rnorm3df(float, float, float);
float norm3df(float, float, float);
float erfcxf(float);
float atomicAdd(float*, float);
double atomicAdd(double*, double);
int atomicAdd(int*, int);
int atomicExch(int* address, int val);
enum deviceMemcpyKind
{
    deviceMemcpyHostToHost,
    deviceMemcpyHostToDevice,
    deviceMemcpyDeviceToHost,
    deviceMemcpyDeviceToDevice,
    deviceMemcpyDefault
};
void deviceMemcpy(void* to, const void* from, size_t size,
                  deviceMemcpyKind kind);
void deviceMemcpyAsync(void* to, const void* from, size_t size,
                       deviceMemcpyKind kind, deviceStream_t stream);
// 注意，size必须是sizeof(int)的整倍数
void deviceMemset(void* to, int val, size_t size);
void deviceFree(void* ptr);

#define Launch_Device_Kernel(kernel, grid, block, sm_memory, stream, ...) \
    kernel(__VA_ARGS__)

struct dim3
{
    unsigned int x;
    unsigned int y;
    unsigned int z;
    dim3(unsigned int ux, unsigned int uy = 1u, unsigned int uz = 1u)
        : x(ux), y(uy), z(uz) {};
};

#define warpSize 0

#endif  // BASIC_BACKEND_H

#ifndef FFT_BACKEND_H
#define FFT_BACKEND_H

#include "fftw3.h"
#ifdef USE_MKL
#include "fftw3_mkl.h"
#endif

#ifdef USE_MKL
#define FFT_LIBRARY_NAME "MKL-FFT"
#elif defined(USE_OPENBLAS)
#define FFT_LIBRARY_NAME "FFTW"
#else
#define FFT_LIBRARY_NAME "FFTW"
#endif

struct _fft_complex
{
    float r, i;
};

#define FFT_COMPLEX _fft_complex
#define REAL(c) c.r
#define IMAGINARY(c) c.i
#define FFT_HANDLE fftwf_plan
#define FFT_SUCCESS 0
#define FFT_RESULT int
#define FFT_SIZE_t int

enum FFT_TYPE
{
    FFT_R2C,
    FFT_C2R
};

#endif  // FFT_BACKEND_H

#ifndef BLAS_BACKEND_H
#define BLAS_BACKEND_H

#ifdef USE_MKL
#include <mkl.h>
#define BLAS_LIBRARY_NAME "MKL-BLAS"
#elif defined(USE_OPENBLAS)
#include <cblas.h>
#include <lapacke.h>
#define BLAS_LIBRARY_NAME "OpenBLAS"
#else
// Placeholder for other BLAS implementations
#endif

#define BLAS_HANDLE int
#define BLAS_SUCCESS 0

enum deviceBlasOperation_t
{
    DEVICE_BLAS_OP_N,
    DEVICE_BLAS_OP_T,
    DEVICE_BLAS_OP_C
};

enum deviceFillMode_t
{
    DEVICE_FILL_MODE_UPPER
};

enum deviceEigMode_t
{
    DEVICE_EIG_MODE_VECTOR
};

#define deviceBlasCreate(handle)
#define deviceBlasDestroy(handle)

#if defined(USE_MKL) || defined(USE_OPENBLAS)
#define deviceBlasSgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, \
                        ldb, C, ldc)                                          \
    do                                                                        \
    {                                                                         \
        for (int i = 0; i < (m) * (n); ++i)                                   \
            (C)[i] = (*(alpha)) * (A)[i] + (*(beta)) * (B)[i];                \
    } while (0)

#define deviceBlasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B,   \
                        ldb, beta, C, ldc)                                   \
    cblas_sgemm(CblasColMajor,                                               \
                (transa == DEVICE_BLAS_OP_N ? CblasNoTrans : CblasTrans),    \
                (transb == DEVICE_BLAS_OP_N ? CblasNoTrans : CblasTrans), m, \
                n, k, *(alpha), A, lda, B, ldb, *(beta), C, ldc)
#endif

#endif  // BLAS_BACKEND_H

#ifndef SOLVER_BACKEND_H
#define SOLVER_BACKEND_H

#ifdef USE_MKL
#include <mkl.h>
#define SOLVER_LIBRARY_NAME "MKL-SOLVER"
#elif defined(USE_OPENBLAS)
#include <lapacke.h>
#define SOLVER_LIBRARY_NAME "LAPACKE"
#endif

#define SOLVER_HANDLE int
#define SOLVER_SUCCESS 0

#define deviceSolverCreate(handle)
#define deviceSolverDestroy(handle)

#endif  // SOLVER_BACKEND_H

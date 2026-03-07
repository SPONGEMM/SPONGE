#pragma once

// -----------------线程并行迭代器定义 ------------------

#define STRINGIFY(x) #x
#define PRAGMA(x) _Pragma(#x)

// ------------------FOR 循环----------------------------
#ifdef GPU_ARCH_NAME
#define SIMPLE_DEVICE_FOR(i, N)                    \
    int i = blockDim.x * blockIdx.x + threadIdx.x; \
    if (i < N)
#else
#define SIMPLE_DEVICE_FOR(i, N) \
    PRAGMA(omp parallel for)    \
    for (int i = 0; i < N; i++)
#endif

// ------------------线程求和----------------------------------

__device__ __forceinline__ void Warp_Sum_To(int* y, int& x, int delta)
{
#ifdef GPU_ARCH_NAME
    const int width = delta;
    for (int offset = width >> 1; offset > 0; offset >>= 1)
    {
        x += deviceShflDown(FULL_MASK, x, offset, width);
    }
    if (threadIdx.x == 0)
#endif
    {
        atomicAdd(y, x);
    }
}
__device__ __forceinline__ void Warp_Sum_To(float* y, float& x, int delta)
{
#ifdef GPU_ARCH_NAME
    const int width = delta;
    for (int offset = width >> 1; offset > 0; offset >>= 1)
    {
        x += deviceShflDown(FULL_MASK, x, offset, width);
    }
    if (threadIdx.x == 0)
#endif
    {
        atomicAdd(y, x);
    }
}
__device__ __forceinline__ void Warp_Sum_To(VECTOR* y, VECTOR& x, int delta)
{
#ifdef GPU_ARCH_NAME
    const int width = delta;
    for (int offset = width >> 1; offset > 0; offset >>= 1)
    {
        x.x += deviceShflDown(FULL_MASK, x.x, offset, width);
        x.y += deviceShflDown(FULL_MASK, x.y, offset, width);
        x.z += deviceShflDown(FULL_MASK, x.z, offset, width);
    }
    if (threadIdx.x == 0)
#endif
    {
        atomicAdd(y, x);
    }
}
__device__ __forceinline__ void Warp_Sum_To(LTMatrix3* y, LTMatrix3& x,
                                            int delta)
{
#ifdef GPU_ARCH_NAME
    const int width = delta;
    for (int offset = width >> 1; offset > 0; offset >>= 1)
    {
        x.a11 += deviceShflDown(FULL_MASK, x.a11, offset, width);
        x.a21 += deviceShflDown(FULL_MASK, x.a21, offset, width);
        x.a22 += deviceShflDown(FULL_MASK, x.a22, offset, width);
        x.a31 += deviceShflDown(FULL_MASK, x.a31, offset, width);
        x.a32 += deviceShflDown(FULL_MASK, x.a32, offset, width);
        x.a33 += deviceShflDown(FULL_MASK, x.a33, offset, width);
    }
    if (threadIdx.x == 0)
#endif
    {
        atomicAdd(y, x);
    }
}

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#elif defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "../../SPONGE/utils/float_classification.hpp"

#if defined(__CUDACC__) || defined(__HIPCC__)
#define TEST_HD __host__ __device__
#else
#define TEST_HD
#endif

template <typename T>
TEST_HD int Classify(T value)
{
    return (SpongeFloat::Is_Finite(value) ? 1 : 0) |
           (SpongeFloat::Is_Nan(value) ? 2 : 0) |
           (SpongeFloat::Is_Inf(value) ? 4 : 0);
}

#if defined(__CUDACC__) || defined(__HIPCC__)
__global__ void Classify_Device(float f, double d, int* result)
{
    result[0] = Classify(f);
    result[1] = Classify(d);
}
#if defined(__HIPCC__)
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaMallocManaged hipMallocManaged
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaFree hipFree
#define cudaSuccess hipSuccess
#endif
#endif

int main(int argc, char** argv)
{
    // Each triple contains raw binary32/binary64 hexadecimal representations
    // and an independently supplied expected classification. Runtime parsing
    // preserves subnormals even when the process enables DAZ/FTZ.
    if (argc < 4 || (argc - 1) % 3 != 0) return 1;
#if defined(__CUDACC__) || defined(__HIPCC__)
    int count = 0;
    const bool have_device =
        cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
    int* result = nullptr;
    if (have_device &&
        cudaMallocManaged(&result, 2 * sizeof(int)) != cudaSuccess)
        return 1;
#endif
    for (int i = 1; i < argc; i += 3)
    {
        const auto fbits =
            static_cast<std::uint32_t>(std::strtoull(argv[i], nullptr, 16));
        const auto dbits =
            static_cast<std::uint64_t>(std::strtoull(argv[i + 1], nullptr, 16));
        const int expected = std::atoi(argv[i + 2]);
        float f;
        double d;
        std::memcpy(&f, &fbits, sizeof(f));
        std::memcpy(&d, &dbits, sizeof(d));
        // Explicit failure paths remain active in Release/NDEBUG builds.
        if (Classify(f) != expected || Classify(d) != expected)
        {
            std::fprintf(stderr, "Host classification failed: %s %s\n", argv[i],
                         argv[i + 1]);
            return 1;
        }
#if defined(__CUDACC__) || defined(__HIPCC__)
        if (have_device)
        {
            Classify_Device<<<1, 1>>>(f, d, result);
            if (cudaDeviceSynchronize() != cudaSuccess ||
                result[0] != expected || result[1] != expected)
            {
                std::fprintf(stderr, "Device classification failed: %s %s\n",
                             argv[i], argv[i + 1]);
                return 1;
            }
        }
#endif
    }
#if defined(__CUDACC__) || defined(__HIPCC__)
    if (!have_device)
    {
        std::puts("Host checks passed; device checks require an available GPU");
        return 77;
    }
    if (cudaFree(result) != cudaSuccess) return 1;
#endif
    std::puts("Floating-point classification checks passed");
    return 0;
}

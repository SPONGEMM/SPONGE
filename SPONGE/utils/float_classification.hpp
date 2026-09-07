#pragma once

#include <cstdint>
#include <cstring>
#include <limits>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define SPONGE_FLOAT_HD __host__ __device__
#else
#define SPONGE_FLOAT_HD
#endif

namespace SpongeFloat
{
static_assert(sizeof(float) == sizeof(std::uint32_t) &&
                  std::numeric_limits<float>::is_iec559 &&
                  std::numeric_limits<float>::digits == 24,
              "SPONGE requires IEEE-754 binary32 float");
static_assert(sizeof(double) == sizeof(std::uint64_t) &&
                  std::numeric_limits<double>::is_iec559 &&
                  std::numeric_limits<double>::digits == 53,
              "SPONGE requires IEEE-754 binary64 double");

// Inspect the representation without floating-point comparisons. Host integer
// barriers prevent fast-math (including LTO) from recognizing a floating-point
// classification idiom and replacing it using finite-math assumptions. CUDA
// and HIP device passes use bit-reinterpretation intrinsics; host passes must
// not select these merely because a GPU backend is enabled.
SPONGE_FLOAT_HD inline std::uint32_t Bits(float value)
{
#if defined(__CUDA_ARCH__) || \
    (defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__)
    return __float_as_uint(value);
#else
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
#if defined(__GNUC__) || defined(__clang__)
    __asm__ __volatile__("" : "+r"(bits));
    return bits;
#else
    volatile std::uint32_t observed = bits;
    return observed;
#endif
#endif
}

SPONGE_FLOAT_HD inline std::uint64_t Bits(double value)
{
#if defined(__CUDA_ARCH__) || \
    (defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__)
    return static_cast<std::uint64_t>(__double_as_longlong(value));
#else
    std::uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
#if defined(__GNUC__) || defined(__clang__)
    __asm__ __volatile__("" : "+r"(bits));
    return bits;
#else
    volatile std::uint64_t observed = bits;
    return observed;
#endif
#endif
}

SPONGE_FLOAT_HD inline bool Is_Finite(float value)
{
    return (Bits(value) & 0x7f800000U) != 0x7f800000U;
}
SPONGE_FLOAT_HD inline bool Is_Finite(double value)
{
    return (Bits(value) & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL;
}
SPONGE_FLOAT_HD inline bool Is_Nan(float value)
{
    return (Bits(value) & 0x7fffffffU) > 0x7f800000U;
}
SPONGE_FLOAT_HD inline bool Is_Nan(double value)
{
    return (Bits(value) & 0x7fffffffffffffffULL) > 0x7ff0000000000000ULL;
}
SPONGE_FLOAT_HD inline bool Is_Inf(float value)
{
    return (Bits(value) & 0x7fffffffU) == 0x7f800000U;
}
SPONGE_FLOAT_HD inline bool Is_Inf(double value)
{
    return (Bits(value) & 0x7fffffffffffffffULL) == 0x7ff0000000000000ULL;
}
}  // namespace SpongeFloat

#undef SPONGE_FLOAT_HD

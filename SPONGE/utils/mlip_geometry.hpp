#pragma once

#include <cfloat>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "float_classification.hpp"

// Geometry used only by the model-independent short-range MLIP path.
namespace MlipGeometry
{
inline float Radius(double cutoff, double skin)
{
    if (!SpongeFloat::Is_Finite(cutoff) || cutoff <= 0 ||
        !SpongeFloat::Is_Finite(skin) || skin < 0)
        throw std::invalid_argument("invalid MLIP cutoff or skin");
    volatile double sum = cutoff + skin;
    // TwoSum recovers the addition's rounding residual. Round upward only
    // when the rounded sum is below the exact sum of the public doubles.
    // Volatile intermediates prevent reassociation under SPONGE's fast-math
    // host flags. The swallowed-addend branch also covers denormal skins on
    // hosts configured to flush denormal arithmetic to zero.
    volatile double b = sum - cutoff;
    volatile double difference = sum - b;
    volatile double left = cutoff - difference;
    volatile double right = skin - b;
    volatile double residual = left + right;
    const bool positive_skin =
        SpongeFloat::Bits(skin) != 0 && (SpongeFloat::Bits(skin) >> 63) == 0;
    if (residual > 0 || (positive_skin && sum == cutoff))
        sum = std::nextafter(double(sum),
                             std::numeric_limits<double>::infinity());
    if (!SpongeFloat::Is_Finite(double(sum)) || sum > FLT_MAX)
        throw std::invalid_argument("MLIP graph radius is not representable");
    float radius = static_cast<float>(sum);
    if (double(radius) < sum) radius = std::nextafter(radius, FLT_MAX);
    return radius;
}

inline double Motion_Limit(double skin) { return skin * 0.5; }

__host__ __device__ inline double Canonical(float raw, float length)
{
    // fmod avoids the FP32 multiply/subtract in the classic wrapping path.
    double value = fmod(double(raw), double(length));
    if (value < 0) value += double(length);
    return value;
}

__host__ __device__ inline int Bin(double value, float spacing, int count)
{
    double index = floor(value / double(spacing));
    return index < 0 ? 0 : (index >= count ? count - 1 : int(index));
}

__host__ __device__ inline float Lower(double value)
{
    float result = float(value);
    if (double(result) > value) result = nextafterf(result, -FLT_MAX);
    return result;
}
__host__ __device__ inline float Upper(double value)
{
    float result = float(value);
    if (double(result) < value) result = nextafterf(result, FLT_MAX);
    return result;
}
}  // namespace MlipGeometry

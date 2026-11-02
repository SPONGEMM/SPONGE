#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "portable_philox.hpp"

// Host-side deterministic distributions backed by one addressable Philox
// stream. A module checkpoints only (seed, completed invocation count); every
// invocation receives an independent stream and therefore has no hidden
// std::normal_distribution cache to serialize.
class SPONGE_PORTABLE_PHILOX_SAMPLER
{
   public:
    SPONGE_PORTABLE_PHILOX_SAMPLER(std::uint64_t seed, std::uint64_t stream)
        : seed_(seed), stream_(stream)
    {
    }

    double Uniform_Open01()
    {
        const std::uint64_t high = Next_Word() >> 5;
        const std::uint64_t low = Next_Word() >> 6;
        const std::uint64_t mantissa = (high << 26) | low;
        return (static_cast<double>(mantissa) + 0.5) / 9007199254740992.0;
    }

    double Normal()
    {
        if (has_spare_normal_)
        {
            has_spare_normal_ = false;
            return spare_normal_;
        }
        const double magnitude = std::sqrt(-2.0 * std::log(Uniform_Open01()));
        const double phase =
            6.283185307179586476925286766559 * Uniform_Open01();
        spare_normal_ = magnitude * std::sin(phase);
        has_spare_normal_ = true;
        return magnitude * std::cos(phase);
    }

    double Gamma(double shape, double scale)
    {
        if (!(shape > 0.0) || !(scale > 0.0))
        {
            throw std::invalid_argument(
                "portable Philox gamma shape and scale must be positive");
        }
        return scale * Gamma_Unit(shape);
    }

   private:
    std::uint32_t Next_Word()
    {
        return SPONGE_PHILOX4X32_10(seed_, stream_, word_offset_++).UInt32();
    }

    double Gamma_Unit(double shape)
    {
        if (shape < 1.0)
        {
            return Gamma_Unit(shape + 1.0) *
                   std::pow(Uniform_Open01(), 1.0 / shape);
        }

        const double d = shape - 1.0 / 3.0;
        const double c = 1.0 / std::sqrt(9.0 * d);
        for (;;)
        {
            const double x = Normal();
            const double base = 1.0 + c * x;
            if (!(base > 0.0)) continue;
            const double v = base * base * base;
            const double u = Uniform_Open01();
            if (u < 1.0 - 0.0331 * x * x * x * x ||
                std::log(u) < 0.5 * x * x + d * (1.0 - v + std::log(v)))
            {
                return d * v;
            }
        }
    }

    std::uint64_t seed_ = 0;
    std::uint64_t stream_ = 0;
    std::uint64_t word_offset_ = 0;
    bool has_spare_normal_ = false;
    double spare_normal_ = 0.0;
};

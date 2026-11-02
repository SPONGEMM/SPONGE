#pragma once

#include <cstdint>

// A backend-independent Philox4x32-10 implementation used by stochastic
// modules whose state must survive H5 restart across CPU/CUDA/HIP processes.
// Random values are addressed by (seed, stream, scalar offset), so no vendor
// RNG object or rank-local device allocation belongs to the checkpoint.
struct SPONGE_PHILOX4X32_10
{
    struct uint4
    {
        std::uint32_t x, y, z, w;
    };

    struct uint2
    {
        std::uint32_t x, y;
    };

    std::uint64_t seed = 0;
    std::uint64_t stream = 0;
    std::uint64_t scalar_offset = 0;

    __host__ __device__ SPONGE_PHILOX4X32_10(std::uint64_t seed_value,
                                             std::uint64_t stream_index,
                                             std::uint64_t offset)
        : seed(seed_value), stream(stream_index), scalar_offset(offset)
    {
    }

    __host__ __device__ void Normal4(float* output) const
    {
        uint4 values = Generate_Block(scalar_offset >> 2);
        const std::uint32_t substate =
            static_cast<std::uint32_t>(scalar_offset & 3U);
        std::uint32_t words[4];
        for (std::uint32_t index = 0; index < 4; ++index)
        {
            const std::uint64_t absolute =
                static_cast<std::uint64_t>(substate) + index;
            if (absolute < 4)
            {
                words[index] =
                    Word(values, static_cast<std::uint32_t>(absolute));
            }
            else
            {
                const uint4 next = Generate_Block((scalar_offset >> 2) + 1);
                words[index] =
                    Word(next, static_cast<std::uint32_t>(absolute - 4));
            }
        }

        const float scale = 1.0f / 4294967295.0f;
        float u1 = static_cast<float>(words[0]) * scale;
        float u2 = static_cast<float>(words[1]) * scale;
        float u3 = static_cast<float>(words[2]) * scale;
        float u4 = static_cast<float>(words[3]) * scale;
        u1 = u1 > 0.0f ? sqrtf(-2.0f * logf(u1)) : 0.0f;
        u2 = 2.0f * 3.141592654f * u2;
        u3 = u3 > 0.0f ? sqrtf(-2.0f * logf(u3)) : 0.0f;
        u4 = 2.0f * 3.141592654f * u4;
        output[0] = u1 * cosf(u2);
        output[1] = u1 * sinf(u2);
        output[2] = u3 * cosf(u4);
        output[3] = u3 * sinf(u4);
    }

    __host__ __device__ std::uint32_t UInt32() const
    {
        const uint4 values = Generate_Block(scalar_offset >> 2);
        return Word(values, static_cast<std::uint32_t>(scalar_offset & 3U));
    }

   private:
    __host__ __device__ static std::uint32_t Word(const uint4& value,
                                                  std::uint32_t index)
    {
        if (index == 0) return value.x;
        if (index == 1) return value.y;
        if (index == 2) return value.z;
        return value.w;
    }

    __host__ __device__ static uint2 Multiply_High_Low(std::uint32_t a,
                                                       std::uint32_t b)
    {
        const std::uint64_t product =
            static_cast<std::uint64_t>(a) * static_cast<std::uint64_t>(b);
        return {static_cast<std::uint32_t>(product),
                static_cast<std::uint32_t>(product >> 32)};
    }

    __host__ __device__ static uint4 Round(const uint4& counter,
                                           const uint2& key)
    {
        const uint2 left = Multiply_High_Low(0xD2511F53U, counter.x);
        const uint2 right = Multiply_High_Low(0xCD9E8D57U, counter.z);
        return {right.y ^ counter.y ^ key.x, right.x,
                left.y ^ counter.w ^ key.y, left.x};
    }

    __host__ __device__ static uint2 Bump_Key(const uint2& key)
    {
        return {key.x + 0x9E3779B9U, key.y + 0xBB67AE85U};
    }

    __host__ __device__ uint4 Generate_Block(std::uint64_t block_offset) const
    {
        uint4 counter = {static_cast<std::uint32_t>(block_offset),
                         static_cast<std::uint32_t>(block_offset >> 32),
                         static_cast<std::uint32_t>(stream),
                         static_cast<std::uint32_t>(stream >> 32)};
        uint2 key = {static_cast<std::uint32_t>(seed),
                     static_cast<std::uint32_t>(seed >> 32)};
        for (int round = 0; round < 10; ++round)
        {
            counter = Round(counter, key);
            if (round != 9) key = Bump_Key(key);
        }
        return counter;
    }
};

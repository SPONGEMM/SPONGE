#ifndef SPONGE_LANE_GROUP_AVX_H
#define SPONGE_LANE_GROUP_AVX_H

#include <immintrin.h>

#include "lane_group.h"

struct LaneMask
{
    unsigned int bits;

    __host__ __device__ __forceinline__ LaneMask() : bits(0) {}
    __host__ __device__ __forceinline__ explicit LaneMask(unsigned int value)
        : bits(value)
    {
    }
};

struct LaneGroup
{
    __host__ __device__ __forceinline__ static int Width() { return 8; }

    __host__ __device__ __forceinline__ static int Lane_Id() { return 0; }

    __host__ __device__ __forceinline__ static LaneMask Active_Mask()
    {
        return LaneMask(0xFFU);
    }

    __host__ __device__ __forceinline__ static LaneMask Ballot(bool predicate)
    {
        return predicate ? Active_Mask() : LaneMask(0);
    }

    __host__ __device__ __forceinline__ static LaneMask Ballot(__m256 predicate)
    {
        return LaneMask(
            static_cast<unsigned int>(_mm256_movemask_ps(predicate)));
    }

    __host__ __device__ __forceinline__ static LaneMask Ballot(
        __m256d predicate)
    {
        return LaneMask(
            static_cast<unsigned int>(_mm256_movemask_pd(predicate)));
    }

    __host__ __device__ __forceinline__ static LaneMask And(LaneMask lhs,
                                                            LaneMask rhs)
    {
        return LaneMask(lhs.bits & rhs.bits);
    }

    __host__ __device__ __forceinline__ static LaneMask Or(LaneMask lhs,
                                                           LaneMask rhs)
    {
        return LaneMask(lhs.bits | rhs.bits);
    }

    __host__ __device__ __forceinline__ static LaneMask Not(LaneMask mask)
    {
        return LaneMask((~mask.bits) & Active_Mask().bits);
    }

    __host__ __device__ __forceinline__ static bool Any(bool predicate)
    {
        return predicate;
    }

    __host__ __device__ __forceinline__ static bool Any(LaneMask mask)
    {
        return mask.bits != 0;
    }

    __host__ __device__ __forceinline__ static bool All(bool predicate)
    {
        return predicate;
    }

    __host__ __device__ __forceinline__ static bool All(LaneMask mask)
    {
        return mask.bits == Active_Mask().bits;
    }

    __host__ __device__ __forceinline__ static int Count(LaneMask mask)
    {
        return LaneGroup_PopCount(mask.bits);
    }

    __host__ __device__ __forceinline__ static int First_Lane(LaneMask mask)
    {
        return mask.bits == 0 ? -1 : __builtin_ctz(mask.bits);
    }

    __host__ __device__ __forceinline__ static LaneMask Lower_Lane_Mask()
    {
        return LaneMask(0);
    }

    __host__ __device__ __forceinline__ static int Prefix_Count(LaneMask mask)
    {
        return Count(And(mask, Lower_Lane_Mask()));
    }

    template <typename T>
    __host__ __device__ __forceinline__ static T Broadcast(T value,
                                                           int src_lane)
    {
        (void)src_lane;
        return value;
    }

    template <typename T>
    __host__ __device__ __forceinline__ static T Shuffle_Down(T value,
                                                              int delta)
    {
        (void)delta;
        return value;
    }

    template <typename T>
    __host__ __device__ __forceinline__ static T Reduce_Sum(T value)
    {
        return value;
    }

    template <typename T>
    __host__ __device__ __forceinline__ static T Reduce_Sum(T value, int width)
    {
        (void)width;
        return value;
    }

    __host__ __device__ __forceinline__ static float Reduce_Sum(__m256 value)
    {
        alignas(32) float lanes[8];
        _mm256_storeu_ps(lanes, value);
        float sum = 0.0f;
        for (int i = 0; i < 8; ++i)
        {
            sum += lanes[i];
        }
        return sum;
    }

    __host__ __device__ __forceinline__ static double Reduce_Sum(__m256d value)
    {
        alignas(32) double lanes[4];
        _mm256_storeu_pd(lanes, value);
        double sum = 0.0;
        for (int i = 0; i < 4; ++i)
        {
            sum += lanes[i];
        }
        return sum;
    }
};

#endif  // SPONGE_LANE_GROUP_AVX_H

#ifndef SPONGE_LANE_GROUP_SSE42_H
#define SPONGE_LANE_GROUP_SSE42_H

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
    __host__ __device__ __forceinline__ static int Width() { return 4; }

    __host__ __device__ __forceinline__ static int Lane_Id() { return 0; }

    __host__ __device__ __forceinline__ static LaneMask Active_Mask()
    {
        return LaneMask(0xFU);
    }

    __host__ __device__ __forceinline__ static LaneMask Ballot(bool predicate)
    {
        return predicate ? Active_Mask() : LaneMask(0);
    }

    __host__ __device__ __forceinline__ static LaneMask Ballot(__m128 predicate)
    {
        return LaneMask(static_cast<unsigned int>(_mm_movemask_ps(predicate)));
    }

    __host__ __device__ __forceinline__ static LaneMask Ballot(
        __m128d predicate)
    {
        return LaneMask(static_cast<unsigned int>(_mm_movemask_pd(predicate)));
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

    __host__ __device__ __forceinline__ static float Reduce_Sum(__m128 value)
    {
        alignas(16) float lanes[4];
        _mm_storeu_ps(lanes, value);
        return lanes[0] + lanes[1] + lanes[2] + lanes[3];
    }

    __host__ __device__ __forceinline__ static double Reduce_Sum(__m128d value)
    {
        alignas(16) double lanes[2];
        _mm_storeu_pd(lanes, value);
        return lanes[0] + lanes[1];
    }
};

#endif  // SPONGE_LANE_GROUP_SSE42_H

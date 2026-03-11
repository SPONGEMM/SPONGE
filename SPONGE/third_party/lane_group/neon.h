#ifndef SPONGE_LANE_GROUP_NEON_H
#define SPONGE_LANE_GROUP_NEON_H

#include <arm_neon.h>

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

    __host__ __device__ __forceinline__ static LaneMask Ballot(
        uint32x4_t predicate)
    {
        unsigned int bits = 0;
        bits |= (vgetq_lane_u32(predicate, 0) >> 31) << 0;
        bits |= (vgetq_lane_u32(predicate, 1) >> 31) << 1;
        bits |= (vgetq_lane_u32(predicate, 2) >> 31) << 2;
        bits |= (vgetq_lane_u32(predicate, 3) >> 31) << 3;
        return LaneMask(bits);
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

    __host__ __device__ __forceinline__ static float Reduce_Sum(
        float32x4_t value)
    {
#if defined(__aarch64__)
        return vaddvq_f32(value);
#else
        float32x2_t sum2 = vadd_f32(vget_low_f32(value), vget_high_f32(value));
        sum2 = vpadd_f32(sum2, sum2);
        return vget_lane_f32(sum2, 0);
#endif
    }

#if defined(__aarch64__)
    __host__ __device__
        __forceinline__ static double Reduce_Sum(float64x2_t value)
    {
        return vaddvq_f64(value);
    }
#endif
};

#endif  // SPONGE_LANE_GROUP_NEON_H

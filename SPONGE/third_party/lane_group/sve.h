#ifndef SPONGE_LANE_GROUP_SVE_H
#define SPONGE_LANE_GROUP_SVE_H

#include <arm_sve.h>

#include "lane_group.h"

struct LaneMask
{
    svbool_t bits;

    __host__ __device__ __forceinline__ LaneMask() : bits(svpfalse_b()) {}

    __host__ __device__ __forceinline__ explicit LaneMask(svbool_t value)
        : bits(value)
    {
    }
};

struct LaneGroup
{
    __host__ __device__ __forceinline__ static int Width() { return svcntw(); }

    // SVE lanes are vector lanes rather than per-thread lanes. Keep this
    // conservative until a lane-local API is introduced for CPU vector kernels.
    __host__ __device__ __forceinline__ static int Lane_Id() { return 0; }

    __host__ __device__ __forceinline__ static LaneMask Active_Mask()
    {
        return LaneMask(svptrue_b32());
    }

    __host__ __device__ __forceinline__ static LaneMask Ballot(bool predicate)
    {
        return predicate ? Active_Mask() : LaneMask(svpfalse_b());
    }

    __host__ __device__ __forceinline__ static LaneMask Ballot(
        svbool_t predicate)
    {
        return LaneMask(predicate);
    }

    __host__ __device__ __forceinline__ static LaneMask And(LaneMask lhs,
                                                            LaneMask rhs)
    {
        return LaneMask(svand_z(svptrue_b32(), lhs.bits, rhs.bits));
    }

    __host__ __device__ __forceinline__ static LaneMask Or(LaneMask lhs,
                                                           LaneMask rhs)
    {
        return LaneMask(svorr_z(svptrue_b32(), lhs.bits, rhs.bits));
    }

    __host__ __device__ __forceinline__ static LaneMask Not(LaneMask mask)
    {
        return LaneMask(svnot_z(svptrue_b32(), mask.bits));
    }

    __host__ __device__ __forceinline__ static bool Any(bool predicate)
    {
        return predicate;
    }

    __host__ __device__ __forceinline__ static bool Any(LaneMask mask)
    {
        return svptest_any(svptrue_b32(), mask.bits);
    }

    __host__ __device__ __forceinline__ static bool All(bool predicate)
    {
        return predicate;
    }

    __host__ __device__ __forceinline__ static bool All(LaneMask mask)
    {
        return svptest_all(svptrue_b32(), mask.bits);
    }

    __host__ __device__ __forceinline__ static int Count(LaneMask mask)
    {
        return svcntp_b32(svptrue_b32(), mask.bits);
    }

    __host__ __device__ __forceinline__ static int First_Lane(LaneMask mask)
    {
        if (!Any(mask))
        {
            return -1;
        }

        svint32_t lane_ids = svindex_s32(0, 1);
        svint32_t sentinel = svdup_n_s32(Width());
        svint32_t selected = svsel_s32(mask.bits, lane_ids, sentinel);
        return svminv_s32(svptrue_b32(), selected);
    }

    // Lower-lane and prefix semantics depend on a lane-local scalar context,
    // which the current CPU vector API does not expose yet.
    __host__ __device__ __forceinline__ static LaneMask Lower_Lane_Mask()
    {
        return LaneMask(svpfalse_b());
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
        svfloat32_t value)
    {
        return svaddv_f32(svptrue_b32(), value);
    }

    __host__ __device__ __forceinline__ static double Reduce_Sum(
        svfloat64_t value)
    {
        return svaddv_f64(svptrue_b64(), value);
    }
};

#endif  // SPONGE_LANE_GROUP_SVE_H

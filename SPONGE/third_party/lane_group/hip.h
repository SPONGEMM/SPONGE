#ifndef SPONGE_LANE_GROUP_HIP_H
#define SPONGE_LANE_GROUP_HIP_H

#include "lane_group.h"

struct LaneMask
{
    device_mask_t bits;

    __host__ __device__ __forceinline__ LaneMask() : bits(0) {}
    __host__ __device__ __forceinline__ explicit LaneMask(device_mask_t value)
        : bits(value)
    {
    }
};

struct LaneGroup
{
    __device__ __forceinline__ static int Width() { return warpSize; }

    __device__ __forceinline__ static int Lane_Id()
    {
        return threadIdx.x & (warpSize - 1);
    }

    __device__ __forceinline__ static LaneMask Active_Mask()
    {
        return LaneMask(deviceActiveMask());
    }

    __device__ __forceinline__ static LaneMask Ballot(bool predicate)
    {
        LaneMask active = Active_Mask();
        return LaneMask(deviceBallot(active.bits, predicate ? 1 : 0));
    }

    __device__ __forceinline__ static LaneMask And(LaneMask lhs, LaneMask rhs)
    {
        return LaneMask(lhs.bits & rhs.bits);
    }

    __device__ __forceinline__ static LaneMask Or(LaneMask lhs, LaneMask rhs)
    {
        return LaneMask(lhs.bits | rhs.bits);
    }

    __device__ __forceinline__ static LaneMask Not(LaneMask mask)
    {
        LaneMask active = Active_Mask();
        return LaneMask((~mask.bits) & active.bits);
    }

    __device__ __forceinline__ static bool Any(bool predicate)
    {
        return Any(Ballot(predicate));
    }

    __device__ __forceinline__ static bool Any(LaneMask mask)
    {
        return mask.bits != 0;
    }

    __device__ __forceinline__ static bool All(bool predicate)
    {
        LaneMask active = Active_Mask();
        return Ballot(predicate).bits == active.bits;
    }

    __device__ __forceinline__ static bool All(LaneMask mask)
    {
        LaneMask active = Active_Mask();
        return mask.bits == active.bits;
    }

    __device__ __forceinline__ static int Count(LaneMask mask)
    {
        return devicePopCount(mask.bits);
    }

    __device__ __forceinline__ static int First_Lane(LaneMask mask)
    {
        return mask.bits == 0 ? -1 : (deviceFindFirstSet(mask.bits) - 1);
    }

    __device__ __forceinline__ static LaneMask Lower_Lane_Mask()
    {
        return LaneMask(deviceLowerLaneMask(Lane_Id()));
    }

    __device__ __forceinline__ static int Prefix_Count(LaneMask mask)
    {
        return Count(And(mask, Lower_Lane_Mask()));
    }

    template <typename T>
    __device__ __forceinline__ static T Broadcast(T value, int src_lane)
    {
        LaneMask active = Active_Mask();
        return deviceShfl(active.bits, value, src_lane, Width());
    }

    template <typename T>
    __device__ __forceinline__ static T Shuffle_Down(T value, int delta)
    {
        LaneMask active = Active_Mask();
        return deviceShflDown(active.bits, value, delta, Width());
    }

    template <typename T>
    __device__ __forceinline__ static T Reduce_Sum(T value)
    {
        LaneMask active = Active_Mask();
        for (int offset = Width() / 2; offset > 0; offset >>= 1)
        {
            value += deviceShflDown(active.bits, value, offset, Width());
        }
        return value;
    }

    template <typename T>
    __device__ __forceinline__ static T Reduce_Sum(T value, int width)
    {
        LaneMask active = Active_Mask();
        for (int offset = width / 2; offset > 0; offset >>= 1)
        {
            value += deviceShflDown(active.bits, value, offset, width);
        }
        return value;
    }
};

#endif  // SPONGE_LANE_GROUP_HIP_H

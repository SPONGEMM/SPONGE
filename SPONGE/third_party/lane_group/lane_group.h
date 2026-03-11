#ifndef SPONGE_LANE_GROUP_H
#define SPONGE_LANE_GROUP_H

#include <stdint.h>

#ifdef USE_HIP
#include "../device_backend/hip_api.h"
#elif defined(USE_CUDA)
#include "../device_backend/cuda_api.h"
#else
#include "../device_backend/cpu_api.h"
#endif

template <typename T>
static __host__ __device__ __forceinline__ int LaneGroup_PopCount(T mask)
{
    int count = 0;
    while (mask != 0)
    {
        count += static_cast<int>(mask & static_cast<T>(1));
        mask >>= 1;
    }
    return count;
}

#endif  // SPONGE_LANE_GROUP_H

#pragma once

#include "../common.h"

// DD remaps exclusion indices but preserves the original row direction.
// A local i < j pair may therefore be stored in either row after packing.
static __host__ __device__ __forceinline__ bool No_PBC_Pair_Excluded(
    int i, int j, const int* starts, const int* list, const int* counts)
{
    for (int k = 0; k < counts[i]; ++k)
    {
        if (list[starts[i] + k] == j) return true;
    }
    for (int k = 0; k < counts[j]; ++k)
    {
        if (list[starts[j] + k] == i) return true;
    }
    return false;
}

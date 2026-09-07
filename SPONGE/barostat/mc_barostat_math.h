#pragma once

#include "../common.h"

__host__ __device__ __forceinline__ LTMatrix3
Get_Reverse_Diagonal_Box_Change(const LTMatrix3 g, const float dt)
{
    LTMatrix3 reverse_g;
    reverse_g.a11 = -g.a11 / (1.0f + dt * g.a11);
    reverse_g.a22 = -g.a22 / (1.0f + dt * g.a22);
    reverse_g.a33 = -g.a33 / (1.0f + dt * g.a33);
    return reverse_g;
}

#pragma once

#include "../common.h"

struct PME_GRID_COORDINATE
{
    int index;
    float fraction;
};

// Input is a finite periodic fraction after x - floor(x). In float precision
// that operation can round a tiny negative input to exactly 1, not just [0, 1).
__host__ __device__ inline PME_GRID_COORDINATE PME_Get_Grid_Coordinate(
    float fraction, int size)
{
    float scaled = fraction * size;
    // Wrap before choosing both the interpolation anchor and its local offset.
    if (scaled >= size) scaled = 0.0f;
    int index = (int)scaled;
    index = index < 0 ? 0 : (index < size ? index : size - 1);
    float offset = scaled - index;
    offset -= floorf(offset);
    return {index, offset};
}

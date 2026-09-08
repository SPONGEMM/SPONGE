#ifndef SPONGE_COMMON_BOUNDARY_H
#define SPONGE_COMMON_BOUNDARY_H

#include <cstdint>
#include <type_traits>

#include "vector.hpp"

enum class BoundaryPolicy : std::uint8_t
{
    Open = 0,
    Periodic = 1
};

struct Boundary
{
    BoundaryPolicy policy = BoundaryPolicy::Open;
    LTMatrix3 cell;
    LTMatrix3 rcell;
};

__host__ __device__ __forceinline__ Boundary
Scale_Boundary(Boundary boundary, float coordinate_scale)
{
    boundary.cell = coordinate_scale * boundary.cell;
    boundary.rcell = (1.0f / coordinate_scale) * boundary.rcell;
    return boundary;
}

template <BoundaryPolicy policy>
__host__ __device__ __forceinline__ VECTOR
Get_Displacement(VECTOR a, VECTOR b, const Boundary& boundary)
{
    const VECTOR dr = a - b;
    if (policy == BoundaryPolicy::Open)
    {
        return dr;
    }
    return dr - floorf(dr * boundary.rcell + 0.5f) * boundary.cell;
}

__host__ __device__ __forceinline__ VECTOR
Get_Displacement(VECTOR a, VECTOR b, const Boundary& boundary)
{
    if (boundary.policy == BoundaryPolicy::Open) return a - b;
    return Get_Displacement<BoundaryPolicy::Periodic>(a, b, boundary);
}

__host__ __device__ __forceinline__ VECTOR
Wrap_Coordinate(VECTOR coordinate, const Boundary& boundary)
{
    if (boundary.policy == BoundaryPolicy::Open) return coordinate;
    return coordinate - floorf(coordinate * boundary.rcell) * boundary.cell;
}

static_assert(std::is_trivially_copyable<Boundary>::value,
              "Boundary must be trivially copyable");

#endif

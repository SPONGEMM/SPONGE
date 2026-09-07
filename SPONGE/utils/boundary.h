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

__host__ __device__ __forceinline__ LTMatrix3
Normalize_Near_Orthogonal_Cell(LTMatrix3 cell, const float tolerance = 1.0e-3f)
{
    cell.a21 = fabsf(cell.a21) < tolerance ? 0.0f : cell.a21;
    cell.a31 = fabsf(cell.a31) < tolerance ? 0.0f : cell.a31;
    cell.a32 = fabsf(cell.a32) < tolerance ? 0.0f : cell.a32;
    return cell;
}

__host__ __device__ __forceinline__ LTMatrix3
Invert_Lower_Triangular_Cell(const LTMatrix3 cell)
{
    LTMatrix3 rcell;
    rcell.a11 = 1.0f / cell.a11;
    rcell.a22 = 1.0f / cell.a22;
    rcell.a33 = 1.0f / cell.a33;
    rcell.a21 = -cell.a21 / (cell.a11 * cell.a22);
    rcell.a31 = (cell.a21 * cell.a32 - cell.a22 * cell.a31) /
                (cell.a11 * cell.a22 * cell.a33);
    rcell.a32 = -cell.a32 / (cell.a22 * cell.a33);
    return rcell;
}

__host__ __device__ __forceinline__ void Get_Cell_Lengths_And_Angles(
    const LTMatrix3 cell, VECTOR* lengths, VECTOR* angles)
{
    const VECTOR va = {cell.a11, 0.0f, 0.0f};
    const VECTOR vb = {cell.a21, cell.a22, 0.0f};
    const VECTOR vc = {cell.a31, cell.a32, cell.a33};
    lengths->x = sqrtf(va * va);
    lengths->y = sqrtf(vb * vb);
    lengths->z = sqrtf(vc * vc);
    const float cos_alpha =
        fmaxf(-1.0f, fminf(1.0f, vb * vc / (lengths->y * lengths->z)));
    const float cos_beta =
        fmaxf(-1.0f, fminf(1.0f, va * vc / (lengths->x * lengths->z)));
    const float cos_gamma =
        fmaxf(-1.0f, fminf(1.0f, va * vb / (lengths->x * lengths->y)));
    angles->x = acosf(cos_alpha);
    angles->y = acosf(cos_beta);
    angles->z = acosf(cos_gamma);
}

#endif

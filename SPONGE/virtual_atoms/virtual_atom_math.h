#pragma once

#include "../common.h"

constexpr float VIRTUAL_ATOM_GEOMETRY_EPSILON = 1.0e-6f;

__device__ __forceinline__ void Virtual_Atom_Add_Source_Force(
    VECTOR* destination, VECTOR contribution)
{
    atomicAdd(&destination->x, contribution.x);
    atomicAdd(&destination->y, contribution.y);
    atomicAdd(&destination->z, contribution.z);
}

__host__ __device__ __forceinline__ VECTOR
Virtual_Atom_Type_0_Position(VECTOR source, float h)
{
    source.z = 2.0f * h - source.z;
    return source;
}

__host__ __device__ __forceinline__ VECTOR
Virtual_Atom_Type_0_Source_Force(VECTOR force_v)
{
    return {force_v.x, force_v.y, -force_v.z};
}

__host__ __device__ __forceinline__ VECTOR
Virtual_Atom_Type_1_Position(VECTOR r1, VECTOR r2, float a, Boundary boundary)
{
    return r1 + a * Get_Displacement(r2, r1, boundary);
}

__host__ __device__ __forceinline__ void Virtual_Atom_Type_1_Source_Forces(
    VECTOR force_v, float a, VECTOR* force_1, VECTOR* force_2)
{
    *force_1 = (1.0f - a) * force_v;
    *force_2 = a * force_v;
}

__host__ __device__ __forceinline__ VECTOR Virtual_Atom_Type_2_Position(
    VECTOR r1, VECTOR r2, VECTOR r3, float a, float b, Boundary boundary)
{
    return r1 + a * Get_Displacement(r2, r1, boundary) +
           b * Get_Displacement(r3, r1, boundary);
}

__host__ __device__ __forceinline__ void Virtual_Atom_Type_2_Source_Forces(
    VECTOR force_v, float a, float b, VECTOR* force_1, VECTOR* force_2,
    VECTOR* force_3)
{
    *force_1 = (1.0f - a - b) * force_v;
    *force_2 = a * force_v;
    *force_3 = b * force_v;
}

__host__ __device__ __forceinline__ bool Virtual_Atom_Type_3_Position(
    VECTOR r1, VECTOR r2, VECTOR r3, float d, float k, Boundary boundary,
    VECTOR* position)
{
    const VECTOR r21 = Get_Displacement(r2, r1, boundary);
    const VECTOR r32 = Get_Displacement(r3, r2, boundary);
    const VECTOR direction = r21 + k * r32;
    const float norm2 = direction * direction;
    if (fabsf(d) <= VIRTUAL_ATOM_GEOMETRY_EPSILON)
    {
        *position = r1;
        return true;
    }
    if (norm2 <= VIRTUAL_ATOM_GEOMETRY_EPSILON * VIRTUAL_ATOM_GEOMETRY_EPSILON)
    {
        *position = r1;
        return false;
    }
    *position = r1 + (d / sqrtf(norm2)) * direction;
    return true;
}

__host__ __device__ __forceinline__ bool Virtual_Atom_Type_3_Source_Forces(
    VECTOR r1, VECTOR r2, VECTOR r3, VECTOR force_v, float d, float k,
    Boundary boundary, VECTOR* force_1, VECTOR* force_2, VECTOR* force_3)
{
    if (fabsf(d) <= VIRTUAL_ATOM_GEOMETRY_EPSILON)
    {
        *force_1 = force_v;
        *force_2 = VECTOR(0.0f);
        *force_3 = VECTOR(0.0f);
        return true;
    }
    const VECTOR r21 = Get_Displacement(r2, r1, boundary);
    const VECTOR r32 = Get_Displacement(r3, r2, boundary);
    const VECTOR direction = r21 + k * r32;
    const float norm2 = direction * direction;
    if (norm2 <= VIRTUAL_ATOM_GEOMETRY_EPSILON * VIRTUAL_ATOM_GEOMETRY_EPSILON)
    {
        *force_1 = VECTOR(0.0f);
        *force_2 = VECTOR(0.0f);
        *force_3 = VECTOR(0.0f);
        return false;
    }
    const float inverse_norm = 1.0f / sqrtf(norm2);
    const VECTOR unit = inverse_norm * direction;
    const VECTOR perpendicular = force_v - (unit * force_v) * unit;
    const VECTOR correction = (d * inverse_norm) * perpendicular;
    *force_1 = force_v - correction;
    *force_2 = (1.0f - k) * correction;
    *force_3 = k * correction;
    return true;
}

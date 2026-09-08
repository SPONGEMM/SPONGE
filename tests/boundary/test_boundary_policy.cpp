#include <cmath>
#include <iostream>

#include "common.h"

namespace
{
bool Near(const float actual, const float expected)
{
    return std::fabs(actual - expected) < 1.0e-6f;
}

bool Check_Vector(const char* label, const VECTOR actual, const VECTOR expected)
{
    if (Near(actual.x, expected.x) && Near(actual.y, expected.y) &&
        Near(actual.z, expected.z))
    {
        return true;
    }
    std::cerr << label << " mismatch: actual=(" << actual.x << ", " << actual.y
              << ", " << actual.z << "), expected=(" << expected.x << ", "
              << expected.y << ", " << expected.z << ")\n";
    return false;
}
}  // namespace

#ifdef USE_CUDA
static __global__ void Boundary_Device_Probe(Boundary boundary, VECTOR* result)
{
    const VECTOR a{9.0f, 0.0f, 0.0f}, b{1.0f, 0.0f, 0.0f};
    result[0] = Get_Displacement(a, b, boundary);
    result[1] = Get_Displacement<BoundaryPolicy::Periodic>(a, b, boundary);
    result[2] = Get_Displacement<BoundaryPolicy::Open>(a, b, boundary);
}

static bool Check_Device_Boundary()
{
    Boundary boundary{BoundaryPolicy::Periodic, LTMatrix3(10, 0, 10, 0, 0, 10),
                      LTMatrix3(0.1f, 0, 0.1f, 0, 0, 0.1f)};
    VECTOR* device_result = NULL;
    if (cudaMalloc((void**)&device_result, 3 * sizeof(VECTOR)) != cudaSuccess)
        return false;
    bool ok = true;
    for (int stage = 0; stage < 3; stage++)
    {
        if (stage == 1)
        {
            boundary.cell.a11 = 20.0f;
            boundary.rcell.a11 = 0.05f;
        }
        if (stage == 2)
        {
            boundary.policy = BoundaryPolicy::Open;
            boundary.cell.a11 = 10.0f;
            boundary.rcell.a11 = 0.1f;
        }
        Boundary_Device_Probe<<<1, 1>>>(boundary, device_result);
        VECTOR result[3];
        if (cudaGetLastError() != cudaSuccess ||
            cudaMemcpy(result, device_result, sizeof(result),
                       cudaMemcpyDeviceToHost) != cudaSuccess)
        {
            ok = false;
            break;
        }
        const VECTOR expected{stage == 0 ? -2.0f : 8.0f, 0, 0};
        ok &= Check_Vector("device dynamic boundary", result[0], expected);
        ok &= Check_Vector("device periodic boundary", result[1],
                           {stage == 1 ? 8.0f : -2.0f, 0, 0});
        ok &= Check_Vector("device open boundary", result[2], {8, 0, 0});
    }
    return cudaFree(device_result) == cudaSuccess && ok;
}
#endif

int main()
{
    const LTMatrix3 cell(10.0f, 0.0f, 20.0f, 0.0f, 0.0f, 30.0f);
    const LTMatrix3 rcell(0.1f, 0.0f, 0.05f, 0.0f, 0.0f, 1.0f / 30.0f);
    Boundary boundary{BoundaryPolicy::Periodic, cell, rcell};
    const Boundary open{BoundaryPolicy::Open, cell, rcell};
    const VECTOR a{9.0f, 0.0f, 0.0f};
    const VECTOR b{1.0f, 0.0f, 0.0f};

    bool ok = true;
    ok &= Check_Vector("open displacement", Get_Displacement(a, b, open),
                       {8.0f, 0.0f, 0.0f});
    ok &= Check_Vector("periodic displacement",
                       Get_Displacement(a, b, boundary), {-2.0f, 0.0f, 0.0f});
    ok &=
        Check_Vector("periodic-only displacement",
                     Get_Displacement<BoundaryPolicy::Periodic>(a, b, boundary),
                     {-2.0f, 0.0f, 0.0f});
    ok &= Check_Vector("coordinate wrapping",
                       Wrap_Coordinate({-1.0f, 21.0f, 61.0f}, boundary),
                       {9.0f, 1.0f, 1.0f});

    ok &= Check_Vector("open coordinate unchanged",
                       Wrap_Coordinate({-1.0f, 21.0f, 61.0f}, open),
                       {-1.0f, 21.0f, 61.0f});
    ok &= Check_Vector("compile-time open displacement",
                       Get_Displacement<BoundaryPolicy::Open>(a, b, boundary),
                       {8.0f, 0.0f, 0.0f});
    // Consumers bind the owner, not a cached copy of its old matrices.
    const Boundary& consumer = boundary;
    boundary.cell.a11 = 20.0f;
    boundary.rcell.a11 = 0.05f;
    ok &= Check_Vector("updated box", Get_Displacement(a, b, consumer),
                       {8.0f, 0.0f, 0.0f});

    const UNSIGNED_INT_VECTOR mesh_a{5, 7, 9};
    const UNSIGNED_INT_VECTOR mesh_b{2, 3, 4};
    const VECTOR mesh_scale{0.5f, 1.0f, 2.0f};
    ok &= Check_Vector("mesh index displacement",
                       Get_Mesh_Index_Displacement(mesh_a, mesh_b, mesh_scale),
                       {1.5f, 4.0f, 10.0f});

#ifdef USE_CUDA
    ok &= Check_Device_Boundary();
#endif
    return ok ? 0 : 1;
}

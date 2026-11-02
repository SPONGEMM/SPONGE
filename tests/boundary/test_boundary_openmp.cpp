#include <cmath>
#include <iostream>

#include "common.h"

// Keep the by-value, shared Boundary and both inlined displacement calls:
// MSVC 19.39 /O2 /openmp:llvm miscompiled the second z subtraction here.
static void Probe(const VECTOR* coordinates, Boundary boundary, float* angles)
{
#pragma omp parallel for
    for (int i = 0; i < 16; ++i)
    {
        const VECTOR a =
            Get_Displacement(coordinates[0], coordinates[1], boundary);
        const VECTOR b =
            Get_Displacement(coordinates[2], coordinates[1], boundary);
        angles[i] = acosf((a * b) * sqrtf(1.0f / (a * a) * (1.0f / (b * b))));
    }
}

int main()
{
    const Boundary boundary{BoundaryPolicy::Open,
                            LTMatrix3(10, 0, 10, 0, 0, 10),
                            LTMatrix3(.1f, 0, .1f, 0, 0, .1f)};
    for (const float shift : {0.0f, 476.0f})
    {
        const VECTOR coordinates[3] = {
            {1, 0, shift}, {0, 0, shift}, {1, 1, shift}};
        float angles[16]{};
        Probe(coordinates, boundary, angles);
        for (const float angle : angles)
        {
            if (!std::isfinite(angle) ||
                std::fabs(angle - .785398163f) > 1.e-5f)
            {
                std::cerr << "shift=" << shift << ", angle=" << angle << '\n';
                return 1;
            }
        }
    }
    return 0;
}

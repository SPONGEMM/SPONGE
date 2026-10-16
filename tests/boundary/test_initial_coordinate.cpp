#include <cmath>
#include <cstring>
#include <iostream>

#include "MD_core/initial_coordinate.hpp"

static int failures = 0;
static void Check(const char* name, VECTOR actual, VECTOR expected, bool exact)
{
    const bool ok = exact ? std::memcmp(&actual, &expected, sizeof(VECTOR)) == 0
                          : std::fabs(actual.x - expected.x) < 2e-6f &&
                                std::fabs(actual.y - expected.y) < 2e-6f &&
                                std::fabs(actual.z - expected.z) < 2e-6f;
    if (!ok)
    {
        std::cerr << name << ": " << actual.x << "," << actual.y << ","
                  << actual.z << " != " << expected.x << "," << expected.y
                  << "," << expected.z << "\n";
        ++failures;
    }
}
static VECTOR Wrap(VECTOR crd, LTMatrix3 cell)
{
    Boundary b{BoundaryPolicy::Periodic, cell,
               Invert_Lower_Triangular_Cell(cell)};
    Move_Crd_Nearest_From_Connectivity({{0}}, {}, &crd, b, {});
    return crd;
}
int main()
{
    const LTMatrix3 box(86.88f, 0, 43.44f, 0, 0, 43.44f);
    const VECTOR inside{78.7425308227539f, 12.34567f, -0.0f};
    Check("interior bitwise", Wrap(inside, box), inside, true);
    const VECTOR below{std::nextafter(box.a11, 0.0f),
                       std::nextafter(box.a22, 0.0f),
                       std::nextafter(box.a33, 0.0f)};
    Check("one ULP below faces", Wrap(below, box), below, true);
    const LTMatrix3 orth(8, 0, 16, 0, 0, 32);
    Check("exact upper faces", Wrap({8, 16, 32}, orth), {0, 0, 0}, true);
    const float above = std::nextafter(8.0f, 9.0f);
    Check("one ULP above face", Wrap({above, 1, 2}, orth), {above - 8.0f, 1, 2},
          true);
    const float below_zero = std::nextafter(0.0f, -1.0f);
    // An infinitesimal negative input translates to a value rounding to L in
    // FP32. Preserve that lattice translation; do not add a non-lattice clamp.
    Check("one ULP below zero", Wrap({below_zero, 1, 2}, orth), {8.0f, 1, 2},
          true);
    Check("multiple boxes", Wrap({25, -30, 99}, orth), {1, 2, 3}, true);
    Check("negative fraction", Wrap({-.25f, 1, 2}, orth), {7.75f, 1, 2}, true);
    const LTMatrix3 tilted(8, 2, 8, -1, 2, 8);
    const VECTOR inner{1.125f, 2.25f, 3.5f};
    Check("triclinic interior", Wrap(inner, tilted), inner, true);
    const VECTOR shifted = inner + VECTOR{2, -3, 4} * tilted;
    Check("triclinic coupled images", Wrap(shifted, tilted), inner, true);
    Check("triclinic upper corner", Wrap(VECTOR{1, 1, 1} * tilted, tilted),
          {0, 0, 0}, true);
    const VECTOR tilt_below{std::nextafter(8.0f, 0.0f), 0, 0};
    Check("triclinic one ULP below", Wrap(tilt_below, tilted), tilt_below,
          true);
    Boundary b{BoundaryPolicy::Periodic, orth,
               Invert_Lower_Triangular_Cell(orth)};
    VECTOR connected[] = {
        {8.25f, 1, 2}, {7.75f, 1, 2}, {7.5f, 1, 2}, {17.25f, 3, 4}};
    CONECT links{{0, {1}}, {1, {0, 2}}, {2, {1}}};
    Move_Crd_Nearest_From_Connectivity({{0, 1, 2}, {3}, {}}, links, connected,
                                       b, {});
    Check("connected anchor", connected[0], {.25f, 1, 2}, true);
    Check("connected neighbor across face", connected[1], {-.25f, 1, 2}, true);
    Check("connected propagation", connected[2], {-.5f, 1, 2}, true);
    Check("independent fragment", connected[3], {1.25f, 3, 4}, true);
    return failures != 0;
}

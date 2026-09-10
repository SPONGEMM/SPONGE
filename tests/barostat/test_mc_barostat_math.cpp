#include <cmath>
#include <iostream>

#include "barostat/mc_barostat_math.h"

namespace
{
bool Near(const float actual, const float expected)
{
    return std::fabs(actual - expected) < 1.0e-6f;
}

bool Check_Restored(const char* label, const float scale,
                    const float reverse_scale)
{
    const float restored = scale * reverse_scale;
    if (Near(restored, 1.0f)) return true;
    std::cerr << label << " mismatch: " << restored << "\n";
    return false;
}
}  // namespace

int main()
{
    const float dt = 0.002f;
    LTMatrix3 forward_g;
    forward_g.a11 = (0.9f - 1.0f) / dt;
    forward_g.a22 = (1.01f - 1.0f) / dt;
    forward_g.a33 = (1.1f - 1.0f) / dt;

    const LTMatrix3 reverse_g = Get_Reverse_Diagonal_Box_Change(forward_g, dt);

    bool ok = true;
    ok &= Check_Restored("x scale", 1.0f + dt * forward_g.a11,
                         1.0f + dt * reverse_g.a11);
    ok &= Check_Restored("y scale", 1.0f + dt * forward_g.a22,
                         1.0f + dt * reverse_g.a22);
    ok &= Check_Restored("z scale", 1.0f + dt * forward_g.a33,
                         1.0f + dt * reverse_g.a33);
    ok &= Near(reverse_g.a21, 0.0f) && Near(reverse_g.a31, 0.0f) &&
          Near(reverse_g.a32, 0.0f);
    return ok ? 0 : 1;
}

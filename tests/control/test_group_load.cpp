#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "external_force/group_load.h"

using SpongeDynamics::Distribute_Group_Load;
using SpongeDynamics::Vec3;
void Check(bool ok, const char* message)
{
    if (!ok)
    {
        std::cerr << message << '\n';
        std::exit(1);
    }
}
void Verify(const std::vector<Vec3>& r, const std::vector<double>& m,
            const Vec3& force, const Vec3& torque)
{
    const auto loads = Distribute_Group_Load(r, m, force, torque);
    double mass = 0;
    Vec3 center{}, sum{}, moment{};
    for (std::size_t i = 0; i < r.size(); ++i)
    {
        mass += m[i];
        for (int k = 0; k < 3; ++k) center[k] += m[i] * r[i][k];
    }
    for (auto& x : center) x /= mass;
    for (std::size_t i = 0; i < r.size(); ++i)
        for (int k = 0; k < 3; ++k)
        {
            sum[k] += loads[i][k];
            const int a = (k + 1) % 3, b = (k + 2) % 3;
            moment[k] += (r[i][a] - center[a]) * loads[i][b] -
                         (r[i][b] - center[b]) * loads[i][a];
        }
    for (int k = 0; k < 3; ++k)
    {
        Check(std::abs(sum[k] - force[k]) < 1e-9, "incorrect total force");
        Check(std::abs(moment[k] - torque[k]) < 1e-9, "incorrect COM torque");
    }
}
int main()
{
    Verify({{0, 0, 0}, {1, 2, 0}, {-1, 0, 2}, {0, 1, 1}}, {1, 3, 7, 2},
           {2, -1, 3}, {4, 2, -5});
    Verify({{100, 200, 300}, {101, 202, 300}, {99, 200, 302}, {100, 201, 301}},
           {1, 3, 7, 2}, {2, -1, 3}, {4, 2, -5});
    Verify({{-1, 0, 0}, {1, 0, 0}}, {2, 2}, {0, 0, 0}, {0, 0, 6});
    const auto pair = Distribute_Group_Load({{-1, 0, 0}, {1, 0, 0}}, {2, 2},
                                            {0, 0, 0}, {0, 0, 6});
    Check(std::abs(pair[0][1] + 3) < 1e-12 && std::abs(pair[1][1] - 3) < 1e-12,
          "torque sign or magnitude");
    Verify({{1, 2, 3}}, {1}, {2, 3, 4}, {0, 0, 0});
    const auto translation = Distribute_Group_Load(
        {{0, 0, 0}, {1, 0, 0}}, {1, 3}, {4, 0, 0}, {0, 0, 0});
    Check(translation[0][0] == 1 && translation[1][0] == 3,
          "total force is not mass weighted");
    bool rejected = false;
    try
    {
        Distribute_Group_Load({{-1, 0, 0}, {1, 0, 0}}, {2, 2}, {0, 0, 0},
                              {1, 0, 0});
    }
    catch (const std::runtime_error&)
    {
        rejected = true;
    }
    Check(rejected, "impossible axial torque accepted");
    rejected = false;
    try
    {
        Distribute_Group_Load({{0, 0, 0}}, {1}, {0, 0, 0}, {0, 0, 1});
    }
    catch (const std::runtime_error&)
    {
        rejected = true;
    }
    Check(rejected, "point torque accepted");
    rejected = false;
    try
    {
        Distribute_Group_Load({{0, 0, 0}},
                              {std::numeric_limits<double>::quiet_NaN()},
                              {0, 0, 0}, {0, 0, 0});
    }
    catch (const std::runtime_error&)
    {
        rejected = true;
    }
    Check(rejected, "nonfinite mass accepted under fast-math");
    for (int i = 1; i <= 100; ++i)
    {
        const double s = 0.1 * i;
        Verify({{std::sin(s), 0, 0},
                {0, 1 + std::cos(s), 0},
                {0, 0, 1},
                {-1, -1, -1}},
               {1, 2, 3, 4}, {2, -3, 4}, {std::cos(s), std::sin(s), 1});
    }
}

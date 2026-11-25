#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>

#include "MD_core/hmr.h"

void Check(bool condition, const char* message)
{
    if (!condition)
    {
        std::cerr << message << '\n';
        std::exit(1);
    }
}
void Near(double a, double b)
{
    Check(std::abs(a - b) < 2e-6, "mass mismatch");
}
template <class F>
void Reject(F function)
{
    bool rejected = false;
    try
    {
        function();
    }
    catch (const std::exception&)
    {
        rejected = true;
    }
    Check(rejected, "invalid HMR input accepted");
}
int main()
{
    using SpongeHMR::Repartition;
    // Independent XH1, XH2, XH3 and XH4 donors, with reversed/duplicate edges.
    for (int count = 1; count <= 4; ++count)
        for (double ratio : {2.5, 3.0})
        {
            std::vector<float> original(count + 1, 1.008f);
            original[0] = count == 4 ? 14.007f : 12.011f;
            const auto saved = original;
            std::vector<std::pair<int, int>> bonds;
            for (int h = 1; h <= count; ++h)
            {
                bonds.emplace_back(0, h);
                bonds.emplace_back(h, 0);
            }
            const auto result = Repartition(original, bonds, {}, 2, ratio);
            const double factor = count == 1 ? 2 : ratio;
            Check(result.hydrogens == count, "H count mismatch");
            for (int h = 1; h <= count; ++h)
                Near(result.mass[h], original[h] * factor);
            Near(result.mass[0],
                 original[0] - count * original[1] * (factor - 1));
            Near(std::accumulate(result.mass.begin(), result.mass.end(), 0.0),
                 std::accumulate(original.begin(), original.end(), 0.0));
            Check(original == saved, "original masses changed");
        }
    // Ring heavy atoms keep the XH1 ratio independently of ring size.
    for (int ring : {5, 6})
    {
        std::vector<float> mass(ring * 2, 1.008f);
        std::vector<std::pair<int, int>> bonds;
        for (int i = 0; i < ring; ++i)
        {
            mass[i] = 12.011f;
            bonds.emplace_back(i, (i + 1) % ring);
            bonds.emplace_back(i, i + ring);
        }
        auto result = Repartition(mass, bonds, {}, 2, 3);
        for (int i = ring; i < ring * 2; ++i) Near(result.mass[i], 2.016);
    }
    // Water detection is independent of ordering, SETTLE H-H edges and sites.
    for (int oxygen = 0; oxygen < 3; ++oxygen)
        for (bool triangle : {false, true})
        {
            std::vector<float> mass{1.008f, 1.008f, 1.008f, 0, 0};
            mass[oxygen] = 15.999f;
            std::vector<std::pair<int, int>> bonds;
            for (int i = 0; i < 3; ++i)
                for (int j = i + 1; j < 3; ++j)
                    if (triangle || i == oxygen || j == oxygen)
                        bonds.emplace_back(i, j);
            bonds.emplace_back(oxygen, 3);
            auto result = Repartition(mass, bonds, {3, 4}, 2, 3);
            Check(result.mass == mass && result.waters == 1 &&
                      result.hydrogens == 0,
                  "water was repartitioned");
        }
    // Methanol's OH is not water.
    auto alcohol = Repartition({12, 16, 1}, {{0, 1}, {1, 2}}, {}, 2, 3);
    Check(alcohol.waters == 0, "alcohol classified as water");
    Near(alcohol.mass[1], 15);
    Near(alcohol.mass[2], 2);
    Reject([&] { Repartition({12, 1}, {}, {}, 2, 3); });
    Reject([&] { Repartition({12, 1, 12}, {{0, 1}, {1, 2}}, {}, 2, 3); });
    Reject([&] { Repartition({1, 1}, {{0, 1}}, {}, 2, 3); });
    Reject([&] { Repartition({12, 1}, {{0, 1}}, {}, 20, 3); });
    Reject([&] { Repartition({0}, {}, {}, 2, 3); });
    Reject([&] { Repartition({12, 1}, {{0, 2}}, {}, 2, 3); });
    Reject([&] { Repartition({12}, {}, {2}, 2, 3); });
    for (double bad : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::quiet_NaN()})
        Reject([&] { Repartition({12, 1}, {{0, 1}}, {}, 2, bad); });
    for (float bad : {-1.f, std::numeric_limits<float>::infinity(),
                      std::numeric_limits<float>::quiet_NaN()})
        Reject([&] { Repartition({bad}, {}, {}, 2, 3); });
    const float normal[] = {12, 1}, changed[] = {11, 2}, permuted[] = {1, 12};
    const auto signature = SpongeHMR::Mass_Signature(normal, 2);
    Check(signature != SpongeHMR::Mass_Signature(changed, 2) &&
              signature != SpongeHMR::Mass_Signature(permuted, 2),
          "mass signature missed a change");
    std::cout << "HMR unit checks passed\n";
}

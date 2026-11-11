#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>

#include "MD_core/integrator/constrained_middle_step.h"
#include "thermostat/velocity_bias/mode_math.h"
using namespace SpongeDynamics;
using Vel = std::vector<Vec3>;
void Check(bool ok, const char* message)
{
    if (!ok) throw std::runtime_error(message);
}
void Near(double a, double b, double tol = 2e-10)
{
    Check(std::isfinite(a) && std::isfinite(b) && std::abs(a - b) <= tol,
          "constrained flow mismatch");
}
template <class F>
void Reject(F f)
{
    bool rejected = false;
    try
    {
        f();
    }
    catch (const std::exception&)
    {
        rejected = true;
    }
    Check(rejected, "invalid constrained flow input accepted");
}
double Norm(const Vel& v, const std::vector<double>& m)
{
    double x = 0;
    for (std::size_t i = 0; i < m.size(); ++i)
        for (double y : v[i]) x += m[i] * y * y;
    return x;
}
void Print(const Vel& v)
{
    for (auto row : v)
        for (double x : row) std::cout << ' ' << x;
    std::cout << '\n';
}
// Test-only adapter: NumPy builds independent dense SVD/QR projectors from
// these same input coordinates. Production code has no streaming/RNG wrapper.
int Stream()
{
    int n, nc, axis, samples;
    double damping, kBT;
    std::cin >> n >> nc >> axis >> samples >> damping >> kBT;
    Check(n >= 2 && n <= 30 && nc >= 0 && nc <= 50 && samples >= 0 &&
              samples <= 100000,
          "invalid probe dimensions");
    std::vector<double> m(n), cosine(n);
    std::vector<DistanceConstraint> pairs(nc);
    Vel q(n), v(n), noise(n);
    Vec3 box;
    for (auto& x : m) std::cin >> x;
    for (auto* a : {&q, &v, &noise})
        for (auto& row : *a)
            for (auto& x : row) std::cin >> x;
    for (auto& x : box) std::cin >> x;
    for (auto& x : cosine) std::cin >> x;
    for (auto& p : pairs) std::cin >> p.a >> p.b >> p.length;
    Check(bool(std::cin), "incomplete probe input");
    auto frame = ConstraintStressPlan(m, pairs).Prepare(q, box, 1e-8);
    ConstrainedFlowFrame flow(frame, cosine, axis);
    const auto tangent = frame.Project(v);
    auto ou = flow.Apply_OU(tangent, noise, damping, kBT);
    std::cout << std::setprecision(17);
    std::cout << flow.Thermal_Degrees_Of_Freedom() << ' ' << ou.heat << '\n';
    Print(tangent);
    Print(flow.Preserved(v));
    Print(flow.Thermal(v));
    Print(ou.velocity);
    const int dim = 3 * n;
    // Each printed row is a column of the whitened (sqrt(M)*v) matrix.
    for (int k = 0; k < dim; ++k)
    {
        Vel b(n);
        b[k / 3][k % 3] = 1 / std::sqrt(m[k / 3]);
        auto p = frame.Project(b), h = flow.Thermal(b);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < 3; ++j)
            {
                p[i][j] *= std::sqrt(m[i]);
                h[i][j] *= std::sqrt(m[i]);
            }
        Print(p);
        Print(h);
    }
    // Independent draws from the target tangent Maxwell distribution, then an
    // OU step with independent noise. Validate full one-step conditional and
    // invariant covariances outside this implementation in NumPy.
    std::mt19937_64 rng(20261028);
    std::normal_distribution<double> normal;
    std::vector<double> sum(3 * dim), square(3 * dim * dim);
    double ksum = 0, ksquare = 0;
    for (int s = 0; s < samples; ++s)
    {
        Vel draw(n), xi(n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < 3; ++j)
            {
                draw[i][j] = normal(rng) * std::sqrt(kBT / m[i]);
                xi[i][j] = normal(rng);
            }
        draw = frame.Project(draw);
        const auto thermal = flow.Thermal(draw);
        const auto after = flow.Apply_OU(draw, xi, damping, kBT).velocity;
        const auto kept = flow.Preserved(draw);
        Vel innovation = after;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < 3; ++j)
                innovation[i][j] -= kept[i][j] + damping * thermal[i][j];
        const Vel* sets[] = {&thermal, &innovation, &after};
        for (int t = 0; t < 3; ++t)
            for (int i = 0; i < dim; ++i)
            {
                const double a = (*sets[t])[i / 3][i % 3] * std::sqrt(m[i / 3]);
                sum[t * dim + i] += a;
                for (int j = 0; j < dim; ++j)
                    square[t * dim * dim + i * dim + j] +=
                        a * (*sets[t])[j / 3][j % 3] * std::sqrt(m[j / 3]);
            }
        const double kinetic = Norm(thermal, m) / kBT;
        ksum += kinetic;
        ksquare += kinetic * kinetic;
    }
    for (double x : sum) std::cout << ' ' << x;
    std::cout << '\n';
    for (double x : square) std::cout << ' ' << x;
    std::cout << '\n' << ksum << ' ' << ksquare << '\n';
    return 0;
}
int main(int argc, char** argv)
{
    try
    {
        if (argc == 2 && std::string(argv[1]) == "--stream") return Stream();
        std::vector<double> m{2, 3, 4, 5};
        Vel q{{1, 2, 3}, {3, 2, 4}, {4, 3, 6}, {6, 1, 7}},
            v{{1, 2, 3}, {4, 1, 2}, {2, 5, 1}, {-2, 3, 1}};
        Vec3 box{20, 20, 20};
        std::vector<double> cs;
        for (const auto& row : q) cs.push_back(std::cos(.3 * row[2]));
        auto frame = ConstraintStressPlan(m, {}).Prepare(q, box, 1e-8);
        ConstrainedFlowFrame flow(frame, cs, 0);
        auto fit = Fit_Cosine(m, q, v, 0, 2, .3);
        auto heat = flow.Thermal(v), keep = flow.Preserved(v);
        Near(Norm(heat, m) * .5, fit.thermal_energy);
        Check(flow.Thermal_Degrees_Of_Freedom() == 8, "free DOF");
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 3; ++j)
            {
                Near(keep[i][j],
                     fit.mean_velocity[j] +
                         (j == 0 ? fit.amplitude * (cs[i] - fit.mean_cosine)
                                 : 0));
                Near(heat[i][j] + keep[i][j], v[i][j]);
            }
        auto identity = flow.Apply_OU(v, Vel(4), 1, 2);
        Check(identity.velocity == v && identity.heat == 0, "OU identity");
        auto cold = flow.Apply_OU(v, Vel(4), 0, 0);
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 3; ++j) Near(cold.velocity[i][j], keep[i][j]);
        Near(cold.heat, -fit.thermal_energy);
        Reject(
            [&]
            { ConstrainedFlowFrame bad(frame, std::vector<double>(4, 1), 0); });
        Reject([&] { ConstrainedFlowFrame bad(frame, cs, 3); });
        Reject([&] { flow.Apply_OU(v, Vel(3), .8, 1); });
        Reject([&] { flow.Apply_OU(v, Vel(4), 1.1, 1); });
        Reject([&] { flow.Apply_OU(v, Vel(4), .8, -1); });
        Reject(
            [&]
            {
                auto bad = v;
                bad[0][0] = std::numeric_limits<double>::quiet_NaN();
                flow.Thermal(bad);
            });
        auto dimer = ConstraintStressPlan({2, 3}, {{0, 1, 1}})
                         .Prepare({{0, 0, 0}, {1, 0, 0}}, box, 1e-8);
        // The raw x mode has variance, but its tangent projection is pure COM.
        Reject([&] { ConstrainedFlowFrame bad(dimer, {1, -1}, 0); });
        ConstrainedFlowFrame transverse(dimer, {1, -1}, 1);
        Reject([&]
               { transverse.Apply_OU({{1, 0, 0}, {0, 0, 0}}, Vel(2), .8, 1); });
        Reject([&]
               { ConstraintStressPlan bad({2, 3}, {{0, 1, 1}, {1, 0, 1}}); });
        Reject(
            [&]
            {
                ConstraintStressPlan({1, 1, 1},
                                     {{0, 1, 1}, {1, 2, 1}, {0, 2, 2}})
                    .Prepare({{0, 0, 0}, {1, 0, 0}, {2, 0, 0}}, box, 1e-8);
            });
        std::cout
            << "constrained flow zero-constraint, rank and OU gates passed\n";
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

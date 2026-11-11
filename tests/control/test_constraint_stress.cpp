#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

#include "constrain/tangent_frame.h"
using namespace SpongeDynamics;
void Check(bool ok, const char* message)
{
    if (!ok)
    {
        std::cerr << message << '\n';
        std::exit(1);
    }
}
void Near(double a, double b, double tolerance = 2e-12)
{
    Check(std::abs(a - b) <= tolerance,
          "constraint observer numerical mismatch");
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
    Check(rejected, "invalid constraint observer input accepted");
}
// Test-only streaming adapter to apply the SAME C++ observer to real executable
// native trajectories. It is not a production input format or viscosity CLI.
int Stream()
{
    int n, c, frames;
    if (!(std::cin >> n >> c >> frames) || n < 2 || n > 10000 || c < 0 ||
        c > 10000 || frames < 1 || frames > 100000)
        throw std::runtime_error("invalid test stream dimensions");
    std::vector<double> m(n);
    std::vector<DistanceConstraint> pairs(c);
    for (auto& x : m) std::cin >> x;
    for (auto& p : pairs) std::cin >> p.a >> p.b >> p.length;
    ConstraintStressPlan plan(m, pairs);
    std::cout << std::setprecision(17);
    for (int frame = 0; frame < frames; ++frame)
    {
        double h, tolerance;
        Vec3 box;
        Stress6 w;
        std::vector<Vec3> q(n), v(n), f(n);
        std::cin >> h >> box[0] >> box[1] >> box[2] >> tolerance;
        for (auto* a : {&q, &v, &f})
            for (auto& row : *a)
                for (auto& x : row) std::cin >> x;
        for (auto& x : w) std::cin >> x;
        if (!std::cin) throw std::runtime_error("incomplete test stream");
        auto r = plan.Evaluate(q, v, f, w, h, box, tolerance);
        std::cout << r.equilibrium.kinetic_energy << ' ' << r.thermal_dof;
        for (auto x : r.constraint_virial) std::cout << ' ' << x;
        for (auto x : r.equilibrium.pressure) std::cout << ' ' << x;
        std::cout << ' ' << r.max_relative_length_error << ' '
                  << r.max_tangent_residual << ' '
                  << r.max_acceleration_residual;
        for (auto* a : {&r.velocity, &r.constraint_force})
            for (auto row : *a)
                for (double x : row) std::cout << ' ' << x;
        std::cout << '\n';
    }
    return 0;
}
int main(int argc, char** argv)
{
    try
    {
        if (argc == 2 && std::string(argv[1]) == "--stream") return Stream();
        const Vec3 box{20, 20, 20};
        const Vec3 e{1. / 3, 2. / 3, 2. / 3}, t{2. / 3, 1. / 3, -2. / 3},
            com{2, -3, 1};
        std::vector<double> m{2, 3};
        const double length = 1.7, speed = .8, mu = 1.2;
        ConstraintStressPlan plan(m, {{0, 1, length}});
        std::vector<Vec3> q(2), v(2), force(2);
        for (int j = 0; j < 3; ++j)
        {
            q[0][j] = .6 * length * e[j];
            q[1][j] = -.4 * length * e[j];
            v[0][j] = com[j] + .6 * (speed * t[j] + .3 * e[j]);
            v[1][j] = com[j] - .4 * (speed * t[j] + .3 * e[j]);
            force[0][j] = .6 * e[j];
            force[1][j] = -.6 * e[j];
        }
        Stress6 w{};
        const int ax[6] = {0, 1, 2, 0, 0, 1}, ay[6] = {0, 1, 2, 1, 2, 2};
        for (int j = 0; j < 6; ++j) w[j] = .6 * length * e[ax[j]] * e[ay[j]];
        auto r = plan.Evaluate(q, v, force, w, .02, box, 1e-9);
        Near(r.equilibrium.kinetic_energy, .5 * mu * speed * speed);
        Check(r.thermal_dof == 2, "dimer DOF");
        for (int j = 0; j < 6; ++j)
        {
            const double ee = e[ax[j]] * e[ay[j]], tt = t[ax[j]] * t[ay[j]];
            Near(r.constraint_virial[j],
                 -(mu * speed * speed + .6 * length) * ee);
            Near(r.equilibrium.pressure[j] * 8000,
                 mu * speed * speed * (tt - ee));
        }
        Near(r.max_tangent_residual, 0);
        Near(r.max_acceleration_residual, 0);
        // A tangential force survives the half kick and changes the required
        // centripetal force; a radial-only test would miss omitted centering.
        auto tangent_force = force;
        for (int j = 0; j < 3; ++j)
        {
            tangent_force[0][j] += .2 * t[j];
            tangent_force[1][j] -= .2 * t[j];
        }
        auto kicked = plan.Evaluate(q, v, tangent_force, w, .02, box, 1e-9);
        const double kicked_speed = speed + .01 * .2 / mu;
        Near(kicked.equilibrium.kinetic_energy,
             .5 * mu * kicked_speed * kicked_speed);
        for (int j = 0; j < 6; ++j)
            Near(kicked.constraint_virial[j],
                 -(mu * kicked_speed * kicked_speed + .6 * length) * e[ax[j]] *
                     e[ay[j]]);
        // Independently wrap one endpoint, reverse pair orientation, and
        // permute atom indices. Pair virial must not depend on these
        // representations.
        auto wrapped = q;
        for (int j = 0; j < 3; ++j) wrapped[0][j] += box[j];
        auto pbc = ConstraintStressPlan(m, {{1, 0, length}})
                       .Evaluate(wrapped, v, force, w, .02, box, 1e-9);
        for (int j = 0; j < 6; ++j)
            Near(pbc.equilibrium.pressure[j], r.equilibrium.pressure[j]);
        auto perm = ConstraintStressPlan({3, 2}, {{1, 0, length}})
                        .Evaluate({q[1], q[0]}, {v[1], v[0]},
                                  {force[1], force[0]}, w, .02, box, 1e-9);
        for (int j = 0; j < 6; ++j)
            Near(perm.constraint_virial[j], r.constraint_virial[j]);
        // Non-collinear rigid three-site body rotating about the plane normal:
        // exact continuous acceleration is -omega^2*r about the mass COM.
        std::vector<double> water_mass{16, 1, 1};
        std::vector<Vec3> water{{0, 0, 0}, {.6, .8, 0}, {-.6, .8, 0}};
        for (auto& a : water) a[1] -= 1.6 / 18;
        std::vector<Vec3> water_v(3), zero(3);
        for (int i = 0; i < 3; ++i)
            water_v[i] = {-2 * water[i][1], 2 * water[i][0], 0};
        ConstraintStressPlan triangle(water_mass,
                                      {{0, 1, 1}, {0, 2, 1}, {1, 2, 1.2}});
        auto wr = triangle.Evaluate(water, water_v, zero, {}, 0, box, 1e-9);
        Check(wr.thermal_dof == 3, "triangle DOF");
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                Near(wr.constraint_force[i][j],
                     -4 * water_mass[i] * water[i][j]);
        Near(wr.equilibrium.pressure[0] + wr.equilibrium.pressure[1] +
                 wr.equilibrium.pressure[2],
             0);
        // Two disconnected molecules exercise component decomposition and the
        // removal of only GLOBAL COM, preserving relative translations.
        auto wm = water_mass;
        wm.insert(wm.end(), water_mass.begin(), water_mass.end());
        auto wq = water, wv = water_v;
        for (int i = 0; i < 3; ++i)
        {
            auto a = water[i];
            a[2] += 5;
            wq.push_back(a);
            auto b = water_v[i];
            b[2] = 3;
            wv.push_back(b);
        }
        ConstraintStressPlan two(wm, {{0, 1, 1},
                                      {0, 2, 1},
                                      {1, 2, 1.2},
                                      {3, 4, 1},
                                      {3, 5, 1},
                                      {4, 5, 1.2}});
        auto tr = two.Evaluate(wq, wv, std::vector<Vec3>(6), {}, 0, box, 1e-9);
        Check(tr.thermal_dof == 9, "two-molecule global COM DOF");
        Near(tr.equilibrium.kinetic_energy,
             2 * wr.equilibrium.kinetic_energy + 40.5);
        // Projected Maxwell draws: mean 2K=dof, variance=2*dof. Constraints
        // remove only their independent rank, without rescaling fluctuations.
        std::mt19937_64 rng(20261028);
        std::normal_distribution<double> normal;
        double sum = 0, square = 0;
        const int samples = 20000;
        for (int s = 0; s < samples; ++s)
        {
            for (int i = 0; i < 3; ++i)
                for (double& x : water_v[i])
                    x = normal(rng) / std::sqrt(water_mass[i]);
            double x =
                2 * triangle.Evaluate(water, water_v, zero, {}, 0, box, 1e-9)
                        .equilibrium.kinetic_energy;
            sum += x;
            square += x * x;
        }
        Near(sum / samples, 3, .07);
        Near(square / samples - std::pow(sum / samples, 2), 6, .3);
        Reject([&] { ConstraintStressPlan bad(m, {{0, 1, 1}, {1, 0, 1}}); });
        Reject([&] { ConstraintStressPlan bad({1, -1}, {}); });
        Reject([&] { ConstraintStressPlan bad(m, {{0, 2, 1}}); });
        Reject([&] { plan.Evaluate(q, v, force, w, .02, {3, 3, 3}, 1e-9); });
        Reject(
            [&]
            {
                auto bad = q;
                bad[0][0] += .01;
                plan.Evaluate(bad, v, force, w, .02, box, 1e-9);
            });
        Reject(
            [&]
            {
                ConstraintStressPlan singular(
                    {1, 1, 1}, {{0, 1, 1}, {1, 2, 1}, {0, 2, 2}});
                singular.Evaluate({{0, 0, 0}, {1, 0, 0}, {2, 0, 0}}, zero, zero,
                                  {}, 0, box, 1e-9);
            });
        // Empty constraints reduce to the existing centered COM stress exactly.
        auto free = ConstraintStressPlan(m, {}).Evaluate(q, v, force, w, .02,
                                                         box, 1e-9);
        auto reference = Centered_Stress(m, v, force, w, .02, 8000);
        for (int j = 0; j < 6; ++j)
            Near(free.equilibrium.pressure[j], reference.pressure[j]);
        std::cout << "constraint observer analytic, periodic, rank and DOF "
                     "gates passed\n";
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

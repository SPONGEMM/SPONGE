#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>

#include "MD_core/integrator/rattle_step.h"
using namespace SpongeDynamics;
using V = std::vector<Vec3>;
const Vec3 box{32, 32, 32};
const std::vector<double> mass{2, 3};
constexpr double length = 1.7, mu = 1.2;
void Check(bool value, const char* message)
{
    if (!value) throw std::runtime_error(message);
}
void Near(double a, double b, double tolerance = 2e-10)
{
    Check(std::isfinite(a) && std::isfinite(b) && std::abs(a - b) < tolerance,
          "mechanical NVE mismatch");
}
void Same(const V& a, const V& b, double tolerance = 2e-10)
{
    Check(a.size() == b.size(), "dimensions");
    for (std::size_t i = 0; i < a.size(); ++i)
        for (int j = 0; j < 3; ++j) Near(a[i][j], b[i][j], tolerance);
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
    Check(rejected, "invalid mechanical NVE accepted");
}
V Position(double theta)
{
    return {{8 + .6 * length * std::cos(theta),
             8 + .6 * length * std::sin(theta), 8},
            {8 - .4 * length * std::cos(theta),
             8 - .4 * length * std::sin(theta), 8}};
}
V Velocity(double theta, double omega)
{
    return {{-.6 * length * omega * std::sin(theta),
             .6 * length * omega * std::cos(theta), 0},
            {.4 * length * omega * std::sin(theta),
             -.4 * length * omega * std::cos(theta), 0}};
}
// Independent angular equation for an anisotropic harmonic relative potential.
// U = L^2/2 (kx cos^2 theta + ky sin^2 theta).
constexpr double kx = .3, ky = .7, kz = .2;
std::array<double, 2> Reference(double h)
{
    std::array<double, 2> x{.25, .6};
    auto rhs = [](auto y)
    {
        return std::array<double, 2>{
            y[1], (kx - ky) / mu * std::sin(y[0]) * std::cos(y[0])};
    };
    auto add = [](auto a, auto b, double t)
    { return std::array<double, 2>{a[0] + t * b[0], a[1] + t * b[1]}; };
    for (int n = 0; n < int(std::round(.8 / h)); ++n)
    {
        const auto a = rhs(x), b = rhs(add(x, a, h / 2)),
                   c = rhs(add(x, b, h / 2)), d = rhs(add(x, c, h));
        for (int j = 0; j < 2; ++j)
            x[j] += h / 6 * (a[j] + 2 * b[j] + 2 * c[j] + d[j]);
    }
    return x;
}
V Force(const V& q)
{
    V f(2);
    const double stiffness[3]{kx, ky, kz};
    for (int j = 0; j < 3; ++j)
    {
        f[0][j] = -stiffness[j] * (q[0][j] - q[1][j]);
        f[1][j] = -f[0][j];
    }
    return f;
}
double Energy(const V& q, const V& v)
{
    const double stiffness[3]{kx, ky, kz};
    double e = 0;
    for (int j = 0; j < 3; ++j)
    {
        e += .5 * stiffness[j] * std::pow(q[0][j] - q[1][j], 2);
        for (int i = 0; i < 2; ++i) e += .5 * mass[i] * v[i][j] * v[i][j];
    }
    return e;
}
V Kick(const ConstraintStressPlan& plan, const V& q, V v, const V& f, double h,
       Stress6* virial = nullptr)
{
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j) v[i][j] += h * f[i][j] / mass[i];
    return plan.Prepare(q, box, 2e-12).Project(v, virial);
}
// Test-only state probe: reproduce the runtime's fresh coordinate/tangent
// preparation so a liquid handoff can evaluate forces at the actual q0.
// This shares the preparation algorithm; it is not an independent oracle.
void Prepare_Stream()
{
    int n, count;
    Check(bool(std::cin >> n >> count) && n >= 2 && n <= 1000000 &&
              count >= 1 && count <= 1000000,
          "invalid preparation dimensions");
    Vec3 cell;
    std::vector<double> masses(n);
    V q(n), u(n);
    std::vector<DistanceConstraint> pairs(count);
    for (double& x : cell) Check(bool(std::cin >> x), "missing box");
    for (double& x : masses) Check(bool(std::cin >> x), "missing mass");
    for (auto* array : {&q, &u})
        for (auto& row : *array)
            for (double& x : row) Check(bool(std::cin >> x), "missing state");
    for (auto& pair : pairs)
    {
        Check(bool(std::cin >> pair.a >> pair.b >> pair.length),
              "missing pair");
        if (pair.b < pair.a) std::swap(pair.a, pair.b);
    }
    std::sort(pairs.begin(), pairs.end(),
              [](const auto& a, const auto& b)
              {
                  return std::tie(a.a, a.b, a.length) <
                         std::tie(b.a, b.b, b.length);
              });
    ConstraintStressPlan plan(masses, pairs);
    q = plan.Drift(q, V(n), 1., cell, 1e-4).position;
    u = plan.Prepare(q, cell, 2e-12).Project(u);
    std::cout << std::setprecision(17);
    for (int i = 0; i < n; ++i)
        std::cout << q[i][0] << ' ' << q[i][1] << ' ' << q[i][2] << ' '
                  << u[i][0] << ' ' << u[i][1] << ' ' << u[i][2] << '\n';
}
int main(int argc, char** argv)
{
    try
    {
        if (argc == 2 && std::string(argv[1]) == "--prepare")
        {
            Prepare_Stream();
            return 0;
        }
        Check(argc == 1, "unknown test option");
        ConstraintStressPlan plan(mass, {{0, 1, length}});
        const auto q = Position(.3), v = Velocity(.3, .8);
        const double h = .04, phi = std::asin(h * .8);
        const auto free =
            Advance_Constrained_NVE(plan, q, v, V(2), box, h, 2e-12);
        Same(free.position, Position(.3 + phi));
        Same(free.staggered_velocity, Velocity(.3 + phi, .8));
        Near(free.drift_kinetic_change, 0);
        const int ax[6]{0, 1, 2, 0, 0, 1}, ay[6]{0, 1, 2, 1, 2, 2};
        const double impulse = mu * length * length * (std::cos(phi) - 1) / h;
        Vec3 a{std::cos(.3), std::sin(.3), 0},
            b{std::cos(.3 + phi), std::sin(.3 + phi), 0};
        for (int j = 0; j < 6; ++j)
            Near(free.mechanical_constraint_impulse_virial[j],
                 impulse * (a[ax[j]] * a[ay[j]] + b[ax[j]] * b[ay[j]]));
        // Mechanical impulse and continuous force-time stress are distinct.
        const auto stress = plan.Evaluate(q, v, V(2), Stress6{}, h, box, 2e-12);
        for (int j = 0; j < 6; ++j)
            Near(stress.constraint_virial[j],
                 -mu * length * length * .8 * .8 * a[ax[j]] * a[ay[j]]);
        // Independent per-atom periodic images and uniform COM boost.
        auto shifted = q, boosted = v;
        shifted[0][0] += 32;
        shifted[1][1] -= 32;
        const Vec3 boost{.2, -.3, .1};
        for (auto& row : boosted)
            for (int j = 0; j < 3; ++j) row[j] += boost[j];
        auto wrapped = Advance_Constrained_NVE(plan, shifted, boosted, V(2),
                                               box, h, 2e-12);
        wrapped.position[0][0] -= 32;
        wrapped.position[1][1] += 32;
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 3; ++j)
            {
                wrapped.position[i][j] -= h * boost[j];
                wrapped.staggered_velocity[i][j] -= boost[j];
            }
        Same(wrapped.position, free.position);
        Same(wrapped.staggered_velocity, free.staggered_velocity);
        for (int j = 0; j < 6; ++j)
            Near(wrapped.mechanical_constraint_impulse_virial[j],
                 free.mechanical_constraint_impulse_virial[j]);
        // No flow geometry is needed in the unconstrained limit. The shared
        // stress plan retains its positive COM-free DOF requirement.
        const std::vector<double> um{2, 5};
        ConstraintStressPlan unconstrained(um, {});
        const V uq{{1, 2, 3}, {1, 4, 3}}, uv{{.2, -.3, .4}, {-.1, .5, .2}},
            uf{{.3, .5, -.2}, {-.4, .1, .6}};
        auto us = uv;
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 3; ++j) us[i][j] -= .5 * h * uf[i][j] / um[i];
        const auto un =
            Advance_Constrained_NVE(unconstrained, uq, us, uf, box, h, 2e-12);
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 3; ++j)
            {
                Near(un.position[i][j],
                     uq[i][j] + h * uv[i][j] + .5 * h * h * uf[i][j] / um[i]);
                Near(un.staggered_velocity[i][j] + .5 * h * uf[i][j] / um[i],
                     uv[i][j] + h * uf[i][j] / um[i]);
            }
        for (double x : un.mechanical_constraint_impulse_virial) Near(x, 0);
        Near(un.drift_kinetic_change, 0);
        // Multiple coupled constraints: known principal-axis rigid rotation.
        std::vector<double> tm{16, 1, 1};
        V tq{{0, 0, 0}, {.6, .8, 0}, {-.6, .8, 0}}, tv(3), eq(3), ev(3);
        for (auto& row : tq) row[1] -= 1.6 / 18;
        for (int i = 0; i < 3; ++i)
        {
            tv[i] = {-.8 * tq[i][1], .8 * tq[i][0], 0};
            eq[i] = {std::cos(phi) * tq[i][0] - std::sin(phi) * tq[i][1],
                     std::sin(phi) * tq[i][0] + std::cos(phi) * tq[i][1], 0};
            ev[i] = {-.8 * eq[i][1], .8 * eq[i][0], 0};
        }
        ConstraintStressPlan triangle(tm, {{0, 1, 1}, {0, 2, 1}, {1, 2, 1.2}});
        const auto tr =
            Advance_Constrained_NVE(triangle, tq, tv, V(3), box, h, 2e-12);
        Same(tr.position, eq);
        Same(tr.staggered_velocity, ev);
        // Explicit physical BAB vs staggered BA for a changing force.
        auto physical_q = q, physical_v = v, sq = q;
        auto sv = Kick(plan, q, v, Force(q), -.5 * h);
        for (int n = 0; n < 30; ++n)
        {
            Stress6 initial_half{}, final_half{};
            auto b1 = Kick(plan, physical_q, physical_v, Force(physical_q),
                           h / 2, &initial_half);
            auto ad = plan.Drift(physical_q, b1, h, box, 2e-12);
            auto b2 = Kick(plan, ad.position, ad.velocity, Force(ad.position),
                           h / 2, &final_half);
            const auto next =
                Advance_Constrained_NVE(plan, sq, sv, Force(sq), box, h, 2e-12);
            Same(next.position, ad.position);
            Same(Kick(plan, next.position, next.staggered_velocity,
                      Force(next.position), h / 2),
                 b2);
            Near(next.drift_kinetic_change, ad.kinetic_change);
            for (int j = 0; j < 6; ++j)
                Near(next.mechanical_constraint_impulse_virial[j] -
                         initial_half[j] + final_half[j],
                     initial_half[j] + ad.impulse_virial[j] + final_half[j]);
            physical_q = ad.position;
            physical_v = b2;
            sq = next.position;
            sv = next.staggered_velocity;
        }
        // Reverse physical velocity (not just the staggered variable).
        for (auto& row : physical_v)
            for (double& x : row) x = -x;
        sv = Kick(plan, sq, physical_v, Force(sq), -h / 2);
        for (int n = 0; n < 30; ++n)
        {
            const auto next =
                Advance_Constrained_NVE(plan, sq, sv, Force(sq), box, h, 2e-12);
            sq = next.position;
            sv = next.staggered_velocity;
        }
        auto reversed = Kick(plan, sq, sv, Force(sq), h / 2);
        for (auto& row : reversed)
            for (double& x : row) x = -x;
        Same(sq, q, 5e-9);
        Same(reversed, v, 5e-9);
        auto exact = Reference(1e-5), finer = Reference(5e-6);
        Near(exact[0], finer[0], 2e-12);
        Near(exact[1], finer[1], 2e-12);
        double prior_angle = 0, prior_energy = 0;
        std::cout << std::setprecision(17);
        for (double dt : {.04, .02, .01})
        {
            auto x = Position(.25), u = Velocity(.25, .6);
            const double e0 = Energy(x, u);
            auto s = Kick(plan, x, u, Force(x), -dt / 2);
            double energy_error = 0;
            for (int n = 0; n < int(std::round(.8 / dt)); ++n)
            {
                const auto next = Advance_Constrained_NVE(plan, x, s, Force(x),
                                                          box, dt, 2e-12);
                x = next.position;
                s = next.staggered_velocity;
                u = Kick(plan, x, s, Force(x), dt / 2);
                energy_error =
                    std::max(energy_error, std::abs(Energy(x, u) - e0));
                for (int j = 0; j < 3; ++j)
                    Near(2 * u[0][j] + 3 * u[1][j], 0, 2e-11);
            }
            const double angle_error = std::abs(
                std::atan2(x[0][1] - x[1][1], x[0][0] - x[1][0]) - exact[0]);
            if (prior_angle)
            {
                Check(prior_angle / angle_error > 3.8 &&
                          prior_angle / angle_error < 4.2,
                      "NVE angle order");
                Check(prior_energy / energy_error > 3.8 &&
                          prior_energy / energy_error < 4.2,
                      "NVE energy order");
            }
            prior_angle = angle_error;
            prior_energy = energy_error;
            std::cout << "dt=" << dt << " angle_error=" << angle_error
                      << " max_energy_error=" << energy_error << '\n';
        }
        for (double bad : {0., -h, std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::quiet_NaN()})
            Reject(
                [&]
                {
                    Advance_Constrained_NVE(plan, q, v, V(2), box, bad, 2e-12);
                });
        Reject(
            [&]
            { Advance_Constrained_NVE(plan, q, V(1), V(2), box, h, 2e-12); });
        Reject([&]
               { Advance_Constrained_NVE(plan, q, v, V(1), box, h, 2e-12); });
        auto bad = v;
        bad[0][0] += 1;
        Reject([&]
               { Advance_Constrained_NVE(plan, q, bad, V(2), box, h, 2e-12); });
        bad = V(2);
        bad[0][0] = std::numeric_limits<double>::quiet_NaN();
        Reject([&]
               { Advance_Constrained_NVE(plan, q, v, bad, box, h, 2e-12); });
        std::cout << "Mechanical NVE analytic, periodic, BAB staging, reversal "
                     "and ODE gates passed\n";
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

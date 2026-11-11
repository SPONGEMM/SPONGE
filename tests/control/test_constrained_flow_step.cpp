#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "MD_core/integrator/constrained_middle_step.h"
using namespace SpongeDynamics;
using Vel = std::vector<Vec3>;
void Check(bool ok, const char* message)
{
    if (!ok) throw std::runtime_error(message);
}
void Near(double a, double b, double tol = 2e-10)
{
    Check(std::isfinite(a) && std::isfinite(b) && std::abs(a - b) < tol,
          "constrained step mismatch");
}
void Same(const Vel& a, const Vel& b, double tol = 2e-10)
{
    Check(a.size() == b.size(), "size");
    for (std::size_t i = 0; i < a.size(); ++i)
        for (int j = 0; j < 3; ++j) Near(a[i][j], b[i][j], tol);
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
    Check(rejected, "invalid drift accepted");
}
const Vec3 box{32, 32, 32};
const std::vector<double> mass{2, 3};
constexpr double length = 1.7, mu = 1.2;
Vel Positions(double theta)
{
    return {{8 + .6 * length * std::cos(theta),
             8 + .6 * length * std::sin(theta), 8},
            {8 - .4 * length * std::cos(theta),
             8 - .4 * length * std::sin(theta), 8}};
}
Vel Velocities(double theta, double omega)
{
    return {{-.6 * length * omega * std::sin(theta),
             .6 * length * omega * std::cos(theta), 0},
            {.4 * length * omega * std::sin(theta),
             -.4 * length * omega * std::cos(theta), 0}};
}
double Energy(const Vel& q, const Vel& v, double f)
{
    double e = -f * (q[0][1] - q[1][1]);
    for (int i = 0; i < 2; ++i)
        for (double x : v[i]) e += .5 * mass[i] * x * x;
    return e;
}
std::array<double, 2> Reference(double step)
{
    std::array<double, 2> state{.25, .6};
    auto rhs = [](const std::array<double, 2>& x)
    {
        return std::array<double, 2>{x[1], .4 / (mu * length) * std::cos(x[0])};
    };
    auto add = [](auto a, auto b, double s)
    { return std::array<double, 2>{a[0] + s * b[0], a[1] + s * b[1]}; };
    for (int n = 0; n < int(std::round(.8 / step)); ++n)
    {
        auto k1 = rhs(state);
        auto k2 = rhs(add(state, k1, .5 * step));
        auto k3 = rhs(add(state, k2, .5 * step));
        auto k4 = rhs(add(state, k3, step));
        for (int j = 0; j < 2; ++j)
            state[j] += step / 6 * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]);
    }
    return state;
}
int main()
{
    try
    {
        ConstraintStressPlan plan(mass, {{0, 1, length}});
        const auto q = Positions(.3), v = Velocities(.3, .8);
        const double h = .04, phi = std::asin(h * .8);
        auto drift = plan.Drift(q, v, h, box, 1e-10);
        Same(drift.position, Positions(.3 + phi));
        Same(drift.velocity, Velocities(.3 + phi, .8));
        Near(drift.kinetic_change, 0);
        const double impulse =
            mu * length * length * (std::sqrt(1 - h * h * .8 * .8) - 1) / h;
        const int ax[6] = {0, 1, 2, 0, 0, 1}, ay[6] = {0, 1, 2, 1, 2, 2};
        Vec3 a{std::cos(.3), std::sin(.3), 0},
            b{std::cos(.3 + phi), std::sin(.3 + phi), 0};
        for (int j = 0; j < 6; ++j)
            Near(drift.impulse_virial[j],
                 impulse * (a[ax[j]] * a[ay[j]] + b[ax[j]] * b[ay[j]]));
        auto reverse =
            plan.Drift(drift.position, drift.velocity, -h, box, 1e-10);
        Same(reverse.position, q);
        Same(reverse.velocity, v);
        for (int j = 0; j < 6; ++j)
            Near(reverse.impulse_virial[j], -drift.impulse_virial[j]);
        // Direct radial-velocity removal has a known pair impulse -mu*u.
        Vel radial(2);
        for (int j = 0; j < 3; ++j)
        {
            radial[0][j] = .6 * a[j];
            radial[1][j] = -.4 * a[j];
        }
        Stress6 radial_virial{};
        Same(plan.Prepare(q, box, 1e-10).Project(radial, &radial_virial),
             Vel(2));
        for (int j = 0; j < 6; ++j)
            Near(radial_virial[j], -mu * length * a[ax[j]] * a[ay[j]]);
        // Coupled three-distance rigid body, unequal masses: a principal-axis
        // planar rotation has the same exact RATTLE angle asin(h*omega).
        const std::vector<double> tm{16, 1, 1};
        Vel tq{{0, 0, 0}, {.6, .8, 0}, {-.6, .8, 0}}, tv(3), expected_q(3),
            expected_v(3);
        const double omega = .8;
        for (auto& row : tq) row[1] -= 1.6 / 18;
        Stress6 expected_virial{};
        for (int i = 0; i < 3; ++i)
        {
            tv[i] = {-omega * tq[i][1], omega * tq[i][0], 0};
            expected_q[i] = {
                std::cos(phi) * tq[i][0] - std::sin(phi) * tq[i][1],
                std::sin(phi) * tq[i][0] + std::cos(phi) * tq[i][1], 0};
            expected_v[i] = {-omega * expected_q[i][1],
                             omega * expected_q[i][0], 0};
            for (int j = 0; j < 6; ++j)
                expected_virial[j] +=
                    tm[i] * (std::cos(phi) - 1) / h *
                    (tq[i][ax[j]] * tq[i][ay[j]] +
                     expected_q[i][ax[j]] * expected_q[i][ay[j]]);
            for (int j = 0; j < 3; ++j)
            {
                tq[i][j] += 5;
                expected_q[i][j] += 5;
            }
        }
        ConstraintStressPlan triangle(tm, {{0, 1, 1}, {0, 2, 1}, {1, 2, 1.2}});
        const auto td = triangle.Drift(tq, tv, h, box, 1e-10);
        Same(td.position, expected_q);
        Same(td.velocity, expected_v);
        for (int j = 0; j < 6; ++j)
            Near(td.impulse_virial[j], expected_virial[j]);
        auto tr = triangle.Drift(td.position, td.velocity, -h, box, 1e-10);
        Same(tr.position, tq);
        Same(tr.velocity, tv);
        auto wrapped = q;
        wrapped[0][0] += 32;
        wrapped[0][1] -= 32;
        auto wd = plan.Drift(wrapped, v, h, box, 1e-10);
        wd.position[0][0] -= 32;
        wd.position[0][1] += 32;
        Same(wd.position, drift.position);
        Same(wd.velocity, drift.velocity);
        for (int j = 0; j < 6; ++j)
            Near(wd.impulse_virial[j], drift.impulse_virial[j]);
        ConstraintStressPlan free(mass, {});
        auto fd = free.Drift(q, v, h, box, 1e-10);
        auto expected = q;
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 3; ++j) expected[i][j] += h * v[i][j];
        Same(fd.position, expected);
        Same(fd.velocity, v);
        Near(fd.kinetic_change, 0);
        for (double x : fd.impulse_virial) Near(x, 0);
        Reject([&] { plan.Drift(q, v, 0, box, 1e-10); });
        for (double bad : {std::numeric_limits<double>::quiet_NaN(),
                           std::numeric_limits<double>::infinity()})
        {
            Reject([&] { plan.Drift(q, v, bad, box, 1e-10); });
            Reject(
                [&]
                {
                    auto x = q;
                    x[0][0] = bad;
                    plan.Drift(x, v, h, box, 1e-10);
                });
            Reject(
                [&]
                {
                    auto x = v;
                    x[0][0] = bad;
                    plan.Drift(q, x, h, box, 1e-10);
                });
            Reject(
                [&]
                {
                    auto x = box;
                    x[0] = bad;
                    plan.Drift(q, v, h, x, 1e-10);
                });
        }
        Reject(
            [&]
            {
                auto bad = v;
                bad[0][0] += 1;
                plan.Drift(q, bad, h, box, 1e-10);
            });
        Reject(
            [&]
            {
                plan.Drift(q, v, 2, box, 1e-10);
            });  // no local position solution
        // Exact finite-force staging check: full BAOAB vs merged half B kicks.
        const Vel force{{0, .4, 0}, {0, -.4, 0}},
            noise{{.4, -.1, .7}, {-.2, .6, -.3}};
        auto frame = plan.Prepare(q, box, 1e-10);
        auto stored = v, halfkick = v;
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 3; ++j)
            {
                stored[i][j] -= .5 * h * force[i][j] / mass[i];
                halfkick[i][j] += .5 * h * force[i][j] / mass[i];
            }
        stored = frame.Project(stored);
        halfkick = frame.Project(halfkick);
        const double wave = 2 * 3.14159265358979323846 / 32;
        auto result = Advance_Constrained_Flow(
            plan, q, stored, force, noise, box, h, 2, 0, wave, .7, .2, 1e-10);
        auto first = plan.Drift(q, halfkick, h / 2, box, 1e-10);
        ConstrainedFlowFrame mid(plan.Prepare(first.position, box, 2e-12),
                                 {std::cos(wave * first.position[0][0]),
                                  std::cos(wave * first.position[1][0])},
                                 2);
        auto ou = mid.Apply_OU(first.velocity, noise, .7, .2);
        auto last = plan.Drift(first.position, ou.velocity, h / 2, box, 2e-12);
        Same(result.position, last.position);
        Same(result.staggered_velocity, last.velocity);
        Near(result.heat, ou.heat);
        // Explicit physical BAOAB with both endpoint kicks, changing force,
        // moving tangent space and nonidentity OU. The reference never stores
        // a staggered velocity between steps. Shared primitives test splitting,
        // while the separate analytic and RK4 tests test the drift dynamics.
        auto field = [](const Vel& x)
        {
            Vel f(2);
            const double stiffness[3] = {.3, .7, .2};
            for (int j = 0; j < 3; ++j)
            {
                f[0][j] = -stiffness[j] * (x[0][j] - x[1][j]);
                f[1][j] = -f[0][j];
            }
            return f;
        };
        auto kick_at = [&](const Vel& x, Vel velocity, const Vel& f,
                           double time, Stress6* virial = nullptr)
        {
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 3; ++j)
                    velocity[i][j] += time * f[i][j] / mass[i];
            return plan.Prepare(x, box, 1e-10).Project(velocity, virial);
        };
        auto physical_q = q, physical_v = v, staggered_q = q;
        auto staggered_v = kick_at(q, v, field(q), -.5 * h);
        for (int n = 0; n < 30; ++n)
        {
            Vel draws(2);
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 3; ++j)
                    draws[i][j] = std::sin((n + 1) * (3 * i + j + 1) * .73);
            const auto f = field(physical_q);
            Stress6 initial_half{}, final_half{};
            auto b1 = kick_at(physical_q, physical_v, f, .5 * h, &initial_half);
            auto a1 = plan.Drift(physical_q, b1, .5 * h, box, 1e-10);
            ConstrainedFlowFrame of(plan.Prepare(a1.position, box, 2e-12),
                                    {std::cos(wave * a1.position[0][0]),
                                     std::cos(wave * a1.position[1][0])},
                                    2);
            auto o = of.Apply_OU(a1.velocity, draws, .93, .1);
            auto a2 = plan.Drift(a1.position, o.velocity, .5 * h, box, 2e-12);
            auto next_force = field(a2.position);
            auto b2 = kick_at(a2.position, a2.velocity, next_force, .5 * h,
                              &final_half);
            auto st = Advance_Constrained_Flow(plan, staggered_q, staggered_v,
                                               field(staggered_q), draws, box,
                                               h, 2, 0, wave, .93, .1, 1e-10);
            Same(st.position, a2.position);
            Same(kick_at(st.position, st.staggered_velocity, field(st.position),
                         .5 * h),
                 b2);
            Near(st.heat, o.heat);
            Near(st.drift_kinetic_change,
                 a1.kinetic_change + a2.kinetic_change);
            for (int j = 0; j < 6; ++j)
            {
                const double physical_interval =
                    initial_half[j] + a1.impulse_virial[j] +
                    a2.impulse_virial[j] + final_half[j];
                Near(st.mechanical_constraint_impulse_virial[j] -
                         initial_half[j] + final_half[j],
                     physical_interval);
            }
            physical_q = a2.position;
            physical_v = b2;
            staggered_q = st.position;
            staggered_v = st.staggered_velocity;
        }
        Reject(
            [&]
            {
                auto bad = stored;
                bad[0][0] += 1;
                Advance_Constrained_Flow(plan, q, bad, force, noise, box, h, 2,
                                         0, wave, .7, .2, 1e-10);
            });
        Reject(
            [&]
            {
                Advance_Constrained_Flow(plan, q, stored, force, noise, box, h,
                                         0, 0, wave, .7, .2, 1e-10);
            });
        Reject(
            [&]
            {
                Advance_Constrained_Flow(plan, q, stored, force, Vel(1), box, h,
                                         2, 0, wave, .7, .2, 1e-10);
            });
        // Independent 2D forced-rotor ODE, not a copy of the RATTLE equations.
        auto exact = Reference(1e-5), fine = Reference(5e-6);
        Near(exact[0], fine[0], 2e-12);
        Near(exact[1], fine[1], 2e-12);
        double previous_error = 0, previous_energy = 0;
        std::cout << std::setprecision(17);
        for (double dt : {.04, .02, .01})
        {
            auto current_q = Positions(.25), current_v = Velocities(.25, .6);
            const double initial_energy = Energy(current_q, current_v, .4);
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 3; ++j)
                    current_v[i][j] -= .5 * dt * force[i][j] / mass[i];
            current_v = plan.Prepare(current_q, box, 1e-10).Project(current_v);
            double max_energy = 0;
            for (int step = 0; step < int(std::round(.8 / dt)); ++step)
            {
                auto next = Advance_Constrained_Flow(plan, current_q, current_v,
                                                     force, Vel(2), box, dt, 2,
                                                     0, wave, 1, 0, 1e-10);
                current_q = next.position;
                current_v = next.staggered_velocity;
                auto physical = current_v;
                for (int i = 0; i < 2; ++i)
                    for (int j = 0; j < 3; ++j)
                        physical[i][j] += .5 * dt * force[i][j] / mass[i];
                physical =
                    plan.Prepare(current_q, box, 2e-12).Project(physical);
                max_energy = std::max(
                    max_energy,
                    std::abs(Energy(current_q, physical, .4) - initial_energy));
            }
            const double theta = std::atan2(current_q[0][1] - current_q[1][1],
                                            current_q[0][0] - current_q[1][0]);
            const double error = std::abs(theta - exact[0]);
            if (previous_error)
            {
                Check(previous_error / error > 3.8 &&
                          previous_error / error < 4.2,
                      "rotor angle not second order");
                Check(previous_energy / max_energy > 3.8 &&
                          previous_energy / max_energy < 4.2,
                      "rotor energy not second order");
            }
            previous_error = error;
            previous_energy = max_energy;
            std::cout << "dt=" << dt << " angle_error=" << error
                      << " max_energy_error=" << max_energy << '\n';
        }
        std::cout << "RATTLE analytic, virial, reversible, periodic, staged OU "
                     "and forced-rotor convergence gates passed\n";
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

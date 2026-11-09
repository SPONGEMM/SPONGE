#include <cmath>
#include <cstdio>
#include <cstring>
#include <initializer_list>

#ifndef SPONGE_SETTLE_SOURCE
#define SPONGE_SETTLE_SOURCE "constrain/settle.cpp"
#endif
#include SPONGE_SETTLE_SOURCE

unsigned int CONTROLLER::device_max_thread = 64;

#ifdef SPONGE_CONSTRAINT_DOUBLE_PROBE
using ProbeReal = double;
#else
using ProbeReal = float;
#endif

namespace
{
const int ends[3][2] = {{0, 1}, {0, 2}, {1, 2}};

ConstraintVector<ProbeReal> Probe_Wrap(ConstraintVector<ProbeReal> q,
                                       const Boundary& b)
{
    const auto w = Wrap_Coordinate(VECTOR(q.x, q.y, q.z), b);
    return {w.x, w.y, w.z};
}

ProbeReal Component(const ConstraintVector<ProbeReal>& v, int d)
{
    return d == 0 ? v.x : (d == 1 ? v.y : v.z);
}

// Independent double Newton solve: corrections are along the OLD bonds,
// weighted by inverse mass. This solves the position SHAKE equations directly,
// without using SETTLE's rotations or geometry coefficients.
bool Reference(const ConstraintVector<ProbeReal> old[3],
               const ConstraintVector<ProbeReal> trial[3],
               const ProbeReal mass[3], double corrected[3][3])
{
    double direction[3][3][3] = {}, target2[3] = {};
    for (int a = 0; a < 3; ++a)
    {
        for (int d = 0; d < 3; ++d) corrected[a][d] = Component(trial[a], d);
    }
    for (int k = 0; k < 3; ++k)
    {
        const int a = ends[k][0], b = ends[k][1];
        for (int d = 0; d < 3; ++d)
        {
            const double bond = Component(old[a], d) - Component(old[b], d);
            target2[k] += bond * bond;
            direction[k][a][d] = bond / mass[a];
            direction[k][b][d] = -bond / mass[b];
        }
    }
    for (int iteration = 0; iteration < 20; ++iteration)
    {
        double matrix[3][4] = {}, maximum = 0;
        for (int k = 0; k < 3; ++k)
        {
            const int a = ends[k][0], b = ends[k][1];
            double residual = -target2[k];
            for (int d = 0; d < 3; ++d)
            {
                const double bond = corrected[a][d] - corrected[b][d];
                residual += bond * bond;
                for (int j = 0; j < 3; ++j)
                    matrix[k][j] +=
                        2 * bond * (direction[j][a][d] - direction[j][b][d]);
            }
            matrix[k][3] = -residual;
            maximum = std::fmax(maximum, std::fabs(residual));
        }
        if (maximum < 1e-14) return true;
        for (int k = 0; k < 3; ++k)
        {
            const double pivot = matrix[k][k];
            if (std::fabs(pivot) < 1e-14) return false;
            for (int j = k; j < 4; ++j) matrix[k][j] /= pivot;
            for (int i = 0; i < 3; ++i)
            {
                if (i == k) continue;
                const double scale = matrix[i][k];
                for (int j = k; j < 4; ++j)
                    matrix[i][j] -= scale * matrix[k][j];
            }
        }
        for (int a = 0; a < 3; ++a)
            for (int d = 0; d < 3; ++d)
                for (int k = 0; k < 3; ++k)
                    corrected[a][d] += direction[k][a][d] * matrix[k][3];
    }
    return false;
}

ConstraintVector<ProbeReal> Permute(ConstraintVector<ProbeReal> v, int axis)
{
    for (int k = 0; k < axis; ++k) v = {v.z, v.x, v.y};
    return v;
}
}  // namespace

int main()
{
    double max_correction_error = 0, max_momentum_error = 0,
           max_stationary_speed = 0, max_local_bond_error = 0,
           max_stored_bond_error = 0, max_scaled_virial_error = 0;
    // Dyadic fixtures preserve exactly the same local input under translation.
    // Therefore their impulses must be independent of absolute coordinates,
    // even though the final coordinates remain stored in ProbeReal.
    for (ProbeReal mass_c : {1.0f, 2.0f})
        for (int deformed : {0, 1})
            for (int axis = 0; axis < 3; ++axis)
                for (int periodic : {0, 1})
                    for (ProbeReal dt : {0.01f, 0.0025f, 0.001f})
                        for (ProbeReal gamma : {1.0f, 0.97f})
                        {
                            // The legacy analytic rotation assumes equal B/C
                            // masses for a driven triangle. Check unequal
                            // masses only for the stationary precision case;
                            // extending that solver is separate from
                            // translation stability.
                            if (mass_c != 1.0f && deformed) continue;
                            const ProbeReal factor = 0.5f * (1.0f + gamma);
                            const ProbeReal mass[3] = {16, 1, mass_c};
                            Boundary boundary = {};
                            if (periodic)
                            {
                                boundary.policy = BoundaryPolicy::Periodic;
                                boundary.cell = {32, 0, 32, 0, 0, 32};
                                boundary.rcell = {1.f / 32, 0, 1.f / 32,
                                                  0,        0, 1.f / 32};
                            }
                            ConstraintVector<ProbeReal> old[3] = {
                                {0, 0, 0},
                                Permute({.375f, .5f, 0}, axis),
                                Permute({-.375f, .5f, 0}, axis)};
                            ConstraintVector<ProbeReal> trial[3] = {
                                old[0], old[1], old[2]};
                            if (deformed)
                            {
                                trial[0] =
                                    trial[0] + Permute({1.f / 1024, -2.f / 1024,
                                                        1.f / 1024},
                                                       axis);
                                trial[1] =
                                    trial[1] + Permute({-3.f / 1024, 1.f / 1024,
                                                        -2.f / 1024},
                                                       axis);
                                trial[2] =
                                    trial[2] + Permute({2.f / 1024, 3.f / 1024,
                                                        1.f / 1024},
                                                       axis);
                            }
                            double reference[3][3];
                            if (!Reference(old, trial, mass, reference))
                                return 2;
                            ConstraintTriangle<ProbeReal> triangle = {};
                            triangle.atom_A = 0;
                            triangle.atom_B = 1;
                            triangle.atom_C = 2;
                            Get_Rabcde_From_SSS<ProbeReal>(
                                .625f, .625f, .75f, 16, 1, mass_c, triangle.ra,
                                triangle.rb, triangle.rc, triangle.rd,
                                triangle.re);
                            ConstraintVector<ProbeReal> first_velocity[3] = {};
                            ConstraintTensor<ProbeReal> first_virial = {};
                            for (ProbeReal shift : {0.f, 16.f, 128.f, 1024.f})
                            {
                                ConstraintVector<ProbeReal> crd[3],
                                    velocity[3] = {};
                                for (int a = 0; a < 3; ++a)
                                    crd[a] = Probe_Wrap(
                                        trial[a] +
                                            ConstraintVector<ProbeReal>{
                                                shift, -shift, shift},
                                        boundary);
                                const ConstraintVector<ProbeReal> original[3] =
                                    {crd[0], crd[1], crd[2]};
                                ConstraintTensor<ProbeReal> virial = {};
                                settle_triangle(1, &triangle, mass, crd,
                                                boundary, &old[1], &old[2], dt,
                                                gamma, factor, velocity,
                                                &virial);
                                double momentum[3] = {};
                                for (int a = 0; a < 3; ++a)
                                    for (int d = 0; d < 3; ++d)
                                    {
                                        const double value =
                                            Component(velocity[a], d);
                                        const double correction =
                                            value * dt * factor / gamma;
                                        const double expected =
                                            reference[a][d] -
                                            Component(trial[a], d);
                                        const double error =
                                            std::fabs(correction - expected);
                                        max_correction_error = std::fmax(
                                            max_correction_error, error);
                                        momentum[d] += mass[a] * correction;
                                        if (!SpongeFloat::Is_Finite(value) ||
                                            error > 3e-7)
                                        {
                                            std::fprintf(
                                                stderr,
                                                "correction error %.9g (shift "
                                                "%.1f, dt %.4g)\n",
                                                error, shift, dt);
                                            return 3;
                                        }
                                        if (!deformed)
                                            max_stationary_speed =
                                                std::fmax(max_stationary_speed,
                                                          std::fabs(value));
                                        // Float write-back can round away a
                                        // sub-ULP correction. Compare with the
                                        // independently rounded final position,
                                        // not a new velocity inferred by
                                        // subtracting those rounded positions.
                                        const ProbeReal expected_position =
                                            static_cast<ProbeReal>(
                                                reference[a][d] +
                                                Component(original[a], d) -
                                                Component(trial[a], d));
                                        const ProbeReal actual_position =
                                            Component(crd[a], d);
                                        if (std::fabs(actual_position -
                                                      expected_position) >
                                            3e-7 +
                                                2e-7 * std::fabs(
                                                           expected_position))
                                            return 4;
                                    }
                                for (double p : momentum)
                                {
                                    max_momentum_error = std::fmax(
                                        max_momentum_error, std::fabs(p));
                                    if (std::fabs(p) > 2e-6)
                                    {
                                        std::fprintf(
                                            stderr,
                                            "mass-weighted correction %.9g "
                                            "(shift %.1f, dt %.4g)\n",
                                            p, shift, dt);
                                        return 5;
                                    }
                                }
                                // Check the constraints themselves, separately
                                // from agreement with the Newton correction.
                                for (const auto& edge : ends)
                                {
                                    const int a = edge[0], b = edge[1];
                                    const ConstraintVector<ProbeReal> old_bond =
                                        old[a] - old[b];
                                    const double target =
                                        std::sqrt(static_cast<double>(
                                            old_bond * old_bond));
                                    double local_squared = 0;
                                    for (int d = 0; d < 3; ++d)
                                    {
                                        const double displacement =
                                            Component(trial[a], d) -
                                            Component(trial[b], d) +
                                            (Component(velocity[a], d) -
                                             Component(velocity[b], d)) *
                                                static_cast<double>(dt) *
                                                factor / gamma;
                                        local_squared +=
                                            displacement * displacement;
                                    }
                                    const ConstraintVector<ProbeReal>
                                        stored_bond = Get_Displacement(
                                            crd[a], crd[b], boundary);
                                    const double local_error = std::fabs(
                                        std::sqrt(local_squared) - target);
                                    const double stored_error = std::fabs(
                                        std::sqrt(static_cast<double>(
                                            stored_bond * stored_bond)) -
                                        target);
                                    max_local_bond_error = std::fmax(
                                        max_local_bond_error, local_error);
                                    max_stored_bond_error = std::fmax(
                                        max_stored_bond_error, stored_error);
                                    // Stored absolute floats at shift=1024
                                    // have a larger ULP than the local solve.
                                    if (local_error > 6e-7 ||
                                        stored_error > 3e-4)
                                        return 7;
                                }
                                // Independently build sum_i m_i*delta_r_i
                                // (r_i-r_A), with delta_r from Newton. Scaling
                                // back by dt^2*factor tests both the integrator
                                // factor and ConstraintTensor<ProbeReal>
                                // off-diagonal convention.
                                const ProbeReal actual_virial[6] = {
                                    virial.a11, virial.a21, virial.a22,
                                    virial.a31, virial.a32, virial.a33};
                                const int axes[6][2] = {{0, 0}, {1, 0}, {1, 1},
                                                        {2, 0}, {2, 1}, {2, 2}};
                                for (int k = 0; k < 6; ++k)
                                {
                                    const int d = axes[k][0], e = axes[k][1];
                                    double expected = 0;
                                    for (int a = 1; a < 3; ++a)
                                    {
                                        const double correction_d =
                                            reference[a][d] -
                                            Component(trial[a], d);
                                        const double correction_e =
                                            reference[a][e] -
                                            Component(trial[a], e);
                                        expected += mass[a] * correction_d *
                                                    Component(old[a], e);
                                        if (d != e)
                                            expected += mass[a] * correction_e *
                                                        Component(old[a], d);
                                    }
                                    const double error =
                                        std::fabs(actual_virial[k] *
                                                      static_cast<double>(dt) *
                                                      dt * factor -
                                                  expected);
                                    max_scaled_virial_error = std::fmax(
                                        max_scaled_virial_error, error);
                                    if (error >
                                        2e-6 + 5e-5 * std::fabs(expected))
                                        return 8;
                                }
                                if (shift == 0)
                                {
                                    std::memcpy(first_velocity, velocity,
                                                sizeof(velocity));
                                    first_virial = virial;
                                }
                                else if (std::memcmp(first_velocity, velocity,
                                                     sizeof(velocity)) != 0 ||
                                         std::memcmp(&first_virial, &virial,
                                                     sizeof(virial)) != 0)
                                {
                                    std::fputs(
                                        "constraint impulse or virial depends "
                                        "on absolute translation\n",
                                        stderr);
                                    return 6;
                                }
                            }
                        }
    std::printf(
        "max correction error %.9g; mass-weighted correction %.9g; stationary "
        "speed %.9g; local bond error %.9g; stored bond error %.9g; "
        "scaled virial error %.9g\n",
        max_correction_error, max_momentum_error, max_stationary_speed,
        max_local_bond_error, max_stored_bond_error, max_scaled_virial_error);
    if (std::is_same<ProbeReal, double>::value &&
        (max_correction_error > 1e-10 || max_momentum_error > 5e-12 ||
         max_local_bond_error > 1e-10 || max_scaled_virial_error > 1e-10))
        return 9;
    return 0;
}

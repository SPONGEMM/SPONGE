#include <cmath>
#include <cstdio>
#include <cstring>

#include "constrain/matrix_constraint.cpp"

unsigned int CONTROLLER::device_max_thread = 64;

// Independent two-row mass-metric velocity projection. Both production
// matrix solvers must converge to it without changing the coordinates.
template <class Real>
bool Check(bool ccma)
{
    using V = ConstraintVector<Real>;
    CONSTRAIN_PAIR pairs[2] = {{0, 1, 1, 1}, {1, 2, 1, 1}};
    V q[3] = {{0, 0, 0}, {1, 0, 0}, {Real(1.6), Real(.8), 0}};
    V saved[3] = {q[0], q[1], q[2]};
    V vel[3] = {{Real(.3), Real(-.2), Real(.4)},
                {Real(-.7), Real(.5), Real(.1)},
                {Real(.8), Real(.9), Real(-.3)}};
    V initial[3] = {vel[0], vel[1], vel[2]};
    Real inv[3] = {Real(1) / 16, 1, 1}, scale[2];
    V unit[2], old[2], total[3] = {};
    Real coefficient[4], matrix[4], rhs[2], solution[2], term[2], next[2];
    const int rows[3] = {0, 2, 4}, columns[4] = {0, 1, 0, 1};
    const int active[3] = {0, 1, 2}, atom_rows[4] = {0, 1, 3, 4},
              incidence[4] = {1, -1, 2, -2};
    const Boundary boundary = {};
    Matrix_Directions<Real>(2, pairs, q, boundary, unit, old);
    for (int i = 0; i < 2; ++i)
        scale[i] = Real(1) / ConstraintSqrt(inv[i] + inv[i + 1]);
    // Cbar's only off-diagonal entry comes from shared atom 1.
    const double c =
        -double(scale[0]) * double(scale[1]) * double(unit[0] * unit[1]);
    coefficient[0] = 0;
    coefficient[1] = -scale[0] * scale[1];
    coefficient[2] = coefficient[1];
    coefficient[3] = 0;
    if (ccma)
    {
        matrix[0] = matrix[3] = Real(1 / (1 - c * c));
        matrix[1] = matrix[2] = Real(-c / (1 - c * c));
    }
    else
        Matrix_Coupling<Real>(2, rows, columns, coefficient, unit, matrix);
    bool converged = false;
    for (int iteration = 0; iteration < 50; ++iteration)
    {
        int bad = 0;
        Matrix_Velocity_Residual<Real>(2, pairs, q, vel, boundary, unit, scale,
                                       rhs, &bad);
        if (!bad)
        {
            converged = true;
            break;
        }
        if (ccma)
            Matrix_Product<Real>(2, rows, columns, matrix, rhs, solution,
                                 nullptr);
        else
        {
            std::memcpy(term, rhs, sizeof(rhs));
            std::memcpy(solution, rhs, sizeof(rhs));
            for (int order = 0; order < 24; ++order)
            {
                Matrix_Product<Real>(2, rows, columns, matrix, term, next,
                                     solution);
                std::memcpy(term, next, sizeof(term));
            }
        }
        Matrix_Apply<Real>(3, active, atom_rows, incidence, unit, scale,
                           solution, inv, vel, total);
    }
    if (!converged || std::memcmp(q, saved, sizeof(q))) return false;
    double b[2][3][3] = {}, rhs_ref[2] = {}, a[2][2] = {};
    for (int i = 0; i < 2; ++i)
    {
        const double d[3] = {double(q[i].x) - q[i + 1].x,
                             double(q[i].y) - q[i + 1].y,
                             double(q[i].z) - q[i + 1].z};
        for (int k = 0; k < 3; ++k)
        {
            b[i][i][k] = d[k];
            b[i][i + 1][k] = -d[k];
        }
    }
    for (int i = 0; i < 2; ++i)
        for (int atom = 0; atom < 3; ++atom)
        {
            const double v[3] = {double(initial[atom].x),
                                 double(initial[atom].y),
                                 double(initial[atom].z)};
            for (int k = 0; k < 3; ++k)
            {
                rhs_ref[i] += b[i][atom][k] * v[k];
                for (int j = 0; j < 2; ++j)
                    a[i][j] +=
                        b[i][atom][k] * double(inv[atom]) * b[j][atom][k];
            }
        }
    const double det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    const double lambda[2] = {
        (a[1][1] * rhs_ref[0] - a[0][1] * rhs_ref[1]) / det,
        (a[0][0] * rhs_ref[1] - a[1][0] * rhs_ref[0]) / det};
    double maximum = 0, momentum[3] = {};
    for (int atom = 0; atom < 3; ++atom)
    {
        const double v[3] = {double(initial[atom].x), double(initial[atom].y),
                             double(initial[atom].z)};
        const double result[3] = {double(vel[atom].x), double(vel[atom].y),
                                  double(vel[atom].z)};
        for (int k = 0; k < 3; ++k)
        {
            const double expected =
                v[k] - double(inv[atom]) * (b[0][atom][k] * lambda[0] +
                                            b[1][atom][k] * lambda[1]);
            maximum = std::fmax(maximum, std::fabs(result[k] - expected));
            momentum[k] += (result[k] - v[k]) / double(inv[atom]);
        }
    }
    const double momentum_limit =
        std::is_same<Real, float>::value ? 2e-6 : 1e-12;
    for (double p : momentum)
        if (std::fabs(p) > momentum_limit) return false;
    std::printf(
        "%s %s velocity reference error %.12g\n", ccma ? "CCMA" : "LINCS",
        std::is_same<Real, float>::value ? "single" : "double", maximum);
    return maximum < (std::is_same<Real, float>::value ? 2e-5 : 1e-8);
}

int main()
{
    return Check<float>(false) && Check<double>(false) && Check<float>(true) &&
                   Check<double>(true)
               ? 0
               : 1;
}

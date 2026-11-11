#include "group_load.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "../utils/float_classification.hpp"

namespace SpongeDynamics
{
namespace
{
void Finite(double value)
{
    if (!SpongeFloat::Is_Finite(value))
        throw std::runtime_error("group_load: nonfinite geometry or load");
}
double Norm(const Vec3& a) { return std::hypot(std::hypot(a[0], a[1]), a[2]); }
Vec3 Cross(const Vec3& a, const Vec3& b)
{
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]};
}
}  // namespace

std::vector<Vec3> Distribute_Group_Load(const std::vector<Vec3>& positions,
                                        const std::vector<double>& masses,
                                        const Vec3& force, const Vec3& torque)
{
    if (positions.empty() || positions.size() != masses.size())
        throw std::runtime_error("group_load: inconsistent or empty selection");
    double total = 0;
    Vec3 center{0, 0, 0};
    for (std::size_t i = 0; i < masses.size(); ++i)
    {
        Finite(masses[i]);
        if (masses[i] <= 0)
            throw std::runtime_error("group_load: masses must be positive");
        total += masses[i];
        for (int k = 0; k < 3; ++k)
        {
            Finite(positions[i][k]);
            // Center relative to an anchor to reduce cancellation after
            // translation.
            center[k] += masses[i] * (positions[i][k] - positions[0][k]);
        }
    }
    Finite(total);
    for (int k = 0; k < 3; ++k)
    {
        Finite(force[k]);
        Finite(torque[k]);
        center[k] /= total;
        Finite(center[k]);
    }
    std::vector<Vec3> offsets(positions.size());
    double inertia[3][3] = {};
    for (std::size_t i = 0; i < positions.size(); ++i)
    {
        auto& r = offsets[i];
        for (int k = 0; k < 3; ++k)
            r[k] = (positions[i][k] - positions[0][k]) - center[k];
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                if (a == b)
                    inertia[a][a] +=
                        masses[i] * (r[(a + 1) % 3] * r[(a + 1) % 3] +
                                     r[(a + 2) % 3] * r[(a + 2) % 3]);
                else
                    inertia[a][b] -= masses[i] * r[a] * r[b];
    }
    double scale = 0;
    for (auto& row : inertia)
        for (double x : row)
        {
            Finite(x);
            scale = std::max(scale, std::abs(x));
        }
    Vec3 alpha{0, 0, 0};
    if (scale > 0)
    {
        // Symmetric Jacobi diagonalization, scaled to avoid absolute cutoffs.
        double a[3][3], eigenvectors[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k) a[j][k] = inertia[j][k] / scale;
        for (int sweep = 0; sweep < 32; ++sweep)
        {
            int p = 0, q = 1;
            if (std::abs(a[0][2]) > std::abs(a[p][q]))
            {
                p = 0;
                q = 2;
            }
            if (std::abs(a[1][2]) > std::abs(a[p][q]))
            {
                p = 1;
                q = 2;
            }
            if (std::abs(a[p][q]) < 1e-15) break;
            const double angle =
                0.5 * std::atan2(2 * a[p][q], a[q][q] - a[p][p]);
            const double c = std::cos(angle), s = std::sin(angle);
            const double app = a[p][p], aqq = a[q][q], apq = a[p][q];
            a[p][p] = c * c * app - 2 * c * s * apq + s * s * aqq;
            a[q][q] = s * s * app + 2 * c * s * apq + c * c * aqq;
            a[p][q] = a[q][p] = 0;
            for (int k = 0; k < 3; ++k)
            {
                if (k != p && k != q)
                {
                    const double kp = a[k][p], kq = a[k][q];
                    a[k][p] = a[p][k] = c * kp - s * kq;
                    a[k][q] = a[q][k] = s * kp + c * kq;
                }
                const double vp = eigenvectors[k][p], vq = eigenvectors[k][q];
                eigenvectors[k][p] = c * vp - s * vq;
                eigenvectors[k][q] = s * vp + c * vq;
            }
        }
        const double largest = std::max({a[0][0], a[1][1], a[2][2]});
        for (int k = 0; k < 3; ++k)
        {
            if (a[k][k] <= 1e-12 * largest) continue;
            double projection = 0;
            for (int j = 0; j < 3; ++j)
                projection += eigenvectors[j][k] * torque[j];
            const double coefficient = (projection / scale) / a[k][k];
            for (int j = 0; j < 3; ++j)
                alpha[j] += coefficient * eigenvectors[j][k];
        }
    }
    Vec3 residual = torque;
    for (int j = 0; j < 3; ++j)
    {
        Finite(alpha[j]);
        for (int k = 0; k < 3; ++k) residual[j] -= inertia[j][k] * alpha[k];
        Finite(residual[j]);
    }
    if (Norm(residual) > 1e-8 * Norm(torque))
        throw std::runtime_error(
            "group_load: requested torque is not representable by the group "
            "geometry");
    std::vector<Vec3> result(positions.size());
    for (std::size_t i = 0; i < result.size(); ++i)
    {
        const auto rotation = Cross(alpha, offsets[i]);
        for (int k = 0; k < 3; ++k)
        {
            result[i][k] =
                (masses[i] / total) * force[k] + masses[i] * rotation[k];
            Finite(result[i][k]);
        }
    }
    return result;
}
}  // namespace SpongeDynamics

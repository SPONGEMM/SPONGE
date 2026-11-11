#include "tangent_frame.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>

#include "../utils/float_classification.hpp"

namespace SpongeDynamics
{
namespace
{
double Dot(const Vec3& a, const Vec3& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
Vec3 Difference(const Vec3& a, const Vec3& b)
{
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}
int Root(std::vector<int>& parent, int i)
{
    while (parent[i] != i)
    {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    return i;
}
// Reject dependent/ill-conditioned constraints; never silently regularize and
// then subtract their nominal count from the temperature degrees of freedom.
void Cholesky(std::vector<double>& a, int n)
{
    double scale = 0;
    for (int i = 0; i < n; ++i) scale = std::max(scale, a[i * n + i]);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j <= i; ++j)
        {
            double value = a[i * n + j];
            for (int k = 0; k < j; ++k) value -= a[i * n + k] * a[j * n + k];
            if (!SpongeFloat::Is_Finite(value) ||
                (i == j && value <= 1e-12 * scale))
                throw std::runtime_error(
                    "dependent or ill-conditioned constraints");
            a[i * n + j] = i == j ? std::sqrt(value) : value / a[j * n + j];
        }
}
std::vector<double> Solve(const std::vector<double>& l, std::vector<double> b)
{
    const int n = b.size();
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < i; ++j) b[i] -= l[i * n + j] * b[j];
        b[i] /= l[i * n + i];
    }
    for (int i = n - 1; i >= 0; --i)
    {
        for (int j = i + 1; j < n; ++j) b[i] -= l[j * n + i] * b[j];
        b[i] /= l[i * n + i];
        if (!SpongeFloat::Is_Finite(b[i]))
            throw std::runtime_error("nonfinite constraint solution");
    }
    return b;
}
}  // namespace

ConstraintStressPlan::ConstraintStressPlan(
    std::vector<double> mass, std::vector<DistanceConstraint> pairs)
    : mass_(std::move(mass)), pairs_(std::move(pairs))
{
    if (mass_.size() < 2 || 3 * mass_.size() <= pairs_.size() + 3)
        throw std::runtime_error("nonpositive constrained thermal DOF");
    for (double m : mass_)
        if (!SpongeFloat::Is_Finite(m) || m <= 0)
            throw std::runtime_error(
                "positive finite constraint masses required");
    std::vector<int> parent(mass_.size());
    std::iota(parent.begin(), parent.end(), 0);
    std::set<std::pair<int, int>> unique;
    for (auto p : pairs_)
    {
        if (p.a < 0 || p.b < 0 || p.a >= int(mass_.size()) ||
            p.b >= int(mass_.size()) || p.a == p.b ||
            !SpongeFloat::Is_Finite(p.length) || p.length <= 0)
            throw std::runtime_error("invalid distance constraint");
        if (!unique.insert(std::minmax(p.a, p.b)).second)
            throw std::runtime_error("duplicate distance constraint");
        parent[Root(parent, p.a)] = Root(parent, p.b);
    }
    std::map<int, std::vector<int>> groups;
    for (int k = 0; k < int(pairs_.size()); ++k)
        groups[Root(parent, pairs_[k].a)].push_back(k);
    for (const auto& group : groups)
    {
        if (group.second.size() > 256)
            throw std::runtime_error(
                "constraint observer component exceeds 256 rows");
        components_.push_back(group.second);
    }
}

ConstraintTangentFrame ConstraintStressPlan::Prepare(
    const std::vector<Vec3>& position, const Vec3& box,
    double length_tolerance) const
{
    const auto n = mass_.size();
    if (position.size() != n || !SpongeFloat::Is_Finite(length_tolerance) ||
        length_tolerance <= 0 || length_tolerance >= 1)
        throw std::runtime_error("invalid constraint geometry frame");
    for (double x : box)
        if (!SpongeFloat::Is_Finite(x) || x <= 0)
            throw std::runtime_error("positive orthogonal box required");
    for (const auto& row : position)
        for (double x : row)
            if (!SpongeFloat::Is_Finite(x))
                throw std::runtime_error("nonfinite constraint coordinates");
    ConstraintTangentFrame frame;
    frame.mass_ = mass_;
    frame.pairs_ = pairs_;
    frame.components_ = components_;
    std::vector<Vec3> unit(pairs_.size());
    std::vector<double> length(pairs_.size());
    for (int k = 0; k < int(pairs_.size()); ++k)
    {
        const auto p = pairs_[k];
        auto d = Difference(position[p.a], position[p.b]);
        for (int j = 0; j < 3; ++j)
        {
            if (p.length * (1 + length_tolerance) >= .5 * box[j])
                throw std::runtime_error("ambiguous periodic constraint image");
            d[j] -= box[j] * std::floor(d[j] / box[j] + .5);
        }
        length[k] = std::sqrt(Dot(d, d));
        double error = std::abs(length[k] / p.length - 1);
        if (!SpongeFloat::Is_Finite(error) || error > length_tolerance)
            throw std::runtime_error(
                "constraint length residual exceeds tolerance");
        frame.max_relative_length_error_ =
            std::max(frame.max_relative_length_error_, error);
        for (int j = 0; j < 3; ++j) unit[k][j] = d[j] / length[k];
    }
    for (const auto& group : components_)
    {
        const int c = group.size();
        std::vector<double> matrix(c * c);
        for (int i = 0; i < c; ++i)
        {
            const auto p = pairs_[group[i]];
            for (int j = 0; j <= i; ++j)
            {
                const auto q = pairs_[group[j]];
                const double weight = (p.a == q.a ? 1 / mass_[p.a] : 0) +
                                      (p.b == q.b ? 1 / mass_[p.b] : 0) -
                                      (p.a == q.b ? 1 / mass_[p.a] : 0) -
                                      (p.b == q.a ? 1 / mass_[p.b] : 0);
                matrix[i * c + j] =
                    weight * Dot(unit[group[i]], unit[group[j]]);
            }
        }
        Cholesky(matrix, c);
        frame.factors_.push_back(std::move(matrix));
    }
    frame.unit_ = std::move(unit);
    frame.length_ = std::move(length);
    return frame;
}

std::vector<Vec3> ConstraintTangentFrame::Project(
    const std::vector<Vec3>& velocity, Stress6* impulse_virial) const
{
    if (impulse_virial) *impulse_virial = {};
    if (velocity.size() != mass_.size())
        throw std::runtime_error("invalid constraint velocity dimensions");
    for (const auto& row : velocity)
        for (double x : row)
            if (!SpongeFloat::Is_Finite(x))
                throw std::runtime_error("nonfinite constraint velocity");
    auto out = velocity;
    for (std::size_t g = 0; g < components_.size(); ++g)
    {
        const auto& group = components_[g];
        const int c = group.size();
        std::vector<double> rhs(c);
        for (int i = 0; i < c; ++i)
        {
            const auto p = pairs_[group[i]];
            rhs[i] = -Dot(unit_[group[i]], Difference(out[p.a], out[p.b]));
        }
        auto projection = Solve(factors_[g], rhs);
        for (int i = 0; i < c; ++i)
        {
            const auto p = pairs_[group[i]];
            for (int j = 0; j < 3; ++j)
            {
                const double impulse = projection[i] * unit_[group[i]][j];
                out[p.a][j] += impulse / mass_[p.a];
                out[p.b][j] -= impulse / mass_[p.b];
            }
            if (impulse_virial)
            {
                const int ax[6] = {0, 1, 2, 0, 0, 1};
                const int ay[6] = {0, 1, 2, 1, 2, 2};
                const int k = group[i];
                for (int j = 0; j < 6; ++j)
                    (*impulse_virial)[j] += projection[i] * length_[k] *
                                            unit_[k][ax[j]] * unit_[k][ay[j]];
            }
        }
    }
    for (const auto& row : out)
        for (double x : row)
            if (!SpongeFloat::Is_Finite(x))
                throw std::runtime_error(
                    "constraint velocity projection overflow");
    if (impulse_virial)
        for (double x : *impulse_virial)
            if (!SpongeFloat::Is_Finite(x))
                throw std::runtime_error("constraint impulse virial overflow");
    return out;
}

ConstrainedDrift ConstraintStressPlan::Drift(
    const std::vector<Vec3>& position, const std::vector<Vec3>& velocity,
    double dt, const Vec3& box, double input_length_tolerance) const
{
    if (!SpongeFloat::Is_Finite(dt) || dt == 0)
        throw std::runtime_error("RATTLE drift requires a finite nonzero step");
    const auto old = Prepare(position, box, input_length_tolerance);
    const auto tangent = old.Project(velocity);
    double residual2 = 0, norm2 = 0;
    for (std::size_t i = 0; i < mass_.size(); ++i)
        for (int j = 0; j < 3; ++j)
        {
            const double residual = tangent[i][j] - velocity[i][j];
            residual2 += mass_[i] * residual * residual;
            norm2 += mass_[i] * velocity[i][j] * velocity[i][j];
        }
    if (!SpongeFloat::Is_Finite(norm2) ||
        residual2 > 1e-20 * std::max(1., norm2))
        throw std::runtime_error("RATTLE drift requires tangent velocity");
    ConstrainedDrift out;
    out.position = position;
    out.velocity = velocity;
    for (std::size_t i = 0; i < mass_.size(); ++i)
        for (int j = 0; j < 3; ++j) out.position[i][j] += dt * velocity[i][j];
    std::vector<double> multipliers(pairs_.size());
    std::vector<Vec3> images(pairs_.size());
    for (int k = 0; k < int(pairs_.size()); ++k)
        for (int j = 0; j < 3; ++j)
            images[k][j] = box[j] * std::floor((position[pairs_[k].a][j] -
                                                position[pairs_[k].b][j]) /
                                                   box[j] +
                                               .5);
    auto displacement = [&](int k)
    {
        const auto p = pairs_[k];
        auto d = Difference(out.position[p.a], out.position[p.b]);
        for (int j = 0; j < 3; ++j) d[j] -= images[k][j];
        return d;
    };
    // Gauss-Seidel Newton solves only the constraints, along fixed old normals.
    // Updating the normals inside this iteration would be a different method.
    constexpr int max_iterations = 256;
    constexpr double length_tolerance = 1e-12;
    bool converged = false;
    for (int iteration = 0; iteration <= max_iterations; ++iteration)
    {
        converged = true;
        for (int k = 0; k < int(pairs_.size()); ++k)
        {
            const auto d = displacement(k);
            const double relative = std::sqrt(Dot(d, d)) / pairs_[k].length;
            // Stay on the local RATTLE branch, including the final check.
            if (!SpongeFloat::Is_Finite(relative) ||
                Dot(d, old.unit_[k]) <= 1e-12 * pairs_[k].length)
                throw std::runtime_error(
                    "RATTLE drift left the local constraint branch");
            if (std::abs(relative - 1) > length_tolerance) converged = false;
        }
        if (converged) break;
        if (iteration == max_iterations)
            throw std::runtime_error("RATTLE position solve did not converge");
        for (int k = 0; k < int(pairs_.size()); ++k)
        {
            const auto p = pairs_[k];
            const auto d = displacement(k);
            const double denominator =
                2 * (1 / mass_[p.a] + 1 / mass_[p.b]) * Dot(d, old.unit_[k]);
            if (!SpongeFloat::Is_Finite(denominator) || denominator <= 0)
                throw std::runtime_error("singular RATTLE position correction");
            const double lambda =
                -(Dot(d, d) - p.length * p.length) / denominator;
            if (!SpongeFloat::Is_Finite(lambda))
                throw std::runtime_error(
                    "nonfinite RATTLE position correction");
            multipliers[k] += lambda;
            for (int j = 0; j < 3; ++j)
            {
                out.position[p.a][j] += lambda * old.unit_[k][j] / mass_[p.a];
                out.position[p.b][j] -= lambda * old.unit_[k][j] / mass_[p.b];
            }
        }
        ++out.iterations;
    }
    // Recover the corrected unconstrained drift velocity from the impulses,
    // not (Q-q)/dt, avoiding cancellation of absolute positions.
    const int ax[6] = {0, 1, 2, 0, 0, 1}, ay[6] = {0, 1, 2, 1, 2, 2};
    for (int k = 0; k < int(pairs_.size()); ++k)
    {
        const auto p = pairs_[k];
        const double impulse = multipliers[k] / dt;
        for (int j = 0; j < 3; ++j)
        {
            out.velocity[p.a][j] += impulse * old.unit_[k][j] / mass_[p.a];
            out.velocity[p.b][j] -= impulse * old.unit_[k][j] / mass_[p.b];
        }
        for (int j = 0; j < 6; ++j)
            out.impulse_virial[j] += impulse * old.length_[k] *
                                     old.unit_[k][ax[j]] * old.unit_[k][ay[j]];
    }
    Stress6 final_impulse{};
    out.velocity = Prepare(out.position, box, 2 * length_tolerance)
                       .Project(out.velocity, &final_impulse);
    for (int j = 0; j < 6; ++j) out.impulse_virial[j] += final_impulse[j];
    for (std::size_t i = 0; i < mass_.size(); ++i)
        for (int j = 0; j < 3; ++j)
            out.kinetic_change += .5 * mass_[i] *
                                  (out.velocity[i][j] - velocity[i][j]) *
                                  (out.velocity[i][j] + velocity[i][j]);
    if (!SpongeFloat::Is_Finite(out.kinetic_change))
        throw std::runtime_error("RATTLE kinetic energy overflow");
    for (double x : out.impulse_virial)
        if (!SpongeFloat::Is_Finite(x))
            throw std::runtime_error("RATTLE impulse virial overflow");
    return out;
}

ConstrainedStress ConstraintStressPlan::Evaluate(
    const std::vector<Vec3>& position, const std::vector<Vec3>& half_velocity,
    const std::vector<Vec3>& force, const Stress6& potential_virial,
    double dt_internal, const Vec3& box, double length_tolerance) const
{
    const auto n = mass_.size();
    if (half_velocity.size() != n || force.size() != n ||
        !SpongeFloat::Is_Finite(dt_internal) || dt_internal < 0)
        throw std::runtime_error("invalid constraint observer frame");
    auto frame = Prepare(position, box, length_tolerance);
    ConstrainedStress out;
    out.velocity = half_velocity;
    out.constraint_force.resize(n);
    out.thermal_dof = frame.Degrees_Of_Freedom() - 3;
    out.max_relative_length_error = frame.max_relative_length_error_;
    for (std::size_t i = 0; i < n; ++i)
        for (int j = 0; j < 3; ++j)
        {
            if (!SpongeFloat::Is_Finite(force[i][j]) ||
                !SpongeFloat::Is_Finite(half_velocity[i][j]))
                throw std::runtime_error("nonfinite constraint frame input");
            out.velocity[i][j] += .5 * dt_internal * force[i][j] / mass_[i];
        }
    out.velocity = frame.Project(out.velocity);
    const auto& unit = frame.unit_;
    const auto& length = frame.length_;
    for (std::size_t g = 0; g < components_.size(); ++g)
    {
        const auto& group = components_[g];
        const int c = group.size();
        std::vector<double> rhs(c);
        for (int i = 0; i < c; ++i)
        {
            const int k = group[i];
            const auto p = pairs_[k];
            const auto v = Difference(out.velocity[p.a], out.velocity[p.b]);
            Vec3 a{};
            for (int j = 0; j < 3; ++j)
                a[j] = force[p.a][j] / mass_[p.a] - force[p.b][j] / mass_[p.b];
            const double radial = Dot(unit[k], v);
            rhs[i] =
                -Dot(unit[k], a) - (Dot(v, v) - radial * radial) / length[k];
            out.max_tangent_residual =
                std::max(out.max_tangent_residual, std::abs(radial));
        }
        auto lambda = Solve(frame.factors_[g], rhs);
        const int ax[6] = {0, 1, 2, 0, 0, 1}, ay[6] = {0, 1, 2, 1, 2, 2};
        for (int i = 0; i < c; ++i)
        {
            const int k = group[i];
            const auto p = pairs_[k];
            for (int j = 0; j < 3; ++j)
            {
                out.constraint_force[p.a][j] += lambda[i] * unit[k][j];
                out.constraint_force[p.b][j] -= lambda[i] * unit[k][j];
            }
            for (int j = 0; j < 6; ++j)
                out.constraint_virial[j] +=
                    lambda[i] * length[k] * unit[k][ax[j]] * unit[k][ay[j]];
        }
    }
    for (int k = 0; k < int(pairs_.size()); ++k)
    {
        const auto p = pairs_[k];
        auto v = Difference(out.velocity[p.a], out.velocity[p.b]);
        Vec3 a{};
        for (int j = 0; j < 3; ++j)
            a[j] = (force[p.a][j] + out.constraint_force[p.a][j]) / mass_[p.a] -
                   (force[p.b][j] + out.constraint_force[p.b][j]) / mass_[p.b];
        const double radial = Dot(unit[k], v);
        const double residual =
            Dot(unit[k], a) + (Dot(v, v) - radial * radial) / length[k];
        out.max_acceleration_residual =
            std::max(out.max_acceleration_residual, std::abs(residual));
    }
    Stress6 total = potential_virial;
    for (int j = 0; j < 6; ++j) total[j] += out.constraint_virial[j];
    out.equilibrium = Centered_Stress(mass_, out.velocity, std::vector<Vec3>(n),
                                      total, 0, box[0] * box[1] * box[2]);
    return out;
}
}  // namespace SpongeDynamics

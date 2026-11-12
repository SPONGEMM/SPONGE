#include "thermal_frame.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "../utils/float_classification.hpp"

namespace SpongeDynamics
{
void ConstrainedFlowFrame::Validate(const std::vector<Vec3>& v) const
{
    if (v.size() != constraints_.Mass().size())
        throw std::runtime_error("invalid constrained flow dimensions");
    for (const auto& row : v)
        for (double x : row)
            if (!SpongeFloat::Is_Finite(x))
                throw std::runtime_error("nonfinite constrained flow input");
}

double ConstrainedFlowFrame::Inner(const std::vector<Vec3>& a,
                                   const std::vector<Vec3>& b) const
{
    double value = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
        for (int j = 0; j < 3; ++j)
            value += constraints_.Mass()[i] * a[i][j] * b[i][j];
    if (!SpongeFloat::Is_Finite(value))
        throw std::runtime_error("constrained flow inner product overflow");
    return value;
}

ConstrainedFlowFrame::ConstrainedFlowFrame(ConstraintTangentFrame constraints,
                                           const std::vector<double>& cosine,
                                           int flow_axis)
    : constraints_(std::move(constraints))
{
    const auto n = constraints_.Mass().size();
    if (cosine.size() != n || flow_axis < 0 || flow_axis > 2 ||
        constraints_.Degrees_Of_Freedom() <= 4)
        throw std::runtime_error(
            "invalid constrained flow mode or thermal DOF");
    for (double c : cosine)
        if (!SpongeFloat::Is_Finite(c))
            throw std::runtime_error("nonfinite constrained flow mode");
    // Equivalent to orthogonalization in sqrt(M)*v coordinates, without
    // assembling a global Jacobian or a global 3N by 3N projector.
    for (int mode = 0; mode < 4; ++mode)
    {
        std::vector<Vec3> raw(n);
        for (std::size_t i = 0; i < n; ++i)
            raw[i][mode < 3 ? mode : flow_axis] = mode < 3 ? 1 : cosine[i];
        const double scale = Inner(raw, raw);
        auto q = constraints_.Project(raw);
        // Two-pass modified Gram-Schmidt, including the global COM modes.
        for (int pass = 0; pass < 2; ++pass)
            for (const auto& b : basis_)
            {
                const double coefficient = Inner(b, q);
                for (std::size_t i = 0; i < n; ++i)
                    for (int j = 0; j < 3; ++j)
                        q[i][j] -= coefficient * b[i][j];
            }
        const double norm2 = Inner(q, q);
        if (scale <= 0 || norm2 <= 1e-12 * scale)
            throw std::runtime_error("dependent constrained flow modes");
        for (auto& row : q)
            for (double& x : row) x /= std::sqrt(norm2);
        basis_.push_back(std::move(q));
    }
}

int ConstrainedFlowFrame::Thermal_Degrees_Of_Freedom() const
{
    return constraints_.Degrees_Of_Freedom() - int(basis_.size());
}

std::vector<Vec3> ConstrainedFlowFrame::Preserved(
    const std::vector<Vec3>& velocity) const
{
    Validate(velocity);
    std::vector<Vec3> out(velocity.size());
    for (const auto& b : basis_)
    {
        const double coefficient = Inner(b, velocity);
        for (std::size_t i = 0; i < out.size(); ++i)
            for (int j = 0; j < 3; ++j) out[i][j] += coefficient * b[i][j];
    }
    Validate(out);
    return out;
}

std::vector<Vec3> ConstrainedFlowFrame::Thermal(
    const std::vector<Vec3>& velocity) const
{
    auto out = constraints_.Project(velocity);
    const auto kept = Preserved(velocity);
    for (std::size_t i = 0; i < out.size(); ++i)
        for (int j = 0; j < 3; ++j) out[i][j] -= kept[i][j];
    Validate(out);
    return out;
}

ConstrainedFlowOU ConstrainedFlowFrame::Apply_OU(
    const std::vector<Vec3>& velocity, const std::vector<Vec3>& normal,
    double damping, double kBT) const
{
    Validate(velocity);
    Validate(normal);
    if (!SpongeFloat::Is_Finite(damping) || damping < 0 || damping > 1 ||
        !SpongeFloat::Is_Finite(kBT) || kBT < 0)
        throw std::runtime_error("invalid constrained flow OU parameters");
    auto tangent = constraints_.Project(velocity);
    auto residual = velocity;
    for (std::size_t i = 0; i < velocity.size(); ++i)
        for (int j = 0; j < 3; ++j) residual[i][j] -= tangent[i][j];
    if (Inner(residual, residual) >
        1e-20 * std::max(1., Inner(velocity, velocity)))
        throw std::runtime_error(
            "constrained flow OU requires tangent velocity");
    const auto kept = Preserved(velocity);
    auto noise = normal;
    for (std::size_t i = 0; i < noise.size(); ++i)
        for (double& x : noise[i]) x /= std::sqrt(constraints_.Mass()[i]);
    noise = Thermal(noise);
    const double sigma = std::sqrt(kBT * (1 - damping) * (1 + damping));
    ConstrainedFlowOU out{velocity, 0};
    if (damping == 1) return out;  // Exact identity, including signed zero.
    for (std::size_t i = 0; i < velocity.size(); ++i)
        for (int j = 0; j < 3; ++j)
            out.velocity[i][j] = kept[i][j] +
                                 damping * (tangent[i][j] - kept[i][j]) +
                                 sigma * noise[i][j];
    Validate(out.velocity);
    // Difference of squares as an increment avoids cancellation of a large
    // preserved-flow energy. This accounts for the actual returned velocities.
    for (std::size_t i = 0; i < velocity.size(); ++i)
        for (int j = 0; j < 3; ++j)
            out.heat += .5 * constraints_.Mass()[i] *
                        (out.velocity[i][j] - velocity[i][j]) *
                        (out.velocity[i][j] + velocity[i][j]);
    if (!SpongeFloat::Is_Finite(out.heat))
        throw std::runtime_error("constrained flow OU heat overflow");
    return out;
}

}  // namespace SpongeDynamics

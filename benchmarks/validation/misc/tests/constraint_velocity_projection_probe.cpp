#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "constrain/settle.h"
#include "constrain/shake.h"
#include "utils/float_classification.hpp"

unsigned int CONTROLLER::device_max_thread = 64;

template <class Real>
struct ProjectionChecks
{
    static constexpr int kAtomCount = 3;
    static constexpr int kPairCount = 3;

    const ConstraintVector<Real> kCoordinates[kAtomCount] = {
        {0.0f, 0.0f, 0.0f},
        {0.8233542f, -0.0292778f, 0.4872141f},
        {-0.6461200f, -0.3741645f, 0.5988970f},
    };
    const ConstraintVector<Real> kVelocities[kAtomCount] = {
        {0.3f, -0.2f, 0.4f},
        {-0.7f, 0.5f, 0.1f},
        {0.8f, 0.9f, -0.3f},
    };
    const Real kMassInverse[kAtomCount] = {1.0f / 16.0f, 1.0f, 1.0f};
    const int kPairs[kPairCount][2] = {{0, 1}, {0, 2}, {1, 2}};

    const ConstraintVector<Real> kRoundoffCoordinates[kAtomCount] = {
        {0.0f, 0.0f, 0.0f},
        {0.5699996948242188f, 0.779998779296875f, -0.25f},
        {-0.5999984741210938f, 0.26000213623046875f, 0.7599983215332031f},
    };
    const ConstraintVector<Real> kRoundoffVelocities[kAtomCount] = {
        {-0.174088716506958f, -0.10837503522634506f, 0.24693767726421356f},
        {-1.4800273180007935f, 1.8255033493041992f, 0.039367739111185074f},
        {-0.05815054848790169f, 1.55351984500885f, -1.609876275062561f},
    };
    const Real kRoundoffMassInverse[kAtomCount] = {
        1.0f / 15.9994f, 1.0f / 1.008f, 1.0f / 1.008f};

    bool Fail(const char* message)
    {
        std::fprintf(stderr, "%s\n", message);
        return false;
    }

    Real Maximum_Relative_Residual(const ConstraintVector<Real>* coordinates,
                                   const ConstraintVector<Real>* velocities)
    {
        Real maximum = 0.0f;
        for (const auto& pair : kPairs)
        {
            const ConstraintVector<Real> displacement =
                coordinates[pair[0]] - coordinates[pair[1]];
            const ConstraintVector<Real> velocity_difference =
                velocities[pair[0]] - velocities[pair[1]];
            const Real numerator =
                std::fabs(displacement * velocity_difference);
            const Real denominator =
                std::sqrt((displacement * displacement) *
                          (velocity_difference * velocity_difference));
            maximum =
                std::fmax(maximum, numerator / std::fmax(denominator, 1e-12f));
        }
        return maximum;
    }

    bool Check_Velocity_Only_Result(
        const char* name, const ConstraintVector<Real>* original_coordinates,
        const ConstraintVector<Real>* projected_coordinates,
        const ConstraintVector<Real>* projected_velocities)
    {
        if (std::memcmp(original_coordinates, projected_coordinates,
                        sizeof(kCoordinates)) != 0)
            return Fail("velocity-only projection changed coordinates");
        const Real residual = Maximum_Relative_Residual(projected_coordinates,
                                                        projected_velocities);
        if (!SpongeFloat::Is_Finite(residual) || residual > 2e-5f)
        {
            std::fprintf(stderr, "%s residual %.9g exceeds tolerance\n", name,
                         static_cast<double>(residual));
            return false;
        }
        return true;
    }

    bool Check_Roundoff_Floor_Result(
        const char* name, const ConstraintVector<Real>* original_coordinates,
        const ConstraintVector<Real>* projected_coordinates,
        const ConstraintVector<Real>* projected_velocities)
    {
        if (std::memcmp(original_coordinates, projected_coordinates,
                        sizeof(kRoundoffCoordinates)) != 0)
            return Fail("roundoff-floor projection changed coordinates");

        bool exercised_roundoff_floor = false;
        for (const auto& pair : kPairs)
        {
            const ConstraintVector<Real> displacement =
                projected_coordinates[pair[0]] - projected_coordinates[pair[1]];
            const ConstraintVector<Real> velocity_i =
                projected_velocities[pair[0]];
            const ConstraintVector<Real> velocity_j =
                projected_velocities[pair[1]];
            const ConstraintVector<Real> velocity_difference =
                velocity_i - velocity_j;
            const Real displacement_squared = displacement * displacement;
            const Real projection = displacement * velocity_difference;
            const Real relative_scale = ConstraintSqrt(
                displacement_squared *
                fmax(velocity_difference * velocity_difference, 1.0e-12f));
            const Real relative_limit = 1.0e-5f * relative_scale;
            const Real velocity_scale =
                fmax(ConstraintSqrt(velocity_i * velocity_i),
                     ConstraintSqrt(velocity_j * velocity_j));
            const Real roundoff_floor = 8.0f * ConstraintEpsilon<Real>() *
                                        ConstraintSqrt(displacement_squared) *
                                        velocity_scale;
            const Real tolerance = fmax(relative_limit, roundoff_floor);
            if (!SpongeFloat::Is_Finite(projection) ||
                !SpongeFloat::Is_Finite(tolerance) ||
                std::fabs(projection) > tolerance)
            {
                std::fprintf(
                    stderr,
                    "%s residual %.9g exceeds attainable tolerance %.9g\n",
                    name, static_cast<double>(std::fabs(projection)),
                    static_cast<double>(tolerance));
                return false;
            }
            exercised_roundoff_floor |=
                std::fabs(projection) > relative_limit &&
                roundoff_floor > relative_limit;
        }
        return std::is_same<Real, double>::value || exercised_roundoff_floor ||
               Fail("roundoff-floor case did not exercise the Real bound");
    }

    bool Check_Settle()
    {
        CONSTRAIN constrain;
        constrain.dt = 0.002f;
        // An unrelated high-degree SHAKE component must not reduce the SETTLE
        // projection step or make an otherwise valid water fail to converge.
        constrain.maximum_constraint_degree = 1024;

        ConstraintTriangle<Real> triangle = {};
        triangle.atom_A = 0;
        triangle.atom_B = 1;
        triangle.atom_C = 2;
        ConstraintVector<Real> delta_velocity[kAtomCount] = {};

        SETTLE_SOLVER<Real> settle;
        settle.is_initialized = 1;
        settle.constrain = &constrain;
        settle.local_atom_numbers = kAtomCount;
        settle.num_triangle_local = 1;
        settle.d_triangles_local = &triangle;
        settle.d_delta_vel_local = delta_velocity;

        ConstraintVector<Real> coordinates[kAtomCount];
        ConstraintVector<Real> velocities[kAtomCount];
        std::memcpy(coordinates, kCoordinates, sizeof(coordinates));
        std::memcpy(velocities, kVelocities, sizeof(velocities));
        const Boundary boundary = Boundary{};
        if (!settle.Project_Velocity_To_Constraint_Manifold(
                velocities, coordinates, kMassInverse, boundary, false))
            return Fail("SETTLE velocity-only projection did not converge");
        if (!Check_Velocity_Only_Result("SETTLE", kCoordinates, coordinates,
                                        velocities))
            return false;

        std::memcpy(coordinates, kRoundoffCoordinates, sizeof(coordinates));
        std::memcpy(velocities, kRoundoffVelocities, sizeof(velocities));
        if (!settle.Project_Velocity_To_Constraint_Manifold(
                velocities, coordinates, kRoundoffMassInverse, boundary, false))
            return Fail("SETTLE rejected a Real-limited attainable projection");
        if (!Check_Roundoff_Floor_Result("SETTLE", kRoundoffCoordinates,
                                         coordinates, velocities))
            return false;

        std::memcpy(coordinates, kCoordinates, sizeof(coordinates));
        std::memcpy(velocities, kVelocities, sizeof(velocities));
        if (!settle.Project_Velocity_To_Constraint_Manifold(
                velocities, coordinates, kMassInverse, boundary))
            return Fail("SETTLE default projection reported failure");
        return std::memcmp(kCoordinates, coordinates, sizeof(coordinates)) !=
                   0 ||
               Fail("SETTLE default projection no longer updates coordinates");
    }

    bool Check_Shake()
    {
        CONSTRAIN constrain;
        constrain.dt = 0.002f;
        constrain.maximum_constraint_degree = 2;
        CONSTRAIN_PAIR pairs[kPairCount] = {};
        for (int index = 0; index < kPairCount; ++index)
        {
            pairs[index].atom_i_serial = kPairs[index][0];
            pairs[index].atom_j_serial = kPairs[index][1];
        }
        constrain.num_pair_local = kPairCount;
        constrain.constrain_pair_local = pairs;

        ConstraintVector<Real> correction[kAtomCount] = {};
        SHAKE_SOLVER<Real> shake;
        shake.is_initialized = 1;
        shake.constrain = &constrain;
        shake.constrain_frc = correction;

        ConstraintVector<Real> coordinates[kAtomCount];
        ConstraintVector<Real> velocities[kAtomCount];
        std::memcpy(coordinates, kCoordinates, sizeof(coordinates));
        std::memcpy(velocities, kVelocities, sizeof(velocities));
        const Boundary boundary = Boundary{};
        if (!shake.Project_Velocity_To_Constraint_Manifold(
                velocities, coordinates, kMassInverse, boundary, kAtomCount,
                false))
            return Fail("SHAKE velocity-only projection did not converge");
        if (!Check_Velocity_Only_Result("SHAKE", kCoordinates, coordinates,
                                        velocities))
            return false;

        std::memcpy(coordinates, kRoundoffCoordinates, sizeof(coordinates));
        std::memcpy(velocities, kRoundoffVelocities, sizeof(velocities));
        if (!shake.Project_Velocity_To_Constraint_Manifold(
                velocities, coordinates, kRoundoffMassInverse, boundary,
                kAtomCount, false))
            return Fail("SHAKE rejected a Real-limited attainable projection");
        if (!Check_Roundoff_Floor_Result("SHAKE", kRoundoffCoordinates,
                                         coordinates, velocities))
            return false;

        std::memcpy(coordinates, kCoordinates, sizeof(coordinates));
        std::memcpy(velocities, kVelocities, sizeof(velocities));
        if (!shake.Project_Velocity_To_Constraint_Manifold(
                velocities, coordinates, kMassInverse, boundary, kAtomCount))
            return Fail("SHAKE default projection reported failure");
        return std::memcmp(kCoordinates, coordinates, sizeof(coordinates)) !=
                   0 ||
               Fail("SHAKE default projection no longer updates coordinates");
    }

    bool Check_Boundary_Projection()
    {
        const LTMatrix3 cell(10.0f, 0.0f, 10.0f, 0.0f, 0.0f, 10.0f);
        const LTMatrix3 rcell(0.1f, 0.0f, 0.1f, 0.0f, 0.0f, 0.1f);
        const Boundary boundaries[] = {{BoundaryPolicy::Open, cell, rcell},
                                       {BoundaryPolicy::Periodic, cell, rcell}};
        const ConstraintVector<Real> expected_displacements[] = {
            {11.0f, 1.0f, 0.0f}, {1.0f, 1.0f, 0.0f}};
        const ConstraintVector<Real> original_coordinates[2] = {
            {21.0f, 0.0f, 0.0f}, {10.0f, -1.0f, 0.0f}};
        const ConstraintVector<Real> original_velocities[2] = {
            {1.0f, 2.0f, 3.0f}, {-1.0f, 0.0f, 1.0f}};
        const Real mass_inverse[2] = {1.0f, 1.0f};
        CONSTRAIN_PAIR pair = {};
        pair.atom_i_serial = 0;
        pair.atom_j_serial = 1;
        CONSTRAIN constrain;
        constrain.dt = 0.002f;
        constrain.maximum_constraint_degree = 1;
        constrain.num_pair_local = 1;
        constrain.constrain_pair_local = &pair;
        ConstraintVector<Real> correction[2] = {};
        SETTLE_SOLVER<Real> settle;
        settle.is_initialized = 1;
        settle.constrain = &constrain;
        settle.local_atom_numbers = 2;
        settle.num_pair_local = 1;
        settle.d_pairs_local = &pair;
        settle.d_delta_vel_local = correction;
        SHAKE_SOLVER<Real> shake;
        shake.is_initialized = 1;
        shake.constrain = &constrain;
        shake.constrain_frc = correction;
        for (int policy = 0; policy < 2; ++policy)
        {
            const ConstraintVector<Real> dr = expected_displacements[policy];
            const ConstraintVector<Real> dv =
                original_velocities[0] - original_velocities[1];
            const ConstraintVector<Real> delta =
                (-0.5f * (dr * dv) / (dr * dr)) * dr;
            const ConstraintVector<Real> expected[2] = {
                original_velocities[0] + delta, original_velocities[1] - delta};
            for (int algorithm = 0; algorithm < 2; ++algorithm)
            {
                ConstraintVector<Real> coordinates[2], velocities[2];
                std::memcpy(coordinates, original_coordinates,
                            sizeof(coordinates));
                std::memcpy(velocities, original_velocities,
                            sizeof(velocities));
                const bool converged =
                    algorithm == 0
                        ? settle.Project_Velocity_To_Constraint_Manifold(
                              velocities, coordinates, mass_inverse,
                              boundaries[policy], false)
                        : shake.Project_Velocity_To_Constraint_Manifold(
                              velocities, coordinates, mass_inverse,
                              boundaries[policy], 2, false);
                if (!converged || std::memcmp(coordinates, original_coordinates,
                                              sizeof(coordinates)) != 0)
                    return Fail(
                        "boundary projection failed or moved coordinates");
                for (int atom = 0; atom < 2; ++atom)
                {
                    const ConstraintVector<Real> error =
                        velocities[atom] - expected[atom];
                    if (error * error > 1.0e-10f)
                        return Fail(
                            "projection used the wrong boundary displacement");
                }
            }
        }
        return true;
    }

    bool Check_Degenerate_Constraint_Status()
    {
        CONSTRAIN constrain;
        constrain.dt = 0.002f;
        constrain.maximum_constraint_degree = 1;

        CONSTRAIN_PAIR pair = {};
        pair.atom_i_serial = 0;
        pair.atom_j_serial = 1;
        constrain.num_pair_local = 1;
        constrain.constrain_pair_local = &pair;

        ConstraintVector<Real> coordinates[2] = {};
        ConstraintVector<Real> velocities[2] = {
            ConstraintVector<Real>(1.0f, 0.0f, 0.0f),
            ConstraintVector<Real>(-1.0f, 0.0f, 0.0f)};
        const Real dynamic_mass_inverse[2] = {1.0f, 1.0f};
        const Real fixed_mass_inverse[2] = {0.0f, 0.0f};
        ConstraintVector<Real> correction[2] = {};
        const Boundary boundary = Boundary{};

        SETTLE_SOLVER<Real> settle;
        settle.is_initialized = 1;
        settle.constrain = &constrain;
        settle.local_atom_numbers = 2;
        settle.num_pair_local = 1;
        settle.d_pairs_local = &pair;
        settle.d_delta_vel_local = correction;
        if (settle.Project_Velocity_To_Constraint_Manifold(
                velocities, coordinates, dynamic_mass_inverse, boundary, false))
            return Fail("SETTLE accepted a degenerate dynamic constraint");
        if (!settle.Project_Velocity_To_Constraint_Manifold(
                velocities, coordinates, fixed_mass_inverse, boundary, false))
            return Fail("SETTLE rejected an all-fixed constraint");

        SHAKE_SOLVER<Real> shake;
        shake.is_initialized = 1;
        shake.constrain = &constrain;
        shake.constrain_frc = correction;
        if (shake.Project_Velocity_To_Constraint_Manifold(
                velocities, coordinates, dynamic_mass_inverse, boundary, 2,
                false))
            return Fail("SHAKE accepted a degenerate dynamic constraint");
        if (!shake.Project_Velocity_To_Constraint_Manifold(
                velocities, coordinates, fixed_mass_inverse, boundary, 2,
                false))
            return Fail("SHAKE rejected an all-fixed constraint");
        return true;
    }

    bool Run()
    {
        return Check_Settle() && Check_Shake() && Check_Boundary_Projection() &&
               Check_Degenerate_Constraint_Status();
    }
};

int main()
{
    return ProjectionChecks<float>().Run() && ProjectionChecks<double>().Run()
               ? 0
               : 1;
}

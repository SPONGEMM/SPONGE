#include <cmath>
#include <cstdio>
#include <cstring>
#include <initializer_list>

#ifndef SPONGE_SETTLE_SOURCE
#define SPONGE_SETTLE_SOURCE "constrain/settle.cpp"
#endif
#include SPONGE_SETTLE_SOURCE

unsigned int CONTROLLER::device_max_thread = 64;

namespace
{
const int ends[3][2] = {{0, 1}, {0, 2}, {1, 2}};

float Component(const VECTOR& v, int d)
{
    return d == 0 ? v.x : (d == 1 ? v.y : v.z);
}

// Independent double Newton solve: corrections are along the OLD bonds,
// weighted by inverse mass. This solves the position SHAKE equations directly,
// without using SETTLE's rotations or geometry coefficients.
bool Reference(const VECTOR old[3], const VECTOR trial[3], const float mass[3],
               double corrected[3][3])
{
    double direction[3][3][3] = {}, target2[3] = {};
    for (int a = 0; a < 3; ++a)
    {
        for (int d = 0; d < 3; ++d)
            corrected[a][d] = Component(trial[a],d);
    }
    for (int k = 0; k < 3; ++k)
    {
        const int a = ends[k][0], b = ends[k][1];
        for (int d = 0; d < 3; ++d)
        {
            const double bond = Component(old[a],d) -
                                Component(old[b],d);
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
                    matrix[k][j] += 2 * bond *
                        (direction[j][a][d] - direction[j][b][d]);
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
                for (int j = k; j < 4; ++j) matrix[i][j] -= scale*matrix[k][j];
            }
        }
        for (int a = 0; a < 3; ++a)
            for (int d = 0; d < 3; ++d)
                for (int k = 0; k < 3; ++k)
                    corrected[a][d] += direction[k][a][d]*matrix[k][3];
    }
    return false;
}

VECTOR Permute(VECTOR v, int axis)
{
    for (int k = 0; k < axis; ++k) v = {v.z, v.x, v.y};
    return v;
}
}  // namespace

int main()
{
    double max_correction_error = 0, max_momentum_error = 0, max_stationary_speed = 0;
    // Dyadic fixtures preserve exactly the same local input under translation.
    // Therefore their impulses must be independent of absolute coordinates,
    // even though the final coordinates remain stored in float.
    for (float mass_c : {1.0f, 2.0f})
    for (int deformed : {0, 1})
    for (int axis = 0; axis < 3; ++axis)
    for (int periodic : {0, 1})
    for (float dt : {0.01f, 0.0025f, 0.001f})
    for (float gamma : {1.0f, 0.97f})
    {
        // The legacy analytic rotation assumes equal B/C masses for a driven
        // triangle. Check unequal masses only for the stationary precision case;
        // extending that solver is separate from translation stability.
        if (mass_c != 1.0f && deformed) continue;
        const float factor = 0.5f*(1.0f + gamma);
        const float mass[3] = {16, 1, mass_c};
        Boundary boundary = {};
        if (periodic)
        {
            boundary.policy = BoundaryPolicy::Periodic;
            boundary.cell = {32,0,32,0,0,32};
            boundary.rcell = {1.f/32,0,1.f/32,0,0,1.f/32};
        }
        VECTOR old[3] = {{0,0,0}, Permute({.375f,.5f,0},axis),
                                  Permute({-.375f,.5f,0},axis)};
        VECTOR trial[3] = {old[0], old[1], old[2]};
        if (deformed)
        {
            trial[0] = trial[0] + Permute({1.f/1024, -2.f/1024, 1.f/1024},axis);
            trial[1] = trial[1] + Permute({-3.f/1024, 1.f/1024, -2.f/1024},axis);
            trial[2] = trial[2] + Permute({2.f/1024, 3.f/1024, 1.f/1024},axis);
        }
        double reference[3][3];
        if (!Reference(old,trial,mass,reference)) return 2;
        CONSTRAIN_TRIANGLE triangle = {};
        triangle.atom_A = 0; triangle.atom_B = 1; triangle.atom_C = 2;
        Get_Rabcde_From_SSS(.625f,.625f,.75f,16,1,mass_c,triangle.ra,
                           triangle.rb,triangle.rc,triangle.rd,triangle.re);
        VECTOR first_velocity[3] = {};
        LTMatrix3 first_virial = {};
        for (float shift : {0.f,16.f,128.f,1024.f})
        {
            VECTOR crd[3], velocity[3] = {};
            for (int a = 0; a < 3; ++a)
                crd[a] = Wrap_Coordinate(trial[a] + VECTOR{shift,-shift,shift},boundary);
            const VECTOR original[3] = {crd[0],crd[1],crd[2]};
            LTMatrix3 virial = {};
            settle_triangle(1,&triangle,mass,crd,boundary,&old[1],&old[2],
                            dt,gamma,factor,velocity,&virial);
            double momentum[3] = {};
            for (int a = 0; a < 3; ++a)
            for (int d = 0; d < 3; ++d)
            {
                const double value = Component(velocity[a],d);
                const double correction = value*dt*factor/gamma;
                const double expected = reference[a][d] -
                    Component(trial[a],d);
                const double error = std::fabs(correction-expected);
                max_correction_error = std::fmax(max_correction_error,error);
                momentum[d] += mass[a]*correction;
                if (!SpongeFloat::Is_Finite(value) || error > 3e-7)
                {
                    std::fprintf(stderr,"correction error %.9g (shift %.1f, dt %.4g)\n",error,shift,dt);
                    return 3;
                }
                if (!deformed) max_stationary_speed = std::fmax(max_stationary_speed,std::fabs(value));
                // Float write-back can round away a sub-ULP correction. Compare
                // with the independently rounded final position, not a new
                // velocity inferred by subtracting those rounded positions.
                const float expected_position = static_cast<float>(
                    reference[a][d] + Component(original[a],d) - Component(trial[a],d));
                const float actual_position = Component(crd[a],d);
                if (std::fabs(actual_position-expected_position) >
                    3e-7 + 2e-7*std::fabs(expected_position)) return 4;
            }
            for (double p : momentum)
            {
                max_momentum_error = std::fmax(max_momentum_error,std::fabs(p));
                if (std::fabs(p) > 2e-6)
                {
                    std::fprintf(stderr,"mass-weighted correction %.9g (shift %.1f, dt %.4g)\n",p,shift,dt);
                    return 5;
                }
            }
            if (shift == 0)
            {
                std::memcpy(first_velocity,velocity,sizeof(velocity));
                first_virial = virial;
            }
            else if (std::memcmp(first_velocity,velocity,sizeof(velocity)) != 0 ||
                     std::memcmp(&first_virial,&virial,sizeof(virial)) != 0)
            {
                std::fputs("constraint impulse or virial depends on absolute translation\n",stderr);
                return 6;
            }
        }
    }
    std::printf("max correction error %.9g; mass-weighted correction %.9g; stationary speed %.9g\n",
                max_correction_error,max_momentum_error,max_stationary_speed);
    return 0;
}

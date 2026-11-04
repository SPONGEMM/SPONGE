#include <cmath>
#include <cstdio>
#include <initializer_list>
#include <vector>

#include "PM_force/PM_force.cpp"

unsigned int CONTROLLER::device_max_thread = 64;

namespace
{
bool Check_R2C_Weights(int nz)
{
    const int count = nz / 2 + 1;
    std::vector<LTMatrix3> coefficients(count, LTMatrix3{1, 2, 3, 4, 5, 6});
    std::vector<FFT_COMPLEX> spectrum(count);
    for (int plane : {0, 1, count - 1})
    {
        for (auto& value : spectrum)
        {
            REAL(value) = 0;
            IMAGINARY(value) = 0;
        }
        REAL(spectrum[plane]) = 1;
        IMAGINARY(spectrum[plane]) = 2;
        LTMatrix3 actual = {};
        PME_Sum_Virial(count, coefficients.data(), spectrum.data(), &actual,
                       nz);
        const bool self_conjugate =
            plane == 0 || (nz % 2 == 0 && plane == nz / 2);
        const float scale = self_conjugate ? 2.5f : 5.f;
        const float values[6] = {actual.a11, actual.a21, actual.a22,
                                 actual.a31, actual.a32, actual.a33};
        for (int i = 0; i < 6; ++i)
            if (std::fabs(values[i] - scale * (i + 1)) > 1e-6)
            {
                std::fprintf(stderr,
                             "R2C weight/sign failure (fftz %d, plane %d)\n",
                             nz, plane);
                return false;
            }
    }
    return true;
}

// One conjugate mode pair has U=exp(-p*m^2)/(pi*m^2*V).
// Differentiate its energy under each Cartesian strain independently of the
// production coefficient formula. F^-T changes reciprocal vectors, det(F)
// changes the cell volume. This also exercises coefficient rebuilding.
bool Check_Box_Coefficients(float lx, float ly, float lz, float xy = 0,
                            float xz = 0, float yz = 0)
{
    const int nx = 4, ny = 4, nz = 5, count = nx * ny * (nz / 2 + 1);
    std::vector<float> bc(count), interpolation(count, 1);
    std::vector<LTMatrix3> coefficients(count);
    const LTMatrix3 inverse = {1 / lx, xy, 1 / ly, xz, yz, 1 / lz};
    const float prefactor = 3.5f, volume = lx * ly * lz;
    up_box_bc(nx, ny, nz, bc.data(), interpolation.data(), coefficients.data(),
              prefactor, inverse, volume);
    const int index = (1 * ny + 1) * (nz / 2 + 1) + 1;
    const double m[3] = {1.0 / lx, xy + 1.0 / ly, xz + yz + 1.0 / lz};
    const int axes[6][2] = {{0, 0}, {1, 0}, {1, 1}, {2, 0}, {2, 1}, {2, 2}};
    const LTMatrix3 actual = coefficients[index];
    const float values[6] = {actual.a11, actual.a21, actual.a22,
                             actual.a31, actual.a32, actual.a33};
    for (int i = 0; i < 6; ++i)
    {
        const int a = axes[i][0], b = axes[i][1];
        auto energy = [&](double strain)
        {
            double k[3] = {m[0], m[1], m[2]};
            const double determinant = 1 + (a == b ? strain : 0);
            k[b] -= strain * m[a] / determinant;
            const double k2 = k[0] * k[0] + k[1] * k[1] + k[2] * k[2];
            return std::exp(-prefactor * k2) /
                   (std::acos(-1.0) * k2 * volume * determinant);
        };
        const double h = 1e-5;
        const double expected =
            -(energy(h) - energy(-h)) / (2 * h) * (a == b ? 1 : 2);
        if (std::fabs(values[i] - expected) > 1e-8 + 3e-6 * std::fabs(expected))
        {
            std::fprintf(stderr,
                         "updated box coefficient %d: %.9g versus %.9g\n", i,
                         values[i], expected);
            return false;
        }
    }
    return true;
}
}  // namespace

int main()
{
    return Check_R2C_Weights(5) && Check_R2C_Weights(6) &&
                   Check_Box_Coefficients(10, 12, 14) &&
                   Check_Box_Coefficients(11, 13, 17) &&
                   Check_Box_Coefficients(10, 12, 14, .013f, -.017f, .019f) &&
                   Check_Box_Coefficients(11, 13, 17, -.021f, .009f, -.011f)
               ? 0
               : 1;
}

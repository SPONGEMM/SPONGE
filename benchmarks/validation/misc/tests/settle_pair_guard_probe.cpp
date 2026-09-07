#include <cmath>
#include <cstdio>
#include <cstring>

#include "constrain/settle.cpp"
#include "utils/float_classification.hpp"

unsigned int CONTROLLER::device_max_thread = 64;

namespace
{

bool Check_Pair(const char* name, const VECTOR& predicted,
                const VECTOR& previous, const float target,
                const bool should_fail)
{
    CONSTRAIN_PAIR pair = {};
    pair.atom_i_serial = 0;
    pair.atom_j_serial = 1;
    pair.constant_r = target;
    pair.constrain_k = 0.5f;
    const int atom_local[2] = {10, 11};
    const float mass[2] = {1.0f, 1.0f};
    VECTOR coordinates[2] = {{0.0f, 0.0f, 0.0f}, predicted};
    VECTOR velocities[2] = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
    VECTOR last_pair[1] = {previous};
    LTMatrix3 virial[1] = {{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}};
    const VECTOR original_coordinates[2] = {coordinates[0], coordinates[1]};
    const VECTOR original_velocities[2] = {velocities[0], velocities[1]};
    const LTMatrix3 original_virial = virial[0];
    const LTMatrix3 cell;
    int invalid_pair = -1;

    settle_pair(1, &pair, atom_local, mass, coordinates, cell, cell, last_pair,
                0.002f, 1.0f, 1.0f, velocities, virial, &invalid_pair);

    if (should_fail)
    {
        const bool unchanged =
            std::memcmp(coordinates, original_coordinates,
                        sizeof(coordinates)) == 0 &&
            std::memcmp(velocities, original_velocities, sizeof(velocities)) ==
                0 &&
            std::memcmp(virial, &original_virial, sizeof(original_virial)) == 0;
        if (invalid_pair == 0 && unchanged) return true;
        std::fprintf(stderr, "%s did not fail before writing state\n", name);
        return false;
    }

    const VECTOR constrained = coordinates[1] - coordinates[0];
    const float distance = std::sqrt(constrained * constrained);
    const bool finite_distance = SpongeFloat::Is_Finite(distance);
    if (invalid_pair == -1 && finite_distance &&
        std::fabs(distance - target) <= 2.0e-6f)
        return true;
    std::fprintf(stderr, "%s rejected a valid pair (distance %.9g)\n", name,
                 static_cast<double>(distance));
    return false;
}

bool Check_Float_Cancellation_Case()
{
    const VECTOR predicted = {1.50678098f, -2.55530214f, -1.57559633f};
    const VECTOR previous = {-0.689187527f, 0.64532584f, 0.32950747f};
    const float r1r1 = predicted * predicted;
    const float r1r2 = predicted * previous;
    const float r2r2 = previous * previous;
    const float float_radicand = r1r2 * r1r2 - r1r1 * r2r2 + r2r2;
    const double precise_radicand =
        Settle_Pair_Precise_Radicand(predicted, previous, 1.0f, NULL);
    if (!(float_radicand < 0.0f) ||
        !Settle_Double_Is_Finite_Nonnegative(precise_radicand))
    {
        std::fprintf(
            stderr,
            "roundoff fixture no longer exercises float cancellation\n");
        return false;
    }
    return Check_Pair("float-cancellation pair", predicted, previous, 1.0f,
                      false);
}

bool Check_Float_False_Positive_Case()
{
    const VECTOR predicted = {0.769227564f, 0.589192092f, 1.73570371f};
    const VECTOR previous = {0.359873652f, 0.728374541f, 0.583062232f};
    const float r1r1 = predicted * predicted;
    const float r1r2 = predicted * previous;
    const float r2r2 = previous * previous;
    const float float_radicand = r1r2 * r1r2 - r1r1 * r2r2 + r2r2;
    const double precise_radicand =
        Settle_Pair_Precise_Radicand(predicted, previous, 1.0f, NULL);
    if (!(float_radicand > 0.0f) ||
        Settle_Double_Is_Finite_Nonnegative(precise_radicand))
    {
        std::fprintf(stderr,
                     "roundoff fixture no longer exercises a float false "
                     "positive\n");
        return false;
    }
    return Check_Pair("float-false-positive pair", predicted, previous, 1.0f,
                      true);
}

bool Check_Precise_K_Overflow()
{
    const VECTOR predicted = {0.0f, 0.0f, 0.0f};
    const VECTOR previous = {FLT_MIN, 0.0f, 0.0f};
    float k = 123.0f;
    return !Settle_Try_Precise_Pair_Solution(predicted, previous, FLT_MAX,
                                             &k) &&
           k == 123.0f;
}

}  // namespace

int main()
{
    const VECTOR old_x = {1.0f, 0.0f, 0.0f};
    const VECTOR ordinary = {1.2f, 0.2f, 0.0f};
    const VECTOR just_inside = {0.0f, std::nextafter(1.0f, 0.0f), 0.0f};
    const VECTOR tangent = {0.0f, 1.0f, 0.0f};
    const VECTOR just_outside = {0.0f, std::nextafter(1.0f, 2.0f), 0.0f};
    const VECTOR observed = {-0.511489868f, 1.50683594f, 0.117248535f};
    const VECTOR observed_previous = {0.13999939f, 0.509994507f, 0.939994812f};

    return Check_Pair("ordinary pair", ordinary, old_x, 1.0f, false) &&
                   Check_Pair("inside-boundary pair", just_inside, old_x, 1.0f,
                              false) &&
                   Check_Pair("tangent pair", tangent, old_x, 1.0f, false) &&
                   Check_Float_Cancellation_Case() &&
                   Check_Float_False_Positive_Case() &&
                   Check_Precise_K_Overflow() &&
                   Check_Pair("outside-boundary pair", just_outside, old_x,
                              1.0f, true) &&
                   Check_Pair("observed negative-radicand pair", observed,
                              observed_previous, 1.08000004f, true)
               ? 0
               : 1;
}

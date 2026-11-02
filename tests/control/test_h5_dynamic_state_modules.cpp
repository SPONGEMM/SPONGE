#include <cmath>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>

#include "../../SPONGE/utils/float_classification.hpp"
#include "barostat/pressure_based_barostat.h"
#include "thermostat/Bussi_thermostat.h"
#include "utils/random/portable_philox_sampler.hpp"
#include "utils/random/restart_rng_state.hpp"

int CONTROLLER::MPI_rank = 0;

static void Require(bool value)
{
    if (!value)
    {
        throw std::runtime_error("requirement failed");
    }
}

static void Test_Bussi_H5_Restart_State_Round_Trips()
{
    BUSSI_THERMOSTAT_INFORMATION source;
    source.is_initialized = 1;
    source.lambda = 0.87f;
    source.random_seed = 12345;
    source.random_invocation_count = 7;

    SpongeH5MD::RestartDynamicState state;
    std::string error;
    Require(source.Export_H5_Restart_State(&state, &error));
    Require(state.rng_states.count("bussi_thermostat") == 1);
    Require(state.rng_state_text.count("bussi_thermostat") == 0);
    Require(
        state.thermostat_float_states["bussi_thermostat"]["lambda"].size() ==
        1);

    BUSSI_THERMOSTAT_INFORMATION target;
    target.is_initialized = 1;
    target.lambda = 1.0f;
    target.random_seed = 67890;
    Require(target.Apply_H5_Restart_State(state, &error));
    Require(std::fabs(target.lambda - 0.87f) < 1.0e-6f);
    Require(target.random_seed == source.random_seed);
    Require(target.random_invocation_count == source.random_invocation_count);
    Require(!target.use_legacy_rng);
}

static void Test_Bussi_H5_Restart_Requires_Lambda()
{
    BUSSI_THERMOSTAT_INFORMATION source;
    source.is_initialized = 1;
    source.lambda = 0.87f;
    source.random_seed = 12345;

    SpongeH5MD::RestartDynamicState state;
    std::string error;
    Require(source.Export_H5_Restart_State(&state, &error));
    state.thermostat_float_states["bussi_thermostat"].erase("lambda");

    BUSSI_THERMOSTAT_INFORMATION target;
    target.is_initialized = 1;
    Require(!target.Apply_H5_Restart_State(state, &error));
    Require(error.find("lambda") != std::string::npos);
}

static void Test_Bussi_Legacy_Text_Rng_Remains_Read_Compatible()
{
    std::default_random_engine expected(31415);
    std::ostringstream encoded;
    encoded << expected;
    SpongeH5MD::RestartDynamicState state;
    state.rng_state_text["bussi_thermostat"] = encoded.str();
    state.thermostat_float_states["bussi_thermostat"]["lambda"] = {0.91f};

    BUSSI_THERMOSTAT_INFORMATION target;
    target.is_initialized = 1;
    std::string error;
    Require(target.Apply_H5_Restart_State(state, &error));
    Require(target.use_legacy_rng);
    Require(target.legacy_engine() == expected());

    SpongeH5MD::RestartDynamicState exported;
    Require(target.Export_H5_Restart_State(&exported, &error));
    Require(exported.rng_state_text.count("bussi_thermostat") == 1);
    Require(exported.rng_states.count("bussi_thermostat") == 0);
}

static void Test_Pressure_Barostat_H5_Restart_State_Round_Trips()
{
    PRESSURE_BASED_BAROSTAT_INFORMATION source;
    source.is_initialized = 1;
    source.g.a11 = 1.0f;
    source.g.a21 = 2.0f;
    source.g.a22 = 3.0f;
    source.g.a31 = 4.0f;
    source.g.a32 = 5.0f;
    source.g.a33 = 6.0f;
    source.random_seed = 24680;
    source.random_invocation_count = 9;

    SpongeH5MD::RestartDynamicState state;
    std::string error;
    Require(source.Export_H5_Restart_State(&state, &error));
    Require(state.rng_states.count("pressure_based_barostat") == 1);
    Require(state.rng_state_text.count("pressure_based_barostat") == 0);
    Require(
        state.barostat_float_states["pressure_based_barostat"]["g"].size() ==
        6);

    PRESSURE_BASED_BAROSTAT_INFORMATION target;
    target.is_initialized = 1;
    target.random_seed = 13579;
    Require(target.Apply_H5_Restart_State(state, &error));
    Require(std::fabs(target.g.a11 - 1.0f) < 1.0e-6f);
    Require(std::fabs(target.g.a21 - 2.0f) < 1.0e-6f);
    Require(std::fabs(target.g.a22 - 3.0f) < 1.0e-6f);
    Require(std::fabs(target.g.a31 - 4.0f) < 1.0e-6f);
    Require(std::fabs(target.g.a32 - 5.0f) < 1.0e-6f);
    Require(std::fabs(target.g.a33 - 6.0f) < 1.0e-6f);
    Require(target.random_seed == source.random_seed);
    Require(target.random_invocation_count == source.random_invocation_count);
    Require(!target.use_legacy_rng);
}

static void Test_Pressure_Barostat_H5_Restart_Requires_Rng()
{
    PRESSURE_BASED_BAROSTAT_INFORMATION source;
    source.is_initialized = true;
    source.g = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    source.random_seed = 24680;

    SpongeH5MD::RestartDynamicState state;
    std::string error;
    Require(source.Export_H5_Restart_State(&state, &error));
    state.rng_states.erase("pressure_based_barostat");

    PRESSURE_BASED_BAROSTAT_INFORMATION target;
    target.is_initialized = true;
    Require(!target.Apply_H5_Restart_State(state, &error));
    Require(error.find("RNG state") != std::string::npos);
}

static void Test_Pressure_Barostat_Legacy_Text_Rng_Remains_Read_Compatible()
{
    std::default_random_engine expected(27182);
    std::ostringstream encoded;
    encoded << expected;
    SpongeH5MD::RestartDynamicState state;
    state.rng_state_text["pressure_based_barostat"] = encoded.str();
    state.barostat_float_states["pressure_based_barostat"]["g"] = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    PRESSURE_BASED_BAROSTAT_INFORMATION target;
    target.is_initialized = true;
    std::string error;
    Require(target.Apply_H5_Restart_State(state, &error));
    Require(target.use_legacy_rng);
    Require(target.generator() == expected());

    SpongeH5MD::RestartDynamicState exported;
    Require(target.Export_H5_Restart_State(&exported, &error));
    Require(exported.rng_state_text.count("pressure_based_barostat") == 1);
    Require(exported.rng_states.count("pressure_based_barostat") == 0);
}

static void Test_Portable_Philox_Sampler_Is_Deterministic()
{
    SPONGE_PORTABLE_PHILOX_SAMPLER first(12345, 7);
    SPONGE_PORTABLE_PHILOX_SAMPLER repeated(12345, 7);
    SPONGE_PORTABLE_PHILOX_SAMPLER next(12345, 8);
    const double first_normal = first.Normal();
    const double repeated_normal = repeated.Normal();
    const double next_normal = next.Normal();
    Require(SpongeFloat::Is_Finite(first_normal));
    Require(first_normal == repeated_normal);
    Require(first_normal != next_normal);
    const double first_gamma = first.Gamma(8.0, 2.0);
    const double repeated_gamma = repeated.Gamma(8.0, 2.0);
    Require(SpongeFloat::Is_Finite(first_gamma));
    Require(first_gamma > 0.0);
    Require(first_gamma == repeated_gamma);
}

static void Test_Portable_Philox_Is_Addressable_And_Round_Trips()
{
    float first[4] = {};
    float repeated[4] = {};
    float next[4] = {};
    SPONGE_PHILOX4X32_10(0x123456789abcdef0ULL, 7, 44).Normal4(first);
    SPONGE_PHILOX4X32_10(0x123456789abcdef0ULL, 7, 44).Normal4(repeated);
    SPONGE_PHILOX4X32_10(0x123456789abcdef0ULL, 7, 48).Normal4(next);
    for (int index = 0; index < 4; ++index)
    {
        Require(SpongeFloat::Is_Finite(first[index]));
        Require(first[index] == repeated[index]);
    }
    Require(first[0] != next[0] || first[1] != next[1] || first[2] != next[2] ||
            first[3] != next[3]);

    const auto encoded = SpongeRestartRng::Counter_Philox_State(
        0x123456789abcdef0ULL, 0x123456789ULL);
    std::uint64_t seed = 0;
    std::uint64_t invocation_count = 0;
    std::string error;
    Require(SpongeRestartRng::Decode_Counter_Philox_State(
        encoded, &seed, &invocation_count, &error));
    Require(seed == 0x123456789abcdef0ULL);
    Require(invocation_count == 0x123456789ULL);

    auto malformed = encoded;
    malformed.state_words[0] = -1;
    Require(!SpongeRestartRng::Decode_Counter_Philox_State(
        malformed, &seed, &invocation_count, &error));
}

static void Test_Portable_Splitmix_State_Round_Trips()
{
    const auto encoded =
        SpongeRestartRng::Splitmix64_State(0x123456789abcdef0ULL);
    std::uint64_t decoded = 0;
    std::string error;
    Require(
        SpongeRestartRng::Decode_Splitmix64_State(encoded, &decoded, &error));
    Require(decoded == 0x123456789abcdef0ULL);
}

int main()
{
    Test_Bussi_H5_Restart_State_Round_Trips();
    Test_Bussi_H5_Restart_Requires_Lambda();
    Test_Bussi_Legacy_Text_Rng_Remains_Read_Compatible();
    Test_Pressure_Barostat_H5_Restart_State_Round_Trips();
    Test_Pressure_Barostat_H5_Restart_Requires_Rng();
    Test_Pressure_Barostat_Legacy_Text_Rng_Remains_Read_Compatible();
    Test_Portable_Philox_Sampler_Is_Deterministic();
    Test_Portable_Philox_Is_Addressable_And_Round_Trips();
    Test_Portable_Splitmix_State_Round_Trips();
    return 0;
}

#pragma once

#include <cstdint>
#include <string>

#include "../h5md/h5_structural_state.hpp"

namespace SpongeRestartRng
{
inline std::int64_t Low_Word(std::uint64_t value)
{
    return static_cast<std::int64_t>(
        static_cast<std::uint32_t>(value & 0xffffffffULL));
}

inline std::int64_t High_Word(std::uint64_t value)
{
    return static_cast<std::int64_t>(static_cast<std::uint32_t>(value >> 32));
}

inline std::uint64_t Join_Words(std::int64_t low, std::int64_t high)
{
    return static_cast<std::uint64_t>(static_cast<std::uint32_t>(low)) |
           (static_cast<std::uint64_t>(static_cast<std::uint32_t>(high)) << 32);
}

inline SpongeH5MD::RestartRngState Counter_Philox_State(
    std::uint64_t seed, std::uint64_t invocation_count)
{
    SpongeH5MD::RestartRngState state;
    state.engine = "sponge.philox4x32-10.counter.v1";
    state.stream_count = 1;
    state.words_per_stream = 4;
    state.state_words = {Low_Word(seed), High_Word(seed),
                         Low_Word(invocation_count),
                         High_Word(invocation_count)};
    return state;
}

inline bool Decode_Counter_Philox_State(
    const SpongeH5MD::RestartRngState& state, std::uint64_t* seed,
    std::uint64_t* invocation_count, std::string* error_message)
{
    auto fail = [error_message](const std::string& message)
    {
        if (error_message != nullptr) *error_message = message;
        return false;
    };
    if (seed == nullptr || invocation_count == nullptr)
    {
        return fail("counter Philox restart output pointer is null");
    }
    if (state.engine != "sponge.philox4x32-10.counter.v1" ||
        state.state_schema_version != 1 || state.stream_count != 1 ||
        state.words_per_stream != 4 || state.state_words.size() != 4)
    {
        return fail("counter Philox restart state has an incompatible schema");
    }
    for (const std::int64_t word : state.state_words)
    {
        if (word < 0 || word > 0xffffffffLL)
        {
            return fail(
                "counter Philox restart state contains a non-uint32 word");
        }
    }
    *seed = Join_Words(state.state_words[0], state.state_words[1]);
    *invocation_count = Join_Words(state.state_words[2], state.state_words[3]);
    if (*invocation_count > UINT64_MAX / 4)
    {
        return fail("counter Philox restart invocation count would overflow");
    }
    return true;
}

inline SpongeH5MD::RestartRngState Splitmix64_State(std::uint64_t value)
{
    SpongeH5MD::RestartRngState state;
    state.engine = "sponge.splitmix64.v1";
    state.stream_count = 1;
    state.words_per_stream = 2;
    state.state_words = {Low_Word(value), High_Word(value)};
    return state;
}

inline bool Decode_Splitmix64_State(const SpongeH5MD::RestartRngState& state,
                                    std::uint64_t* value,
                                    std::string* error_message)
{
    auto fail = [error_message](const std::string& message)
    {
        if (error_message != nullptr) *error_message = message;
        return false;
    };
    if (value == nullptr)
        return fail("SplitMix64 restart output pointer is null");
    if (state.engine != "sponge.splitmix64.v1" ||
        state.state_schema_version != 1 || state.stream_count != 1 ||
        state.words_per_stream != 2 || state.state_words.size() != 2)
    {
        return fail("SplitMix64 restart state has an incompatible schema");
    }
    for (const std::int64_t word : state.state_words)
    {
        if (word < 0 || word > 0xffffffffLL)
        {
            return fail("SplitMix64 restart state contains a non-uint32 word");
        }
    }
    *value = Join_Words(state.state_words[0], state.state_words[1]);
    return true;
}
}  // namespace SpongeRestartRng

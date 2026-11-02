#pragma once

#include <algorithm>
#include <cctype>
#include <cstring>
#include <string>

namespace SpongeH5InputContract
{
static constexpr const char* kTopologyPathKey = "input_h5_topology_path";
static constexpr const char* kProtocolPathKey = "input_h5_protocol_path";
static constexpr const char* kRestartPathKey = "input_h5_restart_path";
static constexpr const char* kRestartLoadKey = "input_h5_restart_load";
static constexpr const char* kTrajectoryPathKey = "input_h5_trajectory_path";
static constexpr const char* kTrajectoryParticleStreamKey =
    "input_h5_trajectory_particle_stream";

static constexpr const char* kTopologySuffix = ".spgt.h5";
static constexpr const char* kProtocolSuffix = ".spgp.h5";
static constexpr const char* kRestartSuffix = ".spgr.h5";
static constexpr const char* kTrajectorySuffix = ".spg.h5md";

static constexpr const char* kDefaultRestartLoad = "structural";
static constexpr const char* kDefaultTrajectoryParticleStream = "all";

enum class RestartLoadPolicy
{
    structural,
    dynamic,
    protocol,
    full,
    custom,
    invalid
};

inline std::string Lowercase(std::string text)
{
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return text;
}

inline bool Ends_With(const std::string& text, const char* suffix)
{
    if (suffix == nullptr)
    {
        return false;
    }
    const std::size_t suffix_size = std::strlen(suffix);
    return text.size() >= suffix_size &&
           text.compare(text.size() - suffix_size, suffix_size, suffix) == 0;
}

inline RestartLoadPolicy Parse_Restart_Load_Policy(const char* value)
{
    const std::string text = Lowercase(
        value == nullptr || value[0] == '\0' ? kDefaultRestartLoad : value);
    if (text == "structural")
    {
        return RestartLoadPolicy::structural;
    }
    if (text == "dynamic")
    {
        return RestartLoadPolicy::dynamic;
    }
    if (text == "protocol")
    {
        return RestartLoadPolicy::protocol;
    }
    if (text == "full")
    {
        return RestartLoadPolicy::full;
    }
    if (text == "custom")
    {
        return RestartLoadPolicy::custom;
    }
    return RestartLoadPolicy::invalid;
}

inline const char* Restart_Load_Policy_Name(const RestartLoadPolicy policy)
{
    switch (policy)
    {
        case RestartLoadPolicy::structural:
            return "structural";
        case RestartLoadPolicy::dynamic:
            return "dynamic";
        case RestartLoadPolicy::protocol:
            return "protocol";
        case RestartLoadPolicy::full:
            return "full";
        case RestartLoadPolicy::custom:
            return "custom";
        case RestartLoadPolicy::invalid:
        default:
            return "invalid";
    }
}

}  // namespace SpongeH5InputContract

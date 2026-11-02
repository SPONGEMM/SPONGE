#pragma once

#include <cctype>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <string>

namespace SpongeH5OutputContract
{
// Parser-visible keys produced by grouped TOML:
// [output.h5.trajectory] path/chunk_size/vds/repair_policy
// [output.h5.restart] path/topology_hash/atom_order_hash/protocol_hash
// [output.h5.observable] path
static constexpr const char* kTrajectoryPathKey = "output_h5_trajectory_path";
static constexpr const char* kTrajectoryVdsKey = "output_h5_trajectory_vds";
static constexpr const char* kTrajectoryChunkSizeKey =
    "output_h5_trajectory_chunk_size";
static constexpr const char* kTrajectoryRepairPolicyKey =
    "output_h5_trajectory_repair_policy";
static constexpr const char* kRestartPathKey = "output_h5_restart_path";
static constexpr const char* kRestartTopologyHashKey =
    "output_h5_restart_topology_hash";
static constexpr const char* kRestartAtomOrderHashKey =
    "output_h5_restart_atom_order_hash";
static constexpr const char* kRestartProtocolHashKey =
    "output_h5_restart_protocol_hash";
static constexpr const char* kObservablePathKey = "output_h5_observable_path";

static constexpr bool kDefaultTrajectoryVds = false;
static constexpr int kDefaultTrajectoryChunkSize = 20;
static constexpr const char* kDefaultTrajectoryRepairPolicy = "strict";

static constexpr const char* kTrajectorySuffix = ".spg.h5md";
static constexpr const char* kRestartSuffix = ".spgr.h5";
static constexpr const char* kObservableSuffix = ".obs.spg.h5md";

inline bool Ends_With(const std::string& value, const std::string& suffix)
{
    if (value.size() < suffix.size())
    {
        return false;
    }
    return value.compare(value.size() - suffix.size(), suffix.size(), suffix) ==
           0;
}

inline bool Has_Recommended_Suffix(const char* path, const char* suffix)
{
    if (path == NULL || suffix == NULL)
    {
        return false;
    }
    return Ends_With(path, suffix);
}

inline const char* Recommended_Suffix_For_Key(const char* key)
{
    if (key == NULL)
    {
        return NULL;
    }
    const std::string key_string = key;
    if (key_string == kTrajectoryPathKey)
    {
        return kTrajectorySuffix;
    }
    if (key_string == kRestartPathKey)
    {
        return kRestartSuffix;
    }
    if (key_string == kObservablePathKey)
    {
        return kObservableSuffix;
    }
    return NULL;
}

inline bool Command_Has_Non_Empty_Value(CONTROLLER* controller, const char* key)
{
    if (controller == NULL || key == NULL || !controller->Command_Exist(key))
    {
        return false;
    }
    const char* value = controller->Command(key);
    return value != NULL && value[0] != '\0';
}

inline bool Any_H5_Output_Enabled(CONTROLLER* controller)
{
    return Command_Has_Non_Empty_Value(controller, kTrajectoryPathKey) ||
           Command_Has_Non_Empty_Value(controller, kRestartPathKey) ||
           Command_Has_Non_Empty_Value(controller, kObservablePathKey);
}

inline bool Is_Canonical_Sha256(const std::string& value)
{
    static constexpr const char* prefix = "sha256:";
    if (value.size() != 71 || value.compare(0, 7, prefix) != 0)
    {
        return false;
    }
    for (std::size_t i = 7; i < value.size(); ++i)
    {
        const char c = value[i];
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')))
        {
            return false;
        }
    }
    return true;
}

inline bool Legacy_Sidecars_Default_Enabled(CONTROLLER* controller)
{
    return !Any_H5_Output_Enabled(controller);
}

inline bool Legacy_Sidecar_Requested(CONTROLLER* controller,
                                     const char* legacy_key)
{
    if (controller == NULL || legacy_key == NULL)
    {
        return false;
    }
    return controller->Command_Exist(legacy_key);
}

inline bool Legacy_Sidecar_Enabled(CONTROLLER* controller,
                                   const char* legacy_key)
{
    return Legacy_Sidecars_Default_Enabled(controller) ||
           Legacy_Sidecar_Requested(controller, legacy_key);
}

inline int Trajectory_Chunk_Size(CONTROLLER* controller)
{
    if (controller != NULL &&
        controller->Command_Exist(kTrajectoryChunkSizeKey))
    {
        const char* text = controller->Command(kTrajectoryChunkSizeKey);
        if (text == NULL || text[0] == '\0')
        {
            return 0;
        }
        char* end = NULL;
        errno = 0;
        const long value = std::strtol(text, &end, 10);
        if (end == text || errno == ERANGE || value > INT_MAX ||
            value < INT_MIN)
        {
            return 0;
        }
        while (end != NULL && *end != '\0')
        {
            if (!std::isspace(static_cast<unsigned char>(*end)))
            {
                return 0;
            }
            ++end;
        }
        return static_cast<int>(value);
    }
    return kDefaultTrajectoryChunkSize;
}
}  // namespace SpongeH5OutputContract

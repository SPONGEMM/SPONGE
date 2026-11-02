#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace SpongeH5OutputPlan
{
struct TrajectoryH5OutputPlan
{
    bool enabled = false;
    std::string path;
    bool vds = SpongeH5OutputContract::kDefaultTrajectoryVds;
    int chunk_size = SpongeH5OutputContract::kDefaultTrajectoryChunkSize;
    std::string repair_policy =
        SpongeH5OutputContract::kDefaultTrajectoryRepairPolicy;
    bool allow_complete_prefix_repair = false;
    std::string derived_shard_root;
    bool has_recommended_suffix = false;
};

struct RestartH5OutputPlan
{
    bool enabled = false;
    std::string path;
    std::string topology_hash;
    std::string atom_order_hash;
    std::string protocol_hash;
    bool has_recommended_suffix = false;
};

struct ObservableH5OutputPlan
{
    bool enabled = false;
    std::string path;
    bool has_recommended_suffix = false;
};

struct LegacySidecarPlan
{
    std::string key;
    bool enabled = false;
    bool explicit_path = false;
    std::string path;
};

struct LegacyOutputPlan
{
    bool default_enabled = true;
    std::vector<LegacySidecarPlan> sidecars;

    bool Enabled(const char* key) const
    {
        if (key == NULL)
        {
            return false;
        }
        for (const auto& sidecar : sidecars)
        {
            if (sidecar.key == key)
            {
                return sidecar.enabled;
            }
        }
        return false;
    }

    bool Explicitly_Requested(const char* key) const
    {
        if (key == NULL)
        {
            return false;
        }
        for (const auto& sidecar : sidecars)
        {
            if (sidecar.key == key)
            {
                return sidecar.explicit_path;
            }
        }
        return false;
    }
};

struct ResolvedOutputPlan
{
    bool valid = true;
    std::string error_message;
    bool any_h5_output_enabled = false;
    TrajectoryH5OutputPlan trajectory;
    RestartH5OutputPlan restart;
    ObservableH5OutputPlan observable;
    LegacyOutputPlan legacy;
};

inline bool Parse_Bool(
    const char* value,
    const bool default_value = SpongeH5OutputContract::kDefaultTrajectoryVds)
{
    if (value == NULL)
    {
        return default_value;
    }
    std::string text = value;
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return text == "1" || text == "true" || text == "yes" || text == "on";
}

inline bool Is_Bool_Text(const char* value)
{
    if (value == NULL)
    {
        return false;
    }
    std::string text = value;
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return text == "1" || text == "true" || text == "yes" || text == "on" ||
           text == "0" || text == "false" || text == "no" || text == "off";
}

inline std::string Command_String(CONTROLLER* controller, const char* key)
{
    if (controller == NULL || key == NULL || !controller->Command_Exist(key))
    {
        return "";
    }
    return controller->Command(key);
}

inline std::string Lowercase(std::string text)
{
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return text;
}

inline std::string Derive_Shards_Root(const std::string& trajectory_path)
{
    if (SpongeH5OutputContract::Ends_With(
            trajectory_path, SpongeH5OutputContract::kTrajectorySuffix))
    {
        return trajectory_path.substr(
                   0,
                   trajectory_path.size() -
                       std::strlen(SpongeH5OutputContract::kTrajectorySuffix)) +
               ".spg.shards";
    }
    return trajectory_path + ".shards";
}

inline LegacyOutputPlan Resolve_Legacy_Output_Plan(CONTROLLER* controller)
{
    static constexpr std::array<const char*, 7> legacy_keys = {
        "mdout", "mdinfo", "crd", "box", "vel", "frc", "rst"};

    LegacyOutputPlan plan;
    plan.default_enabled =
        SpongeH5OutputContract::Legacy_Sidecars_Default_Enabled(controller);
    plan.sidecars.reserve(legacy_keys.size());
    for (const char* key : legacy_keys)
    {
        LegacySidecarPlan sidecar;
        sidecar.key = key;
        sidecar.explicit_path =
            SpongeH5OutputContract::Legacy_Sidecar_Requested(controller, key);
        sidecar.enabled = plan.default_enabled || sidecar.explicit_path;
        if (sidecar.explicit_path)
        {
            sidecar.path = controller->Command(key);
        }
        plan.sidecars.push_back(sidecar);
    }
    return plan;
}

inline void Collect_Explicit_Legacy_Sidecars(const LegacyOutputPlan& plan,
                                             std::vector<std::string>* keys,
                                             std::vector<std::string>* paths)
{
    if (keys == NULL || paths == NULL)
    {
        return;
    }
    keys->clear();
    paths->clear();
    for (const auto& sidecar : plan.sidecars)
    {
        if (sidecar.explicit_path)
        {
            keys->push_back(sidecar.key);
            paths->push_back(sidecar.path);
        }
    }
}

inline ResolvedOutputPlan Resolve_Output_Plan(CONTROLLER* controller,
                                              bool throw_on_error = true)
{
    ResolvedOutputPlan plan;
    if (controller == NULL)
    {
        plan.valid = false;
        plan.error_message = "CONTROLLER is null";
        return plan;
    }

    plan.trajectory.enabled =
        SpongeH5OutputContract::Command_Has_Non_Empty_Value(
            controller, SpongeH5OutputContract::kTrajectoryPathKey);
    if (plan.trajectory.enabled)
    {
        plan.trajectory.path =
            controller->Command(SpongeH5OutputContract::kTrajectoryPathKey);
        plan.trajectory.has_recommended_suffix =
            SpongeH5OutputContract::Has_Recommended_Suffix(
                plan.trajectory.path.c_str(),
                SpongeH5OutputContract::kTrajectorySuffix);
        plan.trajectory.derived_shard_root =
            Derive_Shards_Root(plan.trajectory.path);
    }

    if (controller->Command_Exist(SpongeH5OutputContract::kTrajectoryVdsKey))
    {
        const char* vds_value =
            controller->Command(SpongeH5OutputContract::kTrajectoryVdsKey);
        if (!Is_Bool_Text(vds_value))
        {
            plan.valid = false;
            plan.error_message = "output_h5_trajectory_vds must be boolean";
        }
        else
        {
            plan.trajectory.vds = Parse_Bool(vds_value);
        }
    }

    plan.trajectory.chunk_size =
        SpongeH5OutputContract::Trajectory_Chunk_Size(controller);
    if (plan.trajectory.chunk_size <= 0)
    {
        plan.valid = false;
        plan.error_message =
            "output_h5_trajectory_chunk_size must be greater than 0";
    }

    if (controller->Command_Exist(
            SpongeH5OutputContract::kTrajectoryRepairPolicyKey))
    {
        plan.trajectory.repair_policy = Lowercase(controller->Command(
            SpongeH5OutputContract::kTrajectoryRepairPolicyKey));
    }
    if (plan.trajectory.repair_policy == "strict")
    {
        plan.trajectory.allow_complete_prefix_repair = false;
    }
    else if (plan.trajectory.repair_policy == "complete_prefix" ||
             plan.trajectory.repair_policy == "allow_complete_prefix")
    {
        plan.trajectory.allow_complete_prefix_repair = true;
    }
    else
    {
        plan.valid = false;
        plan.error_message =
            "output_h5_trajectory_repair_policy must be strict or "
            "complete_prefix";
    }
    if (plan.trajectory.allow_complete_prefix_repair &&
        (!plan.trajectory.enabled || !plan.trajectory.vds))
    {
        plan.valid = false;
        plan.error_message =
            "output_h5_trajectory_repair_policy=complete_prefix requires "
            "output_h5_trajectory_path and output_h5_trajectory_vds=true";
    }

    plan.restart.enabled = SpongeH5OutputContract::Command_Has_Non_Empty_Value(
        controller, SpongeH5OutputContract::kRestartPathKey);
    plan.restart.topology_hash = Command_String(
        controller, SpongeH5OutputContract::kRestartTopologyHashKey);
    plan.restart.atom_order_hash = Command_String(
        controller, SpongeH5OutputContract::kRestartAtomOrderHashKey);
    plan.restart.protocol_hash = Command_String(
        controller, SpongeH5OutputContract::kRestartProtocolHashKey);
    if (plan.restart.enabled)
    {
        plan.restart.path =
            controller->Command(SpongeH5OutputContract::kRestartPathKey);
        plan.restart.has_recommended_suffix =
            SpongeH5OutputContract::Has_Recommended_Suffix(
                plan.restart.path.c_str(),
                SpongeH5OutputContract::kRestartSuffix);
    }

    plan.observable.enabled =
        SpongeH5OutputContract::Command_Has_Non_Empty_Value(
            controller, SpongeH5OutputContract::kObservablePathKey);
    if (plan.observable.enabled)
    {
        plan.observable.path =
            controller->Command(SpongeH5OutputContract::kObservablePathKey);
        plan.observable.has_recommended_suffix =
            SpongeH5OutputContract::Has_Recommended_Suffix(
                plan.observable.path.c_str(),
                SpongeH5OutputContract::kObservableSuffix);
    }

    plan.any_h5_output_enabled = plan.trajectory.enabled ||
                                 plan.restart.enabled ||
                                 plan.observable.enabled;
    plan.legacy = Resolve_Legacy_Output_Plan(controller);

    if (!plan.valid && throw_on_error)
    {
        controller->Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                                       "Resolve_Output_Plan",
                                       plan.error_message.c_str());
    }
    return plan;
}
}  // namespace SpongeH5OutputPlan

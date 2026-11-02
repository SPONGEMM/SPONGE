#pragma once

#include <initializer_list>
#include <string>

#include "../control/h5_input_contract.hpp"

namespace SpongeH5InputPlan
{
struct H5Binding
{
    bool enabled = false;
    std::string path;
    bool has_recommended_suffix = false;
};

struct RestartInputPlan
{
    H5Binding binding;
    SpongeH5InputContract::RestartLoadPolicy load_policy =
        SpongeH5InputContract::RestartLoadPolicy::structural;
};

struct TrajectoryInputPlan
{
    H5Binding binding;
    std::string particle_stream =
        SpongeH5InputContract::kDefaultTrajectoryParticleStream;
};

struct ResolvedInputPlan
{
    bool valid = true;
    std::string error_message;
    bool any_h5_input_enabled = false;
    bool legacy_input_allowed = true;
    H5Binding topology;
    H5Binding protocol;
    RestartInputPlan restart;
    TrajectoryInputPlan trajectory;
};

template <typename ControllerType>
inline std::string Command_String(ControllerType* controller, const char* key)
{
    if (controller == nullptr || key == nullptr ||
        !controller->Command_Exist(key))
    {
        return "";
    }
    return controller->Command(key);
}

inline H5Binding Make_Binding(const std::string& path,
                              const char* recommended_suffix)
{
    H5Binding binding;
    binding.enabled = !path.empty();
    binding.path = path;
    binding.has_recommended_suffix =
        binding.enabled &&
        SpongeH5InputContract::Ends_With(path, recommended_suffix);
    return binding;
}

inline void Fail(ResolvedInputPlan* plan, const std::string& message)
{
    if (plan == nullptr || !plan->valid)
    {
        return;
    }
    plan->valid = false;
    plan->error_message = message;
}

template <typename ControllerType>
inline bool Has_Any_Command(ControllerType* controller,
                            std::initializer_list<const char*> keys)
{
    if (controller == nullptr)
    {
        return false;
    }
    for (const char* key : keys)
    {
        if (controller->Command_Exist(key))
        {
            return true;
        }
    }
    return false;
}

template <typename ControllerType>
inline bool Mode_Is_Rerun(ControllerType* controller)
{
    const std::string mode =
        SpongeH5InputContract::Lowercase(Command_String(controller, "mode"));
    return mode == "rerun";
}

template <typename ControllerType>
inline ResolvedInputPlan Resolve_Input_Plan(ControllerType* controller)
{
    ResolvedInputPlan plan;
    plan.topology = Make_Binding(
        Command_String(controller, SpongeH5InputContract::kTopologyPathKey),
        SpongeH5InputContract::kTopologySuffix);
    plan.protocol = Make_Binding(
        Command_String(controller, SpongeH5InputContract::kProtocolPathKey),
        SpongeH5InputContract::kProtocolSuffix);
    plan.restart.binding = Make_Binding(
        Command_String(controller, SpongeH5InputContract::kRestartPathKey),
        SpongeH5InputContract::kRestartSuffix);
    plan.trajectory.binding = Make_Binding(
        Command_String(controller, SpongeH5InputContract::kTrajectoryPathKey),
        SpongeH5InputContract::kTrajectorySuffix);

    const std::string trajectory_stream = Command_String(
        controller, SpongeH5InputContract::kTrajectoryParticleStreamKey);
    if (!trajectory_stream.empty())
    {
        plan.trajectory.particle_stream = trajectory_stream;
    }

    plan.any_h5_input_enabled =
        plan.topology.enabled || plan.protocol.enabled ||
        plan.restart.binding.enabled || plan.trajectory.binding.enabled;
    plan.legacy_input_allowed = !plan.any_h5_input_enabled;

    if (!plan.any_h5_input_enabled)
    {
        return plan;
    }

    if (Has_Any_Command(controller, {"gromacs_top", "gromacs_gro",
                                     "gromacs_include_dir", "gromacs_define"}))
    {
        Fail(&plan,
             "HDF5 input paths support only SPONGE native inputs; GROMACS "
             "input commands (gromacs_top/gromacs_gro/gromacs_include_dir/"
             "gromacs_define) cannot be used when any input_h5_*_path is "
             "configured");
        return plan;
    }

    if (Has_Any_Command(controller, {"amber_parm7", "amber_rst7"}))
    {
        Fail(&plan,
             "HDF5 input paths support only SPONGE native inputs; AMBER input "
             "commands (amber_parm7/amber_rst7) cannot be used when any "
             "input_h5_*_path is configured");
        return plan;
    }

    if (!plan.topology.enabled)
    {
        Fail(&plan, std::string("missing required H5 input binding: ") +
                        SpongeH5InputContract::kTopologyPathKey);
        return plan;
    }
    if (!plan.protocol.enabled)
    {
        Fail(&plan, std::string("missing required H5 input binding: ") +
                        SpongeH5InputContract::kProtocolPathKey);
        return plan;
    }
    const bool rerun_mode = Mode_Is_Rerun(controller);
    if (!plan.restart.binding.enabled &&
        !(rerun_mode && plan.trajectory.binding.enabled))
    {
        Fail(&plan, std::string("missing required H5 input binding: ") +
                        SpongeH5InputContract::kRestartPathKey);
        return plan;
    }

    const std::string load_text =
        Command_String(controller, SpongeH5InputContract::kRestartLoadKey);
    plan.restart.load_policy =
        SpongeH5InputContract::Parse_Restart_Load_Policy(load_text.c_str());
    if (plan.restart.load_policy ==
        SpongeH5InputContract::RestartLoadPolicy::invalid)
    {
        Fail(&plan, "invalid input_h5_restart_load value: " + load_text);
        return plan;
    }
    if (plan.restart.load_policy ==
        SpongeH5InputContract::RestartLoadPolicy::custom)
    {
        Fail(&plan,
             "input_h5_restart_load = custom is reserved until component-list "
             "keys are implemented");
        return plan;
    }

    if (plan.trajectory.binding.enabled && !rerun_mode)
    {
        Fail(&plan,
             "input_h5_trajectory_path is currently only valid with mode = "
             "rerun");
        return plan;
    }

    if (plan.trajectory.binding.enabled &&
        Has_Any_Command(controller, {"crd", "box", "vel"}))
    {
        Fail(&plan,
             "input_h5_trajectory_path is mutually exclusive with legacy rerun "
             "crd/box/vel inputs");
        return plan;
    }

    if (plan.restart.binding.enabled &&
        Has_Any_Command(controller,
                        {"coordinate_in_file", "velocity_in_file", "rst7"}))
    {
        Fail(&plan,
             "input_h5_restart_path is mutually exclusive with legacy "
             "coordinate/velocity restart inputs");
        return plan;
    }

    return plan;
}

template <typename ControllerType>
inline bool Has_H5_Input_Binding(ControllerType* controller)
{
    return Resolve_Input_Plan(controller).any_h5_input_enabled;
}

}  // namespace SpongeH5InputPlan

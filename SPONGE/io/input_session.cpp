#include "input_session.h"

#include "../utils/float_classification.hpp"

namespace SpongeIO
{
namespace
{
bool Requests_H5_Dynamic_State(
    const SpongeH5InputContract::RestartLoadPolicy policy)
{
    return policy == SpongeH5InputContract::RestartLoadPolicy::dynamic ||
           policy == SpongeH5InputContract::RestartLoadPolicy::full;
}

bool Requests_H5_Protocol_State(
    const SpongeH5InputContract::RestartLoadPolicy policy)
{
    return policy == SpongeH5InputContract::RestartLoadPolicy::protocol ||
           policy == SpongeH5InputContract::RestartLoadPolicy::full;
}
}  // namespace

void InputSession::Load_System()
{
    Validate_H5_Input_Plan();
    runtime_.system.Load_Inputs(&runtime_.controller, input_);
}

void InputSession::Apply_H5_Dynamic_Integrator_State(
    const SpongeH5MD::RestartDynamicState& dynamic_state)
{
    const auto mode = dynamic_state.integrator_state_text.find("mode");
    if (mode == dynamic_state.integrator_state_text.end())
    {
        return;
    }
    if (mode->second != Current_MD_Mode_Name(runtime_.md_info))
    {
        const std::string message =
            std::string("Reason:\n\tRestart integrator mode is ") +
            mode->second + ", but current mode is " +
            Current_MD_Mode_Name(runtime_.md_info) + "\n";
        runtime_.controller.Throw_SPONGE_Error(spongeErrorConflictingCommand,
                                               "Apply_H5_Dynamic_Restart_State",
                                               message.c_str());
    }
    const auto step = dynamic_state.integrator_state_text.find("step");
    const auto time = dynamic_state.integrator_state_text.find("time");
    if ((step == dynamic_state.integrator_state_text.end()) !=
        (time == dynamic_state.integrator_state_text.end()))
    {
        runtime_.controller.Throw_SPONGE_Error(
            spongeErrorValueErrorCommand, "Apply_H5_Dynamic_Restart_State",
            "Reason:\n\tRestart integrator state must contain both step and "
            "time\n");
    }
    if (step == dynamic_state.integrator_state_text.end()) return;
    try
    {
        std::size_t step_consumed = 0;
        std::size_t time_consumed = 0;
        const long long checkpoint_step =
            std::stoll(step->second, &step_consumed);
        const double checkpoint_time = std::stod(time->second, &time_consumed);
        if (step_consumed != step->second.size() ||
            time_consumed != time->second.size() || checkpoint_step < 0 ||
            checkpoint_step >= INT_MAX ||
            !SpongeFloat::Is_Finite(checkpoint_time))
        {
            throw std::invalid_argument("invalid integrator step/time");
        }
        runtime_.md_info.sys.steps = static_cast<int>(checkpoint_step + 1);
        runtime_.md_info.sys.start_time =
            checkpoint_time -
            runtime_.md_info.sys.dt_in_ps * runtime_.md_info.sys.steps;
    }
    catch (const std::exception&)
    {
        runtime_.controller.Throw_SPONGE_Error(
            spongeErrorValueErrorCommand, "Apply_H5_Dynamic_Restart_State",
            "Reason:\n\tRestart integrator step/time is invalid\n");
    }
}

void InputSession::Validate_H5_Input_Plan()
{
    if (!input_.Initial(&runtime_.controller))
    {
        runtime_.controller.Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                                               "Validate_H5_Input_Plan",
                                               input_.Last_Error().c_str());
    }
}

void InputSession::Restore_Dynamic_State()
{
    const auto& input_plan = input_.Plan();
    if (!input_plan.restart.binding.enabled ||
        !Requests_H5_Dynamic_State(input_plan.restart.load_policy))
    {
        return;
    }
    const auto* payload = input_.Dynamic_Restart();
    if (payload == nullptr)
    {
        runtime_.controller.Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                               "Apply_H5_Dynamic_Restart_State",
                                               input_.Last_Error().c_str());
    }
    const auto& dynamic_state = *payload;
    if (SpongeH5InputValidation::Has_Unsupported_Dynamic_State(dynamic_state))
    {
        const std::string unsupported =
            SpongeH5InputValidation::Unsupported_Dynamic_State_Reason(
                dynamic_state);
        runtime_.controller.Throw_SPONGE_Error(
            spongeErrorValueErrorCommand, "Apply_H5_Dynamic_Restart_State",
            ("Reason:\n\tRestart contains unsupported dynamic state: " +
             unsupported + "\n")
                .c_str());
    }
    Apply_H5_Dynamic_Integrator_State(dynamic_state);
    std::string error_message;
    if (dynamic_state.has_nose_hoover_chain)
    {
        if (!runtime_.nhc.is_initialized)
        {
            runtime_.controller.Throw_SPONGE_Error(
                spongeErrorConflictingCommand, "Apply_H5_Dynamic_Restart_State",
                "Reason:\n\tRestart contains Nose-Hoover chain state, but the "
                "nose_hoover_chain thermostat is not initialized\n");
        }
        if (!runtime_.nhc.Apply_H5_Restart_State(dynamic_state, &error_message))
        {
            runtime_.controller.Throw_SPONGE_Error(
                spongeErrorValueErrorCommand, "Apply_H5_Dynamic_Restart_State",
                error_message.c_str());
        }
    }
    const bool has_bussi_rng =
        dynamic_state.rng_state_text.count("bussi_thermostat") != 0 ||
        dynamic_state.rng_states.count("bussi_thermostat") != 0;
    if (runtime_.bussi_thermo.is_initialized || has_bussi_rng)
    {
        if (!runtime_.bussi_thermo.Apply_H5_Restart_State(dynamic_state,
                                                          &error_message))
        {
            runtime_.controller.Throw_SPONGE_Error(
                spongeErrorValueErrorCommand, "Apply_H5_Dynamic_Restart_State",
                error_message.c_str());
        }
    }
    const bool has_middle_rng =
        dynamic_state.rng_states.count("middle_langevin") != 0;
    if (runtime_.middle_langevin.is_initialized || has_middle_rng)
    {
        if (!runtime_.middle_langevin.Apply_H5_Restart_State(dynamic_state,
                                                             &error_message))
        {
            runtime_.controller.Throw_SPONGE_Error(
                spongeErrorValueErrorCommand, "Apply_H5_Dynamic_Restart_State",
                error_message.c_str());
        }
    }
    const bool has_andersen_rng =
        dynamic_state.rng_states.count("andersen") != 0;
    if (runtime_.ad_thermo.is_initialized || has_andersen_rng)
    {
        if (!runtime_.ad_thermo.Apply_H5_Restart_State(dynamic_state,
                                                       &error_message))
        {
            runtime_.controller.Throw_SPONGE_Error(
                spongeErrorValueErrorCommand, "Apply_H5_Dynamic_Restart_State",
                error_message.c_str());
        }
    }

    const auto pressure_baro =
        dynamic_state.barostat_float_states.find("pressure_based_barostat");
    if (runtime_.press_baro.is_initialized ||
        pressure_baro != dynamic_state.barostat_float_states.end())
    {
        if (!runtime_.press_baro.Apply_H5_Restart_State(dynamic_state,
                                                        &error_message))
        {
            runtime_.controller.Throw_SPONGE_Error(
                spongeErrorValueErrorCommand, "Apply_H5_Dynamic_Restart_State",
                error_message.c_str());
        }
    }
    const bool has_mc_rng =
        dynamic_state.rng_states.count("monte_carlo_barostat") != 0;
    if (runtime_.mc_baro.is_initialized || has_mc_rng)
    {
        if (!runtime_.mc_baro.Apply_H5_Restart_State(dynamic_state,
                                                     &error_message))
        {
            runtime_.controller.Throw_SPONGE_Error(
                spongeErrorValueErrorCommand, "Apply_H5_Dynamic_Restart_State",
                error_message.c_str());
        }
    }
}

void InputSession::Prepare_Metadynamics()
{
    const auto& input_plan = input_.Plan();
    if (!input_plan.restart.binding.enabled ||
        !Requests_H5_Protocol_State(input_plan.restart.load_policy))
    {
        return;
    }

    const auto* payload = input_.Protocol_Restart();
    if (payload == nullptr)
    {
        runtime_.controller.Throw_SPONGE_Error(
            spongeErrorBadFileFormat,
            "Materialize_H5_Metadynamics_Restart_Text_State",
            input_.Last_Error().c_str());
    }
    const auto& protocol_state = *payload;
    if (protocol_state.metadynamics_states.empty())
    {
        return;
    }

    bool materialized = false;
    std::string error_message;
    const std::string metadynamics_name =
        runtime_.cv_controller.protocol_metadynamics_name.empty()
            ? "meta"
            : runtime_.cv_controller.protocol_metadynamics_name;
    const auto typed_state = std::find_if(
        protocol_state.metadynamics_states.begin(),
        protocol_state.metadynamics_states.end(),
        [&metadynamics_name](const SpongeH5MD::RestartMetadynamicsState& value)
        {
            return value.name == metadynamics_name && value.has_typed_state &&
                   (value.state_schema_version >= 1 ||
                    value.text_states.empty());
        });
    if (typed_state != protocol_state.metadynamics_states.end())
    {
        return;
    }
    if (!SpongeH5MD::Materialize_Metadynamics_Text_State(
            protocol_state, metadynamics_name, "myhill.log", "history.log",
            "sumhill.log", "Meta_Potential.txt", "Meta_directly.txt",
            &materialized, &error_message))
    {
        runtime_.controller.Throw_SPONGE_Error(
            spongeErrorValueErrorCommand,
            "Materialize_H5_Metadynamics_Restart_Text_State",
            error_message.c_str());
    }
}

void InputSession::Restore_Protocol_State()
{
    const auto& input_plan = input_.Plan();
    if (!input_plan.restart.binding.enabled ||
        !Requests_H5_Protocol_State(input_plan.restart.load_policy))
    {
        return;
    }

    const auto* payload = input_.Protocol_Restart();
    if (payload == nullptr)
    {
        runtime_.controller.Throw_SPONGE_Error(
            spongeErrorBadFileFormat, "Apply_H5_Protocol_Restart_State",
            input_.Last_Error().c_str());
    }
    const auto& protocol_state = *payload;
    if (protocol_state.sits_states.empty() &&
        protocol_state.metadynamics_states.empty() &&
        protocol_state.restraint_states.empty() &&
        protocol_state.cv_reference_states.empty())
    {
        runtime_.controller.Throw_SPONGE_Error(
            spongeErrorValueErrorCommand, "Apply_H5_Protocol_Restart_State",
            "Reason:\n\tNo supported protocol restart state is available\n");
    }
    if (!protocol_state.metadynamics_states.empty() &&
        !runtime_.meta.is_initialized)
    {
        runtime_.controller.Throw_SPONGE_Error(
            spongeErrorConflictingCommand, "Apply_H5_Protocol_Restart_State",
            "Reason:\n\tRestart contains metadynamics state, but the meta "
            "module is not initialized\n");
    }
    if (!protocol_state.restraint_states.empty())
    {
        const auto state = std::find_if(
            protocol_state.restraint_states.begin(),
            protocol_state.restraint_states.end(),
            [this](const SpongeH5MD::RestartRestraintState& value)
            { return value.name == runtime_.restrain.h5_restraint_name; });
        if (state == protocol_state.restraint_states.end())
        {
            runtime_.controller.Throw_SPONGE_Error(
                spongeErrorConflictingCommand,
                "Apply_H5_Protocol_Restart_State",
                "Reason:\n\trestart restraint state does not match the "
                "active protocol restraint object\n");
        }
        std::string restraint_error;
        if (!runtime_.restrain.Apply_H5_Reference_Coordinates(
                state->reference_coordinates, &restraint_error))
        {
            runtime_.controller.Throw_SPONGE_Error(
                spongeErrorValueErrorCommand, "Apply_H5_Protocol_Restart_State",
                restraint_error.c_str());
        }
    }
    for (const auto& reference : protocol_state.cv_reference_states)
    {
        const auto active =
            runtime_.cv_controller.protocol_cv_reference.find(reference.name);
        if (active == runtime_.cv_controller.protocol_cv_reference.end() ||
            active->second != reference.reference_coordinates)
        {
            runtime_.controller.Throw_SPONGE_Error(
                spongeErrorConflictingCommand,
                "Apply_H5_Protocol_Restart_State",
                "Reason:\n\trestart CV reference does not match an active "
                "native CV object\n");
        }
    }
    if (!protocol_state.metadynamics_states.empty())
    {
        const std::string metadynamics_name =
            runtime_.cv_controller.protocol_metadynamics_name.empty()
                ? "meta"
                : runtime_.cv_controller.protocol_metadynamics_name;
        const auto state =
            std::find_if(protocol_state.metadynamics_states.begin(),
                         protocol_state.metadynamics_states.end(),
                         [&metadynamics_name](
                             const SpongeH5MD::RestartMetadynamicsState& value)
                         {
                             return value.name == metadynamics_name &&
                                    value.has_typed_state &&
                                    (value.state_schema_version >= 1 ||
                                     value.text_states.empty());
                         });
        const bool has_unmatched_typed_state =
            std::any_of(protocol_state.metadynamics_states.begin(),
                        protocol_state.metadynamics_states.end(),
                        [](const SpongeH5MD::RestartMetadynamicsState& value)
                        {
                            return value.has_typed_state &&
                                   (value.state_schema_version >= 1 ||
                                    value.text_states.empty());
                        });
        if (state == protocol_state.metadynamics_states.end() &&
            has_unmatched_typed_state)
        {
            runtime_.controller.Throw_SPONGE_Error(
                spongeErrorConflictingCommand,
                "Apply_H5_Protocol_Restart_State",
                "Reason:\n\trestart metadynamics state does not match the "
                "active protocol metadynamics object\n");
        }
        if (state != protocol_state.metadynamics_states.end())
        {
            std::string metadynamics_error;
            if (!runtime_.meta.Apply_H5_Restart_State(*state,
                                                      &metadynamics_error))
            {
                runtime_.controller.Throw_SPONGE_Error(
                    spongeErrorValueErrorCommand,
                    "Apply_H5_Protocol_Restart_State",
                    metadynamics_error.c_str());
            }
        }
    }
    if (protocol_state.sits_states.empty())
    {
        return;
    }
    if (runtime_.sits.is_initialized &&
        runtime_.controller.Command_Exist(runtime_.sits.module_name,
                                          "nk_in_file"))
    {
        return;
    }
    std::string error_message;
    if (!runtime_.sits.Apply_H5_Restart_State(protocol_state, &error_message))
    {
        runtime_.controller.Throw_SPONGE_Error(
            spongeErrorValueErrorCommand, "Apply_H5_Protocol_Restart_State",
            error_message.c_str());
    }
}
}  // namespace SpongeIO

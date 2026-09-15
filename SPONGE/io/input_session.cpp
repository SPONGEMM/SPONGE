#include "input_session.h"

#include "../utils/float_classification.hpp"
#include "../utils/h5md/h5_legacy_sidecar.hpp"

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
    Materialize_H5_Native_Topology_Core();
    Materialize_H5_Topology_And_Protocol_Sidecars();
    Materialize_H5_Protocol_Restart_Sidecars();
    runtime_.system.Load_Inputs(&runtime_.controller);
    Materialize_H5_Native_Topology_Forcefield();
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

void InputSession::Materialize_H5_Topology_And_Protocol_Sidecars()
{
    const auto& input_plan = input_.Plan();
    if (!input_plan.any_h5_input_enabled)
    {
        return;
    }

    std::string error_message;
    if (!SpongeH5MD::Inject_Legacy_Sidecar_Commands_From_H5(
            &runtime_.controller, input_plan.topology.path,
            SpongeH5MD::H5_Topology_Sidecar_Command_Keys(),
            "input_h5_topology_path", &error_message))
    {
        runtime_.controller.Throw_SPONGE_Error(
            spongeErrorValueErrorCommand,
            "Materialize_H5_Topology_And_Protocol_Sidecars",
            error_message.c_str());
    }
    if (!SpongeH5MD::Inject_Legacy_Sidecar_Commands_From_H5(
            &runtime_.controller, input_plan.protocol.path,
            SpongeH5MD::H5_Protocol_Sidecar_Command_Keys(),
            "input_h5_protocol_path", &error_message))
    {
        runtime_.controller.Throw_SPONGE_Error(
            spongeErrorValueErrorCommand,
            "Materialize_H5_Topology_And_Protocol_Sidecars",
            error_message.c_str());
    }
}

void InputSession::Materialize_H5_Native_Topology_Core()
{
    const auto& input_plan = input_.Plan();
    if (!input_plan.topology.enabled)
    {
        return;
    }

    const auto* payload = input_.Topology();
    if (payload == nullptr)
    {
        runtime_.controller.Throw_SPONGE_Error(
            spongeErrorBadFileFormat, "Materialize_H5_Native_Topology_Core",
            input_.Last_Error().c_str());
    }
    const auto& state = *payload;
    if (!state.has_mass && !state.has_charge && !state.has_exclusions &&
        !state.has_bonds && !state.has_angles && !state.has_dihedrals &&
        !state.has_impropers && !state.has_lj && !state.has_nb14 &&
        !state.has_gb && !state.has_virtual_atoms && !state.has_urey_bradley &&
        !state.has_cmap && !state.has_lj_soft_core)
    {
        return;
    }
    if (state.has_mass &&
        runtime_.controller.commands.count("mass_in_file") != 0)
    {
        runtime_.controller.Throw_SPONGE_Error(
            spongeErrorConflictingCommand,
            "Materialize_H5_Native_Topology_Core",
            "Reason:\n\tinput.h5.topology provides /atoms/mass, but "
            "mass_in_file is also set. Native H5 topology data and legacy "
            "text topology input cannot both own atom masses\n");
    }
    if (state.has_charge &&
        runtime_.controller.commands.count("charge_in_file") != 0)
    {
        runtime_.controller.Throw_SPONGE_Error(
            spongeErrorConflictingCommand,
            "Materialize_H5_Native_Topology_Core",
            "Reason:\n\tinput.h5.topology provides /atoms/charge, but "
            "charge_in_file is also set. Native H5 topology data and legacy "
            "text topology input cannot both own atom charges\n");
    }
    if (state.has_exclusions &&
        runtime_.controller.commands.count("exclude_in_file") != 0)
    {
        runtime_.controller.Throw_SPONGE_Error(
            spongeErrorConflictingCommand,
            "Materialize_H5_Native_Topology_Core",
            "Reason:\n\tinput.h5.topology provides native exclusions, but "
            "exclude_in_file is also set. Native H5 topology data and legacy "
            "text topology input cannot both own exclusions\n");
    }
    if (state.has_bonds &&
        runtime_.controller.commands.count("bond_in_file") != 0)
    {
        runtime_.controller.Throw_SPONGE_Error(
            spongeErrorConflictingCommand,
            "Materialize_H5_Native_Topology_Core",
            "Reason:\n\tinput.h5.topology provides native bonds, but "
            "bond_in_file is also set. Native H5 topology data and legacy "
            "text topology input cannot both own bonds\n");
    }
    if (state.has_angles &&
        runtime_.controller.commands.count("angle_in_file") != 0)
    {
        runtime_.controller.Throw_SPONGE_Error(
            spongeErrorConflictingCommand,
            "Materialize_H5_Native_Topology_Core",
            "Reason:\n\tinput.h5.topology provides native angles, but "
            "angle_in_file is also set. Native H5 topology data and legacy "
            "text topology input cannot both own angles\n");
    }
    if (state.has_dihedrals &&
        runtime_.controller.commands.count("dihedral_in_file") != 0)
    {
        runtime_.controller.Throw_SPONGE_Error(
            spongeErrorConflictingCommand,
            "Materialize_H5_Native_Topology_Core",
            "Reason:\n\tinput.h5.topology provides native dihedrals, but "
            "dihedral_in_file is also set. Native H5 topology data and legacy "
            "text topology input cannot both own dihedrals\n");
    }
    if (state.has_impropers &&
        (runtime_.controller.commands.count("improper_dihedral_in_file") != 0 ||
         runtime_.controller.commands.count("improper_in_file") != 0))
    {
        runtime_.controller.Throw_SPONGE_Error(
            spongeErrorConflictingCommand,
            "Materialize_H5_Native_Topology_Core",
            "Reason:\n\tinput.h5.topology provides native impropers, but "
            "improper_dihedral_in_file is also set. Native H5 topology data "
            "and legacy text topology input cannot both own impropers\n");
    }
    if (state.has_lj && runtime_.controller.commands.count("LJ_in_file") != 0)
    {
        runtime_.controller.Throw_SPONGE_Error(
            spongeErrorConflictingCommand,
            "Materialize_H5_Native_Topology_Core",
            "Reason:\n\tinput.h5.topology provides native LJ parameters, but "
            "LJ_in_file is also set. Native H5 topology data and legacy text "
            "topology input cannot both own LJ parameters\n");
    }
    if (state.has_nb14 &&
        (runtime_.controller.commands.count("nb14_in_file") != 0 ||
         runtime_.controller.commands.count("nb14_extra_in_file") != 0))
    {
        runtime_.controller.Throw_SPONGE_Error(
            spongeErrorConflictingCommand,
            "Materialize_H5_Native_Topology_Core",
            "Reason:\n\tinput.h5.topology provides native nb14 parameters, "
            "but nb14_in_file or nb14_extra_in_file is also set. Native H5 "
            "topology data and legacy text topology input cannot both own "
            "nb14 parameters\n");
    }
    if (state.has_gb && runtime_.controller.commands.count("gb_in_file") != 0)
    {
        runtime_.controller.Throw_SPONGE_Error(
            spongeErrorConflictingCommand,
            "Materialize_H5_Native_Topology_Core",
            "Reason:\n\tinput.h5.topology provides native GB parameters, but "
            "gb_in_file is also set. Native H5 topology data and legacy text "
            "topology input cannot both own GB parameters\n");
    }
    if (state.has_virtual_atoms &&
        (runtime_.controller.commands.count("virtual_atom_in_file") != 0 ||
         runtime_.controller.commands.count("virtual_atoms_in_file") != 0))
    {
        runtime_.controller.Throw_SPONGE_Error(
            spongeErrorConflictingCommand,
            "Materialize_H5_Native_Topology_Core",
            "Reason:\n\tinput.h5.topology provides native virtual atom "
            "records, but virtual_atom_in_file is also set. Native H5 "
            "topology data and legacy text topology input cannot both own "
            "virtual atoms\n");
    }
    if (state.has_urey_bradley &&
        runtime_.controller.commands.count("urey_bradley_in_file") != 0)
    {
        runtime_.controller.Throw_SPONGE_Error(
            spongeErrorConflictingCommand,
            "Materialize_H5_Native_Topology_Core",
            "Reason:\n\tinput.h5.topology provides native Urey-Bradley "
            "parameters, but urey_bradley_in_file is also set. Native H5 "
            "topology data and legacy text topology input cannot both own "
            "Urey-Bradley parameters\n");
    }
    if (state.has_cmap &&
        runtime_.controller.commands.count("cmap_in_file") != 0)
    {
        runtime_.controller.Throw_SPONGE_Error(
            spongeErrorConflictingCommand,
            "Materialize_H5_Native_Topology_Core",
            "Reason:\n\tinput.h5.topology provides native CMAP parameters, "
            "but cmap_in_file is also set. Native H5 topology data and "
            "legacy text topology input cannot both own CMAP parameters\n");
    }
    if (state.has_lj_soft_core &&
        (runtime_.controller.commands.count("LJ_soft_core_in_file") != 0 ||
         runtime_.controller.commands.count("subsys_division_in_file") != 0))
    {
        runtime_.controller.Throw_SPONGE_Error(
            spongeErrorConflictingCommand,
            "Materialize_H5_Native_Topology_Core",
            "Reason:\n\tinput.h5.topology provides native LJ soft-core "
            "parameters, but LJ_soft_core_in_file or subsys_division_in_file "
            "is also set. Native H5 topology data and legacy text topology "
            "input cannot both own LJ soft-core parameters\n");
    }
    if (state.has_mass)
    {
        runtime_.system.atoms.mass = state.mass;
    }
    if (state.has_charge)
    {
        runtime_.system.atoms.charge = state.charge;
    }
}

void InputSession::Materialize_H5_Native_Topology_Forcefield()
{
    const auto& input_plan = input_.Plan();
    if (!input_plan.topology.enabled)
    {
        return;
    }

    const auto* payload = input_.Topology();
    if (payload == nullptr)
    {
        runtime_.controller.Throw_SPONGE_Error(
            spongeErrorBadFileFormat,
            "Materialize_H5_Native_Topology_Forcefield",
            input_.Last_Error().c_str());
    }
    const auto& state = *payload;
    if (!state.has_exclusions && !state.has_bonds && !state.has_angles &&
        !state.has_dihedrals && !state.has_impropers && !state.has_lj &&
        !state.has_nb14 && !state.has_gb && !state.has_virtual_atoms &&
        !state.has_urey_bradley && !state.has_cmap && !state.has_lj_soft_core)
    {
        return;
    }
    int atom_numbers = 0;
    if (!runtime_.system.atoms.mass.empty())
    {
        atom_numbers = static_cast<int>(runtime_.system.atoms.mass.size());
    }
    else if (!runtime_.system.atoms.charge.empty())
    {
        atom_numbers = static_cast<int>(runtime_.system.atoms.charge.size());
    }
    else if (!runtime_.system.atoms.coordinate.empty())
    {
        atom_numbers =
            static_cast<int>(runtime_.system.atoms.coordinate.size() / 3);
    }
    if (state.atom_count > 0 && atom_numbers > 0 &&
        state.atom_count != atom_numbers)
    {
        runtime_.controller.Throw_SPONGE_Error(
            spongeErrorConflictingCommand,
            "Materialize_H5_Native_Topology_Forcefield",
            "Reason:\n\tinput.h5.topology atom_count does not match the "
            "materialized runtime atom count\n");
    }
    if (state.has_exclusions)
    {
        runtime_.system.exclusions.excluded_atoms =
            state.exclusions.excluded_atoms;
    }
    if (state.has_bonds)
    {
        runtime_.system.classical_force_field.bonds.atom_a = state.bonds.atom_a;
        runtime_.system.classical_force_field.bonds.atom_b = state.bonds.atom_b;
        runtime_.system.classical_force_field.bonds.k = state.bonds.k;
        runtime_.system.classical_force_field.bonds.r0 = state.bonds.r0;
    }
    if (state.has_angles)
    {
        runtime_.system.classical_force_field.angles.atom_a =
            state.angles.atom_a;
        runtime_.system.classical_force_field.angles.atom_b =
            state.angles.atom_b;
        runtime_.system.classical_force_field.angles.atom_c =
            state.angles.atom_c;
        runtime_.system.classical_force_field.angles.k = state.angles.k;
        runtime_.system.classical_force_field.angles.theta0 =
            state.angles.theta0;
    }
    if (state.has_dihedrals)
    {
        runtime_.system.classical_force_field.dihedrals.atom_a =
            state.dihedrals.atom_a;
        runtime_.system.classical_force_field.dihedrals.atom_b =
            state.dihedrals.atom_b;
        runtime_.system.classical_force_field.dihedrals.atom_c =
            state.dihedrals.atom_c;
        runtime_.system.classical_force_field.dihedrals.atom_d =
            state.dihedrals.atom_d;
        runtime_.system.classical_force_field.dihedrals.pk = state.dihedrals.pk;
        runtime_.system.classical_force_field.dihedrals.pn = state.dihedrals.pn;
        runtime_.system.classical_force_field.dihedrals.ipn =
            state.dihedrals.ipn;
        runtime_.system.classical_force_field.dihedrals.gamc =
            state.dihedrals.gamc;
        runtime_.system.classical_force_field.dihedrals.gams =
            state.dihedrals.gams;
    }
    if (state.has_impropers)
    {
        runtime_.system.classical_force_field.impropers.atom_a =
            state.impropers.atom_a;
        runtime_.system.classical_force_field.impropers.atom_b =
            state.impropers.atom_b;
        runtime_.system.classical_force_field.impropers.atom_c =
            state.impropers.atom_c;
        runtime_.system.classical_force_field.impropers.atom_d =
            state.impropers.atom_d;
        runtime_.system.classical_force_field.impropers.pk = state.impropers.pk;
        runtime_.system.classical_force_field.impropers.pn = state.impropers.pn;
        runtime_.system.classical_force_field.impropers.ipn =
            state.impropers.ipn;
        runtime_.system.classical_force_field.impropers.gamc =
            state.impropers.gamc;
        runtime_.system.classical_force_field.impropers.gams =
            state.impropers.gams;
    }
    if (state.has_lj)
    {
        runtime_.system.classical_force_field.lj.atom_type = state.lj.atom_type;
        runtime_.system.classical_force_field.lj.pair_A = state.lj.pair_A;
        runtime_.system.classical_force_field.lj.pair_B = state.lj.pair_B;
        for (float& value : runtime_.system.classical_force_field.lj.pair_A)
        {
            value *= 12.0f;
        }
        for (float& value : runtime_.system.classical_force_field.lj.pair_B)
        {
            value *= 6.0f;
        }
        runtime_.system.classical_force_field.lj.atom_type_numbers =
            state.lj.atom_type_numbers;
    }
    if (state.has_nb14)
    {
        runtime_.system.classical_force_field.nb14.atom_a = state.nb14.atom_a;
        runtime_.system.classical_force_field.nb14.atom_b = state.nb14.atom_b;
        runtime_.system.classical_force_field.nb14.A = state.nb14.A;
        runtime_.system.classical_force_field.nb14.B = state.nb14.B;
        runtime_.system.classical_force_field.nb14.cf_scale_factor =
            state.nb14.cf_scale_factor;
    }
    if (state.has_gb)
    {
        runtime_.system.generalized_born.radius = state.gb.radius;
        runtime_.system.generalized_born.scale_factor = state.gb.scale_factor;
    }
    if (state.has_virtual_atoms)
    {
        runtime_.system.virtual_atoms.records.clear();
        runtime_.system.virtual_atoms.records.reserve(
            state.virtual_atoms.records.size());
        for (const auto& source_record : state.virtual_atoms.records)
        {
            Xponge::VirtualAtomRecord record;
            record.type = source_record.type;
            record.virtual_atom = source_record.virtual_atom;
            record.from = source_record.from;
            record.parameter = source_record.parameter;
            runtime_.system.virtual_atoms.records.push_back(record);
        }
    }
    if (state.has_urey_bradley)
    {
        runtime_.system.classical_force_field.urey_bradley.atom_a =
            state.urey_bradley.atom_a;
        runtime_.system.classical_force_field.urey_bradley.atom_b =
            state.urey_bradley.atom_b;
        runtime_.system.classical_force_field.urey_bradley.atom_c =
            state.urey_bradley.atom_c;
        runtime_.system.classical_force_field.urey_bradley.angle_k =
            state.urey_bradley.angle_k;
        runtime_.system.classical_force_field.urey_bradley.angle_theta0 =
            state.urey_bradley.angle_theta0;
        runtime_.system.classical_force_field.urey_bradley.bond_k =
            state.urey_bradley.bond_k;
        runtime_.system.classical_force_field.urey_bradley.bond_r0 =
            state.urey_bradley.bond_r0;
    }
    if (state.has_cmap)
    {
        runtime_.system.classical_force_field.cmap.atom_a = state.cmap.atom_a;
        runtime_.system.classical_force_field.cmap.atom_b = state.cmap.atom_b;
        runtime_.system.classical_force_field.cmap.atom_c = state.cmap.atom_c;
        runtime_.system.classical_force_field.cmap.atom_d = state.cmap.atom_d;
        runtime_.system.classical_force_field.cmap.atom_e = state.cmap.atom_e;
        runtime_.system.classical_force_field.cmap.cmap_type =
            state.cmap.cmap_type;
        runtime_.system.classical_force_field.cmap.resolution =
            state.cmap.resolution;
        runtime_.system.classical_force_field.cmap.grid_value =
            state.cmap.grid_value;
        runtime_.system.classical_force_field.cmap.interpolation_coeff =
            state.cmap.interpolation_coeff;
        runtime_.system.classical_force_field.cmap.type_offset =
            state.cmap.type_offset;
        runtime_.system.classical_force_field.cmap.unique_type_numbers =
            state.cmap.unique_type_numbers;
        runtime_.system.classical_force_field.cmap.unique_gridpoint_numbers =
            state.cmap.unique_gridpoint_numbers;
    }
    if (state.has_lj_soft_core)
    {
        runtime_.system.classical_force_field.lj_soft_core.atom_numbers =
            state.lj_soft_core.atom_numbers;
        runtime_.system.classical_force_field.lj_soft_core.atom_type_numbers_A =
            state.lj_soft_core.atom_type_numbers_A;
        runtime_.system.classical_force_field.lj_soft_core.atom_type_numbers_B =
            state.lj_soft_core.atom_type_numbers_B;
        runtime_.system.classical_force_field.lj_soft_core.LJ_AA =
            state.lj_soft_core.LJ_AA;
        runtime_.system.classical_force_field.lj_soft_core.LJ_AB =
            state.lj_soft_core.LJ_AB;
        runtime_.system.classical_force_field.lj_soft_core.LJ_BA =
            state.lj_soft_core.LJ_BA;
        runtime_.system.classical_force_field.lj_soft_core.LJ_BB =
            state.lj_soft_core.LJ_BB;
        runtime_.system.classical_force_field.lj_soft_core.atom_LJ_type_A =
            state.lj_soft_core.atom_LJ_type_A;
        runtime_.system.classical_force_field.lj_soft_core.atom_LJ_type_B =
            state.lj_soft_core.atom_LJ_type_B;
        runtime_.system.classical_force_field.lj_soft_core.subsystem_division =
            state.lj_soft_core.subsystem_division;
    }
}

void InputSession::Materialize_H5_Protocol_Restart_Sidecars()
{
    const auto& input_plan = input_.Plan();
    if (!input_plan.restart.binding.enabled)
    {
        return;
    }

    const auto* payload = input_.Protocol_Restart();
    if (payload == nullptr)
    {
        runtime_.controller.Throw_SPONGE_Error(
            spongeErrorBadFileFormat,
            "Materialize_H5_Protocol_Restart_Sidecars",
            input_.Last_Error().c_str());
    }
    const auto& protocol_state = *payload;
    std::vector<SpongeH5MD::LegacySidecarBinding> sidecars;
    std::string error_message;
    if (!SpongeH5MD::Materialize_Protocol_Sidecar_Text_State(
            protocol_state, ".sponge_h5_restart_protocol", &sidecars,
            &error_message))
    {
        runtime_.controller.Throw_SPONGE_Error(
            spongeErrorValueErrorCommand,
            "Materialize_H5_Protocol_Restart_Sidecars", error_message.c_str());
    }

    const auto allowed_keys = SpongeH5MD::H5_Protocol_Sidecar_Command_Keys();
    const bool requests_protocol_state =
        Requests_H5_Protocol_State(input_plan.restart.load_policy);
    for (const auto& sidecar : sidecars)
    {
        if (!SpongeH5MD::Command_Key_Allowed(allowed_keys, sidecar.key))
        {
            const std::string message =
                "unsupported H5 restart protocol sidecar key in "
                "input_h5_restart_path: " +
                sidecar.key;
            runtime_.controller.Throw_SPONGE_Error(
                spongeErrorValueErrorCommand,
                "Materialize_H5_Protocol_Restart_Sidecars", message.c_str());
        }
        if (!requests_protocol_state &&
            sidecar.key != "restrain_coordinate_in_file")
        {
            continue;
        }
        runtime_.controller.original_commands[sidecar.key] = sidecar.path;
        runtime_.controller.commands[sidecar.key] = sidecar.path;
        runtime_.controller.command_check[sidecar.key] = 0;
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

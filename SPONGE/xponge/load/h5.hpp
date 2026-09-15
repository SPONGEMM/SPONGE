#pragma once

#include "../../utils/h5md/h5_legacy_sidecar.hpp"
#include "../../utils/h5md/input_assembler.hpp"
#include "../../utils/h5md/input_context.hpp"
#include "../xponge.h"

namespace Xponge
{
inline void Materialize_H5_Topology_And_Protocol_Sidecars(
    CONTROLLER& controller, SpongeH5MD::InputContext& input)
{
    const auto& input_plan = input.Plan();
    if (!input_plan.any_h5_input_enabled)
    {
        return;
    }

    std::string error_message;
    if (!SpongeH5MD::Inject_Legacy_Sidecar_Commands_From_H5(
            &controller, input_plan.topology.path,
            SpongeH5MD::H5_Topology_Sidecar_Command_Keys(),
            "input_h5_topology_path", &error_message))
    {
        controller.Throw_SPONGE_Error(
            spongeErrorValueErrorCommand,
            "Materialize_H5_Topology_And_Protocol_Sidecars",
            error_message.c_str());
    }
    if (!SpongeH5MD::Inject_Legacy_Sidecar_Commands_From_H5(
            &controller, input_plan.protocol.path,
            SpongeH5MD::H5_Protocol_Sidecar_Command_Keys(),
            "input_h5_protocol_path", &error_message))
    {
        controller.Throw_SPONGE_Error(
            spongeErrorValueErrorCommand,
            "Materialize_H5_Topology_And_Protocol_Sidecars",
            error_message.c_str());
    }
}

inline void Materialize_H5_Native_Topology_Core(System& system,
                                                CONTROLLER& controller,
                                                SpongeH5MD::InputContext& input)
{
    const auto& input_plan = input.Plan();
    if (!input_plan.topology.enabled)
    {
        return;
    }

    const auto* payload = input.Topology();
    if (payload == nullptr)
    {
        controller.Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                      "Materialize_H5_Native_Topology_Core",
                                      input.Last_Error().c_str());
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
    if (state.has_mass && controller.commands.count("mass_in_file") != 0)
    {
        controller.Throw_SPONGE_Error(
            spongeErrorConflictingCommand,
            "Materialize_H5_Native_Topology_Core",
            "Reason:\n\tinput.h5.topology provides /atoms/mass, but "
            "mass_in_file is also set. Native H5 topology data and legacy "
            "text topology input cannot both own atom masses\n");
    }
    if (state.has_charge && controller.commands.count("charge_in_file") != 0)
    {
        controller.Throw_SPONGE_Error(
            spongeErrorConflictingCommand,
            "Materialize_H5_Native_Topology_Core",
            "Reason:\n\tinput.h5.topology provides /atoms/charge, but "
            "charge_in_file is also set. Native H5 topology data and legacy "
            "text topology input cannot both own atom charges\n");
    }
    if (state.has_exclusions &&
        controller.commands.count("exclude_in_file") != 0)
    {
        controller.Throw_SPONGE_Error(
            spongeErrorConflictingCommand,
            "Materialize_H5_Native_Topology_Core",
            "Reason:\n\tinput.h5.topology provides native exclusions, but "
            "exclude_in_file is also set. Native H5 topology data and legacy "
            "text topology input cannot both own exclusions\n");
    }
    if (state.has_bonds && controller.commands.count("bond_in_file") != 0)
    {
        controller.Throw_SPONGE_Error(
            spongeErrorConflictingCommand,
            "Materialize_H5_Native_Topology_Core",
            "Reason:\n\tinput.h5.topology provides native bonds, but "
            "bond_in_file is also set. Native H5 topology data and legacy "
            "text topology input cannot both own bonds\n");
    }
    if (state.has_angles && controller.commands.count("angle_in_file") != 0)
    {
        controller.Throw_SPONGE_Error(
            spongeErrorConflictingCommand,
            "Materialize_H5_Native_Topology_Core",
            "Reason:\n\tinput.h5.topology provides native angles, but "
            "angle_in_file is also set. Native H5 topology data and legacy "
            "text topology input cannot both own angles\n");
    }
    if (state.has_dihedrals &&
        controller.commands.count("dihedral_in_file") != 0)
    {
        controller.Throw_SPONGE_Error(
            spongeErrorConflictingCommand,
            "Materialize_H5_Native_Topology_Core",
            "Reason:\n\tinput.h5.topology provides native dihedrals, but "
            "dihedral_in_file is also set. Native H5 topology data and legacy "
            "text topology input cannot both own dihedrals\n");
    }
    if (state.has_impropers &&
        (controller.commands.count("improper_dihedral_in_file") != 0 ||
         controller.commands.count("improper_in_file") != 0))
    {
        controller.Throw_SPONGE_Error(
            spongeErrorConflictingCommand,
            "Materialize_H5_Native_Topology_Core",
            "Reason:\n\tinput.h5.topology provides native impropers, but "
            "improper_dihedral_in_file is also set. Native H5 topology data "
            "and legacy text topology input cannot both own impropers\n");
    }
    if (state.has_lj && controller.commands.count("LJ_in_file") != 0)
    {
        controller.Throw_SPONGE_Error(
            spongeErrorConflictingCommand,
            "Materialize_H5_Native_Topology_Core",
            "Reason:\n\tinput.h5.topology provides native LJ parameters, but "
            "LJ_in_file is also set. Native H5 topology data and legacy text "
            "topology input cannot both own LJ parameters\n");
    }
    if (state.has_nb14 &&
        (controller.commands.count("nb14_in_file") != 0 ||
         controller.commands.count("nb14_extra_in_file") != 0))
    {
        controller.Throw_SPONGE_Error(
            spongeErrorConflictingCommand,
            "Materialize_H5_Native_Topology_Core",
            "Reason:\n\tinput.h5.topology provides native nb14 parameters, "
            "but nb14_in_file or nb14_extra_in_file is also set. Native H5 "
            "topology data and legacy text topology input cannot both own "
            "nb14 parameters\n");
    }
    if (state.has_gb && controller.commands.count("gb_in_file") != 0)
    {
        controller.Throw_SPONGE_Error(
            spongeErrorConflictingCommand,
            "Materialize_H5_Native_Topology_Core",
            "Reason:\n\tinput.h5.topology provides native GB parameters, but "
            "gb_in_file is also set. Native H5 topology data and legacy text "
            "topology input cannot both own GB parameters\n");
    }
    if (state.has_virtual_atoms &&
        (controller.commands.count("virtual_atom_in_file") != 0 ||
         controller.commands.count("virtual_atoms_in_file") != 0))
    {
        controller.Throw_SPONGE_Error(
            spongeErrorConflictingCommand,
            "Materialize_H5_Native_Topology_Core",
            "Reason:\n\tinput.h5.topology provides native virtual atom "
            "records, but virtual_atom_in_file is also set. Native H5 "
            "topology data and legacy text topology input cannot both own "
            "virtual atoms\n");
    }
    if (state.has_urey_bradley &&
        controller.commands.count("urey_bradley_in_file") != 0)
    {
        controller.Throw_SPONGE_Error(
            spongeErrorConflictingCommand,
            "Materialize_H5_Native_Topology_Core",
            "Reason:\n\tinput.h5.topology provides native Urey-Bradley "
            "parameters, but urey_bradley_in_file is also set. Native H5 "
            "topology data and legacy text topology input cannot both own "
            "Urey-Bradley parameters\n");
    }
    if (state.has_cmap && controller.commands.count("cmap_in_file") != 0)
    {
        controller.Throw_SPONGE_Error(
            spongeErrorConflictingCommand,
            "Materialize_H5_Native_Topology_Core",
            "Reason:\n\tinput.h5.topology provides native CMAP parameters, "
            "but cmap_in_file is also set. Native H5 topology data and "
            "legacy text topology input cannot both own CMAP parameters\n");
    }
    if (state.has_lj_soft_core &&
        (controller.commands.count("LJ_soft_core_in_file") != 0 ||
         controller.commands.count("subsys_division_in_file") != 0))
    {
        controller.Throw_SPONGE_Error(
            spongeErrorConflictingCommand,
            "Materialize_H5_Native_Topology_Core",
            "Reason:\n\tinput.h5.topology provides native LJ soft-core "
            "parameters, but LJ_soft_core_in_file or subsys_division_in_file "
            "is also set. Native H5 topology data and legacy text topology "
            "input cannot both own LJ soft-core parameters\n");
    }
    if (state.has_mass)
    {
        system.atoms.mass = state.mass;
    }
    if (state.has_charge)
    {
        system.atoms.charge = state.charge;
    }
}

inline void Materialize_H5_Native_Topology_Forcefield(
    System& system, CONTROLLER& controller, SpongeH5MD::InputContext& input)
{
    const auto& input_plan = input.Plan();
    if (!input_plan.topology.enabled)
    {
        return;
    }

    const auto* payload = input.Topology();
    if (payload == nullptr)
    {
        controller.Throw_SPONGE_Error(
            spongeErrorBadFileFormat,
            "Materialize_H5_Native_Topology_Forcefield",
            input.Last_Error().c_str());
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
    if (!system.atoms.mass.empty())
    {
        atom_numbers = static_cast<int>(system.atoms.mass.size());
    }
    else if (!system.atoms.charge.empty())
    {
        atom_numbers = static_cast<int>(system.atoms.charge.size());
    }
    else if (!system.atoms.coordinate.empty())
    {
        atom_numbers = static_cast<int>(system.atoms.coordinate.size() / 3);
    }
    if (state.atom_count > 0 && atom_numbers > 0 &&
        state.atom_count != atom_numbers)
    {
        controller.Throw_SPONGE_Error(
            spongeErrorConflictingCommand,
            "Materialize_H5_Native_Topology_Forcefield",
            "Reason:\n\tinput.h5.topology atom_count does not match the "
            "materialized runtime atom count\n");
    }
    if (state.has_exclusions)
    {
        system.exclusions.excluded_atoms = state.exclusions.excluded_atoms;
    }
    if (state.has_bonds)
    {
        system.classical_force_field.bonds.atom_a = state.bonds.atom_a;
        system.classical_force_field.bonds.atom_b = state.bonds.atom_b;
        system.classical_force_field.bonds.k = state.bonds.k;
        system.classical_force_field.bonds.r0 = state.bonds.r0;
    }
    if (state.has_angles)
    {
        system.classical_force_field.angles.atom_a = state.angles.atom_a;
        system.classical_force_field.angles.atom_b = state.angles.atom_b;
        system.classical_force_field.angles.atom_c = state.angles.atom_c;
        system.classical_force_field.angles.k = state.angles.k;
        system.classical_force_field.angles.theta0 = state.angles.theta0;
    }
    if (state.has_dihedrals)
    {
        system.classical_force_field.dihedrals.atom_a = state.dihedrals.atom_a;
        system.classical_force_field.dihedrals.atom_b = state.dihedrals.atom_b;
        system.classical_force_field.dihedrals.atom_c = state.dihedrals.atom_c;
        system.classical_force_field.dihedrals.atom_d = state.dihedrals.atom_d;
        system.classical_force_field.dihedrals.pk = state.dihedrals.pk;
        system.classical_force_field.dihedrals.pn = state.dihedrals.pn;
        system.classical_force_field.dihedrals.ipn = state.dihedrals.ipn;
        system.classical_force_field.dihedrals.gamc = state.dihedrals.gamc;
        system.classical_force_field.dihedrals.gams = state.dihedrals.gams;
    }
    if (state.has_impropers)
    {
        system.classical_force_field.impropers.atom_a = state.impropers.atom_a;
        system.classical_force_field.impropers.atom_b = state.impropers.atom_b;
        system.classical_force_field.impropers.atom_c = state.impropers.atom_c;
        system.classical_force_field.impropers.atom_d = state.impropers.atom_d;
        system.classical_force_field.impropers.pk = state.impropers.pk;
        system.classical_force_field.impropers.pn = state.impropers.pn;
        system.classical_force_field.impropers.ipn = state.impropers.ipn;
        system.classical_force_field.impropers.gamc = state.impropers.gamc;
        system.classical_force_field.impropers.gams = state.impropers.gams;
    }
    if (state.has_lj)
    {
        system.classical_force_field.lj.atom_type = state.lj.atom_type;
        system.classical_force_field.lj.pair_A = state.lj.pair_A;
        system.classical_force_field.lj.pair_B = state.lj.pair_B;
        for (float& value : system.classical_force_field.lj.pair_A)
        {
            value *= 12.0f;
        }
        for (float& value : system.classical_force_field.lj.pair_B)
        {
            value *= 6.0f;
        }
        system.classical_force_field.lj.atom_type_numbers =
            state.lj.atom_type_numbers;
    }
    if (state.has_nb14)
    {
        system.classical_force_field.nb14.atom_a = state.nb14.atom_a;
        system.classical_force_field.nb14.atom_b = state.nb14.atom_b;
        system.classical_force_field.nb14.A = state.nb14.A;
        system.classical_force_field.nb14.B = state.nb14.B;
        system.classical_force_field.nb14.cf_scale_factor =
            state.nb14.cf_scale_factor;
    }
    if (state.has_gb)
    {
        system.generalized_born.radius = state.gb.radius;
        system.generalized_born.scale_factor = state.gb.scale_factor;
    }
    if (state.has_virtual_atoms)
    {
        system.virtual_atoms.records.clear();
        system.virtual_atoms.records.reserve(
            state.virtual_atoms.records.size());
        for (const auto& source_record : state.virtual_atoms.records)
        {
            Xponge::VirtualAtomRecord record;
            record.type = source_record.type;
            record.virtual_atom = source_record.virtual_atom;
            record.from = source_record.from;
            record.parameter = source_record.parameter;
            system.virtual_atoms.records.push_back(record);
        }
    }
    if (state.has_urey_bradley)
    {
        system.classical_force_field.urey_bradley.atom_a =
            state.urey_bradley.atom_a;
        system.classical_force_field.urey_bradley.atom_b =
            state.urey_bradley.atom_b;
        system.classical_force_field.urey_bradley.atom_c =
            state.urey_bradley.atom_c;
        system.classical_force_field.urey_bradley.angle_k =
            state.urey_bradley.angle_k;
        system.classical_force_field.urey_bradley.angle_theta0 =
            state.urey_bradley.angle_theta0;
        system.classical_force_field.urey_bradley.bond_k =
            state.urey_bradley.bond_k;
        system.classical_force_field.urey_bradley.bond_r0 =
            state.urey_bradley.bond_r0;
    }
    if (state.has_cmap)
    {
        system.classical_force_field.cmap.atom_a = state.cmap.atom_a;
        system.classical_force_field.cmap.atom_b = state.cmap.atom_b;
        system.classical_force_field.cmap.atom_c = state.cmap.atom_c;
        system.classical_force_field.cmap.atom_d = state.cmap.atom_d;
        system.classical_force_field.cmap.atom_e = state.cmap.atom_e;
        system.classical_force_field.cmap.cmap_type = state.cmap.cmap_type;
        system.classical_force_field.cmap.resolution = state.cmap.resolution;
        system.classical_force_field.cmap.grid_value = state.cmap.grid_value;
        system.classical_force_field.cmap.interpolation_coeff =
            state.cmap.interpolation_coeff;
        system.classical_force_field.cmap.type_offset = state.cmap.type_offset;
        system.classical_force_field.cmap.unique_type_numbers =
            state.cmap.unique_type_numbers;
        system.classical_force_field.cmap.unique_gridpoint_numbers =
            state.cmap.unique_gridpoint_numbers;
    }
    if (state.has_lj_soft_core)
    {
        system.classical_force_field.lj_soft_core.atom_numbers =
            state.lj_soft_core.atom_numbers;
        system.classical_force_field.lj_soft_core.atom_type_numbers_A =
            state.lj_soft_core.atom_type_numbers_A;
        system.classical_force_field.lj_soft_core.atom_type_numbers_B =
            state.lj_soft_core.atom_type_numbers_B;
        system.classical_force_field.lj_soft_core.LJ_AA =
            state.lj_soft_core.LJ_AA;
        system.classical_force_field.lj_soft_core.LJ_AB =
            state.lj_soft_core.LJ_AB;
        system.classical_force_field.lj_soft_core.LJ_BA =
            state.lj_soft_core.LJ_BA;
        system.classical_force_field.lj_soft_core.LJ_BB =
            state.lj_soft_core.LJ_BB;
        system.classical_force_field.lj_soft_core.atom_LJ_type_A =
            state.lj_soft_core.atom_LJ_type_A;
        system.classical_force_field.lj_soft_core.atom_LJ_type_B =
            state.lj_soft_core.atom_LJ_type_B;
        system.classical_force_field.lj_soft_core.subsystem_division =
            state.lj_soft_core.subsystem_division;
    }
}

inline void Materialize_H5_Protocol_Restart_Sidecars(
    CONTROLLER& controller, SpongeH5MD::InputContext& input)
{
    const auto& input_plan = input.Plan();
    if (!input_plan.restart.binding.enabled)
    {
        return;
    }

    const auto* payload = input.Protocol_Restart();
    if (payload == nullptr)
    {
        controller.Throw_SPONGE_Error(
            spongeErrorBadFileFormat,
            "Materialize_H5_Protocol_Restart_Sidecars",
            input.Last_Error().c_str());
    }
    const auto& protocol_state = *payload;
    std::vector<SpongeH5MD::LegacySidecarBinding> sidecars;
    std::string error_message;
    if (!SpongeH5MD::Materialize_Protocol_Sidecar_Text_State(
            protocol_state, ".sponge_h5_restart_protocol", &sidecars,
            &error_message))
    {
        controller.Throw_SPONGE_Error(
            spongeErrorValueErrorCommand,
            "Materialize_H5_Protocol_Restart_Sidecars", error_message.c_str());
    }

    const auto allowed_keys = SpongeH5MD::H5_Protocol_Sidecar_Command_Keys();
    const bool requests_protocol_state =
        (input_plan.restart.load_policy ==
             SpongeH5InputContract::RestartLoadPolicy::protocol ||
         input_plan.restart.load_policy ==
             SpongeH5InputContract::RestartLoadPolicy::full);
    for (const auto& sidecar : sidecars)
    {
        if (!SpongeH5MD::Command_Key_Allowed(allowed_keys, sidecar.key))
        {
            const std::string message =
                "unsupported H5 restart protocol sidecar key in "
                "input_h5_restart_path: " +
                sidecar.key;
            controller.Throw_SPONGE_Error(
                spongeErrorValueErrorCommand,
                "Materialize_H5_Protocol_Restart_Sidecars", message.c_str());
        }
        if (!requests_protocol_state &&
            sidecar.key != "restrain_coordinate_in_file")
        {
            continue;
        }
        controller.original_commands[sidecar.key] = sidecar.path;
        controller.commands[sidecar.key] = sidecar.path;
        controller.command_check[sidecar.key] = 0;
    }
}

}  // namespace Xponge

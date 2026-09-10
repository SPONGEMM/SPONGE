#include "main.h"

#include "utils/h5md/h5_legacy_sidecar.hpp"
#include "utils/h5md/input_validation.hpp"
#include "utils/h5md/topology_native_h5_reader.hpp"
#include "xponge/load/common.hpp"

#define SUBPACKAGE_HINT \
    "SPONGE, for general-purpose molecular dynamics simulations"
#define THERMOSTAT_IS(name)                              \
    (md_info.mode >= md_info.NVT &&                      \
     (controller.Command_Choice("thermostat", (name)) || \
      controller.Command_Choice("thermostat_mode", (name))))
#define BAROSTAT_IS(name)                              \
    (md_info.mode == md_info.NPT &&                    \
     (controller.Command_Choice("barostat", (name)) || \
      controller.Command_Choice("barostat_mode", (name))))

CONTROLLER controller;
Xponge::System Xponge::system;
MD_INFORMATION md_info;
INITIAL_VELOCITY_INFORMATION initial_velocity;
DOMAIN_INFORMATION dd;
MIDDLE_Langevin_INFORMATION middle_langevin;
ANDERSEN_THERMOSTAT_INFORMATION ad_thermo;
BERENDSEN_THERMOSTAT_INFORMATION bd_thermo;
BUSSI_THERMOSTAT_INFORMATION bussi_thermo;
NOSE_HOOVER_CHAIN_INFORMATION nhc;
PRESSURE_BASED_BAROSTAT_INFORMATION press_baro;
MC_BAROSTAT_INFORMATION mc_baro;
NEIGHBOR_LIST neighbor_list;
LENNARD_JONES_INFORMATION lj;
LJ_SOFT_CORE lj_soft;
SOLVENT_LENNARD_JONES solvent_lj;
Particle_Mesh pm;
ANGLE angle;
UREY_BRADLEY urey_bradley;
BOND bond;
CMAP cmap;
DIHEDRAL dihedral;
IMPROPER_DIHEDRAL improper;
NON_BOND_14 nb14;
RESTRAIN_INFORMATION restrain;
CONSTRAIN constrain;
SETTLE settle;
SHAKE shake;
VIRTUAL_INFORMATION vatom;
COLLECTIVE_VARIABLE_CONTROLLER cv_controller;
STEER_CV steer_cv;
RESTRAIN_CV restrain_cv;
META meta;
LISTED_FORCES listed_forces;
PAIRWISE_FORCE pairwise_force;
HARD_WALL hard_wall;
SOFT_WALLS soft_walls;
LENNARD_JONES_NO_PBC_INFORMATION LJ_NOPBC;
COULOMB_FORCE_NO_PBC_INFORMATION CF_NOPBC;
GENERALIZED_BORN_INFORMATION gb;
SITS_INFORMATION sits;
DIHEDRAL sits_dihedral;
NON_BOND_14 sits_nb14;
CMAP sits_cmap;
STILLINGER_WEBER_INFORMATION sw;
EDIP_INFORMATION edip;
EAM_INFORMATION eam;
TERSOFF_INFORMATION tersoff;
REAXFF reaxff;
QUANTUM_CHEMISTRY qc;
SPONGE_PLUGIN plugin;

deviceStream_t main_stream;

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

std::string Current_MD_Mode_Name()
{
    if (md_info.mode == md_info.RERUN) return "rerun";
    if (md_info.mode == md_info.MINIMIZATION) return "minimization";
    if (md_info.mode == md_info.NVE) return "nve";
    if (md_info.mode == md_info.NVT) return "nvt";
    if (md_info.mode == md_info.NPT) return "npt";
    return "unknown";
}

SpongeH5MD::RestartDynamicState Build_H5_Dynamic_Restart_State()
{
    SpongeH5MD::RestartDynamicState state;
    state.integrator_state_text["mode"] = Current_MD_Mode_Name();
    state.integrator_state_text["step"] = std::to_string(md_info.sys.steps);
    state.integrator_state_text["time"] =
        std::to_string(md_info.sys.Get_Current_Time());

    std::string error_message;
    if (!bussi_thermo.Export_H5_Restart_State(&state, &error_message))
    {
        controller.Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                                      "Build_H5_Dynamic_Restart_State",
                                      error_message.c_str());
    }
    if (!middle_langevin.Export_H5_Restart_State(&state, &error_message))
    {
        controller.Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                                      "Build_H5_Dynamic_Restart_State",
                                      error_message.c_str());
    }
    if (!ad_thermo.Export_H5_Restart_State(&state, &error_message))
    {
        controller.Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                                      "Build_H5_Dynamic_Restart_State",
                                      error_message.c_str());
    }
    if (!press_baro.Export_H5_Restart_State(&state, &error_message))
    {
        controller.Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                                      "Build_H5_Dynamic_Restart_State",
                                      error_message.c_str());
    }
    if (!mc_baro.Export_H5_Restart_State(&state, &error_message))
    {
        controller.Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                                      "Build_H5_Dynamic_Restart_State",
                                      error_message.c_str());
    }
    return state;
}

void Apply_H5_Dynamic_Integrator_State(
    const SpongeH5MD::RestartDynamicState& dynamic_state)
{
    const auto mode = dynamic_state.integrator_state_text.find("mode");
    if (mode == dynamic_state.integrator_state_text.end())
    {
        return;
    }
    if (mode->second != Current_MD_Mode_Name())
    {
        const std::string message =
            std::string("Reason:\n\tRestart integrator mode is ") +
            mode->second + ", but current mode is " + Current_MD_Mode_Name() +
            "\n";
        controller.Throw_SPONGE_Error(spongeErrorConflictingCommand,
                                      "Apply_H5_Dynamic_Restart_State",
                                      message.c_str());
    }
    const auto step = dynamic_state.integrator_state_text.find("step");
    const auto time = dynamic_state.integrator_state_text.find("time");
    if ((step == dynamic_state.integrator_state_text.end()) !=
        (time == dynamic_state.integrator_state_text.end()))
    {
        controller.Throw_SPONGE_Error(
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
            checkpoint_step >= INT_MAX || !std::isfinite(checkpoint_time))
        {
            throw std::invalid_argument("invalid integrator step/time");
        }
        md_info.sys.steps = static_cast<int>(checkpoint_step + 1);
        md_info.sys.start_time =
            checkpoint_time - md_info.sys.dt_in_ps * md_info.sys.steps;
    }
    catch (const std::exception&)
    {
        controller.Throw_SPONGE_Error(
            spongeErrorValueErrorCommand, "Apply_H5_Dynamic_Restart_State",
            "Reason:\n\tRestart integrator step/time is invalid\n");
    }
}

void Validate_H5_Input_Plan()
{
    const auto validation =
        SpongeH5InputValidation::Validate_Input_Bindings(&controller);
    if (!validation.valid)
    {
        controller.Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                                      "Validate_H5_Input_Plan",
                                      validation.error_message.c_str());
    }
}

void Materialize_H5_Topology_And_Protocol_Sidecars()
{
    const auto input_plan = SpongeH5InputPlan::Resolve_Input_Plan(&controller);
    if (!input_plan.valid)
    {
        controller.Throw_SPONGE_Error(
            spongeErrorValueErrorCommand,
            "Materialize_H5_Topology_And_Protocol_Sidecars",
            input_plan.error_message.c_str());
    }
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

void Materialize_H5_Native_Topology_Core()
{
    const auto input_plan = SpongeH5InputPlan::Resolve_Input_Plan(&controller);
    if (!input_plan.valid)
    {
        controller.Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                                      "Materialize_H5_Native_Topology_Core",
                                      input_plan.error_message.c_str());
    }
    if (!input_plan.topology.enabled)
    {
        return;
    }

    SpongeH5MD::TopologyNativeH5Reader reader;
    if (!reader.Open(input_plan.topology.path))
    {
        controller.Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                      "Materialize_H5_Native_Topology_Core",
                                      reader.Last_Error().c_str());
    }
    SpongeH5MD::NativeTopologyCoreState state;
    if (!reader.Read_Core_State(&state))
    {
        controller.Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                      "Materialize_H5_Native_Topology_Core",
                                      reader.Last_Error().c_str());
    }
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
        Xponge::system.atoms.mass = state.mass;
    }
    if (state.has_charge)
    {
        Xponge::system.atoms.charge = state.charge;
    }
}

void Materialize_H5_Native_Topology_Forcefield()
{
    const auto input_plan = SpongeH5InputPlan::Resolve_Input_Plan(&controller);
    if (!input_plan.valid)
    {
        controller.Throw_SPONGE_Error(
            spongeErrorValueErrorCommand,
            "Materialize_H5_Native_Topology_Forcefield",
            input_plan.error_message.c_str());
    }
    if (!input_plan.topology.enabled)
    {
        return;
    }

    SpongeH5MD::TopologyNativeH5Reader reader;
    if (!reader.Open(input_plan.topology.path))
    {
        controller.Throw_SPONGE_Error(
            spongeErrorBadFileFormat,
            "Materialize_H5_Native_Topology_Forcefield",
            reader.Last_Error().c_str());
    }
    SpongeH5MD::NativeTopologyCoreState state;
    if (!reader.Read_Core_State(&state))
    {
        controller.Throw_SPONGE_Error(
            spongeErrorBadFileFormat,
            "Materialize_H5_Native_Topology_Forcefield",
            reader.Last_Error().c_str());
    }
    if (!state.has_exclusions && !state.has_bonds && !state.has_angles &&
        !state.has_dihedrals && !state.has_impropers && !state.has_lj &&
        !state.has_nb14 && !state.has_gb && !state.has_virtual_atoms &&
        !state.has_urey_bradley && !state.has_cmap && !state.has_lj_soft_core)
    {
        return;
    }
    int atom_numbers = 0;
    if (!Xponge::system.atoms.mass.empty())
    {
        atom_numbers = static_cast<int>(Xponge::system.atoms.mass.size());
    }
    else if (!Xponge::system.atoms.charge.empty())
    {
        atom_numbers = static_cast<int>(Xponge::system.atoms.charge.size());
    }
    else if (!Xponge::system.atoms.coordinate.empty())
    {
        atom_numbers =
            static_cast<int>(Xponge::system.atoms.coordinate.size() / 3);
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
        Xponge::system.exclusions.excluded_atoms =
            state.exclusions.excluded_atoms;
    }
    if (state.has_bonds)
    {
        Xponge::system.classical_force_field.bonds.atom_a = state.bonds.atom_a;
        Xponge::system.classical_force_field.bonds.atom_b = state.bonds.atom_b;
        Xponge::system.classical_force_field.bonds.k = state.bonds.k;
        Xponge::system.classical_force_field.bonds.r0 = state.bonds.r0;
    }
    if (state.has_angles)
    {
        Xponge::system.classical_force_field.angles.atom_a =
            state.angles.atom_a;
        Xponge::system.classical_force_field.angles.atom_b =
            state.angles.atom_b;
        Xponge::system.classical_force_field.angles.atom_c =
            state.angles.atom_c;
        Xponge::system.classical_force_field.angles.k = state.angles.k;
        Xponge::system.classical_force_field.angles.theta0 =
            state.angles.theta0;
    }
    if (state.has_dihedrals)
    {
        Xponge::system.classical_force_field.dihedrals.atom_a =
            state.dihedrals.atom_a;
        Xponge::system.classical_force_field.dihedrals.atom_b =
            state.dihedrals.atom_b;
        Xponge::system.classical_force_field.dihedrals.atom_c =
            state.dihedrals.atom_c;
        Xponge::system.classical_force_field.dihedrals.atom_d =
            state.dihedrals.atom_d;
        Xponge::system.classical_force_field.dihedrals.pk = state.dihedrals.pk;
        Xponge::system.classical_force_field.dihedrals.pn = state.dihedrals.pn;
        Xponge::system.classical_force_field.dihedrals.ipn =
            state.dihedrals.ipn;
        Xponge::system.classical_force_field.dihedrals.gamc =
            state.dihedrals.gamc;
        Xponge::system.classical_force_field.dihedrals.gams =
            state.dihedrals.gams;
    }
    if (state.has_impropers)
    {
        Xponge::system.classical_force_field.impropers.atom_a =
            state.impropers.atom_a;
        Xponge::system.classical_force_field.impropers.atom_b =
            state.impropers.atom_b;
        Xponge::system.classical_force_field.impropers.atom_c =
            state.impropers.atom_c;
        Xponge::system.classical_force_field.impropers.atom_d =
            state.impropers.atom_d;
        Xponge::system.classical_force_field.impropers.pk = state.impropers.pk;
        Xponge::system.classical_force_field.impropers.pn = state.impropers.pn;
        Xponge::system.classical_force_field.impropers.ipn =
            state.impropers.ipn;
        Xponge::system.classical_force_field.impropers.gamc =
            state.impropers.gamc;
        Xponge::system.classical_force_field.impropers.gams =
            state.impropers.gams;
    }
    if (state.has_lj)
    {
        Xponge::system.classical_force_field.lj.atom_type = state.lj.atom_type;
        Xponge::system.classical_force_field.lj.pair_A = state.lj.pair_A;
        Xponge::system.classical_force_field.lj.pair_B = state.lj.pair_B;
        for (float& value : Xponge::system.classical_force_field.lj.pair_A)
        {
            value *= 12.0f;
        }
        for (float& value : Xponge::system.classical_force_field.lj.pair_B)
        {
            value *= 6.0f;
        }
        Xponge::system.classical_force_field.lj.atom_type_numbers =
            state.lj.atom_type_numbers;
    }
    if (state.has_nb14)
    {
        Xponge::system.classical_force_field.nb14.atom_a = state.nb14.atom_a;
        Xponge::system.classical_force_field.nb14.atom_b = state.nb14.atom_b;
        Xponge::system.classical_force_field.nb14.A = state.nb14.A;
        Xponge::system.classical_force_field.nb14.B = state.nb14.B;
        Xponge::system.classical_force_field.nb14.cf_scale_factor =
            state.nb14.cf_scale_factor;
    }
    if (state.has_gb)
    {
        Xponge::system.generalized_born.radius = state.gb.radius;
        Xponge::system.generalized_born.scale_factor = state.gb.scale_factor;
    }
    if (state.has_virtual_atoms)
    {
        Xponge::system.virtual_atoms.records.clear();
        Xponge::system.virtual_atoms.records.reserve(
            state.virtual_atoms.records.size());
        for (const auto& source_record : state.virtual_atoms.records)
        {
            Xponge::VirtualAtomRecord record;
            record.type = source_record.type;
            record.virtual_atom = source_record.virtual_atom;
            record.from = source_record.from;
            record.parameter = source_record.parameter;
            Xponge::system.virtual_atoms.records.push_back(record);
        }
    }
    if (state.has_urey_bradley)
    {
        Xponge::system.classical_force_field.urey_bradley.atom_a =
            state.urey_bradley.atom_a;
        Xponge::system.classical_force_field.urey_bradley.atom_b =
            state.urey_bradley.atom_b;
        Xponge::system.classical_force_field.urey_bradley.atom_c =
            state.urey_bradley.atom_c;
        Xponge::system.classical_force_field.urey_bradley.angle_k =
            state.urey_bradley.angle_k;
        Xponge::system.classical_force_field.urey_bradley.angle_theta0 =
            state.urey_bradley.angle_theta0;
        Xponge::system.classical_force_field.urey_bradley.bond_k =
            state.urey_bradley.bond_k;
        Xponge::system.classical_force_field.urey_bradley.bond_r0 =
            state.urey_bradley.bond_r0;
    }
    if (state.has_cmap)
    {
        Xponge::system.classical_force_field.cmap.atom_a = state.cmap.atom_a;
        Xponge::system.classical_force_field.cmap.atom_b = state.cmap.atom_b;
        Xponge::system.classical_force_field.cmap.atom_c = state.cmap.atom_c;
        Xponge::system.classical_force_field.cmap.atom_d = state.cmap.atom_d;
        Xponge::system.classical_force_field.cmap.atom_e = state.cmap.atom_e;
        Xponge::system.classical_force_field.cmap.cmap_type =
            state.cmap.cmap_type;
        Xponge::system.classical_force_field.cmap.resolution =
            state.cmap.resolution;
        Xponge::system.classical_force_field.cmap.grid_value =
            state.cmap.grid_value;
        Xponge::system.classical_force_field.cmap.interpolation_coeff =
            state.cmap.interpolation_coeff;
        Xponge::system.classical_force_field.cmap.type_offset =
            state.cmap.type_offset;
        Xponge::system.classical_force_field.cmap.unique_type_numbers =
            state.cmap.unique_type_numbers;
        Xponge::system.classical_force_field.cmap.unique_gridpoint_numbers =
            state.cmap.unique_gridpoint_numbers;
    }
    if (state.has_lj_soft_core)
    {
        Xponge::system.classical_force_field.lj_soft_core.atom_numbers =
            state.lj_soft_core.atom_numbers;
        Xponge::system.classical_force_field.lj_soft_core.atom_type_numbers_A =
            state.lj_soft_core.atom_type_numbers_A;
        Xponge::system.classical_force_field.lj_soft_core.atom_type_numbers_B =
            state.lj_soft_core.atom_type_numbers_B;
        Xponge::system.classical_force_field.lj_soft_core.LJ_AA =
            state.lj_soft_core.LJ_AA;
        Xponge::system.classical_force_field.lj_soft_core.LJ_AB =
            state.lj_soft_core.LJ_AB;
        Xponge::system.classical_force_field.lj_soft_core.LJ_BA =
            state.lj_soft_core.LJ_BA;
        Xponge::system.classical_force_field.lj_soft_core.LJ_BB =
            state.lj_soft_core.LJ_BB;
        Xponge::system.classical_force_field.lj_soft_core.atom_LJ_type_A =
            state.lj_soft_core.atom_LJ_type_A;
        Xponge::system.classical_force_field.lj_soft_core.atom_LJ_type_B =
            state.lj_soft_core.atom_LJ_type_B;
        Xponge::system.classical_force_field.lj_soft_core.subsystem_division =
            state.lj_soft_core.subsystem_division;
    }
}

void Materialize_H5_Protocol_Restart_Sidecars()
{
    const auto input_plan = SpongeH5InputPlan::Resolve_Input_Plan(&controller);
    if (!input_plan.valid)
    {
        controller.Throw_SPONGE_Error(
            spongeErrorValueErrorCommand,
            "Materialize_H5_Protocol_Restart_Sidecars",
            input_plan.error_message.c_str());
    }
    if (!input_plan.restart.binding.enabled)
    {
        return;
    }

    SpongeH5MD::RestartH5Reader reader;
    if (!reader.Open(input_plan.restart.binding.path))
    {
        controller.Throw_SPONGE_Error(
            spongeErrorBadFileFormat,
            "Materialize_H5_Protocol_Restart_Sidecars",
            reader.Last_Error().c_str());
    }
    SpongeH5MD::RestartProtocolState protocol_state;
    if (!reader.Read_Protocol_State(&protocol_state))
    {
        controller.Throw_SPONGE_Error(
            spongeErrorBadFileFormat,
            "Materialize_H5_Protocol_Restart_Sidecars",
            reader.Last_Error().c_str());
    }
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
        Requests_H5_Protocol_State(input_plan.restart.load_policy);
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

void Apply_H5_Dynamic_Restart_State()
{
    const auto input_plan = SpongeH5InputPlan::Resolve_Input_Plan(&controller);
    if (!input_plan.valid)
    {
        controller.Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                                      "Apply_H5_Dynamic_Restart_State",
                                      input_plan.error_message.c_str());
    }
    if (!input_plan.restart.binding.enabled ||
        !Requests_H5_Dynamic_State(input_plan.restart.load_policy))
    {
        return;
    }
    SpongeH5MD::RestartH5Reader reader;
    if (!reader.Open(input_plan.restart.binding.path))
    {
        controller.Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                      "Apply_H5_Dynamic_Restart_State",
                                      reader.Last_Error().c_str());
    }
    SpongeH5MD::RestartDynamicState dynamic_state;
    if (!reader.Read_Dynamic_State(&dynamic_state))
    {
        controller.Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                      "Apply_H5_Dynamic_Restart_State",
                                      reader.Last_Error().c_str());
    }
    if (SpongeH5InputValidation::Has_Unsupported_Dynamic_State(dynamic_state))
    {
        const std::string unsupported =
            SpongeH5InputValidation::Unsupported_Dynamic_State_Reason(
                dynamic_state);
        controller.Throw_SPONGE_Error(
            spongeErrorValueErrorCommand, "Apply_H5_Dynamic_Restart_State",
            ("Reason:\n\tRestart contains unsupported dynamic state: " +
             unsupported + "\n")
                .c_str());
    }
    Apply_H5_Dynamic_Integrator_State(dynamic_state);
    std::string error_message;
    if (dynamic_state.has_nose_hoover_chain)
    {
        if (!nhc.is_initialized)
        {
            controller.Throw_SPONGE_Error(
                spongeErrorConflictingCommand, "Apply_H5_Dynamic_Restart_State",
                "Reason:\n\tRestart contains Nose-Hoover chain state, but the "
                "nose_hoover_chain thermostat is not initialized\n");
        }
        if (!nhc.Apply_H5_Restart_State(dynamic_state, &error_message))
        {
            controller.Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                                          "Apply_H5_Dynamic_Restart_State",
                                          error_message.c_str());
        }
    }
    const bool has_bussi_rng =
        dynamic_state.rng_state_text.count("bussi_thermostat") != 0 ||
        dynamic_state.rng_states.count("bussi_thermostat") != 0;
    if (bussi_thermo.is_initialized || has_bussi_rng)
    {
        if (!bussi_thermo.Apply_H5_Restart_State(dynamic_state, &error_message))
        {
            controller.Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                                          "Apply_H5_Dynamic_Restart_State",
                                          error_message.c_str());
        }
    }
    const bool has_middle_rng =
        dynamic_state.rng_states.count("middle_langevin") != 0;
    if (middle_langevin.is_initialized || has_middle_rng)
    {
        if (!middle_langevin.Apply_H5_Restart_State(dynamic_state,
                                                    &error_message))
        {
            controller.Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                                          "Apply_H5_Dynamic_Restart_State",
                                          error_message.c_str());
        }
    }
    const bool has_andersen_rng =
        dynamic_state.rng_states.count("andersen") != 0;
    if (ad_thermo.is_initialized || has_andersen_rng)
    {
        if (!ad_thermo.Apply_H5_Restart_State(dynamic_state, &error_message))
        {
            controller.Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                                          "Apply_H5_Dynamic_Restart_State",
                                          error_message.c_str());
        }
    }

    const auto pressure_baro =
        dynamic_state.barostat_float_states.find("pressure_based_barostat");
    if (press_baro.is_initialized ||
        pressure_baro != dynamic_state.barostat_float_states.end())
    {
        if (!press_baro.Apply_H5_Restart_State(dynamic_state, &error_message))
        {
            controller.Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                                          "Apply_H5_Dynamic_Restart_State",
                                          error_message.c_str());
        }
    }
    const bool has_mc_rng =
        dynamic_state.rng_states.count("monte_carlo_barostat") != 0;
    if (mc_baro.is_initialized || has_mc_rng)
    {
        if (!mc_baro.Apply_H5_Restart_State(dynamic_state, &error_message))
        {
            controller.Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                                          "Apply_H5_Dynamic_Restart_State",
                                          error_message.c_str());
        }
    }
}

void Materialize_H5_Metadynamics_Restart_Text_State()
{
    const auto input_plan = SpongeH5InputPlan::Resolve_Input_Plan(&controller);
    if (!input_plan.valid)
    {
        controller.Throw_SPONGE_Error(
            spongeErrorValueErrorCommand,
            "Materialize_H5_Metadynamics_Restart_Text_State",
            input_plan.error_message.c_str());
    }
    if (!input_plan.restart.binding.enabled ||
        !Requests_H5_Protocol_State(input_plan.restart.load_policy))
    {
        return;
    }

    SpongeH5MD::RestartH5Reader reader;
    if (!reader.Open(input_plan.restart.binding.path))
    {
        controller.Throw_SPONGE_Error(
            spongeErrorBadFileFormat,
            "Materialize_H5_Metadynamics_Restart_Text_State",
            reader.Last_Error().c_str());
    }
    SpongeH5MD::RestartProtocolState protocol_state;
    if (!reader.Read_Protocol_State(&protocol_state))
    {
        controller.Throw_SPONGE_Error(
            spongeErrorBadFileFormat,
            "Materialize_H5_Metadynamics_Restart_Text_State",
            reader.Last_Error().c_str());
    }
    if (protocol_state.metadynamics_states.empty())
    {
        return;
    }

    bool materialized = false;
    std::string error_message;
    const std::string metadynamics_name =
        cv_controller.protocol_metadynamics_name.empty()
            ? "meta"
            : cv_controller.protocol_metadynamics_name;
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
        controller.Throw_SPONGE_Error(
            spongeErrorValueErrorCommand,
            "Materialize_H5_Metadynamics_Restart_Text_State",
            error_message.c_str());
    }
}

void Apply_H5_Protocol_Restart_State()
{
    const auto input_plan = SpongeH5InputPlan::Resolve_Input_Plan(&controller);
    if (!input_plan.valid)
    {
        controller.Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                                      "Apply_H5_Protocol_Restart_State",
                                      input_plan.error_message.c_str());
    }
    if (!input_plan.restart.binding.enabled ||
        !Requests_H5_Protocol_State(input_plan.restart.load_policy))
    {
        return;
    }

    SpongeH5MD::RestartH5Reader reader;
    if (!reader.Open(input_plan.restart.binding.path))
    {
        controller.Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                      "Apply_H5_Protocol_Restart_State",
                                      reader.Last_Error().c_str());
    }
    SpongeH5MD::RestartProtocolState protocol_state;
    if (!reader.Read_Protocol_State(&protocol_state))
    {
        controller.Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                      "Apply_H5_Protocol_Restart_State",
                                      reader.Last_Error().c_str());
    }
    if (protocol_state.sits_states.empty() &&
        protocol_state.metadynamics_states.empty() &&
        protocol_state.restraint_states.empty() &&
        protocol_state.cv_reference_states.empty())
    {
        controller.Throw_SPONGE_Error(
            spongeErrorValueErrorCommand, "Apply_H5_Protocol_Restart_State",
            "Reason:\n\tNo supported protocol restart state is available\n");
    }
    if (!protocol_state.metadynamics_states.empty() && !meta.is_initialized)
    {
        controller.Throw_SPONGE_Error(
            spongeErrorConflictingCommand, "Apply_H5_Protocol_Restart_State",
            "Reason:\n\tRestart contains metadynamics state, but the meta "
            "module is not initialized\n");
    }
    if (!protocol_state.restraint_states.empty())
    {
        const auto state =
            std::find_if(protocol_state.restraint_states.begin(),
                         protocol_state.restraint_states.end(),
                         [](const SpongeH5MD::RestartRestraintState& value)
                         { return value.name == restrain.h5_restraint_name; });
        if (state == protocol_state.restraint_states.end())
        {
            controller.Throw_SPONGE_Error(
                spongeErrorConflictingCommand,
                "Apply_H5_Protocol_Restart_State",
                "Reason:\n\trestart restraint state does not match the "
                "active protocol restraint object\n");
        }
        std::string restraint_error;
        if (!restrain.Apply_H5_Reference_Coordinates(
                state->reference_coordinates, &restraint_error))
        {
            controller.Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                                          "Apply_H5_Protocol_Restart_State",
                                          restraint_error.c_str());
        }
    }
    for (const auto& reference : protocol_state.cv_reference_states)
    {
        const auto active =
            cv_controller.protocol_cv_reference.find(reference.name);
        if (active == cv_controller.protocol_cv_reference.end() ||
            active->second != reference.reference_coordinates)
        {
            controller.Throw_SPONGE_Error(
                spongeErrorConflictingCommand,
                "Apply_H5_Protocol_Restart_State",
                "Reason:\n\trestart CV reference does not match an active "
                "native CV object\n");
        }
    }
    if (!protocol_state.metadynamics_states.empty())
    {
        const std::string metadynamics_name =
            cv_controller.protocol_metadynamics_name.empty()
                ? "meta"
                : cv_controller.protocol_metadynamics_name;
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
            controller.Throw_SPONGE_Error(
                spongeErrorConflictingCommand,
                "Apply_H5_Protocol_Restart_State",
                "Reason:\n\trestart metadynamics state does not match the "
                "active protocol metadynamics object\n");
        }
        if (state != protocol_state.metadynamics_states.end())
        {
            std::string metadynamics_error;
            if (!meta.Apply_H5_Restart_State(*state, &metadynamics_error))
            {
                controller.Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                                              "Apply_H5_Protocol_Restart_State",
                                              metadynamics_error.c_str());
            }
        }
    }
    if (protocol_state.sits_states.empty())
    {
        return;
    }
    if (sits.is_initialized &&
        controller.Command_Exist(sits.module_name, "nk_in_file"))
    {
        return;
    }
    std::string error_message;
    if (!sits.Apply_H5_Restart_State(protocol_state, &error_message))
    {
        controller.Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                                      "Apply_H5_Protocol_Restart_State",
                                      error_message.c_str());
    }
}
}  // namespace

int main(int argc, char* argv[])
{
    Main_Initial(argc, argv);
    if (md_info.sys.steps > INT_MAX - md_info.sys.step_limit)
    {
        controller.Throw_SPONGE_Error(
            spongeErrorValueErrorCommand, "main",
            "Reason:\n\trestart step plus step_limit exceeds INT_MAX\n");
    }
    md_info.sys.step_limit += md_info.sys.steps;
    for (; md_info.sys.steps <= md_info.sys.step_limit; md_info.sys.steps++)
    {
        Main_Sync_Dynamic_Targets_To_Controllers();
        Main_Calculate_Force();
        Main_Iteration();
        Main_Print();
    }
    Main_Clear();
    return 0;
}

void Main_Initial(int argc, char* argv[])
{
    controller.Initial(argc, argv, SUBPACKAGE_HINT);
    Validate_H5_Input_Plan();
    Materialize_H5_Native_Topology_Core();
    Materialize_H5_Topology_And_Protocol_Sidecars();
    Materialize_H5_Protocol_Restart_Sidecars();
    Xponge::system.Load_Inputs(&controller);
    Materialize_H5_Native_Topology_Forcefield();
    cv_controller.atom_numbers = Xponge::Load_Get_Atom_Numbers(&Xponge::system);
    cv_controller.Initial(&controller,
                          &md_info.no_direct_interaction_virtual_atom_numbers);
    md_info.Initial(&controller);
    md_info.sys.Update_Targets_By_Schedule(0);
    controller.Step_Print_Initial("potential", "%.2f");
    controller.Step_Print_Initial("eff_pot", "%.7e");
    qc.Initial(&controller, md_info.atom_numbers, md_info.crd);
    cv_controller.atom_numbers = md_info.atom_numbers;
    plugin.Initial(&md_info, &controller, &cv_controller, &neighbor_list);

    if (md_info.mode >= md_info.NVT &&
        (!controller.Command_Exist("thermostat") &&
         !controller.Command_Exist("thermostat_mode")))
    {
        controller.Throw_SPONGE_Error(
            spongeErrorMissingCommand, "Main_Initial",
            "Reason:\n\tthermostat is required for NVT or NPT simulations\n");
    }
    if (THERMOSTAT_IS("middle_langevin") || THERMOSTAT_IS("langevin"))
    {
        middle_langevin.Initial(&controller, md_info.atom_numbers,
                                md_info.sys.target_temperature, md_info.h_mass);
    }
    else if (THERMOSTAT_IS("andersen"))
    {
        ad_thermo.Initial(&controller, md_info.sys.target_temperature,
                          md_info.atom_numbers, md_info.sys.dt_in_ps,
                          md_info.h_mass);
    }
    else if (THERMOSTAT_IS("bussi_thermostat"))
    {
        bussi_thermo.Initial(&controller, md_info.sys.target_temperature);
    }
    else if (THERMOSTAT_IS("berendsen_thermostat"))
    {
        bd_thermo.Initial(&controller, md_info.sys.target_temperature);
    }
    else if (THERMOSTAT_IS("nose_hoover_chain"))
    {
        nhc.Initial(&controller, md_info.atom_numbers,
                    md_info.sys.target_temperature, md_info.h_mass);
    }

    if (md_info.mode == md_info.NPT && !controller.Command_Exist("barostat") &&
        !controller.Command_Exist("barostat_mode"))
    {
        controller.Throw_SPONGE_Error(
            spongeErrorMissingCommand, "Main_Initial",
            "Reason:\n\tbarostat is required for NPT simulations\n");
    }
    if (BAROSTAT_IS("andersen_barostat") || BAROSTAT_IS("bussi_barostat") ||
        BAROSTAT_IS("berendsen_barostat"))
    {
        press_baro.Initial(&controller, md_info.sys.target_pressure,
                           md_info.pbc.cell, &Main_Box_Change);
    }
    if (BAROSTAT_IS("monte_carlo_barostat"))
    {
        mc_baro.Initial(&controller, md_info.atom_numbers,
                        md_info.sys.target_pressure, md_info.sys.box_length,
                        md_info.pbc.cell);
    }

    Apply_H5_Dynamic_Restart_State();

    if (md_info.pbc.pbc)
    {
        lj.Initial(&controller, md_info.nb.cutoff);
        lj_soft.Initial(&controller, md_info.nb.cutoff);
        pm.Initial(&controller, md_info.atom_numbers, md_info.pbc.cell,
                   md_info.pbc.rcell, md_info.sys.box_length, md_info.nb.cutoff,
                   md_info.no_direct_interaction_virtual_atom_numbers);
        pairwise_force.Initial(&controller);
        nb14.Initial(&controller, lj.h_LJ_A, lj.h_LJ_B, lj.h_atom_LJ_type);

        sits.Initial(&controller, md_info.atom_numbers);
        if (sits.is_initialized && sits.selectively_applied)
        {
            sits_dihedral.Initial(&controller, "sits_dihedral");
            sits_nb14.Initial(&controller, lj.h_LJ_A, lj.h_LJ_B,
                              lj.h_atom_LJ_type, "sits_nb14");
            sits_cmap.Initial(&controller, "sits_cmap");
        }
        sits.Check_Solvent(&controller, md_info.atom_numbers,
                           solvent_lj.solvent_numbers);
    }
    else
    {
        LJ_NOPBC.Initial(&controller, md_info.nb.cutoff);
        CF_NOPBC.Initial(&controller, md_info.atom_numbers, md_info.nb.cutoff);
        if (controller.Command_Exist("gb", "in_file") ||
            !Xponge::system.generalized_born.radius.empty())
        {
            gb.Initial(&controller, md_info.nb.cutoff);
        }
        nb14.Initial(&controller, LJ_NOPBC.h_LJ_A, LJ_NOPBC.h_LJ_B,
                     LJ_NOPBC.h_atom_LJ_type);
        sits.Initial(&controller, md_info.atom_numbers);
    }

    bond.Initial(&controller, &md_info.sys.connectivity,
                 &md_info.sys.connected_distance);
    angle.Initial(&controller);
    urey_bradley.Initial(&controller);
    cmap.Initial(&controller);
    dihedral.Initial(&controller);
    improper.Initial(&controller);
    listed_forces.Initial(&controller, &md_info.sys.connectivity,
                          &md_info.sys.connected_distance);

    sw.Initial(&controller, "SW", &neighbor_list.is_needed_full);
    edip.Initial(&controller, "EDIP", &neighbor_list.is_needed_full);
    eam.Initial(&controller, md_info.atom_numbers, "EAM",
                &neighbor_list.is_needed_full);
    tersoff.Initial(&controller, md_info.atom_numbers, "TERSOFF",
                    &neighbor_list.is_needed_full);
    reaxff.Initial(&controller, md_info.atom_numbers, md_info.nb.cutoff,
                   &neighbor_list.cutoff_full, &neighbor_list.is_needed_full);

    if (Xponge::system.positional_restraint.present)
    {
        restrain.Initial(&controller, md_info.atom_numbers, md_info.crd,
                         Xponge::system.positional_restraint);
    }
    else
    {
        restrain.Initial(&controller, md_info.atom_numbers, md_info.crd);
    }
    hard_wall.Initial(&controller, md_info.sys.target_temperature,
                      md_info.sys.target_pressure, md_info.mode == md_info.NPT);
    soft_walls.Initial(&controller, md_info.atom_numbers);

    if (controller.Command_Exist("constrain_mode"))
    {
        constrain.Initial_List(&controller, md_info.sys.connected_distance,
                               md_info.h_mass);
        constrain.Initial_Constrain(&controller, md_info.atom_numbers,
                                    md_info.dt, md_info.sys.box_length,
                                    md_info.h_mass, &md_info.sys.freedom);
        settle.Initial(&controller, &constrain, md_info.h_mass);
        if (controller.Command_Choice("constrain_mode", "SHAKE"))
        {
            shake.Initial_SHAKE(&controller, &constrain);
        }
        if (md_info.mode == md_info.MINIMIZATION)
        {
            constrain.v_factor = 0.0f;
        }
        if (middle_langevin.is_initialized)
        {
            constrain.v_factor = middle_langevin.exp_gamma;
            constrain.x_factor = 0.5 * (1. + middle_langevin.exp_gamma);
        }
    }
    vatom.Initial(&controller, &cv_controller, md_info.atom_numbers,
                  md_info.no_direct_interaction_virtual_atom_numbers,
                  cv_controller.cv_vatom_name, md_info.h_mass,
                  &md_info.sys.freedom, &md_info.sys.connectivity);
    vatom.Coordinate_Refresh(md_info.crd, md_info.pbc.cell, md_info.pbc.rcell);
    initial_velocity.Initial(&controller, &md_info);

    if (md_info.pbc.pbc)
    {
        neighbor_list.Initial(&controller, md_info.atom_numbers,
                              md_info.nb.cutoff, md_info.nb.skin,
                              md_info.pbc.cell, md_info.pbc.rcell);
    }
    steer_cv.Initial(&controller, &cv_controller);
    restrain_cv.Initial(&controller, &cv_controller);
    Materialize_H5_Metadynamics_Restart_Text_State();
    meta.Initial(&controller, &cv_controller);
    Apply_H5_Protocol_Restart_State();

    cv_controller.Print_Initial();
    plugin.After_Initial();
    cv_controller.Input_Check();

    md_info.ug.Initial_Edge(md_info.atom_numbers);
    constrain.update_ug_connectivity(&md_info.ug.connectivity);
    settle.update_ug_connectivity(&md_info.ug.connectivity);
    vatom.update_ug_connectivity(&md_info.ug.connectivity);
    md_info.ug.Read_Update_Group(md_info.atom_numbers);
    md_info.mol.Initial(&controller);
    if (md_info.pbc.pbc)
    {
        solvent_lj.Initial(&controller, &lj, &lj_soft, &md_info,
                           md_info.mode >= md_info.NVT);
    }
    Main_Process_Management();

    if (CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size)
    {
        Main_Refresh_Local_State(true);
        initial_velocity.Finalize(&controller, &md_info, &dd, &settle, &shake);
        plugin.Set_Domain_Information(&dd);
    }

    pm.Get_Atoms(&controller, md_info.crd, md_info.d_charge, dd.atom_numbers,
                 dd.crd, dd.d_charge, dd.atom_local, true, true, true, true);

    controller.Print_First_Line_To_Mdout();
    md_info.output.Initial_H5_Trajectory(&controller);
    md_info.output.Initial_H5_Observable(&controller);
    md_info.output.Initial_H5_Restart(&controller);
    md_info.output.Initial_H5_Nose_Hoover_Chain(
        &controller, nhc.is_initialized ? nhc.chain_length : 0);
    md_info.output.Initial_H5_Sits_Nk(
        &controller, sits.is_initialized ? sits.module_name : NULL,
        sits.is_initialized && sits.classic_sits.is_initialized
            ? sits.classic_sits.k_numbers
            : 0);
    md_info.output.Initial_H5_Metadynamics(&controller, meta.is_initialized);
    md_info.output.Initial_H5_Qc(&controller, qc.is_initialized);
    md_info.output.Initial_H5_Reaxff(
        &controller, reaxff.is_initialized,
        reaxff.eeq.is_initialized
            ? static_cast<std::size_t>(reaxff.eeq.atom_numbers)
            : static_cast<std::size_t>(0));
    md_info.output.Prepare_H5_Swmr_Layout(
        &controller, meta.is_initialized ? meta.h5_object_name.c_str() : NULL,
        qc.is_initialized);
    if (meta.is_initialized)
    {
        md_info.output.Write_H5_Metadynamics_Diagnostic_File(
            &controller, meta.h5_object_name.c_str(), "hills", "myhill.log");
        md_info.output.Write_H5_Metadynamics_Diagnostic_File(
            &controller, meta.h5_object_name.c_str(), "history", "history.log");
        md_info.output.Write_H5_Metadynamics_Diagnostic_File(
            &controller, meta.h5_object_name.c_str(), "edge",
            meta.edge_file_name);
        md_info.output.Write_H5_Metadynamics_Diagnostic_File(
            &controller, meta.h5_object_name.c_str(), "direct_export",
            meta.write_directly_file_name);
    }
    md_info.output.Start_H5_Swmr(&controller);
}

void Main_Calculate_Force()
{
    bool use_reaxff_eeq = reaxff.eeq.is_initialized;
    const int cv_atom_numbers =
        md_info.atom_numbers +
        md_info.no_direct_interaction_virtual_atom_numbers;
    md_info.MD_Reset_Atom_Energy_And_Virial_And_Force();
    qc.Solve_SCF(dd.crd, md_info.sys.box_length, true, md_info.sys.steps);
    if (qc.is_initialized && qc.scf_output_file != NULL)
    {
        fflush(qc.scf_output_file);
        md_info.output.Write_H5_Qc_Scf_Output_File(&controller,
                                                   qc.scf_output_file_name);
    }
    if (md_info.mode == md_info.MINIMIZATION && md_info.min.dynamic_dt)
    {
        md_info.need_potential = 1;
    }
    mc_baro.Ask_For_Calculate_Potential(md_info.sys.steps,
                                        &md_info.need_potential);
    press_baro.Ask_For_Calculate_Pressure(md_info.sys.steps,
                                          &md_info.need_pressure);
    if (press_baro.is_initialized && md_info.output.Check_Mdout_Step())
    {
        md_info.need_pressure = 1;
    }
    if (bd_thermo.is_initialized || bussi_thermo.is_initialized ||
        nhc.is_initialized)
    {
        md_info.need_kinetic = 1;
    }
    sits.Reset_Force_Energy(&md_info.need_potential);

    controller.Get_Time_Recorder("Calculate_Force")->Start();
    pm.Get_Atoms(&controller, md_info.crd, md_info.d_charge, dd.atom_numbers,
                 dd.crd, dd.d_charge, dd.atom_local, false, false, true, false);
    if (CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size)
    {
        dd.Reset_Force_and_Virial(&md_info);
        // QC 梯度必须在 dd.Reset_Force_and_Virial 之后调用
        if (qc.is_initialized && qc.need_gradient)
            qc.Compute_Gradient(dd.frc, dd.crd, md_info.sys.box_length,
                                md_info.need_pressure, dd.d_virial);
        dd.Update_Ghost(&controller);
        neighbor_list.Update(
            dd.atom_local, dd.atom_numbers, dd.ghost_numbers, dd.crd,
            md_info.pbc.cell, md_info.pbc.rcell, md_info.sys.steps,
            neighbor_list.CONDITIONAL_UPDATE, md_info.nb.d_excluded_list_start,
            md_info.nb.d_excluded_list, md_info.nb.d_excluded_numbers);

        reaxff.Calculate_Force(&dd, &md_info, &neighbor_list);

        LJ_NOPBC.LJ_Force_With_Atom_Energy(
            dd.atom_numbers, dd.crd, dd.frc, md_info.need_potential,
            dd.d_energy, dd.d_excluded_list_start, dd.d_excluded_list,
            dd.d_excluded_numbers);
        CF_NOPBC.Coulomb_Force_With_Atom_Energy(
            dd.atom_numbers, dd.crd, dd.d_charge, dd.frc,
            md_info.need_potential, dd.d_energy, dd.d_excluded_list_start,
            dd.d_excluded_list, dd.d_excluded_numbers);
        gb.Get_Effective_Born_Radius(dd.crd);
        gb.GB_Force_With_Atom_Energy(dd.atom_numbers, dd.crd, dd.d_charge,
                                     dd.frc, dd.d_energy);

        if (!use_reaxff_eeq)
        {
            pm.MPI_PME_Excluded_Force_With_Atom_Energy(
                dd.atom_numbers, dd.atom_local, dd.atom_local_id, dd.crd,
                md_info.pbc.cell, md_info.pbc.rcell, dd.d_charge,
                dd.d_excluded_list_start, dd.d_excluded_list,
                dd.d_excluded_numbers, dd.frc, md_info.need_potential,
                dd.d_energy, md_info.need_pressure, dd.d_virial);
        }

        if (sits.is_initialized && sits.selectively_applied)
        {
            sits_dihedral.Dihedral_Force_With_Atom_Energy_And_Virial(
                dd.crd, md_info.pbc.cell, md_info.pbc.rcell,
                sits.pw_select.select_force[0], md_info.need_potential,
                sits.pw_select.select_atom_energy[0], md_info.need_pressure,
                sits.pw_select.select_atom_virial_tensor[0]);
            sits_nb14.Non_Bond_14_LJ_CF_Force_With_Atom_Energy_And_Virial(
                dd.crd, dd.d_charge, md_info.pbc.cell, md_info.pbc.rcell,
                sits.pw_select.select_force[0], md_info.need_potential,
                sits.pw_select.select_atom_energy[0], md_info.need_pressure,
                sits.pw_select.select_atom_virial_tensor[0]);
            sits_cmap.CMAP_Force_With_Atom_Energy_And_Virial(
                dd.crd, md_info.pbc.cell, md_info.pbc.rcell,
                sits.pw_select.select_force[0], md_info.need_potential,
                sits.pw_select.select_atom_energy[0], md_info.need_pressure,
                sits.pw_select.select_atom_virial_tensor[0]);
            sits.SITS_LJ_Direct_CF_Force_With_Atom_Energy_And_Virial(
                md_info.atom_numbers, dd.atom_numbers,
                solvent_lj.local_solvent_numbers, dd.ghost_numbers, dd.crd,
                dd.d_charge, &lj, dd.frc, md_info.pbc.cell, md_info.pbc.rcell,
                neighbor_list.d_nl, md_info.nb.cutoff, pm.beta,
                md_info.need_potential, dd.d_energy, md_info.need_pressure,
                dd.d_virial, pm.d_direct_atom_energy);
            sits.SITS_LJ_Soft_Core_Direct_CF_Force_With_Atom_Energy_And_Virial(
                md_info.atom_numbers, dd.atom_numbers,
                solvent_lj.local_solvent_numbers, dd.ghost_numbers, dd.crd,
                dd.d_charge, &lj_soft, dd.frc, md_info.pbc.cell,
                md_info.pbc.rcell, neighbor_list.d_nl, md_info.nb.cutoff,
                pm.beta, md_info.need_potential, dd.d_energy,
                md_info.need_pressure, dd.d_virial, pm.d_direct_atom_energy);
        }
        else
        {
            lj.LJ_PME_Direct_Force_With_Atom_Energy_And_Virial(
                md_info.atom_numbers, dd.atom_numbers,
                solvent_lj.local_solvent_numbers, dd.ghost_numbers, dd.crd,
                dd.d_charge, dd.frc, md_info.pbc.cell, md_info.pbc.rcell,
                neighbor_list.d_nl, pm.beta, md_info.need_potential,
                dd.d_energy, md_info.need_pressure, dd.d_virial,
                pm.d_direct_atom_energy);

            lj_soft.LJ_Soft_Core_PME_Direct_Force_With_Atom_Energy_And_Virial(
                md_info.atom_numbers, dd.atom_numbers,
                solvent_lj.local_solvent_numbers, dd.ghost_numbers, dd.crd,
                dd.d_charge, dd.frc, md_info.pbc.cell, md_info.pbc.rcell,
                neighbor_list.d_nl, pm.beta, md_info.need_potential,
                dd.d_energy, md_info.need_pressure, dd.d_virial,
                pm.d_direct_atom_energy);
        }
        solvent_lj.LJ_PME_Direct_Force_With_Atom_Energy_And_Virial(
            dd.atom_numbers, dd.res_numbers, dd.res_start, dd.crd, dd.d_charge,
            dd.frc, md_info.pbc.cell, md_info.pbc.rcell, neighbor_list.d_nl,
            pm.beta, md_info.need_potential, dd.d_energy, md_info.need_pressure,
            dd.d_virial, pm.d_direct_atom_energy);

        lj.Long_Range_Correction(
            md_info.need_pressure, dd.d_virial, md_info.need_potential,
            dd.d_energy,
            md_info.pbc.cell.a11 * md_info.pbc.cell.a22 * md_info.pbc.cell.a33);

        lj_soft.Long_Range_Correction(
            md_info.need_pressure, dd.d_virial, md_info.need_potential,
            dd.d_energy,
            md_info.pbc.cell.a11 * md_info.pbc.cell.a22 * md_info.pbc.cell.a33);
        sw.SW_Force_With_Atom_Energy_And_Virial_Full_NL(
            dd.atom_numbers, dd.crd, dd.frc, md_info.pbc.cell,
            md_info.pbc.rcell, neighbor_list.full_neighbor_list.d_nl,
            md_info.need_potential, dd.d_energy, md_info.need_pressure,
            dd.d_virial);
        edip.EDIP_Force_With_Atom_Energy_And_Virial_Full_NL(
            dd.atom_numbers, dd.crd, dd.frc, md_info.pbc.cell,
            md_info.pbc.rcell, neighbor_list.full_neighbor_list.d_nl,
            md_info.need_potential, dd.d_energy, md_info.need_pressure,
            dd.d_virial);
        eam.EAM_Force_With_Atom_Energy_And_Virial(
            dd.atom_numbers, dd.crd, dd.frc, md_info.pbc.cell,
            md_info.pbc.rcell, neighbor_list.full_neighbor_list.d_nl,
            md_info.need_potential, dd.d_energy, md_info.need_pressure,
            dd.d_virial);
        tersoff.TERSOFF_Force_With_Atom_Energy_And_Virial(
            dd.atom_numbers, dd.crd, dd.frc, md_info.pbc.cell,
            md_info.pbc.rcell, neighbor_list.full_neighbor_list.d_nl,
            md_info.need_potential, dd.d_energy, md_info.need_pressure,
            dd.d_virial);
        listed_forces.Compute_Force(dd.atom_numbers, dd.crd, md_info.pbc.cell,
                                    md_info.pbc.rcell, dd.frc,
                                    md_info.need_potential, dd.d_energy,
                                    md_info.need_pressure, dd.d_virial);
        pairwise_force.Compute_Force(
            neighbor_list.d_nl, dd.crd, md_info.pbc.cell, md_info.pbc.rcell,
            md_info.nb.cutoff, pm.beta, dd.d_charge, dd.frc,
            md_info.need_potential, dd.d_energy, md_info.need_pressure,
            dd.d_virial, pm.d_direct_atom_energy);
        angle.Angle_Force_With_Atom_Energy_And_Virial(
            dd.crd, md_info.pbc.cell, md_info.pbc.rcell, dd.frc,
            md_info.need_potential, dd.d_energy, md_info.need_pressure,
            dd.d_virial);
        urey_bradley.Urey_Bradley_Force_With_Atom_Energy_And_Virial(
            dd.crd, md_info.pbc.cell, md_info.pbc.rcell, dd.frc,
            md_info.need_potential, dd.d_energy, md_info.need_pressure,
            dd.d_virial);
        bond.Bond_Force_With_Atom_Energy_And_Virial(
            dd.crd, md_info.pbc.cell, md_info.pbc.rcell, dd.frc,
            md_info.need_potential, dd.d_energy, md_info.need_pressure,
            dd.d_virial);
        cmap.CMAP_Force_With_Atom_Energy_And_Virial(
            dd.crd, md_info.pbc.cell, md_info.pbc.rcell, dd.frc,
            md_info.need_potential, dd.d_energy, md_info.need_pressure,
            dd.d_virial);
        dihedral.Dihedral_Force_With_Atom_Energy_And_Virial(
            dd.crd, md_info.pbc.cell, md_info.pbc.rcell, dd.frc,
            md_info.need_potential, dd.d_energy, md_info.need_pressure,
            dd.d_virial);
        improper.Dihedral_Force_With_Atom_Energy_And_Virial(
            dd.crd, md_info.pbc.cell, md_info.pbc.rcell, dd.frc,
            md_info.need_potential, dd.d_energy, md_info.need_pressure,
            dd.d_virial);
        nb14.Non_Bond_14_LJ_CF_Force_With_Atom_Energy_And_Virial(
            dd.crd, dd.d_charge, md_info.pbc.cell, md_info.pbc.rcell, dd.frc,
            md_info.need_potential, dd.d_energy, md_info.need_pressure,
            dd.d_virial);
        soft_walls.Compute_Force(dd.atom_numbers, dd.crd, dd.frc,
                                 md_info.need_potential, dd.d_energy);
        plugin.Calculate_Force();

        restrain.Restraint(dd.crd, md_info.pbc.cell, md_info.pbc.rcell,
                           md_info.need_potential, dd.d_energy,
                           md_info.need_pressure, dd.d_virial, dd.frc, &md_info,
                           &dd);

        if (CONTROLLER::MPI_size == 1 && CONTROLLER::PM_MPI_size == 1)
        {
            vatom.Coordinate_Refresh_CV(dd.crd, md_info.pbc.cell,
                                        md_info.pbc.rcell);
            if (!use_reaxff_eeq)
            {
                pm.PME_Reciprocal_Force_With_Energy_And_Virial(
                    dd.crd, md_info.pbc.cell, md_info.pbc.rcell, dd.d_charge,
                    dd.frc, md_info.need_pressure, md_info.need_potential,
                    dd.d_virial, dd.d_energy, md_info.sys.steps);
            }

            cv_controller.Compute_CV_For_Print(
                cv_atom_numbers, dd.crd, md_info.pbc.cell, md_info.pbc.rcell,
                md_info.sys.steps, md_info.output.write_mdout_interval,
                md_info.output.print_zeroth_frame);

            steer_cv.Steer(cv_atom_numbers, dd.crd, md_info.pbc.cell,
                           md_info.pbc.rcell, md_info.sys.steps, dd.d_energy,
                           dd.d_virial, dd.frc, md_info.need_potential,
                           md_info.need_pressure);
            restrain_cv.Restraint(
                cv_atom_numbers, dd.crd, md_info.pbc.cell, md_info.pbc.rcell,
                md_info.sys.steps, dd.d_energy, dd.d_virial, dd.frc,
                md_info.need_potential, md_info.need_pressure);
            meta.Do_Metadynamics(cv_atom_numbers, dd.crd, md_info.pbc.cell,
                                 md_info.pbc.rcell, md_info.sys.steps,
                                 md_info.need_potential, md_info.need_pressure,
                                 dd.frc, dd.d_energy, dd.d_virial,
                                 md_info.sys.h_temperature);
            if (meta.is_initialized && meta.potential_update_interval > 0 &&
                md_info.sys.steps % meta.potential_update_interval == 0)
            {
                md_info.output.Write_H5_Metadynamics_Diagnostic_File(
                    &controller, meta.h5_object_name.c_str(), "hills",
                    "myhill.log");
            }
            vatom.Force_Redistribute_CV(dd.crd, md_info.pbc.cell,
                                        md_info.pbc.rcell, dd.frc);
        }
        else
        {
            if (!use_reaxff_eeq)
            {
                pm.Send_Recv_Force(&controller, md_info.frc, dd.frc,
                                   dd.atom_numbers);
            }
        }
        sits.Update_And_Enhance(
            md_info.sys.steps, md_info.sys.d_potential, md_info.need_pressure,
            dd.d_virial, dd.frc,
            1.0f / (CONSTANT_kB * md_info.sys.target_temperature));
        if (sits.is_initialized && sits.classic_sits.h5_nk_pending)
        {
            md_info.output.Append_H5_Sits_Nk_Frame(
                &controller, sits.module_name, sits.classic_sits.nk_record_cpu,
                sits.classic_sits.k_numbers);
            sits.classic_sits.h5_nk_pending = 0;
        }
        vatom.Force_Redistribute(dd.crd, md_info.pbc.cell, md_info.pbc.rcell,
                                 dd.frc);
    }
    else
    {
        if (!use_reaxff_eeq)
        {
            pm.reset_global_force(
                md_info.no_direct_interaction_virtual_atom_numbers);
            vatom.Coordinate_Refresh_CV(pm.g_crd, md_info.pbc.cell,
                                        md_info.pbc.rcell);
            pm.PME_Reciprocal_Force_With_Energy_And_Virial(
                md_info.crd, md_info.pbc.cell, md_info.pbc.rcell,
                md_info.d_charge, md_info.frc, md_info.need_pressure,
                md_info.need_potential, md_info.d_atom_virial_tensor,
                md_info.d_atom_energy, md_info.sys.steps);
            cv_controller.Compute_CV_For_Print(
                cv_atom_numbers, pm.g_crd, md_info.pbc.cell, md_info.pbc.rcell,
                md_info.sys.steps, md_info.output.write_mdout_interval,
                md_info.output.print_zeroth_frame);
            steer_cv.Steer(cv_atom_numbers, pm.g_crd, md_info.pbc.cell,
                           md_info.pbc.rcell, md_info.sys.steps,
                           md_info.d_atom_energy, md_info.d_atom_virial_tensor,
                           pm.g_frc, md_info.need_potential,
                           md_info.need_pressure);
            restrain_cv.Restraint(
                cv_atom_numbers, pm.g_crd, md_info.pbc.cell, md_info.pbc.rcell,
                md_info.sys.steps, md_info.d_atom_energy,
                md_info.d_atom_virial_tensor, pm.g_frc, md_info.need_potential,
                md_info.need_pressure);
            meta.Do_Metadynamics(
                cv_atom_numbers, pm.g_crd, md_info.pbc.cell, md_info.pbc.rcell,
                md_info.sys.steps, md_info.need_potential,
                md_info.need_pressure, pm.g_frc, md_info.d_atom_energy,
                md_info.d_atom_virial_tensor, md_info.sys.h_temperature);
            vatom.Force_Redistribute_CV(pm.g_crd, md_info.pbc.cell,
                                        md_info.pbc.rcell, pm.g_frc);
            pm.add_force_g_to_l(md_info.frc);
            pm.Send_Recv_Force(&controller, md_info.frc, dd.frc,
                               dd.atom_numbers);
        }
    }
    md_info.min.Scale_Force_For_Dynamic_Dt(dd.atom_numbers, dd.d_mass_inverse,
                                           dd.frc, dd.vel, dd.acc);
    controller.Get_Time_Recorder("Calculate_Force")->Stop();
}

void Main_Refresh_Local_State(bool rebuild_dd)
{
    if (rebuild_dd)
    {
        dd.Send_Recv_Dom_Dec(&controller);
        dd.Find_Neighbor_Domain(&controller, &md_info);
        dd.Get_Atoms(&controller, &md_info);
    }
    dd.Get_Ghost(&controller, &md_info);
    dd.Get_Excluded(&controller, &md_info);

    neighbor_list.Update(
        dd.atom_local, dd.atom_numbers, dd.ghost_numbers, dd.crd,
        md_info.pbc.cell, md_info.pbc.rcell, md_info.sys.steps,
        neighbor_list.FORCED_UPDATE, md_info.nb.d_excluded_list_start,
        md_info.nb.d_excluded_list, md_info.nb.d_excluded_numbers);

    middle_langevin.Get_Local(dd.atom_local, dd.atom_numbers);
    ad_thermo.Get_Local(dd.atom_local, dd.atom_numbers);
    nhc.Get_Local(dd.atom_local, dd.atom_numbers);

    lj.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers);
    lj_soft.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers);
    solvent_lj.Get_Local(dd.res_numbers, dd.res_len, dd.atom_numbers,
                         dd.d_mass);
    listed_forces.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers,
                            dd.atom_local_label, dd.atom_local_id);
    pairwise_force.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers,
                             dd.atom_local_label, dd.atom_local_id);

    angle.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers,
                    dd.atom_local_label, dd.atom_local_id);
    urey_bradley.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers,
                           dd.atom_local_label, dd.atom_local_id);
    bond.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers,
                   dd.atom_local_label, dd.atom_local_id);
    cmap.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers,
                   dd.atom_local_label, dd.atom_local_id);
    dihedral.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers,
                       dd.atom_local_label, dd.atom_local_id);
    improper.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers,
                       dd.atom_local_label, dd.atom_local_id);
    nb14.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers,
                   dd.atom_local_label, dd.atom_local_id);
    restrain.Get_Local(dd.atom_local, dd.atom_numbers, dd.atom_local_label,
                       dd.atom_local_id);
    constrain.Get_Local(dd.atom_local_id, dd.atom_local_label, dd.atom_numbers);
    settle.Get_Local(dd.atom_local_id, dd.atom_local_label, dd.atom_numbers);
    vatom.Get_Local(dd.atom_local_id, dd.atom_local_label, dd.atom_numbers);
    sits.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers);
    if (sits.is_initialized && sits.selectively_applied)
    {
        sits_dihedral.Get_Local(dd.atom_local, dd.atom_numbers,
                                dd.ghost_numbers, dd.atom_local_label,
                                dd.atom_local_id);
        sits_nb14.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers,
                            dd.atom_local_label, dd.atom_local_id);
        sits_cmap.Get_Local(dd.atom_local, dd.atom_numbers, dd.ghost_numbers,
                            dd.atom_local_label, dd.atom_local_id);
    }
}

void Main_Iteration()
{
    controller.Get_Time_Recorder("Iteration")->Start();
    if (md_info.need_potential || md_info.need_pressure || md_info.need_kinetic)
    {
        dd.Get_Ek_and_Temperature(&controller, &md_info);
    }
    dd.Get_Potential(&controller, &md_info);
    if (md_info.mode != md_info.RERUN)
    {
        Main_MC_Barostat();
        if (CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size)
        {
            settle.Remember_Last_Coordinates(dd.crd, md_info.pbc.cell,
                                             md_info.pbc.rcell);
            shake.Remember_Last_Coordinates(dd.crd, md_info.pbc.cell,
                                            md_info.pbc.rcell);

            if (md_info.mode == md_info.NVE)
            {
                md_info.nve.Leap_Frog(dd.atom_numbers, dd.vel, dd.crd, dd.frc,
                                      dd.d_mass_inverse, md_info.dt);
            }
            else if (md_info.mode == md_info.MINIMIZATION)
            {
                md_info.min.Gradient_Descent(dd.atom_numbers, dd.crd, dd.frc,
                                             dd.vel, dd.d_mass_inverse);
                constrain.v_factor = fmaxf(FLT_MIN, md_info.min.momentum_keep);
            }
            else if (middle_langevin.is_initialized)
            {
                middle_langevin.MD_Iteration_Leap_Frog(dd.frc, dd.vel, dd.acc,
                                                       dd.crd);
                constrain.v_factor = middle_langevin.exp_gamma;
                constrain.x_factor = 0.5f * middle_langevin.exp_gamma + 0.5f;
            }
            else if (bd_thermo.is_initialized)
            {
                bd_thermo.Record_Temperature(dd.temperature,
                                             md_info.sys.freedom);
                md_info.nve.Leap_Frog(dd.atom_numbers, dd.vel, dd.crd, dd.frc,
                                      dd.d_mass_inverse, md_info.dt);
                bd_thermo.Scale_Velocity(dd.atom_numbers, dd.vel);
            }
            else if (bussi_thermo.is_initialized)
            {
                bussi_thermo.Record_Temperature(dd.temperature,
                                                md_info.sys.freedom);
                md_info.nve.Leap_Frog(dd.atom_numbers, dd.vel, dd.crd, dd.frc,
                                      dd.d_mass_inverse, md_info.dt);
                bussi_thermo.Scale_Velocity(dd.atom_numbers, dd.vel);
            }
            else if (ad_thermo.is_initialized)
            {
                if ((md_info.sys.steps - 1) % ad_thermo.update_interval == 0)
                {
                    ad_thermo.MD_Iteration_Leap_Frog(dd.vel, dd.crd, dd.frc,
                                                     dd.acc, md_info.dt);
                    settle.Project_Velocity_To_Constraint_Manifold(
                        dd.vel, dd.crd, dd.d_mass_inverse, md_info.pbc.cell,
                        md_info.pbc.rcell);
                    shake.Project_Velocity_To_Constraint_Manifold(
                        dd.vel, dd.crd, dd.d_mass_inverse, md_info.pbc.cell,
                        md_info.pbc.rcell, dd.atom_numbers);
                    constrain.v_factor = FLT_MIN;
                    constrain.x_factor = 0.5;
                }
                else
                {
                    md_info.nve.Leap_Frog(dd.atom_numbers, dd.vel, dd.crd,
                                          dd.frc, dd.d_mass_inverse,
                                          md_info.dt);
                    constrain.v_factor = 1.0;
                    constrain.x_factor = 1.0;
                }
            }
            else if (nhc.is_initialized)
            {
                nhc.MD_Iteration_Leap_Frog(dd.vel, dd.crd, dd.frc, dd.acc,
                                           md_info.dt, dd.h_ek_total,
                                           md_info.sys.freedom);
            }

            settle.Do_SETTLE(&controller, dd.atom_local, dd.d_mass, dd.crd,
                             md_info.pbc.cell, md_info.pbc.rcell, dd.vel,
                             md_info.need_pressure, md_info.sys.d_stress);
            shake.Constrain(dd.atom_numbers, dd.crd, dd.vel, dd.d_mass_inverse,
                            dd.d_mass, md_info.pbc.cell, md_info.pbc.rcell,
                            md_info.need_pressure, md_info.sys.d_stress);
            hard_wall.Reflect(dd.atom_numbers, dd.crd, dd.vel);
        }
        if (md_info.need_pressure && !mc_baro.is_initialized)
        {
            md_info.Get_pressure(&controller, dd.atom_numbers, dd.vel,
                                 dd.d_mass, dd.d_virial, main_stream);
            md_info.sys.Get_Density();
            press_baro.Regulate_Pressure(
                md_info.sys.steps, md_info.sys.h_stress, md_info.pbc.cell,
                md_info.dt, md_info.sys.target_pressure,
                md_info.sys.target_temperature);
        }
    }
    else
    {
        md_info.rerun.Iteration();
        if (md_info.rerun.need_box_update)
        {
            Main_Box_Change(md_info.rerun.g, 1, 0, 0);
        }
        md_info.Crd_Vel_Device_to_dd(dd.crd, dd.vel, dd.atom_local_label,
                                     dd.atom_local_id, main_stream);
    }

    if (CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size)
    {
        vatom.Coordinate_Refresh(dd.crd, md_info.pbc.cell, md_info.pbc.rcell);
        if ((md_info.sys.steps + 1) % dd.update_interval == 0 ||
            md_info.mode == md_info.RERUN)
        {
            if (CONTROLLER::PP_MPI_size != 1)
            {
                controller.Get_Time_Recorder("Communication")->Start();
                dd.Exchange_Particles(&controller, &md_info);
                controller.Get_Time_Recorder("Communication")->Stop();
                Main_Refresh_Local_State(false);
            }
            else
            {
                neighbor_list.Update(
                    dd.atom_local, dd.atom_numbers, dd.ghost_numbers, dd.crd,
                    md_info.pbc.cell, md_info.pbc.rcell, md_info.sys.steps,
                    neighbor_list.FORCED_UPDATE,
                    md_info.nb.d_excluded_list_start,
                    md_info.nb.d_excluded_list, md_info.nb.d_excluded_numbers);
            }
        }
    }
    if ((md_info.sys.steps + 1) % dd.update_interval == 0 ||
        md_info.mode == md_info.RERUN)
    {
        controller.Get_Time_Recorder("Communication")->Start();
        pm.Get_Atoms(&controller, md_info.crd, md_info.d_charge,
                     dd.atom_numbers, dd.crd, dd.d_charge, dd.atom_local, true,
                     true, true, true);
        controller.Get_Time_Recorder("Communication")->Stop();
    }
    controller.Get_Time_Recorder("Iteration")->Stop();
}

void Main_Print()
{
    if (md_info.output.Check_Mdout_Step())
    {
        md_info.Step_Print(&controller);
        if (!md_info.pbc.pbc)
        {
            CF_NOPBC.Step_Print(&controller);
            LJ_NOPBC.Step_Print(&controller);
            gb.Step_Print(&controller);
        }
        else
        {
            lj.Step_Print(&controller);
            lj_soft.Step_Print(&controller);
            pm.Step_Print(&controller);
            sits.Step_Print(&controller, 1.0f / md_info.sys.target_temperature /
                                             CONSTANT_kB);
        }
        sits_dihedral.Step_Print(&controller, false);
        sits_nb14.Step_Print(&controller, false);
        sits_cmap.Step_Print(&controller, false);

        sw.Step_Print(&controller);
        edip.Step_Print(&controller);
        eam.Step_Print(&controller);
        tersoff.Step_Print(&controller);
        reaxff.Step_Print(&controller, md_info.d_charge,
                          !md_info.output.h5_reaxff_eeq_snapshot_enabled);
        md_info.output.Append_H5_Reaxff_Frame(&controller);
        if (!reaxff.h_eeq_charges.empty())
        {
            md_info.output.Write_H5_Reaxff_Eeq_Charge_Snapshot(
                &controller, reaxff.h_eeq_charges.data(),
                reaxff.h_eeq_charges.size());
        }
        pairwise_force.Step_Print(&controller);
        angle.Step_Print(&controller);
        urey_bradley.Step_Print(&controller);
        bond.Step_Print(&controller);
        cmap.Step_Print(&controller);
        listed_forces.Step_Print(&controller);
        dihedral.Step_Print(&controller);
        improper.Step_Print(&controller);
        nb14.Step_Print(&controller);

        controller.Step_Print("potential", dd.h_sum_ene_total);

        restrain.Step_Print(&controller);
        if (qc.is_initialized)
        {
            qc.Step_Print(&controller);
            md_info.output.Append_H5_Qc_Frame(&controller);
        }
        cv_controller.Step_Print();
        plugin.Mdout_Print();
        steer_cv.Step_Print(&controller);
        restrain_cv.Step_Print(&controller);
        meta.Step_Print(&controller);
        md_info.output.Append_H5_Metadynamics_Scalar_Frame(
            &controller, meta.potential_local, meta.rbias, meta.rct);
        soft_walls.Step_Print(&controller);
        md_info.output.Append_H5_Observable_Frame(&controller);
        md_info.output.Append_H5_Observable_Only_Frame(&controller);
        controller.Print_To_Screen_And_Mdout();
    }

    if (md_info.output.Check_Trajectory_Step())
    {
        md_info.Crd_Vel_dd_to_Device(dd.crd, dd.vel, dd.atom_local_label,
                                     dd.atom_local_id, main_stream);
        if (md_info.pbc.pbc)
        {
            md_info.mol.Molecule_Crd_Map();
            md_info.Crd_Vel_Device_to_dd(dd.crd, dd.vel, dd.atom_local_label,
                                         dd.atom_local_id, main_stream);
        }
        md_info.output.Append_Crd_Traj_File();
        md_info.output.Append_Vel_Traj_File();
        md_info.output.Append_Box_Traj_File();
        if (md_info.output.h5_trajectory_force_enabled)
        {
            md_info.Frc_dd_to_Host(dd.frc, dd.atom_local_label,
                                   dd.atom_local_id, main_stream);
        }
        md_info.output.Append_H5_Trajectory_Frame(&controller);
        meta.Write_Potential();
#ifdef USE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        if (meta.is_initialized)
        {
            md_info.output.Write_H5_Metadynamics_Diagnostic_File(
                &controller, meta.h5_object_name.c_str(), "potential_export",
                meta.write_potential_file_name);
        }
        nhc.Save_Trajectory_File();
        if (nhc.is_initialized)
        {
            md_info.output.Append_H5_Nose_Hoover_Chain_Frame(
                &controller, nhc.h_coordinate, nhc.h_velocity,
                nhc.chain_length);
        }
    }

    if (md_info.output.is_frc_traj && md_info.output.Check_Force_Step())
    {
        md_info.Frc_dd_to_Host(dd.frc, dd.atom_local_label, dd.atom_local_id,
                               main_stream);
        md_info.output.Append_Frc_Traj_File();
    }

    if (md_info.output.Check_Restart_Step())
    {
        const SpongeH5MD::RestartDynamicState h5_dynamic_state =
            Build_H5_Dynamic_Restart_State();
        SpongeH5MD::RestartMetadynamicsState h5_metadynamics_state;
        const SpongeH5MD::RestartMetadynamicsState*
            h5_metadynamics_state_pointer = NULL;
        std::string h5_metadynamics_error;
        if (meta.is_initialized && md_info.output.h5_restart_enabled &&
            CONTROLLER::MPI_rank == 0)
        {
            if (!meta.Export_H5_Restart_State(&h5_metadynamics_state,
                                              &h5_metadynamics_error))
            {
                controller.Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                                              "Main_Print",
                                              h5_metadynamics_error.c_str());
            }
            h5_metadynamics_state_pointer = &h5_metadynamics_state;
        }
        SpongeH5MD::RestartSitsState h5_sits_state;
        const SpongeH5MD::RestartSitsState* h5_sits_state_pointer = NULL;
        std::string h5_sits_error;
        if (sits.is_initialized && md_info.output.h5_restart_enabled &&
            CONTROLLER::MPI_rank == 0)
        {
            if (!sits.Export_H5_Restart_State(&h5_sits_state, &h5_sits_error))
            {
                controller.Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                                              "Main_Print",
                                              h5_sits_error.c_str());
            }
            if (!h5_sits_state.float_states.empty())
            {
                h5_sits_state_pointer = &h5_sits_state;
            }
        }
        std::vector<float> h5_restraint_reference;
        std::string h5_restraint_error;
        if (!restrain.Export_H5_Reference_Coordinates(&h5_restraint_reference,
                                                      &h5_restraint_error))
        {
            controller.Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                                          "Main_Print",
                                          h5_restraint_error.c_str());
        }
        md_info.output.Export_H5_Restart_File(
            &controller, nhc.is_initialized ? nhc.h_coordinate : NULL,
            nhc.is_initialized ? nhc.h_velocity : NULL,
            nhc.is_initialized ? nhc.chain_length : 0, h5_sits_state_pointer,
            meta.is_initialized ? meta.h5_object_name.c_str() : NULL,
            h5_metadynamics_state_pointer,
            meta.is_initialized ? "myhill.log" : NULL,
            meta.is_initialized ? "history.log" : NULL,
            meta.is_initialized ? meta.edge_file_name : NULL,
            meta.is_initialized ? meta.write_potential_file_name : NULL,
            meta.is_initialized ? meta.write_directly_file_name : NULL,
            restrain.is_initialized ? restrain.h5_restraint_name.c_str() : NULL,
            h5_restraint_reference.empty() ? NULL
                                           : h5_restraint_reference.data(),
            restrain.is_initialized
                ? static_cast<std::size_t>(restrain.atom_numbers)
                : 0,
            &cv_controller.protocol_cv_reference, &h5_dynamic_state);
        if (md_info.output.Should_Write_Legacy_Restart(&controller))
        {
            md_info.output.Export_Restart_File();
            nhc.Save_Restart_File();
        }
    }
    md_info.output.Publish_Output(&controller);
}

void Main_Clear()
{
    md_info.output.Finalize_H5_Trajectory(&controller);
    md_info.output.Finalize_H5_Observable(&controller);
    if (CONTROLLER::MPI_rank == 0)
    {
        const double h5_finalize_total_s =
            md_info.output.h5_trajectory_finalize_elapsed_s +
            md_info.output.h5_observable_finalize_elapsed_s +
            md_info.output.h5_restart_finalize_elapsed_s;
        controller.printf(
            "H5 I/O finalize timing: trajectory=%.9f s, observable=%.9f s, "
            "restart=%.9f s, total=%.9f s\n",
            md_info.output.h5_trajectory_finalize_elapsed_s,
            md_info.output.h5_observable_finalize_elapsed_s,
            md_info.output.h5_restart_finalize_elapsed_s, h5_finalize_total_s);
    }

    const std::string h5_output_failure =
        md_info.output.H5_Output_Failure_Summary();
    if (!h5_output_failure.empty())
    {
        controller.Throw_SPONGE_Error(
            spongeErrorValueErrorCommand, "Main_Clear",
            ("H5 output failure isolation: " + h5_output_failure).c_str());
    }

    controller.Final_Time_Summary(
        md_info.sys.steps, md_info.sys.speed_time_factor,
        md_info.sys.speed_unit_name.c_str(), md_info.mode);

    controller.Clear();
}

float Main_Box_Change(LTMatrix3 g, int scale_box, int scale_crd, int scale_vel)
{
    if (scale_box)
    {
        md_info.pbc.Update_Box(g);
    }
    // 放缩坐标与速度
    if (CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size)
    {
        md_info.Scale_Positions_And_Velocities(
            g, scale_crd, scale_vel, dd.crd,
            dd.vel);  // rescale dd进程原子坐标与速度
        restrain.Update_Refcoord_Scaling(&md_info, g, md_info.dt, dd.atom_local,
                                         dd.atom_numbers, dd.atom_local_label,
                                         dd.atom_local_id);
    }

    // 大幅度放缩盒子时，重新初始化相关模块
    if (scale_box && md_info.pbc.Check_Change_Large())
    {
        Main_Box_Change_Largely();
    }
    else  // 更新域分解盒子
    {
        if (CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size)
        {
            dd.Update_Box(g, md_info.dt);
        }
        if (CONTROLLER::PM_MPI_rank < CONTROLLER::PM_MPI_size &&
            CONTROLLER::PM_MPI_rank != -1)
        {
            pm.Update_Box(md_info.pbc.cell, md_info.pbc.rcell, g, md_info.dt);
        }
    }
    return md_info.sys.Get_Volume();
}

void Main_Box_Change_Largely()
{
    controller.printf(
        "Some modules are based on the meshing methods, and it is more "
        "precise "
        "to re-initialize these modules now for a large box change.\n");

    if (CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size)
    {
        md_info.Crd_Vel_dd_to_Device(dd.crd, dd.vel, dd.atom_local_label,
                                     dd.atom_local_id, main_stream);
    }
    neighbor_list.Clear();
    neighbor_list.Initial(&controller, md_info.atom_numbers, md_info.nb.cutoff,
                          md_info.nb.skin, md_info.pbc.cell, md_info.pbc.rcell);
    pm.Clear();
    pm.Initial(&controller, md_info.atom_numbers, md_info.pbc.cell,
               md_info.pbc.rcell, md_info.sys.box_length, md_info.nb.cutoff,
               md_info.no_direct_interaction_virtual_atom_numbers);
    dd.Free_Buffer();
    dd.Domain_Decomposition(&controller, &md_info);
    pm.Domain_Decomposition(&controller, md_info.sys.box_length,
                            dd.dom_dec_split_num);
    pm.Send_Recv_Dom_Dec(&controller);
    pm.Find_Neighbor_Domain(&controller);
    if (CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size)
    {
        Main_Refresh_Local_State(true);
        plugin.Set_Domain_Information(&dd);
    }
    pm.Get_Atoms(&controller, md_info.crd, md_info.d_charge, dd.atom_numbers,
                 dd.crd, dd.d_charge, dd.atom_local, true, true, true, true);
    MPI_Barrier(MPI_COMM_WORLD);
    controller.printf(
        "------------------------------------------------------------------"
        "----"
        "--------------------------------------\n");
}

void Main_Process_Management()
{
    CONTROLLER::PM_MPI_size = pm.PM_MPI_size;
    CONTROLLER::PP_MPI_size =
        (CONTROLLER::MPI_size - CONTROLLER::PM_MPI_size -
         CONTROLLER::CC_MPI_size) <= 0
            ? 1
            : (CONTROLLER::MPI_size - CONTROLLER::PM_MPI_size -
               CONTROLLER::CC_MPI_size);

    if (CONTROLLER::MPI_size == 1)
    {
        CONTROLLER::pp_comm = MPI_COMM_WORLD;
        CONTROLLER::pm_comm = MPI_COMM_WORLD;
        CONTROLLER::PP_MPI_rank = 0;
        dd.pp_rank = 0;
        if (CONTROLLER::PM_MPI_size != 0)
        {
            CONTROLLER::PM_MPI_rank = 0;
            pm.pm_rank = 0;
        }
        else
        {
            CONTROLLER::PM_MPI_rank = -1;
            pm.pm_rank = -1;
        }
    }
    else if (CONTROLLER::PM_MPI_size == 0)
    {
        CONTROLLER::pp_comm = MPI_COMM_WORLD;
        CONTROLLER::PP_MPI_rank = CONTROLLER::MPI_rank;
        dd.pp_rank = CONTROLLER::PP_MPI_rank;
        pm.pm_rank = -1;
#ifdef USE_XCCL
        xcclUniqueId pp_id;
        if (CONTROLLER::PP_MPI_rank == 0)
        {
            xcclGetUniqueId(&pp_id);
        }
        MPI_Bcast(&pp_id, sizeof(pp_id), MPI_BYTE, 0, CONTROLLER::pp_comm);
        xcclCommInitRank(&CONTROLLER::d_pp_comm, CONTROLLER::PP_MPI_size, pp_id,
                         CONTROLLER::PP_MPI_rank);
#else
        CONTROLLER::d_pp_comm = CONTROLLER::pp_comm;
#endif
    }
    else
    {
        if (CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size)
        {
            MPI_Comm_split(MPI_COMM_WORLD, 0, CONTROLLER::MPI_rank,
                           &CONTROLLER::pp_comm);
            MPI_Comm_rank(CONTROLLER::pp_comm, &dd.pp_rank);
            CONTROLLER::PP_MPI_rank = dd.pp_rank;
#ifdef USE_XCCL
            xcclUniqueId pp_id;
            if (CONTROLLER::PP_MPI_rank == 0)
            {
                xcclGetUniqueId(&pp_id);
            }
            MPI_Bcast(&pp_id, sizeof(pp_id), MPI_BYTE, 0, CONTROLLER::pp_comm);
            xcclCommInitRank(&CONTROLLER::d_pp_comm, CONTROLLER::PP_MPI_size,
                             pp_id, CONTROLLER::PP_MPI_rank);
#else
            CONTROLLER::d_pp_comm = CONTROLLER::pp_comm;
#endif
        }
        else
        {
            CONTROLLER::PP_MPI_rank =
                CONTROLLER::PP_MPI_size;  // PP_MPI_rank 设置>=
                                          // PP_MPI_size，表示非PP进程
            MPI_Comm_split(MPI_COMM_WORLD, 1, CONTROLLER::MPI_rank,
                           &CONTROLLER::pm_comm);
            MPI_Comm_rank(CONTROLLER::pm_comm, &pm.pm_rank);
            CONTROLLER::PM_MPI_rank = pm.pm_rank;
#ifdef USE_XCCL
            xcclUniqueId pm_id;
            if (CONTROLLER::PM_MPI_rank == 0)
            {
                xcclGetUniqueId(&pm_id);
            }
            MPI_Bcast(&pm_id, sizeof(pm_id), MPI_BYTE, 0, CONTROLLER::pm_comm);
            xcclCommInitRank(&CONTROLLER::d_pm_comm, CONTROLLER::PM_MPI_size,
                             pm_id, CONTROLLER::PM_MPI_rank);
#else
            CONTROLLER::d_pm_comm = CONTROLLER::pm_comm;
#endif
        }
    }

    controller.printf(
        "MPI process total: MPI_size=%d, PP_MPI_size=%d, PM_MPI_size=%d\n",
        CONTROLLER::MPI_size, CONTROLLER::PP_MPI_size, CONTROLLER::PM_MPI_size);
    controller.MPI_printf(
        "MPI process partition: MPI_rank=%d, PP_MPI_rank=%d, "
        "PM_MPI_rank=%d\n",
        CONTROLLER::MPI_rank, CONTROLLER::PP_MPI_rank, CONTROLLER::PM_MPI_rank);

    if (CONTROLLER::PP_MPI_size > 1)
    {
        md_info.nb.Excluded_List_Reform(md_info.atom_numbers);
    }
    pm.exclude_factor = CONTROLLER::PP_MPI_size == 1 ? 1.0f : 0.5f;

    deviceStreamCreate(&main_stream);
    dd.Create_Stream();
    pm.Create_Stream();

    dd.Domain_Decomposition(&controller, &md_info);
    pm.Domain_Decomposition(&controller, md_info.sys.box_length,
                            dd.dom_dec_split_num);
    pm.Send_Recv_Dom_Dec(&controller);
    pm.Find_Neighbor_Domain(&controller);
}

void Main_MC_Barostat()
{
    if (mc_baro.is_initialized &&
        md_info.sys.steps % mc_baro.update_interval == 0)
    {
        mc_baro.energy_old = dd.h_sum_ene_total;
        if (CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size)
        {
            deviceMemcpy(mc_baro.frc_backup, dd.frc,
                         sizeof(VECTOR) * dd.atom_numbers,
                         deviceMemcpyDeviceToDevice);
            deviceMemcpy(mc_baro.crd_backup, dd.crd,
                         sizeof(VECTOR) * dd.atom_numbers,
                         deviceMemcpyDeviceToDevice);
        }
        mc_baro.Volume_Change_Attempt(md_info.sys.box_length, md_info.dt);
        Main_Box_Change(mc_baro.g, 1, 0, 0);
        if (CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size)
        {
            dd.Res_Crd_Map(mc_baro.g, md_info.dt);
        }

        Main_Calculate_Force();
        dd.Get_Potential(&controller, &md_info);
        mc_baro.energy_new = dd.h_sum_ene_total;
        mc_baro.extra_term = md_info.sys.target_pressure * mc_baro.DeltaV -
                             md_info.ug.ug_numbers * CONSTANT_kB *
                                 md_info.sys.target_temperature *
                                 logf(mc_baro.VDevided);
        if (mc_baro.couple_dimension != mc_baro.NO &&
            mc_baro.couple_dimension != mc_baro.XYZ)
        {
            mc_baro.extra_term -= mc_baro.surface_number *
                                  mc_baro.surface_tension * mc_baro.DeltaS;
        }
        mc_baro.accept_possibility =
            mc_baro.energy_new - mc_baro.energy_old + mc_baro.extra_term;
        mc_baro.accept_possibility =
            expf(-mc_baro.accept_possibility /
                 (CONSTANT_kB * md_info.sys.target_temperature));

        if (!mc_baro.Check_MC_Barostat_Accept())  // 如果不接受
        {
            mc_baro.g = {-mc_baro.g.a11, 0, -mc_baro.g.a22, 0, 0,
                         -mc_baro.g.a33};
            if (CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size)
            {
                deviceMemcpy(dd.frc, mc_baro.frc_backup,
                             sizeof(VECTOR) * dd.atom_numbers,
                             deviceMemcpyDeviceToDevice);
                deviceMemcpy(dd.crd, mc_baro.crd_backup,
                             sizeof(VECTOR) * dd.atom_numbers,
                             deviceMemcpyDeviceToDevice);
            }
            Main_Box_Change(mc_baro.g, 1, 0, 0);
        }
        mc_baro.Delta_Box_Length_Max_Update();
        dd.h_sum_ene_total = mc_baro.energy_old;  // 恢复能量值
    }
}

void Main_Sync_Dynamic_Targets_To_Controllers()
{
    md_info.sys.Update_Targets_By_Schedule(md_info.sys.steps);
    const float target_temperature = md_info.sys.target_temperature;
    bd_thermo.Set_Target_Temperature(target_temperature);
    bussi_thermo.Set_Target_Temperature(target_temperature);
    ad_thermo.Set_Target_Temperature(target_temperature);
    middle_langevin.Set_Target_Temperature(target_temperature);
    nhc.Set_Target_Temperature(target_temperature);
}

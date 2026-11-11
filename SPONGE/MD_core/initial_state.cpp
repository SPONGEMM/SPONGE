#include "initial_state.h"

#include "../PM_force/PM_force.h"
#include "../bias/restrain_cv.h"
#include "../bias/steer.h"
#include "../constrain/constrain.h"
#include "../constrain/matrix_constraint.h"
#include "../constrain/settle.h"
#include "../constrain/shake.h"
#include "../io/input_session.h"
#include "../plugin/plugin.h"
#include "../thermostat/Berendsen_thermostat.h"
#include "../utils/float_classification.hpp"
#include "../virtual_atoms/virtual_atoms.h"
#include "../xponge/load/common.hpp"

namespace SpongeRuntime
{
void InitialStateBuilder::Require_Stage(Stage expected, const char* operation)
{
    if (stage_ != expected)
    {
        bindings_.runtime.controller.Throw_SPONGE_Error(
            spongeErrorSimulationBreakDown, operation,
            "Reason:\n\tInitial state construction phases are out of order\n");
    }
}

bool InitialStateBuilder::Thermostat_Is(const char* name)
{
    auto& controller = bindings_.runtime.controller;
    auto& md_info = bindings_.runtime.md_info;
    return md_info.mode >= md_info.NVT &&
           (controller.Command_Choice("thermostat", name) ||
            controller.Command_Choice("thermostat_mode", name));
}

bool InitialStateBuilder::Barostat_Is(const char* name)
{
    auto& controller = bindings_.runtime.controller;
    auto& md_info = bindings_.runtime.md_info;
    return md_info.mode == md_info.NPT &&
           (controller.Command_Choice("barostat", name) ||
            controller.Command_Choice("barostat_mode", name));
}

void InitialStateBuilder::Load_Base_State()
{
    Require_Stage(Stage::empty, "Load_Base_State");
    auto& controller = bindings_.runtime.controller;
    auto& md_info = bindings_.runtime.md_info;
    auto& cv_controller = bindings_.runtime.cv_controller;
    auto& input_session = bindings_.input;
    input_session.Load_System();
    cv_controller.atom_numbers =
        Xponge::Load_Get_Atom_Numbers(&bindings_.runtime.system);
    cv_controller.Initial(&controller,
                          &md_info.no_direct_interaction_virtual_atom_numbers);
    md_info.Initial(&controller);
    md_info.sys.Update_Targets_By_Schedule(0);
    stage_ = Stage::base;
}

void InitialStateBuilder::Initialize_Dynamics(float (*box_updater)(LTMatrix3,
                                                                   int, int,
                                                                   int))
{
    Require_Stage(Stage::base, "Initialize_Dynamics");
    auto& controller = bindings_.runtime.controller;
    auto& md_info = bindings_.runtime.md_info;
    auto& middle_langevin = bindings_.runtime.middle_langevin;
    auto& ad_thermo = bindings_.runtime.ad_thermo;
    auto& bussi_thermo = bindings_.runtime.bussi_thermo;
    auto& nhc = bindings_.runtime.nhc;
    auto& press_baro = bindings_.runtime.press_baro;
    auto& mc_baro = bindings_.runtime.mc_baro;
    auto& bd_thermo = bindings_.bd_thermo;
    auto& input_session = bindings_.input;
    if (md_info.mode >= md_info.NVT &&
        (!controller.Command_Exist("thermostat") &&
         !controller.Command_Exist("thermostat_mode")))
    {
        controller.Throw_SPONGE_Error(
            spongeErrorMissingCommand, "Main_Initial",
            "Reason:\n\tthermostat is required for NVT or NPT simulations\n");
    }
    if (Thermostat_Is("middle_langevin") || Thermostat_Is("langevin"))
    {
        middle_langevin.Initial(&controller, md_info.atom_numbers,
                                md_info.sys.target_temperature, md_info.h_mass);
    }
    else if (Thermostat_Is("andersen"))
    {
        ad_thermo.Initial(&controller, md_info.sys.target_temperature,
                          md_info.atom_numbers, md_info.sys.dt_in_ps,
                          md_info.h_mass);
    }
    else if (Thermostat_Is("bussi_thermostat"))
    {
        bussi_thermo.Initial(&controller, md_info.sys.target_temperature);
    }
    else if (Thermostat_Is("berendsen_thermostat"))
    {
        bd_thermo.Initial(&controller, md_info.sys.target_temperature);
    }
    else if (Thermostat_Is("nose_hoover_chain"))
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
    if (Barostat_Is("andersen_barostat") || Barostat_Is("bussi_barostat") ||
        Barostat_Is("berendsen_barostat"))
    {
        press_baro.Initial(&controller, md_info.sys.target_pressure,
                           md_info.pbc.boundary.cell, box_updater);
    }
    if (Barostat_Is("monte_carlo_barostat"))
    {
        mc_baro.Initial(&controller, md_info.atom_numbers,
                        md_info.sys.target_pressure, md_info.sys.box_length,
                        md_info.pbc.boundary.cell);
    }

    auto& r = bindings_.runtime;
    r.rattle_nve.Initial(&controller);
    std::string identity = "mode=" + std::to_string(md_info.mode);
    for (const char* key : {"thermostat", "thermostat_mode", "thermostat_tau",
                            "target_temperature"})
        if (controller.Command_Exist(key))
            identity += ";" + std::string(key) + "=" + controller.Command(key);
    r.velocity_bias.Initial(&controller, md_info.atom_numbers, md_info.h_mass,
                            md_info.sys.box_length, md_info.sys.box_angle,
                            md_info.pbc.pbc, identity);
    r.constrained_middle.Initial(&controller, md_info.atom_numbers,
                                 r.velocity_bias.Parameters());
    r.sampling.Initial(&controller, md_info.sys.box_length,
                       md_info.sys.box_angle, md_info.pbc.pbc);
    r.stress_sampler.Initial(&controller, md_info.atom_numbers, r.sampling,
                             md_info.sys.dt_in_ps);
    r.external_force.Initial(&controller, md_info.atom_numbers, md_info.h_mass,
                             md_info.h_charge, md_info.pbc.pbc,
                             md_info.sys.box_length, md_info.sys.box_angle);
    r.momentum_exchange.Initial(
        &controller, md_info.atom_numbers, md_info.h_mass,
        md_info.sys.box_length, md_info.sys.box_angle, md_info.pbc.pbc,
        identity, md_info.sys.dt_in_ps, md_info.sys.Get_Current_Time(false));
    r.energy_accounting.Initial(
        &controller, md_info.atom_numbers, r.external_force.enabled(),
        r.external_force.Configuration() + ";" +
            r.velocity_bias.Configuration(),
        md_info.sys.dt_in_ps, md_info.sys.Get_Current_Time(false),
        r.velocity_bias.enabled() && md_info.mode == md_info.NVT);
    input_session.Restore_Dynamic_State();
    stage_ = Stage::dynamics;
}

void InitialStateBuilder::Build_Constraints_And_Velocities()
{
    Require_Stage(Stage::dynamics, "Build_Constraints_And_Velocities");
    auto& controller = bindings_.runtime.controller;
    auto& md_info = bindings_.runtime.md_info;
    auto& middle_langevin = bindings_.runtime.middle_langevin;
    auto& cv_controller = bindings_.runtime.cv_controller;
    auto& constrain = bindings_.constrain;
    auto& settle = bindings_.settle;
    auto& shake = bindings_.shake;
    auto& matrix_constraint = bindings_.matrix_constraint;
    auto& vatom = bindings_.vatom;
    auto& initial_velocity = initial_velocity_;
    std::vector<SpongeDynamics::DistanceConstraint> stress_pairs;
    if (controller.Command_Exist("constrain_mode"))
    {
        constrain.Initial_List(
            &controller, md_info.sys.connected_distance, md_info.h_mass, NULL,
            md_info.hmr_enabled ? Xponge::system.atoms.mass.data()
                                : md_info.h_mass);
        constrain.Initial_Constrain(&controller, md_info.atom_numbers,
                                    md_info.dt, md_info.sys.box_length,
                                    md_info.h_mass, &md_info.sys.freedom);
        if (bindings_.runtime.sampling.stress ||
            bindings_.runtime.velocity_bias.enabled() ||
            bindings_.runtime.rattle_nve.Selected())
            for (int i = 0; i < constrain.constrain_pair_numbers; ++i)
            {
                const auto& pair = constrain.h_constrain_pair[i];
                stress_pairs.push_back(
                    {pair.atom_i_serial, pair.atom_j_serial, pair.constant_r});
            }
        if (!bindings_.runtime.rattle_nve.Selected())
            settle.Initial(&controller, &constrain, md_info.h_mass);
        if (controller.Command_Choice("constrain_mode", "SHAKE"))
        {
            shake.Initial_SHAKE(&controller, &constrain);
        }
        else if (controller.Command_Choice("constrain_mode", "LINCS") ||
                 controller.Command_Choice("constrain_mode", "CCMA"))
        {
            matrix_constraint.algorithm =
                controller.Command_Choice("constrain_mode", "LINCS")
                    ? MATRIX_CONSTRAINT::Algorithm::LINCS
                    : MATRIX_CONSTRAINT::Algorithm::CCMA;
            matrix_constraint.Initial(&controller, &constrain, md_info.h_mass,
                                      md_info.crd, md_info.pbc.boundary);
        }
        else if (!controller.Command_Choice("constrain_mode", "SETTLE") &&
                 !bindings_.runtime.rattle_nve.Selected())
        {
            controller.Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                                          "Build_Constraints_And_Velocities",
                                          "Unknown constrain_mode; use SETTLE, "
                                          "SHAKE, LINCS, CCMA or RATTLE");
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
        if (bindings_.runtime.velocity_bias.enabled())
            flow_pairs_ = stress_pairs;
        if (bindings_.runtime.rattle_nve.Selected()) nve_pairs_ = stress_pairs;
    }
    if (bindings_.runtime.sampling.stress)
    {
        if (controller.Command_Choice("constrain_mode", "SETTLE", true) &&
            constrain.constrain_pair_numbers != 0)
            controller.Throw_SPONGE_Error(
                spongeErrorValueErrorCommand,
                "Build_Constraints_And_Velocities",
                "constrained GK has distances not handled by SETTLE");
        std::string solver = controller.Command_Exist("constrain_mode")
                                 ? controller.Command("constrain_mode")
                                 : "none";
        const auto settle_counts = settle.Get_Counts();
        const auto shake_settings = shake.Get_Settings();
        solver +=
            "|settle_pairs=" + std::to_string(settle_counts.pairs) +
            "|settle_triangles=" + std::to_string(settle_counts.triangles) +
            "|shake_pairs=" + std::to_string(constrain.constrain_pair_numbers);
        if (constrain.is_initialized)
            solver += constrain.precision == ConstraintPrecision::Double
                          ? "|precision=double"
                          : "|precision=single";
        if (shake_settings.initialized &&
            (shake_settings.iterations <= 0 ||
             !(shake_settings.step_length > 0) ||
             !SpongeFloat::Is_Finite(shake_settings.step_length)))
            controller.Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                                          "Build_Constraints_And_Velocities",
                                          "constrained GK requires positive "
                                          "finite SHAKE iteration settings");
        if (shake_settings.initialized)
            solver +=
                "|iterations=" + std::to_string(shake_settings.iterations) +
                "|step_length=" +
                SpongeDynamics::Precise_Number(shake_settings.step_length) +
                "|tolerance=" +
                SpongeDynamics::Precise_Number(shake_settings.tolerance);
        if (bindings_.runtime.rattle_nve.Selected())
            solver =
                "RATTLE|rattle_pairs=" + std::to_string(stress_pairs.size()) +
                "|precision=canonical_double_native_force_v1|step=BA_full_"
                "drift";
        bindings_.runtime.stress_sampler.Configure_Constraint_Stress(
            md_info.atom_numbers, md_info.h_mass, std::move(stress_pairs),
            md_info.sys.box_length, md_info.sys.box_angle, md_info.dt, solver);
    }
    vatom.Initial(
        &controller, &cv_controller, md_info.atom_numbers,
        md_info.no_direct_interaction_virtual_atom_numbers,
        cv_controller.cv_vatom_name,
        md_info.hmr_enabled ? Xponge::system.atoms.mass.data() : md_info.h_mass,
        &md_info.sys.freedom, &md_info.sys.connectivity);
    vatom.Coordinate_Refresh(md_info.crd, md_info.pbc.boundary);
    initial_velocity.Initial(&controller, &md_info);

    stage_ = Stage::geometry;
}

void InitialStateBuilder::Restore_Protocol_State()
{
    Require_Stage(Stage::geometry, "Restore_Protocol_State");
    auto& controller = bindings_.runtime.controller;
    auto& cv_controller = bindings_.runtime.cv_controller;
    auto& meta = bindings_.runtime.meta;
    auto& steer_cv = bindings_.steer_cv;
    auto& restrain_cv = bindings_.restrain_cv;
    auto& input_session = bindings_.input;
    steer_cv.Initial(&controller, &cv_controller);
    restrain_cv.Initial(&controller, &cv_controller);
    input_session.Prepare_Metadynamics();
    meta.Initial(&controller, &cv_controller);
    input_session.Restore_Protocol_State();
    input_session.Finish_Initialization();

    stage_ = Stage::protocol;
}

void InitialStateBuilder::Build_Coordinate_Derivatives()
{
    Require_Stage(Stage::protocol, "Build_Coordinate_Derivatives");
    auto& controller = bindings_.runtime.controller;
    auto& md_info = bindings_.runtime.md_info;
    auto& constrain = bindings_.constrain;
    auto& settle = bindings_.settle;
    auto& vatom = bindings_.vatom;
    md_info.ug.Initial_Edge(md_info.atom_numbers);
    constrain.update_ug_connectivity(&md_info.ug.connectivity);
    settle.update_ug_connectivity(&md_info.ug.connectivity);
    vatom.update_ug_connectivity(&md_info.ug.connectivity);
    md_info.ug.Read_Update_Group(md_info.atom_numbers);
    md_info.mol.Initial(&controller,
                        bindings_.runtime.rattle_nve.Selected() ||
                            (bindings_.runtime.velocity_bias.enabled() &&
                             controller.Command_Exist("constrain_mode")));
    stage_ = Stage::coordinates;
}

void InitialStateBuilder::Distribute_State(void (*prepare_processes)(),
                                           void (*refresh_local_state)(bool))
{
    Require_Stage(Stage::coordinates, "Distribute_State");
    auto& controller = bindings_.runtime.controller;
    auto& md_info = bindings_.runtime.md_info;
    auto& dd = bindings_.runtime.dd;
    auto& settle = bindings_.settle;
    auto& shake = bindings_.shake;
    auto& matrix_constraint = bindings_.matrix_constraint;
    auto& pm = bindings_.pm;
    auto& plugin = bindings_.plugin;
    auto& initial_velocity = initial_velocity_;
    prepare_processes();

    if (CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size)
    {
        refresh_local_state(true);
        if (!bindings_.runtime.rattle_nve.Selected())
            initial_velocity.Finalize(&controller, &md_info, &dd, &settle,
                                      &shake, &matrix_constraint);
        if (bindings_.runtime.rattle_nve.Selected())
        {
            if (CONTROLLER::MPI_size != 1 || md_info.mode != md_info.NVE ||
                !md_info.pbc.pbc || bindings_.runtime.velocity_bias.enabled())
                controller.Throw_SPONGE_Error(
                    spongeErrorConflictingCommand, "Distribute_State",
                    "RATTLE requires single-rank periodic NVE without flow");
            md_info.Crd_Vel_dd_to_Device(dd.crd, dd.vel, dd.atom_local_label,
                                         dd.atom_local_id,
                                         bindings_.runtime.main_stream);
            bindings_.runtime.rattle_nve.Configure(
                md_info.atom_numbers, md_info.h_mass, std::move(nve_pairs_),
                md_info.sys.box_length, md_info.sys.box_angle, md_info.dt,
                md_info.sys.dt_in_ps, md_info.crd, md_info.vel,
                initial_velocity.is_initialized
                    ? double(CONSTANT_kB) * md_info.sys.target_temperature
                    : 0);
            if (initial_velocity.is_initialized)
                initial_velocity.is_finalized = true;
            md_info.Crd_Vel_Device_to_dd(dd.crd, dd.vel, dd.atom_local_label,
                                         dd.atom_local_id,
                                         bindings_.runtime.main_stream);
            refresh_local_state(false);
        }
        if (bindings_.runtime.velocity_bias.enabled() &&
            bindings_.runtime.velocity_bias.Parameters().cosine &&
            controller.Command_Exist("constrain_mode"))
        {
            if (CONTROLLER::MPI_size != 1 || md_info.mode != md_info.NVT ||
                !bindings_.runtime.middle_langevin.is_initialized ||
                (!controller.Command_Choice("constrain_mode", "SETTLE") &&
                 !controller.Command_Choice("constrain_mode", "SHAKE")))
                controller.Throw_SPONGE_Error(
                    spongeErrorConflictingCommand, "Distribute_State",
                    "constrained cosine requires single-rank NVT "
                    "middle_langevin and SETTLE/SHAKE distance topology");
            md_info.Crd_Vel_dd_to_Device(dd.crd, dd.vel, dd.atom_local_label,
                                         dd.atom_local_id,
                                         bindings_.runtime.main_stream);
            bindings_.runtime.constrained_middle.Configure_Constraints(
                md_info.h_mass, std::move(flow_pairs_), md_info.sys.box_length,
                md_info.dt, md_info.crd, md_info.vel);
            md_info.Crd_Vel_Device_to_dd(dd.crd, dd.vel, dd.atom_local_label,
                                         dd.atom_local_id,
                                         bindings_.runtime.main_stream);
            refresh_local_state(false);
        }
        plugin.Set_Domain_Information(&dd);
    }

    pm.Get_Atoms(&controller, md_info.crd, md_info.d_charge, dd.atom_numbers,
                 dd.crd, dd.d_charge, dd.atom_local, true, true, true, true);

    stage_ = Stage::distributed;
}

void InitialStateBuilder::Finalize_Run_Range()
{
    Require_Stage(Stage::distributed, "Finalize_Run_Range");
    auto& controller = bindings_.runtime.controller;
    auto& md_info = bindings_.runtime.md_info;
    if (md_info.sys.steps > INT_MAX - md_info.sys.step_limit)
    {
        controller.Throw_SPONGE_Error(
            spongeErrorValueErrorCommand, "main",
            "Reason:\n\trestart step plus step_limit exceeds INT_MAX\n");
    }
    md_info.sys.step_limit += md_info.sys.steps;
    stage_ = Stage::ready;
}

}  // namespace SpongeRuntime

#include "output_session.h"

namespace SpongeIO
{
SpongeH5MD::RestartDynamicState OutputSession::Build_H5_Dynamic_Restart_State()
{
    SpongeH5MD::RestartDynamicState state;
    state.integrator_state_text["mode"] =
        Current_MD_Mode_Name(runtime_.md_info);
    state.integrator_state_text["step"] =
        std::to_string(runtime_.md_info.sys.steps);
    state.integrator_state_text["time"] =
        std::to_string(runtime_.md_info.sys.Get_Current_Time());

    std::string error_message;
    if (!runtime_.bussi_thermo.Export_H5_Restart_State(&state, &error_message))
    {
        runtime_.controller.Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                                               "Build_H5_Dynamic_Restart_State",
                                               error_message.c_str());
    }
    if (!runtime_.middle_langevin.Export_H5_Restart_State(&state,
                                                          &error_message))
    {
        runtime_.controller.Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                                               "Build_H5_Dynamic_Restart_State",
                                               error_message.c_str());
    }
    if (!runtime_.ad_thermo.Export_H5_Restart_State(&state, &error_message))
    {
        runtime_.controller.Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                                               "Build_H5_Dynamic_Restart_State",
                                               error_message.c_str());
    }
    if (!runtime_.press_baro.Export_H5_Restart_State(&state, &error_message))
    {
        runtime_.controller.Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                                               "Build_H5_Dynamic_Restart_State",
                                               error_message.c_str());
    }
    if (!runtime_.mc_baro.Export_H5_Restart_State(&state, &error_message))
    {
        runtime_.controller.Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                                               "Build_H5_Dynamic_Restart_State",
                                               error_message.c_str());
    }
    return state;
}

void OutputSession::Initial()
{
    runtime_.controller.Print_First_Line_To_Mdout();
    runtime_.md_info.output.Initial_H5_Trajectory(&runtime_.controller);
    runtime_.md_info.output.Initial_H5_Observable(&runtime_.controller);
    runtime_.md_info.output.Initial_H5_Restart(&runtime_.controller);
    runtime_.md_info.output.Initial_H5_Nose_Hoover_Chain(
        &runtime_.controller,
        runtime_.nhc.is_initialized ? runtime_.nhc.chain_length : 0);
    runtime_.md_info.output.Initial_H5_Sits_Nk(
        &runtime_.controller,
        runtime_.sits.is_initialized ? runtime_.sits.module_name : NULL,
        runtime_.sits.is_initialized &&
                runtime_.sits.classic_sits.is_initialized
            ? runtime_.sits.classic_sits.k_numbers
            : 0);
    runtime_.md_info.output.Initial_H5_Metadynamics(
        &runtime_.controller, runtime_.meta.is_initialized);
    runtime_.md_info.output.Initial_H5_Reaxff(
        &runtime_.controller, runtime_.reaxff.is_initialized,
        runtime_.reaxff.eeq.is_initialized
            ? static_cast<std::size_t>(runtime_.reaxff.eeq.atom_numbers)
            : static_cast<std::size_t>(0));
    runtime_.md_info.output.Prepare_H5_Swmr_Layout(
        &runtime_.controller, runtime_.meta.is_initialized
                                  ? runtime_.meta.h5_object_name.c_str()
                                  : NULL);
    if (runtime_.meta.is_initialized)
    {
        runtime_.md_info.output.Write_H5_Metadynamics_Diagnostic_File(
            &runtime_.controller, runtime_.meta.h5_object_name.c_str(), "hills",
            "myhill.log");
        runtime_.md_info.output.Write_H5_Metadynamics_Diagnostic_File(
            &runtime_.controller, runtime_.meta.h5_object_name.c_str(),
            "history", "history.log");
        runtime_.md_info.output.Write_H5_Metadynamics_Diagnostic_File(
            &runtime_.controller, runtime_.meta.h5_object_name.c_str(), "edge",
            runtime_.meta.edge_file_name);
        runtime_.md_info.output.Write_H5_Metadynamics_Diagnostic_File(
            &runtime_.controller, runtime_.meta.h5_object_name.c_str(),
            "direct_export", runtime_.meta.write_directly_file_name);
    }
    runtime_.md_info.output.Start_H5_Swmr(&runtime_.controller);
}

bool OutputSession::Observables_Due()
{
    return runtime_.md_info.output.Check_Mdout_Step();
}

bool OutputSession::Trajectory_Due()
{
    return runtime_.md_info.output.Check_Trajectory_Step();
}

bool OutputSession::Force_Due()
{
    return runtime_.md_info.output.is_frc_traj &&
           runtime_.md_info.output.Check_Force_Step();
}

bool OutputSession::Restart_Due()
{
    return runtime_.md_info.output.Check_Restart_Step();
}

void OutputSession::Write_Reaxff()
{
    runtime_.reaxff.Step_Print(
        &runtime_.controller, runtime_.md_info.d_charge,
        !runtime_.md_info.output.h5_reaxff_eeq_snapshot_enabled);
    runtime_.md_info.output.Append_H5_Reaxff_Frame(&runtime_.controller);
    if (!runtime_.reaxff.h_eeq_charges.empty())
    {
        runtime_.md_info.output.Write_H5_Reaxff_Eeq_Charge_Snapshot(
            &runtime_.controller, runtime_.reaxff.h_eeq_charges.data(),
            runtime_.reaxff.h_eeq_charges.size());
    }
}

void OutputSession::Write_Metadynamics_Scalars()
{
    runtime_.md_info.output.Append_H5_Metadynamics_Scalar_Frame(
        &runtime_.controller, runtime_.meta.potential_local,
        runtime_.meta.rbias, runtime_.meta.rct);
}

void OutputSession::Write_Observables()
{
    runtime_.md_info.output.Append_H5_Observable_Frame(&runtime_.controller);
    runtime_.md_info.output.Append_H5_Observable_Only_Frame(
        &runtime_.controller);
    runtime_.controller.Print_To_Screen_And_Mdout();
}

void OutputSession::Write_Trajectory()
{
    runtime_.md_info.Crd_Vel_dd_to_Device(
        runtime_.dd.crd, runtime_.dd.vel, runtime_.dd.atom_local_label,
        runtime_.dd.atom_local_id, runtime_.main_stream);
    if (runtime_.md_info.pbc.pbc)
    {
        runtime_.md_info.mol.Molecule_Crd_Map();
        runtime_.md_info.Crd_Vel_Device_to_dd(
            runtime_.dd.crd, runtime_.dd.vel, runtime_.dd.atom_local_label,
            runtime_.dd.atom_local_id, runtime_.main_stream);
    }
    runtime_.md_info.output.Append_Crd_Traj_File();
    runtime_.md_info.output.Append_Vel_Traj_File();
    runtime_.md_info.output.Append_Box_Traj_File();
    if (runtime_.md_info.output.h5_trajectory_force_enabled)
    {
        runtime_.md_info.Frc_dd_to_Host(
            runtime_.dd.frc, runtime_.dd.atom_local_label,
            runtime_.dd.atom_local_id, runtime_.main_stream);
    }
    runtime_.md_info.output.Append_H5_Trajectory_Frame(&runtime_.controller);
    runtime_.meta.Write_Potential();
#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    if (runtime_.meta.is_initialized)
    {
        runtime_.md_info.output.Write_H5_Metadynamics_Diagnostic_File(
            &runtime_.controller, runtime_.meta.h5_object_name.c_str(),
            "potential_export", runtime_.meta.write_potential_file_name);
    }
    runtime_.nhc.Save_Trajectory_File();
    if (runtime_.nhc.is_initialized)
    {
        runtime_.md_info.output.Append_H5_Nose_Hoover_Chain_Frame(
            &runtime_.controller, runtime_.nhc.h_coordinate,
            runtime_.nhc.h_velocity, runtime_.nhc.chain_length);
    }
}

void OutputSession::Write_Force()
{
    runtime_.md_info.Frc_dd_to_Host(
        runtime_.dd.frc, runtime_.dd.atom_local_label,
        runtime_.dd.atom_local_id, runtime_.main_stream);
    runtime_.md_info.output.Append_Frc_Traj_File();
}

void OutputSession::Write_Metadynamics_Hills()
{
    if (runtime_.meta.is_initialized &&
        runtime_.meta.potential_update_interval > 0 &&
        runtime_.md_info.sys.steps % runtime_.meta.potential_update_interval ==
            0)
    {
        runtime_.md_info.output.Write_H5_Metadynamics_Diagnostic_File(
            &runtime_.controller, runtime_.meta.h5_object_name.c_str(), "hills",
            "myhill.log");
    }
}

void OutputSession::Write_Pending_Sits()
{
    if (runtime_.sits.is_initialized &&
        runtime_.sits.classic_sits.h5_nk_pending)
    {
        runtime_.md_info.output.Append_H5_Sits_Nk_Frame(
            &runtime_.controller, runtime_.sits.module_name,
            runtime_.sits.classic_sits.nk_record_cpu,
            runtime_.sits.classic_sits.k_numbers);
        runtime_.sits.classic_sits.h5_nk_pending = 0;
    }
}

void OutputSession::Publish()
{
    runtime_.md_info.output.Publish_Output(&runtime_.controller);
}

void OutputSession::Close()
{
    runtime_.md_info.output.Finalize_H5_Trajectory(&runtime_.controller);
    runtime_.md_info.output.Finalize_H5_Observable(&runtime_.controller);
    if (CONTROLLER::MPI_rank == 0)
    {
        const double h5_finalize_total_s =
            runtime_.md_info.output.h5_trajectory_finalize_elapsed_s +
            runtime_.md_info.output.h5_observable_finalize_elapsed_s +
            runtime_.md_info.output.h5_restart_finalize_elapsed_s;
        runtime_.controller.printf(
            "H5 I/O finalize timing: trajectory=%.9f s, observable=%.9f s, "
            "restart=%.9f s, total=%.9f s\n",
            runtime_.md_info.output.h5_trajectory_finalize_elapsed_s,
            runtime_.md_info.output.h5_observable_finalize_elapsed_s,
            runtime_.md_info.output.h5_restart_finalize_elapsed_s,
            h5_finalize_total_s);
    }

    const std::string h5_output_failure =
        runtime_.md_info.output.H5_Output_Failure_Summary();
    if (!h5_output_failure.empty())
    {
        runtime_.controller.Throw_SPONGE_Error(
            spongeErrorValueErrorCommand, "Main_Clear",
            ("H5 output failure isolation: " + h5_output_failure).c_str());
    }
}

RestartOutputState OutputSession::Capture_Restart_State()
{
    RestartOutputState state;
    state.dynamic = Build_H5_Dynamic_Restart_State();
    auto& controller = runtime_.controller;
    auto& nhc = runtime_.nhc;
    auto& meta = runtime_.meta;
    auto& sits = runtime_.sits;
    auto& restrain = runtime_.restrain;
    if (nhc.is_initialized && nhc.chain_length > 0)
    {
        state.nhc_coordinates.assign(nhc.h_coordinate,
                                     nhc.h_coordinate + nhc.chain_length);
        state.nhc_velocities.assign(nhc.h_velocity,
                                    nhc.h_velocity + nhc.chain_length);
    }
    std::string error;
    if (meta.is_initialized)
    {
        state.metadynamics.emplace();
        if (!meta.Export_H5_Restart_State(&*state.metadynamics, &error))
        {
            controller.Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                                          "Main_Print", error.c_str());
        }
        state.metadynamics_name = meta.h5_object_name;
        state.metad_hills_file = "myhill.log";
        state.metad_history_file = "history.log";
        state.metad_edge_file = meta.edge_file_name;
        state.metad_potential_file = meta.write_potential_file_name;
        state.metad_direct_file = meta.write_directly_file_name;
    }
    if (sits.is_initialized)
    {
        state.sits.emplace();
        if (!sits.Export_H5_Restart_State(&*state.sits, &error))
        {
            controller.Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                                          "Main_Print", error.c_str());
        }
        if (state.sits->float_states.empty()) state.sits.reset();
    }
    if (!restrain.Export_H5_Reference_Coordinates(&state.restraint_reference,
                                                  &error))
    {
        controller.Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                                      "Main_Print", error.c_str());
    }
    if (restrain.is_initialized)
        state.restraint_name = restrain.h5_restraint_name;
    state.cv_references = runtime_.cv_controller.protocol_cv_reference;
    return state;
}

void OutputSession::Write_Restart()
{
    auto& output = runtime_.md_info.output;
    if (output.h5_restart_enabled && CONTROLLER::MPI_rank == 0)
    {
        output.Export_H5_Restart_File(&runtime_.controller,
                                      Capture_Restart_State());
    }
    if (output.Should_Write_Legacy_Restart(&runtime_.controller))
    {
        output.Export_Restart_File();
        runtime_.nhc.Save_Restart_File();
    }
}
}  // namespace SpongeIO

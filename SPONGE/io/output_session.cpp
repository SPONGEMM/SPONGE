#include "output_session.h"

#include <filesystem>
#include <set>

#include "../MD_core/hmr.h"
#include "../utils/h5md/highfive_backend.hpp"

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
        SpongeDynamics::Precise_Number(runtime_.md_info.sys.Get_Current_Time());
    state.coupling_state_text = runtime_.energy_accounting.Export_Restart();
    if (runtime_.velocity_bias.enabled())
        state.coupling_state_text["velocity_bias_configuration"] =
            runtime_.velocity_bias.Configuration();
    if (runtime_.velocity_bias.Constrained())
        state.coupling_state_text["velocity_bias_constraint_configuration"] =
            runtime_.velocity_bias.Constraint_Identity();
    const auto stress = runtime_.stress_sampler.Export_Restart();
    state.coupling_state_text.insert(stress.begin(), stress.end());
    const auto exchange = runtime_.momentum_exchange.Export_Restart();
    state.coupling_state_text.insert(exchange.begin(), exchange.end());
    runtime_.constrained_middle.Export_Constrained_State(&state);
    runtime_.rattle_nve.Export(&state);

    state.integrator_state_text["mass_signature"] = SpongeHMR::Mass_Signature(
        runtime_.md_info.h_mass, runtime_.md_info.atom_numbers);

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
    Initial_Samples();
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

void OutputSession::Check_Samples_IO(bool ok)
{
    if (!ok)
        runtime_.controller.Throw_SPONGE_Error(
            spongeErrorSimulationBreakDown, "OutputSession::Samples",
            sample_writer_->Last_Error().c_str());
}
void OutputSession::Initial_Samples()
{
    if (!runtime_.sampling.Enabled()) return;
    try
    {
        const auto& path = runtime_.sampling.output;
        const auto target =
            std::filesystem::weakly_canonical(std::filesystem::absolute(path));
        if (std::filesystem::exists(path) || std::filesystem::is_symlink(path))
            runtime_.controller.Throw_SPONGE_Error(
                spongeErrorConflictingCommand, "OutputSession::Samples",
                "sample output already exists; choose a new segment path or "
                "a fresh run directory");
        const std::set<std::string> legacy_paths{
            "crd", "vel", "frc", "box", "mdout", "mdinfo", "rst"};
        for (const auto& command : runtime_.controller.commands)
        {
            const bool path_key =
                command.first.find("_in_file") != std::string::npos ||
                (command.first.size() > 5 &&
                 command.first.compare(command.first.size() - 5, 5, "_path") ==
                     0) ||
                legacy_paths.count(command.first);
            if (!path_key || command.second.empty()) continue;
            // Protect any explicitly named input or output from this new
            // stream.
            if (target == std::filesystem::weakly_canonical(
                              std::filesystem::absolute(command.second)))
                runtime_.controller.Throw_SPONGE_Error(
                    spongeErrorConflictingCommand, "OutputSession::Samples",
                    "sample output collides with another input/output path");
        }
        for (const char* reserved : {"mdout.txt", "mdinfo.txt"})
            if (target == std::filesystem::weakly_canonical(
                              std::filesystem::absolute(reserved)))
                runtime_.controller.Throw_SPONGE_Error(
                    spongeErrorConflictingCommand, "OutputSession::Samples",
                    "sample output collides with a default legacy output");
        SpongeH5OutputPlan::ResolvedOutputPlan plan;
        plan.observable.enabled = true;
        plan.observable.path = path;
        sample_backend_.reset(new SpongeH5MD::HighFiveBackend());
        sample_writer_.reset(
            new SpongeH5MD::ObservableH5Writer(sample_backend_.get()));
        Check_Samples_IO(sample_writer_->Open(plan));

        std::map<std::string, std::string> units;
        if (runtime_.sampling.stress)
        {
            for (const char* key : {"pxx", "pyy", "pzz", "pxy", "pxz", "pyz"})
                units[key] = "Pa";
            units["temperature"] = "K";
            units["volume"] = "nm^3";
            for (const char* key : {"com_vx", "com_vy", "com_vz"})
                units[key] = "nm/ps";
            auto x = runtime_.stress_sampler.Constraint_Units();
            units.insert(x.begin(), x.end());
        }
        if (runtime_.sampling.profile)
        {
            auto x = runtime_.profile_sampler.Units();
            units.insert(x.begin(), x.end());
        }
        if (runtime_.momentum_exchange.enabled())
        {
            auto x = runtime_.momentum_exchange.Units();
            units.insert(x.begin(), x.end());
        }
        if (runtime_.external_force.enabled() ||
            runtime_.energy_accounting.Heat_Enabled())
            units["coupling_work"] = "kcal/mol";
        if (runtime_.energy_accounting.Heat_Enabled())
            units["thermostat_heat"] = "kcal/mol";
        std::vector<std::string> names;
        for (const auto& kv : units) names.push_back(kv.first);
        Check_Samples_IO(
            sample_writer_->Define_Observable_Stream(names, names));
        for (const auto& kv : units)
            Check_Samples_IO(
                sample_writer_->Set_Observable_Unit(kv.first, kv.second));
        auto meta = [&](const std::string& key, const std::string& value)
        {
            Check_Samples_IO(
                sample_writer_->Write_Provenance_String(key, value));
        };
        if (runtime_.energy_accounting.Heat_Enabled())
            meta("thermostat_heat_method", "thermal_substep_v1");
        meta("observable_contract", "sponge.observables.physical_state.v1");
        meta("velocity_time_layer",
             runtime_.sampling.centered ? "force_centered" : "stored");
        if (runtime_.sampling.profile && runtime_.nhc.is_initialized)
            meta("profile_velocity_convention", "native_nhc_stored_v1");
        if (runtime_.velocity_bias.Constrained())
        {
            meta("constraint_profile_configuration",
                 runtime_.velocity_bias.Constraint_Identity());
            meta("thermal_dof",
                 std::to_string(
                     runtime_.velocity_bias.Thermal_Degrees_Of_Freedom()));
        }
        meta("sampling_phase", "post_force_pre_exchange_pre_integrate");
        meta("stress_convention",
             "symmetric internal pressure; COM removed; external-body-force "
             "virial excluded; instantaneous constraint reaction when present");
        meta("dt_ps",
             SpongeDynamics::Precise_Number(runtime_.md_info.sys.dt_in_ps));
        meta("sample_interval", std::to_string(runtime_.sampling.interval));
        meta("atom_count", std::to_string(runtime_.md_info.atom_numbers));
        for (const auto& kv : runtime_.stress_sampler.Constraint_Metadata())
            meta(kv.first, kv.second);
        meta("profile_kind",
             runtime_.sampling.profile
                 ? (runtime_.sampling.mode.cosine ? "cosine" : "slab")
                 : "none");
        meta("bins", std::to_string(runtime_.sampling.mode.bins));
        meta("flow", std::to_string(runtime_.sampling.mode.flow));
        meta("gradient", std::to_string(runtime_.sampling.mode.gradient));
        if (runtime_.momentum_exchange.enabled())
            meta(
                "exchange_interval",
                std::to_string(
                    runtime_.momentum_exchange.Parameters().exchange_interval));
        if (runtime_.constrained_middle.Constrained())
        {
            meta("canonical_integrator_configuration",
                 runtime_.constrained_middle.Constraint_Identity());
            meta("state_precision",
                 "float64 canonical q/s; float32 native forces");
        }
        std::string commands;
        for (const auto& kv : runtime_.controller.commands)
            commands += kv.first + "=" + kv.second + "\n";
        meta("effective_commands", commands);
    }
    catch (const std::exception& e)
    {
        runtime_.controller.Throw_SPONGE_Error(
            spongeErrorValueErrorCommand, "OutputSession::Samples", e.what());
    }
}
void OutputSession::Write_Samples()
{
    if (!runtime_.sampling.Due(runtime_.md_info.sys.steps)) return;
    std::map<std::string, double> values;
    const double dt = runtime_.sampling.centered ? runtime_.md_info.dt : 0;
    if (runtime_.sampling.stress)
        values = runtime_.stress_sampler.Sample_Stress(
            runtime_.dd.atom_numbers, runtime_.dd.vel, runtime_.dd.frc,
            runtime_.dd.d_mass, runtime_.dd.d_virial, dt,
            runtime_.md_info.sys.Get_Volume(), runtime_.dd.crd,
            runtime_.dd.atom_local, runtime_.md_info.sys.box_length,
            runtime_.constrained_middle.Constrained()
                ? runtime_.constrained_middle.Positions()
                : runtime_.rattle_nve.Positions(),
            runtime_.constrained_middle.Constrained()
                ? runtime_.constrained_middle.Velocities()
                : runtime_.rattle_nve.Velocities());
    if (runtime_.sampling.profile)
    {
        auto x = runtime_.profile_sampler.Sample(
            runtime_.dd.atom_numbers, runtime_.dd.crd, runtime_.dd.vel,
            runtime_.dd.frc, runtime_.dd.d_mass, dt,
            runtime_.constrained_middle.Constrained()
                ? runtime_.constrained_middle.View()
                : SpongeDynamics::PhysicalStateView{});
        values.insert(x.begin(), x.end());
    }
    if (runtime_.momentum_exchange.enabled())
    {
        auto x = runtime_.momentum_exchange.Values();
        values.insert(x.begin(), x.end());
    }
    if (runtime_.external_force.enabled() ||
        runtime_.energy_accounting.Heat_Enabled())
        values["coupling_work"] = runtime_.energy_accounting.Coupling_Work();
    if (runtime_.energy_accounting.Heat_Enabled())
        values["thermostat_heat"] =
            runtime_.energy_accounting.Thermostat_Heat();
    Check_Samples_IO(sample_writer_->Append_Observable_Frame(
        runtime_.md_info.sys.steps,
        runtime_.md_info.sys.Get_Current_Time(false), values));
    if (sample_writer_->Observable_Frame_Count() % 64 == 0)
        Check_Samples_IO(sample_writer_->Publish());
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

void OutputSession::Prepare_Force_Output()
{
    const auto& output = runtime_.md_info.output;
    // Use the shared plan: H5 writer state is initialized only on rank zero,
    // but gathering force requires every particle rank to participate.
    if (Force_Due() ||
        (Trajectory_Due() && output.h5_output_plan.trajectory.enabled &&
         output.is_frc_traj))
    {
        runtime_.md_info.Frc_dd_to_Host(
            runtime_.dd.frc, runtime_.dd.atom_local_label,
            runtime_.dd.atom_local_id, runtime_.main_stream);
    }
}

void OutputSession::Write_Trajectory()
{
    runtime_.md_info.Crd_Vel_dd_to_Device(
        runtime_.dd.crd, runtime_.dd.vel, runtime_.dd.atom_local_label,
        runtime_.dd.atom_local_id, runtime_.main_stream);
    // Canonical constrained integrators already publish q through double
    // wrapping. Reimaging here would mutate the native state according to
    // output cadence.
    if (runtime_.md_info.pbc.pbc && !runtime_.rattle_nve.Active() &&
        !runtime_.constrained_middle.Constrained())
    {
        runtime_.md_info.mol.Molecule_Crd_Map();
        runtime_.md_info.Crd_Vel_Device_to_dd(
            runtime_.dd.crd, runtime_.dd.vel, runtime_.dd.atom_local_label,
            runtime_.dd.atom_local_id, runtime_.main_stream);
    }
    runtime_.md_info.output.Append_Crd_Traj_File();
    runtime_.md_info.output.Append_Vel_Traj_File();
    runtime_.md_info.output.Append_Box_Traj_File();
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
    if (sample_writer_)
    {
        Check_Samples_IO(sample_writer_->Finalize());
        Check_Samples_IO(sample_writer_->Close());
        sample_writer_.reset();
        sample_backend_.reset();
    }

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

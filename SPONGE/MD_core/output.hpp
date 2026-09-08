#pragma once

#include <cctype>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "utils/h5md/atomic_file_publish.hpp"
#include "utils/h5md/h5_legacy_sidecar_contract.hpp"
#include "utils/h5md/h5_structural_state.hpp"
#include "utils/h5md/highfive_backend.hpp"
#include "utils/h5md/output_route_helpers.hpp"
#include "utils/h5md/protocol_h5_reader.hpp"
#include "utils/h5md/topology_h5_reader.hpp"

namespace
{
std::string Sanitize_H5MD_Name_For_Output(const std::string& name)
{
    return SpongeH5OutputRoute::Sanitize_Output_Name(name);
}

std::vector<std::string> Make_Unique_H5MD_Output_Names(
    const std::vector<std::string>& names)
{
    return SpongeH5OutputRoute::Make_Unique_Output_Names(names);
}

bool Parse_H5MD_Output_Double(const std::string& text, double* value)
{
    return SpongeH5OutputRoute::Parse_Output_Double(text, value);
}

void Fill_H5MD_Box_Edges(MD_INFORMATION* md_info, float box_edges[9])
{
    if (md_info->pbc.pbc)
    {
        const LTMatrix3 cell =
            md_info->mode == md_info->RERUN
                ? md_info->pbc.Get_Cell(md_info->sys.box_length,
                                        md_info->sys.box_angle)
                : md_info->pbc.boundary.cell;
        box_edges[0] = cell.a11;
        box_edges[1] = 0.0f;
        box_edges[2] = 0.0f;
        box_edges[3] = cell.a21;
        box_edges[4] = cell.a22;
        box_edges[5] = 0.0f;
        box_edges[6] = cell.a31;
        box_edges[7] = cell.a32;
        box_edges[8] = cell.a33;
    }
    else
    {
        box_edges[0] = md_info->sys.box_length.x;
        box_edges[1] = 0.0f;
        box_edges[2] = 0.0f;
        box_edges[3] = 0.0f;
        box_edges[4] = md_info->sys.box_length.y;
        box_edges[5] = 0.0f;
        box_edges[6] = 0.0f;
        box_edges[7] = 0.0f;
        box_edges[8] = md_info->sys.box_length.z;
    }
}

bool Read_H5MD_Text_File_If_Present(const char* file_name, std::string* text)
{
    return SpongeH5OutputRoute::Read_Text_File_If_Present(file_name, text);
}

bool Write_H5_Restart_Text_File_If_Present(SpongeH5MD::RestartH5Writer* writer,
                                           const char* module_name,
                                           const char* component,
                                           const char* file_name)
{
    if (writer == NULL || module_name == NULL || component == NULL)
    {
        return true;
    }
    std::string text;
    if (!Read_H5MD_Text_File_If_Present(file_name, &text))
    {
        return true;
    }
    return writer->Write_Metad_State_Text(module_name, component, text);
}

bool Write_H5_Restart_Protocol_Sidecars_If_Present(
    CONTROLLER* controller, SpongeH5MD::RestartH5Writer* writer)
{
    if (controller == NULL || writer == NULL)
    {
        return true;
    }
    for (const auto& key : SpongeH5MD::H5_Protocol_Sidecar_Command_Keys())
    {
        if (!controller->Command_Exist(key.c_str()))
        {
            continue;
        }
        std::string text;
        if (!Read_H5MD_Text_File_If_Present(controller->Command(key.c_str()),
                                            &text))
        {
            continue;
        }
        if (!writer->Write_Protocol_Sidecar_Text(key, text))
        {
            return false;
        }
    }
    return true;
}

template <typename Writer>
bool Write_H5_Topology_Compatibility_If_Present(CONTROLLER* controller,
                                                Writer* writer,
                                                std::string* error)
{
    constexpr const char* input_key = "input_h5_topology_path";
    if (writer == NULL || !controller->Command_Exist(input_key))
    {
        return true;
    }
    SpongeH5MD::TopologyH5Reader reader;
    if (!reader.Open(controller->Command(input_key)))
    {
        if (error != NULL) *error = reader.Last_Error();
        return false;
    }
    SpongeH5InputMetadata::TopologyMetadata metadata;
    if (!reader.Read_Metadata(&metadata))
    {
        if (error != NULL) *error = reader.Last_Error();
        return false;
    }
    if (!writer->Write_Topology_Compatibility(metadata.topology_hash,
                                              metadata.atom_ordering_hash))
    {
        if (error != NULL) *error = writer->Last_Error();
        return false;
    }
    return true;
}

bool Resolve_H5_Restart_Lineage(
    CONTROLLER* controller,
    SpongeH5OutputPlan::RestartH5OutputPlan* restart_plan, std::string* error)
{
    constexpr const char* topology_key = "input_h5_topology_path";
    constexpr const char* protocol_key = "input_h5_protocol_path";
    if (controller == NULL || restart_plan == NULL || !restart_plan->enabled)
    {
        return true;
    }
    if (controller->Command_Exist(topology_key))
    {
        SpongeH5MD::TopologyH5Reader topology_reader;
        SpongeH5InputMetadata::TopologyMetadata topology;
        if (!topology_reader.Open(controller->Command(topology_key)) ||
            !topology_reader.Read_Metadata(&topology))
        {
            if (error != NULL) *error = topology_reader.Last_Error();
            return false;
        }
        if ((!restart_plan->topology_hash.empty() &&
             restart_plan->topology_hash != topology.topology_hash) ||
            (!restart_plan->atom_order_hash.empty() &&
             restart_plan->atom_order_hash != topology.atom_ordering_hash))
        {
            if (error != NULL)
            {
                *error =
                    "explicit restart lineage does not match the input "
                    "H5 topology bundle";
            }
            return false;
        }
        restart_plan->topology_hash = topology.topology_hash;
        restart_plan->atom_order_hash = topology.atom_ordering_hash;
    }
    if (controller->Command_Exist(protocol_key))
    {
        SpongeH5MD::ProtocolH5Reader protocol_reader;
        SpongeH5InputMetadata::ProtocolMetadata protocol;
        if (!protocol_reader.Open(controller->Command(protocol_key)) ||
            !protocol_reader.Read_Metadata(&protocol))
        {
            if (error != NULL) *error = protocol_reader.Last_Error();
            return false;
        }
        if (protocol.topology_hash != restart_plan->topology_hash)
        {
            if (error != NULL)
            {
                *error =
                    "input H5 protocol topology_hash does not match "
                    "restart lineage";
            }
            return false;
        }
        if (!restart_plan->protocol_hash.empty() &&
            restart_plan->protocol_hash != protocol.protocol_hash)
        {
            if (error != NULL)
            {
                *error =
                    "explicit restart protocol_hash does not match the "
                    "input H5 protocol bundle";
            }
            return false;
        }
        restart_plan->protocol_hash = protocol.protocol_hash;
    }
    if (restart_plan->topology_hash.empty() ||
        restart_plan->atom_order_hash.empty())
    {
        if (error != NULL)
        {
            *error =
                "H5 restart output requires canonical topology_hash and "
                "atom_order_hash lineage; bind input_h5_topology_path or set "
                "output_h5_restart_topology_hash and "
                "output_h5_restart_atom_order_hash";
        }
        return false;
    }
    if (!SpongeH5OutputContract::Is_Canonical_Sha256(
            restart_plan->topology_hash) ||
        !SpongeH5OutputContract::Is_Canonical_Sha256(
            restart_plan->atom_order_hash) ||
        (!restart_plan->protocol_hash.empty() &&
         !SpongeH5OutputContract::Is_Canonical_Sha256(
             restart_plan->protocol_hash)))
    {
        if (error != NULL)
        {
            *error =
                "H5 restart lineage hashes must use canonical "
                "sha256:<64 lowercase hex> encoding";
        }
        return false;
    }
    return true;
}

bool Write_H5_Restart_Lineage(
    const SpongeH5OutputPlan::RestartH5OutputPlan& restart_plan,
    SpongeH5MD::RestartH5Writer* writer, std::string* error)
{
    if (writer == NULL)
    {
        return true;
    }
    if (!writer->Write_Lineage(restart_plan.topology_hash,
                               restart_plan.atom_order_hash,
                               restart_plan.protocol_hash))
    {
        if (error != NULL) *error = writer->Last_Error();
        return false;
    }
    return true;
}

bool Write_H5_Restart_Dynamic_State_If_Present(
    SpongeH5MD::RestartH5Writer* writer,
    const SpongeH5MD::RestartDynamicState* state)
{
    if (writer == NULL || state == NULL)
    {
        return true;
    }
    for (const auto& item : state->rng_state_text)
    {
        if (!writer->Write_Rng_State_Text(item.first, item.second))
        {
            return false;
        }
    }
    for (const auto& item : state->rng_states)
    {
        if (!writer->Write_Rng_State(item.first, item.second))
        {
            return false;
        }
    }
    for (const auto& item : state->integrator_state_text)
    {
        if (!writer->Write_Integrator_State_Text(item.first, item.second))
        {
            return false;
        }
    }
    for (const auto& module : state->thermostat_text_states)
    {
        for (const auto& item : module.second)
        {
            if (!writer->Write_Thermostat_State_Text(module.first, item.first,
                                                     item.second))
            {
                return false;
            }
        }
    }
    for (const auto& module : state->thermostat_float_states)
    {
        for (const auto& item : module.second)
        {
            if (!writer->Write_Thermostat_State_Float(module.first, item.first,
                                                      item.second.data(),
                                                      item.second.size()))
            {
                return false;
            }
        }
    }
    for (const auto& module : state->thermostat_integer_states)
    {
        for (const auto& item : module.second)
        {
            if (!writer->Write_Thermostat_State_Int64(module.first, item.first,
                                                      item.second.data(),
                                                      item.second.size()))
            {
                return false;
            }
        }
    }
    for (const auto& module : state->barostat_text_states)
    {
        for (const auto& item : module.second)
        {
            if (!writer->Write_Barostat_State_Text(module.first, item.first,
                                                   item.second))
            {
                return false;
            }
        }
    }
    for (const auto& module : state->barostat_float_states)
    {
        for (const auto& item : module.second)
        {
            if (!writer->Write_Barostat_State_Float(module.first, item.first,
                                                    item.second.data(),
                                                    item.second.size()))
            {
                return false;
            }
        }
    }
    for (const auto& module : state->barostat_integer_states)
    {
        for (const auto& item : module.second)
        {
            if (!writer->Write_Barostat_State_Int64(module.first, item.first,
                                                    item.second.data(),
                                                    item.second.size()))
            {
                return false;
            }
        }
    }
    return true;
}

bool Is_H5MD_Reaxff_Output_Key(const std::string& name)
{
    return SpongeH5OutputRoute::Is_Reaxff_Output_Key(name);
}

bool H5MD_Output_Key_Exists(const CONTROLLER* controller, const char* name)
{
    if (controller == NULL || name == NULL) return false;
    return SpongeH5OutputRoute::Output_Key_Exists(controller->outputs_key,
                                                  name);
}

double H5MD_Elapsed_Seconds(const std::chrono::steady_clock::time_point start)
{
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(stop - start).count();
}
}  // namespace

void MD_INFORMATION::trajectory_output::Record_H5_Output_Failure(
    const char* family, const char* phase, const std::string& reason)
{
    h5_output_failures.push_back(std::string("family=") + family +
                                 " phase=" + phase + " reason=" + reason);
}

std::string MD_INFORMATION::trajectory_output::H5_Output_Failure_Summary() const
{
    std::string summary;
    for (const std::string& failure : h5_output_failures)
    {
        if (!summary.empty()) summary += "; ";
        summary += failure;
    }
    return summary;
}

void MD_INFORMATION::trajectory_output::Initial(CONTROLLER* controller,
                                                MD_INFORMATION* md_info)
{
    this->md_info = md_info;
    current_crd_synchronized_step = -1;
    h5_output_plan = SpongeH5OutputPlan::Resolve_Output_Plan(controller);

    print_virial = false;
    if (controller->Command_Exist("print_pressure"))
    {
        print_virial = controller->Get_Bool(
            "print_pressure", "MD_INFORMATION::trajectory_output::Initial");
    }
    if (print_virial)
    {
        if (!H5MD_Output_Key_Exists(controller, "pressure"))
        {
            controller->Step_Print_Initial("pressure", "%.2f");
        }
        controller->Step_Print_Initial("Pxx", "%.2f");
        controller->Step_Print_Initial("Pyy", "%.2f");
        controller->Step_Print_Initial("Pzz", "%.2f");
        controller->Step_Print_Initial("Pxy", "%.2f");
        controller->Step_Print_Initial("Pxz", "%.2f");
        controller->Step_Print_Initial("Pyz", "%.2f");
    }
    if (md_info->mode != md_info->RERUN)
    {
        int default_interval = 1000;
        if (controller[0].Command_Exist("write_information_interval"))
        {
            controller->Check_Int("write_information_interval",
                                  "MD_INFORMATION::trajectory_output::Initial");
            default_interval =
                atoi(controller[0].Command("write_information_interval"));
        }
        write_trajectory_interval = default_interval;
        if (controller[0].Command_Exist("write_trajectory_interval"))
        {
            controller->Check_Int("write_trajectory_interval",
                                  "MD_INFORMATION::trajectory_output::Initial");
            write_trajectory_interval =
                atoi(controller[0].Command("write_trajectory_interval"));
        }
        write_mdout_interval = default_interval;
        if (controller[0].Command_Exist("write_mdout_interval"))
        {
            controller->Check_Int("write_mdout_interval",
                                  "MD_INFORMATION::trajectory_output::Initial");
            write_mdout_interval =
                atoi(controller[0].Command("write_mdout_interval"));
        }
        write_restart_file_interval = md_info->sys.step_limit;
        if (controller[0].Command_Exist("write_restart_file_interval"))
        {
            controller->Check_Int("write_restart_file_interval",
                                  "MD_INFORMATION::trajectory_output::Initial");
            write_restart_file_interval =
                atoi(controller[0].Command("write_restart_file_interval"));
        }
        if (controller->Command_Exist(RESTART_COMMAND))
        {
            strcpy(restart_name, controller->Command(RESTART_COMMAND));
        }
        else if (controller->Command_Exist("default_out_file_prefix"))
        {
            strcpy(restart_name,
                   controller->Command("default_out_file_prefix"));
        }
        else
        {
            strcpy(restart_name, RESTART_DEFAULT_FILENAME);
        }
        if (controller->Command_Exist(FRC_TRAJ_COMMAND))
        {
            is_frc_traj = 1;
            Open_File_Safely(&frc_traj, controller->Command(FRC_TRAJ_COMMAND),
                             "wb");
            controller->Set_File_Buffer(frc_traj,
                                        sizeof(VECTOR) * md_info->atom_numbers);
        }
        if (controller->Command_Exist(VEL_TRAJ_COMMAND))
        {
            is_vel_traj = 1;
            Open_File_Safely(&vel_traj, controller->Command(VEL_TRAJ_COMMAND),
                             "wb");
            controller->Set_File_Buffer(vel_traj,
                                        sizeof(VECTOR) * md_info->atom_numbers);
        }
        print_zeroth_frame = false;
        if (controller->Command_Exist("print_zeroth_frame"))
        {
            print_zeroth_frame = controller->Get_Bool(
                "print_zeroth_frame",
                "MD_INFORMATION::trajectory_output::Initial");
        }
        if (controller->Command_Exist("max_restart_export_count"))
        {
            controller->Check_Int("max_restart_export_count",
                                  "MD_INFORMATION::trajectory_output::Initial");
            max_restart_export_count =
                atoi(controller->Command("max_restart_export_count"));
        }
    }
    else
    {
        print_zeroth_frame = true;
        write_trajectory_interval = 0;
        if (controller->Command_Exist("write_trajectory_interval"))
        {
            controller->Check_Int("write_trajectory_interval",
                                  "MD_INFORMATION::trajectory_output::Initial");
            write_trajectory_interval =
                atoi(controller->Command("write_trajectory_interval"));
        }
        write_mdout_interval = 1;
        write_restart_file_interval = 0;
        if (controller->Command_Exist(FRC_TRAJ_COMMAND))
        {
            is_frc_traj = 1;
            Open_File_Safely(&frc_traj, controller->Command(FRC_TRAJ_COMMAND),
                             "wb");
            controller->Set_File_Buffer(frc_traj,
                                        sizeof(VECTOR) * md_info->atom_numbers);
        }
        if (controller->Command_Exist("rerun_output_vel"))
        {
            is_vel_traj = 1;
            Open_File_Safely(&vel_traj, controller->Command("rerun_output_vel"),
                             "wb");
            controller->Set_File_Buffer(vel_traj,
                                        sizeof(VECTOR) * md_info->atom_numbers);
        }
    }
    if (write_trajectory_interval != 0 && md_info->mode == md_info->RERUN &&
        controller->Command_Exist("rerun_output_crd"))
    {
        Open_File_Safely(&crd_traj, controller->Command("rerun_output_crd"),
                         "wb");
        controller->Set_File_Buffer(crd_traj,
                                    sizeof(VECTOR) * md_info->atom_numbers);
    }
    else if (write_trajectory_interval != 0 &&
             h5_output_plan.legacy.Enabled(TRAJ_COMMAND))
    {
        crd_traj = controller->Get_Output_File(true, TRAJ_COMMAND, ".dat",
                                               TRAJ_DEFAULT_FILENAME);
        if (crd_traj != NULL)
        {
            controller->Set_File_Buffer(crd_traj,
                                        sizeof(VECTOR) * md_info->atom_numbers);
        }
    }
    if (write_trajectory_interval != 0 && md_info->mode == md_info->RERUN &&
        controller->Command_Exist("rerun_output_box"))
    {
        Open_File_Safely(&box_traj, controller->Command("rerun_output_box"),
                         "w");
    }
    else if (write_trajectory_interval != 0 &&
             h5_output_plan.legacy.Enabled(BOX_TRAJ_COMMAND))
    {
        box_traj = controller->Get_Output_File(false, BOX_TRAJ_COMMAND, ".box",
                                               BOX_TRAJ_DEFAULT_FILENAME);
        char line[256];
        sprintf(line, "%9.3f %9.3f %9.3f %9.5f %9.5f %9.5f\n",
                md_info->sys.box_length.x, md_info->sys.box_length.y,
                md_info->sys.box_length.z, 90.0f, 90.0f, 90.0f);
        if (box_traj != NULL)
        {
            controller->Set_File_Buffer(box_traj, sizeof(char) * strlen(line));
        }
    }
}

void MD_INFORMATION::trajectory_output::Initial_H5_Trajectory(
    CONTROLLER* controller)
{
    h5_output_plan = SpongeH5OutputPlan::Resolve_Output_Plan(controller);
    if (h5_output_plan.any_h5_output_enabled &&
        h5_output_identity_uuid.empty() && CONTROLLER::MPI_rank == 0)
    {
        h5_output_identity_uuid = SpongeH5MD::Generate_Uuid_V4();
    }
    if (!h5_output_plan.trajectory.enabled)
    {
        return;
    }
    if (CONTROLLER::MPI_rank != 0)
    {
        return;
    }
    h5_trajectory_velocity_enabled = is_vel_traj != 0;
    h5_trajectory_force_enabled = is_frc_traj != 0;
    h5_observable_names =
        Make_Unique_H5MD_Output_Names(controller->outputs_key);
    if (h5_output_plan.trajectory.vds)
    {
        h5_vds_backend_factory.reset(new SpongeH5MD::HighFiveBackendFactory());
        h5_vds_trajectory_writer.reset(new SpongeH5MD::VdsTrajectoryH5Writer(
            h5_vds_backend_factory.get()));
        if (!h5_vds_trajectory_writer->Open(h5_output_plan,
                                            SpongeH5MD::kOutputSchemaVersion,
                                            h5_output_identity_uuid))
        {
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand,
                "MD_INFORMATION::trajectory_output::Initial_H5_Trajectory",
                h5_vds_trajectory_writer->Last_Error().c_str());
        }
        std::string topology_compatibility_error;
        if (!Write_H5_Topology_Compatibility_If_Present(
                controller, h5_vds_trajectory_writer.get(),
                &topology_compatibility_error))
        {
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand,
                "MD_INFORMATION::trajectory_output::Initial_H5_Trajectory",
                topology_compatibility_error.c_str());
        }
        if (!h5_vds_trajectory_writer->Define_Particle_Datasets(
                md_info->atom_numbers, h5_trajectory_velocity_enabled,
                h5_trajectory_force_enabled))
        {
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand,
                "MD_INFORMATION::trajectory_output::Initial_H5_Trajectory",
                h5_vds_trajectory_writer->Last_Error().c_str());
        }
        if (!h5_vds_trajectory_writer->Define_Observable_Stream(
                h5_observable_names, controller->outputs_key))
        {
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand,
                "MD_INFORMATION::trajectory_output::Initial_H5_Trajectory",
                h5_vds_trajectory_writer->Last_Error().c_str());
        }
        h5_trajectory_vds_enabled = true;
        h5_trajectory_enabled = true;
        Write_H5_Legacy_Sidecar_Provenance(
            controller, h5_vds_trajectory_writer.get(),
            "MD_INFORMATION::trajectory_output::Initial_H5_Trajectory");
        return;
    }
    h5_trajectory_backend.reset(new SpongeH5MD::HighFiveBackend());
    h5_trajectory_writer.reset(
        new SpongeH5MD::TrajectoryH5Writer(h5_trajectory_backend.get()));
    if (!h5_trajectory_writer->Open_Single_File(
            h5_output_plan, SpongeH5MD::kOutputSchemaVersion,
            h5_output_identity_uuid))
    {
        controller->Throw_SPONGE_Error(
            spongeErrorValueErrorCommand,
            "MD_INFORMATION::trajectory_output::Initial_H5_Trajectory",
            h5_trajectory_writer->Last_Error().c_str());
    }
    std::string topology_compatibility_error;
    if (!Write_H5_Topology_Compatibility_If_Present(
            controller, h5_trajectory_writer.get(),
            &topology_compatibility_error))
    {
        controller->Throw_SPONGE_Error(
            spongeErrorValueErrorCommand,
            "MD_INFORMATION::trajectory_output::Initial_H5_Trajectory",
            topology_compatibility_error.c_str());
    }
    if (!h5_trajectory_writer->Define_Particle_Datasets(
            md_info->atom_numbers, h5_trajectory_velocity_enabled,
            h5_trajectory_force_enabled))
    {
        controller->Throw_SPONGE_Error(
            spongeErrorValueErrorCommand,
            "MD_INFORMATION::trajectory_output::Initial_H5_Trajectory",
            h5_trajectory_writer->Last_Error().c_str());
    }
    if (!h5_trajectory_writer->Define_Observable_Stream(
            h5_observable_names, controller->outputs_key))
    {
        controller->Throw_SPONGE_Error(
            spongeErrorValueErrorCommand,
            "MD_INFORMATION::trajectory_output::Initial_H5_Trajectory",
            h5_trajectory_writer->Last_Error().c_str());
    }
    h5_trajectory_enabled = true;
    Write_H5_Legacy_Sidecar_Provenance(
        controller, h5_trajectory_writer.get(),
        "MD_INFORMATION::trajectory_output::Initial_H5_Trajectory");
}

void MD_INFORMATION::trajectory_output::Initial_H5_Observable(
    CONTROLLER* controller)
{
    if (!h5_output_plan.trajectory.enabled &&
        !h5_output_plan.observable.enabled)
    {
        h5_output_plan = SpongeH5OutputPlan::Resolve_Output_Plan(controller);
    }
    if (!h5_output_plan.observable.enabled)
    {
        return;
    }
    if (CONTROLLER::MPI_rank != 0)
    {
        return;
    }
    h5_observable_backend.reset(new SpongeH5MD::HighFiveBackend());
    h5_observable_writer.reset(
        new SpongeH5MD::ObservableH5Writer(h5_observable_backend.get()));
    if (h5_output_identity_uuid.empty())
    {
        h5_output_identity_uuid = SpongeH5MD::Generate_Uuid_V4();
    }
    if (!h5_observable_writer->Open(h5_output_plan,
                                    SpongeH5MD::kOutputSchemaVersion,
                                    h5_output_identity_uuid))
    {
        controller->Throw_SPONGE_Error(
            spongeErrorValueErrorCommand,
            "MD_INFORMATION::trajectory_output::Initial_H5_Observable",
            h5_observable_writer->Last_Error().c_str());
    }
    h5_observable_only_names =
        Make_Unique_H5MD_Output_Names(controller->outputs_key);
    if (!h5_observable_writer->Define_Observable_Stream(
            h5_observable_only_names, controller->outputs_key))
    {
        controller->Throw_SPONGE_Error(
            spongeErrorValueErrorCommand,
            "MD_INFORMATION::trajectory_output::Initial_H5_Observable",
            h5_observable_writer->Last_Error().c_str());
    }
    h5_observable_enabled = true;
    Write_H5_Legacy_Sidecar_Provenance(
        controller, h5_observable_writer.get(),
        "MD_INFORMATION::trajectory_output::Initial_H5_Observable");
}

void MD_INFORMATION::trajectory_output::Initial_H5_Restart(
    CONTROLLER* controller)
{
    if (!h5_output_plan.trajectory.enabled &&
        !h5_output_plan.observable.enabled && !h5_output_plan.restart.enabled)
    {
        h5_output_plan = SpongeH5OutputPlan::Resolve_Output_Plan(controller);
    }
    h5_restart_enabled = h5_output_plan.restart.enabled;
    h5_restart_generation = 0;
    if (h5_restart_enabled)
    {
        std::string lineage_error;
        if (!Resolve_H5_Restart_Lineage(controller, &h5_output_plan.restart,
                                        &lineage_error))
        {
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand,
                "MD_INFORMATION::trajectory_output::Initial_H5_Restart",
                lineage_error.c_str());
        }
    }
}

void MD_INFORMATION::trajectory_output::Write_H5_Legacy_Sidecar_Provenance(
    CONTROLLER* controller, SpongeH5MD::TrajectoryH5Writer* writer,
    const char* context)
{
    if (writer == NULL) return;
    std::vector<std::string> keys;
    std::vector<std::string> paths;
    SpongeH5OutputPlan::Collect_Explicit_Legacy_Sidecars(h5_output_plan.legacy,
                                                         &keys, &paths);
    if (keys.empty()) return;
    if (!writer->Write_Legacy_Sidecar_Paths(keys, paths))
    {
        controller->Throw_SPONGE_Error(spongeErrorValueErrorCommand, context,
                                       writer->Last_Error().c_str());
    }
}

void MD_INFORMATION::trajectory_output::Write_H5_Legacy_Sidecar_Provenance(
    CONTROLLER* controller, SpongeH5MD::VdsTrajectoryH5Writer* writer,
    const char* context)
{
    if (writer == NULL) return;
    std::vector<std::string> keys;
    std::vector<std::string> paths;
    SpongeH5OutputPlan::Collect_Explicit_Legacy_Sidecars(h5_output_plan.legacy,
                                                         &keys, &paths);
    if (keys.empty()) return;
    if (!writer->Write_Legacy_Sidecar_Paths(keys, paths))
    {
        controller->Throw_SPONGE_Error(spongeErrorValueErrorCommand, context,
                                       writer->Last_Error().c_str());
    }
}

void MD_INFORMATION::trajectory_output::Write_H5_Legacy_Sidecar_Provenance(
    CONTROLLER* controller, SpongeH5MD::ObservableH5Writer* writer,
    const char* context)
{
    if (writer == NULL) return;
    std::vector<std::string> keys;
    std::vector<std::string> paths;
    SpongeH5OutputPlan::Collect_Explicit_Legacy_Sidecars(h5_output_plan.legacy,
                                                         &keys, &paths);
    if (keys.empty()) return;
    if (!writer->Write_Legacy_Sidecar_Paths(keys, paths))
    {
        controller->Throw_SPONGE_Error(spongeErrorValueErrorCommand, context,
                                       writer->Last_Error().c_str());
    }
}

void MD_INFORMATION::trajectory_output::Write_H5_Legacy_Sidecar_Provenance(
    CONTROLLER* controller, SpongeH5MD::RestartH5Writer* writer,
    const char* context)
{
    if (writer == NULL) return;
    std::vector<std::string> keys;
    std::vector<std::string> paths;
    SpongeH5OutputPlan::Collect_Explicit_Legacy_Sidecars(h5_output_plan.legacy,
                                                         &keys, &paths);
    if (keys.empty()) return;
    if (!writer->Write_Legacy_Sidecar_Paths(keys, paths))
    {
        controller->Throw_SPONGE_Error(spongeErrorValueErrorCommand, context,
                                       writer->Last_Error().c_str());
    }
}

void MD_INFORMATION::trajectory_output::Initial_H5_Nose_Hoover_Chain(
    CONTROLLER* controller, std::size_t chain_length)
{
    if (chain_length == 0 || CONTROLLER::MPI_rank != 0)
    {
        return;
    }
    h5_nhc_chain_length = chain_length;
    bool enabled = false;
    if (h5_trajectory_enabled)
    {
        bool ok =
            h5_trajectory_vds_enabled
                ? h5_vds_trajectory_writer
                      ->Ensure_Nose_Hoover_Chain_Observables(chain_length)
                : h5_trajectory_writer->Ensure_Nose_Hoover_Chain_Observables(
                      chain_length);
        if (!ok)
        {
            const std::string error =
                h5_trajectory_vds_enabled
                    ? h5_vds_trajectory_writer->Last_Error()
                    : h5_trajectory_writer->Last_Error();
            controller->Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                                           "MD_INFORMATION::trajectory_output::"
                                           "Initial_H5_Nose_Hoover_Chain",
                                           error.c_str());
        }
        enabled = true;
    }
    if (h5_observable_enabled)
    {
        if (!h5_observable_writer->Ensure_Nose_Hoover_Chain_Observables(
                chain_length))
        {
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand,
                "MD_INFORMATION::trajectory_output::Initial_H5_Nose_Hoover_"
                "Chain",
                h5_observable_writer->Last_Error().c_str());
        }
        enabled = true;
    }
    h5_nhc_observable_enabled = enabled;
}

void MD_INFORMATION::trajectory_output::Initial_H5_Sits_Nk(
    CONTROLLER* controller, const char* module_name, std::size_t k_count)
{
    if (module_name == NULL || k_count == 0 || CONTROLLER::MPI_rank != 0)
    {
        return;
    }
    h5_sits_module_name = module_name;
    h5_sits_k_count = k_count;
    bool enabled = false;
    if (h5_trajectory_enabled)
    {
        bool ok = h5_trajectory_vds_enabled
                      ? h5_vds_trajectory_writer->Ensure_Sits_Nk_Observable(
                            h5_sits_module_name, h5_sits_k_count)
                      : h5_trajectory_writer->Ensure_Sits_Nk_Observable(
                            h5_sits_module_name, h5_sits_k_count);
        if (!ok)
        {
            const std::string error =
                h5_trajectory_vds_enabled
                    ? h5_vds_trajectory_writer->Last_Error()
                    : h5_trajectory_writer->Last_Error();
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand,
                "MD_INFORMATION::trajectory_output::Initial_H5_Sits_Nk",
                error.c_str());
        }
        enabled = true;
    }
    if (h5_observable_enabled)
    {
        if (!h5_observable_writer->Ensure_Sits_Nk_Observable(
                h5_sits_module_name, h5_sits_k_count))
        {
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand,
                "MD_INFORMATION::trajectory_output::Initial_H5_Sits_Nk",
                h5_observable_writer->Last_Error().c_str());
        }
        enabled = true;
    }
    h5_sits_nk_enabled = enabled;
}

void MD_INFORMATION::trajectory_output::Initial_H5_Metadynamics(
    CONTROLLER* controller, int is_initialized)
{
    if (!is_initialized || CONTROLLER::MPI_rank != 0)
    {
        return;
    }
    bool enabled = false;
    if (h5_trajectory_enabled)
    {
        bool ok = h5_trajectory_vds_enabled
                      ? h5_vds_trajectory_writer->Ensure_Metadynamics_Scalars()
                      : h5_trajectory_writer->Ensure_Metadynamics_Scalars();
        if (!ok)
        {
            const std::string error =
                h5_trajectory_vds_enabled
                    ? h5_vds_trajectory_writer->Last_Error()
                    : h5_trajectory_writer->Last_Error();
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand,
                "MD_INFORMATION::trajectory_output::Initial_H5_Metadynamics",
                error.c_str());
        }
        enabled = true;
    }
    if (h5_observable_enabled)
    {
        if (!h5_observable_writer->Ensure_Metadynamics_Scalars())
        {
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand,
                "MD_INFORMATION::trajectory_output::Initial_H5_Metadynamics",
                h5_observable_writer->Last_Error().c_str());
        }
        enabled = true;
    }
    h5_metadynamics_scalar_enabled = enabled;
}

void MD_INFORMATION::trajectory_output::Initial_H5_Qc(CONTROLLER* controller,
                                                      int is_initialized)
{
    if (!is_initialized || CONTROLLER::MPI_rank != 0)
    {
        return;
    }
    if (!H5MD_Output_Key_Exists(controller, "QC"))
    {
        return;
    }
    h5_qc_spin_square_enabled = H5MD_Output_Key_Exists(controller, "QC_S_sq");
    bool enabled = false;
    if (h5_trajectory_enabled)
    {
        bool ok = h5_trajectory_vds_enabled
                      ? h5_vds_trajectory_writer->Ensure_Qc_Observables(
                            h5_qc_spin_square_enabled)
                      : h5_trajectory_writer->Ensure_Qc_Observables(
                            h5_qc_spin_square_enabled);
        if (!ok)
        {
            const std::string error =
                h5_trajectory_vds_enabled
                    ? h5_vds_trajectory_writer->Last_Error()
                    : h5_trajectory_writer->Last_Error();
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand,
                "MD_INFORMATION::trajectory_output::Initial_H5_Qc",
                error.c_str());
        }
        enabled = true;
    }
    if (h5_observable_enabled)
    {
        if (!h5_observable_writer->Ensure_Qc_Observables(
                h5_qc_spin_square_enabled))
        {
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand,
                "MD_INFORMATION::trajectory_output::Initial_H5_Qc",
                h5_observable_writer->Last_Error().c_str());
        }
        enabled = true;
    }
    h5_qc_scalar_enabled = enabled;
}

void MD_INFORMATION::trajectory_output::Initial_H5_Reaxff(
    CONTROLLER* controller, int is_initialized, std::size_t eeq_atom_count)
{
    if (!is_initialized || CONTROLLER::MPI_rank != 0)
    {
        return;
    }
    h5_reaxff_terms.clear();
    for (const std::string& key : controller->outputs_key)
    {
        if (Is_H5MD_Reaxff_Output_Key(key))
        {
            h5_reaxff_terms.push_back(key);
        }
    }
    const bool has_energy_terms = !h5_reaxff_terms.empty();
    const bool has_eeq_snapshot = eeq_atom_count > 0;
    if (!has_energy_terms && !has_eeq_snapshot) return;

    bool energy_enabled = false;
    bool eeq_snapshot_enabled = false;
    if (h5_trajectory_enabled)
    {
        if (has_energy_terms)
        {
            const bool ok =
                h5_trajectory_vds_enabled
                    ? h5_vds_trajectory_writer->Ensure_Reaxff_Energy_Terms(
                          h5_reaxff_terms)
                    : h5_trajectory_writer->Ensure_Reaxff_Energy_Terms(
                          h5_reaxff_terms);
            if (!ok)
            {
                const std::string error =
                    h5_trajectory_vds_enabled
                        ? h5_vds_trajectory_writer->Last_Error()
                        : h5_trajectory_writer->Last_Error();
                controller->Throw_SPONGE_Error(
                    spongeErrorValueErrorCommand,
                    "MD_INFORMATION::trajectory_output::Initial_H5_Reaxff",
                    error.c_str());
            }
            energy_enabled = true;
        }
        if (has_eeq_snapshot)
        {
            const bool ok =
                h5_trajectory_vds_enabled
                    ? h5_vds_trajectory_writer
                          ->Ensure_Reaxff_Eeq_Charge_Snapshot(eeq_atom_count)
                    : h5_trajectory_writer->Ensure_Reaxff_Eeq_Charge_Snapshot(
                          eeq_atom_count);
            if (!ok)
            {
                const std::string error =
                    h5_trajectory_vds_enabled
                        ? h5_vds_trajectory_writer->Last_Error()
                        : h5_trajectory_writer->Last_Error();
                controller->Throw_SPONGE_Error(
                    spongeErrorValueErrorCommand,
                    "MD_INFORMATION::trajectory_output::Initial_H5_Reaxff",
                    error.c_str());
            }
            eeq_snapshot_enabled = true;
        }
    }
    if (h5_observable_enabled)
    {
        if (has_energy_terms &&
            !h5_observable_writer->Ensure_Reaxff_Energy_Terms(h5_reaxff_terms))
        {
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand,
                "MD_INFORMATION::trajectory_output::Initial_H5_Reaxff",
                h5_observable_writer->Last_Error().c_str());
        }
        if (has_energy_terms) energy_enabled = true;
        if (has_eeq_snapshot &&
            !h5_observable_writer->Ensure_Reaxff_Eeq_Charge_Snapshot(
                eeq_atom_count))
        {
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand,
                "MD_INFORMATION::trajectory_output::Initial_H5_Reaxff",
                h5_observable_writer->Last_Error().c_str());
        }
        if (has_eeq_snapshot) eeq_snapshot_enabled = true;
    }
    h5_reaxff_energy_enabled = energy_enabled;
    h5_reaxff_eeq_snapshot_enabled = eeq_snapshot_enabled;
    h5_reaxff_eeq_atom_count =
        eeq_snapshot_enabled ? eeq_atom_count : static_cast<std::size_t>(0);
}

void MD_INFORMATION::trajectory_output::Prepare_H5_Swmr_Layout(
    CONTROLLER* controller, const char* metadynamics_module_name,
    int qc_is_initialized)
{
    if (CONTROLLER::MPI_rank != 0) return;
    const char* components[] = {"hills", "history", "edge", "direct_export",
                                "potential_export"};
    if (metadynamics_module_name != NULL)
    {
        for (const char* component : components)
        {
            if (h5_trajectory_enabled && !h5_trajectory_vds_enabled &&
                !h5_trajectory_writer->Write_Metadynamics_Diagnostic(
                    metadynamics_module_name, component, ""))
            {
                controller->Throw_SPONGE_Error(
                    spongeErrorValueErrorCommand,
                    "MD_INFORMATION::trajectory_output::Prepare_H5_Swmr_Layout",
                    h5_trajectory_writer->Last_Error().c_str());
            }
            if (h5_observable_enabled &&
                !h5_observable_writer->Write_Metadynamics_Diagnostic(
                    metadynamics_module_name, component, ""))
            {
                controller->Throw_SPONGE_Error(
                    spongeErrorValueErrorCommand,
                    "MD_INFORMATION::trajectory_output::Prepare_H5_Swmr_Layout",
                    h5_observable_writer->Last_Error().c_str());
            }
        }
    }
    if (qc_is_initialized)
    {
        if (h5_trajectory_enabled && !h5_trajectory_vds_enabled &&
            !h5_trajectory_writer->Write_Qc_Scf_Output(""))
        {
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand,
                "MD_INFORMATION::trajectory_output::Prepare_H5_Swmr_Layout",
                h5_trajectory_writer->Last_Error().c_str());
        }
        if (h5_observable_enabled &&
            !h5_observable_writer->Write_Qc_Scf_Output(""))
        {
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand,
                "MD_INFORMATION::trajectory_output::Prepare_H5_Swmr_Layout",
                h5_observable_writer->Last_Error().c_str());
        }
    }
}

void MD_INFORMATION::trajectory_output::Start_H5_Swmr(CONTROLLER* controller)
{
    if (CONTROLLER::MPI_rank != 0) return;
    if (h5_trajectory_enabled && !h5_trajectory_vds_enabled &&
        !h5_trajectory_writer->Start_Swmr_Write())
    {
        controller->Throw_SPONGE_Error(
            spongeErrorValueErrorCommand,
            "MD_INFORMATION::trajectory_output::Start_H5_Swmr",
            h5_trajectory_writer->Last_Error().c_str());
    }
    if (h5_observable_enabled && !h5_observable_writer->Start_Swmr_Write())
    {
        controller->Throw_SPONGE_Error(
            spongeErrorValueErrorCommand,
            "MD_INFORMATION::trajectory_output::Start_H5_Swmr",
            h5_observable_writer->Last_Error().c_str());
    }
}

void MD_INFORMATION::trajectory_output::Append_H5_Observable_Frame(
    CONTROLLER* controller)
{
    if (!h5_trajectory_enabled || CONTROLLER::MPI_rank != 0) return;
    std::map<std::string, double> values;
    for (std::size_t i = 0; i < h5_observable_names.size(); ++i)
    {
        const std::string& original_name = controller->outputs_key[i];
        double value = 0.0;
        if (!Parse_H5MD_Output_Double(
                controller->outputs_content[original_name], &value))
        {
            const std::string error =
                "cannot convert mdout value to H5 observable: " + original_name;
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand,
                "MD_INFORMATION::trajectory_output::Append_H5_Observable_Frame",
                error.c_str());
        }
        values[h5_observable_names[i]] = value;
    }
    if (h5_trajectory_vds_enabled)
    {
        if (!h5_vds_trajectory_writer->Append_Observable_Frame(
                md_info->sys.steps, md_info->sys.Get_Current_Time(false),
                values))
        {
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand,
                "MD_INFORMATION::trajectory_output::Append_H5_Observable_Frame",
                h5_vds_trajectory_writer->Last_Error().c_str());
        }
        return;
    }
    if (!h5_trajectory_writer->Append_Observable_Frame(
            md_info->sys.steps, md_info->sys.Get_Current_Time(false), values))
    {
        controller->Throw_SPONGE_Error(
            spongeErrorValueErrorCommand,
            "MD_INFORMATION::trajectory_output::Append_H5_Observable_Frame",
            h5_trajectory_writer->Last_Error().c_str());
    }
}

void MD_INFORMATION::trajectory_output::Append_H5_Observable_Only_Frame(
    CONTROLLER* controller)
{
    if (!h5_observable_enabled || CONTROLLER::MPI_rank != 0) return;
    std::map<std::string, double> values;
    for (std::size_t i = 0; i < h5_observable_only_names.size(); ++i)
    {
        const std::string& original_name = controller->outputs_key[i];
        double value = 0.0;
        if (!Parse_H5MD_Output_Double(
                controller->outputs_content[original_name], &value))
        {
            const std::string error =
                "cannot convert mdout value to observable-only H5: " +
                original_name;
            controller->Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                                           "MD_INFORMATION::trajectory_output::"
                                           "Append_H5_Observable_Only_Frame",
                                           error.c_str());
        }
        values[h5_observable_only_names[i]] = value;
    }
    if (!h5_observable_writer->Append_Observable_Frame(
            md_info->sys.steps, md_info->sys.Get_Current_Time(false), values))
    {
        const std::string reason = h5_observable_writer->Last_Error();
        h5_observable_writer->Close();
        h5_observable_enabled = false;
        Record_H5_Output_Failure("observable", "append", reason);
    }
}

void MD_INFORMATION::trajectory_output::Append_H5_Nose_Hoover_Chain_Frame(
    CONTROLLER* controller, const float* coordinates, const float* velocities,
    std::size_t chain_length)
{
    if (!h5_nhc_observable_enabled || CONTROLLER::MPI_rank != 0) return;
    if (coordinates == NULL || velocities == NULL ||
        chain_length != h5_nhc_chain_length)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorValueErrorCommand,
            "MD_INFORMATION::trajectory_output::Append_H5_Nose_Hoover_Chain_"
            "Frame",
            "invalid Nose-Hoover chain state for H5 output");
    }
    if (h5_trajectory_enabled)
    {
        bool ok =
            h5_trajectory_vds_enabled
                ? h5_vds_trajectory_writer->Append_Nose_Hoover_Chain_Frame(
                      md_info->sys.steps + 1,
                      md_info->sys.Get_Current_Time(false), coordinates,
                      velocities, chain_length)
                : h5_trajectory_writer->Append_Nose_Hoover_Chain_Frame(
                      md_info->sys.steps + 1,
                      md_info->sys.Get_Current_Time(false), coordinates,
                      velocities, chain_length);
        if (!ok)
        {
            const std::string error =
                h5_trajectory_vds_enabled
                    ? h5_vds_trajectory_writer->Last_Error()
                    : h5_trajectory_writer->Last_Error();
            controller->Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                                           "MD_INFORMATION::trajectory_output::"
                                           "Append_H5_Nose_Hoover_Chain_Frame",
                                           error.c_str());
        }
    }
    if (h5_observable_enabled)
    {
        if (!h5_observable_writer->Append_Nose_Hoover_Chain_Frame(
                md_info->sys.steps + 1, md_info->sys.Get_Current_Time(false),
                coordinates, velocities, chain_length))
        {
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand,
                "MD_INFORMATION::trajectory_output::Append_H5_Nose_Hoover_"
                "Chain_Frame",
                h5_observable_writer->Last_Error().c_str());
        }
    }
}

void MD_INFORMATION::trajectory_output::Append_H5_Sits_Nk_Frame(
    CONTROLLER* controller, const char* module_name, const float* values,
    std::size_t k_count)
{
    if (!h5_sits_nk_enabled || CONTROLLER::MPI_rank != 0) return;
    if (module_name == NULL || values == NULL ||
        module_name != h5_sits_module_name || k_count != h5_sits_k_count)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorValueErrorCommand,
            "MD_INFORMATION::trajectory_output::Append_H5_Sits_Nk_Frame",
            "invalid SITS nk state for H5 output");
    }
    if (h5_trajectory_enabled)
    {
        bool ok =
            h5_trajectory_vds_enabled
                ? (h5_vds_trajectory_writer->Total_Trajectory_Frame_Count() == 0
                       ? true
                       : h5_vds_trajectory_writer->Append_Sits_Nk_Frame(
                             md_info->sys.steps,
                             md_info->sys.Get_Current_Time(false),
                             h5_sits_module_name, values, k_count))
                : h5_trajectory_writer->Append_Sits_Nk_Frame(
                      md_info->sys.steps, md_info->sys.Get_Current_Time(false),
                      h5_sits_module_name, values, k_count);
        if (!ok)
        {
            const std::string error =
                h5_trajectory_vds_enabled
                    ? h5_vds_trajectory_writer->Last_Error()
                    : h5_trajectory_writer->Last_Error();
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand,
                "MD_INFORMATION::trajectory_output::Append_H5_Sits_Nk_Frame",
                error.c_str());
        }
    }
    if (h5_observable_enabled)
    {
        if (!h5_observable_writer->Append_Sits_Nk_Frame(
                md_info->sys.steps, md_info->sys.Get_Current_Time(false),
                h5_sits_module_name, values, k_count))
        {
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand,
                "MD_INFORMATION::trajectory_output::Append_H5_Sits_Nk_Frame",
                h5_observable_writer->Last_Error().c_str());
        }
    }
}

void MD_INFORMATION::trajectory_output::Append_H5_Reaxff_Frame(
    CONTROLLER* controller)
{
    if (!h5_reaxff_energy_enabled || CONTROLLER::MPI_rank != 0) return;
    std::map<std::string, double> values;
    for (const std::string& term : h5_reaxff_terms)
    {
        double value = 0.0;
        if (!Parse_H5MD_Output_Double(controller->outputs_content[term],
                                      &value))
        {
            const std::string error =
                "cannot convert ReaxFF mdout value to H5 observable: " + term;
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand,
                "MD_INFORMATION::trajectory_output::Append_H5_Reaxff_Frame",
                error.c_str());
        }
        values[term] = value;
    }
    if (h5_trajectory_enabled)
    {
        bool ok =
            h5_trajectory_vds_enabled
                ? (h5_vds_trajectory_writer->Total_Trajectory_Frame_Count() == 0
                       ? true
                       : h5_vds_trajectory_writer->Append_Reaxff_Frame(
                             md_info->sys.steps,
                             md_info->sys.Get_Current_Time(false), values))
                : h5_trajectory_writer->Append_Reaxff_Frame(
                      md_info->sys.steps, md_info->sys.Get_Current_Time(false),
                      values);
        if (!ok)
        {
            const std::string error =
                h5_trajectory_vds_enabled
                    ? h5_vds_trajectory_writer->Last_Error()
                    : h5_trajectory_writer->Last_Error();
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand,
                "MD_INFORMATION::trajectory_output::Append_H5_Reaxff_Frame",
                error.c_str());
        }
    }
    if (h5_observable_enabled)
    {
        if (!h5_observable_writer->Append_Reaxff_Frame(
                md_info->sys.steps, md_info->sys.Get_Current_Time(false),
                values))
        {
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand,
                "MD_INFORMATION::trajectory_output::Append_H5_Reaxff_Frame",
                h5_observable_writer->Last_Error().c_str());
        }
    }
}

void MD_INFORMATION::trajectory_output::Write_H5_Reaxff_Eeq_Charge_Snapshot(
    CONTROLLER* controller, const float* values, std::size_t atom_count)
{
    if (!h5_reaxff_eeq_snapshot_enabled || CONTROLLER::MPI_rank != 0) return;
    if (values == NULL || atom_count == 0 ||
        atom_count != h5_reaxff_eeq_atom_count)
    {
        controller->Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                                       "MD_INFORMATION::trajectory_output::"
                                       "Write_H5_Reaxff_Eeq_Charge_Snapshot",
                                       "invalid ReaxFF EEQ charge snapshot");
    }
    if (h5_trajectory_enabled)
    {
        const bool ok =
            h5_trajectory_vds_enabled
                ? h5_vds_trajectory_writer->Write_Reaxff_Eeq_Charge_Snapshot(
                      values, atom_count)
                : h5_trajectory_writer->Write_Reaxff_Eeq_Charge_Snapshot(
                      values, atom_count);
        if (!ok)
        {
            const std::string error =
                h5_trajectory_vds_enabled
                    ? h5_vds_trajectory_writer->Last_Error()
                    : h5_trajectory_writer->Last_Error();
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand,
                "MD_INFORMATION::trajectory_output::"
                "Write_H5_Reaxff_Eeq_Charge_Snapshot",
                error.c_str());
        }
    }
    if (h5_observable_enabled &&
        !h5_observable_writer->Write_Reaxff_Eeq_Charge_Snapshot(values,
                                                                atom_count))
    {
        controller->Throw_SPONGE_Error(
            spongeErrorValueErrorCommand,
            "MD_INFORMATION::trajectory_output::"
            "Write_H5_Reaxff_Eeq_Charge_Snapshot",
            h5_observable_writer->Last_Error().c_str());
    }
}

void MD_INFORMATION::trajectory_output::Append_H5_Qc_Frame(
    CONTROLLER* controller)
{
    if (!h5_qc_scalar_enabled || CONTROLLER::MPI_rank != 0) return;
    double energy = 0.0;
    if (!Parse_H5MD_Output_Double(controller->outputs_content["QC"], &energy))
    {
        controller->Throw_SPONGE_Error(
            spongeErrorValueErrorCommand,
            "MD_INFORMATION::trajectory_output::Append_H5_Qc_Frame",
            "cannot convert QC mdout value to H5 observable");
    }
    double spin_square = 0.0;
    const double* spin_square_ptr = NULL;
    if (h5_qc_spin_square_enabled)
    {
        if (!Parse_H5MD_Output_Double(controller->outputs_content["QC_S_sq"],
                                      &spin_square))
        {
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand,
                "MD_INFORMATION::trajectory_output::Append_H5_Qc_Frame",
                "cannot convert QC_S_sq mdout value to H5 observable");
        }
        spin_square_ptr = &spin_square;
    }
    if (h5_trajectory_enabled)
    {
        bool ok =
            h5_trajectory_vds_enabled
                ? (h5_vds_trajectory_writer->Total_Trajectory_Frame_Count() == 0
                       ? true
                       : h5_vds_trajectory_writer->Append_Qc_Frame(
                             md_info->sys.steps,
                             md_info->sys.Get_Current_Time(false), energy,
                             spin_square_ptr))
                : h5_trajectory_writer->Append_Qc_Frame(
                      md_info->sys.steps, md_info->sys.Get_Current_Time(false),
                      energy, spin_square_ptr);
        if (!ok)
        {
            const std::string error =
                h5_trajectory_vds_enabled
                    ? h5_vds_trajectory_writer->Last_Error()
                    : h5_trajectory_writer->Last_Error();
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand,
                "MD_INFORMATION::trajectory_output::Append_H5_Qc_Frame",
                error.c_str());
        }
    }
    if (h5_observable_enabled)
    {
        if (!h5_observable_writer->Append_Qc_Frame(
                md_info->sys.steps, md_info->sys.Get_Current_Time(false),
                energy, spin_square_ptr))
        {
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand,
                "MD_INFORMATION::trajectory_output::Append_H5_Qc_Frame",
                h5_observable_writer->Last_Error().c_str());
        }
    }
}

void MD_INFORMATION::trajectory_output::Append_H5_Metadynamics_Scalar_Frame(
    CONTROLLER* controller, double meta, double rbias, double rct)
{
    if (!h5_metadynamics_scalar_enabled || CONTROLLER::MPI_rank != 0) return;
    if (h5_trajectory_enabled)
    {
        bool ok =
            h5_trajectory_vds_enabled
                ? (h5_vds_trajectory_writer->Total_Trajectory_Frame_Count() == 0
                       ? true
                       : h5_vds_trajectory_writer
                             ->Append_Metadynamics_Scalar_Frame(
                                 md_info->sys.steps,
                                 md_info->sys.Get_Current_Time(false), meta,
                                 rbias, rct))
                : h5_trajectory_writer->Append_Metadynamics_Scalar_Frame(
                      md_info->sys.steps, md_info->sys.Get_Current_Time(false),
                      meta, rbias, rct);
        if (!ok)
        {
            const std::string error =
                h5_trajectory_vds_enabled
                    ? h5_vds_trajectory_writer->Last_Error()
                    : h5_trajectory_writer->Last_Error();
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand,
                "MD_INFORMATION::trajectory_output::Append_H5_Metadynamics_"
                "Scalar_Frame",
                error.c_str());
        }
    }
    if (h5_observable_enabled)
    {
        if (!h5_observable_writer->Append_Metadynamics_Scalar_Frame(
                md_info->sys.steps, md_info->sys.Get_Current_Time(false), meta,
                rbias, rct))
        {
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand,
                "MD_INFORMATION::trajectory_output::Append_H5_Metadynamics_"
                "Scalar_Frame",
                h5_observable_writer->Last_Error().c_str());
        }
    }
}

void MD_INFORMATION::trajectory_output::Write_H5_Metadynamics_Diagnostic_File(
    CONTROLLER* controller, const char* module_name, const char* component,
    const char* file_name)
{
    if ((!h5_trajectory_enabled && !h5_observable_enabled) ||
        CONTROLLER::MPI_rank != 0)
    {
        return;
    }
    if (module_name == NULL || component == NULL)
    {
        return;
    }
    std::string text;
    if (!Read_H5MD_Text_File_If_Present(file_name, &text))
    {
        return;
    }
    if (h5_trajectory_enabled)
    {
        bool ok = h5_trajectory_vds_enabled
                      ? h5_vds_trajectory_writer->Write_Metadynamics_Diagnostic(
                            module_name, component, text)
                      : h5_trajectory_writer->Write_Metadynamics_Diagnostic(
                            module_name, component, text);
        if (!ok)
        {
            const std::string error =
                h5_trajectory_vds_enabled
                    ? h5_vds_trajectory_writer->Last_Error()
                    : h5_trajectory_writer->Last_Error();
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand,
                "MD_INFORMATION::trajectory_output::Write_H5_Metadynamics_"
                "Diagnostic_File",
                error.c_str());
        }
    }
    if (h5_observable_enabled)
    {
        if (!h5_observable_writer->Write_Metadynamics_Diagnostic(
                module_name, component, text))
        {
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand,
                "MD_INFORMATION::trajectory_output::Write_H5_Metadynamics_"
                "Diagnostic_File",
                h5_observable_writer->Last_Error().c_str());
        }
    }
}

void MD_INFORMATION::trajectory_output::Write_H5_Qc_Scf_Output_File(
    CONTROLLER* controller, const char* file_name)
{
    if ((!h5_trajectory_enabled && !h5_observable_enabled) ||
        CONTROLLER::MPI_rank != 0)
    {
        return;
    }
    std::string text;
    if (!Read_H5MD_Text_File_If_Present(file_name, &text))
    {
        return;
    }
    if (h5_trajectory_enabled)
    {
        bool ok = h5_trajectory_vds_enabled
                      ? h5_vds_trajectory_writer->Write_Qc_Scf_Output(text)
                      : h5_trajectory_writer->Write_Qc_Scf_Output(text);
        if (!ok)
        {
            const std::string error =
                h5_trajectory_vds_enabled
                    ? h5_vds_trajectory_writer->Last_Error()
                    : h5_trajectory_writer->Last_Error();
            controller->Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                                           "MD_INFORMATION::trajectory_output::"
                                           "Write_H5_Qc_Scf_Output_File",
                                           error.c_str());
        }
    }
    if (h5_observable_enabled)
    {
        if (!h5_observable_writer->Write_Qc_Scf_Output(text))
        {
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand,
                "MD_INFORMATION::trajectory_output::Write_H5_Qc_Scf_Output_"
                "File",
                h5_observable_writer->Last_Error().c_str());
        }
    }
}

void MD_INFORMATION::trajectory_output::Append_H5_Trajectory_Frame(
    CONTROLLER* controller)
{
    (void)controller;
    if (!h5_trajectory_enabled || CONTROLLER::MPI_rank != 0) return;
    float box_edges[9];
    Fill_H5MD_Box_Edges(md_info, box_edges);
    const float* velocity =
        h5_trajectory_velocity_enabled ? &md_info->velocity[0].x : NULL;
    const float* force = NULL;
    if (h5_trajectory_force_enabled)
    {
        force = &md_info->force[0].x;
    }
    if (h5_trajectory_vds_enabled)
    {
        if (!h5_vds_trajectory_writer->Append_Particle_Frame(
                md_info->sys.steps + 1, md_info->sys.Get_Current_Time(false),
                &md_info->coordinate[0].x, box_edges, velocity, force))
        {
            const std::string reason = h5_vds_trajectory_writer->Last_Error();
            Record_H5_Output_Failure("trajectory", "append", reason);
            h5_vds_trajectory_writer.reset();
            h5_vds_backend_factory.reset();
            h5_trajectory_vds_enabled = false;
            h5_trajectory_enabled = false;
        }
        return;
    }
    if (!h5_trajectory_writer->Append_Particle_Frame(
            md_info->sys.steps + 1, md_info->sys.Get_Current_Time(false),
            &md_info->coordinate[0].x, box_edges, velocity, force))
    {
        const std::string reason = h5_trajectory_writer->Last_Error();
        h5_trajectory_writer->Close();
        h5_trajectory_enabled = false;
        Record_H5_Output_Failure("trajectory", "append", reason);
    }
}

void MD_INFORMATION::trajectory_output::Publish_H5_Output(
    CONTROLLER* controller)
{
    if (CONTROLLER::MPI_rank != 0) return;
    if (h5_trajectory_enabled && !h5_trajectory_vds_enabled &&
        !h5_trajectory_writer->Publish())
    {
        const std::string reason = h5_trajectory_writer->Last_Error();
        h5_trajectory_writer->Close();
        h5_trajectory_enabled = false;
        Record_H5_Output_Failure("trajectory", "publish", reason);
    }
    if (h5_observable_enabled && !h5_observable_writer->Publish())
    {
        const std::string reason = h5_observable_writer->Last_Error();
        h5_observable_writer->Close();
        h5_observable_enabled = false;
        Record_H5_Output_Failure("observable", "publish", reason);
    }
    (void)controller;
}

void MD_INFORMATION::trajectory_output::Publish_Output(CONTROLLER* controller)
{
    Publish_H5_Output(controller);
    std::string error_message;
    if (!SpongeLegacyIO::OutputFlushCoordinator::Flush_Dirty(&error_message))
    {
        controller->Throw_SPONGE_Error(
            spongeErrorValueErrorCommand,
            "MD_INFORMATION::trajectory_output::Publish_Output",
            error_message.c_str());
    }
}

void MD_INFORMATION::trajectory_output::Finalize_H5_Trajectory(
    CONTROLLER* controller)
{
    (void)controller;
    if (!h5_trajectory_enabled || CONTROLLER::MPI_rank != 0) return;
    if (h5_trajectory_vds_enabled)
    {
        const auto finalize_start = std::chrono::steady_clock::now();
        const bool finalized =
            h5_output_plan.trajectory.allow_complete_prefix_repair
                ? h5_vds_trajectory_writer->Finalize_With_Repair()
                : h5_vds_trajectory_writer->Finalize();
        h5_trajectory_finalize_elapsed_s +=
            H5MD_Elapsed_Seconds(finalize_start);
        if (!finalized)
        {
            Record_H5_Output_Failure("trajectory", "finalize",
                                     h5_vds_trajectory_writer->Last_Error());
        }
        h5_vds_trajectory_writer.reset();
        h5_vds_backend_factory.reset();
        h5_trajectory_vds_enabled = false;
        h5_trajectory_enabled = false;
        return;
    }
    const auto finalize_start = std::chrono::steady_clock::now();
    if (!h5_trajectory_writer->Finalize())
    {
        h5_trajectory_finalize_elapsed_s +=
            H5MD_Elapsed_Seconds(finalize_start);
        Record_H5_Output_Failure("trajectory", "finalize",
                                 h5_trajectory_writer->Last_Error());
    }
    else
    {
        h5_trajectory_finalize_elapsed_s +=
            H5MD_Elapsed_Seconds(finalize_start);
    }
    h5_trajectory_writer->Close();
    h5_trajectory_enabled = false;
}

void MD_INFORMATION::trajectory_output::Finalize_H5_Observable(
    CONTROLLER* controller)
{
    (void)controller;
    if (!h5_observable_enabled || CONTROLLER::MPI_rank != 0) return;
    const auto finalize_start = std::chrono::steady_clock::now();
    if (!h5_observable_writer->Finalize())
    {
        h5_observable_finalize_elapsed_s +=
            H5MD_Elapsed_Seconds(finalize_start);
        Record_H5_Output_Failure("observable", "finalize",
                                 h5_observable_writer->Last_Error());
    }
    else
    {
        h5_observable_finalize_elapsed_s +=
            H5MD_Elapsed_Seconds(finalize_start);
    }
    h5_observable_writer->Close();
    h5_observable_enabled = false;
}

void MD_INFORMATION::trajectory_output::Append_Crd_Traj_File(FILE* fp)
{
    if (md_info->is_initialized && CONTROLLER::MPI_rank == 0)
    {
        md_info->Crd_Vel_Device_To_Host();
        if (fp == NULL)
        {
            fp = crd_traj;
        }
        if (fp != NULL)
        {
            fwrite(&md_info->coordinate[0].x, sizeof(VECTOR),
                   md_info->atom_numbers, fp);
            SpongeLegacyIO::OutputFlushCoordinator::Mark_Dirty(
                fp, "coordinate trajectory");
        }
    }
}

// 20210827用于输出速度和力
void MD_INFORMATION::trajectory_output::Append_Frc_Traj_File(FILE* fp)
{
    if (md_info->is_initialized && CONTROLLER::MPI_rank == 0)
    {
        deviceMemcpy(md_info->force, md_info->frc,
                     sizeof(VECTOR) * md_info->atom_numbers,
                     deviceMemcpyDeviceToHost);
        if (fp == NULL)  // 默认的frc输出位置
            fp = frc_traj;
        if (fp != NULL)
        {
            fwrite(&md_info->force[0].x, sizeof(VECTOR), md_info->atom_numbers,
                   fp);
            SpongeLegacyIO::OutputFlushCoordinator::Mark_Dirty(
                fp, "force trajectory");
        }
    }
}
void MD_INFORMATION::trajectory_output::Append_Vel_Traj_File(FILE* fp)
{
    if (md_info->is_initialized && CONTROLLER::MPI_rank == 0)
    {
        deviceMemcpy(md_info->velocity, md_info->vel,
                     sizeof(VECTOR) * md_info->atom_numbers,
                     deviceMemcpyDeviceToHost);
        if (fp == NULL)  // 默认的vel输出位置
        {
            fp = vel_traj;
            if (fp != NULL)
            {
                fwrite(&md_info->velocity[0].x, sizeof(VECTOR),
                       md_info->atom_numbers, fp);
                SpongeLegacyIO::OutputFlushCoordinator::Mark_Dirty(
                    fp, "velocity trajectory");
            }
        }
        else
        {
            fwrite(&md_info->velocity[0].x, sizeof(VECTOR),
                   md_info->atom_numbers, fp);
            SpongeLegacyIO::OutputFlushCoordinator::Mark_Dirty(
                fp, "velocity trajectory");
        }
    }
}

void MD_INFORMATION::trajectory_output::Append_Box_Traj_File(FILE* fp)
{
    if (md_info->is_initialized && CONTROLLER::MPI_rank == 0)
    {
        if (fp == NULL)
        {
            fp = box_traj;
        }
        if (fp != NULL)
        {
            fprintf(fp, "%9.6f %9.6f %9.6f %9.5f %9.5f %9.5f\n",
                    md_info->sys.box_length.x, md_info->sys.box_length.y,
                    md_info->sys.box_length.z, md_info->sys.box_angle.x,
                    md_info->sys.box_angle.y, md_info->sys.box_angle.z);
            SpongeLegacyIO::OutputFlushCoordinator::Mark_Dirty(
                fp, "box trajectory");
        }
    }
}

bool MD_INFORMATION::trajectory_output::Should_Write_Legacy_Restart(
    CONTROLLER* controller)
{
    return SpongeH5OutputContract::Legacy_Sidecar_Enabled(controller,
                                                          RESTART_COMMAND);
}

void MD_INFORMATION::trajectory_output::Export_H5_Restart_File(
    CONTROLLER* controller, const float* nhc_coordinates,
    const float* nhc_velocities, std::size_t nhc_chain_length,
    const SpongeH5MD::RestartSitsState* sits_state,
    const char* metad_module_name,
    const SpongeH5MD::RestartMetadynamicsState* metad_state,
    const char* metad_hills_file_name, const char* metad_history_file_name,
    const char* metad_edge_file_name, const char* metad_potential_file_name,
    const char* metad_direct_file_name, const char* restraint_name,
    const float* restraint_reference_coordinates,
    std::size_t restraint_atom_count,
    const std::map<std::string, std::vector<float>>* cv_references,
    const SpongeH5MD::RestartDynamicState* dynamic_state)
{
    if (!h5_restart_enabled || !md_info->is_initialized || CONTROLLER::MPI_rank)
    {
        return;
    }
    md_info->Crd_Vel_Device_To_Host();
    float box_edges[9];
    Fill_H5MD_Box_Edges(md_info, box_edges);
    const std::string restart_path = h5_output_plan.restart.path;
    const std::string temporary_path = restart_path + ".tmp";
    std::string publish_error;
    if (!SpongeH5MD::Remove_File_If_Exists(temporary_path, &publish_error))
    {
        h5_restart_enabled = false;
        Record_H5_Output_Failure("restart", "prepare", publish_error);
        return;
    }
    SpongeH5MD::TemporaryFileGuard temporary_file_guard(temporary_path);
    SpongeH5OutputPlan::ResolvedOutputPlan temporary_plan = h5_output_plan;
    temporary_plan.restart.path = temporary_path;
    const int64_t candidate_generation = h5_restart_generation + 1;
    SpongeH5MD::HighFiveBackend backend;
    SpongeH5MD::RestartH5Writer writer(&backend);
    if (!writer.Open(temporary_plan, SpongeH5MD::kInputSchemaVersion,
                     h5_output_identity_uuid))
    {
        controller->Throw_SPONGE_Error(
            spongeErrorValueErrorCommand,
            "MD_INFORMATION::trajectory_output::Export_H5_Restart_File",
            writer.Last_Error().c_str());
    }
    std::string lineage_error;
    if (!Write_H5_Restart_Lineage(h5_output_plan.restart, &writer,
                                  &lineage_error))
    {
        controller->Throw_SPONGE_Error(
            spongeErrorValueErrorCommand,
            "MD_INFORMATION::trajectory_output::Export_H5_Restart_File",
            lineage_error.c_str());
    }
    if (!writer.Write_Restart_Generation(candidate_generation))
    {
        controller->Throw_SPONGE_Error(
            spongeErrorValueErrorCommand,
            "MD_INFORMATION::trajectory_output::Export_H5_Restart_File",
            writer.Last_Error().c_str());
    }
    Write_H5_Legacy_Sidecar_Provenance(
        controller, &writer,
        "MD_INFORMATION::trajectory_output::Export_H5_Restart_File");
    if (!writer.Define_Structural_State(md_info->atom_numbers, true))
    {
        controller->Throw_SPONGE_Error(
            spongeErrorValueErrorCommand,
            "MD_INFORMATION::trajectory_output::Export_H5_Restart_File",
            writer.Last_Error().c_str());
    }
    if (!writer.Write_Structural_State(
            md_info->sys.steps, md_info->sys.Get_Current_Time(),
            &md_info->coordinate[0].x, box_edges, &md_info->velocity[0].x))
    {
        const std::string reason = writer.Last_Error();
        writer.Close();
        h5_restart_enabled = false;
        Record_H5_Output_Failure("restart", "append", reason);
        return;
    }
    if (nhc_coordinates != NULL && nhc_velocities != NULL &&
        nhc_chain_length != 0)
    {
        std::vector<float> nhc_pairs(nhc_chain_length * 2);
        for (std::size_t i = 0; i < nhc_chain_length; ++i)
        {
            nhc_pairs[2 * i] = nhc_coordinates[i];
            nhc_pairs[2 * i + 1] = nhc_velocities[i];
        }
        if (!writer.Write_Nose_Hoover_Chain_State(nhc_pairs.data(),
                                                  nhc_chain_length))
        {
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand,
                "MD_INFORMATION::trajectory_output::Export_H5_Restart_File",
                writer.Last_Error().c_str());
        }
    }
    if (!Write_H5_Restart_Dynamic_State_If_Present(&writer, dynamic_state))
    {
        controller->Throw_SPONGE_Error(
            spongeErrorValueErrorCommand,
            "MD_INFORMATION::trajectory_output::Export_H5_Restart_File",
            writer.Last_Error().c_str());
    }
    if (sits_state != NULL && !sits_state->module_name.empty())
    {
        if (!writer.Write_Sits_State_Schema_Version(
                sits_state->module_name, sits_state->state_schema_version))
        {
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand,
                "MD_INFORMATION::trajectory_output::Export_H5_Restart_File",
                writer.Last_Error().c_str());
        }
        for (const auto& state_entry : sits_state->float_states)
        {
            if (state_entry.second.empty() ||
                !writer.Write_Sits_State(
                    sits_state->module_name, state_entry.first,
                    state_entry.second.data(), state_entry.second.size()))
            {
                controller->Throw_SPONGE_Error(
                    spongeErrorValueErrorCommand,
                    "MD_INFORMATION::trajectory_output::Export_H5_Restart_File",
                    writer.Last_Error().c_str());
            }
        }
    }
    if (restraint_name != NULL && restraint_reference_coordinates != NULL &&
        restraint_atom_count != 0 &&
        !writer.Write_Restraint_Reference(restraint_name,
                                          restraint_reference_coordinates,
                                          restraint_atom_count))
    {
        controller->Throw_SPONGE_Error(
            spongeErrorValueErrorCommand,
            "MD_INFORMATION::trajectory_output::Export_H5_Restart_File",
            writer.Last_Error().c_str());
    }
    if (cv_references != NULL)
    {
        for (const auto& reference : *cv_references)
        {
            if (reference.second.empty() || reference.second.size() % 3 != 0)
            {
                controller->Throw_SPONGE_Error(
                    spongeErrorValueErrorCommand,
                    "MD_INFORMATION::trajectory_output::"
                    "Export_H5_Restart_File",
                    "CV reference coordinate count must be a positive "
                    "multiple of three");
            }
            if (!writer.Write_CV_Reference(reference.first,
                                           reference.second.data(),
                                           reference.second.size() / 3))
            {
                controller->Throw_SPONGE_Error(
                    spongeErrorValueErrorCommand,
                    "MD_INFORMATION::trajectory_output::"
                    "Export_H5_Restart_File",
                    writer.Last_Error().c_str());
            }
        }
    }
    if (!Write_H5_Restart_Protocol_Sidecars_If_Present(controller, &writer))
    {
        controller->Throw_SPONGE_Error(
            spongeErrorValueErrorCommand,
            "MD_INFORMATION::trajectory_output::Export_H5_Restart_File",
            writer.Last_Error().c_str());
    }
    if (metad_state != NULL && !writer.Write_Metadynamics_State(*metad_state))
    {
        controller->Throw_SPONGE_Error(
            spongeErrorValueErrorCommand,
            "MD_INFORMATION::trajectory_output::Export_H5_Restart_File",
            writer.Last_Error().c_str());
    }
    if (metad_module_name != NULL && metad_state == NULL)
    {
        const bool ok =
            Write_H5_Restart_Text_File_If_Present(
                &writer, metad_module_name, "hills", metad_hills_file_name) &&
            Write_H5_Restart_Text_File_If_Present(&writer, metad_module_name,
                                                  "history",
                                                  metad_history_file_name) &&
            Write_H5_Restart_Text_File_If_Present(
                &writer, metad_module_name, "edge", metad_edge_file_name) &&
            Write_H5_Restart_Text_File_If_Present(&writer, metad_module_name,
                                                  "potential_export",
                                                  metad_potential_file_name) &&
            Write_H5_Restart_Text_File_If_Present(&writer, metad_module_name,
                                                  "direct_export",
                                                  metad_direct_file_name);
        if (!ok)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand,
                "MD_INFORMATION::trajectory_output::Export_H5_Restart_File",
                writer.Last_Error().c_str());
        }
    }
    const auto finalize_start = std::chrono::steady_clock::now();
    if (!writer.Finalize())
    {
        h5_restart_finalize_elapsed_s += H5MD_Elapsed_Seconds(finalize_start);
        const std::string reason = writer.Last_Error();
        writer.Close();
        h5_restart_enabled = false;
        Record_H5_Output_Failure("restart", "finalize", reason);
        return;
    }
    h5_restart_finalize_elapsed_s += H5MD_Elapsed_Seconds(finalize_start);
    if (!writer.Close())
    {
        h5_restart_enabled = false;
        Record_H5_Output_Failure("restart", "close", writer.Last_Error());
        return;
    }
    if (!SpongeH5MD::Atomic_Replace_File(temporary_path, restart_path,
                                         &publish_error))
    {
        h5_restart_enabled = false;
        Record_H5_Output_Failure("restart", "publish", publish_error);
        return;
    }
    temporary_file_guard.Release();
    h5_restart_generation = candidate_generation;
}

void MD_INFORMATION::trajectory_output::Export_Restart_File(
    const char* rst7_name)
{
    if (!md_info->is_initialized || CONTROLLER::MPI_rank) return;

    char filename[CHAR_LENGTH_MAX];
    if (rst7_name == NULL)
        strcpy(filename, restart_name);
    else
        strcpy(filename, rst7_name);
    md_info->Crd_Vel_Device_To_Host();
    int export_index = restart_export_count % max_restart_export_count;
    restart_export_count = restart_export_count + 1;
    std::string prefix =
        export_index ? std::to_string(export_index) + "_" + filename : filename;
    if (Xponge::system.source == Xponge::InputSource::kAmber)
    {
        strcpy(filename, prefix.c_str());
        strcat(filename, ".rst7");
        const char* sys_name = md_info->md_name;
        FILE* lin = NULL;
        Open_File_Safely(&lin, filename, "w");
        fprintf(lin, "%s step=%d\n", sys_name, md_info->sys.steps);
        fprintf(lin, "%8d %.10lf\n", md_info->atom_numbers,
                md_info->sys.Get_Current_Time());
        int s = 0;
        for (int i = 0; i < md_info->atom_numbers; i = i + 1)
        {
            fprintf(lin, "%12.7f%12.7f%12.7f", md_info->coordinate[i].x,
                    md_info->coordinate[i].y, md_info->coordinate[i].z);
            s = s + 1;
            if (s == 2)
            {
                s = 0;
                fprintf(lin, "\n");
            }
        }
        if (s == 1)
        {
            s = 0;
            fprintf(lin, "\n");
        }
        for (int i = 0; i < md_info->atom_numbers; i = i + 1)
        {
            fprintf(lin, "%12.7f%12.7f%12.7f", md_info->velocity[i].x,
                    md_info->velocity[i].y, md_info->velocity[i].z);
            s = s + 1;
            if (s == 2)
            {
                s = 0;
                fprintf(lin, "\n");
            }
        }
        if (s == 1)
        {
            s = 0;
            fprintf(lin, "\n");
        }
        fprintf(lin, "%12.7f%12.7f%12.7f", (float)md_info->sys.box_length.x,
                (float)md_info->sys.box_length.y,
                (float)md_info->sys.box_length.z);
        fprintf(lin, "%12.7f%12.7f%12.7f", (float)md_info->sys.box_angle.x,
                (float)md_info->sys.box_angle.y,
                (float)md_info->sys.box_angle.z);
        fclose(lin);
    }
    else
    {
        FILE* lin = NULL;
        FILE* lin2 = NULL;
        std::string buffer;
        buffer = prefix + std::string("_coordinate.txt");
        Open_File_Safely(&lin, buffer.c_str(), "w");
        buffer = prefix + std::string("_velocity.txt");
        Open_File_Safely(&lin2, buffer.c_str(), "w");
        fprintf(lin, "%d %.10lf %d\n", md_info->atom_numbers,
                md_info->sys.Get_Current_Time(), md_info->sys.steps);
        fprintf(lin2, "%d %.10lf %d\n", md_info->atom_numbers,
                md_info->sys.Get_Current_Time(), md_info->sys.steps);
        for (int i = 0; i < md_info->atom_numbers; i++)
        {
            fprintf(lin, "%12.7f %12.7f %12.7f\n", md_info->coordinate[i].x,
                    md_info->coordinate[i].y, md_info->coordinate[i].z);
            fprintf(lin2, "%12.7f %12.7f %12.7f\n", md_info->velocity[i].x,
                    md_info->velocity[i].y, md_info->velocity[i].z);
        }
        fprintf(lin, "%12.7f %12.7f %12.7f %12.7f %12.7f %12.7f",
                md_info->sys.box_length.x, md_info->sys.box_length.y,
                md_info->sys.box_length.z, md_info->sys.box_angle.x,
                md_info->sys.box_angle.y, md_info->sys.box_angle.z);
        fclose(lin);
        fclose(lin2);
    }
}

bool MD_INFORMATION::trajectory_output::Check_Mdout_Step()
{
    return (print_zeroth_frame || md_info->sys.steps) &&
           md_info->sys.steps % write_mdout_interval == 0;
}

bool MD_INFORMATION::trajectory_output::Check_Force_Step()
{
    if (md_info->mode == md_info->RERUN) return Check_Trajectory_Step();
    return md_info->output.write_trajectory_interval &&
           (md_info->output.print_zeroth_frame || md_info->sys.steps) &&
           md_info->sys.steps % md_info->output.write_trajectory_interval == 0;
}

bool MD_INFORMATION::trajectory_output::Check_Trajectory_Step()
{
    return md_info->output.write_trajectory_interval &&
           (md_info->sys.steps + 1) %
                   md_info->output.write_trajectory_interval ==
               0 &&
           md_info->sys.steps != md_info->sys.step_limit;
}

bool MD_INFORMATION::trajectory_output::Check_Restart_Step()
{
    return md_info->output.write_restart_file_interval &&
           (md_info->sys.steps + 1) %
                   md_info->output.write_restart_file_interval ==
               0 &&
           md_info->sys.steps != md_info->sys.step_limit;
}

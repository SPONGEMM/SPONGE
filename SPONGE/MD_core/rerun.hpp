#pragma once

#include "utils/h5md/rerun_frame_selection.hpp"

void MD_INFORMATION::RERUN_information::Initial(CONTROLLER* controller,
                                                MD_INFORMATION* md_info)
{
    this->controller = controller;
    this->md_info = md_info;
    if (md_info->mode == RERUN)
    {
        controller->printf("    Start initializing rerun:\n");
        std::string filename;
        const auto input_plan =
            SpongeH5InputPlan::Resolve_Input_Plan(controller);
        if (!input_plan.valid)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand,
                "MD_INFORMATION::RERUN_information::Initial",
                input_plan.error_message.c_str());
        }
        if (input_plan.trajectory.binding.enabled)
        {
            h5_trajectory_enabled = true;
            h5_next_frame_index = 0;
            if (controller->MPI_rank == 0)
            {
                h5_trajectory_reader.reset(
                    new SpongeH5MD::TrajectoryH5Reader());
                if (!h5_trajectory_reader->Open(
                        input_plan.trajectory.binding.path,
                        input_plan.trajectory.particle_stream))
                {
                    controller->Throw_SPONGE_Error(
                        spongeErrorBadFileFormat,
                        "MD_INFORMATION::RERUN_information::Initial",
                        h5_trajectory_reader->Last_Error().c_str());
                }
                SpongeH5InputMetadata::TrajectoryMetadata metadata;
                if (!h5_trajectory_reader->Read_Metadata(&metadata))
                {
                    controller->Throw_SPONGE_Error(
                        spongeErrorBadFileFormat,
                        "MD_INFORMATION::RERUN_information::Initial",
                        h5_trajectory_reader->Last_Error().c_str());
                }
                if (!metadata.has_position || !metadata.has_box)
                {
                    controller->Throw_SPONGE_Error(
                        spongeErrorBadFileFormat,
                        "MD_INFORMATION::RERUN_information::Initial",
                        "Reason:\n\tH5MD rerun trajectory requires position "
                        "and box datasets\n");
                }
                if (metadata.atom_count != md_info->atom_numbers)
                {
                    controller->Throw_SPONGE_Error(
                        spongeErrorConflictingCommand,
                        "MD_INFORMATION::RERUN_information::Initial",
                        "Reason:\n\tH5MD rerun trajectory atom count does not "
                        "match MD atom count\n");
                }
                h5_trajectory_frame_count =
                    static_cast<int>(metadata.frame_count);
                h5_velocity_present = metadata.has_velocity;
                controller->printf("        Open rerun H5MD trajectory '%s'\n",
                                   input_plan.trajectory.binding.path.c_str());
            }
        }
        else if (!controller->Command_Exist(TRAJ_COMMAND))
        {
            controller->Throw_SPONGE_Error(
                spongeErrorMissingCommand,
                "MD_INFORMATION::RERUN_information::Initial",
                "Reason:\n\tno trajectory information found (command 'crd' "
                "required)");
        }
        else
        {
            filename = controller->Command(TRAJ_COMMAND);
            if (controller->MPI_rank == 0)
            {
                Open_File_Safely(&traj_file, filename.c_str(), "rb");

                controller->printf(
                    "        Open rerun coordinate trajectory '%s'\n",
                    filename.c_str());
                controller->Set_File_Buffer(
                    traj_file, sizeof(VECTOR) * md_info->atom_numbers);
                if (!controller->Command_Exist(BOX_TRAJ_COMMAND))
                {
                    controller->Throw_SPONGE_Error(
                        spongeErrorMissingCommand,
                        "MD_INFORMATION::RERUN_information::Initial",
                        "Reason:\n\tno box information found (command 'box' "
                        "required)");
                }
                filename = controller->Command(BOX_TRAJ_COMMAND);

                Open_File_Safely(&box_file, filename.c_str(), "r");

                controller->printf("        Open rerun box trajectory '%s'\n",
                                   filename.c_str());
                controller->Set_File_Buffer(box_file, sizeof(char) * 50);
                vel_file = NULL;
                if (controller->Command_Exist(VEL_TRAJ_COMMAND))
                {
                    filename = controller->Command(VEL_TRAJ_COMMAND);

                    Open_File_Safely(&vel_file, filename.c_str(), "r");

                    controller->printf(
                        "        Open rerun velocity trajectory '%s'\n",
                        filename.c_str());
                    controller->Set_File_Buffer(
                        vel_file, sizeof(VECTOR) * md_info->atom_numbers);
                }
            }
        }
        start_frame = 0;
        if (controller->Command_Exist("rerun_start"))
        {
            controller->Check_Float(
                "rerun_start", "MD_INFORMATION::RERUN_information::Initial");
            start_frame = atoi(controller->Command("rerun_start"));
        }
        strip_frame = 0;
        if (controller->Command_Exist("rerun_strip"))
        {
            controller->Check_Float(
                "rerun_strip", "MD_INFORMATION::RERUN_information::Initial");
            strip_frame = atoi(controller->Command("rerun_strip"));
        }
        need_box_update = 0;
        if (controller->Command_Exist("rerun_need_box_update"))
        {
            controller->Check_Float(
                "rerun_need_box_update",
                "MD_INFORMATION::RERUN_information::Initial");
            need_box_update =
                atoi(controller->Command("rerun_need_box_update"));
        }
        md_info->sys.step_limit = INT_MAX;
        controller->printf("    End initializing rerun\n\n");
    }
}

bool MD_INFORMATION::RERUN_information::Iteration(int strip)
{
    int n, nvel;
    int scanf_box;
    int loaded_frame = 0;
    if (strip < 0)
    {
        strip = this->strip_frame;
    }
    VECTOR old_box_length = md_info->sys.box_length;
    VECTOR old_box_angle = md_info->sys.box_angle;
    if (CONTROLLER::MPI_rank == 0)
    {
        if (h5_trajectory_enabled)
        {
            const auto selection = SpongeH5MD::Select_Next_H5_Rerun_Frame(
                h5_next_frame_index, strip, h5_trajectory_frame_count);
            h5_next_frame_index = selection.next_frame_index;
            md_info->sys.steps += selection.skipped_frames;
            if (!selection.has_frame)
            {
                md_info->sys.step_limit = md_info->sys.steps;
            }
            else
            {
                SpongeH5MD::RestartStructuralState frame;
                if (!h5_trajectory_reader->Read_Frame(selection.frame_index,
                                                      &frame))
                {
                    controller->Throw_SPONGE_Error(
                        spongeErrorBadFileFormat,
                        "MD_INFORMATION::RERUN_information::Iteration",
                        h5_trajectory_reader->Last_Error().c_str());
                }
                else
                {
                    std::string error_message;
                    if (!SpongeH5MD::Apply_Restart_Structural_State(
                            frame, md_info->atom_numbers, md_info->coordinate,
                            md_info->velocity, &md_info->sys.box_length,
                            &md_info->sys.box_angle, &md_info->sys.start_time,
                            &error_message))
                    {
                        controller->Throw_SPONGE_Error(
                            spongeErrorConflictingCommand,
                            "MD_INFORMATION::RERUN_information::Iteration",
                            error_message.c_str());
                    }
                    else
                    {
                        loaded_frame = 1;
                    }
                }
            }
        }
        else
        {
            for (int i = 0; i < strip; i++)
            {
                n = fread(this->md_info->coordinate, sizeof(VECTOR),
                          this->md_info->atom_numbers, traj_file);
                scanf_box = fscanf(
                    box_file, "%f %f %f %f %f %f", &md_info->sys.box_length.x,
                    &md_info->sys.box_length.y, &md_info->sys.box_length.z,
                    &md_info->sys.box_angle.x, &md_info->sys.box_angle.y,
                    &md_info->sys.box_angle.z);
                if (vel_file != NULL)
                {
                    nvel = fread(this->md_info->velocity, sizeof(VECTOR),
                                 this->md_info->atom_numbers, vel_file);
                }
                md_info->sys.steps += 1;
            }
            n = fread(this->md_info->coordinate, sizeof(VECTOR),
                      this->md_info->atom_numbers, traj_file);
            scanf_box =
                fscanf(box_file, "%f %f %f %f %f %f",
                       &md_info->sys.box_length.x, &md_info->sys.box_length.y,
                       &md_info->sys.box_length.z, &md_info->sys.box_angle.x,
                       &md_info->sys.box_angle.y, &md_info->sys.box_angle.z);
            nvel = this->md_info->atom_numbers;
            if (vel_file != NULL)
            {
                nvel = fread(this->md_info->velocity, sizeof(VECTOR),
                             this->md_info->atom_numbers, vel_file);
            }
            if (n != this->md_info->atom_numbers || scanf_box != 6 ||
                nvel != this->md_info->atom_numbers)
            {
                md_info->sys.step_limit = md_info->sys.steps;
                // close files
                if (traj_file != NULL)
                {
                    fclose(traj_file);
                    traj_file = NULL;
                }
                if (box_file != NULL)
                {
                    fclose(box_file);
                    box_file = NULL;
                }
                if (vel_file != NULL)
                {
                    fclose(vel_file);
                    vel_file = NULL;
                }
            }
            else
            {
                loaded_frame = 1;
            }
        }
        if (loaded_frame && loaded_frame_count > 0 && !need_box_update)
        {
            md_info->sys.box_length = old_box_length;
            md_info->sys.box_angle = old_box_angle;
        }
        if ((h5_trajectory_enabled && !h5_velocity_present) ||
            (!h5_trajectory_enabled && vel_file == NULL))
        {
            memset(this->md_info->velocity, 0,
                   sizeof(VECTOR) * this->md_info->atom_numbers);
        }
    }
#ifdef USE_MPI
    MPI_Bcast(&loaded_frame, sizeof(int), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&md_info->sys.step_limit, sizeof(int), MPI_BYTE, 0,
              MPI_COMM_WORLD);
    MPI_Bcast(this->md_info->coordinate,
              sizeof(VECTOR) * this->md_info->atom_numbers, MPI_BYTE, 0,
              MPI_COMM_WORLD);
    MPI_Bcast(this->md_info->velocity,
              sizeof(VECTOR) * this->md_info->atom_numbers, MPI_BYTE, 0,
              MPI_COMM_WORLD);
    MPI_Bcast(&md_info->sys.box_length, sizeof(VECTOR), MPI_BYTE, 0,
              MPI_COMM_WORLD);
    MPI_Bcast(&md_info->sys.box_angle, sizeof(VECTOR), MPI_BYTE, 0,
              MPI_COMM_WORLD);
#endif
    if (loaded_frame) loaded_frame_count += 1;
    bool box_changed =
        fabsf(old_box_length.x - md_info->sys.box_length.x) > 1e-4 ||
        fabsf(old_box_length.y - md_info->sys.box_length.y) > 1e-4 ||
        fabsf(old_box_length.z - md_info->sys.box_length.z) > 1e-4 ||
        fabsf(old_box_angle.x - md_info->sys.box_angle.x) > 1e-4 ||
        fabsf(old_box_angle.y - md_info->sys.box_angle.y) > 1e-4 ||
        fabsf(old_box_angle.z - md_info->sys.box_angle.z) > 1e-4;
    if (box_changed && md_info->pbc.pbc)
    {
        // 通过盒子变化计算g
        LTMatrix3 new_cell = md_info->pbc.Get_Cell(md_info->sys.box_length,
                                                   md_info->sys.box_angle);
        g = (1 / md_info->dt) * (new_cell * inv(md_info->pbc.boundary.cell) -
                                 LTMatrix3(1, 0, 1, 0, 0, 1));
    }
    deviceMemcpy(this->md_info->crd, this->md_info->coordinate,
                 sizeof(VECTOR) * this->md_info->atom_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(this->md_info->vel, this->md_info->velocity,
                 sizeof(VECTOR) * this->md_info->atom_numbers,
                 deviceMemcpyHostToDevice);
    return box_changed;
}

#pragma once

void MD_INFORMATION::trajectory_output::Initial(CONTROLLER* controller,
                                                MD_INFORMATION* md_info)
{
    this->md_info = md_info;
    current_crd_synchronized_step = -1;

    print_virial = false;
    if (controller->Command_Exist("print_pressure"))
    {
        print_virial = controller->Get_Bool(
            "print_pressure", "MD_INFORMATION::trajectory_output::Initial");
    }
    if (print_virial)
    {
        controller->Step_Print_Initial("pressure", "%.2f");
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
    }
    if (write_trajectory_interval != 0)
    {
        crd_traj = controller->Get_Output_File(true, TRAJ_COMMAND, ".dat",
                                               TRAJ_DEFAULT_FILENAME);
        controller->Set_File_Buffer(crd_traj,
                                    sizeof(VECTOR) * md_info->atom_numbers);
        box_traj = controller->Get_Output_File(false, BOX_TRAJ_COMMAND, ".box",
                                               BOX_TRAJ_DEFAULT_FILENAME);
        char line[256];
        sprintf(line, "%9.3f %9.3f %9.3f %9.5f %9.5f %9.5f\n",
                md_info->sys.box_length.x, md_info->sys.box_length.y,
                md_info->sys.box_length.z, 90.0f, 90.0f, 90.0f);
        controller->Set_File_Buffer(box_traj, sizeof(char) * strlen(line));
    }
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
        fwrite(&md_info->coordinate[0].x, sizeof(VECTOR), md_info->atom_numbers,
               fp);
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
            }
        }
        else
        {
            fwrite(&md_info->velocity[0].x, sizeof(VECTOR),
                   md_info->atom_numbers, fp);
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
        fprintf(fp, "%9.6f %9.6f %9.6f %9.5f %9.5f %9.5f\n",
                md_info->sys.box_length.x, md_info->sys.box_length.y,
                md_info->sys.box_length.z, md_info->sys.box_angle.x,
                md_info->sys.box_angle.y, md_info->sys.box_angle.z);
    }
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
    return md_info->mode == md_info->RERUN ||
           md_info->output.write_trajectory_interval &&
               (md_info->output.print_zeroth_frame || md_info->sys.steps) &&
               md_info->sys.steps % md_info->output.write_trajectory_interval ==
                   0;
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

#pragma once

inline void CONTROLLER::printf(const char* fmt, ...)
{
    if (MPI_rank != 0) return;
    va_list argp;

    va_start(argp, fmt);
    vfprintf(stdout, fmt, argp);
    va_end(argp);

    if (mdinfo != NULL)
    {
        va_start(argp, fmt);
        vfprintf(mdinfo, fmt, argp);
        va_end(argp);
    }
}

inline void CONTROLLER::MPI_printf(const char* fmt, ...)
{
    va_list argp;
    char* temp_local = new char[CHAR_LENGTH_MAX];
    va_start(argp, fmt);
    vsprintf(temp_local, fmt, argp);
    va_end(argp);
#ifdef USE_MPI
    char* temp_for_print;
    int *recvcounts, *displs;
    int local_length = strlen(temp_local);
    int total_length = 0;
    MPI_Reduce(&local_length, &total_length, 1, MPI_INT, MPI_SUM,
               SPONGE_MPI_ROOT, MPI_COMM_WORLD);
    if (CONTROLLER::MPI_rank == 0)
    {
        temp_for_print = new char[total_length + 1];
        recvcounts = new int[CONTROLLER::MPI_size];
        displs = new int[CONTROLLER::MPI_size];
    }

    MPI_Gather(&local_length, 1, MPI_INT, recvcounts, 1, MPI_INT,
               SPONGE_MPI_ROOT, MPI_COMM_WORLD);

    if (CONTROLLER::MPI_rank == 0)
    {
        int current_displacement = 0;
        for (int i = 0; i < CONTROLLER::MPI_size; i++)
        {
            displs[i] = current_displacement;
            current_displacement += recvcounts[i];
        }
    }

    MPI_Gatherv(temp_local, local_length, MPI_CHAR, temp_for_print, recvcounts,
                displs, MPI_CHAR, SPONGE_MPI_ROOT, MPI_COMM_WORLD);

    if (CONTROLLER::MPI_rank == 0)
    {
        temp_for_print[total_length] = 0;
        printf("%s", temp_for_print);
        delete[] temp_for_print;
        delete[] recvcounts;
        delete[] displs;
    }
#else
    printf(temp_local);
#endif
    delete[] temp_local;
}

inline void CONTROLLER::Step_Print_Initial(const char* head, const char* format)
{
    outputs_format.insert(std::pair<std::string, std::string>(head, format));
    outputs_content.insert(std::pair<std::string, std::string>(head, "****"));
    outputs_key.push_back(head);
}

inline void CONTROLLER::Step_Print(const char* head, const float* pointer,
                                   const bool add_to_total)
{
    char temp[CHAR_LENGTH_MAX];
    if (outputs_content.count(head))
    {
        sprintf(temp, outputs_format[head].c_str(), pointer[0]);
        outputs_content[head] = temp;
        if (add_to_total)
        {
            printf_sum += pointer[0];
        }
    }
}

inline void CONTROLLER::Step_Print(const char* head, const float pointer,
                                   const bool add_to_total)
{
    char temp[CHAR_LENGTH_MAX];
    if (outputs_content.count(head))
    {
        sprintf(temp, outputs_format[head].c_str(), pointer);
        outputs_content[head] = temp;
        if (add_to_total)
        {
            printf_sum += pointer;
        }
    }
}

inline void CONTROLLER::Step_Print(const char* head, const double pointer,
                                   const bool add_to_total)
{
    char temp[CHAR_LENGTH_MAX];
    if (outputs_content.count(head))
    {
        sprintf(temp, outputs_format[head].c_str(), pointer);
        outputs_content[head] = temp;
        if (add_to_total)
        {
            printf_sum += (float)pointer;
        }
    }
}

inline void CONTROLLER::Step_Print(const char* head, const int pointer)
{
    char temp[CHAR_LENGTH_MAX];
    if (outputs_content.count(head))
    {
        sprintf(temp, outputs_format[head].c_str(), pointer);
        outputs_content[head] = temp;
    }
}

inline void CONTROLLER::Step_Print(const char* head, const char* pointer)
{
    char temp[CHAR_LENGTH_MAX];
    if (outputs_content.count(head))
    {
        sprintf(temp, outputs_format[head].c_str(), pointer);
        outputs_content[head] = temp;
    }
}

inline void CONTROLLER::Print_First_Line_To_Mdout(FILE* mdout)
{
#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    Get_Time_Recorder("Initialization")->Stop();
    if (MPI_rank == 0)
    {
        if (mdout == NULL)
        {
            mdout = this->mdout;
        }
        if (mdout != NULL)
        {
            this->Set_File_Buffer(mdout,
                                  sizeof(char) * outputs_key.size() * 16);
        }
        char space[4] = " ";
        for (int i = 0; i < outputs_key.size(); i++)
        {
            if (i == outputs_key.size() - 1) space[0] = '\n';
            if (mdout != NULL)
            {
                fprintf(mdout, "%15s%s", outputs_key[i].c_str(), space);
            }
        }
        printf(
            "------------------------------------------------------------------"
            "------------------------------------------\n");
        SpongeLegacyIO::OutputFlushCoordinator::Mark_Dirty(mdout, "mdout");
    }
    core_time.Start();
}

inline void CONTROLLER::Print_To_Screen_And_Mdout(FILE* mdout)
{
    if (MPI_rank != 0) return;
    if (mdout == NULL)
    {
        mdout = this->mdout;
    }
    int line_numbers = 0;
    char space[4] = " ";
    for (int i = 0; i < outputs_key.size(); i++)
    {
        line_numbers++;
        fprintf(stdout, "%15s = %15s, ", outputs_key[i].c_str(),
                outputs_content[outputs_key[i]].c_str());
        if (i == outputs_key.size() - 1) space[0] = '\n';
        if (mdout != NULL)
        {
            fprintf(mdout, "%15s%s", outputs_content[outputs_key[i]].c_str(),
                    space);
        }
        outputs_content[outputs_key[i]] = "****";
        if (line_numbers % 3 == 0) fprintf(stdout, "\n");
    }
    if (line_numbers % 3 != 0) fprintf(stdout, "\n");
    fprintf(stdout,
            "------------------------------------------------------------------"
            "------------------------------------------\n");
    SpongeLegacyIO::OutputFlushCoordinator::Mark_Dirty(mdout, "mdout");
}

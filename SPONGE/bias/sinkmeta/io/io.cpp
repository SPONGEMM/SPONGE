#include "../meta.h"
#include "../util.h"
using sinkmeta::split_sentence;
static void Write_CV_Header(FILE* temp_file, int ndim, const CV_LIST& cvs)
{
    for (int i = 0; i < ndim; ++i)
    {
        const char* cv_name = nullptr;
        if (i < static_cast<int>(cvs.size()) && cvs[i] != nullptr &&
            cvs[i]->module_name[0] != '\0')
        {
            cv_name = cvs[i]->module_name;
        }
        if (cv_name != nullptr)
        {
            fprintf(temp_file, "%s\t", cv_name);
        }
        else
        {
            fprintf(temp_file, "cv%d\t", i + 1);
        }
    }
}
void META::Write_Potential(void)
{
    if (!is_initialized)
    {
        return;
    }
    if (CONTROLLER::MPI_rank == CONTROLLER::MPI_size - 1)
    {
        FILE* temp_file = NULL;
        Open_File_Safely(&temp_file, write_potential_file_name, "w");
        if (subhill || (!usegrid && !use_scatter))
        {
            fprintf(temp_file, "# ");
            Write_CV_Header(temp_file, ndim, cvs);
            fprintf(temp_file, "potential_local\tpotential_backup");
            if (!kde)
            {
                fprintf(temp_file, "\tpotential_raw");
            }
            fprintf(temp_file, "\n");
            vector<float> loop_flag(ndim, 0);
            vector<float> loop_floor(ndim, 0);
            for (int i = 0; i < ndim; ++i)
            {
                loop_floor[i] = cv_mins[i] + 0.5 * cv_deltas[i];
                loop_flag[i] = loop_floor[i];
            }
            int i = 0;
            while (i >= 0)
            {
                Estimate(loop_flag, true, false);  // get potential
                ostringstream ss;
                for (const float& v : loop_flag)
                {
                    ss << v << "\t";
                }
                fprintf(temp_file, "%s%f\t%f", ss.str().c_str(),
                        potential_local, potential_backup);
                if (!kde)
                {
                    if (mgrid != nullptr)
                    {
                        fprintf(temp_file, "\t%f",
                                mgrid->potential[mgrid->Get_Flat_Index(loop_flag)]);
                    }
                    else if (mscatter != nullptr)
                    {
                        fprintf(temp_file, "\t%f",
                                mscatter->potential[mscatter->Get_Index(loop_flag)]);
                    }
                }
                fprintf(temp_file, "\n");
                //  iterate over any dimensions
                i = ndim - 1;
                while (i >= 0)
                {
                    loop_flag[i] += cv_deltas[i];
                    if (loop_flag[i] > cv_maxs[i])
                    {
                        loop_flag[i] = loop_floor[i];
                        --i;
                    }
                    else
                    {
                        break;
                    }
                }
            }
        }
        else if (mgrid != nullptr)
        {
            fprintf(temp_file, "# ");
            Write_CV_Header(temp_file, ndim, cvs);
            fprintf(temp_file, "potential_raw\tpotential_shifted\tvshift\n");
            for (int idx = 0; idx < mgrid->total_size; ++idx)
            {
                ostringstream ss;
                const Axis coor = mgrid->Get_Coordinates(idx);
                float vshift = CalcVshift(coor);
                for (const float& v : coor)
                {
                    ss << v << "\t";
                }
                fprintf(temp_file, "%s%f\t%f\t%f\n", ss.str().c_str(),
                        mgrid->potential[idx],
                        mgrid->potential[idx] - vshift, vshift);
            }
        }
        // In case of pure scattering point!
        else if (mscatter != nullptr)
        {
            fprintf(temp_file, "# ");
            Write_CV_Header(temp_file, ndim, cvs);
            fprintf(temp_file, "potential_raw\tpotential_shifted\n");
            for (int iter = 0; iter < scatter_size; ++iter)
            {
                ostringstream ss;
                const Axis& coor = mscatter->Get_Coordinate(iter);
                float vshift = CalcVshift(coor);
                for (auto& v : coor)
                {
                    ss << v << "\t";
                }
                fprintf(temp_file, "%s%f\t%f\n", ss.str().c_str(),
                        mscatter->potential[iter],
                        mscatter->potential[iter] - vshift);
            }
        }
        fclose(temp_file);
    }
}
void META::Write_Directly(void)
{
    if (!is_initialized || !(use_scatter || usegrid))
    {
        return;
    }
    if (CONTROLLER::MPI_rank == CONTROLLER::MPI_size - 1)
    {
        FILE* temp_file = NULL;
        Open_File_Safely(&temp_file, write_directly_file_name, "w");
        string meta_type;
        if (do_negative)
        {
            string pm = to_string(potential_max);
            meta_type += "sink(kcal): " + pm;
        }
        if (mask)
        {
            meta_type += " mask ";
        }
        if (subhill)
        {
            meta_type += " subhill ";
        }
        else
        {
            meta_type += " d_force";
        }

        fprintf(temp_file, "%dD-Meta X %s\n", ndim, meta_type.c_str());
        for (int i = 0; i < ndim; ++i)
        {
            fprintf(temp_file, "%f\t%f\t%f\n", cv_mins[i], cv_maxs[i],
                    cv_deltas[i]);
        }
        int gridsize = 1;
        for (int i = 0; i < ndim; ++i)
        {
            int num_grid = round((cv_maxs[i] - cv_mins[i]) / cv_deltas[i]);
            fprintf(temp_file, " %d\t", num_grid);
            gridsize *= num_grid;
        }
        if (mscatter != nullptr)
        {
            fprintf(temp_file, "%d\n", scatter_size);
            for (int iter = 0; iter < scatter_size; ++iter)
            {
                ostringstream ss;
                const Axis& coor = mscatter->Get_Coordinate(iter);
                Estimate(coor, true, false);  // get potential
                for (auto& v : coor)
                {
                    ss << v << "\t";
                }
                if (subhill)
                {
                    fprintf(temp_file, "%s%f\t%f\t%f\n", ss.str().c_str(),
                            potential_local, potential_backup,
                            mscatter->potential[iter]);
                }
                else  // restart of catheter will replace the result!
                {
                    float result;
                    result = potential_local;
                    fprintf(temp_file, "%s%f\t", ss.str().c_str(), result);
                    float* data = &mscatter->force[iter * ndim];
                    for (int i = 0; i < ndim; ++i)
                    {
                        fprintf(temp_file, "%f\t", data[i]);
                    }
                    fprintf(temp_file, "%f\n", mscatter->potential[iter]);
                }
            }
        }
        else if (mgrid != nullptr)
        {
            fprintf(temp_file, "%d\n", mgrid->total_size);
            for (int idx = 0; idx < mgrid->total_size; ++idx)
            {
                ostringstream ss;
                vector<float> coor = mgrid->Get_Coordinates(idx);
                Estimate(coor, true, false);  // get potential
                for (auto& v : coor)
                {
                    ss << v << "\t";
                }
                fprintf(temp_file, "%s%f\t", ss.str().c_str(),
                        potential_local);
                float* data = &mgrid->force[idx * ndim];
                for (int i = 0; i < ndim; ++i)
                {
                    fprintf(temp_file, "%f\t", data[i]);
                }
                fprintf(temp_file, "%f\n", mgrid->potential[idx]);
            }
        }
        fclose(temp_file);
    }
}
void META::Read_Potential(CONTROLLER* controller)
{
    FILE* temp_file = NULL;
    Open_File_Safely(&temp_file, read_potential_file_name, "r");
    char temp_char[256];
    int scanf_ret = 0;
    char* get_val = fgets(temp_char, 256, temp_file);  // title line
    Malloc_Safely((void**)&cv_mins, sizeof(float) * ndim);
    Malloc_Safely((void**)&cv_maxs, sizeof(float) * ndim);
    Malloc_Safely((void**)&cv_deltas, sizeof(float) * ndim);
    Malloc_Safely((void**)&n_grids, sizeof(float) * ndim);
    for (int i = 0; i < ndim; ++i)
    {
        scanf_ret = fscanf(temp_file, "%f %f %f\n", &cv_mins[i], &cv_maxs[i],
                           &cv_deltas[i]);
        if (scanf_ret != 3)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat, "META::Read_Potential",
                "Reason:\n\tbad potential input file\n");
        }
        controller->printf(
            "    CV_minimal = %f\n    CV_maximum = %f\n    dCV = %f\n",
            cv_mins[i], cv_maxs[i], cv_deltas[i]);
    }
    for (int i = 0; i < ndim; ++i)
    {
        scanf_ret = fscanf(temp_file, "%d", &n_grids[i]);
        if (scanf_ret != 1)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat, "META::Read_Potential",
                "Reason:\n\tbad potential input file\n");
        }
    }
    scanf_ret = fscanf(temp_file, "%d\n", &scatter_size);
    // Scatter points coordinate
    for (int i = 0; i < ndim; ++i)
    {
        float* ttoorr;
        Malloc_Safely((void**)&ttoorr, sizeof(float) * scatter_size);
        tcoor.push_back(ttoorr);
    }
    vector<float> potential_from_file;
    vector<Gdata> force_from_file;
    sigma_s = cv_sigmas[0];
    for (int j = 0; j < scatter_size; ++j)
    {
        char* grid_val = fgets(temp_char, 256, temp_file);
        std::vector<std::string> words;
        int nwords = split_sentence(temp_char, words);
        Gdata force(ndim, 0.);
        if (nwords < ndim)
        {
            controller->printf("size %d not match %d\n", nwords, ndim);
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat, "META::Read_Potential",
                "Reason:\n\tbad potential input file \n");
        }
        else if (nwords < ndim + 2)
        {
            potential_from_file.push_back(0.);
        }
        else if (subhill && nwords >= ndim + 2)
        {
            potential_from_file.push_back(std::stof(words[nwords - 1]));
        }
        else if (nwords == 2 * ndim + 2)
        {
            potential_from_file.push_back(
                std::stof(words[2 * ndim + 1]));  // raw hill before sink
            if (!subhill)
            {
                for (int i = 0; i < ndim; ++i)
                {
                    force[i] = std::stof(words[1 + ndim + i]);
                }
            }
        }
        for (int i = 0; i < ndim; ++i)
        {
            tcoor[i][j] = std::stof(words[i]);  // coordinate!
        }
        force_from_file.push_back(force);
        if (catheter)
        {
            sigma_r = std::stof(words[ndim]);
            float sr_inv = 1.0 / sigma_r;
            delta_sigma.push_back(0.5 * (sigma_s * sigma_s - sr_inv * sr_inv));
        }
    }
    fclose(temp_file);
    Setgrid(controller);
    vector<float>::iterator max_it =
        max_element(potential_from_file.begin(), potential_from_file.end());
    potential_max = *max_it;
    if (usegrid)
    {
        mgrid->potential = potential_from_file;  // potential
        // calculate derivative force dpotential
        if (!subhill)
        {
            for (int idx = 0; idx < mgrid->total_size; ++idx)
            {
                for (int d = 0; d < ndim; ++d)
                {
                    mgrid->force[idx * ndim + d] = force_from_file[idx][d];
                }
            }
        }
    }
    else if (use_scatter)
    {
        mscatter->potential = potential_from_file;
        if (convmeta)
        {
            max_index = distance(potential_from_file.begin(), max_it);
        }
        if (!subhill)
        {
            mscatter->force.resize(scatter_size * ndim);
            for (int idx = 0; idx < scatter_size; ++idx)
            {
                for (int d = 0; d < ndim; ++d)
                {
                    mscatter->force[idx * ndim + d] = force_from_file[idx][d];
                }
            }
        }
        if (mask)
        {
            for (int index = 0; index < mscatter->size(); ++index)
            {
                const Axis& coor = mscatter->Get_Coordinate(index);
                int gidx = mgrid->Get_Flat_Index(coor);
                mgrid->potential[gidx] = potential_from_file[index];

                for (int d = 0; d < ndim; ++d)
                {
                    mgrid->force[gidx * ndim + d] = force_from_file[index][d];
                }
            }
        }
    }
    if (mgrid != nullptr) mgrid->Sync_To_Device();
    if (mscatter != nullptr) mscatter->Sync_To_Device();
}

void META::Step_Print(CONTROLLER* controller)
{
    if (!is_initialized)
    {
        return;
    }
    if (CONTROLLER::MPI_size == 1 && CONTROLLER::PM_MPI_size == 1)
    {
        controller->Step_Print(this->module_name, potential_local);
        controller->Step_Print("rbias", rbias);
        controller->Step_Print("rct", rct);
        return;
    }
#ifdef USE_MPI
    if (CONTROLLER::MPI_rank == CONTROLLER::MPI_size - 1)
    {
        MPI_Send(&potential_local, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&rbias, 1, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
        MPI_Send(&rct, 1, MPI_FLOAT, 0, 2, MPI_COMM_WORLD);
    }
    if (CONTROLLER::MPI_rank == 0)
    {
        MPI_Recv(&potential_local, 1, MPI_FLOAT, CONTROLLER::MPI_size - 1, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&rbias, 1, MPI_FLOAT, CONTROLLER::MPI_size - 1, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&rct, 1, MPI_FLOAT, CONTROLLER::MPI_size - 1, 2,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        controller->Step_Print(this->module_name, potential_local);
        controller->Step_Print("rbias", rbias);
        controller->Step_Print("rct", rct);
    }
#endif
}

#include "../meta.h"
static int split_sentence(const std::string& line,
                          std::vector<std::string>& words)
{
    words.clear();
    std::istringstream iss(line);
    std::string word;
    while (iss >> word)
    {
        words.push_back(word);
    }
    return static_cast<int>(words.size());
}

static int split_sentence(const char* line, std::vector<std::string>& words)
{
    if (line == nullptr) return 0;
    return split_sentence(std::string(line), words);
}
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
        /*fprintf(temp_file, "%dD-Meta X %f\n", ndim, sum_normal);
        for (int i = 0; i < ndim; ++i)
        {
            fprintf(temp_file, "%f\t%f\t%f\n", cv_mins[i], cv_maxs[i],
        cv_deltas[i]);
        }
        int gridsize = 1;
        for (int i = 0; i < ndim; ++i)
        {
            int num_grid = round((cv_maxs[i] - cv_mins[i]) / cv_deltas[i]);
            if (periods[i] == 0)
            {
                ++num_grid; //  numpoint+1 for non-periodic condition
            }
            fprintf(temp_file, " %d\t", num_grid);
            gridsize *= num_grid;
        }
        fprintf(temp_file, "%d\n", gridsize);
            */
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
                    if (potential_grid != nullptr)
                    {
                        fprintf(temp_file, "\t%f",
                                potential_grid->at(loop_flag));
                    }
                    else if (potential_scatter != nullptr)
                    {
                        fprintf(temp_file, "\t%f",
                                potential_scatter->at(loop_flag));
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
        else if (potential_grid != nullptr)
        {
            fprintf(temp_file, "# ");
            Write_CV_Header(temp_file, ndim, cvs);
            fprintf(temp_file, "potential_raw\tpotential_shifted\tvshift\n");
            for (Grid<float>::iterator g_iter = potential_grid->begin();
                 g_iter != potential_grid->end(); ++g_iter)
            {
                ostringstream ss;
                const Axis coor = g_iter.coordinates();
                float vshift = CalcVshift(coor);
                for (const float& v : coor)
                {
                    ss << v << "\t";
                }
                fprintf(temp_file, "%s%f\t%f\t%f\n", ss.str().c_str(), *g_iter,
                        *g_iter - vshift, vshift);
            }
        }
        // In case of pure scattering point!
        else if (potential_scatter != nullptr)
        {
            fprintf(temp_file, "# ");
            Write_CV_Header(temp_file, ndim, cvs);
            fprintf(temp_file, "potential_raw\tpotential_shifted\n");
            // fprintf(temp_file, "%d\n", scatter_size);
            for (int iter = 0; iter < scatter_size; ++iter)
            {
                ostringstream ss;
                const Axis coor = potential_scatter->GetCoordinate(iter);
                float vshift = CalcVshift(coor);
                for (auto& v : coor)
                {
                    ss << v << "\t";
                }
                fprintf(temp_file, "%s%f\t%f\n", ss.str().c_str(),
                        potential_scatter->data_[iter],
                        potential_scatter->data_[iter] - vshift);
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
            /*if (periods[i] == 0)
            {
                ++num_grid; //  numpoint+1 for non-periodic condition
            }*/
            fprintf(temp_file, " %d\t", num_grid);
            gridsize *= num_grid;
        }
        if (potential_scatter != nullptr)
        {
            // printf("Directly print the %d scatter points to
            // %s\n",scatter_size,write_directly_file_name);
            fprintf(temp_file, "%d\n", scatter_size);
            for (int iter = 0; iter < scatter_size; ++iter)
            {
                ostringstream ss;
                vector<float> coor = potential_scatter->GetCoordinate(iter);
                Estimate(coor, true, false);  // get potential
                for (auto& v : coor)
                {
                    ss << v << "\t";
                }
                if (subhill)
                {
                    fprintf(temp_file, "%s%f\t%f\t%f\n", ss.str().c_str(),
                            potential_local, potential_backup,
                            potential_scatter->data_[iter]);
                }
                else  // restart of catheter will replace the result!
                {
                    float result;
                    result = potential_local;
                    fprintf(temp_file, "%s%f\t", ss.str().c_str(), result);
                    Gdata& data = scatter->data_[iter];
                    for (int i = 0; i < ndim; ++i)
                    {
                        fprintf(temp_file, "%f\t", data[i]);
                    }
                    fprintf(temp_file, "%f\n", potential_scatter->data_[iter]);
                }
            }
        }
        else if (potential_grid != nullptr)
        {
            /*for (int i = 0; i < ndim; ++i)
            {
                fprintf(temp_file, " %d\t", n_grids[i]);
            }*/
            fprintf(temp_file, "%zu\n", potential_grid->size());
            for (Grid<float>::iterator g_iter = potential_grid->begin();
                 g_iter != potential_grid->end(); ++g_iter)
            {
                ostringstream ss;
                vector<float> coor = g_iter.coordinates();
                Estimate(coor, true, false);  // get potential
                for (auto& v : coor)
                {
                    ss << v << "\t";
                }
                fprintf(temp_file, "%s%f\t", ss.str().c_str(),
                        potential_local);  // potential_grid->data_[index]);
                Gdata& data = grid->at(coor);
                for (int i = 0; i < ndim; ++i)
                {
                    fprintf(temp_file, "%f\t", data[i]);
                }
                fprintf(temp_file, "%f\n", *g_iter);
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
        char* grid_val = fgets(temp_char, 256, temp_file);  // grid line
        /*std::string command = string_strip(temp_char);
        std::vector<std::string> words
         = string_split(command, " ");*/
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
            // printf("Success reading line %d\n",j);
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
        potential_grid->data_ = potential_from_file;  // potential
        // calculate derivative force dpotential
        if (!subhill)
        {
            int index = 0;
            for (Grid<float>::iterator it = potential_grid->begin();
                 it != potential_grid->end(); ++it)
            {
                for (int i = 0; i < ndim; ++i)
                {
                    Axis coord = it.coordinates();
                    grid->at(coord)[i] = force_from_file[index][i];
                }
                /*
                for (int i = 0; i < ndim; ++i)
                {
                    vector<int> shift(ndim, 0);
                    shift[i] = 1;
                    // float just = *it;
                    auto shit = it;
                    shit += shift;
                    if (shit != potential_grid->end()) // do not compute
                edge !
                    {
                        grid->at(it.GetIndices())[i] = (*shit - *it) /
                cv_deltas[i];
                    }
                }*/
                ++index;
            }
        }
    }
    else if (use_scatter)
    {
        potential_scatter->data_ = potential_from_file;
        if (convmeta)
        {
            max_index = distance(potential_from_file.begin(), max_it);
            // sum_normal =
            // expf(-normal_lse->at(scatter->GetCoordinate(max_index)));
        }
        if (!subhill)
        {
            scatter->data_ = force_from_file;
        }
        if (mask)
        {
            for (int index = 0; index < potential_scatter->size(); ++index)
            {
                Axis coor = potential_scatter->GetCoordinate(index);
                potential_grid->at(coor) = potential_from_file[index];

                for (int i = 0; i < ndim; ++i)
                {
                    grid->at(coor)[i] = force_from_file[index][i];
                }
            }
        }
    }
}

#ifdef USE_GPU
static __global__ void Add_Frc(const int atom_numbers, VECTOR* frc,
                               VECTOR* cv_grad, float dheight_dcv)
{
    for (int i = blockIdx.x + blockDim.x * threadIdx.x; i < atom_numbers;
         i += gridDim.x * blockDim.x)
    {
        frc[i] = frc[i] - dheight_dcv * cv_grad[i];
    }
}

static __global__ void Add_Potential(float* d_potential, const float to_add)
{
    d_potential[0] += to_add;
}

static __global__ void Add_Virial(LTMatrix3* d_virial, const float dU_dCV,
                                  const LTMatrix3* cv_virial)
{
    d_virial[0] = d_virial[0] - dU_dCV * cv_virial[0];
}
#else
static void Add_Frc(const int atom_numbers, VECTOR* frc, VECTOR* cv_grad,
                    float dheight_dcv)
{
#pragma omp parallel for
    for (int i = 0; i < atom_numbers; i++)
    {
        frc[i] = frc[i] - dheight_dcv * cv_grad[i];
    }
}

static void Add_Potential(float* d_potential, const float to_add)
{
    d_potential[0] += to_add;
}

static void Add_Virial(LTMatrix3* d_virial, const float dU_dCV,
                       const LTMatrix3* cv_virial)
{
    d_virial[0] = d_virial[0] - dU_dCV * cv_virial[0];
}
#endif

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

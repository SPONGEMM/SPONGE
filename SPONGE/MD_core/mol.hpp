#pragma once

static __global__ void Get_Origin(const int residue_numbers, const int* start,
                                  const int* end, const VECTOR* crd,
                                  const float* atom_mass,
                                  const float* residue_mass_inverse,
                                  VECTOR* center_of_mass)
{
#ifdef USE_GPU
    int residue_i = blockDim.x * blockIdx.x + threadIdx.x;
    if (residue_i < residue_numbers)
#else
#pragma omp parallel for
    for (int residue_i = 0; residue_i < residue_numbers; residue_i++)
#endif
    {
        VECTOR origin = {FLT_MAX, FLT_MAX, FLT_MAX};
        for (int atom_i = start[residue_i]; atom_i < end[residue_i];
             atom_i += 1)
        {
            VECTOR crd_i = crd[atom_i];
            origin.x = fminf(origin.x, crd_i.x);
            origin.y = fminf(origin.y, crd_i.y);
            origin.z = fminf(origin.z, crd_i.z);
        }
        center_of_mass[residue_i] = origin;
    }
}

static __global__ void Map_Center_Of_Mass(
    const int residue_numbers, const int* start, const int* end,
    const float scaler, const VECTOR* center_of_mass, const LTMatrix3 cell,
    const LTMatrix3 rcell, VECTOR* crd, int* periodicity)
{
    VECTOR trans_vec;
    VECTOR com;
#ifdef USE_GPU
    int residue_i = blockDim.x * blockIdx.x + threadIdx.x;
    if (residue_i < residue_numbers)
#else
    for (int residue_i = 0; residue_i < residue_numbers; ++residue_i)
#endif
    {
        if (periodicity == NULL || periodicity[residue_i] == 0)
        {
            com = center_of_mass[residue_i];
            VECTOR frac = com * rcell;
            frac.x = frac.x - floorf(frac.x);
            frac.y = frac.y - floorf(frac.y);
            frac.z = frac.z - floorf(frac.z);
            VECTOR mapped = frac * cell;
            trans_vec.x = scaler * mapped.x - com.x;
            trans_vec.y = scaler * mapped.y - com.y;
            trans_vec.z = scaler * mapped.z - com.z;
#ifdef USE_GPU
            for (int atom_i = start[residue_i] + threadIdx.y;
                 atom_i < end[residue_i]; atom_i += blockDim.y)
#else
            for (int atom_i = start[residue_i]; atom_i < end[residue_i];
                 ++atom_i)
#endif
            {
                crd[atom_i] = crd[atom_i] + trans_vec;
            }
        }
        else
        {
#ifdef USE_GPU
            for (int atom_i = start[residue_i] + threadIdx.y;
                 atom_i < end[residue_i]; atom_i += blockDim.y)
#else
            for (int atom_i = start[residue_i]; atom_i < end[residue_i];
                 ++atom_i)
#endif
            {
                com = crd[atom_i];
                VECTOR frac = com * rcell;
                frac.x = frac.x - floorf(frac.x);
                frac.y = frac.y - floorf(frac.y);
                frac.z = frac.z - floorf(frac.z);
                VECTOR mapped = frac * cell;
                trans_vec.x = scaler * mapped.x - com.x;
                trans_vec.y = scaler * mapped.y - com.y;
                trans_vec.z = scaler * mapped.z - com.z;
                crd[atom_i] = crd[atom_i] + trans_vec;
            }
        }
    }
}

static __global__ void Map_Center_Of_Mass(
    const int residue_numbers, const int* start, const int* end,
    const VECTOR scaler, const VECTOR* center_of_mass, const LTMatrix3 cell,
    const LTMatrix3 rcell, VECTOR* crd, int* periodicity)
{
    VECTOR trans_vec;
    VECTOR com;
#ifdef USE_GPU
    int residue_i = blockDim.x * blockIdx.x + threadIdx.x;
    if (residue_i < residue_numbers)
#else
    for (int residue_i = 0; residue_i < residue_numbers; ++residue_i)
#endif
    {
        if (periodicity == NULL || periodicity[residue_i] == 0)
        {
            com = center_of_mass[residue_i];
            VECTOR frac = com * rcell;
            frac.x = frac.x - floorf(frac.x);
            frac.y = frac.y - floorf(frac.y);
            frac.z = frac.z - floorf(frac.z);
            VECTOR mapped = frac * cell;
            trans_vec.x = scaler.x * mapped.x - com.x;
            trans_vec.y = scaler.y * mapped.y - com.y;
            trans_vec.z = scaler.z * mapped.z - com.z;
#ifdef USE_GPU
            for (int atom_i = start[residue_i] + threadIdx.y;
                 atom_i < end[residue_i]; atom_i += blockDim.y)
#else
            for (int atom_i = start[residue_i]; atom_i < end[residue_i];
                 ++atom_i)
#endif
            {
                crd[atom_i] = crd[atom_i] + trans_vec;
            }
        }
        else
        {
#ifdef USE_GPU
            for (int atom_i = start[residue_i] + threadIdx.y;
                 atom_i < end[residue_i]; atom_i += blockDim.y)
#else
            for (int atom_i = start[residue_i]; atom_i < end[residue_i];
                 ++atom_i)
#endif
            {
                com = crd[atom_i];
                VECTOR frac = com * rcell;
                frac.x = frac.x - floorf(frac.x);
                frac.y = frac.y - floorf(frac.y);
                frac.z = frac.z - floorf(frac.z);
                VECTOR mapped = frac * cell;
                trans_vec.x = scaler.x * mapped.x - com.x;
                trans_vec.y = scaler.y * mapped.y - com.y;
                trans_vec.z = scaler.z * mapped.z - com.z;
                crd[atom_i] = crd[atom_i] + trans_vec;
            }
        }
    }
}
//--------------------residue functions-------------------------
void MD_INFORMATION::residue_information::Residue_Crd_Map(VECTOR scaler)
{
    Launch_Device_Kernel(Get_Origin, (residue_numbers + 1023) / 1024, 1024, 0,
                         NULL, residue_numbers, d_res_start, d_res_end,
                         md_info->crd, md_info->d_mass, d_mass_inverse,
                         d_center_of_mass);
    dim3 block_res = {64, 16};
    Launch_Device_Kernel(Map_Center_Of_Mass, (residue_numbers + 63) / 64,
                         block_res, 0, NULL, residue_numbers, d_res_start,
                         d_res_end, scaler, d_center_of_mass, md_info->pbc.cell,
                         md_info->pbc.rcell, md_info->crd, (int*)NULL);
}

void MD_INFORMATION::residue_information::Read_AMBER_Parm7(
    const char* file_name, CONTROLLER controller)
{
    FILE* parm = NULL;
    Open_File_Safely(&parm, file_name, "r");
    controller.printf(
        "    Start reading residue informataion from AMBER parm7:\n");

    while (true)
    {
        char temps[CHAR_LENGTH_MAX];
        char temp_first_str[CHAR_LENGTH_MAX];
        char temp_second_str[CHAR_LENGTH_MAX];
        if (fgets(temps, CHAR_LENGTH_MAX, parm) == NULL)
        {
            break;
        }
        if (sscanf(temps, "%s %s", temp_first_str, temp_second_str) != 2)
        {
            continue;
        }
        // read in atomnumber atomljtypenumber
        if (strcmp(temp_first_str, "%FLAG") == 0 &&
            strcmp(temp_second_str, "POINTERS") == 0)
        {
            char* get_ret = fgets(temps, CHAR_LENGTH_MAX, parm);

            int atom_numbers = 0;
            int scanf_ret = fscanf(parm, "%d", &atom_numbers);
            if (scanf_ret != 1)
            {
                controller.Throw_SPONGE_Error(
                    spongeErrorBadFileFormat,
                    "MD_INFORMATION::residue_information::Read_AMBER_Parm7",
                    "Reason:\n\tthe format of the amber_parm7 is not right\n");
            }
            if (md_info->atom_numbers > 0 &&
                md_info->atom_numbers != atom_numbers)
            {
                controller.Throw_SPONGE_Error(
                    spongeErrorConflictingCommand,
                    "MD_INFORMATION::residue_information::Read_AMBER_Parm7",
                    ATOM_NUMBERS_DISMATCH);
            }
            else if (md_info->atom_numbers == 0)
            {
                md_info->atom_numbers = atom_numbers;
            }
            for (int i = 0; i < 10; i = i + 1)
            {
                int lin;
                scanf_ret = fscanf(parm, "%d\n", &lin);
                if (scanf_ret != 1)
                {
                    controller.Throw_SPONGE_Error(
                        spongeErrorBadFileFormat,
                        "MD_INFORMATION::residue_information::Read_AMBER_Parm7",
                        "Reason:\n\tthe format of the amber_parm7 is not "
                        "right\n");
                }
            }
            scanf_ret = fscanf(parm, "%d\n", &this->residue_numbers);  // NRES
            if (scanf_ret != 1)
            {
                controller.Throw_SPONGE_Error(
                    spongeErrorBadFileFormat,
                    "MD_INFORMATION::residue_information::Read_AMBER_Parm7",
                    "Reason:\n\tthe format of the amber_parm7 is not right\n");
            }
            controller.printf("        residue_numbers is %d\n",
                              this->residue_numbers);

            Malloc_Safely((void**)&h_mass,
                          sizeof(float) * this->residue_numbers);
            Malloc_Safely((void**)&h_mass_inverse,
                          sizeof(float) * this->residue_numbers);
            Malloc_Safely((void**)&h_res_start,
                          sizeof(int) * this->residue_numbers);
            Malloc_Safely((void**)&h_res_end,
                          sizeof(int) * this->residue_numbers);
            Malloc_Safely((void**)&h_momentum,
                          sizeof(float) * this->residue_numbers);
            Malloc_Safely((void**)&h_center_of_mass,
                          sizeof(VECTOR) * this->residue_numbers);
            Malloc_Safely((void**)&h_sigma_of_res_ek, sizeof(float));

            Device_Malloc_Safely((void**)&d_mass,
                                 sizeof(float) * this->residue_numbers);
            Device_Malloc_Safely((void**)&d_mass_inverse,
                                 sizeof(float) * this->residue_numbers);
            Device_Malloc_Safely((void**)&d_res_start,
                                 sizeof(int) * this->residue_numbers);
            Device_Malloc_Safely((void**)&d_res_end,
                                 sizeof(int) * this->residue_numbers);
            Device_Malloc_Safely((void**)&d_momentum,
                                 sizeof(float) * this->residue_numbers);
            Device_Malloc_Safely((void**)&d_center_of_mass,
                                 sizeof(VECTOR) * this->residue_numbers);
            Device_Malloc_Safely((void**)&res_ek_energy,
                                 sizeof(float) * this->residue_numbers);
            Device_Malloc_Safely((void**)&sigma_of_res_ek, sizeof(float));
        }  // FLAG POINTERS

        // residue range read
        if (strcmp(temp_first_str, "%FLAG") == 0 &&
            strcmp(temp_second_str, "RESIDUE_POINTER") == 0)
        {
            char* get_ret = fgets(temps, CHAR_LENGTH_MAX, parm);
            // 注意读进来的数的编号要减1
            int* lin_serial;
            Malloc_Safely((void**)&lin_serial,
                          sizeof(int) * this->residue_numbers);
            for (int i = 0; i < this->residue_numbers; i = i + 1)
            {
                int scanf_ret = fscanf(parm, "%d\n", &lin_serial[i]);
                if (scanf_ret != 1)
                {
                    controller.Throw_SPONGE_Error(
                        spongeErrorBadFileFormat,
                        "MD_INFORMATION::residue_information::Read_AMBER_Parm7",
                        "Reason:\n\tthe format of the amber_parm7 is not "
                        "right\n");
                }
            }
            for (int i = 0; i < this->residue_numbers - 1; i = i + 1)
            {
                h_res_start[i] = lin_serial[i] - 1;
                h_res_end[i] = lin_serial[i + 1] - 1;
            }
            h_res_start[this->residue_numbers - 1] =
                lin_serial[this->residue_numbers - 1] - 1;
            h_res_end[this->residue_numbers - 1] =
                md_info->atom_numbers + 1 - 1;

            free(lin_serial);
        }
    }  // while cycle

    deviceMemcpy(this->d_res_start, h_res_start,
                 sizeof(int) * this->residue_numbers, deviceMemcpyHostToDevice);
    deviceMemcpy(this->d_res_end, h_res_end,
                 sizeof(int) * this->residue_numbers, deviceMemcpyHostToDevice);

    controller.printf(
        "    End reading residue informataion from AMBER parm7\n\n");

    fclose(parm);
}

void MD_INFORMATION::residue_information::Initial(CONTROLLER* controller,
                                                  MD_INFORMATION* md_info)
{
    this->md_info = md_info;
    if (!Xponge::system.residues.atom_numbers.empty())
    {
        residue_numbers = (int)Xponge::system.residues.atom_numbers.size();
        if (md_info->atom_numbers == 0)
        {
            for (int atom_number : Xponge::system.residues.atom_numbers)
            {
                md_info->atom_numbers += atom_number;
            }
        }
        controller->printf("    Start reading residue list from Xponge:\n");
        controller->printf("        residue_numbers is %d\n", residue_numbers);
        Malloc_Safely((void**)&h_mass, sizeof(float) * this->residue_numbers);
        Malloc_Safely((void**)&h_mass_inverse,
                      sizeof(float) * this->residue_numbers);
        Malloc_Safely((void**)&h_res_start,
                      sizeof(int) * this->residue_numbers);
        Malloc_Safely((void**)&h_res_end, sizeof(int) * this->residue_numbers);
        Malloc_Safely((void**)&h_momentum,
                      sizeof(float) * this->residue_numbers);
        Malloc_Safely((void**)&h_center_of_mass,
                      sizeof(VECTOR) * this->residue_numbers);
        Malloc_Safely((void**)&h_sigma_of_res_ek, sizeof(float));

        Device_Malloc_Safely((void**)&d_mass,
                             sizeof(float) * this->residue_numbers);
        Device_Malloc_Safely((void**)&d_mass_inverse,
                             sizeof(float) * this->residue_numbers);
        Device_Malloc_Safely((void**)&d_res_start,
                             sizeof(int) * this->residue_numbers);
        Device_Malloc_Safely((void**)&d_res_end,
                             sizeof(int) * this->residue_numbers);
        Device_Malloc_Safely((void**)&d_momentum,
                             sizeof(float) * this->residue_numbers);
        Device_Malloc_Safely((void**)&d_center_of_mass,
                             sizeof(VECTOR) * this->residue_numbers);
        Device_Malloc_Safely((void**)&res_ek_energy,
                             sizeof(float) * this->residue_numbers);
        Device_Malloc_Safely((void**)&sigma_of_res_ek, sizeof(float));

        int count = 0;
        for (int i = 0; i < residue_numbers; i++)
        {
            h_res_start[i] = count;
            count += Xponge::system.residues.atom_numbers[i];
            h_res_end[i] = count;
        }
        if (count != md_info->atom_numbers)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorConflictingCommand,
                "MD_INFORMATION::residue_information::Initial",
                "Reason:\n\tresidue atom counts in Xponge::system do not sum "
                "to the total atom number\n");
        }
        deviceMemcpy(d_res_start, h_res_start, sizeof(int) * residue_numbers,
                     deviceMemcpyHostToDevice);
        deviceMemcpy(d_res_end, h_res_end, sizeof(int) * residue_numbers,
                     deviceMemcpyHostToDevice);
        controller->printf("    End reading residue list from Xponge\n\n");
        is_initialized = 1;
    }
    else
    {
        controller->Throw_SPONGE_Error(
            spongeErrorMissingCommand,
            "MD_INFORMATION::residue_information::Initial",
            "Reason:\n\tno residue information found in Xponge::system\n");
    }
    if (is_initialized)
    {
        if (md_info->h_mass != NULL)
        {
            for (int i = 0; i < residue_numbers; i++)
            {
                float temp_mass = 0;
                for (int j = h_res_start[i]; j < h_res_end[i]; j++)
                {
                    temp_mass += md_info->h_mass[j];
                }
                this->h_mass[i] = temp_mass;
                if (temp_mass == 0)
                    this->h_mass_inverse[i] = 0;
                else
                    this->h_mass_inverse[i] = 1.0 / temp_mass;
            }
            deviceMemcpy(d_mass_inverse, h_mass_inverse,
                         sizeof(float) * residue_numbers,
                         deviceMemcpyHostToDevice);
            deviceMemcpy(d_mass, h_mass, sizeof(float) * residue_numbers,
                         deviceMemcpyHostToDevice);
        }
        else
        {
            controller->Throw_SPONGE_Error(
                spongeErrorMissingCommand,
                "MD_INFORMATION::residue_information::Initial",
                "Reason:\n\tno mass information found");
        }
    }
}

void MD_INFORMATION::residue_information::Split_Disconnected_By_UG_Connectivity(
    const CONECT* connectivity)
{
    if (!is_initialized || residue_numbers == 0 || md_info == NULL ||
        connectivity == NULL)
    {
        return;
    }
    if (h_res_start == NULL || h_res_end == NULL || md_info->h_mass == NULL)
    {
        return;
    }

    std::vector<int> new_res_start;
    std::vector<int> new_res_end;
    new_res_start.reserve(residue_numbers);
    new_res_end.reserve(residue_numbers);

    for (int res_i = 0; res_i < residue_numbers; res_i++)
    {
        int start = h_res_start[res_i];
        int end = h_res_end[res_i];
        int len = end - start;
        if (len <= 1)
        {
            new_res_start.push_back(start);
            new_res_end.push_back(end);
            continue;
        }

        std::vector<int> local_comp(len, -1);
        int comp_count = 0;
        std::deque<int> queue;
        for (int atom_i = start; atom_i < end; atom_i++)
        {
            int local_idx = atom_i - start;
            if (local_comp[local_idx] != -1)
            {
                continue;
            }
            local_comp[local_idx] = comp_count;
            queue.push_back(atom_i);
            while (!queue.empty())
            {
                int atom = queue.front();
                queue.pop_front();
                auto it = connectivity->find(atom);
                if (it == connectivity->end())
                {
                    continue;
                }
                const std::set<int>& neigh = it->second;
                for (int atom_j : neigh)
                {
                    if (atom_j < start || atom_j >= end)
                    {
                        continue;
                    }
                    int local_j = atom_j - start;
                    if (local_comp[local_j] == -1)
                    {
                        local_comp[local_j] = comp_count;
                        queue.push_back(atom_j);
                    }
                }
            }
            comp_count += 1;
        }

        // Note: residues are stored as contiguous atom ranges. If a connected
        // component is interleaved in atom ordering, it will be split into
        // multiple contiguous residue segments here.
        if (comp_count > 1)
        {
            printf(
                "Residue %d is disconnected (components=%d, atoms=%d). "
                "Splitting into contiguous segments.\n",
                res_i, comp_count, len);
        }

        int seg_start = start;
        int seg_comp = local_comp[0];
        for (int offset = 1; offset < len; offset++)
        {
            if (local_comp[offset] != seg_comp)
            {
                new_res_start.push_back(seg_start);
                new_res_end.push_back(start + offset);
                seg_start = start + offset;
                seg_comp = local_comp[offset];
            }
        }
        new_res_start.push_back(seg_start);
        new_res_end.push_back(end);
    }

    if (static_cast<int>(new_res_start.size()) == residue_numbers)
    {
        bool unchanged = true;
        for (int i = 0; i < residue_numbers; i++)
        {
            if (new_res_start[i] != h_res_start[i] ||
                new_res_end[i] != h_res_end[i])
            {
                unchanged = false;
                break;
            }
        }
        if (unchanged)
        {
            return;
        }
    }

    if (h_mass != NULL) free(h_mass);
    if (h_mass_inverse != NULL) free(h_mass_inverse);
    if (h_res_start != NULL) free(h_res_start);
    if (h_res_end != NULL) free(h_res_end);
    if (h_momentum != NULL) free(h_momentum);
    if (h_center_of_mass != NULL) free(h_center_of_mass);
    if (h_sigma_of_res_ek != NULL) free(h_sigma_of_res_ek);

    if (d_res_start != NULL) Free_Single_Device_Pointer((void**)&d_res_start);
    if (d_res_end != NULL) Free_Single_Device_Pointer((void**)&d_res_end);
    if (d_mass != NULL) Free_Single_Device_Pointer((void**)&d_mass);
    if (d_mass_inverse != NULL)
        Free_Single_Device_Pointer((void**)&d_mass_inverse);
    if (d_momentum != NULL) Free_Single_Device_Pointer((void**)&d_momentum);
    if (d_center_of_mass != NULL)
        Free_Single_Device_Pointer((void**)&d_center_of_mass);
    if (res_ek_energy != NULL)
        Free_Single_Device_Pointer((void**)&res_ek_energy);
    if (sigma_of_res_ek != NULL)
        Free_Single_Device_Pointer((void**)&sigma_of_res_ek);

    residue_numbers = static_cast<int>(new_res_start.size());
    if (residue_numbers == 0)
    {
        return;
    }

    Malloc_Safely((void**)&h_mass, sizeof(float) * residue_numbers);
    Malloc_Safely((void**)&h_mass_inverse, sizeof(float) * residue_numbers);
    Malloc_Safely((void**)&h_res_start, sizeof(int) * residue_numbers);
    Malloc_Safely((void**)&h_res_end, sizeof(int) * residue_numbers);
    Malloc_Safely((void**)&h_momentum, sizeof(float) * residue_numbers);
    Malloc_Safely((void**)&h_center_of_mass, sizeof(VECTOR) * residue_numbers);
    Malloc_Safely((void**)&h_sigma_of_res_ek, sizeof(float));

    Device_Malloc_Safely((void**)&d_mass, sizeof(float) * residue_numbers);
    Device_Malloc_Safely((void**)&d_mass_inverse,
                         sizeof(float) * residue_numbers);
    Device_Malloc_Safely((void**)&d_res_start, sizeof(int) * residue_numbers);
    Device_Malloc_Safely((void**)&d_res_end, sizeof(int) * residue_numbers);
    Device_Malloc_Safely((void**)&d_momentum, sizeof(float) * residue_numbers);
    Device_Malloc_Safely((void**)&d_center_of_mass,
                         sizeof(VECTOR) * residue_numbers);
    Device_Malloc_Safely((void**)&res_ek_energy,
                         sizeof(float) * residue_numbers);
    Device_Malloc_Safely((void**)&sigma_of_res_ek, sizeof(float));

    for (int i = 0; i < residue_numbers; i++)
    {
        h_res_start[i] = new_res_start[i];
        h_res_end[i] = new_res_end[i];
        float temp_mass = 0;
        for (int j = h_res_start[i]; j < h_res_end[i]; j++)
        {
            temp_mass += md_info->h_mass[j];
        }
        h_mass[i] = temp_mass;
        if (temp_mass == 0)
            h_mass_inverse[i] = 0;
        else
            h_mass_inverse[i] = 1.0f / temp_mass;
    }

    memset(h_momentum, 0, sizeof(float) * residue_numbers);
    memset(h_center_of_mass, 0, sizeof(VECTOR) * residue_numbers);
    memset(h_sigma_of_res_ek, 0, sizeof(float));

    deviceMemcpy(d_res_start, h_res_start, sizeof(int) * residue_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(d_res_end, h_res_end, sizeof(int) * residue_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(d_mass, h_mass, sizeof(float) * residue_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(d_mass_inverse, h_mass_inverse,
                 sizeof(float) * residue_numbers, deviceMemcpyHostToDevice);
    deviceMemset(d_momentum, 0, sizeof(float) * residue_numbers);
    deviceMemset(d_center_of_mass, 0, sizeof(VECTOR) * residue_numbers);
    deviceMemset(res_ek_energy, 0, sizeof(float) * residue_numbers);
    deviceMemset(sigma_of_res_ek, 0, sizeof(float));
}

//--------------------molecule functions-------------------------

static void Get_Atom_Group_From_Edges(const int atom_numbers, const int* edges,
                                      const int* first_edge,
                                      const int* edge_next,
                                      CPP_ATOM_GROUP& mol_atoms, int* belongs)
{
    std::deque<int> queue;
    std::vector<int> visited(atom_numbers, 0);
    int atom;
    int edge_count;
    for (int i = 0; i < atom_numbers; i++)
    {
        if (!visited[i])
        {
            std::vector<int> atoms;
            visited[i] = 1;
            queue.push_back(i);
            while (!queue.empty())
            {
                atom = queue[0];
                belongs[atom] = mol_atoms.size();
                atoms.push_back(atom);
                queue.pop_front();
                edge_count = first_edge[atom];
                while (edge_count != -1)
                {
                    atom = edges[edge_count];
                    if (!visited[atom])
                    {
                        queue.push_back(atom);
                        visited[atom] = 1;
                    }
                    edge_count = edge_next[edge_count];
                }
            }
            mol_atoms.push_back(atoms);
        }
    }
}

static void Get_Molecule_Atoms(CONTROLLER* controller, int atom_numbers,
                               CONECT connectivity, CPP_ATOM_GROUP& mol_atoms,
                               std::vector<int>& molecule_belongings)
{
    // 分子拓扑是一个无向图，邻接表进行描述
    int edge_numbers = 0;
    for (int i = 0; i < atom_numbers; i++)
    {
        edge_numbers += connectivity[i].size();
    }
    edge_numbers *= 2;
    int* first_edge = NULL;  // 每个原子的第一个边（链表的头）
    int* edges = NULL;       // 每个边的序号
    int* edge_next = NULL;   // 每个原子的边（链表结构）
    Malloc_Safely((void**)&first_edge, sizeof(int) * atom_numbers);
    Malloc_Safely((void**)&edges, sizeof(int) * edge_numbers);
    Malloc_Safely((void**)&edge_next, sizeof(int) * edge_numbers);
    // 初始化链表
    for (int i = 0; i < atom_numbers; i++)
    {
        first_edge[i] = -1;
    }
    int atom_i, atom_j, edge_count = 0;
    for (int atom_i = 0; atom_i < atom_numbers; atom_i++)
    {
        std::set<int> conect_i = connectivity[atom_i];
        for (auto iter = conect_i.begin(); iter != conect_i.end(); iter++)
        {
            atom_j = *iter;
            edge_next[edge_count] = first_edge[atom_i];
            first_edge[atom_i] = edge_count;
            edges[edge_count] = atom_j;
            edge_count++;
        }
    }
    if (controller->Command_Exist("make_output_whole"))
    {
        std::string temp =
            string_strip(controller->Original_Command("make_output_whole"));
        for (std::string aword : string_split(temp, " "))
        {
            std::vector<std::string> atomij =
                string_split(string_strip(aword), "-");
            if (atomij.size() != 2 || !is_str_int(atomij[0].c_str()) ||
                !is_str_int(atomij[1].c_str()))
            {
                controller->Throw_SPONGE_Error(
                    spongeErrorValueErrorCommand,
                    "Move_Crd_Nearest_From_Exclusions_Host",
                    "Reason:\n\t'make_output_whole' should provide atoms in "
                    "the format of atom_i-atom_j");
            }
            atom_i = atoi(atomij[0].c_str());
            atom_j = atoi(atomij[1].c_str());
            edge_next[edge_count] = first_edge[atom_i];
            first_edge[atom_i] = edge_count;
            edges[edge_count] = atom_j;
            edge_count++;
            edge_next[edge_count] = first_edge[atom_j];
            first_edge[atom_j] = edge_count;
            edges[edge_count] = atom_i;
            edge_count++;
        }
    }
    Get_Atom_Group_From_Edges(atom_numbers, edges, first_edge, edge_next,
                              mol_atoms, &molecule_belongings[0]);
    free(first_edge);
    free(edges);
    free(edge_next);
}

static std::vector<int> Check_Periodic_Molecules(CPP_ATOM_GROUP mol_atoms,
                                                 const CONECT connectivity,
                                                 const VECTOR* crd,
                                                 const LTMatrix3 cell,
                                                 const LTMatrix3 rcell)
{
    std::vector<int> periodic_mols;
    int max_atom_idx = -1;
    for (auto& mol : mol_atoms)
    {
        for (int idx : mol)
        {
            if (idx > max_atom_idx) max_atom_idx = idx;
        }
    }
    if (max_atom_idx < 0) return periodic_mols;

    std::vector<VECTOR> crd_orig(max_atom_idx + 1);
    for (int i = 0; i <= max_atom_idx; ++i) crd_orig[i] = crd[i];

    std::vector<int> mark(max_atom_idx + 1, 0);
    std::vector<int> visited(max_atom_idx + 1, 0);
    std::deque<int> queue;

    for (int i = 0; i < mol_atoms.size(); i++)
    {
        auto& atoms = mol_atoms[i];
        if (atoms.empty())
        {
            periodic_mols.push_back(0);
            continue;
        }
        for (int idx : atoms)
        {
            mark[idx] = 1;
            visited[idx] = 0;
        }

        // anchor mapped to primary box
        int anchor = atoms[0];
        VECTOR frac0 = crd_orig[anchor] * rcell;
        frac0.x = frac0.x - floorf(frac0.x);
        frac0.y = frac0.y - floorf(frac0.y);
        frac0.z = frac0.z - floorf(frac0.z);
        VECTOR mapped_anchor = frac0 * cell;

        std::vector<VECTOR> mapped(max_atom_idx + 1);
        mapped[anchor] = mapped_anchor;
        visited[anchor] = 1;
        queue.clear();
        queue.push_back(anchor);

        while (!queue.empty())
        {
            int atom = queue.front();
            queue.pop_front();
            auto itMap = connectivity.find(atom);
            if (itMap == connectivity.end())
            {
                continue;
            }
            for (auto it = itMap->second.begin(); it != itMap->second.end();
                 ++it)
            {
                int nb = *it;
                if (nb < 0 || nb > max_atom_idx || mark[nb] == 0 || visited[nb])
                    continue;
                VECTOR dr = Get_Periodic_Displacement(
                    crd_orig[nb], crd_orig[atom], cell, rcell);
                mapped[nb] = mapped[atom] + dr;
                visited[nb] = 1;
                queue.push_back(nb);
            }
        }

        VECTOR frac_min = {FLT_MAX, FLT_MAX, FLT_MAX};
        VECTOR frac_max = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
        for (int idx : atoms)
        {
            VECTOR frac = mapped[idx] * rcell;
            frac_min.x = fminf(frac_min.x, frac.x);
            frac_min.y = fminf(frac_min.y, frac.y);
            frac_min.z = fminf(frac_min.z, frac.z);
            frac_max.x = fmaxf(frac_max.x, frac.x);
            frac_max.y = fmaxf(frac_max.y, frac.y);
            frac_max.z = fmaxf(frac_max.z, frac.z);
        }
        periodic_mols.push_back((frac_max.x - frac_min.x) >= 1.0f ||
                                (frac_max.y - frac_min.y) >= 1.0f ||
                                (frac_max.z - frac_min.z) >= 1.0f);

        for (int idx : atoms) mark[idx] = 0;
    }
    return periodic_mols;
}

static void Move_Crd_Nearest_From_Connectivity(
    CPP_ATOM_GROUP mol_atoms, const CONECT connectivity, VECTOR* crd,
    const LTMatrix3 cell, const LTMatrix3 rcell,
    std::vector<int> periodic_molecules)
{
    // 复制原始坐标供最小镜像计算
    int max_atom_idx = -1;
    for (auto& mol : mol_atoms)
    {
        for (int idx : mol)
        {
            if (idx > max_atom_idx) max_atom_idx = idx;
        }
    }
    if (max_atom_idx < 0) return;
    std::vector<VECTOR> crd_orig(max_atom_idx + 1);
    for (int i = 0; i <= max_atom_idx; ++i) crd_orig[i] = crd[i];

    std::vector<int> mark(max_atom_idx + 1, 0);
    std::vector<int> visited(max_atom_idx + 1, 0);
    std::deque<int> queue;

    for (int i = 0; i < mol_atoms.size(); i++)
    {
        auto& atoms = mol_atoms[i];
        if (atoms.empty()) continue;
        for (int idx : atoms)
        {
            mark[idx] = 1;
            visited[idx] = 0;
        }
        // 先把第一个原子映射到主盒子
        int anchor = atoms[0];
        VECTOR frac0 = crd_orig[anchor] * rcell;
        frac0.x = frac0.x - floorf(frac0.x);
        frac0.y = frac0.y - floorf(frac0.y);
        frac0.z = frac0.z - floorf(frac0.z);
        crd[anchor] = frac0 * cell;

        visited[anchor] = 1;
        queue.clear();
        queue.push_back(anchor);
        while (!queue.empty())
        {
            int atom = queue.front();
            queue.pop_front();
            auto itMap = connectivity.find(atom);
            if (itMap == connectivity.end())
            {
                continue;
            }
            for (auto it = itMap->second.begin(); it != itMap->second.end();
                 ++it)
            {
                int nb = *it;
                if (nb < 0 || nb > max_atom_idx || mark[nb] == 0 || visited[nb])
                    continue;
                VECTOR dr = Get_Periodic_Displacement(
                    crd_orig[nb], crd_orig[atom], cell, rcell);
                crd[nb] = crd[atom] + dr;
                visited[nb] = 1;
                queue.push_back(nb);
            }
        }
        for (int idx : atoms) mark[idx] = 0;
    }
}

void MD_INFORMATION::molecule_information::Initial(CONTROLLER* controller)
{
    if (!md_info->pbc.pbc) return;
    // 分子拓扑是一个无向图，邻接表进行描述，通过排除表形成
    CPP_ATOM_GROUP mol_atoms;
    std::vector<int> molecule_belongings(md_info->atom_numbers, 0);
    md_info->res.Split_Disconnected_By_UG_Connectivity(
        &md_info->sys.connectivity);
    Get_Molecule_Atoms(controller, md_info->atom_numbers,
                       md_info->sys.connectivity, mol_atoms,
                       molecule_belongings);
    molecule_numbers = mol_atoms.size();
    Move_Crd_Nearest_From_Connectivity(mol_atoms, md_info->sys.connectivity,
                                       md_info->coordinate, md_info->pbc.cell,
                                       md_info->pbc.rcell, std::vector<int>());
    std::vector<int> h_periodicity = Check_Periodic_Molecules(
        mol_atoms, md_info->sys.connectivity, md_info->coordinate,
        md_info->pbc.cell, md_info->pbc.rcell);
    Device_Malloc_And_Copy_Safely((void**)&d_periodicity, &h_periodicity[0],
                                  sizeof(int) * molecule_numbers);
    deviceMemcpy(md_info->crd, md_info->coordinate,
                 sizeof(VECTOR) * md_info->atom_numbers,
                 deviceMemcpyHostToDevice);

    Malloc_Safely((void**)&h_mass, sizeof(float) * molecule_numbers);
    Malloc_Safely((void**)&h_mass_inverse, sizeof(float) * molecule_numbers);
    Malloc_Safely((void**)&h_atom_start, sizeof(int) * molecule_numbers);
    Malloc_Safely((void**)&h_atom_end, sizeof(int) * molecule_numbers);
    Malloc_Safely((void**)&h_residue_start, sizeof(int) * molecule_numbers);
    Malloc_Safely((void**)&h_residue_end, sizeof(int) * molecule_numbers);
    Malloc_Safely((void**)&h_center_of_mass, sizeof(VECTOR) * molecule_numbers);

    Device_Malloc_Safely((void**)&d_mass, sizeof(float) * molecule_numbers);
    Device_Malloc_Safely((void**)&d_mass_inverse,
                         sizeof(float) * molecule_numbers);
    Device_Malloc_Safely((void**)&d_atom_start, sizeof(int) * molecule_numbers);
    Device_Malloc_Safely((void**)&d_atom_end, sizeof(int) * molecule_numbers);
    Device_Malloc_Safely((void**)&d_residue_start,
                         sizeof(int) * molecule_numbers);
    Device_Malloc_Safely((void**)&d_residue_end,
                         sizeof(int) * molecule_numbers);
    Device_Malloc_Safely((void**)&d_center_of_mass,
                         sizeof(VECTOR) * molecule_numbers);

    int molecule_j = 0;
    h_atom_start[0] = 0;
    // 该判断基于一个分子的所有原子一定在列表里是连续的
    for (int i = 0; i < md_info->atom_numbers; i++)
    {
        if (molecule_belongings[i] != molecule_j)
        {
            if (molecule_belongings[i] < molecule_j)
            {
                char error_reason[CHAR_LENGTH_MAX];
                sprintf(
                    error_reason,
                    "Reason:\n\tthe indexes of atoms in the same one molecule "
                    "should be continous, and atom #%d is not right\n",
                    i);
                controller->Throw_SPONGE_Error(
                    spongeErrorValueErrorCommand,
                    "MD_INFORMATION::molecule_information::Initial",
                    error_reason);
            }
            h_atom_end[molecule_j] = i;
            molecule_j += 1;
            if (molecule_j < molecule_numbers) h_atom_start[molecule_j] = i;
        }
    }
    h_atom_end[molecule_numbers - 1] = md_info->atom_numbers;

    molecule_j = 0;
    h_residue_start[0] = 0;
    // 该判断基于一个分子的所有残基一定在列表里是连续的，且原子在残基里也是连续的
    for (int i = 0; i < md_info->res.residue_numbers; i++)
    {
        if (md_info->res.h_res_start[i] == h_atom_end[molecule_j])
        {
            h_residue_end[molecule_j] = i;
            molecule_j += 1;
            if (molecule_j < molecule_numbers) h_residue_start[molecule_j] = i;
        }
    }
    h_residue_end[molecule_numbers - 1] = md_info->res.residue_numbers;

    for (int i = 0; i < molecule_numbers; i++)
    {
        h_mass[i] = 0;
        for (molecule_j = h_atom_start[i]; molecule_j < h_atom_end[i];
             molecule_j++)
        {
            h_mass[i] += md_info->h_mass[molecule_j];
        }
        h_mass_inverse[i] = 1.0f / h_mass[i];
    }

    deviceMemcpy(d_mass, h_mass, sizeof(float) * molecule_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(d_mass_inverse, h_mass_inverse,
                 sizeof(float) * molecule_numbers, deviceMemcpyHostToDevice);
    deviceMemcpy(d_atom_start, h_atom_start, sizeof(int) * molecule_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(d_atom_end, h_atom_end, sizeof(int) * molecule_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(d_residue_start, h_residue_start,
                 sizeof(int) * molecule_numbers, deviceMemcpyHostToDevice);
    deviceMemcpy(d_residue_end, h_residue_end, sizeof(int) * molecule_numbers,
                 deviceMemcpyHostToDevice);
    if (md_info->pbc.pbc)
    {
        controller->printf("    Molecule numbers: %d\n",
                           md_info->mol.molecule_numbers);
    }
}

void MD_INFORMATION::molecule_information::Molecule_Crd_Map(float scaler)
{
    // 为了有一个分子有很多残基，而其他分子都很小这种情况的并行，先求残基的质心
    Launch_Device_Kernel(
        Get_Origin, (md_info->res.residue_numbers + 1023) / 1024, 1024, 0, NULL,
        md_info->res.residue_numbers, md_info->res.d_res_start,
        md_info->res.d_res_end, md_info->crd, md_info->d_mass,
        md_info->res.d_mass_inverse, md_info->res.d_center_of_mass);
    // 再用残基的质心求分子的质心
    Launch_Device_Kernel(Get_Origin, (molecule_numbers + 1023) / 1024, 1024, 0,
                         NULL, molecule_numbers, d_residue_start, d_residue_end,
                         md_info->res.d_center_of_mass, md_info->res.d_mass,
                         d_mass_inverse, d_center_of_mass);

    dim3 block_mol = {64, 16};
    Launch_Device_Kernel(
        Map_Center_Of_Mass, (molecule_numbers + 63) / 64, block_mol, 0, NULL,
        molecule_numbers, d_atom_start, d_atom_end, scaler, d_center_of_mass,
        md_info->pbc.cell, md_info->pbc.rcell, md_info->crd, d_periodicity);
}

void MD_INFORMATION::molecule_information::Molecule_Crd_Map(VECTOR scaler)
{
    // 为了有一个分子有很多残基，而其他分子都很小这种情况的并行，先求残基的质心
    Launch_Device_Kernel(
        Get_Origin, (md_info->res.residue_numbers + 1023) / 1024, 1024, 0, NULL,
        md_info->res.residue_numbers, md_info->res.d_res_start,
        md_info->res.d_res_end, md_info->crd, md_info->d_mass,
        md_info->res.d_mass_inverse, md_info->res.d_center_of_mass);
    // 再用残基的质心求分子的质心
    Launch_Device_Kernel(Get_Origin, (molecule_numbers + 1023) / 1024, 1024, 0,
                         NULL, molecule_numbers, d_residue_start, d_residue_end,
                         md_info->res.d_center_of_mass, md_info->res.d_mass,
                         d_mass_inverse, d_center_of_mass);
    dim3 block_mol = {64, 16};
    Launch_Device_Kernel(
        Map_Center_Of_Mass, (molecule_numbers + 63) / 64, block_mol, 0, NULL,
        molecule_numbers, d_atom_start, d_atom_end, scaler, d_center_of_mass,
        md_info->pbc.cell, md_info->pbc.rcell, md_info->crd, d_periodicity);
}

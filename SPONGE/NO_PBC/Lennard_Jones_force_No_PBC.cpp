#include "Lennard_Jones_force_No_PBC.h"

#include "../xponge/load/native/lj.hpp"
#include "../xponge/xponge.h"

static __global__ void LJ_Force_Device(const int atom_numbers,
                                       const VECTOR* crd, const int* LJ_types,
                                       const float* LJ_A, const float* LJ_B,
                                       const int* excluded_list_start,
                                       const int* excluded_list,
                                       const int* excluded_atom_numbers,
                                       const float cutoff_square, VECTOR* frc)
{
#ifdef USE_GPU
    int atom_i = blockDim.y * blockIdx.y + threadIdx.y;
    int atom_j = atom_i + 1 + blockDim.x * blockIdx.x + threadIdx.x;
    if (atom_i < atom_numbers && atom_j < atom_numbers)
#else
#pragma omp parallel for
    for (int atom_i = 0; atom_i < atom_numbers; atom_i++)
        for (int atom_j = atom_i + 1; atom_j < atom_numbers; atom_j++)
#endif
    {
        int tocal = 1;
        const int* start = excluded_list + excluded_list_start[atom_i];
        for (int k = 0; tocal == 1 && k < excluded_atom_numbers[atom_i]; k += 1)
        {
            if (start[k] == atom_j) tocal = 0;
        }
        if (tocal == 1)
        {
            VECTOR dr = crd[atom_j] - crd[atom_i];
            float dr2 = dr * dr;
            if (dr2 < cutoff_square)
            {
                float dr_2 = 1. / dr2;
                float dr_4 = dr_2 * dr_2;
                float dr_6 = dr_4 * dr_2;
                float dr_8 = dr_4 * dr_4;

                int type_i = LJ_types[atom_i];
                int type_j = LJ_types[atom_j];
                int type_ij = type_i;
                if (type_i < type_j)
                {
                    type_i = type_j;
                    type_j = type_ij;
                }
                type_ij = (type_i * (type_i + 1) / 2) + type_j;
                float frc_abs = (-LJ_A[type_ij] * dr_6 + LJ_B[type_ij]) * dr_8;
                VECTOR temp_frc = frc_abs * dr;

                atomicAdd(&frc[atom_j].x, -temp_frc.x);
                atomicAdd(&frc[atom_j].y, -temp_frc.y);
                atomicAdd(&frc[atom_j].z, -temp_frc.z);
                atomicAdd(&frc[atom_i].x, temp_frc.x);
                atomicAdd(&frc[atom_i].y, temp_frc.y);
                atomicAdd(&frc[atom_i].z, temp_frc.z);
            }
        }
    }
}

static __global__ void LJ_Force_Energy_Device(
    const int atom_numbers, const VECTOR* crd, const int* LJ_types,
    const float* LJ_A, const float* LJ_B, const int* excluded_list_start,
    const int* excluded_list, const int* excluded_atom_numbers,
    const float cutoff_square, float* atom_ene, VECTOR* frc, float* this_ene)
{
#ifdef USE_GPU
    int atom_i = blockDim.y * blockIdx.y + threadIdx.y;
    int atom_j = atom_i + 1 + blockDim.x * blockIdx.x + threadIdx.x;
    if (atom_i < atom_numbers && atom_j < atom_numbers)
#else
#pragma omp parallel for
    for (int atom_i = 0; atom_i < atom_numbers; atom_i++)
        for (int atom_j = atom_i + 1; atom_j < atom_numbers; atom_j++)
#endif
    {
        int tocal = 1;
        const int* start = excluded_list + excluded_list_start[atom_i];
        for (int k = 0; tocal == 1 && k < excluded_atom_numbers[atom_i]; k += 1)
        {
            if (start[k] == atom_j) tocal = 0;
        }
        if (tocal == 1)
        {
            VECTOR dr = crd[atom_j] - crd[atom_i];
            float dr2 = dr * dr;
            if (dr2 < cutoff_square)
            {
                float dr_2 = 1. / dr2;
                float dr_4 = dr_2 * dr_2;
                float dr_6 = dr_4 * dr_2;
                float dr_8 = dr_4 * dr_4;

                int type_i = LJ_types[atom_i];
                int type_j = LJ_types[atom_j];
                int type_ij = type_i;
                if (type_i < type_j)
                {
                    type_i = type_j;
                    type_j = type_ij;
                }
                type_ij = (type_i * (type_i + 1) / 2) + type_j;
                float temp_ene = (0.083333333 * LJ_A[type_ij] * dr_6 -
                                  0.166666666 * LJ_B[type_ij]) *
                                 dr_6;
                float frc_abs = (-LJ_A[type_ij] * dr_6 + LJ_B[type_ij]) * dr_8;
                VECTOR temp_frc = frc_abs * dr;
                atomicAdd(&frc[atom_j].x, -temp_frc.x);
                atomicAdd(&frc[atom_j].y, -temp_frc.y);
                atomicAdd(&frc[atom_j].z, -temp_frc.z);
                atomicAdd(&frc[atom_i].x, temp_frc.x);
                atomicAdd(&frc[atom_i].y, temp_frc.y);
                atomicAdd(&frc[atom_i].z, temp_frc.z);

                atomicAdd(&atom_ene[atom_i], temp_ene);
                atomicAdd(&this_ene[atom_i], temp_ene);
            }
        }
    }
}

void LENNARD_JONES_NO_PBC_INFORMATION::LJ_Malloc()
{
    Malloc_Safely((void**)&h_LJ_energy_atom, sizeof(float) * atom_numbers);
    Malloc_Safely((void**)&h_atom_LJ_type, sizeof(int) * atom_numbers);
    Malloc_Safely((void**)&h_LJ_A, sizeof(float) * pair_type_numbers);
    Malloc_Safely((void**)&h_LJ_B, sizeof(float) * pair_type_numbers);

    Device_Malloc_And_Copy_Safely((void**)&d_LJ_energy_sum, &h_LJ_energy_sum,
                                  sizeof(float));
    Device_Malloc_And_Copy_Safely((void**)&d_LJ_energy_atom, h_LJ_energy_atom,
                                  sizeof(float) * atom_numbers);
    Device_Malloc_And_Copy_Safely((void**)&d_atom_LJ_type, h_atom_LJ_type,
                                  sizeof(int) * atom_numbers);
    Device_Malloc_And_Copy_Safely((void**)&d_LJ_A, h_LJ_A,
                                  sizeof(float) * pair_type_numbers);
    Device_Malloc_And_Copy_Safely((void**)&d_LJ_B, h_LJ_B,
                                  sizeof(float) * pair_type_numbers);
}

void LENNARD_JONES_NO_PBC_INFORMATION::Initial(CONTROLLER* controller,
                                               float cutoff,
                                               const char* module_name)
{
    if (module_name == NULL)
    {
        strcpy(this->module_name, "LJ");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }
    controller[0].printf("START INITIALIZING LENNADR JONES INFORMATION:\n");
    this->cutoff = cutoff;
    const auto& system_lj = Xponge::system.classical_force_field.lj;
    Xponge::LennardJones local_lj;
    const Xponge::LennardJones* lj_to_use = NULL;
    if (module_name == NULL)
    {
        lj_to_use = &system_lj;
    }
    else if (controller[0].Command_Exist(this->module_name, "in_file"))
    {
        Xponge::Native_Load_LJ(&local_lj, controller, 0, this->module_name);
        lj_to_use = &local_lj;
    }
    if (lj_to_use != NULL && !lj_to_use->atom_type.empty())
    {
        atom_numbers = static_cast<int>(lj_to_use->atom_type.size());
        atom_type_numbers = lj_to_use->atom_type_numbers;
        controller[0].printf("    atom_numbers is %d\n", atom_numbers);
        controller[0].printf("    atom_LJ_type_number is %d\n",
                             atom_type_numbers);
        pair_type_numbers = atom_type_numbers * (atom_type_numbers + 1) / 2;
        LJ_Malloc();

        for (int i = 0; i < pair_type_numbers; i++)
        {
            h_LJ_A[i] = lj_to_use->pair_A[i];
        }
        for (int i = 0; i < pair_type_numbers; i++)
        {
            h_LJ_B[i] = lj_to_use->pair_B[i];
        }
        for (int i = 0; i < atom_numbers; i++)
        {
            h_atom_LJ_type[i] = lj_to_use->atom_type[i];
        }
        Parameter_Host_To_Device();
        is_initialized = 1;
    }
    else if (controller[0].Command_Exist("amber_parm7"))
    {
        Initial_From_AMBER_Parm(controller[0].Command("amber_parm7"),
                                controller[0]);
    }
    if (is_initialized && !is_controller_printf_initialized)
    {
        controller[0].Step_Print_Initial(this->module_name, "%.2f");
        is_controller_printf_initialized = 1;
        controller[0].printf("    structure last modify date is %d\n",
                             last_modify_date);
    }
    controller[0].printf("END INITIALIZING LENNADR JONES INFORMATION\n\n");
}

void LENNARD_JONES_NO_PBC_INFORMATION::Initial_From_AMBER_Parm(
    const char* file_name, CONTROLLER controller)
{
    FILE* parm = NULL;
    Open_File_Safely(&parm, file_name, "r");
    controller.printf("    Start reading LJ information from AMBER file:\n");

    while (true)
    {
        char temps[CHAR_LENGTH_MAX];
        char temp_first_str[CHAR_LENGTH_MAX];
        char temp_second_str[CHAR_LENGTH_MAX];
        if (!fgets(temps, CHAR_LENGTH_MAX, parm))
        {
            break;
        }
        if (sscanf(temps, "%s %s", temp_first_str, temp_second_str) != 2)
        {
            continue;
        }
        if (strcmp(temp_first_str, "%FLAG") == 0 &&
            strcmp(temp_second_str, "POINTERS") == 0)
        {
            char* get_ret = fgets(temps, CHAR_LENGTH_MAX, parm);

            int scanf_ret = fscanf(parm, "%d\n", &atom_numbers);
            controller.printf("        atom_numbers is %d\n", atom_numbers);
            scanf_ret = fscanf(parm, "%d\n", &atom_type_numbers);
            controller.printf("        atom_LJ_type_number is %d\n",
                              atom_type_numbers);
            pair_type_numbers = atom_type_numbers * (atom_type_numbers + 1) / 2;

            LJ_Malloc();
        }
        if (strcmp(temp_first_str, "%FLAG") == 0 &&
            strcmp(temp_second_str, "ATOM_TYPE_INDEX") == 0)
        {
            char* get_ret = fgets(temps, CHAR_LENGTH_MAX, parm);
            printf("        read atom LJ type index\n");
            int atomljtype;
            for (int i = 0; i < atom_numbers; i = i + 1)
            {
                int scanf_ret = fscanf(parm, "%d\n", &atomljtype);
                h_atom_LJ_type[i] = atomljtype - 1;
            }
        }
        if (strcmp(temp_first_str, "%FLAG") == 0 &&
            strcmp(temp_second_str, "LENNARD_JONES_ACOEF") == 0)
        {
            char* get_ret = fgets(temps, CHAR_LENGTH_MAX, parm);
            printf("        read atom LJ A\n");
            double lin;
            for (int i = 0; i < pair_type_numbers; i = i + 1)
            {
                int scanf_ret = fscanf(parm, "%lf\n", &lin);
                h_LJ_A[i] = (float)12. * lin;
            }
        }
        if (strcmp(temp_first_str, "%FLAG") == 0 &&
            strcmp(temp_second_str, "LENNARD_JONES_BCOEF") == 0)
        {
            char* get_ret = fgets(temps, CHAR_LENGTH_MAX, parm);
            printf("        read atom LJ B\n");
            double lin;
            for (int i = 0; i < pair_type_numbers; i = i + 1)
            {
                int scanf_ret = fscanf(parm, "%lf\n", &lin);
                h_LJ_B[i] = (float)6. * lin;
            }
        }
    }
    controller.printf("    End reading LJ information from AMBER file:\n");
    fclose(parm);
    is_initialized = 1;
    Parameter_Host_To_Device();
}

void LENNARD_JONES_NO_PBC_INFORMATION::Parameter_Host_To_Device()
{
    deviceMemcpy(d_LJ_B, h_LJ_B, sizeof(float) * pair_type_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(d_LJ_A, h_LJ_A, sizeof(float) * pair_type_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(d_atom_LJ_type, h_atom_LJ_type, sizeof(int) * atom_numbers,
                 deviceMemcpyHostToDevice);
}

void LENNARD_JONES_NO_PBC_INFORMATION::LJ_Force_With_Atom_Energy(
    const int atom_numbers, const VECTOR* crd, VECTOR* frc,
    const int need_atom_energy, float* atom_energy,
    const int* excluded_list_start, const int* excluded_list,
    const int* excluded_atom_numbers)
{
    if (is_initialized)
    {
        dim3 blockSize = {
            CONTROLLER::device_warp,
            CONTROLLER::device_max_thread / CONTROLLER::device_warp};
        dim3 gridSize = {(atom_numbers + blockSize.x - 1) / blockSize.x,
                         (atom_numbers + blockSize.y - 1) / blockSize.y};
        if (!need_atom_energy)
        {
            Launch_Device_Kernel(LJ_Force_Device, gridSize, blockSize, 0, NULL,
                                 atom_numbers, crd, d_atom_LJ_type, d_LJ_A,
                                 d_LJ_B, excluded_list_start, excluded_list,
                                 excluded_atom_numbers, cutoff * cutoff, frc);
        }
        else
        {
            deviceMemset(d_LJ_energy_atom, 0, sizeof(float) * atom_numbers);
            Launch_Device_Kernel(
                LJ_Force_Energy_Device, gridSize, blockSize, 0, NULL,
                atom_numbers, crd, d_atom_LJ_type, d_LJ_A, d_LJ_B,
                excluded_list_start, excluded_list, excluded_atom_numbers,
                cutoff * cutoff, atom_energy, frc, d_LJ_energy_atom);
        }
    }
}

void LENNARD_JONES_NO_PBC_INFORMATION::Step_Print(CONTROLLER* controller)
{
    if (is_initialized)
    {
        Sum_Of_List(d_LJ_energy_atom, d_LJ_energy_sum, atom_numbers);
        deviceMemcpy(&h_LJ_energy_sum, d_LJ_energy_sum, sizeof(float),
                     deviceMemcpyDeviceToHost);
        controller->Step_Print("LJ", h_LJ_energy_sum, true);
    }
}

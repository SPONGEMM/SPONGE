#include "Coulomb_Force_No_PBC.h"

#include "exclusions.hpp"

static __global__ void Coulomb_Force_Device(
    const int atom_numbers, const VECTOR* crd, const float* charge,
    const int* excluded_list_start, const int* excluded_list,
    const int* excluded_atom_numbers, const float cutoff_square, VECTOR* frc)
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
        if (!No_PBC_Pair_Excluded(atom_i, atom_j, excluded_list_start,
                                  excluded_list, excluded_atom_numbers))
        {
            VECTOR dr = crd[atom_j] - crd[atom_i];
            float dr2 = dr * dr;
            if (dr2 < cutoff_square)
            {
                float dr_2 = 1. / dr2;
                float dr_1 = sqrtf(dr_2);
                float dr_3 = dr_1 * dr_2;
                float chargeij = charge[atom_i] * charge[atom_j];
                float frc_abs = -chargeij * dr_3;
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

static __global__ void Coulomb_Force_Energy_Device(
    const int atom_numbers, const VECTOR* crd, const float* charge,
    const int* excluded_list_start, const int* excluded_list,
    const int* excluded_atom_numbers, const float cutoff_square,
    float* atom_ene, VECTOR* frc, float* this_ene)
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
        if (!No_PBC_Pair_Excluded(atom_i, atom_j, excluded_list_start,
                                  excluded_list, excluded_atom_numbers))
        {
            VECTOR dr = crd[atom_j] - crd[atom_i];
            float dr2 = dr * dr;
            if (dr2 < cutoff_square)
            {
                float dr_2 = 1. / dr2;
                float dr_1 = sqrtf(dr_2);
                float dr_3 = dr_1 * dr_2;
                float chargeij = charge[atom_i] * charge[atom_j];
                float temp_ene = chargeij * dr_1;
                float frc_abs = -chargeij * dr_3;
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

void COULOMB_FORCE_NO_PBC_INFORMATION::Malloc()
{
    Malloc_Safely((void**)&h_Coulomb_energy_atom, sizeof(float) * atom_numbers);
    Device_Malloc_And_Copy_Safely((void**)&d_Coulomb_energy_sum,
                                  h_Coulomb_energy_atom, sizeof(float));
    Device_Malloc_Safely((void**)&d_Coulomb_energy_atom,
                         sizeof(float) * atom_numbers);
}

void COULOMB_FORCE_NO_PBC_INFORMATION::Initial(CONTROLLER* controller,
                                               int atom_numbers, float cutoff,
                                               const char* module_name)
{
    if (module_name == NULL)
    {
        strcpy(this->module_name, "Coulomb");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }
    controller[0].printf("START INITIALIZING COULOMB INFORMATION:\n");
    this->cutoff = cutoff;
    this->atom_numbers = atom_numbers;
    this->is_initialized = 1;
    Malloc();
    if (is_initialized && !is_controller_printf_initialized)
    {
        controller[0].Step_Print_Initial(this->module_name, "%.2f");
        is_controller_printf_initialized = 1;
        controller[0].printf("    structure last modify date is %d\n",
                             last_modify_date);
    }
    controller[0].printf("END INITIALIZING COULOMB INFORMATION\n\n");
}

void COULOMB_FORCE_NO_PBC_INFORMATION::Coulomb_Force_With_Atom_Energy(
    const int atom_numbers, const VECTOR* crd, const float* charge, VECTOR* frc,
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
        if (need_atom_energy == 0)
        {
            Launch_Device_Kernel(Coulomb_Force_Device, gridSize, blockSize, 0,
                                 NULL, atom_numbers, crd, charge,
                                 excluded_list_start, excluded_list,
                                 excluded_atom_numbers, cutoff * cutoff, frc);
        }
        else
        {
            deviceMemset(d_Coulomb_energy_atom, 0,
                         sizeof(float) * atom_numbers);
            Launch_Device_Kernel(Coulomb_Force_Energy_Device, gridSize,
                                 blockSize, 0, NULL, atom_numbers, crd, charge,
                                 excluded_list_start, excluded_list,
                                 excluded_atom_numbers, cutoff * cutoff,
                                 atom_energy, frc, d_Coulomb_energy_atom);
        }
    }
}

void COULOMB_FORCE_NO_PBC_INFORMATION::Step_Print(CONTROLLER* controller)
{
    if (!is_initialized) return;
    Sum_Of_List(d_Coulomb_energy_atom, d_Coulomb_energy_sum, atom_numbers);
    deviceMemcpy(&h_Coulomb_energy_sum, d_Coulomb_energy_sum, sizeof(float),
                 deviceMemcpyDeviceToHost);
    controller->Step_Print("Coulomb", h_Coulomb_energy_sum, true);
}

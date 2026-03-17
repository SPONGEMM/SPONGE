#include "nb14.h"

#include "../xponge/load/native/nb14.hpp"
#include "../xponge/xponge.h"
#define TINY 1e-10

static __global__ void
Dihedral_14_LJ_CF_Force_With_Atom_Energy_And_Virial_Device(
    const int dihedral_14_numbers, const VECTOR* crd, const LTMatrix3 cell,
    const LTMatrix3 rcell, const int local_atom_numbers, const int* a_14,
    const int* b_14, const float* cf_scale_factor, const float* charge,
    const float* lj_A, const float* lj_B, VECTOR* frc, int need_atom_energy,
    float* atom_energy, int need_virial, LTMatrix3* atom_virial,
    float* nb14_cf_ene, float* nb14_lj_ene)
{
#ifdef USE_GPU
    int dihedral_14_i = blockDim.x * blockIdx.x + threadIdx.x;
    if (dihedral_14_i < dihedral_14_numbers)
#else
#pragma omp parallel for
    for (int dihedral_14_i = 0; dihedral_14_i < dihedral_14_numbers;
         dihedral_14_i++)
#endif
    {
        VECTOR r1, r2, dr, temp_frc;
        float dr_abs;
        float dr2;
        float dr_1;
        float dr_2;
        float dr_4;
        float dr_8;
        float dr_14;
        float frc_abs = 0.;

        float ene_lin;
        float ene_lin2;

        int atom_i = a_14[dihedral_14_i];
        int atom_j = b_14[dihedral_14_i];

        r1 = crd[atom_i];
        r2 = crd[atom_j];

        dr = Get_Periodic_Displacement(r2, r1, cell, rcell);

        dr2 = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;

        dr_2 = 1.0 / dr2;
        dr_4 = dr_2 * dr_2;
        dr_8 = dr_4 * dr_4;
        dr_14 = dr_8 * dr_4 * dr_2;
        dr_abs = norm3df(dr.x, dr.y, dr.z);
        dr_1 = 1. / dr_abs;
        // CF
        float charge_i = charge[atom_i];
        if (fabs(charge_i) < TINY) charge_i = TINY;
        float charge_j = charge[atom_j];
        if (fabs(charge_j) < TINY) charge_j = TINY;
        float frc_cf_abs;
        frc_cf_abs = cf_scale_factor[dihedral_14_i] * dr_2 * dr_1;
        frc_cf_abs = -frc_cf_abs * charge_i * charge_j;
        // LJ
        frc_abs = -lj_A[dihedral_14_i] * dr_14 + lj_B[dihedral_14_i] * dr_8;

        frc_abs += frc_cf_abs;
        temp_frc.x = frc_abs * dr.x;
        temp_frc.y = frc_abs * dr.y;
        temp_frc.z = frc_abs * dr.z;

        if (atom_j < local_atom_numbers)
        {
            atomicAdd(&frc[atom_j].x, -temp_frc.x);
            atomicAdd(&frc[atom_j].y, -temp_frc.y);
            atomicAdd(&frc[atom_j].z, -temp_frc.z);
        }
        if (atom_i < local_atom_numbers)
        {
            atomicAdd(&frc[atom_i].x, temp_frc.x);
            atomicAdd(&frc[atom_i].y, temp_frc.y);
            atomicAdd(&frc[atom_i].z, temp_frc.z);
            if (need_atom_energy)
            {
                // 能量
                ene_lin =
                    cf_scale_factor[dihedral_14_i] * dr_1 * charge_i * charge_j;
                ene_lin2 = 0.08333333 * lj_A[dihedral_14_i] * dr_4 * dr_8 -
                           0.1666666 * lj_B[dihedral_14_i] * dr_4 *
                               dr_2;  // LJ的A,B系数已经乘以12和6因此要反乘
                atomicAdd(atom_energy + atom_i, ene_lin + ene_lin2);
                nb14_cf_ene[dihedral_14_i] = ene_lin;
                nb14_lj_ene[dihedral_14_i] = ene_lin2;
            }
            // 维里
            if (need_virial)
            {
                atomicAdd(atom_virial + atom_i,
                          Get_Virial_From_Force_Dis(-temp_frc, dr));
            }
        }
    }
}

void NON_BOND_14::Initial(CONTROLLER* controller, const float* LJ_type_A,
                          const float* LJ_type_B, const int* lj_atom_type,
                          const char* module_name)
{
    if (module_name == NULL)
    {
        strcpy(this->module_name, "nb14");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }

    print_lj_name = this->module_name;
    print_lj_name += "_LJ";
    print_ee_name = this->module_name;
    print_ee_name += "_EE";
    char file_name_suffix[CHAR_LENGTH_MAX];
    const auto& nb14 = Xponge::system.classical_force_field.nb14;
    Xponge::NB14 local_nb14;
    const Xponge::NB14* nb14_to_use = NULL;
    const char* init_source = NULL;

    sprintf(file_name_suffix, "in_file");
    if (module_name == NULL)
    {
        nb14_to_use = &nb14;
        init_source = "Xponge::system";
    }
    else if (controller->Command_Exist(this->module_name, file_name_suffix))
    {
        Xponge::Native_Load_NB14(&local_nb14, lj_atom_type, LJ_type_A,
                                 LJ_type_B, controller, this->module_name);
        nb14_to_use = &local_nb14;
    }

    if (nb14_to_use != NULL)
    {
        nb14_numbers = static_cast<int>(nb14_to_use->atom_a.size());
    }
    if (nb14_numbers > 0)
    {
        if (module_name == NULL)
        {
            controller->printf("START INITIALIZING NB14 (%s):\n", init_source);
        }
        else
        {
            controller->printf("START INITIALIZING NB14 (%s_%s):\n",
                               this->module_name, file_name_suffix);
        }
        controller->printf("    non-bond 14 numbers is %d\n", nb14_numbers);
        Memory_Allocate();
        for (int i = 0; i < nb14_numbers; i++)
        {
            h_atom_a[i] = nb14_to_use->atom_a[i];
            h_atom_b[i] = nb14_to_use->atom_b[i];
            h_A[i] = nb14_to_use->A[i];
            h_B[i] = nb14_to_use->B[i];
            h_cf_scale_factor[i] = nb14_to_use->cf_scale_factor[i];
        }
        Parameter_Host_To_Device();
        is_initialized = 1;
    }
    else
    {
        controller->printf("NB14 IS NOT INITIALIZED\n\n");
    }

    if (is_initialized && !is_controller_printf_initialized)
    {
        controller->Step_Print_Initial(print_lj_name.c_str(), "%.2f");
        controller->Step_Print_Initial(print_ee_name.c_str(), "%.2f");
        is_controller_printf_initialized = 1;
        controller->printf("    structure last modify date is %d\n",
                           last_modify_date);
    }
    if (is_initialized)
    {
        controller->printf("END INITIALIZING NB14\n\n");
    }
}

void NON_BOND_14::Memory_Allocate()
{
    Malloc_Safely((void**)&h_atom_a, sizeof(int) * nb14_numbers);
    Malloc_Safely((void**)&h_atom_b, sizeof(int) * nb14_numbers);

    Malloc_Safely((void**)&h_A, sizeof(float) * nb14_numbers);
    Malloc_Safely((void**)&h_B, sizeof(float) * nb14_numbers);
    Malloc_Safely((void**)&h_cf_scale_factor, sizeof(float) * nb14_numbers);

    Malloc_Safely((void**)&h_nb14_lj_energy_sum, sizeof(float));
    Malloc_Safely((void**)&h_nb14_cf_energy_sum, sizeof(float));
    memset(h_nb14_lj_energy_sum, 0, sizeof(float));
    memset(h_nb14_cf_energy_sum, 0, sizeof(float));
}

void NON_BOND_14::Parameter_Host_To_Device()
{
    Device_Malloc_Safely((void**)&d_atom_a, sizeof(int) * nb14_numbers);
    Device_Malloc_Safely((void**)&d_atom_b, sizeof(int) * nb14_numbers);
    Device_Malloc_Safely((void**)&d_A, sizeof(float) * nb14_numbers);
    Device_Malloc_Safely((void**)&d_B, sizeof(float) * nb14_numbers);
    Device_Malloc_Safely((void**)&d_cf_scale_factor,
                         sizeof(float) * nb14_numbers);
    Device_Malloc_Safely((void**)&d_nb14_cf_energy,
                         sizeof(float) * nb14_numbers);
    Device_Malloc_Safely((void**)&d_nb14_lj_energy,
                         sizeof(float) * nb14_numbers);
    Device_Malloc_Safely((void**)&d_nb14_lj_energy_sum, sizeof(float));
    Device_Malloc_Safely((void**)&d_nb14_cf_energy_sum, sizeof(float));

    Device_Malloc_Safely((void**)&d_atom_a_local, sizeof(int) * nb14_numbers);
    Device_Malloc_Safely((void**)&d_atom_b_local, sizeof(int) * nb14_numbers);
    Device_Malloc_Safely((void**)&d_A_local, sizeof(float) * nb14_numbers);
    Device_Malloc_Safely((void**)&d_B_local, sizeof(float) * nb14_numbers);
    Device_Malloc_Safely((void**)&d_cf_scale_factor_local,
                         sizeof(float) * nb14_numbers);
    Device_Malloc_Safely((void**)&d_num_nb14_local, sizeof(int));
    deviceMemset(d_num_nb14_local, 0, sizeof(int));

    deviceMemcpy(d_atom_a, h_atom_a, sizeof(int) * nb14_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(d_atom_b, h_atom_b, sizeof(int) * nb14_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(d_A, h_A, sizeof(float) * nb14_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(d_B, h_B, sizeof(float) * nb14_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(d_cf_scale_factor, h_cf_scale_factor,
                 sizeof(float) * nb14_numbers, deviceMemcpyHostToDevice);
    deviceMemset(d_nb14_cf_energy, 0, sizeof(float) * nb14_numbers);
    deviceMemset(d_nb14_lj_energy, 0, sizeof(float) * nb14_numbers);
    deviceMemset(d_nb14_lj_energy_sum, 0, sizeof(float));
    deviceMemset(d_nb14_cf_energy_sum, 0, sizeof(float));
}

static __global__ void get_local_device(
    int nb14_numbers, int* d_atom_a, int* d_atom_b, char* atom_local_label,
    int* atom_local_id, int* d_atom_a_local, int* d_atom_b_local, float* d_A,
    float* d_B, float* d_cf_scale_factor, float* d_A_local, float* d_B_local,
    float* d_cf_scale_factor_local, int* d_num_nb14_local)
{
#ifdef USE_GPU
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx != 0) return;
#endif
    d_num_nb14_local[0] = 0;
    for (int i = 0; i < nb14_numbers; i++)
    {
        // 判断第i个angle中原子a是否在local中，如果是则需要在本domain中计算
        if (atom_local_label[d_atom_a[i]] == 1 ||
            atom_local_label[d_atom_b[i]] == 1)
        {
            d_atom_a_local[d_num_nb14_local[0]] = atom_local_id[d_atom_a[i]];
            d_atom_b_local[d_num_nb14_local[0]] = atom_local_id[d_atom_b[i]];
            d_A_local[d_num_nb14_local[0]] = d_A[i];
            d_B_local[d_num_nb14_local[0]] = d_B[i];
            d_cf_scale_factor_local[d_num_nb14_local[0]] = d_cf_scale_factor[i];
            d_num_nb14_local[0]++;
        }
    }
}

void NON_BOND_14::Get_Local(int* atom_local, int local_atom_numbers,
                            int ghost_numbers, char* atom_local_label,
                            int* atom_local_id)
{
    if (!is_initialized) return;
    num_nb14_local = 0;
    this->local_atom_numbers = local_atom_numbers;
    Launch_Device_Kernel(get_local_device, 1, 1, 0, NULL, nb14_numbers,
                         d_atom_a, d_atom_b, atom_local_label, atom_local_id,
                         d_atom_a_local, d_atom_b_local, d_A, d_B,
                         d_cf_scale_factor, d_A_local, d_B_local,
                         d_cf_scale_factor_local, d_num_nb14_local);
    deviceMemcpy(&num_nb14_local, d_num_nb14_local, sizeof(int),
                 deviceMemcpyDeviceToHost);
}

void NON_BOND_14::Non_Bond_14_LJ_CF_Force_With_Atom_Energy_And_Virial(
    const VECTOR* crd, const float* charge, const LTMatrix3 cell,
    const LTMatrix3 rcell, VECTOR* frc, int need_atom_energy,
    float* atom_energy, int need_virial, LTMatrix3* atom_virial)
{
    if (is_initialized)
    {
        Launch_Device_Kernel(
            Dihedral_14_LJ_CF_Force_With_Atom_Energy_And_Virial_Device,
            (nb14_numbers + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, num_nb14_local, crd, cell,
            rcell, this->local_atom_numbers, d_atom_a_local, d_atom_b_local,
            d_cf_scale_factor_local, charge, d_A_local, d_B_local, frc,
            need_atom_energy, atom_energy, need_virial, atom_virial,
            d_nb14_cf_energy, d_nb14_lj_energy);
    }
}

void NON_BOND_14::Step_Print(CONTROLLER* controller, bool print_sum)
{
    if (is_initialized && CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size)
    {
        Sum_Of_List(d_nb14_cf_energy, d_nb14_cf_energy_sum, num_nb14_local);
        deviceMemcpy(h_nb14_cf_energy_sum, d_nb14_cf_energy_sum, sizeof(float),
                     deviceMemcpyDeviceToHost);
        Sum_Of_List(d_nb14_lj_energy, d_nb14_lj_energy_sum, num_nb14_local);
        deviceMemcpy(h_nb14_lj_energy_sum, d_nb14_lj_energy_sum, sizeof(float),
                     deviceMemcpyDeviceToHost);
#ifdef USE_MPI
        MPI_Allreduce(MPI_IN_PLACE, h_nb14_cf_energy_sum, 1, MPI_FLOAT, MPI_SUM,
                      CONTROLLER::pp_comm);
        MPI_Allreduce(MPI_IN_PLACE, h_nb14_lj_energy_sum, 1, MPI_FLOAT, MPI_SUM,
                      CONTROLLER::pp_comm);
#endif
        controller->Step_Print(print_ee_name.c_str(), h_nb14_cf_energy_sum[0],
                               print_sum);
        controller->Step_Print(print_lj_name.c_str(), h_nb14_lj_energy_sum[0],
                               print_sum);
    }
}

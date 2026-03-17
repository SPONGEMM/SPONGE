#include "angle.h"

#include "../xponge/load/native/angle.hpp"
#include "../xponge/xponge.h"

static __global__ void Angle_Force_With_Atom_Energy_And_Virial_Device(
    const int angle_numbers, const VECTOR* crd, const LTMatrix3 cell,
    const LTMatrix3 rcell, const int local_atom_numbers, const int* atom_a,
    const int* atom_b, const int* atom_c, const float* angle_k,
    const float* angle_theta0, VECTOR* frc, int need_atom_energy,
    float* atom_energy, float* angle_energy, int need_virial, LTMatrix3* virial)
{
#ifdef USE_GPU
    int angle_i = blockDim.x * blockIdx.x + threadIdx.x;
    if (angle_i < angle_numbers)
#else
#pragma omp parallel for
    for (int angle_i = 0; angle_i < angle_numbers; angle_i++)
#endif
    {
        int atom_i = atom_a[angle_i];
        int atom_j = atom_b[angle_i];
        int atom_k = atom_c[angle_i];

        float theta0 = angle_theta0[angle_i];
        float k = angle_k[angle_i];
        float k2 = k;  // 复制一份k

        VECTOR drij =
            Get_Periodic_Displacement(crd[atom_i], crd[atom_j], cell, rcell);
        VECTOR drkj =
            Get_Periodic_Displacement(crd[atom_k], crd[atom_j], cell, rcell);

        float rij_2 = 1. / (drij * drij);
        float rkj_2 = 1. / (drkj * drkj);
        float rij_1_rkj_1 = sqrtf(rij_2 * rkj_2);

        float costheta = drij * drkj * rij_1_rkj_1;
        costheta = fmaxf(-0.999999, fminf(costheta, 0.999999));
        float theta = acosf(costheta);

        float dtheta = theta - theta0;
        k = -2 * k * dtheta / sinf(theta);

        float common_factor_cross = k * rij_1_rkj_1;
        float common_factor_self = k * costheta;

        VECTOR fi =
            common_factor_self * rij_2 * drij - common_factor_cross * drkj;
        VECTOR fk =
            common_factor_self * rkj_2 * drkj - common_factor_cross * drij;
        if (atom_k < local_atom_numbers)
        {
            atomicAdd(&frc[atom_k].x, fk.x);
            atomicAdd(&frc[atom_k].y, fk.y);
            atomicAdd(&frc[atom_k].z, fk.z);
        }
        if (atom_i < local_atom_numbers)
        {
            atomicAdd(&frc[atom_i].x, fi.x);
            atomicAdd(&frc[atom_i].y, fi.y);
            atomicAdd(&frc[atom_i].z, fi.z);
            if (need_virial)
            {
                atomicAdd(virial + atom_i,
                          Get_Virial_From_Force_Dis(fi, drij) +
                              Get_Virial_From_Force_Dis(fk, drkj));
            }
            if (need_atom_energy)
            {
                atomicAdd(atom_energy + atom_i, k2 * dtheta * dtheta);
                angle_energy[angle_i] = k2 * dtheta * dtheta;
            }
        }
        fi = -fi - fk;
        if (atom_j < local_atom_numbers)
        {
            atomicAdd(&frc[atom_j].x, fi.x);
            atomicAdd(&frc[atom_j].y, fi.y);
            atomicAdd(&frc[atom_j].z, fi.z);
        }
    }
}

void ANGLE::Initial(CONTROLLER* controller, const char* module_name)
{
    if (module_name == NULL)
    {
        strcpy(this->module_name, "angle");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }

    char file_name_suffix[CHAR_LENGTH_MAX];
    sprintf(file_name_suffix, "in_file");
    const auto& angles = Xponge::system.classical_force_field.angles;
    Xponge::Angles local_angles;
    const Xponge::Angles* angles_to_use = NULL;
    const char* init_source = NULL;

    if (module_name == NULL)
    {
        angles_to_use = &angles;
        init_source = "Xponge::system";
    }
    else if (controller->Command_Exist(this->module_name, file_name_suffix))
    {
        Xponge::Native_Load_Angles(&local_angles, controller,
                                   this->module_name);
        angles_to_use = &local_angles;
    }

    if (angles_to_use != NULL)
    {
        angle_numbers = static_cast<int>(angles_to_use->atom_a.size());
    }
    if (angle_numbers > 0)
    {
        if (module_name == NULL)
        {
            controller->printf("START INITIALIZING ANGLE (%s):\n", init_source);
        }
        else
        {
            controller->printf("START INITIALIZING ANGLE (%s_%s):\n",
                               this->module_name, file_name_suffix);
        }
        controller->printf("    angle_numbers is %d\n", angle_numbers);
        Memory_Allocate();
        for (int i = 0; i < angle_numbers; i++)
        {
            h_atom_a[i] = angles_to_use->atom_a[i];
            h_atom_b[i] = angles_to_use->atom_b[i];
            h_atom_c[i] = angles_to_use->atom_c[i];
            h_angle_k[i] = angles_to_use->k[i];
            h_angle_theta0[i] = angles_to_use->theta0[i];
        }
        Parameter_Host_To_Device();
        is_initialized = 1;
    }
    else
    {
        controller->printf("ANGLE IS NOT INITIALIZED\n\n");
    }
    if (is_initialized && !is_controller_printf_initialized)
    {
        controller->Step_Print_Initial(this->module_name, "%.2f");
        is_controller_printf_initialized = 1;
        controller->printf("    structure last modify date is %d\n",
                           last_modify_date);
    }
    if (is_initialized)
    {
        controller->printf("END INITIALIZING ANGLE\n\n");
    }
}

void ANGLE::Memory_Allocate()
{
    Malloc_Safely((void**)&h_atom_a, sizeof(int) * angle_numbers);
    Malloc_Safely((void**)&h_atom_b, sizeof(int) * angle_numbers);
    Malloc_Safely((void**)&h_atom_c, sizeof(int) * angle_numbers);
    Malloc_Safely((void**)&h_angle_k, sizeof(float) * angle_numbers);
    Malloc_Safely((void**)&h_angle_theta0, sizeof(float) * angle_numbers);
    Malloc_Safely((void**)&h_angle_ene, sizeof(float) * angle_numbers);
    memset(h_angle_ene, 0, sizeof(float) * angle_numbers);
    Malloc_Safely((void**)&h_sigma_of_angle_ene, sizeof(float));
    memset(h_sigma_of_angle_ene, 0, sizeof(float));
}

void ANGLE::Parameter_Host_To_Device()
{
    Device_Malloc_Safely((void**)&d_atom_a, sizeof(int) * angle_numbers);
    Device_Malloc_Safely((void**)&d_atom_b, sizeof(int) * angle_numbers);
    Device_Malloc_Safely((void**)&d_atom_c, sizeof(int) * angle_numbers);
    Device_Malloc_Safely((void**)&d_angle_k, sizeof(float) * angle_numbers);
    Device_Malloc_Safely((void**)&d_angle_theta0,
                         sizeof(float) * angle_numbers);
    Device_Malloc_Safely((void**)&d_angle_ene, sizeof(float) * angle_numbers);
    Device_Malloc_Safely((void**)&d_sigma_of_angle_ene, sizeof(float));

    Device_Malloc_Safely((void**)&d_atom_a_local, sizeof(int) * angle_numbers);
    Device_Malloc_Safely((void**)&d_atom_b_local, sizeof(int) * angle_numbers);
    Device_Malloc_Safely((void**)&d_atom_c_local, sizeof(int) * angle_numbers);
    Device_Malloc_Safely((void**)&d_angle_k_local,
                         sizeof(float) * angle_numbers);
    Device_Malloc_Safely((void**)&d_angle_theta0_local,
                         sizeof(float) * angle_numbers);
    Device_Malloc_Safely((void**)&d_num_angle_local, sizeof(int));
    deviceMemset(d_num_angle_local, 0, sizeof(int));

    deviceMemcpy(d_atom_a, h_atom_a, sizeof(int) * angle_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(d_atom_b, h_atom_b, sizeof(int) * angle_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(d_atom_c, h_atom_c, sizeof(int) * angle_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(d_angle_k, h_angle_k, sizeof(float) * angle_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(d_angle_theta0, h_angle_theta0, sizeof(float) * angle_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(d_angle_ene, h_angle_ene, sizeof(float) * angle_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemset(d_sigma_of_angle_ene, 0, sizeof(float));
}

static __global__ void get_local_device(
    int angle_numbers, int* d_atom_a, int* d_atom_b, int* d_atom_c,
    char* atom_local_label, int* atom_local_id, int* d_atom_a_local,
    int* d_atom_b_local, int* d_atom_c_local, float* d_angle_k,
    float* d_angle_theta0, float* d_angle_k_local, float* d_angle_theta0_local,
    int* d_num_angle_local)
{
#ifdef USE_GPU
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx != 0) return;
#endif
    d_num_angle_local[0] = 0;
    // 判断第i个angle中原子a是否在local中，如果是则需要在本domain中计算
    for (int i = 0; i < angle_numbers; i++)
    {
        if (atom_local_label[d_atom_a[i]] == 1 ||
            atom_local_label[d_atom_b[i]] == 1 ||
            atom_local_label[d_atom_c[i]] == 1)
        {
            d_atom_a_local[d_num_angle_local[0]] = atom_local_id[d_atom_a[i]];
            d_atom_b_local[d_num_angle_local[0]] = atom_local_id[d_atom_b[i]];
            d_atom_c_local[d_num_angle_local[0]] = atom_local_id[d_atom_c[i]];
            d_angle_k_local[d_num_angle_local[0]] = d_angle_k[i];
            d_angle_theta0_local[d_num_angle_local[0]] = d_angle_theta0[i];
            d_num_angle_local[0]++;
        }
    }
}

void ANGLE::Get_Local(int* atom_local, int local_atom_numbers,
                      int ghost_numbers, char* atom_local_label,
                      int* atom_local_id)
{
    if (!is_initialized) return;
    num_angle_local = 0;
    this->local_atom_numbers = local_atom_numbers;
    Launch_Device_Kernel(
        get_local_device, 1, 1, 0, NULL, this->angle_numbers, this->d_atom_a,
        this->d_atom_b, this->d_atom_c, atom_local_label, atom_local_id,
        this->d_atom_a_local, this->d_atom_b_local, this->d_atom_c_local,
        this->d_angle_k, this->d_angle_theta0, this->d_angle_k_local,
        this->d_angle_theta0_local, this->d_num_angle_local);
    deviceMemcpy(&this->num_angle_local, this->d_num_angle_local, sizeof(int),
                 deviceMemcpyDeviceToHost);
}

void ANGLE::Angle_Force_With_Atom_Energy_And_Virial(
    const VECTOR* crd, const LTMatrix3 cell, const LTMatrix3 rcell, VECTOR* frc,
    int need_atom_energy, float* atom_energy, int need_virial,
    LTMatrix3* atom_virial_tensor)
{
    if (is_initialized)
    {
        Launch_Device_Kernel(
            Angle_Force_With_Atom_Energy_And_Virial_Device,
            (angle_numbers + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, this->num_angle_local, crd,
            cell, rcell, this->local_atom_numbers, this->d_atom_a_local,
            this->d_atom_b_local, this->d_atom_c_local, this->d_angle_k_local,
            this->d_angle_theta0_local, frc, need_atom_energy, atom_energy,
            this->d_angle_ene, need_virial, atom_virial_tensor);
    }
}

void ANGLE::Step_Print(CONTROLLER* controller)
{
    if (is_initialized && CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size)
    {
        Sum_Of_List(d_angle_ene, d_sigma_of_angle_ene,
                    num_angle_local);  // 修改为local求和
        deviceMemcpy(h_sigma_of_angle_ene, d_sigma_of_angle_ene, sizeof(float),
                     deviceMemcpyDeviceToHost);
#ifdef USE_MPI
        MPI_Allreduce(MPI_IN_PLACE, h_sigma_of_angle_ene, 1, MPI_FLOAT, MPI_SUM,
                      CONTROLLER::pp_comm);
#endif
        if (CONTROLLER::MPI_rank == 0)
        {
            controller->Step_Print(this->module_name, h_sigma_of_angle_ene,
                                   true);
        }
    }
}

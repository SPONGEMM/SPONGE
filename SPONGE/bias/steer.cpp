#include "steer.h"

static __global__ void steer_force_and_energy(int atom_numbers, float* cv_value,
                                              VECTOR* crd_grads, float weight,
                                              VECTOR* frc, float* energy,
                                              float* self_ene,
                                              int need_potential)
{
#ifdef USE_GPU
    // 能量只第一个线程算
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        if (need_potential)
        {
            atomicAdd(energy, weight * cv_value[0]);
            self_ene[0] = weight * cv_value[0];
        }
    }
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < atom_numbers;
         i += gridDim.x * blockDim.x)
    {
        VECTOR force = -weight * crd_grads[i];
        atomicAdd(&frc[i].x, force.x);
        atomicAdd(&frc[i].y, force.y);
        atomicAdd(&frc[i].z, force.z);
    }
#else
    if (need_potential)
    {
        atomicAdd(energy, weight * cv_value[0]);
        self_ene[0] = weight * cv_value[0];
    }
#pragma omp parallel for
    for (int i = 0; i < atom_numbers; i++)
    {
        VECTOR force = -weight * crd_grads[i];
        atomicAdd(&frc[i].x, force.x);
        atomicAdd(&frc[i].y, force.y);
        atomicAdd(&frc[i].z, force.z);
    }
#endif
}

static __global__ void steer_force_and_energy_and_virial(
    int atom_numbers, float* cv_value, VECTOR* crd_grads, LTMatrix3* cv_virial,
    float weight, VECTOR* frc, float* energy, float* self_ene,
    LTMatrix3* virial, int need_potential)
{
#ifdef USE_GPU
    // 能量只第一个线程算
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        if (need_potential)
        {
            atomicAdd(energy, weight * cv_value[0]);
            self_ene[0] = weight * cv_value[0];
        }
        atomicAdd(virial, -weight * cv_virial[0]);
    }
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < atom_numbers;
         i += blockDim.x * gridDim.x)
    {
        VECTOR force = -weight * crd_grads[i];
        atomicAdd(&frc[i].x, force.x);
        atomicAdd(&frc[i].y, force.y);
        atomicAdd(&frc[i].z, force.z);
    }
#else
    if (need_potential)
    {
        atomicAdd(energy, weight * cv_value[0]);
        self_ene[0] = weight * cv_value[0];
    }
    atomicAdd(virial, -weight * cv_virial[0]);
#pragma omp parallel for
    for (int i = 0; i < atom_numbers; i++)
    {
        VECTOR force = -weight * crd_grads[i];
        atomicAdd(&frc[i].x, force.x);
        atomicAdd(&frc[i].y, force.y);
        atomicAdd(&frc[i].z, force.z);
    }
#endif
}

void STEER_CV::Initial(CONTROLLER* controller,
                       COLLECTIVE_VARIABLE_CONTROLLER* cv_controller)
{
    strcpy(this->module_name, "steer_cv");
    controller->printf("START INITIALIZING STEER CV:\n");
    cv_list = cv_controller->Ask_For_CV("steer", 0);
    CV_numbers = cv_list.size();
    if (CV_numbers)
    {
        weight = cv_controller->Ask_For_Float_Parameter(
            "steer", "weight", cv_list.size(), 1, true, 0, 0, "kcal/mol/CV^2");
        Malloc_Safely((void**)&h_ene, sizeof(float) * CV_numbers);
        Device_Malloc_Safely((void**)&d_ene, sizeof(float) * CV_numbers);
        controller->Step_Print_Initial(this->module_name, "%.2f");
        is_initialized = 1;
        controller->printf("END INITIALIZING STEER CV\n\n");
    }
    else
    {
        controller->printf("STEER CV IS NOT INITIALIZED\n\n");
    }
}

void STEER_CV::Step_Print(CONTROLLER* controller)
{
    if (!is_initialized) return;
    if (CONTROLLER::MPI_size == 1 && CONTROLLER::PM_MPI_size == 1)
    {
        float ret = 0;
        deviceMemcpy(h_ene, d_ene, sizeof(float) * CV_numbers,
                     deviceMemcpyDeviceToHost);
        for (int i = 0; i < CV_numbers; i++)
        {
            ret += h_ene[i];
        }
        controller->Step_Print(module_name, ret);
        return;
    }
    else  // 把最后一个进程号的信息发给0号进程
    {
        float ret = 0;
#ifdef USE_MPI
        if (CONTROLLER::MPI_rank == 0)
        {
            MPI_Recv(&ret, 1, MPI_FLOAT, CONTROLLER::MPI_size - 1, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            controller->Step_Print(module_name, ret);
        }
        if (CONTROLLER::MPI_rank == CONTROLLER::MPI_size - 1)
        {
            deviceMemcpy(h_ene, d_ene, sizeof(float) * CV_numbers,
                         deviceMemcpyDeviceToHost);
            for (int i = 0; i < CV_numbers; i++)
            {
                ret += h_ene[i];
            }
            MPI_Send(&ret, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        }
#endif
    }
}

void STEER_CV::Steer(int atom_numbers, VECTOR* crd, LTMatrix3 cell,
                     LTMatrix3 rcell, int step, float* d_ene,
                     LTMatrix3* d_virial, VECTOR* frc, int need_potential,
                     int need_pressure)
{
    if (!is_initialized) return;
    COLLECTIVE_VARIABLE_PROTOTYPE* cv;
    int need = CV_NEED_CRD_GRADS | CV_NEED_GPU_VALUE;
    if (need_pressure) need |= CV_NEED_VIRIAL;
    for (int i = 0; i < CV_numbers; i++)
    {
        cv = cv_list[i];
        cv->Compute(atom_numbers, crd, cell, rcell, need, step);
        if (!need_pressure)
        {
            Launch_Device_Kernel(
                steer_force_and_energy,
                (atom_numbers + CONTROLLER::device_max_thread - 1) /
                    CONTROLLER::device_max_thread,
                CONTROLLER::device_max_thread, 0, cv->device_stream,
                atom_numbers, cv->d_value, cv->crd_grads, weight[i], frc, d_ene,
                this->d_ene + i, need_potential);
        }
        else
        {
            Launch_Device_Kernel(
                steer_force_and_energy_and_virial,
                (atom_numbers + CONTROLLER::device_max_thread - 1) /
                    CONTROLLER::device_max_thread,
                CONTROLLER::device_max_thread, 0, cv->device_stream,
                atom_numbers, cv->d_value, cv->crd_grads, cv->virial, weight[i],
                frc, d_ene, this->d_ene + i, d_virial, need_potential);
        }
    }
}

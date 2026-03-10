#include "restrain_cv.h"

static __global__ void restrain_force_and_energy(
    int atom_numbers, float* cv_value, VECTOR* crd_grads, float weight,
    float reference, float period, VECTOR* frc, float* energy, float* self_ene,
    int need_potential)
{
    float dCV = cv_value[0] - reference;
    if (period > 0)
    {
        dCV = dCV - floorf(dCV / period + 0.5) * period;
    }
#ifdef USE_GPU
    // 能量只第一个线程算
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        if (need_potential)
        {
            self_ene[0] = weight * dCV * dCV;
            atomicAdd(energy, weight * dCV * dCV);
        }
    }
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < atom_numbers;
         i += gridDim.x * blockDim.x)
    {
        VECTOR force = -2 * weight * dCV * crd_grads[i];
        atomicAdd(&frc[i].x, force.x);
        atomicAdd(&frc[i].y, force.y);
        atomicAdd(&frc[i].z, force.z);
    }
#else
    if (need_potential)
    {
        self_ene[0] = weight * dCV * dCV;
        atomicAdd(energy, weight * dCV * dCV);
    }
#pragma omp parallel for
    for (int i = 0; i < atom_numbers; i++)
    {
        VECTOR force = -2 * weight * dCV * crd_grads[i];
        atomicAdd(&frc[i].x, force.x);
        atomicAdd(&frc[i].y, force.y);
        atomicAdd(&frc[i].z, force.z);
    }
#endif
}

static __global__ void restrain_force_and_energy_and_virial(
    int atom_numbers, float* cv_value, VECTOR* crd_grads, LTMatrix3* cv_virial,
    float weight, float reference, float period, VECTOR* frc, float* energy,
    float* self_ene, LTMatrix3* virial, int need_potential)
{
    float dCV = cv_value[0] - reference;
    if (period > 0)
    {
        dCV = dCV - floorf(dCV / period + 0.5) * period;
    }
#ifdef USE_GPU
    // 能量只第一个线程算
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        if (need_potential)
        {
            self_ene[0] = weight * dCV * dCV;
            atomicAdd(energy, weight * dCV * dCV);
        }
        atomicAdd(virial, -2 * weight * dCV * cv_virial[0]);
    }
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < atom_numbers;
         i += gridDim.x * blockDim.x)
    {
        VECTOR force = -2 * weight * dCV * crd_grads[i];
        atomicAdd(&frc[i].x, force.x);
        atomicAdd(&frc[i].y, force.y);
        atomicAdd(&frc[i].z, force.z);
    }
#else
    if (need_potential)
    {
        self_ene[0] = weight * dCV * dCV;
        atomicAdd(energy, weight * dCV * dCV);
    }
    atomicAdd(virial, -2 * weight * dCV * cv_virial[0]);
#pragma omp parallel for
    for (int i = 0; i < atom_numbers; i++)
    {
        VECTOR force = -2 * weight * dCV * crd_grads[i];
        atomicAdd(&frc[i].x, force.x);
        atomicAdd(&frc[i].y, force.y);
        atomicAdd(&frc[i].z, force.z);
    }
#endif
}

void RESTRAIN_CV::Initial(CONTROLLER* controller,
                          COLLECTIVE_VARIABLE_CONTROLLER* cv_controller)
{
    strcpy(this->module_name, "restrain_cv");
    controller->printf("START INITIALIZING RESTRAIN CV:\n");
    cv_list = cv_controller->Ask_For_CV("restrain", 0);
    CV_numbers = cv_list.size();
    if (CV_numbers)
    {
        weight = cv_controller->Ask_For_Float_Parameter("restrain", "weight",
                                                        cv_list.size(), 1, true,
                                                        0, 0, "kcal/mol/CV^2");
        reference = cv_controller->Ask_For_Float_Parameter(
            "restrain", "reference", cv_list.size(), 1, true, 0, 0, "CV");
        period = cv_controller->Ask_For_Float_Parameter(
            "restrain", "period", cv_list.size(), 1, false, 0, 0, "CV");
        start_step = cv_controller->Ask_For_Int_Parameter(
            "restrain", "start_step", cv_list.size(), 1, false, 0, 0);
        max_step = cv_controller->Ask_For_Int_Parameter(
            "restrain", "max_step", cv_list.size(), 1, false, 0, 0);
        reduce_step = cv_controller->Ask_For_Int_Parameter(
            "restrain", "reduce_step", cv_list.size(), 1, false, 0, 0);
        stop_step = cv_controller->Ask_For_Int_Parameter(
            "restrain", "stop_step", cv_list.size(), 1, false, 0, 0);
        for (int i = 0; i < CV_numbers; i++)
        {
            StringMap error_map = {
                {"i", std::to_string(i)},
                {"start_step", std::to_string(start_step[i])},
                {"max_step", std::to_string(max_step[i])},
                {"reduce_step", std::to_string(reduce_step[i])},
                {"stop_step", std::to_string(stop_step[i])}};
            if (max_step[i] != 0 && max_step[i] < start_step[i])
            {
                controller->Throw_SPONGE_Error(
                    spongeErrorConflictingCommand, "RESTRAIN_CV::Initial",
                    string_format(
                        "Reason:\n\tThe max step (%max_step%) of %i%-th CV is smaller than \
the start step (%start_step%)",
                        error_map)
                        .c_str());
            }
            if (reduce_step[i] != 0 && reduce_step[i] < max_step[i])
            {
                controller->Throw_SPONGE_Error(
                    spongeErrorConflictingCommand, "RESTRAIN_CV::Initial",
                    string_format(
                        "Reason:\n\tThe reducing step (%reduce_step%) of %i%-th CV is smaller than \
the max step (%max_step%)",
                        error_map)
                        .c_str());
            }
            if (reduce_step[i] != 0 && stop_step[i] < reduce_step[i])
            {
                controller->Throw_SPONGE_Error(
                    spongeErrorConflictingCommand, "RESTRAIN_CV::Initial",
                    string_format(
                        "Reason:\n\tThe stop step (%stop_step%) of %i%-th CV is smaller than \
the reduce step (%reduce_step%)",
                        error_map)
                        .c_str());
            }
            if (stop_step[i] != 0 && reduce_step[i] == 0)
            {
                controller->Throw_SPONGE_Error(
                    spongeErrorConflictingCommand, "RESTRAIN_CV::Initial",
                    string_format(
                        "Reason:\n\tThe reduce step (%reduce_step%) of %i%-th CV should be non-zero \
when the stop step is not zero (%stop_step%)",
                        error_map)
                        .c_str());
            }
        }
        Malloc_Safely((void**)&h_ene, sizeof(float) * CV_numbers);
        Device_Malloc_Safely((void**)&d_ene, sizeof(float) * CV_numbers);
        controller->Step_Print_Initial(this->module_name, "%.2f");
        is_initialized = 1;
        controller->printf("END INITIALIZING RESTRAIN CV\n\n");
    }
    else
    {
        controller->printf("RESTRAIN CV IS NOT INITIALIZED\n\n");
    }
}

void RESTRAIN_CV::Step_Print(CONTROLLER* controller)
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

void RESTRAIN_CV::Restraint(int atom_numbers, VECTOR* crd, LTMatrix3 cell,
                            LTMatrix3 rcell, int step, float* d_ene,
                            LTMatrix3* d_virial, VECTOR* frc,
                            int need_potential, int need_pressure)
{
    if (!is_initialized) return;
    COLLECTIVE_VARIABLE_PROTOTYPE* cv;
    int need = CV_NEED_CRD_GRADS | CV_NEED_GPU_VALUE;
    if (need_pressure) need |= CV_NEED_VIRIAL;
    for (int i = 0; i < CV_numbers; i++)
    {
        if (step < start_step[i] || (stop_step[i] > 0 && step >= stop_step[i]))
            continue;
        float local_weight = weight[i];
        if (step < max_step[i] && max_step[i] > start_step[i])
            local_weight *=
                (float)(step - start_step[i]) / (max_step[i] - start_step[i]);
        if (reduce_step[i] != 0 && step > reduce_step[i] &&
            reduce_step[i] < stop_step[i])
            local_weight *=
                (float)(stop_step[i] - step) / (stop_step[i] - reduce_step[i]);
        cv = cv_list[i];
        cv->Compute(atom_numbers, crd, cell, rcell, need, step);
        if (!need_pressure)
        {
            Launch_Device_Kernel(
                restrain_force_and_energy,
                (atom_numbers + CONTROLLER::device_max_thread - 1) /
                    CONTROLLER::device_max_thread,
                CONTROLLER::device_max_thread, 0, cv->device_stream,
                atom_numbers, cv->d_value, cv->crd_grads, local_weight,
                reference[i], period[i], frc, d_ene, this->d_ene + i,
                need_potential);
        }
        else
        {
            Launch_Device_Kernel(
                restrain_force_and_energy_and_virial,
                (atom_numbers + CONTROLLER::device_max_thread - 1) /
                    CONTROLLER::device_max_thread,
                CONTROLLER::device_max_thread, 0, cv->device_stream,
                atom_numbers, cv->d_value, cv->crd_grads, cv->virial,
                local_weight, reference[i], period[i], frc, d_ene,
                this->d_ene + i, d_virial, need_potential);
        }
    }
}

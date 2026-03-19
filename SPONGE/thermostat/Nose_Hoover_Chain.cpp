#include "Nose_Hoover_Chain.h"

static __global__ void Nose_Hoover_Chain_Update(
    int chain_length, float* chain_crd, float* chain_vel, float chain_mass,
    float Ek, float kB_T, float dt, int freedom)
{
    float chain_mass_inverse = 1.0 / chain_mass;
    chain_vel[0] += (2 * Ek - freedom * kB_T) * chain_mass_inverse * dt -
                    chain_vel[0] * chain_vel[1] * dt;
    for (int i = 1; i < chain_length; i++)
    {
        float temp_vel = chain_vel[i - 1];
        chain_vel[i] += (temp_vel * temp_vel - kB_T * chain_mass_inverse) * dt -
                        chain_vel[i] * chain_vel[i + 1] * dt;
    }
    for (int i = 0; i < chain_length; i++)
    {
        chain_crd[i] += chain_vel[i] * dt;
    }
}

static __global__ void MD_Iteration_Leap_Frog_With_NHC(
    const int local_atom_numbers, const float dt,
    const float* inverse_mass_local, VECTOR* vel, VECTOR* crd, VECTOR* frc,
    VECTOR* acc, float chain_vel)
{
#ifdef USE_GPU
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < local_atom_numbers)
#else
#pragma omp parallel for
    for (int i = 0; i < local_atom_numbers; i++)
#endif
    {
        VECTOR temp1 = inverse_mass_local[i] * frc[i] - chain_vel * vel[i];
        temp1 = vel[i] + dt * temp1;
        vel[i] = temp1;
        crd[i] = crd[i] + dt * temp1;
    }
}

static __global__ void MD_Iteration_Leap_Frog_With_NHC_With_Max_Velocity(
    const int local_atom_numbers, const float dt,
    const float* inverse_mass_local, VECTOR* vel, VECTOR* crd, VECTOR* frc,
    VECTOR* acc, float chain_vel, float max_vel)
{
#ifdef USE_GPU
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < local_atom_numbers)
#else
#pragma omp parallel for
    for (int i = 0; i < local_atom_numbers; i++)
#endif
    {
        VECTOR temp1 = inverse_mass_local[i] * frc[i] - chain_vel * vel[i];
        temp1 = vel[i] + dt * temp1;
        temp1 = Make_Vector_Not_Exceed_Value(temp1, max_vel);
        vel[i] = temp1;
        crd[i] = crd[i] + dt * temp1;
    }
}

void NOSE_HOOVER_CHAIN_INFORMATION::Initial(CONTROLLER* controller,
                                            int atom_numbers,
                                            float target_temperature,
                                            const float* atom_mass,
                                            const char* module_name)
{
    controller->printf("START INITIALIZING NOSE HOOVER CHAIN:\n");
    if (module_name == NULL)
    {
        strcpy(this->module_name, "nose_hoover_chain");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }

    float* h_mass_temp = NULL;
    Malloc_Safely((void**)&h_mass_temp, sizeof(float) * atom_numbers);
    deviceMemcpy(h_mass_temp, atom_mass, sizeof(float) * atom_numbers,
                 deviceMemcpyHostToHost);
    for (int i = 0; i < atom_numbers; i = i + 1)
    {
        if (h_mass_temp[i] != 0)
        {
            h_mass_temp[i] = 1.0 / h_mass_temp[i];
        }
    }
    Device_Malloc_Safely((void**)&d_mass_inverse, sizeof(float) * atom_numbers);
    deviceMemcpy(d_mass_inverse, h_mass_temp, sizeof(float) * atom_numbers,
                 deviceMemcpyHostToDevice);
    Device_Malloc_Safely((void**)&d_mass_inverse_local,
                         sizeof(float) * atom_numbers);
    free(h_mass_temp);

    chain_length = 1;
    if (controller[0].Command_Exist(this->module_name, "length"))
    {
        controller->Check_Int(this->module_name, "length",
                              "NOSE_HOOVER_CHAIN_INFORMATION::Initial");
        chain_length = atoi(controller->Command(this->module_name, "length"));
    }
    controller[0].printf("    chain length is %d\n", chain_length);

    Malloc_Safely((void**)&h_coordinate, sizeof(float) * chain_length);
    Malloc_Safely((void**)&h_velocity, sizeof(float) * (chain_length + 1));

    float tauT = 1.0f;
    if (controller[0].Command_Exist("thermostat", "tau"))
    {
        controller->Check_Float("thermostat", "tau",
                                "NOSE_HOOVER_CHAIN_INFORMATION::Initial");
        tauT = atof(controller->Command("thermostat", "tau"));
    }
    controller[0].printf("    time constant tau is %f ps\n", tauT);
    tauT *= CONSTANT_TIME_CONVERTION;
    this->target_temperature = target_temperature;
    h_mass =
        tauT * tauT * target_temperature / 4.0f / CONSTANT_Pi / CONSTANT_Pi;
    kB_T = CONSTANT_kB * target_temperature;
    controller[0].printf("    target temperature is %.2f K\n",
                         target_temperature);
    controller[0].printf("    chain mass is %f\n", h_mass);
    controller->Deprecated("nose_hoover_chain_tau", "thermostat_tau = %VALUE%",
                           "1.5",
                           "The thermostat parameters have been managed "
                           "uniformly since version 1.5");

    if (controller[0].Command_Exist(this->module_name, "restart_input"))
    {
        FILE* fcrd = NULL;
        Open_File_Safely(
            &fcrd, controller[0].Command(this->module_name, "restart_input"),
            "r");
        for (int i = 0; i < chain_length; i++)
        {
            int scan_ret =
                fscanf(fcrd, "%f %f", h_coordinate + i, h_velocity + i);
        }
        fclose(fcrd);
    }
    else
    {
        for (int i = 0; i < chain_length; i++)
        {
            h_coordinate[i] = 0;
            h_velocity[i] = 0;
        }
    }
    h_velocity[chain_length] = 0;
    Device_Malloc_Safely((void**)&coordinate, sizeof(float) * chain_length);
    Device_Malloc_Safely((void**)&velocity, sizeof(float) * (chain_length + 1));
    deviceMemcpy(coordinate, h_coordinate, sizeof(float) * chain_length,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(velocity, h_velocity, sizeof(float) * (chain_length + 1),
                 deviceMemcpyHostToDevice);

    restart_file_name[0] = 0;
    if (controller[0].Command_Exist(this->module_name, "restart_output"))
    {
        strcpy(restart_file_name,
               controller->Command(this->module_name, "restart_output"));
    }

    char tempchar[CHAR_LENGTH_MAX];
    tempchar[0] = 0;
    f_crd_traj = NULL;
    if (controller[0].Command_Exist(this->module_name, "crd"))
    {
        strcpy(tempchar, controller->Command(this->module_name, "crd"));
        Open_File_Safely(&f_crd_traj, tempchar, "w");
        controller->Set_File_Buffer(f_crd_traj,
                                    sizeof(char) * 15 * chain_length);
    }
    tempchar[0] = 0;
    f_vel_traj = NULL;
    if (controller[0].Command_Exist(this->module_name, "vel"))
    {
        strcpy(tempchar, controller->Command(this->module_name, "vel"));
        Open_File_Safely(&f_vel_traj, tempchar, "w");
        controller->Set_File_Buffer(f_vel_traj,
                                    sizeof(char) * 15 * chain_length);
    }

    max_velocity = 0;
    if (controller[0].Command_Exist("velocity_max"))
    {
        controller->Check_Float(this->module_name, "velocity_max",
                                "NOSE_HOOVER_CHAIN_INFORMATION::Initial");
        sscanf(controller[0].Command("velocity_max"), "%f", &max_velocity);
        controller[0].printf("    max velocity is %.2f\n", max_velocity);
    }

    is_initialized = 1;
    if (is_initialized && !is_controller_printf_initialized)
    {
        is_controller_printf_initialized = 1;
        controller[0].printf("    structure last modify date is %d\n",
                             last_modify_date);
    }

    controller->printf("END INITIALIZING NOSE HOOVER CHAIN\n\n");
}

static __global__ void device_get_local(int* atom_local, int local_atom_numbers,
                                        float* d_mass_inverse,
                                        float* d_mass_inverse_local)
{
#ifdef USE_GPU
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < local_atom_numbers;
         i += blockDim.x * gridDim.x)
#else
#pragma omp parallel for
    for (int i = 0; i < local_atom_numbers; ++i)
#endif
    {
        int atom_i = atom_local[i];
        d_mass_inverse_local[i] = d_mass_inverse[atom_i];
    }
}

void NOSE_HOOVER_CHAIN_INFORMATION::Get_Local(int* atom_local,
                                              int local_atom_numbers)
{
    if (!is_initialized) return;
    this->local_atom_numbers = local_atom_numbers;
    Launch_Device_Kernel(
        device_get_local,
        (local_atom_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, atom_local, local_atom_numbers,
        d_mass_inverse, d_mass_inverse_local);
}

void NOSE_HOOVER_CHAIN_INFORMATION::MD_Iteration_Leap_Frog(
    VECTOR* vel, VECTOR* crd, VECTOR* frc, VECTOR* acc, float dt, float Ek,
    int freedom)
{
    if (is_initialized)
    {
        Launch_Device_Kernel(Nose_Hoover_Chain_Update, 1, 1, 0, NULL,
                             chain_length, coordinate, velocity, h_mass, Ek,
                             kB_T, dt, freedom);
        deviceMemcpy(h_coordinate, coordinate, sizeof(float) * chain_length,
                     deviceMemcpyDeviceToHost);
        deviceMemcpy(h_velocity, velocity, sizeof(float) * (chain_length + 1),
                     deviceMemcpyDeviceToHost);
        if (max_velocity <= 0)
        {
            Launch_Device_Kernel(
                MD_Iteration_Leap_Frog_With_NHC,
                (local_atom_numbers + CONTROLLER::device_max_thread - 1) /
                    CONTROLLER::device_max_thread,
                CONTROLLER::device_max_thread, 0, NULL, local_atom_numbers, dt,
                d_mass_inverse_local, vel, crd, frc, acc, h_velocity[0]);
        }
        else
        {
            Launch_Device_Kernel(
                MD_Iteration_Leap_Frog_With_NHC_With_Max_Velocity,
                (local_atom_numbers + CONTROLLER::device_max_thread - 1) /
                    CONTROLLER::device_max_thread,
                CONTROLLER::device_max_thread, 0, NULL, local_atom_numbers, dt,
                d_mass_inverse_local, vel, crd, frc, acc, h_velocity[0],
                max_velocity);
        }
    }
}

void NOSE_HOOVER_CHAIN_INFORMATION::Set_Target_Temperature(
    float target_temperature_new)
{
    if (!is_initialized || !(target_temperature_new > 0.0f) ||
        !(target_temperature > 0.0f))
    {
        return;
    }
    if (fabsf(target_temperature_new - target_temperature) <= FLT_EPSILON)
    {
        return;
    }
    const float ratio = target_temperature_new / target_temperature;
    if (!(ratio > 0.0f) || !isfinite(ratio))
    {
        return;
    }
    h_mass *= ratio;
    kB_T = CONSTANT_kB * target_temperature_new;
    target_temperature = target_temperature_new;
}

void NOSE_HOOVER_CHAIN_INFORMATION::Save_Restart_File()
{
    if (is_initialized && restart_file_name[0] != 0 &&
        CONTROLLER::MPI_rank == 0)
    {
        FILE* frst = NULL;
        Open_File_Safely(&frst, restart_file_name, "w");
        for (int i = 0; i < chain_length; i++)
        {
            fprintf(frst, "%f %f\n", h_coordinate[i], h_velocity[i]);
        }
        fclose(frst);
    }
}

void NOSE_HOOVER_CHAIN_INFORMATION::Save_Trajectory_File()
{
    if (is_initialized && f_crd_traj != NULL && CONTROLLER::MPI_rank == 0)
    {
        for (int i = 0; i < chain_length; i++)
        {
            fprintf(f_crd_traj, "%f ", h_coordinate[i]);
        }
        fprintf(f_crd_traj, "\n");
    }
    if (is_initialized && f_vel_traj != NULL && CONTROLLER::MPI_rank == 0)
    {
        for (int i = 0; i < chain_length; i++)
        {
            fprintf(f_vel_traj, "%f ", h_velocity[i]);
        }
        fprintf(f_vel_traj, "\n");
    }
}

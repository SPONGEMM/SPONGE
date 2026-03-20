#include "Middle_Langevin_MD.h"

// liu. J, Middle Langevin 热浴迭代算法
// 4个原子为一组计算，每组随机数由
static __global__ void MD_Iteration_Leap_Frog_With_LiuJian(
    Philox4_32_10_t* rand_state, float* rand_float,
    const int local_atom_numbers, const float half_dt, const float dt,
    const float exp_gamma, const float* inverse_mass,
    const float* sqrt_mass_inverse, VECTOR* vel, VECTOR* crd, VECTOR* frc,
    VECTOR* acc, VECTOR* random_frc)
{
#ifdef USE_GPU
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int begin = tid * 4;
    if (begin < local_atom_numbers)
#else
#pragma omp parallel for
    for (int begin = 0; begin < local_atom_numbers; begin += 4)
#endif
    {
        if (begin + 4 <= local_atom_numbers)
        {
            int rand_begin = begin / 4 * 3;
            for (int i = rand_begin; i < rand_begin + 3; ++i)
            {
                float4* rand_float4 = reinterpret_cast<float4*>(rand_float);
                device_get_4_normal_distributed_random_numbers(rand_float4,
                                                               rand_state, i);
            }
            for (int i = begin; i < begin + 4; ++i)
            {
                VECTOR temp1 = inverse_mass[i] * frc[i];
                temp1 = vel[i] + dt * temp1;
                VECTOR temp2 = crd[i] + half_dt * temp1;
                temp1 =
                    exp_gamma * temp1 + sqrt_mass_inverse[i] * random_frc[i];
                vel[i] = temp1;
                crd[i] = temp2 + half_dt * temp1;
            }
        }
        else
        {
            int size = local_atom_numbers - begin;
            int rand_begin = begin / 4 * 3;
            int rand_end = rand_begin + (size * 3 + 3) / 4;
            for (int i = rand_begin; i < rand_end; ++i)
            {
                float4* rand_float4 = reinterpret_cast<float4*>(rand_float);
                device_get_4_normal_distributed_random_numbers(rand_float4,
                                                               rand_state, i);
            }
            for (int i = begin; i < begin + size; ++i)
            {
                VECTOR temp1 = inverse_mass[i] * frc[i];
                temp1 = vel[i] + dt * temp1;
                VECTOR temp2 = crd[i] + half_dt * temp1;
                temp1 =
                    exp_gamma * temp1 + sqrt_mass_inverse[i] * random_frc[i];
                vel[i] = temp1;
                crd[i] = temp2 + half_dt * temp1;
            }
        }
    }
}

// liu. J, Middle Langevin 热浴迭代算法
// 4个原子为一组计算，每组随机数由
static __global__ void MD_Iteration_Leap_Frog_With_LiuJian_With_Max_Velocity(
    Philox4_32_10_t* rand_state, float* rand_float,
    const int local_atom_numbers, const float half_dt, const float dt,
    const float exp_gamma, const float* inverse_mass,
    const float* sqrt_mass_inverse, VECTOR* vel, VECTOR* crd, VECTOR* frc,
    VECTOR* acc, VECTOR* random_frc, const float max_vel)
{
#ifdef USE_GPU
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int begin = tid * 4;
    if (begin < local_atom_numbers)
#else
#pragma omp parallel for
    for (int begin = 0; begin < local_atom_numbers; begin += 4)
#endif
    {
        if (begin + 4 <= local_atom_numbers)
        {
            int rand_begin = begin / 4 * 3;
            for (int i = rand_begin; i < rand_begin + 3; ++i)
            {
                float4* rand_float4 = reinterpret_cast<float4*>(rand_float);
                device_get_4_normal_distributed_random_numbers(rand_float4,
                                                               rand_state, i);
            }
            for (int i = begin; i < begin + 4; ++i)
            {
                VECTOR temp1 = inverse_mass[i] * frc[i];
                temp1 = vel[i] + dt * temp1;
                temp1 = Make_Vector_Not_Exceed_Value(temp1, max_vel);
                VECTOR temp2 = crd[i] + half_dt * temp1;
                temp1 =
                    exp_gamma * temp1 + sqrt_mass_inverse[i] * random_frc[i];
                vel[i] = temp1;
                crd[i] = temp2 + half_dt * temp1;
            }
        }
        else
        {
            int size = local_atom_numbers - begin;
            int rand_begin = begin / 4 * 3;
            int rand_end = rand_begin + (size * 3 + 3) / 4;
            for (int i = rand_begin; i < rand_end; ++i)
            {
                float4* rand_float4 = reinterpret_cast<float4*>(rand_float);
                device_get_4_normal_distributed_random_numbers(rand_float4,
                                                               rand_state, i);
            }
            for (int i = begin; i < begin + size; ++i)
            {
                VECTOR temp1 = inverse_mass[i] * frc[i];
                temp1 = vel[i] + dt * temp1;
                temp1 = Make_Vector_Not_Exceed_Value(temp1, max_vel);
                VECTOR temp2 = crd[i] + half_dt * temp1;
                temp1 =
                    exp_gamma * temp1 + sqrt_mass_inverse[i] * random_frc[i];
                vel[i] = temp1;
                crd[i] = temp2 + half_dt * temp1;
            }
        }
    }
}

void MIDDLE_Langevin_INFORMATION::Initial(CONTROLLER* controller,
                                          const int atom_numbers,
                                          const float target_temperature,
                                          const float* h_mass,
                                          const char* module_name)
{
    controller->printf("START INITIALIZING MIDDLE LANGEVIN DYNAMICS:\n");
    if (controller->Command_Choice("thermostat", "langevin") ||
        controller->Command_Choice("thermostat_mode", "langevin"))
    {
        if (controller->Command_Choice("thermostat", "langevin"))
        {
            controller->Deprecated("thermostat", "thermostat = middle_langevin",
                                   "2.0",
                                   "The thermostat 'langevin' has been "
                                   "deprecated, use middle_langevin instead.");
        }
        if (controller->Command_Choice("thermostat_mode", "langevin"))
        {
            controller->Deprecated("thermostat_mode",
                                   "thermostat_mode = middle_langevin", "2.0",
                                   "The thermostat 'langevin' has been "
                                   "deprecated, use middle_langevin instead.");
        }
    }
    if (module_name == NULL)
    {
        strcpy(this->module_name, "middle_langevin");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }

    float* h_mass_temp = NULL;
    this->atom_numbers = atom_numbers;
    this->target_temperature = target_temperature;
    controller->printf("    atom_numbers is %d\n", atom_numbers);
    Malloc_Safely((void**)&h_mass_temp, sizeof(float) * atom_numbers);
    deviceMemcpy(h_mass_temp, h_mass, sizeof(float) * atom_numbers,
                 deviceMemcpyHostToHost);

    gamma_ln = 1.0f;
    if (controller->Command_Exist("thermostat", "tau"))
    {
        controller->Check_Float("thermostat", "tau",
                                "MIDDLE_Langevin_INFORMATION::Initial");
        gamma_ln = atof(controller->Command("thermostat", "tau"));
        gamma_ln = 1.0f / gamma_ln;
    }

    int random_seed = rand();
    if (controller->Command_Exist("thermostat", "seed"))
    {
        controller->Check_Int("thermostat", "seed",
                              "MIDDLE_Langevin_INFORMATION::Initial");
        random_seed = atoi(controller->Command("thermostat", "seed"));
    }

    controller->Deprecated("middle_langevin_gamma",
                           "thermostat_tau = 1.0 / %VALUE%", "1.5",
                           "The thermostat parameters have been managed "
                           "uniformly since version 1.5");
    controller->Deprecated("middle_langevin_seed", "thermostat_seed = %VALUE%",
                           "1.5",
                           "The thermostat parameters have been managed "
                           "uniformly since version 1.5");

    controller->printf("    target temperature is %.2f K\n",
                       target_temperature);
    controller->printf("    friction coefficient is %.2f ps^-1\n", gamma_ln);
    controller->printf("    random seed is %d\n", random_seed);

    dt = 0.001;
    if (controller->Command_Exist("dt")) dt = atof(controller->Command("dt"));
    dt *= CONSTANT_TIME_CONVERTION;
    half_dt = 0.5 * dt;

    float4_numbers = (3 * atom_numbers + 3) / 4;
    Device_Malloc_Safely((void**)&random_force,
                         sizeof(float4) * float4_numbers);
    Device_Malloc_Safely((void**)&rand_state,
                         sizeof(Philox4_32_10_t) * float4_numbers);

    Launch_Device_Kernel(Setup_Rand_Normal_Kernel,
                         (float4_numbers + CONTROLLER::device_max_thread - 1) /
                             CONTROLLER::device_max_thread,
                         CONTROLLER::device_max_thread, 0, NULL, float4_numbers,
                         rand_state, random_seed);

    gamma_ln = gamma_ln / CONSTANT_TIME_CONVERTION;  // 单位换算

    exp_gamma = expf(-gamma_ln * dt);

    float sart_gamma =
        sqrtf((1. - exp_gamma * exp_gamma) * target_temperature * CONSTANT_kB);
    Malloc_Safely((void**)&h_sqrt_mass, sizeof(float) * atom_numbers);
    for (int i = 0; i < atom_numbers; i = i + 1)
    {
        if (h_mass_temp[i] == 0)
            h_sqrt_mass[i] = 0;
        else
            h_sqrt_mass[i] = sart_gamma * sqrtf(1. / h_mass_temp[i]);
    }
    Device_Malloc_And_Copy_Safely((void**)&d_sqrt_mass, h_sqrt_mass,
                                  sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&d_sqrt_mass_local,
                         sizeof(float) * atom_numbers);

    // 确定是否加上速度上限
    max_velocity = 0;
    if (controller->Command_Exist("velocity_max"))
    {
        sscanf(controller->Command("velocity_max"), "%f", &max_velocity);
        controller->printf("    max velocity is %.2f\n", max_velocity);
    }
    // 记录质量的倒数
    for (int i = 0; i < atom_numbers; i = i + 1)
    {
        if (h_mass_temp[i] == 0)
            h_mass_temp[i] = 0;
        else
            h_mass_temp[i] = 1.0f / h_mass_temp[i];
    }
    Device_Malloc_Safely((void**)&d_mass_inverse, sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&d_mass_inverse_local,
                         sizeof(float) * atom_numbers);
    deviceMemcpy(d_mass_inverse, h_mass_temp, sizeof(float) * atom_numbers,
                 deviceMemcpyHostToDevice);
    free(h_mass_temp);
    is_initialized = 1;
    if (is_initialized && !is_controller_printf_initialized)
    {
        is_controller_printf_initialized = 1;
        controller->printf("    structure last modify date is %d\n",
                           last_modify_date);
    }
    controller->printf("END INITIALIZING MIDDLE LANGEVIN DYNAMICS\n\n");
}

static __global__ void device_get_local(int* atom_local, int local_atom_numbers,
                                        float* d_sqrt_mass,
                                        float* d_sqrt_mass_local,
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
        d_sqrt_mass_local[i] = d_sqrt_mass[atom_i];
        d_mass_inverse_local[i] = d_mass_inverse[atom_i];
    }
}

void MIDDLE_Langevin_INFORMATION::Get_Local(int* atom_local,
                                            int local_atom_numbers)
{
    if (!is_initialized) return;
    this->local_atom_numbers = local_atom_numbers;
    this->local_float4_numbers = (local_atom_numbers * 3 + 3) / 4;
    Launch_Device_Kernel(
        device_get_local,
        (local_atom_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, atom_local, local_atom_numbers,
        d_sqrt_mass, d_sqrt_mass_local, d_mass_inverse, d_mass_inverse_local);
}

void MIDDLE_Langevin_INFORMATION::MD_Iteration_Leap_Frog(VECTOR* frc,
                                                         VECTOR* vel,
                                                         VECTOR* acc,
                                                         VECTOR* crd)
{
    if (is_initialized)
    {
        if (max_velocity <= 0)
        {
            Launch_Device_Kernel(MD_Iteration_Leap_Frog_With_LiuJian,
                                 (local_atom_numbers + 32 - 1) / 32, 32, 0,
                                 NULL, rand_state, (float*)random_force,
                                 local_atom_numbers, half_dt, dt, exp_gamma,
                                 d_mass_inverse_local, d_sqrt_mass_local, vel,
                                 crd, frc, acc, random_force);
        }
        else
        {
            Launch_Device_Kernel(
                MD_Iteration_Leap_Frog_With_LiuJian_With_Max_Velocity,
                (local_atom_numbers + 32 - 1) / 32, 32, 0, NULL, rand_state,
                (float*)random_force, local_atom_numbers, half_dt, dt,
                exp_gamma, d_mass_inverse_local, d_sqrt_mass_local, vel, crd,
                frc, acc, random_force, max_velocity);
        }
    }
}

void MIDDLE_Langevin_INFORMATION::Set_Target_Temperature(
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
    float scale = sqrtf(target_temperature_new / target_temperature);
    if (!(scale > 0.0f) || !isfinite(scale))
    {
        return;
    }

    for (int i = 0; i < atom_numbers; i++)
    {
        h_sqrt_mass[i] *= scale;
    }
    Scale_List(d_sqrt_mass, scale, atom_numbers);
    if (local_atom_numbers > 0)
    {
        Scale_List(d_sqrt_mass_local, scale, local_atom_numbers);
    }
    target_temperature = target_temperature_new;
}

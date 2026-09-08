#include "Andersen_thermostat.h"

#include "../utils/float_classification.hpp"
#include "../utils/random/restart_rng_state.hpp"

static __global__ void MD_Iteration_Leap_Frog_With_Andersen(
    const std::uint64_t random_seed,
    const std::uint64_t random_invocation_count, float* rand_float,
    const int local_atom_numbers, const float half_dt, const float dt,
    const float* inverse_mass, const float* factor, VECTOR* vel, VECTOR* crd,
    VECTOR* frc, VECTOR* acc, VECTOR* random_vel)
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
                SPONGE_PHILOX4X32_10 rng(random_seed, i,
                                         random_invocation_count * 4);
                rng.Normal4(rand_float + 4 * i);
            }
            for (int i = begin; i < begin + 4; ++i)
            {
                VECTOR temp1 = inverse_mass[i] * frc[i];
                temp1 = vel[i] + dt * temp1;
                VECTOR temp2 = crd[i] + half_dt * temp1;

                temp1 = factor[i] * random_vel[i];
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
                SPONGE_PHILOX4X32_10 rng(random_seed, i,
                                         random_invocation_count * 4);
                rng.Normal4(rand_float + 4 * i);
            }
            for (int i = begin; i < begin + size; ++i)
            {
                VECTOR temp1 = inverse_mass[i] * frc[i];
                temp1 = vel[i] + dt * temp1;
                VECTOR temp2 = crd[i] + half_dt * temp1;

                temp1 = factor[i] * random_vel[i];
                vel[i] = temp1;
                crd[i] = temp2 + half_dt * temp1;
            }
        }
    }
}

static __global__ void MD_Iteration_Leap_Frog_With_Andersen_With_Max_Velocity(
    const std::uint64_t random_seed,
    const std::uint64_t random_invocation_count, float* rand_float,
    const int local_atom_numbers, const float half_dt, const float dt,
    const float* inverse_mass, const float* factor, VECTOR* vel, VECTOR* crd,
    VECTOR* frc, VECTOR* acc, VECTOR* random_vel, const float max_vel)
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
                SPONGE_PHILOX4X32_10 rng(random_seed, i,
                                         random_invocation_count * 4);
                rng.Normal4(rand_float + 4 * i);
            }
            for (int i = begin; i < begin + 4; ++i)
            {
                VECTOR temp1 = inverse_mass[i] * frc[i];
                temp1 = vel[i] + dt * temp1;
                temp1 = Make_Vector_Not_Exceed_Value(temp1, max_vel);
                VECTOR temp2 = crd[i] + half_dt * temp1;

                temp1 = factor[i] * random_vel[i];
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
                SPONGE_PHILOX4X32_10 rng(random_seed, i,
                                         random_invocation_count * 4);
                rng.Normal4(rand_float + 4 * i);
            }
            for (int i = begin; i < begin + size; ++i)
            {
                VECTOR temp1 = inverse_mass[i] * frc[i];
                temp1 = vel[i] + dt * temp1;
                temp1 = Make_Vector_Not_Exceed_Value(temp1, max_vel);
                VECTOR temp2 = crd[i] + half_dt * temp1;

                temp1 = factor[i] * random_vel[i];
                vel[i] = temp1;
                crd[i] = temp2 + half_dt * temp1;
            }
        }
    }
}

void ANDERSEN_THERMOSTAT_INFORMATION::Initial(CONTROLLER* controller,
                                              float target_temperature,
                                              int atom_numbers, float dt_in_ps,
                                              float* h_mass,
                                              const char* module_name)
{
    controller->printf("START INITIALIZING ANDERSEN THERMOSTAT:\n");
    if (module_name == NULL)
    {
        strcpy(this->module_name, "andersen_thermostat");
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
    controller[0].printf("    target temperature is %.2f K\n",
                         target_temperature);

    int random_seed = time(NULL);
    if (controller[0].Command_Exist("thermostat", "seed"))
    {
        controller->Check_Int("thermostat", "seed",
                              "ANDERSEN_THERMOSTAT_INFORMATION::Initial");
        random_seed = atoi(controller[0].Command("thermostat", "seed"));
    }
    controller[0].printf("    random seed is %d\n", random_seed);
    this->random_seed =
        static_cast<std::uint64_t>(static_cast<std::uint32_t>(random_seed));
    random_invocation_count = 0;
    float4_numbers = (3 * atom_numbers + 3) / 4;
    Device_Malloc_Safely((void**)&random_vel, sizeof(float4) * float4_numbers);
    float factor = sqrtf(CONSTANT_kB * target_temperature);
    Malloc_Safely((void**)&h_factor, sizeof(float) * atom_numbers);

    for (int i = 0; i < atom_numbers; i = i + 1)
    {
        if (h_mass[i] == 0)
            h_factor[i] = 0;
        else
        {
            h_factor[i] = factor * sqrtf(1.0 / h_mass[i]);
        }
    }

    Device_Malloc_And_Copy_Safely((void**)&d_factor, h_factor,
                                  sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&d_factor_local, sizeof(float) * atom_numbers);

    float tau = 1.0f;
    if (controller[0].Command_Exist("thermostat", "tau"))
    {
        controller->Check_Float("thermostat", "tau",
                                "ANDERSEN_THERMOSTAT_INFORMATION::Initial");
        tau = atof(controller->Command("thermostat", "tau"));
    }
    update_interval = roundf(tau / dt_in_ps);
    controller->printf("    The update_interval is %d\n", update_interval);

    controller->Deprecated("andersen_thermostat_seed",
                           "thermostat_seed = %VALUE%", "1.5",
                           "The thermostat parameters have been managed "
                           "uniformly since version 1.5");
    controller->Deprecated("andersen_thermostat_update_interval",
                           "thermostat_tau = %VALUE%", "1.5",
                           "The thermostat parameters have been managed "
                           "uniformly since version 1.5");

    // 确定是否加上速度上限
    max_velocity = 0;
    if (controller[0].Command_Exist("velocity_max"))
    {
        controller->Check_Float("velocity_max",
                                "ANDERSEN_THERMOSTAT_INFORMATION::Initial");
        sscanf(controller[0].Command("velocity_max"), "%f", &max_velocity);
        controller[0].printf("    max velocity is %.2f\n", max_velocity);
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
        controller[0].printf("    structure last modify date is %d\n",
                             last_modify_date);
    }

    controller->printf("END INITIALIZING ANDERSEN THERMOSTAT\n\n");
}

static __global__ void device_get_local(int* atom_local, int local_atom_numbers,
                                        float* d_factor, float* d_factor_local,
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
        d_factor_local[i] = d_factor[atom_i];
        d_mass_inverse_local[i] = d_mass_inverse[atom_i];
    }
}

void ANDERSEN_THERMOSTAT_INFORMATION::Get_Local(int* atom_local,
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
        d_factor, d_factor_local, d_mass_inverse, d_mass_inverse_local);
}

void ANDERSEN_THERMOSTAT_INFORMATION::MD_Iteration_Leap_Frog(
    VECTOR* vel, VECTOR* crd, VECTOR* frc, VECTOR* acc, float dt)
{
    if (is_initialized)
    {
        if (max_velocity <= 0)
        {
            Launch_Device_Kernel(
                MD_Iteration_Leap_Frog_With_Andersen,
                (local_atom_numbers + 32 - 1) / 32, 32, 0, NULL, random_seed,
                random_invocation_count, (float*)random_vel, local_atom_numbers,
                0.5f * dt, dt, d_mass_inverse_local, d_factor_local, vel, crd,
                frc, acc, random_vel);
        }
        else
        {
            Launch_Device_Kernel(
                MD_Iteration_Leap_Frog_With_Andersen_With_Max_Velocity,
                (local_atom_numbers + 32 - 1) / 32, 32, 0, NULL, random_seed,
                random_invocation_count, (float*)random_vel, local_atom_numbers,
                0.5f * dt, dt, d_mass_inverse_local, d_factor_local, vel, crd,
                frc, acc, random_vel, max_velocity);
        }
        random_invocation_count += 1;
    }
}

void ANDERSEN_THERMOSTAT_INFORMATION::Set_Target_Temperature(
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
    if (!(scale > 0.0f) || !SpongeFloat::Is_Finite(scale))
    {
        return;
    }

    for (int i = 0; i < atom_numbers; i++)
    {
        h_factor[i] *= scale;
    }
    Scale_List(d_factor, scale, atom_numbers);
    if (local_atom_numbers > 0)
    {
        Scale_List(d_factor_local, scale, local_atom_numbers);
    }
    target_temperature = target_temperature_new;
}

bool ANDERSEN_THERMOSTAT_INFORMATION::Export_H5_Restart_State(
    SpongeH5MD::RestartDynamicState* state, std::string* error_message) const
{
    if (state == nullptr)
    {
        if (error_message != nullptr)
        {
            *error_message = "Andersen H5 restart state output pointer is null";
        }
        return false;
    }
    if (!is_initialized) return true;
    state->rng_states["andersen"] = SpongeRestartRng::Counter_Philox_State(
        random_seed, random_invocation_count);
    return true;
}

bool ANDERSEN_THERMOSTAT_INFORMATION::Apply_H5_Restart_State(
    const SpongeH5MD::RestartDynamicState& state, std::string* error_message)
{
    if (!is_initialized)
    {
        if (error_message != nullptr)
        {
            *error_message = "Andersen thermostat module is not initialized";
        }
        return false;
    }
    const auto rng = state.rng_states.find("andersen");
    if (rng == state.rng_states.end())
    {
        if (error_message != nullptr)
        {
            *error_message = "Andersen restart is missing typed RNG state";
        }
        return false;
    }
    return SpongeRestartRng::Decode_Counter_Philox_State(
        rng->second, &random_seed, &random_invocation_count, error_message);
}

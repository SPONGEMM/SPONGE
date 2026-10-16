#pragma once

static __global__ void MD_Iteration_Leap_Frog(const int atom_numbers,
                                              VECTOR* vel, VECTOR* crd,
                                              VECTOR* frc,
                                              const float* inverse_mass,
                                              const float dt)
{
    SIMPLE_DEVICE_FOR(i, atom_numbers)
    {
        VECTOR acc_i = inverse_mass[i] * frc[i];
        VECTOR vel_i = vel[i] + dt * acc_i;
        vel[i] = vel_i;
        crd[i] = crd[i] + dt * vel_i;
    }
}

static __global__ void MD_Iteration_Leap_Frog_With_Max_Velocity(
    const int atom_numbers, VECTOR* vel, VECTOR* crd, VECTOR* frc,
    const float* inverse_mass, const float dt, const float max_velocity)
{
    SIMPLE_DEVICE_FOR(i, atom_numbers)
    {
        VECTOR acc_i = inverse_mass[i] * frc[i];
        VECTOR vel_i = vel[i] + dt * acc_i;
        vel_i = Make_Vector_Not_Exceed_Value(vel_i, max_velocity);
        vel[i] = vel_i;
        crd[i] = crd[i] + dt * vel_i;
    }
}

void MD_INFORMATION::NVE_iteration::Leap_Frog(const int atom_numbers,
                                              VECTOR* vel, VECTOR* crd,
                                              VECTOR* frc,
                                              const float* inverse_mass,
                                              const float dt)
{
    if (atom_numbers == 0) return;
    if (max_velocity <= 0)
    {
        Launch_Device_Kernel(
            MD_Iteration_Leap_Frog,
            (atom_numbers + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, atom_numbers, vel, crd, frc,
            inverse_mass, dt);
    }
    else
    {
        Launch_Device_Kernel(
            MD_Iteration_Leap_Frog_With_Max_Velocity,
            (atom_numbers + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, atom_numbers, vel, crd, frc,
            inverse_mass, dt, max_velocity);
    }
}

void MD_INFORMATION::NVE_iteration::Initial(CONTROLLER* controller,
                                            MD_INFORMATION* md_info)
{
    this->md_info = md_info;
    max_velocity = -1;
    if (controller[0].Command_Exist("velocity_max"))
    {
        controller->Check_Float("velocity_max",
                                "MD_INFORMATION::NVE_iteration::Initial");
        max_velocity = atof(controller[0].Command("velocity_max"));
    }
}

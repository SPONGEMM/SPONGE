#pragma once

static __global__ void MD_Iteration_Gradient_Descent(
    const int atom_numbers, VECTOR* crd, VECTOR* frc, const float* mass_inverse,
    const float dt, VECTOR* vel, const float momentum_keep)
{
    SIMPLE_DEVICE_FOR(i, atom_numbers)
    {
        VECTOR vel_i = vel[i] + dt * mass_inverse[i] * frc[i];
        crd[i] = crd[i] + dt * vel_i;
        vel[i] = momentum_keep * vel_i;
    }
}

static __global__ void MD_Iteration_Gradient_Descent_With_Max_Move(
    const int atom_numbers, VECTOR* crd, VECTOR* frc, const float* mass_inverse,
    const float dt, VECTOR* vel, const float momentum_keep, float max_move)
{
    SIMPLE_DEVICE_FOR(i, atom_numbers)
    {
        VECTOR vel_i = vel[i] + dt * mass_inverse[i] * frc[i];
        VECTOR move = dt * vel_i;
        move = Make_Vector_Not_Exceed_Value(move, max_move);
        crd[i] = crd[i] + move;
        vel[i] = momentum_keep * vel_i;
    }
}

static __global__ void Get_Adam_Force(int atom_numbers, float* mass_inverse,
                                      VECTOR* frc, VECTOR* vel, VECTOR* acc,
                                      float beta1, float beta2, float epsilon,
                                      float t, float learning_rate)
{
    SIMPLE_DEVICE_FOR(i, atom_numbers)
    {
        if (mass_inverse[i] == 0.0f)
        {
            frc[i].x = 0.0f;
            frc[i].y = 0.0f;
            frc[i].z = 0.0f;
        }
        else
        {
            const double first_bias = 1.0 - pow(static_cast<double>(beta1),
                                                static_cast<double>(t) + 1.0);
            const double second_bias_sqrt =
                sqrt(1.0 - pow(static_cast<double>(beta2),
                               static_cast<double>(t) + 1.0));
            VECTOR f = frc[i];
            f.x = fminf(fmaxf(f.x, -1e10f), 1e10f);
            f.y = fminf(fmaxf(f.y, -1e10f), 1e10f);
            f.z = fminf(fmaxf(f.z, -1e10f), 1e10f);
            VECTOR moment = vel[i];
            VECTOR root = acc[i];
            double m = static_cast<double>(beta1) * moment.x +
                       (1.0 - static_cast<double>(beta1)) * f.x;
            double r = sqrt(static_cast<double>(beta2) * root.x * root.x +
                            (1.0 - static_cast<double>(beta2)) *
                                static_cast<double>(f.x) * f.x);
            moment.x = static_cast<float>(m);
            root.x = static_cast<float>(r);
            f.x = static_cast<float>(learning_rate * (m / first_bias) /
                                     (r / second_bias_sqrt + epsilon));
            m = static_cast<double>(beta1) * moment.y +
                (1.0 - static_cast<double>(beta1)) * f.y;
            r = sqrt(static_cast<double>(beta2) * root.y * root.y +
                     (1.0 - static_cast<double>(beta2)) *
                         static_cast<double>(f.y) * f.y);
            moment.y = static_cast<float>(m);
            root.y = static_cast<float>(r);
            f.y = static_cast<float>(learning_rate * (m / first_bias) /
                                     (r / second_bias_sqrt + epsilon));
            m = static_cast<double>(beta1) * moment.z +
                (1.0 - static_cast<double>(beta1)) * f.z;
            r = sqrt(static_cast<double>(beta2) * root.z * root.z +
                     (1.0 - static_cast<double>(beta2)) *
                         static_cast<double>(f.z) * f.z);
            moment.z = static_cast<float>(m);
            root.z = static_cast<float>(r);
            f.z = static_cast<float>(learning_rate * (m / first_bias) /
                                     (r / second_bias_sqrt + epsilon));
            vel[i] = moment;
            acc[i] = root;
            frc[i] = f;
        }
    }
}

static __global__ void MD_Iteration_Adam_Move(const int atom_numbers,
                                              VECTOR* crd, const VECTOR* frc,
                                              const float max_move)
{
    SIMPLE_DEVICE_FOR(i, atom_numbers)
    {
        VECTOR move = frc[i];
        if (max_move > 0)
        {
            move = Make_Vector_Not_Exceed_Value(move, max_move);
        }
        crd[i] = crd[i] + move;
    }
}

void MD_INFORMATION::MINIMIZATION_iteration::Initial(CONTROLLER* controller,
                                                     MD_INFORMATION* md_info)
{
    this->md_info = md_info;
    if (md_info->mode == MINIMIZATION)
    {
        controller->printf("    Start initializing minimization:\n");
        max_move = 0.1f;
        if (controller[0].Command_Exist("minimization_max_move"))
        {
            controller->Check_Float(
                "minimization", "max_move",
                "MD_INFORMATION::MINIMIZATION_iteration::Initial");
            max_move = atof(controller[0].Command("minimization_max_move"));
        }
        controller->printf("        minimization max move is %f A\n", max_move);

        momentum_keep = 0;
        if (controller[0].Command_Exist("minimization_momentum_keep"))
        {
            controller->Check_Float(
                "minimization", "momentum_keep",
                "MD_INFORMATION::MINIMIZATION_iteration::Initial");
            momentum_keep =
                atof(controller[0].Command("minimization_momentum_keep"));
        }
        controller->printf("        minimization momentum keep is %f\n",
                           momentum_keep);

        dynamic_dt = 1;
        if (controller[0].Command_Exist("minimization_dynamic_dt"))
        {
            controller->Check_Int(
                "minimization", "dynamic_dt",
                "MD_INFORMATION::MINIMIZATION_iteration::Initial");
            dynamic_dt = atoi(controller[0].Command("minimization_dynamic_dt"));
        }
        controller->printf("        minimization dynamic dt is %d\n",
                           dynamic_dt);

        if (dynamic_dt)
        {
            md_info->dt = 3e-4f;
            momentum_keep = 1;
            beta1 = 0.9;
            if (controller->Command_Exist("minimization", "beta1"))
            {
                controller->Check_Float(
                    "minimization", "beta1",
                    "MD_INFORMATION::MINIMIZATION_iteration::Initial");
                beta1 = atof(controller->Command("minimization", "beta1"));
            }
            controller->printf("        minimization beta1 is %f\n", beta1);

            beta2 = 0.9;
            if (controller->Command_Exist("minimization", "beta1"))
            {
                controller->Check_Float(
                    "minimization", "beta1",
                    "MD_INFORMATION::MINIMIZATION_iteration::Initial");
                beta2 = atof(controller->Command("minimization", "beta1"));
            }
            controller->printf("        minimization beta2 is %f\n", beta2);

            epsilon = 1e-4f;
            if (controller->Command_Exist("minimization", "epsilon"))
            {
                controller->Check_Float(
                    "minimization", "epsilon",
                    "MD_INFORMATION::MINIMIZATION_iteration::Initial");
                epsilon = atof(controller->Command("minimization", "epsilon"));
            }
            controller->printf("        minimization epsilon is %e\n", epsilon);

            if (controller->Command_Exist("minimization", "learning_rate"))
            {
                controller->Check_Float(
                    "minimization", "learning_rate",
                    "MD_INFORMATION::MINIMIZATION_iteration::Initial");
                learning_rate =
                    atof(controller->Command("minimization", "learning_rate"));
            }
            controller->printf("        minimization learning rate is %e A\n",
                               learning_rate);
        }
        else
        {
            md_info->dt = 1e-8f;
            momentum_keep = 0;
            if (controller->Command_Exist("minimization", "momentum_keep"))
            {
                controller->Check_Float(
                    "minimization", "momentum_keep",
                    "MD_INFORMATION::MINIMIZATION_iteration::Initial");
                momentum_keep =
                    atof(controller->Command("minimization", "momentum_keep"));
            }
            controller->printf("        minimization momentum_keep is %f\n",
                               momentum_keep);
        }
        controller->printf("    End initializing minimization\n\n");
    }
}

void MD_INFORMATION::MINIMIZATION_iteration::Gradient_Descent(
    int atom_numbers, VECTOR* crd, VECTOR* frc, VECTOR* vel,
    const float* d_mass_inverse)
{
    if (dynamic_dt)
    {
        Launch_Device_Kernel(
            MD_Iteration_Adam_Move,
            (atom_numbers + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, atom_numbers, crd, frc,
            max_move);
    }
    else if (max_move <= 0)
    {
        Launch_Device_Kernel(
            MD_Iteration_Gradient_Descent,
            (atom_numbers + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, atom_numbers, crd, frc,
            d_mass_inverse, md_info->dt, vel, momentum_keep);
    }
    else
    {
        Launch_Device_Kernel(
            MD_Iteration_Gradient_Descent_With_Max_Move,
            (atom_numbers + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, atom_numbers, crd, frc,
            d_mass_inverse, md_info->dt, vel, momentum_keep, max_move);
    }
}

void MD_INFORMATION::MINIMIZATION_iteration::Scale_Force_For_Dynamic_Dt(
    int atom_numbers, float* d_mass_inverse, VECTOR* frc, VECTOR* vel,
    VECTOR* acc)
{
    if (md_info->mode == MINIMIZATION && dynamic_dt)
    {
        Launch_Device_Kernel(
            Get_Adam_Force,
            (atom_numbers + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, atom_numbers,
            d_mass_inverse, frc, vel, acc, beta1, beta2, epsilon,
            md_info->sys.steps, learning_rate);
    }
}

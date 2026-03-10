#include "pressure_based_barostat.h"

void PRESSURE_BASED_BAROSTAT_INFORMATION::Initial(
    CONTROLLER* controller, float target_pressure, LTMatrix3 cell,
    float (*box_updator)(LTMatrix3, int, int, int))
{
    controller->printf("START INITIALIZING PRESSURE BASED BAROSTAT:\n");
    this->box_updator = box_updator;
    this->extreme_box_updator = extreme_box_updator;
    if (controller->Command_Choice("barostat", "andersen_barostat") ||
        controller->Command_Choice("barostat_mode", "andersen_barostat"))
    {
        this->Algorithm = this->Andersen;
        controller->printf("    The algorithm is Andersen\n");
    }
    else if (controller->Command_Choice("barostat", "berendsen_barostat") ||
             controller->Command_Choice("barostat_mode", "berendsen_barostat"))
    {
        this->Algorithm = this->Berendsen;
        controller->printf("    The algorithm is Berendsen\n");
    }
    else if (controller->Command_Choice("barostat", "bussi_barostat") ||
             controller->Command_Choice("barostat_mode", "bussi_barostat"))
    {
        this->Algorithm = this->Bussi;
        controller->printf("    The algorithm is Bussi\n");
    }
    controller->printf("    The target pressure is %.2f bar\n",
                       target_pressure * CONSTANT_PRES_CONVERTION);

    target_surface_tension = 0;
    if (controller->Command_Exist("target", "surface_tensor"))
    {
        controller->Check_Float("target", "surface_tensor",
                                "PRESSURE_BASED_BAROSTAT_INFORMATION::Initial");
        target_surface_tension =
            atof(controller->Command("target", "surface_tensor"));
        target_surface_tension *= 100 * CONSTANT_PRES_CONVERTION_INVERSE;
    }
    else if (controller->Command_Exist("barostat_target", "surface_tensor"))
    {
        controller->Check_Float("barostat_target", "surface_tensor",
                                "PRESSURE_BASED_BAROSTAT_INFORMATION::Initial");
        target_surface_tension =
            atof(controller->Command("barostat_target", "surface_tensor"));
        target_surface_tension *= 100 * CONSTANT_PRES_CONVERTION_INVERSE;
    }

    if (!controller->Command_Exist("barostat", "isotropy") ||
        controller->Command_Choice("barostat_isotropy", "isotropic"))
    {
        this->Isotropy = this->Isotropic;
        controller->printf("    The isotropy is isotropic\n");
    }
    else if (controller->Command_Choice("barostat_isotropy", "semiisotropic"))
    {
        this->Isotropy = this->Semiisotropic;
        controller->printf("    The isotropy is semiisotropic\n");
    }
    else if (controller->Command_Choice("barostat_isotropy", "semianisotropic"))
    {
        this->Isotropy = this->Semianisotropic;
        controller->printf("    The isotropy is semianisotropic\n");
    }
    else if (controller->Command_Choice("barostat_isotropy", "anisotropic"))
    {
        this->Isotropy = this->Anisotropic;
        controller->printf("    The isotropy is anisotropic\n");
    }

    V0 = cell.a11 * cell.a22 * cell.a33;

    float taup = 1.0f;
    if (controller[0].Command_Exist("barostat", "tau"))
    {
        controller->Check_Float("barostat", "tau",
                                "PRESSURE_BASED_BAROSTAT_INFORMATION::Initial");
        taup = atof(controller[0].Command("barostat", "tau"));
    }
    controller->printf("    The time constant tau is %f ps\n", taup);

    float compressibility = 4.5e-5f;
    if (controller[0].Command_Exist("barostat", "compressibility"))
    {
        controller->Check_Float("barostat", "compressibility",
                                "PRESSURE_BASED_BAROSTAT_INFORMATION::Initial");
        compressibility =
            atof(controller[0].Command("barostat", "compressibility"));
    }
    controller->printf("    The compressibility constant is %f bar^-1\n",
                       compressibility);

    update_interval = 10;
    if (controller->Command_Exist("barostat", "update_interval"))
    {
        controller->Check_Int("barostat", "update_interval",
                              "PRESSURE_BASED_BAROSTAT_INFORMATION::Initial");
        update_interval =
            atoi(controller->Command("barostat", "update_interval"));
    }
    controller->printf("    The update interval is %d\n", update_interval);

    float d = powf(V0, 2.0f);
    piston_mass_inverse = d * taup * taup / compressibility / V0;
    controller->printf("    The piston mass is %f bar ps^2 / A^3\n",
                       piston_mass_inverse);
    taup *= CONSTANT_TIME_CONVERTION;
    compressibility *= CONSTANT_PRES_CONVERTION;
    piston_mass_inverse = V0 * compressibility / taup / taup / d;

    g.a11 = 0;
    if (controller->Command_Exist("barostat", "g11"))
    {
        controller->Check_Float("barostat", "g11",
                                "PRESSURE_BASED_BAROSTAT_INFORMATION::Initial");
        g.a11 = atof(controller->Command("barostat", "g11"));
    }

    g.a21 = 0;
    if (controller->Command_Exist("barostat", "g21"))
    {
        controller->Check_Float("barostat", "g21",
                                "PRESSURE_BASED_BAROSTAT_INFORMATION::Initial");
        g.a21 = atof(controller->Command("barostat", "g21"));
    }

    g.a22 = 0;
    if (controller->Command_Exist("barostat", "g22"))
    {
        controller->Check_Float("barostat", "g22",
                                "PRESSURE_BASED_BAROSTAT_INFORMATION::Initial");
        g.a22 = atof(controller->Command("barostat", "g22"));
    }

    g.a31 = 0;
    if (controller->Command_Exist("barostat", "g31"))
    {
        controller->Check_Float("barostat", "g31",
                                "PRESSURE_BASED_BAROSTAT_INFORMATION::Initial");
        g.a31 = atof(controller->Command("barostat", "g31"));
    }

    g.a32 = 0;
    if (controller->Command_Exist("barostat", "g32"))
    {
        controller->Check_Float("barostat", "g32",
                                "PRESSURE_BASED_BAROSTAT_INFORMATION::Initial");
        g.a32 = atof(controller->Command("barostat", "g32"));
    }
    g.a33 = 0;
    if (controller->Command_Exist("barostat", "g33"))
    {
        controller->Check_Float("barostat", "g33",
                                "PRESSURE_BASED_BAROSTAT_INFORMATION::Initial");
        g.a33 = atof(controller->Command("barostat", "g33"));
    }
    controller->printf(
        "    The initial box velocity is (in A^3/(1/20.455 fs))\n        %8e "
        "%8e %8e\n        %8e %8e %8e\n        %8e %8e %8e\n",
        g.a11, 0.0, 0.0, g.a21, g.a22, 0.0, g.a31, g.a32, g.a33);

    generator = std::default_random_engine(rand());
    distribution = std::normal_distribution<float>(0.0, 1.0);

    x_constant = false;
    y_constant = false;
    z_constant = false;
    if (Isotropy != this->Anisotropic)
    {
        if (controller->Command_Exist("barostat", "x_constant"))
        {
            x_constant = controller->Get_Bool(
                "barostat", "x_constant",
                "PRESSURE_BASED_BAROSTAT_INFORMATION::Initial");
        }
        if (controller->Command_Exist("barostat", "y_constant"))
        {
            y_constant = controller->Get_Bool(
                "barostat", "y_constant",
                "PRESSURE_BASED_BAROSTAT_INFORMATION::Initial");
        }
        if (controller->Command_Exist("barostat", "z_constant"))
        {
            z_constant = controller->Get_Bool(
                "barostat", "z_constant",
                "PRESSURE_BASED_BAROSTAT_INFORMATION::Initial");
        }
    }
    is_initialized = 1;

    controller->Step_Print_Initial("density", "%.4f");
    controller->Step_Print_Initial("pressure", "%.2f");
    if (Isotropy == this->Anisotropic)
        controller->Step_Print_Initial("surface_tensor", "%.2f");

    controller->printf("END INITIALIZING ANDERSEN BAROSTAT\n\n");
}

void PRESSURE_BASED_BAROSTAT_INFORMATION::Control_Velocity_Of_Box(
    float dt, float target_temperature, LTMatrix3 dg)
{
    if (!is_initialized) return;
    float gamma_ln;
    LTMatrix3 random_velocity;
    switch (Algorithm)
    {
        case Andersen:
            gamma_ln = 1.0f / CONSTANT_TIME_CONVERTION;
            gamma_ln = exp(-gamma_ln * update_interval * dt);
            random_velocity = {
                distribution(generator), distribution(generator),
                distribution(generator), distribution(generator),
                distribution(generator), distribution(generator)};
            g = dg + gamma_ln * g +
                sqrt((1 - gamma_ln * gamma_ln) * target_temperature *
                     CONSTANT_kB * piston_mass_inverse) *
                    random_velocity;
            break;

        case Bussi:
            random_velocity = {
                distribution(generator), distribution(generator),
                distribution(generator), distribution(generator),
                distribution(generator), distribution(generator)};
            g = dg +
                sqrt(target_temperature * CONSTANT_kB * piston_mass_inverse) *
                    random_velocity;
            break;

        case Berendsen:
            g = dg;
            break;
    }
}

void PRESSURE_BASED_BAROSTAT_INFORMATION::Ask_For_Calculate_Pressure(
    int steps, int* need_pressure)
{
    if (is_initialized && (steps + 1) % update_interval == 0)
    {
        *need_pressure += 1;
    }
}

void PRESSURE_BASED_BAROSTAT_INFORMATION::Regulate_Pressure(
    int steps, LTMatrix3 h_stress, LTMatrix3 cell, float dt,
    float target_pressure, float target_temperature)
{
    if (is_initialized && (steps + 1) % update_interval == 0)
    {
        if (CONTROLLER::MPI_rank == 0)
        {
            float volume = cell.a11 * cell.a22 * cell.a33;
            LTMatrix3 dg = {
                target_pressure,
                0,
                target_pressure,
                0,
                0,
                target_pressure + target_surface_tension / cell.a33};
            dg = volume * piston_mass_inverse * update_interval * dt *
                 (h_stress - dg);
            this->Control_Velocity_Of_Box(dt, target_temperature, dg);

            switch (Isotropy)
            {
                case Isotropic:
                    this->g.a11 =
                        (this->g.a11 + this->g.a22 + this->g.a33) / 3.0f;
                    this->g.a22 = this->g.a11;
                    this->g.a33 = this->g.a11;
                    this->g.a21 = 0;
                    this->g.a31 = 0;
                    this->g.a32 = 0;
                    break;

                case Semiisotropic:
                    this->g.a11 = 0.5f * (this->g.a11 + this->g.a22);
                    this->g.a22 = this->g.a11;
                    this->g.a21 = 0;
                    this->g.a31 = 0;
                    this->g.a32 = 0;
                    break;

                case Semianisotropic:
                    this->g.a21 = 0;
                    this->g.a31 = 0;
                    this->g.a32 = 0;
                    break;

                case Anisotropic:
                    break;
            }

            if (x_constant) this->g.a11 = 0;
            if (y_constant) this->g.a22 = 0;
            if (z_constant) this->g.a33 = 0;
        }
#ifdef USE_MPI
        MPI_Bcast(&g, 6, MPI_FLOAT, 0, MPI_COMM_WORLD);
#endif
        this->box_updator(update_interval * g, true, SCALE_COORDINATES_BY_ATOM,
                          SCALE_VELOCITIES_BY_ATOM);
    }
}

#include "Berendsen_thermostat.h"

void BERENDSEN_THERMOSTAT_INFORMATION::Initial(CONTROLLER* controller,
                                               float target_temperature,
                                               const char* module_name)
{
    controller->printf("START INITIALIZING BERENDSEN THERMOSTAT:\n");
    if (module_name == NULL)
    {
        strcpy(this->module_name, "berendsen_thermostat");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }
    controller->printf("    The target temperature is %.2f K\n",
                       target_temperature);

    this->target_temperature = target_temperature;

    dt = 1e-3f;
    if (controller[0].Command_Exist("dt"))
        dt = atof(controller[0].Command("dt"));
    controller->printf("    The dt is %f ps\n", dt);

    tauT = 1.0f;
    if (controller[0].Command_Exist("thermostat", "tau"))
    {
        controller->Check_Float("thermostat", "tau",
                                "BERENDSEN_THERMOSTAT_INFORMATION::Initial");
        tauT = atof(controller[0].Command("thermostat", "tau"));
    }

    controller->printf("    The time constant tau is %f ps\n", tauT);

    controller->Deprecated("berendsen_thermostat_tau",
                           "thermostat_tau = %VALUE%", "1.5",
                           "The thermostat parameters have been managed "
                           "uniformly since version 1.5");
    controller->Deprecated("berendsen_thermostat_seed",
                           "thermostat_seed = %VALUE%", "1.5",
                           "The thermostat parameters have been managed "
                           "uniformly since version 1.5");

    is_initialized = 1;
    if (is_initialized && !is_controller_printf_initialized)
    {
        is_controller_printf_initialized = 1;
        controller[0].printf("    structure last modify date is %d\n",
                             last_modify_date);
    }

    controller->printf("END INITIALIZING BERENDSEN THERMOSTAT\n\n");
}

void BERENDSEN_THERMOSTAT_INFORMATION::Record_Temperature(float temperature,
                                                          int freedom)
{
    if (!is_initialized) return;

    // Use conservative guards before sqrtf to avoid NaN propagation under
    // aggressive compiler math optimizations.
    lambda = 1.0f;
    if (!(temperature > 0.0f) || tauT <= 0.0f || freedom <= 0) return;

    float ratio = target_temperature / temperature;
    if (!(ratio > 0.0f) || ratio >= FLT_MAX) return;

    float lambda_square = 1.0f + dt / tauT * (ratio - 1.0f);

    if (lambda_square > 0.0f && lambda_square < FLT_MAX)
    {
        lambda = sqrtf(lambda_square);
    }
    if (!(lambda > 0.0f) || lambda >= FLT_MAX)
    {
        lambda = 1.0f;
    }
    if (lambda > 1.2f)
    {
        lambda = 1.2f;
    }
    else if (lambda < 0.8f)
    {
        lambda = 0.8f;
    }
}

void BERENDSEN_THERMOSTAT_INFORMATION::Scale_Velocity(int atom_numbers,
                                                      VECTOR* vel)
{
    if (is_initialized)
    {
        Scale_List((float*)vel, lambda, 3 * atom_numbers);
    }
}

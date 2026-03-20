#include "Bussi_thermostat.h"

void BUSSI_THERMOSTAT_INFORMATION::Initial(CONTROLLER* controller,
                                           float target_temperature,
                                           const char* module_name)
{
    controller->printf("START INITIALIZING BUSSI THERMOSTAT:\n");
    if (module_name == NULL)
    {
        strcpy(this->module_name, "bussi_thermostat");
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
    {
        dt = atof(controller[0].Command("dt"));
    }
    controller->printf("    The dt is %f ps\n", dt);

    tauT = 1.0f;
    if (controller[0].Command_Exist("thermostat", "tau"))
    {
        controller->Check_Float("thermostat", "tau",
                                "BUSSI_THERMOSTAT_INFORMATION::Initial");
        tauT = atof(controller[0].Command("thermostat", "tau"));
    }
    controller->printf("    The time constant tau is %f ps\n", tauT);

    int seed = time(NULL);
    if (controller[0].Command_Exist("thermostat", "seed"))
    {
        controller->Check_Int("thermostat", "seed",
                              "BUSSI_THERMOSTAT_INFORMATION::Initial");
        seed = atoi(controller[0].Command("thermostat", "seed"));
    }
    controller->printf("    The random seed is %d\n", seed);
    e.seed(seed);
    normal01 = std::normal_distribution<float>(0.0f, 1.0f);

    controller->Deprecated("berendsen_thermostat_tau",
                           "thermostat_tau = %VALUE%", "2.0",
                           "The thermostat parameters have been managed "
                           "uniformly since version 2.0");
    controller->Deprecated("berendsen_thermostat_seed",
                           "thermostat_seed = %VALUE%", "2.0",
                           "The thermostat parameters have been managed "
                           "uniformly since version 2.0");

    is_initialized = 1;
    if (is_initialized && !is_controller_printf_initialized)
    {
        is_controller_printf_initialized = 1;
        controller[0].printf("    structure last modify date is %d\n",
                             last_modify_date);
    }

    controller->printf("END INITIALIZING BUSSI THERMOSTAT\n\n");
}

void BUSSI_THERMOSTAT_INFORMATION::Record_Temperature(float temperature,
                                                      int freedom)
{
    if (!is_initialized) return;

    lambda = 1.0f;
    if (temperature <= 0.0f || tauT <= 0.0f || freedom <= 1) return;

    double ratio = static_cast<double>(target_temperature) /
                   static_cast<double>(temperature);
    if (!(ratio > 0.0) || ratio >= static_cast<double>(FLT_MAX)) return;
    double c = exp(-static_cast<double>(dt) / static_cast<double>(tauT));
    double one_minus_c = 1.0 - c;
    double dof = static_cast<double>(freedom);
    double R = static_cast<double>(normal01(e));
    std::gamma_distribution<double> chi_square_distribution(0.5 * (dof - 1.0),
                                                            2.0);
    double S = chi_square_distribution(e);
    double lambda_square = c + one_minus_c * ratio * (S + R * R) / dof +
                           2.0 * R * sqrt(c * one_minus_c * ratio / dof);

    if (lambda_square > 0.0 && lambda_square < static_cast<double>(FLT_MAX))
    {
        lambda = static_cast<float>(sqrt(lambda_square));
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

void BUSSI_THERMOSTAT_INFORMATION::Scale_Velocity(int atom_numbers, VECTOR* vel)
{
    if (is_initialized)
    {
        Scale_List((float*)vel, lambda, 3 * atom_numbers);
    }
}

void BUSSI_THERMOSTAT_INFORMATION::Set_Target_Temperature(
    float target_temperature_new)
{
    if (!is_initialized || !(target_temperature_new > 0.0f)) return;
    target_temperature = target_temperature_new;
}

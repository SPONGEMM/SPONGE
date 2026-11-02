#include "MC_barostat.h"

#include "../utils/float_classification.hpp"
#include "../utils/random/restart_rng_state.hpp"

std::uint64_t MC_BAROSTAT_INFORMATION::Next_Random_U64()
{
    random_state += 0x9e3779b97f4a7c15ULL;
    std::uint64_t value = random_state;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

float MC_BAROSTAT_INFORMATION::Next_Uniform()
{
    return static_cast<float>(Next_Random_U64() >> 40) * (1.0f / 16777216.0f);
}

int MC_BAROSTAT_INFORMATION::Next_Random_Index(int upper_bound)
{
    if (upper_bound <= 1) return 0;
    const std::uint64_t bound = static_cast<std::uint64_t>(upper_bound);
    const std::uint64_t limit = UINT64_MAX - (UINT64_MAX % bound);
    std::uint64_t value = 0;
    do
    {
        value = Next_Random_U64();
    } while (value >= limit);
    return static_cast<int>(value % bound);
}

void MC_BAROSTAT_INFORMATION::Volume_Change_Attempt(VECTOR boxlength, float dt)
{
    if (CONTROLLER::MPI_rank == 0)
    {
        double nrand = 2.0 * static_cast<double>(Next_Uniform()) - 1.0;

        Delta_Box_Length = {0.0f, 0.0f, 0.0f};
        switch (couple_dimension)
        {
            case NO:
                if (only_direction > 0)
                    xyz = only_direction - 1;
                else
                    xyz = Next_Random_Index(3);
                if (xyz == 0)
                {
                    Delta_Box_Length.x = nrand * Delta_Box_Length_Max[xyz];
                }
                else if (xyz == 1)
                {
                    Delta_Box_Length.y = nrand * Delta_Box_Length_Max[xyz];
                }
                else
                {
                    Delta_Box_Length.z = nrand * Delta_Box_Length_Max[xyz];
                }
                break;
            case XY:
                if (only_direction > 0)
                    xyz = only_direction - 1;
                else
                    xyz = Next_Random_Index(2);
                if (xyz == 0)
                {
                    Delta_Box_Length.z = nrand * Delta_Box_Length_Max[xyz];
                }
                else
                {
                    Delta_Box_Length.y = nrand * Delta_Box_Length_Max[xyz];
                    Delta_Box_Length.x = nrand * Delta_Box_Length_Max[xyz];
                }
                break;
            case XZ:
                if (only_direction > 0)
                    xyz = only_direction - 1;
                else
                    xyz = Next_Random_Index(2);
                if (xyz == 0)
                {
                    Delta_Box_Length.y = nrand * Delta_Box_Length_Max[xyz];
                }
                else
                {
                    Delta_Box_Length.z = nrand * Delta_Box_Length_Max[xyz];
                    Delta_Box_Length.x = nrand * Delta_Box_Length_Max[xyz];
                }
                break;
            case YZ:
                if (only_direction > 0)
                    xyz = only_direction - 1;
                else
                    xyz = Next_Random_Index(2);
                if (xyz == 0)
                {
                    Delta_Box_Length.x = nrand * Delta_Box_Length_Max[xyz];
                }
                else
                {
                    Delta_Box_Length.z = nrand * Delta_Box_Length_Max[xyz];
                    Delta_Box_Length.y = nrand * Delta_Box_Length_Max[xyz];
                }
                break;
            case XYZ:
                xyz = 0;
                Delta_Box_Length.x = nrand * Delta_Box_Length_Max[xyz];
                Delta_Box_Length.y = nrand * Delta_Box_Length_Max[xyz];
                Delta_Box_Length.z = nrand * Delta_Box_Length_Max[xyz];
                break;
        }

        New_Box_Length = boxlength + Delta_Box_Length;
        DeltaS = 0.0f;
        switch (couple_dimension)
        {
            case NO:
                break;
            case XY:
                if (xyz == 1)
                {
                    DeltaS = New_Box_Length.x * New_Box_Length.y -
                             boxlength.x * boxlength.y;
                }
                break;
            case XZ:
                if (xyz == 1)
                {
                    DeltaS = New_Box_Length.x * New_Box_Length.z -
                             boxlength.x * boxlength.z;
                }
                break;
            case YZ:
                if (xyz == 1)
                {
                    DeltaS = New_Box_Length.z * New_Box_Length.y -
                             boxlength.z * boxlength.y;
                }
                break;
            case XYZ:
                break;
        }
        double V = boxlength.x * boxlength.y * boxlength.z;
        newV = New_Box_Length.x * New_Box_Length.y * New_Box_Length.z;
        DeltaV = newV - V;
        VDevided = newV / V;
        VECTOR crd_scale_factor = New_Box_Length / boxlength;
        g = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        g.a11 = (crd_scale_factor.x - 1.0f) / dt;
        g.a22 = (crd_scale_factor.y - 1.0f) / dt;
        g.a33 = (crd_scale_factor.z - 1.0f) / dt;
    }
#ifdef USE_MPI
    MPI_Bcast(&g, 6, MPI_FLOAT, 0, MPI_COMM_WORLD);
#endif
}

int MC_BAROSTAT_INFORMATION::Check_MC_Barostat_Accept()
{
    total_count[xyz] += 1;
    float tmp_rand;
    if (CONTROLLER::MPI_rank == 0)
    {
        tmp_rand = Next_Uniform();
    }
#ifdef USE_MPI
    MPI_Bcast(&tmp_rand, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
#endif
    if (tmp_rand < accept_possibility)  // 接受
    {
        accept = 1;
        accep_count[xyz] += 1;
    }
    else
    {
        accept = 0;
    }
    return accept;
}

void MC_BAROSTAT_INFORMATION::Initial(CONTROLLER* controller, int atom_numbers,
                                      float target_pressure, VECTOR boxlength,
                                      LTMatrix3 cell, const char* module_name)
{
    controller->printf("START INITIALIZING MC BAROSTAT:\n");
    if (module_name == NULL)
    {
        strcpy(this->module_name, "monte_carlo_barostat");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }
    int random_seed = 0;
    if (controller[0].Command_Exist(this->module_name, "seed"))
    {
        controller->Check_Int(this->module_name, "seed",
                              "MC_BAROSTAT_INFORMATION::Initial");
        random_seed = atoi(controller[0].Command(this->module_name, "seed"));
    }
    random_state =
        static_cast<std::uint64_t>(static_cast<std::uint32_t>(random_seed));
    controller->printf("    random seed is %d\n", random_seed);
    for (int dimension = 0; dimension < 3; ++dimension)
    {
        total_count[dimension] = 0;
        accep_count[dimension] = 0;
        accept_rate[dimension] = 0.0f;
    }
    if (cell.a21 != 0.0f || cell.a31 != 0.0f || cell.a32 != 0.0f)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorValueErrorCommand, "MC_BAROSTAT_INFORMATION::Initial",
            "Reason:\n\t MC barostat only supports orthogonal box now.\n");
    }
    controller->printf("    The target pressure is %.2f bar\n",
                       target_pressure * CONSTANT_PRES_CONVERTION);
    float mc_baro_initial_ratio = 0.001;
    if (controller[0].Command_Exist(this->module_name, "initial_ratio"))
    {
        controller->Check_Float(this->module_name, "initial_ratio",
                                "MC_BAROSTAT_INFORMATION::Initial");
        mc_baro_initial_ratio =
            atof(controller[0].Command(this->module_name, "initial_ratio"));
    }
    Delta_Box_Length_Max[0] = mc_baro_initial_ratio * boxlength.x;
    Delta_Box_Length_Max[1] = mc_baro_initial_ratio * boxlength.y;
    Delta_Box_Length_Max[2] = mc_baro_initial_ratio * boxlength.z;
    controller->printf(
        "    The initial max box length to change is %f %f %f Angstrom for x y "
        "z\n",
        Delta_Box_Length_Max[0], Delta_Box_Length_Max[1],
        Delta_Box_Length_Max[2]);

    update_interval = 100;
    if (controller[0].Command_Exist(this->module_name, "update_interval"))
    {
        controller->Check_Int(this->module_name, "update_interval",
                              "MC_BAROSTAT_INFORMATION::Initial");
        update_interval =
            atoi(controller[0].Command(this->module_name, "update_interval"));
    }
    controller->printf("    The update_interval is %d\n", update_interval);

    check_interval = 10;
    if (controller[0].Command_Exist(this->module_name, "check_interval"))
    {
        controller->Check_Int(this->module_name, "check_interval",
                              "MC_BAROSTAT_INFORMATION::Initial");
        check_interval =
            atoi(controller[0].Command(this->module_name, "check_interval"));
    }
    controller->printf("    The check_interval is %d\n", check_interval);

    accept_rate_low = 30;
    if (controller[0].Command_Exist(this->module_name, "accept_rate_low"))
    {
        controller->Check_Float(this->module_name, "accept_rate_low",
                                "MC_BAROSTAT_INFORMATION::Initial");
        accept_rate_low =
            atof(controller[0].Command(this->module_name, "accept_rate_low"));
    }
    controller->printf("    The lowest accept rate is %.2f%%\n",
                       accept_rate_low);

    accept_rate_high = 40;
    if (controller[0].Command_Exist(this->module_name, "accept_rate_high"))
    {
        controller->Check_Float(this->module_name, "accept_rate_high",
                                "MC_BAROSTAT_INFORMATION::Initial");
        accept_rate_high =
            atof(controller[0].Command(this->module_name, "accept_rate_high"));
    }
    controller->printf("    The highest accept rate is %.2f%%\n",
                       accept_rate_high);

    if (!controller->Command_Exist(this->module_name, "couple_dimension") ||
        controller->Command_Choice(this->module_name, "couple_dimension",
                                   "XYZ"))
    {
        couple_dimension = XYZ;
    }
    else if (controller->Command_Choice(this->module_name, "couple_dimension",
                                        "NO"))
    {
        couple_dimension = NO;
    }
    else if (controller->Command_Choice(this->module_name, "couple_dimension",
                                        "XY"))
    {
        couple_dimension = XY;
    }
    else if (controller->Command_Choice(this->module_name, "couple_dimension",
                                        "XZ"))
    {
        couple_dimension = XZ;
    }
    else if (controller->Command_Choice(this->module_name, "couple_dimension",
                                        "YZ"))
    {
        couple_dimension = YZ;
    }
    if (!controller->Command_Exist(this->module_name, "couple_dimension"))
        controller->printf("    The couple dimension is %s (index %d)\n", "XYZ",
                           couple_dimension);
    else
        controller->printf(
            "    The couple dimension is %s (index %d)\n",
            controller->Command(this->module_name, "couple_dimension"),
            couple_dimension);
    if (controller->Command_Exist(this->module_name, "only_direction"))
    {
        if (couple_dimension == NO)
        {
            if (controller->Command_Choice(this->module_name, "only_direction",
                                           "x"))
            {
                only_direction = 1;
            }
            else if (controller->Command_Choice(this->module_name,
                                                "only_direction", "y"))
            {
                only_direction = 2;
            }
            else if (controller->Command_Choice(this->module_name,
                                                "only_direction", "z"))
            {
                only_direction = 3;
            }
        }
        else if (couple_dimension == XYZ)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand,
                "MC_BAROSTAT_INFORMATION::Initial",
                "Reason:\n\tonly_dimension is not valid for isotropic pressure "
                "regulation\n");
        }
        else if (couple_dimension == XY)
        {
            if (controller->Command_Choice(this->module_name, "only_direction",
                                           "z"))
            {
                only_direction = 1;
            }
            else if (controller->Command_Choice(this->module_name,
                                                "only_direction", "xy"))
            {
                only_direction = 2;
            }
        }
        else if (couple_dimension == XZ)
        {
            if (controller->Command_Choice(this->module_name, "only_direction",
                                           "y"))
            {
                only_direction = 1;
            }
            else if (controller->Command_Choice(this->module_name,
                                                "only_direction", "xz"))
            {
                only_direction = 2;
            }
        }
        else if (couple_dimension == YZ)
        {
            if (controller->Command_Choice(this->module_name, "only_direction",
                                           "x"))
            {
                only_direction = 1;
            }
            else if (controller->Command_Choice(this->module_name,
                                                "only_direction", "yz"))
            {
                only_direction = 2;
            }
        }
    }
    if (couple_dimension != NO && couple_dimension != XYZ)
    {
        surface_number = 0;
        if (controller->Command_Exist(this->module_name, "surface_number"))
        {
            controller->Check_Int(this->module_name, "surface_number",
                                  "MC_BAROSTAT_INFORMATION::Initial");
            surface_number =
                atoi(controller->Command(this->module_name, "surface_number"));
        }
        surface_tension = 0.0f;
        if (controller->Command_Exist(this->module_name, "surface_tension"))
        {
            controller->Check_Float(this->module_name, "surface_tension",
                                    "MC_BAROSTAT_INFORMATION::Initial");
            surface_tension =
                atof(controller->Command(this->module_name, "surface_tension"));
        }
        surface_tension *= TENSION_UNIT_FACTOR;
        controller->printf("        The surface number is %d\n",
                           surface_number);
        controller->printf("        The surface tension is %f\n",
                           surface_tension);
    }
    Device_Malloc_Safely((void**)&frc_backup, sizeof(VECTOR) * atom_numbers);
    Device_Malloc_Safely((void**)&crd_backup, sizeof(VECTOR) * atom_numbers);
    is_initialized = 1;
    if (is_initialized && !is_controller_printf_initialized)
    {
        controller->Step_Print_Initial("density", "%.4f");
        is_controller_printf_initialized = 1;
        controller[0].printf("    structure last modify date is %d\n",
                             last_modify_date);
    }
    controller[0].printf("END INITIALIZING MC BAROSTAT\n\n");
}

void MC_BAROSTAT_INFORMATION::Delta_Box_Length_Max_Update()
{
    if (total_count[xyz] % check_interval == 0)
    {
        accept_rate[xyz] = 100.0 * accep_count[xyz] / total_count[xyz];

        if (accept_rate[xyz] < accept_rate_low)
        {
            total_count[xyz] = 0;
            accep_count[xyz] = 0;
            Delta_Box_Length_Max[xyz] *= 0.9;
        }
        if (accept_rate[xyz] > accept_rate_high)
        {
            total_count[xyz] = 0;
            accep_count[xyz] = 0;
            Delta_Box_Length_Max[xyz] *= 1.1;
        }
    }
}

void MC_BAROSTAT_INFORMATION::Ask_For_Calculate_Potential(int steps,
                                                          int* need_potential)
{
    if (is_initialized && steps % update_interval == 0)
    {
        *need_potential = 1;
    }
}

bool MC_BAROSTAT_INFORMATION::Export_H5_Restart_State(
    SpongeH5MD::RestartDynamicState* state, std::string* error_message) const
{
    if (state == nullptr)
    {
        if (error_message != nullptr)
        {
            *error_message =
                "Monte Carlo barostat H5 restart state output pointer is null";
        }
        return false;
    }
    if (!is_initialized) return true;
    const std::string module = "monte_carlo_barostat";
    state->rng_states[module] =
        SpongeRestartRng::Splitmix64_State(random_state);
    state->barostat_float_states[module]["delta_box_length_max"] = {
        Delta_Box_Length_Max[0], Delta_Box_Length_Max[1],
        Delta_Box_Length_Max[2]};
    state->barostat_float_states[module]["accept_rate"] = {
        accept_rate[0], accept_rate[1], accept_rate[2]};
    state->barostat_integer_states[module]["total_count_int64"] = {
        total_count[0], total_count[1], total_count[2]};
    state->barostat_integer_states[module]["accept_count_int64"] = {
        accep_count[0], accep_count[1], accep_count[2]};
    return true;
}

bool MC_BAROSTAT_INFORMATION::Apply_H5_Restart_State(
    const SpongeH5MD::RestartDynamicState& state, std::string* error_message)
{
    auto fail = [error_message](const std::string& message)
    {
        if (error_message != nullptr) *error_message = message;
        return false;
    };
    if (!is_initialized)
    {
        return fail("Monte Carlo barostat module is not initialized");
    }
    const std::string module = "monte_carlo_barostat";
    const auto rng = state.rng_states.find(module);
    if (rng == state.rng_states.end() ||
        !SpongeRestartRng::Decode_Splitmix64_State(rng->second, &random_state,
                                                   error_message))
    {
        if (rng == state.rng_states.end())
        {
            return fail(
                "Monte Carlo barostat restart is missing typed RNG state");
        }
        return false;
    }
    const auto floats = state.barostat_float_states.find(module);
    const auto integers = state.barostat_integer_states.find(module);
    if (floats == state.barostat_float_states.end() ||
        integers == state.barostat_integer_states.end())
    {
        return fail("Monte Carlo barostat restart is missing adaptive state");
    }
    const auto delta = floats->second.find("delta_box_length_max");
    const auto rates = floats->second.find("accept_rate");
    const auto totals = integers->second.find("total_count_int64");
    const auto accepts = integers->second.find("accept_count_int64");
    if (delta == floats->second.end() || rates == floats->second.end() ||
        totals == integers->second.end() || accepts == integers->second.end() ||
        delta->second.size() != 3 || rates->second.size() != 3 ||
        totals->second.size() != 3 || accepts->second.size() != 3)
    {
        return fail("Monte Carlo barostat adaptive state has invalid shape");
    }
    for (int dimension = 0; dimension < 3; ++dimension)
    {
        if (!SpongeFloat::Is_Finite(delta->second[dimension]) ||
            delta->second[dimension] < 0.0f ||
            !SpongeFloat::Is_Finite(rates->second[dimension]) ||
            totals->second[dimension] < 0 || accepts->second[dimension] < 0 ||
            accepts->second[dimension] > totals->second[dimension] ||
            totals->second[dimension] > INT_MAX ||
            accepts->second[dimension] > INT_MAX)
        {
            return fail("Monte Carlo barostat adaptive state is invalid");
        }
        Delta_Box_Length_Max[dimension] = delta->second[dimension];
        accept_rate[dimension] = rates->second[dimension];
        total_count[dimension] = static_cast<int>(totals->second[dimension]);
        accep_count[dimension] = static_cast<int>(accepts->second[dimension]);
    }
    return true;
}

#ifndef BUSSI_THERMOSTAT_H
#define BUSSI_THERMOSTAT_H

#include <cstdint>
#include <random>
#include <sstream>

#include "../common.h"
#include "../control.h"
#include "../utils/h5md/h5_structural_state.hpp"
#include "../utils/random/restart_rng_state.hpp"

// 用于记录与计算Bussi CVR控温相关的信息
struct BUSSI_THERMOSTAT_INFORMATION
{
    char module_name[CHAR_LENGTH_MAX];
    int is_initialized = 0;
    int is_controller_printf_initialized = 0;
    int last_modify_date = 20260227;

    float tauT;                // 弛豫时间（ps）
    float dt;                  // 步长（ps）
    float target_temperature;  // 目标温度
    float lambda;              // 速度缩放系数

    std::uint64_t random_seed = 0;
    std::uint64_t random_invocation_count = 0;
    bool use_legacy_rng = false;
    std::default_random_engine legacy_engine;
    std::normal_distribution<float> legacy_normal01;

    // 初始化
    void Initial(CONTROLLER* controller, float target_temperature,
                 const char* module_name = NULL);

    // 根据当前温度计算Bussi精确CVR缩放系数
    void Record_Temperature(float temperature, int freedom);

    // 按lambda缩放速度
    void Scale_Velocity(int atom_numbers, VECTOR* vel);
    void Set_Target_Temperature(float target_temperature_new);
    bool Export_H5_Restart_State(SpongeH5MD::RestartDynamicState* state,
                                 std::string* error_message) const;
    bool Apply_H5_Restart_State(const SpongeH5MD::RestartDynamicState& state,
                                std::string* error_message);
};

inline bool BUSSI_THERMOSTAT_INFORMATION::Export_H5_Restart_State(
    SpongeH5MD::RestartDynamicState* state, std::string* error_message) const
{
    if (state == nullptr)
    {
        if (error_message != nullptr)
        {
            *error_message = "Bussi H5 restart state output pointer is null";
        }
        return false;
    }
    if (!is_initialized)
    {
        return true;
    }
    const std::string module = "bussi_thermostat";
    if (use_legacy_rng)
    {
        std::ostringstream rng;
        rng << legacy_engine;
        state->rng_state_text[module] = rng.str();
        state->thermostat_text_states[module]["rng_engine"] =
            "std::default_random_engine";
    }
    else
    {
        state->rng_states[module] = SpongeRestartRng::Counter_Philox_State(
            random_seed, random_invocation_count);
    }
    state->thermostat_float_states[module]["lambda"] = {lambda};
    return true;
}

inline bool BUSSI_THERMOSTAT_INFORMATION::Apply_H5_Restart_State(
    const SpongeH5MD::RestartDynamicState& state, std::string* error_message)
{
    const std::string module = "bussi_thermostat";
    const auto typed_rng = state.rng_states.find(module);
    const auto legacy_rng = state.rng_state_text.find(module);
    if (typed_rng != state.rng_states.end() &&
        legacy_rng != state.rng_state_text.end())
    {
        if (error_message != nullptr)
        {
            *error_message =
                "Bussi thermostat restart contains both typed and legacy RNG "
                "state";
        }
        return false;
    }
    if (typed_rng == state.rng_states.end() &&
        legacy_rng == state.rng_state_text.end())
    {
        if (error_message != nullptr)
        {
            *error_message =
                "Current run uses bussi_thermostat, but restart does not "
                "contain Bussi thermostat RNG state";
        }
        return false;
    }
    if (!is_initialized)
    {
        if (error_message != nullptr)
        {
            *error_message =
                "Restart contains Bussi thermostat RNG state, but the "
                "bussi_thermostat module is not initialized";
        }
        return false;
    }
    if (typed_rng != state.rng_states.end())
    {
        std::string decode_error;
        if (!SpongeRestartRng::Decode_Counter_Philox_State(
                typed_rng->second, &random_seed, &random_invocation_count,
                &decode_error))
        {
            if (error_message != nullptr)
            {
                *error_message =
                    "Failed to decode Bussi thermostat RNG state: " +
                    decode_error;
            }
            return false;
        }
        use_legacy_rng = false;
    }
    else
    {
        std::istringstream rng(legacy_rng->second);
        rng >> legacy_engine;
        if (rng.fail())
        {
            if (error_message != nullptr)
            {
                *error_message = "Failed to parse Bussi thermostat RNG state";
            }
            return false;
        }
        legacy_normal01.reset();
        use_legacy_rng = true;
    }
    const auto module_floats = state.thermostat_float_states.find(module);
    if (module_floats == state.thermostat_float_states.end())
    {
        if (error_message != nullptr)
        {
            *error_message = "Bussi thermostat restart state is missing lambda";
        }
        return false;
    }
    const auto lambda_state = module_floats->second.find("lambda");
    if (lambda_state == module_floats->second.end() ||
        lambda_state->second.size() != 1)
    {
        if (error_message != nullptr)
        {
            *error_message =
                "Bussi thermostat restart state is missing a scalar lambda";
        }
        return false;
    }
    lambda = lambda_state->second[0];
    return true;
}

#endif

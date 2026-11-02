#pragma once
#include <cstdint>
#include <random>
#include <sstream>

#include "../common.h"
#include "../control.h"
#include "../utils/h5md/h5_structural_state.hpp"
#include "../utils/random/restart_rng_state.hpp"

// 用于记录与计算控压相关的信息
struct PRESSURE_BASED_BAROSTAT_INFORMATION
{
    bool is_initialized = 0;
    float piston_mass_inverse;  // 拓展自由度的质量的倒数
    LTMatrix3 g;                // 拓展自由度的速度
    float V0;                   // 最初的体积，用于大体积变化时进行重新初始化
    int update_interval;        // 更新间隔
    std::default_random_engine generator;          // 随机数引擎
    std::normal_distribution<float> distribution;  // 随机数分布
    std::uint64_t random_seed = 0;
    std::uint64_t random_invocation_count = 0;
    bool use_legacy_rng = false;
    float target_surface_tension;  // 在xy界面的总表面张力（包含了表面数量）
    bool x_constant, y_constant, z_constant;

    float (*box_updator)(LTMatrix3 g, int scale_box, int scale_crd,
                         int scale_vel);
    void (*extreme_box_updator)();
    enum
    {
        Isotropic,
        Semiisotropic,
        Semianisotropic,
        Anisotropic
    } Isotropy;
    enum
    {
        Andersen,
        Berendsen,
        Bussi
    } Algorithm;

    void Initial(CONTROLLER* controller, float target_pressure, LTMatrix3 cell,
                 float (*box_updator)(LTMatrix3, int, int, int));

    void Control_Velocity_Of_Box(float dt, float target_temperature,
                                 LTMatrix3 dg);
    void Fill_Random_Normal(float* values, int count);

    void Ask_For_Calculate_Pressure(int steps, int* need_pressure);

    void Regulate_Pressure(int steps, LTMatrix3 h_stress, LTMatrix3 cell,
                           float dt, float target_pressure,
                           float target_temperature);
    bool Export_H5_Restart_State(SpongeH5MD::RestartDynamicState* state,
                                 std::string* error_message) const;
    bool Apply_H5_Restart_State(const SpongeH5MD::RestartDynamicState& state,
                                std::string* error_message);
};

inline std::vector<float> H5_Restart_LTMatrix3_To_Vector(
    const LTMatrix3& matrix)
{
    return {matrix.a11, matrix.a21, matrix.a22,
            matrix.a31, matrix.a32, matrix.a33};
}

inline bool H5_Restart_Vector_To_LTMatrix3(const std::vector<float>& values,
                                           LTMatrix3* matrix)
{
    if (matrix == nullptr || values.size() != 6)
    {
        return false;
    }
    matrix->a11 = values[0];
    matrix->a21 = values[1];
    matrix->a22 = values[2];
    matrix->a31 = values[3];
    matrix->a32 = values[4];
    matrix->a33 = values[5];
    return true;
}

inline bool PRESSURE_BASED_BAROSTAT_INFORMATION::Export_H5_Restart_State(
    SpongeH5MD::RestartDynamicState* state, std::string* error_message) const
{
    if (state == nullptr)
    {
        if (error_message != nullptr)
        {
            *error_message =
                "Pressure-based barostat H5 restart state output pointer is "
                "null";
        }
        return false;
    }
    if (!is_initialized)
    {
        return true;
    }
    const std::string module = "pressure_based_barostat";
    if (use_legacy_rng)
    {
        std::ostringstream rng;
        rng << generator;
        state->rng_state_text[module] = rng.str();
        state->barostat_text_states[module]["rng_engine"] =
            "std::default_random_engine";
    }
    else
    {
        state->rng_states[module] = SpongeRestartRng::Counter_Philox_State(
            random_seed, random_invocation_count);
    }
    state->barostat_float_states[module]["g"] =
        H5_Restart_LTMatrix3_To_Vector(g);
    return true;
}

inline bool PRESSURE_BASED_BAROSTAT_INFORMATION::Apply_H5_Restart_State(
    const SpongeH5MD::RestartDynamicState& state, std::string* error_message)
{
    const std::string module = "pressure_based_barostat";
    const auto module_floats = state.barostat_float_states.find(module);
    if (module_floats == state.barostat_float_states.end())
    {
        if (error_message != nullptr)
        {
            *error_message =
                "Current NPT run uses a pressure-based barostat, but restart "
                "does not contain pressure-based barostat dynamic state";
        }
        return false;
    }
    if (!is_initialized)
    {
        if (error_message != nullptr)
        {
            *error_message =
                "Restart contains pressure-based barostat state, but the "
                "current barostat is not pressure-based";
        }
        return false;
    }
    const auto g_state = module_floats->second.find("g");
    if (g_state == module_floats->second.end() ||
        !H5_Restart_Vector_To_LTMatrix3(g_state->second, &g))
    {
        if (error_message != nullptr)
        {
            *error_message =
                "Pressure-based barostat restart state is missing a valid g "
                "matrix";
        }
        return false;
    }
    const auto typed_rng = state.rng_states.find(module);
    const auto legacy_rng = state.rng_state_text.find(module);
    if (typed_rng != state.rng_states.end() &&
        legacy_rng != state.rng_state_text.end())
    {
        if (error_message != nullptr)
        {
            *error_message =
                "Pressure-based barostat restart contains both typed and "
                "legacy RNG state";
        }
        return false;
    }
    if (typed_rng == state.rng_states.end() &&
        legacy_rng == state.rng_state_text.end())
    {
        if (error_message != nullptr)
        {
            *error_message =
                "Pressure-based barostat restart state is missing RNG state";
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
                    "Failed to decode pressure-based barostat RNG state: " +
                    decode_error;
            }
            return false;
        }
        use_legacy_rng = false;
    }
    else
    {
        std::istringstream rng(legacy_rng->second);
        rng >> generator;
        if (rng.fail())
        {
            if (error_message != nullptr)
            {
                *error_message =
                    "Failed to parse pressure-based barostat RNG state";
            }
            return false;
        }
        distribution.reset();
        use_legacy_rng = true;
    }
    return true;
}

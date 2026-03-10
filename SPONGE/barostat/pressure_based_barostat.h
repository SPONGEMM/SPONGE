#pragma once
#include "../common.h"
#include "../control.h"

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

    void Ask_For_Calculate_Pressure(int steps, int* need_pressure);

    void Regulate_Pressure(int steps, LTMatrix3 h_stress, LTMatrix3 cell,
                           float dt, float target_pressure,
                           float target_temperature);
};

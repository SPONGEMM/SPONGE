#ifndef BUSSI_THERMOSTAT_H
#define BUSSI_THERMOSTAT_H

#include "../common.h"
#include "../control.h"

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

    std::default_random_engine e;
    std::normal_distribution<float> normal01;

    // 初始化
    void Initial(CONTROLLER* controller, float target_temperature,
                 const char* module_name = NULL);

    // 根据当前温度计算Bussi精确CVR缩放系数
    void Record_Temperature(float temperature, int freedom);

    // 按lambda缩放速度
    void Scale_Velocity(int atom_numbers, VECTOR* vel);
    void Set_Target_Temperature(float target_temperature_new);
};

#endif

#pragma once

#include "../common.h"
#include "../control.h"

// 用于记录与计算Berendsen控温相关的信息
struct BERENDSEN_THERMOSTAT_INFORMATION
{
    char module_name[CHAR_LENGTH_MAX];
    int is_initialized = 0;
    int is_controller_printf_initialized = 0;
    int last_modify_date = 20260216;

    // 原始版本berendsen控温
    float tauT;                // 弛豫时间（ps）
    float dt;                  // 步长（ps）
    float target_temperature;  // 目标温度
    float lambda;              // 规度系数

    // 初始化
    void Initial(CONTROLLER* controller, float target_temperature,
                 const char* module_name = NULL);

    void Scale_Velocity(int atom_numbers, VECTOR* vel);

    void Record_Temperature(float temperature, int freedom);
    void Set_Target_Temperature(float target_temperature_new);
};

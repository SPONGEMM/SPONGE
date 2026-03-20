#pragma once

#include "../common.h"
#include "../control.h"

// 用于记录与计算Andersen控温相关的信息
struct ANDERSEN_THERMOSTAT_INFORMATION
{
    char module_name[CHAR_LENGTH_MAX];
    int is_initialized = 0;
    int is_controller_printf_initialized = 0;
    int last_modify_date = 20260216;

    int atom_numbers;
    float target_temperature;  // 热浴温度

    int update_interval = 0;  // 更新间隔

    int float4_numbers;                  // 存储随机数的长度
    Philox4_32_10_t* rand_state = NULL;  // 用于记录随机数发生器状态
    VECTOR* random_vel =
        NULL;  // 存储随机速度矢量，要求该数组的长度要能整除4且大于等于atom_numbers
    float* h_factor = NULL;        // 用于计算随机速度的系数
    float* d_factor = NULL;        // 用于计算随机速度的系数
    float* d_mass_inverse = NULL;  // 质量的倒数

    // 使用速度上限的迭代方法而非盲目加大摩擦、降低温度、减少步长
    float max_velocity = 0;

    // 初始化
    void Initial(CONTROLLER* controller, float target_pressure,
                 int atom_numbers, float dt_in_ps, float* h_mass,
                 const char* module_name = NULL);

    /*
        以下用于区域分解
    */

    int local_atom_numbers = 0;    // 局域的原子数
    int local_float4_numbers = 0;  // 存储局域随机数的长度
    float* d_factor_local = NULL;
    float* d_mass_inverse_local = NULL;
    void Get_Local(int* atom_local,
                   int local_atom_numbers);  // 获取局域粒子信息

    void MD_Iteration_Leap_Frog(VECTOR* vel, VECTOR* crd, VECTOR* frc,
                                VECTOR* acc, float dt);
    void Set_Target_Temperature(float target_temperature_new);
};

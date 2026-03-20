#pragma once

#include "../common.h"
#include "../control.h"

// 该方法的主要实现的参考文献
// A unified thermostat scheme for efficient configurational sampling for
// classical/quantum canonical ensembles via molecular dynamics
struct MIDDLE_Langevin_INFORMATION
{
    char module_name[CHAR_LENGTH_MAX];
    int is_initialized = 0;
    int is_controller_printf_initialized = 0;
    int last_modify_date = 20260216;

    int atom_numbers;
    float dt;
    float half_dt;
    float gamma_ln;            // 碰撞频率
    float target_temperature;  // 热浴温度

    float exp_gamma;     // 刘剑动力学中参数( = expf(-gamma_ln*dt));
    int float4_numbers;  // 存储随机数的长度
    Philox4_32_10_t* rand_state = NULL;  // 用于记录随机数发生器状态
    VECTOR* random_force =
        NULL;  // 存储随机力矢量，要求该数组的长度要能整除4且大于等于atom_numbers
    float* d_sqrt_mass = NULL;     // 用于刘剑热浴过程中随机力的原子等效质量
    float* h_sqrt_mass = NULL;     // 用于刘剑热浴过程中随机力的原子等效质量
    float* d_mass_inverse = NULL;  // 质量的倒数

    // 使用速度上限的迭代方法而非盲目加大摩擦、降低温度、减少步长
    float max_velocity;

    // 初始化（质量信息从某个MD_CORE的已初始化质量数组中获得）
    void Initial(CONTROLLER* controller, const int atom_numbers,
                 const float target_temperature, const float* h_mass,
                 const char* module_name = NULL);

    /*
        以下用于区域分解
    */
    int local_atom_numbers = 0;       // 局域的原子数
    int local_float4_numbers = 0;     // 存储局域随机数的长度
    float* d_sqrt_mass_local = NULL;  // 用于刘剑热浴过程中随机力的原子等效质量
    float* d_mass_inverse_local =
        NULL;  // 用于刘剑热浴过程中随机力的原子等效质量
    void Get_Local(int* atom_local,
                   int local_atom_numbers);  // 获取局域粒子信息

    // 迭代算法
    void MD_Iteration_Leap_Frog(VECTOR* frc, VECTOR* vel, VECTOR* acc,
                                VECTOR* crd);
    void Set_Target_Temperature(float target_temperature_new);
};

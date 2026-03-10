#pragma once
#include "../common.h"
#include "../control.h"

// 用于记录与计算Nose-Hoover链控温相关的信息
struct NOSE_HOOVER_CHAIN_INFORMATION
{
    char module_name[CHAR_LENGTH_MAX];
    int is_initialized = 0;
    int is_controller_printf_initialized = 0;
    int last_modify_date = 20260216;

    int chain_length = 0;        // NH链长度
    float* h_coordinate = NULL;  // 拓展自由度的坐标 host
    float* h_velocity = NULL;    // 拓展自由度的速度 host
    float* coordinate = NULL;    // 拓展自由度的坐标 device
    float* velocity = NULL;      // 拓展自由度的速度 device
    float h_mass;                // 拓展自由度的质量
    float kB_T = 0;

    float max_velocity = 0;                       // 最大速度
    char restart_file_name[CHAR_LENGTH_MAX];      // 重启文件名字
    FILE *f_crd_traj = NULL, *f_vel_traj = NULL;  // 坐标和速度轨迹文件

    float* d_mass_inverse = NULL;  // 质量的倒数

    // 初始化
    void Initial(CONTROLLER* controller, int atom_numbers,
                 float target_pressure, const float* atom_mass,
                 const char* module_name = NULL);

    /*
        以下用于区域分解
    */
    int local_atom_numbers = 0;  // 局域的原子数
    float* d_mass_inverse_local = NULL;
    void Get_Local(int* atom_local,
                   int local_atom_numbers);  // 获取局域粒子信息

    void MD_Iteration_Leap_Frog(VECTOR* vel, VECTOR* crd, VECTOR* frc,
                                VECTOR* acc, float dt, float Ek, int freedom);

    void Save_Restart_File();

    void Save_Trajectory_File();
};

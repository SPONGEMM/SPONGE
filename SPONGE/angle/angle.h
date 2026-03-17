#pragma once
#include "../common.h"
#include "../control.h"

struct ANGLE
{
    char module_name[CHAR_LENGTH_MAX];
    int is_initialized = 0;
    int is_controller_printf_initialized = 0;
    int last_modify_date = 20260216;

    // E_angle = k (∠abc - theta0) ^ 2
    int angle_numbers = 0;  // 总键角数
    int* h_atom_a = NULL;   // host键角的原子a
    int* h_atom_b = NULL;   // host键角的原子b
    int* h_atom_c = NULL;   // host键角的原子c
    int* d_atom_a =
        NULL;  // device端的键角的原子a， 区域分解中对应存储区域内的原子a序号
    int* d_atom_b = NULL;
    int* d_atom_c = NULL;
    float* h_angle_k = NULL;  // 键参数
    float* d_angle_k =
        NULL;  // device端的键参数, 区域分解中对应存储区域内的键参数
    float* h_angle_theta0 = NULL;
    float* d_angle_theta0 = NULL;

    float* h_angle_ene = NULL;           // 每个angle能量
    float* d_angle_ene = NULL;           // domain的angle_local能量
    float* d_sigma_of_angle_ene = NULL;  // 总angle能量
    float* h_sigma_of_angle_ene = NULL;  // domain中的总angle能量

    // 初始化模块
    void Initial(CONTROLLER* controller, const char* module_name = NULL);
    // 内存分配
    void Memory_Allocate();
    // 拷贝cpu中的数据到gpu中
    void Parameter_Host_To_Device();

    // 计算angle force并同时计算能量并将其加到原子能量列表上
    void Angle_Force_With_Atom_Energy_And_Virial(
        const VECTOR* crd, const LTMatrix3 cell, const LTMatrix3 rcell,
        VECTOR* frc, int need_atom_energy, float* atom_energy, int need_virial,
        LTMatrix3* atom_virial_tensor);

    /*
        以下用于区域分解
    */
    int* d_atom_a_local = NULL;
    int* d_atom_b_local = NULL;
    int* d_atom_c_local = NULL;
    float* d_angle_k_local = NULL;
    float* d_angle_theta0_local = NULL;

    // 局部信息
    int num_angle_local = 0;  // 进程内angle数
    int local_atom_numbers = 0;
    int* d_num_angle_local = NULL;
    // 局部函数：allocated模块，查询当前进程domain内需要计算的angle序号
    void Get_Local(int* atom_local, int local_atom_numbers, int ghost_numbers,
                   char* atom_local_label,
                   int* atom_local_id);  // 为domain分配angle信息

    // 打印能量
    void Step_Print(CONTROLLER* controller);
};

#pragma once
#include "../common.h"
#include "../control.h"

struct BOND
{
    char module_name[CHAR_LENGTH_MAX];
    int is_initialized = 0;
    int is_controller_printf_initialized = 0;
    int last_modify_date = 20260216;

    // E_bond=k*(|r_a-r_b|-r0)^2
    // 计算bond模块的最基本信息
    // h为host上内存，在initial时被分配空间与赋值
    // d为device上内存，在initial的h赋值结束后，被分配以及赋值
    // 实际计算过程的信息由d上数值为准
    int bond_numbers = 0;
    int* h_atom_a = NULL;
    int* d_atom_a = NULL;
    int* h_atom_b = NULL;
    int* d_atom_b = NULL;
    float* d_k = NULL;
    float* h_k = NULL;
    float* d_r0 = NULL;
    float* h_r0 = NULL;

    // 对每根bond的能量进行存储
    float* h_bond_ene = NULL;
    float* d_bond_ene = NULL;
    // bond的总能量
    float* d_sigma_of_bond_ene = NULL;
    float* h_sigma_of_bond_ene = NULL;

    // 初始化模块
    void Initial(CONTROLLER* controller, CONECT* connectivity,
                 PAIR_DISTANCE* con_dis, const char* module_name = NULL);

    // 内存分配
    void Memory_Allocate();
    // 拷贝cpu中的参数到gpu
    void Parameter_Host_To_Device();

    // 同时计算力，并将能量和维里加到每个原子头上
    void Bond_Force_With_Atom_Energy_And_Virial(
        const VECTOR* crd, const LTMatrix3 cell, const LTMatrix3 rcell,
        VECTOR* frc, int need_atom_energy, float* atom_energy, int need_virial,
        LTMatrix3* atom_virial);

    /*
        以下用于区域分解
    */
    int* d_atom_a_local = NULL;
    int* d_atom_b_local = NULL;
    float* d_k_local = NULL;
    float* d_r0_local = NULL;

    // 局部信息
    int local_atom_numbers = 0;
    int num_bond_local = 0;  // 进程内bond数
    int* d_num_bond_local = NULL;
    // 局部函数：allocated模块，查询当前进程domain内需要计算的bond序号
    void Get_Local(int* atom_local, int local_atom_numbers, int ghost_numbers,
                   char* atom_local_label,
                   int* atom_local_id);  // 为domain分配bond信息
    // 判断atom_i在local中还是在ghost中，还是均不在
    // 打印能量
    void Step_Print(CONTROLLER* controller);
};

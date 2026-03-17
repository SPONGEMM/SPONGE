#pragma once
#include "../common.h"
#include "../control.h"

struct IMPROPER_DIHEDRAL
{
    char module_name[CHAR_LENGTH_MAX];
    int is_initialized = 0;
    int is_controller_printf_initialized = 0;
    int last_modify_date = 20260216;

    // phi = 面abc和面bcd的二面角
    // E_improper_dihedral = pk * (phi - phi0 ) * (phi - phi0 )
    int dihedral_numbers = 0;

    int* h_atom_a = NULL;
    int* d_atom_a = NULL;
    int* h_atom_b = NULL;
    int* d_atom_b = NULL;
    int* h_atom_c = NULL;
    int* d_atom_c = NULL;
    int* h_atom_d = NULL;
    int* d_atom_d = NULL;

    float* h_pk = NULL;
    float* d_pk = NULL;
    float* h_phi0 = NULL;
    float* d_phi0 = NULL;

    float* h_dihedral_ene = NULL;
    float* d_dihedral_ene = NULL;
    float* d_sigma_of_dihedral_ene = NULL;
    float* h_sigma_of_dihedral_ene = NULL;

    // cuda计算分配相关参数
    int threads_per_block = 128;

    // 初始化模块
    void Initial(CONTROLLER* controller, const char* module_name = NULL);

    // 为dihedral中的变量分配空间
    void Memory_Allocate();
    // 拷贝cpu中的数据到gpu
    void Parameter_Host_To_Device();

    // 计算dihedral force并同时计算能量并加到原子能量列表上
    void Dihedral_Force_With_Atom_Energy_And_Virial(
        const VECTOR* crd, const LTMatrix3 cell, const LTMatrix3 rcell,
        VECTOR* frc, int need_atom_energy, float* atom_energy, int need_virial,
        LTMatrix3* atom_virial);

    /*
        以下用于区域分解
    */
    int* d_atom_a_local = NULL;
    int* d_atom_b_local = NULL;
    int* d_atom_c_local = NULL;
    int* d_atom_d_local = NULL;
    float* d_pk_local = NULL;
    float* d_phi0_local = NULL;

    // 局部信息
    int num_dihe_local = 0;  // 进程内dihedral数
    int local_atom_numbers = 0;
    int* d_num_dihe_local = NULL;
    // 局部函数：allocated模块，查询当前进程domain内需要计算的dihedral序号
    void Get_Local(int* atom_local, int local_atom_numbers, int ghost_numbers,
                   char* atom_local_label,
                   int* atom_local_id);  // 为domain分配angle信息
    // 获得能量
    void Step_Print(CONTROLLER* controller, bool print_sum = true);
};

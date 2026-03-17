#pragma once
#include "../common.h"
#include "../control.h"

struct NON_BOND_14
{
    char module_name[CHAR_LENGTH_MAX];
    int is_initialized = 0;
    int is_controller_printf_initialized = 0;
    int last_modify_date = 20260216;
    std::string print_lj_name;
    std::string print_ee_name;

    // r = ab原子的距离
    // E_lj_energy = (A/12 * r^-12 - B/6 * r^-6)
    // E_cf_energy = cf_scale_factor * charge_a * charge_b / r
    // lj_A、lj_B、charge从外部传入，lj_A、lj_B参考LJ，charge参考md_core
    int nb14_numbers = 0;
    int* h_atom_a = NULL;
    int* h_atom_b = NULL;
    int* d_atom_a = NULL;
    int* d_atom_b = NULL;
    float* h_A = NULL;
    float* d_A = NULL;
    float* h_B = NULL;
    float* d_B = NULL;
    float* h_cf_scale_factor = NULL;
    float* d_cf_scale_factor = NULL;

    float* d_nb14_cf_energy = NULL;
    float* d_nb14_lj_energy = NULL;
    float* d_nb14_cf_energy_sum = NULL;
    float* d_nb14_lj_energy_sum = NULL;
    float* h_nb14_cf_energy_sum = NULL;
    float* h_nb14_lj_energy_sum = NULL;

    void Initial(CONTROLLER* controller, const float* LJ_type_A,
                 const float* LJ_type_B, const int* lj_atom_type,
                 const char* module_name = NULL);
    void Memory_Allocate();
    void Parameter_Host_To_Device();

    // 同时计算原子的力、能量和维里
    void Non_Bond_14_LJ_CF_Force_With_Atom_Energy_And_Virial(
        const VECTOR* crd, const float* charge, const LTMatrix3 cell,
        const LTMatrix3 rcell, VECTOR* frc, int need_atom_energy,
        float* atom_energy, int need_virial, LTMatrix3* atom_virial);

    /*
        以下用于区域分解
    */
    int* d_atom_a_local = NULL;
    int* d_atom_b_local = NULL;
    float* d_A_local = NULL;
    float* d_B_local = NULL;
    float* d_cf_scale_factor_local = NULL;

    // 局部信息
    int num_nb14_local = 0;  // 进程内nb14数
    int local_atom_numbers = 0;
    int* d_num_nb14_local = NULL;

    // 局部函数：allocated模块，查询当前进程domain内需要计算的nb14序号
    void Get_Local(int* atom_local, int local_atom_numbers, int ghost_numbers,
                   char* atom_local_label,
                   int* atom_local_id);  // 为domain分配nb14信息
    // 获得能量
    void Step_Print(CONTROLLER* controller, bool print_sum = true);
};

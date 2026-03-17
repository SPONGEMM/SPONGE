#pragma once
#include "../common.h"
#include "../control.h"

// 用于计算LJ_Force时使用的坐标和记录的原子LJ种类序号与原子电荷
#ifndef VECTOR_LJ_DEFINE
#define VECTOR_LJ_DEFINE
#define TWO_DIVIDED_BY_SQRT_PI 1.1283791670218446f
__host__ __device__ __forceinline__ int Get_LJ_Type(int a, int b);
__host__ __device__ __forceinline__ int Get_LJ_Type(unsigned int a,
                                                    unsigned int b);

struct VECTOR_LJ
{
    VECTOR crd;
    int LJ_type;
    float charge;
    friend __host__ __device__ __forceinline__ int Get_LJ_Type(int a, int b)
    {
        int y = (b - a);
        int x = y >> 31;
        y = (y ^ x) - x;
        x = b + a;
        int z = (x + y) >> 1;
        x = (x - y) >> 1;
        return (z * (z + 1) >> 1) + x;
    }
    friend __host__ __device__ __forceinline__ int Get_LJ_Type(unsigned int a,
                                                               unsigned int b)
    {
        int y = (b - a);
        int x = y >> 31;
        y = (y ^ x) - x;
        x = b + a;
        int z = (x + y) >> 1;
        x = (x - y) >> 1;
        return (z * (z + 1) >> 1) + x;
    }
    friend __device__ __host__ __forceinline__ VECTOR Get_Periodic_Displacement(
        VECTOR_LJ uvec_a, VECTOR_LJ uvec_b, LTMatrix3 cell, LTMatrix3 rcell)
    {
        return Get_Periodic_Displacement(uvec_a.crd, uvec_b.crd, cell, rcell);
    }
    friend __device__ __host__ __forceinline__ float Get_LJ_Energy(
        VECTOR_LJ r1, VECTOR_LJ r2, float dr_abs, const float A, const float B)
    {
        float dr_6 = powf(dr_abs, -6.0f);
        return (0.083333333f * A * dr_6 - 0.166666667f * B) * dr_6;
    }
    friend __device__ __host__ __forceinline__ float Get_LJ_Force(
        VECTOR_LJ r1, VECTOR_LJ r2, float dr_abs, const float A, const float B)
    {
        return (B - A * powf(dr_abs, -6.0f)) * powf(dr_abs, -8.0f);
    }
    friend __device__ __host__ __forceinline__ float Get_LJ_Virial(
        VECTOR_LJ r1, VECTOR_LJ r2, float dr_abs, const float A, const float B)
    {
        float dr_6 = powf(dr_abs, -6.0f);
        return -(B - A * dr_6) * dr_6;
    }
    friend __device__ __host__ __forceinline__ float Get_Direct_Coulomb_Energy(
        VECTOR_LJ r1, VECTOR_LJ r2, float dr_abs, const float pme_beta)
    {
        return r1.charge * r2.charge * erfcf(pme_beta * dr_abs) / dr_abs;
    }
    friend __device__ __host__ __forceinline__ float Get_Direct_Coulomb_Force(
        VECTOR_LJ r1, VECTOR_LJ r2, float dr_abs, const float pme_beta)
    {
        float beta_dr = pme_beta * dr_abs;
        return r1.charge * r2.charge * powf(dr_abs, -3.0f) *
               (beta_dr * TWO_DIVIDED_BY_SQRT_PI * expf(-beta_dr * beta_dr) +
                erfcf(beta_dr));
    }
};
__global__ void Copy_LJ_Type_To_New_Crd(const int atom_numbers,
                                        VECTOR_LJ* new_crd, const int* LJ_type);
__global__ void Copy_Crd_And_Charge_To_New_Crd(const int atom_numbers,
                                               const VECTOR* crd,
                                               VECTOR_LJ* new_crd,
                                               const float* charge);
__global__ void Copy_Crd_To_New_Crd(const int atom_numbers, const VECTOR* crd,
                                    VECTOR_LJ* new_crd);
#endif

// 用于记录与计算LJ相关的信息
struct LENNARD_JONES_INFORMATION
{
    char module_name[CHAR_LENGTH_MAX];
    int is_initialized = 0;
    int is_controller_printf_initialized = 0;
    int last_modify_date = 20260216;

    // a = LJ_A between atom[i] and atom[j]
    // b = LJ_B between atom[i] and atom[j]
    // E_lj = a/12 * r^-12 - b/6 * r^-6;
    // F_lj = (a * r^-14 - b * r ^ -6) * dr
    int atom_numbers = 0;       // 原子数
    int atom_type_numbers = 0;  // 原子种类数
    int pair_type_numbers = 0;  // 原子对种类数

    int* h_atom_LJ_type = NULL;  // 原子对应的LJ种类
    int* d_atom_LJ_type = NULL;  // 原子对应的LJ种类

    float* h_LJ_A = NULL;  // LJ的A系数
    float* h_LJ_B = NULL;  // LJ的B系数
    float* d_LJ_A = NULL;  // LJ的A系数
    float* d_LJ_B = NULL;  // LJ的B系数

    float* h_LJ_energy_atom = NULL;  // 每个原子的LJ的能量
    float h_LJ_energy_sum = 0;       // 所有原子的LJ能量和
    float* d_LJ_energy_atom = NULL;  // 每个原子的LJ的能量
    float* d_LJ_energy_sum = NULL;
    ;                            // 所有原子的LJ能量和
    float h_LJ_long_energy = 0;  // 长程修正能量

    // 初始化
    void Initial(CONTROLLER* controller, float cutoff,
                 const char* module_name = NULL);
    // 分配内存
    void LJ_Malloc();
    // 参数传到GPU上
    void Parameter_Host_To_Device();

    float cutoff = 10.0;
    VECTOR_LJ* crd_with_LJ_parameters = NULL;

    /*
        以下用于区域分解
    */
    int local_atom_numbers = 0;
    int ghost_numbers = 0;
    VECTOR_LJ* crd_with_LJ_parameters_local =
        NULL;  // 局域原子的坐标，电荷LJ_type打包
    void Get_Local(int* atom_local, int local_atom_numbers,
                   int ghost_numbers);  // 获取局域粒子信息

    // 可以根据外界传入的need_atom_energy和need_virial，选择性计算能量和维里。其中的维里对PME直接部分计算的原子能量，在和PME其他部分加和后即维里。
    void LJ_PME_Direct_Force_With_Atom_Energy_And_Virial(
        const int atom_numbers, const int local_atom_numbers,
        const int solvent_numbers, const int ghost_numbers, const VECTOR* crd,
        const float* charge, VECTOR* frc, const LTMatrix3 cell,
        const LTMatrix3 rcell, const ATOM_GROUP* nl, const float pme_beta,
        const int need_atom_energy, float* atom_energy, const int need_virial,
        LTMatrix3* atom_virial, float* atom_direct_pme_energy);

#ifdef KPCCL_TASKLOOP
    void LJ_PME_Direct_Force_With_Atom_Energy_And_Virial_Init(
        const int atom_numbers, const int local_atom_numbers,
        const int ghost_numbers, const VECTOR* crd, const float* charge,
        VECTOR* frc, const LTMatrix3 cell, const LTMatrix3 rcell,
        const ATOM_GROUP* nl, const float pme_beta, const int need_atom_energy,
        float* atom_energy, const int need_virial, LTMatrix3* atom_virial,
        float* atom_direct_pme_energy);

    void LJ_PME_Direct_Force_With_Atom_Energy_And_Virial_KPCCL(
        const int atom_numbers, const int local_atom_numbers,
        const int ghost_numbers, const VECTOR* crd, const float* charge,
        VECTOR* frc, const LTMatrix3 cell, const LTMatrix3 rcell,
        const ATOM_GROUP* nl, const float pme_beta, const int need_atom_energy,
        float* atom_energy, const int need_virial, LTMatrix3* atom_virial,
        float* atom_direct_pme_energy);
#endif

    // 长程能量和维里修正
    float long_range_factor = 0;
    float* d_long_range_factor = NULL;
    // 求力的时候对能量和维里的长程修正
    void Long_Range_Correction(int need_pressure, LTMatrix3* d_virial,
                               int need_potential, float* d_potential,
                               const float volume);

    void Step_Print(CONTROLLER* controller);
};

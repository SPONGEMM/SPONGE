#pragma once

#include "../common.h"
#include "../control.h"
// FFT Backend
#include "../utils/fft.hpp"

// 为以后MPI-FFT预留接口，默认使用fft3d
typedef void* MPI_FFT_PLAN;
// #include "fft3d_wrap.h"

#define MAX_PME_MPI_SIZE 100
#define MAX_PP_MPI_SIZE 100

struct Particle_Mesh
{
    char module_name[CHAR_LENGTH_MAX];
    int is_initialized = 0;
    int is_controller_printf_initialized = 0;
    int last_modify_date = 20260216;
    bool print_detail = 0;

    float sum_of_charge_square = 0;
    float square_of_charge_sum = 0;

    // fft维度参数
    int fftx = -1;
    int ffty = -1;
    int fftz = -1;
    int PME_Nall = 0;  // fftx*ffty*fftz 倒空间三维体积
    int PME_Nin = 0;   // ffty*fftz 二维截面面积
    int PME_Nfft = 0;  // fftx*ffty*(fftz/2+1) FFT 数组长度
    char FFT_MPI_TYPE[CHAR_LENGTH_MAX];

    // MPI参数,区域分解参数
    int PM_MPI_size;
    int pm_rank;

    // 主进程初始化与通信使用
    int pm_pp_corres[MAX_PME_MPI_SIZE]
                    [MAX_PP_MPI_SIZE];  // 每个PM进程对应的PP进程号
    int pm_pp_num[MAX_PME_MPI_SIZE];    // 每个PM进程对应的PP进程数

    // 每个进程私有
    int pp_corres_pm_rank;                       // 当前PP进程对应的PM进程号
    int pm_corres_pp_num;                        // 当前PM进程对应的PP进程数
    int pm_corres_pp_rank_set[MAX_PP_MPI_SIZE];  // 当前PM进程对应的PP进程号集合
    int pm_corres_pp_atom_number[MAX_PP_MPI_SIZE];
    int pm_corres_pp_atom_number_prefix[MAX_PP_MPI_SIZE];
    INT_VECTOR pm_dom_dec_split_num = {0, 0, 0};

    int neighbor_num[6];
    int neighbor_dir[6][MAX_PME_MPI_SIZE];
    VECTOR min_corner;
    VECTOR max_corner;
    VECTOR min_corner_set[MAX_PME_MPI_SIZE];
    VECTOR max_corner_set[MAX_PME_MPI_SIZE];

    MPI_FFT_PLAN mpi_fft_plan;
    MPI_FFT_PLAN mpi_fft_plan_forward;
    MPI_FFT_PLAN mpi_fft_plan_backward;

    INT_VECTOR local_lows;
    INT_VECTOR local_highs;
    INT_VECTOR local_length;
    int MPI_PME_Nfft;
    float* MPI_PME_Q = nullptr;
    float* MPI_PME_FQ = nullptr;
    float* MPI_PME_FBCFQ = nullptr;

    // cuda参数
    FFT_HANDLE PME_plan_r2c;
    FFT_HANDLE PME_plan_c2r;

    // 初始化参数
    int max_atom_numbers = 0;
    int atom_numbers = 0;
    int ghost_numbers = 0;
    int num_ghost_dir[6];
    int num_ghost_dir_re[6];
    int* num_ghost_dir_id;

    // 体积相关的物理参数
    float* PME_BC = NULL;  // GPU上的BC数组
    float* PME_BC0 =
        NULL;  // GPU上的BC0数组，也即BC数组在乘上盒子相关信息之前的数组，更新体积的时候用
    LTMatrix3* PME_Virial_BC = NULL;  // GPU上的维里对应的BC0数组

    // 体积无关的物理参数
    UNSIGNED_INT_VECTOR* PME_uxyz = NULL;  // 原子对应的网格点
    VECTOR* PME_frxyz = NULL;    // 原子对应的网格点的相对位置, 已归一化
    float* PME_Q = NULL;         // 网格上的电荷密度
    float* PME_FBCFQ = NULL;     // 网格上的电荷密度乘以BC
    FFT_COMPLEX* PME_FQ = NULL;  // 网格上的电荷密度的傅里叶变换
    int* PME_atom_near =
        NULL;  // 四阶Bspline插值，每个原子附近的网格点 size = atom_numbers*64

    int* PME_atom_near_global = NULL;  // 用于检验电荷插值错误

    // 控制参数
    float beta;
    float cutoff = 10.0;
    float tolerance = 0.00001f;
    int update_interval = 1;
    VECTOR* force_backup;
    bool calculate_reciprocal_part = true;
    bool calculate_excluded_part = true;
    // 排除项能量参数，若使用半近邻表应设为1.0f,
    // 若区域分解使用完整紧邻表应设为0.5f
    float exclude_factor = 1.0f;

    // 非中性时的能量额外项处理
    float neutralizing_factor = 0;  // 系数
    float* charge_sum = NULL;       // 电荷量
    float* charge_square = NULL;    // 电荷平方，用于确定性 self 能量求和

    // 能量参数
    float* d_direct_atom_energy = NULL;      // 每个原子的直接的能量数组
    float* d_correction_atom_energy = NULL;  // 每个原子的修正能量数组
    float* d_reciprocal_ene = NULL;
    float* d_self_ene = NULL;
    float* d_direct_ene = NULL;
    float* d_correction_ene = NULL;
    float* d_ee_ene = NULL;
    float reciprocal_ene = 0;
    float self_ene = 0;
    float direct_ene = 0;
    float correction_ene = 0;
    float ee_ene = 0;

    deviceStream_t pm_stream;
    void Create_Stream();
    void Destroy_Stream();

    // 从局部编号到全局编号的映射
    int* atom_id_l_g = NULL;
    int* atom_id_g_l = NULL;
    VECTOR* g_crd = NULL;  // global 编号的crd
    VECTOR* g_frc = NULL;  // global  编号的frc
    void reset_global_force(int no_direct_interaction_virtual_atom_numbers);
    void add_force_g_to_l(VECTOR* l_frc);

    enum PME_ENERGY_PART
    {
        DIRECT = 1,
        RECIPROCAL = 2,
        CORRECTION = 4,
        SELF = 8,
        TOTAL = 15
    };

    // 初始化PME系统（PME信息）
    void Initial(CONTROLLER* controller, int atom_numbers, LTMatrix3 cell,
                 LTMatrix3 rcell, VECTOR box_length, float cutoff,
                 int no_direct_interaction_virtual_atom_numbers,
                 const char* module_name = NULL);

    // 重初始化PME系统（PME信息）
    void Reinitial(CONTROLLER* controller, LTMatrix3 cell, LTMatrix3 rcell,
                   VECTOR box_length, const char* module_name = NULL);
    // 清除内存
    void Clear();

    /*-----------------------------------------------------------------------------------------
    下面的函数是普通md的需求
    ------------------------------------------------------------------------------------------*/

    // 计算exclude能量和能量，并加到每个原子上
    void PME_Excluded_Force_With_Atom_Energy(
        const VECTOR* crd, const LTMatrix3 cell, const LTMatrix3 rcell,
        const float* charge, const int* excluded_list_start,
        const int* excluded_list, const int* excluded_atom_numbers, VECTOR* frc,
        int need_energy, float* atom_ene, LTMatrix3* atom_virial);
    // 计算倒空间力，并计算自能和倒空间的能量，并结合其他部分计算出PME部分给出的总维里（需要先计算其他部分）
    void PME_Reciprocal_Force_With_Energy_And_Virial(
        const VECTOR* crd, const LTMatrix3 cell, const LTMatrix3 rcell,
        const float* charge, VECTOR* force, int need_virial, int need_energy,
        LTMatrix3* d_virial, float* d_potential, int step);

    void Update_Box(LTMatrix3 cell, LTMatrix3 rcell, LTMatrix3 g, float dt);
    void Step_Print(CONTROLLER* controller);

    void Get_Local(CONTROLLER* controller, int step, VECTOR box_length,
                   float* pme_charge);
    void MPI_PME_Excluded_Force_With_Atom_Energy(
        const int N, const int* id1, const int* id2, const VECTOR* crd,
        const LTMatrix3 cell, const LTMatrix3 rcell, const float* charge,
        const int* excluded_list_start, const int* excluded_list,
        const int* excluded_atom_numbers, VECTOR* frc, int need_energy,
        float* atom_ene, int need_virial, LTMatrix3* atom_virial);
    void MPI_PME_Reciprocal_Force_With_Energy_And_Virial(
        VECTOR* frc, const VECTOR* crd, const LTMatrix3 cell,
        const LTMatrix3 rcell, const float* charge, int need_virial,
        int need_energy, LTMatrix3* d_virial, float* d_potential, int step);

    void Domain_Decomposition(CONTROLLER* controller, VECTOR box_length,
                              INT_VECTOR pp_split_num);
    void Send_Recv_Dom_Dec(CONTROLLER* controller);
    void Find_Neighbor_Domain(CONTROLLER* controller);
    void Get_Atoms(CONTROLLER* controller, VECTOR* pme_crd, float* pme_charge,
                   int pp_atom_numbers, VECTOR* pp_crd, float* pp_charge,
                   int* atom_local, bool atom_number_label, bool charge_label,
                   bool crd_label, bool id_label);
    void Get_Ghost(CONTROLLER* controller, VECTOR* pme_crd, float* pme_charge,
                   LTMatrix3 cell, LTMatrix3 rcell);
    void Update_Ghost(CONTROLLER* controller, VECTOR* pme_crd);
    void Send_Recv_Force(CONTROLLER* controller, VECTOR* frc, VECTOR* pp_frc,
                         int pp_atom_numbers);
    void Distribute_Ghost_Information(CONTROLLER* controller, VECTOR* frc);
};

__global__ void PME_Atom_Near(const VECTOR* crd, int* PME_atom_near,
                              const int PME_Nin, const LTMatrix3 cell,
                              const LTMatrix3 rcell, const int atom_numbers,
                              const int fftx, const int ffty, const int fftz,
                              UNSIGNED_INT_VECTOR* PME_uxyz, VECTOR* PME_frxyz,
                              VECTOR* force_backup);

__global__ void PME_Q_Spread(int* PME_atom_near, const float* charge,
                             const VECTOR* PME_frxyz, float* PME_Q,
                             const int atom_numbers);

__global__ void PME_BCFQ(FFT_COMPLEX* PME_FQ, float* PME_BC, int PME_Nfft);

__global__ void PME_Energy_Product(const int element_number, const float* list1,
                                   const float* list2, float* sum);

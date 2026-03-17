#pragma once

#include "../common.h"
#include "../control.h"

struct CMAP
{
    char module_name[CHAR_LENGTH_MAX];
    int is_initialized = 0;
    int is_controller_printf_initialized = 0;
    int last_modify_date = 20260216;

    // 基本文件信息读入
    int tot_cmap_num = 0;
    int uniq_cmap_num = 0;
    int uniq_gridpoint_num = 0;
    int* h_cmap_resolution = NULL;
    int* d_cmap_resolution = NULL;
    float* h_inter_coeff = NULL;
    float* d_inter_coeff = NULL;
    float* grid_value = NULL;
    int* type_offset = NULL;

    // 插值系数矩阵的逆矩阵，相当于解线性方程组得到插值多项式系数
    /*
    每16个格点可以做一次插值，二元多项式形式为：F(x,y) =
    \sum_{i.j=0,1,2,3}(c_{ij}x^iy^j),格点的取法为4*4,采用PBC。
    对初始数据格点连线长度归一化后，可以得到系数向量(16*1):

                         c = A^{-1}p

    其中p向量是包含格点值(4)，一阶差分(8)，二阶差分(4)的（16*1）向量
    */

    // 分子的坐标和双二面角相关拓扑文件

    int* h_atom_a = NULL;
    int* d_atom_a = NULL;
    int* h_atom_b = NULL;
    int* d_atom_b = NULL;
    int* h_atom_c = NULL;
    int* d_atom_c = NULL;
    int* h_atom_d = NULL;
    int* d_atom_d = NULL;
    int* h_atom_e = NULL;
    int* d_atom_e = NULL;
    int* h_cmap_type = NULL;
    int* d_cmap_type = NULL;
    float** h_coeff_ptr = NULL;
    float** d_coeff_ptr = NULL;

    float* d_cmap_ene = NULL;
    float h_sigma_of_cmap_ene = 0;
    float* d_sigma_of_cmap_ene = NULL;

    // 初始化模块
    void Initial(CONTROLLER* controller, const char* module_name = NULL);
    // 内存分配
    void Memory_Allocate();
    // 计算插值系数
    void Interpolation(CONTROLLER* controller);
    // CUDA计算
    void Parameter_Host_to_Device();

    // 能量和力计算
    void CMAP_Force_With_Atom_Energy_And_Virial(
        const VECTOR* crd, const LTMatrix3 cell, const LTMatrix3 rcell,
        VECTOR* frc, int need_potential, float* atom_energy, int need_pressure,
        LTMatrix3* atom_virial);

    void Step_Print(CONTROLLER* controller, bool print_sum = true);
    /*
    以下用于区域分解
*/
    int* d_atom_a_local = NULL;
    int* d_atom_b_local = NULL;
    int* d_atom_c_local = NULL;
    int* d_atom_d_local = NULL;
    int* d_atom_e_local = NULL;
    int* d_cmap_type_local = NULL;

    int local_atom_numbers = 0;
    int num_cmap_local = 0;
    int* d_num_cmap_local = NULL;
    void Get_Local(int* atom_local, int local_atom_numbers, int ghost_numbers,
                   char* atom_local_label, int* atom_local_id);
};

#pragma once
#include "../collective_variable/collective_variable.h"
#include "../common.h"
#include "../control.h"
// debug

// virtual atom type 0
//  x_v = x_1
//  y_v = y_1
//  z_v = 2 * h - z_1
//          1
//          ↑
//          —    镜面   ↑
//          ↓     |     h
//          V     |     |
//             盒子底部 ↓
struct VIRTUAL_TYPE_0
{
    int virtual_atom;
    int from_1;
    float h_double;
};

struct VIRTUAL_TYPE_0_INFROMATION
{
    int virtual_numbers = 0;
    int local_numbers = 0;
    int* d_local_numbers = NULL;
    VIRTUAL_TYPE_0* h_virtual_type_0 = NULL;
    VIRTUAL_TYPE_0* d_virtual_type_0 = NULL;
    VIRTUAL_TYPE_0* l_virtual_type_0 = NULL;
};

// virtual atom type 1
// r_v1 = a * r_21
//    1 - a - v - 1-a - 2
struct VIRTUAL_TYPE_1
{
    int virtual_atom;
    int from_1;
    int from_2;
    float a;
};

struct VIRTUAL_TYPE_1_INFROMATION
{
    int virtual_numbers = 0;
    int local_numbers = 0;
    int* d_local_numbers = NULL;
    VIRTUAL_TYPE_1* h_virtual_type_1 = NULL;
    VIRTUAL_TYPE_1* d_virtual_type_1 = NULL;
    VIRTUAL_TYPE_1* l_virtual_type_1 = NULL;
};

// virtual atom type 2
// r_v1 = a * r_21 + b * r_31
struct VIRTUAL_TYPE_2
{
    int virtual_atom;
    int from_1;
    int from_2;
    int from_3;
    float a;
    float b;
};

struct VIRTUAL_TYPE_2_INFROMATION
{
    int virtual_numbers = 0;
    int local_numbers = 0;
    int* d_local_numbers = NULL;
    VIRTUAL_TYPE_2* h_virtual_type_2 = NULL;
    VIRTUAL_TYPE_2* d_virtual_type_2 = NULL;
    VIRTUAL_TYPE_2* l_virtual_type_2 = NULL;
};

// virtual atom type 3
// r_v1 =  d * (r_12 + k * r_23)/|r_12 + k * r_23|
//            1
//            ↑  d
//            V
//         ↙ |  ↘
//       2- k - 1-k -3
struct VIRTUAL_TYPE_3
{
    int virtual_atom;
    int from_1;
    int from_2;
    int from_3;
    float d;
    float k;
};

struct VIRTUAL_TYPE_3_INFROMATION
{
    int virtual_numbers = 0;
    int local_numbers = 0;
    int* d_local_numbers = NULL;
    VIRTUAL_TYPE_3* h_virtual_type_3 = NULL;
    VIRTUAL_TYPE_3* d_virtual_type_3 = NULL;
    VIRTUAL_TYPE_3* l_virtual_type_3 = NULL;
};

// virtual atom type 4
// center for a group of atoms with arbitrary weights
// r_v1 = \sum_i w_i * r_i
struct VIRTUAL_TYPE_4
{
    int virtual_atom;
    int atom_numbers;
    int *d_from, *h_from;
    float *h_weight, *d_weight;
};

struct VIRTUAL_TYPE_4_INFROMATION
{
    int virtual_numbers = 0;
    VIRTUAL_TYPE_4* h_virtual_type_4 = NULL;
};

struct VIRTUAL_LAYER_INFORMATION
{
    VIRTUAL_TYPE_0_INFROMATION v0_info;
    VIRTUAL_TYPE_1_INFROMATION v1_info;
    VIRTUAL_TYPE_2_INFROMATION v2_info;
    VIRTUAL_TYPE_3_INFROMATION v3_info;
    VIRTUAL_TYPE_4_INFROMATION v4_info;
};

struct VIRTUAL_INFORMATION
{
    // 模块信息
    char module_name[CHAR_LENGTH_MAX];
    int is_initialized = 0;
    int is_controller_printf_initialized = 0;
    int last_modify_date = 20260216;
    bool need_atomic = false;
    bool local_state_ready = false;

    // 内容信息
    int max_level = 0;  // 最大的虚拟层级

    int* virtual_level =
        NULL;  // 每个原子的虚拟位点层级：0->实原子，1->只依赖于实原子，2->依赖的原子的虚拟等级最高为1，以此类推...

    std::vector<VIRTUAL_LAYER_INFORMATION>
        virtual_layer_info;  // 记录每个层级的信息

    void Initial(CONTROLLER* controller,
                 COLLECTIVE_VARIABLE_CONTROLLER* cv_controller,
                 int atom_numbers, int no_direct_vatom_numbers,
                 CheckMap cv_vatom_name, float* h_mass, int* system_freedom,
                 CONECT* connectivity,
                 const char* module_name = NULL);  // 初始化

    void Force_Redistribute(const VECTOR* crd, Boundary boundary,
                            VECTOR* frc);  // 进行力重分配

    void Coordinate_Refresh(VECTOR* crd,
                            Boundary boundary);  // 更新虚拟位点的坐标

    // 目前的虚原子构建策略中，将CV定义的虚原子独立处理
    // CV构建的虚原子只能是质心；而VIRTUAL_INFORMATION构建的虚原子不可以是V4类型。

    void Force_Redistribute_CV(const VECTOR* crd, Boundary boundary,
                               VECTOR* frc);  // 进行力重分配
    void Coordinate_Refresh_CV(VECTOR* crd,
                               Boundary boundary);  // 更新虚拟位点的坐标

    void Get_Local(const int* atom_local_id, const char* atom_local_label,
                   const int local_atom_numbers);
    void update_ug_connectivity(CONECT* connectivity);
};

#pragma once
#include "../collective_variable/collective_variable.h"
#include "../common.h"
#include "../control.h"
#include "virtual_atom_graph.h"
#include "virtual_atom_math.h"

// Notation for types 1-3:
//   r_ij = Get_Displacement(r_i, r_j, boundary).
//   PBC uses the periodic displacement; NOPBC uses r_i - r_j.
//   F_v denotes the force to redistribute to the source atoms.
// Types 0-3 require an acyclic dependency graph: a target must never be its
// own source. Repeated source indices alone are not self-dependencies.
//
// Virtual atom type 0: reflection in a plane normal to the z axis.
//   Intended: x_v = x_1, y_v = y_1, z_v = 2 * h - z_1.
//   NOTE: the loader stores h_double = 2 * h, but the current position
//   helper computes z_v = 2 * h_double - z_1 (i.e. 4 * h - z_1).
//   This legacy factor-of-two discrepancy is not corrected here.
//   Force: F_1 += (F_v.x, F_v.y, -F_v.z).
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

// Virtual atom type 1: linear interpolation (extrapolation is also allowed).
//   r_v = r_1 + a * r_21.
//   F_1 += (1 - a) * F_v; F_2 += a * F_v.
//   For 0 <= a <= 1:  1 ---- a ---- V ---- (1 - a) ---- 2
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

// Virtual atom type 2: affine combination of three source positions.
//   r_v = r_1 + a * r_21 + b * r_31.
//   F_1 += (1 - a - b) * F_v; F_2 += a * F_v; F_3 += b * F_v.
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
    bool need_atomic = false;
    int* d_local_numbers = NULL;
    VIRTUAL_TYPE_2* h_virtual_type_2 = NULL;
    VIRTUAL_TYPE_2* d_virtual_type_2 = NULL;
    VIRTUAL_TYPE_2* l_virtual_type_2 = NULL;
};

// Virtual atom type 3: signed distance d along a normalized direction.
//   q = r_21 + k * r_32; u = q / |q|; r_v = r_1 + d * u.
//   g = (d / |q|) * (F_v - dot(F_v, u) * u).
//   F_1 += F_v - g; F_2 += (1 - k) * g; F_3 += k * g.
//   For |d| <= epsilon, r_v = r_1 and all force goes to atom 1.
//   Otherwise, |q| <= epsilon is an invalid geometry and triggers an error.
//   epsilon = VIRTUAL_ATOM_GEOMETRY_EPSILON.
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

// Virtual atom type 4: weighted sum of coordinates for a CV atom group.
//   r_v = sum_i(w_i * r_i); F_i += w_i * F_v.
//   The coordinate kernel does not unwrap molecules or normalize weights.
//   Mass-center weights are normalized by the initializer.
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
    bool local_state_ready = false;
    bool has_type_3 = false;
    CONTROLLER* controller = NULL;
    int* d_runtime_error = NULL;

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

    void Reset_Runtime_Error();
    void Throw_If_Runtime_Error(const char* operation);
};

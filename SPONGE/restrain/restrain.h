#pragma once
#include <string>
#include <vector>

#include "../Domain_decomposition/Domain_decomposition.h"
#include "../MD_core/MD_core.h"
#include "../common.h"
#include "../control.h"

namespace Xponge
{
struct PositionalRestraint;
}

struct RESTRAIN_INFORMATION
{
    char module_name[CHAR_LENGTH_MAX];
    int is_initialized = 0;
    int is_controller_printf_initialized = 0;
    int last_modify_date = 20260216;

    enum REFCOORD_SCALING
    {
        REFCOORD_SCALING_NO = 0,
        REFCOORD_SCALING_ALL = 1,
        REFCOORD_SCALING_COM_UG = 2,
        REFCOORD_SCALING_COM_RES = 3,
        REFCOORD_SCALING_COM_MOL = 4
    };

    // 支持各项异性的restrain算法
    // E_restrain = 0.5 * [W.x * (r.x-r_ref.x)^2
    //                    + W.y * (r.y-r_ref.y)^2
    //                    + W.z * (r.z-r_ref.z)^2]
    int restrain_numbers = 0;  // 限制的原子数量
    int atom_numbers = 0;
    int* h_lists = NULL;       // 限制的原子序号
    int* d_lists = NULL;       // 限制的原子序号
    VECTOR* h_weights = NULL;  // 限制力常数(各向异性)
    VECTOR* d_weights = NULL;  // 限制力常数(各向异性)

    int if_single_weight = 0;     // 是否使用单一力常数
    float single_weight = 20.0f;  // 单一限制力常数

    float* d_restrain_ene = NULL;
    float* h_sum_of_restrain_ene = NULL;
    float* d_sum_of_restrain_ene = NULL;

    VECTOR* crd_ref = NULL;        // 限制的参考坐标(在GPU上)
    VECTOR* d_ref_crd_all = NULL;  // 全局参考坐标(在GPU上)

    int refcoord_scaling = REFCOORD_SCALING_NO;
    bool calc_virial = true;
    std::string h5_restraint_name = "default";

    // Restrain初始化(总原子数，GPU上所有原子的坐标，控制，模块名)
    void Initial(CONTROLLER* control, const int atom_numbers, const VECTOR* crd,
                 const char* module_name = NULL);
    void Initial(CONTROLLER* control, const int atom_numbers, const VECTOR* crd,
                 const Xponge::PositionalRestraint& native_restraint,
                 const char* module_name = NULL);

    void Update_Refcoord_Scaling(MD_INFORMATION* md_info, const LTMatrix3 g,
                                 float dt, int* atom_local,
                                 int local_atom_numbers, char* atom_local_label,
                                 int* atom_local_id);

    bool Export_H5_Reference_Coordinates(std::vector<float>* coordinates,
                                         std::string* error) const;
    bool Apply_H5_Reference_Coordinates(const std::vector<float>& coordinates,
                                        std::string* error);

    void Init_Com_Cache_If_Needed(const int atom_numbers,
                                  const MD_INFORMATION& md_info);
    void Update_Group_COM(const int local_atom_numbers, const VECTOR* crd,
                          const int* atom_local, const int* atom_to_group,
                          const float* mass, const int group_numbers,
                          float* d_sum_mass, VECTOR* d_sum_pos, VECTOR* d_com,
                          float* h_sum_mass, VECTOR* h_sum_pos, VECTOR* h_com);

    // 计算Restrain的能量、力和维里
    void Restraint(const VECTOR* crd, Boundary boundary, int need_potential,
                   float* atom_energy, int need_pressure,
                   LTMatrix3* atom_virial, VECTOR* frc, MD_INFORMATION* md_info,
                   DOMAIN_INFORMATION* dd);

    /*
        以下用于区域分解
    */

    int local_restrain_numbers = 0;  // 本地限制的原子数量
    int* d_local_restrain_numbers = NULL;

    int* d_local_restrain_list = NULL;  // 本地限制的本地原子序号
    VECTOR* local_crd_ref = NULL;       // 本地限制的参考坐标
    VECTOR* local_weights = NULL;       // 本地限制的权重

    int com_cache_initialized = 0;
    int cached_atom_numbers = 0;
    int cached_ug_numbers = 0;
    int cached_res_numbers = 0;
    int cached_mol_numbers = 0;

    int* d_atom_to_ug = NULL;
    int* d_atom_to_res = NULL;
    int* d_atom_to_mol = NULL;

    float* d_sum_mass_ug = NULL;
    float* d_sum_mass_res = NULL;
    float* d_sum_mass_mol = NULL;

    VECTOR* d_sum_pos_ug = NULL;
    VECTOR* d_sum_pos_res = NULL;
    VECTOR* d_sum_pos_mol = NULL;

    VECTOR* d_com_ug = NULL;
    VECTOR* d_com_res = NULL;
    VECTOR* d_com_mol = NULL;

    float* h_sum_mass_ug = NULL;
    float* h_sum_mass_res = NULL;
    float* h_sum_mass_mol = NULL;

    VECTOR* h_sum_pos_ug = NULL;
    VECTOR* h_sum_pos_res = NULL;
    VECTOR* h_sum_pos_mol = NULL;

    VECTOR* h_com_ug = NULL;
    VECTOR* h_com_res = NULL;
    VECTOR* h_com_mol = NULL;

    void Get_Local(int* atom_local, int local_atom_numbers,
                   char* atom_local_label, int* atom_local_id);
    void Step_Print(CONTROLLER* controller);
};

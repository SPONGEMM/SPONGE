#pragma once
#include "../common.h"
#include "../control.h"
#include "constrain.h"

struct SHAKE
{
    char module_name[CHAR_LENGTH_MAX];
    int is_initialized = 0;
    int is_controller_printf_initialized = 0;
    int last_modify_date = 20260216;

    CONSTRAIN* constrain;

    // 约束内力，使得主循环中更新后的坐标加上该力（力的方向与更新前的pair方向一致）修正，得到满足约束的坐标。
    VECTOR* constrain_frc = NULL;
    // 每对的维里
    LTMatrix3* d_pair_virial = NULL;
    // 总维里
    LTMatrix3* d_virial = NULL;
    // 进行constrain迭代过程中的不断微调的原子uint坐标
    VECTOR* test_crd = NULL;
    // 主循环中更新前的pair向量信息
    VECTOR* last_pair_dr = NULL;

    float step_length =
        1.0f;  // 迭代求力时选取的步长，步长为1.可以刚好严格求得两体的constrain
    // 但对于三体及以上的情况，步长小一点会更稳定，但随之而来可能要求迭代次数增加
    int iteration_numbers = 25;  // 迭代步数

    // 在初始化后进行初始化
    void Initial_SHAKE(CONTROLLER* controller, CONSTRAIN* constrain,
                       const char* module_name = NULL);

    // 记录更新前的距离
    void Remember_Last_Coordinates(const VECTOR* crd, Boundary boundary);
    // 将速度投影到SHAKE约束流形
    bool Project_Velocity_To_Constraint_Manifold(
        VECTOR* vel, VECTOR* crd, const float* mass_inverse, Boundary boundary,
        int local_atom_numbers, bool update_coordinates = true);
    // 进行约束迭代
    void Constrain(int atom_numbers, VECTOR* crd, VECTOR* vel,
                   const float* mass_inverse, const float* d_mass,
                   Boundary boundary, int need_pressure, LTMatrix3* d_stress);
};

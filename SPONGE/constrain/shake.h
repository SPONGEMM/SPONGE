#pragma once
#include "../common.h"
#include "../control.h"
#include "constrain.h"

template <class Real>
struct SHAKE_SOLVER
{
    using Scalar = Real;
    char module_name[CHAR_LENGTH_MAX];
    int is_initialized = 0;
    int is_controller_printf_initialized = 0;
    int last_modify_date = 20260216;

    SHAKE_SOLVER() = default;
    SHAKE_SOLVER(const SHAKE_SOLVER&) = delete;
    SHAKE_SOLVER& operator=(const SHAKE_SOLVER&) = delete;
    bool owns_buffers = false;
    ~SHAKE_SOLVER()
    {
        if (!owns_buffers) return;
        if (constrain_frc) deviceFree(constrain_frc);
        if (test_crd) deviceFree(test_crd);
        if (last_pair_dr) deviceFree(last_pair_dr);
        if (d_pair_virial) deviceFree(d_pair_virial);
        if (d_virial) deviceFree(d_virial);
    }
    CONSTRAIN* constrain;

    // 约束内力，使得主循环中更新后的坐标加上该力（力的方向与更新前的pair方向一致）修正，得到满足约束的坐标。
    ConstraintVector<Real>* constrain_frc = NULL;
    // 每对的维里
    ConstraintTensor<Real>* d_pair_virial = NULL;
    // 总维里
    ConstraintTensor<Real>* d_virial = NULL;
    // 进行constrain迭代过程中的不断微调的原子uint坐标
    ConstraintVector<Real>* test_crd = NULL;
    // 主循环中更新前的pair向量信息
    ConstraintVector<Real>* last_pair_dr = NULL;

    Real step_length =
        1.0f;  // 迭代求力时选取的步长，步长为1.可以刚好严格求得两体的constrain
    // 但对于三体及以上的情况，步长小一点会更稳定，但随之而来可能要求迭代次数增加
    int iteration_numbers = 25;  // 迭代步数
    bool use_small_groups = true;
    Real tolerance = Real(1e-4);
    void Get_Local(int local_atoms);

    // 在初始化后进行初始化
    void Initial_SHAKE(CONTROLLER* controller, CONSTRAIN* constrain,
                       const char* module_name = NULL);

    // 记录更新前的距离
    void Remember_Last_Coordinates(const ConstraintVector<Real>* crd,
                                   Boundary boundary);
    // 将速度投影到SHAKE约束流形
    bool Project_Velocity_To_Constraint_Manifold(
        ConstraintVector<Real>* vel, ConstraintVector<Real>* crd,
        const Real* mass_inverse, Boundary boundary, int local_atom_numbers,
        bool update_coordinates = true);
    // 进行约束迭代
    void Constrain(int atom_numbers, ConstraintVector<Real>* crd,
                   ConstraintVector<Real>* vel, const Real* mass_inverse,
                   const Real* d_mass, Boundary boundary, int need_pressure,
                   ConstraintTensor<Real>* d_stress);

   private:
    SMALL_CONSTRAINT_GROUPS small_groups;
    CONTROLLER* controller = nullptr;
};

// Float-storage MD boundary; implementation and work arrays use selected
// precision.
struct SHAKE
{
    void Initial_SHAKE(CONTROLLER* controller, CONSTRAIN* constrain,
                       const char* name = nullptr);
    void Get_Local(int n);
    void Constrain(int n, VECTOR* crd, VECTOR* vel, const float* inv,
                   const float* mass, Boundary boundary, int pressure,
                   LTMatrix3* stress);
    bool Project_Velocity_To_Constraint_Manifold(
        VECTOR* vel, VECTOR* crd, const float* inv, Boundary boundary, int n,
        bool update_coordinates = true);
    void Remember_Last_Coordinates(const VECTOR* crd, Boundary boundary);

   private:
    int local_atoms = 0;
    ConstraintDispatch<SHAKE_SOLVER> backend;
};

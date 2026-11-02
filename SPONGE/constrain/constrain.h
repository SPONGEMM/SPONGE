#pragma once
#include "../common.h"
#include "../control.h"

struct CONSTRAIN_PAIR
{
    int atom_i_serial;
    int atom_j_serial;
    float constant_r;
    float
        constrain_k;  // 这个并不是说有个弹性系数来固定，而是迭代时，有个系数k=m1*m2/(m1+m2)
};

struct CONSTRAIN
{
    char module_name[CHAR_LENGTH_MAX];
    int is_initialized = 0;
    int is_controller_printf_initialized = 0;
    int last_modify_date = 20260216;

    int atom_numbers = 0;
    float dt = 0.001f;
    float dt_inverse;

    float v_factor = 1.0f;  // 一个积分步中,一个微小的力F对速度的影响，即dv =
                            // v_factor * F * dt/m
    float x_factor = 1.0f;  // 一个积分步中,一个微小的力F对位移的影响，即dx =
                            // x_factor * F * dt * dt/m
    float constrain_mass = 3.3;  // 对质量小于该值的原子进行限制

    // 在初始化的时候用到，在实际计算中不会使用,在初始化时已经被释放
    int bond_constrain_pair_numbers = 0;
    CONSTRAIN_PAIR* h_bond_pair = NULL;

    // 在实际计算中使用，体系总的constrain pair
    int constrain_pair_numbers = 0;
    int maximum_constraint_degree = 0;
    CONSTRAIN_PAIR* h_constrain_pair = NULL;
    CONSTRAIN_PAIR* d_constrain_pair = NULL;

    // 用于暂时记录bond的信息，便于angle中搜索bond长度
    // 这些指针指向的空间并不由本模块申请且不由本模块释放
    struct BOND_INFORMATION
    {
        int bond_numbers;
        const int* atom_a = NULL;
        const int* atom_b = NULL;
        const float* bond_r = NULL;
    } bond_info;

    // 默认的Initial需要按照下面的顺序：
    // Initial_List
    // Initial_Constrain
    void Initial_List(CONTROLLER* controller, PAIR_DISTANCE con_dis,
                      float* atom_mass, const char* module_name = NULL);
    void Initial_Constrain(CONTROLLER* controller, const int atom_numbers,
                           const float dt, const VECTOR box_length,
                           float* atom_mass, int* system_freedom);

    // 为ug构造connectivity
    void update_ug_connectivity(CONECT* connectivity);

    // device 分配constrain pair
    int num_pair_local = 0;
    int* d_num_pair_local = NULL;
    CONSTRAIN_PAIR* constrain_pair_local = NULL;
    // 获得本设备中的constrain
    void Get_Local(const int* atom_local_id, const char* atom_local_label,
                   const int local_atom_numbers);
};

// Independent connected components with at most three constraints/four atoms.
// The general solvers remain the fallback if any component exceeds this bound.
struct SMALL_CONSTRAINT_GROUP
{
    int pair_count = 0, atom_count = 0;
    int pairs[3] = {}, atoms[4] = {}, a[3] = {}, b[3] = {};
};

struct SMALL_CONSTRAINT_GROUPS
{
    SMALL_CONSTRAINT_GROUP* data = nullptr;
    int count = 0;
    SMALL_CONSTRAINT_GROUPS() = default;
    SMALL_CONSTRAINT_GROUPS(const SMALL_CONSTRAINT_GROUPS&) = delete;
    SMALL_CONSTRAINT_GROUPS& operator=(const SMALL_CONSTRAINT_GROUPS&) = delete;
    ~SMALL_CONSTRAINT_GROUPS();
    void Clear();
    void Build(CONTROLLER* controller, const std::vector<CONSTRAIN_PAIR>& pairs,
               int atoms);
};

// Explicit selection avoids dynamic indexing of short per-thread arrays,
// which otherwise materializes these arrays in CUDA local memory.
static __device__ __forceinline__ VECTOR
Small_Group_Vector(const VECTOR* values, int i)
{
    return i == 0 ? values[0]
                  : (i == 1 ? values[1] : (i == 2 ? values[2] : values[3]));
}

// Test the requested relative length tolerance on the current displacement.
// Evaluate only this scalar predicate in double precision to avoid cancellation
// in r^2 - l^2. Box size and coordinate-storage error must not tighten the
// user's convergence target; later coordinate rounding is a separate accuracy
// issue.
static __device__ __forceinline__ bool Constraint_Within_Tolerance(
    const VECTOR& displacement, float target, float tolerance)
{
    const double x = displacement.x, y = displacement.y, z = displacement.z;
    const double r2 = x * x + y * y + z * z;
    const double lower = double(target) * (1.0 - double(tolerance));
    const double upper = double(target) * (1.0 + double(tolerance));
    // NaNs fail the comparisons. No square root or division is needed.
    return r2 >= lower * lower && r2 <= upper * upper;
}

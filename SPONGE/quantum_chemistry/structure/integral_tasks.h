#ifndef QC_STRUCTURE_INTEGRAL_TASKS_H
#define QC_STRUCTURE_INTEGRAL_TASKS_H

#include "../../common.h"

// 单电子双中心积分任务
struct QC_ONE_E_TASK
{
    int x, y;
};

// 双电子四中心积分任务
struct QC_ERI_TASK
{
    int x, y, z, w;
};

// 积分任务拓扑信息（初始化后基本不变）
struct QC_Integral_Topology
{
    // 单电子积分
    int n_1e_tasks = 0;
    std::vector<QC_ONE_E_TASK> h_1e_tasks;

    // 壳层对（i >= j），用于直接 SCF 筛选
    int n_shell_pairs = 0;
    std::vector<QC_ONE_E_TASK> h_shell_pairs;
    std::vector<float> h_shell_pair_bounds;

    // 壳层对按 (l_i, l_j) 类型分组，用于运行时类型分发
    // 最多支持到 l=4（g 壳层）：5*5=25
    static const int MAX_PAIR_TYPES = 25;
    // 当前实际存在的类型数
    int n_pair_types = 0;
    int pair_type_offset[MAX_PAIR_TYPES] = {};
    int pair_type_count[MAX_PAIR_TYPES] = {};
    int pair_type_l0[MAX_PAIR_TYPES] = {};  // 该类型中第一个壳层的 l
    int pair_type_l1[MAX_PAIR_TYPES] = {};  // 该类型中第二个壳层的 l
    std::vector<int> h_sorted_pair_ids;
    int* d_sorted_pair_ids = NULL;

    // 运行时筛选组合信息
    struct ScreenCombo
    {
        int pair_base_A, n_A;
        int pair_base_B, n_B;
        int n_quartets;      // 该组合中的四元组总数
        int output_offset;   // 在 d_screened_tasks 中的输出偏移
        int same_type;       // 1 表示三角区，0 表示矩形区
        int l0, l1, l2, l3;  // 用于 ERI 内核选择的壳层角动量
    };
    static const int MAX_COMBOS = 512;
    int n_combos = 0;
    ScreenCombo h_combos[MAX_COMBOS];
    int combo_prefix[MAX_COMBOS + 1] = {};  // n_quartets 的前缀和
    int total_quartets = 0;
};

// 积分任务设备缓冲区（初始化分配，SCF 中复用）
struct QC_Integral_Device_Buffers
{
    QC_ONE_E_TASK* d_1e_tasks = NULL;
    QC_ONE_E_TASK* d_shell_pairs = NULL;
    float* d_shell_pair_bounds = NULL;
    int* d_sorted_pair_ids = NULL;
    QC_Integral_Topology::ScreenCombo* d_combos = NULL;

    // 筛选输出缓冲区（每轮 SCF 迭代复用）
    QC_ERI_TASK* d_screened_tasks = NULL;
    int* d_screen_counts = NULL;  // 每个组合的原子计数器 [MAX_COMBOS]
    int screened_buf_capacity = 0;
};

// 积分内核参数与筛选阈值
struct QC_Integral_Kernel_Params
{
    int eri_hr_base = 13;
    int eri_hr_size = 28561;
    int eri_shell_buf_size = 50625;
    float eri_prim_screen_tol = 1e-12f;
    float direct_eri_prim_screen_tol = 1e-10f;
    float eri_shell_screen_tol = 1e-10f;
};

// 积分任务总上下文
struct QC_INTEGRAL_TASKS
{
    using ScreenCombo = QC_Integral_Topology::ScreenCombo;
    static const int MAX_PAIR_TYPES = QC_Integral_Topology::MAX_PAIR_TYPES;
    static const int MAX_COMBOS = QC_Integral_Topology::MAX_COMBOS;

    QC_Integral_Topology topo;
    QC_Integral_Device_Buffers buffers;
    QC_Integral_Kernel_Params params;
};

#endif

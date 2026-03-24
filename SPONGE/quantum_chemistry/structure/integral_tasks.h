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

// 积分任务
struct QC_INTEGRAL_TASKS
{
    // 单电子积分
    int n_1e_tasks = 0;
    std::vector<QC_ONE_E_TASK> h_1e_tasks;
    QC_ONE_E_TASK* d_1e_tasks = NULL;

    // 壳层对（i >= j），用于 direct SCF screening
    int n_shell_pairs = 0;
    std::vector<QC_ONE_E_TASK> h_shell_pairs;
    QC_ONE_E_TASK* d_shell_pairs = NULL;
    std::vector<float> h_shell_pair_bounds;
    float* d_shell_pair_bounds = NULL;

    // 双电子积分 (legacy task list, kept for CPU path and generic kernel)
    int n_eri_tasks = 0;
    std::vector<QC_ERI_TASK> h_eri_tasks;
    QC_ERI_TASK* d_eri_tasks = NULL;

    // 预分桶 (init 时按 shell type 排序，记录每个桶的 offset/count)
    static const int N_BUCKETS = 17;
    int bucket_offset[N_BUCKETS] = {};
    int bucket_count[N_BUCKETS] = {};

    // On-the-fly pair type dispatch:
    // Shell pairs grouped by (l_i, l_j) type.
    // pair_type_id = l_i * (max_l + 1) + l_j
    // sorted_pair_ids[pair_type_offset[t]..+pair_type_count[t]) = pair_ids of type t
    static const int MAX_PAIR_TYPES = 16; // supports up to l=3 (f shells): 4*4=16
    int n_pair_types = 0;       // actual number of types present
    int pair_type_offset[MAX_PAIR_TYPES] = {};
    int pair_type_count[MAX_PAIR_TYPES] = {};
    int pair_type_l0[MAX_PAIR_TYPES] = {};  // l of shell x for this type
    int pair_type_l1[MAX_PAIR_TYPES] = {};  // l of shell y for this type
    std::vector<int> h_sorted_pair_ids;
    int* d_sorted_pair_ids = NULL;

    // Screening output buffer (reused each SCF iteration)
    QC_ERI_TASK* d_screened_tasks = NULL;  // compacted output
    int* d_screen_count = NULL;            // atomic counter (device)
    int screened_buf_capacity = 0;         // allocated size
    int eri_hr_base = 13;
    int eri_hr_size = 28561;
    int eri_shell_buf_size = 50625;
    float eri_prim_screen_tol = 1e-12f;
    float direct_eri_prim_screen_tol = 1e-10f;
    float eri_shell_screen_tol = 1e-10f;
};

#endif

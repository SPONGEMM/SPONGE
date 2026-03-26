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

    // On-the-fly pair type dispatch:
    // Shell pairs grouped by (l_i, l_j) type.
    // pair_type_id = l_i * (max_l + 1) + l_j
    // sorted_pair_ids[pair_type_offset[t]..+pair_type_count[t]) = pair_ids of
    // type t
    static const int MAX_PAIR_TYPES =
        25;                // supports up to l=4 (g shells): 5*5=25
    int n_pair_types = 0;  // actual number of types present
    int pair_type_offset[MAX_PAIR_TYPES] = {};
    int pair_type_count[MAX_PAIR_TYPES] = {};
    int pair_type_l0[MAX_PAIR_TYPES] = {};  // l of shell x for this type
    int pair_type_l1[MAX_PAIR_TYPES] = {};  // l of shell y for this type
    std::vector<int> h_sorted_pair_ids;
    int* d_sorted_pair_ids = NULL;

    // On-the-fly screening combo info
    struct ScreenCombo
    {
        int pair_base_A, n_A;
        int pair_base_B, n_B;
        int n_quartets;      // total quartets in this combo
        int output_offset;   // offset into d_screened_tasks
        int same_type;       // 1=triangular, 0=rectangular
        int l0, l1, l2, l3;  // shell angular momenta for ERI kernel selection
    };
    static const int MAX_COMBOS = 512;  // C(25+1,2)=325 for l_max=4
    int n_combos = 0;
    ScreenCombo h_combos[MAX_COMBOS];
    ScreenCombo* d_combos = NULL;
    int combo_prefix[MAX_COMBOS + 1] = {};  // prefix sum of n_quartets
    int total_quartets = 0;

    // Screening output buffer (reused each SCF iteration)
    QC_ERI_TASK* d_screened_tasks = NULL;
    int* d_screen_counts = NULL;  // per-combo atomic counters [MAX_COMBOS]
    int screened_buf_capacity = 0;
    int eri_hr_base = 13;
    int eri_hr_size = 28561;
    int eri_shell_buf_size = 50625;
    float eri_prim_screen_tol = 1e-12f;
    float direct_eri_prim_screen_tol = 1e-10f;
    float eri_shell_screen_tol = 1e-10f;
};

#endif

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

    // 双电子积分
    int n_eri_tasks = 0;
    std::vector<QC_ERI_TASK> h_eri_tasks;
    QC_ERI_TASK* d_eri_tasks = NULL;
    int eri_hr_base = 13;
    int eri_hr_size = 28561;
    int eri_shell_buf_size = 50625;
    float eri_prim_screen_tol = 1e-12f;
    float direct_eri_prim_screen_tol = 1e-10f;
    float eri_shell_screen_tol = 1e-10f;
};

#endif

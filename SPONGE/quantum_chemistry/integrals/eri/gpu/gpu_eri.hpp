#pragma once

#ifdef USE_GPU

#include "../../../structure/integral_tasks.h"

void QC_Launch_Screen(
    int n_total, const QC_INTEGRAL_TASKS::ScreenCombo* combos,
    const int* combo_prefix, int n_combos, const int* sorted_pair_ids,
    const QC_ONE_E_TASK* shell_pairs, const float* shell_pair_bounds,
    const float* pair_density_coul, const float* pair_density_exx_a,
    const float* pair_density_exx_b, float shell_screen_tol, float exx_scale_a,
    float exx_scale_b, QC_ERI_TASK* output_tasks, int* output_counts);

void QC_Build_Fock_Direct_GPU(
    const QC_INTEGRAL_TASKS& task_ctx, const int* atm, const int* bas,
    const float* env, const int* ao_offsets_cart, const int* ao_offsets_sph,
    const float* norms, const float* shell_pair_bounds,
    const float* pair_density_coul, const float* pair_density_exx_a,
    const float* pair_density_exx_b, float shell_screen_tol,
    const float* P_coul, const float* P_exx_a, const float* P_exx_b,
    float exx_scale_a, float exx_scale_b, int nao, int nao_sph,
    int is_spherical, const float* cart2sph_mat, float* F_a, float* F_b,
    float* global_hr_pool, float prim_screen_tol);

#endif

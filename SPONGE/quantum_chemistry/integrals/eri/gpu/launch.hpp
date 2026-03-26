#pragma once

#include "../../../structure/integral_tasks.h"

#define ERI_KERNEL_PARAMS                                                     \
    int n_tasks, const QC_ERI_TASK *tasks, const int *atm, const int *bas,    \
        const float *env, const int *ao_offsets_cart,                         \
        const int *ao_offsets_sph, const float *norms,                        \
        const float *shell_pair_bounds, const float *pair_density_coul,       \
        const float *pair_density_exx_a, const float *pair_density_exx_b,     \
        float shell_screen_tol, const float *P_coul, const float *P_exx_a,    \
        const float *P_exx_b, float exx_scale_a, float exx_scale_b, int nao,  \
        int nao_sph, int is_spherical, const float *cart2sph_mat, float *F_a, \
        float *F_b, float *global_hr_pool, int hr_base, int hr_size,          \
        int shell_buf_size, float prim_screen_tol

void QC_Launch_ssss(ERI_KERNEL_PARAMS);
void QC_Launch_psss(ERI_KERNEL_PARAMS);
void QC_Launch_spss(ERI_KERNEL_PARAMS);
void QC_Launch_ssps(ERI_KERNEL_PARAMS);
void QC_Launch_sssp(ERI_KERNEL_PARAMS);
void QC_Launch_ppss(ERI_KERNEL_PARAMS);
void QC_Launch_psps(ERI_KERNEL_PARAMS);
void QC_Launch_pssp(ERI_KERNEL_PARAMS);
void QC_Launch_spps(ERI_KERNEL_PARAMS);
void QC_Launch_spsp(ERI_KERNEL_PARAMS);
void QC_Launch_sspp(ERI_KERNEL_PARAMS);
void QC_Launch_sppp(ERI_KERNEL_PARAMS);
void QC_Launch_pspp(ERI_KERNEL_PARAMS);
void QC_Launch_ppsp(ERI_KERNEL_PARAMS);
void QC_Launch_ppps(ERI_KERNEL_PARAMS);
void QC_Launch_pppp(ERI_KERNEL_PARAMS);

void QC_Launch_D_L2(ERI_KERNEL_PARAMS);
void QC_Launch_D_L3(ERI_KERNEL_PARAMS);
void QC_Launch_D_L4(ERI_KERNEL_PARAMS);
void QC_Launch_D_L5(ERI_KERNEL_PARAMS);
void QC_Launch_D_L6(ERI_KERNEL_PARAMS);
void QC_Launch_D_L7(ERI_KERNEL_PARAMS);
void QC_Launch_D_L8(ERI_KERNEL_PARAMS);

void QC_Launch_Rys_L2(ERI_KERNEL_PARAMS);
void QC_Launch_Rys_L3(ERI_KERNEL_PARAMS);
void QC_Launch_Rys_L4(ERI_KERNEL_PARAMS);
void QC_Launch_Rys_L5(ERI_KERNEL_PARAMS);
void QC_Launch_Rys_L6(ERI_KERNEL_PARAMS);
void QC_Launch_Rys_L7(ERI_KERNEL_PARAMS);
void QC_Launch_Rys_L8(ERI_KERNEL_PARAMS);
void QC_Launch_Rys_L9(ERI_KERNEL_PARAMS);
void QC_Launch_Rys_L10(ERI_KERNEL_PARAMS);
void QC_Launch_Rys_L11(ERI_KERNEL_PARAMS);
void QC_Launch_Rys_L12(ERI_KERNEL_PARAMS);
void QC_Launch_Rys_L13(ERI_KERNEL_PARAMS);
void QC_Launch_Rys_L14(ERI_KERNEL_PARAMS);
void QC_Launch_Rys_L15(ERI_KERNEL_PARAMS);
void QC_Launch_Rys_L16(ERI_KERNEL_PARAMS);

void QC_Launch_Screen(
    int n_total, const QC_INTEGRAL_TASKS::ScreenCombo* combos,
    const int* combo_prefix, int n_combos, const int* sorted_pair_ids,
    const QC_ONE_E_TASK* shell_pairs, const float* shell_pair_bounds,
    const float* pair_density_coul, const float* pair_density_exx_a,
    const float* pair_density_exx_b, float shell_screen_tol, float exx_scale_a,
    float exx_scale_b, QC_ERI_TASK* output_tasks, int* output_counts);

#define DEFINE_ERI_LAUNCH(launch_name, kernel_name)                          \
    void launch_name(ERI_KERNEL_PARAMS)                                      \
    {                                                                        \
        const int threads = 256;                                             \
        Launch_Device_Kernel(                                                \
            kernel_name, (n_tasks + threads - 1) / threads, threads, 0, 0,   \
            n_tasks, tasks, atm, bas, env, ao_offsets_cart, ao_offsets_sph,  \
            norms, shell_pair_bounds, pair_density_coul, pair_density_exx_a, \
            pair_density_exx_b, shell_screen_tol, P_coul, P_exx_a, P_exx_b,  \
            exx_scale_a, exx_scale_b, nao, nao_sph, is_spherical,            \
            cart2sph_mat, F_a, F_b, global_hr_pool, hr_base, hr_size,        \
            shell_buf_size, prim_screen_tol);                                \
    }

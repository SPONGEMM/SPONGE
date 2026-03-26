#pragma once

#ifdef USE_GPU

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

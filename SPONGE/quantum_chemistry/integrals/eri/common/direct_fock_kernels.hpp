#pragma once

#include "eri_kernel_utils.hpp"
#include "eri_md_tensor.hpp"

static __global__ void QC_Init_Fock_Kernel(const int n, const float* H_core,
                                           const float* Vxc, const int use_vxc,
                                           float* F)
{
    SIMPLE_DEVICE_FOR(idx, n)
    {
        float v = H_core[idx];
        if (use_vxc) v += Vxc[idx];
        F[idx] = v;
    }
}

static __device__ void QC_Cart2Sph_Shell_ERI(
    const float* U_row_nc_ns, const int nao_s, const int* off_cart,
    const int* off_sph, const int* dims_cart, const int* dims_sph, float* buf0,
    float* buf1)
{
    const int nc0 = dims_cart[0], nc1 = dims_cart[1], nc2 = dims_cart[2],
              nc3 = dims_cart[3];
    const int ns0 = dims_sph[0], ns1 = dims_sph[1], ns2 = dims_sph[2],
              ns3 = dims_sph[3];

    for (int p = 0; p < ns0; p++)
        for (int b = 0; b < nc1; b++)
            for (int c = 0; c < nc2; c++)
                for (int d = 0; d < nc3; d++)
                {
                    double sum = 0.0;
                    for (int a = 0; a < nc0; a++)
                    {
                        sum += (double)U_row_nc_ns[(off_cart[0] + a) * nao_s +
                                                   (off_sph[0] + p)] *
                               (double)buf0[QC_Shell_Buffer_Index(
                                   a, b, c, d, nc1, nc2, nc3)];
                    }
                    buf1[QC_Shell_Buffer_Index(p, b, c, d, nc1, nc2, nc3)] =
                        (float)sum;
                }

    for (int p = 0; p < ns0; p++)
        for (int q = 0; q < ns1; q++)
            for (int c = 0; c < nc2; c++)
                for (int d = 0; d < nc3; d++)
                {
                    double sum = 0.0;
                    for (int b = 0; b < nc1; b++)
                    {
                        sum += (double)U_row_nc_ns[(off_cart[1] + b) * nao_s +
                                                   (off_sph[1] + q)] *
                               (double)buf1[QC_Shell_Buffer_Index(
                                   p, b, c, d, nc1, nc2, nc3)];
                    }
                    buf0[QC_Shell_Buffer_Index(p, q, c, d, ns1, nc2, nc3)] =
                        (float)sum;
                }

    for (int p = 0; p < ns0; p++)
        for (int q = 0; q < ns1; q++)
            for (int r = 0; r < ns2; r++)
                for (int d = 0; d < nc3; d++)
                {
                    double sum = 0.0;
                    for (int c = 0; c < nc2; c++)
                    {
                        sum += (double)U_row_nc_ns[(off_cart[2] + c) * nao_s +
                                                   (off_sph[2] + r)] *
                               (double)buf0[QC_Shell_Buffer_Index(
                                   p, q, c, d, ns1, nc2, nc3)];
                    }
                    buf1[QC_Shell_Buffer_Index(p, q, r, d, ns1, ns2, nc3)] =
                        (float)sum;
                }

    for (int p = 0; p < ns0; p++)
        for (int q = 0; q < ns1; q++)
            for (int r = 0; r < ns2; r++)
                for (int s = 0; s < ns3; s++)
                {
                    double sum = 0.0;
                    for (int d = 0; d < nc3; d++)
                    {
                        sum += (double)U_row_nc_ns[(off_cart[3] + d) * nao_s +
                                                   (off_sph[3] + s)] *
                               (double)buf1[QC_Shell_Buffer_Index(
                                   p, q, r, d, ns1, ns2, nc3)];
                    }
                    buf0[QC_Shell_Buffer_Index(p, q, r, s, ns1, ns2, ns3)] =
                        (float)sum;
                }
}

static __device__ bool QC_Compute_Shell_Quartet_ERI_Buffer(
    const int* sh, const int* atm, const int* bas, const float* env,
    const int* ao_offsets_cart, const int* ao_offsets_sph, const float* norms,
    const int is_spherical, const float* cart2sph_mat, const int nao_sph,
    float* HR, float* shell_eri, float* shell_tmp, int hr_base,
    int shell_buf_size, float prim_screen_tol, int* dims_eff, int* off_eff)
{
    int l[4];
    int np[4];
    int p_exp[4];
    int p_cof[4];
    int dims_cart[4];
    int dims_sph[4];
    int off_cart[4];
    float R[4][3];
    for (int i = 0; i < 4; i++)
    {
        l[i] = bas[sh[i] * 8 + 1];
        np[i] = bas[sh[i] * 8 + 2];
        p_exp[i] = bas[sh[i] * 8 + 5];
        p_cof[i] = bas[sh[i] * 8 + 6];
        dims_cart[i] = (l[i] + 1) * (l[i] + 2) / 2;
        dims_sph[i] = 2 * l[i] + 1;
        dims_eff[i] = QC_Shell_Dim(l[i], is_spherical);
        off_cart[i] = ao_offsets_cart[sh[i]];
        off_eff[i] = is_spherical ? ao_offsets_sph[sh[i]] : off_cart[i];

        const int ptr_R = atm[bas[sh[i] * 8 + 0] * 6 + 1];
        R[i][0] = env[ptr_R + 0];
        R[i][1] = env[ptr_R + 1];
        R[i][2] = env[ptr_R + 2];
    }

    const int shell_size =
        dims_cart[0] * dims_cart[1] * dims_cart[2] * dims_cart[3];
    if (shell_size > shell_buf_size) return false;
    for (int i = 0; i < shell_size; i++) shell_eri[i] = 0.0f;

    int comp_x[4][MAX_CART_SHELL];
    int comp_y[4][MAX_CART_SHELL];
    int comp_z[4][MAX_CART_SHELL];
    for (int s = 0; s < 4; s++)
    {
        for (int c = 0; c < dims_cart[s]; c++)
        {
            QC_Get_Lxyz_Device(l[s], c, comp_x[s][c], comp_y[s][c],
                               comp_z[s][c]);
        }
    }

    const float rab2 = (R[0][0] - R[1][0]) * (R[0][0] - R[1][0]) +
                       (R[0][1] - R[1][1]) * (R[0][1] - R[1][1]) +
                       (R[0][2] - R[1][2]) * (R[0][2] - R[1][2]);
    const float rcd2 = (R[2][0] - R[3][0]) * (R[2][0] - R[3][0]) +
                       (R[2][1] - R[3][1]) * (R[2][1] - R[3][1]) +
                       (R[2][2] - R[3][2]) * (R[2][2] - R[3][2]);

    float E_bra[3][5][5][9];
    float E_ket[3][5][5][9];

    for (int ip = 0; ip < np[0]; ip++)
    {
        for (int jp = 0; jp < np[1]; jp++)
        {
            float ai = env[p_exp[0] + ip];
            float aj = env[p_exp[1] + jp];
            float p = ai + aj;
            float inv_p = 1.0f / p;
            float P[3] = {(ai * R[0][0] + aj * R[1][0]) * inv_p,
                          (ai * R[0][1] + aj * R[1][1]) * inv_p,
                          (ai * R[0][2] + aj * R[1][2]) * inv_p};
            float kab = expf(-(ai * aj * inv_p) * rab2);
            float n_ab = env[p_cof[0] + ip] * env[p_cof[1] + jp] * kab;
            if (fabsf(n_ab) < prim_screen_tol) continue;

            float PA_val[3] = {(P[0] - R[0][0]), (P[1] - R[0][1]),
                               (P[2] - R[0][2])};
            float PB_val[3] = {(P[0] - R[1][0]), (P[1] - R[1][1]),
                               (P[2] - R[1][2])};
            for (int d = 0; d < 3; d++)
                compute_md_coeffs(E_bra[d], l[0], l[1], PA_val[d], PB_val[d],
                                  0.5f * inv_p);

            for (int kp = 0; kp < np[2]; kp++)
            {
                for (int lp = 0; lp < np[3]; lp++)
                {
                    float ak = env[p_exp[2] + kp];
                    float al = env[p_exp[3] + lp];
                    float q = ak + al;
                    float inv_q = 1.0f / q;
                    float Q[3] = {(ak * R[2][0] + al * R[3][0]) * inv_q,
                                  (ak * R[2][1] + al * R[3][1]) * inv_q,
                                  (ak * R[2][2] + al * R[3][2]) * inv_q};
                    float kcd = expf(-(ak * al * inv_q) * rcd2);

                    float pref = 2.0f * PI_25 / (p * q * sqrtf(p + q));
                    float n_abcd = n_ab * env[p_cof[2] + kp] *
                                   env[p_cof[3] + lp] * kcd * pref;
                    if (fabsf(n_abcd) < prim_screen_tol) continue;

                    float alpha = p * q / (p + q);
                    float PQ_val[3] = {(P[0] - Q[0]), (P[1] - Q[1]),
                                       (P[2] - Q[2])};
                    const int L_sum = l[0] + l[1] + l[2] + l[3];
                    float t_arg =
                        alpha * (PQ_val[0] * PQ_val[0] + PQ_val[1] * PQ_val[1] +
                                 PQ_val[2] * PQ_val[2]);
                    compute_hr_tensor(HR, alpha, PQ_val, L_sum, hr_base, t_arg);

                    float QC_val[3] = {(Q[0] - R[2][0]), (Q[1] - R[2][1]),
                                       (Q[2] - R[2][2])};
                    float QD_val[3] = {(Q[0] - R[3][0]), (Q[1] - R[3][1]),
                                       (Q[2] - R[3][2])};
                    for (int d = 0; d < 3; d++)
                        compute_md_coeffs(E_ket[d], l[2], l[3], QC_val[d],
                                          QD_val[d], 0.5f * inv_q);

                    for (int i = 0; i < dims_cart[0]; i++)
                    {
                        int ix = comp_x[0][i], iy = comp_y[0][i],
                            iz = comp_z[0][i];
                        for (int j = 0; j < dims_cart[1]; j++)
                        {
                            int jx = comp_x[1][j], jy = comp_y[1][j],
                                jz = comp_z[1][j];
                            for (int k = 0; k < dims_cart[2]; k++)
                            {
                                int kx = comp_x[2][k], ky = comp_y[2][k],
                                    kz = comp_z[2][k];
                                for (int l_idx = 0; l_idx < dims_cart[3];
                                     l_idx++)
                                {
                                    int lx_l = comp_x[3][l_idx],
                                        ly_l = comp_y[3][l_idx],
                                        lz_l = comp_z[3][l_idx];
                                    float val = 0.0f;
                                    for (int mux = 0; mux <= ix + jx; mux++)
                                    {
                                        auto ex = E_bra[0][ix][jx][mux];
                                        if (ex == 0.0f) continue;
                                        for (int muy = 0; muy <= iy + jy; muy++)
                                        {
                                            auto ey = E_bra[1][iy][jy][muy];
                                            if (ey == 0.0f) continue;
                                            for (int muz = 0; muz <= iz + jz;
                                                 muz++)
                                            {
                                                auto ez = E_bra[2][iz][jz][muz];
                                                auto e_bra_val = ex * ey * ez;
                                                if (e_bra_val == 0.0f) continue;
                                                for (int nux = 0;
                                                     nux <= kx + lx_l; nux++)
                                                {
                                                    auto dx =
                                                        E_ket[0][kx][lx_l][nux];
                                                    if (dx == 0.0f) continue;
                                                    for (int nuy = 0;
                                                         nuy <= ky + ly_l;
                                                         nuy++)
                                                    {
                                                        auto dy =
                                                            E_ket[1][ky][ly_l]
                                                                 [nuy];
                                                        if (dy == 0.0f)
                                                            continue;
                                                        for (int nuz = 0;
                                                             nuz <= kz + lz_l;
                                                             nuz++)
                                                        {
                                                            auto dz =
                                                                E_ket[2][kz]
                                                                     [lz_l]
                                                                     [nuz];
                                                            int tx = mux + nux;
                                                            int ty = muy + nuy;
                                                            int tz = muz + nuz;
                                                            float sign_val =
                                                                ((nux + nuy +
                                                                  nuz) %
                                                                     2 ==
                                                                 0)
                                                                    ? 1.0f
                                                                    : -1.0f;
                                                            val +=
                                                                e_bra_val * dx *
                                                                dy * dz *
                                                                HR[HR_IDX_RUNTIME(
                                                                    tx, ty, tz,
                                                                    0,
                                                                    hr_base)] *
                                                                sign_val;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    shell_eri[QC_Shell_Buffer_Index(
                                        i, j, k, l_idx, dims_cart[1],
                                        dims_cart[2], dims_cart[3])] +=
                                        val * n_abcd;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if (is_spherical)
    {
        QC_Cart2Sph_Shell_ERI(cart2sph_mat, nao_sph, off_cart, off_eff,
                              dims_cart, dims_sph, shell_eri, shell_tmp);
    }

    for (int i = 0; i < dims_eff[0]; i++)
    {
        const float ni = norms[off_eff[0] + i];
        for (int j = 0; j < dims_eff[1]; j++)
        {
            const float nj = norms[off_eff[1] + j];
            for (int k = 0; k < dims_eff[2]; k++)
            {
                const float nk = norms[off_eff[2] + k];
                for (int l_idx = 0; l_idx < dims_eff[3]; l_idx++)
                {
                    const float nl = norms[off_eff[3] + l_idx];
                    const int idx = QC_Shell_Buffer_Index(
                        i, j, k, l_idx, dims_eff[1], dims_eff[2], dims_eff[3]);
                    shell_eri[idx] *= ni * nj * nk * nl;
                }
            }
        }
    }
    return true;
}

static __global__ void QC_Build_Shell_Pair_Bounds_Kernel(
    const int n_pairs, const QC_ONE_E_TASK* shell_pairs, const int* atm,
    const int* bas, const float* env, const int* ao_offsets_cart,
    const int* ao_offsets_sph, const float* norms, const int is_spherical,
    const float* cart2sph_mat, const int nao_sph, float* bounds,
    float* global_hr_pool, int hr_base, int hr_size, int shell_buf_size,
    float prim_screen_tol)
{
    SIMPLE_DEVICE_FOR(task_id, n_pairs)
    {
#ifdef GPU_ARCH_NAME
        const int scratch_id = task_id;
#else
        const int scratch_id = omp_get_thread_num();
#endif
        float* task_pool =
            global_hr_pool + (int)scratch_id * (hr_size + 2 * shell_buf_size);
        float* HR = task_pool;
        float* shell_eri = task_pool + hr_size;
        float* shell_tmp = shell_eri + shell_buf_size;

        QC_ONE_E_TASK pair = shell_pairs[task_id];
        int sh[4] = {pair.x, pair.y, pair.x, pair.y};
        int dims_eff[4];
        int off_eff[4];
        if (!QC_Compute_Shell_Quartet_ERI_Buffer(
                sh, atm, bas, env, ao_offsets_cart, ao_offsets_sph, norms,
                is_spherical, cart2sph_mat, nao_sph, HR, shell_eri, shell_tmp,
                hr_base, shell_buf_size, prim_screen_tol, dims_eff, off_eff))
        {
            bounds[task_id] = 0.0f;
        }
        else
        {
            float max_diag = 0.0f;
            for (int i = 0; i < dims_eff[0]; i++)
                for (int j = 0; j < dims_eff[1]; j++)
                {
                    const float val = fabsf(shell_eri[QC_Shell_Buffer_Index(
                        i, j, i, j, dims_eff[1], dims_eff[2], dims_eff[3])]);
                    max_diag = fmaxf(max_diag, val);
                }
            bounds[task_id] = sqrtf(fmaxf(max_diag, 1e-30f));
        }
    }
}

static __global__ void QC_Build_Shell_Pair_Density_Kernel(
    const int n_pairs, const QC_ONE_E_TASK* shell_pairs,
    const int* ao_offsets_cart, const int* ao_offsets_sph, const int* l_list,
    const int is_spherical, const int nao, const float* P0, float* out0,
    const float* P1, float* out1, const float* P2, float* out2)
{
    SIMPLE_DEVICE_FOR(pair_id, n_pairs)
    {
        const QC_ONE_E_TASK pair = shell_pairs[pair_id];
        const int dim_i = QC_Shell_Dim(l_list[pair.x], is_spherical);
        const int dim_j = QC_Shell_Dim(l_list[pair.y], is_spherical);
        const int off_i =
            is_spherical ? ao_offsets_sph[pair.x] : ao_offsets_cart[pair.x];
        const int off_j =
            is_spherical ? ao_offsets_sph[pair.y] : ao_offsets_cart[pair.y];

        float max0 = 0.0f, max1 = 0.0f, max2 = 0.0f;
        for (int i = 0; i < dim_i; i++)
            for (int j = 0; j < dim_j; j++)
            {
                const int idx = (off_i + i) * nao + (off_j + j);
                if (P0 != NULL) max0 = fmaxf(max0, fabsf(P0[idx]));
                if (P1 != NULL) max1 = fmaxf(max1, fabsf(P1[idx]));
                if (P2 != NULL) max2 = fmaxf(max2, fabsf(P2[idx]));
            }

        if (out0 != NULL) out0[pair_id] = max0;
        if (out1 != NULL) out1[pair_id] = max1;
        if (out2 != NULL) out2[pair_id] = max2;
    }
}

#pragma once

#include "one_e.hpp"

static __device__ void compute_hr_tensor(float* HR, float alpha, float PQ[3],
                                  int L_tot, int hr_base, float t_arg)
{
    // Use double Boys + double seeding to avoid (-2α)^n amplification
    double F_d[17];
    compute_boys_double(F_d, t_arg, L_tot);
    double m2a = -2.0 * (double)alpha;
    double fac = 1.0;
    for (int n = 0; n <= L_tot; n++)
    {
        HR[HR_IDX_RUNTIME(0, 0, 0, n, hr_base)] = (float)(fac * F_d[n]);
        fac *= m2a;
    }
    for (int N = 1; N <= L_tot; N++)
    {
        for (int t = 0; t <= N; t++)
        {
            for (int u = 0; u <= N - t; u++)
            {
                int v = N - t - u;
                int max_n = L_tot - N;
                for (int n = 0; n <= max_n; n++)
                {
                    float val = 0.0f;
                    if (t > 0)
                    {
                        val = PQ[0] *
                              HR[HR_IDX_RUNTIME(t - 1, u, v, n + 1, hr_base)];
                        if (t > 1)
                            val +=
                                (float)(t - 1) *
                                HR[HR_IDX_RUNTIME(t - 2, u, v, n + 1, hr_base)];
                    }
                    else if (u > 0)
                    {
                        val = PQ[1] *
                              HR[HR_IDX_RUNTIME(t, u - 1, v, n + 1, hr_base)];
                        if (u > 1)
                            val +=
                                (float)(u - 1) *
                                HR[HR_IDX_RUNTIME(t, u - 2, v, n + 1, hr_base)];
                    }
                    else if (v > 0)
                    {
                        val = PQ[2] *
                              HR[HR_IDX_RUNTIME(t, u, v - 1, n + 1, hr_base)];
                        if (v > 1)
                            val +=
                                (float)(v - 1) *
                                HR[HR_IDX_RUNTIME(t, u, v - 2, n + 1, hr_base)];
                    }
                    HR[HR_IDX_RUNTIME(t, u, v, n, hr_base)] = val;
                }
            }
        }
    }
}

// Unified ERI Kernel
static __global__ void ERI_Kernel(const int n_tasks, const QC_ERI_TASK* tasks,
                                  const int* atm, const int* bas,
                                  const float* env, const int* ao_loc,
                                  float* out_eri, float* global_hr_pool,
                                  int nao_total, int hr_base, int hr_size,
                                  int shell_buf_size, float prim_screen_tol)
{
    SIMPLE_DEVICE_FOR(task_id, n_tasks)
    {
        float* task_pool =
            global_hr_pool + (int)task_id * (hr_size + shell_buf_size);
        float* HR = task_pool;
        float* shell_eri = task_pool + hr_size;

        QC_ERI_TASK t = tasks[task_id];
        int sh[4] = {t.x, t.y, t.z, t.w};

        int l[4];
        int np[4];
        int p_exp[4];
        int p_cof[4];
        float R[4][3];
        for (int i = 0; i < 4; i++)
        {
            l[i] = bas[sh[i] * 8 + 1];
            np[i] = bas[sh[i] * 8 + 2];
            p_exp[i] = bas[sh[i] * 8 + 5];
            p_cof[i] = bas[sh[i] * 8 + 6];
            int ptr_R = atm[bas[sh[i] * 8 + 0] * 6 + 1];
            R[i][0] = env[ptr_R + 0];
            R[i][1] = env[ptr_R + 1];
            R[i][2] = env[ptr_R + 2];
        }

        int dims[4];
        for (int i = 0; i < 4; i++) dims[i] = (l[i] + 1) * (l[i] + 2) / 2;
        const int shell_size = dims[0] * dims[1] * dims[2] * dims[3];
        if (shell_size <= shell_buf_size)
        {
            for (int i = 0; i < shell_size; i++) shell_eri[i] = 0.0f;

            int comp_x[4][MAX_CART_SHELL];
            int comp_y[4][MAX_CART_SHELL];
            int comp_z[4][MAX_CART_SHELL];
            for (int s = 0; s < 4; s++)
            {
                for (int c = 0; c < dims[s]; c++)
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
                        compute_md_coeffs(E_bra[d], l[0], l[1], PA_val[d],
                                          PB_val[d], 0.5f * inv_p);

                    for (int kp = 0; kp < np[2]; kp++)
                    {
                        for (int lp = 0; lp < np[3]; lp++)
                        {
                            float ak = env[p_exp[2] + kp];
                            float al = env[p_exp[3] + lp];
                            float q = ak + al;
                            float inv_q = 1.0f / q;
                            float Q[3] = {
                                (ak * R[2][0] + al * R[3][0]) * inv_q,
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
                            int L_sum = l[0] + l[1] + l[2] + l[3];
                            float t_arg = alpha * (PQ_val[0] * PQ_val[0] +
                                                   PQ_val[1] * PQ_val[1] +
                                                   PQ_val[2] * PQ_val[2]);
                            compute_hr_tensor(HR, alpha, PQ_val, L_sum,
                                              hr_base, t_arg);

                            float QC_val[3] = {(Q[0] - R[2][0]),
                                               (Q[1] - R[2][1]),
                                               (Q[2] - R[2][2])};
                            float QD_val[3] = {(Q[0] - R[3][0]),
                                               (Q[1] - R[3][1]),
                                               (Q[2] - R[3][2])};
                            for (int d = 0; d < 3; d++)
                                compute_md_coeffs(E_ket[d], l[2], l[3],
                                                  QC_val[d], QD_val[d],
                                                  0.5f * inv_q);

                            for (int i = 0; i < dims[0]; i++)
                            {
                                int ix = comp_x[0][i], iy = comp_y[0][i],
                                    iz = comp_z[0][i];
                                for (int j = 0; j < dims[1]; j++)
                                {
                                    int jx = comp_x[1][j], jy = comp_y[1][j],
                                        jz = comp_z[1][j];
                                    for (int k = 0; k < dims[2]; k++)
                                    {
                                        int kx = comp_x[2][k],
                                            ky = comp_y[2][k],
                                            kz = comp_z[2][k];
                                        for (int l_idx = 0; l_idx < dims[3];
                                             l_idx++)
                                        {
                                            int lx_l = comp_x[3][l_idx],
                                                ly_l = comp_y[3][l_idx],
                                                lz_l = comp_z[3][l_idx];
                                            float val = 0.0f;
                                            for (int mux = 0; mux <= ix + jx;
                                                 mux++)
                                            {
                                                auto ex = E_bra[0][ix][jx][mux];
                                                if (ex == 0.0f) continue;
                                                for (int muy = 0;
                                                     muy <= iy + jy; muy++)
                                                {
                                                    auto ey =
                                                        E_bra[1][iy][jy][muy];
                                                    if (ey == 0.0f) continue;
                                                    for (int muz = 0;
                                                         muz <= iz + jz; muz++)
                                                    {
                                                        auto ez =
                                                            E_bra[2][iz][jz]
                                                                 [muz];
                                                        auto e_bra_val =
                                                            ex * ey * ez;
                                                        if (e_bra_val == 0.0f)
                                                            continue;
                                                        for (int nux = 0;
                                                             nux <= kx + lx_l;
                                                             nux++)
                                                        {
                                                            auto dx =
                                                                E_ket[0][kx]
                                                                     [lx_l]
                                                                     [nux];
                                                            if (dx == 0.0f)
                                                                continue;
                                                            for (int nuy = 0;
                                                                 nuy <=
                                                                 ky + ly_l;
                                                                 nuy++)
                                                            {
                                                                auto dy =
                                                                    E_ket[1][ky]
                                                                         [ly_l]
                                                                         [nuy];
                                                                if (dy == 0.0f)
                                                                    continue;
                                                                for (int nuz =
                                                                         0;
                                                                     nuz <=
                                                                     kz + lz_l;
                                                                     nuz++)
                                                                {
                                                                    auto dz = E_ket
                                                                        [2][kz]
                                                                        [lz_l]
                                                                        [nuz];
                                                                    int tx =
                                                                        mux +
                                                                        nux;
                                                                    int ty =
                                                                        muy +
                                                                        nuy;
                                                                    int tz =
                                                                        muz +
                                                                        nuz;
                                                                    float sign_val =
                                                                        ((nux +
                                                                          nuy +
                                                                          nuz) %
                                                                             2 ==
                                                                         0)
                                                                            ? 1.0f
                                                                            : -1.0f;
                                                                    val +=
                                                                        e_bra_val *
                                                                        dx *
                                                                        dy *
                                                                        dz *
                                                                        HR[HR_IDX_RUNTIME(
                                                                            tx,
                                                                            ty,
                                                                            tz,
                                                                            0,
                                                                            hr_base)] *
                                                                        sign_val;
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                            shell_eri[((i * dims[1] + j) *
                                                           dims[2] +
                                                       k) *
                                                          dims[3] +
                                                      l_idx] += val * n_abcd;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            int off0 = ao_loc[sh[0]];
            int off1 = ao_loc[sh[1]];
            int off2 = ao_loc[sh[2]];
            int off3 = ao_loc[sh[3]];
            const int nao2 = (int)nao_total * nao_total;
            const int nao3 = nao2 * nao_total;
            for (int i = 0; i < dims[0]; i++)
            {
                for (int j = 0; j < dims[1]; j++)
                {
                    for (int k = 0; k < dims[2]; k++)
                    {
                        for (int l_idx = 0; l_idx < dims[3]; l_idx++)
                        {
                            float val =
                                shell_eri[((i * dims[1] + j) * dims[2] + k) *
                                              dims[3] +
                                          l_idx];
                            int p = off0 + i;
                            int q = off1 + j;
                            int r = off2 + k;
                            int s = off3 + l_idx;

                            // Expand shell-quartet symmetry to full AO tensor.
                            out_eri[(int)p * nao3 + (int)q * nao2 +
                                    (int)r * nao_total + s] = val;  // (pq|rs)
                            out_eri[(int)q * nao3 + (int)p * nao2 +
                                    (int)r * nao_total + s] = val;  // (qp|rs)
                            out_eri[(int)p * nao3 + (int)q * nao2 +
                                    (int)s * nao_total + r] = val;  // (pq|sr)
                            out_eri[(int)q * nao3 + (int)p * nao2 +
                                    (int)s * nao_total + r] = val;  // (qp|sr)
                            out_eri[(int)r * nao3 + (int)s * nao2 +
                                    (int)p * nao_total + q] = val;  // (rs|pq)
                            out_eri[(int)s * nao3 + (int)r * nao2 +
                                    (int)p * nao_total + q] = val;  // (sr|pq)
                            out_eri[(int)r * nao3 + (int)s * nao2 +
                                    (int)q * nao_total + p] = val;  // (rs|qp)
                            out_eri[(int)s * nao3 + (int)r * nao2 +
                                    (int)q * nao_total + p] = val;  // (sr|qp)
                        }
                    }
                }
            }
        }
    }
}

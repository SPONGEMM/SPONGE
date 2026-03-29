#pragma once

#include "../structure/matrix.h"
#include "dft.hpp"
#include "grid.hpp"
#include "xc.hpp"

static void QC_Cart2Sph_AO_Batch_Device(
    BLAS_HANDLE blas_handle, int n_batch, int nao_c, int nao_s,
    const float* d_cart2sph_mat, const float* d_ao_vals_c,
    const float* d_ao_gx_c, const float* d_ao_gy_c, const float* d_ao_gz_c,
    float* d_ao_vals_s, float* d_ao_gx_s, float* d_ao_gy_s, float* d_ao_gz_s)
{
    QC_MatMul_RowRow_Blas(blas_handle, n_batch, nao_s, nao_c, d_ao_vals_c,
                          d_cart2sph_mat, d_ao_vals_s);
    QC_MatMul_RowRow_Blas(blas_handle, n_batch, nao_s, nao_c, d_ao_gx_c,
                          d_cart2sph_mat, d_ao_gx_s);
    QC_MatMul_RowRow_Blas(blas_handle, n_batch, nao_s, nao_c, d_ao_gy_c,
                          d_cart2sph_mat, d_ao_gy_s);
    QC_MatMul_RowRow_Blas(blas_handle, n_batch, nao_s, nao_c, d_ao_gz_c,
                          d_cart2sph_mat, d_ao_gz_s);
}

// 对 AO 值施加归一化因子
static __global__ void QC_Apply_Norms_AO_Kernel(const int n_grid, const int nao,
                                                const float* norms,
                                                const float* ao_in,
                                                float* ao_out)
{
    const int total = n_grid * nao;
    SIMPLE_DEVICE_FOR(idx, total)
    {
        const int i = idx % nao;
        ao_out[idx] = ao_in[idx] * norms[i];
    }
}

// 从 Pao 和归一化 AO 计算 ρ (LDA) 或 ρ + σ + ∇ρ (GGA)
template <int deriv_level>
static __global__ void QC_Eval_Rho_Kernel(
    const int n_grid, const int nao, const float* ao_norm, const float* gx_norm,
    const float* gy_norm, const float* gz_norm, const float* Pao, double* rho,
    double* sigma, double* grad_rho_x, double* grad_rho_y, double* grad_rho_z)
{
    SIMPLE_DEVICE_FOR(ig, n_grid)
    {
        double r = 0.0;
        double gx = 0.0, gy = 0.0, gz = 0.0;
        for (int i = 0; i < nao; i++)
        {
            const double pao_i = (double)Pao[i * n_grid + ig];
            r += (double)ao_norm[ig * nao + i] * pao_i;
            if (deriv_level >= 1)
            {
                gx += (double)gx_norm[ig * nao + i] * pao_i;
                gy += (double)gy_norm[ig * nao + i] * pao_i;
                gz += (double)gz_norm[ig * nao + i] * pao_i;
            }
        }
        rho[ig] = r;
        if (deriv_level >= 1)
        {
            gx *= 2.0;
            gy *= 2.0;
            gz *= 2.0;
            sigma[ig] = gx * gx + gy * gy + gz * gz;
            grad_rho_x[ig] = gx;
            grad_rho_y[ig] = gy;
            grad_rho_z[ig] = gz;
        }
    }
}

// 构建加权 AO
// deriv_level=0: W_full = w * vrho * φ_i (LDA, Vxc = W^T @ AO 天然对称)
// deriv_level=1: W_full = w * (vrho*φ_i + 2*vsigma*∇ρ·∇φ_i)
//                W_sigma = w * 2*vsigma*∇ρ·∇φ_i
//                Vxc = W_full^T @ AO + AO^T @ W_sigma
template <int deriv_level>
static __global__ void QC_Build_Weighted_AO_Kernel(
    const int n_grid, const int nao, const float* ao_norm, const float* gx_norm,
    const float* gy_norm, const float* gz_norm, const float* grid_weights,
    const double* rho, const double* exc, const double* vrho,
    const double* vsigma, const double* grad_rho_x, const double* grad_rho_y,
    const double* grad_rho_z, float* W_full, float* W_sigma, double* exc_total)
{
    SIMPLE_DEVICE_FOR(ig, n_grid)
    {
        if (rho[ig] < 1e-10)
        {
            for (int i = 0; i < nao; i++)
            {
                W_full[ig * nao + i] = 0.0f;
                if (deriv_level >= 1) W_sigma[ig * nao + i] = 0.0f;
            }
        }
        else
        {
            const float w = grid_weights[ig];
            atomicAdd(exc_total, (double)w * exc[ig]);
            const double v_rho = vrho[ig];

            if (deriv_level >= 1)
            {
                const double v_sigma = vsigma[ig];
                const double grx = grad_rho_x[ig];
                const double gry = grad_rho_y[ig];
                const double grz = grad_rho_z[ig];
                for (int i = 0; i < nao; i++)
                {
                    const double ai = (double)ao_norm[ig * nao + i];
                    const double sp = 2.0 * v_sigma *
                                      (grx * (double)gx_norm[ig * nao + i] +
                                       gry * (double)gy_norm[ig * nao + i] +
                                       grz * (double)gz_norm[ig * nao + i]);
                    W_full[ig * nao + i] =
                        (float)((double)w * (v_rho * ai + sp));
                    W_sigma[ig * nao + i] = (float)((double)w * sp);
                }
            }
            else
            {
                for (int i = 0; i < nao; i++)
                    W_full[ig * nao + i] =
                        (float)((double)w * v_rho *
                                (double)ao_norm[ig * nao + i]);
            }
        }
    }
}

template <int deriv_level>
static void QC_Build_DFT_VXC_Impl(
    BLAS_HANDLE blas_handle, QC_METHOD method, int is_spherical, int nao_c,
    int nao_s, int total_grid_size, int grid_batch_size, int nbas,
    const float* d_grid_coords, const float* d_grid_weights,
    const float* d_cart2sph_mat, const VECTOR* d_centers, const int* d_l_list,
    const float* d_exps, const float* d_coeffs, const int* d_shell_offsets,
    const int* d_shell_sizes, const int* d_ao_offsets, const float* d_norms,
    const float* d_P, float* d_ao_vals_cart, float* d_ao_grad_x_cart,
    float* d_ao_grad_y_cart, float* d_ao_grad_z_cart, float* d_ao_vals,
    float* d_ao_grad_x, float* d_ao_grad_y, float* d_ao_grad_z, double* d_rho,
    double* d_sigma, double* d_exc, double* d_vrho, double* d_vsigma,
    double* d_exc_total, float* d_Vxc, float* d_ao_norm, float* d_gx_norm,
    float* d_gy_norm, float* d_gz_norm, float* d_Pao, float* d_W_full,
    float* d_W_sigma, double* d_grad_rho_x, double* d_grad_rho_y,
    double* d_grad_rho_z, const float* d_shell_r2_screen)
{
    const int nao = nao_s;
    const int nao2 = nao * nao;
    deviceMemset(d_Vxc, 0, sizeof(float) * nao2);
    deviceMemset(d_exc_total, 0, sizeof(double));
    if (total_grid_size <= 0) return;

    const int batch_size = std::max(1, grid_batch_size);
    const int threads = 128;

    for (int g0 = 0; g0 < total_grid_size; g0 += batch_size)
    {
        const int n_batch = std::min(batch_size, total_grid_size - g0);
        const float* d_coords_batch = d_grid_coords + g0 * 3;
        const float* d_weights_batch = d_grid_weights + g0;
        const int total_ao = n_batch * nao;

        // 1. AO 求值 + Cart2Sph + 归一化
        {
            float* d_vals_use = d_ao_vals;
            float* d_gx_use = d_ao_grad_x;
            float* d_gy_use = d_ao_grad_y;
            float* d_gz_use = d_ao_grad_z;
            int nao_eval = nao_s;
            if (is_spherical)
            {
                d_vals_use = d_ao_vals_cart;
                if (deriv_level >= 1)
                {
                    d_gx_use = d_ao_grad_x_cart;
                    d_gy_use = d_ao_grad_y_cart;
                    d_gz_use = d_ao_grad_z_cart;
                }
                nao_eval = nao_c;
            }
            Launch_Device_Kernel(
                (QC_Eval_AO_Grid_Kernel<deriv_level>),
                (n_batch + threads - 1) / threads, threads, 0, 0, n_batch,
                d_coords_batch, nao_eval, nbas, d_centers, d_l_list, d_exps,
                d_coeffs, d_shell_offsets, d_shell_sizes, d_ao_offsets,
                d_shell_r2_screen, d_vals_use, d_gx_use, d_gy_use, d_gz_use);
            if (is_spherical)
            {
                QC_MatMul_RowRow_Blas(blas_handle, n_batch, nao_s, nao_c,
                                      d_ao_vals_cart, d_cart2sph_mat,
                                      d_ao_vals);
                if (deriv_level >= 1)
                {
                    QC_MatMul_RowRow_Blas(blas_handle, n_batch, nao_s, nao_c,
                                          d_ao_grad_x_cart, d_cart2sph_mat,
                                          d_ao_grad_x);
                    QC_MatMul_RowRow_Blas(blas_handle, n_batch, nao_s, nao_c,
                                          d_ao_grad_y_cart, d_cart2sph_mat,
                                          d_ao_grad_y);
                    QC_MatMul_RowRow_Blas(blas_handle, n_batch, nao_s, nao_c,
                                          d_ao_grad_z_cart, d_cart2sph_mat,
                                          d_ao_grad_z);
                }
            }
            Launch_Device_Kernel(
                QC_Apply_Norms_AO_Kernel, (total_ao + threads - 1) / threads,
                threads, 0, 0, n_batch, nao, d_norms, d_ao_vals, d_ao_norm);
            if (deriv_level >= 1)
            {
                Launch_Device_Kernel(QC_Apply_Norms_AO_Kernel,
                                     (total_ao + threads - 1) / threads,
                                     threads, 0, 0, n_batch, nao, d_norms,
                                     d_ao_grad_x, d_gx_norm);
                Launch_Device_Kernel(QC_Apply_Norms_AO_Kernel,
                                     (total_ao + threads - 1) / threads,
                                     threads, 0, 0, n_batch, nao, d_norms,
                                     d_ao_grad_y, d_gy_norm);
                Launch_Device_Kernel(QC_Apply_Norms_AO_Kernel,
                                     (total_ao + threads - 1) / threads,
                                     threads, 0, 0, n_batch, nao, d_norms,
                                     d_ao_grad_z, d_gz_norm);
            }
        }

        // 2. Pao = P @ AO_norm^T
        {
            const float one = 1.0f, zero = 0.0f;
            deviceBlasSgemm(blas_handle, DEVICE_BLAS_OP_T, DEVICE_BLAS_OP_N,
                            n_batch, nao, nao, &one, d_ao_norm, nao, d_P, nao,
                            &zero, d_Pao, n_batch);
        }

        // 3. ρ (+ σ, ∇ρ for GGA)
        Launch_Device_Kernel((QC_Eval_Rho_Kernel<deriv_level>),
                             (n_batch + threads - 1) / threads, threads, 0, 0,
                             n_batch, nao, d_ao_norm, d_gx_norm, d_gy_norm,
                             d_gz_norm, d_Pao, d_rho, d_sigma, d_grad_rho_x,
                             d_grad_rho_y, d_grad_rho_z);

        // 4. XC 泛函求值
        Launch_Device_Kernel(QC_Eval_XC_Derivs_Kernel,
                             (n_batch + threads - 1) / threads, threads, 0, 0,
                             n_batch, (int)method, d_rho, d_sigma, d_exc,
                             d_vrho, d_vsigma);

        // 5. 加权 AO
        Launch_Device_Kernel((QC_Build_Weighted_AO_Kernel<deriv_level>),
                             (n_batch + threads - 1) / threads, threads, 0, 0,
                             n_batch, nao, d_ao_norm, d_gx_norm, d_gy_norm,
                             d_gz_norm, d_weights_batch, d_rho, d_exc, d_vrho,
                             d_vsigma, d_grad_rho_x, d_grad_rho_y, d_grad_rho_z,
                             d_W_full, d_W_sigma, d_exc_total);

        // 6. Vxc 矩阵累加
        {
            const float one = 1.0f;
            deviceBlasSgemm(blas_handle, DEVICE_BLAS_OP_N, DEVICE_BLAS_OP_T,
                            nao, nao, n_batch, &one, d_ao_norm, nao, d_W_full,
                            nao, &one, d_Vxc, nao);
            if (deriv_level >= 1)
            {
                // GGA: 补全 sigma 项另一侧
                deviceBlasSgemm(blas_handle, DEVICE_BLAS_OP_N, DEVICE_BLAS_OP_T,
                                nao, nao, n_batch, &one, d_W_sigma, nao,
                                d_ao_norm, nao, &one, d_Vxc, nao);
            }
        }
    }
}

// 外部接口：根据方法自动分派到对应模板实例
static void QC_Build_DFT_VXC(
    BLAS_HANDLE blas_handle, QC_METHOD method, int is_spherical, int nao_c,
    int nao_s, int total_grid_size, int grid_batch_size, int nbas,
    const float* d_grid_coords, const float* d_grid_weights,
    const float* d_cart2sph_mat, const VECTOR* d_centers, const int* d_l_list,
    const float* d_exps, const float* d_coeffs, const int* d_shell_offsets,
    const int* d_shell_sizes, const int* d_ao_offsets, const float* d_norms,
    const float* d_P, float* d_ao_vals_cart, float* d_ao_grad_x_cart,
    float* d_ao_grad_y_cart, float* d_ao_grad_z_cart, float* d_ao_vals,
    float* d_ao_grad_x, float* d_ao_grad_y, float* d_ao_grad_z, double* d_rho,
    double* d_sigma, double* d_exc, double* d_vrho, double* d_vsigma,
    double* d_exc_total, float* d_Vxc, float* d_ao_norm, float* d_gx_norm,
    float* d_gy_norm, float* d_gz_norm, float* d_Pao, float* d_W_full,
    float* d_W_sigma, double* d_grad_rho_x, double* d_grad_rho_y,
    double* d_grad_rho_z, const float* d_shell_r2_screen)
{
    if (method == QC_METHOD::LDA)
    {
        QC_Build_DFT_VXC_Impl<0>(
            blas_handle, method, is_spherical, nao_c, nao_s, total_grid_size,
            grid_batch_size, nbas, d_grid_coords, d_grid_weights,
            d_cart2sph_mat, d_centers, d_l_list, d_exps, d_coeffs,
            d_shell_offsets, d_shell_sizes, d_ao_offsets, d_norms, d_P,
            d_ao_vals_cart, d_ao_grad_x_cart, d_ao_grad_y_cart,
            d_ao_grad_z_cart, d_ao_vals, d_ao_grad_x, d_ao_grad_y, d_ao_grad_z,
            d_rho, d_sigma, d_exc, d_vrho, d_vsigma, d_exc_total, d_Vxc,
            d_ao_norm, d_gx_norm, d_gy_norm, d_gz_norm, d_Pao, d_W_full,
            d_W_sigma, d_grad_rho_x, d_grad_rho_y, d_grad_rho_z,
            d_shell_r2_screen);
    }
    else
    {
        QC_Build_DFT_VXC_Impl<1>(
            blas_handle, method, is_spherical, nao_c, nao_s, total_grid_size,
            grid_batch_size, nbas, d_grid_coords, d_grid_weights,
            d_cart2sph_mat, d_centers, d_l_list, d_exps, d_coeffs,
            d_shell_offsets, d_shell_sizes, d_ao_offsets, d_norms, d_P,
            d_ao_vals_cart, d_ao_grad_x_cart, d_ao_grad_y_cart,
            d_ao_grad_z_cart, d_ao_vals, d_ao_grad_x, d_ao_grad_y, d_ao_grad_z,
            d_rho, d_sigma, d_exc, d_vrho, d_vsigma, d_exc_total, d_Vxc,
            d_ao_norm, d_gx_norm, d_gy_norm, d_gz_norm, d_Pao, d_W_full,
            d_W_sigma, d_grad_rho_x, d_grad_rho_y, d_grad_rho_z,
            d_shell_r2_screen);
    }
}

// ====================== UKS BLAS 优化 kernel ======================

// UKS: 从 Pao_a/Pao_b 计算自旋密度和梯度
template <int deriv_level>
static __global__ void QC_Eval_Rho_UKS_Kernel(
    const int n_grid, const int nao, const float* ao_norm, const float* gx_norm,
    const float* gy_norm, const float* gz_norm, const float* Pao_a,
    const float* Pao_b, double* rho_a, double* rho_b, double* sigma_aa,
    double* sigma_ab, double* sigma_bb, double* gra_x, double* gra_y,
    double* gra_z, double* grb_x, double* grb_y, double* grb_z)
{
    SIMPLE_DEVICE_FOR(ig, n_grid)
    {
        double ra = 0.0, rb = 0.0;
        double gax = 0.0, gay = 0.0, gaz = 0.0;
        double gbx = 0.0, gby = 0.0, gbz = 0.0;
        for (int i = 0; i < nao; i++)
        {
            const double pa = (double)Pao_a[i * n_grid + ig];
            const double pb = (double)Pao_b[i * n_grid + ig];
            const double ai = (double)ao_norm[ig * nao + i];
            ra += ai * pa;
            rb += ai * pb;
            if (deriv_level >= 1)
            {
                const double gxi = (double)gx_norm[ig * nao + i];
                const double gyi = (double)gy_norm[ig * nao + i];
                const double gzi = (double)gz_norm[ig * nao + i];
                gax += gxi * pa;
                gay += gyi * pa;
                gaz += gzi * pa;
                gbx += gxi * pb;
                gby += gyi * pb;
                gbz += gzi * pb;
            }
        }
        rho_a[ig] = ra;
        rho_b[ig] = rb;
        if (deriv_level >= 1)
        {
            gax *= 2.0;
            gay *= 2.0;
            gaz *= 2.0;
            gbx *= 2.0;
            gby *= 2.0;
            gbz *= 2.0;
            sigma_aa[ig] = gax * gax + gay * gay + gaz * gaz;
            sigma_ab[ig] = gax * gbx + gay * gby + gaz * gbz;
            sigma_bb[ig] = gbx * gbx + gby * gby + gbz * gbz;
            gra_x[ig] = gax;
            gra_y[ig] = gay;
            gra_z[ig] = gaz;
            grb_x[ig] = gbx;
            grb_y[ig] = gby;
            grb_z[ig] = gbz;
        }
    }
}

// UKS: XC 泛函求值
static __global__ void QC_Eval_XC_UKS_Kernel(
    const int n_grid, const int method_id, const double* rho_a,
    const double* rho_b, const double* sigma_aa, const double* sigma_ab,
    const double* sigma_bb, double* exc, double* v_rho_a, double* v_rho_b,
    double* v_sigma_aa, double* v_sigma_ab, double* v_sigma_bb)
{
    SIMPLE_DEVICE_FOR(ig, n_grid)
    {
        if (rho_a[ig] + rho_b[ig] < 1e-10)
        {
            exc[ig] = 0.0;
            v_rho_a[ig] = v_rho_b[ig] = 0.0;
            v_sigma_aa[ig] = v_sigma_ab[ig] = v_sigma_bb[ig] = 0.0;
        }
        else
        {
            double e = 0.0, vra = 0.0, vrb = 0.0;
            double vsaa = 0.0, vsab = 0.0, vsbb = 0.0;
            QC_Local_UKS_Derivs_FD((QC_METHOD)method_id, rho_a[ig], rho_b[ig],
                                   sigma_aa[ig], sigma_ab[ig], sigma_bb[ig], e,
                                   vra, vrb, vsaa, vsab, vsbb);
            exc[ig] = e;
            v_rho_a[ig] = vra;
            v_rho_b[ig] = vrb;
            v_sigma_aa[ig] = vsaa;
            v_sigma_ab[ig] = vsab;
            v_sigma_bb[ig] = vsbb;
        }
    }
}

// UKS: 构建加权 AO（alpha 和 beta 各一套）
template <int deriv_level>
static __global__ void QC_Build_Weighted_AO_UKS_Kernel(
    const int n_grid, const int nao, const float* ao_norm, const float* gx_norm,
    const float* gy_norm, const float* gz_norm, const float* grid_weights,
    const double* rho_a, const double* rho_b, const double* exc,
    const double* v_rho_a, const double* v_rho_b, const double* v_sigma_aa,
    const double* v_sigma_ab, const double* v_sigma_bb, const double* gra_x,
    const double* gra_y, const double* gra_z, const double* grb_x,
    const double* grb_y, const double* grb_z, float* Wa_full, float* Wa_sigma,
    float* Wb_full, float* Wb_sigma, double* exc_total)
{
    SIMPLE_DEVICE_FOR(ig, n_grid)
    {
        if (rho_a[ig] + rho_b[ig] < 1e-10)
        {
            for (int i = 0; i < nao; i++)
            {
                Wa_full[ig * nao + i] = 0.0f;
                Wb_full[ig * nao + i] = 0.0f;
                if (deriv_level >= 1)
                {
                    Wa_sigma[ig * nao + i] = 0.0f;
                    Wb_sigma[ig * nao + i] = 0.0f;
                }
            }
        }
        else
        {
            const float w = grid_weights[ig];
            atomicAdd(exc_total, (double)w * exc[ig]);

            if (deriv_level >= 1)
            {
                const double gax = 2.0 * v_sigma_aa[ig] * gra_x[ig] +
                                   v_sigma_ab[ig] * grb_x[ig];
                const double gay = 2.0 * v_sigma_aa[ig] * gra_y[ig] +
                                   v_sigma_ab[ig] * grb_y[ig];
                const double gaz = 2.0 * v_sigma_aa[ig] * gra_z[ig] +
                                   v_sigma_ab[ig] * grb_z[ig];
                const double gbx = 2.0 * v_sigma_bb[ig] * grb_x[ig] +
                                   v_sigma_ab[ig] * gra_x[ig];
                const double gby = 2.0 * v_sigma_bb[ig] * grb_y[ig] +
                                   v_sigma_ab[ig] * gra_y[ig];
                const double gbz = 2.0 * v_sigma_bb[ig] * grb_z[ig] +
                                   v_sigma_ab[ig] * gra_z[ig];
                for (int i = 0; i < nao; i++)
                {
                    const double ai = (double)ao_norm[ig * nao + i];
                    const double gxi = (double)gx_norm[ig * nao + i];
                    const double gyi = (double)gy_norm[ig * nao + i];
                    const double gzi = (double)gz_norm[ig * nao + i];
                    const double spa = gax * gxi + gay * gyi + gaz * gzi;
                    const double spb = gbx * gxi + gby * gyi + gbz * gzi;
                    Wa_full[ig * nao + i] =
                        (float)((double)w * (v_rho_a[ig] * ai + spa));
                    Wa_sigma[ig * nao + i] = (float)((double)w * spa);
                    Wb_full[ig * nao + i] =
                        (float)((double)w * (v_rho_b[ig] * ai + spb));
                    Wb_sigma[ig * nao + i] = (float)((double)w * spb);
                }
            }
            else
            {
                for (int i = 0; i < nao; i++)
                {
                    const double ai = (double)ao_norm[ig * nao + i];
                    Wa_full[ig * nao + i] =
                        (float)((double)w * v_rho_a[ig] * ai);
                    Wb_full[ig * nao + i] =
                        (float)((double)w * v_rho_b[ig] * ai);
                }
            }
        }
    }
}

// UKS BLAS 优化版本
static void QC_Build_DFT_VXC_UKS(
    BLAS_HANDLE blas_handle, QC_METHOD method, int is_spherical, int nao_c,
    int nao_s, int total_grid_size, int grid_batch_size, int nbas,
    const float* d_grid_coords, const float* d_grid_weights,
    const float* d_cart2sph_mat, const VECTOR* d_centers, const int* d_l_list,
    const float* d_exps, const float* d_coeffs, const int* d_shell_offsets,
    const int* d_shell_sizes, const int* d_ao_offsets, const float* d_norms,
    const float* d_Pa, const float* d_Pb, float* d_ao_vals_cart,
    float* d_ao_grad_x_cart, float* d_ao_grad_y_cart, float* d_ao_grad_z_cart,
    float* d_ao_vals, float* d_ao_grad_x, float* d_ao_grad_y,
    float* d_ao_grad_z, double* d_exc_total, float* d_Vxc_a, float* d_Vxc_b,
    const float* d_shell_r2_screen,
    // 预分配缓冲
    float* d_ao_norm, float* d_gx_norm, float* d_gy_norm, float* d_gz_norm,
    float* d_Pao, float* d_W_full, float* d_W_sigma, double* d_grad_rho_x,
    double* d_grad_rho_y, double* d_grad_rho_z,
    // UKS 预分配缓冲
    float* d_Pao_b, double* d_rho_a, double* d_rho_b, double* d_sigma_aa,
    double* d_sigma_ab, double* d_sigma_bb, double* d_grb_x, double* d_grb_y,
    double* d_grb_z, double* d_exc, double* d_vra, double* d_vrb,
    double* d_vsaa, double* d_vsab, double* d_vsbb, float* d_Wb_full,
    float* d_Wb_sigma)
{
    const int nao = nao_s;
    const int nao2 = nao * nao;
    deviceMemset(d_Vxc_a, 0, sizeof(float) * nao2);
    deviceMemset(d_Vxc_b, 0, sizeof(float) * nao2);
    deviceMemset(d_exc_total, 0, sizeof(double));
    if (total_grid_size <= 0) return;

    const int batch_size = std::max(1, grid_batch_size);
    const int threads = 128;
    const bool is_gga = (method != QC_METHOD::LDA);

    for (int g0 = 0; g0 < total_grid_size; g0 += batch_size)
    {
        const int n_batch = std::min(batch_size, total_grid_size - g0);
        const float* d_coords_batch = d_grid_coords + g0 * 3;
        const float* d_weights_batch = d_grid_weights + g0;
        const int total_ao = n_batch * nao;

        // 1. AO 求值 + Cart2Sph + 归一化（复用 RKS 逻辑）
        {
            float* d_vals_use = d_ao_vals;
            float* d_gx_use = d_ao_grad_x;
            float* d_gy_use = d_ao_grad_y;
            float* d_gz_use = d_ao_grad_z;
            int nao_eval = nao_s;
            if (is_spherical)
            {
                d_vals_use = d_ao_vals_cart;
                if (is_gga)
                {
                    d_gx_use = d_ao_grad_x_cart;
                    d_gy_use = d_ao_grad_y_cart;
                    d_gz_use = d_ao_grad_z_cart;
                }
                nao_eval = nao_c;
            }
            if (is_gga)
            {
                Launch_Device_Kernel((QC_Eval_AO_Grid_Kernel<1>),
                                     (n_batch + threads - 1) / threads, threads,
                                     0, 0, n_batch, d_coords_batch, nao_eval,
                                     nbas, d_centers, d_l_list, d_exps,
                                     d_coeffs, d_shell_offsets, d_shell_sizes,
                                     d_ao_offsets, d_shell_r2_screen,
                                     d_vals_use, d_gx_use, d_gy_use, d_gz_use);
            }
            else
            {
                Launch_Device_Kernel((QC_Eval_AO_Grid_Kernel<0>),
                                     (n_batch + threads - 1) / threads, threads,
                                     0, 0, n_batch, d_coords_batch, nao_eval,
                                     nbas, d_centers, d_l_list, d_exps,
                                     d_coeffs, d_shell_offsets, d_shell_sizes,
                                     d_ao_offsets, d_shell_r2_screen,
                                     d_vals_use, d_gx_use, d_gy_use, d_gz_use);
            }
            if (is_spherical)
            {
                QC_MatMul_RowRow_Blas(blas_handle, n_batch, nao_s, nao_c,
                                      d_ao_vals_cart, d_cart2sph_mat,
                                      d_ao_vals);
                if (is_gga)
                {
                    QC_MatMul_RowRow_Blas(blas_handle, n_batch, nao_s, nao_c,
                                          d_ao_grad_x_cart, d_cart2sph_mat,
                                          d_ao_grad_x);
                    QC_MatMul_RowRow_Blas(blas_handle, n_batch, nao_s, nao_c,
                                          d_ao_grad_y_cart, d_cart2sph_mat,
                                          d_ao_grad_y);
                    QC_MatMul_RowRow_Blas(blas_handle, n_batch, nao_s, nao_c,
                                          d_ao_grad_z_cart, d_cart2sph_mat,
                                          d_ao_grad_z);
                }
            }
            Launch_Device_Kernel(
                QC_Apply_Norms_AO_Kernel, (total_ao + threads - 1) / threads,
                threads, 0, 0, n_batch, nao, d_norms, d_ao_vals, d_ao_norm);
            if (is_gga)
            {
                Launch_Device_Kernel(QC_Apply_Norms_AO_Kernel,
                                     (total_ao + threads - 1) / threads,
                                     threads, 0, 0, n_batch, nao, d_norms,
                                     d_ao_grad_x, d_gx_norm);
                Launch_Device_Kernel(QC_Apply_Norms_AO_Kernel,
                                     (total_ao + threads - 1) / threads,
                                     threads, 0, 0, n_batch, nao, d_norms,
                                     d_ao_grad_y, d_gy_norm);
                Launch_Device_Kernel(QC_Apply_Norms_AO_Kernel,
                                     (total_ao + threads - 1) / threads,
                                     threads, 0, 0, n_batch, nao, d_norms,
                                     d_ao_grad_z, d_gz_norm);
            }
        }

        // 2. Pao_a = Pa @ AO^T, Pao_b = Pb @ AO^T
        {
            const float one = 1.0f, zero = 0.0f;
            deviceBlasSgemm(blas_handle, DEVICE_BLAS_OP_T, DEVICE_BLAS_OP_N,
                            n_batch, nao, nao, &one, d_ao_norm, nao, d_Pa, nao,
                            &zero, d_Pao, n_batch);
            deviceBlasSgemm(blas_handle, DEVICE_BLAS_OP_T, DEVICE_BLAS_OP_N,
                            n_batch, nao, nao, &one, d_ao_norm, nao, d_Pb, nao,
                            &zero, d_Pao_b, n_batch);
        }

        // 3. UKS 密度
        if (is_gga)
        {
            Launch_Device_Kernel(
                (QC_Eval_Rho_UKS_Kernel<1>), (n_batch + threads - 1) / threads,
                threads, 0, 0, n_batch, nao, d_ao_norm, d_gx_norm, d_gy_norm,
                d_gz_norm, d_Pao, d_Pao_b, d_rho_a, d_rho_b, d_sigma_aa,
                d_sigma_ab, d_sigma_bb, d_grad_rho_x, d_grad_rho_y,
                d_grad_rho_z, d_grb_x, d_grb_y, d_grb_z);
        }
        else
        {
            Launch_Device_Kernel(
                (QC_Eval_Rho_UKS_Kernel<0>), (n_batch + threads - 1) / threads,
                threads, 0, 0, n_batch, nao, d_ao_norm, d_gx_norm, d_gy_norm,
                d_gz_norm, d_Pao, d_Pao_b, d_rho_a, d_rho_b, d_sigma_aa,
                d_sigma_ab, d_sigma_bb, d_grad_rho_x, d_grad_rho_y,
                d_grad_rho_z, d_grb_x, d_grb_y, d_grb_z);
        }

        // 4. UKS XC 泛函
        Launch_Device_Kernel(QC_Eval_XC_UKS_Kernel,
                             (n_batch + threads - 1) / threads, threads, 0, 0,
                             n_batch, (int)method, d_rho_a, d_rho_b, d_sigma_aa,
                             d_sigma_ab, d_sigma_bb, d_exc, d_vra, d_vrb,
                             d_vsaa, d_vsab, d_vsbb);

        // 5. 加权 AO
        if (is_gga)
        {
            Launch_Device_Kernel(
                (QC_Build_Weighted_AO_UKS_Kernel<1>),
                (n_batch + threads - 1) / threads, threads, 0, 0, n_batch, nao,
                d_ao_norm, d_gx_norm, d_gy_norm, d_gz_norm, d_weights_batch,
                d_rho_a, d_rho_b, d_exc, d_vra, d_vrb, d_vsaa, d_vsab, d_vsbb,
                d_grad_rho_x, d_grad_rho_y, d_grad_rho_z, d_grb_x, d_grb_y,
                d_grb_z, d_W_full, d_W_sigma, d_Wb_full, d_Wb_sigma,
                d_exc_total);
        }
        else
        {
            Launch_Device_Kernel(
                (QC_Build_Weighted_AO_UKS_Kernel<0>),
                (n_batch + threads - 1) / threads, threads, 0, 0, n_batch, nao,
                d_ao_norm, d_gx_norm, d_gy_norm, d_gz_norm, d_weights_batch,
                d_rho_a, d_rho_b, d_exc, d_vra, d_vrb, d_vsaa, d_vsab, d_vsbb,
                d_grad_rho_x, d_grad_rho_y, d_grad_rho_z, d_grb_x, d_grb_y,
                d_grb_z, d_W_full, d_W_sigma, d_Wb_full, d_Wb_sigma,
                d_exc_total);
        }

        // 6. Vxc_a, Vxc_b 矩阵累加
        {
            const float one = 1.0f;
            deviceBlasSgemm(blas_handle, DEVICE_BLAS_OP_N, DEVICE_BLAS_OP_T,
                            nao, nao, n_batch, &one, d_ao_norm, nao, d_W_full,
                            nao, &one, d_Vxc_a, nao);
            deviceBlasSgemm(blas_handle, DEVICE_BLAS_OP_N, DEVICE_BLAS_OP_T,
                            nao, nao, n_batch, &one, d_ao_norm, nao, d_Wb_full,
                            nao, &one, d_Vxc_b, nao);
            if (is_gga)
            {
                deviceBlasSgemm(blas_handle, DEVICE_BLAS_OP_N, DEVICE_BLAS_OP_T,
                                nao, nao, n_batch, &one, d_W_sigma, nao,
                                d_ao_norm, nao, &one, d_Vxc_a, nao);
                deviceBlasSgemm(blas_handle, DEVICE_BLAS_OP_N, DEVICE_BLAS_OP_T,
                                nao, nao, n_batch, &one, d_Wb_sigma, nao,
                                d_ao_norm, nao, &one, d_Vxc_b, nao);
            }
        }
    }
}

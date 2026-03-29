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

// 对 AO 值施加归一化因子: ao_norm[ig * nao + i] = ao[ig * nao + i] * norms[i]
static __global__ void QC_Apply_Norms_AO_Kernel(const int n_grid,
                                                const int nao,
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

// 从 Pao 和归一化 AO 计算 ρ, σ, ∇ρ
// Pao[i * n_grid + ig] = Σ_j P[i,j] * φ_j^norm(ig)
// ρ(ig) = Σ_i φ_i^norm(ig) * Pao_i(ig)
// ∇ρ_x(ig) = 2 * Σ_i ∇φ_i^norm_x(ig) * Pao_i(ig)
static __global__ void QC_Eval_Rho_Sigma_BLAS_Kernel(
    const int n_grid, const int nao, const float* ao_norm,
    const float* gx_norm, const float* gy_norm, const float* gz_norm,
    const float* Pao, double* rho, double* sigma, double* grad_rho_x,
    double* grad_rho_y, double* grad_rho_z)
{
    SIMPLE_DEVICE_FOR(ig, n_grid)
    {
        double r = 0.0, gx = 0.0, gy = 0.0, gz = 0.0;
        for (int i = 0; i < nao; i++)
        {
            const double pao_i = (double)Pao[i * n_grid + ig];
            r += (double)ao_norm[ig * nao + i] * pao_i;
            gx += (double)gx_norm[ig * nao + i] * pao_i;
            gy += (double)gy_norm[ig * nao + i] * pao_i;
            gz += (double)gz_norm[ig * nao + i] * pao_i;
        }
        gx *= 2.0;
        gy *= 2.0;
        gz *= 2.0;
        rho[ig] = r;
        sigma[ig] = gx * gx + gy * gy + gz * gz;
        grad_rho_x[ig] = gx;
        grad_rho_y[ig] = gy;
        grad_rho_z[ig] = gz;
    }
}

// 构建加权 AO 矩阵:
//   W_full[ig,i] = w * (vrho*φ_i + 2*vsigma*(∇ρ·∇φ_i))
//   W_sigma[ig,i] = w * 2*vsigma*(∇ρ·∇φ_i)
// Vxc = W_full^T @ AO + AO^T @ W_sigma (精确对称，无需近似对称化)
static __global__ void QC_Build_Weighted_AO_Kernel(
    const int n_grid, const int nao, const float* ao_norm,
    const float* gx_norm, const float* gy_norm, const float* gz_norm,
    const float* grid_weights, const double* rho, const double* exc,
    const double* vrho, const double* vsigma, const double* grad_rho_x,
    const double* grad_rho_y, const double* grad_rho_z, float* W_full,
    float* W_sigma, double* exc_total)
{
    SIMPLE_DEVICE_FOR(ig, n_grid)
    {
        if (rho[ig] < 1e-10)
        {
            for (int i = 0; i < nao; i++)
            {
                W_full[ig * nao + i] = 0.0f;
                W_sigma[ig * nao + i] = 0.0f;
            }
        }
        else
        {
            const float w = grid_weights[ig];
            atomicAdd(exc_total, (double)w * exc[ig]);
            const double v_rho = vrho[ig];
            const double v_sigma = vsigma[ig];
            const double grx = grad_rho_x[ig];
            const double gry = grad_rho_y[ig];
            const double grz = grad_rho_z[ig];

            for (int i = 0; i < nao; i++)
            {
                const double ai = (double)ao_norm[ig * nao + i];
                const double gxi = (double)gx_norm[ig * nao + i];
                const double gyi = (double)gy_norm[ig * nao + i];
                const double gzi = (double)gz_norm[ig * nao + i];

                const double sigma_part =
                    2.0 * v_sigma * (grx * gxi + gry * gyi + grz * gzi);
                W_full[ig * nao + i] =
                    (float)((double)w * (v_rho * ai + sigma_part));
                W_sigma[ig * nao + i] = (float)((double)w * sigma_part);
            }
        }
    }
}

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
    double* d_exc_total, float* d_Vxc,
    // 预分配的 BLAS 优化缓冲
    float* d_ao_norm, float* d_gx_norm, float* d_gy_norm, float* d_gz_norm,
    float* d_Pao, float* d_W_full, float* d_W_sigma, double* d_grad_rho_x,
    double* d_grad_rho_y, double* d_grad_rho_z,
    const float* d_shell_r2_screen)
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

        float* d_vals_use = d_ao_vals;
        float* d_gx_use = d_ao_grad_x;
        float* d_gy_use = d_ao_grad_y;
        float* d_gz_use = d_ao_grad_z;
        int nao_eval = nao_s;

        if (is_spherical)
        {
            d_vals_use = d_ao_vals_cart;
            d_gx_use = d_ao_grad_x_cart;
            d_gy_use = d_ao_grad_y_cart;
            d_gz_use = d_ao_grad_z_cart;
            nao_eval = nao_c;
        }

        // 1. 计算 AO 值和梯度（带壳层 screening）
        Launch_Device_Kernel(
            QC_Eval_AO_Grid_Batch_Kernel, (n_batch + threads - 1) / threads,
            threads, 0, 0, n_batch, d_coords_batch, nao_eval, nbas, d_centers,
            d_l_list, d_exps, d_coeffs, d_shell_offsets, d_shell_sizes,
            d_ao_offsets, d_shell_r2_screen, d_vals_use, d_gx_use, d_gy_use,
            d_gz_use);

        if (is_spherical)
        {
            QC_Cart2Sph_AO_Batch_Device(blas_handle, n_batch, nao_c, nao_s,
                                        d_cart2sph_mat, d_ao_vals_cart,
                                        d_ao_grad_x_cart, d_ao_grad_y_cart,
                                        d_ao_grad_z_cart, d_ao_vals,
                                        d_ao_grad_x, d_ao_grad_y, d_ao_grad_z);
        }

        // 2. 归一化 AO: ao_norm[ig, i] = ao[ig, i] * norms[i]
        const int total_ao = n_batch * nao;
        Launch_Device_Kernel(QC_Apply_Norms_AO_Kernel,
                             (total_ao + threads - 1) / threads, threads, 0, 0,
                             n_batch, nao, d_norms, d_ao_vals, d_ao_norm);
        Launch_Device_Kernel(QC_Apply_Norms_AO_Kernel,
                             (total_ao + threads - 1) / threads, threads, 0, 0,
                             n_batch, nao, d_norms, d_ao_grad_x, d_gx_norm);
        Launch_Device_Kernel(QC_Apply_Norms_AO_Kernel,
                             (total_ao + threads - 1) / threads, threads, 0, 0,
                             n_batch, nao, d_norms, d_ao_grad_y, d_gy_norm);
        Launch_Device_Kernel(QC_Apply_Norms_AO_Kernel,
                             (total_ao + threads - 1) / threads, threads, 0, 0,
                             n_batch, nao, d_norms, d_ao_grad_z, d_gz_norm);

        // 3. Pao = P @ AO_norm^T  (nao × n_batch)
        //    P: nao×nao 行主序, AO_norm: n_batch×nao 行主序
        //    Pao: nao × n_batch 行主序
        //    列主序视角: Pao^T = AO @ P^T
        //    sgemm(N, T): C = AO @ P^T where C=nbatch×nao, AO=nbatch×nao, P=nao×nao
        {
            const float one = 1.0f, zero = 0.0f;
            deviceBlasSgemm(blas_handle, DEVICE_BLAS_OP_T, DEVICE_BLAS_OP_N,
                            n_batch, nao, nao, &one, d_ao_norm, nao, d_P, nao,
                            &zero, d_Pao, n_batch);
        }

        // 4. 从 Pao 计算 ρ, σ, ∇ρ
        Launch_Device_Kernel(
            QC_Eval_Rho_Sigma_BLAS_Kernel, (n_batch + threads - 1) / threads,
            threads, 0, 0, n_batch, nao, d_ao_norm, d_gx_norm, d_gy_norm,
            d_gz_norm, d_Pao, d_rho, d_sigma, d_grad_rho_x, d_grad_rho_y,
            d_grad_rho_z);

        // 5. XC 泛函求值
        Launch_Device_Kernel(QC_Eval_XC_Derivs_Kernel,
                             (n_batch + threads - 1) / threads, threads, 0, 0,
                             n_batch, (int)method, d_rho, d_sigma, d_exc,
                             d_vrho, d_vsigma);

        // 6. 构建加权 AO: W_full 和 W_sigma
        Launch_Device_Kernel(
            QC_Build_Weighted_AO_Kernel, (n_batch + threads - 1) / threads,
            threads, 0, 0, n_batch, nao, d_ao_norm, d_gx_norm, d_gy_norm,
            d_gz_norm, d_weights_batch, d_rho, d_exc, d_vrho, d_vsigma,
            d_grad_rho_x, d_grad_rho_y, d_grad_rho_z, d_W_full, d_W_sigma,
            d_exc_total);

        // 7. Vxc += W_full^T @ AO_norm (vrho + 单侧 sigma)
        //    列主序: Vxc^T += AO_norm_col @ W_full_col^T
        {
            const float one = 1.0f;
            deviceBlasSgemm(blas_handle, DEVICE_BLAS_OP_N, DEVICE_BLAS_OP_T,
                            nao, nao, n_batch, &one, d_ao_norm, nao,
                            d_W_full, nao, &one, d_Vxc, nao);
        }

        // 8. Vxc += AO_norm^T @ W_sigma (补全另一侧 sigma)
        //    列主序: Vxc^T += W_sigma_col @ AO_norm_col^T
        {
            const float one = 1.0f;
            deviceBlasSgemm(blas_handle, DEVICE_BLAS_OP_N, DEVICE_BLAS_OP_T,
                            nao, nao, n_batch, &one, d_W_sigma, nao,
                            d_ao_norm, nao, &one, d_Vxc, nao);
        }
    }

}

// TODO: UKS 版本待同步优化
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
    const float* d_shell_r2_screen)
{
    const int nao2 = nao_s * nao_s;
    deviceMemset(d_Vxc_a, 0, sizeof(float) * nao2);
    deviceMemset(d_Vxc_b, 0, sizeof(float) * nao2);
    deviceMemset(d_exc_total, 0, sizeof(double));
    if (total_grid_size <= 0) return;

    const int batch_size = std::max(1, grid_batch_size);
    const int threads = 128;

    for (int g0 = 0; g0 < total_grid_size; g0 += batch_size)
    {
        const int n_batch = std::min(batch_size, total_grid_size - g0);
        const float* d_coords_batch = d_grid_coords + g0 * 3;
        const float* d_weights_batch = d_grid_weights + g0;

        float* d_vals_use = d_ao_vals;
        float* d_gx_use = d_ao_grad_x;
        float* d_gy_use = d_ao_grad_y;
        float* d_gz_use = d_ao_grad_z;
        int nao_eval = nao_s;

        if (is_spherical)
        {
            d_vals_use = d_ao_vals_cart;
            d_gx_use = d_ao_grad_x_cart;
            d_gy_use = d_ao_grad_y_cart;
            d_gz_use = d_ao_grad_z_cart;
            nao_eval = nao_c;
        }

        Launch_Device_Kernel(
            QC_Eval_AO_Grid_Batch_Kernel, (n_batch + threads - 1) / threads,
            threads, 0, 0, n_batch, d_coords_batch, nao_eval, nbas, d_centers,
            d_l_list, d_exps, d_coeffs, d_shell_offsets, d_shell_sizes,
            d_ao_offsets, d_shell_r2_screen, d_vals_use, d_gx_use, d_gy_use,
            d_gz_use);

        if (is_spherical)
        {
            QC_Cart2Sph_AO_Batch_Device(blas_handle, n_batch, nao_c, nao_s,
                                        d_cart2sph_mat, d_ao_vals_cart,
                                        d_ao_grad_x_cart, d_ao_grad_y_cart,
                                        d_ao_grad_z_cart, d_ao_vals,
                                        d_ao_grad_x, d_ao_grad_y, d_ao_grad_z);
        }

        Launch_Device_Kernel(
            QC_Build_Vxc_UKS_Kernel, (n_batch + threads - 1) / threads,
            threads, 0, 0, n_batch, nao_s, (int)method, d_ao_vals,
            d_ao_grad_x, d_ao_grad_y, d_ao_grad_z, d_weights_batch, d_Pa, d_Pb,
            d_norms, d_Vxc_a, d_Vxc_b, d_exc_total);
    }
}

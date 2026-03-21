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
    double* d_exc_total, float* d_Vxc)
{
    const int nao2 = nao_s * nao_s;
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

        Launch_Device_Kernel(
            QC_Eval_AO_Grid_Batch_Kernel, (n_batch + threads - 1) / threads,
            threads, 0, 0, n_batch, d_coords_batch, nao_eval, nbas, d_centers,
            d_l_list, d_exps, d_coeffs, d_shell_offsets, d_shell_sizes,
            d_ao_offsets, d_vals_use, d_gx_use, d_gy_use, d_gz_use);

        if (is_spherical)
        {
            QC_Cart2Sph_AO_Batch_Device(blas_handle, n_batch, nao_c, nao_s,
                                        d_cart2sph_mat, d_ao_vals_cart,
                                        d_ao_grad_x_cart, d_ao_grad_y_cart,
                                        d_ao_grad_z_cart, d_ao_vals,
                                        d_ao_grad_x, d_ao_grad_y, d_ao_grad_z);
        }

        Launch_Device_Kernel(
            QC_Eval_Rho_Sigma_Kernel, (n_batch + threads - 1) / threads,
            threads, 0, 0, n_batch, nao_s, d_ao_vals, d_ao_grad_x, d_ao_grad_y,
            d_ao_grad_z, d_P, d_norms, d_rho, d_sigma);

        Launch_Device_Kernel(QC_Eval_XC_Derivs_Kernel,
                             (n_batch + threads - 1) / threads, threads, 0, 0,
                             n_batch, (int)method, d_rho, d_sigma, d_exc,
                             d_vrho, d_vsigma);

        Launch_Device_Kernel(
            QC_Build_Vxc_Kernel, (n_batch + threads - 1) / threads, threads, 0,
            0, n_batch, nao_s, d_ao_vals, d_ao_grad_x, d_ao_grad_y, d_ao_grad_z,
            d_weights_batch, d_P, d_rho, d_exc, d_vrho, d_vsigma, d_norms,
            d_Vxc, d_exc_total);
    }
}

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
    float* d_ao_grad_z, double* d_exc_total, float* d_Vxc_a, float* d_Vxc_b)
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
            d_ao_offsets, d_vals_use, d_gx_use, d_gy_use, d_gz_use);

        if (is_spherical)
        {
            QC_Cart2Sph_AO_Batch_Device(blas_handle, n_batch, nao_c, nao_s,
                                        d_cart2sph_mat, d_ao_vals_cart,
                                        d_ao_grad_x_cart, d_ao_grad_y_cart,
                                        d_ao_grad_z_cart, d_ao_vals,
                                        d_ao_grad_x, d_ao_grad_y, d_ao_grad_z);
        }

        Launch_Device_Kernel(
            QC_Build_Vxc_UKS_Kernel, (n_batch + threads - 1) / threads, threads,
            0, 0, n_batch, nao_s, (int)method, d_ao_vals, d_ao_grad_x,
            d_ao_grad_y, d_ao_grad_z, d_weights_batch, d_Pa, d_Pb, d_norms,
            d_Vxc_a, d_Vxc_b, d_exc_total);
    }
}

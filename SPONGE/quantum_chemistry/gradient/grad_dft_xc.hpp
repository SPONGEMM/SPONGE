#pragma once

// 依赖: 此文件需要在 vxc.hpp 相关 kernel 可用之后 include
// (dft.hpp 定义了 QC_Eval_AO_Grid_Kernel, QC_Eval_Rho_Kernel 等)

// ====================== DFT XC 网格梯度 ======================
// dE_xc/dR_A = -2 Σ_g Σ_{μ∈A} (∂φ_μ/∂r_d) · W_pao_μ(g)
//
// LDA:  W_pao_μ = w · v_ρ · Pao_μ
// GGA:  W_pao_μ = w · v_ρ · Pao_μ + 2·w·v_σ · (∇ρ · GPao_μ)
//        其中 GPao_μ = Σ_ν P_μν · ∇φ_ν
//
// 注: GGA 缺少涉及 AO 二阶导数的 term(a)，待后续实现
// ==============================================================

// 构建加权 Pao (LDA + GGA term b)
template <int deriv_level>
static __global__ void QC_Build_W_Pao_Kernel(
    const int n_grid, const int nao, const float* weights,
    const double* vrho, const double* vsigma, const double* rho,
    const double* grad_rho_x, const double* grad_rho_y,
    const double* grad_rho_z, const float* Pao,
    const float* GPao_scratch, // 累积 GPao_rho (GGA only)
    float* W_pao)
{
    SIMPLE_DEVICE_FOR(idx, n_grid * nao)
    {
        const int g = idx % n_grid;
        const int mu = idx / n_grid;
        float val = 0.0f;
        if (rho[g] >= 1e-20)
        {
            float w_vrho = (float)(weights[g] * vrho[g]);
            val = w_vrho * Pao[idx];
            if (deriv_level >= 1 && vsigma != nullptr)
            {
                float w_vsigma2 = (float)(2.0 * weights[g] * vsigma[g]);
                val += w_vsigma2 * GPao_scratch[idx];
            }
        }
        W_pao[idx] = val;
    }
}

// 累加到原子梯度
static __global__ void QC_XC_Grad_Accumulate_Kernel(
    const int n_grid, const int nao, const int nbas,
    const int* shell_atom, const int* ao_offsets,
    const float* gx_norm, const float* gy_norm, const float* gz_norm,
    const float* W_pao, double* grad)
{
    SIMPLE_DEVICE_FOR(ig, n_grid)
    {
        for (int ish = 0; ish < nbas; ish++)
        {
            const int atom = shell_atom[ish];
            const int ao0 = ao_offsets[ish];
            const int ao1 = (ish + 1 < nbas) ? ao_offsets[ish + 1] : nao;
            for (int mu = ao0; mu < ao1; mu++)
            {
                const float wp = W_pao[mu * n_grid + ig];
                if (fabsf(wp) < 1e-30f) continue;
                atomicAdd(&grad[atom * 3 + 0],
                          (double)(-2.0f * gx_norm[ig * nao + mu] * wp));
                atomicAdd(&grad[atom * 3 + 1],
                          (double)(-2.0f * gy_norm[ig * nao + mu] * wp));
                atomicAdd(&grad[atom * 3 + 2],
                          (double)(-2.0f * gz_norm[ig * nao + mu] * wp));
            }
        }
    }
}

// 构建 GPao_rho: Σ_dir ∇ρ_dir · (P @ ∇φ_dir)
// 逐方向累积到 scratch 缓冲
static __global__ void QC_Accumulate_GPao_Rho_Kernel(
    const int n_grid, const int nao, const double* grad_rho_dir,
    const float* Pgao_dir, // P @ grad_dir_norm^T, [nao x n_grid]
    float* GPao_rho,       // [nao x n_grid], 累加
    bool first_dir)
{
    SIMPLE_DEVICE_FOR(idx, n_grid * nao)
    {
        const int g = idx % n_grid;
        float val = (float)grad_rho_dir[g] * Pgao_dir[idx];
        if (first_dir)
            GPao_rho[idx] = val;
        else
            GPao_rho[idx] += val;
    }
}

// RKS XC 梯度主函数
template <int deriv_level>
static void QC_Build_DFT_XC_Gradient_RKS_Impl(
    BLAS_HANDLE blas_handle, QC_METHOD method, int is_spherical, int nao_c,
    int nao_s, int total_grid_size, int grid_batch_size, int nbas,
    const float* d_grid_coords, const float* d_grid_weights,
    const float* d_cart2sph_mat, const VECTOR* d_centers, const int* d_l_list,
    const float* d_exps, const float* d_coeffs, const int* d_shell_offsets,
    const int* d_shell_sizes, const int* d_ao_offsets, const float* d_norms,
    const float* d_P,
    // DFT buffers (reused from VXC build)
    float* d_ao_vals_cart, float* d_ao_grad_x_cart, float* d_ao_grad_y_cart,
    float* d_ao_grad_z_cart, float* d_ao_vals, float* d_ao_grad_x,
    float* d_ao_grad_y, float* d_ao_grad_z, double* d_rho, double* d_sigma,
    double* d_exc, double* d_vrho, double* d_vsigma, float* d_ao_norm,
    float* d_gx_norm, float* d_gy_norm, float* d_gz_norm, float* d_Pao,
    double* d_grad_rho_x, double* d_grad_rho_y, double* d_grad_rho_z,
    const float* d_shell_r2_screen,
    // Gradient specific
    const int* d_shell_atom, const int* d_ao_offsets_grad,
    float* d_W_pao, float* d_GPao_scratch,
    double* d_grad)
{
    const int nao = nao_s;
    if (total_grid_size <= 0) return;
    if (std::getenv("SPONGE_DEBUG_GRAD"))
        std::fprintf(stderr, "XC_GRAD: total_grid=%d nao=%d nbas=%d\n",
                     total_grid_size, nao, nbas);
    const int batch_size = std::max(1, grid_batch_size);
    const int threads = 128;

    for (int g0 = 0; g0 < total_grid_size; g0 += batch_size)
    {
        const int n_batch = std::min(batch_size, total_grid_size - g0);
        const float* d_coords_batch = d_grid_coords + g0 * 3;
        const float* d_weights_batch = d_grid_weights + g0;
        const int total_ao = n_batch * nao;

        // ====== 步骤 1-4: 与 VXC build 完全相同 ======

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

        // 2. Pao = P^T @ AO_norm^T
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

        // ====== 步骤 5-6: XC 梯度特有 ======

        // 5. 构建 W_pao
        const bool is_gga = (method != QC_METHOD::LDA);
        if (is_gga)
        {
            // GGA: 计算 GPao_rho = Σ_dir ∇ρ_dir · (P @ ∇φ_dir_norm^T)
            const float one = 1.0f, zero = 0.0f;
            const float* grad_dirs[3] = {d_gx_norm, d_gy_norm, d_gz_norm};
            const double* drho_dirs[3] = {d_grad_rho_x, d_grad_rho_y,
                                          d_grad_rho_z};
            for (int dir = 0; dir < 3; dir++)
            {
                // Pgao_dir = P^T @ grad_dir_norm^T → d_GPao_scratch (临时)
                // 先算到 d_W_pao 作为临时区
                deviceBlasSgemm(blas_handle, DEVICE_BLAS_OP_T, DEVICE_BLAS_OP_N,
                                n_batch, nao, nao, &one, grad_dirs[dir], nao,
                                d_P, nao, &zero, d_W_pao, n_batch);
                // 累积 GPao_rho += ∇ρ_dir · Pgao_dir
                const int tot = n_batch * nao;
                Launch_Device_Kernel(
                    QC_Accumulate_GPao_Rho_Kernel,
                    (tot + threads - 1) / threads, threads, 0, 0, n_batch, nao,
                    drho_dirs[dir], d_W_pao, d_GPao_scratch, dir == 0);
            }
        }

        // 构建最终 W_pao = w·v_ρ·Pao (+ 2·w·v_σ·GPao_rho for GGA)
        {
            const int tot = n_batch * nao;
            if (is_gga)
                Launch_Device_Kernel(
                    (QC_Build_W_Pao_Kernel<1>),
                    (tot + threads - 1) / threads, threads, 0, 0, n_batch, nao,
                    d_weights_batch, d_vrho, d_vsigma, d_rho, d_grad_rho_x,
                    d_grad_rho_y, d_grad_rho_z, d_Pao, d_GPao_scratch, d_W_pao);
            else
                Launch_Device_Kernel(
                    (QC_Build_W_Pao_Kernel<0>),
                    (tot + threads - 1) / threads, threads, 0, 0, n_batch, nao,
                    d_weights_batch, d_vrho, d_vsigma, d_rho, d_grad_rho_x,
                    d_grad_rho_y, d_grad_rho_z, d_Pao, d_GPao_scratch, d_W_pao);
        }

        // 6. 累加到原子梯度
        Launch_Device_Kernel(
            QC_XC_Grad_Accumulate_Kernel,
            (n_batch + threads - 1) / threads, threads, 0, 0, n_batch, nao,
            nbas, d_shell_atom, d_ao_offsets_grad,
            d_gx_norm, d_gy_norm, d_gz_norm, d_W_pao, d_grad);
    }
}

// RKS XC 梯度入口 (根据泛函类型选择 LDA / GGA)
static void QC_Build_DFT_XC_Gradient_RKS(
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
    float* d_ao_norm, float* d_gx_norm, float* d_gy_norm, float* d_gz_norm,
    float* d_Pao, double* d_grad_rho_x, double* d_grad_rho_y,
    double* d_grad_rho_z, const float* d_shell_r2_screen,
    const int* d_shell_atom, const int* d_ao_offsets_grad,
    float* d_W_pao, float* d_GPao_scratch,
    double* d_grad)
{
    const bool is_gga = (method != QC_METHOD::LDA);

#define XC_GRAD_ARGS                                                           \
    blas_handle, method, is_spherical, nao_c, nao_s, total_grid_size,          \
        grid_batch_size, nbas, d_grid_coords, d_grid_weights, d_cart2sph_mat,  \
        d_centers, d_l_list, d_exps, d_coeffs, d_shell_offsets, d_shell_sizes, \
        d_ao_offsets, d_norms, d_P, d_ao_vals_cart, d_ao_grad_x_cart,          \
        d_ao_grad_y_cart, d_ao_grad_z_cart, d_ao_vals, d_ao_grad_x,            \
        d_ao_grad_y, d_ao_grad_z, d_rho, d_sigma, d_exc, d_vrho, d_vsigma,    \
        d_ao_norm, d_gx_norm, d_gy_norm, d_gz_norm, d_Pao, d_grad_rho_x,      \
        d_grad_rho_y, d_grad_rho_z, d_shell_r2_screen, d_shell_atom,           \
        d_ao_offsets_grad, d_W_pao, d_GPao_scratch, d_grad

    // XC 梯度始终需要 AO 梯度 (∂φ/∂r)，所以固定 deriv_level=1
    // LDA vs GGA 的区别在 Impl 内部通过 method 判断
    QC_Build_DFT_XC_Gradient_RKS_Impl<1>(XC_GRAD_ARGS);

#undef XC_GRAD_ARGS
}

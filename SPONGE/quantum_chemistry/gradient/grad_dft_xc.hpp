#pragma once

// 依赖: 此文件需要在 vxc.hpp 相关 kernel 可用之后 include
// (dft.hpp 定义了 QC_Eval_AO_Grid_Kernel, QC_Eval_Rho_Kernel 等)

static __global__ void QC_Double_Accumulate_Kernel(int n, double* dst,
                                                   const double* src)
{
    SIMPLE_DEVICE_FOR(i, n) { dst[i] += src[i]; }
}

// DFT XC 网格梯度
// dE_xc/dR_A_d = -2 Σ_g Σ_{μ∈A} ∂φ_μ/∂r_d · W_pao_μ(g)
//              + -2 Σ_g Σ_{μ∈A} H_d_μ(g) · W_pao_a_μ(g)  [GGA term a]
//
// W_pao  = w·v_ρ·Pao + 2·w·v_σ·(∇ρ·GPao)         [LDA + GGA term b]
// W_pao_a = 2·w·v_σ·Pao                             [GGA term a weight]
// H_d_μ  = Σ_dir ∇ρ_dir · ∂²φ_μ/(∂r_dir·∂r_d)    [Hessian·∇ρ contraction]

// AO 二阶导数与 ∇ρ 的收缩:
// H_d[μ,g] = Σ_dir ∇ρ_dir(g) · ∂²φ_μ/(∂r_dir · ∂r_d)(g)
// 输出 3 个缓冲 [n_grid × nao_cart], 与 ao_grad_x 布局一致
static __global__ void QC_Eval_AO_Hessian_DotGradRho_Kernel(
    const int n_grid, const int nao, const int nbas, const float* grid_coords,
    const VECTOR* centers, const int* l_list, const float* exps_arr,
    const float* coeffs_arr, const int* shell_offsets, const int* shell_sizes,
    const int* ao_offsets, const float* shell_r2_screen,
    const double* grad_rho_x, const double* grad_rho_y,
    const double* grad_rho_z, float* hess_x, float* hess_y, float* hess_z)
{
    SIMPLE_DEVICE_FOR(ig, n_grid)
    {
        const float gx = grid_coords[ig * 3 + 0];
        const float gy = grid_coords[ig * 3 + 1];
        const float gz = grid_coords[ig * 3 + 2];
        const float grx = (float)grad_rho_x[ig];
        const float gry = (float)grad_rho_y[ig];
        const float grz = (float)grad_rho_z[ig];

        for (int i = 0; i < nao; i++)
            hess_x[ig * nao + i] = hess_y[ig * nao + i] = hess_z[ig * nao + i] =
                0.0f;

        for (int ish = 0; ish < nbas; ish++)
        {
            const VECTOR c = centers[ish];
            const float dx = gx - c.x, dy = gy - c.y, dz = gz - c.z;
            const float r2 = dx * dx + dy * dy + dz * dz;
            if (r2 > shell_r2_screen[ish]) continue;

            const int l = l_list[ish];
            const int ncart = (l + 1) * (l + 2) / 2;
            const int ao_off = ao_offsets[ish];

            float px[6], py[6], pz[6];
            px[0] = py[0] = pz[0] = 1.0f;
            for (int k = 1; k <= 5; k++)
            {
                px[k] = px[k - 1] * dx;
                py[k] = py[k - 1] * dy;
                pz[k] = pz[k - 1] * dz;
            }

            for (int ip = 0; ip < shell_sizes[ish]; ip++)
            {
                const int pidx = shell_offsets[ish] + ip;
                const float a = exps_arr[pidx];
                const float e = coeffs_arr[pidx] * expf(-a * r2);
                if (fabsf(e) < 1e-20f) continue;
                const float a2 = 2.0f * a;
                const float a4 = 4.0f * a * a;

                for (int ic = 0; ic < ncart; ic++)
                {
                    int lx, ly, lz;
                    QC_Get_Lxyz_Device(l, ic, lx, ly, lz);

                    // D0: polynomial value
                    const float d0x = px[lx], d0y = py[ly], d0z = pz[lz];
                    // D1: first derivative factor (same as AO grad kernel)
                    const float d1x = (lx > 0 ? (float)lx * px[lx - 1] : 0.0f) -
                                      a2 * px[lx + 1];
                    const float d1y = (ly > 0 ? (float)ly * py[ly - 1] : 0.0f) -
                                      a2 * py[ly + 1];
                    const float d1z = (lz > 0 ? (float)lz * pz[lz - 1] : 0.0f) -
                                      a2 * pz[lz + 1];
                    // D2: second derivative factor (diagonal)
                    const float d2xx =
                        (lx > 1 ? (float)(lx * (lx - 1)) * px[lx - 2] : 0.0f) -
                        a2 * (float)(2 * lx + 1) * px[lx] + a4 * px[lx + 2];
                    const float d2yy =
                        (ly > 1 ? (float)(ly * (ly - 1)) * py[ly - 2] : 0.0f) -
                        a2 * (float)(2 * ly + 1) * py[ly] + a4 * py[ly + 2];
                    const float d2zz =
                        (lz > 1 ? (float)(lz * (lz - 1)) * pz[lz - 2] : 0.0f) -
                        a2 * (float)(2 * lz + 1) * pz[lz] + a4 * pz[lz + 2];

                    // H_x = grx·∂²φ/∂x² + gry·∂²φ/∂x∂y + grz·∂²φ/∂x∂z
                    //      = [grx·D2xx·D0y·D0z + D1x·(gry·D1y·D0z +
                    //      grz·D0y·D1z)]·e
                    const float cross_yz = gry * d1y * d0z + grz * d0y * d1z;
                    const float hx =
                        e * (grx * d2xx * d0y * d0z + d1x * cross_yz);

                    // H_y = grx·∂²φ/∂y∂x + gry·∂²φ/∂y² + grz·∂²φ/∂y∂z
                    const float cross_xz = grx * d1x * d0z + grz * d0x * d1z;
                    const float hy =
                        e * (gry * d0x * d2yy * d0z + d1y * cross_xz);

                    // H_z = grx·∂²φ/∂z∂x + gry·∂²φ/∂z∂y + grz·∂²φ/∂z²
                    const float cross_xy = grx * d1x * d0y + gry * d0x * d1y;
                    const float hz =
                        e * (grz * d0x * d0y * d2zz + d1z * cross_xy);

                    const int i = ao_off + ic;
                    hess_x[ig * nao + i] += hx;
                    hess_y[ig * nao + i] += hy;
                    hess_z[ig * nao + i] += hz;
                }
            }
        }
    }
}

// 构建加权 Pao (LDA + GGA term b)
template <int deriv_level>
static __global__ void QC_Build_W_Pao_Kernel(
    const int n_grid, const int nao, const float* weights, const double* vrho,
    const double* vsigma, const double* rho, const double* grad_rho_x,
    const double* grad_rho_y, const double* grad_rho_z, const float* Pao,
    const float* GPao_scratch,  // 累积 GPao_rho (GGA only)
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
    const int n_grid, const int nao, const int nbas, const int* shell_atom,
    const int* ao_offsets, const float* gx_norm, const float* gy_norm,
    const float* gz_norm, const float* W_pao, double* grad)
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

// GGA term(a) 的权重: W_pao_a = 2·w·v_σ·Pao
static __global__ void QC_Build_W_Pao_TermA_Kernel(
    const int n_grid, const int nao, const float* weights, const double* vsigma,
    const double* rho, const float* Pao, float* W_pao)
{
    SIMPLE_DEVICE_FOR(idx, n_grid * nao)
    {
        const int g = idx % n_grid;
        float val = 0.0f;
        if (rho[g] >= 1e-20)
            val = (float)(2.0 * weights[g] * vsigma[g]) * Pao[idx];
        W_pao[idx] = val;
    }
}

// 构建 GPao_rho: Σ_dir ∇ρ_dir · (P @ ∇φ_dir)
// 逐方向累积到 scratch 缓冲
static __global__ void QC_Accumulate_GPao_Rho_Kernel(
    const int n_grid, const int nao, const double* grad_rho_dir,
    const float* Pgao_dir,  // P @ grad_dir_norm^T, [nao x n_grid]
    float* GPao_rho,        // [nao x n_grid], 累加
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
    BLAS_HANDLE blas_handle, QC_METHOD method, const QC_MOLECULE& mol,
    QC_DFT& dft, const QC_CARTESIAN_TO_SPHERICAL& cart2sph,
    const float* d_norms, const float* d_P, const int* d_shell_atom,
    const int* d_ao_offsets_grad, float* d_W_pao, float* d_GPao_scratch,
    double* d_grad)
{
    // 局部别名: mol
    const int is_spherical = mol.is_spherical;
    const int nao_c = mol.nao_cart;
    const int nao_s = mol.nao;
    const int nao = nao_s;
    const int nbas = mol.nbas;
    const VECTOR* d_centers = mol.d_centers;
    const int* d_l_list = mol.d_l_list;
    const float* d_exps = mol.d_exps;
    const float* d_coeffs = mol.d_coeffs;
    const int* d_shell_offsets = mol.d_shell_offsets;
    const int* d_shell_sizes = mol.d_shell_sizes;
    const int* d_ao_offsets = mol.d_ao_offsets;
    // 局部别名: dft
    const int total_grid_size = dft.max_grid_size;
    const float* d_grid_coords = dft.d_grid_coords;
    const float* d_grid_weights = dft.d_grid_weights;
    const float* d_shell_r2_screen = dft.d_shell_r2_screen;
    float* d_ao_vals_cart = dft.d_ao_vals_cart;
    float* d_ao_grad_x_cart = dft.d_ao_grad_x_cart;
    float* d_ao_grad_y_cart = dft.d_ao_grad_y_cart;
    float* d_ao_grad_z_cart = dft.d_ao_grad_z_cart;
    float* d_ao_vals = dft.d_ao_vals;
    float* d_ao_grad_x = dft.d_ao_grad_x;
    float* d_ao_grad_y = dft.d_ao_grad_y;
    float* d_ao_grad_z = dft.d_ao_grad_z;
    double* d_rho = dft.d_rho;
    double* d_sigma = dft.d_sigma;
    double* d_exc = dft.d_exc;
    double* d_vrho = dft.d_vrho;
    double* d_vsigma = dft.d_vsigma;
    float* d_ao_norm = dft.d_ao_norm;
    float* d_gx_norm = dft.d_gx_norm;
    float* d_gy_norm = dft.d_gy_norm;
    float* d_gz_norm = dft.d_gz_norm;
    float* d_Pao = dft.d_Pao;
    double* d_grad_rho_x = dft.d_grad_rho_x;
    double* d_grad_rho_y = dft.d_grad_rho_y;
    double* d_grad_rho_z = dft.d_grad_rho_z;
    // 局部别名: cart2sph
    const float* d_cart2sph_mat = cart2sph.d_cart2sph_mat;

    if (total_grid_size <= 0) return;
    const int batch_size = std::max(1, dft.grid_batch_size);
    const int threads = 128;

    for (int g0 = 0; g0 < total_grid_size; g0 += batch_size)
    {
        const int n_batch = std::min(batch_size, total_grid_size - g0);
        const float* d_coords_batch = d_grid_coords + g0 * 3;
        const float* d_weights_batch = d_grid_weights + g0;
        const int total_ao = n_batch * nao;

        // 步骤 1-4: 与 VXC build 完全相同

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

        // 步骤 5-6: XC 梯度特有

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
                Launch_Device_Kernel(QC_Accumulate_GPao_Rho_Kernel,
                                     (tot + threads - 1) / threads, threads, 0,
                                     0, n_batch, nao, drho_dirs[dir], d_W_pao,
                                     d_GPao_scratch, dir == 0);
            }
        }

        // 构建最终 W_pao = w·v_ρ·Pao (+ 2·w·v_σ·GPao_rho for GGA)
        {
            const int tot = n_batch * nao;
            if (is_gga)
                Launch_Device_Kernel(
                    (QC_Build_W_Pao_Kernel<1>), (tot + threads - 1) / threads,
                    threads, 0, 0, n_batch, nao, d_weights_batch, d_vrho,
                    d_vsigma, d_rho, d_grad_rho_x, d_grad_rho_y, d_grad_rho_z,
                    d_Pao, d_GPao_scratch, d_W_pao);
            else
                Launch_Device_Kernel(
                    (QC_Build_W_Pao_Kernel<0>), (tot + threads - 1) / threads,
                    threads, 0, 0, n_batch, nao, d_weights_batch, d_vrho,
                    d_vsigma, d_rho, d_grad_rho_x, d_grad_rho_y, d_grad_rho_z,
                    d_Pao, d_GPao_scratch, d_W_pao);
        }

        // 6. 累加到原子梯度 (主项: v_ρ + GGA term b)
        Launch_Device_Kernel(
            QC_XC_Grad_Accumulate_Kernel, (n_batch + threads - 1) / threads,
            threads, 0, 0, n_batch, nao, nbas, d_shell_atom, d_ao_offsets_grad,
            d_gx_norm, d_gy_norm, d_gz_norm, d_W_pao, d_grad);

        // 7. GGA term(a): AO Hessian · ∇ρ 贡献
        if (is_gga)
        {
            // 7a. 计算 H_d = Σ_dir ∇ρ_dir · ∂²φ/(∂dir∂d) (Cartesian)
            float* d_hx = d_ao_grad_x;
            float* d_hy = d_ao_grad_y;
            float* d_hz = d_ao_grad_z;
            int nao_hess = nao_s;
            if (is_spherical)
            {
                d_hx = d_ao_grad_x_cart;
                d_hy = d_ao_grad_y_cart;
                d_hz = d_ao_grad_z_cart;
                nao_hess = nao_c;
            }
            Launch_Device_Kernel(QC_Eval_AO_Hessian_DotGradRho_Kernel,
                                 (n_batch + threads - 1) / threads, threads, 0,
                                 0, n_batch, nao_hess, nbas, d_coords_batch,
                                 d_centers, d_l_list, d_exps, d_coeffs,
                                 d_shell_offsets, d_shell_sizes, d_ao_offsets,
                                 d_shell_r2_screen, d_grad_rho_x, d_grad_rho_y,
                                 d_grad_rho_z, d_hx, d_hy, d_hz);

            // 7b. Cart2Sph (球形基时)
            if (is_spherical)
            {
                QC_MatMul_RowRow_Blas(blas_handle, n_batch, nao_s, nao_c, d_hx,
                                      d_cart2sph_mat, d_ao_grad_x);
                QC_MatMul_RowRow_Blas(blas_handle, n_batch, nao_s, nao_c, d_hy,
                                      d_cart2sph_mat, d_ao_grad_y);
                QC_MatMul_RowRow_Blas(blas_handle, n_batch, nao_s, nao_c, d_hz,
                                      d_cart2sph_mat, d_ao_grad_z);
                d_hx = d_ao_grad_x;
                d_hy = d_ao_grad_y;
                d_hz = d_ao_grad_z;
            }
            // 归一化: hx_norm = hx * norms
            Launch_Device_Kernel(QC_Apply_Norms_AO_Kernel,
                                 (total_ao + threads - 1) / threads, threads, 0,
                                 0, n_batch, nao, d_norms, d_hx, d_gx_norm);
            Launch_Device_Kernel(QC_Apply_Norms_AO_Kernel,
                                 (total_ao + threads - 1) / threads, threads, 0,
                                 0, n_batch, nao, d_norms, d_hy, d_gy_norm);
            Launch_Device_Kernel(QC_Apply_Norms_AO_Kernel,
                                 (total_ao + threads - 1) / threads, threads, 0,
                                 0, n_batch, nao, d_norms, d_hz, d_gz_norm);

            // 7c. W_pao_a = 2·w·v_σ·Pao (term a weight)
            {
                const int tot = n_batch * nao;
                Launch_Device_Kernel(QC_Build_W_Pao_TermA_Kernel,
                                     (tot + threads - 1) / threads, threads, 0,
                                     0, n_batch, nao, d_weights_batch, d_vsigma,
                                     d_rho, d_Pao, d_W_pao);
            }

            // 7d. 累加到原子梯度
            Launch_Device_Kernel(QC_XC_Grad_Accumulate_Kernel,
                                 (n_batch + threads - 1) / threads, threads, 0,
                                 0, n_batch, nao, nbas, d_shell_atom,
                                 d_ao_offsets_grad, d_gx_norm, d_gy_norm,
                                 d_gz_norm, d_W_pao, d_grad);
        }
    }
}

// RKS XC 梯度入口 (根据泛函类型选择 LDA / GGA)
static void QC_Build_DFT_XC_Gradient_RKS(
    BLAS_HANDLE blas_handle, QC_METHOD method, const QC_MOLECULE& mol,
    QC_DFT& dft, const QC_CARTESIAN_TO_SPHERICAL& cart2sph,
    const float* d_norms, const float* d_P, const int* d_shell_atom,
    const int* d_ao_offsets_grad, float* d_W_pao, float* d_GPao_scratch,
    double* d_grad)
{
    QC_Build_DFT_XC_Gradient_RKS_Impl<1>(
        blas_handle, method, mol, dft, cart2sph, d_norms, d_P, d_shell_atom,
        d_ao_offsets_grad, d_W_pao, d_GPao_scratch, d_grad);
}

// UKS XC 梯度
// 对 alpha 和 beta 各处理一次
// GGA: g_eff_α = 2·v_σαα·∇ρα + v_σαβ·∇ρβ
//      g_eff_β = 2·v_σββ·∇ρβ + v_σαβ·∇ρα

// 单方向 eff_grad: out = 2*vs_same*gr_this + vsab*gr_other
static __global__ void QC_Build_UKS_Eff_Grad_One_Kernel(
    const int n_grid, const double* vs_same, const double* vsab,
    const double* gr_this, const double* gr_other, double* out)
{
    SIMPLE_DEVICE_FOR(ig, n_grid)
    {
        out[ig] = 2.0 * vs_same[ig] * gr_this[ig] + vsab[ig] * gr_other[ig];
    }
}

// UKS W_pao: w·v_ρσ·Paoσ + w·GPaoσ_eff
static __global__ void QC_Build_W_Pao_UKS_Kernel(
    const int n_grid, const int nao, const float* weights,
    const double* vrho_spin, const double* rho_total, const float* Pao_spin,
    const double* eff_x, const double* eff_y, const double* eff_z,
    const float* GPao_scratch,  // 已按 eff_grad 计算的 GPao
    bool is_gga, float* W_pao)
{
    SIMPLE_DEVICE_FOR(idx, n_grid * nao)
    {
        const int g = idx % n_grid;
        float val = 0.0f;
        if (rho_total[g] >= 1e-20)
        {
            val = (float)(weights[g] * vrho_spin[g]) * Pao_spin[idx];
            if (is_gga) val += (float)weights[g] * GPao_scratch[idx];
        }
        W_pao[idx] = val;
    }
}

// UKS term(a) 权重: 只有 w·Paoσ (v_σ 已含在 eff_grad 里)
static __global__ void QC_Build_W_Pao_TermA_UKS_Kernel(
    const int n_grid, const int nao, const float* weights,
    const double* rho_total, const float* Pao_spin, float* W_pao)
{
    SIMPLE_DEVICE_FOR(idx, n_grid * nao)
    {
        const int g = idx % n_grid;
        W_pao[idx] =
            (rho_total[g] >= 1e-20) ? (float)weights[g] * Pao_spin[idx] : 0.0f;
    }
}

static void QC_Build_DFT_XC_Gradient_UKS(
    BLAS_HANDLE blas_handle, QC_METHOD method, const QC_MOLECULE& mol,
    QC_DFT& dft, const QC_CARTESIAN_TO_SPHERICAL& cart2sph,
    const float* d_norms, const float* d_Pa, const float* d_Pb,
    const int* d_shell_atom, const int* d_ao_offsets_grad, float* d_W_pao,
    float* d_GPao_scratch, double* d_grad)
{
    const int nao = mol.nao;
    if (dft.max_grid_size <= 0) return;
    const int batch_size = std::max(1, dft.grid_batch_size);
    const int threads = 128;
    const bool is_gga = (method != QC_METHOD::LDA);

    // 局部别名简化内核调用
    const int is_spherical = mol.is_spherical;
    const int nao_c = mol.nao_cart, nao_s = mol.nao;
    const int nbas = mol.nbas;
    const float* d_grid_coords = dft.d_grid_coords;
    const float* d_grid_weights = dft.d_grid_weights;
    const float* d_cart2sph_mat = cart2sph.d_cart2sph_mat;
    const VECTOR* d_centers = mol.d_centers;
    const int* d_l_list = mol.d_l_list;
    const float* d_exps = mol.d_exps;
    const float* d_coeffs = mol.d_coeffs;
    const int* d_shell_offsets = mol.d_shell_offsets;
    const int* d_shell_sizes = mol.d_shell_sizes;
    const int* d_ao_offsets = mol.d_ao_offsets;
    const float* d_shell_r2_screen = dft.d_shell_r2_screen;
    float* d_ao_vals_cart = dft.d_ao_vals_cart;
    float* d_ao_grad_x_cart = dft.d_ao_grad_x_cart;
    float* d_ao_grad_y_cart = dft.d_ao_grad_y_cart;
    float* d_ao_grad_z_cart = dft.d_ao_grad_z_cart;
    float* d_ao_vals = dft.d_ao_vals;
    float* d_ao_grad_x = dft.d_ao_grad_x;
    float* d_ao_grad_y = dft.d_ao_grad_y;
    float* d_ao_grad_z = dft.d_ao_grad_z;
    float* d_ao_norm = dft.d_ao_norm;
    float* d_gx_norm = dft.d_gx_norm;
    float* d_gy_norm = dft.d_gy_norm;
    float* d_gz_norm = dft.d_gz_norm;
    float* d_Pao_a = dft.d_Pao;
    float* d_Pao_b = dft.d_Pao_b;
    double* d_rho_a = dft.d_rho_a;
    double* d_rho_b = dft.d_rho_b;
    double* d_sigma_aa = dft.d_sigma_aa;
    double* d_sigma_ab = dft.d_sigma_ab;
    double* d_sigma_bb = dft.d_sigma_bb;
    double* d_gra_x = dft.d_grad_rho_x;
    double* d_gra_y = dft.d_grad_rho_y;
    double* d_gra_z = dft.d_grad_rho_z;
    double* d_grb_x = dft.d_grb_x;
    double* d_grb_y = dft.d_grb_y;
    double* d_grb_z = dft.d_grb_z;
    double* d_exc = dft.d_exc_buf;
    double* d_vra = dft.d_vra;
    double* d_vrb = dft.d_vrb;
    double* d_vsaa = dft.d_vsaa;
    double* d_vsab = dft.d_vsab;
    double* d_vsbb = dft.d_vsbb;
    const int total_grid_size = dft.max_grid_size;

    for (int g0 = 0; g0 < total_grid_size; g0 += batch_size)
    {
        const int n_batch = std::min(batch_size, total_grid_size - g0);
        const float* d_coords_batch = d_grid_coords + g0 * 3;
        const float* d_weights_batch = d_grid_weights + g0;
        const int total_ao = n_batch * nao;

        // 1. AO 评估 (共享, deriv_level=1)
        {
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
                (QC_Eval_AO_Grid_Kernel<1>), (n_batch + threads - 1) / threads,
                threads, 0, 0, n_batch, d_coords_batch, nao_eval, nbas,
                d_centers, d_l_list, d_exps, d_coeffs, d_shell_offsets,
                d_shell_sizes, d_ao_offsets, d_shell_r2_screen, d_vals_use,
                d_gx_use, d_gy_use, d_gz_use);
            if (is_spherical)
            {
                QC_MatMul_RowRow_Blas(blas_handle, n_batch, nao_s, nao_c,
                                      d_ao_vals_cart, d_cart2sph_mat,
                                      d_ao_vals);
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
            Launch_Device_Kernel(
                QC_Apply_Norms_AO_Kernel, (total_ao + threads - 1) / threads,
                threads, 0, 0, n_batch, nao, d_norms, d_ao_vals, d_ao_norm);
            Launch_Device_Kernel(
                QC_Apply_Norms_AO_Kernel, (total_ao + threads - 1) / threads,
                threads, 0, 0, n_batch, nao, d_norms, d_ao_grad_x, d_gx_norm);
            Launch_Device_Kernel(
                QC_Apply_Norms_AO_Kernel, (total_ao + threads - 1) / threads,
                threads, 0, 0, n_batch, nao, d_norms, d_ao_grad_y, d_gy_norm);
            Launch_Device_Kernel(
                QC_Apply_Norms_AO_Kernel, (total_ao + threads - 1) / threads,
                threads, 0, 0, n_batch, nao, d_norms, d_ao_grad_z, d_gz_norm);
        }

        // 2. Pao_alpha 和 Pao_beta
        {
            const float one = 1.0f, zero = 0.0f;
            deviceBlasSgemm(blas_handle, DEVICE_BLAS_OP_T, DEVICE_BLAS_OP_N,
                            n_batch, nao, nao, &one, d_ao_norm, nao, d_Pa, nao,
                            &zero, d_Pao_a, n_batch);
            deviceBlasSgemm(blas_handle, DEVICE_BLAS_OP_T, DEVICE_BLAS_OP_N,
                            n_batch, nao, nao, &one, d_ao_norm, nao, d_Pb, nao,
                            &zero, d_Pao_b, n_batch);
        }

        // 3. UKS 密度 + 梯度
        Launch_Device_Kernel((QC_Eval_Rho_UKS_Kernel<1>),
                             (n_batch + threads - 1) / threads, threads, 0, 0,
                             n_batch, nao, d_ao_norm, d_gx_norm, d_gy_norm,
                             d_gz_norm, d_Pao_a, d_Pao_b, d_rho_a, d_rho_b,
                             d_sigma_aa, d_sigma_ab, d_sigma_bb, d_gra_x,
                             d_gra_y, d_gra_z, d_grb_x, d_grb_y, d_grb_z);

        // 4. UKS XC 泛函求值
        Launch_Device_Kernel(QC_Eval_XC_UKS_Kernel,
                             (n_batch + threads - 1) / threads, threads, 0, 0,
                             n_batch, (int)method, d_rho_a, d_rho_b, d_sigma_aa,
                             d_sigma_ab, d_sigma_bb, d_exc, d_vra, d_vrb,
                             d_vsaa, d_vsab, d_vsbb);

        // 用 d_rho_a 临时存储 rho_total = rho_a + rho_b (用于密度截断)
        // 注意：这会覆盖 d_rho_a，但后面不再使用 rho_a 的值
        Launch_Device_Kernel(QC_Double_Accumulate_Kernel, (n_batch + 255) / 256,
                             256, 0, 0, n_batch, d_rho_a, d_rho_b);
        double* d_rho_total = d_rho_a;  // alias

        // 处理 alpha 和 beta
        // 不覆盖 ∇ρ，eff_grad 存入 sigma 缓冲
        double* d_eff_tmp[3] = {d_sigma_aa, d_sigma_ab, d_sigma_bb};

        for (int spin = 0; spin < 2; spin++)
        {
            const float* d_Pao_spin = (spin == 0) ? d_Pao_a : d_Pao_b;
            const double* d_vrho_spin = (spin == 0) ? d_vra : d_vrb;
            const float* d_P_spin = (spin == 0) ? d_Pa : d_Pb;
            const float* d_P_other = (spin == 0) ? d_Pb : d_Pa;
            // ∇ρ_this 和 ∇ρ_other (原始值，未被覆盖)
            const double* grt[3] = {(spin == 0) ? d_gra_x : d_grb_x,
                                    (spin == 0) ? d_gra_y : d_grb_y,
                                    (spin == 0) ? d_gra_z : d_grb_z};
            const double* gro[3] = {(spin == 0) ? d_grb_x : d_gra_x,
                                    (spin == 0) ? d_grb_y : d_gra_y,
                                    (spin == 0) ? d_grb_z : d_gra_z};
            const double* d_vs_same = (spin == 0) ? d_vsaa : d_vsbb;

            // 5a. 计算 eff_grad_σ = 2·v_σσσ·∇ρσ + v_σαβ·∇ρ_other
            //     存入 d_eff_tmp (复用 sigma 缓冲)
            if (is_gga)
            {
                Launch_Device_Kernel(QC_Build_UKS_Eff_Grad_One_Kernel,
                                     (n_batch + threads - 1) / threads, threads,
                                     0, 0, n_batch, d_vs_same, d_vsab, grt[0],
                                     gro[0], d_eff_tmp[0]);
                Launch_Device_Kernel(QC_Build_UKS_Eff_Grad_One_Kernel,
                                     (n_batch + threads - 1) / threads, threads,
                                     0, 0, n_batch, d_vs_same, d_vsab, grt[1],
                                     gro[1], d_eff_tmp[1]);
                Launch_Device_Kernel(QC_Build_UKS_Eff_Grad_One_Kernel,
                                     (n_batch + threads - 1) / threads, threads,
                                     0, 0, n_batch, d_vs_same, d_vsab, grt[2],
                                     gro[2], d_eff_tmp[2]);
            }

            // 5b. GGA term b: GPao = eff_σ · Pgaoσ
            if (is_gga)
            {
                const float one = 1.0f, zero = 0.0f;
                const float* gd[3] = {d_gx_norm, d_gy_norm, d_gz_norm};
                // 自旋项: Σ_dir eff_dir · Pgaoσ_dir
                for (int dir = 0; dir < 3; dir++)
                {
                    deviceBlasSgemm(blas_handle, DEVICE_BLAS_OP_T,
                                    DEVICE_BLAS_OP_N, n_batch, nao, nao, &one,
                                    gd[dir], nao, d_P_spin, nao, &zero, d_W_pao,
                                    n_batch);
                    Launch_Device_Kernel(QC_Accumulate_GPao_Rho_Kernel,
                                         (total_ao + threads - 1) / threads,
                                         threads, 0, 0, n_batch, nao,
                                         d_eff_tmp[dir], d_W_pao,
                                         d_GPao_scratch, dir == 0);
                }
                // 注: 不需要交叉项 v_σαβ*∇ρσ·Pgao_other
                // 因为 alpha eff 含 v_σαβ*∇ρβ → 收缩 Pgaoα 给出 ∇ρβ·d(∇ρα)/dR
                //      beta eff  含 v_σαβ*∇ρα → 收缩 Pgaoβ 给出 ∇ρα·d(∇ρβ)/dR
                // 两通道合计恰好覆盖 dσαβ/dR 的两项，无需额外交叉项
            }

            // 6. W_pao = w·v_ρσ·Paoσ + w·GPaoσ_total
            Launch_Device_Kernel(
                QC_Build_W_Pao_UKS_Kernel, (total_ao + threads - 1) / threads,
                threads, 0, 0, n_batch, nao, d_weights_batch, d_vrho_spin,
                d_rho_total, d_Pao_spin, d_eff_tmp[0], d_eff_tmp[1],
                d_eff_tmp[2], d_GPao_scratch, is_gga, d_W_pao);

            // 7. 累加主项 + term b
            Launch_Device_Kernel(QC_XC_Grad_Accumulate_Kernel,
                                 (n_batch + threads - 1) / threads, threads, 0,
                                 0, n_batch, nao, nbas, d_shell_atom,
                                 d_ao_offsets_grad, d_gx_norm, d_gy_norm,
                                 d_gz_norm, d_W_pao, d_grad);

            // 8. GGA term a: Hessian with eff_grad
            if (is_gga)
            {
                float* d_hx = d_ao_grad_x;
                float* d_hy = d_ao_grad_y;
                float* d_hz = d_ao_grad_z;
                int nao_hess = nao_s;
                if (is_spherical)
                {
                    d_hx = d_ao_grad_x_cart;
                    d_hy = d_ao_grad_y_cart;
                    d_hz = d_ao_grad_z_cart;
                    nao_hess = nao_c;
                }
                Launch_Device_Kernel(
                    QC_Eval_AO_Hessian_DotGradRho_Kernel,
                    (n_batch + threads - 1) / threads, threads, 0, 0, n_batch,
                    nao_hess, nbas, d_coords_batch, d_centers, d_l_list, d_exps,
                    d_coeffs, d_shell_offsets, d_shell_sizes, d_ao_offsets,
                    d_shell_r2_screen, d_eff_tmp[0], d_eff_tmp[1], d_eff_tmp[2],
                    d_hx, d_hy, d_hz);

                if (is_spherical)
                {
                    QC_MatMul_RowRow_Blas(blas_handle, n_batch, nao_s, nao_c,
                                          d_hx, d_cart2sph_mat, d_ao_grad_x);
                    QC_MatMul_RowRow_Blas(blas_handle, n_batch, nao_s, nao_c,
                                          d_hy, d_cart2sph_mat, d_ao_grad_y);
                    QC_MatMul_RowRow_Blas(blas_handle, n_batch, nao_s, nao_c,
                                          d_hz, d_cart2sph_mat, d_ao_grad_z);
                    d_hx = d_ao_grad_x;
                    d_hy = d_ao_grad_y;
                    d_hz = d_ao_grad_z;
                }
                // Hessian norms 写入空闲缓冲区 (不覆盖 d_gx_norm 等!)
                // d_ao_vals, d_ao_norm, d_ao_grad_z 此时空闲 (Pao 已算完)
                float* d_hx_norm = d_ao_vals;
                float* d_hy_norm = d_ao_norm;
                // d_hz_norm: 复用 d_GPao_scratch 的前 n_batch*nao 个 float
                // (GPao_scratch 是 float[nao*n_batch]，此时已用完)
                float* d_hz_norm = d_GPao_scratch;
                Launch_Device_Kernel(QC_Apply_Norms_AO_Kernel,
                                     (total_ao + threads - 1) / threads,
                                     threads, 0, 0, n_batch, nao, d_norms, d_hx,
                                     d_hx_norm);
                Launch_Device_Kernel(QC_Apply_Norms_AO_Kernel,
                                     (total_ao + threads - 1) / threads,
                                     threads, 0, 0, n_batch, nao, d_norms, d_hy,
                                     d_hy_norm);
                Launch_Device_Kernel(QC_Apply_Norms_AO_Kernel,
                                     (total_ao + threads - 1) / threads,
                                     threads, 0, 0, n_batch, nao, d_norms, d_hz,
                                     d_hz_norm);

                Launch_Device_Kernel(
                    QC_Build_W_Pao_TermA_UKS_Kernel,
                    (total_ao + threads - 1) / threads, threads, 0, 0, n_batch,
                    nao, d_weights_batch, d_rho_total, d_Pao_spin, d_W_pao);

                Launch_Device_Kernel(QC_XC_Grad_Accumulate_Kernel,
                                     (n_batch + threads - 1) / threads, threads,
                                     0, 0, n_batch, nao, nbas, d_shell_atom,
                                     d_ao_offsets_grad, d_hx_norm, d_hy_norm,
                                     d_hz_norm, d_W_pao, d_grad);
            }
        }
    }
}

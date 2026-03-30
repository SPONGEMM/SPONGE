#include "quantum_chemistry.h"
#include "gradient/grad_one_e.hpp"
#include "gradient/grad_workspace.h"
#include "gradient/gradient.hpp"
#include "integrals/eri/common/direct_fock_kernels.hpp"
#include "integrals/eri/eri_backend.hpp"
#include "gradient/grad_eri.hpp"

static void _debug_print_grad(const char* label, const int natm,
                               const double* grad)
{
    if (!std::getenv("SPONGE_DEBUG_GRAD")) return;
    std::fprintf(stderr, "%s (Ha/Bohr)\n", label);
    for (int ia = 0; ia < natm; ia++)
        std::fprintf(stderr, "  atom %d : (% .10e, % .10e, % .10e)\n", ia,
                     grad[ia * 3 + 0], grad[ia * 3 + 1], grad[ia * 3 + 2]);
}

void QUANTUM_CHEMISTRY::Compute_Gradient(VECTOR* frc, const VECTOR box_length)
{
    if (!is_initialized) return;
    const int natm = mol.natm;
    const int nao = mol.nao;
    const int nao2 = mol.nao2;

    // SCF 收敛后的 Fock 矩阵可能是 DIIS 外推的，特征值不够精确。
    // 用收敛密度重建 Fock 并对角化，获取准确的轨道能量用于 W 矩阵。
    // 必须关闭 level shift，否则特征值会被系统性偏移。
    const double saved_ls = scf_ws.runtime.level_shift;
    scf_ws.runtime.level_shift = 0.0;
    Build_Fock(scf_ws.runtime.max_scf_iter);
    Diagonalize_And_Build_Density();
    scf_ws.runtime.level_shift = saved_ls;

    // 清零梯度累加器
    deviceMemset(grad_ws.d_grad, 0, sizeof(double) * natm * 3);

    // 1. 构建能量加权密度矩阵 W
    {
        // UHF: d_W 已被 beta 特征值覆盖，alpha 特征值在 d_W_alpha
        const float* alpha_epsilon = scf_ws.runtime.unrestricted
                                         ? scf_ws.ortho.d_W_alpha
                                         : scf_ws.ortho.d_W;
        float* d_D_tmp = scf_ws.alpha.d_F;
        QC_Build_Energy_Weighted_Density(
            blas_handle, nao, scf_ws.runtime.n_alpha,
            scf_ws.runtime.occ_factor, scf_ws.alpha.d_C, alpha_epsilon,
            grad_ws.d_W_density, d_D_tmp);

        if (scf_ws.runtime.unrestricted && grad_ws.d_W_density_beta)
        {
            float* d_D_tmp_b = scf_ws.beta.d_F;
            QC_Build_Energy_Weighted_Density(
                blas_handle, nao, scf_ws.runtime.n_beta, 1.0f,
                scf_ws.beta.d_C, scf_ws.ortho.d_W,
                grad_ws.d_W_density_beta, d_D_tmp_b);
            // 合并 W = W_alpha + W_beta，1e 梯度 kernel 只接受一个 W
            for (int i = 0; i < nao2; i++)
                grad_ws.d_W_density[i] += grad_ws.d_W_density_beta[i];
        }
    }

    // 2. 核排斥梯度
    {
        const int threads = 256;
        const VECTOR box_bohr(box_length.x * CONSTANT_ANGSTROM_TO_BOHR,
                              box_length.y * CONSTANT_ANGSTROM_TO_BOHR,
                              box_length.z * CONSTANT_ANGSTROM_TO_BOHR);
        Launch_Device_Kernel(QC_Nuclear_Gradient_Kernel,
                             (natm + threads - 1) / threads, threads, 0, 0,
                             natm, mol.d_Z, mol.d_atm, mol.d_env, box_bohr,
                             grad_ws.d_grad);
    }
    _debug_print_grad("AFTER_NUCLEAR", natm, grad_ws.d_grad);

    // 3. 单电子积分导数: Tr[P·dH/dR] - Tr[W·dS/dR]
    // 1e kernel 在 Cartesian 基中计算积分，需要 Cartesian 版的 P, W, norms。
    // 对 spherical 基组: 变换 P_sph → P_cart, W_sph → W_cart,
    //                   并从 Cartesian 重叠矩阵构建 norms_cart。
    {
        const float* d_P_use = scf_ws.direct.d_P_coul;
        const float* d_W_use = grad_ws.d_W_density;
        const float* d_norms_use = scf_ws.ortho.d_norms;
        int nao_1e = mol.nao;

        // 临时 Cartesian 缓冲（spherical 时使用）
        float* d_P_cart = nullptr;
        float* d_W_cart = nullptr;
        float* d_norms_cart = nullptr;

        if (mol.is_spherical)
        {
            const int nao_c = mol.nao_cart;
            const int nao_c2 = nao_c * nao_c;
            nao_1e = nao_c;

            // 分配临时缓冲
            d_P_cart = (float*)malloc(sizeof(float) * nao_c2);
            d_W_cart = (float*)malloc(sizeof(float) * nao_c2);
            d_norms_cart = (float*)malloc(sizeof(float) * nao_c);

            // Sph2Cart: M_cart = C · M_sph · C^T
            // C = cart2sph_mat [nao_c × nao_s], C^T = [nao_s × nao_c]
            // M_cart = C · M_sph · C^T
            const float* C = cart2sph.d_cart2sph_mat;
            const int nao_s = mol.nao;
            float* tmp = (float*)malloc(sizeof(float) * nao_c * nao_s);

            auto sph2cart = [&](const float* M_sph, float* M_cart) {
                // tmp = C · M_sph  [nao_c × nao_s]
                const float one = 1.0f, zero = 0.0f;
                deviceBlasSgemm(blas_handle, DEVICE_BLAS_OP_N, DEVICE_BLAS_OP_N,
                                nao_c, nao_s, nao_s, &one, C, nao_c, M_sph,
                                nao_s, &zero, tmp, nao_c);
                // M_cart = tmp · C^T  [nao_c × nao_c]
                deviceBlasSgemm(blas_handle, DEVICE_BLAS_OP_N, DEVICE_BLAS_OP_T,
                                nao_c, nao_c, nao_s, &one, tmp, nao_c, C,
                                nao_c, &zero, M_cart, nao_c);
            };

            sph2cart(d_P_use, d_P_cart);
            sph2cart(d_W_use, d_W_cart);
            free(tmp);

            d_P_use = d_P_cart;
            d_W_use = d_W_cart;

            // Cartesian norms from Cartesian overlap: S_cart = C · S_sph · C^T
            float* S_cart = (float*)malloc(sizeof(float) * nao_c2);
            {
                float* tmp2 = (float*)malloc(sizeof(float) * nao_c * nao_s);
                const float one2 = 1.0f, zero2 = 0.0f;
                deviceBlasSgemm(blas_handle, DEVICE_BLAS_OP_N, DEVICE_BLAS_OP_N,
                                nao_c, nao_s, nao_s, &one2, C, nao_c,
                                scf_ws.core.d_S, nao_s, &zero2, tmp2, nao_c);
                deviceBlasSgemm(blas_handle, DEVICE_BLAS_OP_N, DEVICE_BLAS_OP_T,
                                nao_c, nao_c, nao_s, &one2, tmp2, nao_c, C,
                                nao_c, &zero2, S_cart, nao_c);
                free(tmp2);
            }
            for (int i = 0; i < nao_c; i++)
                d_norms_cart[i] =
                    1.0f / sqrtf(fmaxf(S_cart[i * nao_c + i], 1e-20f));
            free(S_cart);

            d_norms_use = d_norms_cart;
        }

        const int chunk_size = ONE_E_BATCH_SIZE;
        for (int i = 0; i < task_ctx.topo.n_1e_tasks; i += chunk_size)
        {
            int current_chunk =
                std::min(chunk_size, task_ctx.topo.n_1e_tasks - i);
            QC_ONE_E_TASK* task_ptr = task_ctx.buffers.d_1e_tasks + i;
            Launch_Device_Kernel(
                OneE_Grad_Kernel, (current_chunk + 63) / 64, 64, 0, 0,
                current_chunk, task_ptr, mol.d_centers, mol.d_l_list,
                mol.d_exps, mol.d_coeffs, mol.d_shell_offsets,
                mol.d_shell_sizes, mol.d_ao_offsets, mol.d_atm, mol.d_env,
                mol.natm, nao_1e, grad_ws.d_shell_atom, d_P_use, d_W_use,
                d_norms_use, grad_ws.d_grad);
        }

        if (d_P_cart) free(d_P_cart);
        if (d_W_cart) free(d_W_cart);
        if (d_norms_cart) free(d_norms_cart);
    }
    _debug_print_grad("AFTER_1E", natm, grad_ws.d_grad);

    // 4. 双电子积分导数: Tr[Γ·dERI/dR]
#ifndef USE_GPU
    {
        QC_Build_ERI_Gradient_CPU(
            task_ctx, mol.nbas, mol.d_atm, mol.d_bas, mol.d_env,
            mol.d_ao_offsets, mol.d_ao_offsets_sph, scf_ws.ortho.d_norms,
            task_ctx.buffers.d_shell_pair_bounds,
            scf_ws.direct.d_pair_density_coul,
            scf_ws.direct.d_pair_density_exx,
            scf_ws.runtime.unrestricted ? scf_ws.direct.d_pair_density_exx_b
                                        : (const float*)nullptr,
            task_ctx.params.eri_shell_screen_tol, scf_ws.direct.d_P_coul,
            scf_ws.alpha.d_P,
            scf_ws.runtime.unrestricted ? scf_ws.beta.d_P
                                        : (const float*)nullptr,
            scf_ws.runtime.unrestricted ? dft.exx_fraction
                                        : (0.5f * dft.exx_fraction),
            scf_ws.runtime.unrestricted ? dft.exx_fraction : 0.0f,
            nao, mol.nao_sph, mol.is_spherical,
            cart2sph.d_cart2sph_mat, grad_ws.d_shell_atom, grad_ws.d_grad,
            task_ctx.params.eri_hr_base, task_ctx.params.eri_hr_size,
            task_ctx.params.eri_shell_buf_size,
            task_ctx.params.direct_eri_prim_screen_tol,
            scf_ws.direct.fock_thread_count);
    }
#endif
    _debug_print_grad("AFTER_2E", natm, grad_ws.d_grad);

    // 5. DFT XC 网格梯度
    if (dft.enable_dft) Build_DFT_XC_Gradient();
    _debug_print_grad("AFTER_XC", natm, grad_ws.d_grad);

    // 6. 将梯度写入 MD 力数组
    {
        const int threads = 256;
        Launch_Device_Kernel(QC_Writeback_Gradient_Kernel,
                             (natm + threads - 1) / threads, threads, 0, 0,
                             natm, d_atom_local, grad_ws.d_grad, frc);
    }
}

#include "quantum_chemistry.h"
#include "gradient/grad_one_e.hpp"
#include "gradient/grad_workspace.h"
#include "gradient/gradient.hpp"
#include "integrals/eri/common/direct_fock_kernels.hpp"
#include "integrals/eri/eri_backend.hpp"
#include "gradient/grad_eri.hpp"
#include "gradient/grad_ri.hpp"

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
    if (!std::getenv("SPONGE_SKIP_REFOCK"))
    {
        const double saved_ls = scf_ws.runtime.level_shift;
        scf_ws.runtime.level_shift = 0.0;
        Build_Fock(scf_ws.runtime.max_scf_iter);
        Diagonalize_And_Build_Density();
        scf_ws.runtime.level_shift = saved_ls;
    }

    // Cartesian norms（spherical 基组时用于 1e 和 2e 梯度 kernel）
    float* d_norms_cart = nullptr;

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

        if (std::getenv("SPONGE_DEBUG_GRAD"))
        {
            // 检查原始 norms (在 sph2cart 变换前)
            std::fprintf(stderr, "1E_GRAD: nao=%d nao_cart=%d nao_sph=%d is_sph=%d\n",
                         mol.nao, mol.nao_cart, mol.nao_sph, mol.is_spherical);
            std::fprintf(stderr, "  sph_norms[0..4]=%.6f %.6f %.6f %.6f %.6f\n",
                         scf_ws.ortho.d_norms[0], scf_ws.ortho.d_norms[1],
                         scf_ws.ortho.d_norms[2], scf_ws.ortho.d_norms[3],
                         mol.nao>4?scf_ws.ortho.d_norms[4]:0.0f);
            std::fprintf(stderr, "  S_diag[0..4]=%.4f %.4f %.4f %.4f %.4f\n",
                         scf_ws.core.d_S[0], scf_ws.core.d_S[mol.nao+1],
                         scf_ws.core.d_S[2*mol.nao+2], scf_ws.core.d_S[3*mol.nao+3],
                         mol.nao>4?scf_ws.core.d_S[4*mol.nao+4]:0.0f);
            std::fprintf(stderr, "  P[0..4]=%.4e %.4e %.4e %.4e %.4e\n",
                         d_P_use[0], d_P_use[1], d_P_use[2], d_P_use[3], d_P_use[4]);
            std::fprintf(stderr, "  ao_off[0..5]=%d %d %d %d %d %d\n",
                         mol.d_ao_offsets[0], mol.d_ao_offsets[1], mol.d_ao_offsets[2],
                         mol.d_ao_offsets[3], mol.d_ao_offsets[4],
                         mol.nbas>5?mol.d_ao_offsets[5]:0);
        }

        // 临时 Cartesian 缓冲（spherical 时使用, norms_cart 延迟到 2e 后释放）
        float* d_P_cart = nullptr;
        float* d_W_cart = nullptr;

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

            // Cartesian norms: 直接从原始 Cartesian 重叠矩阵构建
            // cart2sph.d_S_cart 保存了 Prepare_Integrals 之前的 Cartesian S
            // 注意: d_S_cart 在 Prepare_Integrals 中未被 norms 缩放
            const float* S_cart = cart2sph.d_S_cart;
            for (int i = 0; i < nao_c; i++)
                d_norms_cart[i] =
                    1.0f / sqrtf(fmaxf(S_cart[i * nao_c + i], 1e-20f));

            if (std::getenv("SPONGE_DEBUG_GRAD"))
                std::fprintf(stderr, "  norms_cart[0..4]=%.6f %.6f %.6f %.6f %.6f  S_cart_diag=%.4f %.4f\n",
                             d_norms_cart[0], d_norms_cart[1], d_norms_cart[2],
                             d_norms_cart[3], nao_c>4?d_norms_cart[4]:0.f,
                             S_cart[0], S_cart[nao_c+1]);

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
        // d_norms_cart 延迟释放 (2e gradient 也需要)
    }
    _debug_print_grad("AFTER_1E", natm, grad_ws.d_grad);

    // 4. 双电子积分导数: Tr[Γ·dERI/dR]
    // ERI gradient 在 Cartesian 中计算导数积分并用 norms 归一化。
    // 对 spherical 基组: 需要 Cartesian norms 和 Cartesian P。
    const float* d_norms_eri = mol.is_spherical ? d_norms_cart
                                                : scf_ws.ortho.d_norms;
    // Sph2Cart 变换 P_coul 和 P_exx (spherical → Cartesian)
    float* d_Pcoul_cart = nullptr;
    float* d_Pexx_a_cart = nullptr;
    const float* d_Pcoul_eri = scf_ws.direct.d_P_coul;
    const float* d_Pexx_a_eri = scf_ws.alpha.d_P;
    if (mol.is_spherical)
    {
        const int nao_c = mol.nao_cart;
        const int nao_s = mol.nao;
        const float* C = cart2sph.d_cart2sph_mat;
        d_Pcoul_cart = (float*)malloc(sizeof(float) * nao_c * nao_c);
        d_Pexx_a_cart = (float*)malloc(sizeof(float) * nao_c * nao_c);
        float* tmp = (float*)malloc(sizeof(float) * nao_c * nao_s);
        auto s2c = [&](const float* M_sph, float* M_cart) {
            const float one = 1.0f, zero = 0.0f;
            deviceBlasSgemm(blas_handle, DEVICE_BLAS_OP_N, DEVICE_BLAS_OP_N,
                            nao_c, nao_s, nao_s, &one, C, nao_c, M_sph,
                            nao_s, &zero, tmp, nao_c);
            deviceBlasSgemm(blas_handle, DEVICE_BLAS_OP_N, DEVICE_BLAS_OP_T,
                            nao_c, nao_c, nao_s, &one, tmp, nao_c, C,
                            nao_c, &zero, M_cart, nao_c);
        };
        s2c(d_Pcoul_eri, d_Pcoul_cart);
        s2c(d_Pexx_a_eri, d_Pexx_a_cart);
        free(tmp);
        d_Pcoul_eri = d_Pcoul_cart;
        d_Pexx_a_eri = d_Pexx_a_cart;
    }
#ifndef USE_GPU
    if (scf_ws.ri.enabled)
    {
        Build_RI_Gradient();
    }
    else
    {
        QC_Build_ERI_Gradient_CPU(
            task_ctx, mol.nbas, mol.d_atm, mol.d_bas, mol.d_env,
            mol.d_ao_offsets, mol.d_ao_offsets_sph, d_norms_eri,
            task_ctx.buffers.d_shell_pair_bounds,
            scf_ws.direct.d_pair_density_coul,
            scf_ws.direct.d_pair_density_exx,
            scf_ws.runtime.unrestricted ? scf_ws.direct.d_pair_density_exx_b
                                        : (const float*)nullptr,
            task_ctx.params.eri_shell_screen_tol, d_Pcoul_eri,
            d_Pexx_a_eri,
            scf_ws.runtime.unrestricted ? scf_ws.beta.d_P
                                        : (const float*)nullptr,
            scf_ws.runtime.unrestricted ? dft.exx_fraction
                                        : (0.5f * dft.exx_fraction),
            scf_ws.runtime.unrestricted ? dft.exx_fraction : 0.0f,
            mol.is_spherical ? mol.nao_cart : nao, mol.nao_sph, mol.is_spherical,
            cart2sph.d_cart2sph_mat, grad_ws.d_shell_atom, grad_ws.d_grad,
            task_ctx.params.eri_hr_base, task_ctx.params.eri_hr_size,
            task_ctx.params.eri_shell_buf_size,
            task_ctx.params.direct_eri_prim_screen_tol,
            scf_ws.direct.fock_thread_count);
    }
#endif
    if (d_norms_cart) free(d_norms_cart);
    if (d_Pcoul_cart) free(d_Pcoul_cart);
    if (d_Pexx_a_cart) free(d_Pexx_a_cart);
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

// RI (Density Fitting) 解析梯度
// 构建有效密度并调用二中心/三中心导数内核
void QUANTUM_CHEMISTRY::Build_RI_Gradient()
{
#ifndef USE_GPU
    auto& ri = scf_ws.ri;
    const int natm = mol.natm;
    const int nao = mol.nao;
    const int nao2 = mol.nao2;
    const int naux = ri.naux;

    // 目前仅支持 stored 模式
    if (ri.direct)
    {
        std::fprintf(stderr,
                     "[QC-RI] WARNING: RI gradient not yet supported in "
                     "direct mode, skipping 2e gradient.\n");
        return;
    }

    // ---- 下载所有需要的数据到 host ----
    std::vector<double> h_metric_inv_sqrt((size_t)naux * naux);
    std::vector<double> h_eri3c((size_t)naux * nao2);
    std::vector<double> h_g_vec(naux);
    std::vector<float> h_P(nao2);
    std::vector<float> h_orb_norms(nao);

    deviceMemcpy(h_metric_inv_sqrt.data(), ri.d_metric_inv_sqrt,
                 sizeof(double) * naux * naux, deviceMemcpyDeviceToHost);
    deviceMemcpy(h_eri3c.data(), ri.d_eri3c,
                 sizeof(double) * (size_t)naux * nao2,
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(h_g_vec.data(), ri.d_g_vec, sizeof(double) * naux,
                 deviceMemcpyDeviceToHost);

    // Coulomb 密度矩阵
    const float* d_P_coul = scf_ws.runtime.unrestricted
                                ? scf_ws.direct.d_Ptot
                                : scf_ws.alpha.d_P;
    deviceMemcpy(h_P.data(), d_P_coul, sizeof(float) * nao2,
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(h_orb_norms.data(), scf_ws.ortho.d_norms,
                 sizeof(float) * nao, deviceMemcpyDeviceToHost);

    // RI-K 数据
    const bool need_exx = (dft.exx_fraction != 0.0f);
    const int nocc = scf_ws.runtime.n_alpha;
    const int M = naux * nao;
    std::vector<float> h_B_occ;
    std::vector<float> h_C_occ;

    if (need_exx && nocc > 0 && ri.d_B_occ != nullptr)
    {
        // 重新计算 B_occ (用当前 alpha C)
        const float one_f = 1.0f, zero_f = 0.0f;
        deviceBlasSgemm(blas_handle, DEVICE_BLAS_OP_T, DEVICE_BLAS_OP_T, M,
                        nocc, nao, &one_f, ri.d_B, nao, scf_ws.alpha.d_C, nao,
                        &zero_f, ri.d_B_occ, M);

        h_B_occ.resize((size_t)M * nocc);
        deviceMemcpy(h_B_occ.data(), ri.d_B_occ,
                     sizeof(float) * (size_t)M * nocc,
                     deviceMemcpyDeviceToHost);

        h_C_occ.resize((size_t)nao * nao);
        deviceMemcpy(h_C_occ.data(), scf_ws.alpha.d_C,
                     sizeof(float) * (size_t)nao * nao,
                     deviceMemcpyDeviceToHost);
    }

    // 壳层到原子映射 (host)
    std::vector<int> h_shell_atom_aux(ri.naux_bas);
    for (int ish = 0; ish < ri.naux_bas; ish++)
        h_shell_atom_aux[ish] = ri.h_aux_bas[ish * 8 + 0];

    std::vector<int> h_shell_atom_orb(mol.nbas);
    for (int ish = 0; ish < mol.nbas; ish++)
        h_shell_atom_orb[ish] = mol.h_bas[ish * 8 + 0];

    // 辅助基 host 数据
    std::vector<float> h_aux_norms(naux);
    deviceMemcpy(h_aux_norms.data(), ri.d_aux_norms, sizeof(float) * naux,
                 deviceMemcpyDeviceToHost);

    // 轨道基中心 (host)
    std::vector<VECTOR> h_orb_centers(mol.nbas);
    deviceMemcpy(h_orb_centers.data(), mol.d_centers,
                 sizeof(VECTOR) * mol.nbas, deviceMemcpyDeviceToHost);

    // 轨道基参数 (host)
    std::vector<int> h_orb_l_list(mol.nbas);
    std::vector<float> h_orb_exps(mol.h_exps.size());
    std::vector<float> h_orb_coeffs(mol.h_coeffs.size());
    std::vector<int> h_orb_shell_offsets(mol.nbas);
    std::vector<int> h_orb_shell_sizes(mol.nbas);
    deviceMemcpy(h_orb_l_list.data(), mol.d_l_list, sizeof(int) * mol.nbas,
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(h_orb_exps.data(), mol.d_exps,
                 sizeof(float) * mol.h_exps.size(),
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(h_orb_coeffs.data(), mol.d_coeffs,
                 sizeof(float) * mol.h_coeffs.size(),
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(h_orb_shell_offsets.data(), mol.d_shell_offsets,
                 sizeof(int) * mol.nbas, deviceMemcpyDeviceToHost);
    deviceMemcpy(h_orb_shell_sizes.data(), mol.d_shell_sizes,
                 sizeof(int) * mol.nbas, deviceMemcpyDeviceToHost);

    // 调用 RI 梯度驱动
    QC_Build_RI_Gradient_Stored(
        natm, nao, mol.nao_cart, naux, ri.naux_cart, ri.naux_bas, mol.nbas,
        mol.is_spherical,
        // 辅助基
        ri.h_aux_centers.data(), ri.h_aux_l_list.data(), ri.h_aux_exps.data(),
        ri.h_aux_coeffs.data(), ri.h_aux_shell_offsets.data(),
        ri.h_aux_shell_sizes.data(), ri.h_aux_ao_offsets.data(),
        ri.h_aux_ao_offsets_sph.data(), h_aux_norms.data(),
        ri.h_U_aux.data(),
        // 轨道基
        h_orb_centers.data(), h_orb_l_list.data(), h_orb_exps.data(),
        h_orb_coeffs.data(), h_orb_shell_offsets.data(),
        h_orb_shell_sizes.data(), mol.h_ao_offsets.data(),
        mol.is_spherical ? mol.h_ao_offsets_sph.data()
                         : mol.h_ao_offsets.data(),
        h_orb_norms.data(),
        mol.is_spherical ? ri.h_U_orb.data() : nullptr,
        // SCF 数据
        h_metric_inv_sqrt.data(), h_eri3c.data(),
        h_g_vec.data(), h_P.data(),
        need_exx && nocc > 0 ? h_B_occ.data() : nullptr,
        need_exx && nocc > 0 ? h_C_occ.data() : nullptr, nocc,
        dft.exx_fraction,
        // 特征分解数据 (D2_K Daleckii-Kreĭn)
        ri.h_eigval.data(), ri.h_eigvec.data(), ri.naux_eff,
        // 壳层到原子映射
        h_shell_atom_aux.data(), h_shell_atom_orb.data(),
        // 输出
        grad_ws.d_grad);

    _debug_print_grad("AFTER_RI_2E", natm, grad_ws.d_grad);
#endif
}

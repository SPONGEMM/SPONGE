#include "quantum_chemistry.h"
#include "gradient/grad_one_e.hpp"
#include "gradient/grad_workspace.h"
#include "gradient/gradient.hpp"
#include "integrals/eri/common/direct_fock_kernels.hpp"
#include "integrals/eri/eri_backend.hpp"
#include "gradient/grad_eri.hpp"
#include "gradient/grad_ri.hpp"

std::vector<float> QC_Build_Cart2Sph_Mat_Host(const std::vector<int>& l_list,
                                              int nao_cart, int nao_sph);

void QUANTUM_CHEMISTRY::Compute_Gradient(VECTOR* frc, const VECTOR* crd,
                                          const VECTOR box_length,
                                          int need_virial,
                                          LTMatrix3* atom_virial)
{
    if (!is_initialized) return;
    const int natm = mol.natm;
    const int nao = mol.nao;
    const int nao2 = mol.nao2;

    // 用收敛密度重建 Fock 并对角化，获取准确的轨道能量用于 W 矩阵。
    // DIIS 外推的 Fock 矩阵特征值不够精确，且必须关闭 level shift。
    {
        const double saved_ls = scf_ws.runtime.level_shift;
        scf_ws.runtime.level_shift = 0.0;
        Build_Fock(scf_ws.runtime.max_scf_iter);
        Diagonalize_And_Build_Density();
        scf_ws.runtime.level_shift = saved_ls;
    }

    deviceMemset(grad_ws.d_grad, 0, sizeof(double) * natm * 3);

    // 1. 构建能量加权密度矩阵 W
    {
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

    // 3. 单电子积分导数: Tr[P·dH/dR] - Tr[W·dS/dR]
    {
#ifndef USE_GPU
        if (mol.is_spherical)
        {
            std::vector<int> h_shell_atom(mol.nbas);
            for (int ish = 0; ish < mol.nbas; ish++)
                h_shell_atom[ish] = mol.h_bas[ish * 8 + 0];
            const std::vector<float> h_cart2sph = QC_Build_Cart2Sph_Mat_Host(
                mol.h_l_list, mol.nao_cart, mol.nao_sph);

            QC_Build_OneE_Gradient_Spherical_CPU(
                task_ctx.topo.h_1e_tasks, mol.h_centers, mol.h_l_list,
                mol.h_exps, mol.h_coeffs, mol.h_shell_offsets,
                mol.h_shell_sizes, mol.h_ao_offsets, mol.h_ao_offsets_sph,
                mol.h_atm, mol.h_env, h_shell_atom, scf_ws.direct.d_P_coul,
                grad_ws.d_W_density, scf_ws.ortho.d_norms,
                h_cart2sph.data(), mol.natm, mol.nao_sph, grad_ws.d_grad);
        }
        else
#endif
        {
            const float* d_P_use = scf_ws.direct.d_P_coul;
            const float* d_W_use = grad_ws.d_W_density;
            const float* d_norms_use = scf_ws.ortho.d_norms;
            int nao_1e = mol.nao;

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
        }
    }

    // 4. 双电子积分导数: Tr[Γ·dERI/dR]
    // grad_eri 内部在 Cartesian shell buffer 上计算导数积分，
    // is_spherical 时内部做 cart2sph，因此始终传入 SCF AO 基的密度和 norms。
#ifndef USE_GPU
    if (scf_ws.ri.enabled)
    {
        Build_RI_Gradient();
    }
    else
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

    // 5. DFT XC 网格梯度
    if (dft.enable_dft) Build_DFT_XC_Gradient();

    // 6. 将梯度写入 MD 力数组
    {
        const int threads = 256;
        Launch_Device_Kernel(QC_Writeback_Gradient_Kernel,
                             (natm + threads - 1) / threads, threads, 0, 0,
                             natm, d_atom_local, grad_ws.d_grad, crd, frc,
                             need_virial, atom_virial);
    }
}

// RI (Density Fitting) 解析梯度
void QUANTUM_CHEMISTRY::Build_RI_Gradient()
{
#ifndef USE_GPU
    auto& ri = scf_ws.ri;
    const int natm = mol.natm;
    const int nao = mol.nao;
    const int nao2 = mol.nao2;
    const int naux = ri.naux;

    if (ri.direct)
    {
        fprintf(stderr,
                "[QC-RI] WARNING: RI gradient not yet supported in "
                "direct mode, skipping 2e gradient.\n");
        return;
    }

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

    const float* d_P_coul = scf_ws.runtime.unrestricted
                                ? scf_ws.direct.d_Ptot
                                : scf_ws.alpha.d_P;
    deviceMemcpy(h_P.data(), d_P_coul, sizeof(float) * nao2,
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(h_orb_norms.data(), scf_ws.ortho.d_norms,
                 sizeof(float) * nao, deviceMemcpyDeviceToHost);

    const bool need_exx = (dft.exx_fraction != 0.0f);
    const int nocc = scf_ws.runtime.n_alpha;
    const int M = naux * nao;
    std::vector<float> h_B_occ;
    std::vector<float> h_C_occ;

    if (need_exx && nocc > 0 && ri.d_B_occ != nullptr)
    {
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

    std::vector<int> h_shell_atom_aux(ri.naux_bas);
    for (int ish = 0; ish < ri.naux_bas; ish++)
        h_shell_atom_aux[ish] = ri.h_aux_bas[ish * 8 + 0];

    std::vector<int> h_shell_atom_orb(mol.nbas);
    for (int ish = 0; ish < mol.nbas; ish++)
        h_shell_atom_orb[ish] = mol.h_bas[ish * 8 + 0];

    std::vector<float> h_aux_norms(naux);
    deviceMemcpy(h_aux_norms.data(), ri.d_aux_norms, sizeof(float) * naux,
                 deviceMemcpyDeviceToHost);

    std::vector<VECTOR> h_orb_centers(mol.nbas);
    deviceMemcpy(h_orb_centers.data(), mol.d_centers,
                 sizeof(VECTOR) * mol.nbas, deviceMemcpyDeviceToHost);

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

    QC_Build_RI_Gradient_Stored(
        natm, nao, mol.nao_cart, naux, ri.naux_cart, ri.naux_bas, mol.nbas,
        mol.is_spherical,
        ri.h_aux_centers.data(), ri.h_aux_l_list.data(), ri.h_aux_exps.data(),
        ri.h_aux_coeffs.data(), ri.h_aux_shell_offsets.data(),
        ri.h_aux_shell_sizes.data(), ri.h_aux_ao_offsets.data(),
        ri.h_aux_ao_offsets_sph.data(), h_aux_norms.data(),
        ri.h_U_aux.data(),
        h_orb_centers.data(), h_orb_l_list.data(), h_orb_exps.data(),
        h_orb_coeffs.data(), h_orb_shell_offsets.data(),
        h_orb_shell_sizes.data(), mol.h_ao_offsets.data(),
        mol.is_spherical ? mol.h_ao_offsets_sph.data()
                         : mol.h_ao_offsets.data(),
        h_orb_norms.data(),
        mol.is_spherical ? ri.h_U_orb.data() : nullptr,
        h_metric_inv_sqrt.data(), h_eri3c.data(),
        h_g_vec.data(), h_P.data(),
        need_exx && nocc > 0 ? h_B_occ.data() : nullptr,
        need_exx && nocc > 0 ? h_C_occ.data() : nullptr, nocc,
        dft.exx_fraction,
        ri.h_eigval.data(), ri.h_eigvec.data(), ri.naux_eff,
        h_shell_atom_aux.data(), h_shell_atom_orb.data(),
        grad_ws.d_grad);
#endif
}

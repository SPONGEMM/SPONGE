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

    // 清零梯度累加器
    deviceMemset(grad_ws.d_grad, 0, sizeof(double) * natm * 3);

    // 1. 构建能量加权密度矩阵 W
    {
        float* d_D_tmp = scf_ws.alpha.d_F;
        QC_Build_Energy_Weighted_Density(
            blas_handle, nao, scf_ws.runtime.n_alpha,
            scf_ws.runtime.occ_factor, scf_ws.alpha.d_C, scf_ws.ortho.d_W,
            grad_ws.d_W_density, d_D_tmp);

        if (scf_ws.runtime.unrestricted && grad_ws.d_W_density_beta)
        {
            float* d_D_tmp_b = scf_ws.beta.d_F;
            QC_Build_Energy_Weighted_Density(
                blas_handle, nao, scf_ws.runtime.n_beta, 1.0f,
                scf_ws.beta.d_C, scf_ws.ortho.d_W,
                grad_ws.d_W_density_beta, d_D_tmp_b);
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
    {
        const float* d_P_use = scf_ws.direct.d_P_coul;
        const float* d_W_use = grad_ws.d_W_density;

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
                mol.natm, mol.nao, grad_ws.d_shell_atom, d_P_use, d_W_use,
                scf_ws.ortho.d_norms, grad_ws.d_grad);
        }
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
            0.0f, nao, mol.nao_sph, mol.is_spherical,
            cart2sph.d_cart2sph_mat, grad_ws.d_shell_atom, grad_ws.d_grad,
            task_ctx.params.eri_hr_base, task_ctx.params.eri_hr_size,
            task_ctx.params.eri_shell_buf_size,
            task_ctx.params.direct_eri_prim_screen_tol,
            scf_ws.direct.fock_thread_count);
    }
#endif
    _debug_print_grad("AFTER_2E", natm, grad_ws.d_grad);

    // 5. DFT XC 网格梯度
    // TODO: 实现 grad_xc.hpp

    // 6. 将梯度写入 MD 力数组
    {
        const int threads = 256;
        Launch_Device_Kernel(QC_Writeback_Gradient_Kernel,
                             (natm + threads - 1) / threads, threads, 0, 0,
                             natm, d_atom_local, grad_ws.d_grad, frc);
    }
}

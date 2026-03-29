#pragma once

#include "../integrals/eri/common/direct_fock_kernels.hpp"
#include "../integrals/eri/cpu/task_filter.hpp"
#include "../integrals/eri/eri_backend.hpp"

void QUANTUM_CHEMISTRY::Build_Fock(int iter)
{
    const int threads = 256;
    const int total = mol.nao2;

    if (dft.enable_dft) Build_DFT_VXC();

    Launch_Device_Kernel(QC_Init_Fock_Kernel, (total + threads - 1) / threads,
                         threads, 0, 0, total, scf_ws.core.d_H_core, dft.d_Vxc,
                         dft.enable_dft, scf_ws.alpha.d_F);
    if (scf_ws.runtime.unrestricted)
    {
        Launch_Device_Kernel(QC_Init_Fock_Kernel,
                             (total + threads - 1) / threads, threads, 0, 0,
                             total, scf_ws.core.d_H_core, dft.d_Vxc_beta,
                             dft.enable_dft, scf_ws.beta.d_F);
    }

#ifndef USE_GPU
    if (scf_ws.alpha.d_F_double)
        for (int i = 0; i < total; i++)
            scf_ws.alpha.d_F_double[i] = (double)scf_ws.alpha.d_F[i];
    if (scf_ws.beta.d_F_double && scf_ws.runtime.unrestricted)
        for (int i = 0; i < total; i++)
            scf_ws.beta.d_F_double[i] = (double)scf_ws.beta.d_F[i];
#endif

#ifdef USE_GPU
    float* d_F_build = scf_ws.alpha.d_F;
    float* d_F_b_build =
        scf_ws.runtime.unrestricted ? scf_ws.beta.d_F : (float*)nullptr;
#else
    const int thread_total = scf_ws.direct.fock_thread_count * total;
    deviceMemset(scf_ws.direct.d_F_thread, 0, sizeof(double) * thread_total);
    if (scf_ws.runtime.unrestricted)
        deviceMemset(scf_ws.direct.d_F_b_thread, 0,
                     sizeof(double) * thread_total);
    double* d_F_build = scf_ws.direct.d_F_thread;
    double* d_F_b_build = scf_ws.runtime.unrestricted
                              ? scf_ws.direct.d_F_b_thread
                              : (double*)nullptr;
#endif

    const float exx_scale_a = scf_ws.runtime.unrestricted
                                  ? dft.exx_fraction
                                  : (0.5f * dft.exx_fraction);
    const float exx_scale_b =
        scf_ws.runtime.unrestricted ? dft.exx_fraction : 0.0f;
    const bool need_exx = (dft.exx_fraction != 0.0f);

    Launch_Device_Kernel(
        QC_Build_Shell_Pair_Density_Kernel,
        (task_ctx.topo.n_shell_pairs + threads - 1) / threads, threads, 0, 0,
        task_ctx.topo.n_shell_pairs, task_ctx.buffers.d_shell_pairs,
        mol.d_ao_offsets, mol.d_ao_offsets_sph, mol.d_l_list, mol.is_spherical,
        mol.nao, scf_ws.direct.d_P_coul, scf_ws.direct.d_pair_density_coul,
        need_exx ? scf_ws.alpha.d_P : (const float*)nullptr,
        scf_ws.direct.d_pair_density_exx,
        (need_exx && scf_ws.runtime.unrestricted) ? scf_ws.beta.d_P
                                                  : (const float*)nullptr,
        scf_ws.direct.d_pair_density_exx_b);
    const float shell_screen_tol = QC_Effective_Shell_Screen_Tol(
        task_ctx.params.eri_shell_screen_tol, iter);
    const float prim_screen_tol = QC_Effective_Prim_Screen_Tol(
        task_ctx.params.direct_eri_prim_screen_tol, iter);

#ifdef USE_GPU
    QC_Build_Fock_Direct_GPU(
        task_ctx, mol.d_atm, mol.d_bas, mol.d_env, mol.d_ao_offsets,
        mol.d_ao_offsets_sph, scf_ws.ortho.d_norms,
        task_ctx.buffers.d_shell_pair_bounds, scf_ws.direct.d_pair_density_coul,
        scf_ws.direct.d_pair_density_exx,
        scf_ws.runtime.unrestricted ? scf_ws.direct.d_pair_density_exx_b
                                    : (const float*)nullptr,
        shell_screen_tol, scf_ws.direct.d_P_coul, scf_ws.alpha.d_P,
        scf_ws.runtime.unrestricted ? scf_ws.beta.d_P : (const float*)nullptr,
        exx_scale_a, exx_scale_b, mol.nao, mol.nao_sph, mol.is_spherical,
        cart2sph.d_cart2sph_mat, d_F_build, d_F_b_build,
        scf_ws.direct.d_hr_pool, prim_screen_tol);

    if (scf_ws.alpha.d_F_double != NULL)
        QC_Float_To_Double_Copy(total, scf_ws.alpha.d_F,
                                scf_ws.alpha.d_F_double);
    if (scf_ws.runtime.unrestricted && scf_ws.beta.d_F_double != NULL)
        QC_Float_To_Double_Copy(total, scf_ws.beta.d_F, scf_ws.beta.d_F_double);
#else
    QC_Build_Fock_Direct_CPU(
        task_ctx, mol.nbas, mol.d_atm, mol.d_bas, mol.d_env, mol.d_ao_offsets,
        mol.d_ao_offsets_sph, scf_ws.ortho.d_norms,
        task_ctx.buffers.d_shell_pair_bounds, scf_ws.direct.d_pair_density_coul,
        scf_ws.direct.d_pair_density_exx,
        scf_ws.runtime.unrestricted ? scf_ws.direct.d_pair_density_exx_b
                                    : (const float*)nullptr,
        shell_screen_tol, scf_ws.direct.d_P_coul, scf_ws.alpha.d_P,
        scf_ws.runtime.unrestricted ? scf_ws.beta.d_P : (const float*)nullptr,
        exx_scale_a, exx_scale_b, mol.nao, mol.nao_sph, mol.is_spherical,
        cart2sph.d_cart2sph_mat, d_F_build, d_F_b_build,
        scf_ws.direct.d_hr_pool,
        (QC_Angular_Term_CPU*)scf_ws.direct.h_cpu_bra_terms,
        (QC_Angular_Term_CPU*)scf_ws.direct.h_cpu_ket_terms,
        task_ctx.params.eri_hr_base, task_ctx.params.eri_hr_size,
        task_ctx.params.eri_shell_buf_size, prim_screen_tol,
        scf_ws.direct.fock_thread_count);
#endif

#ifndef USE_GPU
    Launch_Device_Kernel(
        QC_Reduce_Thread_Fock_Kernel, (total + threads - 1) / threads, threads,
        0, 0, total, scf_ws.direct.fock_thread_count, scf_ws.direct.d_F_thread,
        scf_ws.alpha.d_F, scf_ws.alpha.d_F_double);
    if (scf_ws.runtime.unrestricted)
    {
        Launch_Device_Kernel(QC_Reduce_Thread_Fock_Kernel,
                             (total + threads - 1) / threads, threads, 0, 0,
                             total, scf_ws.direct.fock_thread_count,
                             scf_ws.direct.d_F_b_thread, scf_ws.beta.d_F,
                             scf_ws.beta.d_F_double);
    }
#endif
}

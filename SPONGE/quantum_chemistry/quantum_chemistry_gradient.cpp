// clang-format off
#include "quantum_chemistry.h"
#include "gradient/grad_one_e.hpp"
#include "gradient/grad_workspace.h"
#include "gradient/gradient.hpp"
#include "integrals/eri/common/direct_fock_kernels.hpp"
#include "integrals/eri/common/eri_rys.hpp"
#include "integrals/eri/eri_backend.hpp"
#include "integrals/ri/ri_3center.hpp"
#include "gradient/grad_eri.hpp"
#include "gradient/grad_ri.hpp"
#include "ecp/ecp_integrals.h"
// clang-format on

std::vector<float> QC_Build_Cart2Sph_Mat_Host(const std::vector<int>& l_list,
                                              int nao_cart, int nao_sph);

static __global__ void QC_Float_Accumulate_Kernel(int n, float* dst,
                                                  const float* src)
{
    SIMPLE_DEVICE_FOR(i, n) { dst[i] += src[i]; }
}

static __global__ void QC_Weight_By_Norms_Kernel(int nao, const float* P,
                                                 const float* norms, float* out)
{
    SIMPLE_DEVICE_FOR(idx, nao * nao)
    {
        int i = idx / nao, j = idx % nao;
        out[idx] = P[idx] * norms[i] * norms[j];
    }
}

void QUANTUM_CHEMISTRY::Compute_Gradient(VECTOR* frc, const VECTOR* crd,
                                         const VECTOR box_length,
                                         int need_virial,
                                         LTMatrix3* atom_virial)
{
    if (!is_initialized) return;
    const int natm = mol.natm;
    const int nao = mol.nao;
    const int nao2 = mol.nao2;

    // 获取准确轨道能量用于 W 矩阵: 使用 SCF 缓存的 DIIS 前 Fock，
    // 避免昂贵的 Fock 重建。缓存在 Solve_SCF 每轮 Apply_DIIS 前保存。
    {
        const double saved_ls = scf_ws.runtime.level_shift;
        scf_ws.runtime.level_shift = 0.0;
        if (scf_ws.alpha.d_F_for_grad)
        {
            // 将缓存的 pre-DIIS Fock 恢复到 d_F_double，供对角化使用
            deviceMemcpy(scf_ws.alpha.d_F_double, scf_ws.alpha.d_F_for_grad,
                         sizeof(double) * nao2, deviceMemcpyDeviceToDevice);
            if (scf_ws.runtime.unrestricted && scf_ws.beta.d_F_for_grad)
                deviceMemcpy(scf_ws.beta.d_F_double, scf_ws.beta.d_F_for_grad,
                             sizeof(double) * nao2, deviceMemcpyDeviceToDevice);
        }
        else
        {
            // 回退: 无缓存时重建 Fock
            Build_Fock(1);
        }
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
            blas_handle, nao, scf_ws.runtime.n_alpha, scf_ws.runtime.occ_factor,
            scf_ws.alpha.d_C, alpha_epsilon, grad_ws.d_W_density, d_D_tmp);

        if (scf_ws.runtime.unrestricted && grad_ws.d_W_density_beta)
        {
            float* d_D_tmp_b = scf_ws.beta.d_F;
            QC_Build_Energy_Weighted_Density(
                blas_handle, nao, scf_ws.runtime.n_beta, 1.0f, scf_ws.beta.d_C,
                scf_ws.ortho.d_W, grad_ws.d_W_density_beta, d_D_tmp_b);
            Launch_Device_Kernel(QC_Float_Accumulate_Kernel, (nao2 + 255) / 256,
                                 256, 0, 0, nao2, grad_ws.d_W_density,
                                 grad_ws.d_W_density_beta);
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
                grad_ws.d_W_density, scf_ws.ortho.d_norms, h_cart2sph.data(),
                mol.natm, mol.nao_sph, grad_ws.d_grad);
        }
        else
#endif
        {
            const int n_tasks = task_ctx.topo.n_1e_tasks;
            const float* d_P_use;
            const float* d_W_use;
            const float* d_norms_use;
            int nao_1e;

            if (mol.is_spherical)
            {
                // 1e 梯度核使用 Cartesian ao_offsets, 需要匹配的密度矩阵
                const int nc = mol.nao_cart, ns = mol.nao;
                std::vector<float> h_norms(ns), h_C(ns * nc);
                std::vector<float> h_P(ns * ns), h_W(ns * ns);
                deviceMemcpy(h_norms.data(), scf_ws.ortho.d_norms,
                             sizeof(float) * ns, deviceMemcpyDeviceToHost);
                deviceMemcpy(h_C.data(), cart2sph.d_cart2sph_mat,
                             sizeof(float) * ns * nc, deviceMemcpyDeviceToHost);
                deviceMemcpy(h_P.data(), scf_ws.direct.d_P_coul,
                             sizeof(float) * ns * ns, deviceMemcpyDeviceToHost);
                deviceMemcpy(h_W.data(), grad_ws.d_W_density,
                             sizeof(float) * ns * ns, deviceMemcpyDeviceToHost);

                std::vector<float> h_Pc, h_Wc;
                QC_Sph2Cart_Density_Host(ns, nc, h_norms, h_C, h_P, h_Pc);
                QC_Sph2Cart_Density_Host(ns, nc, h_norms, h_C, h_W, h_Wc);

                deviceMemcpy(grad_ws.d_P_cart, h_Pc.data(),
                             sizeof(float) * nc * nc, deviceMemcpyHostToDevice);
                deviceMemcpy(grad_ws.d_W_cart, h_Wc.data(),
                             sizeof(float) * nc * nc, deviceMemcpyHostToDevice);

                d_P_use = grad_ws.d_P_cart;
                d_W_use = grad_ws.d_W_cart;
                d_norms_use = grad_ws.d_norms_ones;
                nao_1e = nc;
            }
            else
            {
                d_P_use = scf_ws.direct.d_P_coul;
                d_W_use = grad_ws.d_W_density;
                d_norms_use = scf_ws.ortho.d_norms;
                nao_1e = mol.nao;
            }

            Launch_Device_Kernel(OneE_ST_Grad_Kernel, (n_tasks + 63) / 64, 64,
                                 0, 0, n_tasks, task_ctx.buffers.d_1e_tasks,
                                 mol.d_centers, mol.d_l_list, mol.d_exps,
                                 mol.d_coeffs, mol.d_shell_offsets,
                                 mol.d_shell_sizes, mol.d_ao_offsets, nao_1e,
                                 grad_ws.d_shell_atom, d_P_use, d_W_use,
                                 d_norms_use, grad_ws.d_grad);

            const int v_total = n_tasks * mol.natm;
            Launch_Device_Kernel(
                OneE_V_Grad_Kernel, (v_total + 63) / 64, 64, 0, 0, n_tasks,
                task_ctx.buffers.d_1e_tasks, mol.d_centers, mol.d_l_list,
                mol.d_exps, mol.d_coeffs, mol.d_shell_offsets,
                mol.d_shell_sizes, mol.d_ao_offsets, mol.d_atm, mol.d_env,
                mol.natm, nao_1e, grad_ws.d_shell_atom, d_P_use, d_norms_use,
                grad_ws.d_grad);
        }
    }

    // 3b. ECP 梯度: 复用 1e 梯度已计算的 Cartesian 密度
    if (mol.has_ecp)
    {
        if (!mol.is_spherical)
        {
            // 非球谐基: 需要计算 P_cart = P .* (norms * norms')
            const int nc2 = mol.nao_cart * mol.nao_cart;
            Launch_Device_Kernel(QC_Weight_By_Norms_Kernel, (nc2 + 255) / 256,
                                 256, 0, 0, mol.nao_cart,
                                 scf_ws.direct.d_P_coul, scf_ws.ortho.d_norms,
                                 grad_ws.d_P_cart);
        }
        // 球谐基: grad_ws.d_P_cart 已在 1e 梯度中填充
        QC_Compute_ECP_Gradient(mol, task_ctx, grad_ws.d_shell_atom,
                                grad_ws.d_P_cart, grad_ws.d_grad);
    }

    // 4. 双电子积分导数: Tr[Γ·dERI/dR]
    // grad_eri 内部在 Cartesian shell buffer 上计算导数积分，
    // is_spherical 时内部做 cart2sph，因此始终传入 SCF AO 基的密度和 norms。
    const float grad_shell_screen_tol =
        fmaxf(task_ctx.params.eri_shell_screen_tol, 1.0e-7f);
    const float grad_prim_screen_tol =
        fmaxf(task_ctx.params.direct_eri_prim_screen_tol, 1.0e-7f);
    if (scf_ws.ri.enabled)
    {
        Build_RI_Gradient();
    }
    else
    {
#ifndef USE_GPU
        QC_Build_ERI_Gradient_CPU(task_ctx, mol, cart2sph, scf_ws,
                                  dft.exx_fraction, grad_shell_screen_tol,
                                  grad_prim_screen_tol, grad_ws.d_shell_atom,
                                  grad_ws.d_grad);
#else
        // GPU ERI gradient: reuse screening infrastructure, launch gradient
        // kernel
        {
            // 0. Refresh pair density for screening (Fock build may have
            //    left stale incremental values)
            {
                const int threads_pd = 256;
                const bool need_exx_pd = (dft.exx_fraction != 0.0f);
                Launch_Device_Kernel(
                    QC_Build_Shell_Pair_Density_Kernel,
                    (task_ctx.topo.n_shell_pairs + threads_pd - 1) / threads_pd,
                    threads_pd, 0, 0, task_ctx.topo.n_shell_pairs,
                    task_ctx.buffers.d_shell_pairs, mol.d_ao_offsets,
                    mol.d_ao_offsets_sph, mol.d_l_list, mol.is_spherical, nao,
                    scf_ws.direct.d_P_coul, scf_ws.direct.d_pair_density_coul,
                    need_exx_pd ? scf_ws.alpha.d_P : (const float*)nullptr,
                    scf_ws.direct.d_pair_density_exx,
                    (need_exx_pd && scf_ws.runtime.unrestricted)
                        ? scf_ws.beta.d_P
                        : (const float*)nullptr,
                    scf_ws.direct.d_pair_density_exx_b);
            }

            // 1. Run screening (same as Fock build)
            deviceMemset(task_ctx.buffers.d_screen_counts, 0,
                         sizeof(int) * task_ctx.topo.n_combos);

            const int cp_needed = task_ctx.topo.n_combos + 1;
            deviceMemcpy(grad_ws.d_combo_prefix_grad,
                         (void*)task_ctx.topo.combo_prefix,
                         sizeof(int) * cp_needed, deviceMemcpyHostToDevice);

            const float exx_a = scf_ws.runtime.unrestricted
                                    ? dft.exx_fraction
                                    : (0.5f * dft.exx_fraction);
            const float exx_b =
                scf_ws.runtime.unrestricted ? dft.exx_fraction : 0.0f;

            QC_Launch_Screen(
                task_ctx.topo.total_quartets, task_ctx.buffers.d_combos,
                grad_ws.d_combo_prefix_grad, task_ctx.topo.n_combos,
                task_ctx.buffers.d_sorted_pair_ids,
                task_ctx.buffers.d_shell_pairs,
                task_ctx.buffers.d_shell_pair_bounds,
                scf_ws.direct.d_pair_density_coul,
                scf_ws.direct.d_pair_density_exx,
                scf_ws.runtime.unrestricted ? scf_ws.direct.d_pair_density_exx_b
                                            : (const float*)nullptr,
                grad_shell_screen_tol, exx_a, exx_b,
                task_ctx.buffers.d_screened_tasks,
                task_ctx.buffers.d_screen_counts);

            int h_counts[QC_INTEGRAL_TASKS::MAX_COMBOS] = {};
            deviceMemcpy(h_counts, task_ctx.buffers.d_screen_counts,
                         sizeof(int) * task_ctx.topo.n_combos,
                         deviceMemcpyDeviceToHost);

            // Count total screened tasks
            int total_screened = 0;
            for (int ci = 0; ci < task_ctx.topo.n_combos; ci++)
            {
                total_screened += h_counts[ci];
            }

            if (total_screened > 0)
            {
                // 2. 清零多副本梯度缓冲 (在 Memory_Allocate 中已预分配)
                const int grad_size = natm * 3;
                const size_t copies_needed =
                    (size_t)QC_GRAD_N_COPIES * (size_t)grad_size;
                deviceMemset(grad_ws.d_grad_copies, 0,
                             sizeof(double) * copies_needed);

                const int gamma_buf_size = grad_ws.grad_gamma_buf_size;

                // 3. Launch gradient kernel per combo
                const int threads = QC_GRAD_ERI_THREADS;
                for (int ci = 0; ci < task_ctx.topo.n_combos; ci++)
                {
                    const int n = h_counts[ci];
                    if (n == 0) continue;
                    const QC_ERI_TASK* d_tasks =
                        task_ctx.buffers.d_screened_tasks +
                        task_ctx.topo.h_combos[ci].output_offset;
                    const int max_blocks_pool =
                        grad_ws.grad_gamma_pool_slots / threads;
                    const int blocks = std::min({QC_GRAD_GAMMA_POOL_BLOCKS,
                                                 (n + threads - 1) / threads,
                                                 max_blocks_pool});

                    Launch_Device_Kernel(
                        QC_ERI_Grad_Kernel, blocks, threads, 0, 0, n, d_tasks,
                        mol.d_atm, mol.d_bas, mol.d_env, mol.d_ao_offsets,
                        mol.d_ao_offsets_sph, scf_ws.ortho.d_norms,
                        task_ctx.buffers.d_shell_pair_bounds,
                        scf_ws.direct.d_pair_density_coul,
                        scf_ws.direct.d_pair_density_exx,
                        scf_ws.runtime.unrestricted
                            ? scf_ws.direct.d_pair_density_exx_b
                            : (const float*)nullptr,
                        grad_shell_screen_tol, scf_ws.direct.d_P_coul,
                        scf_ws.alpha.d_P,
                        scf_ws.runtime.unrestricted ? scf_ws.beta.d_P
                                                    : (const float*)nullptr,
                        exx_a, exx_b, nao, mol.nao_sph, mol.is_spherical,
                        cart2sph.d_cart2sph_mat, grad_ws.d_grad_gamma_pool,
                        gamma_buf_size, grad_ws.d_shell_atom,
                        grad_ws.d_grad_copies, QC_GRAD_N_COPIES, natm,
                        grad_prim_screen_tol);
                }

                // 4. Reduce gradient copies
                Launch_Device_Kernel(QC_Reduce_Grad_Copies_Kernel,
                                     (grad_size + 255) / 256, 256, 0, 0,
                                     grad_size, QC_GRAD_N_COPIES,
                                     grad_ws.d_grad_copies, grad_ws.d_grad);
            }
        }
#endif
    }

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
// 支持 stored 模式（直接下载预存的 eri3c）和 direct 模式（逐 shell pair
// 重新计算 3c 积分）。 两种模式最终都构建相同的 D3_eff / D2_eff
// 有效密度，调用相同的梯度内核。
void QUANTUM_CHEMISTRY::Build_RI_Gradient()
{
#ifndef USE_GPU
    auto& ri = scf_ws.ri;
    const int natm = mol.natm;
    const int nao = mol.nao;
    const int nao2 = mol.nao2;
    const int naux = ri.naux;

    // 复用 RI_Precompute 阶段缓存的 host metric
    const std::vector<double>& h_metric_inv_sqrt = ri.h_metric_inv_sqrt;
    std::vector<float> h_P(nao2);
    std::vector<float> h_orb_norms(nao);

    const float* d_P_coul =
        scf_ws.runtime.unrestricted ? scf_ws.direct.d_Ptot : scf_ws.alpha.d_P;
    deviceMemcpy(h_P.data(), d_P_coul, sizeof(float) * nao2,
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(h_orb_norms.data(), scf_ws.ortho.d_norms, sizeof(float) * nao,
                 deviceMemcpyDeviceToHost);

    const bool need_exx = (dft.exx_fraction != 0.0f);
    const int nocc = scf_ws.runtime.n_alpha;
    const int M = naux * nao;

    // 计算 max shell cart sizes（workspace 分配用，从 host 数据）
    int max_aux_cart = 0;
    for (int i = 0; i < ri.naux_bas; i++)
    {
        int nc = (ri.h_aux_l_list[i] + 1) * (ri.h_aux_l_list[i] + 2) / 2;
        if (nc > max_aux_cart) max_aux_cart = nc;
    }
    int max_orb_cart = 0;
    for (int i = 0; i < mol.nbas; i++)
    {
        int nc = (mol.h_l_list[i] + 1) * (mol.h_l_list[i] + 2) / 2;
        if (nc > max_orb_cart) max_orb_cart = nc;
    }

    // 启动梯度内核的公共 lambda（两种模式共用）
    auto launch_grad_kernels = [&](const std::vector<double>& D2_eff,
                                   const std::vector<double>& D3_eff)
    {
        QC_Launch_RI_Grad_Kernels(mol, ri, scf_ws.ortho.d_norms, grad_ws,
                                  max_aux_cart, max_orb_cart, D2_eff, D3_eff);
    };

    if (ri.direct)
    {
        // Direct 模式：增量累积，不存储完整 eri3c 张量
        //   Pass 1: 逐 shell pair 累积 d_vec + B_occ
        //   Pass 2 (仅 EXX): 逐 shell pair 累积 Z_K
        //   内存: O(naux·nao·nocc) + O(naux²)，而非 O(naux·nao²)

        // 复用 RI_Precompute 阶段缓存
        const std::vector<double>& h_metric_inv = ri.h_metric_inv;

        std::vector<float> h_C_occ;
        if (need_exx && nocc > 0)
        {
            h_C_occ.resize((size_t)nao * nao);
            deviceMemcpy(h_C_occ.data(), scf_ws.alpha.d_C,
                         sizeof(float) * (size_t)nao * nao,
                         deviceMemcpyDeviceToHost);
        }

        // 转换密度为 double
        std::vector<double> h_D(nao2);
        for (int i = 0; i < nao2; i++) h_D[i] = (double)h_P[i];

        // GPU 3c 缓冲
        int max_l_cart = 0;
        for (int sh = 0; sh < mol.nbas; sh++)
            if (mol.h_l_list[sh] > max_l_cart) max_l_cart = mol.h_l_list[sh];
        const int max_cart = (max_l_cart + 1) * (max_l_cart + 2) / 2;
        const long long buf_3c_size =
            (long long)ri.naux_cart * max_cart * max_cart;
        double* d_3c_buf = NULL;
        Device_Malloc_Safely((void**)&d_3c_buf, sizeof(double) * buf_3c_size);
        std::vector<QC_RI_3C_TASK> h_tasks(ri.naux_bas);
        QC_RI_3C_TASK* d_tasks = NULL;
        Device_Malloc_Safely((void**)&d_tasks,
                             sizeof(QC_RI_3C_TASK) * ri.naux_bas);
        const int threads = 256;

        // 辅助 lambda: 计算一个 shell pair 的 block_sph
        // 复用 cart2sph + 归一化逻辑
        auto compute_block_sph = [&](int mu_sh, int nu_sh, int dmc, int dnc,
                                     int dms, int dns, int off_mu_s,
                                     int off_nu_s, std::vector<double>& out)
        {
            for (int P = 0; P < ri.naux_bas; P++)
                h_tasks[P] = {P, mu_sh, nu_sh};
            deviceMemcpy(d_tasks, h_tasks.data(),
                         sizeof(QC_RI_3C_TASK) * ri.naux_bas,
                         deviceMemcpyHostToDevice);
            const long long buf_n = (long long)ri.naux_cart * dmc * dnc;
            deviceMemset(d_3c_buf, 0, sizeof(double) * buf_n);
            Launch_Device_Kernel(
                QC_RI_3Center_Kernel, (ri.naux_bas + threads - 1) / threads,
                threads, 0, 0, ri.naux_bas, d_tasks, ri.d_aux_centers,
                ri.d_aux_l_list, ri.d_aux_exps, ri.d_aux_coeffs,
                ri.d_aux_shell_offsets, ri.d_aux_shell_sizes,
                ri.d_aux_ao_offsets, mol.d_centers, mol.d_l_list, mol.d_exps,
                mol.d_coeffs, mol.d_shell_offsets, mol.d_shell_sizes,
                mol.d_ao_offsets, ri.naux_cart, dmc, dnc,
                mol.h_ao_offsets[mu_sh], mol.h_ao_offsets[nu_sh], false,
                d_3c_buf);
            std::vector<double> h_bc(buf_n);
            deviceMemcpy(h_bc.data(), d_3c_buf, sizeof(double) * buf_n,
                         deviceMemcpyDeviceToHost);

            out.assign((size_t)naux * dms * dns, 0.0);
            const int Pc = ri.naux_cart;
            if (!mol.is_spherical)
            {
                for (int ps = 0; ps < naux; ps++)
                    for (int pc = 0; pc < Pc; pc++)
                    {
                        double u = (double)ri.h_U_aux[pc * naux + ps];
                        if (u == 0.0) continue;
                        for (int ij = 0; ij < dms * dns; ij++)
                            out[ps * dms * dns + ij] +=
                                u * h_bc[(long long)pc * dmc * dnc + ij];
                    }
            }
            else
            {
                std::vector<double> t1(Pc * dmc * dns, 0.0);
                for (int P = 0; P < Pc; P++)
                    for (int i = 0; i < dmc; i++)
                        for (int js = 0; js < dns; js++)
                            for (int jc = 0; jc < dnc; jc++)
                                t1[P * dmc * dns + i * dns + js] +=
                                    h_bc[(long long)P * dmc * dnc + i * dnc +
                                         jc] *
                                    (double)ri.h_U_orb
                                        [(mol.h_ao_offsets[nu_sh] + jc) * nao +
                                         off_nu_s + js];
                std::vector<double> t2(Pc * dms * dns, 0.0);
                for (int P = 0; P < Pc; P++)
                    for (int is_ = 0; is_ < dms; is_++)
                        for (int js = 0; js < dns; js++)
                            for (int ic = 0; ic < dmc; ic++)
                                t2[P * dms * dns + is_ * dns + js] +=
                                    (double)ri.h_U_orb
                                        [(mol.h_ao_offsets[mu_sh] + ic) * nao +
                                         off_mu_s + is_] *
                                    t1[P * dmc * dns + ic * dns + js];
                for (int ps = 0; ps < naux; ps++)
                    for (int pc = 0; pc < Pc; pc++)
                    {
                        double u = (double)ri.h_U_aux[pc * naux + ps];
                        if (u == 0.0) continue;
                        for (int mn = 0; mn < dms * dns; mn++)
                            out[ps * dms * dns + mn] +=
                                u * t2[pc * dms * dns + mn];
                    }
            }
            for (int P = 0; P < naux; P++)
            {
                double ps = (double)ri.h_aux_norms[P];
                for (int i = 0; i < dms; i++)
                {
                    double ms = ps * (double)h_orb_norms[off_mu_s + i];
                    for (int j = 0; j < dns; j++)
                        out[P * dms * dns + i * dns + j] *=
                            ms * (double)h_orb_norms[off_nu_s + j];
                }
            }
        };

        // Pass 1: 累积 d_vec 和 B_occ, 同时缓存 3c 积分块
        std::vector<double> h_d_vec(naux, 0.0);
        std::vector<double> h_B_occ_d((size_t)M * nocc, 0.0);

        // 缓存 3c 积分块以复用于 Pass 2 (避免重复 GPU kernel 调用)
        const int n_shell_pairs = mol.nbas * (mol.nbas + 1) / 2;
        std::vector<std::vector<double>> blk_cache(
            need_exx && nocc > 0 ? n_shell_pairs : 0);

        for (int mu_sh = 0; mu_sh < mol.nbas; mu_sh++)
        {
            const int l_mu = mol.h_l_list[mu_sh];
            const int dmc = (l_mu + 1) * (l_mu + 2) / 2;
            const int dms = mol.is_spherical ? (2 * l_mu + 1) : dmc;
            const int off_mu_s = mol.is_spherical ? mol.h_ao_offsets_sph[mu_sh]
                                                  : mol.h_ao_offsets[mu_sh];
            for (int nu_sh = 0; nu_sh <= mu_sh; nu_sh++)
            {
                const int l_nu = mol.h_l_list[nu_sh];
                const int dnc = (l_nu + 1) * (l_nu + 2) / 2;
                const int dns = mol.is_spherical ? (2 * l_nu + 1) : dnc;
                const int off_nu_s = mol.is_spherical
                                         ? mol.h_ao_offsets_sph[nu_sh]
                                         : mol.h_ao_offsets[nu_sh];

                std::vector<double> blk;
                compute_block_sph(mu_sh, nu_sh, dmc, dnc, dms, dns, off_mu_s,
                                  off_nu_s, blk);

                // d_vec[P] += block·D
                for (int P = 0; P < naux; P++)
                    for (int i = 0; i < dms; i++)
                        for (int j = 0; j < dns; j++)
                        {
                            double v = blk[P * dms * dns + i * dns + j];
                            h_d_vec[P] +=
                                v * h_D[(off_mu_s + i) * nao + (off_nu_s + j)];
                            if (mu_sh != nu_sh)
                                h_d_vec[P] +=
                                    v *
                                    h_D[(off_nu_s + j) * nao + (off_mu_s + i)];
                        }

                // B_occ 累积 (仅 EXX)
                if (need_exx && nocc > 0)
                {
                    // B_block = metric_inv_sqrt @ block
                    std::vector<double> B_blk(naux * dms * dns, 0.0);
                    for (int P = 0; P < naux; P++)
                        for (int Q = 0; Q < naux; Q++)
                        {
                            double w = h_metric_inv_sqrt[(size_t)P * naux + Q];
                            if (w == 0.0) continue;
                            for (int mn = 0; mn < dms * dns; mn++)
                                B_blk[P * dms * dns + mn] +=
                                    w * blk[Q * dms * dns + mn];
                        }
                    // B_occ[(P*nao+μ) + M*oc] += Σ_j B_blk[P,i,j] * C[ν,oc]
                    for (int P = 0; P < naux; P++)
                        for (int i = 0; i < dms; i++)
                            for (int j = 0; j < dns; j++)
                            {
                                double b = B_blk[P * dms * dns + i * dns + j];
                                if (b == 0.0) continue;
                                int mu_idx = off_mu_s + i;
                                int nu_idx = off_nu_s + j;
                                for (int oc = 0; oc < nocc; oc++)
                                {
                                    h_B_occ_d[(size_t)(P * nao + mu_idx) +
                                              (size_t)M * oc] +=
                                        b * (double)h_C_occ[nu_idx * nao + oc];
                                    if (mu_sh != nu_sh)
                                        h_B_occ_d[(size_t)(P * nao + nu_idx) +
                                                  (size_t)M * oc] +=
                                            b *
                                            (double)h_C_occ[mu_idx * nao + oc];
                                }
                            }

                    // 缓存 blk 用于 Pass 2
                    const int pair_idx = mu_sh * (mu_sh + 1) / 2 + nu_sh;
                    blk_cache[pair_idx] = std::move(blk);
                }
            }
        }

        // g_vec = metric_inv @ d_vec
        std::vector<double> h_g(naux, 0.0);
        for (int P = 0; P < naux; P++)
            for (int Q = 0; Q < naux; Q++)
                h_g[P] += h_metric_inv[(size_t)P * naux + Q] * h_d_vec[Q];

        // Pass 2 (仅 EXX): 累积 Z_K (复用缓存的 3c 积分块)
        std::vector<double> h_Z_K;
        if (need_exx && nocc > 0)
        {
            // R[P', oc, m]
            std::vector<double> R((size_t)naux * nocc * nao, 0.0);
            for (int Pp = 0; Pp < naux; Pp++)
                for (int oc = 0; oc < nocc; oc++)
                    for (int m = 0; m < nao; m++)
                    {
                        double sum = 0.0;
                        for (int n = 0; n < nao; n++)
                            sum += h_B_occ_d[(size_t)(Pp * nao + n) +
                                             (size_t)M * oc] *
                                   h_D[m * nao + n];
                        R[(long long)Pp * nocc * nao + oc * nao + m] = sum;
                    }

            h_Z_K.assign((size_t)naux * naux, 0.0);

            const int max_sh_sph =
                mol.is_spherical ? (2 * max_l_cart + 1) : max_cart;
            std::vector<double> T((size_t)naux * max_sh_sph * nocc);
            std::vector<double> Tt((size_t)naux * max_sh_sph * nocc);

            for (int mu_sh = 0; mu_sh < mol.nbas; mu_sh++)
            {
                const int l_mu = mol.h_l_list[mu_sh];
                const int dms = mol.is_spherical
                                    ? (2 * l_mu + 1)
                                    : ((l_mu + 1) * (l_mu + 2) / 2);
                const int off_mu_s = mol.is_spherical
                                         ? mol.h_ao_offsets_sph[mu_sh]
                                         : mol.h_ao_offsets[mu_sh];
                for (int nu_sh = 0; nu_sh <= mu_sh; nu_sh++)
                {
                    const int l_nu = mol.h_l_list[nu_sh];
                    const int dns = mol.is_spherical
                                        ? (2 * l_nu + 1)
                                        : ((l_nu + 1) * (l_nu + 2) / 2);
                    const int off_nu_s = mol.is_spherical
                                             ? mol.h_ao_offsets_sph[nu_sh]
                                             : mol.h_ao_offsets[nu_sh];

                    // 从缓存取出 blk (避免重复计算 3c 积分)
                    const int pair_idx = mu_sh * (mu_sh + 1) / 2 + nu_sh;
                    const std::vector<double>& blk = blk_cache[pair_idx];

                    // T[Q, i, oc] = Σ_j blk[Q,i,j] * C[off_nu+j, oc]
                    const size_t T_size = (size_t)naux * dms * nocc;
                    std::fill_n(T.begin(), T_size, 0.0);
                    for (int Q = 0; Q < naux; Q++)
                        for (int i = 0; i < dms; i++)
                            for (int j = 0; j < dns; j++)
                            {
                                double v = blk[Q * dms * dns + i * dns + j];
                                if (v == 0.0) continue;
                                for (int oc = 0; oc < nocc; oc++)
                                    T[(long long)Q * dms * nocc + i * nocc +
                                      oc] +=
                                        v *
                                        (double)
                                            h_C_occ[(off_nu_s + j) * nao + oc];
                            }

                    // Z_K[P',Q'] += -Σ_{i,oc} R[P',oc,off_mu+i] * T[Q',i,oc]
                    for (int Pp = 0; Pp < naux; Pp++)
                        for (int Q = 0; Q < naux; Q++)
                        {
                            double z = 0.0;
                            for (int i = 0; i < dms; i++)
                                for (int oc = 0; oc < nocc; oc++)
                                    z += R[(long long)Pp * nocc * nao +
                                           oc * nao + (off_mu_s + i)] *
                                         T[(long long)Q * dms * nocc +
                                           i * nocc + oc];
                            h_Z_K[(size_t)Pp * naux + Q] -= z;
                        }

                    if (mu_sh != nu_sh)
                    {
                        const size_t Tt_size = (size_t)naux * dns * nocc;
                        std::fill_n(Tt.begin(), Tt_size, 0.0);
                        for (int Q = 0; Q < naux; Q++)
                            for (int i = 0; i < dms; i++)
                                for (int j = 0; j < dns; j++)
                                {
                                    double v = blk[Q * dms * dns + i * dns + j];
                                    if (v == 0.0) continue;
                                    for (int oc = 0; oc < nocc; oc++)
                                        Tt[(long long)Q * dns * nocc +
                                           j * nocc + oc] +=
                                            v *
                                            (double)
                                                h_C_occ[(off_mu_s + i) * nao +
                                                        oc];
                                }
                        for (int Pp = 0; Pp < naux; Pp++)
                            for (int Q = 0; Q < naux; Q++)
                            {
                                double z = 0.0;
                                for (int j = 0; j < dns; j++)
                                    for (int oc = 0; oc < nocc; oc++)
                                        z += R[(long long)Pp * nocc * nao +
                                               oc * nao + (off_nu_s + j)] *
                                             Tt[(long long)Q * dns * nocc +
                                                j * nocc + oc];
                                h_Z_K[(size_t)Pp * naux + Q] -= z;
                            }
                    }
                }
            }
            blk_cache.clear();  // 释放缓存
        }

        // B_occ: double -> float (deferred until after Pass 2 which uses
        // h_B_occ_d)
        std::vector<float> h_B_occ;
        if (need_exx && nocc > 0)
        {
            h_B_occ.resize((size_t)M * nocc);
            for (size_t idx = 0; idx < (size_t)M * nocc; idx++)
                h_B_occ[idx] = (float)h_B_occ_d[idx];
        }

        deviceFree(d_tasks);
        deviceFree(d_3c_buf);

        // 构建 D3_eff 和 D2_eff
        std::vector<double> D3_eff;
        QC_Build_D3_eff(nao, naux, h_g.data(), h_P.data(),
                        h_metric_inv_sqrt.data(),
                        need_exx && nocc > 0 ? h_B_occ.data() : nullptr,
                        need_exx && nocc > 0 ? h_C_occ.data() : nullptr, nocc,
                        dft.exx_fraction, D3_eff);
        std::vector<double> D2_eff;
        QC_Build_D2_eff_FromZK(naux, h_g.data(),
                               need_exx && nocc > 0 ? h_B_occ.data() : nullptr,
                               nocc, dft.exx_fraction,
                               need_exx && nocc > 0 ? h_Z_K.data() : nullptr,
                               ri.h_eigval.data(), ri.h_eigvec.data(), D2_eff);

        launch_grad_kernels(D2_eff, D3_eff);
    }
    else
    {
        // Stored 模式：下载预存的 eri3c，调用原始函数
        std::vector<double> h_eri3c((size_t)naux * nao2);
        deviceMemcpy(h_eri3c.data(), ri.d_eri3c,
                     sizeof(double) * (size_t)naux * nao2,
                     deviceMemcpyDeviceToHost);

        // h_g_vec 仅 stored 模式需要 (direct 模式自行计算 h_g)
        std::vector<double> h_g_vec(naux);
        deviceMemcpy(h_g_vec.data(), ri.d_g_vec, sizeof(double) * naux,
                     deviceMemcpyDeviceToHost);

        std::vector<float> h_B_occ;
        std::vector<float> h_C_occ;
        if (need_exx && nocc > 0 && ri.d_B != nullptr && ri.d_B_occ != nullptr)
        {
            const float one_f = 1.0f, zero_f = 0.0f;
            deviceBlasSgemm(blas_handle, DEVICE_BLAS_OP_T, DEVICE_BLAS_OP_T, M,
                            nocc, nao, &one_f, ri.d_B, nao, scf_ws.alpha.d_C,
                            nao, &zero_f, ri.d_B_occ, M);
            h_B_occ.resize((size_t)M * nocc);
            deviceMemcpy(h_B_occ.data(), ri.d_B_occ,
                         sizeof(float) * (size_t)M * nocc,
                         deviceMemcpyDeviceToHost);
            h_C_occ.resize((size_t)nao * nao);
            deviceMemcpy(h_C_occ.data(), scf_ws.alpha.d_C,
                         sizeof(float) * (size_t)nao * nao,
                         deviceMemcpyDeviceToHost);
        }

        // 构建 D3_eff 和 D2_eff
        std::vector<double> D3_eff;
        QC_Build_D3_eff(nao, naux, h_g_vec.data(), h_P.data(),
                        h_metric_inv_sqrt.data(),
                        need_exx && nocc > 0 ? h_B_occ.data() : nullptr,
                        need_exx && nocc > 0 ? h_C_occ.data() : nullptr, nocc,
                        dft.exx_fraction, D3_eff);
        std::vector<double> D2_eff;
        QC_Build_D2_eff_Stored(
            nao, naux, h_g_vec.data(), h_eri3c.data(), h_P.data(),
            need_exx && nocc > 0 ? h_B_occ.data() : nullptr,
            need_exx && nocc > 0 ? h_C_occ.data() : nullptr, nocc,
            dft.exx_fraction, ri.h_eigval.data(), ri.h_eigvec.data(), D2_eff);

        launch_grad_kernels(D2_eff, D3_eff);
    }
#endif
}

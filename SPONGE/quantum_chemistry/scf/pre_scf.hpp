#pragma once

// ============================ 坐标同步 ===========================
// 从 MD 坐标更新 QC 的原子环境与壳层中心（含周期边界修正）
// ================================================================
static __global__ void QC_Update_Env_From_Crd_Kernel(
    const int natm, const int* atom_local, const VECTOR* crd, const int* atm,
    float* env, const float to_bohr, const VECTOR box_length)
{
    SIMPLE_DEVICE_FOR(i, natm)
    {
        const int md_idx = atom_local[i];
        const VECTOR r = crd[md_idx];
        const int ptr_coord = atm[i * 6 + 1];
        const VECTOR prev(env[ptr_coord + 0] / to_bohr,
                          env[ptr_coord + 1] / to_bohr,
                          env[ptr_coord + 2] / to_bohr);
        const VECTOR dr = Get_Periodic_Displacement(r, prev, box_length);
        env[ptr_coord + 0] = (prev.x + dr.x) * to_bohr;
        env[ptr_coord + 1] = (prev.y + dr.y) * to_bohr;
        env[ptr_coord + 2] = (prev.z + dr.z) * to_bohr;
    }
}

static __global__ void QC_Update_Centers_From_Env_Kernel(const int nbas,
                                                         const int* bas,
                                                         const int* atm,
                                                         const float* env,
                                                         VECTOR* centers)
{
    SIMPLE_DEVICE_FOR(ish, nbas)
    {
        const int iatm = bas[ish * 8 + 0];
        const int ptr_coord = atm[iatm * 6 + 1];
        centers[ish] = {env[ptr_coord + 0], env[ptr_coord + 1],
                        env[ptr_coord + 2]};
    }
}

void QUANTUM_CHEMISTRY::Update_Coordinates_From_MD(const VECTOR* crd,
                                                   const VECTOR box_length)
{
    const int threads = 256;
    Launch_Device_Kernel(QC_Update_Env_From_Crd_Kernel,
                         (mol.natm + threads - 1) / threads, threads, 0, 0,
                         mol.natm, d_atom_local, crd, mol.d_atm, mol.d_env,
                         CONSTANT_ANGSTROM_TO_BOHR, box_length);
    Launch_Device_Kernel(QC_Update_Centers_From_Env_Kernel,
                         (mol.nbas + threads - 1) / threads, threads, 0, 0,
                         mol.nbas, mol.d_bas, mol.d_atm, mol.d_env,
                         mol.d_centers);
}

// ========================== SCF 状态重置 =========================
// 清零收敛标志、能量缓存并重置 DIIS 历史
// 保留密度矩阵 P（复用上一步 MD 的收敛密度作为初猜）
// ================================================================
void QUANTUM_CHEMISTRY::Reset_SCF_State()
{
    scf_ws.diis.diis_hist_count = scf_ws.diis.diis_hist_head = 0;
    scf_ws.diis.diis_hist_count_b = scf_ws.diis.diis_hist_head_b = 0;

    deviceMemset(scf_ws.core.d_scf_energy, 0, sizeof(double));
    deviceMemset(scf_ws.runtime.d_prev_energy, 0, sizeof(double));
    deviceMemset(scf_ws.runtime.d_delta_e, 0, sizeof(double));
    deviceMemset(scf_ws.runtime.d_density_residual, 0, sizeof(double));
    deviceMemset(scf_ws.runtime.d_e, 0, sizeof(double));
    if (scf_ws.runtime.unrestricted)
        deviceMemset(scf_ws.runtime.d_e_b, 0, sizeof(double));
    deviceMemset(scf_ws.runtime.d_pvxc, 0, sizeof(double));
    deviceMemset(scf_ws.runtime.d_converged, 0, sizeof(int));
}

// =========================== 单电子积分 ===========================
// 计算 S/T/V 单电子积分，并在球谐基下执行笛卡尔到球谐变换
// ================================================================
void QUANTUM_CHEMISTRY::Compute_OneE_Integrals()
{
    const int nao_c = mol.nao_cart;
    float* p_S = mol.is_spherical ? cart2sph.d_S_cart : scf_ws.core.d_S;
    float* p_T = mol.is_spherical ? cart2sph.d_T_cart : scf_ws.core.d_T;
    float* p_V = mol.is_spherical ? cart2sph.d_V_cart : scf_ws.core.d_V;

    deviceMemset(p_S, 0, sizeof(float) * nao_c * nao_c);
    deviceMemset(p_T, 0, sizeof(float) * nao_c * nao_c);
    deviceMemset(p_V, 0, sizeof(float) * nao_c * nao_c);

    const int chunk_size = ONE_E_BATCH_SIZE;
    for (int i = 0; i < task_ctx.topo.n_1e_tasks; i += chunk_size)
    {
        int current_chunk = std::min(chunk_size, task_ctx.topo.n_1e_tasks - i);
        QC_ONE_E_TASK* task_ptr = task_ctx.buffers.d_1e_tasks + i;
        Launch_Device_Kernel(
            OneE_Kernel, (current_chunk + 63) / 64, 64, 0, 0, current_chunk,
            task_ptr, mol.d_centers, mol.d_l_list, mol.d_exps, mol.d_coeffs,
            mol.d_shell_offsets, mol.d_shell_sizes, mol.d_ao_offsets, mol.d_atm,
            mol.d_env, mol.natm, p_S, p_T, p_V, nao_c);
    }
    Cart2Sph_OneE_Integrals();
}

// ============================ 核排斥能 ===========================
// 累加核间库仑排斥能，结果写入设备侧 d_nuc_energy_dev
// ================================================================
static __global__ void QC_Accumulate_Nuclear_Repulsion_Kernel(
    const int natm, const int* z_nuc, const int* atm, const float* env,
    double* e_nuc, const VECTOR box_length)
{
    SIMPLE_DEVICE_FOR(i, natm)
    {
        const int ptr_i = atm[i * 6 + 1];
        const double zi = (double)z_nuc[i];
        const VECTOR ri(env[ptr_i + 0], env[ptr_i + 1], env[ptr_i + 2]);
        double local = 0.0;
        for (int j = i + 1; j < natm; j++)
        {
            const int ptr_j = atm[j * 6 + 1];
            const double zj = (double)z_nuc[j];
            const VECTOR rj(env[ptr_j + 0], env[ptr_j + 1], env[ptr_j + 2]);
            const VECTOR dr = Get_Periodic_Displacement(ri, rj, box_length);
            const double r = sqrt((double)dr.x * dr.x + (double)dr.y * dr.y +
                                  (double)dr.z * dr.z);
            local += zi * zj / fmax(r, 1e-12);
        }
        atomicAdd(e_nuc, local);
    }
}

void QUANTUM_CHEMISTRY::Compute_Nuclear_Repulsion(const VECTOR box_length)
{
    deviceMemset(scf_ws.core.d_nuc_energy_dev, 0, sizeof(double));
    const int threads = 256;
    const VECTOR box_bohr(box_length.x * CONSTANT_ANGSTROM_TO_BOHR,
                          box_length.y * CONSTANT_ANGSTROM_TO_BOHR,
                          box_length.z * CONSTANT_ANGSTROM_TO_BOHR);
    Launch_Device_Kernel(QC_Accumulate_Nuclear_Repulsion_Kernel,
                         (mol.natm + threads - 1) / threads, threads, 0, 0,
                         mol.natm, mol.d_Z, mol.d_atm, mol.d_env,
                         scf_ws.core.d_nuc_energy_dev, box_bohr);
}

// =========================== 积分预处理 ===========================
// 归一化单电子积分并构建 Hcore；双电子积分在 Build_Fock 中 direct 计算
// ================================================================
static __global__ void QC_Build_Norms_From_S_Kernel(const int nao,
                                                    const float* S,
                                                    float* norms)
{
    SIMPLE_DEVICE_FOR(i, nao)
    {
        float sii = S[i * nao + i];
        norms[i] = 1.0f / sqrtf(fmaxf(sii, 1e-20f));
    }
}

static __global__ void QC_Scale_OneE_And_Build_Hcore_Kernel(const int nao,
                                                            const float* norms,
                                                            float* S, float* T,
                                                            float* V,
                                                            float* H_core)
{
    const int total = nao * nao;
    SIMPLE_DEVICE_FOR(idx, total)
    {
        int i = idx / nao;
        int j = idx - i * nao;
        float scale = norms[i] * norms[j];
        S[idx] *= scale;
        T[idx] *= scale;
        V[idx] *= scale;
        H_core[idx] = T[idx] + V[idx];
    }
}

void QUANTUM_CHEMISTRY::Prepare_Integrals()
{
    const int nao = mol.nao;
    const int nao2 = mol.nao2;
    const int threads = 256;

    // 单电子积分归一化合并至 Hcore
    Launch_Device_Kernel(QC_Build_Norms_From_S_Kernel,
                         (nao + threads - 1) / threads, threads, 0, 0, nao,
                         scf_ws.core.d_S, scf_ws.ortho.d_norms);
    Launch_Device_Kernel(QC_Scale_OneE_And_Build_Hcore_Kernel,
                         (nao2 + threads - 1) / threads, threads, 0, 0, nao,
                         scf_ws.ortho.d_norms, scf_ws.core.d_S, scf_ws.core.d_T,
                         scf_ws.core.d_V, scf_ws.core.d_H_core);

    if (task_ctx.topo.n_shell_pairs <= 0) return;

    int chunk_size = ERI_BATCH_SIZE;
#ifndef USE_GPU
    chunk_size = std::max(1, task_ctx.topo.n_shell_pairs);
#endif
    for (int i = 0; i < task_ctx.topo.n_shell_pairs; i += chunk_size)
    {
        const int current_chunk =
            std::min(chunk_size, task_ctx.topo.n_shell_pairs - i);
        Launch_Device_Kernel(
            QC_Build_Shell_Pair_Bounds_Kernel,
            (current_chunk + threads - 1) / threads, threads, 0, 0,
            current_chunk, task_ctx.buffers.d_shell_pairs + i, mol.d_atm,
            mol.d_bas, mol.d_env, mol.d_ao_offsets, mol.d_ao_offsets_sph,
            scf_ws.ortho.d_norms, mol.is_spherical, cart2sph.d_cart2sph_mat,
            mol.nao_sph, task_ctx.buffers.d_shell_pair_bounds + i,
            scf_ws.direct.d_hr_pool, task_ctx.params.eri_hr_base,
            task_ctx.params.eri_hr_size, task_ctx.params.eri_shell_buf_size,
            task_ctx.params.eri_prim_screen_tol);
    }
    task_ctx.topo.h_shell_pair_bounds.resize(
        (size_t)task_ctx.topo.n_shell_pairs);
    deviceMemcpy(task_ctx.topo.h_shell_pair_bounds.data(),
                 task_ctx.buffers.d_shell_pair_bounds,
                 sizeof(float) * task_ctx.topo.n_shell_pairs,
                 deviceMemcpyDeviceToHost);
}

// ========================= 重叠正交化矩阵 =========================
// 对重叠矩阵 S 做 double 精度本征分解，并构建正交化变换矩阵 X
// ================================================================
void QUANTUM_CHEMISTRY::Build_Overlap_X()
{
    const int nao = mol.nao;
    const int nao2 = mol.nao2;

    QC_Float_To_Double(nao2, scf_ws.core.d_S, scf_ws.ortho.d_dwork_nao2_1);

    int info = 0;
    QC_Diagonalize_Double(solver_handle, nao, scf_ws.ortho.d_dwork_nao2_1,
                          scf_ws.ortho.d_dW_double,
                          scf_ws.ortho.d_solver_work_double,
                          scf_ws.ortho.lwork_double, &info);

    QC_Double_To_Float(nao, scf_ws.ortho.d_dW_double, scf_ws.ortho.d_W);

    std::vector<double> h_W(nao);
    deviceMemcpy(h_W.data(), scf_ws.ortho.d_dW_double, sizeof(double) * nao,
                 deviceMemcpyDeviceToHost);
    const double lindep_thresh = scf_ws.ortho.lindep_threshold;
    int nao_eff = 0;
    for (int k = 0; k < nao; k++)
        if (h_W[k] >= lindep_thresh) nao_eff++;
    scf_ws.ortho.nao_eff = nao_eff;

    deviceMemset(scf_ws.ortho.d_X, 0, sizeof(double) * nao2);
    QC_Build_X_Canonical(nao, nao_eff, scf_ws.ortho.d_dwork_nao2_1,
                         scf_ws.ortho.d_dW_double, lindep_thresh,
                         scf_ws.ortho.d_X);
}

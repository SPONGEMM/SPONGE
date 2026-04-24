#pragma once

#include "../ecp/ecp_integrals.h"

// 坐标同步
// 从 MD 坐标更新 QC 的原子环境与壳层中心（含周期边界修正）
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

static __global__ void QC_Update_Atom_Coords_From_Env_Kernel(
    const int natm, const int* atm, const float* env, VECTOR* atom_coords)
{
    SIMPLE_DEVICE_FOR(iat, natm)
    {
        const int ptr_coord = atm[iat * 6 + 1];
        atom_coords[iat] = {env[ptr_coord + 0], env[ptr_coord + 1],
                            env[ptr_coord + 2]};
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
    // 同步原子坐标 (按原子索引, ECP 内核使用)
    Launch_Device_Kernel(QC_Update_Atom_Coords_From_Env_Kernel,
                         (mol.natm + threads - 1) / threads, threads, 0, 0,
                         mol.natm, mol.d_atm, mol.d_env, mol.d_atom_coords);

    // RI: 同步辅助基坐标（原子相同，atm 格式相同）
    if (scf_ws.ri.enabled)
    {
        auto& ri = scf_ws.ri;
        Launch_Device_Kernel(
            QC_Update_Env_From_Crd_Kernel, (mol.natm + threads - 1) / threads,
            threads, 0, 0, mol.natm, d_atom_local, crd, ri.d_aux_atm,
            ri.d_aux_env, CONSTANT_ANGSTROM_TO_BOHR, box_length);
        Launch_Device_Kernel(QC_Update_Centers_From_Env_Kernel,
                             (ri.naux_bas + threads - 1) / threads, threads, 0,
                             0, ri.naux_bas, ri.d_aux_bas, ri.d_aux_atm,
                             ri.d_aux_env, ri.d_aux_centers);
    }
}

// SCF 状态重置
// 清零收敛标志、能量缓存并重置 DIIS 历史
// 保留密度矩阵 P（复用上一步 MD 的收敛密度作为初猜）
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

    // Reset incremental Fock state
    const size_t fb = sizeof(float) * mol.nao2;
    const size_t db = sizeof(double) * mol.nao2;
    if (scf_ws.direct.d_P_coul_prev)
        deviceMemset(scf_ws.direct.d_P_coul_prev, 0, fb);
    if (scf_ws.direct.d_F_eri_accum)
        deviceMemset(scf_ws.direct.d_F_eri_accum, 0, db);
    if (scf_ws.direct.d_F_eri_accum_f)
        deviceMemset(scf_ws.direct.d_F_eri_accum_f, 0, fb);
    if (scf_ws.direct.d_P_exx_prev)
        deviceMemset(scf_ws.direct.d_P_exx_prev, 0, fb);
    if (scf_ws.direct.d_P_exx_b_prev)
        deviceMemset(scf_ws.direct.d_P_exx_b_prev, 0, fb);
    if (scf_ws.direct.d_F_eri_b_accum)
        deviceMemset(scf_ws.direct.d_F_eri_b_accum, 0, db);
    if (scf_ws.direct.d_F_eri_b_accum_f)
        deviceMemset(scf_ws.direct.d_F_eri_b_accum_f, 0, fb);
}

// 单电子积分
// 计算 S/T/V 单电子积分，并在球谐基下执行笛卡尔到球谐变换
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

// ECP 矩阵
// 计算 V_ECP 并变换到有效基下, 归一化后加入 H_core
void QUANTUM_CHEMISTRY::Compute_ECP_Matrix()
{
    if (!mol.has_ecp) return;

    const int nao_c = mol.nao_cart;
    const int nao = mol.nao;

    // 在 Cartesian 基下计算 V_ECP
    float* d_V_ECP_cart = scf_ws.core.d_V_ECP;
    if (mol.is_spherical)
    {
        // 复用 cart2sph 临时缓冲 (d_V_cart)
        d_V_ECP_cart = cart2sph.d_V_cart;
    }
    deviceMemset(d_V_ECP_cart, 0, sizeof(float) * nao_c * nao_c);
    QC_Compute_V_ECP(mol, task_ctx, d_V_ECP_cart);

    // 球谐变换
    if (mol.is_spherical)
        Cart2Sph_Single_Matrix(d_V_ECP_cart, scf_ws.core.d_V_ECP);
}

// 核排斥能
// 累加核间库仑排斥能，结果写入设备侧 d_nuc_energy_dev
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

// 积分预处理
// 归一化单电子积分并构建 Hcore；双电子积分在 Build_Fock 中 direct 计算

// 解析计算 AO 归一化因子 — 不依赖 1e 积分结果
// 对于同壳同中心的对角重叠，S_μμ 有解析公式：
//   S_μμ = F(lx,ly,lz) × Σ_{p,q} c_p c_q (π/γ)^{3/2} / (2γ)^L
// 其中 F = (2lx-1)!!(2ly-1)!!(2lz-1)!!，γ = α_p + α_q
//
// 球谐基下：S_μμ^sph = Σ_a c2s[a,μ]² × S_aa^cart
static __global__ void QC_Compute_Analytical_Norms_Kernel(
    const int nao_eff, const int nbas, const int* l_list, const float* exps,
    const float* coeffs, const int* shell_offsets, const int* shell_sizes,
    const int* ao_offsets_cart, const int* ao_offsets_sph,
    const int is_spherical, const float* cart2sph_mat, const int nao_sph,
    float* norms)
{
    // (2n-1)!! for n = 0..4
    const float DFACT[5] = {1.0f, 1.0f, 3.0f, 15.0f, 105.0f};

    SIMPLE_DEVICE_FOR(mu, nao_eff)
    {
        // 找到 μ 所属的壳层
        int sh = 0;
        for (int s = 0; s < nbas; s++)
        {
            int off = is_spherical ? ao_offsets_sph[s] : ao_offsets_cart[s];
            int dim = is_spherical ? (2 * l_list[s] + 1)
                                   : ((l_list[s] + 1) * (l_list[s] + 2) / 2);
            if (mu >= off && mu < off + dim)
            {
                sh = s;
                break;
            }
        }

        const int l = l_list[sh];
        const int np = shell_sizes[sh];
        const int p_off = shell_offsets[sh];
        const int n_cart = (l + 1) * (l + 2) / 2;
        const int oc = ao_offsets_cart[sh];

        // 计算每个笛卡尔分量的 S_prim 权重（只依赖指数和系数）
        // S_aa^cart = F(lx,ly,lz) × Σ_{p,q} c_p c_q × (π/γ)^{3/2} / (2γ)^L
        // 对于同一壳层的所有笛卡尔分量，Σ 部分相同
        float S_shell = 0.0f;
        for (int ip = 0; ip < np; ip++)
        {
            float ai = exps[p_off + ip];
            float ci = coeffs[p_off + ip];
            for (int jp = 0; jp < np; jp++)
            {
                float aj = exps[p_off + jp];
                float cj = coeffs[p_off + jp];
                float g = ai + aj;
                float inv_2g = 0.5f / g;
                // (π/γ)^{3/2} / (2γ)^L
                float val = ci * cj * powf((float)CONSTANT_Pi / g, 1.5f);
                for (int k = 0; k < l; k++) val *= inv_2g;
                S_shell += val;
            }
        }

        float S_diag;
        if (!is_spherical)
        {
            // 笛卡尔基：直接用 F(lx,ly,lz) × S_shell
            int idx = mu - oc;
            int comp_off = QC_Comp_Offset(l);
            int lx = QC_COMP_LX_DEVICE[comp_off + idx];
            int ly = QC_COMP_LY_DEVICE[comp_off + idx];
            int lz = QC_COMP_LZ_DEVICE[comp_off + idx];
            S_diag = DFACT[lx] * DFACT[ly] * DFACT[lz] * S_shell;
        }
        else
        {
            // 球谐基：S_μμ = Σ_a c2s[a,μ]² × F(lx_a,ly_a,lz_a) × S_shell
            int os = ao_offsets_sph[sh];
            int s_idx = mu - os;
            float sum = 0.0f;
            int comp_off = QC_Comp_Offset(l);
            for (int a = 0; a < n_cart; a++)
            {
                float c2s = cart2sph_mat[(oc + a) * nao_sph + (os + s_idx)];
                if (c2s == 0.0f) continue;
                int lx = QC_COMP_LX_DEVICE[comp_off + a];
                int ly = QC_COMP_LY_DEVICE[comp_off + a];
                int lz = QC_COMP_LZ_DEVICE[comp_off + a];
                sum += c2s * c2s * DFACT[lx] * DFACT[ly] * DFACT[lz];
            }
            S_diag = sum * S_shell;
        }
        norms[mu] = 1.0f / sqrtf(fmaxf(S_diag, 1e-20f));
    }
}

void QUANTUM_CHEMISTRY::Compute_Analytical_Norms()
{
    const int nao = mol.nao;
    const int threads = 256;
    Launch_Device_Kernel(
        QC_Compute_Analytical_Norms_Kernel, (nao + threads - 1) / threads,
        threads, 0, 0, nao, mol.nbas, mol.d_l_list, mol.d_exps, mol.d_coeffs,
        mol.d_shell_offsets, mol.d_shell_sizes, mol.d_ao_offsets,
        mol.d_ao_offsets_sph, mol.is_spherical, cart2sph.d_cart2sph_mat,
        mol.nao_sph, scf_ws.ortho.d_norms);
}

void QUANTUM_CHEMISTRY::Build_Shell_Pair_Bounds()
{
    if (task_ctx.topo.n_shell_pairs <= 0) return;
    // 固定 grid 大小，scratch 池按池槽数分配 (见 QC_BOUNDS_POOL_SLOTS)
    const int threads = 64;
    const int blocks = QC_BOUNDS_POOL_SLOTS / threads;
    const int n = task_ctx.topo.n_shell_pairs;
    Launch_Device_Kernel(
        QC_Build_Shell_Pair_Bounds_Kernel, blocks, threads, 0, 0, n,
        task_ctx.buffers.d_shell_pairs, mol.d_atm, mol.d_bas, mol.d_env,
        mol.d_ao_offsets, mol.d_ao_offsets_sph, scf_ws.ortho.d_norms,
        mol.is_spherical, cart2sph.d_cart2sph_mat, mol.nao_sph,
        task_ctx.buffers.d_shell_pair_bounds, scf_ws.direct.d_hr_pool,
        task_ctx.params.eri_hr_base, task_ctx.params.eri_hr_size,
        task_ctx.params.eri_shell_buf_size,
        task_ctx.params.eri_prim_screen_tol);
    task_ctx.topo.h_shell_pair_bounds.resize((size_t)n);
    deviceMemcpy(task_ctx.topo.h_shell_pair_bounds.data(),
                 task_ctx.buffers.d_shell_pair_bounds, sizeof(float) * n,
                 deviceMemcpyDeviceToHost);
}

static __global__ void QC_Scale_OneE_And_Build_Hcore_Kernel(
    const int nao, const float* norms, float* S, float* T, float* V,
    float* V_ECP, float* H_core)
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
        if (V_ECP)
        {
            V_ECP[idx] *= scale;
            H_core[idx] += V_ECP[idx];
        }
    }
}

void QUANTUM_CHEMISTRY::Prepare_Integrals()
{
    const int nao = mol.nao;
    const int nao2 = mol.nao2;
    const int threads = 256;

    // norms 已在 Compute_Analytical_Norms 中计算完毕
    // 归一化 S/T/V(/V_ECP) 并构建 Hcore = T + V (+ V_ECP)
    Launch_Device_Kernel(QC_Scale_OneE_And_Build_Hcore_Kernel,
                         (nao2 + threads - 1) / threads, threads, 0, 0, nao,
                         scf_ws.ortho.d_norms, scf_ws.core.d_S, scf_ws.core.d_T,
                         scf_ws.core.d_V, scf_ws.core.d_V_ECP,
                         scf_ws.core.d_H_core);
}

// 重叠正交化矩阵
// 对重叠矩阵 S 做 double 精度本征分解，并构建正交化变换矩阵 X
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

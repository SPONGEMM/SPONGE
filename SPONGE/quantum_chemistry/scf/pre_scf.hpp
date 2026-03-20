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
#ifndef USE_GPU
    if (CONTROLLER::MPI_rank == 0 && mol.natm > 0 && mol.nbas > 0)
    {
        const int md_idx = atom_local[0];
        const int ptr_coord = mol.h_atm[1];
        float env_xyz[3];
        VECTOR center0;
        deviceMemcpy(env_xyz, mol.d_env + ptr_coord, sizeof(env_xyz),
                     deviceMemcpyDeviceToHost);
        deviceMemcpy(&center0, mol.d_centers, sizeof(VECTOR),
                     deviceMemcpyDeviceToHost);
        printf(
            "QC Coord Check | atom0 qc->md=%d | md(A)=(%.9f, %.9f, %.9f) | env(Bohr)=(%.9f, %.9f, %.9f) | center0(Bohr)=(%.9f, %.9f, %.9f)\n",
            md_idx, crd[md_idx].x, crd[md_idx].y, crd[md_idx].z,
            (double)env_xyz[0], (double)env_xyz[1], (double)env_xyz[2],
            (double)center0.x, (double)center0.y, (double)center0.z);
        fflush(stdout);
    }
#endif
}

// ========================== SCF 状态重置 =========================
// 清零本轮 SCF 的收敛标志、能量缓存与密度矩阵，并重置 DIIS 历史
// ================================================================
void QUANTUM_CHEMISTRY::Reset_SCF_State()
{
    const int nao2 = mol.nao2;

    scf_ws.diis_hist_count = scf_ws.diis_hist_head = 0;
    scf_ws.diis_hist_count_b = scf_ws.diis_hist_head_b = 0;

    deviceMemset(scf_ws.d_scf_energy, 0, sizeof(double));
    deviceMemset(scf_ws.d_prev_energy, 0, sizeof(double));
    deviceMemset(scf_ws.d_delta_e, 0, sizeof(double));
    deviceMemset(scf_ws.d_density_residual, 0, sizeof(double));
    deviceMemset(scf_ws.d_e, 0, sizeof(double));
    if (scf_ws.unrestricted) deviceMemset(scf_ws.d_e_b, 0, sizeof(double));
    deviceMemset(scf_ws.d_pvxc, 0, sizeof(double));
    deviceMemset(scf_ws.d_converged, 0, sizeof(int));
    deviceMemset(scf_ws.d_P, 0, sizeof(float) * nao2);
    if (scf_ws.unrestricted)
    {
        deviceMemset(scf_ws.d_P_b, 0, sizeof(float) * nao2);
        deviceMemset(scf_ws.d_Ptot, 0, sizeof(float) * nao2);
    }
}

// =========================== 单电子积分 ===========================
// 计算 S/T/V 单电子积分，并在球谐基下执行笛卡尔到球谐变换
// ================================================================
void QUANTUM_CHEMISTRY::Compute_OneE_Integrals()
{
    const int nao_c = mol.nao_cart;
    float* p_S = mol.is_spherical ? cart2sph.d_S_cart : scf_ws.d_S;
    float* p_T = mol.is_spherical ? cart2sph.d_T_cart : scf_ws.d_T;
    float* p_V = mol.is_spherical ? cart2sph.d_V_cart : scf_ws.d_V;

    deviceMemset(p_S, 0, sizeof(float) * nao_c * nao_c);
    deviceMemset(p_T, 0, sizeof(float) * nao_c * nao_c);
    deviceMemset(p_V, 0, sizeof(float) * nao_c * nao_c);

    const int chunk_size = ONE_E_BATCH_SIZE;
    for (int i = 0; i < task_ctx.n_1e_tasks; i += chunk_size)
    {
        int current_chunk = std::min(chunk_size, task_ctx.n_1e_tasks - i);
        QC_ONE_E_TASK* task_ptr = task_ctx.d_1e_tasks + i;
        Launch_Device_Kernel(
            OneE_Kernel, (current_chunk + 63) / 64, 64, 0, 0, current_chunk,
            task_ptr, mol.d_centers, mol.d_l_list, mol.d_exps, mol.d_coeffs,
            mol.d_shell_offsets, mol.d_shell_sizes, mol.d_ao_offsets, mol.d_atm,
            mol.d_env, mol.natm, p_S, p_T, p_V, nao_c);
    }
    // Dump S_cart BEFORE cart2sph
#ifndef USE_GPU
    if (mol.is_spherical && CONTROLLER::MPI_rank == 0)
    {
        FILE* fp = fopen("/tmp/sponge_S_cart.bin", "wb");
        fwrite(cart2sph.d_S_cart, sizeof(float),
               (size_t)mol.nao_cart * mol.nao_cart, fp);
        fclose(fp);
        printf("DUMPED S_cart to /tmp/sponge_S_cart.bin (nao_c=%d)\n",
               mol.nao_cart);
        fflush(stdout);
    }
#endif
    Cart2Sph_OneE_Integrals();

    // Dump S_sph AFTER cart2sph, BEFORE normalization
#ifndef USE_GPU
    if (mol.is_spherical && CONTROLLER::MPI_rank == 0)
    {
        FILE* fp = fopen("/tmp/sponge_S_sph_pre_norm.bin", "wb");
        fwrite(scf_ws.d_S, sizeof(float), (size_t)mol.nao * mol.nao, fp);
        fclose(fp);
        printf("DUMPED S_sph (pre-norm) to /tmp/sponge_S_sph_pre_norm.bin (nao=%d)\n",
               mol.nao);
        fflush(stdout);
    }
#endif
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
    deviceMemset(scf_ws.d_nuc_energy_dev, 0, sizeof(double));
    const int threads = 256;
    Launch_Device_Kernel(QC_Accumulate_Nuclear_Repulsion_Kernel,
                         (mol.natm + threads - 1) / threads, threads, 0, 0,
                         mol.natm, mol.d_Z, mol.d_atm, mol.d_env,
                         scf_ws.d_nuc_energy_dev, box_length);
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
                         scf_ws.d_S, scf_ws.d_norms);
    Launch_Device_Kernel(QC_Scale_OneE_And_Build_Hcore_Kernel,
                         (nao2 + threads - 1) / threads, threads, 0, 0, nao,
                         scf_ws.d_norms, scf_ws.d_S, scf_ws.d_T, scf_ws.d_V,
                         scf_ws.d_H_core);

    if (task_ctx.n_shell_pairs <= 0) return;

    int chunk_size = ERI_BATCH_SIZE;
#ifndef USE_GPU
    chunk_size = std::max(1, task_ctx.n_shell_pairs);
#endif
    for (int i = 0; i < task_ctx.n_shell_pairs; i += chunk_size)
    {
        const int current_chunk =
            std::min(chunk_size, task_ctx.n_shell_pairs - i);
        Launch_Device_Kernel(
            QC_Build_Shell_Pair_Bounds_Kernel,
            (current_chunk + threads - 1) / threads, threads, 0, 0,
            current_chunk, task_ctx.d_shell_pairs + i, mol.d_atm, mol.d_bas,
            mol.d_env, mol.d_ao_offsets, mol.d_ao_offsets_sph, scf_ws.d_norms,
            mol.is_spherical, cart2sph.d_cart2sph_mat, mol.nao_sph,
            task_ctx.d_shell_pair_bounds + i, d_hr_pool, task_ctx.eri_hr_base,
            task_ctx.eri_hr_size, task_ctx.eri_shell_buf_size,
            task_ctx.eri_prim_screen_tol);
    }

#ifndef USE_GPU
    if (CONTROLLER::MPI_rank == 0)
    {
        FILE* fp = fopen("/tmp/sponge_S_sph.bin", "wb");
        fwrite(scf_ws.d_S, sizeof(float), (size_t)nao2, fp);
        fclose(fp);
        printf("DUMPED S_sph to /tmp/sponge_S_sph.bin (nao=%d)\n", nao);
        fflush(stdout);
    }
#endif
}

// ========================= 重叠正交化矩阵 =========================
// 对重叠矩阵 S 做本征分解，并构建正交化变换矩阵 X
// ================================================================
static __global__ void QC_Build_X_From_EigCol_Kernel(const int nao,
                                                     const float* U_col,
                                                     const float* W,
                                                     const float eig_floor,
                                                     double* X_row)
{
#ifdef USE_GPU
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nao || j >= nao) return;
#else
#pragma omp parallel for collapse(2)
    for (int i = 0; i < nao; i++)
        for (int j = 0; j < nao; j++)
#endif
    {
        double sum = 0.0;
        for (int k = 0; k < nao; k++)
        {
            float wk = fmaxf(W[k], eig_floor);
            float uik = U_col[i + k * nao];
            float ujk = U_col[j + k * nao];
            sum += (double)uik * (double)ujk / sqrt((double)wk);
        }
        X_row[i * nao + j] = sum;
    }
}

void QUANTUM_CHEMISTRY::Build_Overlap_X()
{
    const int nao = mol.nao;
    const int nao2 = mol.nao2;

    deviceMemcpy(scf_ws.d_Work, scf_ws.d_S, sizeof(float) * nao2,
                 deviceMemcpyDeviceToDevice);
    QC_Diagonalize(solver_handle, mol.nao, scf_ws.d_Work, scf_ws.d_W,
                   scf_ws.d_solver_work, scf_ws.lwork, scf_ws.d_solver_iwork,
                   scf_ws.liwork, scf_ws.d_info);

    const dim3 block2d(16, 16);
    const dim3 grid2d((nao + block2d.x - 1) / block2d.x,
                      (nao + block2d.y - 1) / block2d.y);
    Launch_Device_Kernel(QC_Build_X_From_EigCol_Kernel, grid2d, block2d, 0, 0,
                         nao, scf_ws.d_Work, scf_ws.d_W,
                         scf_ws.overlap_eig_floor, scf_ws.d_X);

#ifndef USE_GPU
    if (CONTROLLER::MPI_rank == 0)
    {
        FILE* fp = fopen("/tmp/sponge_X.bin", "wb");
        fwrite(scf_ws.d_X, sizeof(double), (size_t)nao2, fp);
        fclose(fp);
        printf("DUMPED X to /tmp/sponge_X.bin (nao=%d)\n", nao);
        fflush(stdout);
    }
#endif

    // Always print overlap eigenvalue diagnostics on CPU
#ifndef USE_GPU
    if (CONTROLLER::MPI_rank == 0)
    {
        scf_ws.h_W.resize(nao);
        deviceMemcpy(scf_ws.h_W.data(), scf_ws.d_W, sizeof(float) * nao,
                     deviceMemcpyDeviceToHost);
        printf("Overlap eigenvalues (smallest 10):");
        for (int i = 0; i < std::min(nao, 10); i++)
            printf(" %.6e", (double)scf_ws.h_W[i]);
        printf("\n");
        int n_below_floor = 0;
        for (int i = 0; i < nao; i++)
            if (scf_ws.h_W[i] < scf_ws.overlap_eig_floor) n_below_floor++;
        printf("Overlap eig_floor=%.1e | n_below=%d | nao=%d\n",
               (double)scf_ws.overlap_eig_floor, n_below_floor, nao);
        fflush(stdout);
    }
#endif
    if (scf_ws.print_iter && CONTROLLER::MPI_rank == 0)
    {
        // Compute X^T * S * X in double to check orthogonality
        // Tmp = S * X, Fp = X^T * Tmp
        for (int i = 0; i < nao; i++)
            for (int j = 0; j < nao; j++)
            {
                double s = 0.0;
                for (int k = 0; k < nao; k++)
                    s += (double)scf_ws.d_S[i*nao+k] * scf_ws.d_X[k*nao+j];
                scf_ws.d_Tmp[i*nao+j] = (float)s;
            }
        for (int i = 0; i < nao; i++)
            for (int j = 0; j < nao; j++)
            {
                double s = 0.0;
                for (int k = 0; k < nao; k++)
                    s += scf_ws.d_X[k*nao+i] * (double)scf_ws.d_Tmp[k*nao+j];
                scf_ws.d_Fp[i*nao+j] = (float)s;
            }

        scf_ws.h_W.resize(nao);
        scf_ws.h_Fp.resize(nao2);
        scf_ws.h_Work.resize(nao2);
        deviceMemcpy(scf_ws.h_W.data(), scf_ws.d_W, sizeof(float) * nao,
                     deviceMemcpyDeviceToHost);
        deviceMemcpy(scf_ws.h_Work.data(), scf_ws.d_Work, sizeof(float) * nao2,
                     deviceMemcpyDeviceToHost);
        deviceMemcpy(scf_ws.h_Fp.data(), scf_ws.d_Fp, sizeof(float) * nao2,
                     deviceMemcpyDeviceToHost);

        std::vector<float> h_S(nao2, 0.0f);
        deviceMemcpy(h_S.data(), scf_ws.d_S, sizeof(float) * nao2,
                     deviceMemcpyDeviceToHost);

        double max_diag_dev = 0.0;
        double max_offdiag = 0.0;
        double s_max_asym = 0.0;
        for (int i = 0; i < nao; ++i)
        {
            for (int j = i + 1; j < nao; ++j)
            {
                s_max_asym =
                    fmax(s_max_asym,
                         fabs((double)h_S[i * nao + j] -
                              (double)h_S[j * nao + i]));
            }
        }
        for (int i = 0; i < nao; ++i)
        {
            for (int j = 0; j < nao; ++j)
            {
                const double v = (double)scf_ws.h_Fp[i * nao + j];
                if (i == j)
                    max_diag_dev = fmax(max_diag_dev, fabs(v - 1.0));
                else
                    max_offdiag = fmax(max_offdiag, fabs(v));
            }
        }

        std::vector<double> x_ref(nao2, 0.0);
        std::vector<double> tmp_ref(nao2, 0.0);
        std::vector<double> xsx_ref(nao2, 0.0);
        double eig_res_col = 0.0;
        double eig_res_row = 0.0;
        double u_ortho_diag_dev = 0.0;
        double u_ortho_offdiag = 0.0;
        for (int i = 0; i < nao; ++i)
        {
            for (int k = 0; k < nao; ++k)
            {
                double lhs_col = 0.0;
                double lhs_row = 0.0;
                for (int j = 0; j < nao; ++j)
                {
                    lhs_col += (double)h_S[i * nao + j] *
                               (double)scf_ws.h_Work[j + k * nao];
                    lhs_row += (double)h_S[i * nao + j] *
                               (double)scf_ws.h_Work[j * nao + k];
                }
                const double rhs_col =
                    (double)scf_ws.h_W[k] *
                    (double)scf_ws.h_Work[i + k * nao];
                const double rhs_row =
                    (double)scf_ws.h_W[k] *
                    (double)scf_ws.h_Work[i * nao + k];
                eig_res_col =
                    fmax(eig_res_col, fabs(lhs_col - rhs_col));
                eig_res_row =
                    fmax(eig_res_row, fabs(lhs_row - rhs_row));
            }
            for (int j = 0; j < nao; ++j)
            {
                double dot = 0.0;
                for (int k = 0; k < nao; ++k)
                {
                    dot += (double)scf_ws.h_Work[k + i * nao] *
                           (double)scf_ws.h_Work[k + j * nao];
                }
                if (i == j)
                    u_ortho_diag_dev =
                        fmax(u_ortho_diag_dev, fabs(dot - 1.0));
                else
                    u_ortho_offdiag = fmax(u_ortho_offdiag, fabs(dot));
            }
        }
        for (int i = 0; i < nao; ++i)
        {
            for (int j = 0; j < nao; ++j)
            {
                double sum = 0.0;
                for (int k = 0; k < nao; ++k)
                {
                    const double wk = fmax((double)scf_ws.h_W[k],
                                           (double)scf_ws.overlap_eig_floor);
                    const double uik =
                        (double)scf_ws.h_Work[i + k * nao];
                    const double ujk =
                        (double)scf_ws.h_Work[j + k * nao];
                    sum += uik * ujk / sqrt(wk);
                }
                x_ref[i * nao + j] = sum;
            }
        }
        for (int i = 0; i < nao; ++i)
        {
            for (int j = 0; j < nao; ++j)
            {
                double sum = 0.0;
                for (int k = 0; k < nao; ++k)
                    sum += (double)h_S[i * nao + k] * x_ref[k * nao + j];
                tmp_ref[i * nao + j] = sum;
            }
        }
        for (int i = 0; i < nao; ++i)
        {
            for (int j = 0; j < nao; ++j)
            {
                double sum = 0.0;
                for (int k = 0; k < nao; ++k)
                    sum += x_ref[i * nao + k] * tmp_ref[k * nao + j];
                xsx_ref[i * nao + j] = sum;
            }
        }

        double ref_max_diag_dev = 0.0;
        double ref_max_offdiag = 0.0;
        for (int i = 0; i < nao; ++i)
        {
            for (int j = 0; j < nao; ++j)
            {
                const double v = xsx_ref[i * nao + j];
                if (i == j)
                    ref_max_diag_dev = fmax(ref_max_diag_dev, fabs(v - 1.0));
                else
                    ref_max_offdiag = fmax(ref_max_offdiag, fabs(v));
            }
        }

        printf(
            "Overlap X Check | eig_min=%.6e | eig_max=%.6e | eig_floor=%.6e | s_max_asym=%.6e | eig_res_col=%.6e | eig_res_row=%.6e | u_ortho_diag_dev=%.6e | u_ortho_offdiag=%.6e | max_diag_dev=%.6e | max_offdiag=%.6e | ref_max_diag_dev=%.6e | ref_max_offdiag=%.6e\n",
            (double)scf_ws.h_W.front(), (double)scf_ws.h_W.back(),
            (double)scf_ws.overlap_eig_floor, s_max_asym, eig_res_col,
            eig_res_row, u_ortho_diag_dev, u_ortho_offdiag, max_diag_dev,
            max_offdiag, ref_max_diag_dev, ref_max_offdiag);
        fflush(stdout);
    }
}

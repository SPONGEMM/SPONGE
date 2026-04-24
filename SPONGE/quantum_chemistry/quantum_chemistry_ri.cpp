#include "basis/basis.h"
#include "integrals/eri/common/direct_fock_kernels.hpp"
#include "integrals/ri/ri_2center.hpp"
#include "integrals/ri/ri_3center.hpp"
#include "integrals/ri/ri_metric.hpp"
#include "quantum_chemistry.h"

static __global__ void QC_Scale_RI_Metric_Kernel(const int naux,
                                                 const float* aux_norms,
                                                 double* metric)
{
    const int total = naux * naux;
    SIMPLE_DEVICE_FOR(idx, total)
    {
        const int P = idx / naux;
        const int Q = idx - P * naux;
        metric[idx] *= (double)aux_norms[P] * (double)aux_norms[Q];
    }
}

static __global__ void QC_Scale_RI_3Center_Kernel(const int naux, const int nao,
                                                  const float* aux_norms,
                                                  const float* orb_norms,
                                                  double* eri3c)
{
    const long long total = (long long)naux * nao * nao;
    SIMPLE_DEVICE_FOR(idx, total)
    {
        const int nu = (int)(idx % nao);
        const long long tmp = idx / nao;
        const int mu = (int)(tmp % nao);
        const int P = (int)(tmp / nao);
        eri3c[idx] *= (double)aux_norms[P] * (double)orb_norms[mu] *
                      (double)orb_norms[nu];
    }
}

static void QC_Build_RI_Aux_Norms(QUANTUM_CHEMISTRY* qc)
{
    auto& ri = qc->scf_ws.ri;
    const int threads = 256;
    const int Pc = ri.naux_cart;
    const int Ps = ri.naux;
    const long long naux2_cart = (long long)Pc * Pc;

    float* d_S_cart = NULL;
    float* d_T_dummy = NULL;
    float* d_V_dummy = NULL;
    QC_ONE_E_TASK* d_tasks = NULL;
    Device_Malloc_Safely((void**)&d_S_cart, sizeof(float) * naux2_cart);
    Device_Malloc_Safely((void**)&d_T_dummy, sizeof(float) * naux2_cart);
    Device_Malloc_Safely((void**)&d_V_dummy, sizeof(float) * naux2_cart);
    deviceMemset(d_S_cart, 0, sizeof(float) * naux2_cart);
    deviceMemset(d_T_dummy, 0, sizeof(float) * naux2_cart);
    deviceMemset(d_V_dummy, 0, sizeof(float) * naux2_cart);

    std::vector<QC_ONE_E_TASK> h_tasks;
    h_tasks.reserve((size_t)ri.naux_bas * (size_t)ri.naux_bas);
    for (int i = 0; i < ri.naux_bas; i++)
        for (int j = 0; j < ri.naux_bas; j++) h_tasks.push_back({i, j});

    Device_Malloc_Safely((void**)&d_tasks,
                         sizeof(QC_ONE_E_TASK) * h_tasks.size());
    deviceMemcpy(d_tasks, h_tasks.data(),
                 sizeof(QC_ONE_E_TASK) * h_tasks.size(),
                 deviceMemcpyHostToDevice);

    Launch_Device_Kernel(
        OneE_Kernel, ((int)h_tasks.size() + threads - 1) / threads, threads, 0,
        0, (int)h_tasks.size(), d_tasks, ri.d_aux_centers, ri.d_aux_l_list,
        ri.d_aux_exps, ri.d_aux_coeffs, ri.d_aux_shell_offsets,
        ri.d_aux_shell_sizes, ri.d_aux_ao_offsets, ri.d_aux_atm, ri.d_aux_env,
        0, d_S_cart, d_T_dummy, d_V_dummy, Pc);

    std::vector<float> h_S_final((size_t)Ps * Ps, 0.0f);
    std::vector<float> h_S_cart(naux2_cart);
    deviceMemcpy(h_S_cart.data(), d_S_cart, sizeof(float) * naux2_cart,
                 deviceMemcpyDeviceToHost);
    std::vector<double> tmp((size_t)Ps * Pc, 0.0);
    for (int i = 0; i < Ps; i++)
        for (int j = 0; j < Pc; j++)
            for (int k = 0; k < Pc; k++)
                tmp[(size_t)i * Pc + j] +=
                    (double)ri.h_U_aux[(size_t)k * Ps + i] *
                    (double)h_S_cart[(size_t)k * Pc + j];
    for (int i = 0; i < Ps; i++)
        for (int j = 0; j < Ps; j++)
            for (int k = 0; k < Pc; k++)
                h_S_final[(size_t)i * Ps + j] +=
                    (float)(tmp[(size_t)i * Pc + k] *
                            (double)ri.h_U_aux[(size_t)k * Ps + j]);

    ri.h_aux_norms.resize(Ps);
    for (int i = 0; i < Ps; i++)
    {
        const float sii = h_S_final[(size_t)i * Ps + i];
        // 辅助基线性相关时对角元可能为零，加安全下限避免除零
        constexpr float kNormSafeFloor = 1e-20f;
        ri.h_aux_norms[i] = 1.0f / sqrtf(fmaxf(sii, kNormSafeFloor));
    }

    if (ri.d_aux_norms) deviceFree(ri.d_aux_norms);
    Device_Malloc_And_Copy_Safely((void**)&ri.d_aux_norms,
                                  (void*)ri.h_aux_norms.data(),
                                  sizeof(float) * ri.h_aux_norms.size());

    deviceFree(d_tasks);
    deviceFree(d_S_cart);
    deviceFree(d_T_dummy);
    deviceFree(d_V_dummy);
}

void QUANTUM_CHEMISTRY::Initial_Auxiliary_Basis(CONTROLLER* controller)
{
    QC_BASIS_SET* aux_basis = QC_Get_JKFIT_Basis(orbital_basis_name.c_str());
    if (aux_basis == nullptr)
    {
        controller->Throw_Formatted_SPONGE_Error(
            spongeErrorValueErrorCommand, "QUANTUM_CHEMISTRY::Initial",
            "Reason:\n    No JKFIT auxiliary basis available for orbital "
            "basis \"%s\". RI requires a matching auxiliary basis.\n",
            orbital_basis_name.c_str());
    }
    aux_basis->Initialize();

    auto& ri = scf_ws.ri;
    ri.naux_cart = 0;
    ri.naux = 0;
    ri.naux_bas = 0;

    // 收集每个原子的元素符号（复用 mol 中的原子序数）
    for (int i = 0; i < mol.natm; ++i)
    {
        int Z = mol.h_Z[i];
        auto it_sym = QC_SYMBOL_FROM_Z.find(Z);
        if (it_sym == QC_SYMBOL_FROM_Z.end())
        {
            controller->Throw_Formatted_SPONGE_Error(
                spongeErrorValueErrorCommand, "QUANTUM_CHEMISTRY::Initial",
                "Reason:\n    Unknown atomic number %d in auxiliary basis "
                "setup\n",
                Z);
        }
        const std::string& sym = it_sym->second;

        auto it_basis = aux_basis->data.find(sym);
        if (it_basis == aux_basis->data.end())
        {
            controller->Throw_Formatted_SPONGE_Error(
                spongeErrorValueErrorCommand, "QUANTUM_CHEMISTRY::Initial",
                "Reason:\n    JKFIT auxiliary basis not available for element "
                "%s (Z=%d)\n",
                sym.c_str(), Z);
        }
        const auto& shells = it_basis->second;

        // 辅助基的 atm 条目
        int ptr_coord = ri.h_aux_env.size();
        ri.h_aux_env.push_back(0.0f);  // 坐标占位，后续 Update_Coordinates 填充
        ri.h_aux_env.push_back(0.0f);
        ri.h_aux_env.push_back(0.0f);
        ri.h_aux_atm.push_back(Z);
        ri.h_aux_atm.push_back(ptr_coord);
        ri.h_aux_atm.push_back(1);
        ri.h_aux_atm.push_back(0);
        ri.h_aux_atm.push_back(0);
        ri.h_aux_atm.push_back(0);

        for (const auto& shell : shells)
        {
            int ptr_exp = ri.h_aux_env.size();
            ri.h_aux_env.insert(ri.h_aux_env.end(), shell.exps.begin(),
                                shell.exps.end());
            int ptr_coeff = ri.h_aux_env.size();
            ri.h_aux_env.insert(ri.h_aux_env.end(), shell.coeffs.begin(),
                                shell.coeffs.end());

            ri.h_aux_bas.push_back(i);  // atom index
            ri.h_aux_bas.push_back(shell.l);
            ri.h_aux_bas.push_back(shell.exps.size());
            ri.h_aux_bas.push_back(1);  // nctr
            ri.h_aux_bas.push_back(0);
            ri.h_aux_bas.push_back(ptr_exp);
            ri.h_aux_bas.push_back(ptr_coeff);
            ri.h_aux_bas.push_back(0);

            int cart_dim = (shell.l + 1) * (shell.l + 2) / 2;
            int sph_dim = 2 * shell.l + 1;

            ri.h_aux_l_list.push_back(shell.l);
            ri.h_aux_shell_sizes.push_back(shell.exps.size());
            ri.h_aux_shell_offsets.push_back(ri.h_aux_exps.size());

            ri.h_aux_exps.insert(ri.h_aux_exps.end(), shell.exps.begin(),
                                 shell.exps.end());
            ri.h_aux_coeffs.insert(ri.h_aux_coeffs.end(), shell.coeffs.begin(),
                                   shell.coeffs.end());
            ri.h_aux_centers.push_back(VECTOR(0.0f));

            ri.h_aux_ao_offsets.push_back(ri.naux_cart);
            ri.h_aux_ao_offsets_sph.push_back(ri.naux);

            ri.naux_cart += cart_dim;
            ri.naux += sph_dim;
            ri.naux_bas++;
        }
    }

    ri.h_U_aux =
        QC_Build_Cart2Sph_Mat_Host(ri.h_aux_l_list, ri.naux_cart, ri.naux);

    // 拷贝到 device
    Device_Malloc_And_Copy_Safely((void**)&ri.d_aux_l_list,
                                  (void*)ri.h_aux_l_list.data(),
                                  sizeof(int) * ri.h_aux_l_list.size());
    Device_Malloc_And_Copy_Safely((void**)&ri.d_aux_shell_offsets,
                                  (void*)ri.h_aux_shell_offsets.data(),
                                  sizeof(int) * ri.h_aux_shell_offsets.size());
    Device_Malloc_And_Copy_Safely((void**)&ri.d_aux_shell_sizes,
                                  (void*)ri.h_aux_shell_sizes.data(),
                                  sizeof(int) * ri.h_aux_shell_sizes.size());
    Device_Malloc_And_Copy_Safely((void**)&ri.d_aux_ao_offsets,
                                  (void*)ri.h_aux_ao_offsets.data(),
                                  sizeof(int) * ri.h_aux_ao_offsets.size());
    Device_Malloc_And_Copy_Safely((void**)&ri.d_aux_ao_offsets_sph,
                                  (void*)ri.h_aux_ao_offsets_sph.data(),
                                  sizeof(int) * ri.h_aux_ao_offsets_sph.size());
    Device_Malloc_And_Copy_Safely((void**)&ri.d_aux_exps,
                                  (void*)ri.h_aux_exps.data(),
                                  sizeof(float) * ri.h_aux_exps.size());
    Device_Malloc_And_Copy_Safely((void**)&ri.d_aux_coeffs,
                                  (void*)ri.h_aux_coeffs.data(),
                                  sizeof(float) * ri.h_aux_coeffs.size());
    Device_Malloc_And_Copy_Safely((void**)&ri.d_aux_centers,
                                  (void*)ri.h_aux_centers.data(),
                                  sizeof(VECTOR) * ri.h_aux_centers.size());
    Device_Malloc_And_Copy_Safely((void**)&ri.d_aux_atm,
                                  (void*)ri.h_aux_atm.data(),
                                  sizeof(int) * ri.h_aux_atm.size());
    Device_Malloc_And_Copy_Safely((void**)&ri.d_aux_bas,
                                  (void*)ri.h_aux_bas.data(),
                                  sizeof(int) * ri.h_aux_bas.size());
    Device_Malloc_And_Copy_Safely((void**)&ri.d_aux_env,
                                  (void*)ri.h_aux_env.data(),
                                  sizeof(float) * ri.h_aux_env.size());
}

void QUANTUM_CHEMISTRY::RI_Memory_Allocate()
{
    auto& ri = scf_ws.ri;
    const int naux = ri.naux;
    const int nao = mol.nao;
    const int nao2 = mol.nao2;
    const long long naux2 = (long long)naux * naux;
    const long long n3c = (long long)naux * nao2;

    // 根据用户设置或自动选择 stored/direct 模式
    const double mem_3c_mb = n3c * sizeof(double) / 1e6;
    if (ri.mode == QC_RI_WORKSPACE::DF_DIRECT)
    {
        ri.direct = true;
    }
    else if (ri.mode == QC_RI_WORKSPACE::DF_STORED)
    {
        ri.direct = false;
    }
    else  // DF_AUTO
    {
        ri.direct = (mem_3c_mb > 512.0);
    }

    // 二中心 metric（两种模式均需要）
    Device_Malloc_Safely((void**)&ri.d_metric, sizeof(double) * naux2);
    Device_Malloc_Safely((void**)&ri.d_metric_inv, sizeof(double) * naux2);
    Device_Malloc_Safely((void**)&ri.d_metric_inv_sqrt, sizeof(double) * naux2);
    deviceMemset(ri.d_metric, 0, sizeof(double) * naux2);

    // RI-J scratch
    Device_Malloc_Safely((void**)&ri.d_d_vec, sizeof(double) * naux);
    Device_Malloc_Safely((void**)&ri.d_g_vec, sizeof(double) * naux);

    // RI-J persistent scratch
    Device_Malloc_Safely((void**)&ri.d_P_double, sizeof(double) * nao2);
    Device_Malloc_Safely((void**)&ri.d_J_double, sizeof(double) * nao2);
    Device_Malloc_Safely((void**)&ri.d_J_float, sizeof(float) * nao2);

    // RI-K scratch: B_occ [naux × nao × nocc] + K accumulation + B_flat
    const int nocc = std::max(scf_ws.runtime.n_alpha, scf_ws.runtime.n_beta);
    if (nocc > 0 && dft.exx_fraction != 0.0f)
    {
        const long long n_bocc = (long long)naux * nao * nocc;
        Device_Malloc_Safely((void**)&ri.d_B_occ, sizeof(float) * n_bocc);
        Device_Malloc_Safely((void**)&ri.d_B_flat, sizeof(float) * n_bocc);
        Device_Malloc_Safely((void**)&ri.d_K_scratch, sizeof(float) * nao2);
    }

    // cart2sph 轨道变换矩阵（stored 和 direct 模式都需要）
    if (mol.is_spherical)
    {
        ri.h_U_orb =
            QC_Build_Cart2Sph_Mat_Host(mol.h_l_list, mol.nao_cart, mol.nao);
    }
    else
    {
        ri.h_U_orb.clear();
    }

    if (!ri.direct)
    {
        // Stored 模式：预存 eri3c 和 B
        Device_Malloc_Safely((void**)&ri.d_eri3c, sizeof(double) * n3c);
        deviceMemset(ri.d_eri3c, 0, sizeof(double) * n3c);
        Device_Malloc_Safely((void**)&ri.d_B, sizeof(float) * n3c);
    }
    // Direct 模式：d_3c_buf 在 Build_Fock_RI_Direct 内临时分配/释放
}

void QUANTUM_CHEMISTRY::RI_Precompute()
{
    auto& ri = scf_ws.ri;
    const int naux = ri.naux;
    const int nao = mol.nao;
    const int nao2 = mol.nao2;
    const int threads = 256;

    QC_Build_RI_Aux_Norms(this);

    const int Pc = ri.naux_cart, Ps = ri.naux;
    const int Mc = mol.nao_cart, Ms = mol.nao;

    // 上传 U_aux/U_orb 到 device (float → double, 用于 DGEMM cart2sph)
    double* d_U_aux = NULL;
    double* d_U_orb = NULL;
    {
        Device_Malloc_Safely((void**)&d_U_aux, sizeof(double) * Pc * Ps);
        std::vector<double> h_Ud(Pc * Ps);
        for (int i = 0; i < Pc * Ps; i++) h_Ud[i] = (double)ri.h_U_aux[i];
        deviceMemcpy(d_U_aux, h_Ud.data(), sizeof(double) * Pc * Ps,
                     deviceMemcpyHostToDevice);
        if (mol.is_spherical && Mc > 0 && Ms > 0)
        {
            Device_Malloc_Safely((void**)&d_U_orb, sizeof(double) * Mc * Ms);
            h_Ud.resize(Mc * Ms);
            for (int i = 0; i < Mc * Ms; i++) h_Ud[i] = (double)ri.h_U_orb[i];
            deviceMemcpy(d_U_orb, h_Ud.data(), sizeof(double) * Mc * Ms,
                         deviceMemcpyHostToDevice);
        }
    }

    // 1. 计算二中心 metric (P|Q)
    {
        const long long n2c_cart = (long long)Pc * Pc;
        double* d_metric_cart = NULL;
        QC_ONE_E_TASK* d_2c_tasks = NULL;
        Device_Malloc_Safely((void**)&d_metric_cart, sizeof(double) * n2c_cart);
        deviceMemset(d_metric_cart, 0, sizeof(double) * n2c_cart);

        std::vector<QC_ONE_E_TASK> h_2c_tasks;
        for (int i = 0; i < ri.naux_bas; i++)
            for (int j = 0; j <= i; j++) h_2c_tasks.push_back({i, j});
        const int n_2c = h_2c_tasks.size();

        Device_Malloc_Safely((void**)&d_2c_tasks, sizeof(QC_ONE_E_TASK) * n_2c);
        deviceMemcpy(d_2c_tasks, h_2c_tasks.data(),
                     sizeof(QC_ONE_E_TASK) * n_2c, deviceMemcpyHostToDevice);

        Launch_Device_Kernel(
            QC_RI_2Center_Kernel, (n_2c + threads - 1) / threads, threads, 0, 0,
            n_2c, d_2c_tasks, ri.d_aux_centers, ri.d_aux_l_list, ri.d_aux_exps,
            ri.d_aux_coeffs, ri.d_aux_shell_offsets, ri.d_aux_shell_sizes,
            ri.d_aux_ao_offsets, ri.naux_cart, d_metric_cart);

        deviceFree(d_2c_tasks);

        // M_sph = U^T @ M_cart @ U  (device DGEMM)
        // d_U_aux 列优先 = U^T[Ps,Pc], lda=Ps; op=N → U^T, op=T → U
        // d_metric_cart 列优先 = M[Pc,Pc] (对称), lda=Pc
        const double one = 1.0, zero = 0.0;
        double* d_tmp = NULL;
        Device_Malloc_Safely((void**)&d_tmp, sizeof(double) * Ps * Pc);
        // Step A: tmp[Ps,Pc] = U^T @ M_cart
        deviceBlasDgemm(blas_handle, DEVICE_BLAS_OP_N, DEVICE_BLAS_OP_N, Ps, Pc,
                        Pc, &one, d_U_aux, Ps, d_metric_cart, Pc, &zero, d_tmp,
                        Ps);
        // Step B: M_sph[Ps,Ps] = tmp @ U = tmp @ (U^T)^T
        deviceBlasDgemm(blas_handle, DEVICE_BLAS_OP_N, DEVICE_BLAS_OP_T, Ps, Ps,
                        Pc, &one, d_tmp, Ps, d_U_aux, Ps, &zero, ri.d_metric,
                        Ps);
        deviceFree(d_tmp);
        deviceFree(d_metric_cart);
        Launch_Device_Kernel(QC_Scale_RI_Metric_Kernel,
                             ((long long)naux * naux + threads - 1) / threads,
                             threads, 0, 0, naux, ri.d_aux_norms, ri.d_metric);
        // 2c metric done
    }

    // ---- 2. 计算三中心积分 (P|μν)（仅 stored 模式）----
    if (!ri.direct)
    {
        const long long n3c_cart =
            (long long)ri.naux_cart * mol.nao_cart * mol.nao_cart;
        double* d_eri3c_cart = NULL;
        QC_RI_3C_TASK* d_3c_tasks = NULL;
        Device_Malloc_Safely((void**)&d_eri3c_cart, sizeof(double) * n3c_cart);
        deviceMemset(d_eri3c_cart, 0, sizeof(double) * n3c_cart);

        // Schwarz 筛选: |(P|μν)| ≤ sqrt(P|P) × sqrt(μν|μν)
        // 辅助基 bound: 从 2c metric 对角元素取 sqrt（归一化后）
        std::vector<double> h_metric_diag(Ps);
        {
            std::vector<double> h_metric_full((size_t)Ps * Ps);
            deviceMemcpy(h_metric_full.data(), ri.d_metric,
                         sizeof(double) * Ps * Ps, deviceMemcpyDeviceToHost);
            for (int p = 0; p < Ps; p++)
                h_metric_diag[p] =
                    sqrt(fabs(h_metric_full[(size_t)p * Ps + p]));
        }
        // 每个辅助 shell 的最大 bound
        std::vector<float> aux_shell_bound(ri.naux_bas, 0.0f);
        for (int sh = 0; sh < ri.naux_bas; sh++)
        {
            int off = mol.is_spherical ? ri.h_aux_ao_offsets_sph[sh]
                                       : ri.h_aux_ao_offsets[sh];
            int dim = mol.is_spherical ? (2 * ri.h_aux_l_list[sh] + 1)
                                       : ((ri.h_aux_l_list[sh] + 1) *
                                          (ri.h_aux_l_list[sh] + 2) / 2);
            for (int i = 0; i < dim; i++)
                aux_shell_bound[sh] = std::max(aux_shell_bound[sh],
                                               (float)h_metric_diag[off + i]);
        }
        // 轨道基 pair bound: 从 task_ctx 获取（Prepare_Integrals 中已计算）
        // 建立 (mu_sh, nu_sh) → pair_bound 映射
        std::vector<std::vector<float>> orb_pair_bound(
            mol.nbas, std::vector<float>(mol.nbas, 0.0f));
        for (int pid = 0; pid < task_ctx.topo.n_shell_pairs; pid++)
        {
            const auto& p = task_ctx.topo.h_shell_pairs[pid];
            float b = task_ctx.topo.h_shell_pair_bounds[pid];
            orb_pair_bound[p.x][p.y] = b;
            orb_pair_bound[p.y][p.x] = b;
        }

        const float screen_tol = task_ctx.params.eri_shell_screen_tol;
        std::vector<QC_RI_3C_TASK> h_3c_tasks;
        int n_screened = 0;
        for (int P = 0; P < ri.naux_bas; P++)
            for (int mu = 0; mu < mol.nbas; mu++)
                for (int nu = 0; nu <= mu; nu++)
                {
                    if (aux_shell_bound[P] * orb_pair_bound[mu][nu] <
                        screen_tol)
                    {
                        n_screened++;
                        continue;
                    }
                    h_3c_tasks.push_back({P, mu, nu});
                }
        const int n_3c = h_3c_tasks.size();

        Device_Malloc_Safely((void**)&d_3c_tasks, sizeof(QC_RI_3C_TASK) * n_3c);
        deviceMemcpy(d_3c_tasks, h_3c_tasks.data(),
                     sizeof(QC_RI_3C_TASK) * n_3c, deviceMemcpyHostToDevice);

        Launch_Device_Kernel(
            QC_RI_3Center_Kernel, (n_3c + threads - 1) / threads, threads, 0, 0,
            n_3c, d_3c_tasks, ri.d_aux_centers, ri.d_aux_l_list, ri.d_aux_exps,
            ri.d_aux_coeffs, ri.d_aux_shell_offsets, ri.d_aux_shell_sizes,
            ri.d_aux_ao_offsets, mol.d_centers, mol.d_l_list, mol.d_exps,
            mol.d_coeffs, mol.d_shell_offsets, mol.d_shell_sizes,
            mol.d_ao_offsets, ri.naux_cart, mol.nao_cart, mol.nao_cart, 0, 0,
            true, d_eri3c_cart);

        deviceFree(d_3c_tasks);
        // 3c kernel done

        // Cart2Sph 变换: 全部在 device 上用 DGEMM
        // d_U_aux 列优先 = U_aux^T[Ps,Pc], lda=Ps; op=N → U^T, op=T → U
        // d_U_orb 列优先 = U_orb^T[Ms,Mc], lda=Ms; op=N → U_orb^T, op=T → U_orb
        // d_eri3c_cart 行优先 [Pc,Mc,Mc] = 列优先视为 T^T 各种切片
        const double one = 1.0, zero = 0.0;

        if (!mol.is_spherical)
        {
            // 仅 P 变换: T_sph[Ps,Mc²] = U_aux^T @ T_cart[Pc,Mc²]
            // 列优先: T_sph^T[Mc²,Ps] = T_cart^T[Mc²,Pc] @ U[Pc,Ps]
            deviceBlasDgemm(blas_handle, DEVICE_BLAS_OP_N, DEVICE_BLAS_OP_T,
                            Mc * Mc, Ps, Pc, &one, d_eri3c_cart, Mc * Mc,
                            d_U_aux, Ps, &zero, ri.d_eri3c, Mc * Mc);
        }
        else
        {
            // Step 1 (ν): T1[Pc*Mc, Ms] = T_cart[Pc*Mc, Mc] @ U_orb[Mc, Ms]
            // 列优先: T1^T[Ms, Pc*Mc] = U_orb^T[Ms,Mc] @ T_cart^T[Mc, Pc*Mc]
            const long long n_step1 = (long long)Pc * Mc * Ms;
            double* d_step1 = NULL;
            Device_Malloc_Safely((void**)&d_step1, sizeof(double) * n_step1);
            deviceBlasDgemm(blas_handle, DEVICE_BLAS_OP_N, DEVICE_BLAS_OP_N, Ms,
                            Pc * Mc, Mc, &one, d_U_orb, Ms, d_eri3c_cart, Mc,
                            &zero, d_step1, Ms);

            // Step 2 (μ): 对每个 P, T2_P[Ms,Ms] = U_orb^T @ T1_P (行优先乘法)
            // 列优先: T2_P^T[Ms,Ms] = T1_P^T[Ms,Mc] @ U_orb[Mc,Ms]
            const long long n_step2 = (long long)Pc * Ms * Ms;
            double* d_step2 = NULL;
            Device_Malloc_Safely((void**)&d_step2, sizeof(double) * n_step2);
            for (int P = 0; P < Pc; P++)
            {
                deviceBlasDgemm(
                    blas_handle, DEVICE_BLAS_OP_N, DEVICE_BLAS_OP_T, Ms, Ms, Mc,
                    &one, d_step1 + (long long)P * Mc * Ms, Ms, d_U_orb, Ms,
                    &zero, d_step2 + (long long)P * Ms * Ms, Ms);
            }
            deviceFree(d_step1);

            // Step 3 (P): T_sph[Ps, Ms²] = U_aux^T @ T2[Pc, Ms²]
            // 列优先: T_sph^T[Ms²,Ps] = T2^T[Ms²,Pc] @ U[Pc,Ps]
            deviceBlasDgemm(blas_handle, DEVICE_BLAS_OP_N, DEVICE_BLAS_OP_T,
                            Ms * Ms, Ps, Pc, &one, d_step2, Ms * Ms, d_U_aux,
                            Ps, &zero, ri.d_eri3c, Ms * Ms);
            deviceFree(d_step2);
        }
        deviceFree(d_eri3c_cart);

        Launch_Device_Kernel(
            QC_Scale_RI_3Center_Kernel,
            ((long long)naux * nao * nao + threads - 1) / threads, threads, 0,
            0, naux, nao, ri.d_aux_norms, scf_ws.ortho.d_norms, ri.d_eri3c);
    }

    // 3. 特征分解 → (P|Q)^{-1/2}
    ri.naux_eff = QC_RI_Build_Metric_InvSqrt(solver_handle, blas_handle, naux,
                                             ri.d_metric, ri.d_metric_inv_sqrt,
                                             &ri.h_eigval, &ri.h_eigvec);

    // 4. 构建 (P|Q)^{-1} (RI-J 用)
    QC_RI_Build_Metric_Inv(solver_handle, blas_handle, naux, ri.d_metric,
                           ri.d_metric_inv, ri.naux_eff);

    // 缓存 metric_inv / inv_sqrt 到 host: 供 direct-mode Build_Fock 和
    // RI 梯度复用，消除每轮 SCF 的 O(naux²) D2H 拷贝
    ri.h_metric_inv.resize((size_t)naux * naux);
    ri.h_metric_inv_sqrt.resize((size_t)naux * naux);
    deviceMemcpy(ri.h_metric_inv.data(), ri.d_metric_inv,
                 sizeof(double) * naux * naux, deviceMemcpyDeviceToHost);
    deviceMemcpy(ri.h_metric_inv_sqrt.data(), ri.d_metric_inv_sqrt,
                 sizeof(double) * naux * naux, deviceMemcpyDeviceToHost);

    // ---- 5. 构建 B 张量（仅 stored 模式）----
    if (!ri.direct)
    {
        double* d_B_double = NULL;
        const long long n3c = (long long)naux * nao2;
        Device_Malloc_Safely((void**)&d_B_double, sizeof(double) * n3c);

        // eri3c is stored with row-major indexing [P, mu*nao+nu], which is
        // equivalent to a col-major matrix [nao2 x naux]. Build B in the same
        // physical layout so downstream kernels can keep using row-major
        // indexing on [naux x nao2].
        const double one = 1.0, zero = 0.0;
        deviceBlasDgemm(blas_handle, DEVICE_BLAS_OP_N, DEVICE_BLAS_OP_N, nao2,
                        naux, naux, &one, ri.d_eri3c, nao2,
                        ri.d_metric_inv_sqrt, naux, &zero, d_B_double, nao2);
        QC_Double_To_Float((int)n3c, d_B_double, ri.d_B);
        deviceFree(d_B_double);
    }

    deviceFree(d_U_aux);
    deviceFree(d_U_orb);
}

// F[i] += scale * K[i]
static __global__ void QC_Scaled_Add_Kernel(const int n, const float scale,
                                            const float* src, float* dst)
{
    SIMPLE_DEVICE_FOR(idx, n) { dst[idx] += scale * src[idx]; }
}

// B_occ[P*nao + mu + i*M] → B_flat[mu + (P*nocc + i)*nao]
// 将 (naux*nao, nocc) 重排为 (nao, naux*nocc)，使 K = B_flat * B_flat^T
static __global__ void QC_RI_Permute_B_Occ_Kernel(const int total,
                                                  const int nao, const int naux,
                                                  const int nocc,
                                                  const float* __restrict__ src,
                                                  float* __restrict__ dst)
{
    const int M = naux * nao;
    SIMPLE_DEVICE_FOR(idx, total)
    {
        const int mu = idx % nao;
        const int rem = idx / nao;
        const int P = rem / nocc;
        const int i = rem % nocc;
        dst[idx] = src[P * nao + mu + (long long)i * M];
    }
}

// Stored-mode Build_Fock_RI: 使用预存的 d_eri3c 和 d_B
static void Build_Fock_RI_Stored(QUANTUM_CHEMISTRY* qc);
// Direct-mode Build_Fock_RI: 每轮在线计算 3c 积分
static void Build_Fock_RI_Direct(QUANTUM_CHEMISTRY* qc);

// RI-JK Fock 构建（入口）
void QUANTUM_CHEMISTRY::Build_Fock_RI()
{
    auto& ri = scf_ws.ri;
    const int nao2 = mol.nao2;
    const int threads = 256;

    // 0. F = H_core + Vxc
    if (dft.enable_dft) Build_DFT_VXC();

    Launch_Device_Kernel(QC_Init_Fock_Kernel, (nao2 + threads - 1) / threads,
                         threads, 0, 0, nao2, scf_ws.core.d_H_core, dft.d_Vxc,
                         dft.enable_dft, scf_ws.alpha.d_F);
    if (scf_ws.runtime.unrestricted)
    {
        Launch_Device_Kernel(QC_Init_Fock_Kernel,
                             (nao2 + threads - 1) / threads, threads, 0, 0,
                             nao2, scf_ws.core.d_H_core, dft.d_Vxc_beta,
                             dft.enable_dft, scf_ws.beta.d_F);
    }

    if (ri.direct)
        Build_Fock_RI_Direct(this);
    else
        Build_Fock_RI_Stored(this);

    // F double 精度拷贝 (DIIS 用)
    if (scf_ws.alpha.d_F_double)
        QC_Float_To_Double_Copy(nao2, scf_ws.alpha.d_F,
                                scf_ws.alpha.d_F_double);
    if (scf_ws.runtime.unrestricted && scf_ws.beta.d_F_double)
        QC_Float_To_Double_Copy(nao2, scf_ws.beta.d_F, scf_ws.beta.d_F_double);
}

// Stored mode
// RI-K helper: B_occ → permute → single GEMM → K
// 将 naux 次小 SGEMM 替换为 1 次 permute kernel + 1 次大 SGEMM
static void RI_K_Build_Single_GEMM(BLAS_HANDLE blas_handle, QC_RI_WORKSPACE& ri,
                                   const int naux, const int nao,
                                   const int nocc, const float* d_C, float* d_F,
                                   const float neg_exx, const int threads)
{
    const int nao2 = nao * nao;
    const int M = naux * nao;
    const float one_f = 1.0f, zero_f = 0.0f;

    // B_occ[M, nocc] = B^T[M, nao] * C^T[:nocc, nao]
    deviceBlasSgemm(blas_handle, DEVICE_BLAS_OP_T, DEVICE_BLAS_OP_T, M, nocc,
                    nao, &one_f, ri.d_B, nao, d_C, nao, &zero_f, ri.d_B_occ, M);

    // Permute: B_occ[(naux*nao), nocc] → B_flat[nao, (naux*nocc)]
    // B_occ[P*nao+mu, i] → B_flat[mu, P*nocc+i]
    const int total = nao * naux * nocc;
    Launch_Device_Kernel(QC_RI_Permute_B_Occ_Kernel,
                         (total + threads - 1) / threads, threads, 0, 0, total,
                         nao, naux, nocc, ri.d_B_occ, ri.d_B_flat);

    // K = B_flat × B_flat^T  (single large GEMM)
    // K[nao, nao] = B_flat[nao, naux*nocc] × B_flat^T[naux*nocc, nao]
    const int K_dim = naux * nocc;
    deviceBlasSgemm(blas_handle, DEVICE_BLAS_OP_N, DEVICE_BLAS_OP_T, nao, nao,
                    K_dim, &one_f, ri.d_B_flat, nao, ri.d_B_flat, nao, &zero_f,
                    ri.d_K_scratch, nao);

    // F += neg_exx * K
    Launch_Device_Kernel(QC_Scaled_Add_Kernel, (nao2 + threads - 1) / threads,
                         threads, 0, 0, nao2, neg_exx, ri.d_K_scratch, d_F);
}

static void Build_Fock_RI_Stored(QUANTUM_CHEMISTRY* qc)
{
    auto& ri = qc->scf_ws.ri;
    auto& scf_ws = qc->scf_ws;
    auto& mol = qc->mol;
    auto& dft = qc->dft;
    auto blas_handle = qc->blas_handle;
    const int naux = ri.naux;
    const int nao = mol.nao;
    const int nao2 = mol.nao2;
    const int threads = 256;

    // ---- RI-J（使用持久化缓冲，无 malloc/free）----
    {
        const float* d_P_coul = scf_ws.runtime.unrestricted
                                    ? scf_ws.direct.d_Ptot
                                    : scf_ws.alpha.d_P;

        QC_Float_To_Double(nao2, d_P_coul, ri.d_P_double);

        const double one = 1.0, zero = 0.0;
        // d_vec[P] = eri3c^T * P
        deviceBlasDgemm(blas_handle, DEVICE_BLAS_OP_T, DEVICE_BLAS_OP_N, naux,
                        1, nao2, &one, ri.d_eri3c, nao2, ri.d_P_double, nao2,
                        &zero, ri.d_d_vec, naux);

        // g = (P|Q)^{-1} * d_vec
        deviceBlasDgemm(blas_handle, DEVICE_BLAS_OP_N, DEVICE_BLAS_OP_N, naux,
                        1, naux, &one, ri.d_metric_inv, naux, ri.d_d_vec, naux,
                        &zero, ri.d_g_vec, naux);

        // J = eri3c * g
        deviceBlasDgemm(blas_handle, DEVICE_BLAS_OP_N, DEVICE_BLAS_OP_N, nao2,
                        1, naux, &one, ri.d_eri3c, nao2, ri.d_g_vec, naux,
                        &zero, ri.d_J_double, nao2);

        QC_Double_To_Float(nao2, ri.d_J_double, ri.d_J_float);
        QC_Add_Matrix(nao2, scf_ws.alpha.d_F, ri.d_J_float, scf_ws.alpha.d_F);
        if (scf_ws.runtime.unrestricted)
            QC_Add_Matrix(nao2, scf_ws.beta.d_F, ri.d_J_float, scf_ws.beta.d_F);
    }

    // ---- RI-K: permute + single GEMM（替代 naux 次小 SGEMM）----
    if (dft.exx_fraction != 0.0f)
    {
        const float neg_exx = -dft.exx_fraction;

        const int nocc_a = scf_ws.runtime.n_alpha;
        if (nocc_a > 0)
            RI_K_Build_Single_GEMM(blas_handle, ri, naux, nao, nocc_a,
                                   scf_ws.alpha.d_C, scf_ws.alpha.d_F, neg_exx,
                                   threads);

        if (scf_ws.runtime.unrestricted)
        {
            const int nocc_b = scf_ws.runtime.n_beta;
            if (nocc_b > 0)
                RI_K_Build_Single_GEMM(blas_handle, ri, naux, nao, nocc_b,
                                       scf_ws.beta.d_C, scf_ws.beta.d_F,
                                       neg_exx, threads);
        }
    }
}

// Direct mode
// 在线计算 3c 积分，不存储 eri3c/B 张量。
// 策略：逐轨道 shell pair 启动 GPU 3c kernel，结果到临时缓冲，
//       host 端做 cart2sph + 收缩到 d_vec / B_occ。
// RI-J: pass1 累加 d_P, solve g, pass2 累加 J（需两趟 3c）
// RI-K: 一趟累加 B_occ，最后 K = B_occ^T B_occ
//
// 当前实现：先在 host 上完成全部 shell-pair 循环 + 收缩。
// 后续可优化为 GPU kernel batch + device-side 收缩。
static void Build_Fock_RI_Direct(QUANTUM_CHEMISTRY* qc)
{
    auto& ri = qc->scf_ws.ri;
    auto& scf_ws = qc->scf_ws;
    auto& mol = qc->mol;
    auto& dft = qc->dft;
    auto blas_handle = qc->blas_handle;
    const int naux = ri.naux;
    const int nao = mol.nao;
    const int nao2 = mol.nao2;
    const int threads = 256;

    // 下载到 host
    std::vector<float> h_D_f(nao2);
    const float* d_P_coul =
        scf_ws.runtime.unrestricted ? scf_ws.direct.d_Ptot : scf_ws.alpha.d_P;
    deviceMemcpy(h_D_f.data(), d_P_coul, sizeof(float) * nao2,
                 deviceMemcpyDeviceToHost);
    std::vector<double> h_D(nao2);
    for (int i = 0; i < nao2; i++) h_D[i] = (double)h_D_f[i];

    // 复用 RI_Precompute 阶段缓存的 host metric (不随 SCF 迭代改变)
    const std::vector<double>& h_inv = ri.h_metric_inv;
    const std::vector<double>& h_inv_sqrt = ri.h_metric_inv_sqrt;
    std::vector<float> h_orb_norms(nao);
    deviceMemcpy(h_orb_norms.data(), scf_ws.ortho.d_norms, sizeof(float) * nao,
                 deviceMemcpyDeviceToHost);

    // 临时 3c 缓冲（direct 模式按 shell-pair 紧凑输出）
    int max_l_cart = 0;
    for (int sh = 0; sh < mol.nbas; sh++)
        if (mol.h_l_list[sh] > max_l_cart) max_l_cart = mol.h_l_list[sh];
    const int max_cart = (max_l_cart + 1) * (max_l_cart + 2) / 2;
    const long long buf_3c_size = (long long)ri.naux_cart * max_cart * max_cart;
    double* d_3c_buf = NULL;
    Device_Malloc_Safely((void**)&d_3c_buf, sizeof(double) * buf_3c_size);

    // 准备 RI-J: d_vec, J (host)
    std::vector<double> h_d_vec(naux, 0.0);
    std::vector<double> h_J(nao2, 0.0);

    // 准备 RI-K: B_occ (host)
    const bool need_exx = (dft.exx_fraction != 0.0f);
    const int nocc_a = scf_ws.runtime.n_alpha;
    const int nocc_b_val =
        scf_ws.runtime.unrestricted ? scf_ws.runtime.n_beta : 0;

    std::vector<float> h_C_a, h_C_b;
    std::vector<double> h_B_occ_a, h_B_occ_b;
    if (need_exx && nocc_a > 0)
    {
        h_C_a.resize(nao * nao);
        deviceMemcpy(h_C_a.data(), scf_ws.alpha.d_C, sizeof(float) * nao * nao,
                     deviceMemcpyDeviceToHost);
        h_B_occ_a.assign((long long)naux * nao * nocc_a, 0.0);
    }
    if (need_exx && nocc_b_val > 0)
    {
        h_C_b.resize(nao * nao);
        deviceMemcpy(h_C_b.data(), scf_ws.beta.d_C, sizeof(float) * nao * nao,
                     deviceMemcpyDeviceToHost);
        h_B_occ_b.assign((long long)naux * nao * nocc_b_val, 0.0);
    }

    // pass 1: 逐 shell pair 计算 3c 积分
    // 用 GPU kernel 计算笛卡尔 3c block，下载，cart2sph，收缩
    // 对每个 shell pair (mu_sh, nu_sh):
    //   tasks = {(P_sh, mu_sh, nu_sh) | P_sh = 0..naux_bas-1}
    //   launch kernel → d_3c_buf [naux_cart × dim_mu_c × dim_nu_c]
    //   download, cart2sph → block_sph [naux × dim_mu_s × dim_nu_s]
    //   d_vec[P] += Σ_{μν} block_sph[P,μ,ν] * D[off_mu+μ, off_nu+ν]
    //   B_occ[P,off_mu+μ,i] += Σ_ν block_B[P,μ,ν] * C[off_nu+ν,i]

    // 分配 GPU 3c 任务缓冲（一次性，所有 P_sh × 1 pair）
    std::vector<QC_RI_3C_TASK> h_tasks(ri.naux_bas);
    QC_RI_3C_TASK* d_tasks = NULL;
    Device_Malloc_Safely((void**)&d_tasks, sizeof(QC_RI_3C_TASK) * ri.naux_bas);

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
            const int off_nu_s = mol.is_spherical ? mol.h_ao_offsets_sph[nu_sh]
                                                  : mol.h_ao_offsets[nu_sh];

            // 构建 tasks
            for (int P = 0; P < ri.naux_bas; P++)
                h_tasks[P] = {P, mu_sh, nu_sh};
            deviceMemcpy(d_tasks, h_tasks.data(),
                         sizeof(QC_RI_3C_TASK) * ri.naux_bas,
                         deviceMemcpyHostToDevice);

            // 清零缓冲，启动 kernel
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

            // 下载笛卡尔 block
            std::vector<double> h_block_cart(buf_n);
            deviceMemcpy(h_block_cart.data(), d_3c_buf, sizeof(double) * buf_n,
                         deviceMemcpyDeviceToHost);

            // Cart2sph: block_cart[Pc, dmc, dnc] → block_sph[Ps, dms, dns]
            std::vector<double> block_sph(naux * dms * dns, 0.0);
            if (!mol.is_spherical)
            {
                const int Pc = ri.naux_cart;
                for (int ps = 0; ps < naux; ps++)
                    for (int pc = 0; pc < Pc; pc++)
                    {
                        double u = (double)ri.h_U_aux[pc * naux + ps];
                        if (u == 0.0) continue;
                        for (int i = 0; i < dms; i++)
                            for (int j = 0; j < dns; j++)
                                block_sph[ps * dms * dns + i * dns + j] +=
                                    u * h_block_cart[(long long)pc * dmc * dnc +
                                                     (long long)i * dnc + j];
                    }
            }
            else
            {
                // 先提取笛卡尔 block: [Pc, dmc, dnc]
                const int Pc = ri.naux_cart;
                std::vector<double> bc(Pc * dmc * dnc, 0.0);
                for (int P = 0; P < Pc; P++)
                    for (int i = 0; i < dmc; i++)
                        for (int j = 0; j < dnc; j++)
                            bc[P * dmc * dnc + i * dnc + j] =
                                h_block_cart[(long long)P * dmc * dnc +
                                             (long long)i * dnc + j];

                // ν 变换: T1[Pc, dmc, dns]
                std::vector<double> t1(Pc * dmc * dns, 0.0);
                for (int P = 0; P < Pc; P++)
                    for (int i = 0; i < dmc; i++)
                        for (int js = 0; js < dns; js++)
                            for (int jc = 0; jc < dnc; jc++)
                                t1[P * dmc * dns + i * dns + js] +=
                                    bc[P * dmc * dnc + i * dnc + jc] *
                                    (double)ri.h_U_orb
                                        [(mol.h_ao_offsets[nu_sh] + jc) * nao +
                                         off_nu_s + js];

                // μ 变换: T2[Pc, dms, dns]
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

                // P 变换: block_sph[Ps, dms, dns]
                for (int ps = 0; ps < naux; ps++)
                    for (int pc = 0; pc < Pc; pc++)
                    {
                        double u = (double)ri.h_U_aux[pc * naux + ps];
                        if (u == 0.0) continue;
                        for (int mn = 0; mn < dms * dns; mn++)
                            block_sph[ps * dms * dns + mn] +=
                                u * t2[pc * dms * dns + mn];
                    }
            }

            for (int P = 0; P < naux; P++)
            {
                const double p_scale = (double)ri.h_aux_norms[P];
                for (int i = 0; i < dms; i++)
                {
                    const double mu_scale = (double)h_orb_norms[off_mu_s + i];
                    for (int j = 0; j < dns; j++)
                    {
                        block_sph[P * dms * dns + i * dns + j] *=
                            p_scale * mu_scale *
                            (double)h_orb_norms[off_nu_s + j];
                    }
                }
            }

            // inv_sqrt @ block → B_block [naux, dms, dns]
            std::vector<double> B_block(naux * dms * dns, 0.0);
            for (int ps = 0; ps < naux; ps++)
                for (int qs = 0; qs < naux; qs++)
                {
                    double w = h_inv_sqrt[ps * naux + qs];
                    if (w == 0.0) continue;
                    for (int mn = 0; mn < dms * dns; mn++)
                        B_block[ps * dms * dns + mn] +=
                            w * block_sph[qs * dms * dns + mn];
                }

            // RI-J: d_vec[P] += Σ_{μν} block_sph[P,μ,ν] * D[μ,ν]
            for (int P = 0; P < naux; P++)
                for (int i = 0; i < dms; i++)
                    for (int j = 0; j < dns; j++)
                    {
                        double val = block_sph[P * dms * dns + i * dns + j];
                        h_d_vec[P] +=
                            val * h_D[(off_mu_s + i) * nao + (off_nu_s + j)];
                        if (mu_sh != nu_sh)
                            h_d_vec[P] +=
                                val *
                                h_D[(off_nu_s + j) * nao + (off_mu_s + i)];
                    }

            // RI-K: B_occ[P, off_mu+i, occ] += Σ_j B_block[P,i,j] * C[off_nu+j,
            // occ]
            if (need_exx && nocc_a > 0)
            {
                for (int P = 0; P < naux; P++)
                    for (int i = 0; i < dms; i++)
                        for (int j = 0; j < dns; j++)
                        {
                            double b = B_block[P * dms * dns + i * dns + j];
                            if (b == 0.0) continue;
                            int mu_idx = off_mu_s + i;
                            int nu_idx = off_nu_s + j;
                            for (int oc = 0; oc < nocc_a; oc++)
                            {
                                // C is row-major [nao × nao]
                                // C[ν, occ] = h_C_a[ν * nao + occ]
                                h_B_occ_a[(long long)P * nao * nocc_a +
                                          mu_idx * nocc_a + oc] +=
                                    b * (double)h_C_a[nu_idx * nao + oc];
                                if (mu_sh != nu_sh)
                                    h_B_occ_a[(long long)P * nao * nocc_a +
                                              nu_idx * nocc_a + oc] +=
                                        b * (double)h_C_a[mu_idx * nao + oc];
                            }
                        }
            }
            if (need_exx && nocc_b_val > 0)
            {
                for (int P = 0; P < naux; P++)
                    for (int i = 0; i < dms; i++)
                        for (int j = 0; j < dns; j++)
                        {
                            double b = B_block[P * dms * dns + i * dns + j];
                            if (b == 0.0) continue;
                            int mu_idx = off_mu_s + i;
                            int nu_idx = off_nu_s + j;
                            for (int oc = 0; oc < nocc_b_val; oc++)
                            {
                                h_B_occ_b[(long long)P * nao * nocc_b_val +
                                          mu_idx * nocc_b_val + oc] +=
                                    b * (double)h_C_b[nu_idx * nao + oc];
                                if (mu_sh != nu_sh)
                                    h_B_occ_b[(long long)P * nao * nocc_b_val +
                                              nu_idx * nocc_b_val + oc] +=
                                        b * (double)h_C_b[mu_idx * nao + oc];
                            }
                        }
            }
        }
    }
    deviceFree(d_tasks);

    // RI-J: g = inv @ d, J[μν] = Σ_P g_P block_sph[P,μ,ν]
    // 但 pass2 需要再算一遍 3c 积分... 用 metric_inv 直接算:
    // J[μν] = Σ_P (Σ_Q inv[P,Q] d_Q) block_sph[P,μ,ν]
    // 我们已经在 pass1 中积累了 d_vec。现在需要 pass2。
    // 但 pass2 和 pass1 一样需要遍历所有 shell pair。
    // 优化：可以在 pass1 同时积累 J（如果 g 已知）。
    // 但 g 依赖 d_vec 的完整值，所以必须两趟。
    //
    // 方案 B：用 metric_inv_sqrt，一趟就够。
    // d' = inv_sqrt @ d → g' = d'（因为 inv = inv_sqrt @ inv_sqrt）
    // J = Σ_P g'_P * B_block[P] = 两趟都需要 B_block...
    //
    // 最终方案：pass1 积累 d_vec，算 g_vec，pass2 重新计算 3c 积累 J。
    // 为避免重复计算 3c，我们在 pass1 中同时积累 J 的贡献。
    //
    // 实际上更聪明的做法：
    // J[μν] = Σ_PQ (P|μν) (P|Q)^{-1} (Q|ρσ) D[ρσ]
    //       = Σ_P (P|μν) g_P  where g = inv @ d
    // 需要 g 才能算 J，但 g 需要完整的 d。
    // 所以确实需要两趟。
    //
    // 但注意 RI-K 的 B_occ 在 pass1 中已经一趟算完。
    // 对 RI-J，我们可以重新计算 3c 做 pass2。
    // 或者：在 pass1 中存储每个 block_sph[P, μ_sh, ν_sh] 到一个
    // 压缩格式中。但这等于变相存储了 3c 张量。
    //
    // 最简洁的做法：pass1 计算 d_vec。然后:
    //   g = inv @ d
    //   pass2: 重新计算每个 shell pair 的 3c block，
    //          J[off_mu+i, off_nu+j] += Σ_P g[P] * block[P,i,j]

    // 先求 g
    std::vector<double> h_g(naux, 0.0);
    for (int P = 0; P < naux; P++)
        for (int Q = 0; Q < naux; Q++)
            h_g[P] += h_inv[P * naux + Q] * h_d_vec[Q];

    // pass 2: RI-J 第二趟，重新计算 3c 积分以构建 J
    Device_Malloc_Safely((void**)&d_tasks, sizeof(QC_RI_3C_TASK) * ri.naux_bas);
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
            const int off_nu_s = mol.is_spherical ? mol.h_ao_offsets_sph[nu_sh]
                                                  : mol.h_ao_offsets[nu_sh];

            // 复用 pass1 的 kernel 调用逻辑
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

            std::vector<double> h_block_cart(buf_n);
            deviceMemcpy(h_block_cart.data(), d_3c_buf, sizeof(double) * buf_n,
                         deviceMemcpyDeviceToHost);

            // cart2sph (同 pass1 逻辑)
            std::vector<double> block_sph(naux * dms * dns, 0.0);
            if (!mol.is_spherical)
            {
                const int Pc = ri.naux_cart;
                for (int ps = 0; ps < naux; ps++)
                    for (int pc = 0; pc < Pc; pc++)
                    {
                        double u = (double)ri.h_U_aux[pc * naux + ps];
                        if (u == 0.0) continue;
                        for (int i = 0; i < dms; i++)
                            for (int j = 0; j < dns; j++)
                                block_sph[ps * dms * dns + i * dns + j] +=
                                    u * h_block_cart[(long long)pc * dmc * dnc +
                                                     (long long)i * dnc + j];
                    }
            }
            else
            {
                const int Pc = ri.naux_cart;
                std::vector<double> bc(Pc * dmc * dnc, 0.0);
                for (int P = 0; P < Pc; P++)
                    for (int i = 0; i < dmc; i++)
                        for (int j = 0; j < dnc; j++)
                            bc[P * dmc * dnc + i * dnc + j] =
                                h_block_cart[(long long)P * dmc * dnc +
                                             (long long)i * dnc + j];
                std::vector<double> t1(Pc * dmc * dns, 0.0);
                for (int P = 0; P < Pc; P++)
                    for (int i = 0; i < dmc; i++)
                        for (int js = 0; js < dns; js++)
                            for (int jc = 0; jc < dnc; jc++)
                                t1[P * dmc * dns + i * dns + js] +=
                                    bc[P * dmc * dnc + i * dnc + jc] *
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
                            block_sph[ps * dms * dns + mn] +=
                                u * t2[pc * dms * dns + mn];
                    }
            }

            for (int P = 0; P < naux; P++)
            {
                const double p_scale = (double)ri.h_aux_norms[P];
                for (int i = 0; i < dms; i++)
                {
                    const double mu_scale = (double)h_orb_norms[off_mu_s + i];
                    for (int j = 0; j < dns; j++)
                    {
                        block_sph[P * dms * dns + i * dns + j] *=
                            p_scale * mu_scale *
                            (double)h_orb_norms[off_nu_s + j];
                    }
                }
            }

            // J[off_mu+i, off_nu+j] += Σ_P g[P] * block_sph[P,i,j]
            for (int i = 0; i < dms; i++)
                for (int j = 0; j < dns; j++)
                {
                    double val = 0.0;
                    for (int P = 0; P < naux; P++)
                        val += h_g[P] * block_sph[P * dms * dns + i * dns + j];
                    h_J[(off_mu_s + i) * nao + (off_nu_s + j)] += val;
                    if (mu_sh != nu_sh)
                        h_J[(off_nu_s + j) * nao + (off_mu_s + i)] += val;
                }
        }
    }
    deviceFree(d_tasks);
    deviceFree(d_3c_buf);

    // 上传 J 到 device，加到 F
    {
        std::vector<float> h_J_f(nao2);
        for (int i = 0; i < nao2; i++) h_J_f[i] = (float)h_J[i];
        float* d_J_f = NULL;
        Device_Malloc_Safely((void**)&d_J_f, sizeof(float) * nao2);
        deviceMemcpy(d_J_f, h_J_f.data(), sizeof(float) * nao2,
                     deviceMemcpyHostToDevice);
        QC_Add_Matrix(nao2, scf_ws.alpha.d_F, d_J_f, scf_ws.alpha.d_F);
        if (scf_ws.runtime.unrestricted)
            QC_Add_Matrix(nao2, scf_ws.beta.d_F, d_J_f, scf_ws.beta.d_F);
        deviceFree(d_J_f);
    }

    // RI-K
    if (need_exx)
    {
        const float neg_exx = -dft.exx_fraction;

        auto build_K_from_B_occ =
            [&](const std::vector<double>& h_B_occ, int nocc, float* d_F)
        {
            if (nocc <= 0) return;
            // K[μ,ν] = Σ_{P,i} B_occ[P,μ,i] * B_occ[P,ν,i]
            std::vector<float> h_K(nao2, 0.0f);
            for (int P = 0; P < naux; P++)
                for (int mu = 0; mu < nao; mu++)
                    for (int nu = 0; nu <= mu; nu++)
                    {
                        double sum = 0.0;
                        for (int oc = 0; oc < nocc; oc++)
                            sum += h_B_occ[(long long)P * nao * nocc +
                                           mu * nocc + oc] *
                                   h_B_occ[(long long)P * nao * nocc +
                                           nu * nocc + oc];
                        h_K[mu * nao + nu] += (float)sum;
                        if (mu != nu) h_K[nu * nao + mu] += (float)sum;
                    }
            // 上传并加到 F
            float* d_K = NULL;
            Device_Malloc_Safely((void**)&d_K, sizeof(float) * nao2);
            deviceMemcpy(d_K, h_K.data(), sizeof(float) * nao2,
                         deviceMemcpyHostToDevice);
            Launch_Device_Kernel(QC_Scaled_Add_Kernel,
                                 (nao2 + threads - 1) / threads, threads, 0, 0,
                                 nao2, neg_exx, d_K, d_F);
            deviceFree(d_K);
        };

        build_K_from_B_occ(h_B_occ_a, nocc_a, scf_ws.alpha.d_F);
        if (scf_ws.runtime.unrestricted)
            build_K_from_B_occ(h_B_occ_b, nocc_b_val, scf_ws.beta.d_F);
    }
}

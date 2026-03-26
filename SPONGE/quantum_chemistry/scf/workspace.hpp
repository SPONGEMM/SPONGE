#pragma once

#include <stdexcept>

void QUANTUM_CHEMISTRY::Build_SCF_Workspace()
{
    const int nao = mol.nao;
    const int nao2 = mol.nao2;
    const bool unrestricted = scf_ws.unrestricted;
    const int diis_space = scf_ws.diis_space;
    const int spin_e = mol.multiplicity - 1;

    scf_ws.h_X.resize(nao2);
    scf_ws.h_F.resize(nao2);
    scf_ws.h_C.resize(nao2);
    scf_ws.h_P.resize(nao2);
    scf_ws.h_P_new.resize(nao2);
    scf_ws.h_W.resize((int)nao);
    scf_ws.h_Work.resize(nao2);
    scf_ws.h_Fp.resize(nao2);
    scf_ws.h_Tmp.resize(nao2);
    if (unrestricted)
    {
        scf_ws.h_F_b.resize(nao2);
        scf_ws.h_C_b.resize(nao2);
        scf_ws.h_P_b.resize(nao2);
        scf_ws.h_P_b_new.resize(nao2);
        scf_ws.h_W_b.resize((int)nao);
        scf_ws.h_Fp_b.resize(nao2);
    }
    else
    {
        scf_ws.h_F_b.clear();
        scf_ws.h_C_b.clear();
        scf_ws.h_P_b.clear();
        scf_ws.h_P_b_new.clear();
        scf_ws.h_W_b.clear();
        scf_ws.h_Fp_b.clear();
    }

    auto alloc_zero_float = [](float** ptr, int count)
    {
        if (count == 0)
        {
            *ptr = NULL;
            return;
        }
        Device_Malloc_Safely((void**)ptr, sizeof(float) * count);
        deviceMemset(*ptr, 0, sizeof(float) * count);
    };
    auto alloc_zero_double = [](double** ptr, int count)
    {
        if (count == 0)
        {
            *ptr = NULL;
            return;
        }
        Device_Malloc_Safely((void**)ptr, sizeof(double) * count);
        deviceMemset(*ptr, 0, sizeof(double) * count);
    };
    auto alloc_zero_int = [](int** ptr, int count)
    {
        if (count == 0)
        {
            *ptr = NULL;
            return;
        }
        Device_Malloc_Safely((void**)ptr, sizeof(int) * count);
        deviceMemset(*ptr, 0, sizeof(int) * count);
    };
    auto alloc_from_host_float =
        [](float** ptr, const std::vector<float>& h_buf)
    {
        if (h_buf.empty())
        {
            *ptr = NULL;
            return;
        }
        Device_Malloc_And_Copy_Safely((void**)ptr, (void*)h_buf.data(),
                                      sizeof(float) * h_buf.size());
    };

    alloc_zero_float(&scf_ws.d_norms, (int)nao);
    alloc_zero_double(&scf_ws.d_X, nao2);
    alloc_from_host_float(&scf_ws.d_W, scf_ws.h_W);
    alloc_from_host_float(&scf_ws.d_Work, scf_ws.h_Work);
    alloc_from_host_float(&scf_ws.d_F, scf_ws.h_F);
    alloc_from_host_float(&scf_ws.d_P, scf_ws.h_P);
    alloc_from_host_float(&scf_ws.d_P_new, scf_ws.h_P_new);
    alloc_from_host_float(&scf_ws.d_Tmp, scf_ws.h_Tmp);
    alloc_from_host_float(&scf_ws.d_Fp, scf_ws.h_Fp);
    alloc_from_host_float(&scf_ws.d_C, scf_ws.h_C);

    if (unrestricted)
    {
        alloc_from_host_float(&scf_ws.d_F_b, scf_ws.h_F_b);
        alloc_from_host_float(&scf_ws.d_P_b, scf_ws.h_P_b);
        alloc_from_host_float(&scf_ws.d_P_b_new, scf_ws.h_P_b_new);
        alloc_from_host_float(&scf_ws.d_Fp_b, scf_ws.h_Fp_b);
        alloc_from_host_float(&scf_ws.d_C_b, scf_ws.h_C_b);
        alloc_zero_float(&scf_ws.d_Ptot, nao2);
    }
    else
    {
        scf_ws.d_F_b = NULL;
        scf_ws.d_P_b = NULL;
        scf_ws.d_P_b_new = NULL;
        scf_ws.d_Ptot = NULL;
        scf_ws.d_Fp_b = NULL;
        scf_ws.d_C_b = NULL;
    }

    alloc_zero_double(&scf_ws.d_e, 1);
    if (unrestricted)
    {
        alloc_zero_double(&scf_ws.d_e_b, 1);
    }
    else
    {
        scf_ws.d_e_b = NULL;
    }
    alloc_zero_double(&scf_ws.d_pvxc, 1);
    alloc_zero_double(&scf_ws.d_prev_energy, 1);
    alloc_zero_double(&scf_ws.d_delta_e, 1);
    alloc_zero_double(&scf_ws.d_density_residual, 1);
    alloc_zero_int(&scf_ws.d_converged, 1);
    alloc_zero_float(&scf_ws.d_pair_density_coul, task_ctx.topo.n_shell_pairs);
    alloc_zero_float(&scf_ws.d_pair_density_exx, task_ctx.topo.n_shell_pairs);
    if (unrestricted)
    {
        alloc_zero_float(&scf_ws.d_pair_density_exx_b,
                         task_ctx.topo.n_shell_pairs);
    }
    else
    {
        scf_ws.d_pair_density_exx_b = NULL;
    }

#ifdef USE_GPU
    scf_ws.fock_thread_count = 1;
    scf_ws.d_F_thread = NULL;
    scf_ws.d_F_b_thread = NULL;
#else
    scf_ws.fock_thread_count = std::max(1, omp_get_max_threads());
    alloc_zero_double(&scf_ws.d_F_thread, scf_ws.fock_thread_count * nao2);
    if (unrestricted)
    {
        alloc_zero_double(&scf_ws.d_F_b_thread,
                          scf_ws.fock_thread_count * nao2);
    }
    else
    {
        scf_ws.d_F_b_thread = NULL;
    }
    alloc_zero_double(&scf_ws.d_F_double, nao2);
    if (unrestricted)
        alloc_zero_double(&scf_ws.d_F_b_double, nao2);
    else
        scf_ws.d_F_b_double = NULL;
#endif

    // Double workspace for diag/DIIS
    alloc_zero_double(&scf_ws.d_dwork_nao2_1, nao2);
    alloc_zero_double(&scf_ws.d_dwork_nao2_2, nao2);
    alloc_zero_double(&scf_ws.d_dwork_nao2_3, nao2);
    alloc_zero_double(&scf_ws.d_dwork_nao2_4, nao2);
    alloc_zero_double(&scf_ws.d_dW_double, nao);
    // Double solver workspace
    {
        scf_ws.lwork_double = 0;
        double* tmp_work = NULL;
        QC_Diagonalize_Double_Workspace_Size(
            solver_handle, nao, scf_ws.d_dwork_nao2_1, scf_ws.d_dW_double,
            &tmp_work, &scf_ws.lwork_double);
        if (tmp_work)
        {
            deviceFree(tmp_work);
            tmp_work = NULL;
        }
        if (scf_ws.lwork_double > 0)
            Device_Malloc_Safely((void**)&scf_ws.d_solver_work_double,
                                 sizeof(double) * scf_ws.lwork_double);
    }

    scf_ws.lwork = 0;
    scf_ws.liwork = 0;
    int solver_stat = QC_Diagonalize_Workspace_Size(
        solver_handle, nao, scf_ws.d_Work, scf_ws.d_W, &scf_ws.d_solver_work,
        (void**)&scf_ws.d_solver_iwork, &scf_ws.lwork, &scf_ws.liwork);
    if (solver_stat != 0 || scf_ws.lwork <= 0)
    {
        throw std::runtime_error("QC_Diagonalize_Workspace_Size failed");
    }

    scf_ws.d_diis_f_hist.clear();
    scf_ws.d_diis_e_hist.clear();
    scf_ws.d_diis_f_hist_b.clear();
    scf_ws.d_diis_e_hist_b.clear();
    if (scf_ws.use_diis)
    {
        alloc_zero_double(&scf_ws.d_diis_err, nao2);
        alloc_zero_float(&scf_ws.d_diis_w1, nao2);
        alloc_zero_float(&scf_ws.d_diis_w2, nao2);
        alloc_zero_float(&scf_ws.d_diis_w3, nao2);
        alloc_zero_float(&scf_ws.d_diis_w4, nao2);
        alloc_zero_double(&scf_ws.d_diis_accum, 1);
        scf_ws.d_diis_f_hist.assign((int)diis_space, nullptr);
        scf_ws.d_diis_e_hist.assign((int)diis_space, nullptr);
        for (int i = 0; i < diis_space; i++)
        {
            alloc_zero_double(&scf_ws.d_diis_f_hist[(int)i], nao2);
            alloc_zero_double(&scf_ws.d_diis_e_hist[(int)i], nao2);
        }

        // ADIIS density history
        scf_ws.d_adiis_d_hist.assign((int)diis_space, nullptr);
        for (int i = 0; i < diis_space; i++)
            alloc_zero_double(&scf_ws.d_adiis_d_hist[(int)i], nao2);

        if (unrestricted)
        {
            scf_ws.d_diis_f_hist_b.assign((int)diis_space, nullptr);
            scf_ws.d_diis_e_hist_b.assign((int)diis_space, nullptr);
            scf_ws.d_adiis_d_hist_b.assign((int)diis_space, nullptr);
            for (int i = 0; i < diis_space; i++)
            {
                alloc_zero_double(&scf_ws.d_diis_f_hist_b[(int)i], nao2);
                alloc_zero_double(&scf_ws.d_diis_e_hist_b[(int)i], nao2);
                alloc_zero_double(&scf_ws.d_adiis_d_hist_b[(int)i], nao2);
            }
        }
    }
    else
    {
        scf_ws.d_diis_err = NULL;
        scf_ws.d_diis_w1 = NULL;
        scf_ws.d_diis_w2 = NULL;
        scf_ws.d_diis_w3 = NULL;
        scf_ws.d_diis_w4 = NULL;
        scf_ws.d_diis_accum = NULL;
    }

    scf_ws.n_alpha = (mol.nelectron + (unrestricted ? spin_e : 0)) / 2;
    scf_ws.n_beta = unrestricted ? (mol.nelectron - scf_ws.n_alpha) : 0;
    scf_ws.occ_factor = unrestricted ? 1.0f : 2.0f;
    scf_ws.d_P_coul = unrestricted ? scf_ws.d_Ptot : scf_ws.d_P;
}

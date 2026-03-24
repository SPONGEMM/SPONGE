#pragma once

static __global__ void QC_Mix_Density_Kernel(const int nao2, const int iter,
                                             const float mix,
                                             const float* P_new_row,
                                             float* P_row)
{
    SIMPLE_DEVICE_FOR(idx, nao2)
    {
        if (iter == 0)
            P_row[idx] = P_new_row[idx];
        else
            P_row[idx] = (1.0f - mix) * P_row[idx] + mix * P_new_row[idx];
    }
}

bool QUANTUM_CHEMISTRY::Mix_And_Check_Convergence(int iter, int md_step)
{
    const int mix_threads = 256;

    const float effective_mix = scf_ws.density_mixing;
    Launch_Device_Kernel(QC_Mix_Density_Kernel,
                         (mol.nao2 + mix_threads - 1) / mix_threads,
                         mix_threads, 0, 0, (int)mol.nao2, iter, effective_mix,
                         scf_ws.d_P_new, scf_ws.d_P);

    if (scf_ws.unrestricted)
    {
        Launch_Device_Kernel(
            QC_Mix_Density_Kernel, (mol.nao2 + mix_threads - 1) / mix_threads,
            mix_threads, 0, 0, mol.nao2, iter, scf_ws.density_mixing,
            scf_ws.d_P_b_new, scf_ws.d_P_b);
        QC_Add_Matrix((int)mol.nao2, scf_ws.d_P, scf_ws.d_P_b, scf_ws.d_Ptot);
    }

    if ((scf_ws.print_iter || scf_ws.profile_stages) &&
        CONTROLLER::MPI_rank == 0)
    {
        double h_energy = 0.0;
        double h_delta_e = 0.0;
        deviceMemcpy(&h_energy, scf_ws.d_scf_energy, sizeof(double),
                     deviceMemcpyDeviceToHost);
        deviceMemcpy(&h_delta_e, scf_ws.d_delta_e, sizeof(double),
                     deviceMemcpyDeviceToHost);
        FILE* out = (scf_output_file != NULL) ? scf_output_file : stdout;
        fprintf(out, "Step %6d | SCF Iter %3d | E(Ha)=%.12f | dE(Ha)=%+.6e",
                md_step, iter + 1, h_energy, h_delta_e);
        if (scf_ws.profile_stages)
        {
            fprintf(out,
                    " | t_pre=%.4fs | t_fock=%.4fs | t_filter=%.4fs | "
                    "t_energy=%.4fs | t_diis=%.4fs | t_diag=%.4fs | quartets=%d",
                    scf_ws.last_pre_scf_s, scf_ws.last_build_fock_s,
                    scf_ws.last_fock_filter_s,
                    scf_ws.last_accumulate_energy_s, scf_ws.last_apply_diis_s,
                    scf_ws.last_diag_density_s, scf_ws.last_active_eri_tasks);
        }
        fprintf(out, "\n");
        fflush(out);
    }

    int h_converged = 0;
    deviceMemcpy(&h_converged, scf_ws.d_converged, sizeof(int),
                 deviceMemcpyDeviceToHost);
    return h_converged != 0;
}

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

    // DIIS 外推成功时直接使用新密度，不做混合
    const float mix =
        scf_ws.diis.diis_extrapolated ? 1.0f : scf_ws.runtime.density_mixing;

    Launch_Device_Kernel(QC_Mix_Density_Kernel,
                         (mol.nao2 + mix_threads - 1) / mix_threads,
                         mix_threads, 0, 0, (int)mol.nao2, iter, mix,
                         scf_ws.alpha.d_P_new, scf_ws.alpha.d_P);

    if (scf_ws.runtime.unrestricted)
    {
        Launch_Device_Kernel(QC_Mix_Density_Kernel,
                             (mol.nao2 + mix_threads - 1) / mix_threads,
                             mix_threads, 0, 0, mol.nao2, iter, mix,
                             scf_ws.beta.d_P_new, scf_ws.beta.d_P);
        QC_Add_Matrix((int)mol.nao2, scf_ws.alpha.d_P, scf_ws.beta.d_P,
                      scf_ws.direct.d_Ptot);
    }

    if (scf_ws.runtime.print_iter && CONTROLLER::MPI_rank == 0)
    {
        double h_energy = 0.0;
        double h_delta_e = 0.0;
        deviceMemcpy(&h_energy, scf_ws.core.d_scf_energy, sizeof(double),
                     deviceMemcpyDeviceToHost);
        deviceMemcpy(&h_delta_e, scf_ws.runtime.d_delta_e, sizeof(double),
                     deviceMemcpyDeviceToHost);
        FILE* out = (scf_output_file != NULL) ? scf_output_file : stdout;
        fprintf(out, "Step %6d | SCF Iter %3d | E(Ha)=%.12f | dE(Ha)=%+.6e",
                md_step, iter + 1, h_energy, h_delta_e);
        fprintf(out, "\n");
        fflush(out);
    }

    int h_converged = 0;
    deviceMemcpy(&h_converged, scf_ws.runtime.d_converged, sizeof(int),
                 deviceMemcpyDeviceToHost);
    return h_converged != 0;
}

#pragma once

bool QUANTUM_CHEMISTRY::Check_Convergence(int iter, int md_step)
{
    const int nao2 = (int)mol.nao2;

    // 每步直接用新密度替换旧密度（与 PySCF 一致）
    deviceMemcpy(scf_ws.alpha.d_P, scf_ws.alpha.d_P_new, sizeof(float) * nao2,
                 deviceMemcpyDeviceToDevice);

    if (scf_ws.runtime.unrestricted)
    {
        deviceMemcpy(scf_ws.beta.d_P, scf_ws.beta.d_P_new, sizeof(float) * nao2,
                     deviceMemcpyDeviceToDevice);
        QC_Add_Matrix(nao2, scf_ws.alpha.d_P, scf_ws.beta.d_P,
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

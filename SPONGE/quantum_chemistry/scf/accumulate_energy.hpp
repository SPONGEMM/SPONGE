#pragma once

static __global__ void QC_Combine_SCF_Energy_Kernel(
    const int use_dft, const double* d_e_a, const double* d_e_b,
    const double* d_e_nuc, const double* d_exc, const double* d_pvxc,
    double* d_total_e)
{
    const double exc = use_dft ? d_exc[0] : 0.0;
    const double pvxc = use_dft ? d_pvxc[0] : 0.0;
    double e = d_e_a[0] + d_e_nuc[0] + exc - 0.5 * pvxc;
    if (d_e_b != nullptr) e += d_e_b[0];
    d_total_e[0] = e;
}

static __global__ void QC_Update_Convergence_Flag_Kernel(
    const int iter, const double tol, const double* d_curr_e, double* d_prev_e,
    double* d_delta_e, int* d_converged)
{
    const double curr_e = d_curr_e[0];
    const double delta_e = (iter > 0) ? (curr_e - d_prev_e[0]) : 0.0;
    d_delta_e[0] = delta_e;
    if (iter > 0)
    {
        const double denom = fmax(fabs(curr_e), 1.0);
        if (fabs(delta_e) / denom < tol)
        {
            d_converged[0] = 1;
        }
    }
    d_prev_e[0] = curr_e;
}

void QUANTUM_CHEMISTRY::Accumulate_SCF_Energy(int iter)
{
    deviceMemset(scf_ws.runtime.d_e, 0, sizeof(double));
    QC_Elec_Energy_Accumulate(mol.nao2, scf_ws.alpha.d_P, scf_ws.core.d_H_core,
                              scf_ws.alpha.d_F, scf_ws.runtime.d_e);

    if (scf_ws.runtime.unrestricted)
    {
        deviceMemset(scf_ws.runtime.d_e_b, 0, sizeof(double));
        QC_Elec_Energy_Accumulate(mol.nao2, scf_ws.beta.d_P,
                                  scf_ws.core.d_H_core, scf_ws.beta.d_F,
                                  scf_ws.runtime.d_e_b);
    }

    deviceMemset(scf_ws.runtime.d_pvxc, 0, sizeof(double));
    if (dft.enable_dft)
    {
        QC_Mat_Dot_Accumulate(mol.nao2, scf_ws.alpha.d_P, dft.d_Vxc,
                              scf_ws.runtime.d_pvxc);
        if (scf_ws.runtime.unrestricted)
        {
            QC_Mat_Dot_Accumulate(mol.nao2, scf_ws.beta.d_P, dft.d_Vxc_beta,
                                  scf_ws.runtime.d_pvxc);
        }
    }

    Launch_Device_Kernel(QC_Combine_SCF_Energy_Kernel, 1, 1, 0, 0,
                         dft.enable_dft, scf_ws.runtime.d_e,
                         scf_ws.runtime.unrestricted ? scf_ws.runtime.d_e_b
                                                     : (const double*)nullptr,
                         scf_ws.core.d_nuc_energy_dev, dft.d_exc_total,
                         scf_ws.runtime.d_pvxc, scf_ws.core.d_scf_energy);

    Launch_Device_Kernel(QC_Update_Convergence_Flag_Kernel, 1, 1, 0, 0, iter,
                         scf_ws.runtime.energy_tol, scf_ws.core.d_scf_energy,
                         scf_ws.runtime.d_prev_energy, scf_ws.runtime.d_delta_e,
                         scf_ws.runtime.d_converged);
}

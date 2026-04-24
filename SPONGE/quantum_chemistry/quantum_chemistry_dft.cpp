// clang-format off
#include "quantum_chemistry.h"
#include "dft/dft.hpp"
#include "dft/grid.hpp"
#include "dft/xc.hpp"
#include "dft/vxc.hpp"
#include "gradient/grad_dft_xc.hpp"
// clang-format on

void QUANTUM_CHEMISTRY::Build_DFT_VXC()
{
    if (scf_ws.runtime.unrestricted)
        QC_Build_DFT_VXC_UKS(blas_handle, method, mol, dft, cart2sph,
                             scf_ws.ortho.d_norms, scf_ws.alpha.d_P,
                             scf_ws.beta.d_P);
    else
        QC_Build_DFT_VXC_RKS(blas_handle, method, mol, dft, cart2sph,
                             scf_ws.ortho.d_norms, scf_ws.alpha.d_P);
}

void QUANTUM_CHEMISTRY::Build_DFT_XC_Gradient()
{
    const int* d_ao_off_eff =
        mol.is_spherical ? mol.d_ao_offsets_sph : mol.d_ao_offsets;

    if (scf_ws.runtime.unrestricted)
        QC_Build_DFT_XC_Gradient_UKS(
            blas_handle, method, mol, dft, cart2sph, scf_ws.ortho.d_norms,
            scf_ws.alpha.d_P, scf_ws.beta.d_P, grad_ws.d_shell_atom,
            d_ao_off_eff, dft.d_W_full, dft.d_W_sigma, grad_ws.d_grad);
    else
        QC_Build_DFT_XC_Gradient_RKS(
            blas_handle, method, mol, dft, cart2sph, scf_ws.ortho.d_norms,
            scf_ws.alpha.d_P, grad_ws.d_shell_atom, d_ao_off_eff, dft.d_W_full,
            dft.d_W_sigma, grad_ws.d_grad);
}

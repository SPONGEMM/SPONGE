#include "quantum_chemistry.h"


#include "dft/ao.hpp"
#include "dft/dft.hpp"
#include "dft/grid.hpp"
#include "dft/vxc.hpp"
#include "dft/xc.hpp"

void QUANTUM_CHEMISTRY::Build_DFT_VXC()
{
    if (scf_ws.unrestricted)
    {
        QC_Build_DFT_VXC_UKS(
            blas_handle, method, mol.is_spherical, mol.nao_cart, mol.nao,
            dft.max_grid_size, dft.grid_batch_size, mol.nbas,
            dft.d_grid_coords, dft.d_grid_weights, cart2sph.d_cart2sph_mat,
            mol.d_centers, mol.d_l_list, mol.d_exps, mol.d_coeffs,
            mol.d_shell_offsets, mol.d_shell_sizes, mol.d_ao_offsets,
            scf_ws.d_norms, scf_ws.d_P, scf_ws.d_P_b, dft.d_ao_vals_cart,
            dft.d_ao_grad_x_cart, dft.d_ao_grad_y_cart,
            dft.d_ao_grad_z_cart, dft.d_ao_vals, dft.d_ao_grad_x,
            dft.d_ao_grad_y, dft.d_ao_grad_z, dft.d_exc_total, dft.d_Vxc,
            dft.d_Vxc_beta);
    }
    else
    {
        QC_Build_DFT_VXC(
            blas_handle, method, mol.is_spherical, mol.nao_cart, mol.nao,
            dft.max_grid_size, dft.grid_batch_size, mol.nbas,
            dft.d_grid_coords, dft.d_grid_weights, cart2sph.d_cart2sph_mat,
            mol.d_centers, mol.d_l_list, mol.d_exps, mol.d_coeffs,
            mol.d_shell_offsets, mol.d_shell_sizes, mol.d_ao_offsets,
            scf_ws.d_norms, scf_ws.d_P, dft.d_ao_vals_cart,
            dft.d_ao_grad_x_cart, dft.d_ao_grad_y_cart,
            dft.d_ao_grad_z_cart, dft.d_ao_vals, dft.d_ao_grad_x,
            dft.d_ao_grad_y, dft.d_ao_grad_z, dft.d_rho, dft.d_sigma,
            dft.d_exc, dft.d_vrho, dft.d_vsigma, dft.d_exc_total,
            dft.d_Vxc);
    }
}

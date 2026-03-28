#ifndef QC_GUESS_SAP_H
#define QC_GUESS_SAP_H

// Superposition of Atomic Potentials (SAP) initial guess
//
// References:
//   [1] S. Lehtola, "Assessment of Initial Guesses for Self-Consistent Field
//       Calculations. Superposition of Atomic Potentials: Simple yet Efficient",
//       J. Chem. Theory Comput. 15, 1593-1604 (2019).
//       DOI: 10.1021/acs.jctc.8b01089
//
//   [2] S. Lehtola, L. Visscher, E. Engel, "Efficient implementation of the
//       superposition of atomic potentials initial guess for electronic
//       structure calculations in Gaussian basis sets",
//       J. Chem. Phys. 152, 144105 (2020).
//       DOI: 10.1063/5.0004046
//
// SAP fitting parameters from sap_helfem_large basis set (Psi4/BSE),
// originally computed by S. Lehtola using the HelFEM finite-element program.
//
// The atomic potential is expanded as:
//   V_A(r) = -Z_eff(r)/r = -sum_k c_k * erf(sqrt(alpha_k) * r) / r
//
// Matrix elements <mu|V_SAP|nu> are computed analogously to nuclear attraction
// integrals with Gaussian charge distributions. For each fitting term k on
// atom A with exponent alpha_k:
//   Boys argument:  T = p * alpha_k / (p + alpha_k) * |P - A|^2
//   Prefactor:      2 * pi / (p + alpha_k)
// where p = a + b is the sum of basis Gaussian exponents.

#include "../structure/molecule.h"
#include "../structure/scf_workspace.h"

// Compute the SAP potential matrix V_SAP and build initial density from it.
// Diagonalizes (H_core + V_SAP) to obtain initial MOs, then populates P.
void QC_Build_SAP_Guess(const QC_MOLECULE& mol,
                        const QC_SCF_Runtime_State& runtime,
                        const float* d_H_core, const float* d_S,
                        float* d_P, float* d_P_beta);

#endif

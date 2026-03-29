#ifndef QC_GUESS_SAP_H
#define QC_GUESS_SAP_H

// Superposition of Atomic Potentials (SAP) initial guess
//
// Reference:
//   S. Lehtola, "Comment on 'Efficient implementation of the superposition
//   of atomic potentials initial guess for electronic structure calculations
//   in Gaussian basis sets'", arXiv:2603.16989 (2026).
//
// SAP 通过对核吸引积分的 Boys 函数施加修正来实现，无需显式三中心积分。
// 修正公式（Eq. 11）:
//   F_m(T) → F_m(T) - Σ_k c̃_k (α_k/(ζ+α_k))^(m+1/2) F_m(T·α_k/(ζ+α_k))
// 其中 c̃_k = c_k / Z_C, ζ = p (basis pair exponent), T = p * |P-C|²
//
// SAP 拟合参数来自 sap_helfem_large（Psi4/BSE），
// 由 S. Lehtola 使用 HelFEM 有限元程序计算。

#include "../structure/integral_tasks.h"
#include "../structure/molecule.h"

// Compute the SAP potential matrix V_SAP in Cartesian basis.
void QC_Compute_V_SAP(const QC_MOLECULE& mol, const QC_INTEGRAL_TASKS& task_ctx,
                      float* d_V_SAP);

#endif

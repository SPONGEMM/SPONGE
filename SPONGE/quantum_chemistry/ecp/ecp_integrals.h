#pragma once

#include <vector>

#include "../structure/integral_tasks.h"
#include "../structure/molecule.h"

// 球谐→笛卡尔密度变换 (含归一化):
//   M_cart[nc×nc] = C^T (N · M_sph · N) C
// 在 CPU 上执行 (矩阵通常 <100×100, 远小于 PCIe 开销)
// h_norms: [ns], h_C: [ns×nc] row-major, h_M_sph: [ns×ns]
void QC_Sph2Cart_Density_Host(int ns, int nc, const std::vector<float>& h_norms,
                              const std::vector<float>& h_C,
                              const std::vector<float>& h_M_sph,
                              std::vector<float>& h_M_cart);

// 计算 ECP 矩阵 V_ECP (Cartesian 基)
// 输出到 d_V_ECP[nao_cart × nao_cart]，调用前需 memset 为 0
void QC_Compute_V_ECP(const QC_MOLECULE& mol, const QC_INTEGRAL_TASKS& task_ctx,
                      float* d_V_ECP);

// 计算 ECP 梯度: Tr[P · dV_ECP/dR] 对各原子的贡献
// 累加到 d_grad[natm * 3] (double 精度, Hartree/Bohr)
// d_P_cart_eff: Cartesian 基密度矩阵 (含归一化, nao_cart × nao_cart)
//   球谐基时为 C^T(NPN)C, 笛卡尔基时为 P.*norms*norms'
void QC_Compute_ECP_Gradient(const QC_MOLECULE& mol,
                             const QC_INTEGRAL_TASKS& task_ctx,
                             const int* d_shell_atom, const float* d_P_cart_eff,
                             double* d_grad);

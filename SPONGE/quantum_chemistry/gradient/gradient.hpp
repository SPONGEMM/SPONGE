#pragma once

#include "grad_nuclear.hpp"
#include "grad_wdensity.hpp"

// 力/维里回写 kernel
// 将 QC 原子梯度（Hartree/Bohr）转换为 MD 力（kcal/mol/Å）
// 并累加到 MD 力数组；可选地计算维里张量
static __global__ void QC_Writeback_Gradient_Kernel(
    const int natm, const int* atom_local, const double* grad,
    const VECTOR* crd, VECTOR* frc, const int need_virial,
    LTMatrix3* atom_virial)
{
    SIMPLE_DEVICE_FOR(i, natm)
    {
        const int md_idx = atom_local[i];
        const double ha_bohr_to_kcal_ang =
            627.509474 * CONSTANT_ANGSTROM_TO_BOHR;
        const float fx = (float)(-grad[i * 3 + 0] * ha_bohr_to_kcal_ang);
        const float fy = (float)(-grad[i * 3 + 1] * ha_bohr_to_kcal_ang);
        const float fz = (float)(-grad[i * 3 + 2] * ha_bohr_to_kcal_ang);
        atomicAdd(&frc[md_idx].x, fx);
        atomicAdd(&frc[md_idx].y, fy);
        atomicAdd(&frc[md_idx].z, fz);

        if (need_virial && atom_virial != NULL)
        {
            const VECTOR f_vec = {fx, fy, fz};
            const VECTOR r_vec = crd[md_idx];
            atomicAdd(atom_virial + md_idx,
                      Get_Virial_From_Force_Dis(f_vec, r_vec));
        }
    }
}

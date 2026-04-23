#pragma once

#include "../structure/ecp.h"

extern QC_ECP_SET* QC_ECP_DEF2_PTR;
extern QC_ECP_SET* QC_ECP_LANL2DZ_PTR;

// 根据轨道基组名自动选择 ECP
// 返回 NULL 表示该基组不需要 / 无对应 ECP
static inline QC_ECP_SET* QC_Get_Auto_ECP(const char* basis_name)
{
    // def2 系列 → def2-ecp
    if (strstr(basis_name, "def2") != nullptr) return QC_ECP_DEF2_PTR;
    return nullptr;
}

// auto 模式下 ECP 是否应当作用于该原子。
// def2-ECP 的 Turbomole/Ahlrichs 官方规范只覆盖 Rb(37)–Rn(86)；
// K(19)–Kr(36) 虽在 SPONGE 的 def2_ecp.cpp 数据表中有 ECP10MDF
// 记录，但 Turbomole/ORCA/PySCF 等默认走全电子，SPONGE auto 模式
// 须与之一致，故这里按 Z 过滤。显式 qc_ecp=def2-ecp 不过滤。
static inline bool QC_Auto_ECP_Applies(const char* basis_name, int Z)
{
    if (strstr(basis_name, "def2") != nullptr) return Z >= 37;
    return true;
}

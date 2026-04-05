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

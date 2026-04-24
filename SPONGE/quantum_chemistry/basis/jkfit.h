#pragma once
#include <algorithm>
#include <cstring>
#include <string>

#include "jkfit/def2_universal_jkfit.h"
#include "prototype.h"

// 辅助基组 orbital basis → JKFIT 映射
// 返回对应的 JKFIT 辅助基组指针，若无匹配则返回 nullptr
inline QC_BASIS_SET* QC_Get_JKFIT_Basis(const char* orbital_basis_name)
{
    std::string name(orbital_basis_name);
    std::transform(name.begin(), name.end(), name.begin(), ::tolower);

    // def2 系列统一使用 def2-universal-JKFIT
    if (name == "def2-svp" || name == "def2-svpd" || name == "def2-tzvp" ||
        name == "def2-tzvpd" || name == "def2-tzvpp" || name == "def2-qzvp" ||
        name == "ma-def2-svp" || name == "ma-def2-tzvp")
    {
        return QC_BASIS_DEF2_UNIVERSAL_JKFIT_PTR;
    }

    // Pople 系列: def2-universal-JKFIT 作为通用辅助基组
    if (name == "sto-3g" || name == "3-21g" || name == "6-31g" ||
        name == "6-31g*" || name == "6-31g**" || name == "6-31+g" ||
        name == "6-31+g*" || name == "6-31+g**" || name == "6-31++g" ||
        name == "6-31++g**" || name == "6-311g" || name == "6-311g*" ||
        name == "6-311g**" || name == "6-311+g*" || name == "6-311++g**")
    {
        return QC_BASIS_DEF2_UNIVERSAL_JKFIT_PTR;
    }

    // cc 系列: def2-universal-JKFIT 作为通用辅助基组
    if (name == "cc-pvdz" || name == "cc-pvtz" || name == "aug-cc-pvdz" ||
        name == "aug-cc-pvtz")
    {
        return QC_BASIS_DEF2_UNIVERSAL_JKFIT_PTR;
    }

    return nullptr;
}

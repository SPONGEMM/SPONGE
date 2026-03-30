#ifndef QC_GRADIENT_WORKSPACE_H
#define QC_GRADIENT_WORKSPACE_H

#include "../../common.h"

struct QC_GRAD_WORKSPACE
{
    double* d_grad = NULL;           // [natm * 3], double 精度避免累积误差
    int* d_shell_atom = NULL;        // [nbas], 从 bas[ish*8+0] 预计算
    int* d_shell_atom_aux = NULL;    // [naux_bas], RI 梯度用
    float* d_W_density = NULL;       // [nao * nao] 能量加权密度矩阵
    float* d_W_density_beta = NULL;  // UHF beta 通道
};

#endif

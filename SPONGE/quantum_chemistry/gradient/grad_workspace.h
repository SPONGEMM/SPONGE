#ifndef QC_GRADIENT_WORKSPACE_H
#define QC_GRADIENT_WORKSPACE_H

#include "../../common.h"

// 解析梯度工作空间
struct QC_GRAD_WORKSPACE
{
    // 原子梯度累加器 [natm * 3]，double 精度避免 float 累积误差
    double* d_grad = NULL;

    // 壳层到原子映射 [nbas]，从 bas[ish*8+0] 预计算
    int* d_shell_atom = NULL;

    // 能量加权密度矩阵 W [nao * nao]
    // W_μν = occ_factor × Σ_i ε_i × C_μi × C_νi
    float* d_W_density = NULL;

    // UHF 时 beta 通道的 W
    float* d_W_density_beta = NULL;
};

#endif

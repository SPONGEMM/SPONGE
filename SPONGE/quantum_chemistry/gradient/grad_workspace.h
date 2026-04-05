#pragma once

#include "../../common.h"

struct QC_GRAD_WORKSPACE
{
    double* d_grad = NULL;           // [natm * 3], double 精度避免累积误差
    int* d_shell_atom = NULL;        // [nbas], 从 bas[ish*8+0] 预计算
    int* d_shell_atom_aux = NULL;    // [naux_bas], RI 梯度用
    float* d_W_density = NULL;       // [nao * nao] 能量加权密度矩阵
    float* d_W_density_beta = NULL;  // UHF beta 通道
    // 球谐→笛卡尔 1e 梯度缓冲 (is_spherical 时预分配, 避免每次梯度 malloc)
    float* d_P_cart = NULL;          // [nao_cart²]
    float* d_W_cart = NULL;          // [nao_cart²]
    float* d_norms_ones = NULL;      // [nao_cart], all 1.0
};


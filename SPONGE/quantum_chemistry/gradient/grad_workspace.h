#pragma once

#include "../../common.h"

// GPU ERI 梯度多副本缓冲的 copy 数（降低 atomicAdd 竞争）
static constexpr int QC_GRAD_N_COPIES = 64;

struct QC_GRAD_WORKSPACE
{
    double* d_grad = NULL;           // [natm * 3], double 精度避免累积误差
    int* d_shell_atom = NULL;        // [nbas], 从 bas[ish*8+0] 预计算
    int* d_shell_atom_aux = NULL;    // [naux_bas], RI 梯度用
    float* d_W_density = NULL;       // [nao * nao] 能量加权密度矩阵
    float* d_W_density_beta = NULL;  // UHF beta 通道
    // 球谐→笛卡尔 1e 梯度缓冲 (is_spherical 时预分配, 避免每次梯度 malloc)
    float* d_P_cart = NULL;           // [nao_cart²]
    float* d_W_cart = NULL;           // [nao_cart²]
    float* d_norms_ones = NULL;       // [nao_cart], all 1.0
    float* d_grad_gamma_pool = NULL;  // [slots * 2 * buf_size], ERI 梯度临时池
    int grad_gamma_buf_size = 0;      // 单个 gamma buffer 元素数
    int grad_gamma_pool_slots = 0;    // 当前已分配 slots 数

    // GPU ERI 梯度持久化缓冲（原先是 function-local static，现移入 workspace
    // 使生命周期随 QUANTUM_CHEMISTRY 实例管理）
    int* d_combo_prefix_grad = NULL;  // [n_combos + 1]
    double* d_grad_copies = NULL;     // [QC_GRAD_N_COPIES * natm * 3]
};

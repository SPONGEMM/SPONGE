#pragma once

#include "../../common.h"

// RI (Density Fitting) 工作空间
// 支持两种模式:
//   stored: 预存 eri3c [naux×nao²] 和 B [naux×nao²]（快，内存大）
//   direct: 在线计算 3c 积分，不存储 eri3c/B（省内存，每轮多计算一次 3c）
struct QC_RI_WORKSPACE
{
    bool enabled = false;
    bool direct = false;  // 运行时标志: true 使用 direct 模式

    // 用户设置的模式偏好: auto/stored/direct
    // auto: 由 3c 张量大小自动决定（默认）
    // stored/direct: 强制对应模式
    enum DF_MODE
    {
        DF_AUTO,
        DF_STORED,
        DF_DIRECT
    };
    DF_MODE mode = DF_AUTO;

    // 辅助基组信息
    int naux = 0;       // 辅助基函数总数（球谐）
    int naux_cart = 0;  // 辅助基函数总数（笛卡尔）
    int naux_bas = 0;   // 辅助壳层总数

    std::vector<int> h_aux_l_list;
    int* d_aux_l_list = NULL;

    std::vector<int> h_aux_shell_offsets;
    int* d_aux_shell_offsets = NULL;

    std::vector<int> h_aux_shell_sizes;
    int* d_aux_shell_sizes = NULL;

    std::vector<int> h_aux_ao_offsets;  // 笛卡尔 AO 偏移
    int* d_aux_ao_offsets = NULL;

    std::vector<int> h_aux_ao_offsets_sph;  // 球谐 AO 偏移
    int* d_aux_ao_offsets_sph = NULL;

    std::vector<float> h_aux_exps;
    float* d_aux_exps = NULL;

    std::vector<float> h_aux_coeffs;
    float* d_aux_coeffs = NULL;

    std::vector<VECTOR> h_aux_centers;
    VECTOR* d_aux_centers = NULL;

    std::vector<float> h_aux_norms;
    float* d_aux_norms = NULL;

    // 二中心 metric (P|Q) 及其分解（两种模式均存储）
    double* d_metric = NULL;           // [naux × naux] 对称
    double* d_metric_inv = NULL;       // [naux × naux] (P|Q)^{-1} (RI-J)
    double* d_metric_inv_sqrt = NULL;  // [naux × naux] (P|Q)^{-1/2} (RI-K)

    // stored 模式专用
    double* d_eri3c = NULL;  // [naux × nao × nao] double（仅 stored）
    float* d_B = NULL;       // [naux × nao × nao] float（仅 stored）

    // 每轮迭代 scratch
    double* d_d_vec = NULL;  // [naux] 密度拟合向量
    double* d_g_vec = NULL;  // [naux] 求解后的拟合系数
    float* d_B_occ = NULL;   // [naux × nao × nocc] B 与 MO 系数收缩

    // ---- 持久化 scratch（消除每轮 malloc/free）----
    double* d_P_double = NULL;  // [nao²] RI-J 密度矩阵 double 缓冲
    double* d_J_double = NULL;  // [nao²] RI-J Coulomb double 缓冲
    float* d_J_float = NULL;    // [nao²] RI-J Coulomb float 缓冲
    float* d_K_scratch = NULL;  // [nao²] RI-K exchange 累加缓冲
    float* d_B_flat = NULL;     // [nao × naux × nocc] RI-K 重排缓冲

    // 辅助基 cart2sph（host，用于在线变换）
    std::vector<float> h_U_aux;  // [naux_cart × naux] 行优先
    std::vector<float> h_U_orb;  // [nao_cart × nao] 行优先

    // 特征分解数据（梯度用 D2_K Daleckii-Kreĭn 公式）
    std::vector<double> h_eigval;  // [naux] metric 特征值
    std::vector<double> h_eigvec;  // [naux × naux] 列优先特征向量

    // host 侧缓存: RI_Precompute 后不变，避免每轮 Build_Fock_RI_Direct D2H
    std::vector<double> h_metric_inv;       // [naux × naux]
    std::vector<double> h_metric_inv_sqrt;  // [naux × naux]

    // 内存管理
    int naux_eff = 0;  // 去除线性依赖后的有效辅助基维度

    // 辅助基 atm/bas/env（积分内核用，同 PySCF 格式）
    std::vector<int> h_aux_atm;
    int* d_aux_atm = NULL;
    std::vector<int> h_aux_bas;
    int* d_aux_bas = NULL;
    std::vector<float> h_aux_env;
    float* d_aux_env = NULL;
};

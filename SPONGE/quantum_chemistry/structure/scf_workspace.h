#ifndef QC_STRUCTURE_SCF_WORKSPACE_H
#define QC_STRUCTURE_SCF_WORKSPACE_H

#include "../../common.h"

// 持久 AO 核心矩阵与能量结果缓存
struct QC_SCF_Core_Matrices
{
    float* d_S = NULL;
    float* d_T = NULL;
    float* d_V = NULL;
    float* d_H_core = NULL;
    double* d_scf_energy = NULL;
    double* d_nuc_energy_dev = NULL;
};

// 单个自旋通道的 Fock、密度与轨道系数工作区
struct QC_SCF_Spin_Channel
{
    std::vector<float> h_F;
    float* d_F = NULL;
    std::vector<float> h_P;
    float* d_P = NULL;
    std::vector<float> h_P_new;
    float* d_P_new = NULL;
    std::vector<float> h_C;
    float* d_C = NULL;
    double* d_F_double = NULL;
};

// 重叠正交化、本征分解与双精度临时缓冲
struct QC_SCF_Ortho_Workspace
{
    std::vector<float> h_X;
    double* d_X = NULL;
    std::vector<float> h_W;
    float* d_W = NULL;
    std::vector<float> h_Work;
    float* d_Work = NULL;
    float* d_solver_work = NULL;
    int* d_solver_iwork = NULL;
    float* d_norms = NULL;

    double* d_dwork_nao2_1 = NULL;
    double* d_dwork_nao2_2 = NULL;
    double* d_dwork_nao2_3 = NULL;
    double* d_dwork_nao2_4 = NULL;
    double* d_dW_double = NULL;
    double* d_solver_work_double = NULL;

    int lwork = 0;
    int liwork = 0;
    int lwork_double = 0;
    double lindep_threshold = 1e-6;
    int nao_eff = 0;
};

// DIIS/EDIIS/ADIIS/MESA 迭代加速相关缓冲与历史状态
//
// MESA 算法参考:
//   S. Lehtola, "OpenOrbitalOptimizer - a reusable open source library for
//   self-consistent field calculations", arXiv:2503.23034 (2025).
//
// EDIIS 参考:
//   K. N. Kudin, G. E. Scuseria, E. Cancès, J. Chem. Phys. 116, 8255 (2002).
//
// ADIIS 参考:
//   X. Hu, W. Yang, J. Chem. Phys. 132, 054109 (2010).
struct QC_SCF_DIIS_Workspace
{
    double* d_diis_err = NULL;
    float *d_diis_w1 = NULL, *d_diis_w2 = NULL, *d_diis_w3 = NULL,
          *d_diis_w4 = NULL;
    // CDIIS: Fock 和误差向量历史
    std::vector<double*> d_diis_f_hist;
    std::vector<double*> d_diis_e_hist;
    std::vector<double*> d_diis_f_hist_b;
    std::vector<double*> d_diis_e_hist_b;

    // EDIIS/ADIIS: 密度矩阵历史和 SCF 能量历史
    std::vector<double*> d_diis_d_hist;
    std::vector<double*> d_diis_d_hist_b;
    std::vector<double> energy_hist;

    int diis_hist_count = 0;
    int diis_hist_head = 0;
    int diis_hist_count_b = 0;
    int diis_hist_head_b = 0;

    // MESA → CDIIS 切换阈值
    double mesa_to_cdiis_threshold = 1e-1;

    double* d_diis_accum = NULL;
};

// direct SCF 的 pair density、线程私有 Fock 与 ERI 工作池
struct QC_SCF_Direct_Workspace
{
    float* d_pair_density_coul = NULL;
    float* d_pair_density_exx = NULL;
    float* d_pair_density_exx_b = NULL;
    float* d_hr_pool = NULL;
    void* h_cpu_bra_terms = NULL;
    void* h_cpu_ket_terms = NULL;

    int fock_thread_count = 1;
    double* d_F_thread = NULL;
    double* d_F_b_thread = NULL;

    float* d_Ptot = NULL;
    float* d_P_coul = NULL;
};

// SCF 配置、收敛状态与能量累计缓冲
struct QC_SCF_Runtime_State
{
    bool unrestricted = false;
    int n_alpha = 0;
    int n_beta = 0;
    float occ_factor = 2.0f;
    int max_scf_iter = 100;
    bool use_diis = true;
    int diis_start_iter = 8;
    int diis_space = 6;
    double diis_reg = 1e-10;
    double energy_tol = 1e-6;
    double level_shift = 0.25;
    bool print_iter = false;

    double spin_square = 0.0;
    double spin_square_exact = 0.0;

    double* d_e = NULL;
    double* d_e_b = NULL;
    double* d_pvxc = NULL;
    double* d_prev_energy = NULL;
    double* d_delta_e = NULL;
    double* d_density_residual = NULL;
    int* d_converged = NULL;
};

// SCF 工作空间（初始化分配，循环中复用）
struct QC_SCF_WORKSPACE
{
    QC_SCF_Core_Matrices core;
    QC_SCF_Spin_Channel alpha;
    QC_SCF_Spin_Channel beta;
    QC_SCF_Ortho_Workspace ortho;
    QC_SCF_DIIS_Workspace diis;
    QC_SCF_Direct_Workspace direct;
    QC_SCF_Runtime_State runtime;
};

#endif

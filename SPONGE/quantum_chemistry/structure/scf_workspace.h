#ifndef QC_STRUCTURE_SCF_WORKSPACE_H
#define QC_STRUCTURE_SCF_WORKSPACE_H

#include "../../common.h"

// SCF 工作空间（初始化分配，循环中复用）
struct QC_SCF_WORKSPACE
{
    // 持久 AO 核心矩阵与最终结果缓存
    float* d_S = NULL;
    float* d_T = NULL;
    float* d_V = NULL;
    float* d_H_core = NULL;
    double* d_scf_energy = NULL;
    double* d_nuc_energy_dev = NULL;

    // 重叠正交化与本征分解工作区
    std::vector<float> h_X;
    std::vector<double> h_X_double;
    float* d_X = NULL;
    std::vector<float> h_W;
    float* d_W = NULL;
    std::vector<float> h_Work;
    float* d_Work = NULL;
    std::vector<float> h_U;
    float* d_solver_work = NULL;
    int* d_solver_iwork = NULL;
    float* d_norms = NULL;

    // 主自旋（alpha/限制性）Fock、密度与分子轨道系数工作区
    std::vector<float> h_F;
    float* d_F = NULL;
    std::vector<float> h_P;
    float* d_P = NULL;
    std::vector<float> h_P_new;
    float* d_P_new = NULL;
    std::vector<float> h_Tmp;
    float* d_Tmp = NULL;
    std::vector<float> h_Fp;
    float* d_Fp = NULL;
    std::vector<float> h_C;
    float* d_C = NULL;

    // 次自旋（非限制性的beta）Fock、密度与分子轨道系数工作区
    std::vector<float> h_F_b;
    float* d_F_b = NULL;
    std::vector<float> h_P_b;
    float* d_P_b = NULL;
    std::vector<float> h_P_b_new;
    float* d_P_b_new = NULL;
    float* d_Ptot = NULL;
    std::vector<float> h_Fp_b;
    float* d_Fp_b = NULL;
    std::vector<float> h_C_b;
    float* d_C_b = NULL;
    std::vector<float> h_W_b;
    std::vector<float> h_Work_b;
    std::vector<float> h_Tmp_b;

    // DIIS 误差与历史向量缓冲
    float *d_diis_err = NULL, *d_diis_w1 = NULL, *d_diis_w2 = NULL,
          *d_diis_w3 = NULL, *d_diis_w4 = NULL;
    std::vector<float*> d_diis_f_hist;
    std::vector<float*> d_diis_e_hist;
    std::vector<float*> d_diis_f_hist_b;
    std::vector<float*> d_diis_e_hist_b;

    // 能量累计、收敛状态与线性求解信息
    double *d_e = NULL, *d_e_b = NULL, *d_pvxc = NULL, *d_prev_energy = NULL,
           *d_delta_e = NULL, *d_density_residual = NULL,
           *d_diis_accum = NULL, *d_diis_B = NULL, *d_diis_rhs = NULL;
    int *d_converged = NULL, *d_diis_info = NULL, *d_info = NULL;
    int lwork = 0;
    int liwork = 0;

    // direct SCF shell-pair density screening buffers
    float* d_pair_density_coul = NULL;
    float* d_pair_density_exx = NULL;
    float* d_pair_density_exx_b = NULL;

    // CPU direct SCF thread-private Fock accumulation buffers
    int fock_thread_count = 1;
    float* d_F_thread = NULL;
    float* d_F_b_thread = NULL;

    // SCF 配置与每轮 Solve_SCF 写入的派生参数
    bool unrestricted = false;
    int n_alpha = 0;
    int n_beta = 0;
    float occ_factor = 2.0f;
    float density_mixing = 0.20f;
    int max_scf_iter = 100;
    bool use_diis = true;
    int diis_start_iter = 8;
    int diis_space = 6;
    double diis_reg = 1e-10;
    double energy_tol = 1e-6;
    float overlap_eig_floor = 1e-10f;
    bool print_iter = false;
    bool profile_build_fock = false;
    float* d_P_coul = NULL;

    // DIIS 循环状态
    int diis_hist_count = 0;
    int diis_hist_head = 0;
    int diis_hist_count_b = 0;
    int diis_hist_head_b = 0;
    double diis_best_energy = 1e30;
    int diis_stagnant_count = 0;
    int diis_cooldown = 0;
};

#endif

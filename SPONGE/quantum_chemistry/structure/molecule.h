#pragma once

#include "../../common.h"

// 原子序数转元素符号查找表（当前覆盖 H-Kr）
static const std::map<int, std::string> QC_SYMBOL_FROM_Z = {
    {0, "X"},   {1, "H"},   {2, "He"},  {3, "Li"},  {4, "Be"},  {5, "B"},
    {6, "C"},   {7, "N"},   {8, "O"},   {9, "F"},   {10, "Ne"}, {11, "Na"},
    {12, "Mg"}, {13, "Al"}, {14, "Si"}, {15, "P"},  {16, "S"},  {17, "Cl"},
    {18, "Ar"}, {19, "K"},  {20, "Ca"}, {21, "Sc"}, {22, "Ti"}, {23, "V"},
    {24, "Cr"}, {25, "Mn"}, {26, "Fe"}, {27, "Co"}, {28, "Ni"}, {29, "Cu"},
    {30, "Zn"}, {31, "Ga"}, {32, "Ge"}, {33, "As"}, {34, "Se"}, {35, "Br"},
    {36, "Kr"}};

// 元素符号转原子序数查找表（当前覆盖 H-Kr）
static const std::map<std::string, int> QC_Z_FROM_SYMBOL = {
    {"H", 1},   {"He", 2},  {"Li", 3},  {"Be", 4},  {"B", 5},   {"C", 6},
    {"N", 7},   {"O", 8},   {"F", 9},   {"Ne", 10}, {"Na", 11}, {"Mg", 12},
    {"Al", 13}, {"Si", 14}, {"P", 15},  {"S", 16},  {"Cl", 17}, {"Ar", 18},
    {"K", 19},  {"Ca", 20}, {"Sc", 21}, {"Ti", 22}, {"V", 23},  {"Cr", 24},
    {"Mn", 25}, {"Fe", 26}, {"Co", 27}, {"Ni", 28}, {"Cu", 29}, {"Zn", 30},
    {"Ga", 31}, {"Ge", 32}, {"As", 33}, {"Se", 34}, {"Br", 35}, {"Kr", 36}};

// 量化壳层
struct QC_SHELL
{
    int l;
    std::vector<float> exps;
    std::vector<float> coeffs;
};

// 量化分子
struct QC_MOLECULE
{
    // 原子数
    int natm;
    // 电子总数
    int nelectron;
    // 笛卡尔 AO 总数
    int nao_cart;
    // 球谐 AO 总数
    int nao_sph;
    // 是否使用球谐基（1: 是，0: 否）
    int is_spherical;
    // 当前有效 AO 总数（由 is_spherical 选择球谐或笛卡尔）
    int nao = 0;
    // 当前有效 AO 总数平方
    int nao2 = 0;
    // 壳层总数
    int nbas;
    // 分子总电荷
    int charge;
    // 自旋多重度
    int multiplicity;
    // 壳层数据
    std::vector<QC_SHELL> shells;

    // 每个原子的（有效）核电荷数
    std::vector<int> h_Z;
    int* d_Z = NULL;

    // 每个壳层中心坐标
    std::vector<VECTOR> h_centers;
    VECTOR* d_centers = NULL;

    // 每个壳层的角量子数
    std::vector<int> h_l_list;
    int* d_l_list = NULL;

    // 所有壳层指数参数拼接
    std::vector<float> h_exps;
    float* d_exps = NULL;

    // 所有壳层收缩系数拼接
    std::vector<float> h_coeffs;
    float* d_coeffs = NULL;

    // 每个壳层的起始偏移
    std::vector<int> h_shell_offsets;
    int* d_shell_offsets = NULL;

    // 每个壳层的高斯函数数量
    std::vector<int> h_shell_sizes;
    int* d_shell_sizes = NULL;

    // 每个壳层的原子轨道起始偏移（笛卡尔）
    std::vector<int> h_ao_offsets;
    int* d_ao_offsets = NULL;

    // 每个壳层的原子轨道起始偏移（球谐/有效 AO）
    std::vector<int> h_ao_offsets_sph;
    int* d_ao_offsets_sph = NULL;

    // 原子参数数组（积分内核使用）
    std::vector<int> h_atm;
    int* d_atm = NULL;

    // 基函数参数数组（积分内核使用）
    std::vector<int> h_bas;
    int* d_bas = NULL;

    // 环境参数数组（积分内核使用）
    std::vector<float> h_env;
    float* d_env = NULL;

    // 原子坐标 (按原子索引, ECP/梯度内核使用)
    std::vector<VECTOR> h_atom_coords;  // [natm]
    VECTOR* d_atom_coords = NULL;

    // ECP 数据
    bool has_ecp = false;
    // 每个原子的核心电子数和 l_max (-1 = 无 ECP)
    std::vector<int> h_ecp_n_core;  // [natm]
    std::vector<int> h_ecp_l_max;   // [natm]
    int* d_ecp_l_max = NULL;
    // 扁平化通道数据 (device kernel 使用)
    int ecp_total_channels = 0;
    int ecp_total_terms = 0;
    std::vector<float> h_ecp_d;              // 所有 d_k 拼接
    std::vector<float> h_ecp_zeta;           // 所有 ζ_k 拼接
    std::vector<int> h_ecp_n;                // 所有 n_k 拼接
    std::vector<int> h_ecp_l;                // 每个 channel 的 l 值
    std::vector<int> h_ecp_channel_offsets;  // 每个 channel 在扁平数组中的偏移
    std::vector<int> h_ecp_channel_sizes;    // 每个 channel 的 term 数
    std::vector<int>
        h_ecp_atom_channel_range;  // [natm+1] 原子 i 的 channel 范围
    // Device 指针
    float* d_ecp_d = NULL;
    float* d_ecp_zeta = NULL;
    int* d_ecp_n = NULL;
    int* d_ecp_l = NULL;
    int* d_ecp_channel_offsets = NULL;
    int* d_ecp_channel_sizes = NULL;
    int* d_ecp_atom_channel_range = NULL;
};

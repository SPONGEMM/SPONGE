#ifndef QC_STRUCTURE_DFT_H
#define QC_STRUCTURE_DFT_H

#include "../../common.h"

// DFT 配置
struct QC_DFT
{
    // 杂化泛函的精确交换占比
    float exx_fraction = 1.0f;
    // 是否启用 DFT 计算路径
    int enable_dft = 0;
    // 分子积分网格控制参数
    int dft_radial_points = 60;
    int dft_angular_points = 194;

    // 网格点坐标、权重与容量
    std::vector<float> h_grid_coords;   // [max_grid_capacity * 3]
    std::vector<float> h_grid_weights;  // [max_grid_capacity]
    float* d_grid_coords = NULL;        // [max_grid_capacity * 3]
    float* d_grid_weights = NULL;       // [max_grid_capacity]
    int max_grid_capacity = 0;
    int max_grid_size = 0;
    int grid_batch_size = 8192;

    // DFT 网格上的 AO 值与梯度（按批处理）
    float* d_ao_vals = NULL;
    float* d_ao_grad_x = NULL;
    float* d_ao_grad_y = NULL;
    float* d_ao_grad_z = NULL;
    // 球谐基变换前的笛卡尔中间量
    float* d_ao_vals_cart = NULL;
    float* d_ao_grad_x_cart = NULL;
    float* d_ao_grad_y_cart = NULL;
    float* d_ao_grad_z_cart = NULL;

    // 密度与泛函计算中间量
    double* d_rho = NULL;
    double* d_sigma = NULL;
    double* d_exc = NULL;
    double* d_vrho = NULL;
    double* d_vsigma = NULL;

    // 交换关联项输出
    float* d_Vxc = NULL;
    float* d_Vxc_beta = NULL;
    double* d_exc_total = NULL;
};

#endif

#pragma once

#include "../../common.h"

// 笛卡尔变球谐的缓冲变量
struct QC_CARTESIAN_TO_SPHERICAL
{
    // 笛卡尔缓冲变量
    float* d_S_cart = NULL;
    float* d_T_cart = NULL;
    float* d_V_cart = NULL;
    float* d_cart2sph_mat = NULL;
    float* d_cart2sph_1e_tmp = NULL;
};

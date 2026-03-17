#include "cmap.h"

#include "../xponge/load/native/cmap.hpp"
#include "../xponge/xponge.h"

// clang-format off
// 由于求导带来的系数矩阵的逆矩阵A_inv
static const float A_inv[16][16] =
{ { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
{ 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, },
{ -3, 0, 3, 0, 0, 0, 0, 0, -2, 0, -1, 0, 0, 0, 0, 0, },
{ 2, 0, -2, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, },
{ 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, },
{ 0, 0, 0, 0, -3, 0, 3, 0, 0, 0, 0, 0, -2, 0, -1, 0, },
{ 0, 0, 0, 0, 2, 0, -2, 0, 0, 0, 0, 0, 1, 0, 1, 0, },
{ -3, 3, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
{ 0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, -2, -1, 0, 0, },
{ 9, -9, -9, 9, 6, 3, -6, -3, 6, -6, 3, -3, 4, 2, 2, 1, },
{ -6, 6, 6, -6, -4, -2, 4, 2, -3, 3, -3, 3, -2, -1, -2, -1, },
{ 2, -2, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
{ 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 1, 1, 0, 0, },
{ -6, 6, 6, -6, -3, -3, 3, 3, -4, 4, -2, 2, -2, -2, -1, -1, },
{ 4, -4, -4, 4, 2, 2, -2, -2, 2, -2, 2, -2, 1, 1, 1, 1 } };
// clang-format on

void CMAP::Initial(CONTROLLER* controller, const char* module_name)
{
    if (module_name == NULL)
    {
        strcpy(this->module_name, "cmap");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }

    const auto& cmap = Xponge::system.classical_force_field.cmap;
    Xponge::CMap local_cmap;
    const Xponge::CMap* cmap_to_use = NULL;
    const char* init_source = NULL;
    if (module_name == NULL)
    {
        cmap_to_use = &cmap;
        init_source = "Xponge::system";
    }
    else if (controller->Command_Exist(this->module_name, "in_file"))
    {
        Xponge::Native_Load_CMap(&local_cmap, controller, this->module_name);
        cmap_to_use = &local_cmap;
    }

    tot_cmap_num = 0;
    uniq_cmap_num = 0;
    uniq_gridpoint_num = 0;
    if (cmap_to_use != NULL)
    {
        tot_cmap_num = static_cast<int>(cmap_to_use->atom_a.size());
        uniq_cmap_num = cmap_to_use->unique_type_numbers;
        uniq_gridpoint_num = cmap_to_use->unique_gridpoint_numbers;
    }
    if (tot_cmap_num > 0)
    {
        if (module_name == NULL)
        {
            controller->printf("START INITIALIZING CMAP (%s):\n", init_source);
        }
        else
        {
            controller->printf("START INITIALIZING CMAP (%s_in_file):\n",
                               this->module_name);
        }
        controller->printf(
            "    total CMAP number is %d\n    unique CMAP number is %d\n",
            tot_cmap_num, uniq_cmap_num);
        this->Memory_Allocate();
        Malloc_Safely((void**)&grid_value, sizeof(float) * uniq_gridpoint_num);
        Malloc_Safely((void**)&h_inter_coeff,
                      sizeof(float) * 16 * uniq_gridpoint_num);
        for (int i = 0; i < uniq_cmap_num; i++)
        {
            h_cmap_resolution[i] = cmap_to_use->resolution[i];
            type_offset[i] = cmap_to_use->type_offset[i];
        }
        memcpy(grid_value, cmap_to_use->grid_value.data(),
               sizeof(float) * uniq_gridpoint_num);
        for (int i = 0; i < tot_cmap_num; i++)
        {
            h_atom_a[i] = cmap_to_use->atom_a[i];
            h_atom_b[i] = cmap_to_use->atom_b[i];
            h_atom_c[i] = cmap_to_use->atom_c[i];
            h_atom_d[i] = cmap_to_use->atom_d[i];
            h_atom_e[i] = cmap_to_use->atom_e[i];
            h_cmap_type[i] = cmap_to_use->cmap_type[i];
        }
        is_initialized = 1;
    }

    if (is_initialized && !is_controller_printf_initialized)
    {
        controller->Step_Print_Initial(this->module_name, "%.6f");
        is_controller_printf_initialized = 1;
        controller->printf("    structure last modify date is %d\n",
                           last_modify_date);
    }
    if (is_initialized)
    {
        // 完成插值系数计算，完成初始化
        this->Interpolation(controller);
        Parameter_Host_to_Device();
        controller->printf("END INITIALIZING CMAP\n\n");
    }
    else
    {
        controller->printf("CMAP IS NOT INITIALIZED\n\n");
    }
}

void CMAP::Parameter_Host_to_Device()
{
    Device_Malloc_And_Copy_Safely((void**)&d_atom_a, h_atom_a,
                                  sizeof(int) * tot_cmap_num);
    Device_Malloc_And_Copy_Safely((void**)&d_atom_b, h_atom_b,
                                  sizeof(int) * tot_cmap_num);
    Device_Malloc_And_Copy_Safely((void**)&d_atom_c, h_atom_c,
                                  sizeof(int) * tot_cmap_num);
    Device_Malloc_And_Copy_Safely((void**)&d_atom_d, h_atom_d,
                                  sizeof(int) * tot_cmap_num);
    Device_Malloc_And_Copy_Safely((void**)&d_atom_e, h_atom_e,
                                  sizeof(int) * tot_cmap_num);
    Device_Malloc_And_Copy_Safely((void**)&d_cmap_type, h_cmap_type,
                                  sizeof(int) * tot_cmap_num);
    Device_Malloc_And_Copy_Safely((void**)&d_coeff_ptr, h_coeff_ptr,
                                  sizeof(float*) * tot_cmap_num);
    Device_Malloc_And_Copy_Safely((void**)&d_inter_coeff, h_inter_coeff,
                                  sizeof(int) * 16 * uniq_gridpoint_num);
    Device_Malloc_And_Copy_Safely((void**)&d_cmap_resolution, h_cmap_resolution,
                                  sizeof(int) * uniq_cmap_num);
    for (int i = 0; i < tot_cmap_num; i++)
    {
        h_coeff_ptr[i] = d_inter_coeff + type_offset[h_cmap_type[i]];
    }
    Device_Malloc_And_Copy_Safely((void**)&d_coeff_ptr, h_coeff_ptr,
                                  sizeof(float*) * tot_cmap_num);

    Device_Malloc_Safely((void**)&d_atom_a_local, sizeof(int) * tot_cmap_num);
    Device_Malloc_Safely((void**)&d_atom_b_local, sizeof(int) * tot_cmap_num);
    Device_Malloc_Safely((void**)&d_atom_c_local, sizeof(int) * tot_cmap_num);
    Device_Malloc_Safely((void**)&d_atom_d_local, sizeof(int) * tot_cmap_num);
    Device_Malloc_Safely((void**)&d_atom_e_local, sizeof(int) * tot_cmap_num);
    Device_Malloc_Safely((void**)&d_cmap_type_local,
                         sizeof(int) * tot_cmap_num);
    Device_Malloc_Safely((void**)&d_num_cmap_local, sizeof(int));
    deviceMemset(d_num_cmap_local, 0, sizeof(int));
}

void CMAP::Memory_Allocate()
{
    Malloc_Safely((void**)&h_cmap_resolution, sizeof(int) * uniq_cmap_num);
    Malloc_Safely((void**)&h_cmap_type, sizeof(float) * tot_cmap_num);
    Malloc_Safely((void**)&h_atom_a, sizeof(int) * tot_cmap_num);
    Malloc_Safely((void**)&h_atom_b, sizeof(int) * tot_cmap_num);
    Malloc_Safely((void**)&h_atom_c, sizeof(int) * tot_cmap_num);
    Malloc_Safely((void**)&h_atom_d, sizeof(int) * tot_cmap_num);
    Malloc_Safely((void**)&h_atom_e, sizeof(int) * tot_cmap_num);
    Malloc_Safely((void**)&type_offset, sizeof(int) * uniq_cmap_num);
    Malloc_Safely((void**)&h_coeff_ptr, sizeof(float*) * tot_cmap_num);
    Device_Malloc_Safely((void**)&d_sigma_of_cmap_ene, sizeof(float));
    Device_Malloc_Safely((void**)&d_cmap_ene, sizeof(float) * tot_cmap_num);
}

void CMAP::Interpolation(CONTROLLER* controller)
{
    // 临时储存节点的值和差分
    float f[4][4];
    float p[16];

    controller->printf("    Start Interpolating the CMAP Grid Value\n");
    // 首先从统一读入的CMAP格点数据中截取出需要插值的数据
    float* temp_grid_value = grid_value;      // 临时格点数据
    float* temp_inter_coeff = h_inter_coeff;  // 临时储存插值系数
    int temp_reso;                            // 标记格点分辨率

    int phi_index = 0, psi_index = 0;
    // 插值数据结构为：
    //                           psi
    //                 - - - - - ... - - - - -
    //                 - - - - - ... - - - - -
    //            phi        .
    //                       .
    //                         .
    //                 - - - - - ... - - - - -
    // 规模为 resolution*resolution
    for (int k = 0; k < uniq_cmap_num; k++)
    {
        temp_reso = h_cmap_resolution[k];
        for (int i = 0; i < temp_reso * temp_reso; i++)
        {
            // 对每个单元进行插值
            psi_index = i % (temp_reso);
            phi_index = (i - psi_index) / (temp_reso);
            for (int m = 0; m < 4; m++)
            {
                for (int n = 0; n < 4; n++)
                {
                    // 引入周期性的读取方式
                    if (phi_index + m - 1 >= 0 && psi_index + n - 1 >= 0)
                        f[m][n] = temp_grid_value
                            [((phi_index + m - 1) % (temp_reso)) * temp_reso +
                             (psi_index + n - 1) % temp_reso];
                    else if ((phi_index + m - 1 < 0 && psi_index + n - 1 >= 0))
                        f[m][n] = temp_grid_value
                            [((phi_index + m + 23) % (temp_reso)) * temp_reso +
                             (psi_index + n - 1) % temp_reso];
                    else if ((phi_index + m - 1 >= 0 && psi_index + n - 1 < 0))
                        f[m][n] = temp_grid_value
                            [((phi_index + m - 1) % (temp_reso)) * temp_reso +
                             (psi_index + n + 23) % temp_reso];
                    else
                        f[m][n] = temp_grid_value
                            [((phi_index + m + 23) % (temp_reso)) * temp_reso +
                             (psi_index + n + 23) % temp_reso];
                }
            }
            // 格点值以及一阶二阶差分
            p[0] = f[1][1];
            p[1] = f[2][1];
            p[2] = f[1][2];
            p[3] = f[2][2];
            p[4] = (f[2][1] - f[0][1]) / 2;
            p[5] = (f[3][1] - f[1][1]) / 2;
            p[6] = (f[2][2] - f[0][2]) / 2;
            p[7] = (f[3][2] - f[1][2]) / 2;
            p[8] = (f[1][2] - f[1][0]) / 2;
            p[9] = (f[2][2] - f[2][0]) / 2;
            p[10] = (f[1][3] - f[1][1]) / 2;
            p[11] = (f[2][3] - f[2][1]) / 2;
            p[12] = (f[2][2] + f[0][0] - f[2][0] - f[0][2]) / 4;
            p[13] = (f[3][2] + f[1][0] - f[3][0] - f[1][2]) / 4;
            p[14] = (f[2][3] + f[0][1] - f[2][1] - f[0][3]) / 4;
            p[15] = (f[3][3] + f[1][1] - f[3][1] - f[1][3]) / 4;

            // 系数矩阵（size:4*4）的对应关系为列指标对应y次数，行指标对应x次数，原始数据（size:reso*reso）行指标对应x坐标，列指标对应y坐标
            for (int q = 0; q < 16; q++)
            {
                // 手动矩阵乘法
                temp_inter_coeff[i * 16 + q] = 0;
                for (int j = 0; j < 16; j++)
                    temp_inter_coeff[i * 16 + q] += (A_inv[q][j]) * p[j];
            }
        }
        temp_grid_value += temp_reso * temp_reso;
        temp_inter_coeff += temp_reso * temp_reso * 16;
    }
    controller->printf("    End Interpolating CMAP Grid Value\n");
}

static __global__
    __launch_bounds__(1024) void CMAP_Force_With_Atom_Energy_And_Virial_Device(
        const int cmap_numbers, const VECTOR* crd, const LTMatrix3 cell,
        const LTMatrix3 rcell, const int* atom_a, const int* atom_b,
        const int* atom_c, const int* atom_d, const int* atom_e,
        const int* cmap_type, const int* resolution, float** inter_coeff_ptr,
        VECTOR* frc, int need_potential, float* ene, float* cmap_ene,
        int need_pressure, LTMatrix3* virial)
{
#ifdef USE_GPU
    int cmap_i = blockDim.x * blockIdx.x + threadIdx.x;
    if (cmap_i < cmap_numbers)
#else
#pragma omp parallel for
    for (int cmap_i = 0; cmap_i < cmap_numbers; cmap_i++)
#endif
    {
        int atom_i = atom_a[cmap_i];
        int atom_j = atom_b[cmap_i];
        int atom_k = atom_c[cmap_i];
        int atom_l = atom_d[cmap_i];
        int atom_m = atom_e[cmap_i];
        VECTOR ri = crd[atom_i];
        VECTOR rj = crd[atom_j];
        VECTOR rk = crd[atom_k];
        VECTOR rl = crd[atom_l];
        VECTOR rm = crd[atom_m];
        // 计算phi
        VECTOR drij = Get_Periodic_Displacement(ri, rj, cell, rcell);
        VECTOR drkj = Get_Periodic_Displacement(rk, rj, cell, rcell);
        VECTOR drkl = Get_Periodic_Displacement(rk, rl, cell, rcell);

        // 法向量夹角
        VECTOR r1_phi = drij ^ drkj;
        VECTOR r2_phi = drkl ^ drkj;

        float r1_1_phi = rnorm3df(r1_phi.x, r1_phi.y, r1_phi.z);
        float r2_1_phi = rnorm3df(r2_phi.x, r2_phi.y, r2_phi.z);
        // float r1_2_phi = r1_1_phi * r1_1_phi;
        // float r2_2_phi = r2_1_phi * r2_1_phi;
        float r1_1_r2_1_phi = r1_1_phi * r2_1_phi;

        float phi = r1_phi * r2_phi * r1_1_r2_1_phi;
        phi = fmaxf(-0.999999, fminf(phi, 0.999999));
        phi = acosf(phi);

        // acosf()只能返回[0,pi],需要确定其正负，最终phi落在[-pi,pi]

        phi = CONSTANT_Pi - phi;

        float sign_phi = (r2_phi ^ r1_phi) * drkj;
        phi = copysignf(phi, sign_phi);

        float cos_phi = cosf(phi);
        float sin_phi = sinf(phi);

        // 计算psi
        VECTOR drjk = Get_Periodic_Displacement(rj, rk, cell, rcell);
        VECTOR drlk = Get_Periodic_Displacement(rl, rk, cell, rcell);
        VECTOR drlm = Get_Periodic_Displacement(rl, rm, cell, rcell);

        // 法向量夹角
        VECTOR r1_psi = drjk ^ drlk;
        VECTOR r2_psi = drlm ^ drlk;

        float r1_1_psi = rnorm3df(r1_psi.x, r1_psi.y, r1_psi.z);
        float r2_1_psi = rnorm3df(r2_psi.x, r2_psi.y, r2_psi.z);
        // float r1_2_psi = r1_1_psi * r1_1_psi;
        // float r2_2_psi = r2_1_psi * r2_1_psi;
        float r1_1_r2_1_psi = r1_1_psi * r2_1_psi;

        float psi = r1_psi * r2_psi * r1_1_r2_1_psi;
        psi = fmaxf(-0.999999, fminf(psi, 0.999999));
        psi = acosf(psi);

        // 同理将psi映射到[-pi,pi]
        psi = CONSTANT_Pi - psi;
        float sign_psi = (r2_psi ^ r1_psi) * drlk;
        psi = copysignf(psi, sign_psi);

        float cos_psi = cosf(psi);
        float sin_psi = sinf(psi);

        // 计算能量
        // 首先将phi,psi
        // 对pi归一化,单位为(pi/resolution),并确定其所属格点以及在格内的位置
        int temp_reso = resolution[cmap_type[cmap_i]];
        phi = phi / (2.0 * CONSTANT_Pi / temp_reso);
        psi = psi / (2.0 * CONSTANT_Pi / temp_reso);

        float parm_phi = phi - floorf(phi);
        float parm_psi = psi - floorf(psi);
        int locate_phi = (int)floorf(phi) + 12;
        int locate_psi = (int)floorf(psi) + 12;

        // 定义幂次
        float parm_phi_2 = parm_phi * parm_phi;
        float parm_phi_3 = parm_phi_2 * parm_phi;
        float parm_psi_2 = parm_psi * parm_psi;
        float parm_psi_3 = parm_psi_2 * parm_psi;

        // 用于定位的中间变量
        float* inter_coeff = inter_coeff_ptr[cmap_i];
        int locate = 16 * (locate_phi * temp_reso + locate_psi);

        // 计算能量对有符号归一化二面角的偏微分
        float dE_dphi =
            (inter_coeff[locate + 4] + parm_psi * inter_coeff[locate + 5] +
             parm_psi_2 * inter_coeff[locate + 6] +
             parm_psi_3 * inter_coeff[locate + 7]) +
            2 * parm_phi *
                (inter_coeff[locate + 8] + parm_psi * inter_coeff[locate + 9] +
                 parm_psi_2 * inter_coeff[locate + 10] +
                 parm_psi_3 * inter_coeff[locate + 11]) +
            3 * parm_phi_2 *
                (inter_coeff[locate + 12] +
                 parm_psi * inter_coeff[locate + 13] +
                 parm_psi_2 * inter_coeff[locate + 14] +
                 parm_psi_3 * inter_coeff[locate + 15]);

        float dE_dpsi =
            inter_coeff[locate + 1] + 2 * parm_psi * inter_coeff[locate + 2] +
            3 * parm_psi_2 * inter_coeff[locate + 3] +
            parm_phi * (inter_coeff[locate + 5] +
                        2 * parm_psi * inter_coeff[locate + 6] +
                        3 * parm_psi_2 * inter_coeff[locate + 7]) +
            parm_phi_2 * (inter_coeff[locate + 9] +
                          2 * parm_psi * inter_coeff[locate + 10] +
                          3 * parm_psi_2 * inter_coeff[locate + 11]) +
            parm_phi_3 * (inter_coeff[locate + 13] +
                          2 * parm_psi * inter_coeff[locate + 14] +
                          3 * parm_psi_2 * inter_coeff[locate + 15]);

        // 将有符号归一化二面角映射回弧度制二面角
        dE_dphi = dE_dphi / (2.0 * CONSTANT_Pi / temp_reso);
        dE_dpsi = dE_dpsi / (2.0 * CONSTANT_Pi / temp_reso);

        // phi角部分
        VECTOR temp_phi_A = drij ^ drjk;
        VECTOR temp_phi_B = drlk ^ drjk;

        VECTOR dphi_dri =
            -sqrtf(drjk * drjk) / (temp_phi_A * temp_phi_A) * temp_phi_A;
        VECTOR dphi_drj =
            +sqrtf(drjk * drjk) / (temp_phi_A * temp_phi_A) * temp_phi_A +
            drij * drjk / (temp_phi_A * temp_phi_A * sqrtf(drjk * drjk)) *
                temp_phi_A -
            drlk * drjk / (temp_phi_B * temp_phi_B * sqrtf(drjk * drjk)) *
                temp_phi_B;
        VECTOR dphi_drk =
            -sqrtf(drjk * drjk) / (temp_phi_B * temp_phi_B) * temp_phi_B -
            drij * drjk / (temp_phi_A * temp_phi_A * sqrtf(drjk * drjk)) *
                temp_phi_A +
            drlk * drjk / (temp_phi_B * temp_phi_B * sqrtf(drjk * drjk)) *
                temp_phi_B;
        VECTOR dphi_drl =
            +sqrtf(drjk * drjk) / (temp_phi_B * temp_phi_B) * temp_phi_B;
        VECTOR dphi_drm = {0, 0, 0};

        // psi角部分
        VECTOR drml = Get_Periodic_Displacement(rm, rl, cell, rcell);

        VECTOR temp_psi_A = drjk ^ drkl;
        VECTOR temp_psi_B = drml ^ drkl;

        VECTOR dpsi_dri = {0, 0, 0};
        VECTOR dpsi_drj =
            -sqrtf(drkl * drkl) / (temp_psi_A * temp_psi_A) * temp_psi_A;
        VECTOR dpsi_drk =
            sqrtf(drkl * drkl) / (temp_psi_A * temp_psi_A) * temp_psi_A +
            drjk * drkl / (temp_psi_A * temp_psi_A * sqrtf(drkl * drkl)) *
                temp_psi_A -
            drml * drkl / (temp_psi_B * temp_psi_B * sqrtf(drkl * drkl)) *
                temp_psi_B;
        VECTOR dpsi_drl =
            -sqrtf(drkl * drkl) / (temp_psi_B * temp_psi_B) * temp_psi_B -
            drjk * drkl / (temp_psi_A * temp_psi_A * sqrtf(drkl * drkl)) *
                temp_psi_A +
            drml * drkl / (temp_psi_B * temp_psi_B * sqrtf(drkl * drkl)) *
                temp_psi_B;
        VECTOR dpsi_drm =
            sqrtf(drkl * drkl) / (temp_psi_B * temp_psi_B) * temp_psi_B;

        // 计算力
        VECTOR fi = -(dE_dphi * dphi_dri + dE_dpsi * dpsi_dri);
        VECTOR fj = -(dE_dphi * dphi_drj + dE_dpsi * dpsi_drj);
        VECTOR fk = -(dE_dphi * dphi_drk + dE_dpsi * dpsi_drk);
        VECTOR fl = -(dE_dphi * dphi_drl + dE_dpsi * dpsi_drl);
        VECTOR fm = -(dE_dphi * dphi_drm + dE_dpsi * dpsi_drm);

        atomicAdd(&frc[atom_i].x, fi.x);
        atomicAdd(&frc[atom_i].y, fi.y);
        atomicAdd(&frc[atom_i].z, fi.z);
        atomicAdd(&frc[atom_j].x, fj.x);
        atomicAdd(&frc[atom_j].y, fj.y);
        atomicAdd(&frc[atom_j].z, fj.z);
        atomicAdd(&frc[atom_k].x, fk.x);
        atomicAdd(&frc[atom_k].y, fk.y);
        atomicAdd(&frc[atom_k].z, fk.z);
        atomicAdd(&frc[atom_l].x, fl.x);
        atomicAdd(&frc[atom_l].y, fl.y);
        atomicAdd(&frc[atom_l].z, fl.z);
        atomicAdd(&frc[atom_m].x, fm.x);
        atomicAdd(&frc[atom_m].y, fm.y);
        atomicAdd(&frc[atom_m].z, fm.z);

        //[1,phi,phi^2,phi^3]multiply inter_coeff(4*4,row priority)multiply
        //[1,psi,psi^2,psi^3]T
        if (need_potential)
        {
            float Energy =
                inter_coeff[locate] + parm_psi * inter_coeff[locate + 1] +
                parm_psi_2 * inter_coeff[locate + 2] +
                parm_psi_3 * inter_coeff[locate + 3] +
                parm_phi * (inter_coeff[locate + 4] +
                            parm_psi * inter_coeff[locate + 5] +
                            parm_psi_2 * inter_coeff[locate + 6] +
                            parm_psi_3 * inter_coeff[locate + 7]) +
                parm_phi_2 * (inter_coeff[locate + 8] +
                              parm_psi * inter_coeff[locate + 9] +
                              parm_psi_2 * inter_coeff[locate + 10] +
                              parm_psi_3 * inter_coeff[locate + 11]) +
                parm_phi_3 * (inter_coeff[locate + 12] +
                              parm_psi * inter_coeff[locate + 13] +
                              parm_psi_2 * inter_coeff[locate + 14] +
                              parm_psi_3 * inter_coeff[locate + 15]);
            atomicAdd(&ene[atom_i], Energy);
            cmap_ene[cmap_i] = Energy;
        }
        if (need_pressure)
        {
            atomicAdd(virial + atom_i, Get_Virial_From_Force_Dis(fi, ri) +
                                           Get_Virial_From_Force_Dis(fj, rj) +
                                           Get_Virial_From_Force_Dis(fk, rk) +
                                           Get_Virial_From_Force_Dis(fl, rl) +
                                           Get_Virial_From_Force_Dis(fm, rm));
        }
    }
}

void CMAP::CMAP_Force_With_Atom_Energy_And_Virial(
    const VECTOR* crd, const LTMatrix3 cell, const LTMatrix3 rcell, VECTOR* frc,
    int need_potential, float* atom_energy, int need_pressure,
    LTMatrix3* atom_virial)
{
    if (is_initialized)
    {
        Launch_Device_Kernel(
            CMAP_Force_With_Atom_Energy_And_Virial_Device,
            (tot_cmap_num + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, this->num_cmap_local, crd,
            cell, rcell, this->d_atom_a_local, this->d_atom_b_local,
            this->d_atom_c_local, this->d_atom_d_local, this->d_atom_e_local,
            this->d_cmap_type_local, this->d_cmap_resolution, this->d_coeff_ptr,
            frc, need_potential, atom_energy, d_cmap_ene, need_pressure,
            atom_virial);
    }
}

void CMAP::Step_Print(CONTROLLER* controller, bool print_sum)
{
    if (is_initialized && CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size)
    {
        Sum_Of_List(d_cmap_ene, d_sigma_of_cmap_ene, num_cmap_local);
        deviceMemcpy(&h_sigma_of_cmap_ene, d_sigma_of_cmap_ene, sizeof(float),
                     deviceMemcpyDeviceToHost);
#ifdef USE_MPI
        MPI_Allreduce(MPI_IN_PLACE, &h_sigma_of_cmap_ene, 1, MPI_FLOAT, MPI_SUM,
                      CONTROLLER::pp_comm);
#endif
        controller->Step_Print(this->module_name, h_sigma_of_cmap_ene,
                               print_sum);
    }
}

static __global__ void get_local_device(
    int tot_cmap_num, const int* d_atom_a, const int* d_atom_b,
    const int* d_atom_c, const int* d_atom_d, const int* d_atom_e,
    const int* d_cmap_type, const char* atom_local_label,
    const int* atom_local_id, int* d_atom_a_local, int* d_atom_b_local,
    int* d_atom_c_local, int* d_atom_d_local, int* d_atom_e_local,
    int* d_cmap_type_local, int* d_num_cmap_local)
{
#ifdef USE_GPU
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx != 0) return;
#endif
    d_num_cmap_local[0] = 0;
    for (int i = 0; i < tot_cmap_num; i++)
    {
        if (atom_local_label[d_atom_a[i]] == 1 ||
            atom_local_label[d_atom_b[i]] == 1 ||
            atom_local_label[d_atom_c[i]] == 1 ||
            atom_local_label[d_atom_d[i]] == 1 ||
            atom_local_label[d_atom_e[i]] == 1)
        {
            d_atom_a_local[d_num_cmap_local[0]] = atom_local_id[d_atom_a[i]];
            d_atom_b_local[d_num_cmap_local[0]] = atom_local_id[d_atom_b[i]];
            d_atom_c_local[d_num_cmap_local[0]] = atom_local_id[d_atom_c[i]];
            d_atom_d_local[d_num_cmap_local[0]] = atom_local_id[d_atom_d[i]];
            d_atom_e_local[d_num_cmap_local[0]] = atom_local_id[d_atom_e[i]];
            d_cmap_type_local[d_num_cmap_local[0]] = d_cmap_type[i];
            d_num_cmap_local[0]++;
        }
    }
}

void CMAP::Get_Local(int* atom_local, int local_atom_numbers, int ghost_numbers,
                     char* atom_local_label, int* atom_local_id)
{
    if (!is_initialized) return;
    num_cmap_local = 0;
    this->local_atom_numbers = local_atom_numbers;
    Launch_Device_Kernel(get_local_device, 1, 1, 0, NULL, this->tot_cmap_num,
                         this->d_atom_a, this->d_atom_b, this->d_atom_c,
                         this->d_atom_d, this->d_atom_e, this->d_cmap_type,
                         atom_local_label, atom_local_id, this->d_atom_a_local,
                         this->d_atom_b_local, this->d_atom_c_local,
                         this->d_atom_d_local, this->d_atom_e_local,
                         this->d_cmap_type_local, this->d_num_cmap_local);
    deviceMemcpy(&num_cmap_local, d_num_cmap_local, sizeof(int),
                 deviceMemcpyDeviceToHost);
}

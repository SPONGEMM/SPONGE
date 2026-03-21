#include "quantum_chemistry.h"

#include "internal/qc_compile_config.hpp"
#include "scf/matrix.hpp"

// 用于将归一化笛卡尔基映射到归一化实球谐基
// l=2（d 轨道）
static const float CART2SPH_MAT_D[6][5] = {
    {0.00000000f, 0.00000000f, -0.31539157f, 0.00000000f, 0.54627422f},
    {1.09254843f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f},
    {0.00000000f, 0.00000000f, 0.00000000f, 1.09254843f, 0.00000000f},
    {0.00000000f, 0.00000000f, -0.31539157f, 0.00000000f, -0.54627422f},
    {0.00000000f, 1.09254843f, 0.00000000f, 0.00000000f, 0.00000000f},
    {0.00000000f, 0.00000000f, 0.63078313f, 0.00000000f, 0.00000000f},
};

// l=3（f 轨道）
static const float CART2SPH_MAT_F[10][7] = {
    {0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, -0.45704580f,
     0.00000000f, 0.59004359f},
    {1.77013077f, 0.00000000f, -0.45704580f, 0.00000000f, 0.00000000f,
     0.00000000f, 0.00000000f},
    {0.00000000f, 0.00000000f, 0.00000000f, -1.11952900f, 0.00000000f,
     1.44530572f, 0.00000000f},
    {0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, -0.45704580f,
     0.00000000f, -1.77013077f},
    {0.00000000f, 2.89061144f, 0.00000000f, 0.00000000f, 0.00000000f,
     0.00000000f, 0.00000000f},
    {0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 1.82818320f,
     0.00000000f, 0.00000000f},
    {-0.59004359f, 0.00000000f, -0.45704580f, 0.00000000f, 0.00000000f,
     0.00000000f, 0.00000000f},
    {0.00000000f, 0.00000000f, 0.00000000f, -1.11952900f, 0.00000000f,
     -1.44530572f, 0.00000000f},
    {0.00000000f, 0.00000000f, 1.82818320f, 0.00000000f, 0.00000000f,
     0.00000000f, 0.00000000f},
    {0.00000000f, 0.00000000f, 0.00000000f, 0.74635267f, 0.00000000f,
     0.00000000f, 0.00000000f},
};

// l=4（g 轨道）
static const float CART2SPH_MAT_G[15][9] = {
    {0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.31735664f,
     0.00000000f, -0.47308735f, 0.00000000f, 0.62583574f},
    {2.50334294f, 0.00000000f, -0.94617470f, 0.00000000f, 0.00000000f,
     0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f},
    {0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f,
     -2.00713963f, 0.00000000f, 1.77013077f, 0.00000000f},
    {0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.63471328f,
     0.00000000f, 0.00000000f, 0.00000000f, -3.75501441f},
    {0.00000000f, 5.31039231f, 0.00000000f, -2.00713963f, 0.00000000f,
     0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f},
    {0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, -2.53885313f,
     0.00000000f, 2.83852409f, 0.00000000f, 0.00000000f},
    {-2.50334294f, 0.00000000f, -0.94617470f, 0.00000000f, 0.00000000f,
     0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f},
    {0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f,
     -2.00713963f, 0.00000000f, -5.31039231f, 0.00000000f},
    {0.00000000f, 0.00000000f, 5.67704817f, 0.00000000f, 0.00000000f,
     0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f},
    {0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f,
     2.67618617f, 0.00000000f, 0.00000000f, 0.00000000f},
    {0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.31735664f,
     0.00000000f, 0.47308735f, 0.00000000f, 0.62583574f},
    {0.00000000f, -1.77013077f, 0.00000000f, -2.00713963f, 0.00000000f,
     0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f},
    {0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, -2.53885313f,
     0.00000000f, -2.83852409f, 0.00000000f, 0.00000000f},
    {0.00000000f, 0.00000000f, 0.00000000f, 2.67618617f, 0.00000000f,
     0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f},
    {0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.84628438f,
     0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f},
};

static __global__ void QC_Cart2Sph_MatMul_UT_RowRow_Kernel(
    const int m, const int n, const int kdim, const float* U_row_k_m,
    const float* B_row_k_n, float* C_row_m_n)
{
    SIMPLE_DEVICE_FOR(idx, m * n)
    {
        int i = idx / n;
        int j = idx - i * n;
        double sum = 0.0;
        for (int k = 0; k < kdim; k++)
        {
            sum += (double)U_row_k_m[k * m + i] * (double)B_row_k_n[k * n + j];
        }
        C_row_m_n[i * n + j] = (float)sum;
    }
}

void QUANTUM_CHEMISTRY::Build_Cart2Sph_Matrix()
{
    int nao_c = mol.nao_cart;
    int nao_s = mol.nao_sph;
    std::vector<float> cart2sph_mat(nao_c * nao_s, 0.0f);

    int offset_c = 0;
    int offset_s = 0;

    for (int k = 0; k < mol.h_l_list.size(); k++)
    {
        int l = mol.h_l_list[k];
        int dim_c = (l + 1) * (l + 2) / 2;
        int dim_s = 2 * l + 1;

        switch (l)
        {
            case 0:  // s 轨道
                cart2sph_mat[offset_c * nao_s + offset_s] = 0.28209479f;
                break;
            case 1:  // p 轨道
                cart2sph_mat[(offset_c + 0) * nao_s + (offset_s + 0)] =
                    0.48860251f;
                cart2sph_mat[(offset_c + 1) * nao_s + (offset_s + 1)] =
                    0.48860251f;
                cart2sph_mat[(offset_c + 2) * nao_s + (offset_s + 2)] =
                    0.48860251f;
                break;
            case 2:  // d 轨道
                for (int i = 0; i < 6; i++)
                    for (int j = 0; j < 5; j++)
                        cart2sph_mat[(offset_c + i) * nao_s + (offset_s + j)] =
                            CART2SPH_MAT_D[i][j];
                break;
            case 3:  // f 轨道
                for (int i = 0; i < 10; i++)
                    for (int j = 0; j < 7; j++)
                        cart2sph_mat[(offset_c + i) * nao_s + (offset_s + j)] =
                            CART2SPH_MAT_F[i][j];
                break;
            case 4:  // g 轨道
                for (int i = 0; i < 15; i++)
                    for (int j = 0; j < 9; j++)
                        cart2sph_mat[(offset_c + i) * nao_s + (offset_s + j)] =
                            CART2SPH_MAT_G[i][j];
                break;
            default:
                printf("Error: l=%d not supported in Cart2Sph transform\n", l);
                exit(1);
        }
        offset_c += dim_c;
        offset_s += dim_s;
    }
    if (cart2sph_mat.empty())
    {
        cart2sph.d_cart2sph_mat = nullptr;
    }
    else
    {
        Device_Malloc_Safely((void**)&cart2sph.d_cart2sph_mat,
                             sizeof(float) * cart2sph_mat.size());
        deviceMemcpy(cart2sph.d_cart2sph_mat, cart2sph_mat.data(),
                     sizeof(float) * cart2sph_mat.size(),
                     deviceMemcpyHostToDevice);
    }
}

void QUANTUM_CHEMISTRY::Cart2Sph_OneE_Integrals()
{
    if (!mol.is_spherical) return;
    const int nao_c = mol.nao_cart;
    const int nao_s = mol.nao_sph;
    const int threads = 256;
    const int total = nao_s * nao_s;
    auto cart2sph_1e = [&](float* d_src, float* d_dst)
    {
        QC_MatMul_RowRow_Blas(blas_handle, nao_c, nao_s, nao_c, d_src,
                              cart2sph.d_cart2sph_mat,
                              cart2sph.d_cart2sph_1e_tmp);
        Launch_Device_Kernel(QC_Cart2Sph_MatMul_UT_RowRow_Kernel,
                             (total + threads - 1) / threads, threads, 0, 0,
                             nao_s, nao_s, nao_c, cart2sph.d_cart2sph_mat,
                             cart2sph.d_cart2sph_1e_tmp, d_dst);
    };
    cart2sph_1e(cart2sph.d_S_cart, scf_ws.d_S);
    cart2sph_1e(cart2sph.d_T_cart, scf_ws.d_T);
    cart2sph_1e(cart2sph.d_V_cart, scf_ws.d_V);
}

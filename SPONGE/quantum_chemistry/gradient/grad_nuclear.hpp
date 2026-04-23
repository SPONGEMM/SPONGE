#pragma once

// 核排斥梯度
// dE_nuc/dR_Ax = -Z_A × Σ_{B≠A} Z_B × (R_Ax - R_Bx) / |R_AB|³

static __global__ void QC_Nuclear_Gradient_Kernel(
    const int natm, const int* z_nuc, const int* atm, const float* env,
    const VECTOR box_length, double* grad)
{
    SIMPLE_DEVICE_FOR(i, natm)
    {
        const int ptr_i = atm[i * 6 + 1];
        const double zi = (double)z_nuc[i];
        const VECTOR ri(env[ptr_i + 0], env[ptr_i + 1], env[ptr_i + 2]);

        double gx = 0.0, gy = 0.0, gz = 0.0;

        for (int j = 0; j < natm; j++)
        {
            if (j == i) continue;
            const int ptr_j = atm[j * 6 + 1];
            const double zj = (double)z_nuc[j];
            const VECTOR rj(env[ptr_j + 0], env[ptr_j + 1], env[ptr_j + 2]);
            const VECTOR dr = Get_Periodic_Displacement(ri, rj, box_length);
            const double r2 =
                (double)dr.x * dr.x + (double)dr.y * dr.y + (double)dr.z * dr.z;
            const double r = sqrt(r2);
            const double r3_inv = 1.0 / fmax(r * r2, 1e-30);

            gx += zj * (double)dr.x * r3_inv;
            gy += zj * (double)dr.y * r3_inv;
            gz += zj * (double)dr.z * r3_inv;
        }

        atomicAdd(&grad[i * 3 + 0], -zi * gx);
        atomicAdd(&grad[i * 3 + 1], -zi * gy);
        atomicAdd(&grad[i * 3 + 2], -zi * gz);
    }
}

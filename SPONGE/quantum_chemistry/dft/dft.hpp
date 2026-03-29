#pragma once

#include "../integrals/one_e.hpp"

static inline __host__ __device__ void QC_Local_Vrho_Vsigma_FD(
    QC_METHOD method, double rho, double sigma, double& e, double& vrho,
    double& vsigma);
static inline __host__ __device__ void QC_Local_UKS_Derivs_FD(
    QC_METHOD method, double rho_a, double rho_b, double sigma_aa,
    double sigma_ab, double sigma_bb, double& e, double& v_rho_a,
    double& v_rho_b, double& v_sigma_aa, double& v_sigma_ab,
    double& v_sigma_bb);

// =============================================================================
// Device DFT Kernels
// deriv_level: 0 = LDA (仅值), 1 = GGA (值+梯度), 2 = meta-GGA (预留)
// =============================================================================

// AO 求值: 在网格点上计算基函数值和（可选的）梯度
template <int deriv_level>
static __global__ void QC_Eval_AO_Grid_Kernel(
    const int n_grid_batch, const float* grid_coords, const int nao,
    const int nbas, const VECTOR* centers, const int* l_list, const float* exps,
    const float* coeffs, const int* shell_offsets, const int* shell_sizes,
    const int* ao_offsets, const float* shell_r2_screen, float* ao_vals,
    float* ao_grad_x, float* ao_grad_y, float* ao_grad_z)
{
    SIMPLE_DEVICE_FOR(ig, n_grid_batch)
    {
        const float x = grid_coords[ig * 3 + 0];
        const float y = grid_coords[ig * 3 + 1];
        const float z = grid_coords[ig * 3 + 2];

        for (int i = 0; i < nao; i++)
        {
            ao_vals[ig * nao + i] = 0.0f;
            if (deriv_level >= 1)
            {
                ao_grad_x[ig * nao + i] = 0.0f;
                ao_grad_y[ig * nao + i] = 0.0f;
                ao_grad_z[ig * nao + i] = 0.0f;
            }
        }

        for (int ish = 0; ish < nbas; ish++)
        {
            const VECTOR c = centers[ish];
            const float dx = x - c.x;
            const float dy = y - c.y;
            const float dz = z - c.z;
            const float r2 = dx * dx + dy * dy + dz * dz;
            if (r2 > shell_r2_screen[ish]) continue;

            const int l = l_list[ish];
            const int ncart = (l + 1) * (l + 2) / 2;
            const int ao_off = ao_offsets[ish];

            float px[6], py[6], pz[6];
            px[0] = py[0] = pz[0] = 1.0f;
            for (int k = 1; k <= 5; k++)
            {
                px[k] = px[k - 1] * dx;
                py[k] = py[k - 1] * dy;
                pz[k] = pz[k - 1] * dz;
            }

            for (int ip = 0; ip < shell_sizes[ish]; ip++)
            {
                const int pidx = shell_offsets[ish] + ip;
                const float alpha = exps[pidx];
                const float e = coeffs[pidx] * expf(-alpha * r2);
                if (fabsf(e) < 1e-20f) continue;

                for (int ic = 0; ic < ncart; ic++)
                {
                    int lx, ly, lz;
                    QC_Get_Lxyz_Device(l, ic, lx, ly, lz);
                    const float poly = px[lx] * py[ly] * pz[lz];
                    const int i = ao_off + ic;
                    ao_vals[ig * nao + i] += e * poly;

                    if (deriv_level >= 1)
                    {
                        const float dpx =
                            (lx > 0 ? (float)lx * px[lx - 1] : 0.0f) * py[ly] *
                            pz[lz];
                        const float dpy =
                            px[lx] * (ly > 0 ? (float)ly * py[ly - 1] : 0.0f) *
                            pz[lz];
                        const float dpz =
                            px[lx] * py[ly] *
                            (lz > 0 ? (float)lz * pz[lz - 1] : 0.0f);
                        ao_grad_x[ig * nao + i] +=
                            e * (dpx - 2.0f * alpha * dx * poly);
                        ao_grad_y[ig * nao + i] +=
                            e * (dpy - 2.0f * alpha * dy * poly);
                        ao_grad_z[ig * nao + i] +=
                            e * (dpz - 2.0f * alpha * dz * poly);
                    }
                }
            }
        }
    }
}

// Kernel 2: Compute electron density ρ and gradient σ = |∇ρ|²
static __global__ void QC_Eval_Rho_Sigma_Kernel(
    const int n_grid, const int nao,
    const float* ao_vals,  // [n_grid * nao]
    const float* ao_grad_x, const float* ao_grad_y, const float* ao_grad_z,
    const float* P,      // [nao * nao] density matrix
    const float* norms,  // [nao] normalization factors
    double* rho,         // [n_grid] OUTPUT
    double* sigma        // [n_grid] OUTPUT
)
{
    SIMPLE_DEVICE_FOR(ig, n_grid)
    {
        // Apply normalization to AO values
        double rho_val = 0.0;
        double grad_x = 0.0, grad_y = 0.0, grad_z = 0.0;

        for (int i = 0; i < nao; i++)
        {
            const float ni = norms[i];
            const double ao_i = (double)(ao_vals[ig * nao + i] * ni);
            const double gx_i = (double)(ao_grad_x[ig * nao + i] * ni);
            const double gy_i = (double)(ao_grad_y[ig * nao + i] * ni);
            const double gz_i = (double)(ao_grad_z[ig * nao + i] * ni);

            // Compute density contribution: ρ = Σ_ij P_ij φ_i φ_j
            double tmp = 0.0;
            for (int j = 0; j < nao; j++)
            {
                const float nj = norms[j];
                tmp += (double)P[i * nao + j] *
                       (double)(ao_vals[ig * nao + j] * nj);
            }
            rho_val += ao_i * tmp;

            // Gradient contribution: ∇ρ = 2 Σ_ij P_ij ∇φ_i φ_j
            grad_x += gx_i * tmp;
            grad_y += gy_i * tmp;
            grad_z += gz_i * tmp;
        }

        grad_x *= 2.0;
        grad_y *= 2.0;
        grad_z *= 2.0;

        rho[ig] = rho_val;
        sigma[ig] = grad_x * grad_x + grad_y * grad_y + grad_z * grad_z;
    }
}

// Kernel 3: Evaluate local XC energy density and derivatives.
static __global__ void QC_Eval_XC_Derivs_Kernel(
    const int n_grid, const int method_id, const double* rho,
    const double* sigma, double* exc, double* vrho, double* vsigma)
{
    SIMPLE_DEVICE_FOR(ig, n_grid)
    {
        const double rho_val = rho[ig];
        if (rho_val < 1e-10)
        {
            exc[ig] = 0.0;
            vrho[ig] = 0.0;
            vsigma[ig] = 0.0;
        }
        else
        {
            double e = 0.0, v_rho = 0.0, v_sigma = 0.0;
            QC_Local_Vrho_Vsigma_FD((QC_METHOD)method_id, rho_val, sigma[ig], e,
                                    v_rho, v_sigma);
            exc[ig] = e;
            vrho[ig] = v_rho;
            vsigma[ig] = v_sigma;
        }
    }
}

// Kernel 4: Build Vxc matrix from functional derivatives
// Uses atomicAdd for accumulation since multiple grid points contribute to same
// matrix element.
static __global__ void QC_Build_Vxc_Kernel(
    const int n_grid, const int nao, const float* ao_vals,
    const float* ao_grad_x, const float* ao_grad_y, const float* ao_grad_z,
    const float* grid_weights, const float* P, const double* rho,
    const double* exc,               // ε_xc
    const double* vrho,              // ∂E/∂ρ
    const double* vsigma,            // ∂E/∂σ
    const float* norms, float* Vxc,  // [nao * nao] OUTPUT (accumulated)
    double* exc_total)
{
    SIMPLE_DEVICE_FOR(ig, n_grid)
    {
        if (rho[ig] >= 1e-10)  // Skip near-zero density
        {
            const float w = grid_weights[ig];
            const double v_rho = vrho[ig];
            const double v_sigma = vsigma[ig];
            atomicAdd(exc_total, (double)w * exc[ig]);

            // Precompute gradient of rho at this grid point
            double grad_rho_x = 0.0, grad_rho_y = 0.0, grad_rho_z = 0.0;
            for (int i = 0; i < nao; i++)
            {
                const float ni = norms[i];
                const double gix = (double)(ao_grad_x[ig * nao + i] * ni);
                const double giy = (double)(ao_grad_y[ig * nao + i] * ni);
                const double giz = (double)(ao_grad_z[ig * nao + i] * ni);
                double tmp_i = 0.0;
                for (int j = 0; j < nao; j++)
                {
                    const float nj = norms[j];
                    tmp_i += (double)P[i * nao + j] *
                             (double)(ao_vals[ig * nao + j] * nj);
                }
                grad_rho_x += gix * tmp_i;
                grad_rho_y += giy * tmp_i;
                grad_rho_z += giz * tmp_i;
            }
            grad_rho_x *= 2.0;
            grad_rho_y *= 2.0;
            grad_rho_z *= 2.0;

            // Build Vxc matrix elements
            for (int i = 0; i < nao; i++)
            {
                const float ni = norms[i];
                const double ai = (double)(ao_vals[ig * nao + i] * ni);
                const double gix = (double)(ao_grad_x[ig * nao + i] * ni);
                const double giy = (double)(ao_grad_y[ig * nao + i] * ni);
                const double giz = (double)(ao_grad_z[ig * nao + i] * ni);

                for (int j = 0; j <= i; j++)
                {
                    const float nj = norms[j];
                    const double aj = (double)(ao_vals[ig * nao + j] * nj);
                    const double gjx = (double)(ao_grad_x[ig * nao + j] * nj);
                    const double gjy = (double)(ao_grad_y[ig * nao + j] * nj);
                    const double gjz = (double)(ao_grad_z[ig * nao + j] * nj);

                    // V_xc contribution
                    const double term_rho = v_rho * ai * aj;
                    const double term_sigma =
                        2.0 * v_sigma *
                        (grad_rho_x * (ai * gjx + gix * aj) +
                         grad_rho_y * (ai * gjy + giy * aj) +
                         grad_rho_z * (ai * gjz + giz * aj));

                    const float vij =
                        (float)((double)w * (term_rho + term_sigma));

                    // Atomic accumulation (symmetric matrix)
                    atomicAdd(&Vxc[i * nao + j], vij);
                    if (i != j) atomicAdd(&Vxc[j * nao + i], vij);
                }
            }
        }
    }
}

// UKS batched kernel: compute rho/spin-gradient invariants, local XC
// derivatives, XC energy accumulation, and Vxc(alpha/beta) matrix assembly.
static __global__ void QC_Build_Vxc_UKS_Kernel(
    const int n_grid, const int nao, const int method_id, const float* ao_vals,
    const float* ao_grad_x, const float* ao_grad_y, const float* ao_grad_z,
    const float* grid_weights, const float* P_a, const float* P_b,
    const float* norms, float* Vxc_a, float* Vxc_b, double* exc_total)
{
    SIMPLE_DEVICE_FOR(ig, n_grid)
    {
        double rho_a = 0.0, rho_b = 0.0;
        double gra_x = 0.0, gra_y = 0.0, gra_z = 0.0;
        double grb_x = 0.0, grb_y = 0.0, grb_z = 0.0;

        for (int i = 0; i < nao; i++)
        {
            const float ni = norms[i];
            const double ai = (double)(ao_vals[ig * nao + i] * ni);
            const double gix = (double)(ao_grad_x[ig * nao + i] * ni);
            const double giy = (double)(ao_grad_y[ig * nao + i] * ni);
            const double giz = (double)(ao_grad_z[ig * nao + i] * ni);

            double tmp_a = 0.0, tmp_b = 0.0;
            for (int j = 0; j < nao; j++)
            {
                const double aoj = (double)(ao_vals[ig * nao + j] * norms[j]);
                tmp_a += (double)P_a[i * nao + j] * aoj;
                tmp_b += (double)P_b[i * nao + j] * aoj;
            }

            rho_a += ai * tmp_a;
            rho_b += ai * tmp_b;
            gra_x += gix * tmp_a;
            gra_y += giy * tmp_a;
            gra_z += giz * tmp_a;
            grb_x += gix * tmp_b;
            grb_y += giy * tmp_b;
            grb_z += giz * tmp_b;
        }
        gra_x *= 2.0;
        gra_y *= 2.0;
        gra_z *= 2.0;
        grb_x *= 2.0;
        grb_y *= 2.0;
        grb_z *= 2.0;

        if (rho_a + rho_b >= 1e-10)
        {
            const double sigma_aa =
                gra_x * gra_x + gra_y * gra_y + gra_z * gra_z;
            const double sigma_ab =
                gra_x * grb_x + gra_y * grb_y + gra_z * grb_z;
            const double sigma_bb =
                grb_x * grb_x + grb_y * grb_y + grb_z * grb_z;

            double e_loc = 0.0;
            double v_rho_a = 0.0, v_rho_b = 0.0;
            double v_sigma_aa = 0.0, v_sigma_ab = 0.0, v_sigma_bb = 0.0;
            QC_Local_UKS_Derivs_FD((QC_METHOD)method_id, rho_a, rho_b, sigma_aa,
                                   sigma_ab, sigma_bb, e_loc, v_rho_a, v_rho_b,
                                   v_sigma_aa, v_sigma_ab, v_sigma_bb);
            atomicAdd(exc_total, (double)grid_weights[ig] * e_loc);

            const double gax = 2.0 * v_sigma_aa * gra_x + v_sigma_ab * grb_x;
            const double gay = 2.0 * v_sigma_aa * gra_y + v_sigma_ab * grb_y;
            const double gaz = 2.0 * v_sigma_aa * gra_z + v_sigma_ab * grb_z;
            const double gbx = 2.0 * v_sigma_bb * grb_x + v_sigma_ab * gra_x;
            const double gby = 2.0 * v_sigma_bb * grb_y + v_sigma_ab * gra_y;
            const double gbz = 2.0 * v_sigma_bb * grb_z + v_sigma_ab * gra_z;

            const float w = grid_weights[ig];
            for (int i = 0; i < nao; i++)
            {
                const float ni = norms[i];
                const double ai = (double)(ao_vals[ig * nao + i] * ni);
                const double gix = (double)(ao_grad_x[ig * nao + i] * ni);
                const double giy = (double)(ao_grad_y[ig * nao + i] * ni);
                const double giz = (double)(ao_grad_z[ig * nao + i] * ni);
                for (int j = 0; j <= i; j++)
                {
                    const float nj = norms[j];
                    const double aj = (double)(ao_vals[ig * nao + j] * nj);
                    const double gjx = (double)(ao_grad_x[ig * nao + j] * nj);
                    const double gjy = (double)(ao_grad_y[ig * nao + j] * nj);
                    const double gjz = (double)(ao_grad_z[ig * nao + j] * nj);

                    const double dpx = ai * gjx + gix * aj;
                    const double dpy = ai * gjy + giy * aj;
                    const double dpz = ai * gjz + giz * aj;

                    const float vaij =
                        (float)((double)w * (v_rho_a * ai * aj + gax * dpx +
                                             gay * dpy + gaz * dpz));
                    const float vbij =
                        (float)((double)w * (v_rho_b * ai * aj + gbx * dpx +
                                             gby * dpy + gbz * dpz));

                    atomicAdd(&Vxc_a[i * nao + j], vaij);
                    atomicAdd(&Vxc_b[i * nao + j], vbij);
                    if (i != j)
                    {
                        atomicAdd(&Vxc_a[j * nao + i], vaij);
                        atomicAdd(&Vxc_b[j * nao + i], vbij);
                    }
                }
            }
        }
    }
}

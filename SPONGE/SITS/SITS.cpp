#include "SITS.h"

#include "../utils/h5md/input_assembler.hpp"
#include "sits_h5_input.hpp"

template <bool need_force, bool need_energy, bool need_virial,
          bool need_coulomb>
static __global__ void Selective_Lennard_Jones_And_Direct_Coulomb_Device(
    const int local_atom_numbers, const int solvent_numbers,
    const ATOM_GROUP* nl, float* atom_ene_LJ, const VECTOR_LJ* crd,
    const Boundary boundary, const float* LJ_type_A, const float* LJ_type_B,
    const int* atom_sys_mark, const float cutoff, VECTOR* frc,
    VECTOR* frc_enhancing, const float pme_beta, float* atom_energy,
    float* atom_energy_enhancing, LTMatrix3* atom_virial,
    LTMatrix3* atom_virial_enhancing, float* atom_direct_cf_energy,
    const float pwwp_factor)
{
#ifdef USE_GPU
    int atom_i = blockDim.y * blockIdx.x + threadIdx.y;
    if (atom_i < local_atom_numbers - solvent_numbers)
#else
#pragma omp parallel for
    for (int atom_i = 0; atom_i < local_atom_numbers - solvent_numbers;
         atom_i++)
#endif
    {
        ATOM_GROUP nl_i = nl[atom_i];
        VECTOR_LJ r1 = crd[atom_i];
        int atom_mark_i = atom_sys_mark[atom_i];
        VECTOR frc_record = {0.0f, 0.0f, 0.0f},
               frc_enhancing_record = {0.0f, 0.0f, 0.0f};
        LTMatrix3 virial_record = {0, 0, 0, 0, 0, 0},
                  virial_enhancing = {0, 0, 0, 0, 0, 0};
        float energy_lj = 0.0f, energy_enhancing = 0.0f, energy_coulomb = 0.0f;
#ifdef USE_GPU
        for (int j = threadIdx.x; j < nl_i.atom_numbers; j += blockDim.x)
#else
        for (int j = 0; j < nl_i.atom_numbers; j++)
#endif
        {
            int atom_j = nl_i.atom_serial[j];
            float ij_factor = atom_j < local_atom_numbers ? 1.0f : 0.5f;
            VECTOR_LJ r2 = crd[atom_j];
            VECTOR dr =
                Get_Displacement<BoundaryPolicy::Periodic>(r2, r1, boundary);
            float dr_abs = norm3df(dr.x, dr.y, dr.z);
            if (dr_abs < cutoff)
            {
                int atom_mark_j = atom_sys_mark[atom_j] + atom_mark_i;
                int atom_pair_LJ_type = Get_LJ_Type(r1.LJ_type, r2.LJ_type);
                float A = LJ_type_A[atom_pair_LJ_type];
                float B = LJ_type_B[atom_pair_LJ_type];
                float factor = 0;
                if (atom_mark_j == 0)
                {
                    factor = 1;
                }
                else if (atom_mark_j == 1)
                {
                    factor = pwwp_factor;
                }
                if (need_force)
                {
                    float frc_abs = Get_LJ_Force(r1, r2, dr_abs, A, B);
                    if (need_coulomb)
                    {
                        float frc_cf_abs =
                            Get_Direct_Coulomb_Force(r1, r2, dr_abs, pme_beta);
                        frc_abs = frc_abs - frc_cf_abs;
                    }
                    VECTOR frc_lin = frc_abs * dr;
                    frc_record = frc_record + frc_lin;
                    if (atom_j < local_atom_numbers)
                        atomicAdd(frc + atom_j, -frc_lin);
                    frc_lin = factor * frc_lin;
                    frc_enhancing_record = frc_enhancing_record + frc_lin;
                    if (need_virial)
                    {
                        LTMatrix3 virial0 =
                            Get_Virial_From_Force_Dis(frc_lin, dr);
                        virial_record = virial_record + ij_factor * virial0;
                        virial_enhancing =
                            virial_enhancing + ij_factor * factor * virial0;
                    }
                }
                if (need_coulomb && need_energy)
                {
                    float energy_lin =
                        Get_Direct_Coulomb_Energy(r1, r2, dr_abs, pme_beta);
                    energy_coulomb += ij_factor * energy_lin;
                    energy_enhancing += ij_factor * factor * energy_lin;
                }
                if (need_energy)
                {
                    float energy_lin = Get_LJ_Energy(r1, r2, dr_abs, A, B);
                    energy_lj += ij_factor * energy_lin;
                    energy_enhancing += ij_factor * factor * energy_lin;
                }
            }
        }
        if (need_force)
        {
            Warp_Sum_To(frc + atom_i, frc_record, warpSize);
            Warp_Sum_To(frc_enhancing + atom_i, frc_enhancing_record, warpSize);
        }
        if (need_coulomb && need_energy)
        {
            Warp_Sum_To(atom_direct_cf_energy + atom_i, energy_coulomb,
                        warpSize);
        }
        if (need_energy)
        {
            Warp_Sum_To(atom_energy + atom_i, energy_lj, warpSize);
#ifdef USE_GPU
            if (threadIdx.x == 0)
#endif
                atomicAdd(atom_ene_LJ + atom_i, energy_lj);
            Warp_Sum_To(atom_energy_enhancing + atom_i, energy_enhancing,
                        warpSize);
        }
        if (need_virial)
        {
            Warp_Sum_To(atom_virial + atom_i, virial_record, warpSize);
            Warp_Sum_To(atom_virial_enhancing + atom_i, virial_enhancing,
                        warpSize);
        }
    }
}

template <bool need_force, bool need_energy, bool need_virial,
          bool need_coulomb, bool need_du_dlambda>
static __global__ void
Selective_Lennard_Jones_And_Direct_Coulomb_Soft_Core_Device(
    const int local_atom_numbers, const int solvent_numbers,
    const ATOM_GROUP* nl, float* atom_ene_LJ, const VECTOR_LJ_SOFT_TYPE* crd,
    const Boundary boundary, const int* atom_sys_mark, const float* LJ_type_AA,
    const float* LJ_type_AB, const float* LJ_type_BA, const float* LJ_type_BB,
    const float cutoff, VECTOR* frc, VECTOR* frc_enhancing,
    const float pme_beta, float* atom_energy, float* atom_energy_enhancing,
    LTMatrix3* atom_virial, LTMatrix3* atom_virial_enhancing,
    float* atom_direct_cf_energy, float* atom_du_dlambda_lj,
    float* atom_du_dlambda_direct, float* atom_du_dlambda_enhancing,
    const float lambda, const float alpha, const float p,
    const float input_sigma_6, const float input_sigma_6_min,
    const float pwwp_factor)
{
    float lambda_ = 1.0 - lambda;
    float alpha_lambda_p = alpha * powf(lambda, p);
    float alpha_lambda__p = alpha * powf(lambda_, p);
#ifdef USE_GPU
    int atom_i = blockDim.y * blockIdx.x + threadIdx.y;
    if (atom_i < local_atom_numbers - solvent_numbers)
#else
#pragma omp parallel for firstprivate(lambda, alpha_lambda_p, alpha_lambda__p)
    for (int atom_i = 0; atom_i < local_atom_numbers - solvent_numbers;
         atom_i++)
#endif
    {
        ATOM_GROUP nl_i = nl[atom_i];
        VECTOR_LJ_SOFT_TYPE r1 = crd[atom_i];
        VECTOR frc_record = {0., 0., 0.},
               frc_enhancing_record = {0.0f, 0.0f, 0.0f};
        LTMatrix3 virial_record = {0, 0, 0, 0, 0, 0},
                  virial_enhancing = {0, 0, 0, 0, 0, 0};
        float energy_total = 0., energy_enhancing = 0.0f;
        float energy_coulomb = 0.;
        float du_dlambda_lj = 0.;
        float du_dlambda_direct = 0.;
        // float du_dlambda_enhancing = 0.0f;
        int atom_mark_i = atom_sys_mark[atom_i];
#ifdef USE_GPU
        for (int j = threadIdx.x; j < nl_i.atom_numbers; j += blockDim.x)
#else
        for (int j = 0; j < nl_i.atom_numbers; j++)
#endif
        {
            int atom_j = nl_i.atom_serial[j];
            float ij_factor = atom_j < local_atom_numbers ? 1.0f : 0.5f;
            VECTOR_LJ_SOFT_TYPE r2 = crd[atom_j];
            VECTOR dr =
                Get_Displacement<BoundaryPolicy::Periodic>(r2, r1, boundary);
            float dr_abs = norm3df(dr.x, dr.y, dr.z);
            if (dr_abs < cutoff)
            {
                int atom_mark_j = atom_sys_mark[atom_j] + atom_mark_i;
                float factor = 0;
                if (atom_mark_j == 0)
                {
                    factor = 1;
                }
                else if (atom_mark_j == 1)
                {
                    factor = pwwp_factor;
                }
                int atom_pair_LJ_type_A = Get_LJ_Type(r1.LJ_type, r2.LJ_type);
                int atom_pair_LJ_type_B =
                    Get_LJ_Type(r1.LJ_type_B, r2.LJ_type_B);
                float AA = LJ_type_AA[atom_pair_LJ_type_A];
                float AB = LJ_type_AB[atom_pair_LJ_type_A];
                float BA = LJ_type_BA[atom_pair_LJ_type_B];
                float BB = LJ_type_BB[atom_pair_LJ_type_B];
                if (BA * AA != 0 || BA + AA == 0)
                {
                    if (need_force)
                    {
                        float frc_abs =
                            lambda_ * Get_LJ_Force(r1, r2, dr_abs, AA, AB) +
                            lambda * Get_LJ_Force(r1, r2, dr_abs, BA, BB);
                        if (need_coulomb)
                        {
                            float frc_cf_abs = Get_Direct_Coulomb_Force(
                                r1, r2, dr_abs, pme_beta);
                            frc_abs = frc_abs - frc_cf_abs;
                        }
                        VECTOR frc_lin = frc_abs * dr;
                        frc_record = frc_record + frc_lin;
                        frc_enhancing_record =
                            frc_enhancing_record + factor * frc_lin;
                        if (atom_j < local_atom_numbers)
                        {
                            atomicAdd(frc + atom_j, -frc_lin);
                            atomicAdd(frc_enhancing + atom_j,
                                      -factor * frc_lin);
                        }
                        if (need_virial)
                        {
                            LTMatrix3 virial0 =
                                Get_Virial_From_Force_Dis(frc_lin, dr);
                            virial_record = virial_record + ij_factor * virial0;
                            virial_enhancing =
                                virial_enhancing + ij_factor * factor * virial0;
                        }
                    }
                    if (need_coulomb && need_energy)
                    {
                        float ene =
                            Get_Direct_Coulomb_Energy(r1, r2, dr_abs, pme_beta);
                        energy_coulomb += ij_factor * ene;
                        energy_enhancing += ij_factor * factor * ene;
                    }
                    if (need_energy)
                    {
                        float ene =
                            lambda_ * Get_LJ_Energy(r1, r2, dr_abs, AA, AB) +
                            lambda * Get_LJ_Energy(r1, r2, dr_abs, BA, BB);
                        energy_total += ij_factor * ene;
                        energy_enhancing += ij_factor * factor * ene;
                    }
                    if (need_du_dlambda)
                    {
                        du_dlambda_lj += Get_LJ_Energy(r1, r2, dr_abs, BA, BB) -
                                         Get_LJ_Energy(r1, r2, dr_abs, AA, AB);
                        if (need_coulomb)
                        {
                            du_dlambda_direct += Get_Direct_Coulomb_dU_dlambda(
                                r1, r2, dr_abs, pme_beta);
                        }
                    }
                }
                else
                {
                    float sigma_A = Get_Soft_Core_Sigma(AA, AB, input_sigma_6,
                                                        input_sigma_6_min);
                    float sigma_B = Get_Soft_Core_Sigma(BA, BB, input_sigma_6,
                                                        input_sigma_6_min);
                    float dr_softcore_A = Get_Soft_Core_Distance(
                        AA, AB, sigma_A, dr_abs, alpha, p, lambda);
                    float dr_softcore_B = Get_Soft_Core_Distance(
                        BB, BA, sigma_B, dr_abs, alpha, p, 1 - lambda);
                    if (need_force)
                    {
                        float frc_abs =
                            lambda_ * Get_Soft_Core_LJ_Force(r1, r2, dr_abs,
                                                             dr_softcore_A, AA,
                                                             AB) +
                            lambda * Get_Soft_Core_LJ_Force(
                                         r1, r2, dr_abs, dr_softcore_B, BA, BB);
                        if (need_coulomb)
                        {
                            float frc_cf_abs =
                                lambda_ * Get_Soft_Core_Direct_Coulomb_Force(
                                              r1, r2, dr_abs, dr_softcore_A,
                                              pme_beta) +
                                lambda * Get_Soft_Core_Direct_Coulomb_Force(
                                             r1, r2, dr_abs, dr_softcore_B,
                                             pme_beta);
                            frc_abs = frc_abs - frc_cf_abs;
                        }
                        VECTOR frc_lin = frc_abs * dr;
                        frc_record = frc_record + frc_lin;
                        frc_enhancing_record =
                            frc_enhancing_record + factor * frc_lin;
                        if (atom_j < local_atom_numbers)
                        {
                            atomicAdd(frc + atom_j, -frc_lin);
                            atomicAdd(frc_enhancing + atom_j,
                                      -factor * frc_lin);
                        }
                        if (need_virial)
                        {
                            LTMatrix3 virial0 =
                                Get_Virial_From_Force_Dis(frc_lin, dr);
                            virial_record = virial_record + ij_factor * virial0;
                            virial_enhancing =
                                virial_enhancing + ij_factor * factor * virial0;
                        }
                    }
                    if (need_coulomb && need_energy)
                    {
                        float ene =
                            lambda_ * Get_Direct_Coulomb_Energy(
                                          r1, r2, dr_softcore_A, pme_beta) +
                            lambda * Get_Direct_Coulomb_Energy(
                                         r1, r2, dr_softcore_B, pme_beta);
                        energy_coulomb += ij_factor * ene;
                        energy_enhancing += ij_factor * factor * ene;
                    }
                    if (need_energy)
                    {
                        float ene =
                            lambda_ *
                                Get_LJ_Energy(r1, r2, dr_softcore_A, AA, AB) +
                            lambda *
                                Get_LJ_Energy(r1, r2, dr_softcore_B, BA, BB);
                        energy_total += ij_factor * ene;
                        energy_enhancing += ij_factor * factor * ene;
                    }
                    if (need_du_dlambda)
                    {
                        du_dlambda_lj +=
                            Get_LJ_Energy(r1, r2, dr_softcore_B, BA, BB) -
                            Get_LJ_Energy(r1, r2, dr_softcore_A, AA, AB);
                        du_dlambda_lj +=
                            Get_Soft_Core_dU_dlambda(
                                Get_LJ_Force(r1, r2, dr_softcore_A, AA, AB),
                                sigma_A, dr_softcore_A, alpha, p, lambda) -
                            Get_Soft_Core_dU_dlambda(
                                Get_LJ_Force(r1, r2, dr_softcore_B, BA, BB),
                                sigma_B, dr_softcore_B, alpha, p, lambda_);
                        if (need_coulomb)
                        {
                            du_dlambda_direct +=
                                Get_Direct_Coulomb_Energy(r1, r2, dr_softcore_B,
                                                          pme_beta) -
                                Get_Direct_Coulomb_Energy(r1, r2, dr_softcore_A,
                                                          pme_beta);
                            du_dlambda_direct +=
                                Get_Soft_Core_dU_dlambda(
                                    Get_Direct_Coulomb_Force(
                                        r1, r2, dr_softcore_B, pme_beta),
                                    sigma_B, dr_softcore_B, alpha, p, lambda_) -
                                Get_Soft_Core_dU_dlambda(
                                    Get_Direct_Coulomb_Force(
                                        r1, r2, dr_softcore_A, pme_beta),
                                    sigma_A, dr_softcore_A, alpha, p, lambda);
                            du_dlambda_direct +=
                                lambda * Get_Direct_Coulomb_dU_dlambda(
                                             r1, r2, dr_softcore_B, pme_beta) +
                                lambda_ * Get_Direct_Coulomb_dU_dlambda(
                                              r1, r2, dr_softcore_A, pme_beta);
                        }
                    }
                }
            }
        }
        if (need_force)
        {
            Warp_Sum_To(frc + atom_i, frc_record, warpSize);
            Warp_Sum_To(frc_enhancing + atom_i, frc_enhancing_record, warpSize);
        }
        if (need_coulomb && need_energy)
        {
            Warp_Sum_To(atom_direct_cf_energy + atom_i, energy_coulomb,
                        warpSize);
        }
        if (need_energy)
        {
            Warp_Sum_To(atom_energy + atom_i, energy_total, warpSize);
#ifdef USE_GPU
            if (threadIdx.x == 0)
#endif
                atomicAdd(atom_ene_LJ + atom_i, energy_total);
            Warp_Sum_To(atom_energy_enhancing + atom_i, energy_enhancing,
                        warpSize);
        }
        if (need_virial)
        {
            Warp_Sum_To(atom_virial + atom_i, virial_record, warpSize);
            Warp_Sum_To(atom_virial_enhancing + atom_i, virial_enhancing,
                        warpSize);
        }
        if (need_du_dlambda)
        {
            Warp_Sum_To(atom_du_dlambda_lj, du_dlambda_lj, warpSize);
            if (need_coulomb)
            {
                Warp_Sum_To(atom_du_dlambda_direct, du_dlambda_direct,
                            warpSize);
            }
        }
    }
}

static __device__ float log_add_log(float a, float b)
{
    return fmaxf(a, b) + logf(1.0 + expf(-fabsf(a - b)));
}

static __global__ void SITS_Record_Ene_Device(float* ene_record,
                                              const float* enhancing_energy,
                                              const float pe_a,
                                              const float pe_b)
{
    *ene_record = pe_a * *enhancing_energy + pe_b;
}

static __global__ void SITS_Update_gf_Device(const int kn, float* gf,
                                             const float* ene_record,
                                             const float* log_nk,
                                             const float* beta_k)
{
#ifdef USE_GPU
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < kn)
#else
#pragma omp parallel for
    for (int i = 0; i < kn; i++)
#endif
    {
        gf[i] = -beta_k[i] * ene_record[0] + log_nk[i];
    }
}

static __global__ void SITS_Update_gfsum_Device(const int kn, float* gfsum,
                                                const float* gf)
{
    float temp = -FLT_MAX;
    for (int i = 0; i < kn; i = i + 1)
    {
        temp = log_add_log(temp, gf[i]);
    }
    gfsum[0] = temp;
}

static __global__ void SITS_Update_log_pk_Device(const int kn, float* log_pk,
                                                 const float* gf,
                                                 const float* gfsum,
                                                 const int reset)
{
#ifdef USE_GPU
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < kn)
#else
#pragma omp parallel for
    for (int i = 0; i < kn; i++)
#endif
    {
        float gfi = gf[i];
        log_pk[i] =
            ((float)reset) * gfi +
            ((float)(1 - reset)) * log_add_log(log_pk[i], gfi - gfsum[0]);
    }
}

static __global__ void SITS_Update_log_mk_inverse_Device(
    const int kn, float* log_weight, float* log_mk_inverse, float* log_norm_old,
    float* log_norm, const float* log_pk, const float* log_nk)
{
#ifdef USE_GPU
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < kn - 1)
#else
#pragma omp parallel for
    for (int i = 0; i < kn - 1; i++)
#endif
    {
        log_weight[i] = (log_pk[i] + log_pk[i + 1]) * 0.5;
        log_mk_inverse[i] = log_nk[i] - log_nk[i + 1];
        log_norm_old[i] = log_norm[i];
        log_norm[i] = log_add_log(log_norm[i], log_weight[i]);
        log_mk_inverse[i] =
            log_add_log(log_mk_inverse[i] + log_norm_old[i] - log_norm[i],
                        log_pk[i + 1] - log_pk[i] + log_mk_inverse[i] +
                            log_weight[i] - log_norm[i]);
    }
}

static __global__ void SITS_Update_log_nk_inverse_Device(
    const int kn, float* log_nk_inverse, const float* log_mk_inverse)
{
    for (int i = 0; i < kn - 1; i++)
    {
        log_nk_inverse[i + 1] = log_nk_inverse[i] + log_mk_inverse[i];
    }
}

static __global__ void SITS_Update_nk_Device(const int kn, float* log_nk,
                                             float* nk,
                                             const float* log_nk_inverse)
{
#ifdef USE_GPU
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < kn)
#else
#pragma omp parallel for
    for (int i = 0; i < kn; i++)
#endif
    {
        log_nk[i] = -log_nk_inverse[i];
        nk[i] = exp(log_nk[i]);
    }
}

static __global__ void SITS_For_Enhanced_Force_Calculate_NkExpBetakU_Device(
    const int k_numbers, const float* beta_k, const float* log_nk,
    float* nkexpbetaku, const float* ene, const float beta0, const float pe_a,
    const float pe_b)
{
#ifdef USE_GPU
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < k_numbers)
#else
#pragma omp parallel for
    for (int i = 0; i < k_numbers; i++)
#endif
    {
        nkexpbetaku[i] =
            -(beta_k[i] - beta0) * (pe_a * ene[0] + pe_b) + log_nk[i];
    }
}

static __global__ void SITS_For_Enhanced_Force_Sum_Of_Above_And_Below_Device(
    const int k_numbers, const float* nkexpbetaku, const float* beta_k,
    float* d_bias, float pe_a, float pe_b, float* sum_of_above,
    float* sum_of_below, float* factor, float beta0, float fb_bias,
    const float* h_enhancing_energy)
{
    float above = -FLT_MAX;
    float below = -FLT_MAX;
    for (int i = 0; i < k_numbers; i++)
    {
        above = log_add_log(above, logf(beta_k[i]) + nkexpbetaku[i]);
        below = log_add_log(below, nkexpbetaku[i]);
    }
    sum_of_above[0] = above;
    sum_of_below[0] = below;
    factor[0] = expf(above - below - logf(beta0)) + fb_bias;
    d_bias[0] =
        -below / beta0 / pe_a + fb_bias * (h_enhancing_energy[0] + pe_b / pe_a);
}

static __global__ void SITS_For_Enhanced_Force_Protein_Water_Device(
    const int atom_numbers, VECTOR* md_frc, const VECTOR* enhancing_frc,
    float* md_ene, const float* bias, const int need_pressure,
    LTMatrix3* md_virial, const LTMatrix3* virial_enhancing,
    const float factor_minus_one)
{
#ifdef USE_GPU
    if (blockIdx.x == 0 && threadIdx.x == 0)
#endif
    {
        md_ene[0] = md_ene[0] + bias[0];
        if (need_pressure)
        {
            md_virial[0] =
                md_virial[0] + factor_minus_one * virial_enhancing[0];
        }
    }
#ifdef USE_GPU
    __syncthreads();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < atom_numbers)
#else
#pragma omp parallel for
    for (int i = 0; i < atom_numbers; i++)
#endif
    {
        md_frc[i] = md_frc[i] + factor_minus_one * enhancing_frc[i];
    }
}

static __global__ void ESITS_Get_Current_Fb(const float* enhancing_energy,
                                            float* factor, const float pe_a,
                                            const float pe_b,
                                            const float beta_high,
                                            const float beta_low, float* d_bias)
{
    float ene = enhancing_energy[0];
    if (ene > pe_b)
    {
        factor[0] =
            beta_high - (beta_high - beta_low) * pe_a / (ene - pe_b + pe_a);
        d_bias[0] = -pe_a * logf(enhancing_energy[0] - pe_b + pe_a);
    }
    else
    {
        factor[0] = beta_low;
        d_bias[0] = -enhancing_energy[0];
    }
}

static __global__ void AMD_Get_Current_Fb(const float* enhancing_energy,
                                          float* factor, const float pe_a,
                                          const float pe_b, float* d_bias)
{
    float ene = enhancing_energy[0];
    if (ene < pe_b)
    {
        ene = pe_b - ene;
        factor[0] = 1.0f - ene * (2 * pe_a + ene) / (pe_a + ene) / (pe_a + ene);
        d_bias[0] = ene * ene / (pe_a + ene);
    }
    else
    {
        factor[0] = 1.0f;
        d_bias[0] = 0;
    }
}

static __global__ void GAMD_Get_Current_Fb(const float* enhancing_energy,
                                           float* factor, const float pe_a,
                                           const float pe_b, float* d_bias)
{
    float ene = enhancing_energy[0];
    if (ene < pe_b)
    {
        factor[0] = 1.0f - pe_a * ene;
        d_bias[0] = 0.5 * pe_a * ene * ene;
    }
    else
    {
        factor[0] = 1.0f;
        d_bias[0] = 0;
    }
}

static void SITS_Get_Current_Fb(const int atom_numbers,
                                const float* energy_enhancing, float* d_bias,
                                const int k_numbers, float* nkexpbetaku,
                                const float* beta_k, const float* log_nk,
                                const float beta0, float* sum_a, float* sum_b,
                                float* factor, const float fb_bias,
                                const float pe_a, const float pe_b,
                                const float pwwp_enhance_factor)
{
    Launch_Device_Kernel(SITS_For_Enhanced_Force_Calculate_NkExpBetakU_Device,
                         (k_numbers + 63) / 64, 64, 0, NULL, k_numbers, beta_k,
                         log_nk, nkexpbetaku, energy_enhancing, beta0, pe_a,
                         pe_b);

    Launch_Device_Kernel(SITS_For_Enhanced_Force_Sum_Of_Above_And_Below_Device,
                         1, 1, 0, NULL, k_numbers, nkexpbetaku, beta_k, d_bias,
                         pe_a, pe_b, sum_a, sum_b, factor, beta0, fb_bias,
                         energy_enhancing);
}

void CLASSIC_SITS_INFORMATION::Initial(CONTROLLER* controller,
                                       SITS_INFORMATION* sits)
{
    is_initialized = 1;
    sits_controller = sits;
    record_count = 0;
    fb_interval = 1;
    Device_Malloc_Safely((void**)&d_bias, sizeof(float));
    if (controller->Command_Exist(sits->module_name, "fb_interval"))
    {
        controller->Check_Int(sits->module_name, "fb_interval",
                              "CLASSIC_SITS_INFORMATION::Initial");
        fb_interval =
            atoi(controller->Command(sits->module_name, "fb_interval"));
    }
    controller->printf("    SITS fb update interval set to %d\n", fb_interval);
    if (sits->sits_mode == SITS_MODE_AMD)
    {
        if (controller->Command_Exist(sits->module_name, "pe_a"))
        {
            controller->Check_Float(sits->module_name, "pe_a",
                                    "CLASSIC_SITS_INFORMATION::Initial");
            pe_a = atof(controller->Command(sits->module_name, "pe_a"));
        }
        else
        {
            controller->Throw_SPONGE_Error(
                spongeErrorMissingCommand, "CLASSIC_SITS_INFORMATION::Initial",
                "Reason:\n\tAlpha (pe_a) is required for the Accelerated MD");
        }
        controller->printf("    AMD alpha (pe_a) set to %f\n", pe_a);

        if (controller->Command_Exist(sits->module_name, "pe_b"))
        {
            controller->Check_Float(sits->module_name, "pe_b",
                                    "CLASSIC_SITS_INFORMATION::Initial");
            pe_b = atof(controller->Command(sits->module_name, "pe_b"));
        }
        else
        {
            controller->Throw_SPONGE_Error(
                spongeErrorMissingCommand, "CLASSIC_SITS_INFORMATION::Initial",
                "Reason:\n\tE (pe_b) is required for the Accelerated MD");
        }
        controller->printf("    AMD E (pe_b) set to %f\n", pe_b);

        k_numbers = 0;
        nk_fix = 1;
        record_interval = 1;
        update_interval = INT_MAX;
        Memory_Allocate();
    }
    else if (sits->sits_mode == SITS_MODE_GAMD)
    {
        if (controller->Command_Exist(sits->module_name, "pe_a"))
        {
            controller->Check_Float(sits->module_name, "pe_a",
                                    "CLASSIC_SITS_INFORMATION::Initial");
            pe_a = atof(controller->Command(sits->module_name, "pe_a"));
        }
        else
        {
            controller->Throw_SPONGE_Error(spongeErrorMissingCommand,
                                           "CLASSIC_SITS_INFORMATION::Initial",
                                           "Reason:\n\tk (pe_a) is required "
                                           "for the Gaussian Accelerated MD");
        }
        controller->printf("    GAMD k (pe_a) set to %f\n", pe_a);

        if (controller->Command_Exist(sits->module_name, "pe_b"))
        {
            controller->Check_Float(sits->module_name, "pe_b",
                                    "CLASSIC_SITS_INFORMATION::Initial");
            pe_b = atof(controller->Command(sits->module_name, "pe_b"));
        }
        else
        {
            controller->Throw_SPONGE_Error(
                spongeErrorMissingCommand, "CLASSIC_SITS_INFORMATION::Initial",
                "Reason:\n\tE (pe_b) is required for the Accelerated MD");
        }
        controller->printf("    GAMD E (pe_b) set to %f\n", pe_b);

        k_numbers = 0;
        nk_fix = 1;
        record_interval = 1;
        update_interval = INT_MAX;
        Memory_Allocate();
    }
    else if (sits->sits_mode == SITS_MODE_EMPIRICAL)
    {
        if (controller->Command_Exist(sits->module_name, "pe_a"))
        {
            controller->Check_Float(sits->module_name, "pe_a",
                                    "CLASSIC_SITS_INFORMATION::Initial");
            pe_a = atof(controller->Command(sits->module_name, "pe_a"));
        }
        else
        {
            pe_a = 1.0;
        }
        controller->printf("    SITS_pe_a set to %f\n", pe_a);

        if (controller->Command_Exist(sits->module_name, "pe_b"))
        {
            controller->Check_Float(sits->module_name, "pe_b",
                                    "CLASSIC_SITS_INFORMATION::Initial");
            pe_b = atof(controller->Command(sits->module_name, "pe_b"));
        }
        else
        {
            pe_b = 0.0;
        }
        controller->printf("    SITS_pe_b set to %f\n", pe_b);

        if (!controller->Command_Exist(sits->module_name, "T_low") ||
            !controller->Command_Exist(sits->module_name, "T_high"))
        {
            controller->Throw_SPONGE_Error(
                spongeErrorMissingCommand, "CLASSIC_SITS_INFORMATION::Initial",
                "Reason:\n\tSITS_T_high and SITS_T_low are required for "
                "empirical SITS");
        }
        controller->Check_Float(sits->module_name, "T_low",
                                "CLASSIC_SITS_INFORMATION::Initial");
        controller->Check_Float(sits->module_name, "T_high",
                                "CLASSIC_SITS_INFORMATION::Initial");
        T_low = atof(controller->Command(sits->module_name, "T_low"));
        T_high = atof(controller->Command(sits->module_name, "T_high"));
        controller->printf("    SITS_T_high set to %f\n", T_high);
        controller->printf("    SITS_T_low set to %f\n", T_low);

        k_numbers = 0;
        nk_fix = 1;
        record_interval = 1;
        update_interval = INT_MAX;
        Memory_Allocate();
    }
    else if (sits->sits_mode != SITS_MODE_OBSERVATION)
    {
        if (controller->Command_Exist(sits->module_name, "k_numbers"))
        {
            controller->Check_Int(sits->module_name, "k_numbers",
                                  "CLASSIC_SITS_INFORMATION::Initial");
            k_numbers = atoi(controller->Command("SITS_k_numbers"));
            if (k_numbers <= 0)
            {
                controller->Throw_SPONGE_Error(
                    spongeErrorValueErrorCommand,
                    "CLASSIC_SITS_INFORMATION::Initial",
                    "Reason:\n\tSITS k numbers cannot be smaller than 0\n");
            }
        }
        else
        {
            k_numbers = 40;
        }
        controller->printf("    k numbers is %d\n", k_numbers);
        Memory_Allocate();

        controller->printf("    Read %s temperature information.\n",
                           sits->module_name);
        float* beta_k_tmp;
        Malloc_Safely((void**)&beta_k_tmp, sizeof(float) * k_numbers);
        if (controller->Command_Exist(sits->module_name, "T_low"))
        {
            if (!controller->Command_Exist(sits->module_name, "T_high"))
            {
                controller->Throw_SPONGE_Error(
                    spongeErrorMissingCommand,
                    "CLASSIC_SITS_INFORMATION::Initial",
                    "Reason:\n\tSITS T high must be explicitly given with SITS "
                    "T low in mdin\n");
            }
            controller->Check_Float(sits->module_name, "T_low",
                                    "CLASSIC_SITS_INFORMATION::Initial");
            controller->Check_Float(sits->module_name, "T_high",
                                    "CLASSIC_SITS_INFORMATION::Initial");
            T_low = atof(controller->Command(sits->module_name, "T_low"));
            T_high = atof(controller->Command(sits->module_name, "T_high"));
            float T_space = (T_high - T_low) / (k_numbers - 1);
            for (int i = 0; i < k_numbers; ++i)
            {
                beta_k_tmp[i] = 1.0 / (CONSTANT_kB * (T_low + T_space * i));
            }
        }
        else if (controller->Command_Exist(sits->module_name, "T"))
        {
            const char* char_pt = controller->Command(sits->module_name, "T");
            for (int i = 0; i < k_numbers; ++i)
            {
                float tmp_T;
                sscanf(char_pt, "%f", &tmp_T);
                if (i != k_numbers - 1)
                {
                    while (*char_pt != '/' && *char_pt != '\0') ++char_pt;
                    if (*char_pt == '/') ++char_pt;
                    if (*char_pt == '\0')
                    {
                        controller->Throw_SPONGE_Error(
                            spongeErrorValueErrorCommand,
                            "CLASSIC_SITS_INFORMATION::Initial",
                            "Reason:\n\tthe number of temperatures SITS_T != "
                            "SITS_k_numbers\n");
                    }
                }
                beta_k_tmp[i] = 1.0 / (CONSTANT_kB * tmp_T);
            }
        }
        else
        {
            controller->Throw_SPONGE_Error(
                spongeErrorMissingCommand, "CLASSIC_SITS_INFORMATION::Initial",
                "Reason:\n\tSITS T must be explicitly given in mdin.\n");
        }
        deviceMemcpy(beta_k, beta_k_tmp, sizeof(float) * k_numbers,
                     deviceMemcpyHostToDevice);
        free(beta_k_tmp);
        if (controller->Command_Exist(sits->module_name, "record_interval"))
        {
            controller->Check_Int(sits->module_name, "record_interval",
                                  "CLASSIC_SITS_INFORMATION::Initial");
            record_interval =
                atoi(controller->Command(sits->module_name, "record_interval"));
        }
        else
        {
            record_interval = 1;
        }
        controller->printf("    SITS record interval set to %d\n",
                           record_interval);

        if (controller->Command_Exist(sits->module_name, "update_interval"))
        {
            controller->Check_Int(sits->module_name, "update_interval",
                                  "CLASSIC_SITS_INFORMATION::Initial");
            update_interval =
                atoi(controller->Command(sits->module_name, "update_interval"));
        }
        else
        {
            update_interval = 100;
        }
        controller->printf("    SITS update interval set to %d\n",
                           update_interval);

        if (controller->Command_Exist(sits->module_name, "pe_a"))
        {
            controller->Check_Float(sits->module_name, "pe_a",
                                    "CLASSIC_SITS_INFORMATION::Initial");
            pe_a = atof(controller->Command(sits->module_name, "pe_a"));
        }
        else
        {
            pe_a = 1.0;
        }
        controller->printf("    SITS_pe_a set to %f\n", pe_a);

        if (controller->Command_Exist(sits->module_name, "pe_b"))
        {
            controller->Check_Float(sits->module_name, "pe_b",
                                    "CLASSIC_SITS_INFORMATION::Initial");
            pe_b = atof(controller->Command(sits->module_name, "pe_b"));
        }
        else
        {
            pe_b = 0.0;
        }
        controller->printf("    SITS_pe_b set to %f\n", pe_b);

        if (controller->Command_Exist(sits->module_name, "fb_bias"))
        {
            controller->Check_Float(sits->module_name, "fb_bias",
                                    "CLASSIC_SITS_INFORMATION::Initial");
            fb_bias = atof(controller->Command(sits->module_name, "fb_bias"));
        }
        else
        {
            fb_bias = 0.0;
        }
        controller->printf("    SITS_fb_bias set to %f\n", fb_bias);

        reset = 1;

        int nk_rest;
        if (sits->sits_mode == SITS_MODE_ITERATION)
        {
            nk_rest = 0;
        }
        else
        {
            nk_rest = 1;
        }
        if (controller->Command_Exist(sits->module_name, "nk_rest"))
        {
            nk_rest = controller->Get_Bool(sits->module_name, "nk_rest",
                                           "CLASSIC_SITS_INFORMATION::Initial");
        }
        float* beta_lin;
        Malloc_Safely((void**)&beta_lin, sizeof(float) * k_numbers);

        for (int i = 0; i < k_numbers; ++i) beta_lin[i] = -FLT_MAX;

        deviceMemcpy(log_norm_old, beta_lin, sizeof(float) * k_numbers,
                     deviceMemcpyHostToDevice);
        deviceMemcpy(log_norm, beta_lin, sizeof(float) * k_numbers,
                     deviceMemcpyHostToDevice);
        deviceMemset(log_nk_inverse, 0, sizeof(float) * k_numbers);

        if (nk_rest == 0)
        {
            for (int i = 0; i < k_numbers; ++i)
            {
                beta_lin[i] = 0.0;
            }
        }
        else
        {
            const auto h5_nk = sits->h5_restart_state.float_states.find("nk");
            if (sits->has_h5_restart_state &&
                h5_nk != sits->h5_restart_state.float_states.end())
            {
                if (h5_nk->second.size() != static_cast<std::size_t>(k_numbers))
                {
                    controller->Throw_SPONGE_Error(
                        spongeErrorValueErrorCommand,
                        "CLASSIC_SITS_INFORMATION::Initial",
                        "Reason:\n\tH5 SITS nk count does not match "
                        "SITS_k_numbers\n");
                }
                controller->printf("    Read Nk from native H5 restart\n");
                for (int i = 0; i < k_numbers; ++i)
                {
                    const float value = h5_nk->second[i];
                    if (!(value > 0.0f) || !std::isfinite(value))
                    {
                        controller->Throw_SPONGE_Error(
                            spongeErrorValueErrorCommand,
                            "CLASSIC_SITS_INFORMATION::Initial",
                            "Reason:\n\tH5 SITS nk values must be finite "
                            "and positive\n");
                    }
                    beta_lin[i] = logf(value);
                }
            }
            else if (controller->Command_Exist(sits->module_name, "nk_in_file"))
            {
                FILE* nk_read_file;
                controller->printf(
                    "    Read Nk from %s\n",
                    controller->Command(sits->module_name, "nk_in_file"));
                Open_File_Safely(
                    &nk_read_file,
                    controller->Command(sits->module_name, "nk_in_file"), "r");
                for (int i = 0; i < k_numbers; ++i)
                {
                    if (fscanf(nk_read_file, "%f", beta_lin + i) != 1 ||
                        !(beta_lin[i] > 0.0f) || !std::isfinite(beta_lin[i]))
                    {
                        fclose(nk_read_file);
                        controller->Throw_SPONGE_Error(
                            spongeErrorBadFileFormat,
                            "CLASSIC_SITS_INFORMATION::Initial",
                            "Reason:\n\tSITS_nk_in_file must contain finite "
                            "positive values\n");
                    }
                    beta_lin[i] = logf(beta_lin[i]);
                }
                fclose(nk_read_file);
            }
            else
            {
                controller->Throw_SPONGE_Error(
                    spongeErrorMissingCommand,
                    "CLASSIC_SITS_INFORMATION::Initial",
                    "Reason:\n\tSITS_nk_in_file must be given when "
                    "SITS_nk_rest = 1 or SITS_mode = production\n");
            }
        }
        deviceMemcpy(log_nk, beta_lin, sizeof(float) * k_numbers,
                     deviceMemcpyHostToDevice);

        for (int i = 0; i < k_numbers; ++i)
        {
            beta_lin[i] = -beta_lin[i];
        }
        deviceMemcpy(log_nk_inverse, beta_lin, sizeof(float) * k_numbers,
                     deviceMemcpyHostToDevice);

        for (int i = 0; i < k_numbers; ++i)
        {
            beta_lin[i] = expf(-beta_lin[i]);
        }
        deviceMemcpy(Nk, beta_lin, sizeof(float) * k_numbers,
                     deviceMemcpyHostToDevice);

        free(beta_lin);
        Reset_List(factor, 1.0, 1);

        if (controller->Command_Exist(sits->module_name, "nk_fix"))
        {
            nk_fix = controller->Get_Bool(sits->module_name, "nk_fix",
                                          "CLASSIC_SITS_INFORMATION::Initial");
        }
        else if (sits->sits_mode == SITS_MODE_ITERATION)
        {
            nk_fix = 0;
        }
        else
        {
            nk_fix = 1;
        }
        controller->printf("    SITS nk fix is: %d\n", nk_fix);
        if (nk_fix == 0)
        {
            if (controller->Command_Exist(sits->module_name, "nk_rest_file"))
            {
                strcpy(nk_rest_file_name,
                       controller->Command(sits->module_name, "nk_rest_file"));
            }
            else if (controller->Command_Exist("default_out_file_prefix"))
            {
                strcpy(nk_rest_file_name,
                       controller->Command("default_out_file_prefix"));
                strcat(nk_rest_file_name, "_");
                strcat(nk_rest_file_name, sits->module_name);
                strcat(nk_rest_file_name, "_nk_rest.txt");
            }
            else
            {
                strcpy(nk_rest_file_name, sits->module_name);
                strcat(nk_rest_file_name, "_nk_rest.txt");
            }
            controller->printf("    Restart Nk will be written in %s\n",
                               nk_rest_file_name);
            std::string default_name = sits->module_name;
            default_name += "_nk_traj.dat";
            if (CONTROLLER::MPI_rank == 0)
            {
                nk_traj_file = controller->Get_Output_File(
                    true, sits->module_name, "nk_traj_file", "_nk_traj.dat",
                    default_name.c_str());
            }
        }
    }
}

void CLASSIC_SITS_INFORMATION::Memory_Allocate()
{
    Malloc_Safely((void**)&nk_record_cpu, sizeof(float) * k_numbers);
    Malloc_Safely((void**)&log_norm_record_cpu, sizeof(float) * k_numbers);

    Device_Malloc_Safely((void**)&ene_recorded, sizeof(float));
    Device_Malloc_Safely((void**)&gf, sizeof(float) * k_numbers);
    Device_Malloc_Safely((void**)&gfsum, sizeof(float));
    Device_Malloc_Safely((void**)&log_weight, sizeof(float) * k_numbers);
    Device_Malloc_Safely((void**)&log_mk_inverse, sizeof(float) * k_numbers);
    Device_Malloc_Safely((void**)&log_norm_old, sizeof(float) * k_numbers);
    Device_Malloc_And_Copy_Safely((void**)&log_norm, log_norm_record_cpu,
                                  sizeof(float) * k_numbers);
    Device_Malloc_Safely((void**)&log_pk, sizeof(float) * k_numbers);
    Device_Malloc_Safely((void**)&log_nk_inverse, sizeof(float) * k_numbers);
    Device_Malloc_Safely((void**)&log_nk, sizeof(float) * k_numbers);

    Device_Malloc_Safely((void**)&beta_k, sizeof(float) * k_numbers);
    Device_Malloc_Safely((void**)&NkExpBetakU, sizeof(float) * k_numbers);
    Device_Malloc_And_Copy_Safely((void**)&Nk, nk_record_cpu,
                                  sizeof(float) * k_numbers);
    Device_Malloc_Safely((void**)&sum_a, sizeof(float));
    Device_Malloc_Safely((void**)&sum_b, sizeof(float));
    Device_Malloc_And_Copy_Safely((void**)&factor, &sits_controller->h_factor,
                                  sizeof(float));
}

void CLASSIC_SITS_INFORMATION::SITS_Record_Ene()
{
    Launch_Device_Kernel(SITS_Record_Ene_Device, 1, 1, 0, NULL, ene_recorded,
                         sits_controller->pw_select.select_energy[0], pe_a,
                         pe_b);
}

void CLASSIC_SITS_INFORMATION::SITS_Update_gf()
{
    Launch_Device_Kernel(SITS_Update_gf_Device, (k_numbers + 63) / 64, 64, 0,
                         NULL, k_numbers, gf, ene_recorded, log_nk, beta_k);
}

void CLASSIC_SITS_INFORMATION::SITS_Update_gfsum()
{
    Launch_Device_Kernel(SITS_Update_gfsum_Device, 1, 1, 0, NULL, k_numbers,
                         gfsum, gf);
}

void CLASSIC_SITS_INFORMATION::SITS_Update_log_pk()
{
    Launch_Device_Kernel(SITS_Update_log_pk_Device, (k_numbers + 63) / 64, 64,
                         0, NULL, k_numbers, log_pk, gf, gfsum, reset);
}

void CLASSIC_SITS_INFORMATION::SITS_Update_log_mk_inverse()
{
    Launch_Device_Kernel(SITS_Update_log_mk_inverse_Device,
                         (k_numbers + 63) / 64, 64, 0, NULL, k_numbers,
                         log_weight, log_mk_inverse, log_norm_old, log_norm,
                         log_pk, log_nk);
}

void CLASSIC_SITS_INFORMATION::SITS_Update_log_nk_inverse()
{
    Launch_Device_Kernel(SITS_Update_log_nk_inverse_Device, 1, 1, 0, NULL,
                         k_numbers, log_nk_inverse, log_mk_inverse);
}

void CLASSIC_SITS_INFORMATION::SITS_Update_nk()
{
    Launch_Device_Kernel(SITS_Update_nk_Device, (k_numbers + 63) / 64, 64, 0,
                         NULL, k_numbers, log_nk, Nk, log_nk_inverse);
}

void CLASSIC_SITS_INFORMATION::SITS_Update_Fb(float beta_0, int step)
{
    if (!is_initialized ||
        sits_controller->sits_mode == SITS_MODE_OBSERVATION ||
        step % fb_interval != 0)
    {
        return;
    }
    if (sits_controller->sits_mode < SITS_MODE_EMPIRICAL)
    {
        SITS_Get_Current_Fb(sits_controller->atom_numbers,
                            sits_controller->pw_select.select_energy[0], d_bias,
                            k_numbers, NkExpBetakU, beta_k, log_nk, beta_0,
                            sum_a, sum_b, factor, fb_bias, pe_a, pe_b,
                            sits_controller->pwwp_enhance_factor);
        deviceMemcpy(&sits_controller->h_factor, factor, sizeof(float),
                     deviceMemcpyDeviceToHost);
    }
    else if (sits_controller->sits_mode == SITS_MODE_EMPIRICAL)
    {
        Launch_Device_Kernel(ESITS_Get_Current_Fb, 1, 1, 0, NULL,
                             sits_controller->pw_select.select_energy[0],
                             factor, pe_a, pe_b,
                             1.0f / (beta_0 * T_low * CONSTANT_kB),
                             1.0f / (beta_0 * T_high * CONSTANT_kB), d_bias);
        deviceMemcpy(&sits_controller->h_factor, factor, sizeof(float),
                     deviceMemcpyDeviceToHost);
    }
    else if (sits_controller->sits_mode == SITS_MODE_AMD)
    {
        Launch_Device_Kernel(AMD_Get_Current_Fb, 1, 1, 0, NULL,
                             sits_controller->pw_select.select_energy[0],
                             factor, pe_a, pe_b, d_bias);
        deviceMemcpy(&sits_controller->h_factor, factor, sizeof(float),
                     deviceMemcpyDeviceToHost);
    }
    else if (sits_controller->sits_mode == SITS_MODE_GAMD)
    {
        Launch_Device_Kernel(GAMD_Get_Current_Fb, 1, 1, 0, NULL,
                             sits_controller->pw_select.select_energy[0],
                             factor, pe_a, pe_b, d_bias);
        deviceMemcpy(&sits_controller->h_factor, factor, sizeof(float),
                     deviceMemcpyDeviceToHost);
    }
}

void CLASSIC_SITS_INFORMATION::SITS_Update_Common(const float beta)
{
    if (sits_controller->sits_mode != SITS_MODE_EMPIRICAL)
    {
        SITS_Record_Ene();
        SITS_Update_gf();
        SITS_Update_gfsum();
        SITS_Update_log_pk();
        reset = 0;
        record_count++;
    }
}

void CLASSIC_SITS_INFORMATION::SITS_Update_Nk()
{
    if (sits_controller->sits_mode != SITS_MODE_EMPIRICAL)
    {
        SITS_Update_log_mk_inverse();
        SITS_Update_log_nk_inverse();
        SITS_Update_nk();

        record_count = 0;
        reset = 1;

        SITS_Write_Nk_Norm();
    }
}

void CLASSIC_SITS_INFORMATION::SITS_Write_Nk_Norm()
{
#ifdef USE_MPI
    if (CONTROLLER::MPI_rank != 0) return;
#endif
    deviceMemcpy(nk_record_cpu, Nk, sizeof(float) * k_numbers,
                 deviceMemcpyDeviceToHost);
    h5_nk_pending = 1;
    h5_nk_step = sits_controller->is_initialized
                     ? sits_controller->classic_sits.record_count
                     : 0;
    if (nk_traj_file != NULL)
    {
        fwrite(nk_record_cpu, sizeof(float), k_numbers, nk_traj_file);
        SpongeLegacyIO::OutputFlushCoordinator::Mark_Dirty(
            nk_traj_file, "SITS nk trajectory");
    }

    Open_File_Safely(&nk_rest_file, nk_rest_file_name, "w");
    for (int i = 0; i < k_numbers; ++i)
    {
        fprintf(nk_rest_file, "%e ", nk_record_cpu[i]);
    }
    fclose(nk_rest_file);
}

void SITS_INFORMATION::Initial(CONTROLLER* controller, int atom_numbers_,
                               const BoundaryPolicy boundary_policy,
                               const char* given_module_name)
{
    if (given_module_name == NULL)
    {
        strcpy(module_name, "SITS");
        strcpy(print_aa_kab_name, "SITS");
        strcpy(print_bias_name, "SITS");
        strcpy(print_fb_name, "SITS");
    }
    else
    {
        strcpy(module_name, given_module_name);
        strcpy(print_aa_kab_name, given_module_name);
        strcpy(print_bias_name, given_module_name);
        strcpy(print_fb_name, given_module_name);
    }
    const SpongeH5MD::SitsH5Input h5_input =
        SpongeH5MD::Load_H5_SITS_Input(controller, module_name, atom_numbers_);
    has_h5_restart_state = h5_input.has_restart_state;
    h5_restart_state = h5_input.restart_state;
    strcat(print_aa_kab_name, "_AA_kAB");
    strcat(print_bias_name, "_bias");
    strcat(print_fb_name, "_fb");
    if (controller->Command_Exist(module_name, "mode"))
    {
        if (controller->Command_Choice(module_name, "mode", "observation"))
        {
            controller->printf(
                "START INITIALIZING %s\n    %s mode = observation\n",
                module_name, module_name);
            is_initialized = 1;
            sits_mode = SITS_MODE_OBSERVATION;
        }
        else if (controller->Command_Choice(module_name, "mode", "iteration"))
        {
            controller->printf(
                "START INITIALIZING %s\n    %s mode = iteration\n", module_name,
                module_name);
            is_initialized = 1;
            sits_mode = SITS_MODE_ITERATION;
        }
        else if (controller->Command_Choice(module_name, "mode", "production"))
        {
            controller->printf(
                "START INITIALIZING %s\n    %s mode = production\n",
                module_name, module_name);
            is_initialized = 1;
            sits_mode = SITS_MODE_PRODUCTION;
        }
        else if (controller->Command_Choice(module_name, "mode", "empirical"))
        {
            controller->printf(
                "START INITIALIZING %s\n    %s mode = empirical\n", module_name,
                module_name);
            is_initialized = 1;
            sits_mode = SITS_MODE_EMPIRICAL;
        }
        else if (controller->Command_Choice(module_name, "mode", "amd"))
        {
            controller->printf(
                "START INITIALIZING %s\n    %s mode = AMD (Accelerated MD)\n",
                module_name, module_name);
            is_initialized = 1;
            sits_mode = SITS_MODE_AMD;
        }
        else if (controller->Command_Choice(module_name, "mode", "gamd"))
        {
            controller->printf(
                "START INITIALIZING %s\n    %s mode = GAMD (Gaussian "
                "Accelerated MD)\n",
                module_name, module_name);
            is_initialized = 1;
            sits_mode = SITS_MODE_GAMD;
        }
        else
        {
            return;
        }
        atom_numbers = atom_numbers_;
        controller->printf("\tAtom numbers is %d\n", atom_numbers);
        Memory_Allocate();

        pw_select.Initial();
        pw_select.Add_One_Energy(atom_numbers);
        pw_select.Add_One_Force(atom_numbers);
        pw_select.Add_One_Virial(atom_numbers);

        if (controller->Command_Exist(module_name, "cross_enhance_factor"))
        {
            controller->Check_Float(module_name, "cross_enhance_factor",
                                    "SITS_INFORMATION::Initial");
            pwwp_enhance_factor =
                atof(controller->Command(module_name, "cross_enhance_factor"));
        }
        else
        {
            pwwp_enhance_factor = 0.5;
        }
        controller->printf("\tpwwp enhance factor set to %f\n",
                           pwwp_enhance_factor);

        this->selectively_applied = true;
        if (h5_input.has_atom_indices ||
            controller->Command_Exist(module_name, "atom_in_file") ||
            controller->Command_Exist(module_name, "atom_numbers"))
        {
            controller->printf("    Set atom atribution information\n");
            int* atom_sys_mark_cpu;
            Malloc_Safely((void**)&atom_sys_mark_cpu,
                          sizeof(int) * atom_numbers);
            for (int i = 0; i < atom_numbers; i++) atom_sys_mark_cpu[i] = 1;
            if (h5_input.has_atom_indices)
            {
                controller->printf(
                    "    Read atom selection from native H5 SITS protocol\n");
                for (const int atom : h5_input.atom_indices)
                {
                    atom_sys_mark_cpu[atom] = 0;
                }
            }
            else if (controller->Command_Exist(module_name, "atom_in_file"))
            {
                controller->printf("    reading %s_atom_in_file\n",
                                   module_name);
                FILE* fr = NULL;
                int temp_atom;
                Open_File_Safely(
                    &fr, controller->Command(module_name, "atom_in_file"), "r");
                while (fscanf(fr, "%d", &temp_atom) != EOF)
                {
                    atom_sys_mark_cpu[temp_atom] = 0;
                }
                fclose(fr);
            }
            else if (strcmp(controller->Command(module_name, "atom_numbers"),
                            "ITS") == 0 ||
                     strcmp(controller->Command(module_name, "atom_numbers"),
                            "ALL") == 0)
            {
                this->selectively_applied = false;
                for (int i = 0; i < atom_numbers; i++)
                {
                    atom_sys_mark_cpu[i] = 0;
                }
            }
            else
            {
                controller->Check_Int(module_name, "atom_numbers",
                                      "SITS_INFORMATION::Initial");
                int protein_numbers =
                    atoi(controller->Command(module_name, "atom_numbers"));
                for (int i = 0; i < protein_numbers; i++)
                {
                    atom_sys_mark_cpu[i] = 0;
                }
                for (int i = protein_numbers; i < atom_numbers; i++)
                {
                    atom_sys_mark_cpu[i] = 1;
                }
            }
            deviceMemcpy(atom_sys_mark, atom_sys_mark_cpu,
                         sizeof(int) * atom_numbers, deviceMemcpyHostToDevice);
            free(atom_sys_mark_cpu);
        }
        else
        {
            controller->Throw_SPONGE_Error(
                spongeErrorMissingCommand, "SITS_INFORMATION::Initial",
                "Reason:\n\tAtom information must be given in the form of "
                "SITS_atom_in_file or SITS_atom_numbers\n");
        }

        if (boundary_policy == BoundaryPolicy::Open && selectively_applied)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorConflictingCommand, "SITS_INFORMATION::Initial",
                "Reason:\n\tselective SITS requires periodic boundary "
                "conditions; NOPBC supports only full-system ITS/ALL\n");
        }

        classic_sits.Initial(controller, this);

        h_factor = 1.0f;

        controller->Step_Print_Initial(print_aa_kab_name, "%.2f");
        controller->Step_Print_Initial(print_bias_name, "%.4f");
        controller->Step_Print_Initial(print_fb_name, "%.4f");

        controller->printf("END INTIALIZING %s\n\n", module_name);
    }
    else
    {
        is_initialized = 0;
        return;
    }
}

void SITS_INFORMATION::Memory_Allocate()
{
    Device_Malloc_Safely((void**)&atom_sys_mark, sizeof(int) * atom_numbers);
    Device_Malloc_Safely((void**)&atom_sys_mark_local,
                         sizeof(int) * atom_numbers);
}

bool SITS_INFORMATION::Export_H5_Restart_State(
    SpongeH5MD::RestartSitsState* state, std::string* error_message) const
{
    auto fail = [error_message](const std::string& message)
    {
        if (error_message != NULL)
        {
            *error_message = message;
        }
        return false;
    };
    if (state == NULL) return fail("SITS restart state output is null");
    *state = {};
    if (!is_initialized || !classic_sits.is_initialized)
    {
        return true;
    }
    state->module_name = module_name;
    if (classic_sits.k_numbers <= 0) return true;

    const std::size_t count = static_cast<std::size_t>(classic_sits.k_numbers);
    auto& nk = state->float_states["nk"];
    auto& log_norm = state->float_states["log_norm"];
    auto& log_nk = state->float_states["log_nk"];
    nk.resize(count);
    log_norm.resize(count);
    log_nk.resize(count);
    deviceMemcpy(nk.data(), classic_sits.Nk, sizeof(float) * count,
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(log_norm.data(), classic_sits.log_norm, sizeof(float) * count,
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(log_nk.data(), classic_sits.log_nk, sizeof(float) * count,
                 deviceMemcpyDeviceToHost);
    for (std::size_t index = 0; index < count; ++index)
    {
        if (!(nk[index] > 0.0f) || !std::isfinite(nk[index]) ||
            !std::isfinite(log_norm[index]) || !std::isfinite(log_nk[index]))
        {
            return fail(
                "SITS restart state contains a non-finite or "
                "non-positive value");
        }
    }
    return true;
}

bool SITS_INFORMATION::Apply_H5_Restart_State(
    const SpongeH5MD::RestartProtocolState& state, std::string* error_message)
{
    auto fail = [error_message](const std::string& message)
    {
        if (error_message != NULL)
        {
            *error_message = message;
        }
        return false;
    };

    if (!is_initialized || !classic_sits.is_initialized)
    {
        return fail("SITS module is not initialized");
    }

    std::vector<float> nk_values;
    if (!SpongeH5MD::Extract_Sits_Nk_Protocol_State(state, module_name,
                                                    classic_sits.k_numbers,
                                                    &nk_values, error_message))
    {
        return false;
    }

    const SpongeH5MD::RestartSitsState* module_state = NULL;
    for (const auto& candidate : state.sits_states)
    {
        if (candidate.module_name == module_name)
        {
            module_state = &candidate;
            break;
        }
    }
    if (module_state == NULL)
    {
        return fail("H5 restart does not contain the active SITS module");
    }
    std::vector<float> log_nk_values(nk_values.size());
    const auto log_nk_state = module_state->float_states.find("log_nk");
    if (log_nk_state != module_state->float_states.end())
    {
        if (log_nk_state->second.size() != nk_values.size())
        {
            return fail("H5 SITS log_nk count does not match nk count");
        }
        log_nk_values = log_nk_state->second;
    }
    else
    {
        for (std::size_t i = 0; i < nk_values.size(); ++i)
        {
            log_nk_values[i] = logf(nk_values[i]);
        }
    }
    const std::vector<float>* log_norm_values = NULL;
    const auto log_norm_state = module_state->float_states.find("log_norm");
    if (log_norm_state != module_state->float_states.end())
    {
        if (log_norm_state->second.size() != nk_values.size())
        {
            return fail("H5 SITS log_norm count does not match nk count");
        }
        for (const float value : log_norm_state->second)
        {
            if (!std::isfinite(value))
            {
                return fail("H5 SITS log_norm values must be finite");
            }
        }
        log_norm_values = &log_norm_state->second;
    }
    std::vector<float> log_nk_inverse_values(nk_values.size());
    for (std::size_t i = 0; i < nk_values.size(); ++i)
    {
        if (!std::isfinite(log_nk_values[i]))
        {
            return fail("H5 SITS log_nk values must be finite");
        }
    }
    for (std::size_t i = 0; i < nk_values.size(); ++i)
    {
        classic_sits.nk_record_cpu[i] = nk_values[i];
        log_nk_inverse_values[i] = -log_nk_values[i];
    }

    deviceMemcpy(classic_sits.Nk, nk_values.data(),
                 sizeof(float) * nk_values.size(), deviceMemcpyHostToDevice);
    deviceMemcpy(classic_sits.log_nk, log_nk_values.data(),
                 sizeof(float) * log_nk_values.size(),
                 deviceMemcpyHostToDevice);
    deviceMemcpy(classic_sits.log_nk_inverse, log_nk_inverse_values.data(),
                 sizeof(float) * log_nk_inverse_values.size(),
                 deviceMemcpyHostToDevice);
    if (log_norm_values != NULL)
    {
        for (std::size_t i = 0; i < nk_values.size(); ++i)
        {
            classic_sits.log_norm_record_cpu[i] = (*log_norm_values)[i];
        }
        deviceMemcpy(classic_sits.log_norm, log_norm_values->data(),
                     sizeof(float) * log_norm_values->size(),
                     deviceMemcpyHostToDevice);
        deviceMemcpy(classic_sits.log_norm_old, log_norm_values->data(),
                     sizeof(float) * log_norm_values->size(),
                     deviceMemcpyHostToDevice);
    }
    return true;
}

void SITS_INFORMATION::Reset_Force_Energy(int* md_need_potential)
{
    if (!is_initialized) return;
    md_need_potential[0] += 1;

    deviceMemset(pw_select.select_atom_energy[0], 0,
                 sizeof(float) * atom_numbers);
    deviceMemset(pw_select.select_energy[0], 0, sizeof(float));
    deviceMemset(pw_select.select_force[0], 0, sizeof(VECTOR) * atom_numbers);
    deviceMemset(pw_select.select_atom_virial[0], 0,
                 sizeof(float) * atom_numbers);
    deviceMemset(pw_select.select_virial[0], 0, sizeof(float));
}

void SITS_INFORMATION::Update_And_Enhance(const int step,
                                          float* d_total_potential,
                                          int need_pressure,
                                          LTMatrix3* d_total_virial,
                                          VECTOR* frc, float beta0)
{
    if (!is_initialized) return;
    if (selectively_applied)
    {
        Sum_Of_List(pw_select.select_atom_energy[0], pw_select.select_energy[0],
                    atom_numbers);
#ifdef USE_MPI
        if (CONTROLLER::PP_MPI_size != 1)
            D_MPI_Allreduce_IN_PLACE(pw_select.select_energy[0], 1, D_MPI_FLOAT,
                                     D_MPI_SUM, CONTROLLER::d_pp_comm, NULL);
#endif
        if (need_pressure)
        {
            Sum_Of_List(pw_select.select_atom_virial[0],
                        pw_select.select_virial[0], atom_numbers);
#ifdef USE_MPI
            if (CONTROLLER::PP_MPI_size != 1)
                D_MPI_Allreduce_IN_PLACE(pw_select.select_virial[0], 6,
                                         D_MPI_FLOAT, D_MPI_SUM,
                                         CONTROLLER::d_pp_comm, NULL);
#endif
        }
    }
    else
    {
        deviceMemcpy(pw_select.select_energy[0], d_total_potential,
                     sizeof(float), deviceMemcpyDeviceToDevice);
        deviceMemcpy(pw_select.select_force[0], frc,
                     sizeof(VECTOR) * atom_numbers, deviceMemcpyDeviceToDevice);
        if (need_pressure)
        {
            deviceMemcpy(pw_select.select_virial[0], d_total_virial,
                         sizeof(float), deviceMemcpyDeviceToDevice);
        }
    }
    if (sits_mode != SITS_MODE_OBSERVATION && !classic_sits.nk_fix &&
        step % classic_sits.record_interval == 0)
    {
        classic_sits.SITS_Update_Common(beta0);
        if (classic_sits.record_count % classic_sits.update_interval == 0)
        {
            classic_sits.SITS_Update_Nk();
        }
    }
    if (sits_mode != SITS_MODE_OBSERVATION)
    {
        classic_sits.SITS_Update_Fb(beta0, step);
    }
    Launch_Device_Kernel(SITS_For_Enhanced_Force_Protein_Water_Device,
                         (atom_numbers + CONTROLLER::device_max_thread - 1) /
                             CONTROLLER::device_max_thread,
                         CONTROLLER::device_max_thread, 0, NULL, atom_numbers,
                         frc, pw_select.select_force[0], d_total_potential,
                         classic_sits.d_bias, need_pressure, d_total_virial,
                         pw_select.select_virial_tensor[0], h_factor - 1);
}

void SITS_INFORMATION::SITS_LJ_Direct_CF_Force_With_Atom_Energy_And_Virial(
    const int atom_numbers, const int local_atom_numbers,
    const int solvent_numbers, const int ghost_numbers, const VECTOR* crd,
    const float* charge, LENNARD_JONES_INFORMATION* lj_info, VECTOR* md_frc,
    const Boundary boundary, const ATOM_GROUP* nl, const float cutoff,
    const float pme_beta, const int need_potential, float* atom_energy,
    const int need_pressure, LTMatrix3* atom_virial, float* coulomb_atom_ene)
{
    if (is_initialized && lj_info->is_initialized)
    {
        Launch_Device_Kernel(
            Copy_Crd_And_Charge_To_New_Crd,
            (this->atom_numbers + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL,
            local_atom_numbers + ghost_numbers, crd,
            lj_info->crd_with_LJ_parameters_local, charge);

        if (need_potential)
        {
            deviceMemset(coulomb_atom_ene, 0,
                         sizeof(float) * (local_atom_numbers + ghost_numbers));
            deviceMemset(lj_info->d_LJ_energy_atom, 0,
                         sizeof(float) * (local_atom_numbers + ghost_numbers));
            deviceMemset(classic_sits.d_bias, 0, sizeof(float));
        }
        if (!local_atom_numbers) return;

        auto f = Selective_Lennard_Jones_And_Direct_Coulomb_Device<true, false,
                                                                   false, true>;
        dim3 blockSize = {
            CONTROLLER::device_warp,
            CONTROLLER::device_max_thread / CONTROLLER::device_warp};
        dim3 gridSize = (local_atom_numbers + blockSize.y - 1) / blockSize.y;

        if (need_potential && !need_pressure)
        {
            f = Selective_Lennard_Jones_And_Direct_Coulomb_Device<true, true,
                                                                  false, true>;
        }
        else if (need_potential && need_pressure)
        {
            f = Selective_Lennard_Jones_And_Direct_Coulomb_Device<true, true,
                                                                  true, true>;
        }
        else if (!need_potential && need_pressure)
        {
            f = Selective_Lennard_Jones_And_Direct_Coulomb_Device<true, false,
                                                                  true, true>;
        }
        else
        {
            f = Selective_Lennard_Jones_And_Direct_Coulomb_Device<true, false,
                                                                  false, true>;
        }

        Launch_Device_Kernel(
            f, gridSize, blockSize, 0, NULL, local_atom_numbers,
            solvent_numbers, nl, lj_info->d_LJ_energy_atom,
            lj_info->crd_with_LJ_parameters_local, boundary, lj_info->d_LJ_A,
            lj_info->d_LJ_B, atom_sys_mark_local, cutoff, md_frc,
            pw_select.select_force[0], pme_beta, atom_energy,
            pw_select.select_atom_energy[0], atom_virial,
            pw_select.select_atom_virial_tensor[0], coulomb_atom_ene,
            pwwp_enhance_factor);
    }
}

void SITS_INFORMATION::
    SITS_LJ_Soft_Core_Direct_CF_Force_With_Atom_Energy_And_Virial(
        const int atom_numbers, const int local_atom_numbers,
        const int solvent_numbers, const int ghost_numbers, const VECTOR* crd,
        const float* charge, LJ_SOFT_CORE* lj_info, VECTOR* md_frc,
        const Boundary boundary, const ATOM_GROUP* nl, const float cutoff,
        const float pme_beta, const int need_potential, float* atom_energy,
        const int need_pressure, LTMatrix3* atom_virial,
        float* coulomb_atom_ene)
{
    if (is_initialized && lj_info->is_initialized)
    {
        Launch_Device_Kernel(
            Copy_Crd_And_Charge_To_New_Crd,
            (this->atom_numbers + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL,
            local_atom_numbers + ghost_numbers, crd,
            lj_info->crd_with_LJ_parameters_local, charge);

        if (need_potential)
        {
            deviceMemset(coulomb_atom_ene, 0,
                         sizeof(float) * (local_atom_numbers + ghost_numbers));
            deviceMemset(lj_info->d_LJ_energy_atom, 0,
                         sizeof(float) * (local_atom_numbers + ghost_numbers));
            deviceMemset(classic_sits.d_bias, 0, sizeof(float));
        }
        if (!local_atom_numbers) return;

        auto f = Selective_Lennard_Jones_And_Direct_Coulomb_Soft_Core_Device<
            true, false, false, true, false>;
        dim3 blockSize = {
            CONTROLLER::device_warp,
            CONTROLLER::device_max_thread / CONTROLLER::device_warp};
        dim3 gridSize = (local_atom_numbers + blockSize.y - 1) / blockSize.y;

        if (need_potential && !need_pressure)
        {
            f = Selective_Lennard_Jones_And_Direct_Coulomb_Soft_Core_Device<
                true, true, false, true, false>;
        }
        else if (need_potential && need_pressure)
        {
            f = Selective_Lennard_Jones_And_Direct_Coulomb_Soft_Core_Device<
                true, true, true, true, false>;
        }
        else if (!need_potential && need_pressure)
        {
            f = Selective_Lennard_Jones_And_Direct_Coulomb_Soft_Core_Device<
                true, false, true, true, false>;
        }
        else
        {
            f = Selective_Lennard_Jones_And_Direct_Coulomb_Soft_Core_Device<
                true, false, false, true, false>;
        }
        Launch_Device_Kernel(
            f, gridSize, blockSize, 0, NULL, local_atom_numbers,
            solvent_numbers, nl, lj_info->d_LJ_energy_atom,
            lj_info->crd_with_LJ_parameters_local, boundary,
            atom_sys_mark_local, lj_info->d_LJ_AA, lj_info->d_LJ_AB,
            lj_info->d_LJ_BA, lj_info->d_LJ_BB, cutoff, md_frc,
            pw_select.select_force[0], pme_beta, atom_energy,
            pw_select.select_atom_energy[0], atom_virial,
            pw_select.select_atom_virial_tensor[0], coulomb_atom_ene, NULL,
            NULL, NULL, lj_info->lambda, lj_info->alpha, lj_info->p,
            lj_info->sigma_6, lj_info->sigma_6_min, pwwp_enhance_factor);
    }
}

void SITS_INFORMATION::Step_Print(CONTROLLER* controller, const float beta0)
{
    if (!is_initialized) return;
    float bias;
    deviceMemcpy(&bias, classic_sits.d_bias, sizeof(float),
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(&h_enhancing_energy, pw_select.select_energy[0], sizeof(float),
                 deviceMemcpyDeviceToHost);
    controller->Step_Print(print_aa_kab_name, h_enhancing_energy);
    controller->Step_Print(print_bias_name, bias);
    controller->Step_Print(print_fb_name, h_factor);
}

static __global__ void Check_Solvent_Atom_Included(int atom_numbers,
                                                   int solvent_numbers,
                                                   int* atom_sys_mark,
                                                   int* errored)
{
#ifdef USE_GPU
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < solvent_numbers)
#else
#pragma omp parallel for
    for (int i = 0; i < solvent_numbers; i++)
#endif
    {
        if (!atom_sys_mark[atom_numbers - solvent_numbers + i]) errored[0] = 1;
    }
}

void SITS_INFORMATION::Check_Solvent(CONTROLLER* controller, int atom_numbers,
                                     int solvent_numbers)
{
    if (!is_initialized || solvent_numbers == 0) return;
    int *errored, h_errored;
    Device_Malloc_Safely((void**)&errored, sizeof(int));
    deviceMemset(errored, 0, sizeof(int));
    Launch_Device_Kernel(Check_Solvent_Atom_Included,
                         (solvent_numbers + CONTROLLER::device_max_thread - 1) /
                             CONTROLLER::device_max_thread,
                         CONTROLLER::device_max_thread, 0, NULL, atom_numbers,
                         solvent_numbers, atom_sys_mark, errored);

    deviceMemcpy(&h_errored, errored, sizeof(int), deviceMemcpyDeviceToHost);
    if (h_errored == 1)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorConflictingCommand, "SITS_INFORMATION::Check_Solvent",
            "Reason:\n\tYou are trying to apply SITS to the solvents. If YOU "
            "KNOW WHAT YOU ARE DOING, set the command 'solvent_LJ' to 0 to run "
            "the simulation.");
    }
    Free_Single_Device_Pointer((void**)&errored);
}

void SELECT::Initial()
{
    select_atom_energy.clear();
    select_energy.clear();
    select_force.clear();
    select_atom_virial.clear();
    select_virial.clear();
}

int SELECT::Add_One_Energy(int atom_numbers)
{
    float* tmp_atom_energy;
    float* tmp_energy;
    Device_Malloc_Safely((void**)&tmp_atom_energy,
                         sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&tmp_energy, sizeof(float));
    select_atom_energy.push_back(tmp_atom_energy);
    select_energy.push_back(tmp_energy);
    return select_energy.size() - 1;
}

int SELECT::Add_One_Force(int atom_numbers)
{
    VECTOR* tmp_force;
    Device_Malloc_Safely((void**)&tmp_force, sizeof(VECTOR) * atom_numbers);
    select_force.push_back(tmp_force);
    return (select_force.size() - 1);
}

int SELECT::Add_One_Virial(int atom_numbers)
{
    float* tmp_atom_virial;
    float* tmp_virial;
    LTMatrix3* tmp_atom_virial_tensor;
    LTMatrix3* tmp_virial_tensor;
    Device_Malloc_Safely((void**)&tmp_atom_virial,
                         sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&tmp_virial, sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&tmp_atom_virial_tensor,
                         sizeof(LTMatrix3) * atom_numbers);
    Device_Malloc_Safely((void**)&tmp_virial_tensor,
                         sizeof(LTMatrix3) * atom_numbers);
    select_atom_virial.push_back(tmp_atom_virial);
    select_virial.push_back(tmp_virial);
    select_atom_virial_tensor.push_back(tmp_atom_virial_tensor);
    select_virial_tensor.push_back(tmp_virial_tensor);
    return select_virial.size() - 1;
}

static __global__ void get_local_device(int* atom_local, int local_atom_numbers,
                                        int ghost_numbers, int* atom_sys_mark,
                                        int* atom_sys_mark_local)
{
    int total = local_atom_numbers + ghost_numbers;
    SIMPLE_DEVICE_FOR(i, total)
    {
        atom_sys_mark_local[i] = atom_sys_mark[atom_local[i]];
    }
}

void SITS_INFORMATION::Get_Local(int* atom_local, int local_atom_numbers_,
                                 int ghost_numbers_)
{
    if (is_initialized)
    {
        local_atom_numbers = local_atom_numbers_;
        ghost_numbers = ghost_numbers_;
        Launch_Device_Kernel(get_local_device,
                             (local_atom_numbers + ghost_numbers +
                              CONTROLLER::device_max_thread - 1) /
                                 CONTROLLER::device_max_thread,
                             CONTROLLER::device_max_thread, 0, NULL, atom_local,
                             local_atom_numbers, ghost_numbers, atom_sys_mark,
                             atom_sys_mark_local);
    }
}

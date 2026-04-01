#include "eeq.h"

#define COULOMB_CONSTANT (332.05221729f)
#ifdef USE_CPU
#define EEQ_SIMPLE_DEVICE_FOR(i, N)                           \
    PRAGMA(omp parallel for schedule(static) if ((N) >= 512)) \
    for (int i = 0; i < N; i++)
#else
#define EEQ_SIMPLE_DEVICE_FOR(i, N) SIMPLE_DEVICE_FOR(i, N)
#endif

#ifndef USE_CPU
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#endif

void REAXFF_EEQ::Initial(CONTROLLER* controller, int atom_numbers,
                         const char* parameter_in_file,
                         const char* type_in_file)
{
    if (parameter_in_file == NULL || type_in_file == NULL)
    {
        controller->printf(
            "REAXFF_EEQ IS NOT INITIALIZED (missing input files)\n\n");
        return;
    }

    this->atom_numbers = atom_numbers;
    controller->printf("START INITIALIZING REAXFF_EEQ\n");

    FILE* fp_p;
    Open_File_Safely(&fp_p, parameter_in_file, "r");
    char line[1024];
    auto throw_bad_format = [&](const char* file_name, const char* reason)
    {
        char error_msg[1024];
        sprintf(error_msg, "Reason:\n\t%s in file %s\n", reason, file_name);
        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                       "REAXFF_EEQ::Initial", error_msg);
    };
    auto read_line_or_throw =
        [&](FILE* file, const char* file_name, const char* stage)
    {
        if (fgets(line, 1024, file) == NULL)
        {
            char reason[512];
            sprintf(reason, "failed to read %s", stage);
            throw_bad_format(file_name, reason);
        }
    };

    read_line_or_throw(fp_p, parameter_in_file, "parameter header line 1");
    read_line_or_throw(fp_p, parameter_in_file, "general parameter count line");
    int n_gen_params = 0;
    if (sscanf(line, "%d", &n_gen_params) != 1)
    {
        throw_bad_format(parameter_in_file,
                         "failed to parse number of general parameters");
    }

    for (int i = 0; i < n_gen_params; i++)
    {
        read_line_or_throw(fp_p, parameter_in_file, "general parameter block");
    }

    read_line_or_throw(fp_p, parameter_in_file, "atom type count line");
    int n_atom_types = 0;
    if (sscanf(line, "%d", &n_atom_types) != 1)
    {
        throw_bad_format(parameter_in_file,
                         "failed to parse number of atom types");
    }
    this->atom_type_numbers = n_atom_types;

    read_line_or_throw(fp_p, parameter_in_file, "atom type header line 1");
    read_line_or_throw(fp_p, parameter_in_file, "atom type header line 2");
    read_line_or_throw(fp_p, parameter_in_file, "atom type header line 3");

    std::map<std::string, int> type_map;
    Malloc_Safely((void**)&h_chi, sizeof(float) * n_atom_types);
    Malloc_Safely((void**)&h_eta, sizeof(float) * n_atom_types);
    Malloc_Safely((void**)&h_gamma, sizeof(float) * n_atom_types);
    Malloc_Safely((void**)&h_shield,
                  sizeof(float) * n_atom_types * n_atom_types);

    for (int i = 0; i < n_atom_types; i++)
    {
        char atom_name[16];
        float dummy;

        read_line_or_throw(fp_p, parameter_in_file,
                           "atom type parameter line 1");
        if (sscanf(line, "%s %f %f %f %f %f %f %f %f", atom_name, &dummy,
                   &dummy, &dummy, &dummy, &dummy, &h_gamma[i], &dummy,
                   &dummy) != 9)
        {
            char reason[512];
            sprintf(reason,
                    "failed to parse atom type block line 1 for type index %d",
                    i + 1);
            throw_bad_format(parameter_in_file, reason);
        }
        type_map[std::string(atom_name)] = i;

        read_line_or_throw(fp_p, parameter_in_file,
                           "atom type parameter line 2");
        if (sscanf(line, "%f %f %f %f %f %f %f", &dummy, &dummy, &dummy, &dummy,
                   &dummy, &h_chi[i], &h_eta[i]) != 7)
        {
            char reason[512];
            sprintf(reason,
                    "failed to parse atom type block line 2 for type index %d",
                    i + 1);
            throw_bad_format(parameter_in_file, reason);
        }

        h_chi[i] *= CONSTANT_EV_TO_KCAL_MOL;
        h_eta[i] *= CONSTANT_EV_TO_KCAL_MOL * 2.0f;

        read_line_or_throw(fp_p, parameter_in_file,
                           "atom type parameter line 3");
        read_line_or_throw(fp_p, parameter_in_file,
                           "atom type parameter line 4");
    }
    fclose(fp_p);

    FILE* fp_t;
    Open_File_Safely(&fp_t, type_in_file, "r");
    int check_atom_numbers = 0;
    read_line_or_throw(fp_t, type_in_file, "atom number line");
    if (sscanf(line, "%d", &check_atom_numbers) != 1)
    {
        throw_bad_format(type_in_file, "failed to parse atom numbers");
    }
    if (check_atom_numbers != atom_numbers)
    {
        char reason[512];
        sprintf(reason, "atom numbers (%d) does not match system (%d)",
                check_atom_numbers, atom_numbers);
        throw_bad_format(type_in_file, reason);
    }

    Malloc_Safely((void**)&h_atom_type, sizeof(int) * atom_numbers);
    for (int i = 0; i < atom_numbers; i++)
    {
        char type_name[16];
        read_line_or_throw(fp_t, type_in_file, "atom type entry line");
        if (sscanf(line, "%s", type_name) != 1)
        {
            char reason[512];
            sprintf(reason, "failed to parse atom type at index %d", i + 1);
            throw_bad_format(type_in_file, reason);
        }
        if (type_map.find(std::string(type_name)) == type_map.end())
        {
            char reason[512];
            sprintf(reason, "atom type %s not found in parameter file %s",
                    type_name, parameter_in_file);
            throw_bad_format(type_in_file, reason);
        }
        h_atom_type[i] = type_map[std::string(type_name)];
    }
    fclose(fp_t);

    for (int i = 0; i < n_atom_types; i++)
    {
        for (int j = 0; j < n_atom_types; j++)
        {
            h_shield[i * n_atom_types + j] =
                powf(h_gamma[i] * h_gamma[j], -1.5f);
        }
    }

    Device_Malloc_And_Copy_Safely((void**)&d_chi, h_chi,
                                  sizeof(float) * n_atom_types);
    Device_Malloc_And_Copy_Safely((void**)&d_eta, h_eta,
                                  sizeof(float) * n_atom_types);
    Device_Malloc_And_Copy_Safely((void**)&d_gamma, h_gamma,
                                  sizeof(float) * n_atom_types);
    Device_Malloc_And_Copy_Safely((void**)&d_shield, h_shield,
                                  sizeof(float) * n_atom_types * n_atom_types);
    Device_Malloc_And_Copy_Safely((void**)&d_atom_type, h_atom_type,
                                  sizeof(int) * atom_numbers);

    Device_Malloc_Safely((void**)&d_b, sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&d_r, sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&d_p, sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&d_Ap, sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&d_q, sizeof(float) * atom_numbers);

    Device_Malloc_Safely((void**)&d_s, sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&d_t, sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&d_z, sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&d_temp_sum, sizeof(float));
    Malloc_Safely((void**)&h_h_numnbrs, sizeof(int) * atom_numbers);
    Malloc_Safely((void**)&h_h_firstnbrs, sizeof(int) * atom_numbers);
    Device_Malloc_Safely((void**)&d_h_numnbrs, sizeof(int) * atom_numbers);
    Device_Malloc_Safely((void**)&d_h_firstnbrs, sizeof(int) * atom_numbers);
    deviceMemset(d_q, 0, sizeof(float) * atom_numbers);
    deviceMemset(d_s, 0, sizeof(float) * atom_numbers);
    deviceMemset(d_t, 0, sizeof(float) * atom_numbers);

    // Device-side CG scalar buffers
    Device_Malloc_Safely((void**)&d_rr_old, sizeof(float));
    Device_Malloc_Safely((void**)&d_rr_new, sizeof(float));
    Device_Malloc_Safely((void**)&d_pAp_buf, sizeof(float));
    Device_Malloc_Safely((void**)&d_cg_alpha, sizeof(float));
    Device_Malloc_Safely((void**)&d_cg_beta, sizeof(float));

    // Charge history for extrapolation
    Device_Malloc_Safely((void**)&d_s_hist,
                         sizeof(float) * HIST_SIZE * atom_numbers);
    Device_Malloc_Safely((void**)&d_t_hist,
                         sizeof(float) * HIST_SIZE * atom_numbers);
    nprev = 0;

    is_initialized = 1;
    controller->Step_Print_Initial("REAXFF_EEQ", "%14.7e");
    controller->printf("END INITIALIZING REAXFF_EEQ\n\n");
}

// =====================================================================
// Shared kernels (CPU + GPU)
// =====================================================================

static __global__ void EEQ_Matrix_Vector_Multiply(
    int atom_numbers, const int* __restrict__ firstnbrs,
    const int* __restrict__ numnbrs, const int* __restrict__ jlist,
    const float* __restrict__ h_val, const int* __restrict__ atom_types,
    const float* __restrict__ eta, const float* __restrict__ p,
    float* __restrict__ Ap)
{
    EEQ_SIMPLE_DEVICE_FOR(i, atom_numbers)
    {
        int type_i = atom_types[i];
        float sum = eta[type_i] * p[i];
        int begin = firstnbrs[i];
        int end = begin + numnbrs[i];
        for (int idx = begin; idx < end; idx++)
        {
            sum += h_val[idx] * p[jlist[idx]];
        }
        Ap[i] = sum;
    }
}

static __global__ void EEQ_Count_H_Matrix_Entries(
    int atom_numbers, const VECTOR* crd, const int* atom_types,
    const float* shield, int atom_type_numbers, const ATOM_GROUP* nl,
    const LTMatrix3 cell, const LTMatrix3 rcell, float cutoff, int* numnbrs)
{
    EEQ_SIMPLE_DEVICE_FOR(i, atom_numbers)
    {
        int count = 0;
        ATOM_GROUP nl_i = nl[i];
        VECTOR ri = crd[i];
        int type_i = atom_types[i];
        for (int j_idx = 0; j_idx < nl_i.atom_numbers; j_idx++)
        {
            int atom_j = nl_i.atom_serial[j_idx];
            int type_j = atom_types[atom_j];
            VECTOR rj = crd[atom_j];
            VECTOR drij = Get_Periodic_Displacement(ri, rj, cell, rcell);
            float r2 = drij.x * drij.x + drij.y * drij.y + drij.z * drij.z;
            float r = sqrtf(r2);
            if (r < cutoff)
            {
                float shield_ij = shield[type_i * atom_type_numbers + type_j];
                if (shield_ij >= 0.0f) count++;
            }
        }
        numnbrs[i] = count;
    }
}

static __global__ void EEQ_Fill_H_Matrix(
    int atom_numbers, const VECTOR* crd, const int* atom_types,
    const float* shield, int atom_type_numbers, const ATOM_GROUP* nl,
    const LTMatrix3 cell, const LTMatrix3 rcell, float cutoff,
    const int* firstnbrs, int* jlist, float* h_val)
{
    EEQ_SIMPLE_DEVICE_FOR(i, atom_numbers)
    {
        ATOM_GROUP nl_i = nl[i];
        VECTOR ri = crd[i];
        int type_i = atom_types[i];
        int write_idx = firstnbrs[i];
        for (int j_idx = 0; j_idx < nl_i.atom_numbers; j_idx++)
        {
            int atom_j = nl_i.atom_serial[j_idx];
            int type_j = atom_types[atom_j];
            VECTOR rj = crd[atom_j];
            VECTOR drij = Get_Periodic_Displacement(ri, rj, cell, rcell);
            float r2 = drij.x * drij.x + drij.y * drij.y + drij.z * drij.z;
            float r = sqrtf(r2);
            if (r < cutoff)
            {
                float x = r / cutoff;
                float x2 = x * x;
                float x4 = x2 * x2;
                float x5 = x4 * x;
                float x6 = x5 * x;
                float x7 = x6 * x;
                float taper =
                    20.0f * x7 - 70.0f * x6 + 84.0f * x5 - 35.0f * x4 + 1.0f;
                float shield_ij = shield[type_i * atom_type_numbers + type_j];
                jlist[write_idx] = atom_j;
                h_val[write_idx] =
                    taper * (COULOMB_CONSTANT / cbrtf(r2 * r + shield_ij));
                write_idx++;
            }
        }
    }
}

static __global__ void Vector_Update_P(int n, float* p, const float* r,
                                       float beta)
{
    EEQ_SIMPLE_DEVICE_FOR(i, n) { p[i] = r[i] + beta * p[i]; }
}

static __global__ void Vector_Update_X_R(int n, float* x, float* r,
                                         const float* p, const float* Ap,
                                         float alpha)
{
    EEQ_SIMPLE_DEVICE_FOR(i, n)
    {
        x[i] += alpha * p[i];
        r[i] -= alpha * Ap[i];
    }
}

static __global__ void Vector_Subtract(int n, float* out, const float* a,
                                       const float* b)
{
    EEQ_SIMPLE_DEVICE_FOR(i, n) { out[i] = a[i] - b[i]; }
}

static __global__ void Vector_Copy(int n, float* dst, const float* src)
{
    EEQ_SIMPLE_DEVICE_FOR(i, n) { dst[i] = src[i]; }
}

static __global__ void Setup_B_Chi(int n, float* b, const int* atom_types,
                                   const float* chi)
{
    EEQ_SIMPLE_DEVICE_FOR(i, n) { b[i] = -chi[atom_types[i]]; }
}

static __global__ void Setup_B_One(int n, float* b)
{
    EEQ_SIMPLE_DEVICE_FOR(i, n) { b[i] = 1.0f; }
}

static __global__ void Vector_Scale_Add(int n, float* q, const float* t,
                                        const float* s, float mu)
{
    EEQ_SIMPLE_DEVICE_FOR(i, n) { q[i] = t[i] + mu * s[i]; }
}

static __global__ void EEQ_Convert_Charge_Unit(int n, float* q_out,
                                               const float* q_in, float scale)
{
    EEQ_SIMPLE_DEVICE_FOR(i, n) { q_out[i] = q_in[i] * scale; }
}

static __global__ void Elementwise_Multiply(int n, float* out, const float* a,
                                            const float* b)
{
    EEQ_SIMPLE_DEVICE_FOR(i, n) { out[i] = a[i] * b[i]; }
}

#ifndef USE_CPU

static __device__ __forceinline__ float EEQ_Warp_Reduce_Sum(float value)
{
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1)
    {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

static __device__ __forceinline__ float EEQ_Block_Reduce_Sum(float value)
{
    __shared__ float warp_sums[32];
    int lane = threadIdx.x & (warpSize - 1);
    int warp_id = threadIdx.x >> 5;
    int warp_count = (blockDim.x + warpSize - 1) / warpSize;

    value = EEQ_Warp_Reduce_Sum(value);
    if (lane == 0) warp_sums[warp_id] = value;
    __syncthreads();

    float block_sum = (threadIdx.x < warp_count) ? warp_sums[lane] : 0.0f;
    if (warp_id == 0)
    {
        block_sum = EEQ_Warp_Reduce_Sum(block_sum);
    }
    return block_sum;
}

static __global__ void Dot_Product_Reduce_Kernel(int n, const float* a,
                                                 const float* __restrict__ b,
                                                 float* out)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    float value = 0.0f;
    if (i < n) value = a[i] * b[i];
    float block_sum = EEQ_Block_Reduce_Sum(value);
    if (threadIdx.x == 0) atomicAdd(out, block_sum);
}

static __global__ void Initialize_Preconditioned_CG_State(
    int n, const float* __restrict__ r, float* __restrict__ z,
    float* __restrict__ p, const float* __restrict__ eta,
    const int* __restrict__ atom_types, float* rz_out)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    float value = 0.0f;
    if (i < n)
    {
        float diag = eta[atom_types[i]];
        float zi = (diag != 0.0f) ? r[i] / diag : r[i];
        z[i] = zi;
        p[i] = zi;
        value = r[i] * zi;
    }
    float block_sum = EEQ_Block_Reduce_Sum(value);
    if (threadIdx.x == 0) atomicAdd(rz_out, block_sum);
}

static __global__ void Update_X_R_Precondition_Dot_Kernel(
    int n, float* __restrict__ x, float* __restrict__ r,
    const float* __restrict__ p, const float* __restrict__ Ap,
    float* __restrict__ z, const float* __restrict__ eta,
    const int* __restrict__ atom_types, const float* __restrict__ d_alpha,
    float* rz_out)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    float value = 0.0f;
    float alpha = *d_alpha;
    if (i < n)
    {
        float ri = r[i] - alpha * Ap[i];
        x[i] += alpha * p[i];
        r[i] = ri;
        float diag = eta[atom_types[i]];
        float zi = (diag != 0.0f) ? ri / diag : ri;
        z[i] = zi;
        value = ri * zi;
    }
    float block_sum = EEQ_Block_Reduce_Sum(value);
    if (threadIdx.x == 0) atomicAdd(rz_out, block_sum);
}

#endif

// Polynomial extrapolation coefficients (oldest-to-newest order)
// nprev=k uses row k-1: Newton forward difference formula
static const float EXTRAP_COEFFS[5][5] = {
    {1.0f, 0.0f, 0.0f, 0.0f, 0.0f},     {-1.0f, 2.0f, 0.0f, 0.0f, 0.0f},
    {1.0f, -3.0f, 3.0f, 0.0f, 0.0f},    {-1.0f, 4.0f, -6.0f, 4.0f, 0.0f},
    {1.0f, -5.0f, 10.0f, -10.0f, 5.0f},
};

static __global__ void Extrapolate_Vector_Kernel(int n, float* out,
                                                 const float* hist, int stride,
                                                 int nprev, float c0, float c1,
                                                 float c2, float c3, float c4)
{
    EEQ_SIMPLE_DEVICE_FOR(i, n)
    {
        float val = c0 * hist[i];
        if (nprev >= 2) val += c1 * hist[stride + i];
        if (nprev >= 3) val += c2 * hist[2 * stride + i];
        if (nprev >= 4) val += c3 * hist[3 * stride + i];
        if (nprev >= 5) val += c4 * hist[4 * stride + i];
        out[i] = val;
    }
}

// Jacobi preconditioner: z[i] = r[i] / eta[type_i]
static __global__ void Jacobi_Precondition(int n, float* z, const float* r,
                                           const float* eta,
                                           const int* atom_types)
{
    EEQ_SIMPLE_DEVICE_FOR(i, n)
    {
        float diag = eta[atom_types[i]];
        z[i] = (diag != 0.0f) ? r[i] / diag : r[i];
    }
}

static __global__ void EEQ_Distribute_Energy_Kernel(
    int n, float* d_energy, const float* d_charge, const int* atom_types,
    const float* d_chi, const float* d_eta, const float* d_Aq)
{
    EEQ_SIMPLE_DEVICE_FOR(i, n)
    {
        int type_i = atom_types[i];
        float qi = d_charge[i];
        float e_pol_i = d_chi[type_i] * qi + 0.5f * d_eta[type_i] * qi * qi;
        float e_ele_i = 0.5f * qi * (d_Aq[i] - d_eta[type_i] * qi);
        float en_i = e_pol_i + e_ele_i;
        atomicAdd(&d_energy[i], en_i);
    }
}

static __global__ void EEQ_Calculate_Force_Kernel(
    int atom_numbers, const VECTOR* crd, const int* atom_types,
    const float* shield, int atom_type_numbers, const float* d_charge,
    VECTOR* frc, const ATOM_GROUP* nl, const LTMatrix3 cell,
    const LTMatrix3 rcell, float cutoff, LTMatrix3* atom_virial)
{
    EEQ_SIMPLE_DEVICE_FOR(i, atom_numbers)
    {
        int type_i = atom_types[i];
        float qi = d_charge[i];
        if (fabsf(qi) >= 1e-10f)
        {
            ATOM_GROUP nl_i = nl[i];
            VECTOR ri = crd[i];

            for (int j_idx = 0; j_idx < nl_i.atom_numbers; j_idx++)
            {
                int atom_j = nl_i.atom_serial[j_idx];
                if (atom_j <= i) continue;

                float qj = d_charge[atom_j];
                if (fabsf(qj) < 1e-10f) continue;

                int type_j = atom_types[atom_j];

                VECTOR rj = crd[atom_j];
                VECTOR drij = Get_Periodic_Displacement(ri, rj, cell, rcell);
                float r2 = drij.x * drij.x + drij.y * drij.y + drij.z * drij.z;
                float r = sqrtf(r2);

                if (r < cutoff)
                {
                    float inv_cutoff = 1.0f / cutoff;
                    float x = r * inv_cutoff;
                    float x2 = x * x;
                    float x4 = x2 * x2;
                    float x5 = x4 * x;
                    float x6 = x5 * x;
                    float x7 = x6 * x;

                    // Taper: T(x) = 20x^7 - 70x^6 + 84x^5 - 35x^4 + 1
                    float taper_val = 20.0f * x7 - 70.0f * x6 + 84.0f * x5 -
                                      35.0f * x4 + 1.0f;
                    // dT/dr = (1/cutoff) * (140x^6 - 420x^5 + 420x^4 - 140x^3)
                    float x3 = x2 * x;
                    float dtaper_dr = inv_cutoff * (140.0f * x6 - 420.0f * x5 +
                                                    420.0f * x4 - 140.0f * x3);

                    float shield_ij =
                        shield[type_i * atom_type_numbers + type_j];
                    float u = r2 * r + shield_ij;  // r^3 + shield
                    float u_cbrt = cbrtf(u);       // u^(1/3)
                    float inv_u_cbrt = 1.0f / u_cbrt;

                    // H = taper * C / u^(1/3)
                    // dH/dr = C * (dtaper/dr / u^(1/3) - taper * r^2 / u^(4/3))
                    float dH_dr =
                        COULOMB_CONSTANT * (dtaper_dr * inv_u_cbrt -
                                            taper_val * r2 * inv_u_cbrt / u);

                    float force_mag = -qi * qj * dH_dr / r;

                    float fx = force_mag * drij.x;
                    float fy = force_mag * drij.y;
                    float fz = force_mag * drij.z;

                    atomicAdd(&frc[i].x, fx);
                    atomicAdd(&frc[i].y, fy);
                    atomicAdd(&frc[i].z, fz);
                    atomicAdd(&frc[atom_j].x, -fx);
                    atomicAdd(&frc[atom_j].y, -fy);
                    atomicAdd(&frc[atom_j].z, -fz);
                    if (atom_virial)
                    {
                        VECTOR fij = {fx, fy, fz};
                        atomicAdd(atom_virial + i,
                                  Get_Virial_From_Force_Dis(fij, drij));
                    }
                }
            }
        }
    }
}

static __global__ void EEQ_Calculate_Epol_Kernel(int n, float* out,
                                                 const int* types,
                                                 const float* chi,
                                                 const float* eta,
                                                 const float* q)
{
    EEQ_SIMPLE_DEVICE_FOR(i, n)
    {
        int t = types[i];
        out[i] = chi[t] * q[i] + 0.5f * eta[t] * q[i] * q[i];
    }
}

static __global__ void EEQ_Calculate_Eele_Kernel(int n, float* out,
                                                 const int* types,
                                                 const float* eta,
                                                 const float* q,
                                                 const float* Aq)
{
    EEQ_SIMPLE_DEVICE_FOR(i, n)
    {
        out[i] = 0.5f * q[i] * (Aq[i] - eta[types[i]] * q[i]);
    }
}

// =====================================================================
// GPU-only kernels: device-side CG scalar operations
// =====================================================================
#ifndef USE_CPU

static __global__ void CG_Compute_Alpha_Kernel(const float* rr_old,
                                               const float* pAp, float* alpha)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        float b = *pAp;
        *alpha = (b != 0.0f) ? *rr_old / b : 0.0f;
    }
}

static __global__ void CG_Compute_Beta_Kernel(float* rr_old,
                                              const float* rr_new, float* beta)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        float old_val = *rr_old;
        float new_val = *rr_new;
        *beta = (old_val != 0.0f) ? new_val / old_val : 0.0f;
        *rr_old = new_val;
    }
}

static __global__ void CG_Update_X_R_Kernel(int n, float* x, float* r,
                                            const float* p, const float* Ap,
                                            const float* d_alpha)
{
    float alpha = *d_alpha;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
    {
        x[i] += alpha * p[i];
        r[i] -= alpha * Ap[i];
    }
}

static __global__ void CG_Update_P_Kernel(int n, float* p, const float* r,
                                          const float* d_beta)
{
    float beta = *d_beta;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
    {
        p[i] = r[i] + beta * p[i];
    }
}

#endif  // !USE_CPU

// =====================================================================
// Calculate_Charges implementation
// =====================================================================

void REAXFF_EEQ::Calculate_Charges(int atom_numbers, float* d_charge,
                                   const VECTOR* d_crd, const LTMatrix3 cell,
                                   const LTMatrix3 rcell,
                                   const ATOM_GROUP* fnl_d_nl, float cutoff,
                                   float* d_energy, VECTOR* frc,
                                   int need_virial, LTMatrix3* atom_virial)
{
    if (!is_initialized || fnl_d_nl == NULL) return;

    dim3 blockSize = {std::min(160u, CONTROLLER::device_max_thread)};
    dim3 gridSize = {(atom_numbers + blockSize.x - 1) / blockSize.x};

    // ---- Build H matrix CSR ----
    Launch_Device_Kernel(EEQ_Count_H_Matrix_Entries, gridSize, blockSize, 0,
                         NULL, atom_numbers, d_crd, d_atom_type, d_shield,
                         atom_type_numbers, fnl_d_nl, cell, rcell, cutoff,
                         d_h_numnbrs);

    int total_nnz = 0;
#ifndef USE_CPU
    {
        thrust::device_ptr<int> d_numnbrs_ptr(d_h_numnbrs);
        thrust::device_ptr<int> d_firstnbrs_ptr(d_h_firstnbrs);
        thrust::exclusive_scan(d_numnbrs_ptr, d_numnbrs_ptr + atom_numbers,
                               d_firstnbrs_ptr);
        total_nnz =
            (int)thrust::reduce(d_numnbrs_ptr, d_numnbrs_ptr + atom_numbers);
    }
#else
    deviceMemcpy(h_h_numnbrs, d_h_numnbrs, sizeof(int) * atom_numbers,
                 deviceMemcpyDeviceToHost);
    for (int i = 0; i < atom_numbers; i++)
    {
        h_h_firstnbrs[i] = total_nnz;
        total_nnz += h_h_numnbrs[i];
    }
    deviceMemcpy(d_h_firstnbrs, h_h_firstnbrs, sizeof(int) * atom_numbers,
                 deviceMemcpyHostToDevice);
#endif

    if (total_nnz > h_matrix_capacity)
    {
        if (d_h_jlist != NULL) deviceFree(d_h_jlist);
        if (d_h_val != NULL) deviceFree(d_h_val);
        h_matrix_capacity = total_nnz;
        if (h_matrix_capacity > 0)
        {
            Device_Malloc_Safely((void**)&d_h_jlist,
                                 sizeof(int) * h_matrix_capacity);
            Device_Malloc_Safely((void**)&d_h_val,
                                 sizeof(float) * h_matrix_capacity);
        }
    }
    if (total_nnz > 0)
    {
        Launch_Device_Kernel(EEQ_Fill_H_Matrix, gridSize, blockSize, 0, NULL,
                             atom_numbers, d_crd, d_atom_type, d_shield,
                             atom_type_numbers, fnl_d_nl, cell, rcell, cutoff,
                             d_h_firstnbrs, d_h_jlist, d_h_val);
    }

    // ---- CG solver ----
#ifndef USE_CPU
    // GPU path: Jacobi-preconditioned CG, device-side scalars
    auto solve = [&](float* x, float* b_in, bool warm)
    {
        if (!warm)
        {
            deviceMemset(x, 0, sizeof(float) * atom_numbers);
            Launch_Device_Kernel(Vector_Copy, gridSize, blockSize, 0, NULL,
                                 atom_numbers, d_r, b_in);
        }
        else
        {
            Launch_Device_Kernel(EEQ_Matrix_Vector_Multiply, gridSize,
                                 blockSize, 0, NULL, atom_numbers,
                                 d_h_firstnbrs, d_h_numnbrs, d_h_jlist, d_h_val,
                                 d_atom_type, d_eta, x, d_Ap);
            Launch_Device_Kernel(Vector_Subtract, gridSize, blockSize, 0, NULL,
                                 atom_numbers, d_r, b_in, d_Ap);
        }

        deviceMemset(d_rr_old, 0, sizeof(float));
        Initialize_Preconditioned_CG_State<<<gridSize, blockSize>>>(
            atom_numbers, d_r, d_z, d_p, d_eta, d_atom_type, d_rr_old);

        const int check_interval = 5;
        float h_rz = 0;

        for (int iter = 0; iter < max_iter; iter++)
        {
            // Check convergence every check_interval iterations
            if (iter % check_interval == 0)
            {
                deviceMemcpy(&h_rz, d_rr_old, sizeof(float),
                             deviceMemcpyDeviceToHost);
                if (fabsf(h_rz) < tolerance * tolerance) break;
            }

            Launch_Device_Kernel(EEQ_Matrix_Vector_Multiply, gridSize,
                                 blockSize, 0, NULL, atom_numbers,
                                 d_h_firstnbrs, d_h_numnbrs, d_h_jlist, d_h_val,
                                 d_atom_type, d_eta, d_p, d_Ap);

            deviceMemset(d_pAp_buf, 0, sizeof(float));
            Dot_Product_Reduce_Kernel<<<gridSize, blockSize>>>(
                atom_numbers, d_p, d_Ap, d_pAp_buf);

            // alpha = rz_old / pAp (on device)
            CG_Compute_Alpha_Kernel<<<1, 1>>>(d_rr_old, d_pAp_buf, d_cg_alpha);

            deviceMemset(d_rr_new, 0, sizeof(float));
            Update_X_R_Precondition_Dot_Kernel<<<gridSize, blockSize>>>(
                atom_numbers, x, d_r, d_p, d_Ap, d_z, d_eta, d_atom_type,
                d_cg_alpha, d_rr_new);

            // beta = rz_new/rz_old, rz_old = rz_new (on device)
            CG_Compute_Beta_Kernel<<<1, 1>>>(d_rr_old, d_rr_new, d_cg_beta);

            // p = z + beta*p
            CG_Update_P_Kernel<<<gridSize, blockSize>>>(atom_numbers, d_p, d_z,
                                                        d_cg_beta);
        }
    };
#else
    // CPU path: Jacobi-preconditioned CG with host-side scalars
    auto solve = [&](float* x, float* b_in, bool warm)
    {
        if (!warm)
        {
            deviceMemset(x, 0, sizeof(float) * atom_numbers);
            deviceMemcpy(d_r, b_in, sizeof(float) * atom_numbers,
                         deviceMemcpyDeviceToDevice);
        }
        else
        {
            Launch_Device_Kernel(EEQ_Matrix_Vector_Multiply, gridSize,
                                 blockSize, 0, NULL, atom_numbers,
                                 d_h_firstnbrs, d_h_numnbrs, d_h_jlist, d_h_val,
                                 d_atom_type, d_eta, x, d_Ap);
            Launch_Device_Kernel(Vector_Subtract, gridSize, blockSize, 0, NULL,
                                 atom_numbers, d_r, b_in, d_Ap);
        }

        // z = M^{-1} * r
        Launch_Device_Kernel(Jacobi_Precondition, gridSize, blockSize, 0, NULL,
                             atom_numbers, d_z, d_r, d_eta, d_atom_type);

        deviceMemcpy(d_p, d_z, sizeof(float) * atom_numbers,
                     deviceMemcpyDeviceToDevice);

        float rz_old = 0, rz_new = 0;
        Launch_Device_Kernel(Elementwise_Multiply, gridSize, blockSize, 0, NULL,
                             atom_numbers, d_q, d_r, d_z);
        Sum_Of_List(d_q, d_temp_sum, atom_numbers);
        deviceMemcpy(&rz_old, d_temp_sum, sizeof(float),
                     deviceMemcpyDeviceToHost);

        for (int iter = 0; iter < max_iter; iter++)
        {
            if (fabsf(rz_old) < tolerance * tolerance) break;

            Launch_Device_Kernel(EEQ_Matrix_Vector_Multiply, gridSize,
                                 blockSize, 0, NULL, atom_numbers,
                                 d_h_firstnbrs, d_h_numnbrs, d_h_jlist, d_h_val,
                                 d_atom_type, d_eta, d_p, d_Ap);

            float p_dot_Ap = 0;
            Launch_Device_Kernel(Elementwise_Multiply, gridSize, blockSize, 0,
                                 NULL, atom_numbers, d_q, d_p, d_Ap);
            Sum_Of_List(d_q, d_temp_sum, atom_numbers);
            deviceMemcpy(&p_dot_Ap, d_temp_sum, sizeof(float),
                         deviceMemcpyDeviceToHost);

            float alpha = rz_old / p_dot_Ap;
            Launch_Device_Kernel(Vector_Update_X_R, gridSize, blockSize, 0,
                                 NULL, atom_numbers, x, d_r, d_p, d_Ap, alpha);

            // z = M^{-1} * r
            Launch_Device_Kernel(Jacobi_Precondition, gridSize, blockSize, 0,
                                 NULL, atom_numbers, d_z, d_r, d_eta,
                                 d_atom_type);

            Launch_Device_Kernel(Elementwise_Multiply, gridSize, blockSize, 0,
                                 NULL, atom_numbers, d_q, d_r, d_z);
            Sum_Of_List(d_q, d_temp_sum, atom_numbers);
            deviceMemcpy(&rz_new, d_temp_sum, sizeof(float),
                         deviceMemcpyDeviceToHost);

            if (fabsf(rz_new) < tolerance * tolerance) break;

            float beta = rz_new / rz_old;
            Launch_Device_Kernel(Vector_Update_P, gridSize, blockSize, 0, NULL,
                                 atom_numbers, d_p, d_z, beta);
            rz_old = rz_new;
        }
    };
#endif

    Launch_Device_Kernel(Setup_B_Chi, gridSize, blockSize, 0, NULL,
                         atom_numbers, d_b, d_atom_type, d_chi);
    bool warm = nprev > 0;
    if (warm)
    {
        const float* c = EXTRAP_COEFFS[nprev - 1];
        Launch_Device_Kernel(Extrapolate_Vector_Kernel, gridSize, blockSize, 0,
                             NULL, atom_numbers, d_t, d_t_hist, atom_numbers,
                             nprev, c[0], c[1], c[2], c[3], c[4]);
    }
    solve(d_t, d_b, warm);

    Launch_Device_Kernel(Setup_B_One, gridSize, blockSize, 0, NULL,
                         atom_numbers, d_b);
    if (warm)
    {
        const float* c = EXTRAP_COEFFS[nprev - 1];
        Launch_Device_Kernel(Extrapolate_Vector_Kernel, gridSize, blockSize, 0,
                             NULL, atom_numbers, d_s, d_s_hist, atom_numbers,
                             nprev, c[0], c[1], c[2], c[3], c[4]);
    }
    solve(d_s, d_b, warm);

    // Update extrapolation history
    if (nprev < HIST_SIZE)
    {
        deviceMemcpy(d_t_hist + nprev * atom_numbers, d_t,
                     sizeof(float) * atom_numbers, deviceMemcpyDeviceToDevice);
        deviceMemcpy(d_s_hist + nprev * atom_numbers, d_s,
                     sizeof(float) * atom_numbers, deviceMemcpyDeviceToDevice);
        nprev++;
    }
    else
    {
        for (int k = 0; k < HIST_SIZE - 1; k++)
        {
            deviceMemcpy(
                d_t_hist + k * atom_numbers, d_t_hist + (k + 1) * atom_numbers,
                sizeof(float) * atom_numbers, deviceMemcpyDeviceToDevice);
            deviceMemcpy(
                d_s_hist + k * atom_numbers, d_s_hist + (k + 1) * atom_numbers,
                sizeof(float) * atom_numbers, deviceMemcpyDeviceToDevice);
        }
        deviceMemcpy(d_t_hist + (HIST_SIZE - 1) * atom_numbers, d_t,
                     sizeof(float) * atom_numbers, deviceMemcpyDeviceToDevice);
        deviceMemcpy(d_s_hist + (HIST_SIZE - 1) * atom_numbers, d_s,
                     sizeof(float) * atom_numbers, deviceMemcpyDeviceToDevice);
    }

    float sum_t = 0, sum_s = 0;
    Sum_Of_List(d_t, d_temp_sum, atom_numbers);
    deviceMemcpy(&sum_t, d_temp_sum, sizeof(float), deviceMemcpyDeviceToHost);
    Sum_Of_List(d_s, d_temp_sum, atom_numbers);
    deviceMemcpy(&sum_s, d_temp_sum, sizeof(float), deviceMemcpyDeviceToHost);

    float Qtot = 0;
    float mu = (Qtot - sum_t) / sum_s;

    Launch_Device_Kernel(Vector_Scale_Add, gridSize, blockSize, 0, NULL,
                         atom_numbers, d_q, d_t, d_s, mu);

    Launch_Device_Kernel(EEQ_Matrix_Vector_Multiply, gridSize, blockSize, 0,
                         NULL, atom_numbers, d_h_firstnbrs, d_h_numnbrs,
                         d_h_jlist, d_h_val, d_atom_type, d_eta, d_q, d_Ap);

    Launch_Device_Kernel(EEQ_Calculate_Epol_Kernel, gridSize, blockSize, 0,
                         NULL, atom_numbers, d_r, d_atom_type, d_chi, d_eta,
                         d_q);

    float sum_epol = 0;
    Sum_Of_List(d_r, d_temp_sum, atom_numbers);
    deviceMemcpy(&sum_epol, d_temp_sum, sizeof(float),
                 deviceMemcpyDeviceToHost);

    Launch_Device_Kernel(EEQ_Calculate_Eele_Kernel, gridSize, blockSize, 0,
                         NULL, atom_numbers, d_r, d_atom_type, d_eta, d_q,
                         d_Ap);

    float sum_eele = 0;
    Sum_Of_List(d_r, d_temp_sum, atom_numbers);
    deviceMemcpy(&sum_eele, d_temp_sum, sizeof(float),
                 deviceMemcpyDeviceToHost);

    h_energy = sum_epol + sum_eele;
    if (d_energy != NULL)
    {
        Launch_Device_Kernel(EEQ_Distribute_Energy_Kernel, gridSize, blockSize,
                             0, NULL, atom_numbers, d_energy, d_q, d_atom_type,
                             d_chi, d_eta, d_Ap);
    }

    if (frc != NULL)
    {
        Launch_Device_Kernel(EEQ_Calculate_Force_Kernel, gridSize, blockSize, 0,
                             NULL, atom_numbers, d_crd, d_atom_type, d_shield,
                             atom_type_numbers, d_q, frc, fnl_d_nl, cell, rcell,
                             cutoff, need_virial ? atom_virial : NULL);
    }

    Launch_Device_Kernel(EEQ_Convert_Charge_Unit, gridSize, blockSize, 0, NULL,
                         atom_numbers, d_charge, d_q,
                         CONSTANT_SPONGE_CHARGE_SCALE);
}

void REAXFF_EEQ::Step_Print(CONTROLLER* controller)

{
    if (!is_initialized) return;
    controller->Step_Print("REAXFF_EEQ", h_energy, true);
}

void REAXFF_EEQ::Print_Charges(const float* d_charge)
{
    if (!is_initialized) return;
    float* h_q = NULL;
    Malloc_Safely((void**)&h_q, sizeof(float) * atom_numbers);
    deviceMemcpy(h_q, d_charge, sizeof(float) * atom_numbers,
                 deviceMemcpyDeviceToHost);

    FILE* fp = fopen("eeq_charges.txt", "w");
    if (fp)
    {
        for (int i = 0; i < atom_numbers; i++)
        {
            float q_elementary = h_q[i] / CONSTANT_SPONGE_CHARGE_SCALE;
            fprintf(fp, "%d %.6f\n", i + 1, q_elementary);
        }
        fclose(fp);
    }
    free(h_q);
}

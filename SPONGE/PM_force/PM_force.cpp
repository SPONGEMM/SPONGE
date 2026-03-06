#include "PM_force.h"
/*
    2025-10-14 SPONGE Particle Mesh算法
    目前支持单进程Particle-Mesh-Ewald 与 PMC-IZ
   计算，在头文件预留了MPI-FFT多进程接口
    单进程下，PM进程与PP进程设置为同一个进程；多进程下PM独享一个进程

    未来可能的改进：使用PSWF作为分裂核与插值核，改进Particle
   Mesh算法，需要修改：
    - 插值核 与 修正系数计算。可以引用NUFFT相关代码
    - 近程项、长程修正项等
*/

// constants
#define PI 3.1415926f
#define INVSQRTPI 0.56418958835977f
#define TWO_DIVIDED_BY_SQRT_PI 1.1283791670218446f
static __device__ float PME_Ma[4] = {1.0 / 6.0, -0.5, 0.5, -1.0 / 6.0};
static __device__ float PME_Mb[4] = {0, 0.5, -1, 0.5};
static __device__ float PME_Mc[4] = {0, 0.5, 0, -0.5};
static __device__ float PME_Md[4] = {0, 1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0};
static __device__ float PME_dMa[4] = {0.5, -1.5, 1.5, -0.5};
static __device__ float PME_dMb[4] = {0, 1, -2, 1};
static __device__ float PME_dMc[4] = {0, 0.5, 0, -0.5};

// 计算B样条插值的递归函数 (compact-window function, or SI kernel)
static float M_(float u, int n)
{
    if (n == 2)
    {
        if (u > 2 || u < 0) return 0;
        return 1 - abs(u - 1);
    }
    else
        return u / (n - 1) * M_(u, n - 1) +
               (n - u) / (n - 1) * M_(u - 1, n - 1);
}

// 修正B样条插值的递归函数 (influence function)
static float getb(int k, int NFFT, int B_order)
{
    FFT_COMPLEX tempc, tempc2, res;
    float tempf;
    REAL(tempc2) = 0;
    IMAGINARY(tempc2) = 0;

    REAL(tempc) = 0;
    IMAGINARY(tempc) = 2 * (B_order - 1) * PI * k / NFFT;
    res = expc(tempc);

    for (int kk = 0; kk < (B_order - 1); kk++)
    {
        REAL(tempc) = 0;
        IMAGINARY(tempc) = 2.0f * PI * k / NFFT * kk;
        tempc = expc(tempc);
        tempf = M_(kk + 1, B_order);
        REAL(tempc2) += tempf * REAL(tempc);
        IMAGINARY(tempc2) += tempf * IMAGINARY(tempc);
    }
    res = divc(res, tempc2);
    return REAL(res) * REAL(res) + IMAGINARY(res) * IMAGINARY(res);
}

// PMC_IZ Method
static __global__ void Build_PMC_IZ_C(const int PME_Nfft, int fftx, int ffty,
                                      int fftz,
                                      float box_length_inverse_x_square,
                                      float box_length_inverse_y_square,
                                      float grid_length_of_z, float beta,
                                      float scalor, FFT_COMPLEX* C)
{
    SIMPLE_DEVICE_FOR(tid, PME_Nfft)
    {
        int ffta = (fftx / 2 + 1);
        int grid_x = tid % ffta;
        int grid_y = (tid % (ffta * ffty)) / ffta;
        int grid_z = tid / ffty / ffta;
        if (grid_x >= fftx / 2)
        {
            grid_x = fftx - grid_x;
        }
        if (grid_y >= ffty / 2)
        {
            grid_y = ffty - grid_y;
        }
        if (grid_z >= fftz / 2)
        {
            grid_z = fftz - grid_z;
        }
        float z = grid_length_of_z * grid_z;
        float A = 2.0f * CONSTANT_Pi *
                  sqrtf(grid_x * grid_x * box_length_inverse_x_square +
                        grid_y * grid_y * box_length_inverse_y_square);
        float AB = A / beta / 2.0f;
        float zb2 = z * beta;
        float AB_minus_zb2 = AB - zb2;
        float AB_plus_zb2 = AB + zb2;
        float temp_f =
            expf(-A * z) * (erfcf(AB_minus_zb2) +
                            expf(2.0f * A * z - AB_plus_zb2 * AB_plus_zb2) *
                                erfcxf(AB_plus_zb2));
        temp_f = temp_f / A;
        if (grid_x == 0 && grid_y == 0)
        {
            temp_f =
                2.0f / sqrtf(CONSTANT_Pi) / beta * (1.0f - expf(-zb2 * zb2)) -
                2.0f * z * erff(zb2);
        }
        REAL(C[tid]) = scalor * temp_f;
        IMAGINARY(C[tid]) = 0.;
    }
}

static __global__ void Build_PMC_IZ_BC_Final(const int Nfft, int fftx, int ffty,
                                             int fftz, const FFT_COMPLEX* C,
                                             const FFT_COMPLEX* B, float* BC)
{
    SIMPLE_DEVICE_FOR(tid, Nfft)
    {
        int fftc = fftz / 2 + 1;
        int ffta = fftx / 2 + 1;
        int zi = tid % fftc;
        int yi = (tid / fftc) % ffty;
        int xi = tid / fftc / ffty;
        if (xi >= fftx / 2)
        {
            xi = fftx - xi;
        }
        int ti = zi * ffta * ffty + yi * ffta + xi;
        float b = REAL(B[ti]);
        BC[tid] = REAL(C[ti]) / (b * b);
    }
}

static void Build_PMC_IZ_BC(CONTROLLER* controller, int fftx, int ffty,
                            int fftz, int PME_Nfft, int PME_Nall, int PME_Nin,
                            float box_length_inverse_x_square,
                            float box_length_inverse_y_square,
                            float grid_length_of_z, float beta, float scalor,
                            float** BC)
{
    Device_Malloc_Safely((void**)BC, sizeof(float) * PME_Nfft);
    FFT_SIZE_t n2d[2] = {ffty, fftx};
    FFT_RESULT result;
    FFT_HANDLE plan_2d_many_c2r, plan_3d_temp_r2c;

    result = SPONGE_FFT_WRAPPER::Make_FFT_Plan(&plan_2d_many_c2r, fftz, 2, n2d,
                                               FFT_C2R);
    if (result != FFT_SUCCESS)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorMallocFailed, "Build_PMC_IZ_Efficient_Potential",
            "Reason:\n\tFail to create the batched 2D FFT plan");
    }
    FFT_SIZE_t n3d[3] = {fftz, ffty, fftx};

    result = SPONGE_FFT_WRAPPER::Make_FFT_Plan(&plan_3d_temp_r2c, 1, 3, n3d,
                                               FFT_R2C);
    if (result != FFT_SUCCESS)
    {
        controller->Throw_SPONGE_Error(spongeErrorMallocFailed,
                                       "Build_PMC_IZ_Efficient_Potential",
                                       "Reason:\n\tFail to create the "
                                       "temporary 3D Real to Complex FFT plan");
    }
    FFT_COMPLEX *B, *C;
    float *d_FB, *h_FB, *FC;
    int temp_Nfft = (fftx / 2 + 1) * ffty * fftz;
    Device_Malloc_Safely((void**)&B, sizeof(FFT_COMPLEX) * temp_Nfft);
    Device_Malloc_Safely((void**)&C, sizeof(FFT_COMPLEX) * temp_Nfft);
    Device_Malloc_Safely((void**)&FC, sizeof(float) * PME_Nall);
    Malloc_Safely((void**)&h_FB, sizeof(float) * PME_Nall);

    for (int i = 0; i < PME_Nall; i = i + 1)
    {
        h_FB[i] = 0.;
    }
    float temp_b_spline[3] = {1. / 6., 2. / 3., 1. / 6.};
    for (int k = -1; k <= 1; k = k + 1)
    {
        for (int j = -1; j <= 1; j = j + 1)
        {
            for (int i = -1; i <= 1; i = i + 1)
            {
                float weight = temp_b_spline[k + 1] * temp_b_spline[j + 1] *
                               temp_b_spline[i + 1];
                int kk, jj, ii;
                if (k < 0)
                {
                    kk = k + fftz;
                }
                else
                {
                    kk = k;
                }
                if (j < 0)
                {
                    jj = j + ffty;
                }
                else
                {
                    jj = j;
                }
                if (i < 0)
                {
                    ii = i + fftx;
                }
                else
                {
                    ii = i;
                }
                h_FB[ii + jj * fftx + kk * fftx * ffty] = weight;
            }
        }
    }
    Device_Malloc_And_Copy_Safely((void**)&d_FB, h_FB,
                                  sizeof(float) * PME_Nall);
    SPONGE_FFT_WRAPPER::R2C(plan_3d_temp_r2c, d_FB, B);
    Launch_Device_Kernel(Build_PMC_IZ_C,
                         (temp_Nfft + CONTROLLER::device_max_thread - 1) /
                             CONTROLLER::device_max_thread,
                         CONTROLLER::device_max_thread, 0, NULL, temp_Nfft,
                         fftx, ffty, fftz, box_length_inverse_x_square,
                         box_length_inverse_y_square, grid_length_of_z, beta,
                         scalor, C);

    SPONGE_FFT_WRAPPER::C2R(plan_2d_many_c2r, C, FC);
    SPONGE_FFT_WRAPPER::R2C(plan_3d_temp_r2c, FC, C);
    Launch_Device_Kernel(Build_PMC_IZ_BC_Final,
                         (PME_Nfft + CONTROLLER::device_max_thread - 1) /
                             CONTROLLER::device_max_thread,
                         CONTROLLER::device_max_thread, 0, NULL, PME_Nfft, fftx,
                         ffty, fftz, C, B, BC[0]);

    Free_Single_Device_Pointer((void**)&FC);
    Free_Single_Device_Pointer((void**)&C);
    Free_Single_Device_Pointer((void**)&B);
    Free_Host_And_Device_Pointer((void**)&h_FB, (void**)&d_FB);
    SPONGE_FFT_WRAPPER::Destroy_FFT_Plan(&plan_2d_many_c2r);
    SPONGE_FFT_WRAPPER::Destroy_FFT_Plan(&plan_3d_temp_r2c);
}

// 根据截断距离和精度要求计算Ewald屏蔽参数beta
static float Get_Beta(float cutoff, float tolerance)
{
    float beta, low, high, tempf;
    int ilow, ihigh;

    high = 1.0;
    ihigh = 1;

    while (1)
    {
        tempf = erfc(high * cutoff) / cutoff;
        if (tempf <= tolerance) break;
        high *= 2;
        ihigh++;
    }

    ihigh += 50;
    low = 0.0;
    for (ilow = 1; ilow < ihigh; ilow++)
    {
        beta = (low + high) / 2;
        tempf = erfc(beta * cutoff) / cutoff;
        if (tempf >= tolerance)
            low = beta;
        else
            high = beta;
    }
    return beta;
}

// ene += factor * charge_sum^2
static __global__ void device_add(float* ene, float factor, float* charge_sum)
{
    ene[0] += factor * charge_sum[0] * charge_sum[0];
}

static __global__ void charge_square_kernel(int element_number,
                                            const float* charge,
                                            float* charge_square)
{
    SIMPLE_DEVICE_FOR(i, element_number)
    {
        float q = charge[i];
        charge_square[i] = q * q;
    }
}

//--------Particle Mesh Ewald Method----------

void Particle_Mesh::Initial(CONTROLLER* controller, int atom_numbers,
                            LTMatrix3 cell, LTMatrix3 rcell, VECTOR box_length,
                            float cutoff,
                            int no_direct_interaction_virtual_atom_numbers,
                            const char* module_name)
{
    if (module_name == NULL)
    {
        strcpy(this->module_name, "PM");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }

    controller->printf("START INITIALIZING PME:\n");
    this->cutoff = cutoff;

    controller->printf("    PME backend library: %s\n", FFT_LIBRARY_NAME);

    tolerance = 0.00001;
    if (controller->Command_Exist(this->module_name, "Direct_Tolerance"))
    {
        controller->Check_Float(this->module_name, "Direct_Tolerance",
                                "Particle_Mesh::Initial");
        tolerance =
            atof(controller->Command(this->module_name, "Direct_Tolerance"));
    }

    if (CONTROLLER::PP_MPI_size == 1)
    {
        exclude_factor = 1.0f;
    }
    else
    {
        exclude_factor = 0.5f;
    }

    fftx = -1;
    ffty = -1;
    fftz = -1;
    if (controller->Command_Exist(this->module_name, "fftx"))
    {
        controller->Check_Int(this->module_name, "fftx",
                              "Particle_Mesh::Initial");
        fftx = atoi(controller->Command(this->module_name, "fftx"));
    }
    if (controller->Command_Exist(this->module_name, "ffty"))
    {
        controller->Check_Int(this->module_name, "ffty",
                              "Particle_Mesh::Initial");
        ffty = atoi(controller->Command(this->module_name, "ffty"));
    }
    if (controller->Command_Exist(this->module_name, "fftz"))
    {
        controller->Check_Int(this->module_name, "fftz",
                              "Particle_Mesh::Initial");
        fftz = atoi(controller->Command(this->module_name, "fftz"));
    }

    PM_MPI_size = 0;
    if (controller->Command_Exist(this->module_name, "MPI_size"))
    {
        controller->Check_Int(this->module_name, "MPI_size",
                              "Particle_Mesh::Initial");
        PM_MPI_size = atoi(controller->Command(this->module_name, "MPI_size"));
    }
    else
    {
        PM_MPI_size = controller->PM_MPI_size;
    }
    if (!PM_MPI_size)
    {
        controller->printf("PM RECI NOT INITIALIZED");
    }
    // 2025-10-14: temporary disable multi-process PME
    if (PM_MPI_size > 1)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorValueErrorCommand, "Particle_Mesh::Initial",
            "Reason:\n\t Multi-process PME is not supported yet.");
    }

    this->atom_numbers = atom_numbers;
    this->max_atom_numbers = atom_numbers;
    Device_Malloc_Safely((void**)&num_ghost_dir_id,
                         sizeof(int) * max_atom_numbers * 6);

    float volume = cell.a11 * cell.a22 * cell.a33;

    float grid_spacing = 1;
    if (controller->Command_Exist(this->module_name, "grid_spacing"))
    {
        controller->Check_Float(this->module_name, "grid_spacing",
                                "Particle_Mesh::Initial");
        grid_spacing =
            atof(controller->Command(this->module_name, "grid_spacing"));
    }
    controller->printf("    grid_spacing: %f Angstrom\n", grid_spacing);
    if (fftx < 0) fftx = Get_Fft_Patameter(box_length.x / grid_spacing);

    if (ffty < 0) ffty = Get_Fft_Patameter(box_length.y / grid_spacing);

    if (fftz < 0) fftz = Get_Fft_Patameter(box_length.z / grid_spacing);

    controller->printf("    fftx: %d\n", fftx);
    controller->printf("    ffty: %d\n", ffty);
    controller->printf("    fftz: %d\n", fftz);

    PME_Nall = fftx * ffty * fftz;
    PME_Nin = ffty * fftz;
    PME_Nfft = fftx * ffty * (fftz / 2 + 1);

    beta = Get_Beta(cutoff, tolerance);
    controller->printf("    beta: %f\n", beta);

    neutralizing_factor = -0.5 * CONSTANT_Pi / (beta * beta * volume);
    Device_Malloc_Safely((void**)&charge_sum, sizeof(float));
    Device_Malloc_Safely((void**)&charge_square,
                         sizeof(float) * atom_numbers);

    int i, kx, ky, kz, index;
    FFT_RESULT errP1, errP2;
    update_interval = 1;
    if (controller->Command_Exist("PME", "update_interval"))
    {
        controller->Check_Int("PME", "update_interval",
                              "Particle_Mesh::Initial");
        update_interval = atoi(controller->Command("PME", "update_interval"));
    }
    Device_Malloc_Safely((void**)&force_backup, sizeof(VECTOR) * atom_numbers);
    deviceMemset(force_backup, 0, sizeof(VECTOR) * atom_numbers);
    Device_Malloc_Safely((void**)&PME_uxyz,
                         sizeof(UNSIGNED_INT_VECTOR) * atom_numbers);
    Device_Malloc_Safely((void**)&PME_frxyz, sizeof(VECTOR) * atom_numbers);
    Reset_List((int*)PME_uxyz, 1 << 30, 3 * atom_numbers);

    Device_Malloc_Safely((void**)&PME_Q, sizeof(float) * PME_Nall);
    Device_Malloc_Safely((void**)&PME_FQ, sizeof(FFT_COMPLEX) * PME_Nfft);
    Device_Malloc_Safely((void**)&PME_FBCFQ, sizeof(float) * PME_Nall);

    Device_Malloc_Safely((void**)&PME_atom_near,
                         sizeof(int) * 64 * atom_numbers);
    deviceMemset(PME_atom_near, 0, sizeof(int) * 64 * atom_numbers);

    FFT_SIZE_t n3d[3] = {fftx, ffty, fftz};
    errP1 =
        SPONGE_FFT_WRAPPER::Make_FFT_Plan(&PME_plan_r2c, 1, 3, n3d, FFT_R2C);
    errP2 =
        SPONGE_FFT_WRAPPER::Make_FFT_Plan(&PME_plan_c2r, 1, 3, n3d, FFT_C2R);
    if (errP1 != FFT_SUCCESS || errP2 != FFT_SUCCESS)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorValueErrorCommand, "Particle_Mesh::Initial",
            "Reason:\n\tError occurs when create fft plan of PME");
    }

    Device_Malloc_And_Copy_Safely((void**)&d_reciprocal_ene, &reciprocal_ene,
                                  sizeof(float));
    Device_Malloc_And_Copy_Safely((void**)&d_self_ene, &self_ene,
                                  sizeof(float));
    Device_Malloc_And_Copy_Safely((void**)&d_direct_ene, &direct_ene,
                                  sizeof(float));
    Device_Malloc_And_Copy_Safely((void**)&d_correction_ene, &correction_ene,
                                  sizeof(float));
    Device_Malloc_And_Copy_Safely((void**)&d_ee_ene, &ee_ene, sizeof(float));
    Device_Malloc_Safely((void**)&d_direct_atom_energy,
                         sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&d_correction_atom_energy,
                         sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&atom_id_l_g, sizeof(int) * atom_numbers);
    Device_Malloc_Safely((void**)&atom_id_g_l, sizeof(int) * atom_numbers);
    Device_Malloc_Safely(
        (void**)&g_crd,
        sizeof(VECTOR) *
            (atom_numbers + no_direct_interaction_virtual_atom_numbers));
    Device_Malloc_Safely(
        (void**)&g_frc,
        sizeof(VECTOR) *
            (atom_numbers + no_direct_interaction_virtual_atom_numbers));
    deviceMemset(atom_id_l_g, 0, sizeof(int) * atom_numbers);
    deviceMemset(atom_id_g_l, 0, sizeof(int) * atom_numbers);
    deviceMemset(g_crd, 0,
                 sizeof(VECTOR) * (atom_numbers +
                                   no_direct_interaction_virtual_atom_numbers));
    deviceMemset(g_frc, 0,
                 sizeof(VECTOR) * (atom_numbers +
                                   no_direct_interaction_virtual_atom_numbers));
    deviceMemset(d_direct_atom_energy, 0, sizeof(float) * atom_numbers);
    deviceMemset(d_correction_atom_energy, 0, sizeof(float) * atom_numbers);

    calculate_reciprocal_part = true;
    if (controller->Command_Exist("PME", "calculate_reciprocal_part"))
    {
        calculate_reciprocal_part = controller->Get_Bool(
            "PME", "calculate_reciprocal_part", "Particle_Mesh::Initial");
    }
    calculate_excluded_part = true;
    if (controller->Command_Exist("PME", "calculate_excluded_part"))
    {
        calculate_excluded_part = controller->Get_Bool(
            "PME", "calculate_excluded_part", "Particle_Mesh::Initial");
    }
    bool use_pmc_iz = false;
    if (controller->Command_Exist("PME", "replaced_by_PMC_IZ"))
    {
        use_pmc_iz = controller->Get_Bool("PME", "replaced_by_PMC_IZ",
                                          "Particle_Mesh::Initial");
    }

    // 计算B-Spline修正系数 * 泊松算子因子， 用于倒空间乘法
    if (calculate_reciprocal_part)
    {
        if (use_pmc_iz)
        {
            controller->printf("    PMC-IZ will be used instead of PME\n");
            if (controller->Command_Choice("mode", "npt"))
            {
                controller->Throw_SPONGE_Error(
                    spongeErrorConflictingCommand, "Particle_Mesh::Initial",
                    "Reason:\n\tPMC-IZ can not be used in NPT mode");
            }
            Build_PMC_IZ_BC(
                controller, fftx, ffty, fftz, PME_Nfft, PME_Nall, PME_Nin,
                1.0f / box_length.x / box_length.x,
                1.0f / box_length.y / box_length.y, box_length.z / fftz, beta,
                CONSTANT_Pi / PME_Nall / box_length.x / box_length.y, &PME_BC);
        }
        else
        {
            float *B1 = NULL, *B2 = NULL, *B3 = NULL, *h_PME_BC = NULL,
                  *h_PME_BC0 = NULL;
            LTMatrix3* h_PME_virial_BC = NULL;
            B1 = (float*)malloc(sizeof(float) * fftx);
            ;
            B2 = (float*)malloc(sizeof(float) * ffty);
            B3 = (float*)malloc(sizeof(float) * fftz);
            h_PME_BC0 = (float*)malloc(sizeof(float) * PME_Nfft);
            h_PME_BC = (float*)malloc(sizeof(float) * PME_Nfft);
            h_PME_virial_BC = (LTMatrix3*)malloc(sizeof(LTMatrix3) * PME_Nfft);
            if (B1 == NULL || B2 == NULL || B3 == NULL || h_PME_BC0 == NULL ||
                h_PME_BC == NULL)
            {
                controller->Throw_SPONGE_Error(
                    spongeErrorMallocFailed, "Particle_Mesh::Initial",
                    "Reason:\n\tError occurs when malloc PME_BC of PME");
            }
            for (kx = 0; kx < fftx; kx++)
            {
                B1[kx] = getb(kx, fftx, 4);
            }

            for (ky = 0; ky < ffty; ky++)
            {
                B2[ky] = getb(ky, ffty, 4);
            }

            for (kz = 0; kz < fftz; kz++)
            {
                B3[kz] = getb(kz, fftz, 4);
            }

            float kxrp, kyrp, kzrp;
            float mprefactor = PI * PI / beta / beta;
            float msq;
            VECTOR m;
            for (kx = 0; kx < fftx; kx++)
            {
                kxrp = kx;
                if (kx > fftx / 2) kxrp = kx - fftx;
                for (ky = 0; ky < ffty; ky++)
                {
                    kyrp = ky;
                    if (ky > ffty / 2) kyrp = ky - ffty;
                    for (kz = 0; kz <= fftz / 2; kz++)
                    {
                        kzrp = kz;
                        m = {kxrp, kyrp, kzrp};
                        m = MultiplyTranspose(m, rcell);
                        msq = m * m;

                        index = kx * ffty * (fftz / 2 + 1) +
                                ky * (fftz / 2 + 1) + kz;

                        if (kx + ky + kz == 0)
                        {
                            h_PME_BC[index] = 0;
                            h_PME_virial_BC[index] = {0, 0, 0, 0, 0, 0};
                        }
                        else
                        {
                            h_PME_BC[index] = (float)1.0 / PI / msq *
                                              exp(-mprefactor * msq) / volume;
                            h_PME_virial_BC[index].a11 =
                                1 -
                                2 / msq * (1 + mprefactor * msq) * m.x * m.x;
                            h_PME_virial_BC[index].a21 =
                                0 -
                                2 / msq * (1 + mprefactor * msq) * m.y * m.x;
                            h_PME_virial_BC[index].a22 =
                                1 -
                                2 / msq * (1 + mprefactor * msq) * m.y * m.y;
                            h_PME_virial_BC[index].a31 =
                                0 -
                                2 / msq * (1 + mprefactor * msq) * m.z * m.x;
                            h_PME_virial_BC[index].a32 =
                                0 -
                                2 / msq * (1 + mprefactor * msq) * m.z * m.y;
                            h_PME_virial_BC[index].a33 =
                                1 -
                                2 / msq * (1 + mprefactor * msq) * m.z * m.z;
                        }
                        h_PME_BC0[index] = B1[kx] * B2[ky] * B3[kz];
                        h_PME_BC[index] *= h_PME_BC0[index];
                        h_PME_virial_BC[index] =
                            0.5f * h_PME_BC[index] * h_PME_virial_BC[index];
                    }
                }
            }

            Device_Malloc_Safely((void**)&PME_BC, sizeof(float) * PME_Nfft);
            Device_Malloc_Safely((void**)&PME_BC0, sizeof(float) * PME_Nfft);
            Device_Malloc_Safely((void**)&PME_Virial_BC,
                                 sizeof(LTMatrix3) * PME_Nfft);
            deviceMemcpy(PME_BC, h_PME_BC, sizeof(float) * PME_Nfft,
                         deviceMemcpyHostToDevice);
            deviceMemcpy(PME_BC0, h_PME_BC0, sizeof(float) * PME_Nfft,
                         deviceMemcpyHostToDevice);
            deviceMemcpy(PME_Virial_BC, h_PME_virial_BC,
                         sizeof(LTMatrix3) * PME_Nfft,
                         deviceMemcpyHostToDevice);
            free(B1);
            free(B2);
            free(B3);
            free(h_PME_BC0);
            free(h_PME_BC);
            free(h_PME_virial_BC);
        }
    }
    is_initialized = 1;
    if (is_initialized && !is_controller_printf_initialized)
    {
        controller->Step_Print_Initial(this->module_name, "%.2f");
        if (controller->Command_Exist(this->module_name, "print_detail"))
        {
            print_detail = controller->Get_Bool(
                this->module_name, "print_detail", "Particle_Mesh::Initial");
            if (print_detail)
            {
                controller->Step_Print_Initial("PM_direct", "%.2f");
                controller->Step_Print_Initial("PM_reciprocal", "%.2f");
                controller->Step_Print_Initial("PM_self", "%.2f");
                controller->Step_Print_Initial("PM_correction", "%.2f");
            }
        }
        is_controller_printf_initialized = 1;
        controller->printf("    structure last modify date is %d\n",
                           last_modify_date);
    }
    controller->printf("END INITIALIZING PME\n\n");
}

void Particle_Mesh::Clear()
{
    if (is_initialized)
    {
        is_initialized = 0;
        Free_Single_Device_Pointer((void**)&PME_uxyz);
        Free_Single_Device_Pointer((void**)&PME_frxyz);
        Free_Single_Device_Pointer((void**)&PME_Q);
        Free_Single_Device_Pointer((void**)&PME_FQ);
        Free_Single_Device_Pointer((void**)&PME_FBCFQ);
        Free_Single_Device_Pointer((void**)&PME_BC);
        Free_Single_Device_Pointer((void**)&PME_Virial_BC);
        Free_Single_Device_Pointer((void**)&PME_BC0);
        Free_Single_Device_Pointer((void**)&charge_sum);
        Free_Single_Device_Pointer((void**)&charge_square);
        Free_Single_Device_Pointer((void**)&num_ghost_dir_id);

        Free_Single_Device_Pointer((void**)&atom_id_l_g);
        Free_Single_Device_Pointer((void**)&atom_id_g_l);
        Free_Single_Device_Pointer((void**)&g_crd);
        Free_Single_Device_Pointer((void**)&g_frc);

        // Free_Single_Device_Pointer((void**)&MPI_PME_Q);
        // Free_Single_Device_Pointer((void**)&MPI_PME_FQ);
        // Free_Single_Device_Pointer((void**)&MPI_PME_FBCFQ);

        Free_Single_Device_Pointer((void**)&PME_atom_near);
        Free_Single_Device_Pointer((void**)&force_backup);

        SPONGE_FFT_WRAPPER::Destroy_FFT_Plan(&PME_plan_r2c);
        SPONGE_FFT_WRAPPER::Destroy_FFT_Plan(&PME_plan_c2r);

        Free_Host_And_Device_Pointer(NULL, (void**)&d_reciprocal_ene);
        Free_Host_And_Device_Pointer(NULL, (void**)&d_self_ene);
        Free_Host_And_Device_Pointer(NULL, (void**)&d_direct_ene);
        Free_Host_And_Device_Pointer(NULL, (void**)&d_correction_ene);
        Free_Host_And_Device_Pointer(NULL, (void**)&d_ee_ene);
        Free_Single_Device_Pointer((void**)&d_direct_atom_energy);
        Free_Single_Device_Pointer((void**)&d_correction_atom_energy);
    }
}

// 计算每个原子所在的网格点以及其周围64个网格点的索引
__global__ void PME_Atom_Near(const VECTOR* crd, int* PME_atom_near,
                              const int PME_Nin, const LTMatrix3 cell,
                              const LTMatrix3 rcell, const int atom_numbers,
                              const int fftx, const int ffty, const int fftz,
                              UNSIGNED_INT_VECTOR* PME_uxyz, VECTOR* PME_frxyz,
                              VECTOR* force_backup)
{
    SIMPLE_DEVICE_FOR(atom, atom_numbers)
    {
        force_backup[atom] = {0.0f, 0.0f, 0.0f};
        UNSIGNED_INT_VECTOR* temp_uxyz = &PME_uxyz[atom];
        VECTOR frac_crd = crd[atom] * rcell;
        frac_crd = frac_crd - floorf(frac_crd);
        if (!isfinite(frac_crd.x) || !isfinite(frac_crd.y) ||
            !isfinite(frac_crd.z))
        {
            frac_crd = {0.0f, 0.0f, 0.0f};
        }
        int k, tempux, tempuy, tempuz;
        frac_crd.x *= fftx;
        tempux = (int)frac_crd.x;
        tempux = tempux < 0 ? 0 : (tempux < fftx ? tempux : fftx - 1);
        PME_frxyz[atom].x = frac_crd.x - tempux;
        PME_frxyz[atom].x = PME_frxyz[atom].x - floorf(PME_frxyz[atom].x);
        frac_crd.y *= ffty;
        tempuy = (int)frac_crd.y;
        tempuy = tempuy < 0 ? 0 : (tempuy < ffty ? tempuy : ffty - 1);
        PME_frxyz[atom].y = frac_crd.y - tempuy;
        PME_frxyz[atom].y = PME_frxyz[atom].y - floorf(PME_frxyz[atom].y);
        frac_crd.z *= fftz;
        tempuz = (int)frac_crd.z;
        tempuz = tempuz < 0 ? 0 : (tempuz < fftz ? tempuz : fftz - 1);
        PME_frxyz[atom].z = frac_crd.z - tempuz;
        PME_frxyz[atom].z = PME_frxyz[atom].z - floorf(PME_frxyz[atom].z);
        if (tempux != (*temp_uxyz).uint_x || tempuy != (*temp_uxyz).uint_y ||
            tempuz != (*temp_uxyz).uint_z)
        {
            (*temp_uxyz).uint_x = tempux;
            (*temp_uxyz).uint_y = tempuy;
            (*temp_uxyz).uint_z = tempuz;
            int* temp_near = PME_atom_near + atom * 64;
            int kx, ky, kz;
            for (k = 0; k < 64; k++)
            {
                kx = k / 16;
                ky = (k - 16 * kx) / 4;
                kz = k % 4;

                kx = tempux - kx;

                if (kx < 0) kx += fftx;
                if (kx >= fftx) kx -= fftx;
                ky = tempuy - ky;
                if (ky < 0) ky += ffty;
                if (ky >= ffty) ky -= ffty;
                kz = tempuz - kz;
                if (kz < 0) kz += fftz;
                if (kz >= fftz) kz -= fftz;
                temp_near[k] = kx * PME_Nin + ky * fftz + kz;
            }
        }
    }
}

// 将原子电荷分配到其周围的64个网格点上
__global__ void PME_Q_Spread(int* PME_atom_near, const float* charge,
                             const VECTOR* PME_frxyz, float* PME_Q,
                             const int atom_numbers, const int PME_Nall)
{
    SIMPLE_DEVICE_FOR(atom, atom_numbers)
    {
        int k;
        float tempf, tempQ, tempf2;
        int* temp_near = PME_atom_near + atom * 64;
        VECTOR temp_frxyz = PME_frxyz[atom];
        float tempcharge = charge[atom];

        unsigned int kx;
#ifdef USE_GPU
        for (k = threadIdx.y; k < 64; k = k + blockDim.y)
#else
        for (k = 0; k < 64; k++)
#endif
        {
            kx = k / 16;
            tempf = temp_frxyz.x;
            tempf2 = tempf * tempf;
            tempf = PME_Ma[kx] * tempf * tempf2 + PME_Mb[kx] * tempf2 +
                    PME_Mc[kx] * tempf + PME_Md[kx];

            tempQ = tempcharge * tempf;

            kx = (k - kx * 16) / 4;
            tempf = temp_frxyz.y;
            tempf2 = tempf * tempf;
            tempf = PME_Ma[kx] * tempf * tempf2 + PME_Mb[kx] * tempf2 +
                    PME_Mc[kx] * tempf + PME_Md[kx];

            tempQ = tempQ * tempf;

            kx = k % 4;
            tempf = temp_frxyz.z;
            tempf2 = tempf * tempf;
            tempf = PME_Ma[kx] * tempf * tempf2 + PME_Mb[kx] * tempf2 +
                    PME_Mc[kx] * tempf + PME_Md[kx];
            tempQ = tempQ * tempf;

            int near_index = temp_near[k];
            if ((unsigned int)near_index < (unsigned int)PME_Nall)
            {
                atomicAdd(&PME_Q[near_index], tempQ);
            }
        }
    }
}

// 对FFT后的电荷密度进行修正
__global__ void PME_BCFQ(FFT_COMPLEX* PME_FQ, float* PME_BC, int PME_Nfft)
{
    SIMPLE_DEVICE_FOR(index, PME_Nfft)
    {
        float tempf = PME_BC[index];
        FFT_COMPLEX tempc = PME_FQ[index];
        REAL(PME_FQ[index]) = REAL(tempc) * tempf;
        IMAGINARY(PME_FQ[index]) = IMAGINARY(tempc) * tempf;
    }
}

// 计算每个原子受力
static __global__ void PME_Final(int* PME_atom_near, const float* charge,
                                 const float* PME_Q, VECTOR* force,
                                 const VECTOR* PME_frxyz, const LTMatrix3 rcell,
                                 const int fftx, const int ffty, const int fftz,
                                 const int atom_numbers, const int PME_Nall)
{
#ifdef GPU_ARCH_NAME
    int atom = blockDim.y * blockIdx.x + threadIdx.y;
    if (atom < atom_numbers)
#else
#pragma omp parallel for
    for (int atom = 0; atom < atom_numbers; atom++)
#endif
    {
        int k, kx;
        float tempdx, tempdy, tempdz, tempx, tempy, tempz, tempdQf;
        VECTOR tempdQ;
        float tempf, tempf2;
        float temp_charge = charge[atom];
        int* temp_near = PME_atom_near + atom * 64;
        VECTOR temp_frxyz = PME_frxyz[atom];
        VECTOR tempnv = {0, 0, 0};
#ifdef USE_GPU
        for (k = threadIdx.x; k < 64; k = k + blockDim.x)
#else
        for (k = 0; k < 64; k++)
#endif
        {
            int near_index = temp_near[k];
            if ((unsigned int)near_index >= (unsigned int)PME_Nall)
            {
                continue;
            }
            tempdQf = -PME_Q[near_index] * temp_charge;

            kx = k / 16;
            tempf = temp_frxyz.x;
            tempf2 = tempf * tempf;
            tempx = PME_Ma[kx] * tempf * tempf2 + PME_Mb[kx] * tempf2 +
                    PME_Mc[kx] * tempf + PME_Md[kx];
            tempdx = PME_dMa[kx] * tempf2 + PME_dMb[kx] * tempf + PME_dMc[kx];

            kx = (k - kx * 16) / 4;
            tempf = temp_frxyz.y;
            tempf2 = tempf * tempf;
            tempy = PME_Ma[kx] * tempf * tempf2 + PME_Mb[kx] * tempf2 +
                    PME_Mc[kx] * tempf + PME_Md[kx];
            tempdy = PME_dMa[kx] * tempf2 + PME_dMb[kx] * tempf + PME_dMc[kx];

            kx = k % 4;
            tempf = temp_frxyz.z;
            tempf2 = tempf * tempf;
            tempz = PME_Ma[kx] * tempf * tempf2 + PME_Mb[kx] * tempf2 +
                    PME_Mc[kx] * tempf + PME_Md[kx];
            tempdz = PME_dMa[kx] * tempf2 + PME_dMb[kx] * tempf + PME_dMc[kx];

            tempdQ.x = tempdx * tempy * tempz * fftx;
            tempdQ.y = tempdy * tempx * tempz * ffty;
            tempdQ.z = tempdz * tempx * tempy * fftz;
            tempdQ = tempdQf * MultiplyTranspose(tempdQ, rcell);
            tempnv = tempnv + tempdQ;
        }
        Warp_Sum_To(force + atom, tempnv, 8);
    }
}

// sum += list1 * list2
__global__ void PME_Energy_Product(const int element_number, const float* list1,
                                   const float* list2, float* sum)
{
#ifdef USE_GPU
    if (threadIdx.x == 0)
    {
        sum[0] = 0.;
    }
    __syncthreads();
#else
    sum[0] = 0;
#endif
    float lin = 0.0;
#ifdef USE_GPU
    for (int i = threadIdx.x; i < element_number; i = i + blockDim.x)
#else
#pragma omp parallel for reduction(+ : lin)
    for (int i = 0; i < element_number; i++)
#endif
    {
        lin = lin + list1[i] * list2[i];
    }
    atomicAdd(sum, lin);
}

static __global__ void PME_Excluded_Force_With_Atom_Energy_Correction(
    const int atom_numbers, const VECTOR* crd, const LTMatrix3 cell,
    const LTMatrix3 rcell, const float* charge, const float pme_beta,
    const int* excluded_list_start, const int* excluded_list,
    const int* excluded_atom_numbers, VECTOR* frc, float* atom_ene,
    float* this_ene, LTMatrix3* atom_virial)
{
    SIMPLE_DEVICE_FOR(atom_i, atom_numbers)
    {
        int excluded_numbers = excluded_atom_numbers[atom_i];
        if (excluded_numbers > 0)
        {
            int list_start = excluded_list_start[atom_i];
            int list_end = list_start + excluded_numbers;
            int atom_j;

            float charge_i = charge[atom_i];
            float charge_j;
            float dr_abs;
            float beta_dr;

            VECTOR r1 = crd[atom_i], r2;
            VECTOR dr;
            float dr2;

            float frc_abs = 0.;
            VECTOR frc_lin;
            VECTOR frc_record = {0., 0., 0.};
            LTMatrix3 virial_record = {0, 0, 0, 0, 0, 0};
            float ene_lin = 0.;

            for (int i = list_start; i < list_end; i = i + 1)
            {
                atom_j = excluded_list[i];
                r2 = crd[atom_j];
                charge_j = charge[atom_j];

                dr = Get_Periodic_Displacement(r2, r1, cell, rcell);
                dr2 = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;
                // 假设剔除表中的原子对距离总是小于cutoff的，正常体系

                dr_abs = sqrtf(dr2);
                beta_dr = pme_beta * dr_abs;
                frc_abs = beta_dr * TWO_DIVIDED_BY_SQRT_PI *
                              expf(-beta_dr * beta_dr) +
                          erfcf(beta_dr);
                frc_abs = (frc_abs - 1.) / dr2 / dr_abs;
                frc_abs = -charge_i * charge_j * frc_abs;
                frc_lin = frc_abs * dr;
                ene_lin -= charge_i * charge_j * erff(beta_dr) / dr_abs;
                frc_record = frc_record + frc_lin;
                atomicAdd(frc + atom_j, -frc_lin);
                virial_record =
                    virial_record - Get_Virial_From_Force_Dis(frc_lin, dr);
            }  // atom_j cycle
            atomicAdd(frc + atom_i, frc_record);
            atomicAdd(atom_virial + atom_i, virial_record);
            atomicAdd(atom_ene + atom_i, ene_lin);
            this_ene[atom_i] = ene_lin;
        }  // if need excluded
    }
}

void Particle_Mesh::PME_Excluded_Force_With_Atom_Energy(
    const VECTOR* crd, const LTMatrix3 cell, const LTMatrix3 rcell,
    const float* charge, const int* excluded_list_start,
    const int* excluded_list, const int* excluded_atom_numbers, VECTOR* frc,
    int need_energy, float* atom_ene, LTMatrix3* atom_virial)
{
    if (is_initialized && calculate_excluded_part)
    {
        if (need_energy)
            deviceMemset(d_correction_atom_energy, 0,
                         sizeof(float) * atom_numbers);
        if (CONTROLLER::MPI_rank != 0) return;
        Launch_Device_Kernel(
            PME_Excluded_Force_With_Atom_Energy_Correction,
            (atom_numbers + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, atom_numbers, crd, cell,
            rcell, charge, beta, excluded_list_start, excluded_list,
            excluded_atom_numbers, frc, atom_ene, d_correction_atom_energy,
            atom_virial);
    }
}

static __global__ void PME_Add_Energy_To_Potential(float* d_ene,
                                                   float* d_self_ene,
                                                   float* d_reciprocal_ene)
{
    d_ene[0] += d_self_ene[0] + d_reciprocal_ene[0];
}

static __global__ void device_add_force(const int atom_numbers,
                                        float update_interval, VECTOR* force,
                                        const VECTOR* force_backup)
{
    SIMPLE_DEVICE_FOR(atom_i, atom_numbers)
    {
        force[atom_i] = force[atom_i] + update_interval * force_backup[atom_i];
    }
}

static __global__ void PME_Sum_Virial(const int nfft,
                                      const LTMatrix3* virial_BC,
                                      const FFT_COMPLEX* FQ, LTMatrix3* virial,
                                      int fftz)
{
    LTMatrix3 vir = {0, 0, 0, 0, 0, 0};
#ifdef USE_GPU
    int tid = blockDim.x * blockIdx.x * blockDim.y + threadIdx.x * blockDim.y +
              threadIdx.y;
    for (int index = tid; index < nfft;
         index += blockDim.x * blockDim.y * gridDim.x)
    {
        int fftc = fftz / 2 + 1;
        int nz = index % fftc;
        float factor = (nz == 0 || nz == fftc - 1) ? 0.5f : 1.0f;
        FFT_COMPLEX FQ0 = FQ[index];
        LTMatrix3 vir0 =
            factor * (REAL(FQ0) * REAL(FQ0) + IMAGINARY(FQ0) * IMAGINARY(FQ0)) *
            virial_BC[index];
        vir = vir - vir0;
    }
#else
    float v11 = 0.0f, v21 = 0.0f, v22 = 0.0f;
    float v31 = 0.0f, v32 = 0.0f, v33 = 0.0f;
#pragma omp parallel for reduction(+ : v11, v21, v22, v31, v32, v33)
    for (int index = 0; index < nfft; index++)
    {
        int fftc = fftz / 2 + 1;
        int nz = index % fftc;
        float factor = (nz == 0 || nz == fftc - 1) ? 0.5f : 1.0f;
        FFT_COMPLEX FQ0 = FQ[index];
        LTMatrix3 vir0 =
            factor * (REAL(FQ0) * REAL(FQ0) + IMAGINARY(FQ0) * IMAGINARY(FQ0)) *
            virial_BC[index];
        v11 -= vir0.a11;
        v21 -= vir0.a21;
        v22 -= vir0.a22;
        v31 -= vir0.a31;
        v32 -= vir0.a32;
        v33 -= vir0.a33;
    }
    vir = {v11, v21, v22, v31, v32, v33};
#endif
    Warp_Sum_To(virial, vir, warpSize);
}

void Particle_Mesh::PME_Reciprocal_Force_With_Energy_And_Virial(
    const VECTOR* crd, const LTMatrix3 cell, const LTMatrix3 rcell,
    const float* charge, VECTOR* force, int need_virial, int need_energy,
    LTMatrix3* d_virial, float* d_potential, int step)
{
    if (is_initialized && calculate_reciprocal_part)
    {
        if (need_energy)
        {
            deviceMemset(d_reciprocal_ene, 0, sizeof(float));
            deviceMemset(d_self_ene, 0, sizeof(float));
        }
        if (step % update_interval == 0)
        {
            // 计算插值索引
            deviceMemset(PME_Q, 0, sizeof(float) * PME_Nall);
            Launch_Device_Kernel(
                PME_Atom_Near,
                (atom_numbers + CONTROLLER::device_max_thread - 1) /
                    CONTROLLER::device_max_thread,
                CONTROLLER::device_max_thread, 0, NULL, crd, PME_atom_near,
                PME_Nin, cell, rcell, atom_numbers, fftx, ffty, fftz, PME_uxyz,
                PME_frxyz, force_backup);

            dim3 blockSize = {CONTROLLER::device_max_thread / 64, 64};

            // 电荷Bspline插值
            Launch_Device_Kernel(PME_Q_Spread,
                                 (atom_numbers + blockSize.x - 1) / blockSize.x,
                                 blockSize, 0, NULL, PME_atom_near, charge,
                                 PME_frxyz, PME_Q, atom_numbers, PME_Nall);

            // do FFT
            SPONGE_FFT_WRAPPER::R2C(PME_plan_r2c, PME_Q, PME_FQ);

            // 修正Bspline插值
            blockSize = {
                CONTROLLER::device_warp,
                CONTROLLER::device_max_thread / CONTROLLER::device_warp};
            if (need_virial)
                Launch_Device_Kernel(
                    PME_Sum_Virial,
                    (PME_Nfft + 4 * CONTROLLER::device_max_thread - 1) /
                        CONTROLLER::device_max_thread,
                    blockSize, 0, NULL, PME_Nfft, PME_Virial_BC, PME_FQ,
                    d_virial, fftz);

            Launch_Device_Kernel(
                PME_BCFQ,
                (PME_Nfft + CONTROLLER::device_max_thread - 1) /
                    CONTROLLER::device_max_thread,
                CONTROLLER::device_max_thread, 0, NULL, PME_FQ, PME_BC,
                PME_Nfft);

            // do inverse FFT
            SPONGE_FFT_WRAPPER::C2R(PME_plan_c2r, PME_FQ, PME_FBCFQ);

            // 计算势能和力
            blockSize = {8, CONTROLLER::device_max_thread / 8};
            Launch_Device_Kernel(PME_Final,
                                 (atom_numbers + blockSize.x - 1) / blockSize.x,
                                 blockSize, 0, NULL, PME_atom_near, charge,
                                 PME_FBCFQ, force_backup, PME_frxyz, rcell,
                                 fftx, ffty, fftz, atom_numbers, PME_Nall);

            Launch_Device_Kernel(
                device_add_force,
                (atom_numbers + CONTROLLER::device_max_thread - 1) /
                    CONTROLLER::device_max_thread,
                CONTROLLER::device_max_thread, 0, NULL, atom_numbers,
                update_interval, force, force_backup);
        }
        if (need_energy)
        {
            Launch_Device_Kernel(PME_Energy_Product, 1,
                                 CONTROLLER::device_max_thread, 0, NULL,
                                 PME_Nall, PME_Q, PME_FBCFQ, d_reciprocal_ene);
            Scale_List(d_reciprocal_ene, 0.5f, 1);

            Launch_Device_Kernel(
                charge_square_kernel,
                (atom_numbers + CONTROLLER::device_max_thread - 1) /
                    CONTROLLER::device_max_thread,
                CONTROLLER::device_max_thread, 0, NULL, atom_numbers, charge,
                charge_square);
            Sum_Of_List(charge_square, d_self_ene, atom_numbers);

            Scale_List(d_self_ene, -beta / sqrt(PI), 1);

            Sum_Of_List(charge, charge_sum, atom_numbers);

            Launch_Device_Kernel(device_add, 1, 1, 0, NULL, d_self_ene,
                                 neutralizing_factor, charge_sum);

            Launch_Device_Kernel(PME_Add_Energy_To_Potential, 1, 1, 0, NULL,
                                 d_potential, d_self_ene, d_reciprocal_ene);
        }
    }
}

// 计算PME的位移势能和Virial张量
static __global__ void up_box_bc(int fftx, int ffty, int fftz, float* PME_BC,
                                 float* PME_BC0, LTMatrix3* PME_virial_BC,
                                 float mprefactor, LTMatrix3 rcell,
                                 float volume)
{
    float kxrp, kyrp, kzrp;
    int ky, kz, index;
    float msq;
    VECTOR m;
    LTMatrix3 virial_bc_local;
    float bc_local;
#ifdef USE_GPU
    for (int kx = blockIdx.x * blockDim.x + threadIdx.x; kx < fftx;
         kx += blockDim.x * gridDim.x)
#else
#pragma omp parallel for firstprivate(kxrp, kyrp, kzrp, ky, kz, index, msq, m, \
                                          virial_bc_local, bc_local)
    for (int kx = 0; kx < fftx; kx++)
#endif
    {
        kxrp = kx;
        if (kx > fftx / 2) kxrp = kx - fftx;
#ifdef USE_GPU
        for (ky = blockIdx.y * blockDim.y + threadIdx.y; ky < ffty;
             ky += blockDim.y * gridDim.y)
#else
        for (ky = 0; ky < ffty; ky++)
#endif
        {
            kyrp = ky;
            if (ky > ffty / 2) kyrp = ky - ffty;
#ifdef USE_GPU
            for (kz = threadIdx.z; kz <= fftz / 2; kz += blockDim.z)
#else
            for (kz = 0; kz <= fftz / 2; kz++)
#endif
            {
                kzrp = kz;
                m = {kxrp, kyrp, kzrp};
                m = MultiplyTranspose(m, rcell);
                msq = m * m;

                index = kx * ffty * (fftz / 2 + 1) + ky * (fftz / 2 + 1) + kz;

                if (kx + ky + kz == 0)
                {
                    PME_BC[index] = 0;
                    PME_virial_BC[index] = {0, 0, 0, 0, 0, 0};
                }
                else
                {
                    bc_local = (float)1.0 / PI / msq * exp(mprefactor * msq) /
                               volume * PME_BC0[index];
                    virial_bc_local.a11 =
                        1 - 2 / msq * (1 + mprefactor * msq) * m.x * m.x;
                    virial_bc_local.a21 =
                        0 - 2 / msq * (1 + mprefactor * msq) * m.y * m.x;
                    virial_bc_local.a22 =
                        1 - 2 / msq * (1 + mprefactor * msq) * m.y * m.y;
                    virial_bc_local.a31 =
                        0 - 2 / msq * (1 + mprefactor * msq) * m.z * m.x;
                    virial_bc_local.a32 =
                        0 - 2 / msq * (1 + mprefactor * msq) * m.z * m.y;
                    virial_bc_local.a33 =
                        1 - 2 / msq * (1 + mprefactor * msq) * m.z * m.z;
                    PME_virial_BC[index] = 0.5f * bc_local * virial_bc_local;
                    PME_BC[index] = bc_local;
                }
            }
        }
    }
}

static void Scale_Positions_Device(const LTMatrix3 g, VECTOR* crd, float dt)
{
    VECTOR r_dash;
    r_dash.x = crd[0].x +
               dt * (crd[0].x * g.a11 + crd[0].y * g.a21 + crd[0].z * g.a31);
    r_dash.y = crd[0].y + dt * (crd[0].y * g.a22 + crd[0].z * g.a32);
    r_dash.z = crd[0].z + dt * crd[0].z * g.a33;
    crd[0] = r_dash;
}

void Particle_Mesh::Update_Box(LTMatrix3 cell, LTMatrix3 rcell, LTMatrix3 g,
                               float dt)
{
    float volume = cell.a11 * cell.a22 * cell.a33;
    neutralizing_factor = -0.5 * CONSTANT_Pi / (beta * beta * volume);
    float mprefactor = PI * PI / -beta / beta;
    dim3 blockSize = {8, 8, CONTROLLER::device_max_thread / 64};
    dim3 gridSize = {64, 64};
    Launch_Device_Kernel(up_box_bc, gridSize, blockSize, 0, NULL, fftx, ffty,
                         fftz, PME_BC, PME_BC0, PME_Virial_BC, mprefactor,
                         rcell, volume);
    Scale_Positions_Device(g, &min_corner, dt);
    Scale_Positions_Device(g, &max_corner, dt);
}

//-------domain-decomposition and communication----------------

// 找出n的所有因子，存入factor_set，并按从大到小排序
static void find_factor(std::vector<int>& factor_set, int n)
{
    for (int i = 1; i <= std::sqrt(n); ++i)
    {
        if (n % i == 0)
        {
            factor_set.push_back(i);
            if (i != n / i)
            {
                factor_set.push_back(n / i);
            }
        }
    }
    std::sort(factor_set.begin(), factor_set.end(), std::greater<int>());
}

// Domain Decomposition
void Particle_Mesh::Domain_Decomposition(CONTROLLER* controller,
                                         VECTOR box_length,
                                         INT_VECTOR pp_split_num)
{
    // 如果设置PM进程数=0， 直接返回；区域分割只在主进程上做一次
    if (controller->MPI_rank != 0 || !PM_MPI_size)
    {
        return;
    }
    // 如果PP进程数不能被PM进程数整除，报错
    if (controller->PP_MPI_size % PM_MPI_size)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorValueErrorCommand,
            "Particle_Mesh_Ewald::Domain_Decomposition",
            "Reason:\n\tThe number of PP processes must be divisible by the "
            "number of PM processes.");
        return;
    }

    int nx = pp_split_num.int_x;
    int ny = pp_split_num.int_y;
    int nz = pp_split_num.int_z;

    std::vector<int> fac_set_x;
    std::vector<int> fac_set_y;
    std::vector<int> fac_set_z;
    find_factor(fac_set_x, nx);
    find_factor(fac_set_y, ny);
    find_factor(fac_set_z, nz);
    int pm_size;
    for (int& tmpx : fac_set_x)
    {
        pm_size = controller->PM_MPI_size;
        if (pm_size % tmpx == 0 && pm_dom_dec_split_num.int_x == 0)
        {
            pm_size /= tmpx;
            for (int& tmpy : fac_set_y)
            {
                if (pm_size % tmpy == 0 && pm_dom_dec_split_num.int_y == 0)
                {
                    pm_size /= tmpy;
                    for (int& tmpz : fac_set_z)
                    {
                        if (pm_size == tmpz)
                        {
                            pm_dom_dec_split_num.int_x = tmpx;
                            pm_dom_dec_split_num.int_y = tmpy;
                            pm_dom_dec_split_num.int_z = tmpz;
                        }
                    }
                }
            }
        }
    }

    nx = pm_dom_dec_split_num.int_x;
    ny = pm_dom_dec_split_num.int_y;
    nz = pm_dom_dec_split_num.int_z;
    std::cout << "pm_nx= " << nx << ", pm_ny= " << ny << ", pm_nz= " << nz
              << std::endl;
    for (int k = 0; k < nz; ++k)
    {
        for (int j = 0; j < ny; ++j)
        {
            for (int i = 0; i < nx; ++i)
            {
                int rank_id = i + j * nx + k * nx * ny;
                min_corner_set[rank_id].x = box_length.x / nx * i;
                min_corner_set[rank_id].y = box_length.y / ny * j;
                min_corner_set[rank_id].z = box_length.z / nz * k;
                max_corner_set[rank_id].x = box_length.x / nx * (i + 1);
                max_corner_set[rank_id].y = box_length.y / ny * (j + 1);
                max_corner_set[rank_id].z = box_length.z / nz * (k + 1);
            }
        }
    }

    // 若单进程， PM与PP共享同一进程下，
    if (controller->MPI_size == 1 && PM_MPI_size == 1)
    {
        pm_pp_num[0] = 1;
        pm_pp_corres[0][0] = 0;
        return;
    }

    int nx_ = pp_split_num.int_x / pm_dom_dec_split_num.int_x;
    int ny_ = pp_split_num.int_y / pm_dom_dec_split_num.int_y;
    int nz_ = pp_split_num.int_z / pm_dom_dec_split_num.int_z;
    for (int i = 0; i < controller->PM_MPI_size; ++i)
    {
        pm_pp_num[i] = 0;
    }
    for (int k = 0; k < pp_split_num.int_z; ++k)
    {
        for (int j = 0; j < pp_split_num.int_y; ++j)
        {
            for (int i = 0; i < pp_split_num.int_x; ++i)
            {
                int pp_rank_id = i + j * pp_split_num.int_x +
                                 k * pp_split_num.int_x * pp_split_num.int_y;
                int pm_rank_id = i / nx_ +
                                 j / ny_ * pm_dom_dec_split_num.int_x +
                                 k / nz_ * pm_dom_dec_split_num.int_x *
                                     pm_dom_dec_split_num.int_y;
                pm_pp_corres[pm_rank_id][pm_pp_num[pm_rank_id]] = pp_rank_id;
                pm_pp_num[pm_rank_id]++;
            }
        }
    }
}

void Particle_Mesh::Send_Recv_Dom_Dec(CONTROLLER* controller)
{
    // 如果设置PM进程数=0， 直接返回
    if (!PM_MPI_size)
    {
        return;
    }
    // 如果PM与PP共用一个进程，则不需要通信
    if (controller->MPI_size == 1 && PM_MPI_size == 1)
    {
        strcpy(this->FFT_MPI_TYPE, "DISABLE");
        return;
    }
#ifdef USE_MPI
    // PP进程与PM进程分割
    if (controller->MPI_rank == 0)
    {
        // 发送PP进程对应的PM进程号
        for (int pm_id = 0; pm_id < controller->PM_MPI_size; ++pm_id)
        {
            int pm_rank_tot = pm_id + controller->PP_MPI_size;
            for (int i = 0; i < pm_pp_num[pm_id]; ++i)
            {
                if (pm_pp_corres[pm_id][i] != 0)
                {
                    MPI_Send(&pm_rank_tot, sizeof(int), MPI_BYTE,
                             pm_pp_corres[pm_id][i], pm_pp_corres[pm_id][i],
                             MPI_COMM_WORLD);
                }
                else
                {
                    pp_corres_pm_rank = pm_rank_tot;
                }
            }
        }
        // 发送PM进程对应的pp进程数与进程索引集合；发送域分割信息
        for (int pm_id = 0; pm_id < controller->PM_MPI_size; ++pm_id)
        {
            int pm_rank_tot = pm_id + controller->PP_MPI_size;
            MPI_Send(&min_corner_set[pm_id], sizeof(VECTOR), MPI_BYTE,
                     pm_rank_tot, 0, MPI_COMM_WORLD);
            MPI_Send(&max_corner_set[pm_id], sizeof(VECTOR), MPI_BYTE,
                     pm_rank_tot, 1, MPI_COMM_WORLD);
            MPI_Send(&pm_pp_num[pm_id], sizeof(int), MPI_BYTE, pm_rank_tot, 2,
                     MPI_COMM_WORLD);
            MPI_Send(pm_pp_corres[pm_id], pm_pp_num[pm_id] * sizeof(int),
                     MPI_BYTE, pm_rank_tot, 3, MPI_COMM_WORLD);
            MPI_Send(&pm_dom_dec_split_num, sizeof(INT_VECTOR), MPI_BYTE,
                     pm_rank_tot, 4, MPI_COMM_WORLD);
        }
    }
    else
    {
        if (controller->MPI_rank < controller->PP_MPI_size)
        {
            MPI_Recv(&pp_corres_pm_rank, sizeof(int), MPI_BYTE, 0,
                     controller->MPI_rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        else
        {
            MPI_Recv(&min_corner, sizeof(VECTOR), MPI_BYTE, 0, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&max_corner, sizeof(VECTOR), MPI_BYTE, 0, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&pm_corres_pp_num, sizeof(int), MPI_BYTE, 0, 2,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(pm_corres_pp_rank_set, pm_corres_pp_num * sizeof(int),
                     MPI_BYTE, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&pm_dom_dec_split_num, sizeof(INT_VECTOR), MPI_BYTE, 0, 4,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (pm_dom_dec_split_num.int_y == 1 &&
                pm_dom_dec_split_num.int_z == 1)
            {
                strcpy(this->FFT_MPI_TYPE, "SLAB");
            }
            else
            {
                strcpy(this->FFT_MPI_TYPE, "BRICK");
            }
        }
    }
#endif
}

void Particle_Mesh::Find_Neighbor_Domain(CONTROLLER* controller)
{
    if (!PM_MPI_size)
    {
        return;
    }
    if (controller->PM_MPI_size == 1 ||
        controller->MPI_rank < controller->PP_MPI_size)
    {
        for (int dir = 0; dir < 6; ++dir)
        {
            neighbor_num[dir] = 0;
        }
        return;
    }
    int nx = pm_dom_dec_split_num.int_x;
    int ny = pm_dom_dec_split_num.int_y;
    int nz = pm_dom_dec_split_num.int_z;

    neighbor_num[0] = nx == 1 ? 0 : 1;
    neighbor_num[1] = nx == 1 ? 0 : 1;
    neighbor_num[2] = ny == 1 ? 0 : 1;
    neighbor_num[3] = ny == 1 ? 0 : 1;
    neighbor_num[4] = nz == 1 ? 0 : 1;
    neighbor_num[5] = nz == 1 ? 0 : 1;

    int rank_id = pm_rank;
    int i = rank_id % (nx);
    int j = (rank_id / nx) % ny;
    int k = rank_id / (nx * ny);

    if (nx > 1)
    {
        neighbor_dir[0][0] = (i + 1) % nx + j * nx + k * nx * ny;
        neighbor_dir[1][0] = (i - 1 + nx) % nx + j * nx + k * nx * ny;
    }

    if (ny > 1)
    {
        neighbor_dir[2][0] = i % nx + ((j + 1) % ny) * nx + k * nx * ny;
        neighbor_dir[3][0] = i % nx + ((j - 1 + ny) % ny) * nx + k * nx * ny;
    }

    if (nz > 1)
    {
        neighbor_dir[4][0] = i % nx + j * nx + ((k + 1) % nz) * nx * ny;
        neighbor_dir[5][0] = i % nx + j * nx + ((k - 1 + nz) % nz) * nx * ny;
    }
}

static __global__ void inverse_global_and_local(const int* A, int* B, int N)
{
#ifdef USE_GPU
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
#else
#pragma omp parallel for
    for (int i = 0; i < N; i++)
#endif
    {
        int v = A[i];  // v ∈ [0, N)
        B[v] = i;
    }
}

static __global__ void crd_local_to_global(VECTOR* l_crd, VECTOR* g_crd,
                                           int* atom_id_l_g, int N)
{
#ifdef USE_GPU
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
#else
#pragma omp parallel for
    for (int i = 0; i < N; i++)
#endif
    {
        int g_id = atom_id_l_g[i];
        g_crd[g_id] = l_crd[i];
    }
}

void Particle_Mesh::Get_Atoms(CONTROLLER* controller, VECTOR* pme_crd,
                              float* pme_charge, int pp_atom_numbers,
                              VECTOR* pp_crd, float* pp_charge, int* atom_local,
                              bool atom_number_label, bool charge_label,
                              bool crd_label, bool id_label)
{
    if (!PM_MPI_size)
    {
        return;
    }
    // 若单进程PM与PP共享同一进程下，什么也不做。共享进程下直接传入dd.crd等计算，不再重复拷贝内存
    if (controller->MPI_size == 1 && PM_MPI_size == 1)
    {
        return;
    }
#ifdef USE_MPI
    // 先阻塞通信原子数
    if (controller->MPI_rank < controller->PP_MPI_size)
    {
        if (atom_number_label)
        {
            MPI_Send(&pp_atom_numbers, sizeof(int), MPI_BYTE, pp_corres_pm_rank,
                     0, MPI_COMM_WORLD);
        }
    }
    else
    {
        if (atom_number_label)
        {
            int prefix = 0;
            // 接收PP进程对应的原子数
            for (int i = 0; i < pm_corres_pp_num; ++i)
            {
                int pp_rank = pm_corres_pp_rank_set[i];
                {
                    MPI_Recv(&pm_corres_pp_atom_number[i], sizeof(int),
                             MPI_BYTE, pp_rank, 0, MPI_COMM_WORLD,
                             MPI_STATUS_IGNORE);
                    pm_corres_pp_atom_number_prefix[i] = prefix;
                    prefix += pm_corres_pp_atom_number[i];
                }
                this->atom_numbers = prefix;
            }
        }
    }
    // 通信坐标与电荷

    if (controller->MPI_rank < controller->PP_MPI_size)
    {
        D_MPI_GroupStart();
        if (crd_label)
        {
            D_MPI_Send(pp_crd, pp_atom_numbers * sizeof(VECTOR), D_MPI_BYTE,
                       pp_corres_pm_rank, 1, controller->D_MPI_COMM_WORLD,
                       pm_stream);
        }
        if (charge_label)
        {
            D_MPI_Send(pp_charge, pp_atom_numbers * sizeof(float), D_MPI_BYTE,
                       pp_corres_pm_rank, 2, controller->D_MPI_COMM_WORLD,
                       pm_stream);
        }
        if (id_label)
        {
            D_MPI_Send(atom_local, pp_atom_numbers * sizeof(int), D_MPI_BYTE,
                       pp_corres_pm_rank, 3, controller->D_MPI_COMM_WORLD,
                       pm_stream);
        }
        D_MPI_GroupEnd();
#ifdef USE_GPU
        deviceStreamSynchronize(pm_stream);
#endif
    }
    else
    {
        D_MPI_GroupStart();
        for (int i = 0; i < pm_corres_pp_num; ++i)
        {
            int pp_rank = pm_corres_pp_rank_set[i];
            if (crd_label)
            {
                D_MPI_Recv(pme_crd + pm_corres_pp_atom_number_prefix[i],
                           pm_corres_pp_atom_number[i] * sizeof(VECTOR),
                           D_MPI_BYTE, pp_rank, 1, controller->D_MPI_COMM_WORLD,
                           pm_stream);
            }
            if (charge_label)
            {
                D_MPI_Recv(pme_charge + pm_corres_pp_atom_number_prefix[i],
                           pm_corres_pp_atom_number[i] * sizeof(float),
                           D_MPI_BYTE, pp_rank, 2, controller->D_MPI_COMM_WORLD,
                           pm_stream);
            }
            if (id_label)
            {
                D_MPI_Recv(atom_id_l_g + pm_corres_pp_atom_number_prefix[i],
                           pm_corres_pp_atom_number[i] * sizeof(int),
                           D_MPI_BYTE, pp_rank, 3, controller->D_MPI_COMM_WORLD,
                           pm_stream);
            }
        }
        D_MPI_GroupEnd();
#ifdef USE_GPU
        deviceStreamSynchronize(pm_stream);
#endif
        // 反转local与global的映射关系
        if (id_label)
        {
            Launch_Device_Kernel(
                inverse_global_and_local,
                (this->atom_numbers + CONTROLLER::device_max_thread - 1) /
                    CONTROLLER::device_max_thread,
                CONTROLLER::device_max_thread, 0, NULL, atom_id_l_g,
                atom_id_g_l, this->atom_numbers);
        }
        if (crd_label)
        {
            Launch_Device_Kernel(
                crd_local_to_global,
                (this->atom_numbers + CONTROLLER::device_max_thread - 1) /
                    CONTROLLER::device_max_thread,
                CONTROLLER::device_max_thread, 0, NULL, pme_crd, g_crd,
                atom_id_l_g, this->atom_numbers);
        }
    }
#endif
}

// 目前只做单进程PME，暂时不考虑ghost，同样也不考虑get_local

void Particle_Mesh::Send_Recv_Force(CONTROLLER* controller, VECTOR* frc,
                                    VECTOR* pp_frc, int pp_atom_numbers)
{
    if (!PM_MPI_size)
    {
        return;
    }
#ifdef USE_MPI

    if (controller->MPI_rank < controller->PP_MPI_size)
    {
        D_MPI_GroupStart();
        D_MPI_Recv(frc, sizeof(VECTOR) * pp_atom_numbers, D_MPI_BYTE,
                   pp_corres_pm_rank, controller->MPI_rank,
                   controller->D_MPI_COMM_WORLD, pm_stream);
        D_MPI_GroupEnd();
#ifdef USE_GPU
        deviceStreamSynchronize(pm_stream);
#endif
        Launch_Device_Kernel(
            device_add_force,
            (pp_atom_numbers + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, pp_atom_numbers, 1, pp_frc,
            frc);
    }
    else
    {
        for (int i = 0; i < pm_corres_pp_num; ++i)
        {
            int pp_rank = pm_corres_pp_rank_set[i];
            int prefix = pm_corres_pp_atom_number_prefix[i];
            D_MPI_GroupStart();
            D_MPI_Send(frc + prefix,
                       sizeof(VECTOR) * pm_corres_pp_atom_number[i], D_MPI_BYTE,
                       pp_rank, pp_rank, controller->D_MPI_COMM_WORLD,
                       pm_stream);
            D_MPI_GroupEnd();
#ifdef USE_GPU
            deviceStreamSynchronize(pm_stream);
#endif
        }
    }
#endif
}

void Particle_Mesh::Create_Stream() { deviceStreamCreate(&pm_stream); }

void Particle_Mesh::Destroy_Stream() { deviceStreamDestroy(pm_stream); }

static __global__ void MPI_PME_Excluded_Force_With_Atom_Energy_Correction(
    const int atom_numbers, const VECTOR* crd, const LTMatrix3 cell,
    const LTMatrix3 rcell, const float* charge, const float pme_beta,
    const int* excluded_list_start, const int* excluded_list,
    const int* excluded_atom_numbers, VECTOR* frc, float* atom_ene,
    float* this_ene, LTMatrix3* atom_virial, int need_energy, int need_virial,
    const int* local2global, const int* global2local, const float factor)
{
    SIMPLE_DEVICE_FOR(local_i, atom_numbers)
    {
        int global_i = local_i;
        int excluded_numbers = excluded_atom_numbers[global_i];
        if (excluded_numbers > 0)
        {
            int list_start = excluded_list_start[global_i];
            int list_end = list_start + excluded_numbers;
            int local_j, global_j;

            float charge_i = charge[local_i];
            float charge_j;
            float dr_abs;
            float beta_dr;

            VECTOR r1 = crd[local_i], r2;
            VECTOR dr;
            float dr2;

            float frc_abs = 0.;
            VECTOR frc_lin;
            VECTOR frc_record = {0., 0., 0.};
            LTMatrix3 virial_record = {0, 0, 0, 0, 0, 0};
            float ene_lin = 0.;

            for (int i = list_start; i < list_end; i = i + 1)
            {
                global_j = excluded_list[i];

                // if(global_j<global_i)
                //     continue;
                //     //排除表重整化后，引入这一步用于排除一些重复计算
                local_j = global_j;
                r2 = crd[local_j];
                charge_j = charge[local_j];

                dr = Get_Periodic_Displacement(r2, r1, cell, rcell);
                dr2 = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;
                // 假设剔除表中的原子对距离总是小于cutoff的，正常体系

                dr_abs = sqrtf(dr2);
                beta_dr = pme_beta * dr_abs;
                frc_abs = beta_dr * TWO_DIVIDED_BY_SQRT_PI *
                              expf(-beta_dr * beta_dr) +
                          erfcf(beta_dr);
                frc_abs = (frc_abs - 1.) / dr2 / dr_abs;
                frc_abs = -charge_i * charge_j * frc_abs;
                frc_lin = frc_abs * dr;
                if (factor > 0.6f) atomicAdd(frc + local_j, -frc_lin);
                frc_record = frc_record + frc_lin;
                if (need_energy)
                    ene_lin -=
                        factor * charge_i * charge_j * erff(beta_dr) / dr_abs;
                if (need_virial)
                    virial_record =
                        virial_record -
                        factor * Get_Virial_From_Force_Dis(frc_lin, dr);
            }  // atom_j cycle
            atomicAdd(frc + local_i, frc_record);
            if (need_energy)
            {
                atomicAdd(atom_ene + local_i, ene_lin);
                this_ene[local_i] = ene_lin;
            }
            if (need_virial) atomicAdd(atom_virial + local_i, virial_record);
        }  // if need excluded
    }
}

void Particle_Mesh::MPI_PME_Excluded_Force_With_Atom_Energy(
    const int local_atom_numbers, const int* atom_local,
    const int* atom_local_id, const VECTOR* crd, const LTMatrix3 cell,
    const LTMatrix3 rcell, const float* charge, const int* excluded_list_start,
    const int* excluded_list, const int* excluded_atom_numbers, VECTOR* frc,
    int need_energy, float* atom_ene, int need_virial, LTMatrix3* atom_virial)
{
    if (is_initialized && calculate_excluded_part)
    {
        if (need_energy)
            deviceMemset(d_correction_atom_energy, 0,
                         sizeof(float) * local_atom_numbers);

        Launch_Device_Kernel(
            MPI_PME_Excluded_Force_With_Atom_Energy_Correction,
            (local_atom_numbers + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, local_atom_numbers, crd,
            cell, rcell, charge, beta, excluded_list_start, excluded_list,
            excluded_atom_numbers, frc, atom_ene, d_correction_atom_energy,
            atom_virial, need_energy, need_virial, atom_local, atom_local_id,
            exclude_factor);
    }
}

void Particle_Mesh::Step_Print(CONTROLLER* controller)
{
    if (!is_initialized) return;
    // 单进程, PM与PP共享同一进程情况
    if (CONTROLLER::MPI_size == 1 && CONTROLLER::PM_MPI_size == 1)
    {
        Sum_Of_List(d_correction_atom_energy, d_correction_ene, atom_numbers);
        Sum_Of_List(d_direct_atom_energy, d_direct_ene, atom_numbers);
        deviceMemcpy(&direct_ene, d_direct_ene, sizeof(float),
                     deviceMemcpyDeviceToHost);
        deviceMemcpy(&correction_ene, d_correction_ene, sizeof(float),
                     deviceMemcpyDeviceToHost);
        deviceMemcpy(&self_ene, d_self_ene, sizeof(float),
                     deviceMemcpyDeviceToHost);
        deviceMemcpy(&reciprocal_ene, d_reciprocal_ene, sizeof(float),
                     deviceMemcpyDeviceToHost);
        ee_ene = direct_ene + reciprocal_ene + self_ene + correction_ene;
        controller->Step_Print("PM", ee_ene, true);
        if (print_detail)
        {
            controller->Step_Print("PM_direct", direct_ene);
            controller->Step_Print("PM_reciprocal", reciprocal_ene);
            controller->Step_Print("PM_self", self_ene);
            controller->Step_Print("PM_correction", correction_ene);
        }
        return;
    }
    if (CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size)
    {
        Sum_Of_List(d_correction_atom_energy, d_correction_ene, atom_numbers);
        Sum_Of_List(d_direct_atom_energy, d_direct_ene, atom_numbers);
        self_ene = 0;
        reciprocal_ene = 0;
        deviceMemcpy(&direct_ene, d_direct_ene, sizeof(float),
                     deviceMemcpyDeviceToHost);
        deviceMemcpy(&correction_ene, d_correction_ene, sizeof(float),
                     deviceMemcpyDeviceToHost);
    }
    else
    {
        direct_ene = 0;
        correction_ene = 0;
        deviceMemcpy(&self_ene, d_self_ene, sizeof(float),
                     deviceMemcpyDeviceToHost);
        deviceMemcpy(&reciprocal_ene, d_reciprocal_ene, sizeof(float),
                     deviceMemcpyDeviceToHost);
    }
    ee_ene = direct_ene + reciprocal_ene + self_ene + correction_ene;
#ifdef USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, &ee_ene, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
#endif
    controller->Step_Print("PM", ee_ene, true);
    if (print_detail)
    {
#ifdef USE_MPI
        MPI_Allreduce(MPI_IN_PLACE, &self_ene, 1, MPI_FLOAT, MPI_SUM,
                      MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &reciprocal_ene, 1, MPI_FLOAT, MPI_SUM,
                      MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &direct_ene, 1, MPI_FLOAT, MPI_SUM,
                      MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &correction_ene, 1, MPI_FLOAT, MPI_SUM,
                      MPI_COMM_WORLD);
#endif
        controller->Step_Print("PM_direct", direct_ene);
        controller->Step_Print("PM_reciprocal", reciprocal_ene);
        controller->Step_Print("PM_self", self_ene);
        controller->Step_Print("PM_correction", correction_ene);
    }
}

void Particle_Mesh::reset_global_force(
    int no_direct_interaction_virtual_atom_numbers)
{
    deviceMemset(g_frc, 0,
                 sizeof(VECTOR) * (atom_numbers +
                                   no_direct_interaction_virtual_atom_numbers));
}

static __global__ void add_global_to_local_force(const VECTOR* g_frc,
                                                 VECTOR* l_frc,
                                                 const int* atom_id_g_l, int N)
{
#ifdef USE_GPU
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
#else
#pragma omp parallel for
    for (int i = 0; i < N; i++)
#endif
    {
        int l_id = atom_id_g_l[i];
        l_frc[l_id] = l_frc[l_id] + g_frc[i];
    }
}

void Particle_Mesh::add_force_g_to_l(VECTOR* l_frc)
{
    Launch_Device_Kernel(add_global_to_local_force,
                         (atom_numbers + CONTROLLER::device_max_thread - 1) /
                             CONTROLLER::device_max_thread,
                         CONTROLLER::device_max_thread, 0, NULL, g_frc, l_frc,
                         atom_id_g_l, atom_numbers);
}

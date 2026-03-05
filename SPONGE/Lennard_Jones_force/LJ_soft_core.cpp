#include "LJ_soft_core.h"

__global__ void Copy_LJ_Type_And_Mask_To_New_Crd(const int atom_numbers,
                                                 VECTOR_LJ_SOFT_TYPE* new_crd,
                                                 const int* LJ_type_A,
                                                 const int* LJ_type_B,
                                                 const int* mask)
{
#ifdef USE_GPU
    int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
    if (atom_i < atom_numbers)
#else
#pragma omp parallel for
    for (int atom_i = 0; atom_i < atom_numbers; atom_i++)
#endif
    {
        new_crd[atom_i].LJ_type = LJ_type_A[atom_i];
        new_crd[atom_i].LJ_type_B = LJ_type_B[atom_i];
        new_crd[atom_i].mask = mask[atom_i];
    }
}

static __global__ void device_add(float* variable, const float adder)
{
    variable[0] += adder;
}

__global__ void Copy_Crd_And_Charge_To_New_Crd(const int atom_numbers,
                                               const VECTOR* crd,
                                               VECTOR_LJ_SOFT_TYPE* new_crd,
                                               const float* charge)
{
#ifdef USE_GPU
    int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
    if (atom_i < atom_numbers)
#else
#pragma omp parallel for
    for (int atom_i = 0; atom_i < atom_numbers; atom_i++)
#endif
    {
        new_crd[atom_i].crd = crd[atom_i];
        new_crd[atom_i].charge = charge[atom_i];
    }
}

__global__ void Copy_Crd_And_Charge_To_New_Crd(const int atom_numbers,
                                               const VECTOR* crd,
                                               VECTOR_LJ_SOFT_TYPE* new_crd,
                                               const float* charge,
                                               const float* charge_BA)
{
#ifdef USE_GPU
    int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
    if (atom_i < atom_numbers)
#else
#pragma omp parallel for
    for (int atom_i = 0; atom_i < atom_numbers; atom_i++)
#endif
    {
        new_crd[atom_i].crd = crd[atom_i];
        new_crd[atom_i].charge = charge[atom_i];
        new_crd[atom_i].charge_BA = charge_BA[atom_i];
    }
}
__global__ void Copy_Crd_To_New_Crd(const int atom_numbers, const VECTOR* crd,
                                    VECTOR_LJ_SOFT_TYPE* new_crd)
{
#ifdef USE_GPU
    int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
    if (atom_i < atom_numbers)
#else
#pragma omp parallel for
    for (int atom_i = 0; atom_i < atom_numbers; atom_i++)
#endif
    {
        new_crd[atom_i].crd = crd[atom_i];
    }
}

static __global__ void Total_C6_Get(int atom_numbers, int* atom_lj_type_A,
                                    int* atom_lj_type_B, float* d_lj_Ab,
                                    float* d_lj_Bb, float* d_factor,
                                    const float lambda)
{
    int j;
    float temp_sum = 0.0;
    int xA, yA, xB, yB;
    int itype_A, jtype_A, itype_B, jtype_B, atom_pair_LJ_type_A,
        atom_pair_LJ_type_B;
    float lambda_ = 1.0 - lambda;
#ifdef USE_GPU
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < atom_numbers;
         i += gridDim.x * blockDim.x)
#else
#pragma omp parallel for firstprivate(                         \
        j, xA, yA, xB, yB, itype_A, jtype_A, itype_B, jtype_B, \
            atom_pair_LJ_type_A, atom_pair_LJ_type_B, lambda)  \
    reduction(+ : temp_sum)
    for (int i = 0; i < atom_numbers; i++)
#endif
    {
        itype_A = atom_lj_type_A[i];
        itype_B = atom_lj_type_B[i];
#ifdef USE_GPU
        for (j = blockIdx.y * blockDim.y + threadIdx.y; j < atom_numbers;
             j += gridDim.y * blockDim.y)
#else
        for (j = 0; j < atom_numbers; j++)
#endif
        {
            jtype_A = atom_lj_type_A[j];
            jtype_B = atom_lj_type_B[j];
            yA = (jtype_A - itype_A);
            xA = yA >> 31;
            yA = (yA ^ xA) - xA;
            xA = jtype_A + itype_A;
            jtype_A = (xA + yA) >> 1;
            xA = (xA - yA) >> 1;
            atom_pair_LJ_type_A = (jtype_A * (jtype_A + 1) >> 1) + xA;

            yB = (jtype_B - itype_B);
            xB = yB >> 31;
            yB = (yB ^ xB) - xB;
            xB = jtype_B + itype_B;
            jtype_B = (xB + yB) >> 1;
            xB = (xB - yB) >> 1;
            atom_pair_LJ_type_B = (jtype_B * (jtype_B + 1) >> 1) + xB;

            temp_sum += lambda_ * d_lj_Ab[atom_pair_LJ_type_A];
            temp_sum += lambda * d_lj_Bb[atom_pair_LJ_type_B];
        }
    }
    atomicAdd(d_factor, temp_sum);
}

static __global__ void Total_C6_B_A_Get(int atom_numbers, int* atom_lj_type_A,
                                        int* atom_lj_type_B, float* d_lj_Ab,
                                        float* d_lj_Bb, float* d_factor)
{
    int j;
    float temp_sum = 0.0;
    int xA, yA, xB, yB;
    int itype_A, jtype_A, itype_B, jtype_B, atom_pair_LJ_type_A,
        atom_pair_LJ_type_B;
#ifdef USE_GPU
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < atom_numbers;
         i += gridDim.x * blockDim.x)
#else
#pragma omp parallel for firstprivate(                         \
        j, xA, yA, xB, yB, itype_A, jtype_A, itype_B, jtype_B, \
            atom_pair_LJ_type_A, atom_pair_LJ_type_B) reduction(+ : temp_sum)
    for (int i = 0; i < atom_numbers; i++)
#endif
    {
        itype_A = atom_lj_type_A[i];
        itype_B = atom_lj_type_B[i];
#ifdef USE_GPU
        for (j = blockIdx.y * blockDim.y + threadIdx.y; j < atom_numbers;
             j += gridDim.y * blockDim.y)
#else
        for (j = 0; j < atom_numbers; j++)
#endif
        {
            jtype_A = atom_lj_type_A[j];
            jtype_B = atom_lj_type_B[j];
            yA = (jtype_A - itype_A);
            xA = yA >> 31;
            yA = (yA ^ xA) - xA;
            xA = jtype_A + itype_A;
            jtype_A = (xA + yA) >> 1;
            xA = (xA - yA) >> 1;
            atom_pair_LJ_type_A = (jtype_A * (jtype_A + 1) >> 1) + xA;

            yB = (jtype_B - itype_B);
            xB = yB >> 31;
            yB = (yB ^ xB) - xB;
            xB = jtype_B + itype_B;
            jtype_B = (xB + yB) >> 1;
            xB = (xB - yB) >> 1;
            atom_pair_LJ_type_B = (jtype_B * (jtype_B + 1) >> 1) + xB;

            temp_sum +=
                d_lj_Bb[atom_pair_LJ_type_B] - d_lj_Ab[atom_pair_LJ_type_A];
        }
    }
    atomicAdd(d_factor, temp_sum);
}

template <bool need_force, bool need_energy, bool need_virial,
          bool need_coulomb, bool need_du_dlambda>
static __global__ void Lennard_Jones_And_Direct_Coulomb_Soft_Core_CUDA(
    const int atom_numbers, const int solvent_numbers, const ATOM_GROUP* nl,
    const VECTOR_LJ_SOFT_TYPE* crd, const LTMatrix3 cell, const LTMatrix3 rcell,
    const float* LJ_type_AA, const float* LJ_type_AB, const float* LJ_type_BA,
    const float* LJ_type_BB, const float cutoff, VECTOR* frc,
    const float pme_beta, float* atom_energy, LTMatrix3* atom_virial,
    float* atom_direct_cf_energy, float* atom_du_dlambda_lj,
    float* atom_du_dlambda_direct, const float lambda, const float alpha,
    const float p, const float input_sigma_6, const float input_sigma_6_min,
    float* this_energy)
{
    float lambda_ = 1.0 - lambda;
    float alpha_lambda_p = alpha * powf(lambda, p);
    float alpha_lambda__p = alpha * powf(lambda_, p);
#ifdef USE_GPU
    int atom_i = blockDim.y * blockIdx.x + threadIdx.y;
    if (atom_i < atom_numbers - solvent_numbers)
#else
#pragma omp parallel for firstprivate(lambda, alpha_lambda_p, alpha_lambda__p)
    for (int atom_i = 0; atom_i < atom_numbers - solvent_numbers; atom_i++)
#endif
    {
        ATOM_GROUP nl_i = nl[atom_i];
        VECTOR_LJ_SOFT_TYPE r1 = crd[atom_i];
        VECTOR frc_record = {0., 0., 0.};
        LTMatrix3 virial_record = {0, 0, 0, 0, 0, 0};
        float energy_lj = 0.;
        float energy_coulomb = 0.;
        float du_dlambda_lj = 0.;
        float du_dlambda_direct = 0.;
#ifdef USE_GPU
        for (int j = threadIdx.x; j < nl_i.atom_numbers; j += blockDim.x)
#else
        for (int j = 0; j < nl_i.atom_numbers; j++)
#endif
        {
            int atom_j = nl_i.atom_serial[j];
            float ij_factor = atom_j < atom_numbers ? 1.0f : 0.5f;
            VECTOR_LJ_SOFT_TYPE r2 = crd[atom_j];
            VECTOR dr = Get_Periodic_Displacement(r2, r1, cell, rcell);
            float dr_abs = norm3df(dr.x, dr.y, dr.z);
            if (dr_abs < cutoff)
            {
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
                        if (atom_j < atom_numbers)
                            atomicAdd(frc + atom_j, -frc_lin);
                        if (need_virial)
                        {
                            virial_record =
                                virial_record -
                                ij_factor *
                                    Get_Virial_From_Force_Dis(frc_lin, dr);
                        }
                    }
                    if (need_coulomb && need_energy)
                    {
                        energy_coulomb +=
                            ij_factor *
                            Get_Direct_Coulomb_Energy(r1, r2, dr_abs, pme_beta);
                    }
                    if (need_energy)
                    {
                        energy_lj +=
                            ij_factor *
                            (lambda_ * Get_LJ_Energy(r1, r2, dr_abs, AA, AB) +
                             lambda * Get_LJ_Energy(r1, r2, dr_abs, BA, BB));
                    }
                    if (need_du_dlambda)
                    {
                        du_dlambda_lj +=
                            ij_factor * (Get_LJ_Energy(r1, r2, dr_abs, BA, BB) -
                                         Get_LJ_Energy(r1, r2, dr_abs, AA, AB));
                        if (need_coulomb)
                        {
                            du_dlambda_direct +=
                                ij_factor * Get_Direct_Coulomb_dU_dlambda(
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
                        if (atom_j < atom_numbers)
                            atomicAdd(frc + atom_j, -frc_lin);
                        if (need_virial)
                        {
                            virial_record =
                                virial_record -
                                ij_factor *
                                    Get_Virial_From_Force_Dis(frc_lin, dr);
                        }
                    }
                    if (need_coulomb && need_energy)
                    {
                        energy_coulomb +=
                            ij_factor *
                            (lambda_ * Get_Direct_Coulomb_Energy(
                                           r1, r2, dr_softcore_A, pme_beta) +
                             lambda * Get_Direct_Coulomb_Energy(
                                          r1, r2, dr_softcore_B, pme_beta));
                    }
                    if (need_energy)
                    {
                        energy_lj +=
                            ij_factor *
                            (lambda_ *
                                 Get_LJ_Energy(r1, r2, dr_softcore_A, AA, AB) +
                             lambda *
                                 Get_LJ_Energy(r1, r2, dr_softcore_B, BA, BB));
                    }
                    if (need_du_dlambda)
                    {
                        du_dlambda_lj +=
                            ij_factor *
                            (Get_LJ_Energy(r1, r2, dr_softcore_B, BA, BB) -
                             Get_LJ_Energy(r1, r2, dr_softcore_A, AA, AB));
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
                                ij_factor *
                                (Get_Direct_Coulomb_Energy(
                                     r1, r2, dr_softcore_B, pme_beta) -
                                 Get_Direct_Coulomb_Energy(
                                     r1, r2, dr_softcore_A, pme_beta));
                            du_dlambda_direct +=
                                ij_factor *
                                (Get_Soft_Core_dU_dlambda(
                                     Get_Direct_Coulomb_Force(
                                         r1, r2, dr_softcore_B, pme_beta),
                                     sigma_B, dr_softcore_B, alpha, p,
                                     lambda_) -
                                 Get_Soft_Core_dU_dlambda(
                                     Get_Direct_Coulomb_Force(
                                         r1, r2, dr_softcore_A, pme_beta),
                                     sigma_A, dr_softcore_A, alpha, p, lambda));
                            du_dlambda_direct +=
                                ij_factor *
                                (lambda * Get_Direct_Coulomb_dU_dlambda(
                                              r1, r2, dr_softcore_B, pme_beta) +
                                 lambda_ *
                                     Get_Direct_Coulomb_dU_dlambda(
                                         r1, r2, dr_softcore_A, pme_beta));
                        }
                    }
                }
            }
        }
        if (need_force)
        {
            Warp_Sum_To(frc + atom_i, frc_record, warpSize);
        }
        if (need_energy)
        {
            float energy_total = energy_lj;
            if (need_coulomb)
            {
                energy_total += energy_coulomb;
            }
            Warp_Sum_To(atom_energy + atom_i, energy_total, warpSize);
            Warp_Sum_To(this_energy + atom_i, energy_lj, warpSize);
        }
        if (need_coulomb && need_energy)
        {
            Warp_Sum_To(atom_direct_cf_energy + atom_i, energy_coulomb,
                        warpSize);
        }
        if (need_virial)
        {
            Warp_Sum_To(atom_virial + atom_i, virial_record, warpSize);
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

void LJ_SOFT_CORE::Initial(CONTROLLER* controller, float cutoff,
                           char* module_name)
{
    if (module_name == NULL)
    {
        strcpy(this->module_name, "LJ_soft_core");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }
    controller->printf(
        "START INITIALIZING FEP SOFT CORE FOR LJ AND COULOMB:\n");
    if (controller->Command_Exist(this->module_name, "in_file"))
    {
        if (controller->Command_Exist("lambda_lj"))
        {
            this->lambda = atof(controller->Command("lambda_lj"));
            controller->printf("    FEP lj lambda: %f\n", this->lambda);
        }
        else
        {
            char error_reason[CHAR_LENGTH_MAX];
            sprintf(error_reason,
                    "Reason:\n\t'lambda_lj' is required for the calculation of "
                    "LJ_soft_core\n");
            controller->Throw_SPONGE_Error(spongeErrorMissingCommand,
                                           "LJ_SOFT_CORE::Initial",
                                           error_reason);
        }

        if (controller->Command_Exist("soft_core_alpha"))
        {
            this->alpha = atof(controller->Command("soft_core_alpha"));
            controller->printf("    FEP soft core alpha: %f\n", this->alpha);
        }
        else
        {
            controller->printf(
                "    FEP soft core alpha is set to default value 0.5\n");
            this->alpha = 0.5;
        }

        if (controller->Command_Exist("soft_core_powfer"))
        {
            this->p = atof(controller->Command("soft_core_powfer"));
            controller->printf("    FEP soft core powfer: %f\n", this->p);
        }
        else
        {
            controller->printf(
                "    FEP soft core powfer is set to default value 1.0.\n");
            this->p = 1.0;
        }

        if (controller->Command_Exist("soft_core_sigma"))
        {
            this->sigma = atof(controller->Command("soft_core_sigma"));
            controller->printf("    FEP soft core sigma: %f\n", this->sigma);
        }
        else
        {
            controller->printf(
                "    FEP soft core sigma is set to default value 3.0\n");
            this->sigma = 3.0;
        }
        if (controller->Command_Exist("soft_core_sigma_min"))
        {
            this->sigma_min = atof(controller->Command("soft_core_sigma_min"));
            controller->printf("    FEP soft core sigma min: %f\n",
                               this->sigma_min);
        }
        else
        {
            controller->printf(
                "    FEP soft core sigma min is set to default value 0.0\n");
            this->sigma_min = 0.0;
        }

        FILE* fp = NULL;
        Open_File_Safely(&fp, controller->Command(this->module_name, "in_file"),
                         "r");

        int toscan = fscanf(fp, "%d %d %d", &atom_numbers, &atom_type_numbers_A,
                            &atom_type_numbers_B);
        controller->printf("    atom_numbers is %d\n", atom_numbers);
        controller->printf(
            "    atom_LJ_type_number_A is %d, atom_LJ_type_number_B is %d\n",
            atom_type_numbers_A, atom_type_numbers_B);
        pair_type_numbers_A =
            atom_type_numbers_A * (atom_type_numbers_A + 1) / 2;
        pair_type_numbers_B =
            atom_type_numbers_B * (atom_type_numbers_B + 1) / 2;
        LJ_Soft_Core_Malloc();

        for (int i = 0; i < pair_type_numbers_A; i++)
        {
            toscan = fscanf(fp, "%f", h_LJ_AA + i);
            h_LJ_AA[i] *= 12.0f;
        }
        for (int i = 0; i < pair_type_numbers_A; i++)
        {
            toscan = fscanf(fp, "%f", h_LJ_AB + i);
            h_LJ_AB[i] *= 6.0f;
        }
        for (int i = 0; i < pair_type_numbers_B; ++i)
        {
            toscan = fscanf(fp, "%f", h_LJ_BA + i);
            h_LJ_BA[i] *= 12.0f;
        }
        for (int i = 0; i < pair_type_numbers_B; ++i)
        {
            toscan = fscanf(fp, "%f", h_LJ_BB + i);
            h_LJ_BB[i] *= 6.0f;
        }
        for (int i = 0; i < atom_numbers; i++)
        {
            toscan =
                fscanf(fp, "%d %d", h_atom_LJ_type_A + i, h_atom_LJ_type_B + i);
        }
        fclose(fp);

        if (controller->Command_Exist("subsys_division_in_file"))
        {
            FILE* fp = NULL;
            controller->printf(
                "    Start reading subsystem division information:\n");
            Open_File_Safely(
                &fp, controller->Command("subsys_division_in_file"), "r");
            int atom_numbers = 0;
            char lin[CHAR_LENGTH_MAX];
            char* get_ret = fgets(lin, CHAR_LENGTH_MAX, fp);
            toscan = sscanf(lin, "%d", &atom_numbers);
            if (this->atom_numbers > 0 && this->atom_numbers != atom_numbers)
            {
                controller->Throw_SPONGE_Error(
                    spongeErrorConflictingCommand, "LJ_SOFT_CORE::Initial",
                    "Reason:\n\t'atom_numbers' (the number of atoms) is "
                    "diiferent in different input files\n");
            }
            else if (this->atom_numbers == 0)
            {
                this->atom_numbers = atom_numbers;
            }
            for (int i = 0; i < atom_numbers; i++)
            {
                toscan = fscanf(fp, "%d", &h_subsys_division[i]);
            }
            controller->printf(
                "    End reading subsystem division information\n\n");
            fclose(fp);
        }
        else
        {
            controller->printf("    subsystem mask is set to 0 as default\n");
            for (int i = 0; i < atom_numbers; i++)
            {
                h_subsys_division[i] = 0;
            }
        }

        Parameter_Host_To_Device();
        is_initialized = 1;
        alpha_lambda_p = alpha * powf(lambda, p);
        alpha_lambda_p_ = alpha * powf(1 - lambda, p);
        sigma_6 = powf(sigma, 6);
        sigma_6_min = powf(sigma_min, 6);
        alpha_lambda_p_1 = alpha * powf(lambda, p - 1);
        alpha_lambda_p_1_ = alpha * powf(1.0 - lambda, p - 1);
    }
    if (is_initialized)
    {
        this->cutoff = cutoff;
        Device_Malloc_Safely((void**)&crd_with_parameters,
                             sizeof(VECTOR_LJ_SOFT_TYPE) * atom_numbers);
        Launch_Device_Kernel(
            Copy_LJ_Type_And_Mask_To_New_Crd,
            (atom_numbers + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, atom_numbers,
            crd_with_parameters, d_atom_LJ_type_A, d_atom_LJ_type_B,
            d_subsys_division);
        controller->printf("    Start initializing long range LJ correction\n");
        long_range_factor = 0;
        float* d_factor = NULL;
        Device_Malloc_Safely((void**)&d_factor, sizeof(float));
        deviceMemset(d_factor, 0, sizeof(float));

        dim3 gridSize = {4, 4};
        dim3 blockSize = {32, 32};
        Launch_Device_Kernel(Total_C6_Get, gridSize, blockSize, 0, NULL,
                             atom_numbers, d_atom_LJ_type_A, d_atom_LJ_type_B,
                             d_LJ_AB, d_LJ_BB, d_factor, this->lambda);

        deviceMemcpy(&long_range_factor, d_factor, sizeof(float),
                     deviceMemcpyDeviceToHost);
        deviceMemset(d_factor, 0, sizeof(float));

        Launch_Device_Kernel(Total_C6_B_A_Get, gridSize, blockSize, 0, NULL,
                             atom_numbers, d_atom_LJ_type_A, d_atom_LJ_type_B,
                             d_LJ_AB, d_LJ_BB, d_factor);
        deviceMemcpy(&long_range_factor_TI, d_factor, sizeof(float),
                     deviceMemcpyDeviceToHost);
        Free_Single_Device_Pointer((void**)&d_factor);

        long_range_factor *=
            -2.0f / 3.0f * CONSTANT_Pi / cutoff / cutoff / cutoff / 6.0f;
        long_range_factor_TI *=
            -2.0f / 3.0f * CONSTANT_Pi / cutoff / cutoff / cutoff / 6.0f;
        controller->printf("        long range correction factor is: %e\n",
                           long_range_factor);
        controller->printf("    End initializing long range LJ correction\n");
    }
    if (is_initialized && !is_controller_printf_initialized)
    {
        controller->Step_Print_Initial("LJ_soft", "%.2f");
        controller->Step_Print_Initial("LJ_soft_short", "%.2f");
        controller->Step_Print_Initial("LJ_soft_long", "%.2f");
        is_controller_printf_initialized = 1;
        controller->printf("    structure last modify date is %d\n",
                           last_modify_date);
    }
    controller->printf(
        "END INITIALIZING LENNADR JONES SOFT CORE INFORMATION\n\n");
}

void LJ_SOFT_CORE::LJ_Soft_Core_Malloc()
{
    Malloc_Safely((void**)&h_LJ_energy_atom, sizeof(float) * atom_numbers);
    Malloc_Safely((void**)&h_atom_LJ_type_A, sizeof(int) * atom_numbers);
    Malloc_Safely((void**)&h_atom_LJ_type_B, sizeof(int) * atom_numbers);
    Malloc_Safely((void**)&h_LJ_AA, sizeof(float) * pair_type_numbers_A);
    Malloc_Safely((void**)&h_LJ_AB, sizeof(float) * pair_type_numbers_A);
    Malloc_Safely((void**)&h_LJ_BA, sizeof(float) * pair_type_numbers_B);
    Malloc_Safely((void**)&h_LJ_BB, sizeof(float) * pair_type_numbers_B);
    Malloc_Safely((void**)&h_subsys_division, sizeof(int) * atom_numbers);

    Device_Malloc_And_Copy_Safely((void**)&d_LJ_energy_sum, &h_LJ_energy_sum,
                                  sizeof(float));
    Device_Malloc_And_Copy_Safely((void**)&d_LJ_energy_atom, h_LJ_energy_atom,
                                  sizeof(float) * atom_numbers);

    Malloc_Safely((void**)&h_LJ_energy_atom_intersys,
                  sizeof(float) * atom_numbers);
    Malloc_Safely((void**)&h_LJ_energy_atom_intrasys,
                  sizeof(float) * atom_numbers);

    Device_Malloc_And_Copy_Safely((void**)&d_LJ_energy_atom_intersys,
                                  h_LJ_energy_atom_intersys,
                                  sizeof(float) * atom_numbers);
    Device_Malloc_And_Copy_Safely((void**)&d_LJ_energy_atom_intrasys,
                                  h_LJ_energy_atom_intrasys,
                                  sizeof(float) * atom_numbers);

    Device_Malloc_And_Copy_Safely((void**)&d_direct_ene_sum_intersys,
                                  &h_direct_ene_sum_intersys, sizeof(float));
    Device_Malloc_And_Copy_Safely((void**)&d_direct_ene_sum_intrasys,
                                  &h_direct_ene_sum_intrasys, sizeof(float));
    Device_Malloc_And_Copy_Safely((void**)&d_LJ_energy_sum_intersys,
                                  &h_LJ_energy_sum_intersys, sizeof(float));
    Device_Malloc_And_Copy_Safely((void**)&d_LJ_energy_sum_intrasys,
                                  &h_LJ_energy_sum_intrasys, sizeof(float));

    Malloc_Safely((void**)&h_sigma_of_dH_dlambda_lj, sizeof(float));
    Malloc_Safely((void**)&h_sigma_of_dH_dlambda_direct, sizeof(float));

    Device_Malloc_And_Copy_Safely((void**)&d_sigma_of_dH_dlambda_lj,
                                  h_sigma_of_dH_dlambda_lj, sizeof(float));
    Device_Malloc_And_Copy_Safely((void**)&d_sigma_of_dH_dlambda_direct,
                                  h_sigma_of_dH_dlambda_direct, sizeof(float));
}

void LJ_SOFT_CORE::Parameter_Host_To_Device()
{
    Device_Malloc_And_Copy_Safely((void**)&d_LJ_AA, h_LJ_AA,
                                  sizeof(float) * pair_type_numbers_A);
    Device_Malloc_And_Copy_Safely((void**)&d_LJ_AB, h_LJ_AB,
                                  sizeof(float) * pair_type_numbers_A);
    Device_Malloc_And_Copy_Safely((void**)&d_LJ_BA, h_LJ_BA,
                                  sizeof(float) * pair_type_numbers_B);
    Device_Malloc_And_Copy_Safely((void**)&d_LJ_BB, h_LJ_BB,
                                  sizeof(float) * pair_type_numbers_B);

    Device_Malloc_And_Copy_Safely((void**)&d_atom_LJ_type_A, h_atom_LJ_type_A,
                                  sizeof(int) * atom_numbers);
    Device_Malloc_And_Copy_Safely((void**)&d_atom_LJ_type_B, h_atom_LJ_type_B,
                                  sizeof(int) * atom_numbers);
    Device_Malloc_And_Copy_Safely((void**)&d_subsys_division, h_subsys_division,
                                  sizeof(int) * atom_numbers);
    Device_Malloc_Safely((void**)&crd_with_LJ_parameters_local,
                         sizeof(VECTOR_LJ_SOFT_TYPE) * atom_numbers);
}

void LJ_SOFT_CORE::LJ_Soft_Core_PME_Direct_Force_With_Atom_Energy_And_Virial(
    const int atom_numbers, const int local_atom_numbers,
    const int solvent_numbers, const int ghost_numbers, const VECTOR* crd,
    const float* charge, VECTOR* frc, const LTMatrix3 cell,
    const LTMatrix3 rcell, const ATOM_GROUP* nl, const float pme_beta,
    const int need_atom_energy, float* atom_energy, const int need_virial,
    LTMatrix3* atom_lj_virial, float* atom_direct_pme_energy)
{
    if (is_initialized)
    {
        Launch_Device_Kernel(
            Copy_Crd_And_Charge_To_New_Crd,
            (this->atom_numbers + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL,
            this->local_atom_numbers + this->ghost_numbers, crd,
            crd_with_LJ_parameters_local, charge);

        if (need_atom_energy)
        {
            deviceMemset(d_LJ_energy_atom, 0, sizeof(float) * atom_numbers);
            deviceMemset(atom_direct_pme_energy, 0,
                         sizeof(float) * atom_numbers);
        }

        if (atom_numbers == 0 || local_atom_numbers == 0) return;

        dim3 blockSize = {
            CONTROLLER::device_warp,
            CONTROLLER::device_max_thread / CONTROLLER::device_warp};
        dim3 gridSize = (atom_numbers + blockSize.y - 1) / blockSize.y;

        auto f =
            Lennard_Jones_And_Direct_Coulomb_Soft_Core_CUDA<true, false, false,
                                                            true, false>;

        if (!need_atom_energy && !need_virial)
        {
            f = Lennard_Jones_And_Direct_Coulomb_Soft_Core_CUDA<
                true, false, false, true, false>;
        }
        else if (need_atom_energy && !need_virial)
        {
            f = Lennard_Jones_And_Direct_Coulomb_Soft_Core_CUDA<
                true, true, false, true, false>;
        }
        else if (!need_atom_energy && need_virial)
        {
            f = Lennard_Jones_And_Direct_Coulomb_Soft_Core_CUDA<
                true, false, true, true, false>;
        }
        else
        {
            f = Lennard_Jones_And_Direct_Coulomb_Soft_Core_CUDA<
                true, true, true, true, false>;
        }
        Launch_Device_Kernel(
            f, gridSize, blockSize, 0, NULL, local_atom_numbers,
            solvent_numbers, nl, crd_with_LJ_parameters_local, cell, rcell,
            d_LJ_AA, d_LJ_AB, d_LJ_BA, d_LJ_BB, cutoff, frc, pme_beta,
            atom_energy, atom_lj_virial, atom_direct_pme_energy, NULL, NULL,
            lambda, alpha, p, sigma_6, sigma_6_min, d_LJ_energy_atom);
    }
}

float LJ_SOFT_CORE::Get_Partial_H_Partial_Lambda_With_Columb_Direct(
    const int solvent_numbers, const VECTOR* crd, const LTMatrix3 cell,
    const LTMatrix3 rcell, const float* charge, const ATOM_GROUP* nl,
    const float* charge_B_A, const float pme_beta, const int charge_perturbated)
{
    if (is_initialized)
    {
        Launch_Device_Kernel(
            Copy_Crd_And_Charge_To_New_Crd,
            (this->atom_numbers + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL,
            this->local_atom_numbers + this->ghost_numbers, crd,
            crd_with_parameters, charge, charge_B_A);

        deviceMemset(d_sigma_of_dH_dlambda_lj, 0, sizeof(float));

        dim3 blockSize = {
            CONTROLLER::device_warp,
            CONTROLLER::device_max_thread / CONTROLLER::device_warp};
        dim3 gridSize = (atom_numbers + blockSize.y - 1) / blockSize.y;
        auto f =
            Lennard_Jones_And_Direct_Coulomb_Soft_Core_CUDA<false, false, false,
                                                            true, true>;

        if (charge_perturbated > 0)
        {
            deviceMemset(d_sigma_of_dH_dlambda_direct, 0, sizeof(float));
            f = Lennard_Jones_And_Direct_Coulomb_Soft_Core_CUDA<
                false, false, false, true, true>;
        }
        else
        {
            f = Lennard_Jones_And_Direct_Coulomb_Soft_Core_CUDA<
                false, false, false, false, true>;
        }
        Launch_Device_Kernel(
            f, gridSize, blockSize, 0, NULL, local_atom_numbers,
            solvent_numbers, nl, crd_with_LJ_parameters_local, cell, rcell,
            d_LJ_AA, d_LJ_AB, d_LJ_BA, d_LJ_BB, cutoff, NULL, pme_beta, NULL,
            NULL, NULL, d_sigma_of_dH_dlambda_lj, d_sigma_of_dH_dlambda_direct,
            lambda, alpha, p, sigma_6, sigma_6_min, NULL);

        deviceMemcpy(h_sigma_of_dH_dlambda_lj, d_sigma_of_dH_dlambda_lj,
                     sizeof(float), deviceMemcpyDeviceToHost);
        deviceMemcpy(h_sigma_of_dH_dlambda_direct, d_sigma_of_dH_dlambda_direct,
                     sizeof(float), deviceMemcpyDeviceToHost);
#ifdef USE_MPI
        MPI_Allreduce(MPI_IN_PLACE, h_sigma_of_dH_dlambda_lj, 1, MPI_FLOAT,
                      MPI_SUM, CONTROLLER::pp_comm);
        MPI_Allreduce(MPI_IN_PLACE, h_sigma_of_dH_dlambda_direct, 1, MPI_FLOAT,
                      MPI_SUM, CONTROLLER::pp_comm);
#endif
        return *h_sigma_of_dH_dlambda_lj +
               long_range_factor_TI / cell.a11 / cell.a22 / cell.a33;
    }
    else
    {
        return NAN;
    }
}

void LJ_SOFT_CORE::Step_Print(CONTROLLER* controller)
{
    if (!is_initialized || CONTROLLER::MPI_rank >= CONTROLLER::PP_MPI_size)
        return;
    Sum_Of_List(d_LJ_energy_atom, d_LJ_energy_sum, atom_numbers);
    deviceMemcpy(&h_LJ_energy_sum, d_LJ_energy_sum, sizeof(float),
                 deviceMemcpyDeviceToHost);
#ifdef USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, &h_LJ_energy_sum, 1, MPI_FLOAT, MPI_SUM,
                  CONTROLLER::pp_comm);
#endif
    controller->Step_Print("LJ_soft_short", h_LJ_energy_sum);
    controller->Step_Print("LJ_soft_long", h_LJ_long_energy);
    controller->Step_Print("LJ_soft", h_LJ_energy_sum + h_LJ_long_energy, true);
}

static __global__ void Long_Range_Virial_Correction(LTMatrix3* d_virial,
                                                    const float factor)
{
    d_virial->a11 += factor;
    d_virial->a22 += factor;
    d_virial->a33 += factor;
}

void LJ_SOFT_CORE::Long_Range_Correction(int need_pressure, LTMatrix3* d_virial,
                                         int need_potential, float* d_potential,
                                         const float volume)
{
    if (is_initialized && CONTROLLER::PP_MPI_rank == 0)
    {
        if (need_pressure > 0)
        {
            Launch_Device_Kernel(Long_Range_Virial_Correction, 1, 1, 0, NULL,
                                 d_virial, 2 * long_range_factor / volume);
        }
        if (need_potential > 0)
        {
            Launch_Device_Kernel(device_add, 1, 1, 0, NULL, d_potential,
                                 long_range_factor / volume);
            h_LJ_long_energy = long_range_factor / volume;
        }
    }
}

static __global__ void get_local_device(
    int* atom_local, int local_atom_numbers, int ghost_numbers,
    int* d_atom_LJ_type_A, int* d_atom_LJ_type_B, int* d_mask,
    VECTOR_LJ_SOFT_TYPE* crd_with_LJ_parameters_local)
{
    SIMPLE_DEVICE_FOR(i, local_atom_numbers + ghost_numbers)
    {
        int atom_i = atom_local[i];
        crd_with_LJ_parameters_local[i].LJ_type = d_atom_LJ_type_A[atom_i];
        crd_with_LJ_parameters_local[i].LJ_type_B = d_atom_LJ_type_B[atom_i];
        crd_with_LJ_parameters_local[i].mask = d_mask[atom_i];
    }
}

void LJ_SOFT_CORE::Get_Local(int* atom_local, int local_atom_numbers,
                             int ghost_numbers)
{
    if (!is_initialized) return;
    this->local_atom_numbers = local_atom_numbers;
    this->ghost_numbers = ghost_numbers;
    Launch_Device_Kernel(get_local_device,
                         (local_atom_numbers + ghost_numbers +
                          CONTROLLER::device_max_thread - 1) /
                             CONTROLLER::device_max_thread,
                         CONTROLLER::device_max_thread, 0, NULL, atom_local,
                         local_atom_numbers, ghost_numbers, d_atom_LJ_type_A,
                         d_atom_LJ_type_B, d_subsys_division,
                         crd_with_LJ_parameters_local);
}

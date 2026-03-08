#include "solvent_LJ.h"

__global__ void Vector_Soft_Core_To_Hard_Core(
    int atom_numbers, VECTOR_LJ* hard_core_crd,
    const VECTOR_LJ_SOFT_TYPE* soft_core_crd)
{
#ifdef USE_GPU
    int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
    if (atom_i < atom_numbers)
#else
#pragma omp parallel for
    for (int atom_i = 0; atom_i < atom_numbers; atom_i++)
#endif
    {
        VECTOR_LJ new_crd;
        VECTOR_LJ_SOFT_TYPE crd = soft_core_crd[atom_i];
        new_crd.crd = crd.crd;
        new_crd.charge = crd.charge;
        new_crd.LJ_type = crd.LJ_type;
        hard_core_crd[atom_i] = new_crd;
    }
}

#ifdef USE_GPU
template <int WAT_POINTS, bool need_force, bool need_energy, bool need_virial,
          bool need_coulomb>
static __global__ void Lennard_Jones_And_Direct_Coulomb_Device(
    const int atom_numbers, const ATOM_GROUP* nl,
    const int solvent_start_residue, const int res_numbers,
    const int* d_res_start, const VECTOR_LJ* crd, const LTMatrix3 cell,
    const LTMatrix3 rcell, const float* LJ_type_A, const float* LJ_type_B,
    const float cutoff, VECTOR* frc, const float pme_beta, float* atom_energy,
    LTMatrix3* atom_lj_virial, float* atom_direct_cf_energy, float* this_energy)
{
    __shared__ float r1s_x[128];
    __shared__ float r1s_y[128];
    __shared__ float r1s_z[128];
    __shared__ int r1s_lj_type[128];
    __shared__ float r1s_charge[128];
    int residue_i =
        blockDim.y * blockIdx.x + threadIdx.y + solvent_start_residue;
    if (residue_i < res_numbers)
    {
        VECTOR frc_record[4] = {{0.0f, 0.0f, 0.0f},
                                {0.0f, 0.0f, 0.0f},
                                {0.0f, 0.0f, 0.0f},
                                {0.0f, 0.0f, 0.0f}};
        VECTOR frc_record_j;
        int atom_i = d_res_start[residue_i];
        if (threadIdx.x < WAT_POINTS)
        {
            const int shared_idx = threadIdx.y * WAT_POINTS + threadIdx.x;
            VECTOR_LJ r1 = crd[atom_i + threadIdx.x];
            r1s_x[shared_idx] = r1.crd.x;
            r1s_y[shared_idx] = r1.crd.y;
            r1s_z[shared_idx] = r1.crd.z;
            r1s_lj_type[shared_idx] = r1.LJ_type;
            r1s_charge[shared_idx] = r1.charge;
        }
        deviceSyncWarp();
        ATOM_GROUP nl_i = nl[atom_i];
        LTMatrix3 virial_record = {0, 0, 0, 0, 0, 0};
        float energy_lj = 0.;
        float energy_coulomb = 0.;
        float energy_total = 0.0f;
        for (int j = threadIdx.x; j < nl_i.atom_numbers; j += blockDim.x)
        {
            int atom_j = nl_i.atom_serial[j];
            float ij_factor = atom_j < atom_numbers ? 1.0f : 0.5f;
            VECTOR_LJ r2 = crd[atom_j];
            frc_record_j = {0.0f, 0.0f, 0.0f};
            for (int i = 0; i < WAT_POINTS; i++)
            {
                const int shared_idx = threadIdx.y * WAT_POINTS + i;
                VECTOR_LJ r1 = {
                    {r1s_x[shared_idx], r1s_y[shared_idx], r1s_z[shared_idx]},
                    r1s_lj_type[shared_idx],
                    r1s_charge[shared_idx]};
                VECTOR dr = Get_Periodic_Displacement(r2, r1, cell, rcell);
                float dr_abs = norm3df(dr.x, dr.y, dr.z);
                if (dr_abs < cutoff)
                {
                    int atom_pair_LJ_type = Get_LJ_Type(r1.LJ_type, r2.LJ_type);
                    float A = LJ_type_A[atom_pair_LJ_type];
                    float B = LJ_type_B[atom_pair_LJ_type];
                    if (need_force)
                    {
                        float frc_abs = Get_LJ_Force(r1, r2, dr_abs, A, B);
                        if (need_coulomb)
                        {
                            float frc_cf_abs = Get_Direct_Coulomb_Force(
                                r1, r2, dr_abs, pme_beta);
                            frc_abs = frc_abs - frc_cf_abs;
                        }
                        VECTOR frc_lin = frc_abs * dr;
                        frc_record[i] = frc_record[i] + frc_lin;
                        frc_record_j = frc_record_j - frc_lin;
                        if (need_virial)
                        {
                            virial_record =
                                virial_record -
                                ij_factor *
                                    Get_Virial_From_Force_Dis(frc_lin, dr);
                        }
                    }
                    if (need_energy)
                    {
                        energy_lj +=
                            ij_factor * Get_LJ_Energy(r1, r2, dr_abs, A, B);
                        if (need_coulomb)
                        {
                            energy_coulomb +=
                                ij_factor * Get_Direct_Coulomb_Energy(
                                                r1, r2, dr_abs, pme_beta);
                        }
                    }
                }
            }
            if (need_force && atom_j < atom_numbers)
            {
                atomicAdd(frc + atom_j, frc_record_j);
            }
        }
        energy_total = energy_lj + energy_coulomb;
        if (need_force)
        {
            for (int i = 0; i < WAT_POINTS; i++)
            {
                Warp_Sum_To(frc + atom_i + i, frc_record[i], warpSize);
            }
        }
        if (need_energy)
        {
            Warp_Sum_To(atom_energy + atom_i, energy_total, warpSize);
            Warp_Sum_To(this_energy + atom_i, energy_lj, warpSize);
            if (need_coulomb)
                Warp_Sum_To(atom_direct_cf_energy + atom_i, energy_coulomb,
                            warpSize);
        }
        if (need_virial)
        {
            Warp_Sum_To(atom_lj_virial + atom_i, virial_record, warpSize);
        }
    }
}
#endif

void SOLVENT_LENNARD_JONES::Initial(CONTROLLER* controller,
                                    LENNARD_JONES_INFORMATION* lj,
                                    LJ_SOFT_CORE* lj_soft,
                                    MD_INFORMATION* md_info,
                                    bool default_enable,
                                    const char* module_name)
{
    if (module_name == NULL)
    {
        strcpy(this->module_name, "solvent_LJ");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }
    bool enable;
#ifdef USE_GPU
    if (!controller->Command_Exist(this->module_name))
    {
        printf("md_info->ug.ug_numbers: %d\n", md_info->ug.ug_numbers);
        enable = default_enable && (md_info->ug.ug_numbers >= 10);
    }
    else
    {
        enable = controller->Get_Bool(this->module_name,
                                      "SOLVENT_LENNARD_JONES::Initial");
    }
    if (enable)
    {
        water_points = md_info->ug.ug[md_info->ug.ug_numbers - 1].atom_numbers;
    }
    if (water_points != 3 && water_points != 4)
    {
        enable = false;
    }
#else
    enable = false;
#endif
    if (enable)
    {
        controller->printf("START INITIALIZING SOLVENT LJ:\n");
        this->lj_info = lj;
        this->lj_soft_info = lj_soft;
        solvent_numbers = 0;
        solvent_start = md_info->ug.ug_numbers;
        for (int i = md_info->ug.ug_numbers - 1; i >= 0; i -= 1)
        {
            int res_atom_numbers = md_info->ug.ug[i].atom_numbers;
            float mass_O = md_info->h_mass[md_info->ug.ug[i].atom_serial[0]];
            float mass_H1 = md_info->h_mass[md_info->ug.ug[i].atom_serial[1]];
            float mass_H2 = md_info->h_mass[md_info->ug.ug[i].atom_serial[2]];
            if (res_atom_numbers == water_points &&
                (mass_O > 15.9f && mass_O < 16.1f) &&
                (mass_H1 > 1.007f && mass_H1 < 1.009f) &&
                (mass_H2 > 1.007f && mass_H2 < 1.009f))
            {
                solvent_numbers += res_atom_numbers;
                solvent_start -= 1;
            }
            else
            {
                break;
            }
        }
        controller->printf("    the solvent is %d-point\n", water_points);
        controller->printf(
            "    the number of solvent atoms is %d (started from Residue "
            "#%d)\n",
            solvent_numbers, solvent_start);
        if (solvent_numbers > 0)
        {
            is_initialized = 1;
            if (lj_soft_info->is_initialized)
            {
                Device_Malloc_Safely(
                    (void**)&soft_to_hard_crd,
                    sizeof(VECTOR_LJ) * lj_soft_info->atom_numbers);
            }
            Device_Malloc_Safely((void**)&d_solvent_start_local, sizeof(int));
            Device_Malloc_Safely((void**)&d_local_solvent_numbers, sizeof(int));
        }
    }
    if (!is_initialized)
    {
        solvent_numbers = 0;
        local_solvent_numbers = 0;
        controller->printf("SOLVENT LJ IS NOT INITIALIZED\n\n");
    }
    else if (!is_controller_printf_initialized)
    {
        is_controller_printf_initialized = 1;
        controller->printf("    structure last modify date is %d\n",
                           last_modify_date);
        controller->printf("END INITIALIZING SOLVENT LJ\n\n");
    }
    else
    {
        controller->printf("END INITIALIZING SOLVENT LJ\n\n");
    }
}

/*
    从输入读入local信息
    atom_numbers: local_原子数
    residue_numbers: local_残基数
    d_res_start: local_残基起始位置
*/
void SOLVENT_LENNARD_JONES::LJ_PME_Direct_Force_With_Atom_Energy_And_Virial(
    const int atom_numbers, const int residue_numbers, const int* d_res_start,
    const VECTOR* crd, const float* charge, VECTOR* frc, const LTMatrix3 cell,
    const LTMatrix3 rcell, const ATOM_GROUP* nl, const float pme_beta,
    const int need_atom_energy, float* atom_energy, const int need_virial,
    LTMatrix3* atom_lj_virial, float* atom_direct_pme_energy)
{
    if (is_initialized)
    {
#ifdef USE_GPU
        dim3 blockSize = {
            static_cast<unsigned int>(CONTROLLER::device_warp),
            static_cast<unsigned int>(min(
                8u, static_cast<unsigned int>(CONTROLLER::device_max_thread /
                                              CONTROLLER::device_warp)))};
        dim3 gridSize(static_cast<unsigned int>(
            (residue_numbers - solvent_start_local + blockSize.y - 1) /
            blockSize.y));

        switch (water_points)
        {
            case 3:
                if (lj_info->is_initialized)
                {
                    auto f =
                        Lennard_Jones_And_Direct_Coulomb_Device<3, true, false,
                                                                false, true>;
                    if (!need_atom_energy && !need_virial)
                    {
                        f = Lennard_Jones_And_Direct_Coulomb_Device<
                            3, true, false, false, true>;
                    }
                    else if (need_atom_energy && !need_virial)
                    {
                        f = Lennard_Jones_And_Direct_Coulomb_Device<
                            3, true, true, false, true>;
                    }
                    else if (!need_atom_energy && need_virial)
                    {
                        f = Lennard_Jones_And_Direct_Coulomb_Device<
                            3, true, false, true, true>;
                    }
                    else
                    {
                        f = Lennard_Jones_And_Direct_Coulomb_Device<
                            3, true, true, true, true>;
                    }
                    Launch_Device_Kernel(
                        f, gridSize, blockSize, 0, NULL, atom_numbers, nl,
                        solvent_start_local, residue_numbers, d_res_start,
                        lj_info->crd_with_LJ_parameters_local, cell, rcell,
                        lj_info->d_LJ_A, lj_info->d_LJ_B, lj_info->cutoff, frc,
                        pme_beta, atom_energy, atom_lj_virial,
                        atom_direct_pme_energy, lj_info->d_LJ_energy_atom);
                }
                else if (lj_soft_info->is_initialized)
                {
                    Launch_Device_Kernel(
                        Vector_Soft_Core_To_Hard_Core,
                        (atom_numbers + CONTROLLER::device_max_thread - 1) /
                            CONTROLLER::device_max_thread,
                        CONTROLLER::device_max_thread, 0, NULL, atom_numbers,
                        soft_to_hard_crd,
                        lj_soft_info->crd_with_LJ_parameters_local);

                    auto f =
                        Lennard_Jones_And_Direct_Coulomb_Device<3, true, false,
                                                                false, true>;
                    if (!need_atom_energy && !need_virial)
                    {
                        f = Lennard_Jones_And_Direct_Coulomb_Device<
                            3, true, false, false, true>;
                    }
                    else if (need_atom_energy && !need_virial)
                    {
                        f = Lennard_Jones_And_Direct_Coulomb_Device<
                            3, true, true, false, true>;
                    }
                    else if (!need_atom_energy && need_virial)
                    {
                        f = Lennard_Jones_And_Direct_Coulomb_Device<
                            3, true, false, true, true>;
                    }
                    else
                    {
                        f = Lennard_Jones_And_Direct_Coulomb_Device<
                            3, true, true, true, true>;
                    }
                    Launch_Device_Kernel(
                        f, gridSize, blockSize, 0, NULL, atom_numbers, nl,
                        solvent_start_local, residue_numbers, d_res_start,
                        soft_to_hard_crd, cell, rcell, lj_soft_info->d_LJ_AA,
                        lj_soft_info->d_LJ_AB, lj_soft_info->cutoff, frc,
                        pme_beta, atom_energy, atom_lj_virial,
                        atom_direct_pme_energy, lj_soft_info->d_LJ_energy_atom);
                }
                break;
            case 4:
                if (lj_info->is_initialized)
                {
                    auto f =
                        Lennard_Jones_And_Direct_Coulomb_Device<4, true, false,
                                                                false, true>;
                    if (!need_atom_energy && !need_virial)
                    {
                        f = Lennard_Jones_And_Direct_Coulomb_Device<
                            4, true, false, false, true>;
                    }
                    else if (need_atom_energy && !need_virial)
                    {
                        f = Lennard_Jones_And_Direct_Coulomb_Device<
                            4, true, true, false, true>;
                    }
                    else if (!need_atom_energy && need_virial)
                    {
                        f = Lennard_Jones_And_Direct_Coulomb_Device<
                            4, true, false, true, true>;
                    }
                    else
                    {
                        f = Lennard_Jones_And_Direct_Coulomb_Device<
                            4, true, true, true, true>;
                    }
                    Launch_Device_Kernel(
                        f, gridSize, blockSize, 0, NULL, atom_numbers, nl,
                        solvent_start_local, residue_numbers, d_res_start,
                        lj_info->crd_with_LJ_parameters_local, cell, rcell,
                        lj_info->d_LJ_A, lj_info->d_LJ_B, lj_info->cutoff, frc,
                        pme_beta, atom_energy, atom_lj_virial,
                        atom_direct_pme_energy, lj_info->d_LJ_energy_atom);
                }
                else if (lj_soft_info->is_initialized)
                {
                    Launch_Device_Kernel(
                        Vector_Soft_Core_To_Hard_Core,
                        (atom_numbers + CONTROLLER::device_max_thread - 1) /
                            CONTROLLER::device_max_thread,
                        CONTROLLER::device_max_thread, 0, NULL, atom_numbers,
                        soft_to_hard_crd,
                        lj_soft_info->crd_with_LJ_parameters_local);

                    auto f =
                        Lennard_Jones_And_Direct_Coulomb_Device<4, true, false,
                                                                false, true>;
                    if (!need_atom_energy && !need_virial)
                    {
                        f = Lennard_Jones_And_Direct_Coulomb_Device<
                            4, true, false, false, true>;
                    }
                    else if (need_atom_energy && !need_virial)
                    {
                        f = Lennard_Jones_And_Direct_Coulomb_Device<
                            4, true, true, false, true>;
                    }
                    else if (!need_atom_energy && need_virial)
                    {
                        f = Lennard_Jones_And_Direct_Coulomb_Device<
                            4, true, false, true, true>;
                    }
                    else
                    {
                        f = Lennard_Jones_And_Direct_Coulomb_Device<
                            4, true, true, true, true>;
                    }
                    Launch_Device_Kernel(
                        f, gridSize, blockSize, 0, NULL, atom_numbers, nl,
                        solvent_start_local, residue_numbers, d_res_start,
                        soft_to_hard_crd, cell, rcell, lj_soft_info->d_LJ_AA,
                        lj_soft_info->d_LJ_AB, lj_soft_info->cutoff, frc,
                        pme_beta, atom_energy, atom_lj_virial,
                        atom_direct_pme_energy, lj_soft_info->d_LJ_energy_atom);
                }
                break;
        }
#endif
    }
}

static __global__ void get_local_device(const int local_res_numbers,
                                        const int* d_res_len,
                                        const int water_points,
                                        int* d_solvent_start_local,
                                        int* d_local_solvent_numbers,
                                        const int atom_numbers, float* d_mass)
{
#ifdef USE_GPU
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx != 0) return;
#endif
    d_solvent_start_local[0] = local_res_numbers;
    d_local_solvent_numbers[0] = 0;
    int cnt = 0;
    for (int i = local_res_numbers - 1; i >= 0; i -= 1)
    {
        int res_start = atom_numbers - water_points * (cnt + 1);
        float atom_O = d_mass[res_start];
        float atom_H1 = d_mass[res_start + 1];
        float atom_H2 = d_mass[res_start + 2];
        if (d_res_len[i] == water_points &&
            (atom_O > 15.8f && atom_O < 16.2f) &&
            (atom_H1 > 1.007f && atom_H1 < 1.009f) &&
            (atom_H2 > 1.007f && atom_H2 < 1.009f))
        {
            d_solvent_start_local[0] -= 1;
            d_local_solvent_numbers[0] += water_points;
            cnt++;
        }
        else
        {
            break;
        }
    }
}

void SOLVENT_LENNARD_JONES::Get_Local(const int local_res_numbers,
                                      const int* d_res_len,
                                      const int atom_numbers, float* local_mass)
{
    if (!is_initialized) return;

    Launch_Device_Kernel(get_local_device, 1, 1, 0, NULL, local_res_numbers,
                         d_res_len, water_points, d_solvent_start_local,
                         d_local_solvent_numbers, atom_numbers, local_mass);
    deviceMemcpy(&solvent_start_local, d_solvent_start_local, sizeof(int),
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(&local_solvent_numbers, d_local_solvent_numbers, sizeof(int),
                 deviceMemcpyDeviceToHost);
}

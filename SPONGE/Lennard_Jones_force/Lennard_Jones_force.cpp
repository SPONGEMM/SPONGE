#include "Lennard_Jones_force.h"

#include "../xponge/load/native/lj.hpp"
#include "../xponge/xponge.h"
// #include "assert.h"

// 由LJ坐标和转化系数求距离
__global__ void Copy_LJ_Type_To_New_Crd(const int atom_numbers,
                                        VECTOR_LJ* new_crd, const int* LJ_type)
{
    SIMPLE_DEVICE_FOR(atom_i, atom_numbers)
    {
        new_crd[atom_i].LJ_type = LJ_type[atom_i];
    }
}

__global__ void Copy_Crd_And_Charge_To_New_Crd(const int atom_numbers,
                                               const VECTOR* crd,
                                               VECTOR_LJ* new_crd,
                                               const float* charge)
{
    SIMPLE_DEVICE_FOR(atom_i, atom_numbers)
    {
        new_crd[atom_i].crd = crd[atom_i];
        new_crd[atom_i].charge = charge[atom_i];
    }
}

__global__ void Copy_Crd_To_New_Crd(const int atom_numbers, const VECTOR* crd,
                                    VECTOR_LJ* new_crd)
{
    SIMPLE_DEVICE_FOR(atom_i, atom_numbers)
    {
        new_crd[atom_i].crd = crd[atom_i];
    }
}

static __global__ void device_add(float* variable, const float adder)
{
    variable[0] += adder;
}

template <bool need_force, bool need_energy, bool need_virial,
          bool need_coulomb>
static __global__ void Lennard_Jones_And_Direct_Coulomb_Device(
    const int local_atom_numbers, const int solvent_numbers,
    const ATOM_GROUP* nl, const VECTOR_LJ* crd, const LTMatrix3 cell,
    const LTMatrix3 rcell, const float* LJ_type_A, const float* LJ_type_B,
    const float cutoff, VECTOR* frc, const float pme_beta, float* atom_energy,
    LTMatrix3* atom_virial, float* atom_direct_cf_energy, float* atom_LJ_ene)
{
#ifdef USE_GPU
    int atom_i = 0 + blockDim.y * blockIdx.x + threadIdx.y;
    if (atom_i < local_atom_numbers - solvent_numbers)
#else
#pragma omp parallel for schedule(dynamic)
    for (int atom_i = 0; atom_i < local_atom_numbers - solvent_numbers;
         atom_i++)
#endif
    {
        VECTOR frc_record = {0.0f, 0.0f, 0.0f};
        LTMatrix3 virial = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float energy_lj = 0.0f;
        float energy_coulomb = 0.0f;
        float energy_total = 0.0f;
        ATOM_GROUP nl_i = nl[atom_i];
        VECTOR_LJ r1 = crd[atom_i];
#ifdef USE_GPU
        for (int j = threadIdx.x; j < nl_i.atom_numbers; j += blockDim.x)
#else
        for (int j = 0; j < nl_i.atom_numbers; j += 1)
#endif
        {
            int atom_j = nl_i.atom_serial[j];
            float ij_factor = atom_j < local_atom_numbers ? 1.0f : 0.5f;
            VECTOR_LJ r2 = crd[atom_j];
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
                        float frc_cf_abs =
                            Get_Direct_Coulomb_Force(r1, r2, dr_abs, pme_beta);
                        frc_abs = frc_abs - frc_cf_abs;
                    }
                    VECTOR frc_lin = frc_abs * dr;
                    frc_record = frc_record + frc_lin;
                    if (atom_j < local_atom_numbers)
                    {
                        atomicAdd(frc + atom_j, -frc_lin);
                    }
                    if (need_virial)
                    {
                        virial = virial - ij_factor * Get_Virial_From_Force_Dis(
                                                          frc_lin, dr);
                    }
                }
                if (need_energy)
                {
                    energy_lj +=
                        ij_factor * Get_LJ_Energy(r1, r2, dr_abs, A, B);
                    if (need_coulomb)
                    {
                        energy_coulomb +=
                            ij_factor *
                            Get_Direct_Coulomb_Energy(r1, r2, dr_abs, pme_beta);
                    }
                }
            }
        }
        energy_total = energy_lj + energy_coulomb;
        if (need_force)
        {
            Warp_Sum_To(frc + atom_i, frc_record, warpSize);
        }
        if (need_energy)
        {
            Warp_Sum_To(atom_energy + atom_i, energy_total, warpSize);
            Warp_Sum_To(atom_LJ_ene + atom_i, energy_lj, warpSize);
            if (need_coulomb)
                Warp_Sum_To(atom_direct_cf_energy + atom_i, energy_coulomb,
                            warpSize);
        }
        if (need_virial)
        {
            Warp_Sum_To(atom_virial + atom_i, virial, warpSize);
        }
    }
}

void LENNARD_JONES_INFORMATION::LJ_Malloc()
{
    Malloc_Safely((void**)&h_atom_LJ_type, sizeof(int) * atom_numbers);
    Malloc_Safely((void**)&h_LJ_A, sizeof(float) * pair_type_numbers);
    Malloc_Safely((void**)&h_LJ_B, sizeof(float) * pair_type_numbers);
    Malloc_Safely((void**)&h_LJ_energy_atom, sizeof(float) * atom_numbers);
}

static __global__ void Total_C6_Get(int atom_numbers, int* atom_lj_type,
                                    float* d_lj_b, float* d_factor)
{
    int j;
    double temp_sum = 0;
    int x, y;
    int itype, jtype, atom_pair_LJ_type;
#ifdef USE_GPU
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < atom_numbers;
         i += gridDim.x * blockDim.x)
#else
#pragma omp parallel for firstprivate( \
        j, x, y, itype, jtype, atom_pair_LJ_type) reduction(+ : temp_sum)
    for (int i = 0; i < atom_numbers; i++)
#endif
    {
        itype = atom_lj_type[i];
        double temp_small_sum = 0;
#ifdef USE_GPU
        for (j = blockIdx.y * blockDim.y + threadIdx.y; j < atom_numbers;
             j += gridDim.y * blockDim.y)
#else
        for (j = 0; j < atom_numbers; j++)
#endif
        {
            jtype = atom_lj_type[j];
            y = (jtype - itype);
            x = y >> 31;
            y = (y ^ x) - x;
            x = jtype + itype;
            jtype = (x + y) >> 1;
            x = (x - y) >> 1;
            atom_pair_LJ_type = (jtype * (jtype + 1) >> 1) + x;
            temp_small_sum += d_lj_b[atom_pair_LJ_type];
        }
        temp_sum += temp_small_sum;
    }
    atomicAdd(d_factor, temp_sum);
}

void LENNARD_JONES_INFORMATION::Initial(CONTROLLER* controller, float cutoff,
                                        const char* module_name)
{
    if (module_name == NULL)
    {
        strcpy(this->module_name, "LJ");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }
    controller->printf("START INITIALIZING LENNADR JONES INFORMATION:\n");
    const auto& lj = Xponge::system.classical_force_field.lj;
    Xponge::LennardJones local_lj;
    const Xponge::LennardJones* lj_to_use = NULL;
    if (module_name == NULL)
    {
        lj_to_use = &lj;
    }
    else if (controller->Command_Exist(this->module_name, "in_file"))
    {
        Xponge::Native_Load_LJ(&local_lj, controller, 0, this->module_name);
        lj_to_use = &local_lj;
    }
    if (lj_to_use != NULL)
    {
        atom_numbers = static_cast<int>(lj_to_use->atom_type.size());
        atom_type_numbers = lj_to_use->atom_type_numbers;
    }
    if (atom_numbers > 0)
    {
        controller->printf("    atom_numbers is %d\n", atom_numbers);
        controller->printf("    atom_LJ_type_number is %d\n",
                           atom_type_numbers);
        pair_type_numbers = atom_type_numbers * (atom_type_numbers + 1) / 2;
        LJ_Malloc();

        for (int i = 0; i < pair_type_numbers; i++)
        {
            h_LJ_A[i] = lj_to_use->pair_A[i];
            h_LJ_B[i] = lj_to_use->pair_B[i];
        }
        for (int i = 0; i < atom_numbers; i++)
        {
            h_atom_LJ_type[i] = lj_to_use->atom_type[i];
        }
        Parameter_Host_To_Device();
        is_initialized = 1;
    }
    if (is_initialized)
    {
        this->cutoff = cutoff;
        Device_Malloc_Safely((void**)&crd_with_LJ_parameters,
                             sizeof(VECTOR_LJ) * atom_numbers);
        Launch_Device_Kernel(
            Copy_LJ_Type_To_New_Crd,
            (this->atom_numbers + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, atom_numbers,
            crd_with_LJ_parameters, d_atom_LJ_type);
        controller->printf("    Start initializing long range LJ correction\n");
        long_range_factor = 0;

        Device_Malloc_And_Copy_Safely((void**)&d_long_range_factor,
                                      &long_range_factor, sizeof(float));
        deviceMemset(d_long_range_factor, 0, sizeof(float));

        dim3 gridSize = {(atom_numbers + CONTROLLER::device_max_thread - 1) /
                             CONTROLLER::device_max_thread,
                         1};
        dim3 blockSize = {
            CONTROLLER::device_warp,
            CONTROLLER::device_max_thread / CONTROLLER::device_warp};
        Launch_Device_Kernel(Total_C6_Get, gridSize, blockSize, 0, NULL,
                             atom_numbers, d_atom_LJ_type, d_LJ_B,
                             d_long_range_factor);

        deviceMemcpy(&long_range_factor, d_long_range_factor, sizeof(float),
                     deviceMemcpyDeviceToHost);
        printf("        Total C6 factor is %e\n", long_range_factor);

        long_range_factor *=
            -2.0f / 3.0f * CONSTANT_Pi / cutoff / cutoff / cutoff / 6.0f;
        controller->printf("        long range correction factor is: %e\n",
                           long_range_factor);
        controller->printf("    End initializing long range LJ correction\n");
    }
    if (is_initialized && !is_controller_printf_initialized)
    {
        controller->Step_Print_Initial("LJ_short", "%.2f");
        controller->Step_Print_Initial("LJ_long", "%.2f");
        controller->Step_Print_Initial("LJ", "%.2f");
        is_controller_printf_initialized = 1;
        controller->printf("    structure last modify date is %d\n",
                           last_modify_date);
    }
    controller->printf("END INITIALIZING LENNADR JONES INFORMATION\n\n");
}

static __global__ void get_local_device(int* atom_local, int local_atom_numbers,
                                        int ghost_numbers, int* d_atom_LJ_type,
                                        VECTOR_LJ* crd_with_LJ_parameters_local)
{
    SIMPLE_DEVICE_FOR(i, local_atom_numbers + ghost_numbers)
    {
        int atom_i = atom_local[i];
        crd_with_LJ_parameters_local[i].LJ_type = d_atom_LJ_type[atom_i];
    }
}

void LENNARD_JONES_INFORMATION::Get_Local(int* atom_local,
                                          int local_atom_numbers,
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
                         local_atom_numbers, ghost_numbers, d_atom_LJ_type,
                         crd_with_LJ_parameters_local);
}

static __global__ void Long_Range_Virial_Correction(LTMatrix3* d_virial,
                                                    const float factor)
{
    d_virial[0].a11 += factor;
    d_virial[0].a22 += factor;
    d_virial[0].a33 += factor;
}

void LENNARD_JONES_INFORMATION::Long_Range_Correction(int need_pressure,
                                                      LTMatrix3* d_virial,
                                                      int need_potential,
                                                      float* d_potential,
                                                      const float volume)
{
    if (is_initialized && CONTROLLER::PP_MPI_rank == 0)
    {
        if (need_pressure)
        {
            Launch_Device_Kernel(Long_Range_Virial_Correction, 1, 1, 0, 0,
                                 d_virial, 2 * long_range_factor / volume);
        }
        if (need_potential)
        {
            Launch_Device_Kernel(device_add, 1, 1, 0, 0, d_potential,
                                 long_range_factor / volume);

            h_LJ_long_energy = long_range_factor / volume;
        }
    }
}

void LENNARD_JONES_INFORMATION::Parameter_Host_To_Device()
{
    Device_Malloc_And_Copy_Safely((void**)&d_atom_LJ_type, h_atom_LJ_type,
                                  sizeof(int) * atom_numbers);
    Device_Malloc_And_Copy_Safely((void**)&d_LJ_A, h_LJ_A,
                                  sizeof(float) * pair_type_numbers);
    Device_Malloc_And_Copy_Safely((void**)&d_LJ_B, h_LJ_B,
                                  sizeof(float) * pair_type_numbers);
    Device_Malloc_And_Copy_Safely((void**)&d_LJ_energy_sum, h_LJ_energy_atom,
                                  sizeof(float));
    Device_Malloc_Safely((void**)&d_LJ_energy_atom,
                         sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&crd_with_LJ_parameters_local,
                         sizeof(VECTOR_LJ) * atom_numbers);
}

void LENNARD_JONES_INFORMATION::LJ_PME_Direct_Force_With_Atom_Energy_And_Virial(
    const int atom_numbers, const int local_atom_numbers,
    const int solvent_numbers, const int ghost_numbers, const VECTOR* crd,
    const float* charge, VECTOR* frc, const LTMatrix3 cell,
    const LTMatrix3 rcell, const ATOM_GROUP* nl, const float pme_beta,
    const int need_atom_energy, float* atom_energy, const int need_virial,
    LTMatrix3* atom_virial, float* atom_direct_pme_energy)
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
            deviceMemset(atom_direct_pme_energy, 0,
                         sizeof(float) * this->atom_numbers);
            deviceMemset(d_LJ_energy_atom, 0,
                         sizeof(float) * this->atom_numbers);
        }

        if (atom_numbers == 0 || local_atom_numbers == 0) return;

        dim3 blockSize = {
            CONTROLLER::device_warp,
            CONTROLLER::device_max_thread / CONTROLLER::device_warp};
        dim3 gridSize = (atom_numbers + blockSize.y - 1) / blockSize.y;
        auto f =
            Lennard_Jones_And_Direct_Coulomb_Device<true, false, false, true>;
        if (!need_atom_energy && !need_virial)
        {
            f = Lennard_Jones_And_Direct_Coulomb_Device<true, false, false,
                                                        true>;
        }
        else if (need_atom_energy && !need_virial)
        {
            f = Lennard_Jones_And_Direct_Coulomb_Device<true, true, false,
                                                        true>;
        }
        else if (!need_atom_energy && need_virial)
        {
            f = Lennard_Jones_And_Direct_Coulomb_Device<true, false, true,
                                                        true>;
        }
        else
        {
            f = Lennard_Jones_And_Direct_Coulomb_Device<true, true, true, true>;
        }
        Launch_Device_Kernel(
            f, gridSize, blockSize, 0, NULL, local_atom_numbers,
            solvent_numbers, nl, crd_with_LJ_parameters_local, cell, rcell,
            d_LJ_A, d_LJ_B, cutoff, frc, pme_beta, atom_energy, atom_virial,
            atom_direct_pme_energy, d_LJ_energy_atom);
    }
}

void LENNARD_JONES_INFORMATION::Step_Print(CONTROLLER* controller)
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
    controller->Step_Print("LJ_short", h_LJ_energy_sum);
    controller->Step_Print("LJ_long", h_LJ_long_energy);
    controller->Step_Print("LJ", h_LJ_energy_sum + h_LJ_long_energy, true);
}

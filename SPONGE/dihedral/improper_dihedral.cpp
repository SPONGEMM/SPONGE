#include "improper_dihedral.h"

#include "../xponge/load/native/improper_dihedral.hpp"
#include "../xponge/xponge.h"

static __global__ void Dihedral_Force_With_Atom_Energy_And_Virial_Device(
    const int dihedral_numbers, const VECTOR* crd, const LTMatrix3 cell,
    const LTMatrix3 rcell, const int local_atom_numbers, const int* atom_a,
    const int* atom_b, const int* atom_c, const int* atom_d, const float* pk,
    const float* phi0, VECTOR* frc, int need_atom_energy, float* ene,
    float* di_ene, int need_virial, LTMatrix3* virial)
{
#ifdef USE_GPU
    int dihedral_i = blockDim.x * blockIdx.x + threadIdx.x;
    if (dihedral_i < dihedral_numbers)
#else
#pragma omp parallel for
    for (int dihedral_i = 0; dihedral_i < dihedral_numbers; dihedral_i++)
#endif
    {
        int atom_i = atom_a[dihedral_i];
        int atom_j = atom_b[dihedral_i];
        int atom_k = atom_c[dihedral_i];
        int atom_l = atom_d[dihedral_i];

        float temp_pk = pk[dihedral_i];
        float temp_phi0 = phi0[dihedral_i];

        VECTOR drij =
            Get_Periodic_Displacement(crd[atom_i], crd[atom_j], cell, rcell);
        VECTOR drkj =
            Get_Periodic_Displacement(crd[atom_k], crd[atom_j], cell, rcell);
        VECTOR drkl =
            Get_Periodic_Displacement(crd[atom_k], crd[atom_l], cell, rcell);

        VECTOR r1 = drij ^ drkj;
        VECTOR r2 = drkl ^ drkj;

        float r1_1 = rnorm3df(r1.x, r1.y, r1.z);
        float r2_1 = rnorm3df(r2.x, r2.y, r2.z);
        float r1_2 = r1_1 * r1_1;
        float r2_2 = r2_1 * r2_1;
        float r1_1_r2_1 = r1_1 * r2_1;
        // PHI, pay attention to the var NAME
        float phi = r1 * r2 * r1_1_r2_1;
        phi = fmaxf(-0.999999, fminf(phi, 0.999999));
        phi = acosf(phi);

        float sign = (r2 ^ r1) * drkj;
        phi = copysignf(phi, sign);

        phi = CONSTANT_Pi - phi;

        float delta_phi = phi - temp_phi0;
        if (delta_phi > CONSTANT_Pi)
        {
            delta_phi -= 2.0f * CONSTANT_Pi;
        }
        else if (delta_phi < -CONSTANT_Pi)
        {
            delta_phi += 2.0f * CONSTANT_Pi;
        }
        float sin_phi = sinf(phi);
        float cos_phi = cosf(phi);

        // Here and following var name "phi" corespongding to the declaration of
        // phi aka, the var with the comment line "PHI, pay attention to the var
        // NAME" The real dihedral = Pi - ArcCos(so-called "phi") d(real
        // dihedral) = 1/sin(real dihedral) * d(so-called  "phi")
        float dE_dphi = -2.0f * temp_pk * delta_phi / sin_phi;

        VECTOR dphi_dr1 = r1_1_r2_1 * r2 + cos_phi * r1_2 * r1;
        VECTOR dphi_dr2 = r1_1_r2_1 * r1 + cos_phi * r2_2 * r2;

        VECTOR dE_dri = dE_dphi * drkj ^ dphi_dr1;
        VECTOR dE_drl = dE_dphi * dphi_dr2 ^ drkj;
        VECTOR dE_drj_part = dE_dphi * ((drij ^ dphi_dr1) + (drkl ^ dphi_dr2));

        VECTOR fi = dE_dri;
        VECTOR fj = dE_drj_part - dE_dri;
        VECTOR fk = -dE_drl - dE_drj_part;
        VECTOR fl = dE_drl;
        if (atom_i < local_atom_numbers)
        {
            atomicAdd(&frc[atom_i].x, fi.x);
            atomicAdd(&frc[atom_i].y, fi.y);
            atomicAdd(&frc[atom_i].z, fi.z);
            if (need_atom_energy)
            {
                atomicAdd(&ene[atom_i], temp_pk * delta_phi * delta_phi);
                di_ene[dihedral_i] = (temp_pk * delta_phi * delta_phi);
            }

            if (need_virial)
            {
                atomicAdd(virial + atom_i,
                          Get_Virial_From_Force_Dis(dE_drl, drkl) +
                              Get_Virial_From_Force_Dis(dE_dri, drij) +
                              Get_Virial_From_Force_Dis(dE_drj_part, drkj));
            }
        }

        if (atom_j < local_atom_numbers)
        {
            atomicAdd(&frc[atom_j].x, fj.x);
            atomicAdd(&frc[atom_j].y, fj.y);
            atomicAdd(&frc[atom_j].z, fj.z);
        }
        if (atom_k < local_atom_numbers)
        {
            atomicAdd(&frc[atom_k].x, fk.x);
            atomicAdd(&frc[atom_k].y, fk.y);
            atomicAdd(&frc[atom_k].z, fk.z);
        }
        if (atom_l < local_atom_numbers)
        {
            atomicAdd(&frc[atom_l].x, fl.x);
            atomicAdd(&frc[atom_l].y, fl.y);
            atomicAdd(&frc[atom_l].z, fl.z);
        }
    }
}

void IMPROPER_DIHEDRAL::Initial(CONTROLLER* controller, const char* module_name)
{
    if (module_name == NULL)
    {
        strcpy(this->module_name, "improper_dihedral");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }

    char file_name_suffix[CHAR_LENGTH_MAX];
    sprintf(file_name_suffix, "in_file");
    const auto& impropers = Xponge::system.classical_force_field.impropers;
    Xponge::Torsions local_impropers;
    const Xponge::Torsions* impropers_to_use = NULL;
    const char* init_source = NULL;

    if (module_name == NULL)
    {
        impropers_to_use = &impropers;
        init_source = "Xponge::system";
    }
    else if (controller[0].Command_Exist(this->module_name, file_name_suffix))
    {
        Xponge::Native_Load_Impropers(&local_impropers, controller,
                                      this->module_name);
        impropers_to_use = &local_impropers;
    }

    if (impropers_to_use != NULL)
    {
        dihedral_numbers = static_cast<int>(impropers_to_use->atom_a.size());
    }
    if (dihedral_numbers > 0)
    {
        if (module_name == NULL)
        {
            controller[0].printf("START INITIALIZING IMPROPER DIHEDRAL (%s):\n",
                                 init_source);
        }
        else
        {
            controller[0].printf(
                "START INITIALIZING IMPROPER DIHEDRAL (%s_%s):\n",
                this->module_name, file_name_suffix);
        }
        controller[0].printf("    dihedral_numbers is %d\n", dihedral_numbers);
        Memory_Allocate();
        for (int i = 0; i < dihedral_numbers; i++)
        {
            h_atom_a[i] = impropers_to_use->atom_a[i];
            h_atom_b[i] = impropers_to_use->atom_b[i];
            h_atom_c[i] = impropers_to_use->atom_c[i];
            h_atom_d[i] = impropers_to_use->atom_d[i];
            h_pk[i] = impropers_to_use->pk[i];
            h_phi0[i] = impropers_to_use->gamc[i];
        }
        Parameter_Host_To_Device();
        is_initialized = 1;
    }
    else
    {
        controller[0].printf("IMPROPER DIHEDRAL IS NOT INITIALIZED\n\n");
    }

    if (is_initialized && !is_controller_printf_initialized)
    {
        controller[0].Step_Print_Initial(this->module_name, "%.2f");
        is_controller_printf_initialized = 1;
        controller[0].printf("    structure last modify date is %d\n",
                             last_modify_date);
    }
    if (is_initialized)
    {
        controller[0].printf("END INITIALIZING IMPROPER DIHEDRAL\n\n");
    }
}

void IMPROPER_DIHEDRAL::Memory_Allocate()
{
    Malloc_Safely((void**)&h_atom_a, sizeof(int) * dihedral_numbers);
    Malloc_Safely((void**)&h_atom_b, sizeof(int) * dihedral_numbers);
    Malloc_Safely((void**)&h_atom_c, sizeof(int) * dihedral_numbers);
    Malloc_Safely((void**)&h_atom_d, sizeof(int) * dihedral_numbers);
    Malloc_Safely((void**)&h_pk, sizeof(float) * dihedral_numbers);
    Malloc_Safely((void**)&h_phi0, sizeof(float) * dihedral_numbers);
    Malloc_Safely((void**)&h_dihedral_ene, sizeof(float) * dihedral_numbers);
    memset(h_dihedral_ene, 0, sizeof(float) * dihedral_numbers);
    Malloc_Safely((void**)&h_sigma_of_dihedral_ene, sizeof(float));
    memset(h_sigma_of_dihedral_ene, 0, sizeof(float));
}

void IMPROPER_DIHEDRAL::Parameter_Host_To_Device()
{
    Device_Malloc_Safely((void**)&d_atom_a, sizeof(int) * dihedral_numbers);
    Device_Malloc_Safely((void**)&d_atom_b, sizeof(int) * dihedral_numbers);
    Device_Malloc_Safely((void**)&d_atom_c, sizeof(int) * dihedral_numbers);
    Device_Malloc_Safely((void**)&d_atom_d, sizeof(int) * dihedral_numbers);
    Device_Malloc_Safely((void**)&d_pk, sizeof(float) * dihedral_numbers);
    Device_Malloc_Safely((void**)&d_phi0, sizeof(float) * dihedral_numbers);
    Device_Malloc_Safely((void**)&d_dihedral_ene,
                         sizeof(float) * dihedral_numbers);
    Device_Malloc_Safely((void**)&d_sigma_of_dihedral_ene, sizeof(float));

    Device_Malloc_Safely((void**)&d_atom_a_local,
                         sizeof(int) * dihedral_numbers);
    Device_Malloc_Safely((void**)&d_atom_b_local,
                         sizeof(int) * dihedral_numbers);
    Device_Malloc_Safely((void**)&d_atom_c_local,
                         sizeof(int) * dihedral_numbers);
    Device_Malloc_Safely((void**)&d_atom_d_local,
                         sizeof(int) * dihedral_numbers);
    Device_Malloc_Safely((void**)&d_pk_local, sizeof(float) * dihedral_numbers);
    Device_Malloc_Safely((void**)&d_phi0_local,
                         sizeof(float) * dihedral_numbers);
    Device_Malloc_Safely((void**)&d_num_dihe_local, sizeof(int));
    deviceMemset(d_num_dihe_local, 0, sizeof(int));

    deviceMemcpy(d_atom_a, h_atom_a, sizeof(int) * dihedral_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(d_atom_b, h_atom_b, sizeof(int) * dihedral_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(d_atom_c, h_atom_c, sizeof(int) * dihedral_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(d_atom_d, h_atom_d, sizeof(int) * dihedral_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(d_pk, h_pk, sizeof(float) * dihedral_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(d_phi0, h_phi0, sizeof(float) * dihedral_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemset(d_dihedral_ene, 0, sizeof(float) * dihedral_numbers);
    deviceMemset(d_sigma_of_dihedral_ene, 0, sizeof(float));
}

static __global__ void get_local_device(
    int dihedral_numbers, int* d_atom_a, int* d_atom_b, int* d_atom_c,
    int* d_atom_d, char* atom_local_label, int* atom_local_id,
    int* d_atom_a_local, int* d_atom_b_local, int* d_atom_c_local,
    int* d_atom_d_local, float* d_pk, float* d_phi0, float* d_pk_local,
    float* d_phi0_local, int* d_num_dihe_local)
{
#ifdef USE_GPU
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx != 0) return;
#endif
    d_num_dihe_local[0] = 0;
    // 判断第i个dihedral中原子a是否在local中，如果是则需要在本domain中计算
    for (int i = 0; i < dihedral_numbers; i++)
    {
        if (atom_local_label[d_atom_a[i]] == 1 ||
            atom_local_label[d_atom_b[i]] == 1 ||
            atom_local_label[d_atom_c[i]] == 1 ||
            atom_local_label[d_atom_d[i]] == 1)
        {
            d_atom_a_local[d_num_dihe_local[0]] = atom_local_id[d_atom_a[i]];
            d_atom_b_local[d_num_dihe_local[0]] = atom_local_id[d_atom_b[i]];
            d_atom_c_local[d_num_dihe_local[0]] = atom_local_id[d_atom_c[i]];
            d_atom_d_local[d_num_dihe_local[0]] = atom_local_id[d_atom_d[i]];
            d_pk_local[d_num_dihe_local[0]] = d_pk[i];
            d_phi0_local[d_num_dihe_local[0]] = d_phi0[i];
            d_num_dihe_local[0]++;
        }
    }
}

void IMPROPER_DIHEDRAL::Get_Local(int* atom_local, int local_atom_numbers,
                                  int ghost_numbers, char* atom_local_label,
                                  int* atom_local_id)
{
    if (!is_initialized) return;
    num_dihe_local = 0;
    this->local_atom_numbers = local_atom_numbers;
    Launch_Device_Kernel(get_local_device, 1, 1, 0, NULL, dihedral_numbers,
                         d_atom_a, d_atom_b, d_atom_c, d_atom_d,
                         atom_local_label, atom_local_id, d_atom_a_local,
                         d_atom_b_local, d_atom_c_local, d_atom_d_local, d_pk,
                         d_phi0, d_pk_local, d_phi0_local, d_num_dihe_local);
    deviceMemcpy(&num_dihe_local, d_num_dihe_local, sizeof(int),
                 deviceMemcpyDeviceToHost);
}

void IMPROPER_DIHEDRAL::Dihedral_Force_With_Atom_Energy_And_Virial(
    const VECTOR* crd, const LTMatrix3 cell, const LTMatrix3 rcell, VECTOR* frc,
    int need_atom_energy, float* atom_energy, int need_virial,
    LTMatrix3* atom_virial)
{
    if (is_initialized)
    {
        Launch_Device_Kernel(
            Dihedral_Force_With_Atom_Energy_And_Virial_Device,
            (dihedral_numbers + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, this->num_dihe_local, crd,
            cell, rcell, this->local_atom_numbers, this->d_atom_a_local,
            this->d_atom_b_local, this->d_atom_c_local, this->d_atom_d_local,
            this->d_pk_local, this->d_phi0_local, frc, need_atom_energy,
            atom_energy, d_dihedral_ene, need_virial, atom_virial);
    }
}

void IMPROPER_DIHEDRAL::Step_Print(CONTROLLER* controller, bool print_sum)
{
    if (is_initialized && CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size)
    {
        Sum_Of_List(d_dihedral_ene, d_sigma_of_dihedral_ene,
                    num_dihe_local);  // 修改为局部求和
        deviceMemcpy(h_sigma_of_dihedral_ene, d_sigma_of_dihedral_ene,
                     sizeof(float), deviceMemcpyDeviceToHost);
#ifdef USE_MPI
        MPI_Allreduce(MPI_IN_PLACE, h_sigma_of_dihedral_ene, 1, MPI_FLOAT,
                      MPI_SUM, CONTROLLER::pp_comm);
#endif
        if (CONTROLLER::MPI_rank == 0)
        {
            controller->Step_Print(this->module_name, h_sigma_of_dihedral_ene,
                                   print_sum);
        }
    }
}

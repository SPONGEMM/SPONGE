#include "bond.h"

#include "../xponge/load/native/bond.hpp"
#include "../xponge/xponge.h"

// 由于，大部分情况下bond的energy和virial计算耗时不显著，为简化bond模块的逻辑复杂度，
// 将bond
// 对原子上的力（frc）、原子上的能量（atom_energy）、原子上的维力值（atom_virial）
// 一并计算。
// 对于简易和轻度修改，可以不用考虑能量与维力值的计算。
//   在不使用涉及维力系数的模拟中，可以不用计算正确的维力值
//   在能量数值不影响模拟的过程中，可以不用计算正确的能量值
//   只有力是最基本的计算要求

static __global__ void Bond_Force_With_Atom_Energy_And_Virial_Device(
    const int bond_numbers, const VECTOR* crd, const LTMatrix3 cell,
    const LTMatrix3 rcell, const int local_atom_numbers, const int* atom_a,
    const int* atom_b, const float* bond_k, const float* bond_r0, VECTOR* frc,
    int need_atom_energy, float* atom_energy, int need_virial,
    LTMatrix3* atom_virial, float* bond_ene)
{
#ifdef USE_GPU
    int bond_i = blockDim.x * blockIdx.x + threadIdx.x;
    if (bond_i < bond_numbers)
#else
#pragma omp parallel for
    for (int bond_i = 0; bond_i < bond_numbers; bond_i++)
#endif
    {
        // 获取第bond_i根键的两个连接的原子编号
        // 和键强度、平衡长度
        int atom_i = atom_a[bond_i];
        int atom_j = atom_b[bond_i];
        float k = bond_k[bond_i];
        float r0 = bond_r0[bond_i];

        // 获取该对原子的考虑周期性边界的最短位置矢量（dr），和最短距离abs_r
        VECTOR dr =
            Get_Periodic_Displacement(crd[atom_i], crd[atom_j], cell, rcell);
        float abs_r = norm3df(dr.x, dr.y, dr.z);
        float tempf2 = abs_r - r0;
        float tempf = 2 * tempf2 * k;
        VECTOR f = -tempf / abs_r * dr;
        // 将计算得到的力加到对应的原子身上
        if (atom_j < local_atom_numbers) atomicAdd(frc + atom_j, -f);
        if (atom_i < local_atom_numbers)
        {
            atomicAdd(frc + atom_i, f);
            // 将计算得到的能量和维力值加到该bond中的其中一个原子身上
            // 原理上，该bond能量是不可分的。但是，一般情况，bond相连的两个原子
            // 总是被看作属于一个分子来讨论，因此可以直接将能量和维力值加到其中一个原子上
            if (need_atom_energy)
            {
                atomicAdd(atom_energy + atom_i, k * tempf2 * tempf2);
                bond_ene[bond_i] = k * tempf2 * tempf2;
            }
            if (need_virial)
            {
                atomicAdd(atom_virial + atom_i,
                          Get_Virial_From_Force_Dis(f, dr));
            }
        }
    }
}

void BOND::Initial(CONTROLLER* controller, CONECT* connectivity,
                   PAIR_DISTANCE* con_dis, const char* module_name)
{
    // 给予bond模块一个默认名字：bond
    if (module_name == NULL)
    {
        strcpy(this->module_name, "bond");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }

    char file_name_suffix[CHAR_LENGTH_MAX];
    sprintf(file_name_suffix, "in_file");
    const auto& bonds = Xponge::system.classical_force_field.bonds;
    Xponge::Bonds local_bonds;
    const Xponge::Bonds* bonds_to_use = NULL;
    const char* init_source = NULL;
    if (module_name == NULL)
    {
        bonds_to_use = &bonds;
        init_source = "Xponge::system";
    }
    else if (controller->Command_Exist(this->module_name, file_name_suffix))
    {
        Xponge::Native_Load_Bonds(&local_bonds, controller, this->module_name);
        bonds_to_use = &local_bonds;
    }

    if (bonds_to_use != NULL)
    {
        bond_numbers = static_cast<int>(bonds_to_use->atom_a.size());
    }
    if (bond_numbers > 0)
    {
        if (module_name == NULL)
        {
            controller->printf("START INITIALIZING BOND (%s):\n", init_source);
        }
        else
        {
            controller->printf("START INITIALIZING BOND (%s_%s):\n",
                               this->module_name, file_name_suffix);
        }
        controller->printf("    bond_numbers is %d\n", bond_numbers);
        Memory_Allocate();
        for (int i = 0; i < bond_numbers; i++)
        {
            h_atom_a[i] = bonds_to_use->atom_a[i];
            h_atom_b[i] = bonds_to_use->atom_b[i];
            h_k[i] = bonds_to_use->k[i];
            h_r0[i] = bonds_to_use->r0[i];
        }
        Parameter_Host_To_Device();
        is_initialized = 1;
    }
    else
    {
        controller->printf("BOND IS NOT INITIALIZED\n\n");
    }

    // 初始化了，且第一次加载用于间隔输出的信息
    if (is_initialized && !is_controller_printf_initialized)
    {
        controller->Step_Print_Initial(this->module_name, "%.2f");
        is_controller_printf_initialized = 1;
        controller->printf("    structure last modify date is %d\n",
                           last_modify_date);
    }

    // 初始化完成
    if (is_initialized && connectivity)
    {
        for (int i = 0; i < bond_numbers; i += 1)
        {
            connectivity[0][h_atom_a[i]].insert(h_atom_b[i]);
            connectivity[0][h_atom_b[i]].insert(h_atom_a[i]);
            if (h_atom_a[i] < h_atom_b[i])
            {
                con_dis[0][std::pair<int, int>(h_atom_a[i], h_atom_b[i])] =
                    h_r0[i];
            }
            else
            {
                con_dis[0][std::pair<int, int>(h_atom_b[i], h_atom_a[i])] =
                    h_r0[i];
            }
        }
        controller->printf("END INITIALIZING BOND\n\n");
    }
}

void BOND::Memory_Allocate()
{
    Malloc_Safely((void**)&h_atom_a, sizeof(int) * this->bond_numbers);
    Malloc_Safely((void**)&h_atom_b, sizeof(int) * this->bond_numbers);
    Malloc_Safely((void**)&h_k, sizeof(float) * this->bond_numbers);
    Malloc_Safely((void**)&h_r0, sizeof(float) * this->bond_numbers);
    Malloc_Safely((void**)&h_bond_ene, sizeof(float) * this->bond_numbers);
    memset(h_bond_ene, 0, sizeof(float) * this->bond_numbers);
    Malloc_Safely((void**)&h_sigma_of_bond_ene, sizeof(float));
    memset(h_sigma_of_bond_ene, 0, sizeof(float));
}

void BOND::Parameter_Host_To_Device()
{
    Device_Malloc_Safely((void**)&d_atom_a, sizeof(int) * this->bond_numbers);
    Device_Malloc_Safely((void**)&d_atom_b, sizeof(int) * this->bond_numbers);
    Device_Malloc_Safely((void**)&d_k, sizeof(float) * this->bond_numbers);
    Device_Malloc_Safely((void**)&d_r0, sizeof(float) * this->bond_numbers);
    Device_Malloc_Safely((void**)&d_bond_ene,
                         sizeof(float) * this->bond_numbers);
    Device_Malloc_Safely((void**)&d_sigma_of_bond_ene, sizeof(float));

    Device_Malloc_Safely((void**)&d_atom_a_local,
                         sizeof(int) * this->bond_numbers);
    Device_Malloc_Safely((void**)&d_atom_b_local,
                         sizeof(int) * this->bond_numbers);
    Device_Malloc_Safely((void**)&d_k_local,
                         sizeof(float) * this->bond_numbers);
    Device_Malloc_Safely((void**)&d_r0_local,
                         sizeof(float) * this->bond_numbers);
    Device_Malloc_Safely((void**)&d_num_bond_local, sizeof(int));
    deviceMemset(d_num_bond_local, 0, sizeof(int));

    deviceMemcpy(d_atom_a, h_atom_a, sizeof(int) * this->bond_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(d_atom_b, h_atom_b, sizeof(int) * this->bond_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(d_k, h_k, sizeof(float) * this->bond_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(d_r0, h_r0, sizeof(float) * this->bond_numbers,
                 deviceMemcpyHostToDevice);
    deviceMemset(d_bond_ene, 0, sizeof(float) * this->bond_numbers);
    deviceMemset(d_sigma_of_bond_ene, 0, sizeof(float));
}

static __global__ void get_local_device(
    int bond_numbers, int* d_atom_a, int* d_atom_b, char* atom_local_label,
    int* atom_local_id, int* d_atom_a_local, int* d_atom_b_local, float* d_k,
    float* d_r0, float* d_k_local, float* d_r0_local, int* d_num_bond_local)
{
#ifdef USE_GPU
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx != 0) return;
#endif
    d_num_bond_local[0] = 0;
    for (int i = 0; i < bond_numbers; i++)
    {
        if (atom_local_label[d_atom_a[i]] == 1 ||
            atom_local_label[d_atom_b[i]] == 1)
        {
            d_atom_a_local[d_num_bond_local[0]] = atom_local_id[d_atom_a[i]];
            d_atom_b_local[d_num_bond_local[0]] = atom_local_id[d_atom_b[i]];
            d_k_local[d_num_bond_local[0]] = d_k[i];
            d_r0_local[d_num_bond_local[0]] = d_r0[i];
            d_num_bond_local[0]++;
        }
    }
}

void BOND::Get_Local(int* atom_local, int local_atom_numbers, int ghost_numbers,
                     char* atom_local_label, int* atom_local_id)
{
    if (!is_initialized) return;
    num_bond_local = 0;
    this->local_atom_numbers = local_atom_numbers;
    Launch_Device_Kernel(
        get_local_device, 1, 1, 0, NULL, this->bond_numbers, this->d_atom_a,
        this->d_atom_b, atom_local_label, atom_local_id, this->d_atom_a_local,
        this->d_atom_b_local, this->d_k, this->d_r0, this->d_k_local,
        this->d_r0_local, this->d_num_bond_local);
    deviceMemcpy(&this->num_bond_local, this->d_num_bond_local, sizeof(int),
                 deviceMemcpyDeviceToHost);
}

void BOND::Bond_Force_With_Atom_Energy_And_Virial(
    const VECTOR* crd, const LTMatrix3 cell, const LTMatrix3 rcell, VECTOR* frc,
    int need_atom_energy, float* atom_energy, int need_virial,
    LTMatrix3* atom_virial)
{
    if (is_initialized)
    {
        Launch_Device_Kernel(
            Bond_Force_With_Atom_Energy_And_Virial_Device,
            (bond_numbers + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, this->num_bond_local, crd,
            cell, rcell, this->local_atom_numbers, this->d_atom_a_local,
            this->d_atom_b_local, this->d_k_local, this->d_r0_local, frc,
            need_atom_energy, atom_energy, need_virial, atom_virial,
            this->d_bond_ene);
    }
}

void BOND::Step_Print(CONTROLLER* controller)
{
    if (is_initialized && CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size)
    {
        Sum_Of_List(d_bond_ene, d_sigma_of_bond_ene,
                    num_bond_local);  // 局部求和
        deviceMemcpy(h_sigma_of_bond_ene, d_sigma_of_bond_ene, sizeof(float),
                     deviceMemcpyDeviceToHost);
#ifdef USE_MPI
        MPI_Allreduce(MPI_IN_PLACE, h_sigma_of_bond_ene, 1, MPI_FLOAT, MPI_SUM,
                      CONTROLLER::pp_comm);
#endif
        controller->Step_Print(this->module_name, h_sigma_of_bond_ene, true);
    }
}

#include "constrain.h"

#include <set>

#include "../xponge/xponge.h"

void CONSTRAIN::Initial_Constrain(CONTROLLER* controller,
                                  const int atom_numbers, const float dt,
                                  const VECTOR box_length, float* atom_mass,
                                  int* system_freedom)
{
    // 从传入的参数复制基本信息
    this->atom_numbers = atom_numbers;
    this->dt = dt;
    this->dt_inverse = 1.0 / dt;

    int extra_numbers = 0;
    FILE* fp = NULL;
    // 读文件第一个数确认constrain数量，为分配内存做准备
    if (controller->Command_Exist(this->module_name, "in_file"))
    {
        Open_File_Safely(&fp, controller->Command(this->module_name, "in_file"),
                         "r");
        int scanf_ret = fscanf(fp, "%d", &extra_numbers);
    }

    constrain_pair_numbers = bond_constrain_pair_numbers + extra_numbers;
    system_freedom[0] -= constrain_pair_numbers;
    controller->printf("    constrain pair number is %d\n",
                       constrain_pair_numbers);

    Malloc_Safely((void**)&h_constrain_pair,
                  sizeof(CONSTRAIN_PAIR) * constrain_pair_numbers);
    for (int i = 0; i < bond_constrain_pair_numbers; i = i + 1)
    {
        h_constrain_pair[i] = h_bond_pair[i];
        h_constrain_pair[i].constrain_k =
            h_constrain_pair[i].constrain_k / this->x_factor;
    }
    // 读文件存入
    if (fp != NULL)
    {
        int atom_i, atom_j;
        int count = bond_constrain_pair_numbers;
        for (int i = 0; i < extra_numbers; i = i + 1)
        {
            int scanf_ret = fscanf(fp, "%d %d %f", &atom_i, &atom_j,
                                   &h_constrain_pair[count].constant_r);
            h_constrain_pair[count].atom_i_serial = atom_i;
            h_constrain_pair[count].atom_j_serial = atom_j;
            h_constrain_pair[count].constrain_k =
                atom_mass[atom_i] * atom_mass[atom_j] /
                (atom_mass[atom_i] + atom_mass[atom_j]) / this->x_factor;
            count += 1;
        }
        fclose(fp);
        fp = NULL;
    }

    Device_Malloc_And_Copy_Safely(
        (void**)&d_constrain_pair, h_constrain_pair,
        sizeof(CONSTRAIN_PAIR) * constrain_pair_numbers);
    Device_Malloc_Safely((void**)&constrain_pair_local,
                         sizeof(CONSTRAIN_PAIR) * atom_numbers);
    Device_Malloc_Safely((void**)&d_num_pair_local, sizeof(int));
    // 清空初始化时使用的临时变量
    if (h_bond_pair != NULL)
    {
        free(h_bond_pair);
        h_bond_pair = NULL;
    }
    if (is_initialized && !is_controller_printf_initialized)
    {
        is_controller_printf_initialized = 1;
        controller->printf("    structure last modify date is %d\n",
                           last_modify_date);
    }
    controller->printf("END INITIALIZING CONSTRAIN\n\n");
    is_initialized = 1;
}

void CONSTRAIN::Initial_List(CONTROLLER* controller, PAIR_DISTANCE con_dis,
                             float* atom_mass, const char* module_name)
{
    controller->printf("START INITIALIZING CONSTRAIN:\n");
    if (module_name == NULL)
    {
        strcpy(this->module_name, "constrain");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }
    constrain_mass = 3.3f;
    if (controller->Command_Exist(this->module_name, "in_file"))
        constrain_mass = 0.0f;
    if (controller->Command_Exist(this->module_name, "mass"))
    {
        controller->Check_Float(this->module_name, "mass",
                                "CONSTRAIN::Add_HBond_To_Constrain_Pair");
        constrain_mass = atof(controller->Command(this->module_name, "mass"));
    }
    const auto& explicit_constraints =
        Xponge::system.classical_force_field.constraints;
    // 预先分配一个足够大的CONSTRAIN_PAIR用于临时存储
    Malloc_Safely((void**)&h_bond_pair,
                  sizeof(CONSTRAIN_PAIR) *
                      (con_dis.size() + explicit_constraints.r0.size()));
    int s = 0;
    float mass_a, mass_b;
    int atom_a, atom_b;
    float r0;
    std::set<std::pair<int, int>> added_pairs;
    for (auto& i : con_dis)
    {
        atom_a = i.first.first;
        atom_b = i.first.second;
        r0 = i.second;
        mass_a = atom_mass[atom_a];
        mass_b = atom_mass[atom_b];
        if ((mass_a < constrain_mass && mass_a > 0) ||
            (mass_b < constrain_mass && mass_b > 0))
        {
            h_bond_pair[s].atom_i_serial = atom_a;
            h_bond_pair[s].atom_j_serial = atom_b;
            h_bond_pair[s].constant_r = r0;
            h_bond_pair[s].constrain_k = mass_a * mass_b / (mass_a + mass_b);
            added_pairs.insert({atom_a, atom_b});
            s = s + 1;
        }
    }
    for (std::size_t i = 0; i < explicit_constraints.r0.size(); i++)
    {
        atom_a = explicit_constraints.atom_a[i];
        atom_b = explicit_constraints.atom_b[i];
        if (atom_b < atom_a)
        {
            std::swap(atom_a, atom_b);
        }
        if (added_pairs.count({atom_a, atom_b}) > 0)
        {
            continue;
        }
        mass_a = atom_mass[atom_a];
        mass_b = atom_mass[atom_b];
        h_bond_pair[s].atom_i_serial = atom_a;
        h_bond_pair[s].atom_j_serial = atom_b;
        h_bond_pair[s].constant_r = explicit_constraints.r0[i];
        h_bond_pair[s].constrain_k = mass_a * mass_b / (mass_a + mass_b);
        added_pairs.insert({atom_a, atom_b});
        s = s + 1;
    }
    bond_constrain_pair_numbers = s;
    if (controller->Command_Exist(this->module_name, "angle") &&
        controller->Get_Bool(this->module_name, "angle",
                             "CONSTRAIN::Add_HAngle_To_Constrain_Pair"))
    {
        controller->Throw_SPONGE_Error(
            spongeErrorNotImplemented, "CONSTRAIN::Initial_List",
            "Reason:\n\tConstraints for angle is not supported from v1.4\n");
    }
}

static __global__ void get_local_device(int constrain_pair_numbers,
                                        CONSTRAIN_PAIR* d_constrain_pair,
                                        const int* atom_local_id,
                                        const char* atom_local_label,
                                        CONSTRAIN_PAIR* constrain_pair_local,
                                        int* d_num_pair_local)
{
#ifdef USE_GPU
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx != 0) return;
#endif
    d_num_pair_local[0] = 0;
    for (int i = 0; i < constrain_pair_numbers; i++)
    {
        int atom_a = d_constrain_pair[i].atom_i_serial;
        int atom_b = d_constrain_pair[i].atom_j_serial;
        if (atom_local_label[atom_a])
        {
            constrain_pair_local[d_num_pair_local[0]] = d_constrain_pair[i];
            constrain_pair_local[d_num_pair_local[0]].atom_i_serial =
                atom_local_id[atom_a];
            constrain_pair_local[d_num_pair_local[0]].atom_j_serial =
                atom_local_id[atom_b];
            d_num_pair_local[0] += 1;
        }
    }
}

void CONSTRAIN::Get_Local(const int* atom_local_id,
                          const char* atom_local_label,
                          const int local_atom_numbers)
{
    if (!is_initialized) return;
    num_pair_local = 0;
    Launch_Device_Kernel(get_local_device, 1, 1, 0, NULL,
                         constrain_pair_numbers, d_constrain_pair,
                         atom_local_id, atom_local_label, constrain_pair_local,
                         d_num_pair_local);
    deviceMemcpy(&num_pair_local, d_num_pair_local, sizeof(int),
                 deviceMemcpyDeviceToHost);
}

void CONSTRAIN::update_ug_connectivity(CONECT* connectivity)
{
    if (!is_initialized) return;
    for (int i = 0; i < constrain_pair_numbers; i++)
    {
        CONSTRAIN_PAIR p = h_constrain_pair[i];
        (*connectivity)[p.atom_i_serial].insert(p.atom_j_serial);
        (*connectivity)[p.atom_j_serial].insert(p.atom_i_serial);
    }
}

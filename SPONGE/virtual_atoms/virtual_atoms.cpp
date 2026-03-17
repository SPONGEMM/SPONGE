#include "virtual_atoms.h"

#include "../xponge/load/native/virtual_atoms.hpp"
#include "../xponge/xponge.h"

static __global__ void v0_Coordinate_Refresh(const int virtual_numbers,
                                             const VIRTUAL_TYPE_0* v_info,
                                             VECTOR* crd, const LTMatrix3 cell,
                                             const LTMatrix3 rcell)
{
#ifdef USE_GPU
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < virtual_numbers)
#else
#pragma omp parallel for
    for (int i = 0; i < virtual_numbers; i++)
#endif
    {
        VIRTUAL_TYPE_0 v_temp = v_info[i];
        int atom_v = v_temp.virtual_atom;
        int atom_1 = v_temp.from_1;
        float h = v_temp.h_double;
        VECTOR temp = crd[atom_1];
        temp.z = 2 * h - temp.z;
        crd[atom_v] = temp;
    }
}

static __global__ void v1_Coordinate_Refresh(const int virtual_numbers,
                                             const VIRTUAL_TYPE_1* v_info,
                                             VECTOR* crd, const LTMatrix3 cell,
                                             const LTMatrix3 rcell)
{
#ifdef USE_GPU
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < virtual_numbers)
#else
#pragma omp parallel for
    for (int i = 0; i < virtual_numbers; i++)
#endif
    {
        VIRTUAL_TYPE_1 v_temp = v_info[i];
        int atom_v = v_temp.virtual_atom;
        int atom_1 = v_temp.from_1;
        int atom_2 = v_temp.from_2;
        float a = v_temp.a;
        VECTOR rv1 = a * Get_Periodic_Displacement(crd[atom_2], crd[atom_1],
                                                   cell, rcell);
        crd[atom_v] = crd[atom_1] + rv1;
    }
}

static __global__ void v2_Coordinate_Refresh(const int virtual_numbers,
                                             const VIRTUAL_TYPE_2* v_info,
                                             VECTOR* crd, const LTMatrix3 cell,
                                             const LTMatrix3 rcell)
{
#ifdef USE_GPU
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < virtual_numbers)
#else
#pragma omp parallel for
    for (int i = 0; i < virtual_numbers; i++)
#endif
    {
        VIRTUAL_TYPE_2 v_temp = v_info[i];
        int atom_v = v_temp.virtual_atom;
        int atom_1 = v_temp.from_1;
        int atom_2 = v_temp.from_2;
        int atom_3 = v_temp.from_3;
        float a = v_temp.a;
        float b = v_temp.b;

        const VECTOR r1 = crd[atom_1];
        const VECTOR r2 = crd[atom_2];
        const VECTOR r3 = crd[atom_3];

        VECTOR rv1 = a * Get_Periodic_Displacement(r2, r1, cell, rcell) +
                     b * Get_Periodic_Displacement(r3, r1, cell, rcell);

        crd[atom_v] = crd[atom_1] + rv1;
    }
}

static __global__ void v3_Coordinate_Refresh(const int virtual_numbers,
                                             const VIRTUAL_TYPE_3* v_info,
                                             VECTOR* crd, const LTMatrix3 cell,
                                             const LTMatrix3 rcell)
{
#ifdef USE_GPU
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < virtual_numbers)
#else
#pragma omp parallel for
    for (int i = 0; i < virtual_numbers; i++)
#endif
    {
        VIRTUAL_TYPE_3 v_temp = v_info[i];
        int atom_v = v_temp.virtual_atom;
        int atom_1 = v_temp.from_1;
        int atom_2 = v_temp.from_2;
        int atom_3 = v_temp.from_3;
        float d = v_temp.d;
        float k = v_temp.k;
        const VECTOR r1 = crd[atom_1];
        const VECTOR r2 = crd[atom_2];
        const VECTOR r3 = crd[atom_3];

        VECTOR r21 = Get_Periodic_Displacement(r2, r1, cell, rcell);
        VECTOR r32 = Get_Periodic_Displacement(r3, r2, cell, rcell);

        VECTOR temp = r21 + k * r32;
        temp = d * rnorm3df(temp.x, temp.y, temp.z) * temp;
        crd[atom_v] = crd[atom_1] + temp;
    }
}

static __global__ void v4_Coordinate_Refresh(const int atom_numbers,
                                             const int virtual_atom,
                                             const int* from_atoms,
                                             const float* weight,
                                             VECTOR* coordinate)
{
    VECTOR new_position = {0, 0, 0};
#ifdef USE_GPU
    int i = blockDim.x * blockDim.y * blockIdx.x + blockDim.y * threadIdx.x +
            threadIdx.y;
    if (i < atom_numbers)
    {
        new_position = new_position + weight[i] * coordinate[from_atoms[i]];
    }
#else
    float px = 0.0f, py = 0.0f, pz = 0.0f;
#pragma omp parallel for reduction(+ : px, py, pz)
    for (int i = 0; i < atom_numbers; i++)
    {
        VECTOR p = weight[i] * coordinate[from_atoms[i]];
        px += p.x;
        py += p.y;
        pz += p.z;
    }
    new_position = {px, py, pz};
#endif
    Warp_Sum_To(coordinate + virtual_atom, new_position, warpSize);
}

static __global__ void v0_Force_Redistribute(
    const int virtual_numbers, const VIRTUAL_TYPE_0* v_info, const VECTOR* crd,
    const LTMatrix3 cell, const LTMatrix3 rcell, VECTOR* force)
{
#ifdef USE_GPU
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < virtual_numbers)
#else
#pragma omp parallel for
    for (int i = 0; i < virtual_numbers; i++)
#endif
    {
        VIRTUAL_TYPE_0 v_temp = v_info[i];
        int atom_v = v_temp.virtual_atom;
        int atom_1 = v_temp.from_1;
        VECTOR force_v = force[atom_v];
        atomicAdd(&force[atom_1].x, force_v.x);
        atomicAdd(&force[atom_1].y, force_v.y);
        atomicAdd(&force[atom_1].z, -force_v.z);
        force_v.x = 0.0f;
        force_v.y = 0.0f;
        force_v.z = 0.0f;
        force[atom_v] = force_v;
    }
}

static __global__ void v1_Force_Redistribute(
    const int virtual_numbers, const VIRTUAL_TYPE_1* v_info, const VECTOR* crd,
    const LTMatrix3 cell, const LTMatrix3 rcell, VECTOR* force)
{
#ifdef USE_GPU
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < virtual_numbers)
#else
#pragma omp parallel for
    for (int i = 0; i < virtual_numbers; i++)
#endif
    {
        VIRTUAL_TYPE_1 v_temp = v_info[i];
        int atom_v = v_temp.virtual_atom;
        int atom_1 = v_temp.from_1;
        int atom_2 = v_temp.from_2;
        float a = v_temp.a;
        VECTOR force_v = force[atom_v];
        atomicAdd(&force[atom_1].x, a * force_v.x);
        atomicAdd(&force[atom_1].y, a * force_v.y);
        atomicAdd(&force[atom_1].z, a * force_v.z);

        atomicAdd(&force[atom_2].x, (1 - a) * force_v.x);
        atomicAdd(&force[atom_2].y, (1 - a) * force_v.y);
        atomicAdd(&force[atom_2].z, (1 - a) * force_v.z);

        force_v.x = 0.0f;
        force_v.y = 0.0f;
        force_v.z = 0.0f;
        force[atom_v] = force_v;
    }
}

static __global__ void v2_Force_Redistribute(
    const int virtual_numbers, const VIRTUAL_TYPE_2* v_info, const VECTOR* crd,
    const LTMatrix3 cell, const LTMatrix3 rcell, VECTOR* force)
{
#ifdef USE_GPU
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < virtual_numbers)
#else
#pragma omp parallel for
    for (int i = 0; i < virtual_numbers; i++)
#endif
    {
        VIRTUAL_TYPE_2 v_temp = v_info[i];
        int atom_v = v_temp.virtual_atom;
        int atom_1 = v_temp.from_1;
        int atom_2 = v_temp.from_2;
        int atom_3 = v_temp.from_3;
        float a = v_temp.a;
        float b = v_temp.b;
        VECTOR force_v = force[atom_v];
        atomicAdd(&force[atom_1].x, (1 - a - b) * force_v.x);
        atomicAdd(&force[atom_1].y, (1 - a - b) * force_v.y);
        atomicAdd(&force[atom_1].z, (1 - a - b) * force_v.z);

        atomicAdd(&force[atom_2].x, a * force_v.x);
        atomicAdd(&force[atom_2].y, a * force_v.y);
        atomicAdd(&force[atom_2].z, a * force_v.z);

        atomicAdd(&force[atom_3].x, b * force_v.x);
        atomicAdd(&force[atom_3].y, b * force_v.y);
        atomicAdd(&force[atom_3].z, b * force_v.z);

        force_v.x = 0.0f;
        force_v.y = 0.0f;
        force_v.z = 0.0f;
        force[atom_v] = force_v;
    }
}

static __global__ void v2_Force_Redistribute_No_Atomic(
    const int virtual_numbers, const VIRTUAL_TYPE_2* v_info, const VECTOR* crd,
    const LTMatrix3 cell, const LTMatrix3 rcell, VECTOR* force)
{
#ifdef USE_GPU
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < virtual_numbers)
#else
#pragma omp parallel for
    for (int i = 0; i < virtual_numbers; i++)
#endif
    {
        VIRTUAL_TYPE_2 v_temp = v_info[i];
        int atom_v = v_temp.virtual_atom;
        int atom_1 = v_temp.from_1;
        int atom_2 = v_temp.from_2;
        int atom_3 = v_temp.from_3;
        float a = v_temp.a;
        float b = v_temp.b;
        VECTOR force_v = force[atom_v];

        force[atom_1].x += (1 - a - b) * force_v.x;
        force[atom_1].y += (1 - a - b) * force_v.y;
        force[atom_1].z += (1 - a - b) * force_v.z;

        force[atom_2].x += a * force_v.x;
        force[atom_2].y += a * force_v.y;
        force[atom_2].z += a * force_v.z;

        force[atom_3].x += b * force_v.x;
        force[atom_3].y += b * force_v.y;
        force[atom_3].z += b * force_v.z;

        force[atom_v] = {0, 0, 0};
    }
}

static __global__ void v3_Force_Redistribute(
    const int virtual_numbers, const VIRTUAL_TYPE_3* v_info, const VECTOR* crd,
    const LTMatrix3 cell, const LTMatrix3 rcell, VECTOR* force)
{
#ifdef USE_GPU
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < virtual_numbers)
#else
#pragma omp parallel for
    for (int i = 0; i < virtual_numbers; i++)
#endif
    {
        VIRTUAL_TYPE_3 v_temp = v_info[i];
        int atom_v = v_temp.virtual_atom;
        int atom_1 = v_temp.from_1;
        int atom_2 = v_temp.from_2;
        int atom_3 = v_temp.from_3;
        float d = v_temp.d;
        float k = v_temp.k;
        VECTOR force_v = force[atom_v];

        const VECTOR r1 = crd[atom_1];
        const VECTOR r2 = crd[atom_2];
        const VECTOR r3 = crd[atom_3];
        const VECTOR rv = crd[atom_v];

        VECTOR r21 = Get_Periodic_Displacement(r2, r1, cell, rcell);
        VECTOR r32 = Get_Periodic_Displacement(r3, r2, cell, rcell);
        VECTOR rv1 = Get_Periodic_Displacement(rv, r1, cell, rcell);

        VECTOR temp = r21 + k * r32;
        float factor = d * rnorm3df(temp.x, temp.y, temp.z);

        temp = (rv1 * force_v) / (rv1 * rv1) * rv1;
        temp = factor * (force_v - temp);
        VECTOR force_1 = force_v - temp;
        VECTOR force_2 = (1 - k) * temp;
        VECTOR force_3 = k * temp;

        atomicAdd(&force[atom_1].x, force_1.x);
        atomicAdd(&force[atom_1].y, force_1.y);
        atomicAdd(&force[atom_1].z, force_1.z);

        atomicAdd(&force[atom_2].x, force_2.x);
        atomicAdd(&force[atom_2].y, force_2.y);
        atomicAdd(&force[atom_2].z, force_2.z);

        atomicAdd(&force[atom_3].x, force_3.x);
        atomicAdd(&force[atom_3].y, force_3.y);
        atomicAdd(&force[atom_3].z, force_3.z);

        force_v.x = 0.0f;
        force_v.y = 0.0f;
        force_v.z = 0.0f;
        force[atom_v] = force_v;
    }
}

static __global__ void v4_Force_Redistribute(const int atom_numbers,
                                             const int virtual_atom,
                                             const int* from_atoms,
                                             const float* weight, VECTOR* frc)
{
    VECTOR new_force = frc[virtual_atom];
    float this_weight;
    float* this_frc;
#ifdef USE_GPU
    for (int i = threadIdx.x; i < atom_numbers; i += blockDim.x)
#else
#pragma omp parallel for private(new_force) firstprivate(this_weight, this_frc)
    for (int i = 0; i < atom_numbers; i++)
#endif
    {
        this_weight = weight[i];
        this_frc = &frc[from_atoms[i]].x;
        atomicAdd(this_frc, this_weight * new_force.x);
        atomicAdd(this_frc + 1, this_weight * new_force.y);
        atomicAdd(this_frc + 2, this_weight * new_force.z);
    }
#ifdef USE_GPU
    __syncthreads();
    if (threadIdx.x == 0)
#endif
    {
        new_force.x = 0;
        new_force.y = 0;
        new_force.z = 0;
        frc[virtual_atom] = new_force;
    }
}

void VIRTUAL_INFORMATION::Initial(CONTROLLER* controller,
                                  COLLECTIVE_VARIABLE_CONTROLLER* cv_controller,
                                  int atom_numbers, int no_direct_vatom_numbers,
                                  CheckMap cv_vatom_name, float* h_mass,
                                  int* system_freedom, CONECT* connectivity,
                                  const char* module_name)
{
    if (module_name == NULL)
    {
        strcpy(this->module_name, "virtual_atom");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }
    const auto& system_virtual_atoms = Xponge::system.virtual_atoms.records;
    Xponge::VirtualAtoms local_virtual_atoms;
    const std::vector<Xponge::VirtualAtomRecord>* records_to_use = NULL;
    if (module_name == NULL)
    {
        records_to_use = &system_virtual_atoms;
    }
    else if (controller->Command_Exist(this->module_name, "in_file"))
    {
        Xponge::Native_Load_Virtual_Atoms(&local_virtual_atoms, controller,
                                          this->module_name);
        records_to_use = &local_virtual_atoms.records;
    }
    bool has_in_file = records_to_use != NULL && !records_to_use->empty();
    if (has_in_file || no_direct_vatom_numbers > 0)
    {
        controller->printf("START INITIALIZING VIRTUAL ATOM\n");
        Malloc_Safely((void**)&virtual_level,
                      sizeof(int) * (atom_numbers + no_direct_vatom_numbers));
        for (int i = 0; i < atom_numbers + no_direct_vatom_numbers; i++)
        {
            virtual_level[i] = 0;
        }

        int virtual_type;
        int virtual_atom;

        // 文件会从头到尾读三遍，分别确定每个原子的虚拟等级（因为可能存在坐标依赖于虚原子的虚原子，所以不得不如此做）
        // 第一遍确定虚拟原子的层级
        controller->printf("    Start reading virtual levels\n");
        if (has_in_file)
        {
            for (const auto& record : *records_to_use)
            {
                virtual_type = record.type;
                virtual_atom = record.virtual_atom;
                switch (virtual_type)
                {
                    case 0:
                        virtual_level[virtual_atom] =
                            virtual_level[record.from[0]] + 1;
                        break;

                    case 1:
                        virtual_level[virtual_atom] =
                            std::max(virtual_level[record.from[0]],
                                     virtual_level[record.from[1]]) +
                            1;
                        break;

                    case 2:
                        virtual_level[virtual_atom] =
                            std::max(virtual_level[record.from[0]],
                                     virtual_level[record.from[1]]);
                        virtual_level[virtual_atom] =
                            std::max(virtual_level[virtual_atom],
                                     virtual_level[record.from[2]]) +
                            1;
                        // 添加信息至成键信息
                        connectivity[0][virtual_atom].insert(record.from[0]);
                        connectivity[0][record.from[0]].insert(virtual_atom);
                        break;

                    case 3:
                        virtual_level[virtual_atom] =
                            std::max(virtual_level[record.from[0]],
                                     virtual_level[record.from[1]]);
                        virtual_level[virtual_atom] =
                            std::max(virtual_level[virtual_atom],
                                     virtual_level[record.from[2]]) +
                            1;
                        // 添加信息至成键信息
                        connectivity[0][virtual_atom].insert(record.from[0]);
                        connectivity[0][record.from[0]].insert(virtual_atom);
                        break;

                    default:
                        controller->Throw_SPONGE_Error(
                            spongeErrorBadFileFormat,
                            "VIRTUAL_INFORMATION::Initial",
                            "Reason:\n\tvirtual_atom_in_file contains an "
                            "unsupported virtual atom type\n");
                }
            }
        }
        // 利用CV信息补全虚拟原子层级
        for (CheckMap::iterator iter = cv_vatom_name.begin();
             iter != cv_vatom_name.end(); iter++)
        {
            virtual_atom = iter->second + atom_numbers;
            std::vector<int> h_from =
                cv_controller->Ask_For_Indefinite_Length_Int_Parameter(
                    iter->first.c_str(), "atom");
            for (int i = 0; i < h_from.size(); i++)
            {
                if (h_from[i] >= atom_numbers + cv_vatom_name.size())
                {
                    char error_reason[CHAR_LENGTH_MAX];
                    sprintf(error_reason,
                            "Reason:\n\tError: atom id (%d) >= atom_numbers + "
                            "cv_virtual_atom_numbers (%d)\n",
                            h_from[i],
                            atom_numbers + (int)cv_vatom_name.size());
                    controller->Throw_SPONGE_Error(
                        spongeErrorOverflow, "VIRTUAL_INFORMATION::Initial",
                        error_reason);
                }
                virtual_level[virtual_atom] = std::max(
                    virtual_level[virtual_atom], virtual_level[h_from[i]]);
            }
            virtual_level[virtual_atom] += 1;
        }
        // 层级初始化
        max_level = 0;
        int total_virtual_atoms = 0;
        for (int i = 0; i < (atom_numbers + no_direct_vatom_numbers); i++)
        {
            int vli = virtual_level[i];
            if (vli > 0)
            {
                total_virtual_atoms++;
            }
            if (vli > max_level)
            {
                for (int j = 0; j < vli - max_level; j++)
                {
                    VIRTUAL_LAYER_INFORMATION virtual_layer;
                    virtual_layer_info.push_back(virtual_layer);
                }
                max_level = vli;
            }
        }
        system_freedom[0] -=
            3 * (total_virtual_atoms - no_direct_vatom_numbers);
        controller->printf("        Virtual Atoms Max Level is %d\n",
                           max_level);
        controller->printf("        Virtual Atoms Number is %d\n",
                           total_virtual_atoms);
        controller->printf("            FF Virtual Atoms Number is %d\n",
                           total_virtual_atoms - no_direct_vatom_numbers);
        controller->printf("            CV Virtual Atoms Number is %d\n",
                           no_direct_vatom_numbers);
        controller->printf("    End reading virtual levels\n");
        // 第二遍确定虚拟原子每一层的个数
        controller->printf(
            "    Start reading virtual type numbers in different levels\n");
        if (has_in_file)
        {
            for (const auto& record : *records_to_use)
            {
                virtual_type = record.type;
                virtual_atom = record.virtual_atom;
                VIRTUAL_LAYER_INFORMATION* temp_vl =
                    &virtual_layer_info[virtual_level[virtual_atom] - 1];
                switch (virtual_type)
                {
                    case 0:
                        temp_vl->v0_info.virtual_numbers += 1;
                        break;
                    case 1:
                        temp_vl->v1_info.virtual_numbers += 1;
                        break;
                    case 2:
                        temp_vl->v2_info.virtual_numbers += 1;
                        break;
                    case 3:
                        temp_vl->v3_info.virtual_numbers += 1;
                        break;
                    default:
                        break;
                }
            }
        }

        for (CheckMap::iterator iter = cv_vatom_name.begin();
             iter != cv_vatom_name.end(); iter++)
        {
            virtual_atom = iter->second + atom_numbers;
            std::string strs =
                cv_controller->Command(iter->first.c_str(), "vatom_type");
            VIRTUAL_LAYER_INFORMATION* temp_vl =
                &virtual_layer_info[virtual_level[virtual_atom] - 1];
            if (strs == "center_of_mass" || strs == "center")
            {
                temp_vl->v4_info.virtual_numbers += 1;
            }
        }

        // 每层的每种虚拟原子初始化
        for (int layer = 0; layer < max_level; layer++)
        {
            controller->printf("        Virutual level %d:\n", layer);
            VIRTUAL_LAYER_INFORMATION* temp_vl = &virtual_layer_info[layer];
            if (temp_vl->v0_info.virtual_numbers > 0)
            {
                controller->printf(
                    "            Virtual type 0 atom numbers is %d\n",
                    temp_vl->v0_info.virtual_numbers);
                Malloc_Safely(
                    (void**)&temp_vl->v0_info.h_virtual_type_0,
                    sizeof(VIRTUAL_TYPE_0) * temp_vl->v0_info.virtual_numbers);
            }
            if (temp_vl->v1_info.virtual_numbers > 0)
            {
                controller->printf(
                    "            Virtual type 1 atom numbers is %d\n",
                    temp_vl->v1_info.virtual_numbers);
                Malloc_Safely(
                    (void**)&temp_vl->v1_info.h_virtual_type_1,
                    sizeof(VIRTUAL_TYPE_1) * temp_vl->v1_info.virtual_numbers);
            }
            if (temp_vl->v2_info.virtual_numbers > 0)
            {
                controller->printf(
                    "            Virtual type 2 atom numbers is %d\n",
                    temp_vl->v2_info.virtual_numbers);
                Malloc_Safely(
                    (void**)&temp_vl->v2_info.h_virtual_type_2,
                    sizeof(VIRTUAL_TYPE_2) * temp_vl->v2_info.virtual_numbers);
            }
            if (temp_vl->v3_info.virtual_numbers > 0)
            {
                controller->printf(
                    "            Virtual type 3 atom numbers is %d\n",
                    temp_vl->v3_info.virtual_numbers);
                Malloc_Safely(
                    (void**)&temp_vl->v3_info.h_virtual_type_3,
                    sizeof(VIRTUAL_TYPE_3) * temp_vl->v3_info.virtual_numbers);
            }
            if (temp_vl->v4_info.virtual_numbers > 0)
            {
                controller->printf(
                    "            Virtual type 4 atom numbers is %d\n",
                    temp_vl->v4_info.virtual_numbers);
                Malloc_Safely(
                    (void**)&temp_vl->v4_info.h_virtual_type_4,
                    sizeof(VIRTUAL_TYPE_4) * temp_vl->v4_info.virtual_numbers);
            }
        }
        controller->printf(
            "    End reading virtual type numbers in different levels\n");
        // 第三遍将所有信息填入
        controller->printf(
            "    Start reading information for every virtual atom\n");
        if (has_in_file)
        {
            std::map<int, int> count0, count1, count2, count3;
            for (int i = 0; i < virtual_layer_info.size(); i++)
            {
                count0[i] = 0;
                count1[i] = 0;
                count2[i] = 0;
                count3[i] = 0;
            }
            for (const auto& record : *records_to_use)
            {
                virtual_type = record.type;
                virtual_atom = record.virtual_atom;
                int this_level = virtual_level[virtual_atom] - 1;
                VIRTUAL_LAYER_INFORMATION* temp_vl =
                    &virtual_layer_info[this_level];
                switch (virtual_type)
                {
                    case 0:
                        temp_vl->v0_info.h_virtual_type_0[count0[this_level]]
                            .virtual_atom = record.virtual_atom;
                        temp_vl->v0_info.h_virtual_type_0[count0[this_level]]
                            .from_1 = record.from[0];
                        temp_vl->v0_info.h_virtual_type_0[count0[this_level]]
                            .h_double = 2 * record.parameter[0];
                        count0[this_level]++;
                        break;

                    case 1:
                        temp_vl->v1_info.h_virtual_type_1[count1[this_level]]
                            .virtual_atom = record.virtual_atom;
                        temp_vl->v1_info.h_virtual_type_1[count1[this_level]]
                            .from_1 = record.from[0];
                        temp_vl->v1_info.h_virtual_type_1[count1[this_level]]
                            .from_2 = record.from[1];
                        temp_vl->v1_info.h_virtual_type_1[count1[this_level]]
                            .a = record.parameter[0];
                        count1[this_level]++;
                        break;

                    case 2:
                        temp_vl->v2_info.h_virtual_type_2[count2[this_level]]
                            .virtual_atom = record.virtual_atom;
                        temp_vl->v2_info.h_virtual_type_2[count2[this_level]]
                            .from_1 = record.from[0];
                        temp_vl->v2_info.h_virtual_type_2[count2[this_level]]
                            .from_2 = record.from[1];
                        temp_vl->v2_info.h_virtual_type_2[count2[this_level]]
                            .from_3 = record.from[2];
                        temp_vl->v2_info.h_virtual_type_2[count2[this_level]]
                            .a = record.parameter[0];
                        temp_vl->v2_info.h_virtual_type_2[count2[this_level]]
                            .b = record.parameter[1];
                        count2[this_level]++;
                        break;

                    case 3:
                        temp_vl->v3_info.h_virtual_type_3[count3[this_level]]
                            .virtual_atom = record.virtual_atom;
                        temp_vl->v3_info.h_virtual_type_3[count3[this_level]]
                            .from_1 = record.from[0];
                        temp_vl->v3_info.h_virtual_type_3[count3[this_level]]
                            .from_2 = record.from[1];
                        temp_vl->v3_info.h_virtual_type_3[count3[this_level]]
                            .from_3 = record.from[2];
                        temp_vl->v3_info.h_virtual_type_3[count3[this_level]]
                            .d = record.parameter[0];
                        temp_vl->v3_info.h_virtual_type_3[count3[this_level]]
                            .k = record.parameter[1];
                        count3[this_level]++;
                        break;

                    default:
                        break;
                }
            }
        }
        std::map<int, int> count4;
        for (int i = 0; i < virtual_layer_info.size(); i++)
        {
            count4[i] = 0;
        }
        for (CheckMap::iterator iter = cv_vatom_name.begin();
             iter != cv_vatom_name.end(); iter++)
        {
            virtual_atom = iter->second + atom_numbers;
            std::string virtual_type =
                cv_controller->Command(iter->first.c_str(), "vatom_type");
            int this_level = virtual_level[virtual_atom] - 1;
            VIRTUAL_LAYER_INFORMATION* temp_vl =
                &virtual_layer_info[this_level];
            temp_vl->v4_info.h_virtual_type_4[count4[this_level]].virtual_atom =
                virtual_atom;
            std::vector<int> h_from =
                cv_controller->Ask_For_Indefinite_Length_Int_Parameter(
                    iter->first.c_str(), "atom");
            temp_vl->v4_info.h_virtual_type_4[count4[this_level]].atom_numbers =
                h_from.size();
            Malloc_Safely(
                (void**)&temp_vl->v4_info.h_virtual_type_4[count4[this_level]]
                    .h_from,
                sizeof(int) *
                    temp_vl->v4_info.h_virtual_type_4[count4[this_level]]
                        .atom_numbers);
            memcpy(temp_vl->v4_info.h_virtual_type_4[count4[this_level]].h_from,
                   &h_from[0], sizeof(int) * h_from.size());
            if (virtual_type == "center")
            {
                Device_Malloc_And_Copy_Safely(
                    (void**)&temp_vl->v4_info
                        .h_virtual_type_4[count4[this_level]]
                        .d_from,
                    temp_vl->v4_info.h_virtual_type_4[count4[this_level]]
                        .h_from,
                    sizeof(int) *
                        temp_vl->v4_info.h_virtual_type_4[count4[this_level]]
                            .atom_numbers);
                Malloc_Safely(
                    (void**)&temp_vl->v4_info
                        .h_virtual_type_4[count4[this_level]]
                        .h_weight,
                    sizeof(float) *
                        temp_vl->v4_info.h_virtual_type_4[count4[this_level]]
                            .atom_numbers);
                std::vector<float> weights =
                    cv_controller->Ask_For_Indefinite_Length_Float_Parameter(
                        iter->first.c_str(), "weight");
                if (weights.size() != h_from.size())
                {
                    std::string error_reason =
                        "Reason:\n\tthe number of weights is not equal to the "
                        "number of atoms for the CV virtual atom ";
                    error_reason += iter->first;
                    error_reason += "\n";
                    cv_controller->Throw_SPONGE_Error(
                        spongeErrorConflictingCommand,
                        "VIRTUAL_INFORMATION::Initial", error_reason.c_str());
                }
                for (int i = 0; i < weights.size(); i++)
                {
                    temp_vl->v4_info.h_virtual_type_4[count4[this_level]]
                        .h_weight[i] = weights[i];
                }
                Device_Malloc_And_Copy_Safely(
                    (void**)&temp_vl->v4_info
                        .h_virtual_type_4[count4[this_level]]
                        .d_weight,
                    temp_vl->v4_info.h_virtual_type_4[count4[this_level]]
                        .h_weight,
                    sizeof(float) *
                        temp_vl->v4_info.h_virtual_type_4[count4[this_level]]
                            .atom_numbers);
            }
            else if (virtual_type == "center_of_mass")
            {
                Device_Malloc_And_Copy_Safely(
                    (void**)&temp_vl->v4_info
                        .h_virtual_type_4[count4[this_level]]
                        .d_from,
                    temp_vl->v4_info.h_virtual_type_4[count4[this_level]]
                        .h_from,
                    sizeof(int) *
                        temp_vl->v4_info.h_virtual_type_4[count4[this_level]]
                            .atom_numbers);
                Malloc_Safely(
                    (void**)&temp_vl->v4_info
                        .h_virtual_type_4[count4[this_level]]
                        .h_weight,
                    sizeof(float) *
                        temp_vl->v4_info.h_virtual_type_4[count4[this_level]]
                            .atom_numbers);
                float total_mass = 0;
                int atom_i;
                for (int i = 0;
                     i < temp_vl->v4_info.h_virtual_type_4[count4[this_level]]
                             .atom_numbers;
                     i++)
                {
                    atom_i =
                        temp_vl->v4_info.h_virtual_type_4[count4[this_level]]
                            .h_from[i];
                    total_mass += h_mass[atom_i];
                }
                for (int i = 0;
                     i < temp_vl->v4_info.h_virtual_type_4[count4[this_level]]
                             .atom_numbers;
                     i++)
                {
                    atom_i =
                        temp_vl->v4_info.h_virtual_type_4[count4[this_level]]
                            .h_from[i];
                    temp_vl->v4_info.h_virtual_type_4[count4[this_level]]
                        .h_weight[i] = h_mass[atom_i] / total_mass;
                }
                Device_Malloc_And_Copy_Safely(
                    (void**)&temp_vl->v4_info
                        .h_virtual_type_4[count4[this_level]]
                        .d_weight,
                    temp_vl->v4_info.h_virtual_type_4[count4[this_level]]
                        .h_weight,
                    sizeof(float) *
                        temp_vl->v4_info.h_virtual_type_4[count4[this_level]]
                            .atom_numbers);
            }
            count4[this_level]++;
        }
        // 每层的数据信息传到cuda上去
        for (int layer = 0; layer < max_level; layer++)
        {
            VIRTUAL_LAYER_INFORMATION* temp_vl = &virtual_layer_info[layer];
            if (temp_vl->v0_info.virtual_numbers > 0)
                Device_Malloc_And_Copy_Safely(
                    (void**)&temp_vl->v0_info.d_virtual_type_0,
                    temp_vl->v0_info.h_virtual_type_0,
                    sizeof(VIRTUAL_TYPE_0) * temp_vl->v0_info.virtual_numbers);
            Device_Malloc_Safely(
                (void**)&temp_vl->v0_info.l_virtual_type_0,
                sizeof(VIRTUAL_TYPE_0) * temp_vl->v0_info.virtual_numbers);
            Device_Malloc_Safely((void**)&temp_vl->v0_info.d_local_numbers,
                                 sizeof(int));
            if (temp_vl->v1_info.virtual_numbers > 0)
                Device_Malloc_And_Copy_Safely(
                    (void**)&temp_vl->v1_info.d_virtual_type_1,
                    temp_vl->v1_info.h_virtual_type_1,
                    sizeof(VIRTUAL_TYPE_1) * temp_vl->v1_info.virtual_numbers);
            Device_Malloc_Safely(
                (void**)&temp_vl->v1_info.l_virtual_type_1,
                sizeof(VIRTUAL_TYPE_1) * temp_vl->v1_info.virtual_numbers);
            Device_Malloc_Safely((void**)&temp_vl->v1_info.d_local_numbers,
                                 sizeof(int));
            if (temp_vl->v2_info.virtual_numbers > 0)
            {
                Device_Malloc_And_Copy_Safely(
                    (void**)&temp_vl->v2_info.d_virtual_type_2,
                    temp_vl->v2_info.h_virtual_type_2,
                    sizeof(VIRTUAL_TYPE_2) * temp_vl->v2_info.virtual_numbers);
                Device_Malloc_Safely(
                    (void**)&temp_vl->v2_info.l_virtual_type_2,
                    sizeof(VIRTUAL_TYPE_2) * temp_vl->v2_info.virtual_numbers);
                Device_Malloc_Safely((void**)&temp_vl->v2_info.d_local_numbers,
                                     sizeof(int));
            }
            if (temp_vl->v3_info.virtual_numbers > 0)
            {
                Device_Malloc_And_Copy_Safely(
                    (void**)&temp_vl->v3_info.d_virtual_type_3,
                    temp_vl->v3_info.d_virtual_type_3,
                    sizeof(VIRTUAL_TYPE_3) * temp_vl->v3_info.virtual_numbers);
                Device_Malloc_Safely(
                    (void**)&temp_vl->v3_info.l_virtual_type_3,
                    sizeof(VIRTUAL_TYPE_3) * temp_vl->v3_info.virtual_numbers);
                Device_Malloc_Safely((void**)&temp_vl->v3_info.d_local_numbers,
                                     sizeof(int));
            }
        }
        controller->printf(
            "    End reading information for every virtual atom\n");

        is_initialized = 1;
        if (is_initialized && !is_controller_printf_initialized)
        {
            is_controller_printf_initialized = 1;
            controller->printf("    structure last modify date is %d\n",
                               last_modify_date);
        }

        for (int layer = 0; layer < max_level; layer++)
        {
            std::vector<bool> mark(atom_numbers, 0);
            VIRTUAL_LAYER_INFORMATION* temp_vl = &virtual_layer_info[layer];
            VIRTUAL_TYPE_2* v_info = temp_vl->v2_info.h_virtual_type_2;
            int virtual_numbers = temp_vl->v2_info.local_numbers;
            for (int i = 0; i < virtual_numbers; ++i)
            {
                for (auto x :
                     {v_info[i].from_1, v_info[i].from_2, v_info[i].from_3})
                {
                    if (!mark[x])
                    {
                        mark[x] = 1;
                    }
                    else
                    {
                        need_atomic = true;
                    }
                }
            }
        }
        controller->printf("END INITIALIZING VIRTUAL ATOM\n\n");
    }
    else
    {
        controller->printf("VIRTUAL ATOM IS NOT INITIALIZED\n\n");
    }
}

void VIRTUAL_INFORMATION::Coordinate_Refresh(VECTOR* crd, const LTMatrix3 cell,
                                             const LTMatrix3 rcell)
{
    if (is_initialized)
    {
        // 每层之间需要串行计算，层内并行计算
        for (int layer = 0; layer < max_level; layer++)
        {
            VIRTUAL_LAYER_INFORMATION* temp_vl = &virtual_layer_info[layer];
            if (temp_vl->v0_info.local_numbers > 0)
                Launch_Device_Kernel(v0_Coordinate_Refresh,
                                     (temp_vl->v0_info.local_numbers +
                                      CONTROLLER::device_max_thread - 1) /
                                         CONTROLLER::device_max_thread,
                                     CONTROLLER::device_max_thread, 0, NULL,
                                     temp_vl->v0_info.local_numbers,
                                     temp_vl->v0_info.l_virtual_type_0, crd,
                                     cell, rcell);

            if (temp_vl->v1_info.local_numbers > 0)
                Launch_Device_Kernel(v1_Coordinate_Refresh,
                                     (temp_vl->v1_info.local_numbers +
                                      CONTROLLER::device_max_thread - 1) /
                                         CONTROLLER::device_max_thread,
                                     CONTROLLER::device_max_thread, 0, NULL,
                                     temp_vl->v1_info.local_numbers,
                                     temp_vl->v1_info.l_virtual_type_1, crd,
                                     cell, rcell);

            if (temp_vl->v2_info.local_numbers > 0)
                Launch_Device_Kernel(v2_Coordinate_Refresh,
                                     (temp_vl->v2_info.local_numbers +
                                      CONTROLLER::device_max_thread - 1) /
                                         CONTROLLER::device_max_thread,
                                     CONTROLLER::device_max_thread, 0, NULL,
                                     temp_vl->v2_info.local_numbers,
                                     temp_vl->v2_info.l_virtual_type_2, crd,
                                     cell, rcell);

            if (temp_vl->v3_info.local_numbers > 0)
                Launch_Device_Kernel(v3_Coordinate_Refresh,
                                     (temp_vl->v3_info.local_numbers +
                                      CONTROLLER::device_max_thread - 1) /
                                         CONTROLLER::device_max_thread,
                                     CONTROLLER::device_max_thread, 0, NULL,
                                     temp_vl->v3_info.local_numbers,
                                     temp_vl->v3_info.l_virtual_type_3, crd,
                                     cell, rcell);
        }
    }
}

void VIRTUAL_INFORMATION::Force_Redistribute(const VECTOR* crd,
                                             const LTMatrix3 cell,
                                             const LTMatrix3 rcell, VECTOR* frc)
{
    if (is_initialized)
    {
        // 每层之间需要串行逆向计算，层内并行计算
        for (int layer = max_level - 1; layer >= 0; layer--)
        {
            VIRTUAL_LAYER_INFORMATION* temp_vl = &virtual_layer_info[layer];
            if (temp_vl->v0_info.local_numbers > 0)
            {
                Launch_Device_Kernel(v0_Force_Redistribute,
                                     (temp_vl->v0_info.local_numbers +
                                      CONTROLLER::device_max_thread - 1) /
                                         CONTROLLER::device_max_thread,
                                     CONTROLLER::device_max_thread, 0, NULL,
                                     temp_vl->v0_info.local_numbers,
                                     temp_vl->v0_info.l_virtual_type_0, crd,
                                     cell, rcell, frc);
            }
            if (temp_vl->v1_info.local_numbers > 0)
            {
                Launch_Device_Kernel(v1_Force_Redistribute,
                                     (temp_vl->v1_info.local_numbers +
                                      CONTROLLER::device_max_thread - 1) /
                                         CONTROLLER::device_max_thread,
                                     CONTROLLER::device_max_thread, 0, NULL,
                                     temp_vl->v1_info.local_numbers,
                                     temp_vl->v1_info.l_virtual_type_1, crd,
                                     cell, rcell, frc);
            }
            if (temp_vl->v3_info.local_numbers > 0)
            {
                Launch_Device_Kernel(v3_Force_Redistribute,
                                     (temp_vl->v3_info.local_numbers +
                                      CONTROLLER::device_max_thread - 1) /
                                         CONTROLLER::device_max_thread,
                                     CONTROLLER::device_max_thread, 0, NULL,
                                     temp_vl->v3_info.local_numbers,
                                     temp_vl->v3_info.l_virtual_type_3, crd,
                                     cell, rcell, frc);
            }

            if (temp_vl->v2_info.local_numbers > 0)
            {
                if (need_atomic)
                {
                    Launch_Device_Kernel(v2_Force_Redistribute,
                                         (temp_vl->v2_info.local_numbers +
                                          CONTROLLER::device_max_thread - 1) /
                                             CONTROLLER::device_max_thread,
                                         CONTROLLER::device_max_thread, 0, NULL,
                                         temp_vl->v2_info.local_numbers,
                                         temp_vl->v2_info.l_virtual_type_2, crd,
                                         cell, rcell, frc);
                }
                else
                {
                    Launch_Device_Kernel(v2_Force_Redistribute_No_Atomic,
                                         (temp_vl->v2_info.local_numbers +
                                          CONTROLLER::device_max_thread - 1) /
                                             CONTROLLER::device_max_thread,
                                         CONTROLLER::device_max_thread, 0, NULL,
                                         temp_vl->v2_info.local_numbers,
                                         temp_vl->v2_info.l_virtual_type_2, crd,
                                         cell, rcell, frc);
                }
            }
        }
    }
}

void VIRTUAL_INFORMATION::Coordinate_Refresh_CV(VECTOR* crd,
                                                const LTMatrix3 cell,
                                                const LTMatrix3 rcell)
{
    if (is_initialized)
    {
        // 每层之间需要串行计算，层内并行计算
        for (int layer = 0; layer < max_level; layer++)
        {
            VIRTUAL_LAYER_INFORMATION* temp_vl = &virtual_layer_info[layer];
            // 预留v4质心接口
            VIRTUAL_TYPE_4* temp_vl4;
            for (int iv4 = 0; iv4 < temp_vl->v4_info.virtual_numbers; iv4++)
            {
                temp_vl4 = temp_vl->v4_info.h_virtual_type_4 + iv4;
                Launch_Device_Kernel(
                    v4_Coordinate_Refresh, 1, CONTROLLER::device_warp, 0, NULL,
                    temp_vl4->atom_numbers, temp_vl4->virtual_atom,
                    temp_vl4->d_from, temp_vl4->d_weight, crd);
            }
        }
    }
}

void VIRTUAL_INFORMATION::Force_Redistribute_CV(const VECTOR* crd,
                                                const LTMatrix3 cell,
                                                const LTMatrix3 rcell,
                                                VECTOR* frc)
{
    if (is_initialized)
    {
        // 每层之间需要串行逆向计算，层内并行计算
        for (int layer = max_level - 1; layer >= 0; layer--)
        {
            VIRTUAL_LAYER_INFORMATION* temp_vl = &virtual_layer_info[layer];
            // 预留v4质心接口
            VIRTUAL_TYPE_4* temp_vl4;
            for (int iv4 = 0; iv4 < temp_vl->v4_info.virtual_numbers; iv4++)
            {
                temp_vl4 = temp_vl->v4_info.h_virtual_type_4 + iv4;
                Launch_Device_Kernel(
                    v4_Force_Redistribute, 1, CONTROLLER::device_warp, 0, NULL,
                    temp_vl4->atom_numbers, temp_vl4->virtual_atom,
                    temp_vl4->d_from, temp_vl4->d_weight, frc);
            }
        }
    }
}

static __global__ void get_local_device_V0(int virtual_numbers,
                                           int* local_numbers,
                                           VIRTUAL_TYPE_0* d_virtual_type_0,
                                           VIRTUAL_TYPE_0* l_virtual_type_0,
                                           const int* atom_local_id,
                                           const char* atom_local_label)
{
    local_numbers[0] = 0;
    for (int cluster = 0; cluster < virtual_numbers; cluster++)
    {
        int vatom = d_virtual_type_0[cluster].virtual_atom;
        int from1 = d_virtual_type_0[cluster].from_1;
        if (atom_local_label[vatom])
        {
            l_virtual_type_0[local_numbers[0]] = d_virtual_type_0[cluster];
            l_virtual_type_0[local_numbers[0]].virtual_atom =
                atom_local_id[vatom];
            l_virtual_type_0[local_numbers[0]].from_1 = atom_local_id[from1];
            local_numbers[0] += 1;
        }
    }
}

static __global__ void get_local_device_V1(int virtual_numbers,
                                           int* local_numbers,
                                           VIRTUAL_TYPE_1* d_virtual_type_1,
                                           VIRTUAL_TYPE_1* l_virtual_type_1,
                                           const int* atom_local_id,
                                           const char* atom_local_label)
{
    local_numbers[0] = 0;
    for (int cluster = 0; cluster < virtual_numbers; cluster++)
    {
        int vatom = d_virtual_type_1[cluster].virtual_atom;
        int from1 = d_virtual_type_1[cluster].from_1;
        int from2 = d_virtual_type_1[cluster].from_2;
        if (atom_local_label[vatom])
        {
            l_virtual_type_1[local_numbers[0]] = d_virtual_type_1[cluster];
            l_virtual_type_1[local_numbers[0]].virtual_atom =
                atom_local_id[vatom];
            l_virtual_type_1[local_numbers[0]].from_1 = atom_local_id[from1];
            l_virtual_type_1[local_numbers[0]].from_2 = atom_local_id[from2];
            local_numbers[0] += 1;
        }
    }
}

static __global__ void get_local_device_V2(int virtual_numbers,
                                           int* local_numbers,
                                           VIRTUAL_TYPE_2* d_virtual_type_2,
                                           VIRTUAL_TYPE_2* l_virtual_type_2,
                                           const int* atom_local_id,
                                           const char* atom_local_label)
{
    local_numbers[0] = 0;
    for (int cluster = 0; cluster < virtual_numbers; cluster++)
    {
        int vatom = d_virtual_type_2[cluster].virtual_atom;
        int from1 = d_virtual_type_2[cluster].from_1;
        int from2 = d_virtual_type_2[cluster].from_2;
        int from3 = d_virtual_type_2[cluster].from_3;
        if (atom_local_label[vatom])
        {
            l_virtual_type_2[local_numbers[0]] = d_virtual_type_2[cluster];
            l_virtual_type_2[local_numbers[0]].virtual_atom =
                atom_local_id[vatom];
            l_virtual_type_2[local_numbers[0]].from_1 = atom_local_id[from1];
            l_virtual_type_2[local_numbers[0]].from_2 = atom_local_id[from2];
            l_virtual_type_2[local_numbers[0]].from_3 = atom_local_id[from3];
            local_numbers[0] += 1;
        }
    }
}

static __global__ void get_local_device_V3(int virtual_numbers,
                                           int* local_numbers,
                                           VIRTUAL_TYPE_3* d_virtual_type_3,
                                           VIRTUAL_TYPE_3* l_virtual_type_3,
                                           const int* atom_local_id,
                                           const char* atom_local_label)
{
    local_numbers[0] = 0;
    for (int cluster = 0; cluster < virtual_numbers; cluster++)
    {
        int vatom = d_virtual_type_3[cluster].virtual_atom;
        int from1 = d_virtual_type_3[cluster].from_1;
        int from2 = d_virtual_type_3[cluster].from_2;
        int from3 = d_virtual_type_3[cluster].from_3;
        if (atom_local_label[vatom])
        {
            l_virtual_type_3[local_numbers[0]] = d_virtual_type_3[cluster];
            l_virtual_type_3[local_numbers[0]].virtual_atom =
                atom_local_id[vatom];
            l_virtual_type_3[local_numbers[0]].from_1 = atom_local_id[from1];
            l_virtual_type_3[local_numbers[0]].from_2 = atom_local_id[from2];
            l_virtual_type_3[local_numbers[0]].from_3 = atom_local_id[from3];
            local_numbers[0] += 1;
        }
    }
}

// 预留get_local_device_V4接口

void VIRTUAL_INFORMATION::Get_Local(const int* atom_local_id,
                                    const char* atom_local_label,
                                    const int local_atom_numbers)
{
    if (!is_initialized) return;
    // 每层之间需要串行计算，层内并行计算
    for (int layer = 0; layer < max_level; layer++)
    {
        VIRTUAL_LAYER_INFORMATION* temp_vl = &virtual_layer_info[layer];

        if (temp_vl->v0_info.virtual_numbers > 0)
        {
            Launch_Device_Kernel(get_local_device_V0, 1, 1, 0, NULL,
                                 temp_vl->v0_info.virtual_numbers,
                                 temp_vl->v0_info.d_local_numbers,
                                 temp_vl->v0_info.d_virtual_type_0,
                                 temp_vl->v0_info.l_virtual_type_0,
                                 atom_local_id, atom_local_label);
            deviceMemcpy(&temp_vl->v0_info.local_numbers,
                         temp_vl->v0_info.d_local_numbers, sizeof(int),
                         deviceMemcpyDeviceToHost);
        }

        if (temp_vl->v1_info.virtual_numbers > 0)
        {
            Launch_Device_Kernel(get_local_device_V1, 1, 1, 0, NULL,
                                 temp_vl->v1_info.virtual_numbers,
                                 temp_vl->v1_info.d_local_numbers,
                                 temp_vl->v1_info.d_virtual_type_1,
                                 temp_vl->v1_info.l_virtual_type_1,
                                 atom_local_id, atom_local_label);
            deviceMemcpy(&temp_vl->v1_info.local_numbers,
                         temp_vl->v1_info.d_local_numbers, sizeof(int),
                         deviceMemcpyDeviceToHost);
        }

        if (temp_vl->v2_info.virtual_numbers > 0)
        {
            Launch_Device_Kernel(get_local_device_V2, 1, 1, 0, NULL,
                                 temp_vl->v2_info.virtual_numbers,
                                 temp_vl->v2_info.d_local_numbers,
                                 temp_vl->v2_info.d_virtual_type_2,
                                 temp_vl->v2_info.l_virtual_type_2,
                                 atom_local_id, atom_local_label);
            deviceMemcpy(&temp_vl->v2_info.local_numbers,
                         temp_vl->v2_info.d_local_numbers, sizeof(int),
                         deviceMemcpyDeviceToHost);
        }

        if (temp_vl->v3_info.virtual_numbers > 0)
        {
            Launch_Device_Kernel(get_local_device_V3, 1, 1, 0, NULL,
                                 temp_vl->v3_info.virtual_numbers,
                                 temp_vl->v3_info.d_local_numbers,
                                 temp_vl->v3_info.d_virtual_type_3,
                                 temp_vl->v3_info.l_virtual_type_3,
                                 atom_local_id, atom_local_label);
            deviceMemcpy(&temp_vl->v3_info.local_numbers,
                         temp_vl->v3_info.d_local_numbers, sizeof(int),
                         deviceMemcpyDeviceToHost);
        }

        // 预留v4质心接口
    }
}

void VIRTUAL_INFORMATION::update_ug_connectivity(CONECT* connectivity)
{
    if (!is_initialized) return;
    for (int i = 0; i < virtual_layer_info[0].v0_info.virtual_numbers; i++)
    {
        int atomv =
            virtual_layer_info[0].v0_info.h_virtual_type_0[i].virtual_atom;
        int atom1 = virtual_layer_info[0].v0_info.h_virtual_type_0[i].from_1;
        (*connectivity)[atomv].insert(atom1);
        (*connectivity)[atom1].insert(atomv);
    }
    for (int i = 0; i < virtual_layer_info[0].v1_info.virtual_numbers; i++)
    {
        int atomv =
            virtual_layer_info[0].v1_info.h_virtual_type_1[i].virtual_atom;
        int atom1 = virtual_layer_info[0].v1_info.h_virtual_type_1[i].from_1;
        int atom2 = virtual_layer_info[0].v1_info.h_virtual_type_1[i].from_2;
        (*connectivity)[atomv].insert(atom1);
        (*connectivity)[atomv].insert(atom2);
        (*connectivity)[atom1].insert(atomv);
        (*connectivity)[atom1].insert(atom2);
        (*connectivity)[atom2].insert(atom1);
        (*connectivity)[atom2].insert(atomv);
    }
    for (int i = 0; i < virtual_layer_info[0].v2_info.virtual_numbers; i++)
    {
        int atomv =
            virtual_layer_info[0].v2_info.h_virtual_type_2[i].virtual_atom;
        int atom1 = virtual_layer_info[0].v2_info.h_virtual_type_2[i].from_1;
        int atom2 = virtual_layer_info[0].v2_info.h_virtual_type_2[i].from_2;
        int atom3 = virtual_layer_info[0].v2_info.h_virtual_type_2[i].from_3;
        (*connectivity)[atomv].insert(atom1);
        (*connectivity)[atomv].insert(atom2);
        (*connectivity)[atomv].insert(atom3);
        (*connectivity)[atom1].insert(atomv);
        (*connectivity)[atom1].insert(atom2);
        (*connectivity)[atom1].insert(atom3);
        (*connectivity)[atom2].insert(atom1);
        (*connectivity)[atom2].insert(atomv);
        (*connectivity)[atom2].insert(atom3);
        (*connectivity)[atom3].insert(atom1);
        (*connectivity)[atom3].insert(atom2);
        (*connectivity)[atom3].insert(atomv);
    }
    for (int i = 0; i < virtual_layer_info[0].v3_info.virtual_numbers; i++)
    {
        int atomv =
            virtual_layer_info[0].v3_info.h_virtual_type_3[i].virtual_atom;
        int atom1 = virtual_layer_info[0].v3_info.h_virtual_type_3[i].from_1;
        int atom2 = virtual_layer_info[0].v3_info.h_virtual_type_3[i].from_2;
        int atom3 = virtual_layer_info[0].v3_info.h_virtual_type_3[i].from_3;
        (*connectivity)[atomv].insert(atom1);
        (*connectivity)[atomv].insert(atom2);
        (*connectivity)[atomv].insert(atom3);
        (*connectivity)[atom1].insert(atomv);
        (*connectivity)[atom1].insert(atom2);
        (*connectivity)[atom1].insert(atom3);
        (*connectivity)[atom2].insert(atom1);
        (*connectivity)[atom2].insert(atomv);
        (*connectivity)[atom2].insert(atom3);
        (*connectivity)[atom3].insert(atom1);
        (*connectivity)[atom3].insert(atom2);
        (*connectivity)[atom3].insert(atomv);
    }
}

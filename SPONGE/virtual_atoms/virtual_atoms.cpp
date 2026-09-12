#include "virtual_atoms.h"

#include "../xponge/load/native/virtual_atoms.hpp"
#include "../xponge/xponge.h"

static __global__ void v0_Coordinate_Refresh(const int virtual_numbers,
                                             const VIRTUAL_TYPE_0* v_info,
                                             VECTOR* crd, Boundary boundary)
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
        crd[atom_v] = Virtual_Atom_Type_0_Position(crd[atom_1], v_temp.h);
    }
}

static __global__ void v1_Coordinate_Refresh(const int virtual_numbers,
                                             const VIRTUAL_TYPE_1* v_info,
                                             VECTOR* crd, Boundary boundary)
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
        crd[atom_v] =
            Virtual_Atom_Type_1_Position(crd[atom_1], crd[atom_2], a, boundary);
    }
}

static __global__ void v2_Coordinate_Refresh(const int virtual_numbers,
                                             const VIRTUAL_TYPE_2* v_info,
                                             VECTOR* crd, Boundary boundary)
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

        crd[atom_v] = Virtual_Atom_Type_2_Position(crd[atom_1], crd[atom_2],
                                                   crd[atom_3], a, b, boundary);
    }
}

static __global__ void v3_Coordinate_Refresh(const int virtual_numbers,
                                             const VIRTUAL_TYPE_3* v_info,
                                             VECTOR* crd, Boundary boundary,
                                             int* geometry_error)
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
        VECTOR position;
        if (!Virtual_Atom_Type_3_Position(crd[atom_1], crd[atom_2], crd[atom_3],
                                          d, k, boundary, &position))
        {
            atomicExch(geometry_error, atom_v);
        }
        crd[atom_v] = position;
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
    for (int delta = warpSize >> 1; delta > 0; delta >>= 1)
    {
        new_position.x +=
            deviceShflDown(FULL_MASK, new_position.x, delta, warpSize);
        new_position.y +=
            deviceShflDown(FULL_MASK, new_position.y, delta, warpSize);
        new_position.z +=
            deviceShflDown(FULL_MASK, new_position.z, delta, warpSize);
    }
    if (threadIdx.x == 0)
    {
        coordinate[virtual_atom] = new_position;
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
    coordinate[virtual_atom] = new_position;
#endif
}

static __global__ void v0_Force_Redistribute(const int virtual_numbers,
                                             const VIRTUAL_TYPE_0* v_info,
                                             const VECTOR* crd,
                                             Boundary boundary, VECTOR* force)
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
        Virtual_Atom_Add_Source_Force(
            &force[atom_1], Virtual_Atom_Type_0_Source_Force(force_v));
        force_v.x = 0.0f;
        force_v.y = 0.0f;
        force_v.z = 0.0f;
        force[atom_v] = force_v;
    }
}

static __global__ void v1_Force_Redistribute(const int virtual_numbers,
                                             const VIRTUAL_TYPE_1* v_info,
                                             const VECTOR* crd,
                                             Boundary boundary, VECTOR* force)
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
        VECTOR force_1, force_2;
        Virtual_Atom_Type_1_Source_Forces(force_v, a, &force_1, &force_2);
        Virtual_Atom_Add_Source_Force(&force[atom_1], force_1);
        Virtual_Atom_Add_Source_Force(&force[atom_2], force_2);

        force_v.x = 0.0f;
        force_v.y = 0.0f;
        force_v.z = 0.0f;
        force[atom_v] = force_v;
    }
}

static __global__ void v2_Force_Redistribute(const int virtual_numbers,
                                             const VIRTUAL_TYPE_2* v_info,
                                             const VECTOR* crd,
                                             Boundary boundary, VECTOR* force)
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
        VECTOR force_1, force_2, force_3;
        Virtual_Atom_Type_2_Source_Forces(force_v, a, b, &force_1, &force_2,
                                          &force_3);
        Virtual_Atom_Add_Source_Force(&force[atom_1], force_1);
        Virtual_Atom_Add_Source_Force(&force[atom_2], force_2);
        Virtual_Atom_Add_Source_Force(&force[atom_3], force_3);

        force_v.x = 0.0f;
        force_v.y = 0.0f;
        force_v.z = 0.0f;
        force[atom_v] = force_v;
    }
}

static __global__ void v2_Force_Redistribute_No_Atomic(
    const int virtual_numbers, const VIRTUAL_TYPE_2* v_info, const VECTOR* crd,
    Boundary boundary, VECTOR* force)
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
        const VECTOR force_v = force[v_temp.virtual_atom];
        VECTOR force_1, force_2, force_3;
        Virtual_Atom_Type_2_Source_Forces(force_v, v_temp.a, v_temp.b, &force_1,
                                          &force_2, &force_3);
        force[v_temp.from_1] = force[v_temp.from_1] + force_1;
        force[v_temp.from_2] = force[v_temp.from_2] + force_2;
        force[v_temp.from_3] = force[v_temp.from_3] + force_3;
        force[v_temp.virtual_atom] = VECTOR(0.0f);
    }
}

static __global__ void v3_Force_Redistribute(const int virtual_numbers,
                                             const VIRTUAL_TYPE_3* v_info,
                                             const VECTOR* crd,
                                             Boundary boundary, VECTOR* force,
                                             int* geometry_error)
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
        VECTOR force_1, force_2, force_3;
        if (!Virtual_Atom_Type_3_Source_Forces(r1, r2, r3, force_v, d, k,
                                               boundary, &force_1, &force_2,
                                               &force_3))
        {
            atomicExch(geometry_error, atom_v);
        }

        Virtual_Atom_Add_Source_Force(&force[atom_1], force_1);
        Virtual_Atom_Add_Source_Force(&force[atom_2], force_2);
        Virtual_Atom_Add_Source_Force(&force[atom_3], force_3);

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
#pragma omp parallel for firstprivate(new_force) private(this_weight, this_frc)
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
    this->controller = controller;
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

        // 先统一校验力场虚原子定义并由依赖图计算层级。输入顺序不影响结果。
        controller->printf("    Start reading virtual levels\n");
        if (has_in_file)
        {
            VirtualAtomGraph graph;
            std::string graph_error;
            if (!Build_Virtual_Atom_Graph(*records_to_use, atom_numbers, &graph,
                                          &graph_error))
            {
                const std::string reason = "Reason:\n\t" + graph_error + "\n";
                controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                               "VIRTUAL_INFORMATION::Initial",
                                               reason.c_str());
            }
            for (int atom = 0; atom < atom_numbers; ++atom)
            {
                virtual_level[atom] = graph.atom_levels[atom];
            }
            for (const auto& record : *records_to_use)
            {
                for (const int source : record.from)
                {
                    (*connectivity)[record.virtual_atom].insert(source);
                    (*connectivity)[source].insert(record.virtual_atom);
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
                if (h_from[i] < 0 ||
                    h_from[i] >= atom_numbers + cv_vatom_name.size())
                {
                    char error_reason[CHAR_LENGTH_MAX];
                    sprintf(error_reason,
                            "Reason:\n\tError: atom id (%d) is outside [0, "
                            "atom_numbers + cv_virtual_atom_numbers (%d))\n",
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
                        has_type_3 = true;
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
                            .h = record.parameter[0];
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
        // Type-2 sites are common in water models. Preserve the non-atomic
        // fast path only when no source atom is shared by two records in the
        // same layer. Localization can remove records, but cannot introduce a
        // new overlap, so this global per-layer decision remains safe.
        for (VIRTUAL_LAYER_INFORMATION& layer_info : virtual_layer_info)
        {
            VIRTUAL_TYPE_2_INFROMATION& v2_info = layer_info.v2_info;
            std::vector<unsigned char> source_seen(atom_numbers, 0);
            for (int i = 0; i < v2_info.virtual_numbers; ++i)
            {
                const VIRTUAL_TYPE_2& record = v2_info.h_virtual_type_2[i];
                const int sources[] = {record.from_1, record.from_2,
                                       record.from_3};
                for (const int source : sources)
                {
                    if (source_seen[source]) v2_info.need_atomic = true;
                    source_seen[source] = 1;
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
                    temp_vl->v3_info.h_virtual_type_3,
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

        Device_Malloc_Safely((void**)&d_runtime_error, sizeof(int));
        if (has_type_3) Reset_Runtime_Error();

        is_initialized = 1;
        if (is_initialized && !is_controller_printf_initialized)
        {
            is_controller_printf_initialized = 1;
            controller->printf("    structure last modify date is %d\n",
                               last_modify_date);
        }

        controller->printf("END INITIALIZING VIRTUAL ATOM\n\n");
    }
    else
    {
        controller->printf("VIRTUAL ATOM IS NOT INITIALIZED\n\n");
    }
}

void VIRTUAL_INFORMATION::Coordinate_Refresh(VECTOR* crd, Boundary boundary)
{
    if (is_initialized)
    {
        // 每层之间需要串行计算，层内并行计算
        for (int layer = 0; layer < max_level; layer++)
        {
            VIRTUAL_LAYER_INFORMATION* temp_vl = &virtual_layer_info[layer];
            const int v0_numbers = local_state_ready
                                       ? temp_vl->v0_info.local_numbers
                                       : temp_vl->v0_info.virtual_numbers;
            const VIRTUAL_TYPE_0* v0_info =
                local_state_ready ? temp_vl->v0_info.l_virtual_type_0
                                  : temp_vl->v0_info.d_virtual_type_0;
            if (v0_numbers > 0)
                Launch_Device_Kernel(
                    v0_Coordinate_Refresh,
                    (v0_numbers + CONTROLLER::device_max_thread - 1) /
                        CONTROLLER::device_max_thread,
                    CONTROLLER::device_max_thread, 0, NULL, v0_numbers, v0_info,
                    crd, boundary);

            const int v1_numbers = local_state_ready
                                       ? temp_vl->v1_info.local_numbers
                                       : temp_vl->v1_info.virtual_numbers;
            const VIRTUAL_TYPE_1* v1_info =
                local_state_ready ? temp_vl->v1_info.l_virtual_type_1
                                  : temp_vl->v1_info.d_virtual_type_1;
            if (v1_numbers > 0)
                Launch_Device_Kernel(
                    v1_Coordinate_Refresh,
                    (v1_numbers + CONTROLLER::device_max_thread - 1) /
                        CONTROLLER::device_max_thread,
                    CONTROLLER::device_max_thread, 0, NULL, v1_numbers, v1_info,
                    crd, boundary);

            const int v2_numbers = local_state_ready
                                       ? temp_vl->v2_info.local_numbers
                                       : temp_vl->v2_info.virtual_numbers;
            const VIRTUAL_TYPE_2* v2_info =
                local_state_ready ? temp_vl->v2_info.l_virtual_type_2
                                  : temp_vl->v2_info.d_virtual_type_2;
            if (v2_numbers > 0)
                Launch_Device_Kernel(
                    v2_Coordinate_Refresh,
                    (v2_numbers + CONTROLLER::device_max_thread - 1) /
                        CONTROLLER::device_max_thread,
                    CONTROLLER::device_max_thread, 0, NULL, v2_numbers, v2_info,
                    crd, boundary);

            const int v3_numbers = local_state_ready
                                       ? temp_vl->v3_info.local_numbers
                                       : temp_vl->v3_info.virtual_numbers;
            const VIRTUAL_TYPE_3* v3_info =
                local_state_ready ? temp_vl->v3_info.l_virtual_type_3
                                  : temp_vl->v3_info.d_virtual_type_3;
            if (v3_numbers > 0)
                Launch_Device_Kernel(
                    v3_Coordinate_Refresh,
                    (v3_numbers + CONTROLLER::device_max_thread - 1) /
                        CONTROLLER::device_max_thread,
                    CONTROLLER::device_max_thread, 0, NULL, v3_numbers, v3_info,
                    crd, boundary, d_runtime_error);
        }
    }
}

void VIRTUAL_INFORMATION::Force_Redistribute(const VECTOR* crd,
                                             Boundary boundary, VECTOR* frc)
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
                                     boundary, frc);
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
                                     boundary, frc);
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
                                     boundary, frc, d_runtime_error);
            }

            if (temp_vl->v2_info.local_numbers > 0)
            {
                const int blocks = (temp_vl->v2_info.local_numbers +
                                    CONTROLLER::device_max_thread - 1) /
                                   CONTROLLER::device_max_thread;
                if (temp_vl->v2_info.need_atomic)
                {
                    Launch_Device_Kernel(v2_Force_Redistribute, blocks,
                                         CONTROLLER::device_max_thread, 0, NULL,
                                         temp_vl->v2_info.local_numbers,
                                         temp_vl->v2_info.l_virtual_type_2, crd,
                                         boundary, frc);
                }
                else
                {
                    Launch_Device_Kernel(
                        v2_Force_Redistribute_No_Atomic, blocks,
                        CONTROLLER::device_max_thread, 0, NULL,
                        temp_vl->v2_info.local_numbers,
                        temp_vl->v2_info.l_virtual_type_2, crd, boundary, frc);
                }
            }
        }
    }
}

void VIRTUAL_INFORMATION::Reset_Runtime_Error()
{
    if (d_runtime_error == NULL) return;
    const int no_error = -1;
    deviceMemcpy(d_runtime_error, &no_error, sizeof(int),
                 deviceMemcpyHostToDevice);
}

void VIRTUAL_INFORMATION::Throw_If_Runtime_Error(const char* operation)
{
    if (d_runtime_error == NULL || controller == NULL) return;
    int atom = -1;
    deviceMemcpy(&atom, d_runtime_error, sizeof(int), deviceMemcpyDeviceToHost);
    if (atom >= 0)
    {
        char reason[CHAR_LENGTH_MAX];
        snprintf(reason, sizeof(reason),
                 "Reason:\n\tvirtual atom %d has a degenerate type-3 "
                 "geometry during %s\n",
                 atom, operation);
        controller->Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                                       "VIRTUAL_INFORMATION", reason);
    }
}

void VIRTUAL_INFORMATION::Coordinate_Refresh_CV(VECTOR* crd, Boundary boundary)
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
                                                Boundary boundary, VECTOR* frc)
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

static __global__ void get_local_device_V0(
    int virtual_numbers, int* local_numbers, VIRTUAL_TYPE_0* d_virtual_type_0,
    VIRTUAL_TYPE_0* l_virtual_type_0, const int* atom_local_id,
    const char* atom_local_label, const int local_atom_numbers,
    int* localization_error)
{
    local_numbers[0] = 0;
    for (int cluster = 0; cluster < virtual_numbers; cluster++)
    {
        int vatom = d_virtual_type_0[cluster].virtual_atom;
        int from1 = d_virtual_type_0[cluster].from_1;
        if (atom_local_label[vatom])
        {
            if (atom_local_id[vatom] < 0 ||
                atom_local_id[vatom] >= local_atom_numbers ||
                atom_local_id[from1] < 0 ||
                atom_local_id[from1] >= local_atom_numbers)
            {
                atomicExch(localization_error, vatom);
                continue;
            }
            l_virtual_type_0[local_numbers[0]] = d_virtual_type_0[cluster];
            l_virtual_type_0[local_numbers[0]].virtual_atom =
                atom_local_id[vatom];
            l_virtual_type_0[local_numbers[0]].from_1 = atom_local_id[from1];
            local_numbers[0] += 1;
        }
    }
}

static __global__ void get_local_device_V1(
    int virtual_numbers, int* local_numbers, VIRTUAL_TYPE_1* d_virtual_type_1,
    VIRTUAL_TYPE_1* l_virtual_type_1, const int* atom_local_id,
    const char* atom_local_label, const int local_atom_numbers,
    int* localization_error)
{
    local_numbers[0] = 0;
    for (int cluster = 0; cluster < virtual_numbers; cluster++)
    {
        int vatom = d_virtual_type_1[cluster].virtual_atom;
        int from1 = d_virtual_type_1[cluster].from_1;
        int from2 = d_virtual_type_1[cluster].from_2;
        if (atom_local_label[vatom])
        {
            if (atom_local_id[vatom] < 0 ||
                atom_local_id[vatom] >= local_atom_numbers ||
                atom_local_id[from1] < 0 ||
                atom_local_id[from1] >= local_atom_numbers ||
                atom_local_id[from2] < 0 ||
                atom_local_id[from2] >= local_atom_numbers)
            {
                atomicExch(localization_error, vatom);
                continue;
            }
            l_virtual_type_1[local_numbers[0]] = d_virtual_type_1[cluster];
            l_virtual_type_1[local_numbers[0]].virtual_atom =
                atom_local_id[vatom];
            l_virtual_type_1[local_numbers[0]].from_1 = atom_local_id[from1];
            l_virtual_type_1[local_numbers[0]].from_2 = atom_local_id[from2];
            local_numbers[0] += 1;
        }
    }
}

static __global__ void get_local_device_V2(
    int virtual_numbers, int* local_numbers, VIRTUAL_TYPE_2* d_virtual_type_2,
    VIRTUAL_TYPE_2* l_virtual_type_2, const int* atom_local_id,
    const char* atom_local_label, const int local_atom_numbers,
    int* localization_error)
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
            if (atom_local_id[vatom] < 0 ||
                atom_local_id[vatom] >= local_atom_numbers ||
                atom_local_id[from1] < 0 ||
                atom_local_id[from1] >= local_atom_numbers ||
                atom_local_id[from2] < 0 ||
                atom_local_id[from2] >= local_atom_numbers ||
                atom_local_id[from3] < 0 ||
                atom_local_id[from3] >= local_atom_numbers)
            {
                atomicExch(localization_error, vatom);
                continue;
            }
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

static __global__ void get_local_device_V3(
    int virtual_numbers, int* local_numbers, VIRTUAL_TYPE_3* d_virtual_type_3,
    VIRTUAL_TYPE_3* l_virtual_type_3, const int* atom_local_id,
    const char* atom_local_label, const int local_atom_numbers,
    int* localization_error)
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
            if (atom_local_id[vatom] < 0 ||
                atom_local_id[vatom] >= local_atom_numbers ||
                atom_local_id[from1] < 0 ||
                atom_local_id[from1] >= local_atom_numbers ||
                atom_local_id[from2] < 0 ||
                atom_local_id[from2] >= local_atom_numbers ||
                atom_local_id[from3] < 0 ||
                atom_local_id[from3] >= local_atom_numbers)
            {
                atomicExch(localization_error, vatom);
                continue;
            }
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
    if (has_type_3)
        Throw_If_Runtime_Error("coordinate refresh or force redistribution");
    Reset_Runtime_Error();
    local_state_ready = true;
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
                                 atom_local_id, atom_local_label,
                                 local_atom_numbers, d_runtime_error);
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
                                 atom_local_id, atom_local_label,
                                 local_atom_numbers, d_runtime_error);
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
                                 atom_local_id, atom_local_label,
                                 local_atom_numbers, d_runtime_error);
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
                                 atom_local_id, atom_local_label,
                                 local_atom_numbers, d_runtime_error);
            deviceMemcpy(&temp_vl->v3_info.local_numbers,
                         temp_vl->v3_info.d_local_numbers, sizeof(int),
                         deviceMemcpyDeviceToHost);
        }

        // 预留v4质心接口
    }
    int localization_error = -1;
    deviceMemcpy(&localization_error, d_runtime_error, sizeof(int),
                 deviceMemcpyDeviceToHost);
    if (localization_error >= 0 && controller != NULL)
    {
        char reason[CHAR_LENGTH_MAX];
        snprintf(reason, sizeof(reason),
                 "Reason:\n\tvirtual atom %d and all of its sources must "
                 "belong to the same local update group\n",
                 localization_error);
        controller->Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                                       "VIRTUAL_INFORMATION::Get_Local",
                                       reason);
    }
}

void VIRTUAL_INFORMATION::update_ug_connectivity(CONECT* connectivity)
{
    if (!is_initialized) return;
    auto connect = [connectivity](int atomv, std::initializer_list<int> sources)
    {
        for (int source : sources)
        {
            (*connectivity)[atomv].insert(source);
            (*connectivity)[source].insert(atomv);
        }
    };
    for (int layer = 0; layer < max_level; ++layer)
    {
        VIRTUAL_LAYER_INFORMATION& info = virtual_layer_info[layer];
        for (int i = 0; i < info.v0_info.virtual_numbers; ++i)
        {
            const auto& v = info.v0_info.h_virtual_type_0[i];
            connect(v.virtual_atom, {v.from_1});
        }
        for (int i = 0; i < info.v1_info.virtual_numbers; ++i)
        {
            const auto& v = info.v1_info.h_virtual_type_1[i];
            connect(v.virtual_atom, {v.from_1, v.from_2});
        }
        for (int i = 0; i < info.v2_info.virtual_numbers; ++i)
        {
            const auto& v = info.v2_info.h_virtual_type_2[i];
            connect(v.virtual_atom, {v.from_1, v.from_2, v.from_3});
        }
        for (int i = 0; i < info.v3_info.virtual_numbers; ++i)
        {
            const auto& v = info.v3_info.h_virtual_type_3[i];
            connect(v.virtual_atom, {v.from_1, v.from_2, v.from_3});
        }
    }
}

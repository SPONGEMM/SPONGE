#include "restrain.h"

#include "../xponge/ir/protocol.h"

void RESTRAIN_INFORMATION::Init_Com_Cache_If_Needed(
    const int atom_numbers, const MD_INFORMATION& md_info)
{
    if (this->com_cache_initialized &&
        this->cached_atom_numbers == atom_numbers)
        return;

    this->cached_atom_numbers = atom_numbers;
    this->com_cache_initialized = 1;

    if (md_info.ug.ug_numbers > 0 && md_info.ug.ug != NULL)
    {
        this->cached_ug_numbers = md_info.ug.ug_numbers;
        int* h_atom_to_ug = NULL;
        Malloc_Safely((void**)&h_atom_to_ug, sizeof(int) * atom_numbers);
        for (int i = 0; i < atom_numbers; ++i) h_atom_to_ug[i] = 0;
        for (int ug_i = 0; ug_i < md_info.ug.ug_numbers; ++ug_i)
        {
            for (int idx = 0; idx < md_info.ug.ug[ug_i].atom_numbers; ++idx)
            {
                int atom = md_info.ug.ug[ug_i].atom_serial[idx];
                h_atom_to_ug[atom] = ug_i;
            }
        }
        Device_Malloc_Safely((void**)&this->d_atom_to_ug,
                             sizeof(int) * atom_numbers);
        deviceMemcpy(this->d_atom_to_ug, h_atom_to_ug,
                     sizeof(int) * atom_numbers, deviceMemcpyHostToDevice);
        free(h_atom_to_ug);

        Device_Malloc_Safely((void**)&this->d_sum_mass_ug,
                             sizeof(float) * this->cached_ug_numbers);
        Device_Malloc_Safely((void**)&this->d_sum_pos_ug,
                             sizeof(VECTOR) * this->cached_ug_numbers);
        Device_Malloc_Safely((void**)&this->d_com_ug,
                             sizeof(VECTOR) * this->cached_ug_numbers);
        Malloc_Safely((void**)&this->h_sum_mass_ug,
                      sizeof(float) * this->cached_ug_numbers);
        Malloc_Safely((void**)&this->h_sum_pos_ug,
                      sizeof(VECTOR) * this->cached_ug_numbers);
        Malloc_Safely((void**)&this->h_com_ug,
                      sizeof(VECTOR) * this->cached_ug_numbers);
    }

    if (md_info.res.is_initialized && md_info.res.residue_numbers > 0 &&
        md_info.res.h_res_start != NULL && md_info.res.h_res_end != NULL)
    {
        this->cached_res_numbers = md_info.res.residue_numbers;
        int* h_atom_to_res = NULL;
        Malloc_Safely((void**)&h_atom_to_res, sizeof(int) * atom_numbers);
        for (int i = 0; i < atom_numbers; ++i) h_atom_to_res[i] = 0;
        for (int res_i = 0; res_i < md_info.res.residue_numbers; ++res_i)
        {
            for (int atom = md_info.res.h_res_start[res_i];
                 atom < md_info.res.h_res_end[res_i]; ++atom)
            {
                h_atom_to_res[atom] = res_i;
            }
        }
        Device_Malloc_Safely((void**)&this->d_atom_to_res,
                             sizeof(int) * atom_numbers);
        deviceMemcpy(this->d_atom_to_res, h_atom_to_res,
                     sizeof(int) * atom_numbers, deviceMemcpyHostToDevice);
        free(h_atom_to_res);

        Device_Malloc_Safely((void**)&this->d_sum_mass_res,
                             sizeof(float) * this->cached_res_numbers);
        Device_Malloc_Safely((void**)&this->d_sum_pos_res,
                             sizeof(VECTOR) * this->cached_res_numbers);
        Device_Malloc_Safely((void**)&this->d_com_res,
                             sizeof(VECTOR) * this->cached_res_numbers);
        Malloc_Safely((void**)&this->h_sum_mass_res,
                      sizeof(float) * this->cached_res_numbers);
        Malloc_Safely((void**)&this->h_sum_pos_res,
                      sizeof(VECTOR) * this->cached_res_numbers);
        Malloc_Safely((void**)&this->h_com_res,
                      sizeof(VECTOR) * this->cached_res_numbers);
    }

    if (md_info.mol.is_initialized && md_info.mol.molecule_numbers > 0 &&
        md_info.mol.h_atom_start != NULL && md_info.mol.h_atom_end != NULL)
    {
        this->cached_mol_numbers = md_info.mol.molecule_numbers;
        int* h_atom_to_mol = NULL;
        Malloc_Safely((void**)&h_atom_to_mol, sizeof(int) * atom_numbers);
        for (int i = 0; i < atom_numbers; ++i) h_atom_to_mol[i] = 0;
        for (int mol_i = 0; mol_i < md_info.mol.molecule_numbers; ++mol_i)
        {
            for (int atom = md_info.mol.h_atom_start[mol_i];
                 atom < md_info.mol.h_atom_end[mol_i]; ++atom)
            {
                h_atom_to_mol[atom] = mol_i;
            }
        }
        Device_Malloc_Safely((void**)&this->d_atom_to_mol,
                             sizeof(int) * atom_numbers);
        deviceMemcpy(this->d_atom_to_mol, h_atom_to_mol,
                     sizeof(int) * atom_numbers, deviceMemcpyHostToDevice);
        free(h_atom_to_mol);

        Device_Malloc_Safely((void**)&this->d_sum_mass_mol,
                             sizeof(float) * this->cached_mol_numbers);
        Device_Malloc_Safely((void**)&this->d_sum_pos_mol,
                             sizeof(VECTOR) * this->cached_mol_numbers);
        Device_Malloc_Safely((void**)&this->d_com_mol,
                             sizeof(VECTOR) * this->cached_mol_numbers);
        Malloc_Safely((void**)&this->h_sum_mass_mol,
                      sizeof(float) * this->cached_mol_numbers);
        Malloc_Safely((void**)&this->h_sum_pos_mol,
                      sizeof(VECTOR) * this->cached_mol_numbers);
        Malloc_Safely((void**)&this->h_com_mol,
                      sizeof(VECTOR) * this->cached_mol_numbers);
    }
}

static __global__ void Accumulate_Group_COM_Sum(
    const int atom_numbers, const VECTOR* crd, const int* atom_local,
    const int* atom_to_group, const float* mass, float* sum_mass,
    VECTOR* sum_pos)
{
    SIMPLE_DEVICE_FOR(i, atom_numbers)
    {
        int atom_global = atom_local[i];
        int group = atom_to_group[atom_global];
        float m = mass[atom_global];
        float mx = m * crd[i].x;
        float my = m * crd[i].y;
        float mz = m * crd[i].z;
#ifdef USE_GPU
        atomicAdd(sum_mass + group, m);
        atomicAdd(&sum_pos[group].x, mx);
        atomicAdd(&sum_pos[group].y, my);
        atomicAdd(&sum_pos[group].z, mz);
#else
#pragma omp atomic
        sum_mass[group] += m;
#pragma omp atomic
        sum_pos[group].x += mx;
#pragma omp atomic
        sum_pos[group].y += my;
#pragma omp atomic
        sum_pos[group].z += mz;
#endif
    }
}

void RESTRAIN_INFORMATION::Update_Group_COM(
    const int local_atom_numbers, const VECTOR* crd, const int* atom_local,
    const int* atom_to_group, const float* mass, const int group_numbers,
    float* d_sum_mass, VECTOR* d_sum_pos, VECTOR* d_com, float* h_sum_mass,
    VECTOR* h_sum_pos, VECTOR* h_com)
{
    deviceMemset(d_sum_mass, 0, sizeof(float) * group_numbers);
    deviceMemset(d_sum_pos, 0, sizeof(VECTOR) * group_numbers);
    Launch_Device_Kernel(
        Accumulate_Group_COM_Sum,
        (local_atom_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, local_atom_numbers, crd,
        atom_local, atom_to_group, mass, d_sum_mass, d_sum_pos);

    deviceMemcpy(h_sum_mass, d_sum_mass, sizeof(float) * group_numbers,
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(h_sum_pos, d_sum_pos, sizeof(VECTOR) * group_numbers,
                 deviceMemcpyDeviceToHost);
#ifdef USE_MPI
    if (CONTROLLER::PP_MPI_size > 1)
    {
        MPI_Allreduce(MPI_IN_PLACE, h_sum_mass, group_numbers, MPI_FLOAT,
                      MPI_SUM, CONTROLLER::pp_comm);
        MPI_Allreduce(MPI_IN_PLACE, h_sum_pos, group_numbers * 3, MPI_FLOAT,
                      MPI_SUM, CONTROLLER::pp_comm);
    }
#endif
    for (int i = 0; i < group_numbers; ++i)
    {
        if (h_sum_mass[i] > 0.0f)
        {
            h_com[i].x = h_sum_pos[i].x / h_sum_mass[i];
            h_com[i].y = h_sum_pos[i].y / h_sum_mass[i];
            h_com[i].z = h_sum_pos[i].z / h_sum_mass[i];
        }
        else
        {
            h_com[i].x = 0.0f;
            h_com[i].y = 0.0f;
            h_com[i].z = 0.0f;
        }
    }
    deviceMemcpy(d_com, h_com, sizeof(VECTOR) * group_numbers,
                 deviceMemcpyHostToDevice);
}

static __global__ void restrain_force_with_atom_energy_and_virial(
    const int restrain_numbers, const int* restrain_list, const VECTOR* crd,
    const VECTOR* crd_ref, const int if_single_weight,
    const float single_weight, const VECTOR* weight_list, Boundary boundary,
    int need_atom_energy, float* atom_energy, int need_virial,
    LTMatrix3* atom_virial, VECTOR* frc, float* res_ene,
    const int refcoord_scaling, const int* atom_local, const int* atom_to_group,
    const VECTOR* group_com)
{
#ifdef USE_GPU
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < restrain_numbers)
#else
#pragma omp parallel for
    for (int i = 0; i < restrain_numbers; i++)
#endif
    {
        int atom_i = restrain_list[i];
        VECTOR r_i = crd[atom_i];
        VECTOR dr = Get_Displacement(crd_ref[i], r_i, boundary);
        VECTOR temp_force;
        if (if_single_weight)
        {
            temp_force = single_weight * dr;
        }
        else
        {
            temp_force = wiseproduct(weight_list[i], dr);
        }
        if (need_atom_energy)
        {
            atom_energy[atom_i] += 0.5 * temp_force * dr;
            res_ene[i] = 0.5 * temp_force * dr;
        }
        if (need_virial)
        {
            if (refcoord_scaling == RESTRAIN_INFORMATION::REFCOORD_SCALING_ALL)
            {
                atom_virial[atom_i] = atom_virial[atom_i] -
                                      Get_Virial_From_Force_Dis(temp_force, dr);
            }
            else if (refcoord_scaling ==
                         RESTRAIN_INFORMATION::REFCOORD_SCALING_COM_UG ||
                     refcoord_scaling ==
                         RESTRAIN_INFORMATION::REFCOORD_SCALING_COM_RES ||
                     refcoord_scaling ==
                         RESTRAIN_INFORMATION::REFCOORD_SCALING_COM_MOL)
            {
                int atom_global = atom_local[atom_i];
                int group = atom_to_group[atom_global];
                VECTOR com = group_com[group];
                VECTOR dr_com = Get_Displacement(com, r_i, boundary);
                atom_virial[atom_i] =
                    atom_virial[atom_i] -
                    Get_Virial_From_Force_Dis(temp_force, dr_com);
            }
            else
            {
                atom_virial[atom_i] =
                    atom_virial[atom_i] -
                    Get_Virial_From_Force_Dis(temp_force, r_i);
            }
        }
        frc[atom_i] = frc[atom_i] + temp_force;
    }
}

static __global__ void Gather_Ref_From_All_Device(const int restrain_numbers,
                                                  const int* d_lists,
                                                  const VECTOR* ref_all,
                                                  VECTOR* ref_list)
{
    SIMPLE_DEVICE_FOR(i, restrain_numbers)
    {
        ref_list[i] = ref_all[d_lists[i]];
    }
}

static __global__ void Rescale_Ref_All_Device(const int atom_numbers,
                                              const LTMatrix3 g, const float dt,
                                              VECTOR* ref_all)
{
    SIMPLE_DEVICE_FOR(atom_i, atom_numbers)
    {
        VECTOR r = ref_all[atom_i], r_dash;
        r_dash.x = r.x + dt * (r.x * g.a11 + r.y * g.a21 + r.z * g.a31);
        r_dash.y = r.y + dt * (r.y * g.a22 + r.z * g.a32);
        r_dash.z = r.z + dt * r.z * g.a33;
        ref_all[atom_i] = r_dash;
    }
}

static __global__ void Rescale_Ref_By_Group_Range_Device(
    const int group_numbers, const int* group_start, const int* group_end,
    const VECTOR* ref_all, const float* mass, const LTMatrix3 g, const float dt,
    VECTOR* ref_all_out)
{
#ifdef USE_GPU
    int gidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (gidx < group_numbers)
#else
#pragma omp parallel for
    for (int gidx = 0; gidx < group_numbers; gidx++)
#endif
    {
        double msum = 0.0;
        VECTOR com = {0.0f, 0.0f, 0.0f};
        for (int atom = group_start[gidx]; atom < group_end[gidx]; atom++)
        {
            double m = mass[atom];
            com.x += ref_all[atom].x * m;
            com.y += ref_all[atom].y * m;
            com.z += ref_all[atom].z * m;
            msum += m;
        }
        if (msum > 0.0)
        {
            com.x /= msum;
            com.y /= msum;
            com.z /= msum;
        }
        VECTOR com_dash;
        com_dash.x =
            com.x + dt * (com.x * g.a11 + com.y * g.a21 + com.z * g.a31);
        com_dash.y = com.y + dt * (com.y * g.a22 + com.z * g.a32);
        com_dash.z = com.z + dt * com.z * g.a33;
        VECTOR delta = {com_dash.x - com.x, com_dash.y - com.y,
                        com_dash.z - com.z};
        for (int atom = group_start[gidx]; atom < group_end[gidx]; atom++)
        {
            ref_all_out[atom].x += delta.x;
            ref_all_out[atom].y += delta.y;
            ref_all_out[atom].z += delta.z;
        }
    }
}

static __global__ void Rescale_Ref_By_UG_Device(
    const int ug_numbers, const ATOM_GROUP* ug, const VECTOR* ref_all,
    const float* mass, const LTMatrix3 g, const float dt, VECTOR* ref_all_out)
{
#ifdef USE_GPU
    int gidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (gidx < ug_numbers)
#else
#pragma omp parallel for
    for (int gidx = 0; gidx < ug_numbers; gidx++)
#endif
    {
        double msum = 0.0;
        VECTOR com = {0.0f, 0.0f, 0.0f};
        for (int idx = 0; idx < ug[gidx].atom_numbers; idx++)
        {
            int atom = ug[gidx].atom_serial[idx];
            double m = mass[atom];
            com.x += ref_all[atom].x * m;
            com.y += ref_all[atom].y * m;
            com.z += ref_all[atom].z * m;
            msum += m;
        }
        if (msum > 0.0)
        {
            com.x /= msum;
            com.y /= msum;
            com.z /= msum;
        }
        VECTOR com_dash;
        com_dash.x =
            com.x + dt * (com.x * g.a11 + com.y * g.a21 + com.z * g.a31);
        com_dash.y = com.y + dt * (com.y * g.a22 + com.z * g.a32);
        com_dash.z = com.z + dt * com.z * g.a33;
        VECTOR delta = {com_dash.x - com.x, com_dash.y - com.y,
                        com_dash.z - com.z};
        for (int idx = 0; idx < ug[gidx].atom_numbers; idx++)
        {
            int atom = ug[gidx].atom_serial[idx];
            ref_all_out[atom].x += delta.x;
            ref_all_out[atom].y += delta.y;
            ref_all_out[atom].z += delta.z;
        }
    }
}

// 读取rst7
static void Import_Information_From_Rst7(const char* file_name,
                                         int* atom_numbers, float* sys_time,
                                         VECTOR** crd, VECTOR** vel,
                                         VECTOR* box_length,
                                         CONTROLLER controller)
{
    FILE* fin = NULL;
    Open_File_Safely(&fin, file_name, "r");
    controller.printf(
        "    Start reading restrain reference coordinate from AMBERFILE\n");
    char lin[CHAR_LENGTH_MAX];
    char* get_ret = fgets(lin, CHAR_LENGTH_MAX, fin);
    get_ret = fgets(lin, CHAR_LENGTH_MAX, fin);
    int has_vel = 0;
    int scanf_ret = sscanf(lin, "%d %f", &atom_numbers[0], &sys_time[0]);
    if (scanf_ret == 2)
    {
        has_vel = 1;
    }
    else
    {
        sys_time[0] = 0.;
    }
    controller.printf("        atom_numbers is %d\n", atom_numbers[0]);
    if (has_vel == 1)
    {
        controller.printf("        system_start_time is %f\n", sys_time[0]);
    }

    VECTOR *h_crd = NULL, *h_vel = NULL;
    Malloc_Safely((void**)&h_crd, sizeof(VECTOR) * atom_numbers[0]);
    Malloc_Safely((void**)&h_vel, sizeof(VECTOR) * atom_numbers[0]);

    Device_Malloc_Safely((void**)&crd[0], sizeof(VECTOR) * atom_numbers[0]);
    Device_Malloc_Safely((void**)&vel[0], sizeof(VECTOR) * atom_numbers[0]);
    for (int i = 0; i < atom_numbers[0]; i = i + 1)
    {
        scanf_ret =
            fscanf(fin, "%f %f %f", &h_crd[i].x, &h_crd[i].y, &h_crd[i].z);
    }
    if (has_vel == 1)
    {
        for (int i = 0; i < atom_numbers[0]; i = i + 1)
        {
            scanf_ret =
                fscanf(fin, "%f %f %f", &h_vel[i].x, &h_vel[i].y, &h_vel[i].z);
        }
    }
    else
    {
        for (int i = 0; i < atom_numbers[0]; i = i + 1)
        {
            h_vel[i].x = 0.0;
            h_vel[i].y = 0.0;
            h_vel[i].z = 0.0;
        }
    }
    scanf_ret =
        fscanf(fin, "%f %f %f", &box_length->x, &box_length->y, &box_length->z);
    controller.printf("        system size is %f %f %f\n", box_length->x,
                      box_length->y, box_length->z);
    deviceMemcpy(crd[0], h_crd, sizeof(VECTOR) * atom_numbers[0],
                 deviceMemcpyHostToDevice);
    deviceMemcpy(vel[0], h_vel, sizeof(VECTOR) * atom_numbers[0],
                 deviceMemcpyHostToDevice);
    controller.printf(
        "    End reading restrain reference coordinate from AMBERFILE\n");
    free(h_crd), free(h_vel);
    fclose(fin);
}

void RESTRAIN_INFORMATION::Initial(CONTROLLER* controller,
                                   const int atom_numbers, const VECTOR* crd,
                                   const char* module_name)
{
    if (module_name == NULL)
    {
        strcpy(this->module_name, "restrain");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }
    if (controller->Command_Exist("restrain_h5_object_name"))
    {
        h5_restraint_name = controller->Command("restrain_h5_object_name");
    }
    if (controller->Command_Exist(this->module_name, "atom_id"))
    {
        controller->printf("START INITIALIZING RESTRAIN:\n");

        this->atom_numbers = atom_numbers;

        // ---------------- initialize restrain reference crd information
        // ------------------------------
        // 用来临时储存最后不需要的内容
        int temp_atom;
        float ref_time;
        int* temp_atom_lists = NULL;
        Malloc_Safely((void**)&temp_atom_lists, sizeof(int) * atom_numbers);
        VECTOR *d_vel = NULL, h_boxlength;
        Device_Malloc_Safely((void**)&d_vel, sizeof(VECTOR) * atom_numbers);

        // 读参考原子id
        controller->printf("    reading %s_atom_id\n", this->module_name);
        this->restrain_numbers = 0;
        FILE* fr = NULL;
        Open_File_Safely(&fr, controller->Command(this->module_name, "atom_id"),
                         "r");

        while (fscanf(fr, "%d", &temp_atom) != EOF)
        {
            temp_atom_lists[this->restrain_numbers] = temp_atom;
            this->restrain_numbers++;
        }
        fclose(fr);
        controller->printf("        atom_number is %d\n", restrain_numbers);
        Malloc_Safely((void**)&this->h_lists,
                      sizeof(int) * this->restrain_numbers);
        deviceMemcpy(this->h_lists, temp_atom_lists,
                     sizeof(int) * this->restrain_numbers,
                     deviceMemcpyHostToHost);
        Device_Malloc_And_Copy_Safely((void**)&this->d_lists, this->h_lists,
                                      sizeof(int) * this->restrain_numbers);
        Device_Malloc_Safely((void**)&this->d_restrain_ene,
                             sizeof(float) * this->restrain_numbers);
        Device_Malloc_Safely((void**)&this->d_sum_of_restrain_ene,
                             sizeof(float));

        Malloc_Safely((void**)&this->h_sum_of_restrain_ene, sizeof(float));
        deviceMemset(this->d_restrain_ene, 0,
                     sizeof(float) * this->restrain_numbers);
        deviceMemset(this->d_sum_of_restrain_ene, 0, sizeof(float));
        memset(this->h_sum_of_restrain_ene, 0, sizeof(float));

        Device_Malloc_Safely((void**)&this->d_local_restrain_numbers,
                             sizeof(int));
        deviceMemset(this->d_local_restrain_numbers, 0, sizeof(int));
        Device_Malloc_Safely((void**)&this->d_local_restrain_list,
                             sizeof(int) * atom_numbers);
        Device_Malloc_Safely((void**)&local_crd_ref,
                             sizeof(VECTOR) * atom_numbers);

        refcoord_scaling = REFCOORD_SCALING_NO;
        if (controller->Command_Exist(this->module_name, "refcoord_scaling"))
        {
            const char* scaling =
                controller->Command(this->module_name, "refcoord_scaling");
            if (controller->Command_Choice(this->module_name,
                                           "refcoord_scaling", "no"))
            {
                refcoord_scaling = REFCOORD_SCALING_NO;
            }
            else if (controller->Command_Choice(this->module_name,
                                                "refcoord_scaling", "all"))
            {
                refcoord_scaling = REFCOORD_SCALING_ALL;
            }
            else if (controller->Command_Choice(this->module_name,
                                                "refcoord_scaling", "com_ug"))
            {
                refcoord_scaling = REFCOORD_SCALING_COM_UG;
            }
            else if (controller->Command_Choice(this->module_name,
                                                "refcoord_scaling", "com_res"))
            {
                refcoord_scaling = REFCOORD_SCALING_COM_RES;
            }
            else if (controller->Command_Choice(this->module_name,
                                                "refcoord_scaling", "com_mol"))
            {
                refcoord_scaling = REFCOORD_SCALING_COM_MOL;
            }
            else
            {
                controller->Throw_SPONGE_Error(
                    spongeErrorValueErrorCommand,
                    "RESTRAIN_INFORMATION::Initial",
                    "Reason:\n\trefcoord_scaling should be no/all/com_ug/"
                    "com_res/com_mol\n");
            }
            controller->printf("    %s_refcoord_scaling is %s\n",
                               this->module_name, scaling);
        }

        if (controller->Command_Exist(this->module_name, "calc_virial"))
        {
            calc_virial = controller->Get_Bool(this->module_name, "calc_virial",
                                               "RESTRAIN_INFORMATION::Initial");
            controller->printf("    %s_calc_virial is %d\n", this->module_name,
                               calc_virial ? 1 : 0);
        }

        if (controller->Command_Exist(this->module_name, "coordinate_in_file"))
        {
            controller->printf(
                "    reading restrain reference from %s\n",
                controller->Command(this->module_name, "coordinate_in_file"));
            VECTOR* h_crd = NULL;
            Malloc_Safely((void**)&h_crd, sizeof(VECTOR) * atom_numbers);
            FILE* fp = NULL;
            Open_File_Safely(
                &fp,
                controller->Command(this->module_name, "coordinate_in_file"),
                "r");
            int temp_atom_numbers = 0;
            int scanf_ret = fscanf(fp, "%d", &temp_atom_numbers);
            controller->printf("        atom_numbers is %d\n",
                               temp_atom_numbers);
            for (int i = 0; i < atom_numbers; i++)
            {
                scanf_ret = fscanf(fp, "%f %f %f", &h_crd[i].x, &h_crd[i].y,
                                   &h_crd[i].z);
            }
            Device_Malloc_Safely((void**)&d_ref_crd_all,
                                 sizeof(VECTOR) * atom_numbers);
            deviceMemcpy(d_ref_crd_all, h_crd, sizeof(VECTOR) * atom_numbers,
                         deviceMemcpyHostToDevice);
            free(h_crd);
            fclose(fp);
        }
        else if (controller->Command_Exist(this->module_name, "amber_rst7"))
        {
            controller->printf(
                "    reading restrain reference from %s\n",
                controller->Command(this->module_name, "amber_rst7"));
            Import_Information_From_Rst7(
                controller->Command(this->module_name, "amber_rst7"),
                &temp_atom, &ref_time, &d_ref_crd_all, &d_vel, &h_boxlength,
                controller[0]);
        }
        else
        {
            controller->printf(
                "    restrain reference coordinate copy from input "
                "coordinate\n");
            Device_Malloc_Safely((void**)&d_ref_crd_all,
                                 sizeof(VECTOR) * atom_numbers);
            deviceMemcpy(d_ref_crd_all, crd, sizeof(VECTOR) * atom_numbers,
                         deviceMemcpyDeviceToDevice);
        }

        Device_Malloc_Safely((void**)&crd_ref,
                             sizeof(VECTOR) * this->restrain_numbers);
        Launch_Device_Kernel(
            Gather_Ref_From_All_Device,
            (this->restrain_numbers + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, this->restrain_numbers,
            this->d_lists, this->d_ref_crd_all, this->crd_ref);

        // ---------------- initialize restrain weight information
        // -------------------------------------
        this->single_weight = 20.0f;
        // 单一各向同性约束力常数
        if (controller->Command_Exist(this->module_name, "single_weight"))
        {
            controller->Check_Float(this->module_name, "single_weight",
                                    "RESTRAIN_INFORMATION::Initial");
            this->single_weight =
                atof(controller->Command(this->module_name, "single_weight"));
            this->if_single_weight = 1;
            controller->printf("    %s_single_weight is %.0f\n",
                               this->module_name, this->single_weight);
        }
        else
        {
            controller->printf("    using isotropic restrain weight\n");
            Malloc_Safely((void**)&this->h_weights,
                          sizeof(VECTOR) * this->restrain_numbers);

            FILE* wr = NULL;
            Open_File_Safely(
                &wr, controller->Command(this->module_name, "weight_in_file"),
                "r");
            for (int i = 0; i < this->restrain_numbers; i++)
            {
                int scanf_ret =
                    fscanf(wr, "%f %f %f", &this->h_weights[i].x,
                           &this->h_weights[i].y, &this->h_weights[i].z);
            }
            Device_Malloc_And_Copy_Safely(
                (void**)&this->d_weights, this->h_weights,
                sizeof(VECTOR) * this->restrain_numbers);
            Device_Malloc_Safely((void**)&this->local_weights,
                                 sizeof(VECTOR) * atom_numbers);
            this->if_single_weight = 0;
        }

        Free_Single_Device_Pointer((void**)&d_vel);
        free(temp_atom_lists);
        is_initialized = 1;

        if (is_initialized && !is_controller_printf_initialized)
        {
            controller->Step_Print_Initial(this->module_name, "%.2f");
            is_controller_printf_initialized = 1;
            controller->printf("    structure last modify date is %d\n",
                               last_modify_date);
        }
        controller->printf("END INITIALIZING RESTRAIN\n\n");
    }
    else
    {
        controller->printf("RESTRAIN IS NOT INITIALIZED\n\n");
    }
}

void RESTRAIN_INFORMATION::Initial(
    CONTROLLER* controller, const int atom_numbers, const VECTOR* crd,
    const Xponge::PositionalRestraint& native_restraint,
    const char* module_name)
{
    if (!native_restraint.present)
    {
        Initial(controller, atom_numbers, crd, module_name);
        return;
    }
    if (module_name == NULL)
    {
        strcpy(this->module_name, "restrain");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }
    controller->printf("START INITIALIZING NATIVE H5 RESTRAIN:\n");
    this->atom_numbers = atom_numbers;
    this->h5_restraint_name = native_restraint.name;
    this->restrain_numbers =
        static_cast<int>(native_restraint.atom_indices.size());
    if (atom_numbers <= 0 || restrain_numbers <= 0)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorValueErrorCommand,
            "RESTRAIN_INFORMATION::Initial(native H5)",
            "Reason:\n\tnative positional restraint requires positive atom "
            "and restraint counts\n");
    }

    Malloc_Safely((void**)&this->h_lists, sizeof(int) * this->restrain_numbers);
    deviceMemcpy(this->h_lists, native_restraint.atom_indices.data(),
                 sizeof(int) * this->restrain_numbers, deviceMemcpyHostToHost);
    Device_Malloc_And_Copy_Safely((void**)&this->d_lists, this->h_lists,
                                  sizeof(int) * this->restrain_numbers);
    Device_Malloc_Safely((void**)&this->d_restrain_ene,
                         sizeof(float) * this->restrain_numbers);
    Device_Malloc_Safely((void**)&this->d_sum_of_restrain_ene, sizeof(float));
    Malloc_Safely((void**)&this->h_sum_of_restrain_ene, sizeof(float));
    deviceMemset(this->d_restrain_ene, 0,
                 sizeof(float) * this->restrain_numbers);
    deviceMemset(this->d_sum_of_restrain_ene, 0, sizeof(float));
    memset(this->h_sum_of_restrain_ene, 0, sizeof(float));
    Device_Malloc_Safely((void**)&this->d_local_restrain_numbers, sizeof(int));
    deviceMemset(this->d_local_restrain_numbers, 0, sizeof(int));
    Device_Malloc_Safely((void**)&this->d_local_restrain_list,
                         sizeof(int) * atom_numbers);
    Device_Malloc_Safely((void**)&this->local_crd_ref,
                         sizeof(VECTOR) * atom_numbers);

    auto set_refcoord_scaling = [&](const std::string& scaling)
    {
        if (scaling == "no" || scaling == "none")
            this->refcoord_scaling = REFCOORD_SCALING_NO;
        else if (scaling == "all")
            this->refcoord_scaling = REFCOORD_SCALING_ALL;
        else if (scaling == "com_ug")
            this->refcoord_scaling = REFCOORD_SCALING_COM_UG;
        else if (scaling == "com_res")
            this->refcoord_scaling = REFCOORD_SCALING_COM_RES;
        else if (scaling == "com_mol")
            this->refcoord_scaling = REFCOORD_SCALING_COM_MOL;
        else
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand,
                "RESTRAIN_INFORMATION::Initial(native H5)",
                "Reason:\n\trefcoord_scaling should be no/all/com_ug/"
                "com_res/com_mol\n");
    };
    this->refcoord_scaling = REFCOORD_SCALING_NO;
    if (controller->Command_Exist(this->module_name, "refcoord_scaling"))
    {
        set_refcoord_scaling(
            controller->Command(this->module_name, "refcoord_scaling"));
    }
    else if (native_restraint.has_refcoord_scaling_default)
    {
        set_refcoord_scaling(native_restraint.refcoord_scaling_default);
    }

    this->calc_virial = native_restraint.has_calc_virial_default
                            ? native_restraint.calc_virial_default
                            : true;
    if (controller->Command_Exist(this->module_name, "calc_virial"))
    {
        this->calc_virial =
            controller->Get_Bool(this->module_name, "calc_virial",
                                 "RESTRAIN_INFORMATION::Initial(native H5)");
    }

    Device_Malloc_Safely((void**)&this->d_ref_crd_all,
                         sizeof(VECTOR) * atom_numbers);
    if (native_restraint.reference_coordinates.empty())
    {
        deviceMemcpy(this->d_ref_crd_all, crd, sizeof(VECTOR) * atom_numbers,
                     deviceMemcpyDeviceToDevice);
    }
    else
    {
        if (native_restraint.reference_coordinates.size() !=
            static_cast<std::size_t>(atom_numbers) * 3)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand,
                "RESTRAIN_INFORMATION::Initial(native H5)",
                "Reason:\n\tnative positional restraint reference must have "
                "shape [atom_count,3]\n");
        }
        std::vector<VECTOR> reference(atom_numbers);
        for (int atom = 0; atom < atom_numbers; ++atom)
        {
            reference[atom].x =
                native_restraint.reference_coordinates[3 * atom];
            reference[atom].y =
                native_restraint.reference_coordinates[3 * atom + 1];
            reference[atom].z =
                native_restraint.reference_coordinates[3 * atom + 2];
        }
        deviceMemcpy(this->d_ref_crd_all, reference.data(),
                     sizeof(VECTOR) * atom_numbers, deviceMemcpyHostToDevice);
    }
    Device_Malloc_Safely((void**)&this->crd_ref,
                         sizeof(VECTOR) * this->restrain_numbers);
    Launch_Device_Kernel(
        Gather_Ref_From_All_Device,
        (this->restrain_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, this->restrain_numbers,
        this->d_lists, this->d_ref_crd_all, this->crd_ref);

    this->single_weight = 20.0f;
    if (controller->Command_Exist(this->module_name, "single_weight"))
    {
        controller->Check_Float(this->module_name, "single_weight",
                                "RESTRAIN_INFORMATION::Initial(native H5)");
        this->single_weight =
            atof(controller->Command(this->module_name, "single_weight"));
        this->if_single_weight = 1;
    }
    else if (native_restraint.has_single_weight_default)
    {
        this->single_weight = native_restraint.single_weight_default;
        this->if_single_weight = 1;
    }
    else
    {
        if (native_restraint.weight.size() !=
            static_cast<std::size_t>(this->restrain_numbers) * 3)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand,
                "RESTRAIN_INFORMATION::Initial(native H5)",
                "Reason:\n\tnative positional restraint weight must have "
                "shape [restraint_count,3]\n");
        }
        Malloc_Safely((void**)&this->h_weights,
                      sizeof(VECTOR) * this->restrain_numbers);
        for (int row = 0; row < this->restrain_numbers; ++row)
        {
            this->h_weights[row].x = native_restraint.weight[3 * row];
            this->h_weights[row].y = native_restraint.weight[3 * row + 1];
            this->h_weights[row].z = native_restraint.weight[3 * row + 2];
        }
        Device_Malloc_And_Copy_Safely((void**)&this->d_weights, this->h_weights,
                                      sizeof(VECTOR) * this->restrain_numbers);
        Device_Malloc_Safely((void**)&this->local_weights,
                             sizeof(VECTOR) * atom_numbers);
        this->if_single_weight = 0;
    }

    this->is_initialized = 1;
    if (!this->is_controller_printf_initialized)
    {
        controller->Step_Print_Initial(this->module_name, "%.2f");
        this->is_controller_printf_initialized = 1;
        controller->printf("    structure last modify date is %d\n",
                           last_modify_date);
    }
    controller->printf("END INITIALIZING NATIVE H5 RESTRAIN\n\n");
}

bool RESTRAIN_INFORMATION::Export_H5_Reference_Coordinates(
    std::vector<float>* coordinates, std::string* error) const
{
    if (coordinates == NULL)
    {
        if (error != NULL) *error = "restraint reference output is null";
        return false;
    }
    coordinates->clear();
    if (!is_initialized) return true;
    if (atom_numbers <= 0 || d_ref_crd_all == NULL)
    {
        if (error != NULL)
            *error = "initialized restraint has no global reference state";
        return false;
    }
    coordinates->resize(static_cast<std::size_t>(atom_numbers) * 3);
    deviceMemcpy(coordinates->data(), d_ref_crd_all,
                 sizeof(VECTOR) * atom_numbers, deviceMemcpyDeviceToHost);
    return true;
}

bool RESTRAIN_INFORMATION::Apply_H5_Reference_Coordinates(
    const std::vector<float>& coordinates, std::string* error)
{
    if (!is_initialized || atom_numbers <= 0 || d_ref_crd_all == NULL ||
        crd_ref == NULL)
    {
        if (error != NULL)
            *error =
                "restart contains restraint reference state, but the "
                "positional restraint module is not initialized";
        return false;
    }
    if (coordinates.size() != static_cast<std::size_t>(atom_numbers) * 3)
    {
        if (error != NULL)
        {
            *error =
                "restraint reference atom count does not match the "
                "initialized system";
        }
        return false;
    }
    deviceMemcpy(d_ref_crd_all, coordinates.data(),
                 sizeof(VECTOR) * atom_numbers, deviceMemcpyHostToDevice);
    Launch_Device_Kernel(
        Gather_Ref_From_All_Device,
        (restrain_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, restrain_numbers, d_lists,
        d_ref_crd_all, crd_ref);
    return true;
}

void RESTRAIN_INFORMATION::Update_Refcoord_Scaling(
    MD_INFORMATION* md_info, const LTMatrix3 g, float dt, int* atom_local,
    int local_atom_numbers, char* atom_local_label, int* atom_local_id)
{
    if (!is_initialized) return;
    if (refcoord_scaling == REFCOORD_SCALING_NO) return;
    if (md_info == NULL) return;
    if (md_info->mode != md_info->NPT) return;
    if (d_ref_crd_all == NULL || crd_ref == NULL) return;
    if (dt == 0.0f) return;

    switch (refcoord_scaling)
    {
        case REFCOORD_SCALING_ALL:
            Launch_Device_Kernel(
                Rescale_Ref_All_Device,
                (atom_numbers + CONTROLLER::device_max_thread - 1) /
                    CONTROLLER::device_max_thread,
                CONTROLLER::device_max_thread, 0, NULL, atom_numbers, g, dt,
                d_ref_crd_all);
            break;
        case REFCOORD_SCALING_COM_RES:
            if (!md_info->res.is_initialized ||
                md_info->res.residue_numbers <= 0 ||
                md_info->res.d_res_start == NULL ||
                md_info->res.d_res_end == NULL || md_info->d_mass == NULL)
            {
                printf("restrain refcoord_scaling com_res is not ready.\n");
                return;
            }
            Launch_Device_Kernel(Rescale_Ref_By_Group_Range_Device,
                                 (md_info->res.residue_numbers +
                                  CONTROLLER::device_max_thread - 1) /
                                     CONTROLLER::device_max_thread,
                                 CONTROLLER::device_max_thread, 0, NULL,
                                 md_info->res.residue_numbers,
                                 md_info->res.d_res_start,
                                 md_info->res.d_res_end, d_ref_crd_all,
                                 md_info->d_mass, g, dt, d_ref_crd_all);
            break;
        case REFCOORD_SCALING_COM_MOL:
            if (!md_info->mol.is_initialized ||
                md_info->mol.molecule_numbers <= 0 ||
                md_info->mol.d_atom_start == NULL ||
                md_info->mol.d_atom_end == NULL || md_info->d_mass == NULL)
            {
                printf("restrain refcoord_scaling com_mol is not ready.\n");
                return;
            }
            Launch_Device_Kernel(Rescale_Ref_By_Group_Range_Device,
                                 (md_info->mol.molecule_numbers +
                                  CONTROLLER::device_max_thread - 1) /
                                     CONTROLLER::device_max_thread,
                                 CONTROLLER::device_max_thread, 0, NULL,
                                 md_info->mol.molecule_numbers,
                                 md_info->mol.d_atom_start,
                                 md_info->mol.d_atom_end, d_ref_crd_all,
                                 md_info->d_mass, g, dt, d_ref_crd_all);
            break;
        case REFCOORD_SCALING_COM_UG:
            if (md_info->ug.ug_numbers <= 0 || md_info->ug.d_ug == NULL ||
                md_info->d_mass == NULL)
            {
                printf("restrain refcoord_scaling com_ug is not ready.\n");
                return;
            }
            Launch_Device_Kernel(
                Rescale_Ref_By_UG_Device,
                (md_info->ug.ug_numbers + CONTROLLER::device_max_thread - 1) /
                    CONTROLLER::device_max_thread,
                CONTROLLER::device_max_thread, 0, NULL, md_info->ug.ug_numbers,
                md_info->ug.d_ug, d_ref_crd_all, md_info->d_mass, g, dt,
                d_ref_crd_all);
            break;
        default:
            return;
    }

    Launch_Device_Kernel(
        Gather_Ref_From_All_Device,
        (this->restrain_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, this->restrain_numbers,
        this->d_lists, this->d_ref_crd_all, this->crd_ref);

    if (atom_local_label != NULL && atom_local_id != NULL)
    {
        Get_Local(atom_local, local_atom_numbers, atom_local_label,
                  atom_local_id);
    }
}

static __global__ void get_local_device(
    int restrain_numbers, char* atom_local_label, int* atom_local_id,
    int* d_lists, int* local_restrain_list, VECTOR* crd_ref,
    VECTOR* local_crd_ref, VECTOR* d_weights, VECTOR* local_weights,
    int* d_local_restrain_numbers, int if_single_weight)
{
#ifdef USE_GPU
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx != 0) return;
#endif
    // 遍历所有限制的原子全局序号，如果在当前区域，则设置local
    d_local_restrain_numbers[0] = 0;
    for (int i = 0; i < restrain_numbers; i++)
    {
        if (atom_local_label[d_lists[i]] == 1)
        {
            // local_list 存储dd的本地编号
            local_restrain_list[d_local_restrain_numbers[0]] =
                atom_local_id[d_lists[i]];
            local_crd_ref[d_local_restrain_numbers[0]] = crd_ref[i];
            if (!if_single_weight)
            {
                local_weights[d_local_restrain_numbers[0]] = d_weights[i];
            }
            d_local_restrain_numbers[0]++;
        }
    }
}

void RESTRAIN_INFORMATION::Get_Local(int* atom_local, int local_atom_numbers,
                                     char* atom_local_label, int* atom_local_id)
{
    if (!is_initialized) return;
    local_restrain_numbers = 0;
    Launch_Device_Kernel(get_local_device, 1, 1, 0, NULL, restrain_numbers,
                         atom_local_label, atom_local_id, this->d_lists,
                         this->d_local_restrain_list, this->crd_ref,
                         this->local_crd_ref, this->d_weights,
                         this->local_weights, this->d_local_restrain_numbers,
                         this->if_single_weight);
    deviceMemcpy(&local_restrain_numbers, this->d_local_restrain_numbers,
                 sizeof(int), deviceMemcpyDeviceToHost);
}

void RESTRAIN_INFORMATION::Restraint(const VECTOR* crd, Boundary boundary,
                                     int need_potential, float* atom_energy,
                                     int need_pressure, LTMatrix3* atom_virial,
                                     VECTOR* frc, MD_INFORMATION* md_info,
                                     DOMAIN_INFORMATION* dd)
{
    if (is_initialized)
    {
        const int* atom_local = NULL;
        const int* atom_to_group = NULL;
        const VECTOR* group_com = NULL;
        int effective_scaling = this->refcoord_scaling;
        int need_virial = need_pressure && this->calc_virial;

        if (need_virial)
        {
            if (dd == NULL || md_info == NULL)
            {
                effective_scaling = REFCOORD_SCALING_NO;
            }
            else
            {
                atom_local = dd->atom_local;
            }
            if (md_info != NULL)
            {
                this->Init_Com_Cache_If_Needed(this->atom_numbers, *md_info);
            }
            if (this->refcoord_scaling == REFCOORD_SCALING_COM_UG &&
                this->d_atom_to_ug != NULL && this->d_com_ug != NULL)
            {
                if (atom_local == NULL)
                {
                    effective_scaling = REFCOORD_SCALING_NO;
                }
                else
                {
                    this->Update_Group_COM(
                        dd->atom_numbers, crd, atom_local, this->d_atom_to_ug,
                        md_info->d_mass, this->cached_ug_numbers,
                        this->d_sum_mass_ug, this->d_sum_pos_ug, this->d_com_ug,
                        this->h_sum_mass_ug, this->h_sum_pos_ug,
                        this->h_com_ug);
                    atom_to_group = this->d_atom_to_ug;
                    group_com = this->d_com_ug;
                }
            }
            else if (this->refcoord_scaling == REFCOORD_SCALING_COM_RES &&
                     this->d_atom_to_res != NULL && this->d_com_res != NULL)
            {
                if (atom_local == NULL)
                {
                    effective_scaling = REFCOORD_SCALING_NO;
                }
                else
                {
                    this->Update_Group_COM(
                        dd->atom_numbers, crd, atom_local, this->d_atom_to_res,
                        md_info->d_mass, this->cached_res_numbers,
                        this->d_sum_mass_res, this->d_sum_pos_res,
                        this->d_com_res, this->h_sum_mass_res,
                        this->h_sum_pos_res, this->h_com_res);
                    atom_to_group = this->d_atom_to_res;
                    group_com = this->d_com_res;
                }
            }
            else if (this->refcoord_scaling == REFCOORD_SCALING_COM_MOL &&
                     this->d_atom_to_mol != NULL && this->d_com_mol != NULL)
            {
                if (atom_local == NULL)
                {
                    effective_scaling = REFCOORD_SCALING_NO;
                }
                else
                {
                    this->Update_Group_COM(
                        dd->atom_numbers, crd, atom_local, this->d_atom_to_mol,
                        md_info->d_mass, this->cached_mol_numbers,
                        this->d_sum_mass_mol, this->d_sum_pos_mol,
                        this->d_com_mol, this->h_sum_mass_mol,
                        this->h_sum_pos_mol, this->h_com_mol);
                    atom_to_group = this->d_atom_to_mol;
                    group_com = this->d_com_mol;
                }
            }
            else if (this->refcoord_scaling == REFCOORD_SCALING_COM_UG ||
                     this->refcoord_scaling == REFCOORD_SCALING_COM_RES ||
                     this->refcoord_scaling == REFCOORD_SCALING_COM_MOL)
            {
                effective_scaling = REFCOORD_SCALING_NO;
            }
        }

        Launch_Device_Kernel(
            restrain_force_with_atom_energy_and_virial,
            (local_restrain_numbers + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, local_restrain_numbers,
            this->d_local_restrain_list, crd, this->local_crd_ref,
            this->if_single_weight, this->single_weight, this->local_weights,
            boundary, need_potential, atom_energy, need_virial, atom_virial,
            frc, this->d_restrain_ene, effective_scaling, atom_local,
            atom_to_group, group_com);
    }
}

void RESTRAIN_INFORMATION::Step_Print(CONTROLLER* controller)
{
    if (is_initialized && CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size)
    {
        Sum_Of_List(d_restrain_ene, d_sum_of_restrain_ene,
                    local_restrain_numbers);
        deviceMemcpy(h_sum_of_restrain_ene, d_sum_of_restrain_ene,
                     sizeof(float), deviceMemcpyDeviceToHost);
#ifdef USE_MPI
        MPI_Allreduce(MPI_IN_PLACE, h_sum_of_restrain_ene, 1, MPI_FLOAT,
                      MPI_SUM, CONTROLLER::pp_comm);
#endif
        controller->Step_Print(this->module_name, h_sum_of_restrain_ene[0],
                               true);
    }
}

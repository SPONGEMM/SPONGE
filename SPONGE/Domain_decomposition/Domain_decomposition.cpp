#include "Domain_decomposition.h"

#ifdef USE_CPU  // CPU 线程并行内存分配
extern int max_omp_threads;
extern int frc_size;
extern int atom_energy_size;
extern int atom_virial_size;
#endif

void DOMAIN_INFORMATION::Domain_Decomposition(CONTROLLER* controller,
                                              MD_INFORMATION* md_info)
{
    strcpy(this->module_name, "DOM_DEC");
    controller->printf("DOMAIN DECOMPOSITION BEGIN\n");
    Device_Malloc_Safely((void**)&d_sum_ene_local, sizeof(float));
    Device_Malloc_Safely((void**)&d_sum_ene_total, sizeof(float));
    Device_Malloc_Safely((void**)&d_ek_local, sizeof(float));
    Device_Malloc_Safely((void**)&d_ek_total, sizeof(float));
    deviceMemset(d_ek_local, 0, sizeof(float));
    deviceMemset(d_ek_total, 0, sizeof(float));

    update_interval = 100;
    if (controller->Command_Exist(this->module_name, "update_interval"))
    {
        update_interval =
            atoi(controller->Command(this->module_name, "update_interval"));
    }
    controller->printf("    domain decomposition update_interval = %d\n",
                       update_interval);

    // One PP MPI Process
    if (controller->PP_MPI_size == 1)
    {
        dom_dec_split_num.int_x = 1;
        dom_dec_split_num.int_y = 1;
        dom_dec_split_num.int_z = 1;
        min_corner.x = 0;
        min_corner.y = 0;
        min_corner.z = 0;
        max_corner = md_info->sys.box_length;
        controller->printf("DOMAIN DECOMPOSITION END\n");
        controller->printf("nx=%d, ny=%d, nz=%d\n", dom_dec_split_num.int_x,
                           dom_dec_split_num.int_y, dom_dec_split_num.int_z);
        return;
    }

#ifdef USE_MPI
    VECTOR box_length = md_info->sys.box_length;
    VECTOR box_angle = md_info->sys.box_angle;
    int nx, ny, nz;

    if (controller->Command_Exist(this->module_name, "split_nx") ||
        controller->Command_Exist(this->module_name, "split_ny") ||
        controller->Command_Exist(this->module_name, "split_nz"))
    {
        controller->Check_Int(this->module_name, "split_nx",
                              "Particle_Mesh_Ewald::Initial");
        controller->Check_Int(this->module_name, "split_ny",
                              "Particle_Mesh_Ewald::Initial");
        controller->Check_Int(this->module_name, "split_nz",
                              "Particle_Mesh_Ewald::Initial");
        dom_dec_split_num.int_x =
            atoi(controller->Command(this->module_name, "split_nx"));
        dom_dec_split_num.int_y =
            atoi(controller->Command(this->module_name, "split_ny"));
        dom_dec_split_num.int_z =
            atoi(controller->Command(this->module_name, "split_nz"));
    }
    else
    {
        int PP_MPI_size = controller->PP_MPI_size;
        VECTOR area;
        area.x = box_length.y * box_length.z *
                 (float)std::sin(CONSTANT_Pi * box_angle.x / 180.0);
        area.y = box_length.x * box_length.z *
                 (float)std::sin(CONSTANT_Pi * box_angle.y / 180.0);
        area.z = box_length.x * box_length.y *
                 (float)std::sin(CONSTANT_Pi * box_angle.z / 180.0);

        float best_area = 2 * (area.x + area.y + area.z);
        float area_tmp;

        int nremain;
        int cntx, cnty, cntz;
        nx = 1;
        while (nx <= PP_MPI_size)
        {
            if (PP_MPI_size % nx == 0)
            {
                nremain = PP_MPI_size / nx;
                ny = 1;
                while (ny <= nremain)
                {
                    if (nremain % ny == 0)
                    {
                        nz = nremain / ny;
                        cntx = nx == 1 ? 0 : 2;
                        cnty = ny == 1 ? 0 : 2;
                        cntz = nz == 1 ? 0 : 2;
                        area_tmp = cntx * area.x / ny / nz +
                                   cnty * area.y / nx / nz +
                                   cntz * area.z / nx / ny;
                        // area_tmp = 2*area.x / ny / nz + 2*area.y / nx / nz +
                        // 2*area.z / nx / ny;
                        if (area_tmp < best_area)
                        {
                            best_area = area_tmp;
                            // printf("best_area=%.4f\n", best_area);
                            dom_dec_split_num.int_x = nx;
                            dom_dec_split_num.int_y = ny;
                            dom_dec_split_num.int_z = nz;
                        }
                    }
                    ny++;
                }
            }
            nx++;
        }
    }

    /*
        在确定表面积最小的分割方式后，将各个进程对应的min_corner,max_corner确定下来
    */

    nx = dom_dec_split_num.int_x;
    ny = dom_dec_split_num.int_y;
    nz = dom_dec_split_num.int_z;

    int rank_id;
    for (int k = 0; k < nz; ++k)
    {
        for (int j = 0; j < ny; ++j)
        {
            for (int i = 0; i < nx; ++i)
            {
                rank_id = i + j * nx + k * nx * ny;
                min_corner_set[rank_id].x = box_length.x / nx * i;
                min_corner_set[rank_id].y = box_length.y / ny * j;
                min_corner_set[rank_id].z = box_length.z / nz * k;
                max_corner_set[rank_id].x = box_length.x / nx * (i + 1);
                max_corner_set[rank_id].y = box_length.y / ny * (j + 1);
                max_corner_set[rank_id].z = box_length.z / nz * (k + 1);
            }
        }
    }

    controller->printf("DOMAIN DECOMPOSITION END\n");
    controller->printf("pp_nx=%d, pp_ny=%d, pp_nz=%d\n",
                       dom_dec_split_num.int_x, dom_dec_split_num.int_y,
                       dom_dec_split_num.int_z);
#endif
}

void DOMAIN_INFORMATION::Send_Recv_Dom_Dec(CONTROLLER* controller)
{
    if (controller->PP_MPI_size == 1)
    {
        return;
    }
#ifdef USE_MPI
    if (pp_rank == 0)
    {
        min_corner = min_corner_set[0];
        max_corner = max_corner_set[0];
        for (int r_id = 1; r_id < controller->PP_MPI_size; ++r_id)
        {
            MPI_Send(&min_corner_set[r_id], sizeof(VECTOR), MPI_BYTE, r_id, 0,
                     CONTROLLER::pp_comm);
            MPI_Send(&max_corner_set[r_id], sizeof(VECTOR), MPI_BYTE, r_id, 1,
                     CONTROLLER::pp_comm);
            MPI_Send(&dom_dec_split_num, sizeof(INT_VECTOR), MPI_BYTE, r_id, 2,
                     CONTROLLER::pp_comm);
        }
    }
    else
    {
        MPI_Recv(&min_corner, sizeof(VECTOR), MPI_BYTE, 0, 0,
                 CONTROLLER::pp_comm, MPI_STATUS_IGNORE);
        MPI_Recv(&max_corner, sizeof(VECTOR), MPI_BYTE, 0, 1,
                 CONTROLLER::pp_comm, MPI_STATUS_IGNORE);
        MPI_Recv(&dom_dec_split_num, sizeof(INT_VECTOR), MPI_BYTE, 0, 2,
                 CONTROLLER::pp_comm, MPI_STATUS_IGNORE);
    }
    controller->printf("FINISH SEND/RECEIVE THE DOMAIN INFORMATION\n");
    printf(
        "rank=%d, min_corner=(%.3f, %.3f, %.3f), max_corner=(%.3f, "
        "%.3f,%.3f)\n",
        pp_rank, min_corner.x, min_corner.y, min_corner.z, max_corner.x,
        max_corner.y, max_corner.z);
#endif
}

void DOMAIN_INFORMATION::Find_Neighbor_Domain(CONTROLLER* controller,
                                              MD_INFORMATION* md_info)
{
    if (controller->PP_MPI_size == 1)
    {
        for (int dir = 0; dir < 6; ++dir)
        {
            h_neighbor_num[dir] = 0;
        }
        return;
    }

#ifdef USE_MPI
    int nx = dom_dec_split_num.int_x;
    int ny = dom_dec_split_num.int_y;
    int nz = dom_dec_split_num.int_z;

    // 初始化六个方向的邻居数，因为是均匀切分，因此不存在错位的情况，邻居数只能是0或1
    // 这里默认周期性边界
    h_neighbor_num[0] = nx == 1 ? 0 : 1;
    h_neighbor_num[1] = nx == 1 ? 0 : 1;
    h_neighbor_num[2] = ny == 1 ? 0 : 1;
    h_neighbor_num[3] = ny == 1 ? 0 : 1;
    h_neighbor_num[4] = nz == 1 ? 0 : 1;
    h_neighbor_num[5] = nz == 1 ? 0 : 1;

    // 获取当前domain的空间坐标
    int rank_id = pp_rank;
    int i = rank_id % (nx);
    int j = (rank_id / nx) % ny;
    int k = rank_id / (nx * ny);

    // x方向邻居
    if (nx > 1)
    {
        h_neighbor_dir[0][0] = (i + 1) % nx + j * nx + k * nx * ny;
        h_neighbor_dir[1][0] = (i - 1 + nx) % nx + j * nx + k * nx * ny;
    }
    // y方向邻居
    if (ny > 1)
    {
        h_neighbor_dir[2][0] = i % nx + ((j + 1) % ny) * nx + k * nx * ny;
        h_neighbor_dir[3][0] = i % nx + ((j - 1 + ny) % ny) * nx + k * nx * ny;
    }
    // z方向邻居
    if (nz > 1)
    {
        h_neighbor_dir[4][0] = i % nx + j * nx + ((k + 1) % nz) * nx * ny;
        h_neighbor_dir[5][0] = i % nx + j * nx + ((k - 1 + nz) % nz) * nx * ny;
    }
    is_initialized = 1;
    controller->printf("FINISH INITIALIZING THE DOMAIN INFORMATION\n");
#endif
}

// 归属于当前区域的粒子和残基信息获取
static __global__ void get_atom_and_residues(
    INT_VECTOR dom_dec_split_num, int ug_numbers, ATOM_GROUP* ug,
    VECTOR* md_info_crd, VECTOR* md_info_vel, float* md_info_d_mass,
    float* md_info_d_mass_inverse, float* md_info_d_charge, LTMatrix3 rcell,
    LTMatrix3 cell, VECTOR min_corner, VECTOR dom_box_length, int* res_start,
    int* res_len, int* res_numbers, int* atom_local, char* atom_local_label,
    int* atom_local_id, int* atom_numbers, VECTOR* crd, VECTOR* vel,
    float* d_mass, float* d_mass_inverse, float* d_charge, VECTOR box_length)
{
#ifdef USE_GPU
    // 以下内容在gpu上串行执行一次
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > 0) return;
#endif

    res_numbers[0] = 0;
    atom_numbers[0] = 0;
    int atom_res_start;  // 残基的首粒子坐标

    // 对残基中粒子归属的判断
    for (int idx = 0; idx < ug_numbers; ++idx)
    {
        atom_res_start = ug[idx].atom_serial[0];

        VECTOR frac_crd_i = md_info_crd[atom_res_start] * rcell -
                            floorf(md_info_crd[atom_res_start] * rcell);
        VECTOR crd_i = wiseproduct(frac_crd_i, box_length);
        // VECTOR crd_i = md_info_crd[atom_res_start] -
        //                floorf(md_info_crd[atom_res_start] * rcell) * cell;
        VECTOR dr = crd_i - min_corner;

        if (dr.x >= 0 && dr.x < dom_box_length.x && dr.y >= 0 &&
            dr.y < dom_box_length.y && dr.z >= 0 && dr.z < dom_box_length.z)
        {
            res_start[res_numbers[0]] = atom_numbers[0];
            res_len[res_numbers[0]] = ug[idx].atom_numbers;
            res_numbers[0] += 1;
            for (int _atom_i = 0; _atom_i < ug[idx].atom_numbers; _atom_i++)
            {
                int atom_i = ug[idx].atom_serial[_atom_i];
                atom_local[atom_numbers[0]] = atom_i;
                crd[atom_numbers[0]] = md_info_crd[atom_i];
                vel[atom_numbers[0]] = md_info_vel[atom_i];
                d_mass[atom_numbers[0]] = md_info_d_mass[atom_i];
                d_mass_inverse[atom_numbers[0]] =
                    md_info_d_mass_inverse[atom_i];
                d_charge[atom_numbers[0]] = md_info_d_charge[atom_i];
                atom_local_label[atom_i] = 1;
                atom_local_id[atom_i] = atom_numbers[0];
                atom_numbers[0] += 1;
            }
        }
    }
}

// 归属于当前区域的粒子和残基信息获取
static __global__ void get_atom_and_residues_single_domain(
    INT_VECTOR dom_dec_split_num, int ug_numbers, ATOM_GROUP* ug,
    VECTOR* md_info_crd, VECTOR* md_info_vel, float* md_info_d_mass,
    float* md_info_d_mass_inverse, float* md_info_d_charge, LTMatrix3 rcell,
    LTMatrix3 cell, VECTOR min_corner, VECTOR dom_box_length, int* res_start,
    int* res_len, int* res_numbers, int* atom_local, char* atom_local_label,
    int* atom_local_id, int* atom_numbers, VECTOR* crd, VECTOR* vel,
    float* d_mass, float* d_mass_inverse, float* d_charge, int tot_atom_numbers)
{
#ifdef USE_GPU
    // 以下内容在gpu上串行执行一次
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > 0) return;
#endif

    (void)tot_atom_numbers;
    res_numbers[0] = 0;
    atom_numbers[0] = 0;

    // 单域模式下也按残基顺序填充，保证res_start/res_len与局域原子数组一致
    for (int idx = 0; idx < ug_numbers; ++idx)
    {
        res_start[res_numbers[0]] = atom_numbers[0];
        res_len[res_numbers[0]] = ug[idx].atom_numbers;
        res_numbers[0] += 1;
        for (int _atom_i = 0; _atom_i < ug[idx].atom_numbers; _atom_i++)
        {
            int atom_i = ug[idx].atom_serial[_atom_i];
            atom_local[atom_numbers[0]] = atom_i;
            crd[atom_numbers[0]] = md_info_crd[atom_i];
            vel[atom_numbers[0]] = md_info_vel[atom_i];
            d_mass[atom_numbers[0]] = md_info_d_mass[atom_i];
            d_mass_inverse[atom_numbers[0]] = md_info_d_mass_inverse[atom_i];
            d_charge[atom_numbers[0]] = md_info_d_charge[atom_i];
            atom_local_label[atom_i] = 1;
            atom_local_id[atom_i] = atom_numbers[0];
            atom_numbers[0] += 1;
        }
    }
}

static __global__ void set_char_array(char* array, int size, char value)
{
    SIMPLE_DEVICE_FOR(idx, size) { array[idx] = value; }
}

void DOMAIN_INFORMATION::Get_Atoms(CONTROLLER* controller,
                                   MD_INFORMATION* md_info)
{
    max_atom_numbers = md_info->atom_numbers +
                       md_info->no_direct_interaction_virtual_atom_numbers;
    max_res_numbers = md_info->ug.ug_numbers;

    controller->printf("max_atom_numbers=%d, max_res_numbers=%d\n",
                       this->max_atom_numbers, this->max_res_numbers);
    Device_Malloc_Safely((void**)&atom_local, sizeof(int) * max_atom_numbers);
    deviceMemset(atom_local, -1, sizeof(int) * max_atom_numbers);

    // 局域粒子的label，如果粒子在当前区域中，即设为1，反之为0
    Device_Malloc_Safely((void**)&atom_local_label,
                         sizeof(char) * max_atom_numbers);
    Launch_Device_Kernel(set_char_array, (max_atom_numbers + 255) / 256, 256, 0,
                         NULL, atom_local_label, max_atom_numbers, 0);
    // 局域粒子的local_id，如果不在当前区域内，则置为-1
    Device_Malloc_Safely((void**)&atom_local_id,
                         sizeof(int) * max_atom_numbers);
    deviceMemset(atom_local_id, -1, sizeof(int) * max_atom_numbers);
    // 局域残基信息的buffer
    Device_Malloc_Safely((void**)&res_start, sizeof(int) * max_res_numbers);
    Device_Malloc_Safely((void**)&res_len, sizeof(int) * max_res_numbers);
    Device_Malloc_Safely((void**)&d_center_of_mass,
                         sizeof(float) * max_res_numbers * 3);
    Device_Malloc_Safely((void**)&crd, sizeof(VECTOR) * max_atom_numbers);
    Device_Malloc_Safely((void**)&vel, sizeof(VECTOR) * max_atom_numbers);
    Device_Malloc_Safely((void**)&acc, sizeof(VECTOR) * max_atom_numbers);
    deviceMemset(crd, 0, sizeof(VECTOR) * max_atom_numbers);
    deviceMemset(vel, 0, sizeof(VECTOR) * max_atom_numbers);
    deviceMemset(acc, 0, sizeof(VECTOR) * max_atom_numbers);
#ifndef USE_CPU
    Device_Malloc_Safely((void**)&frc, sizeof(VECTOR) * max_atom_numbers);
    Device_Malloc_Safely((void**)&d_virial,
                         sizeof(LTMatrix3) * max_atom_numbers);
    Device_Malloc_Safely((void**)&d_energy, sizeof(float) * max_atom_numbers);
    deviceMemset(frc, 0, sizeof(VECTOR) * max_atom_numbers);
    deviceMemset(d_virial, 0, sizeof(LTMatrix3) * max_atom_numbers);
    deviceMemset(d_energy, 0, sizeof(float) * max_atom_numbers);
#else
    // 每个区域的frc最起码要预留非直接作用的虚拟原子数量+10000个粒子的空间
    tmp_frc_size = (md_info->atom_numbers) / controller->PP_MPI_size + 10000 +
                   md_info->no_direct_interaction_virtual_atom_numbers;
    Device_Malloc_Safely((void**)&frc,
                         sizeof(VECTOR) * tmp_frc_size * max_omp_threads);
    Device_Malloc_Safely((void**)&frc_buffer,
                         sizeof(VECTOR) * max_atom_numbers);
    Device_Malloc_Safely((void**)&d_virial,
                         sizeof(LTMatrix3) * tmp_frc_size * max_omp_threads);
    Device_Malloc_Safely((void**)&d_energy,
                         sizeof(float) * tmp_frc_size * max_omp_threads);
    deviceMemset(frc, 0, sizeof(VECTOR) * tmp_frc_size * max_omp_threads);
    deviceMemset(frc_buffer, 0, sizeof(VECTOR) * max_atom_numbers);
    deviceMemset(d_virial, 0,
                 sizeof(LTMatrix3) * tmp_frc_size * max_omp_threads);
    deviceMemset(d_energy, 0, sizeof(float) * tmp_frc_size * max_omp_threads);
    frc_size = tmp_frc_size;
    atom_energy_size = tmp_frc_size;
    atom_virial_size = tmp_frc_size;
#endif
    Device_Malloc_Safely((void**)&d_mass, sizeof(float) * max_atom_numbers);
    Device_Malloc_Safely((void**)&d_mass_inverse,
                         sizeof(float) * max_atom_numbers);
    Device_Malloc_Safely((void**)&d_charge, sizeof(float) * max_atom_numbers);
    Device_Malloc_Safely((void**)&d_ek, sizeof(float) * max_atom_numbers);
    if (d_ek_local == NULL)
        Device_Malloc_Safely((void**)&d_ek_local, sizeof(float));
        deviceMemset(d_ek_local, 0, sizeof(float));
    
    if (d_ek_total == NULL)
        Device_Malloc_Safely((void**)&d_ek_total, sizeof(float));
        deviceMemset(d_ek_total, 0, sizeof(float));
    Device_Malloc_Safely((void**)&d_num_ghost_dir, sizeof(int) * 6);
    Malloc_Safely((void**)&h_num_ghost_dir_id,
                  sizeof(int) * max_atom_numbers * 6);
    Device_Malloc_Safely((void**)&d_num_ghost_dir_id,
                         sizeof(int) * max_atom_numbers * 6);
    Device_Malloc_Safely((void**)&d_num_ghost_dir_re, sizeof(int) * 6);
    Device_Malloc_Safely((void**)&d_num_ghost_res_dir, sizeof(int) * 6);
    Device_Malloc_Safely((void**)&d_num_ghost_res_dir_re, sizeof(int) * 6);
    deviceMemset(d_num_ghost_dir, 0, sizeof(int) * 6);
    deviceMemset(d_num_ghost_dir_id, -1, sizeof(int) * max_atom_numbers * 6);
    deviceMemset(d_num_ghost_dir_re, 0, sizeof(int) * 6);
    deviceMemset(d_num_ghost_res_dir, 0, sizeof(int) * 6);
    deviceMemset(d_num_ghost_res_dir_re, 0, sizeof(int) * 6);

    Device_Malloc_Safely((void**)&d_excluded_numbers,
                         sizeof(int) * max_atom_numbers);
    Device_Malloc_Safely((void**)&d_excluded_list_start,
                         sizeof(int) * max_atom_numbers);
    Device_Malloc_Safely((void**)&d_excluded_list,
                         sizeof(int) * md_info->nb.excluded_atom_numbers);

    Device_Malloc_Safely((void**)&d_atom_numbers, sizeof(int));
    Device_Malloc_Safely((void**)&d_res_numbers, sizeof(int));
    Device_Malloc_Safely((void**)&d_ghost_numbers, sizeof(int));
    Device_Malloc_Safely((void**)&d_ghost_res_numbers, sizeof(int));
    deviceMemset(d_atom_numbers, 0, sizeof(int));
    deviceMemset(d_res_numbers, 0, sizeof(int));
    deviceMemset(d_ghost_numbers, 0, sizeof(int));
    deviceMemset(d_ghost_res_numbers, 0, sizeof(int));

    VECTOR dom_box_length = max_corner - min_corner;  // 当前区域的box尺寸

    if (CONTROLLER::PP_MPI_size != 1)  // if (CONTROLLER::PP_MPI_rank != 1)  //
                                       // if (CONTROLLER::PP_MPI_size != 1)
    {
        Launch_Device_Kernel(
            get_atom_and_residues, 1, 1, 0, NULL, dom_dec_split_num,
            md_info->ug.ug_numbers, md_info->ug.d_ug, md_info->crd,
            md_info->vel, md_info->d_mass, md_info->d_mass_inverse,
            md_info->d_charge, md_info->pbc.rcell, md_info->pbc.cell,
            min_corner, dom_box_length, this->res_start, this->res_len,
            this->d_res_numbers, this->atom_local, this->atom_local_label,
            this->atom_local_id, this->d_atom_numbers, this->crd, this->vel,
            this->d_mass, this->d_mass_inverse, this->d_charge,
            md_info->sys.box_length);
    }
    else
    {
        Launch_Device_Kernel(
            get_atom_and_residues_single_domain, 1, 1, 0, NULL,
            dom_dec_split_num, md_info->ug.ug_numbers, md_info->ug.d_ug,
            md_info->crd, md_info->vel, md_info->d_mass,
            md_info->d_mass_inverse, md_info->d_charge, md_info->pbc.rcell,
            md_info->pbc.cell, min_corner, dom_box_length, this->res_start,
            this->res_len, this->d_res_numbers, this->atom_local,
            this->atom_local_label, this->atom_local_id, this->d_atom_numbers,
            this->crd, this->vel, this->d_mass, this->d_mass_inverse,
            this->d_charge, md_info->atom_numbers);
    }

    deviceMemcpy(&this->atom_numbers, d_atom_numbers, sizeof(int),
                 deviceMemcpyDeviceToHost);
    deviceMemcpy(&this->res_numbers, d_res_numbers, sizeof(int),
                 deviceMemcpyDeviceToHost);

    printf("rank_id=%d, atom_numbers=%d, residue_numbers=%d\n", pp_rank,
           this->atom_numbers, this->res_numbers);
}

static __global__ void plan_decider(int res_numbers, int* res_start,
                                    int* res_len, VECTOR* crd, LTMatrix3 rcell,
                                    LTMatrix3 cell, VECTOR min_corner,
                                    VECTOR max_corner, float cutoff,
                                    VECTOR box_length, unsigned int* plan)
{
    SIMPLE_DEVICE_FOR(i, res_numbers)
    {
        int atom_res_start = res_start[i];
        VECTOR frac_crd_i =
            crd[atom_res_start] * rcell - floorf(crd[atom_res_start] * rcell);
        VECTOR crd_i = wiseproduct(frac_crd_i, box_length);
        // VECTOR crd_i =
        //     crd[atom_res_start] - floorf(crd[atom_res_start] * rcell) * cell;
        VECTOR dr1 = crd_i - max_corner + cutoff;
        VECTOR dr2 = box_length + cutoff - crd_i;

        int plan_res = 0;
        if (crd_i.x < min_corner.x + cutoff)
            plan_res |= send_west;
        else if (crd_i.x > max_corner.x - cutoff)
            plan_res |= send_east << (dr1.x > dr2.x);
        if (crd_i.y < min_corner.y + cutoff)
            plan_res |= send_south;
        else if (crd_i.y > max_corner.y - cutoff)
            plan_res |= send_north << (dr1.y > dr2.y);
        if (crd_i.z < min_corner.z + cutoff)
            plan_res |= send_down;
        else if (crd_i.z > max_corner.z - cutoff)
            plan_res |= send_up << (dr1.z > dr2.z);
        plan[i] = plan_res;
    }
}

// 20251002 设置传输buffer信息
static __global__ void ghost_to_buffer(
    unsigned int dir, int* d_num_ghost_res_dir_re, int max_res_trans,
    int* res_start, unsigned int* plan, int* res_len, int* ghost_buffer,
    VECTOR* crd_buffer, unsigned int* plan_buffer, int* res_len_buffer,
    int* atom_local, VECTOR* crd, int* d_num_ghost_dir, int* d_num_ghost_dir_id,
    int* d_num_ghost_res_dir, int max_atom_numbers)
{
#ifdef USE_GPU
    // 以下内容在gpu上串行执行一次
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 0) return;
#endif
    // 遍历残基，填充需要传递的buffer
    //  int atom_start_i = 0;
    //  int plan_res_i, res_len_i;
    // 修正近邻通信中的粒子重复传递现象
    int dir_res = 0;
    if (dir % 2) dir_res = d_num_ghost_res_dir_re[dir];
    for (int i = 0; i < max_res_trans - dir_res; ++i)
    {
        int atom_start_i = res_start
            [i];  // 由于在传递信息时无法传递残基的起点，因此在遍历残基时对残基起点重新赋值
        int plan_res_i = plan[i];
        int res_len_i = res_len[i];

        if (plan_res_i & (1 << dir))
        {
            for (int atom_i = atom_start_i; atom_i < (atom_start_i + res_len_i);
                 ++atom_i)
            {
                ghost_buffer[d_num_ghost_dir[dir]] = atom_local[atom_i];
                crd_buffer[d_num_ghost_dir[dir]] = crd[atom_i];
                d_num_ghost_dir_id[d_num_ghost_dir[dir] +
                                   dir * max_atom_numbers] = atom_i;
                d_num_ghost_dir[dir]++;
            }
            res_len_buffer[d_num_ghost_res_dir[dir]] = res_len_i;
            plan_buffer[d_num_ghost_res_dir[dir]] = plan_res_i;
            d_num_ghost_res_dir[dir]++;
        }
        // atom_start_i += res_len_i;
    }
}

static __global__ void refresh_res(int res_numbers, int ghost_res_numbers,
                                   int* res_start, int* res_len, int start_idx,
                                   int res_start_idx)
{
#ifdef USE_GPU
    // 以下内容在gpu上串行执行一次
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 0) return;
#endif
    int temp_start_idx = start_idx;
    int res_end = res_numbers + ghost_res_numbers;
    for (int res_i = res_start_idx; res_i < res_end; ++res_i)
    {
        res_start[res_i] = temp_start_idx;
        temp_start_idx += res_len[res_i];
    }
}

static __global__ void save_ghost_id(int atom_numbers, int total_atom_numbers,
                                     int* atom_local, int* atom_local_id,
                                     float* d_charge, float* md_info_d_charge)
{
#ifdef USE_GPU
    for (int i = atom_numbers + blockIdx.x * blockDim.x + threadIdx.x;
         i < total_atom_numbers; i += blockDim.x * gridDim.x)
#else
#pragma omp parallel for
    for (int i = atom_numbers; i < total_atom_numbers; ++i)
#endif
    {
        int atom_i = atom_local[i];
        atom_local_id[atom_i] = i;
        d_charge[i] = md_info_d_charge[atom_i];
    }
}

void DOMAIN_INFORMATION::Get_Ghost(CONTROLLER* controller,
                                   MD_INFORMATION* md_info)
{
    if (controller->PP_MPI_size == 1)
    {
        return;
    }
#ifdef USE_MPI
    float cutoff = md_info->nb.cutoff + md_info->nb.skin;
    VECTOR box_length = md_info->sys.box_length;

    // 粒子的传输计划，plan中存储了当前进程中每个粒子所需的近邻传输方式，存储在device上
    unsigned int* plan;
    Device_Malloc_Safely((void**)&plan, max_res_numbers * sizeof(unsigned int));

    Launch_Device_Kernel(plan_decider, (max_res_numbers + 255) / 256, 256, 0,
                         NULL, res_numbers, res_start, res_len, crd,
                         md_info->pbc.rcell, md_info->pbc.cell, min_corner,
                         max_corner, cutoff, box_length, plan);

    deviceMemset(d_num_ghost_dir, 0, sizeof(int) * 6);
    deviceMemset(d_num_ghost_res_dir, 0, sizeof(int) * 6);
    for (unsigned int dir = 0; dir < 6; ++dir)
    {
        h_num_ghost_dir[dir] = 0;
        h_num_ghost_res_dir[dir] = 0;
        if (!h_neighbor_num[dir]) continue;

        int max_atom_trans =
            atom_numbers +
            ghost_numbers;  // 在粒子传递过程中，可能传递的最大粒子数
        // ghost 也算进max_res_trans中，在判断ghost时需要考虑已经在ghost中的残基
        int max_res_trans =
            res_numbers +
            ghost_res_numbers;  // 在粒子传递过程中，可能传递的最大残基数目

        // 用于传递粒子/残基信息的buffer  都是device内存
        int* ghost_buffer;
        int* res_len_buffer;
        unsigned int* plan_buffer;
        VECTOR* crd_buffer;
        Device_Malloc_Safely((void**)&ghost_buffer,
                             sizeof(int) * max_atom_trans);
        Device_Malloc_Safely((void**)&res_len_buffer,
                             sizeof(int) * max_res_trans);
        Device_Malloc_Safely((void**)&plan_buffer,
                             sizeof(unsigned int) * max_res_trans);
        Device_Malloc_Safely((void**)&crd_buffer,
                             sizeof(VECTOR) * max_atom_trans);
        Launch_Device_Kernel(
            ghost_to_buffer, 1, 1, 0, NULL, dir, d_num_ghost_res_dir_re,
            max_res_trans, res_start, plan, res_len, ghost_buffer, crd_buffer,
            plan_buffer, res_len_buffer, atom_local, crd, d_num_ghost_dir,
            d_num_ghost_dir_id, d_num_ghost_res_dir, max_atom_numbers);
        deviceMemcpy(&h_num_ghost_dir[dir], &d_num_ghost_dir[dir], sizeof(int),
                     deviceMemcpyDeviceToHost);
        deviceMemcpy(&h_num_ghost_res_dir[dir], &d_num_ghost_res_dir[dir],
                     sizeof(int), deviceMemcpyDeviceToHost);
        deviceMemcpy(&h_num_ghost_res_dir_re[dir], &d_num_ghost_res_dir_re[dir],
                     sizeof(int), deviceMemcpyDeviceToHost);
        int send_dir = dir;
        int recv_dir = dir % 2 ? (dir - 1) : (dir + 1);
        int send_neighbor = h_neighbor_dir[send_dir][0];
        int recv_neighbor = h_neighbor_dir[recv_dir][0];

        // host and device request/status holders
        MPI_Request reqs[2];
        MPI_Status status[2];
        D_MPI_Request d_reqs[4];
        D_MPI_Status d_status[4];

#ifdef USE_GPU
        // create device streams for device-side point-to-point operations
        for (int _i = 0; _i < 4; ++_i) deviceStreamCreate(&d_reqs[_i]);
#endif
        // begin host MPI communication
        // 信息传递顺序为：传递east/接收west，传递west/接收east，传递north/接收south，传递south/接收north，传递up/接收down，传递down/接收up
        // 传递send_dir方向即将传输的Ghost粒子的数目
        MPI_Isend(&h_num_ghost_dir[send_dir], sizeof(int), MPI_BYTE,
                  send_neighbor, 0, CONTROLLER::pp_comm, &reqs[0]);
        // 接收recv_dir方向即将接收的Ghost粒子数目
        MPI_Irecv(&h_num_ghost_dir_re[recv_dir], sizeof(int), MPI_BYTE,
                  recv_neighbor, 0, CONTROLLER::pp_comm, &reqs[1]);
        MPI_Waitall(2, reqs, status);
        // end host MPI communication
        int start_idx = atom_numbers + ghost_numbers;
        // begin device MPI communication
        D_MPI_GroupStart();
        // 向send_dir方向传输的Ghost粒子的id
        D_MPI_Isend(ghost_buffer, sizeof(int) * h_num_ghost_dir[send_dir],
                    D_MPI_BYTE, send_neighbor, 1, CONTROLLER::d_pp_comm,
                    d_reqs[0]);
        // 接收recv_dir方向Ghost粒子的id
        D_MPI_Irecv(&atom_local[start_idx],
                    sizeof(int) * h_num_ghost_dir_re[recv_dir], D_MPI_BYTE,
                    recv_neighbor, 1, CONTROLLER::d_pp_comm, d_reqs[1]);

        // 向send_dir方向传输的Ghost粒子的坐标
        D_MPI_Isend(crd_buffer, sizeof(VECTOR) * h_num_ghost_dir[send_dir],
                    D_MPI_BYTE, send_neighbor, 2, CONTROLLER::d_pp_comm,
                    d_reqs[2]);
        // 接收recv_dir方向Ghost粒子的坐标
        D_MPI_Irecv(&crd[start_idx],
                    sizeof(VECTOR) * h_num_ghost_dir_re[recv_dir], D_MPI_BYTE,
                    recv_neighbor, 2, CONTROLLER::d_pp_comm, d_reqs[3]);
        D_MPI_GroupEnd();
        // wait for device-side operations to complete
        D_MPI_Waitall(4, d_reqs, d_status);
        //  end device MPI communication
        ghost_numbers += h_num_ghost_dir_re[recv_dir];  // 更新ghost粒子数目

        // begin host MPI communication
        // 传递/接收Ghost残基的数目信息
        int res_start_idx = res_numbers + ghost_res_numbers;
        MPI_Isend(&h_num_ghost_res_dir[send_dir], sizeof(int), MPI_BYTE,
                  send_neighbor, 3, CONTROLLER::pp_comm, &reqs[0]);
        MPI_Irecv(&h_num_ghost_res_dir_re[recv_dir], sizeof(int), MPI_BYTE,
                  recv_neighbor, 3, CONTROLLER::pp_comm, &reqs[1]);
        MPI_Waitall(2, reqs, status);
        // end host MPI communication

        // begin device MPI communication
        // 向send_dir方向传输的Ghost粒子的plan
        D_MPI_GroupStart();
        D_MPI_Isend(
            plan_buffer, sizeof(unsigned int) * h_num_ghost_res_dir[send_dir],
            D_MPI_BYTE, send_neighbor, 4, CONTROLLER::d_pp_comm, d_reqs[0]);
        // 接收recv_dir方向Ghost粒子的plan
        D_MPI_Irecv(&plan[res_start_idx],
                    sizeof(unsigned int) * h_num_ghost_res_dir_re[recv_dir],
                    D_MPI_BYTE, recv_neighbor, 4, CONTROLLER::d_pp_comm,
                    d_reqs[1]);

        D_MPI_Isend(res_len_buffer, h_num_ghost_res_dir[send_dir] * sizeof(int),
                    D_MPI_BYTE, send_neighbor, 5, CONTROLLER::d_pp_comm,
                    d_reqs[2]);
        D_MPI_Irecv(&res_len[res_start_idx],
                    h_num_ghost_res_dir_re[recv_dir] * sizeof(int), D_MPI_BYTE,
                    recv_neighbor, 5, CONTROLLER::d_pp_comm, d_reqs[3]);
        D_MPI_GroupEnd();
        // wait for device-side operations to complete
        D_MPI_Waitall(4, d_reqs, d_status);
        //  end device MPI communication
#ifdef USE_GPU
        // destroy device streams created for this exchange
        for (int _i = 0; _i < 4; ++_i) deviceStreamDestroy(d_reqs[_i]);
#endif
        ghost_res_numbers +=
            h_num_ghost_res_dir_re[recv_dir];  // 更新ghost残基数目

        Launch_Device_Kernel(refresh_res, 1, 1, 0, NULL, res_numbers,
                             ghost_res_numbers, res_start, res_len, start_idx,
                             res_start_idx);
    }

    int total_atom_numbers = atom_numbers + ghost_numbers;
    Launch_Device_Kernel(save_ghost_id, (max_atom_numbers + 255) / 256, 256, 0,
                         NULL, atom_numbers, total_atom_numbers, atom_local,
                         atom_local_id, d_charge, md_info->d_charge);
#endif
}

static __global__ void device_get_excluded(
    int* d_excluded_numbers, int* d_excluded_list_start, int* d_excluded_list,
    int* atom_local, int* atom_local_id, int atom_numbers,
    int* md_info_nb_d_excluded_numbers, int* md_info_nb_d_excluded_list_start,
    int* md_info_nb_d_excluded_list)
{
    int count = 0;
#ifdef USE_GPU
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 0) return;
#endif
    for (int i = 0; i < atom_numbers; ++i)
    {
        int atom_i = atom_local[i];
        d_excluded_numbers[i] = md_info_nb_d_excluded_numbers[atom_i];
        d_excluded_list_start[i] = count;
        if (d_excluded_numbers[i] > 0)
        {
            int start_global = md_info_nb_d_excluded_list_start[atom_i];
            int end_global =
                start_global + md_info_nb_d_excluded_numbers[atom_i];
            for (int j = start_global; j < end_global; ++j)
            {
                int atom_j = md_info_nb_d_excluded_list[j];
                d_excluded_list[count] = atom_local_id[atom_j];
                count++;
            }
        }
    }
}

void DOMAIN_INFORMATION::Get_Excluded(CONTROLLER* controller,
                                      MD_INFORMATION* md_info)
{
    Launch_Device_Kernel(
        device_get_excluded, 1, 1, 0, NULL, d_excluded_numbers,
        d_excluded_list_start, d_excluded_list, atom_local, atom_local_id,
        atom_numbers, md_info->nb.d_excluded_numbers,
        md_info->nb.d_excluded_list_start, md_info->nb.d_excluded_list);
}

static __global__ void set_crd_buffer(int ghost_number_dir,
                                      int* ghost_id_buffer, VECTOR* crd,
                                      VECTOR* crd_buffer)
{
    SIMPLE_DEVICE_FOR(idx, ghost_number_dir)
    {
        int atom_i = ghost_id_buffer[idx];
        crd_buffer[idx] = crd[atom_i];
    }
}

void DOMAIN_INFORMATION::Update_Ghost(CONTROLLER* controller)
{
    if (controller->PP_MPI_size == 1)
    {
        return;
    }
#ifdef USE_MPI
    int start_idx = atom_numbers;
    for (int dir = 0; dir < 6; ++dir)
    {
        if (!h_neighbor_num[dir]) continue;
        // dir方向上需要更新的Ghost粒子的坐标
        deviceMemcpy(&h_num_ghost_dir[dir], &d_num_ghost_dir[dir], sizeof(int),
                     deviceMemcpyDeviceToHost);
        VECTOR* crd_buffer;
        int ghost_number_dir = h_num_ghost_dir[dir];
        Device_Malloc_Safely((void**)&crd_buffer,
                             sizeof(VECTOR) * ghost_number_dir);
        int* ghost_id_buffer = d_num_ghost_dir_id + dir * max_atom_numbers;
        Launch_Device_Kernel(set_crd_buffer, (ghost_number_dir + 255) / 256,
                             256, 0, NULL, ghost_number_dir, ghost_id_buffer,
                             crd, crd_buffer);
        int send_dir = dir;
        int recv_dir = dir % 2 ? (dir - 1) : (dir + 1);
        int send_neighbor = h_neighbor_dir[send_dir][0];
        int recv_neighbor = h_neighbor_dir[recv_dir][0];

        // 信息传递顺序为：传递east/接收west，传递west/接收east，传递north/接收south，传递south/接收south，传递up/接收down，传递down/接收up
        D_MPI_Request d_reqs[2];
        D_MPI_Status d_status[2];
#ifdef USE_GPU
        // create device streams for device-side point-to-point operations
        for (int _i = 0; _i < 2; ++_i) deviceStreamCreate(&d_reqs[_i]);
#endif
        D_MPI_GroupStart();
        D_MPI_Isend(crd_buffer, sizeof(VECTOR) * h_num_ghost_dir[send_dir],
                    D_MPI_BYTE, send_neighbor, 0, CONTROLLER::d_pp_comm,
                    d_reqs[0]);
        D_MPI_Irecv(&crd[start_idx],
                    sizeof(VECTOR) * h_num_ghost_dir_re[recv_dir], D_MPI_BYTE,
                    recv_neighbor, 0, CONTROLLER::d_pp_comm, d_reqs[1]);
        D_MPI_GroupEnd();
        D_MPI_Waitall(2, d_reqs, d_status);
#ifdef USE_GPU
        // destroy device streams created for this exchange
        for (int _i = 0; _i < 2; ++_i) deviceStreamDestroy(d_reqs[_i]);
#endif
        start_idx += h_num_ghost_dir_re[recv_dir];
    }
#endif
}

static __global__ void set_frc_buffer(int ghost_number_dir,
                                      int* ghost_id_buffer, VECTOR* frc,
                                      VECTOR* frc_buffer)
{
    SIMPLE_DEVICE_FOR(idx, ghost_number_dir)
    {
        int atom_i = ghost_id_buffer[idx];
        frc[atom_i] = frc[atom_i] + frc_buffer[idx];
    }
}

static __global__ void add_frc(int atom_numbers, VECTOR* frc, VECTOR* frc_)
{
    SIMPLE_DEVICE_FOR(idx, atom_numbers) { frc[idx] = frc[idx] + frc_[idx]; }
}

static __global__ void sync_local_charge_from_global_charge_device(
    int local_atom_numbers, const int* atom_local, const float* global_charge,
    float* local_charge)
{
#ifdef USE_GPU
    SIMPLE_DEVICE_FOR(i, local_atom_numbers)
#else
#pragma omp parallel for
    for (int i = 0; i < local_atom_numbers; i++)
#endif
    {
        int global_id = atom_local[i];
        local_charge[i] = global_charge[global_id];
    }
}

void DOMAIN_INFORMATION::Sync_Local_Charge_From_Global(
    const float* global_charge)
{
    if (atom_numbers <= 0 || atom_local == NULL || d_charge == NULL ||
        global_charge == NULL)
    {
        return;
    }
    Launch_Device_Kernel(
        sync_local_charge_from_global_charge_device,
        (atom_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, atom_numbers, atom_local,
        global_charge, d_charge);
}

// 似乎在pp进程已被弃用，pm进程还有同名函数
void DOMAIN_INFORMATION::Distribute_Ghost_Information(CONTROLLER* controller,
                                                      VECTOR* frc_)
{
    if (controller->PP_MPI_size == 1)
    {
        return;
    }
#ifdef USE_MPI
    // 这个函数用于将Ghost粒子的信息更新回各自的local区域，这里信息的传递需要与get_ghost和update_ghost相反，否则会遗漏力
    int start_idx = atom_numbers + ghost_numbers;
    for (int dir = 5; dir >= 0; --dir)
    {
        if (!h_neighbor_num[dir]) continue;
        deviceMemcpy(&h_num_ghost_dir[dir], &d_num_ghost_dir[dir], sizeof(int),
                     deviceMemcpyDeviceToHost);
        deviceMemcpy(&h_num_ghost_dir_re[dir], &d_num_ghost_dir_re[dir],
                     sizeof(int), deviceMemcpyDeviceToHost);
        // 注意，这里信息的流向与其它Ghost信息传递过程是相反的
        int recv_dir = dir;
        int send_dir = dir % 2 ? (dir - 1) : (dir + 1);
        int recv_neighbor = h_neighbor_dir[recv_dir][0];
        int send_neighbor = h_neighbor_dir[send_dir][0];
        start_idx -= h_num_ghost_dir_re[send_dir];
        D_MPI_Request d_reqs[2];
        D_MPI_Status d_status[2];
#ifdef USE_GPU
        for (int _i = 0; _i < 2; ++_i) deviceStreamCreate(&d_reqs[_i]);
#endif
        D_MPI_GroupStart();
        D_MPI_Isend(&frc_[start_idx],
                    h_num_ghost_dir_re[send_dir] * sizeof(VECTOR), D_MPI_BYTE,
                    send_neighbor, 0, CONTROLLER::d_pp_comm, d_reqs[0]);
        D_MPI_Irecv(frc_buffer, h_num_ghost_dir[recv_dir] * sizeof(VECTOR),
                    D_MPI_BYTE, recv_neighbor, 0, CONTROLLER::d_pp_comm,
                    d_reqs[1]);
        D_MPI_GroupEnd();
        D_MPI_Waitall(2, d_reqs, d_status);
#ifdef USE_GPU
        for (int _i = 0; _i < 2; ++_i) deviceStreamDestroy(d_reqs[_i]);
#endif
        int num_ghost_dir_recv = h_num_ghost_dir[recv_dir];
        int* num_ghost_dir_id_recv =
            d_num_ghost_dir_id + recv_dir * max_atom_numbers;
        // 将接收到的粒子受力更新到力的buffer中
        Launch_Device_Kernel(set_frc_buffer, (num_ghost_dir_recv + 255) / 256,
                             256, 0, NULL, num_ghost_dir_recv,
                             num_ghost_dir_id_recv, frc_, frc_buffer);
    }
    // 将力更新回本地粒子
    Launch_Device_Kernel(add_frc, (atom_numbers + 255) / 256, 256, 0, NULL,
                         atom_numbers, this->frc, frc_);
#endif
}

static __global__ void manage_buffer_and_rerange(
    int* pass_num_dir, int* pass_res_dir, int res_numbers, int* res_start,
    int* res_len, int* atom_local, char* atom_local_label, int* atom_local_id,
    int* atom_buffer, VECTOR* crd, VECTOR* vel, VECTOR* crd_buffer,
    VECTOR* vel_buffer, int* res_len_buffer, float* d_charge, float* d_mass,
    float* d_mass_inverse, VECTOR min_corner, VECTOR max_corner,
    LTMatrix3 rcell, int dir, VECTOR box_length)
{
#ifdef USE_GPU
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 0) return;
#endif
    for (int i = 0; i < res_numbers; ++i)
    {
        int atom_start_i = res_start[i];
        int res_len_i = res_len[i];
        VECTOR crd_i = crd[atom_start_i];
        VECTOR frac_max_corner = wisediv(max_corner, box_length);
        VECTOR frac_min_corner = wisediv(min_corner, box_length);
        VECTOR d_max = (crd_i * rcell - frac_max_corner);
        d_max = d_max - floorf(d_max + 0.5f);
        VECTOR d_min = (crd_i * rcell - frac_min_corner);
        d_min = d_min - floorf(d_min + 0.5f);
        if ((dir == 0 && d_max.x > 0) || (dir == 1 && d_min.x < 0) ||
            (dir == 2 && d_max.y > 0) || (dir == 3 && d_min.y < 0) ||
            (dir == 4 && d_max.z > 0) || (dir == 5 && d_min.z < 0))
        {
            for (int atom_i = atom_start_i; atom_i < (atom_start_i + res_len_i);
                 ++atom_i)
            {
                atom_buffer[pass_num_dir[0]] = atom_local[atom_i];
                atom_local_label[atom_local[atom_i]] = 0;
                atom_local_id[atom_local[atom_i]] = -1;
                crd_buffer[pass_num_dir[0]] = crd[atom_i];
                vel_buffer[pass_num_dir[0]] = vel[atom_i];
                pass_num_dir[0] += 1;
            }
            res_len_buffer[pass_res_dir[0]] = res_len_i;
            pass_res_dir[0] += 1;
        }
        else
        {
            for (int atom_i = atom_start_i; atom_i < (atom_start_i + res_len_i);
                 ++atom_i)
            {
                atom_local_id[atom_local[atom_i]] = atom_i - pass_num_dir[0];
                atom_local[atom_i - pass_num_dir[0]] = atom_local[atom_i];
                crd[atom_i - pass_num_dir[0]] = crd[atom_i];
                vel[atom_i - pass_num_dir[0]] = vel[atom_i];
                d_charge[atom_i - pass_num_dir[0]] = d_charge[atom_i];
                d_mass[atom_i - pass_num_dir[0]] = d_mass[atom_i];
                d_mass_inverse[atom_i - pass_num_dir[0]] =
                    d_mass_inverse[atom_i];
            }
            res_len[i - pass_res_dir[0]] = res_len[i];
            res_start[i - pass_res_dir[0]] =
                atom_start_i - pass_num_dir[0];  // 暂时不处理res_start
        }
    }
}

static __global__ void reset_local(int start_idx, int atom_numbers,
                                   int* atom_local, char* atom_local_label,
                                   int* atom_local_id, float* d_charge,
                                   float* d_mass, float* d_mass_inverse,
                                   float* md_info_d_charge,
                                   float* md_info_d_mass,
                                   float* md_info_d_mass_inverse)
{
    int sum = atom_numbers - start_idx;
    SIMPLE_DEVICE_FOR(j, sum)
    {
        int atom_j = atom_local[j + start_idx];
        d_charge[j + start_idx] = md_info_d_charge[atom_j];
        d_mass[j + start_idx] = md_info_d_mass[atom_j];
        d_mass_inverse[j + start_idx] = md_info_d_mass_inverse[atom_j];
        // 粒子label置为1
        atom_local_label[atom_j] = 1;
        atom_local_id[atom_j] = j + start_idx;
    }
}

static __global__ void update_res_start(int start_idx_res, int res_numbers,
                                        int* res_start, int* res_len,
                                        int* start_idx)
{
#ifdef USE_GPU
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 0) return;
#endif
    for (int j = start_idx_res; j < res_numbers; ++j)
    {
        res_start[j] = start_idx[0];
        start_idx[0] += res_len[j];
    }
}

void DOMAIN_INFORMATION::Exchange_Particles(CONTROLLER* controller,
                                            MD_INFORMATION* md_info)
{
    if (controller->PP_MPI_size == 1)
    {
        return;
    }
#ifdef USE_MPI
    // Ghost清零
    ghost_numbers = 0;
    // Ghost_res清零
    ghost_res_numbers = 0;
    for (int dir = 0; dir < 6; ++dir)
    {
        if (!h_neighbor_num[dir]) continue;
        // 传递/接收原子/残基的buffer
        int pass_num_dir = 0;
        int pass_res_dir = 0;
        int *d_pass_num_dir, *d_pass_res_dir;
        Device_Malloc_And_Copy_Safely((void**)&d_pass_num_dir, &pass_num_dir,
                                      sizeof(int));
        Device_Malloc_And_Copy_Safely((void**)&d_pass_res_dir, &pass_res_dir,
                                      sizeof(int));

        int recv_num_dir = 0;
        int recv_res_dir = 0;

        // buffer on device
        int* res_len_buffer;
        int* atom_buffer;
        VECTOR* crd_buffer;
        VECTOR* vel_buffer;
        Device_Malloc_Safely((void**)&res_len_buffer,
                             sizeof(int) * (res_numbers));
        Device_Malloc_Safely((void**)&atom_buffer,
                             sizeof(int) * (atom_numbers));
        Device_Malloc_Safely((void**)&crd_buffer,
                             sizeof(VECTOR) * (atom_numbers));
        Device_Malloc_Safely((void**)&vel_buffer,
                             sizeof(VECTOR) * (atom_numbers));

        Launch_Device_Kernel(manage_buffer_and_rerange, 1, 1, 0, NULL,
                             d_pass_num_dir, d_pass_res_dir, res_numbers,
                             res_start, res_len, atom_local, atom_local_label,
                             atom_local_id, atom_buffer, crd, vel, crd_buffer,
                             vel_buffer, res_len_buffer, d_charge, d_mass,
                             d_mass_inverse, min_corner, max_corner,
                             md_info->pbc.rcell, dir, md_info->sys.box_length);

        deviceMemcpy(&pass_num_dir, d_pass_num_dir, sizeof(int),
                     deviceMemcpyDeviceToHost);
        deviceMemcpy(&pass_res_dir, d_pass_res_dir, sizeof(int),
                     deviceMemcpyDeviceToHost);
#ifndef USE_CPU
        deviceFree(d_pass_num_dir);
        deviceFree(d_pass_res_dir);
#endif
        atom_numbers -= pass_num_dir;
        res_numbers -= pass_res_dir;
        // controller->MPI_printf("before rank %d, pass_num = %d, recv_num =
        // %d\n", controller->MPI_rank, pass_num_dir, recv_num_dir);
        MPI_Request reqs[2];
        MPI_Status status[2];
        D_MPI_Request d_reqs[6];
        D_MPI_Status d_status[6];
#ifdef USE_GPU
        // create device streams for device-side point-to-point operations
        for (int _i = 0; _i < 6; ++_i) deviceStreamCreate(&d_reqs[_i]);
#endif

        int send_dir = dir;
        int recv_dir = dir % 2 ? (dir - 1) : (dir + 1);
        int send_neighbor = h_neighbor_dir[send_dir][0];
        int recv_neighbor = h_neighbor_dir[recv_dir][0];
        MPI_Isend(&pass_num_dir, sizeof(int), MPI_BYTE, send_neighbor, 0,
                  CONTROLLER::pp_comm, &reqs[0]);
        MPI_Irecv(&recv_num_dir, sizeof(int), MPI_BYTE, recv_neighbor, 0,
                  CONTROLLER::pp_comm, &reqs[1]);
        MPI_Waitall(2, reqs, status);

        // start atom device buffer communication
        int start_idx = atom_numbers;
        D_MPI_GroupStart();
        D_MPI_Isend(atom_buffer, pass_num_dir * sizeof(int), D_MPI_BYTE,
                    send_neighbor, 1, CONTROLLER::d_pp_comm, d_reqs[0]);
        D_MPI_Irecv(&atom_local[start_idx], recv_num_dir * sizeof(int),
                    D_MPI_BYTE, recv_neighbor, 1, CONTROLLER::d_pp_comm,
                    d_reqs[1]);

        D_MPI_Isend(crd_buffer, pass_num_dir * sizeof(VECTOR), D_MPI_BYTE,
                    send_neighbor, 2, CONTROLLER::d_pp_comm, d_reqs[2]);
        D_MPI_Irecv(&crd[start_idx], recv_num_dir * sizeof(VECTOR), D_MPI_BYTE,
                    recv_neighbor, 2, CONTROLLER::d_pp_comm, d_reqs[3]);

        D_MPI_Isend(vel_buffer, pass_num_dir * sizeof(VECTOR), D_MPI_BYTE,
                    send_neighbor, 3, CONTROLLER::d_pp_comm, d_reqs[4]);
        D_MPI_Irecv(&vel[start_idx], recv_num_dir * sizeof(VECTOR), D_MPI_BYTE,
                    recv_neighbor, 3, CONTROLLER::d_pp_comm, d_reqs[5]);
        D_MPI_GroupEnd();
        D_MPI_Waitall(6, d_reqs, d_status);
        // end device buffer communication
        atom_numbers += recv_num_dir;

        // 残基信息传递
        MPI_Isend(&pass_res_dir, sizeof(int), MPI_BYTE, send_neighbor, 4,
                  CONTROLLER::pp_comm, &reqs[0]);
        MPI_Irecv(&recv_res_dir, sizeof(int), MPI_BYTE, recv_neighbor, 4,
                  CONTROLLER::pp_comm, &reqs[1]);
        MPI_Waitall(2, reqs, status);

        int start_idx_res = res_numbers;
        D_MPI_GroupStart();
        D_MPI_Isend(res_len_buffer, pass_res_dir * sizeof(int), D_MPI_BYTE,
                    send_neighbor, 5, CONTROLLER::d_pp_comm, d_reqs[0]);
        D_MPI_Irecv(&res_len[start_idx_res], recv_res_dir * sizeof(int),
                    D_MPI_BYTE, recv_neighbor, 5, CONTROLLER::d_pp_comm,
                    d_reqs[1]);
        D_MPI_GroupEnd();
        D_MPI_Waitall(2, d_reqs, d_status);
#ifdef USE_GPU
        // destroy device streams created for this exchange
        for (int _i = 0; _i < 6; ++_i) deviceStreamDestroy(d_reqs[_i]);
#endif
        res_numbers += recv_res_dir;

        Launch_Device_Kernel(
            reset_local, (atom_numbers - start_idx + 255) / 256, 256, 0, NULL,
            start_idx, atom_numbers, atom_local, atom_local_label,
            atom_local_id, d_charge, d_mass, d_mass_inverse, md_info->d_charge,
            md_info->d_mass, md_info->d_mass_inverse);

        int* d_start_idx;
        Device_Malloc_And_Copy_Safely((void**)&d_start_idx, &start_idx,
                                      sizeof(int));
        Launch_Device_Kernel(update_res_start, 1, 1, 0, NULL, start_idx_res,
                             res_numbers, res_start, res_len, d_start_idx);
        deviceMemcpy(&start_idx, d_start_idx, sizeof(int),
                     deviceMemcpyDeviceToHost);
#ifndef USE_CPU
        deviceFree(d_start_idx);
#endif
    }
#endif
    // controller->printf("end_exchange_particles\n");
}

void DOMAIN_INFORMATION::Reset_Force_and_Virial(MD_INFORMATION* md_info)
{
    // frc, d_virial and d_energy are allocated (on CPU path) with size
    // tmp_frc_size * max_omp_threads (see Get_Atoms). Use that allocated
    // capacity when resetting to avoid writing past the end of the buffer.
#ifdef USE_CPU
    extern int max_omp_threads;
    int allocated_atoms = tmp_frc_size * max_omp_threads;
    deviceMemset(frc, 0, sizeof(VECTOR) * allocated_atoms);
    deviceMemset(d_virial, 0, sizeof(LTMatrix3) * allocated_atoms);
    deviceMemset(d_energy, 0, sizeof(float) * allocated_atoms);
#else
    deviceMemset(frc, 0, sizeof(VECTOR) * max_atom_numbers);
    deviceMemset(d_virial, 0, sizeof(LTMatrix3) * max_atom_numbers);
    deviceMemset(d_energy, 0, sizeof(float) * max_atom_numbers);
#endif
}

void DOMAIN_INFORMATION::Free_Buffer()
{
    atom_numbers = 0;
    res_numbers = 0;
    ghost_numbers = 0;
    ghost_res_numbers = 0;
    max_atom_numbers = 0;
    max_res_numbers = 0;
    tmp_frc_size = 0;

    Free_Single_Device_Pointer((void**)&d_atom_numbers);
    Free_Single_Device_Pointer((void**)&d_res_numbers);
    Free_Single_Device_Pointer((void**)&d_ghost_numbers);
    Free_Single_Device_Pointer((void**)&d_ghost_res_numbers);

    Free_Single_Device_Pointer((void**)&atom_local);
    Free_Single_Device_Pointer((void**)&atom_local_id);
    Free_Single_Device_Pointer((void**)&atom_local_label);

    Free_Single_Device_Pointer((void**)&res_len);
    Free_Single_Device_Pointer((void**)&res_start);
    Free_Single_Device_Pointer((void**)&d_center_of_mass);

    Free_Single_Device_Pointer((void**)&crd);
    Free_Single_Device_Pointer((void**)&vel);
    Free_Single_Device_Pointer((void**)&acc);
    Free_Single_Device_Pointer((void**)&frc);
    Free_Single_Device_Pointer((void**)&frc_buffer);
    Free_Single_Device_Pointer((void**)&d_mass);
    Free_Single_Device_Pointer((void**)&d_mass_inverse);
    Free_Single_Device_Pointer((void**)&d_charge);

    Free_Single_Device_Pointer((void**)&d_ek);
    Free_Single_Device_Pointer((void**)&d_ek_local);
    Free_Single_Device_Pointer((void**)&d_ek_total);
    Free_Single_Device_Pointer((void**)&d_sum_ene_local);
    Free_Single_Device_Pointer((void**)&d_sum_ene_total);

    Free_Single_Device_Pointer((void**)&d_virial);
    Free_Single_Device_Pointer((void**)&d_energy);

    Free_Single_Device_Pointer((void**)&d_excluded_list_start);
    Free_Single_Device_Pointer((void**)&d_excluded_numbers);
    Free_Single_Device_Pointer((void**)&d_excluded_list);

    Free_Single_Device_Pointer((void**)&frc_buffer);
    free(h_num_ghost_dir_id);
    Free_Single_Device_Pointer((void**)&d_num_ghost_dir);
    Free_Single_Device_Pointer((void**)&d_num_ghost_dir_id);
    Free_Single_Device_Pointer((void**)&d_num_ghost_dir_re);
    Free_Single_Device_Pointer((void**)&d_num_ghost_res_dir);
    Free_Single_Device_Pointer((void**)&d_num_ghost_res_dir_re);
}

// Ek calculation
static __global__ void MD_Atom_Ek(const int atom_numbers, float* ek,
                                  const VECTOR* atom_vel,
                                  const float* atom_mass)
{
    SIMPLE_DEVICE_FOR(atom_i, atom_numbers)
    {
        VECTOR v = atom_vel[atom_i];
        ek[atom_i] = 0.5 * v * v * atom_mass[atom_i];
    }
}

void DOMAIN_INFORMATION::Get_Ek_and_Temperature(CONTROLLER* controller,
                                                MD_INFORMATION* md_info)
{
    if (CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size)
    {
        Launch_Device_Kernel(
            MD_Atom_Ek,
            (atom_numbers + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, atom_numbers, d_ek, vel,
            d_mass);

        Sum_Of_List(d_ek, d_ek_local, atom_numbers);
    }
    else
    {
        deviceMemset(d_ek_local, 0, sizeof(float));
    }

    // 如果只有一个PP进程，直接赋值，不需要MPI通信
    if (CONTROLLER::PP_MPI_size == 1)
    {
        deviceMemcpy(d_ek_total, d_ek_local, sizeof(float),
                     deviceMemcpyDeviceToDevice);
    }
    else
    {
#ifdef USE_MPI
        D_MPI_Allreduce(d_ek_local, d_ek_total, 1, D_MPI_FLOAT, D_MPI_SUM,
                        CONTROLLER::D_MPI_COMM_WORLD, temp_stream);
        D_MPI_Barrier(CONTROLLER::D_MPI_COMM_WORLD, temp_stream);
#endif
    }
    deviceMemcpy(&h_ek_total, d_ek_total, sizeof(float),
                 deviceMemcpyDeviceToHost);
    temperature = 2.0f * h_ek_total / (md_info->sys.freedom * CONSTANT_kB);
    md_info->sys.h_temperature = temperature;
}

void DOMAIN_INFORMATION::Create_Stream() { deviceStreamCreate(&temp_stream); }
void DOMAIN_INFORMATION::Destroy_Stream() { deviceStreamDestroy(temp_stream); }

// ------------ barostat related functions -----------------------

static void Scale_Positions_Device(const LTMatrix3 g, VECTOR* crd, float dt)
{
    VECTOR r_dash;
    r_dash.x = crd[0].x +
               dt * (crd[0].x * g.a11 + crd[0].y * g.a21 + crd[0].z * g.a31);
    r_dash.y = crd[0].y + dt * (crd[0].y * g.a22 + crd[0].z * g.a32);
    r_dash.z = crd[0].z + dt * crd[0].z * g.a33;
    crd[0] = r_dash;
}

void DOMAIN_INFORMATION::Update_Box(LTMatrix3 g, float dt)
{
    Scale_Positions_Device(g, &min_corner, dt);
    Scale_Positions_Device(g, &max_corner, dt);
}

// 最终势能需要广播到所有进程
void DOMAIN_INFORMATION::Get_Potential(CONTROLLER* controller,
                                       MD_INFORMATION* md_info)
{
    if (md_info->need_potential)
    {
        if (CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size)
        {
            Sum_Of_List(d_energy, d_sum_ene_local, atom_numbers);
        }
        else
        {
            Sum_Of_List(md_info->d_atom_energy, d_sum_ene_local,
                        md_info->atom_numbers);
        }
#ifdef USE_MPI
        D_MPI_Allreduce(d_sum_ene_local, d_sum_ene_total, 1, D_MPI_FLOAT,
                        D_MPI_SUM, CONTROLLER::D_MPI_COMM_WORLD, temp_stream);
        D_MPI_Barrier(CONTROLLER::D_MPI_COMM_WORLD, temp_stream);
#else
        deviceMemcpy(d_sum_ene_total, d_sum_ene_local, sizeof(float),
                     deviceMemcpyDeviceToDevice);
#endif
        deviceMemcpy(&h_sum_ene_total, d_sum_ene_total, sizeof(float),
                     deviceMemcpyDeviceToHost);
        deviceMemcpy(md_info->sys.d_potential, d_sum_ene_total, sizeof(float),
                     deviceMemcpyDeviceToDevice);
        md_info->sys.h_potential = h_sum_ene_total;
    }
}

static __global__ void Get_Origin(const int res_numbers, const int* res_start,
                                  const int* res_len, const VECTOR* crd,
                                  VECTOR* com)
{
#ifdef USE_GPU
    int residue_i = blockDim.x * blockIdx.x + threadIdx.x;
    if (residue_i < res_numbers)
#else
#pragma omp parallel for
    for (int residue_i = 0; residue_i < res_numbers; ++residue_i)
#endif
    {
        VECTOR origin = {FLT_MAX, FLT_MAX, FLT_MAX};
        for (int i = 0; i < res_len[residue_i]; ++i)
        {
            int atom_i = res_start[residue_i] + i;
            {
                VECTOR crd_i = crd[atom_i];
                origin.x = fminf(origin.x, crd_i.x);
                origin.y = fminf(origin.y, crd_i.y);
                origin.z = fminf(origin.z, crd_i.z);
            }
            com[residue_i] = origin;
        }
    }
}

static __global__ void Map_Origin(const int res_numbers, const int* res_start,
                                  const int* res_len, const VECTOR* com,
                                  const LTMatrix3 g, const float dt,
                                  VECTOR* crd)
{
#ifdef USE_GPU
    int residue_i = blockDim.x * blockIdx.x + threadIdx.x;
    if (residue_i < res_numbers)
#else
#pragma omp parallel for
    for (int residue_i = 0; residue_i < res_numbers; ++residue_i)
#endif
    {
        VECTOR trans_vec;
        VECTOR r = com[residue_i];
        trans_vec.x = dt * (r.x * g.a11 + r.y * g.a21 + r.z * g.a31);
        trans_vec.y = dt * (r.y * g.a22 + r.z * g.a32);
        trans_vec.z = dt * r.z * g.a33;
        for (int i = 0; i < res_len[residue_i]; ++i)
        {
            int atom_i = res_start[residue_i] + i;
            crd[atom_i] = crd[atom_i] + trans_vec;
        }
    }
}

void DOMAIN_INFORMATION::Res_Crd_Map(LTMatrix3 g, float dt)
{
    Launch_Device_Kernel(Get_Origin,
                         (res_numbers + CONTROLLER::device_max_thread - 1) /
                             CONTROLLER::device_max_thread,
                         CONTROLLER::device_max_thread, 0, NULL, res_numbers,
                         res_start, res_len, crd, d_center_of_mass);
    Launch_Device_Kernel(Map_Origin,
                         (res_numbers + CONTROLLER::device_max_thread - 1) /
                             CONTROLLER::device_max_thread,
                         CONTROLLER::device_max_thread, 0, NULL, res_numbers,
                         res_start, res_len, d_center_of_mass, g, dt, crd);
}

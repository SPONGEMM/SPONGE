#include "neighbor_list.h"

#define MAX_GRID_NEIGHBORS 192

static __global__ void Find_Neighor_Grids_Device(
    int grid_numbers, int* neighbor_grid_numbers, int* neighbor_grids, int Nx,
    int Ny, int Nz, float grid_length, LTMatrix3 cell, LTMatrix3 rcell)
{
    SIMPLE_DEVICE_FOR(grid_i, grid_numbers)
    {
        int NyNz = Ny * Nz;
        int local = 0;
        int dx, dy, dz;
        int* neighbor_i = neighbor_grids + MAX_GRID_NEIGHBORS * grid_i;
        INT_VECTOR grid_i_xyz = {grid_i / NyNz, (grid_i % NyNz) / Nz,
                                 grid_i % Nz};
        INT_VECTOR grid_j_xyz, temp_delta;
        int dxy = cell.a21 / grid_length;
        int dxz = cell.a31 / grid_length;
        int dyz = cell.a32 / grid_length;
        int cx, cy;
        for (int grid_j = 0; grid_j < grid_numbers; grid_j += 1)
        {
            grid_j_xyz = {grid_j / NyNz, (grid_j % NyNz) / Nz, grid_j % Nz};
            for (int k = 0; k < 27; k += 1)
            {
                dx = k / 9 - 1;
                dy = (k % 9) / 3 - 1;
                dz = k % 3 - 1;
                temp_delta = {
                    grid_i_xyz.int_x + dx * Nx + dy * dxy + dz * dxz -
                        grid_j_xyz.int_x,
                    grid_i_xyz.int_y + dy * Ny + dz * dyz - grid_j_xyz.int_y,
                    grid_i_xyz.int_z + dz * Nz - grid_j_xyz.int_z};
                cx = dy * cell.a21 + dz * cell.a31 == 0 ? -2 : -3;
                cy = dz * cell.a32 == 0 ? -2 : -3;
                if (temp_delta.int_x <= 2 && temp_delta.int_x >= cx &&
                    temp_delta.int_y <= 2 && temp_delta.int_y >= cy &&
                    temp_delta.int_z <= 2 && temp_delta.int_z >= -2)
                {
                    neighbor_i[local] = grid_j;
                    local += 1;
                    break;
                }
            }
        }
        neighbor_grid_numbers[grid_i] = local;
    }
}

void NEIGHBOR_LIST::GRIDS::Initial(CONTROLLER* controller,
                                   int max_atom_in_grid_numbers,
                                   int max_ghost_in_grid_numbers,
                                   LTMatrix3 cell, LTMatrix3 rcell,
                                   float grid_length)
{
    controller->printf("    initializing grids\n");
    Nx = floorf(cell.a11 / grid_length);
    Ny = floorf(cell.a22 / grid_length);
    Nz = floorf(cell.a33 / grid_length);
    controller->printf("        Nx: %d        Ny: %d        Nz: %d\n", Nx, Ny,
                       Nz);
    controller->printf("        Max number of atoms in one grid: %d\n",
                       max_atom_in_grid_numbers);
    controller->printf("        Max number of ghosts in one grid: %d\n",
                       max_ghost_in_grid_numbers);
    grid_numbers = Nx * Ny * Nz;
    Malloc_Safely((void**)&h_neighbor_grid_numbers, sizeof(int) * grid_numbers);
    Malloc_Safely((void**)&h_neighbor_grids,
                  sizeof(int) * grid_numbers * MAX_GRID_NEIGHBORS);
    Malloc_Safely((void**)&h_grid_atoms,
                  sizeof(int) * grid_numbers * max_atom_in_grid_numbers);
    Malloc_Safely((void**)&h_grid_atom_numbers, sizeof(int) * grid_numbers);

    Malloc_Safely((void**)&h_grid_ghosts,
                  sizeof(int) * grid_numbers * max_ghost_in_grid_numbers);
    Malloc_Safely((void**)&h_grid_ghost_numbers, sizeof(int) * grid_numbers);
    memset(h_grid_ghost_numbers, 0, sizeof(int) * grid_numbers);
    Device_Malloc_And_Copy_Safely(
        (void**)&d_grid_ghosts, h_grid_ghosts,
        sizeof(int) * grid_numbers * max_ghost_in_grid_numbers);
    Device_Malloc_And_Copy_Safely((void**)&d_grid_ghost_numbers,
                                  h_grid_ghost_numbers,
                                  sizeof(int) * grid_numbers);
    Device_Malloc_Safely(
        (void**)&d_grid_ghost_crd,
        sizeof(VECTOR) * grid_numbers * max_ghost_in_grid_numbers);

    Device_Malloc_And_Copy_Safely((void**)&d_neighbor_grid_numbers,
                                  h_neighbor_grid_numbers,
                                  sizeof(int) * grid_numbers);
    Device_Malloc_And_Copy_Safely(
        (void**)&d_neighbor_grids, h_neighbor_grids,
        sizeof(int) * grid_numbers * MAX_GRID_NEIGHBORS);
    Device_Malloc_And_Copy_Safely(
        (void**)&d_grid_atoms, h_grid_atoms,
        sizeof(int) * grid_numbers * max_atom_in_grid_numbers);
    Device_Malloc_Safely(
        (void**)&d_grid_atom_crd,
        sizeof(VECTOR) * grid_numbers * max_atom_in_grid_numbers);
    Device_Malloc_And_Copy_Safely((void**)&d_grid_atom_numbers,
                                  h_grid_atom_numbers,
                                  sizeof(int) * grid_numbers);

    Launch_Device_Kernel(Find_Neighor_Grids_Device,
                         (grid_numbers + CONTROLLER::device_max_thread - 1) /
                             CONTROLLER::device_max_thread,
                         CONTROLLER::device_max_thread, 0, NULL, grid_numbers,
                         d_neighbor_grid_numbers, d_neighbor_grids, Nx, Ny, Nz,
                         grid_length, cell, rcell);
}

void NEIGHBOR_LIST::GRIDS::Clear()
{
    Free_Host_And_Device_Pointer((void**)&h_neighbor_grid_numbers,
                                 (void**)&d_neighbor_grid_numbers);
    Free_Host_And_Device_Pointer((void**)&h_neighbor_grids,
                                 (void**)&d_neighbor_grids);
    Free_Host_And_Device_Pointer((void**)&h_grid_atoms, (void**)&d_grid_atoms);
    Free_Host_And_Device_Pointer((void**)&h_grid_atom_numbers,
                                 (void**)&d_grid_atom_numbers);
    Free_Host_And_Device_Pointer((void**)&h_grid_ghosts,
                                 (void**)&d_grid_ghosts);
    Free_Host_And_Device_Pointer((void**)&h_grid_ghost_numbers,
                                 (void**)&d_grid_ghost_numbers);
    Free_Single_Device_Pointer((void**)&d_grid_ghost_crd);
    Free_Single_Device_Pointer((void**)&d_grid_atom_crd);
}

void NEIGHBOR_LIST::UPDATOR::Initial(CONTROLLER* controller, int atom_numbers)
{
    controller->printf("    initializing updator\n");
    time_recorder = controller->Get_Time_Recorder("neighbor searching");
    refresh_interval = 0;
    if (controller->Command_Exist("neighbor_list", "refresh_interval"))
    {
        controller->Check_Int("neighbor_list", "refresh_interval",
                              "NEIGHBOR_LIST::Initial");
        refresh_interval =
            atoi(controller->Command("neighbor_list", "refresh_interval"));
    }
    controller->printf(
        "        The interval to refresh the neighbor list: %d\n",
        refresh_interval);

    h_need_update = 1;
    if (refresh_interval <= 0)
    {
        skin_permit = 0.5f;
        if (controller->Command_Exist("neighbor_list", "skin_permit"))
        {
            controller->Check_Float("neighbor_list", "skin_permit",
                                    "NEIGHBOR_LIST::Initial");
            skin_permit =
                atof(controller->Command("neighbor_list", "skin_permit"));
        }
        controller->printf(
            "        The permit of the skin to refresh the neighbor list: %f\n",
            skin_permit);
        Device_Malloc_Safely((void**)&old_crd, sizeof(VECTOR) * atom_numbers);
    }
    Device_Malloc_And_Copy_Safely((void**)&d_need_update, &h_need_update,
                                  sizeof(int));
}

static __global__ void Check_Refresh(int h_need_update, int atom_numbers,
                                     VECTOR* crd, VECTOR* crd_old,
                                     LTMatrix3 cell, LTMatrix3 rcell,
                                     int* d_need_refresh, float permit_square)
{
    SIMPLE_DEVICE_FOR(tid, atom_numbers)
    {
        VECTOR dr =
            Get_Periodic_Displacement(crd[tid], crd_old[tid], cell, rcell);
        if (dr * dr > permit_square)
        {
            d_need_refresh[0] = 1;
        }
    }
}

void NEIGHBOR_LIST::UPDATOR::Check(int atom_numbers, float skin, VECTOR* crd,
                                   LTMatrix3 cell, LTMatrix3 rcell)
{
    if (atom_numbers <= 0) return;
    Launch_Device_Kernel(Check_Refresh,
                         (atom_numbers + CONTROLLER::device_max_thread - 1) /
                             CONTROLLER::device_max_thread,
                         CONTROLLER::device_max_thread, 0, NULL, h_need_update,
                         atom_numbers, crd, old_crd, cell, rcell, d_need_update,
                         skin * skin * skin_permit * skin_permit);
}

static __global__ void Clear_Bucket(const int* need, int grid_numbers,
                                    int* grid_atom_numbers,
                                    int* grid_ghost_numbers)
{
    if (need[0] == 0) return;
    SIMPLE_DEVICE_FOR(tid, grid_numbers)
    {
        grid_atom_numbers[tid] = 0;
        grid_ghost_numbers[tid] = 0;
    }
}

static __global__ void Put_Atom_In_Grids(
    const int* need, const int need_copy, const int* atom_local,
    const int atom_numbers, const int ghost_numbers, const int grid_numbers,
    const VECTOR* crd, VECTOR* old_crd, const LTMatrix3 cell,
    const LTMatrix3 rcell, const float grid_length, const int Nx, const int Ny,
    const int Nz, int* grid_atoms, int* grid_atom_numbers, VECTOR* grid_crd,
    ATOM_GROUP* nl, const int max_grid_atoms, int* neighbor_grid_overflow,
    int* grid_ghosts, int* grid_ghost_numbers, VECTOR* grid_ghost_crd,
    const int max_grid_ghosts, int* neighbor_grid_ghost_overflow)
{
    if (need[0] == 0) return;
    SIMPLE_DEVICE_FOR(tid, atom_numbers + ghost_numbers)
    {
        VECTOR local_crd = crd[tid];
        if (need_copy && tid < atom_numbers)
        {
            old_crd[tid] = local_crd;
        }

        float k3 = floorf(local_crd.z / cell.a33);
        local_crd.z -= k3 * cell.a33;
        local_crd.y -= k3 * cell.a32;
        float k2 = floorf(local_crd.y / cell.a22);
        local_crd.y -= k2 * cell.a22;
        local_crd.x -= k3 * cell.a31 + k2 * cell.a21;
        local_crd.x -= floorf(local_crd.x / cell.a11) * cell.a11;
        int nx = local_crd.x / grid_length;
        int ny = local_crd.y / grid_length;
        int nz = local_crd.z / grid_length;

        nx = nx < 0 ? 0 : (nx < Nx ? nx : Nx - 1);
        ny = ny < 0 ? 0 : (ny < Ny ? ny : Ny - 1);
        nz = nz < 0 ? 0 : (nz < Nz ? nz : Nz - 1);
        int grid_id = nx * Ny * Nz + ny * Nz + nz;
        if (grid_id < 0 || grid_id >= grid_numbers)
        {
            neighbor_grid_overflow[0] = 1;
        }
        else if (tid < atom_numbers)
        {
            int k1 = atomicAdd(grid_atom_numbers + grid_id, 1);
            if (k1 >= max_grid_atoms)
            {
                neighbor_grid_overflow[0] = 1;
                atomicAdd(grid_atom_numbers + grid_id, -1);
            }
            else
            {
                grid_atoms[max_grid_atoms * grid_id + k1] = tid;
                grid_crd[max_grid_atoms * grid_id + k1] = crd[tid];
                nl[tid].atom_numbers = 0;
                nl[tid].ghost_numbers = 0;
            }
        }
        else
        {
            int k1 = atomicAdd(grid_ghost_numbers + grid_id, 1);
            if (k1 >= max_grid_ghosts)
            {
                neighbor_grid_ghost_overflow[0] = 1;
                atomicAdd(grid_ghost_numbers + grid_id, -1);
            }
            else
            {
                grid_ghosts[max_grid_ghosts * grid_id + k1] = tid;
                grid_ghost_crd[max_grid_ghosts * grid_id + k1] = crd[tid];
            }
        }
    }
}

#ifdef USE_GPU

static __global__ void Find_Neighbors_Gridly(
    int* atom_local, int atom_numbers, const int* need, int grid_numbers,
    int* grid_neighbor_numbers, int* grid_neighbors, VECTOR* grid_crd,
    LTMatrix3 cell, LTMatrix3 rcell, int max_atom_numbers_in_grid,
    ATOM_GROUP* nl, float cutoff_skin_square, int* grid_atom_numbers,
    int* grid_atoms, int max_neighbor_numbers, int* neighbor_list_overflow,
    VECTOR* grid_ghost_crd, int max_ghost_numbers_in_grid,
    int* grid_ghost_numbers, int* grid_ghosts)
{
    if (need[0] == 0) return;
    extern __shared__ unsigned char shared_mem[];
    VECTOR* sh_crd = reinterpret_cast<VECTOR*>(shared_mem);
    int* sh_atoms = reinterpret_cast<int*>(sh_crd + max_atom_numbers_in_grid);

    const int lane = threadIdx.x & (warpSize - 1);
    int warps_per_block = blockDim.x / warpSize;
    if (warps_per_block == 0)
    {
        warps_per_block = 1;
    }
    const int warp_id = threadIdx.x / warpSize;
    const int lane_stride = blockDim.x < warpSize ? blockDim.x : warpSize;
    const int lane_index = blockDim.x < warpSize ? threadIdx.x : lane;

    unsigned int warp_mask = 0xFFFFFFFF;
    if (blockDim.x < warpSize)
    {
        warp_mask = (1U << blockDim.x) - 1;
    }

    for (int grid_i = blockIdx.x; grid_i < grid_numbers; grid_i += gridDim.x)
    {
        int atom_numbers_in_grid_i = grid_atom_numbers[grid_i];
        if (atom_numbers_in_grid_i == 0)
        {
            __syncthreads();
            continue;
        }

        int* bucket_i = grid_atoms + grid_i * max_atom_numbers_in_grid;
        VECTOR* grid_crd_i = grid_crd + grid_i * max_atom_numbers_in_grid;

        for (int idx = threadIdx.x; idx < atom_numbers_in_grid_i;
             idx += blockDim.x)
        {
            sh_atoms[idx] = bucket_i[idx];
            sh_crd[idx] = grid_crd_i[idx];
        }
        __syncthreads();

        int neighbor_count = grid_neighbor_numbers[grid_i];
        if (neighbor_count == 0)
        {
            __syncthreads();
            continue;
        }
        for (int jj = warp_id; jj < neighbor_count; jj += warps_per_block)
        {
            int grid_j = grid_neighbors[grid_i * MAX_GRID_NEIGHBORS + jj];
            int atom_numbers_in_grid_j = grid_atom_numbers[grid_j];
            if (atom_numbers_in_grid_j == 0)
            {
                continue;
            }

            int* bucket_j = grid_atoms + grid_j * max_atom_numbers_in_grid;
            VECTOR* grid_crd_j = grid_crd + grid_j * max_atom_numbers_in_grid;

            for (int j_base = 0; j_base < atom_numbers_in_grid_j;
                 j_base += lane_stride)
            {
                int j = j_base + lane_index;
                bool active = j < atom_numbers_in_grid_j;
                int atom_j = 0;
                int global_j = 0;
                VECTOR crd_j = {0, 0, 0};
                if (active)
                {
                    atom_j = bucket_j[j];
                    global_j = atom_local[atom_j];
                    crd_j = grid_crd_j[j];
                }

                for (int i = 0; i < atom_numbers_in_grid_i; ++i)
                {
                    int atom_i = sh_atoms[i];
                    int global_i = atom_local[atom_i];
                    bool is_neighbor = false;
                    if (active && global_j > global_i)
                    {
                        VECTOR dr = Get_Periodic_Displacement(sh_crd[i], crd_j,
                                                              cell, rcell);
                        float dr2 = dr * dr;
                        if (dr2 < cutoff_skin_square)
                        {
                            is_neighbor = true;
                        }
                    }

                    unsigned int mask = __ballot_sync(warp_mask, is_neighbor);
                    if (mask != 0)
                    {
                        int count = __popc(mask);
                        int base_slot = 0;
                        int leader_lane = __ffs(mask) - 1;
                        if (lane == leader_lane)
                        {
                            base_slot =
                                atomicAdd(&nl[atom_i].atom_numbers, count);
                            if (base_slot + count > max_neighbor_numbers)
                            {
                                atomicExch(neighbor_list_overflow, 1);
                            }
                        }
                        base_slot =
                            __shfl_sync(warp_mask, base_slot, leader_lane);

                        if (is_neighbor)
                        {
                            int rank = __popc(mask & ((1 << lane) - 1));
                            if (base_slot + rank < max_neighbor_numbers)
                            {
                                nl[atom_i].atom_serial[base_slot + rank] =
                                    atom_j;
                            }
                        }
                    }
                }
            }
        }

        __syncthreads();

        for (int jj = warp_id; jj < neighbor_count; jj += warps_per_block)
        {
            int grid_j = grid_neighbors[grid_i * MAX_GRID_NEIGHBORS + jj];
            int ghost_numbers_in_grid_j = grid_ghost_numbers[grid_j];
            if (ghost_numbers_in_grid_j == 0)
            {
                continue;
            }

            int* bucket_j = grid_ghosts + grid_j * max_ghost_numbers_in_grid;
            VECTOR* grid_ghost_crd_j =
                grid_ghost_crd + grid_j * max_ghost_numbers_in_grid;

            for (int j_base = 0; j_base < ghost_numbers_in_grid_j;
                 j_base += lane_stride)
            {
                int j = j_base + lane_index;
                bool active = j < ghost_numbers_in_grid_j;
                int atom_j = 0;
                VECTOR crd_j = {0, 0, 0};
                if (active)
                {
                    atom_j = bucket_j[j];
                    crd_j = grid_ghost_crd_j[j];
                }

                for (int i = 0; i < atom_numbers_in_grid_i; ++i)
                {
                    int atom_i = sh_atoms[i];
                    bool is_neighbor = false;
                    if (active)
                    {
                        VECTOR dr = Get_Periodic_Displacement(sh_crd[i], crd_j,
                                                              cell, rcell);
                        float dr2 = dr * dr;
                        if (dr2 < cutoff_skin_square)
                        {
                            is_neighbor = true;
                        }
                    }

                    unsigned int mask = __ballot_sync(warp_mask, is_neighbor);
                    if (mask != 0)
                    {
                        int count = __popc(mask);
                        int base_slot = 0;
                        int leader_lane = __ffs(mask) - 1;
                        if (lane == leader_lane)
                        {
                            base_slot =
                                atomicAdd(&nl[atom_i].atom_numbers, count);
                            atomicAdd(&nl[atom_i].ghost_numbers, count);
                            if (base_slot + count > max_neighbor_numbers)
                            {
                                atomicExch(neighbor_list_overflow, 1);
                            }
                        }
                        base_slot =
                            __shfl_sync(warp_mask, base_slot, leader_lane);

                        if (is_neighbor)
                        {
                            int rank = __popc(mask & ((1 << lane) - 1));
                            if (base_slot + rank < max_neighbor_numbers)
                            {
                                nl[atom_i].atom_serial[base_slot + rank] =
                                    atom_j;
                            }
                        }
                    }
                }
            }
        }

        __syncthreads();
    }
}

#else
static __global__ void Find_Neighbors_Gridly(
    int* atom_local, int atom_numbers, const int* need, int grid_numbers,
    int* grid_neighbor_numbers, int* grid_neighbors, VECTOR* grid_crd,
    LTMatrix3 cell, LTMatrix3 rcell, int max_atom_numbers_in_grid,
    ATOM_GROUP* nl, float cutoff_skin_square, int* grid_atom_numbers,
    int* grid_atoms, int max_neighbor_numbers, int* neighbor_list_overflow,
    VECTOR* grid_ghost_crd, int max_ghost_numbers_in_grid,
    int* grid_ghost_numbers, int* grid_ghosts)
{
    if (need[0] == 0) return;
#pragma omp parallel for schedule(dynamic)
    for (int grid_i = 0; grid_i < grid_numbers; grid_i++)

    {
        int* bucket_i = grid_atoms + grid_i * max_atom_numbers_in_grid;
        int atom_numbers_in_grid_i = grid_atom_numbers[grid_i];
        VECTOR* grid_crd_i = grid_crd + grid_i * max_atom_numbers_in_grid;
        for (int jj = 0; jj < grid_neighbor_numbers[grid_i]; ++jj)
        {
            int grid_j = grid_neighbors[jj + MAX_GRID_NEIGHBORS * grid_i];
            int* bucket_j = grid_atoms + grid_j * max_atom_numbers_in_grid;
            int atom_numbers_in_grid_j = grid_atom_numbers[grid_j];

            VECTOR* grid_crd_j = grid_crd + grid_j * max_atom_numbers_in_grid;
            for (int i = 0; i < atom_numbers_in_grid_i; i++)
            {
                int atom_i = bucket_i[i];
                int global_i = atom_local[atom_i];
                VECTOR crd_i = grid_crd_i[i];

                int* nl_atom_numbers_ptr = &nl[atom_i].atom_numbers;
                int* nl_atom_serial_ptr = nl[atom_i].atom_serial;
                for (int j = 0; j < atom_numbers_in_grid_j; ++j)
                {
                    int atom_j = bucket_j[j];
                    if (atom_local[atom_j] <= global_i) continue;
                    VECTOR crd_j = grid_crd_j[j];
                    VECTOR dr =
                        Get_Periodic_Displacement(crd_i, crd_j, cell, rcell);
                    float dr2 = dr * dr;
                    if (dr2 < cutoff_skin_square)
                    {
                        if (*nl_atom_numbers_ptr < max_neighbor_numbers)
                        {
                            nl_atom_serial_ptr[*nl_atom_numbers_ptr] = atom_j;
                            *nl_atom_numbers_ptr += 1;
                        }
                        else
                        {
                            neighbor_list_overflow[0] = 1;
                        }
                    }
                }
            }
        }

        for (int jj = 0; jj < grid_neighbor_numbers[grid_i]; ++jj)
        {
            int grid_j = grid_neighbors[jj + MAX_GRID_NEIGHBORS * grid_i];
            int ghost_numbers_in_grid_j = grid_ghost_numbers[grid_j];
            if (ghost_numbers_in_grid_j == 0) continue;
            int* bucket_j = grid_ghosts + grid_j * max_ghost_numbers_in_grid;
            VECTOR* grid_ghost_crd_j =
                grid_ghost_crd + grid_j * max_ghost_numbers_in_grid;
            for (int i = 0; i < atom_numbers_in_grid_i; i++)
            {
                int atom_i = bucket_i[i];
                VECTOR crd_i = grid_crd_i[i];

                int* nl_atom_numbers_ptr = &nl[atom_i].atom_numbers;
                int* nl_ghost_numbers_ptr = &nl[atom_i].ghost_numbers;
                int* nl_atom_serial_ptr = nl[atom_i].atom_serial;
                for (int j = 0; j < ghost_numbers_in_grid_j; ++j)
                {
                    int atom_j = bucket_j[j];
                    VECTOR crd_j = grid_ghost_crd_j[j];
                    VECTOR dr =
                        Get_Periodic_Displacement(crd_i, crd_j, cell, rcell);
                    float dr2 = dr * dr;
                    if (dr2 < cutoff_skin_square)
                    {
                        if (*nl_atom_numbers_ptr < max_neighbor_numbers)
                        {
                            nl_atom_serial_ptr[*nl_atom_numbers_ptr] = atom_j;
                            *nl_atom_numbers_ptr += 1;
                            *nl_ghost_numbers_ptr += 1;
                        }
                        else
                        {
                            neighbor_list_overflow[0] = 1;
                        }
                    }
                }
            }
        }
    }
}
#endif

static __global__ void Delete_Excluded_Atoms_Serial_In_Neighbor_List(
    const int* need, const int local_atom_numbers, int* atom_local,
    ATOM_GROUP* nl, const int* excluded_list_start, const int* excluded_list,
    const int* excluded_atom_numbers)
{
    if (need[0] == 0) return;
    SIMPLE_DEVICE_FOR(atom_i_local, local_atom_numbers)
    {
        int atom_i = atom_local[atom_i_local];
        int excluded_number = excluded_atom_numbers[atom_i];
        int list_start_i = excluded_number > 0 ? excluded_list_start[atom_i] : 0;
        int list_end_i = list_start_i + excluded_number;
        int atom_min_i = excluded_number > 0 ? excluded_list[list_start_i] : 0;
        int atom_max_i =
            excluded_number > 0 ? excluded_list[list_end_i - 1] : -1;

        ATOM_GROUP nl_i = nl[atom_i_local];
        int atomnumbers_in_nl_lin = nl_i.atom_numbers;
        int atom_j_local, atom_j;
        for (int i = 0; i < atomnumbers_in_nl_lin; ++i)
        {
            atom_j_local = nl_i.atom_serial[i];
            atom_j = atom_local[atom_j_local];
            bool is_excluded = false;

            if (excluded_number > 0 && atom_j >= atom_min_i && atom_j <= atom_max_i)
            {
                for (int j = list_start_i; j < list_end_i; ++j)
                {
                    if (atom_j == excluded_list[j])
                    {
                        is_excluded = true;
                        break;
                    }
                }
            }

            if (!is_excluded && atom_j_local >= local_atom_numbers)
            {
                int excluded_number_j = excluded_atom_numbers[atom_j];
                if (excluded_number_j > 0)
                {
                    int list_start_j = excluded_list_start[atom_j];
                    int list_end_j = list_start_j + excluded_number_j;
                    int atom_min_j = excluded_list[list_start_j];
                    int atom_max_j = excluded_list[list_end_j - 1];
                    if (atom_i >= atom_min_j && atom_i <= atom_max_j)
                    {
                        for (int j = list_start_j; j < list_end_j; ++j)
                        {
                            if (atom_i == excluded_list[j])
                            {
                                is_excluded = true;
                                break;
                            }
                        }
                    }
                }
            }

            if (is_excluded)
            {
                atomnumbers_in_nl_lin = atomnumbers_in_nl_lin - 1;
                nl_i.atom_serial[i] = nl_i.atom_serial[atomnumbers_in_nl_lin];
                i--;
            }
        }
        nl[atom_i_local].atom_numbers = atomnumbers_in_nl_lin;
    }
}

void NEIGHBOR_LIST::UPDATOR::Update(
    int* atom_local, int local_atom_numbers, int ghost_numbers, int need_copy,
    VECTOR* crd, LTMatrix3 cell, LTMatrix3 rcell, NEIGHBOR_LIST::GRIDS* grids,
    int max_atom_in_grid_numbers, int max_ghost_in_grid_numbers,
    int max_neighbor_numbers, float grid_length, int* d_neighbor_grid_overflow,
    int* d_neighbor_grid_ghost_overflow, int* d_neighbor_list_overflow,
    ATOM_GROUP* d_nl, int* excluded_list_start, int* excluded_list,
    int* excluded_numbers)
{
    int total_atom_numbers = local_atom_numbers + ghost_numbers;
    if (total_atom_numbers <= 0) return;
    Launch_Device_Kernel(
        Clear_Bucket,
        (grids->grid_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, d_need_update,
        grids->grid_numbers, grids->d_grid_atom_numbers,
        grids->d_grid_ghost_numbers);

    Launch_Device_Kernel(
        Put_Atom_In_Grids,
        (total_atom_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, d_need_update, need_copy,
        atom_local, local_atom_numbers, ghost_numbers, grids->grid_numbers, crd,
        old_crd, cell, rcell, grid_length, grids->Nx, grids->Ny, grids->Nz,
        grids->d_grid_atoms, grids->d_grid_atom_numbers, grids->d_grid_atom_crd,
        d_nl, max_atom_in_grid_numbers, d_neighbor_grid_overflow,
        grids->d_grid_ghosts, grids->d_grid_ghost_numbers,
        grids->d_grid_ghost_crd, max_ghost_in_grid_numbers,
        d_neighbor_grid_ghost_overflow);
    Launch_Device_Kernel(
        Find_Neighbors_Gridly, grids->grid_numbers,
        CONTROLLER::device_max_thread,
        (size_t)(max_atom_in_grid_numbers * (sizeof(VECTOR) + sizeof(int))),
        NULL, atom_local, local_atom_numbers, d_need_update,
        grids->grid_numbers, grids->d_neighbor_grid_numbers,
        grids->d_neighbor_grids, grids->d_grid_atom_crd, cell, rcell,
        max_atom_in_grid_numbers, d_nl, grid_length * grid_length * 4.0f,
        grids->d_grid_atom_numbers, grids->d_grid_atoms, max_neighbor_numbers,
        d_neighbor_list_overflow, grids->d_grid_ghost_crd,
        max_ghost_in_grid_numbers, grids->d_grid_ghost_numbers,
        grids->d_grid_ghosts);

    Launch_Device_Kernel(
        Delete_Excluded_Atoms_Serial_In_Neighbor_List,
        (local_atom_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, d_need_update,
        local_atom_numbers, atom_local, d_nl, excluded_list_start,
        excluded_list, excluded_numbers);
}

void NEIGHBOR_LIST::UPDATOR::Clear()
{
    Free_Host_And_Device_Pointer(NULL, (void**)&d_need_update);
    if (refresh_interval <= 0)
    {
        Free_Single_Device_Pointer((void**)&old_crd);
    }
}

void NEIGHBOR_LIST::Initial(CONTROLLER* controller, int atom_numbers,
                            float cutoff, float skin, LTMatrix3 cell,
                            LTMatrix3 rcell)
{
    this->atom_numbers = atom_numbers;
    this->cutoff = cutoff;
    this->skin = skin;
    controller->printf("START INITIALIZING NEIGHBOR LIST:\n");
    h_neighbor_grid_overflow = 0;
    h_neighbor_list_overflow = 0;
    h_neighbor_grid_ghost_overflow = 0;
    Device_Malloc_And_Copy_Safely((void**)&d_neighbor_grid_overflow,
                                  &h_neighbor_grid_overflow, sizeof(int));
    Device_Malloc_And_Copy_Safely((void**)&d_neighbor_list_overflow,
                                  &h_neighbor_list_overflow, sizeof(int));
    Device_Malloc_And_Copy_Safely((void**)&d_neighbor_grid_ghost_overflow,
                                  &h_neighbor_grid_ghost_overflow, sizeof(int));

    throw_error_when_overflow = 0;
    if (controller->Command_Exist("neighbor_list", "throw_error_when_overflow"))
        throw_error_when_overflow =
            controller->Get_Bool("neighbor_list", "throw_error_when_overflow",
                                 "NEIGHBOR_LIST::Initial");

    max_neighbor_numbers = 1200;
    if (controller->Command_Exist("neighbor_list", "max_neighbor_numbers"))
    {
        controller->Check_Int("neighbor_list", "max_neighbor_numbers",
                              "NEIGHBOR_LIST::Initial");
        max_neighbor_numbers =
            atoi(controller->Command("neighbor_list", "max_neighbor_numbers"));
    }
    controller->printf("    Max number of neighbors for one atom: %d\n",
                       max_neighbor_numbers);

    Malloc_Safely((void**)&h_nl, sizeof(ATOM_GROUP) * atom_numbers);
    Device_Malloc_Safely((void**)&d_temp,
                         sizeof(int) * atom_numbers * max_neighbor_numbers);
    for (int i = 0; i < atom_numbers; i++)
    {
        h_nl[i].atom_numbers = 0;
        h_nl[i].atom_serial = d_temp + max_neighbor_numbers * i;
    }
    Device_Malloc_And_Copy_Safely((void**)&d_nl, h_nl,
                                  sizeof(ATOM_GROUP) * atom_numbers);

    check_overflow_interval = 150;
    if (controller->Command_Exist("neighbor_list", "check_overflow_interval"))
    {
        controller->Check_Int("neighbor_list", "check_overflow_interval",
                              "NEIGHBOR_LIST::Initial");
        check_overflow_interval = atoi(
            controller->Command("neighbor_list", "check_overflow_interval"));
    }

    max_atom_in_grid_numbers = 150;
    if (controller->Command_Exist("neighbor_list", "max_atom_in_grid_numbers"))
    {
        controller->Check_Int("neighbor_list", "max_atom_in_grid_numbers",
                              "NEIGHBOR_LIST::Initial");
        max_atom_in_grid_numbers = atoi(
            controller->Command("neighbor_list", "max_atom_in_grid_numbers"));
    }
    max_ghost_in_grid_numbers = 150;
    if (controller->Command_Exist("neighbor_list", "max_ghost_in_grid_numbers"))
    {
        controller->Check_Int("neighbor_list", "max_ghost_in_grid_numbers",
                              "NEIGHBOR_LIST::Initial");
        max_ghost_in_grid_numbers = atoi(
            controller->Command("neighbor_list", "max_ghost_in_grid_numbers"));
    }

    grids.Initial(controller, max_atom_in_grid_numbers,
                  max_ghost_in_grid_numbers, cell, rcell,
                  (cutoff + skin) * 0.5f);

    updator.Initial(controller, atom_numbers);

    is_initialized = 1;

    if (grids.Nx <= 2 || grids.Ny <= 2 || grids.Nz <= 2)
    {
        controller->Throw_SPONGE_Error(spongeErrorMallocFailed,
                                       "NEIGHBOR_LIST::Initial",
                                       "the box is too small.");
    }
    if (this->cutoff_full > 0.0f)
    {
        controller->printf("    cutoff_full (from module): %f\n",
                           this->cutoff_full);
    }
    controller->printf("    is_needed_half: %s\n",
                       is_needed_half ? "true" : "false");
    controller->printf("    is_needed_full: %s\n",
                       is_needed_full ? "true" : "false");

    if (this->is_needed_full)
    {
        controller->printf("    Initializing full neighbor list...\n");
        full_neighbor_list.Initial(atom_numbers, max_neighbor_numbers);
    }

    controller->printf("END INITIALIZING NEIGHBOR LIST\n\n");
}

void NEIGHBOR_LIST::Update(int* atom_local, int local_atom_numbers,
                           int ghost_numbers, VECTOR* crd, LTMatrix3 cell,
                           LTMatrix3 rcell, int step, int update,
                           int* excluded_list_start, int* excluded_list,
                           int* excluded_numbers)
{
    if (!is_initialized) return;
    if (local_atom_numbers <= 0 && ghost_numbers <= 0) return;
    updator.time_recorder->Start();
    if (update == NEIGHBOR_LIST_UPDATE_PARAMETER::FORCED_UPDATE)
    {
        deviceMemset(updator.d_need_update, -1, sizeof(int));
    }
    else if (updator.refresh_interval <= 0)
    {
        deviceMemset(updator.d_need_update, 0, sizeof(int));
        updator.Check(local_atom_numbers, skin, crd, cell, rcell);
    }
    else if ((step + 1) % updator.refresh_interval == 0)
    {
        deviceMemset(updator.d_need_update, -1, sizeof(int));
    }
    else
    {
        deviceMemset(updator.d_need_update, 0, sizeof(int));
    }
    if (this->is_needed_half)
    {
        updator.Update(atom_local, local_atom_numbers, ghost_numbers,
                       updator.refresh_interval <= 0, crd, cell, rcell, &grids,
                       max_atom_in_grid_numbers, max_ghost_in_grid_numbers,
                       max_neighbor_numbers, 0.5f * (cutoff + skin),
                       d_neighbor_grid_overflow, d_neighbor_grid_ghost_overflow,
                       d_neighbor_list_overflow, this->d_nl,
                       excluded_list_start, excluded_list, excluded_numbers);
    }

    if (this->is_needed_full && full_neighbor_list.is_initialized)
    {
        if (this->cutoff_full > 0.0f)
        {
            full_neighbor_list.Build_From_Half_With_Cutoff(
                this->d_nl, local_atom_numbers, crd, cell, rcell,
                this->cutoff_full + skin);
        }
        else
        {
            full_neighbor_list.Build_From_Half(this->d_nl, local_atom_numbers);
        }
    }
    updator.time_recorder->Stop();
}

void NEIGHBOR_LIST::Check_Overflow(CONTROLLER* controller, int steps,
                                   const LTMatrix3 cell, const LTMatrix3 rcell,
                                   LTMatrix3* cell0)
{
    if (is_initialized && (steps + 1) % check_overflow_interval == 0)
    {
        bool re_initializing = 0;
        deviceMemcpy(&h_neighbor_grid_overflow, d_neighbor_grid_overflow,
                     sizeof(int), deviceMemcpyDeviceToHost);
        if (h_neighbor_grid_overflow)
        {
            re_initializing = 1;
            controller->commands["neighbor_list_max_atom_in_grid_numbers"] =
                std::to_string((int)(max_atom_in_grid_numbers * 1.1));
            controller->printf(
                "Overflow occured in neighbor searching.\n\
SPONGE will re-initialize the neighbor list module with %s = %d\n\n",
                "neighbor_list_max_atom_in_grid_numbers",
                (int)(max_atom_in_grid_numbers * 1.1));
        }
        deviceMemcpy(&h_neighbor_grid_ghost_overflow,
                     d_neighbor_grid_ghost_overflow, sizeof(int),
                     deviceMemcpyDeviceToHost);
        if (h_neighbor_grid_ghost_overflow)
        {
            re_initializing = 1;
            controller->commands["neighbor_list_max_ghost_in_grid_numbers"] =
                std::to_string((int)(max_ghost_in_grid_numbers * 1.1));
            controller->printf(
                "Overflow occured in neighbor searching.\n\
SPONGE will re-initialize the neighbor list module with %s = %d\n\n",
                "neighbor_list_max_ghost_in_grid_numbers",
                (int)(max_ghost_in_grid_numbers * 1.1));
        }

        int h_full_overflow = 0;
        if (full_neighbor_list.is_initialized)
        {
            deviceMemcpy(&h_full_overflow, full_neighbor_list.d_overflow,
                         sizeof(int), deviceMemcpyDeviceToHost);
        }

        deviceMemcpy(&h_neighbor_list_overflow, d_neighbor_list_overflow,
                     sizeof(int), deviceMemcpyDeviceToHost);
        if (h_neighbor_list_overflow || h_full_overflow)
        {
            re_initializing = 1;
            controller->commands["neighbor_list_max_neighbor_numbers"] =
                std::to_string((int)(max_neighbor_numbers * 1.1));
            controller->printf(
                "Overflow occured in neighbor searching.\n\
SPONGE will re-initialize the neighbor list module with %s = %d.\n\n",
                "neighbor_list_max_neighbor_numbers",
                (int)(max_neighbor_numbers * 1.1));
        }
        if (re_initializing)
        {
            if (!throw_error_when_overflow)
            {
                this->Clear();
                this->Initial(controller, atom_numbers, cutoff, skin, cell,
                              rcell);
                cell0[0] = cell;
                printf(
                    "--------------------------------------------------"
                    "--------"
                    "--------------------------------------------------"
                    "\n");
            }
            else
            {
                controller->Throw_SPONGE_Error(
                    spongeErrorOverflow, "NEIGHBOR_LIST::Check_Overflow",
                    "Reason:\n\tOverflow occured in neighbor "
                    "searching");
            }
        }
    }
}

void NEIGHBOR_LIST::Clear()
{
    if (is_initialized)
    {
        Free_Single_Device_Pointer((void**)&d_temp);
        Free_Host_And_Device_Pointer((void**)&h_nl, (void**)&d_nl);
        Free_Host_And_Device_Pointer(NULL, (void**)&d_neighbor_grid_overflow);
        Free_Host_And_Device_Pointer(NULL, (void**)&d_neighbor_list_overflow);
        Free_Host_And_Device_Pointer(NULL,
                                     (void**)&d_neighbor_grid_ghost_overflow);
        full_neighbor_list.Clear();
        grids.Clear();
        updator.Clear();
    }
}

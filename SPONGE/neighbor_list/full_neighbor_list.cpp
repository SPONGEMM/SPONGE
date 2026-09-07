#include "full_neighbor_list.h"

void FULL_NEIGHBOR_LIST::Initial(int atom_numbers, int max_neighbor_numbers)
{
    if (is_initialized) return;

    this->atom_numbers = atom_numbers;
    this->max_neighbor_numbers = max_neighbor_numbers;

    Malloc_Safely((void**)&h_nl, sizeof(ATOM_GROUP) * atom_numbers);
    Device_Malloc_Safely((void**)&d_temp,
                         sizeof(int) * atom_numbers * max_neighbor_numbers);
    for (int i = 0; i < atom_numbers; i++)
    {
        h_nl[i].atom_numbers = 0;
        h_nl[i].ghost_numbers = 0;
        h_nl[i].atom_serial = d_temp + max_neighbor_numbers * i;
    }
    Device_Malloc_And_Copy_Safely((void**)&d_nl, h_nl,
                                  sizeof(ATOM_GROUP) * atom_numbers);
    int h_overflow = 0;
    Device_Malloc_And_Copy_Safely((void**)&d_overflow, &h_overflow,
                                  sizeof(int));
    is_initialized = true;
}

static __global__ void Build_Full_Neighbor_List_Kernel(
    const ATOM_GROUP* half_nl, ATOM_GROUP* full_nl, int atom_numbers,
    int max_neighbor_numbers, int* overflow_flag)
{
    SIMPLE_DEVICE_FOR(atom_i, atom_numbers)
    {
        ATOM_GROUP h_nl_i = half_nl[atom_i];
        for (int k = 0; k < h_nl_i.atom_numbers; k++)
        {
            int atom_j = h_nl_i.atom_serial[k];
            int slot_i = atomicAdd(&full_nl[atom_i].atom_numbers, 1);
            if (slot_i < max_neighbor_numbers)
            {
                full_nl[atom_i].atom_serial[slot_i] = atom_j;
            }
            else
            {
                atomicExch(overflow_flag, 1);
            }
            if (atom_j < atom_numbers)
            {
                int slot_j = atomicAdd(&full_nl[atom_j].atom_numbers, 1);
                if (slot_j < max_neighbor_numbers)
                {
                    full_nl[atom_j].atom_serial[slot_j] = atom_i;
                }
                else
                {
                    atomicExch(overflow_flag, 1);
                }
            }
        }
    }
}

void FULL_NEIGHBOR_LIST::Build_From_Half(const ATOM_GROUP* half_nl,
                                         int atom_numbers)
{
    if (!is_initialized) return;
    if (atom_numbers != this->atom_numbers) return;
    for (int i = 0; i < atom_numbers; i++)
    {
        h_nl[i].atom_numbers = 0;
        h_nl[i].ghost_numbers = 0;
    }
    deviceMemcpy(d_nl, h_nl, sizeof(ATOM_GROUP) * atom_numbers,
                 deviceMemcpyHostToDevice);
    Launch_Device_Kernel(Build_Full_Neighbor_List_Kernel,
                         (atom_numbers + CONTROLLER::device_max_thread - 1) /
                             CONTROLLER::device_max_thread,
                         CONTROLLER::device_max_thread, 0, NULL, half_nl, d_nl,
                         atom_numbers, max_neighbor_numbers, d_overflow);
}

static __global__ void Build_Full_Neighbor_List_With_Cutoff_Kernel(
    const ATOM_GROUP* half_nl, ATOM_GROUP* full_nl, int atom_numbers,
    int max_neighbor_numbers, int* overflow_flag, const VECTOR* crd,
    const Boundary boundary, float cutoff)
{
    SIMPLE_DEVICE_FOR(atom_i, atom_numbers)
    {
        ATOM_GROUP h_nl_i = half_nl[atom_i];
        VECTOR ri = crd[atom_i];
        for (int k = 0; k < h_nl_i.atom_numbers; k++)
        {
            int atom_j = h_nl_i.atom_serial[k];
            VECTOR rj = crd[atom_j];
            VECTOR dr =
                Get_Displacement<BoundaryPolicy::Periodic>(ri, rj, boundary);
            float dist_sq = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;

            if (dist_sq <= cutoff * cutoff)
            {
                int slot_i = atomicAdd(&full_nl[atom_i].atom_numbers, 1);
                if (slot_i < max_neighbor_numbers)
                {
                    full_nl[atom_i].atom_serial[slot_i] = atom_j;
                }
                else
                {
                    atomicExch(overflow_flag, 1);
                }

                if (atom_j < atom_numbers)
                {
                    int slot_j = atomicAdd(&full_nl[atom_j].atom_numbers, 1);
                    if (slot_j < max_neighbor_numbers)
                    {
                        full_nl[atom_j].atom_serial[slot_j] = atom_i;
                    }
                    else
                    {
                        atomicExch(overflow_flag, 1);
                    }
                }
            }
        }
    }
}

void FULL_NEIGHBOR_LIST::Build_From_Half_With_Cutoff(const ATOM_GROUP* half_nl,
                                                     int atom_numbers,
                                                     const VECTOR* crd,
                                                     const Boundary boundary,
                                                     float cutoff)
{
    if (!is_initialized) return;
    if (atom_numbers != this->atom_numbers) return;

    for (int i = 0; i < atom_numbers; i++)
    {
        h_nl[i].atom_numbers = 0;
        h_nl[i].ghost_numbers = 0;
    }
    deviceMemcpy(d_nl, h_nl, sizeof(ATOM_GROUP) * atom_numbers,
                 deviceMemcpyHostToDevice);

    Launch_Device_Kernel(Build_Full_Neighbor_List_With_Cutoff_Kernel,
                         (atom_numbers + CONTROLLER::device_max_thread - 1) /
                             CONTROLLER::device_max_thread,
                         CONTROLLER::device_max_thread, 0, NULL, half_nl, d_nl,
                         atom_numbers, max_neighbor_numbers, d_overflow, crd,
                         boundary, cutoff);
}

void FULL_NEIGHBOR_LIST::Clear()
{
    if (is_initialized)
    {
        Free_Single_Device_Pointer((void**)&d_temp);
        Free_Host_And_Device_Pointer((void**)&h_nl, (void**)&d_nl);
        Free_Single_Device_Pointer((void**)&d_overflow);
        is_initialized = false;
    }
}

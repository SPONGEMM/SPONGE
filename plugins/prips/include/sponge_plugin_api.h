#pragma once

#include <stdint.h>

extern "C"
{
    enum
    {
        SPONGE_PRIPS_API_VERSION = 2,
    };

    typedef struct SPONGE_PLUGIN_API
    {
        uint32_t api_version;
        int device_type;

        const char* (*get_command)(const char* key);
        void (*log_message)(const char* message);

        int (*get_mpi_rank)();
        int (*get_atom_numbers)();
        int (*get_steps)();
        void* (*get_coordinate_ptr)();
        void* (*get_force_ptr)();

        int (*get_neighbor_list_max_numbers)();
        int (*get_neighbor_list_count)(int atom_index);
        void* (*get_neighbor_list_index_ptr)();

        int (*get_local_atom_numbers)();
        int (*get_local_ghost_numbers)();
        int (*get_local_pp_rank)();
        int (*get_local_max_atom_numbers)();
        void* (*get_atom_local_ptr)();
        void* (*get_atom_local_label_ptr)();
        void* (*get_atom_local_id_ptr)();
        void* (*get_local_coordinate_ptr)();
        void* (*get_local_force_ptr)();
    } SPONGE_PLUGIN_API;
}

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

    enum
    {
        SPONGE_LOCAL_MP_API_VERSION = 1,
        SPONGE_MLIP_FEATURE_FP32 = 1,
        SPONGE_MLIP_NEIGHBOR_READY = 0,
        SPONGE_MLIP_NEIGHBOR_UNAVAILABLE = 1,
        SPONGE_MLIP_NEIGHBOR_INVALID_ARGUMENT = -1,
    };

    // Device CSR: one row per owned target, owned-plus-ghost source indices.
    // q = image_shift translates the SOURCE: r_source + q @ cell.
    // Receiver-minus-sender vector: r_target - r_source - q @ cell.
    typedef struct SPONGE_MLIP_NEIGHBOR_VIEW
    {
        uint64_t neighbor_epoch;
        int32_t owned_count;
        int32_t edge_count;
        const int32_t* row_ptr;             // owned_count + 1 elements
        const int32_t* source_local_index;  // edge_count elements
        const int32_t* image_shift;         // 3 * edge_count elements
    } SPONGE_MLIP_NEIGHBOR_VIEW;

    // Local-MP contract; the existing PRIPS v2 table remains frozen.
    // CUDA requires stream-scoped exchange and owned-output submission.
    // Edge assembly is required only by models using edge gradients.
    // CPU hosts leave CUDA callbacks null and reject active local-MP setup.
    typedef struct SPONGE_LOCAL_MP_API_V1
    {
        uint32_t api_version;
        const SPONGE_PLUGIN_API* prips_api;
        int (*register_mlip_capabilities)(double cutoff_angstrom,
                                          double skin_angstrom,
                                          int interaction_depth,
                                          int feature_dtype,
                                          uint64_t capability_hash);
        int (*get_pp_rank)();
        int (*get_pp_size)();
        int (*get_device_index)();
        uint64_t (*get_domain_epoch)();
        int (*get_need_potential)();
        int (*get_need_pressure)();
        int (*get_cell)(float cell_row_major[9], uint8_t* pbc_mask);
        // Zero means unavailable. Borrowed CSR pointers live until epoch
        // changes.
        uint64_t (*get_mlip_neighbor_epoch)();
        int (*get_mlip_neighbor_view)(SPONGE_MLIP_NEIGHBOR_VIEW* view);
        // Process-local CUDA stream handle; zero is the default stream.
        // Caller joins all tensor producers onto producer_stream first.
        // Success is HOST completion: rows are ready and all native borrows
        // and scratch consumers have finished, including reverse additions.
        // Calls are serial on one plan; epoch must match its current mapping.
        int (*exchange_on_stream)(float* rows, int row_count, int feature_width,
                                  int reverse, uint64_t domain_epoch,
                                  uintptr_t producer_stream);
        // Selected IDs index current candidate CSR entries; g=dE/dv,
        // v=r_target-r_source-q@cell in model units. HOST completion returns
        // owned forces and optional W=-sym(sum(v outer g)) in model units.
        // All ranks participate, including empty selected sets.
        int (*assemble_selected_edge_gradients)(
            uint64_t domain_epoch, uint64_t neighbor_epoch, int selected_count,
            const int64_t* selected_candidate_ids, const float* gradients_model,
            float* owned_forces_model, float* virial_model_9, int need_virial,
            double length_to_angstrom, uintptr_t producer_stream);
        // Borrowed selected-device FP32 inputs in runtime units until HOST
        // completion. Add owned xyz forces, optional local scalar energy and
        // physical row-major W[9]; native packs [xx,xy+yx,yy,xz+zx,yz+zy,zz].
        // No MPI reduction; no alias with destinations. Null force requires
        // zero owned atoms. Invalid metadata/device fails before mutation.
        int (*submit_owned_outputs)(uint64_t domain_epoch, int owned_count,
                                    const float* forces, const float* energy,
                                    const float* virial9,
                                    uintptr_t producer_stream);
    } SPONGE_LOCAL_MP_API_V1;

    // Independent named exports; no existing function table changes.
    // Setup returns READY (+1) after local model preparation succeeds.
    // INACTIVE (0) has no local-MP driver; FAILED (-1) is not admissible.
    typedef int32_t (*SPONGE_PREPARE_LOCAL_MP_SETUP_V1)();
    // Enable before Initial_Stable: zero means checked errors are enabled.
    typedef int32_t (*SPONGE_ENABLE_CHECKED_RUNTIME_ERRORS_V1)();
    // Zero: no error. One: sticky fatal error. UTF-8 strings remain plugin
    // owned and stable until unload; the host must not free them.
    typedef int32_t (*SPONGE_GET_RUNTIME_ERROR_V1)(const char** stage,
                                                   const char** message);
}

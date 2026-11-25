#ifndef MLIP_NEIGHBOR_LIST_H
#define MLIP_NEIGHBOR_LIST_H

#include <cstdint>

#include "neighbor_list.h"

// Directed, image-aware CSR graph for finite-cutoff ML potentials. The graph
// rows belong to owned atoms; source indices address the caller's owned+ghost
// local array. Device pointers returned by the getters remain valid until the
// next successful rebuild or Clear().
class MLIP_NEIGHBOR_LIST
{
   public:
    MLIP_NEIGHBOR_LIST() = default;
    ~MLIP_NEIGHBOR_LIST();
    MLIP_NEIGHBOR_LIST(const MLIP_NEIGHBOR_LIST&) = delete;
    MLIP_NEIGHBOR_LIST& operator=(const MLIP_NEIGHBOR_LIST&) = delete;

    void Initial(CONTROLLER* controller, int max_local_atoms, double cutoff,
                 double skin);
    // atom_local must contain unique global IDs across owned and ghost rows;
    // periodic copies are represented by image_shift rather than duplicate
    // local entries.
    bool Update(const int* atom_local, int owned_count, int ghost_count,
                const VECTOR* crd, const Boundary& boundary,
                uint64_t domain_epoch, bool force_rebuild = false);
    void Clear();

    uint64_t Get_Neighbor_Epoch() const { return neighbor_epoch_; }
    int Get_Edge_Count() const { return edge_count_; }
    const int* Get_Row_Ptr() const { return row_ptr_; }
    const int* Get_Source_Local_Index() const { return source_local_index_; }
    const int* Get_Image_Shift() const { return image_shift_; }

   private:
    void Build(const int* atom_local, int owned_count, int ghost_count,
               const VECTOR* crd, const Boundary& boundary,
               uint64_t domain_epoch);
    void Ensure_Reference_Capacity(int local_count);
    void Ensure_Row_Capacity(int owned_count);
    void Ensure_Edge_Capacity(int edge_count);

    bool initialized_ = false;
    double skin_ = 0.0f;
    float graph_radius_ = 0.0f;
    float grid_length_ = 0.0f;
    int owned_count_ = 0;
    int ghost_count_ = 0;
    int edge_count_ = 0;
    int row_capacity_ = 0;
    int edge_capacity_ = 0;
    int staging_row_capacity_ = 0;
    int staging_edge_capacity_ = 0;
    int row_scratch_capacity_ = 0;
    int reference_capacity_ = 0;
    uint64_t neighbor_epoch_ = 0;
    uint64_t domain_epoch_ = 0;
    Boundary boundary_;
    bool has_graph_ = false;

    // The grid object is model-specific scratch and never aliases the
    // classical half/full neighbor-list state.
    NEIGHBOR_LIST::GRIDS buckets_;
    int nx_ = 0, ny_ = 0, nz_ = 0;

    int* row_ptr_ = nullptr;
    int* source_local_index_ = nullptr;
    int* image_shift_ = nullptr;
    int* staging_row_ptr_ = nullptr;
    int* staging_source_local_index_ = nullptr;
    int* staging_image_shift_ = nullptr;

    uint64_t* row_count_ = nullptr;
    uint64_t* row_offset_ = nullptr;
    VECTOR* reference_crd_ = nullptr;
    int* reference_atom_local_ = nullptr;
    int* d_owned_overflow_ = nullptr;
    int* d_ghost_overflow_ = nullptr;
    int* d_build_error_ = nullptr;
    int* d_rebuild_needed_ = nullptr;
};

#endif  // MLIP_NEIGHBOR_LIST_H

#pragma once

#include "../../plugins/prips/include/sponge_plugin_api.h"
#include "../common.h"

// Model-neutral assembly of derivatives dE/dv on selected candidate CSR edges.
// This workspace never aliases native mixed force or energy/virial buffers.
// The caller reverses Local_Forces exactly once, then copies owned outputs.
class MLIP_EDGE_OUTPUT
{
   public:
    MLIP_EDGE_OUTPUT() = default;
    ~MLIP_EDGE_OUTPUT() { Clear(); }
    MLIP_EDGE_OUTPUT(const MLIP_EDGE_OUTPUT&) = delete;
    MLIP_EDGE_OUTPUT& operator=(const MLIP_EDGE_OUTPUT&) = delete;
    void Assemble(const SPONGE_MLIP_NEIGHBOR_VIEW& graph, int ghost_count,
                  const VECTOR* coordinates, const LTMatrix3& cell,
                  int selected_count, const int64_t* selected_candidate_ids,
                  const float* gradients_model, bool need_virial,
                  double length_to_angstrom);
    float* Local_Forces() const { return local_forces_; }
    void Copy_Outputs(float* owned_forces_model, float* virial_model_9,
                      bool need_virial) const;
    void Clear();

   private:
    int* target_rows_ = nullptr;
    float* local_forces_ = nullptr;
    float* virial_partials_ = nullptr;
    float* virial_9_ = nullptr;
    int* error_ = nullptr;
    int target_capacity_ = 0;
    int local_capacity_ = 0;
    int partial_capacity_ = 0;
    int owned_count_ = 0;
    uint64_t cached_neighbor_epoch_ = 0;
};

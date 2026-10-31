#pragma once

#include "../common.h"

// Owned-atom snapshot for the validity of a shared coordinate halo. Independent
// of all neighbor-list snapshots; record only after rebuilding halo coverage.
class DD_HALO_REFERENCE
{
   public:
    DD_HALO_REFERENCE() = default;
    ~DD_HALO_REFERENCE();
    DD_HALO_REFERENCE(const DD_HALO_REFERENCE&) = delete;
    DD_HALO_REFERENCE& operator=(const DD_HALO_REFERENCE&) = delete;

    // Coordinate/ID producers must precede these operations on stream. Callers
    // must explicitly join other producer streams before calling. Both methods
    // complete their work on stream before returning; Clear has no pending
    // work.
    void Record(int owned_count, const int* atom_local, const VECTOR* crd,
                const Boundary& boundary, deviceStream_t stream);
    // A nonfinite coordinate raises an error, not a normal refresh request.
    bool Needs_Refresh(int owned_count, const int* atom_local,
                       const VECTOR* crd, const Boundary& boundary,
                       double displacement_limit, deviceStream_t stream);
    void Clear();

   private:
    void Ensure_Capacity(int count);
    VECTOR* reference_crd_ = nullptr;
    int* reference_ids_ = nullptr;
    int* status_ = nullptr;
    int capacity_ = 0;
    int owned_count_ = 0;
    Boundary boundary_{};
    bool valid_ = false;
};

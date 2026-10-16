#pragma once

#include "../neighbor_list/mlip_edge_output.h"

struct DOMAIN_INFORMATION;
struct MD_INFORMATION;
struct CONTROLLER;
class MLIP_NEIGHBOR_LIST;

// Native process-local service behind the context-free plugin ABI. DD retains
// ownership of the domain and transport; graph pointers are borrowed from main.
class LOCAL_MP_SERVICE
{
   public:
    LOCAL_MP_SERVICE() = default;
    ~LOCAL_MP_SERVICE() { Clear(); }
    LOCAL_MP_SERVICE(const LOCAL_MP_SERVICE&) = delete;
    LOCAL_MP_SERVICE& operator=(const LOCAL_MP_SERVICE&) = delete;

    void Initial(MD_INFORMATION* md, CONTROLLER* controller);
    void Set_Domain(DOMAIN_INFORMATION* domain);
    DOMAIN_INFORMATION* Domain() const { return domain_; }
    uint64_t Domain_Epoch() const { return domain_epoch_; }
    uint64_t Neighbor_Epoch() const;
    int Neighbor_View(SPONGE_MLIP_NEIGHBOR_VIEW* view);
    void Invalidate_Neighbor_View();
    void Publish_Neighbor_View(const MLIP_NEIGHBOR_LIST& graph, int owned_count,
                               int ghost_count);
    // Called after force work completes, before controller/device shutdown.
    void Clear();
#ifdef USE_CUDA
    int Exchange_On_Stream(float* rows, int row_count, int width, int reverse,
                           uint64_t epoch, uintptr_t producer_stream);
    int Submit_Owned_Outputs(uint64_t domain_epoch, int owned_count,
                             const float* forces, const float* energy,
                             const float* virial9, uintptr_t producer_stream);
    int Assemble_Selected_Edge_Gradients(
        uint64_t domain_epoch, uint64_t neighbor_epoch, int selected_count,
        const int64_t* selected_ids, const float* gradients_model,
        float* owned_forces_model, float* virial_model_9, int need_virial,
        double length_to_angstrom, uintptr_t producer_stream);
#endif

   private:
    MD_INFORMATION* md_ = nullptr;
    CONTROLLER* controller_ = nullptr;
    DOMAIN_INFORMATION* domain_ = nullptr;
    uint64_t domain_epoch_ = 0;
    SPONGE_MLIP_NEIGHBOR_VIEW neighbor_view_ = {};
    uint64_t neighbor_domain_epoch_ = 0;
    int neighbor_ghost_count_ = 0;
    static int PP_Rank();
    int Device_Index() const;
    bool Tensor_Shape_Valid(float* rows, int row_count, int width);
#ifdef USE_CUDA
    struct ExecutionScope
    {
        cudaEvent_t producer_ready = nullptr;
        int device = -1;
        bool busy = false;
    } execution_;
    MLIP_EDGE_OUTPUT edge_output_;
#endif
};

// The existing ABI supports one simulation context per process.
LOCAL_MP_SERVICE& Local_MP_Service();

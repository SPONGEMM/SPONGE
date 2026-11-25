#include "local_mp_service.h"

#include <stdexcept>

#include "../Domain_decomposition/Domain_decomposition.h"
#include "../MD_core/MD_core.h"
#include "../neighbor_list/mlip_neighbor_list.h"

LOCAL_MP_SERVICE& Local_MP_Service()
{
    static LOCAL_MP_SERVICE service;
    return service;
}

void LOCAL_MP_SERVICE::Initial(MD_INFORMATION* md, CONTROLLER* controller)
{
    Clear();
    md_ = md;
    controller_ = controller;
}

void LOCAL_MP_SERVICE::Set_Domain(DOMAIN_INFORMATION* domain)
{
    Invalidate_Neighbor_View();
    domain_ = domain;
    ++domain_epoch_;
}

int LOCAL_MP_SERVICE::PP_Rank()
{
    return CONTROLLER::PP_MPI_rank >= 0 &&
                   CONTROLLER::PP_MPI_rank < CONTROLLER::PP_MPI_size
               ? CONTROLLER::PP_MPI_rank
               : -1;
}

int LOCAL_MP_SERVICE::Device_Index() const
{
    return controller_ == nullptr ? -1 : controller_->working_device;
}

void LOCAL_MP_SERVICE::Clear()
{
    Invalidate_Neighbor_View();
#ifdef USE_CUDA
    int previous = -1;
    if (execution_.producer_ready != nullptr)
    {
        cudaGetDevice(&previous);
        cudaSetDevice(execution_.device);
    }
    edge_output_.Clear();
    if (execution_.producer_ready != nullptr)
    {
        cudaEventDestroy(execution_.producer_ready);
        if (previous >= 0) cudaSetDevice(previous);
    }
    execution_ = {};
#endif
    domain_epoch_ = 0;
    domain_ = nullptr;
    controller_ = nullptr;
    md_ = nullptr;
}

bool LOCAL_MP_SERVICE::Tensor_Shape_Valid(float* local_rows, int row_count,
                                          int feature_width)
{
    return PP_Rank() >= 0 && domain_ != NULL && controller_ != NULL &&
           row_count >= 0 && (row_count == 0 || local_rows != NULL) &&
           feature_width > 0 &&
           row_count == domain_->atom_numbers + domain_->ghost_numbers;
}

uint64_t LOCAL_MP_SERVICE::Neighbor_Epoch() const
{
    if (domain_ == nullptr || neighbor_domain_epoch_ != domain_epoch_ ||
        neighbor_view_.owned_count != domain_->atom_numbers ||
        neighbor_ghost_count_ != domain_->ghost_numbers)
        return 0;
    return neighbor_view_.neighbor_epoch;
}

int LOCAL_MP_SERVICE::Neighbor_View(SPONGE_MLIP_NEIGHBOR_VIEW* view)
{
    if (view == nullptr) return SPONGE_MLIP_NEIGHBOR_INVALID_ARGUMENT;
    *view = {};
    if (Neighbor_Epoch() == 0) return SPONGE_MLIP_NEIGHBOR_UNAVAILABLE;
    *view = neighbor_view_;
    return SPONGE_MLIP_NEIGHBOR_READY;
}

#ifdef USE_CUDA
int LOCAL_MP_SERVICE::Exchange_On_Stream(float* rows, int row_count, int width,
                                         int reverse, uint64_t epoch,
                                         uintptr_t producer_stream)
{
    auto& context = execution_;
    if (context.busy || epoch != domain_epoch_ ||
        (reverse != 0 && reverse != 1) ||
        !Tensor_Shape_Valid(rows, row_count, width))
        return 1;
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess || device != Device_Index())
        return 1;
    if (context.producer_ready == nullptr)
    {
        if (cudaEventCreateWithFlags(&context.producer_ready,
                                     cudaEventDisableTiming) != cudaSuccess)
            return 1;
        context.device = device;
    }
    if (context.device != device) return 1;
    context.busy = true;
    try
    {
        auto error =
            cudaEventRecord(context.producer_ready,
                            reinterpret_cast<deviceStream_t>(producer_stream));
        if (error == cudaSuccess)
            error = cudaEventSynchronize(context.producer_ready);
        if (error != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(error));
        // This wait also applies to size-one and zero-traffic calls.
        if (reverse)
            domain_->Accumulate_Ghost_Tensor(controller_, rows, width);
        else
            domain_->Update_Ghost_Tensor(controller_, rows, width);
        // Both native paths pack/unpack/add on stream 0 and join their
        // transport streams. Wait terminal writes, not just NCCL completion.
        error = deviceStreamSynchronize(0);
        if (error != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(error));
        context.busy = false;
        return 0;
    }
    catch (const std::exception& error)
    {
        // Distributed failures cannot safely return a reusable slab to Python.
        controller_->Throw_SPONGE_Error(spongeErrorSimulationBreakDown,
                                        "PluginExchangeOnStream", error.what());
        return 1;
    }
}

static bool EdgeOutputPointerOnDevice(const void* pointer, int device)
{
    cudaPointerAttributes attributes = {};
    return pointer != nullptr &&
           cudaPointerGetAttributes(&attributes, pointer) == cudaSuccess &&
           attributes.type == cudaMemoryTypeDevice &&
           attributes.device == device;
}

static __global__ void AddOwnedOutputs(int count, const float* forces,
                                       const float* energy, const float* virial,
                                       VECTOR* destination,
                                       float* destination_energy,
                                       float* destination_virial)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
    {
        destination[i].x += forces[3 * i];
        destination[i].y += forces[3 * i + 1];
        destination[i].z += forces[3 * i + 2];
    }
    if (i == 0)
    {
        if (energy != nullptr) destination_energy[0] += energy[0];
        if (virial != nullptr)
        {
            destination_virial[0] += virial[0];
            destination_virial[1] = __fadd_rn(destination_virial[1],
                                              __fadd_rn(virial[1], virial[3]));
            destination_virial[2] += virial[4];
            destination_virial[3] = __fadd_rn(destination_virial[3],
                                              __fadd_rn(virial[2], virial[6]));
            destination_virial[4] = __fadd_rn(destination_virial[4],
                                              __fadd_rn(virial[5], virial[7]));
            destination_virial[5] += virial[8];
        }
    }
}

int LOCAL_MP_SERVICE::Submit_Owned_Outputs(uint64_t epoch, int owned_count,
                                           const float* forces,
                                           const float* energy,
                                           const float* virial9,
                                           uintptr_t producer_stream)
{
    auto& context = execution_;
    if (context.busy || domain_ == nullptr || controller_ == nullptr ||
        PP_Rank() < 0 || epoch != domain_epoch_ || owned_count < 0 ||
        owned_count != domain_->atom_numbers)
        return 1;
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess || device != Device_Index() ||
        (owned_count > 0 &&
         (!EdgeOutputPointerOnDevice(forces, device) ||
          !EdgeOutputPointerOnDevice(domain_->frc, device))) ||
        (owned_count == 0 && forces != nullptr &&
         !EdgeOutputPointerOnDevice(forces, device)) ||
        (energy != nullptr &&
         (!EdgeOutputPointerOnDevice(energy, device) ||
          !EdgeOutputPointerOnDevice(domain_->d_mlip_energy, device))) ||
        (virial9 != nullptr &&
         (!EdgeOutputPointerOnDevice(virial9, device) ||
          !EdgeOutputPointerOnDevice(domain_->d_mlip_virial, device))))
        return 1;
    if (context.producer_ready == nullptr)
    {
        if (cudaEventCreateWithFlags(&context.producer_ready,
                                     cudaEventDisableTiming) != cudaSuccess)
            return 1;
        context.device = device;
    }
    if (context.device != device) return 1;
    context.busy = true;
    try
    {
        auto status =
            cudaEventRecord(context.producer_ready,
                            reinterpret_cast<cudaStream_t>(producer_stream));
        if (status == cudaSuccess)
            status = cudaEventSynchronize(context.producer_ready);
        if (status != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(status));
        Launch_Device_Kernel(AddOwnedOutputs, owned_count / 128 + 1, 128, 0,
                             nullptr, owned_count, forces, energy, virial9,
                             domain_->frc, domain_->d_mlip_energy,
                             domain_->d_mlip_virial);
        status = cudaGetLastError();
        if (status == cudaSuccess) status = deviceStreamSynchronize(0);
        if (status != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(status));
        context.busy = false;
        return 0;
    }
    catch (const std::exception& error)
    {
        controller_->Throw_SPONGE_Error(spongeErrorSimulationBreakDown,
                                        "PluginSubmitOwnedOutputs",
                                        error.what());
        return 1;
    }
}

int LOCAL_MP_SERVICE::Assemble_Selected_Edge_Gradients(
    uint64_t domain_epoch, uint64_t neighbor_epoch, int selected_count,
    const int64_t* selected_ids, const float* gradients_model,
    float* owned_forces_model, float* virial_model_9, int need_virial,
    double length_to_angstrom, uintptr_t producer_stream)
{
    // Reuse the execution scope to serialize borrows of the same halo plan.
    auto& context = execution_;
    try
    {
        const auto& graph = neighbor_view_;
        if (context.busy || domain_epoch != domain_epoch_ ||
            neighbor_epoch == 0 || neighbor_epoch != Neighbor_Epoch() ||
            selected_count < 0 || (need_virial != 0 && need_virial != 1) ||
            PP_Rank() < 0 || !SpongeFloat::Is_Finite(length_to_angstrom) ||
            length_to_angstrom <= 0)
            throw std::invalid_argument(
                "invalid MLIP edge output epoch or metadata");
        int device = -1;
        if (cudaGetDevice(&device) != cudaSuccess || device != Device_Index() ||
            (selected_count > 0 &&
             (!EdgeOutputPointerOnDevice(selected_ids, device) ||
              !EdgeOutputPointerOnDevice(gradients_model, device))) ||
            (graph.owned_count > 0 &&
             !EdgeOutputPointerOnDevice(owned_forces_model, device)) ||
            (need_virial && !EdgeOutputPointerOnDevice(virial_model_9, device)))
            throw std::invalid_argument(
                "MLIP edge output requires selected-device CUDA tensors");
        if (context.producer_ready == nullptr)
        {
            if (cudaEventCreateWithFlags(&context.producer_ready,
                                         cudaEventDisableTiming) != cudaSuccess)
                throw std::runtime_error(
                    "MLIP edge output producer event creation failed");
            context.device = device;
        }
        if (context.device != device)
            throw std::invalid_argument(
                "MLIP edge output execution device changed");
        context.busy = true;
        auto status =
            cudaEventRecord(context.producer_ready,
                            reinterpret_cast<deviceStream_t>(producer_stream));
        if (status == cudaSuccess)
            status = cudaEventSynchronize(context.producer_ready);
        if (status != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(status));
        edge_output_.Assemble(graph, domain_->ghost_numbers, domain_->crd,
                              md_->pbc.boundary.cell, selected_count,
                              selected_ids, gradients_model, need_virial != 0,
                              length_to_angstrom);
        // Always participate, even with no selected edges or no owned atoms.
        // Never reverse dd.frc: it also contains classical force contributions.
        domain_->Accumulate_Ghost_Tensor(controller_,
                                         edge_output_.Local_Forces(), 3);
        edge_output_.Copy_Outputs(owned_forces_model, virial_model_9,
                                  need_virial != 0);
        status = deviceStreamSynchronize(0);
        if (status != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(status));
        context.busy = false;
        return 0;
    }
    catch (const std::exception& error)
    {
        // Abort collectively rather than leave peers waiting in force reverse.
        controller_->Throw_SPONGE_Error(spongeErrorSimulationBreakDown,
                                        "PluginAssembleSelectedEdgeGradients",
                                        error.what());
        return 1;
    }
}
#endif

void LOCAL_MP_SERVICE::Invalidate_Neighbor_View()
{
    neighbor_view_ = {};
    neighbor_domain_epoch_ = 0;
    neighbor_ghost_count_ = 0;
}

void LOCAL_MP_SERVICE::Publish_Neighbor_View(const MLIP_NEIGHBOR_LIST& graph,
                                             int owned_count, int ghost_count)
{
    Invalidate_Neighbor_View();
    if (domain_ == nullptr || graph.Get_Neighbor_Epoch() == 0 ||
        owned_count != domain_->atom_numbers ||
        ghost_count != domain_->ghost_numbers)
        throw std::invalid_argument("MLIP graph publication domain mismatch");
    neighbor_view_ = {graph.Get_Neighbor_Epoch(),     owned_count,
                      graph.Get_Edge_Count(),         graph.Get_Row_Ptr(),
                      graph.Get_Source_Local_Index(), graph.Get_Image_Shift()};
    neighbor_ghost_count_ = ghost_count;
    neighbor_domain_epoch_ = domain_epoch_;
}

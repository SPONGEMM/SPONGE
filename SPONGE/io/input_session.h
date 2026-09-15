#pragma once

#include "../utils/h5md/input_context.hpp"
#include "runtime_state.h"

namespace SpongeIO
{
class InputSession
{
   public:
    explicit InputSession(const RuntimeState& runtime) : runtime_(runtime) {}

    void Load_System();
    // Thermostats and barostats must already be initialized.
    void Restore_Dynamic_State();
    void Prepare_Metadynamics();
    // Called after MetaD and other protocol modules have been initialized.
    void Restore_Protocol_State();
    void Finish_Initialization() { input_.Clear(); }

   private:
    void Validate_H5_Input_Plan();
    void Materialize_H5_Native_Topology_Core();
    void Materialize_H5_Native_Topology_Forcefield();
    void Materialize_H5_Topology_And_Protocol_Sidecars();
    void Materialize_H5_Protocol_Restart_Sidecars();
    void Apply_H5_Dynamic_Integrator_State(
        const SpongeH5MD::RestartDynamicState& state);

    RuntimeState runtime_;
    SpongeH5MD::InputContext input_;
};
}  // namespace SpongeIO

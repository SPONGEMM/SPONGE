#pragma once

#include "../utils/h5md/input_context.hpp"
#include "runtime_state.h"

struct SPONGE_PLUGIN;

namespace SpongeIO
{
class InputSession
{
   public:
    explicit InputSession(const RuntimeState& runtime) : runtime_(runtime) {}

    void Load_System(SPONGE_PLUGIN& plugin);
    // Thermostats and barostats must already be initialized.
    void Restore_Dynamic_State();
    void Prepare_Metadynamics();
    // Called after MetaD and other protocol modules have been initialized.
    void Restore_Protocol_State();
    void Finish_Initialization() { input_.Clear(); }

   private:
    void Load_Structure(SPONGE_PLUGIN& plugin);
    void Validate_H5_Input_Plan();
    void Apply_H5_Dynamic_Integrator_State(
        const SpongeH5MD::RestartDynamicState& state);

    RuntimeState runtime_;
    SpongeH5MD::InputContext input_;
};
}  // namespace SpongeIO

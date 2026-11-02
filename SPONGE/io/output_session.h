#pragma once

#include "restart_output_state.h"
#include "runtime_state.h"

namespace SpongeIO
{
// Synchronous runtime routing for canonical H5 and explicit legacy outputs.
// Events are invoked at their original simulation phases, not rescheduled.
class OutputSession
{
   public:
    explicit OutputSession(const RuntimeState& runtime) : runtime_(runtime) {}
    void Initial();
    bool Observables_Due();
    bool Trajectory_Due();
    bool Force_Due();
    bool Restart_Due();
    void Write_Reaxff();
    void Write_Metadynamics_Scalars();
    void Write_Observables();
    void Write_Trajectory();
    void Write_Force();
    void Write_Restart();
    void Write_Metadynamics_Hills();
    void Write_Pending_Sits();
    void Publish();
    void Close();

   private:
    SpongeH5MD::RestartDynamicState Build_H5_Dynamic_Restart_State();
    RestartOutputState Capture_Restart_State();
    RuntimeState runtime_;
};
}  // namespace SpongeIO

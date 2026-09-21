#pragma once

#include "../common.h"
#include "initial_velocity.h"

namespace SpongeIO
{
struct RuntimeState;
class InputSession;
}  // namespace SpongeIO

struct BERENDSEN_THERMOSTAT_INFORMATION;
struct CONSTRAIN;
struct MATRIX_CONSTRAINT;
struct VIRTUAL_INFORMATION;
struct Particle_Mesh;
struct SPONGE_PLUGIN;
struct STEER_CV;
struct RESTRAIN_CV;

namespace SpongeRuntime
{
struct InitialStateBindings
{
    SpongeIO::RuntimeState& runtime;
    SpongeIO::InputSession& input;
    BERENDSEN_THERMOSTAT_INFORMATION& bd_thermo;
    CONSTRAIN& constrain;
    SETTLE& settle;
    SHAKE& shake;
    MATRIX_CONSTRAINT& matrix_constraint;
    VIRTUAL_INFORMATION& vatom;
    Particle_Mesh& pm;
    SPONGE_PLUGIN& plugin;
    STEER_CV& steer_cv;
    RESTRAIN_CV& restrain_cv;
};

// Builds the initial physical state in dependency order. Force-field parameter
// loading and backend resource initialization stay with their owning modules.
// The bindings and the objects they reference must outlive this builder.
class InitialStateBuilder
{
   public:
    explicit InitialStateBuilder(const InitialStateBindings& bindings)
        : bindings_(bindings)
    {
    }
    InitialStateBuilder(const InitialStateBuilder&) = delete;
    InitialStateBuilder& operator=(const InitialStateBuilder&) = delete;

    void Load_Base_State();
    void Initialize_Dynamics(float (*box_updater)(LTMatrix3, int, int, int));
    // Requires force-field connectivity and restraint initialization.
    void Build_Constraints_And_Velocities();
    void Restore_Protocol_State();
    // Requires protocol validation and plugin After_Initial hooks to finish.
    void Build_Coordinate_Derivatives();
    void Distribute_State(void (*prepare_processes)(),
                          void (*refresh_local_state)(bool));
    void Finalize_Run_Range();

   private:
    enum class Stage
    {
        empty,
        base,
        dynamics,
        geometry,
        protocol,
        coordinates,
        distributed,
        ready
    };

    void Require_Stage(Stage expected, const char* operation);
    bool Thermostat_Is(const char* name);
    bool Barostat_Is(const char* name);

    InitialStateBindings bindings_;
    INITIAL_VELOCITY_INFORMATION initial_velocity_;
    Stage stage_ = Stage::empty;
};
}  // namespace SpongeRuntime

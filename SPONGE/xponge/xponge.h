#pragma once

#include "../control.h"
#include "./ir/forcefield.h"
#include "./ir/md_core.h"
#include "./ir/protocol.h"

namespace SpongeH5MD
{
class InputContext;
}

namespace Xponge
{

enum class InputSource
{
    kUnknown,
    kNative,
    kAmber,
    kGromacs,
};

struct System
{
    Atoms atoms;
    Box box;
    Residues residues;
    Exclusions exclusions;
    ClassicalForceField classical_force_field;
    GeneralizedBorn generalized_born;
    VirtualAtoms virtual_atoms;
    PositionalRestraint positional_restraint;
    InputSource source = InputSource::kUnknown;
    double start_time = 0.0;

    void Load_Inputs(CONTROLLER* controller);
    // The context must already contain a validated launch plan.
    void Load_Inputs(CONTROLLER* controller, SpongeH5MD::InputContext& input);
};

void Load_Native_Inputs(System* system, CONTROLLER* controller);
void Load_Amber_Inputs(System* system, CONTROLLER* controller);
void Load_Gromacs_Inputs(System* system, CONTROLLER* controller);
extern System system;

}  // namespace Xponge

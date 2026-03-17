#pragma once

#include "../control.h"
#include "./ir/forcefield.h"
#include "./ir/md_core.h"

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
    InputSource source = InputSource::kUnknown;
    double start_time = 0.0;

    void Load_Inputs(CONTROLLER* controller);
};

void Load_Native_Inputs(System* system, CONTROLLER* controller);
void Load_Amber_Inputs(System* system, CONTROLLER* controller);
void Load_Gromacs_Inputs(System* system, CONTROLLER* controller);
extern System system;

}  // namespace Xponge

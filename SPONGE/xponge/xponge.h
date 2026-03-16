#pragma once

#include "../control.h"
#include "./ir/md_core.h"

namespace Xponge {

struct System {
    Atoms atoms;
    Box box;
    Residues residues;
    Exclusions exclusions;

    void Load_Inputs(CONTROLLER* controller);
};

void Load_Native_Inputs(System* system, CONTROLLER* controller);
void Load_Amber_Inputs(System* system, CONTROLLER* controller);
extern System system;

}  // namespace Xponge

#pragma once

#include "./common.hpp"
#include "./native/angle.hpp"
#include "./native/bond.hpp"
#include "./native/cmap.hpp"
#include "./native/dihedral.hpp"
#include "./native/gb.hpp"
#include "./native/improper_dihedral.hpp"
#include "./native/lj.hpp"
#include "./native/lj_soft.hpp"
#include "./native/md_core.hpp"
#include "./native/nb14.hpp"
#include "./native/urey_bradley.hpp"
#include "./native/virtual_atoms.hpp"

namespace Xponge
{

static void Native_Load_Classical_Force_Field(System* system,
                                              CONTROLLER* controller)
{
    Native_Load_Bonds(system, controller);
    Native_Load_Angles(system, controller);
    Native_Load_Dihedrals(system, controller);
    Native_Load_Impropers(system, controller);
    Native_Load_LJ(system, controller);
    Native_Load_NB14(system, controller);
    Native_Load_Urey_Bradley(system, controller);
    Native_Load_CMap(system, controller);
    Native_Load_LJ_Soft_Core(system, controller);
}

void Load_Native_Inputs(System* system, CONTROLLER* controller)
{
    system->source = InputSource::kNative;
    Native_Load_Mass(system, controller);
    Native_Load_Charge(system, controller);
    Native_Load_Coordinate_And_Velocity(system, controller);
    Native_Load_Residues(system, controller);
    Native_Load_Exclusions(system, controller);
    system->generalized_born = GeneralizedBorn{};
    system->virtual_atoms = VirtualAtoms{};
    Load_Reset_Classical_Force_Field(&system->classical_force_field);
    Native_Load_Classical_Force_Field(system, controller);
    Native_Load_Generalized_Born(system, controller);
    Native_Load_Virtual_Atoms(system, controller);
}

}  // namespace Xponge

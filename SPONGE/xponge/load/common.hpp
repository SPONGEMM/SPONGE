#pragma once

#include "../xponge.h"

namespace Xponge
{

static int Load_Get_Atom_Numbers(const System* system)
{
    if (!system->atoms.mass.empty())
    {
        return static_cast<int>(system->atoms.mass.size());
    }
    if (!system->atoms.charge.empty())
    {
        return static_cast<int>(system->atoms.charge.size());
    }
    if (!system->atoms.coordinate.empty())
    {
        return static_cast<int>(system->atoms.coordinate.size() / 3);
    }
    if (!system->atoms.velocity.empty())
    {
        return static_cast<int>(system->atoms.velocity.size() / 3);
    }
    return 0;
}

static int Load_Ensure_Atom_Numbers(System* system, int atom_numbers,
                                    CONTROLLER* controller,
                                    const char* error_by)
{
    int current_atom_numbers = Load_Get_Atom_Numbers(system);
    if (current_atom_numbers > 0 && current_atom_numbers != atom_numbers)
    {
        controller->Throw_SPONGE_Error(spongeErrorConflictingCommand, error_by,
                                       "Reason:\n\t'atom_numbers' is different "
                                       "in different input files\n");
    }
    return atom_numbers;
}

static void Load_Reset_Classical_Force_Field(ClassicalForceField* ff)
{
    ff->bonds = Bonds{};
    ff->angles = Angles{};
    ff->dihedrals = Torsions{};
    ff->impropers = Torsions{};
    ff->nb14 = NB14{};
    ff->lj = LennardJones{};
    ff->cmap = CMap{};
    ff->urey_bradley = UreyBradley{};
    ff->lj_soft_core = LJSoftCore{};
}

}  // namespace Xponge

#pragma once

#include "../common.hpp"

namespace Xponge
{

static void Native_Load_Dihedrals(Torsions* dihedrals, CONTROLLER* controller,
                                  const char* module_name = "dihedral")
{
    if (!controller->Command_Exist(module_name, "in_file"))
    {
        return;
    }

    FILE* fp = NULL;
    Open_File_Safely(&fp, controller->Command(module_name, "in_file"), "r");
    int dihedral_numbers = 0;
    if (fscanf(fp, "%d", &dihedral_numbers) != 1)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorBadFileFormat, "Xponge::Native_Load_Dihedrals",
            "Reason:\n\tthe format of dihedral_in_file is not right\n");
    }
    dihedrals->atom_a.resize(dihedral_numbers);
    dihedrals->atom_b.resize(dihedral_numbers);
    dihedrals->atom_c.resize(dihedral_numbers);
    dihedrals->atom_d.resize(dihedral_numbers);
    dihedrals->pk.resize(dihedral_numbers);
    dihedrals->pn.resize(dihedral_numbers);
    dihedrals->ipn.resize(dihedral_numbers);
    dihedrals->gamc.resize(dihedral_numbers);
    dihedrals->gams.resize(dihedral_numbers);
    for (int i = 0; i < dihedral_numbers; i++)
    {
        float phase = 0.0f;
        if (fscanf(fp, "%d %d %d %d %d %f %f", &dihedrals->atom_a[i],
                   &dihedrals->atom_b[i], &dihedrals->atom_c[i],
                   &dihedrals->atom_d[i], &dihedrals->ipn[i], &dihedrals->pk[i],
                   &phase) != 7)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat, "Xponge::Native_Load_Dihedrals",
                "Reason:\n\tthe format of dihedral_in_file is not right\n");
        }
        dihedrals->pn[i] = static_cast<float>(dihedrals->ipn[i]);
        dihedrals->gamc[i] = cosf(phase) * dihedrals->pk[i];
        dihedrals->gams[i] = sinf(phase) * dihedrals->pk[i];
    }
    fclose(fp);
}

static void Native_Load_Dihedrals(System* system, CONTROLLER* controller)
{
    Native_Load_Dihedrals(&system->classical_force_field.dihedrals, controller);
}

}  // namespace Xponge

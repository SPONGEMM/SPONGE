#pragma once

#include "../common.hpp"

namespace Xponge
{

static void Native_Load_Impropers(Torsions* impropers, CONTROLLER* controller,
                                  const char* module_name = "improper_dihedral")
{
    if (!controller->Command_Exist(module_name, "in_file"))
    {
        return;
    }

    FILE* fp = NULL;
    Open_File_Safely(&fp, controller->Command(module_name, "in_file"), "r");
    int improper_numbers = 0;
    if (fscanf(fp, "%d", &improper_numbers) != 1)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorBadFileFormat, "Xponge::Native_Load_Impropers",
            "Reason:\n\tthe format of improper_dihedral_in_file is not "
            "right\n");
    }
    impropers->atom_a.resize(improper_numbers);
    impropers->atom_b.resize(improper_numbers);
    impropers->atom_c.resize(improper_numbers);
    impropers->atom_d.resize(improper_numbers);
    impropers->pk.resize(improper_numbers);
    impropers->pn.assign(improper_numbers, 0.0f);
    impropers->ipn.assign(improper_numbers, 0);
    impropers->gamc.resize(improper_numbers);
    impropers->gams.assign(improper_numbers, 0.0f);
    for (int i = 0; i < improper_numbers; i++)
    {
        if (fscanf(fp, "%d %d %d %d %f %f", &impropers->atom_a[i],
                   &impropers->atom_b[i], &impropers->atom_c[i],
                   &impropers->atom_d[i], &impropers->pk[i],
                   &impropers->gamc[i]) != 6)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat, "Xponge::Native_Load_Impropers",
                "Reason:\n\tthe format of improper_dihedral_in_file is not "
                "right\n");
        }
    }
    fclose(fp);
}

static void Native_Load_Impropers(System* system, CONTROLLER* controller)
{
    Native_Load_Impropers(&system->classical_force_field.impropers, controller);
}

}  // namespace Xponge

#pragma once

#include "../common.hpp"

namespace Xponge
{

static void Native_Load_Bonds(Bonds* bonds, CONTROLLER* controller,
                              const char* module_name = "bond")
{
    if (!controller->Command_Exist(module_name, "in_file"))
    {
        return;
    }

    FILE* fp = NULL;
    Open_File_Safely(&fp, controller->Command(module_name, "in_file"), "r");
    int bond_numbers = 0;
    if (fscanf(fp, "%d", &bond_numbers) != 1)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorBadFileFormat, "Xponge::Native_Load_Bonds",
            "Reason:\n\tthe format of bond_in_file is not right\n");
    }
    bonds->atom_a.resize(bond_numbers);
    bonds->atom_b.resize(bond_numbers);
    bonds->k.resize(bond_numbers);
    bonds->r0.resize(bond_numbers);
    for (int i = 0; i < bond_numbers; i++)
    {
        if (fscanf(fp, "%d %d %f %f", &bonds->atom_a[i], &bonds->atom_b[i],
                   &bonds->k[i], &bonds->r0[i]) != 4)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat, "Xponge::Native_Load_Bonds",
                "Reason:\n\tthe format of bond_in_file is not right\n");
        }
    }
    fclose(fp);
}

static void Native_Load_Bonds(System* system, CONTROLLER* controller)
{
    Native_Load_Bonds(&system->classical_force_field.bonds, controller);
}

}  // namespace Xponge

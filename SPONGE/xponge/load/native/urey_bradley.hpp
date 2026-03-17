#pragma once

#include "../common.hpp"

namespace Xponge
{

static void Native_Load_Urey_Bradley(UreyBradley* urey_bradley,
                                     CONTROLLER* controller,
                                     const char* module_name = "urey_bradley")
{
    if (!controller->Command_Exist(module_name, "in_file"))
    {
        return;
    }

    FILE* fp = NULL;
    Open_File_Safely(&fp, controller->Command(module_name, "in_file"), "r");
    int term_numbers = 0;
    if (fscanf(fp, "%d", &term_numbers) != 1)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorBadFileFormat, "Xponge::Native_Load_Urey_Bradley",
            "Reason:\n\tthe format of urey_bradley_in_file is not right\n");
    }

    urey_bradley->atom_a.resize(term_numbers);
    urey_bradley->atom_b.resize(term_numbers);
    urey_bradley->atom_c.resize(term_numbers);
    urey_bradley->angle_k.resize(term_numbers);
    urey_bradley->angle_theta0.resize(term_numbers);
    urey_bradley->bond_k.resize(term_numbers);
    urey_bradley->bond_r0.resize(term_numbers);

    for (int i = 0; i < term_numbers; i++)
    {
        if (fscanf(fp, "%d %d %d %f %f %f %f", &urey_bradley->atom_a[i],
                   &urey_bradley->atom_b[i], &urey_bradley->atom_c[i],
                   &urey_bradley->angle_k[i], &urey_bradley->angle_theta0[i],
                   &urey_bradley->bond_k[i], &urey_bradley->bond_r0[i]) != 7)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat, "Xponge::Native_Load_Urey_Bradley",
                "Reason:\n\tthe format of urey_bradley_in_file is not right\n");
        }
    }
    fclose(fp);
}

static void Native_Load_Urey_Bradley(System* system, CONTROLLER* controller)
{
    Native_Load_Urey_Bradley(&system->classical_force_field.urey_bradley,
                             controller);
}

}  // namespace Xponge

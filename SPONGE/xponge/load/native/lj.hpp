#pragma once

#include "../common.hpp"

namespace Xponge
{

static void Native_Load_LJ(LennardJones* lj, CONTROLLER* controller,
                           int atom_numbers_hint = 0,
                           const char* module_name = "LJ")
{
    if (!controller->Command_Exist(module_name, "in_file"))
    {
        return;
    }

    FILE* fp = NULL;
    Open_File_Safely(&fp, controller->Command(module_name, "in_file"), "r");
    int atom_numbers = 0;
    int atom_type_numbers = 0;
    if (fscanf(fp, "%d %d", &atom_numbers, &atom_type_numbers) != 2)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorBadFileFormat, "Xponge::Native_Load_LJ",
            "Reason:\n\tthe format of LJ_in_file is not right\n");
    }
    if (atom_numbers_hint > 0 && atom_numbers_hint != atom_numbers)
    {
        controller->Throw_SPONGE_Error(spongeErrorConflictingCommand,
                                       "Xponge::Native_Load_LJ",
                                       "Reason:\n\t'atom_numbers' is different "
                                       "in different input files\n");
    }
    lj->atom_type_numbers = atom_type_numbers;
    int pair_type_numbers = atom_type_numbers * (atom_type_numbers + 1) / 2;
    lj->pair_A.resize(pair_type_numbers);
    lj->pair_B.resize(pair_type_numbers);
    lj->atom_type.resize(atom_numbers);
    for (int i = 0; i < pair_type_numbers; i++)
    {
        if (fscanf(fp, "%f", &lj->pair_A[i]) != 1)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat, "Xponge::Native_Load_LJ",
                "Reason:\n\tthe format of LJ_in_file is not right\n");
        }
        lj->pair_A[i] *= 12.0f;
    }
    for (int i = 0; i < pair_type_numbers; i++)
    {
        if (fscanf(fp, "%f", &lj->pair_B[i]) != 1)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat, "Xponge::Native_Load_LJ",
                "Reason:\n\tthe format of LJ_in_file is not right\n");
        }
        lj->pair_B[i] *= 6.0f;
    }
    for (int i = 0; i < atom_numbers; i++)
    {
        if (fscanf(fp, "%d", &lj->atom_type[i]) != 1)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat, "Xponge::Native_Load_LJ",
                "Reason:\n\tthe format of LJ_in_file is not right\n");
        }
    }
    fclose(fp);
}

static void Native_Load_LJ(System* system, CONTROLLER* controller)
{
    Native_Load_LJ(&system->classical_force_field.lj, controller,
                   Load_Get_Atom_Numbers(system));
}

}  // namespace Xponge

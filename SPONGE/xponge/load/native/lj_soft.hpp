#pragma once

#include "../common.hpp"

namespace Xponge
{

static void Native_Load_LJ_Soft_Core(LJSoftCore* lj_soft,
                                     CONTROLLER* controller,
                                     const char* module_name = "LJ_soft_core")
{
    if (!controller->Command_Exist(module_name, "in_file"))
    {
        return;
    }

    FILE* fp = NULL;
    Open_File_Safely(&fp, controller->Command(module_name, "in_file"), "r");
    if (fscanf(fp, "%d %d %d", &lj_soft->atom_numbers,
               &lj_soft->atom_type_numbers_A,
               &lj_soft->atom_type_numbers_B) != 3)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorBadFileFormat, "Xponge::Native_Load_LJ_Soft_Core",
            "Reason:\n\tthe format of LJ_soft_core_in_file is not right\n");
    }

    int pair_type_numbers_A =
        lj_soft->atom_type_numbers_A * (lj_soft->atom_type_numbers_A + 1) / 2;
    int pair_type_numbers_B =
        lj_soft->atom_type_numbers_B * (lj_soft->atom_type_numbers_B + 1) / 2;
    lj_soft->LJ_AA.resize(pair_type_numbers_A);
    lj_soft->LJ_AB.resize(pair_type_numbers_A);
    lj_soft->LJ_BA.resize(pair_type_numbers_B);
    lj_soft->LJ_BB.resize(pair_type_numbers_B);
    lj_soft->atom_LJ_type_A.resize(lj_soft->atom_numbers);
    lj_soft->atom_LJ_type_B.resize(lj_soft->atom_numbers);

    for (int i = 0; i < pair_type_numbers_A; i++)
    {
        if (fscanf(fp, "%f", &lj_soft->LJ_AA[i]) != 1)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat, "Xponge::Native_Load_LJ_Soft_Core",
                "Reason:\n\tthe format of LJ_soft_core_in_file is not right\n");
        }
        lj_soft->LJ_AA[i] *= 12.0f;
    }
    for (int i = 0; i < pair_type_numbers_A; i++)
    {
        if (fscanf(fp, "%f", &lj_soft->LJ_AB[i]) != 1)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat, "Xponge::Native_Load_LJ_Soft_Core",
                "Reason:\n\tthe format of LJ_soft_core_in_file is not right\n");
        }
        lj_soft->LJ_AB[i] *= 6.0f;
    }
    for (int i = 0; i < pair_type_numbers_B; i++)
    {
        if (fscanf(fp, "%f", &lj_soft->LJ_BA[i]) != 1)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat, "Xponge::Native_Load_LJ_Soft_Core",
                "Reason:\n\tthe format of LJ_soft_core_in_file is not right\n");
        }
        lj_soft->LJ_BA[i] *= 12.0f;
    }
    for (int i = 0; i < pair_type_numbers_B; i++)
    {
        if (fscanf(fp, "%f", &lj_soft->LJ_BB[i]) != 1)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat, "Xponge::Native_Load_LJ_Soft_Core",
                "Reason:\n\tthe format of LJ_soft_core_in_file is not right\n");
        }
        lj_soft->LJ_BB[i] *= 6.0f;
    }
    for (int i = 0; i < lj_soft->atom_numbers; i++)
    {
        if (fscanf(fp, "%d %d", &lj_soft->atom_LJ_type_A[i],
                   &lj_soft->atom_LJ_type_B[i]) != 2)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat, "Xponge::Native_Load_LJ_Soft_Core",
                "Reason:\n\tthe format of LJ_soft_core_in_file is not right\n");
        }
    }
    fclose(fp);

    if (controller->Command_Exist("subsys_division_in_file"))
    {
        lj_soft->subsystem_division.resize(lj_soft->atom_numbers);
        fp = NULL;
        Open_File_Safely(&fp, controller->Command("subsys_division_in_file"),
                         "r");
        int atom_numbers = 0;
        if (fscanf(fp, "%d", &atom_numbers) != 1)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat, "Xponge::Native_Load_LJ_Soft_Core",
                "Reason:\n\tthe format of subsys_division_in_file is not "
                "right\n");
        }
        if (atom_numbers != lj_soft->atom_numbers)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorConflictingCommand,
                "Xponge::Native_Load_LJ_Soft_Core",
                "Reason:\n\t'atom_numbers' is different in different input "
                "files\n");
        }
        for (int i = 0; i < atom_numbers; i++)
        {
            if (fscanf(fp, "%d", &lj_soft->subsystem_division[i]) != 1)
            {
                controller->Throw_SPONGE_Error(
                    spongeErrorBadFileFormat,
                    "Xponge::Native_Load_LJ_Soft_Core",
                    "Reason:\n\tthe format of subsys_division_in_file is not "
                    "right\n");
            }
        }
        fclose(fp);
    }
}

static void Native_Load_LJ_Soft_Core(System* system, CONTROLLER* controller)
{
    Native_Load_LJ_Soft_Core(&system->classical_force_field.lj_soft_core,
                             controller);
}

}  // namespace Xponge

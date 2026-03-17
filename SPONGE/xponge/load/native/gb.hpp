#pragma once

#include "../common.hpp"

namespace Xponge
{

static void Native_Load_Generalized_Born(GeneralizedBorn* gb,
                                         CONTROLLER* controller,
                                         const char* module_name = "gb")
{
    if (!controller->Command_Exist(module_name, "in_file"))
    {
        return;
    }

    FILE* fp = NULL;
    Open_File_Safely(&fp, controller->Command(module_name, "in_file"), "r");
    int atom_numbers = 0;
    if (fscanf(fp, "%d", &atom_numbers) != 1)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorBadFileFormat, "Xponge::Native_Load_Generalized_Born",
            "Reason:\n\tthe format of gb_in_file is not right\n");
    }
    gb->radius.resize(atom_numbers);
    gb->scale_factor.resize(atom_numbers);
    for (int i = 0; i < atom_numbers; i++)
    {
        if (fscanf(fp, "%f %f", &gb->radius[i], &gb->scale_factor[i]) != 2)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat,
                "Xponge::Native_Load_Generalized_Born",
                "Reason:\n\tthe format of gb_in_file is not right\n");
        }
    }
    fclose(fp);
}

static void Native_Load_Generalized_Born(System* system, CONTROLLER* controller)
{
    Native_Load_Generalized_Born(&system->generalized_born, controller);
}

}  // namespace Xponge

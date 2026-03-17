#pragma once

#include "../common.hpp"

namespace Xponge
{

static void Native_Load_Angles(Angles* angles, CONTROLLER* controller,
                               const char* module_name = "angle")
{
    if (!controller->Command_Exist(module_name, "in_file"))
    {
        return;
    }

    FILE* fp = NULL;
    Open_File_Safely(&fp, controller->Command(module_name, "in_file"), "r");
    int angle_numbers = 0;
    if (fscanf(fp, "%d", &angle_numbers) != 1)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorBadFileFormat, "Xponge::Native_Load_Angles",
            "Reason:\n\tthe format of angle_in_file is not right\n");
    }
    angles->atom_a.resize(angle_numbers);
    angles->atom_b.resize(angle_numbers);
    angles->atom_c.resize(angle_numbers);
    angles->k.resize(angle_numbers);
    angles->theta0.resize(angle_numbers);
    for (int i = 0; i < angle_numbers; i++)
    {
        if (fscanf(fp, "%d %d %d %f %f", &angles->atom_a[i], &angles->atom_b[i],
                   &angles->atom_c[i], &angles->k[i], &angles->theta0[i]) != 5)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat, "Xponge::Native_Load_Angles",
                "Reason:\n\tthe format of angle_in_file is not right\n");
        }
    }
    fclose(fp);
}

static void Native_Load_Angles(System* system, CONTROLLER* controller)
{
    Native_Load_Angles(&system->classical_force_field.angles, controller);
}

}  // namespace Xponge

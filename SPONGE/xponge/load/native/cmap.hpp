#pragma once

#include "../common.hpp"

namespace Xponge
{

static void Native_Load_CMap(CMap* cmap, CONTROLLER* controller,
                             const char* module_name = "cmap")
{
    if (!controller->Command_Exist(module_name, "in_file"))
    {
        return;
    }

    FILE* fp = NULL;
    Open_File_Safely(&fp, controller->Command(module_name, "in_file"), "r");
    int total_cmap_numbers = 0;
    if (fscanf(fp, "%d", &total_cmap_numbers) != 1)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorBadFileFormat, "Xponge::Native_Load_CMap",
            "Reason:\n\tthe format of cmap_in_file is not right\n");
    }
    cmap->unique_type_numbers = 0;
    if (fscanf(fp, "%d", &cmap->unique_type_numbers) != 1)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorBadFileFormat, "Xponge::Native_Load_CMap",
            "Reason:\n\tthe format of cmap_in_file is not right\n");
    }

    cmap->atom_a.resize(total_cmap_numbers);
    cmap->atom_b.resize(total_cmap_numbers);
    cmap->atom_c.resize(total_cmap_numbers);
    cmap->atom_d.resize(total_cmap_numbers);
    cmap->atom_e.resize(total_cmap_numbers);
    cmap->cmap_type.resize(total_cmap_numbers);
    cmap->resolution.resize(cmap->unique_type_numbers);
    cmap->type_offset.resize(cmap->unique_type_numbers);
    cmap->unique_gridpoint_numbers = 0;

    for (int i = 0; i < cmap->unique_type_numbers; i++)
    {
        if (fscanf(fp, "%d", &cmap->resolution[i]) != 1)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat, "Xponge::Native_Load_CMap",
                "Reason:\n\tthe format of cmap_in_file is not right\n");
        }
        cmap->type_offset[i] = 16 * cmap->unique_gridpoint_numbers;
        cmap->unique_gridpoint_numbers +=
            cmap->resolution[i] * cmap->resolution[i];
    }

    cmap->grid_value.resize(cmap->unique_gridpoint_numbers);
    for (int i = 0; i < cmap->unique_gridpoint_numbers; i++)
    {
        if (fscanf(fp, "%f", &cmap->grid_value[i]) != 1)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat, "Xponge::Native_Load_CMap",
                "Reason:\n\tthe format of cmap_in_file is not right\n");
        }
    }

    for (int i = 0; i < total_cmap_numbers; i++)
    {
        if (fscanf(fp, "%d %d %d %d %d %d", &cmap->atom_a[i], &cmap->atom_b[i],
                   &cmap->atom_c[i], &cmap->atom_d[i], &cmap->atom_e[i],
                   &cmap->cmap_type[i]) != 6)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat, "Xponge::Native_Load_CMap",
                "Reason:\n\tthe format of cmap_in_file is not right\n");
        }
    }
    fclose(fp);
}

static void Native_Load_CMap(System* system, CONTROLLER* controller)
{
    Native_Load_CMap(&system->classical_force_field.cmap, controller);
}

}  // namespace Xponge

#pragma once

#include "../common.hpp"

namespace Xponge
{

static void Native_Load_Mass(System* system, CONTROLLER* controller)
{
    if (controller->Command_Exist("mass_in_file"))
    {
        FILE* fp = NULL;
        Open_File_Safely(&fp, controller->Command("mass_in_file"), "r");
        int atom_numbers = 0;
        if (fscanf(fp, "%d", &atom_numbers) != 1)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat, "Xponge::Native_Load_Mass",
                "Reason:\n\tthe format of mass_in_file is not right\n");
        }
        Load_Ensure_Atom_Numbers(system, atom_numbers, controller,
                                 "Xponge::Native_Load_Mass");
        system->atoms.mass.resize(atom_numbers);
        for (int i = 0; i < atom_numbers; i++)
        {
            if (fscanf(fp, "%f", &system->atoms.mass[i]) != 1)
            {
                controller->Throw_SPONGE_Error(
                    spongeErrorBadFileFormat, "Xponge::Native_Load_Mass",
                    "Reason:\n\tthe format of mass_in_file is not right\n");
            }
        }
        fclose(fp);
        return;
    }

    int atom_numbers = Load_Get_Atom_Numbers(system);
    if (atom_numbers > 0)
    {
        system->atoms.mass.assign(atom_numbers, 20.0f);
        return;
    }

    controller->Throw_SPONGE_Error(
        spongeErrorMissingCommand, "Xponge::Native_Load_Mass",
        "Reason:\n\tno atom_numbers found. No mass_in_file is provided\n");
}

static void Native_Load_Charge(System* system, CONTROLLER* controller)
{
    if (controller->Command_Exist("charge_in_file"))
    {
        FILE* fp = NULL;
        Open_File_Safely(&fp, controller->Command("charge_in_file"), "r");
        int atom_numbers = 0;
        if (fscanf(fp, "%d", &atom_numbers) != 1)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat, "Xponge::Native_Load_Charge",
                "Reason:\n\tthe format of charge_in_file is not right\n");
        }
        Load_Ensure_Atom_Numbers(system, atom_numbers, controller,
                                 "Xponge::Native_Load_Charge");
        system->atoms.charge.resize(atom_numbers);
        for (int i = 0; i < atom_numbers; i++)
        {
            if (fscanf(fp, "%f", &system->atoms.charge[i]) != 1)
            {
                controller->Throw_SPONGE_Error(
                    spongeErrorBadFileFormat, "Xponge::Native_Load_Charge",
                    "Reason:\n\tthe format of charge_in_file is not right\n");
            }
        }
        fclose(fp);
        return;
    }

    int atom_numbers = Load_Get_Atom_Numbers(system);
    if (atom_numbers > 0)
    {
        system->atoms.charge.assign(atom_numbers, 0.0f);
        return;
    }

    controller->Throw_SPONGE_Error(
        spongeErrorMissingCommand, "Xponge::Native_Load_Charge",
        "Reason:\n\tno atom_numbers found. No charge_in_file is provided\n");
}

static void Native_Load_Coordinate_And_Velocity(System* system,
                                                CONTROLLER* controller)
{
    if (!controller->Command_Exist("coordinate_in_file"))
    {
        controller->Throw_SPONGE_Error(
            spongeErrorMissingCommand,
            "Xponge::Native_Load_Coordinate_And_Velocity",
            "Reason:\n\tno coordinate information found");
    }

    FILE* fp = NULL;
    Open_File_Safely(&fp, controller->Command("coordinate_in_file"), "r");

    char line[CHAR_LENGTH_MAX];
    if (fgets(line, CHAR_LENGTH_MAX, fp) == NULL)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorBadFileFormat,
            "Xponge::Native_Load_Coordinate_And_Velocity",
            "Reason:\n\tthe format of coordinate_in_file is not right\n");
    }
    int atom_numbers = 0;
    double start_time = 0.0;
    int scanf_ret = sscanf(line, "%d %lf", &atom_numbers, &start_time);
    if (scanf_ret < 1)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorBadFileFormat,
            "Xponge::Native_Load_Coordinate_And_Velocity",
            "Reason:\n\tthe format of coordinate_in_file is not right\n");
    }
    system->start_time = (scanf_ret == 2) ? start_time : 0.0;
    Load_Ensure_Atom_Numbers(system, atom_numbers, controller,
                             "Xponge::Native_Load_Coordinate_And_Velocity");

    system->atoms.coordinate.resize(3 * atom_numbers);
    for (int i = 0; i < atom_numbers; i++)
    {
        if (fscanf(fp, "%f %f %f", &system->atoms.coordinate[3 * i],
                   &system->atoms.coordinate[3 * i + 1],
                   &system->atoms.coordinate[3 * i + 2]) != 3)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat,
                "Xponge::Native_Load_Coordinate_And_Velocity",
                "Reason:\n\tthe format of coordinate_in_file is not right\n");
        }
    }

    system->box.box_length.resize(3);
    system->box.box_angle.resize(3);
    if (fscanf(fp, "%f %f %f", &system->box.box_length[0],
               &system->box.box_length[1], &system->box.box_length[2]) != 3)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorBadFileFormat,
            "Xponge::Native_Load_Coordinate_And_Velocity",
            "Reason:\n\tthe format of coordinate_in_file is not right\n");
    }
    if (fscanf(fp, "%f %f %f", &system->box.box_angle[0],
               &system->box.box_angle[1], &system->box.box_angle[2]) != 3)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorBadFileFormat,
            "Xponge::Native_Load_Coordinate_And_Velocity",
            "Reason:\n\tthe format of coordinate_in_file is not right\n");
    }
    fclose(fp);

    if (controller->Command_Exist("velocity_in_file"))
    {
        FILE* vfp = NULL;
        Open_File_Safely(&vfp, controller->Command("velocity_in_file"), "r");
        int vel_atom_numbers = 0;
        if (fscanf(vfp, "%d", &vel_atom_numbers) != 1)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat,
                "Xponge::Native_Load_Coordinate_And_Velocity",
                "Reason:\n\tthe format of velocity_in_file is not right\n");
        }
        Load_Ensure_Atom_Numbers(system, vel_atom_numbers, controller,
                                 "Xponge::Native_Load_Coordinate_And_Velocity");
        system->atoms.velocity.resize(3 * vel_atom_numbers);
        for (int i = 0; i < vel_atom_numbers; i++)
        {
            if (fscanf(vfp, "%f %f %f", &system->atoms.velocity[3 * i],
                       &system->atoms.velocity[3 * i + 1],
                       &system->atoms.velocity[3 * i + 2]) != 3)
            {
                controller->Throw_SPONGE_Error(
                    spongeErrorBadFileFormat,
                    "Xponge::Native_Load_Coordinate_And_Velocity",
                    "Reason:\n\tthe format of velocity_in_file is not right\n");
            }
        }
        fclose(vfp);
        return;
    }

    system->atoms.velocity.assign(3 * atom_numbers, 0.0f);
}

static void Native_Load_Residues(System* system, CONTROLLER* controller)
{
    int atom_numbers = Load_Get_Atom_Numbers(system);
    if (!controller->Command_Exist("residue_in_file"))
    {
        system->residues.atom_numbers.assign(atom_numbers, 1);
        return;
    }

    FILE* fp = NULL;
    Open_File_Safely(&fp, controller->Command("residue_in_file"), "r");
    int residue_atom_numbers = 0;
    int residue_numbers = 0;
    if (fscanf(fp, "%d %d", &residue_atom_numbers, &residue_numbers) != 2)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorBadFileFormat, "Xponge::Native_Load_Residues",
            "Reason:\n\tthe format of residue_in_file is not right\n");
    }
    Load_Ensure_Atom_Numbers(system, residue_atom_numbers, controller,
                             "Xponge::Native_Load_Residues");
    system->residues.atom_numbers.resize(residue_numbers);
    for (int i = 0; i < residue_numbers; i++)
    {
        if (fscanf(fp, "%d", &system->residues.atom_numbers[i]) != 1)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat, "Xponge::Native_Load_Residues",
                "Reason:\n\tthe format of residue_in_file is not right\n");
        }
    }
    fclose(fp);
}

static void Native_Load_Exclusions(System* system, CONTROLLER* controller)
{
    int atom_numbers = Load_Get_Atom_Numbers(system);
    system->exclusions.excluded_atoms.assign(atom_numbers, {});

    if (!controller->Command_Exist("exclude_in_file"))
    {
        return;
    }

    FILE* fp = NULL;
    Open_File_Safely(&fp, controller->Command("exclude_in_file"), "r");
    int exclude_atom_numbers = 0;
    int excluded_atom_numbers = 0;
    if (fscanf(fp, "%d %d", &exclude_atom_numbers, &excluded_atom_numbers) != 2)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorBadFileFormat, "Xponge::Native_Load_Exclusions",
            "Reason:\n\tThe format of exclude_in_file is not right\n");
    }
    Load_Ensure_Atom_Numbers(system, exclude_atom_numbers, controller,
                             "Xponge::Native_Load_Exclusions");

    int count = 0;
    for (int i = 0; i < atom_numbers; i++)
    {
        int excluded_numbers = 0;
        if (fscanf(fp, "%d", &excluded_numbers) != 1)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat, "Xponge::Native_Load_Exclusions",
                "Reason:\n\tThe format of exclude_in_file is not right\n");
        }
        system->exclusions.excluded_atoms[i].resize(excluded_numbers);
        for (int j = 0; j < excluded_numbers; j++)
        {
            if (fscanf(fp, "%d", &system->exclusions.excluded_atoms[i][j]) != 1)
            {
                controller->Throw_SPONGE_Error(
                    spongeErrorBadFileFormat, "Xponge::Native_Load_Exclusions",
                    "Reason:\n\tThe format of exclude_in_file is not right\n");
            }
            count++;
        }
    }
    fclose(fp);

    if (count != excluded_atom_numbers)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorBadFileFormat, "Xponge::Native_Load_Exclusions",
            "Reason:\n\tThe format of exclude_in_file is not right "
            "(excluded_atom_numbers is not right)\n");
    }
}

}  // namespace Xponge

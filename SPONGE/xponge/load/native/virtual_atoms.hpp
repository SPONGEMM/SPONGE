#pragma once

#include <sstream>

#include "../common.hpp"

namespace Xponge
{

static void Native_Load_Virtual_Atoms(VirtualAtoms* virtual_atoms,
                                      CONTROLLER* controller,
                                      const char* module_name = "virtual_atom")
{
    if (!controller->Command_Exist(module_name, "in_file"))
    {
        return;
    }

    FILE* fp = NULL;
    Open_File_Safely(&fp, controller->Command(module_name, "in_file"), "r");
    char line[CHAR_LENGTH_MAX];
    int line_numbers = 0;
    while (fgets(line, CHAR_LENGTH_MAX, fp) != NULL)
    {
        line_numbers++;
        std::stringstream stream(line);
        VirtualAtomRecord record;
        if (!(stream >> record.type >> record.virtual_atom))
        {
            continue;
        }
        switch (record.type)
        {
            case 0:
            {
                int from = 0;
                float h = 0.0f;
                if (!(stream >> from >> h))
                {
                    controller->Throw_SPONGE_Error(
                        spongeErrorBadFileFormat,
                        "Xponge::Native_Load_Virtual_Atoms",
                        "Reason:\n\tthe format of virtual_atom_in_file is not "
                        "right\n");
                }
                record.from.push_back(from);
                record.parameter.push_back(h);
                break;
            }
            case 1:
            {
                int from1 = 0, from2 = 0;
                float a = 0.0f;
                if (!(stream >> from1 >> from2 >> a))
                {
                    controller->Throw_SPONGE_Error(
                        spongeErrorBadFileFormat,
                        "Xponge::Native_Load_Virtual_Atoms",
                        "Reason:\n\tthe format of virtual_atom_in_file is not "
                        "right\n");
                }
                record.from.push_back(from1);
                record.from.push_back(from2);
                record.parameter.push_back(a);
                break;
            }
            case 2:
            case 3:
            {
                int from1 = 0, from2 = 0, from3 = 0;
                float a = 0.0f, b = 0.0f;
                if (!(stream >> from1 >> from2 >> from3 >> a >> b))
                {
                    controller->Throw_SPONGE_Error(
                        spongeErrorBadFileFormat,
                        "Xponge::Native_Load_Virtual_Atoms",
                        "Reason:\n\tthe format of virtual_atom_in_file is not "
                        "right\n");
                }
                record.from.push_back(from1);
                record.from.push_back(from2);
                record.from.push_back(from3);
                record.parameter.push_back(a);
                record.parameter.push_back(b);
                break;
            }
            default:
                controller->Throw_SPONGE_Error(
                    spongeErrorBadFileFormat,
                    "Xponge::Native_Load_Virtual_Atoms",
                    "Reason:\n\tvirtual_atom_in_file contains an unsupported "
                    "virtual atom type\n");
        }
        virtual_atoms->records.push_back(record);
    }
    fclose(fp);
}

static void Native_Load_Virtual_Atoms(System* system, CONTROLLER* controller)
{
    Native_Load_Virtual_Atoms(&system->virtual_atoms, controller);
}

}  // namespace Xponge

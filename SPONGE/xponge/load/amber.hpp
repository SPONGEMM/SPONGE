#pragma once

#include "../xponge.h"

namespace Xponge {

static int Amber_Get_Atom_Numbers(const System* system)
{
    if (!system->atoms.mass.empty())
    {
        return static_cast<int>(system->atoms.mass.size());
    }
    if (!system->atoms.charge.empty())
    {
        return static_cast<int>(system->atoms.charge.size());
    }
    if (!system->atoms.coordinate.empty())
    {
        return static_cast<int>(system->atoms.coordinate.size() / 3);
    }
    return 0;
}

static void Amber_Ensure_Atom_Numbers(System* system, int atom_numbers,
                                      CONTROLLER* controller,
                                      const char* error_by)
{
    int current_atom_numbers = Amber_Get_Atom_Numbers(system);
    if (current_atom_numbers > 0 && current_atom_numbers != atom_numbers)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorConflictingCommand, error_by,
            "Reason:\n\t'atom_numbers' is different in different input files\n");
    }
}

static std::vector<std::string> Amber_Read_Section(
    const std::vector<std::string>& lines, std::size_t* index)
{
    std::vector<std::string> values;
    while (*index < lines.size())
    {
        const std::string& line = lines[*index];
        if (line.rfind("%FLAG", 0) == 0)
        {
            return values;
        }
        if (line.rfind("%FORMAT", 0) == 0 || line.empty())
        {
            (*index)++;
            continue;
        }
        std::istringstream iss(line);
        std::string token;
        while (iss >> token)
        {
            values.push_back(token);
        }
        (*index)++;
    }
    return values;
}

static void Amber_Load_Parm7(System* system, CONTROLLER* controller)
{
    if (!controller->Command_Exist("amber_parm7"))
    {
        return;
    }

    std::ifstream fin(controller->Command("amber_parm7"));
    if (!fin.is_open())
    {
        controller->Throw_SPONGE_Error(
            spongeErrorBadFileFormat, "Xponge::Amber_Load_Parm7",
            "Reason:\n\tfailed to open amber_parm7\n");
    }

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(fin, line))
    {
        lines.push_back(line);
    }

    int atom_numbers = 0;
    int residue_numbers = 0;
    std::vector<int> excluded_numbers;
    std::vector<int> excluded_list;

    for (std::size_t i = 0; i < lines.size(); i++)
    {
        const std::string& line = lines[i];
        if (line.rfind("%FLAG", 0) != 0)
        {
            continue;
        }
        std::string current_flag = line.substr(6);
        i++;
        std::vector<std::string> values = Amber_Read_Section(lines, &i);

        if (current_flag == "POINTERS")
        {
            if (values.size() < 12)
            {
                controller->Throw_SPONGE_Error(
                    spongeErrorBadFileFormat, "Xponge::Amber_Load_Parm7",
                    "Reason:\n\tthe format of amber_parm7 is not right\n");
            }
            atom_numbers = std::stoi(values[0]);
            residue_numbers = std::stoi(values[11]);
            Amber_Ensure_Atom_Numbers(system, atom_numbers, controller,
                                      "Xponge::Amber_Load_Parm7");
        }
        else if (current_flag == "MASS")
        {
            system->atoms.mass.resize(values.size());
            for (std::size_t i = 0; i < values.size(); i++)
            {
                system->atoms.mass[i] = std::stof(values[i]);
            }
        }
        else if (current_flag == "CHARGE")
        {
            system->atoms.charge.resize(values.size());
            for (std::size_t i = 0; i < values.size(); i++)
            {
                system->atoms.charge[i] = std::stof(values[i]);
            }
        }
        else if (current_flag == "RESIDUE_POINTER")
        {
            system->residues.atom_numbers.clear();
            if (residue_numbers == 0)
            {
                residue_numbers = static_cast<int>(values.size());
            }
            system->residues.atom_numbers.resize(residue_numbers);
            for (int i = 0; i < residue_numbers; i++)
            {
                int start = std::stoi(values[i]) - 1;
                int end = (i + 1 < residue_numbers)
                              ? std::stoi(values[i + 1]) - 1
                              : atom_numbers;
                system->residues.atom_numbers[i] = end - start;
            }
        }
        else if (current_flag == "NUMBER_EXCLUDED_ATOMS")
        {
            excluded_numbers.resize(values.size());
            for (std::size_t i = 0; i < values.size(); i++)
            {
                excluded_numbers[i] = std::stoi(values[i]);
            }
        }
        else if (current_flag == "EXCLUDED_ATOMS_LIST")
        {
            excluded_list.resize(values.size());
            for (std::size_t i = 0; i < values.size(); i++)
            {
                excluded_list[i] = std::stoi(values[i]);
            }
        }
        i--;
    }

    if (atom_numbers > 0)
    {
        Amber_Ensure_Atom_Numbers(system, atom_numbers, controller,
                                  "Xponge::Amber_Load_Parm7");
    }
    if (system->atoms.mass.empty() && atom_numbers > 0)
    {
        system->atoms.mass.assign(atom_numbers, 20.0f);
    }
    if (system->atoms.charge.empty() && atom_numbers > 0)
    {
        system->atoms.charge.assign(atom_numbers, 0.0f);
    }
    if (system->residues.atom_numbers.empty() && atom_numbers > 0)
    {
        system->residues.atom_numbers.assign(atom_numbers, 1);
    }

    system->exclusions.excluded_atoms.assign(atom_numbers, {});
    int count = 0;
    for (int i = 0; i < atom_numbers && i < static_cast<int>(excluded_numbers.size());
         i++)
    {
        for (int j = 0; j < excluded_numbers[i] && count < static_cast<int>(excluded_list.size());
             j++)
        {
            int excluded_atom = excluded_list[count++];
            if (excluded_atom == 0)
            {
                system->exclusions.excluded_atoms[i].clear();
                break;
            }
            system->exclusions.excluded_atoms[i].push_back(excluded_atom - 1);
        }
        std::sort(system->exclusions.excluded_atoms[i].begin(),
                  system->exclusions.excluded_atoms[i].end());
    }
}

static void Amber_Load_Rst7(System* system, CONTROLLER* controller)
{
    if (!controller->Command_Exist("amber_rst7"))
    {
        return;
    }

    FILE* fin = NULL;
    Open_File_Safely(&fin, controller->Command("amber_rst7"), "r");

    char line[CHAR_LENGTH_MAX];
    fgets(line, CHAR_LENGTH_MAX, fin);
    fgets(line, CHAR_LENGTH_MAX, fin);

    int atom_numbers = 0;
    double start_time = 0.0;
    int has_vel = 0;
    int scanf_ret = sscanf(line, "%d %lf", &atom_numbers, &start_time);
    Amber_Ensure_Atom_Numbers(system, atom_numbers, controller,
                              "Xponge::Amber_Load_Rst7");
    if (scanf_ret == 2)
    {
        has_vel = 1;
    }

    system->atoms.coordinate.resize(3 * atom_numbers);
    system->atoms.velocity.resize(3 * atom_numbers, 0.0f);

    for (int i = 0; i < atom_numbers; i++)
    {
        if (fscanf(fin, "%f %f %f", &system->atoms.coordinate[3 * i],
                   &system->atoms.coordinate[3 * i + 1],
                   &system->atoms.coordinate[3 * i + 2]) != 3)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat, "Xponge::Amber_Load_Rst7",
                "Reason:\n\tthe format of amber_rst7 is not right\n");
        }
    }

    int amber_irest = 1;
    if (controller->Command_Exist("amber_irest"))
    {
        amber_irest =
            controller->Get_Bool("amber_irest", "Xponge::Amber_Load_Rst7");
    }
    if (has_vel)
    {
        for (int i = 0; i < atom_numbers; i++)
        {
            if (fscanf(fin, "%f %f %f", &system->atoms.velocity[3 * i],
                       &system->atoms.velocity[3 * i + 1],
                       &system->atoms.velocity[3 * i + 2]) != 3)
            {
                controller->Throw_SPONGE_Error(
                    spongeErrorBadFileFormat, "Xponge::Amber_Load_Rst7",
                    "Reason:\n\tthe format of amber_rst7 is not right\n");
            }
        }
    }
    if (!has_vel || amber_irest == 0)
    {
        system->atoms.velocity.assign(3 * atom_numbers, 0.0f);
    }

    system->box.box_length.resize(3);
    system->box.box_angle.resize(3);
    if (fscanf(fin, "%f %f %f", &system->box.box_length[0],
               &system->box.box_length[1], &system->box.box_length[2]) != 3)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorBadFileFormat, "Xponge::Amber_Load_Rst7",
            "Reason:\n\tthe format of amber_rst7 is not right\n");
    }
    if (fscanf(fin, "%f %f %f", &system->box.box_angle[0],
               &system->box.box_angle[1], &system->box.box_angle[2]) != 3)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorBadFileFormat, "Xponge::Amber_Load_Rst7",
            "Reason:\n\tthe format of amber_rst7 is not right\n");
    }
    fclose(fin);
}

void Load_Amber_Inputs(System* system, CONTROLLER* controller)
{
    Amber_Load_Parm7(system, controller);
    Amber_Load_Rst7(system, controller);
}

}  // namespace Xponge

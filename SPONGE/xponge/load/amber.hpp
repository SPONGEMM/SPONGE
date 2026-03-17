#pragma once

#include "../xponge.h"

namespace Xponge
{

static std::string Amber_Trim(const std::string& value)
{
    std::size_t begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos)
    {
        return "";
    }
    std::size_t end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

static int Amber_Parse_Int(const std::string& value)
{
    return std::stoi(value);
}

static float Amber_Parse_Float(const std::string& value)
{
    return std::stof(value);
}

static void Amber_Require_Section_Size(const std::vector<std::string>& values,
                                       std::size_t expected_count,
                                       CONTROLLER* controller,
                                       const char* error_by)
{
    if (values.size() < expected_count)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorBadFileFormat, error_by,
            "Reason:\n\tthe format of amber_parm7 is not right\n");
    }
}

static int Amber_Get_Atom_Numbers(const System* system);
static std::vector<std::string> Amber_Read_Section(
    const std::vector<std::string>& lines, std::size_t* index);

static void Amber_Load_Classical_Force_Field(System* system,
                                             CONTROLLER* controller)
{
    if (!controller->Command_Exist("amber_parm7"))
    {
        return;
    }

    std::ifstream fin(controller->Command("amber_parm7"));
    if (!fin.is_open())
    {
        controller->Throw_SPONGE_Error(
            spongeErrorBadFileFormat,
            "Xponge::Amber_Load_Classical_Force_Field",
            "Reason:\n\tfailed to open amber_parm7\n");
    }

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(fin, line))
    {
        lines.push_back(line);
    }

    ClassicalForceField* ff = &system->classical_force_field;
    ff->bonds = Bonds{};
    ff->angles = Angles{};
    ff->dihedrals = Torsions{};
    ff->impropers = Torsions{};
    ff->nb14 = NB14{};
    ff->lj = LennardJones{};
    ff->cmap = CMap{};

    int atom_numbers = Amber_Get_Atom_Numbers(system);
    int atom_type_numbers = 0;
    int bond_with_hydrogen = 0;
    int bond_without_hydrogen = 0;
    int angle_with_hydrogen = 0;
    int angle_without_hydrogen = 0;
    int dihedral_with_hydrogen = 0;
    int dihedral_without_hydrogen = 0;
    int bond_type_numbers = 0;
    int angle_type_numbers = 0;
    int dihedral_type_numbers = 0;

    std::vector<float> bond_type_k;
    std::vector<float> bond_type_r0;
    std::vector<float> angle_type_k;
    std::vector<float> angle_type_theta0;
    std::vector<float> dihedral_type_pk;
    std::vector<float> dihedral_type_phase;
    std::vector<float> dihedral_type_periodicity;
    std::vector<float> scee_scale_factor;
    std::vector<float> scnb_scale_factor;

    std::vector<int> raw_dihedral_a;
    std::vector<int> raw_dihedral_b;
    std::vector<int> raw_dihedral_c;
    std::vector<int> raw_dihedral_d;
    std::vector<int> raw_dihedral_type;

    for (std::size_t i = 0; i < lines.size(); i++)
    {
        const std::string& current_line = lines[i];
        if (current_line.rfind("%FLAG", 0) != 0)
        {
            continue;
        }
        std::string current_flag = Amber_Trim(current_line.substr(6));
        i++;
        std::vector<std::string> values = Amber_Read_Section(lines, &i);

        if (current_flag == "POINTERS")
        {
            Amber_Require_Section_Size(
                values, 18, controller,
                "Xponge::Amber_Load_Classical_Force_Field");
            atom_numbers = Amber_Parse_Int(values[0]);
            atom_type_numbers = Amber_Parse_Int(values[1]);
            bond_with_hydrogen = Amber_Parse_Int(values[2]);
            bond_without_hydrogen = Amber_Parse_Int(values[3]);
            angle_with_hydrogen = Amber_Parse_Int(values[4]);
            angle_without_hydrogen = Amber_Parse_Int(values[5]);
            dihedral_with_hydrogen = Amber_Parse_Int(values[6]);
            dihedral_without_hydrogen = Amber_Parse_Int(values[7]);
            bond_type_numbers = Amber_Parse_Int(values[15]);
            angle_type_numbers = Amber_Parse_Int(values[16]);
            dihedral_type_numbers = Amber_Parse_Int(values[17]);
        }
        else if (current_flag == "BOND_FORCE_CONSTANT")
        {
            Amber_Require_Section_Size(
                values, static_cast<std::size_t>(bond_type_numbers), controller,
                "Xponge::Amber_Load_Classical_Force_Field");
            bond_type_k.resize(bond_type_numbers);
            for (int j = 0; j < bond_type_numbers; j++)
            {
                bond_type_k[j] = Amber_Parse_Float(values[j]);
            }
        }
        else if (current_flag == "BOND_EQUIL_VALUE")
        {
            Amber_Require_Section_Size(
                values, static_cast<std::size_t>(bond_type_numbers), controller,
                "Xponge::Amber_Load_Classical_Force_Field");
            bond_type_r0.resize(bond_type_numbers);
            for (int j = 0; j < bond_type_numbers; j++)
            {
                bond_type_r0[j] = Amber_Parse_Float(values[j]);
            }
        }
        else if (current_flag == "BONDS_INC_HYDROGEN" ||
                 current_flag == "BONDS_WITHOUT_HYDROGEN")
        {
            Amber_Require_Section_Size(
                values, values.size(), controller,
                "Xponge::Amber_Load_Classical_Force_Field");
            std::size_t tuple_numbers = values.size() / 3;
            ff->bonds.atom_a.reserve(ff->bonds.atom_a.size() + tuple_numbers);
            ff->bonds.atom_b.reserve(ff->bonds.atom_b.size() + tuple_numbers);
            ff->bonds.k.reserve(ff->bonds.k.size() + tuple_numbers);
            ff->bonds.r0.reserve(ff->bonds.r0.size() + tuple_numbers);
            for (std::size_t j = 0; j + 2 < values.size(); j += 3)
            {
                int type_index = Amber_Parse_Int(values[j + 2]) - 1;
                ff->bonds.atom_a.push_back(Amber_Parse_Int(values[j]) / 3);
                ff->bonds.atom_b.push_back(Amber_Parse_Int(values[j + 1]) / 3);
                ff->bonds.k.push_back(bond_type_k[type_index]);
                ff->bonds.r0.push_back(bond_type_r0[type_index]);
            }
        }
        else if (current_flag == "ANGLE_FORCE_CONSTANT")
        {
            Amber_Require_Section_Size(
                values, static_cast<std::size_t>(angle_type_numbers),
                controller, "Xponge::Amber_Load_Classical_Force_Field");
            angle_type_k.resize(angle_type_numbers);
            for (int j = 0; j < angle_type_numbers; j++)
            {
                angle_type_k[j] = Amber_Parse_Float(values[j]);
            }
        }
        else if (current_flag == "ANGLE_EQUIL_VALUE")
        {
            Amber_Require_Section_Size(
                values, static_cast<std::size_t>(angle_type_numbers),
                controller, "Xponge::Amber_Load_Classical_Force_Field");
            angle_type_theta0.resize(angle_type_numbers);
            for (int j = 0; j < angle_type_numbers; j++)
            {
                angle_type_theta0[j] = Amber_Parse_Float(values[j]);
            }
        }
        else if (current_flag == "ANGLES_INC_HYDROGEN" ||
                 current_flag == "ANGLES_WITHOUT_HYDROGEN")
        {
            std::size_t tuple_numbers = values.size() / 4;
            ff->angles.atom_a.reserve(ff->angles.atom_a.size() + tuple_numbers);
            ff->angles.atom_b.reserve(ff->angles.atom_b.size() + tuple_numbers);
            ff->angles.atom_c.reserve(ff->angles.atom_c.size() + tuple_numbers);
            ff->angles.k.reserve(ff->angles.k.size() + tuple_numbers);
            ff->angles.theta0.reserve(ff->angles.theta0.size() + tuple_numbers);
            for (std::size_t j = 0; j + 3 < values.size(); j += 4)
            {
                int type_index = Amber_Parse_Int(values[j + 3]) - 1;
                ff->angles.atom_a.push_back(Amber_Parse_Int(values[j]) / 3);
                ff->angles.atom_b.push_back(Amber_Parse_Int(values[j + 1]) / 3);
                ff->angles.atom_c.push_back(Amber_Parse_Int(values[j + 2]) / 3);
                ff->angles.k.push_back(angle_type_k[type_index]);
                ff->angles.theta0.push_back(angle_type_theta0[type_index]);
            }
        }
        else if (current_flag == "CMAP_COUNT" ||
                 current_flag == "CHARMM_CMAP_COUNT")
        {
            Amber_Require_Section_Size(
                values, 2, controller,
                "Xponge::Amber_Load_Classical_Force_Field");
            ff->cmap.atom_a.resize(Amber_Parse_Int(values[0]));
            ff->cmap.atom_b.resize(ff->cmap.atom_a.size());
            ff->cmap.atom_c.resize(ff->cmap.atom_a.size());
            ff->cmap.atom_d.resize(ff->cmap.atom_a.size());
            ff->cmap.atom_e.resize(ff->cmap.atom_a.size());
            ff->cmap.cmap_type.resize(ff->cmap.atom_a.size());
            ff->cmap.unique_type_numbers = Amber_Parse_Int(values[1]);
            ff->cmap.resolution.resize(ff->cmap.unique_type_numbers);
            ff->cmap.type_offset.resize(ff->cmap.unique_type_numbers);
            ff->cmap.unique_gridpoint_numbers = 0;
            ff->cmap.grid_value.clear();
        }
        else if (current_flag == "CMAP_RESOLUTION" ||
                 current_flag == "CHARMM_CMAP_RESOLUTION")
        {
            Amber_Require_Section_Size(
                values, static_cast<std::size_t>(ff->cmap.unique_type_numbers),
                controller, "Xponge::Amber_Load_Classical_Force_Field");
            ff->cmap.unique_gridpoint_numbers = 0;
            for (int j = 0; j < ff->cmap.unique_type_numbers; j++)
            {
                ff->cmap.resolution[j] = Amber_Parse_Int(values[j]);
                ff->cmap.type_offset[j] =
                    16 * ff->cmap.unique_gridpoint_numbers;
                ff->cmap.unique_gridpoint_numbers +=
                    ff->cmap.resolution[j] * ff->cmap.resolution[j];
            }
            ff->cmap.grid_value.reserve(ff->cmap.unique_gridpoint_numbers);
        }
        else if (current_flag.rfind("CMAP_PARAMETER", 0) == 0 ||
                 current_flag.rfind("CHARMM_CMAP_PARAMETER", 0) == 0)
        {
            for (const std::string& value : values)
            {
                ff->cmap.grid_value.push_back(Amber_Parse_Float(value));
            }
        }
        else if (current_flag == "CMAP_INDEX" ||
                 current_flag == "CHARMM_CMAP_INDEX")
        {
            Amber_Require_Section_Size(
                values, ff->cmap.atom_a.size() * 6, controller,
                "Xponge::Amber_Load_Classical_Force_Field");
            for (std::size_t j = 0; j < ff->cmap.atom_a.size(); j++)
            {
                ff->cmap.atom_a[j] = Amber_Parse_Int(values[j * 6]) - 1;
                ff->cmap.atom_b[j] = Amber_Parse_Int(values[j * 6 + 1]) - 1;
                ff->cmap.atom_c[j] = Amber_Parse_Int(values[j * 6 + 2]) - 1;
                ff->cmap.atom_d[j] = Amber_Parse_Int(values[j * 6 + 3]) - 1;
                ff->cmap.atom_e[j] = Amber_Parse_Int(values[j * 6 + 4]) - 1;
                ff->cmap.cmap_type[j] = Amber_Parse_Int(values[j * 6 + 5]) - 1;
            }
        }
        else if (current_flag == "DIHEDRAL_FORCE_CONSTANT")
        {
            Amber_Require_Section_Size(
                values, static_cast<std::size_t>(dihedral_type_numbers),
                controller, "Xponge::Amber_Load_Classical_Force_Field");
            dihedral_type_pk.resize(dihedral_type_numbers);
            for (int j = 0; j < dihedral_type_numbers; j++)
            {
                dihedral_type_pk[j] = Amber_Parse_Float(values[j]);
            }
        }
        else if (current_flag == "DIHEDRAL_PHASE")
        {
            Amber_Require_Section_Size(
                values, static_cast<std::size_t>(dihedral_type_numbers),
                controller, "Xponge::Amber_Load_Classical_Force_Field");
            dihedral_type_phase.resize(dihedral_type_numbers);
            for (int j = 0; j < dihedral_type_numbers; j++)
            {
                dihedral_type_phase[j] = Amber_Parse_Float(values[j]);
            }
        }
        else if (current_flag == "DIHEDRAL_PERIODICITY")
        {
            Amber_Require_Section_Size(
                values, static_cast<std::size_t>(dihedral_type_numbers),
                controller, "Xponge::Amber_Load_Classical_Force_Field");
            dihedral_type_periodicity.resize(dihedral_type_numbers);
            for (int j = 0; j < dihedral_type_numbers; j++)
            {
                dihedral_type_periodicity[j] = Amber_Parse_Float(values[j]);
            }
        }
        else if (current_flag == "SCEE_SCALE_FACTOR")
        {
            Amber_Require_Section_Size(
                values, static_cast<std::size_t>(dihedral_type_numbers),
                controller, "Xponge::Amber_Load_Classical_Force_Field");
            scee_scale_factor.resize(dihedral_type_numbers);
            for (int j = 0; j < dihedral_type_numbers; j++)
            {
                scee_scale_factor[j] = Amber_Parse_Float(values[j]);
            }
        }
        else if (current_flag == "SCNB_SCALE_FACTOR")
        {
            Amber_Require_Section_Size(
                values, static_cast<std::size_t>(dihedral_type_numbers),
                controller, "Xponge::Amber_Load_Classical_Force_Field");
            scnb_scale_factor.resize(dihedral_type_numbers);
            for (int j = 0; j < dihedral_type_numbers; j++)
            {
                scnb_scale_factor[j] = Amber_Parse_Float(values[j]);
            }
        }
        else if (current_flag == "DIHEDRALS_INC_HYDROGEN" ||
                 current_flag == "DIHEDRALS_WITHOUT_HYDROGEN")
        {
            std::size_t tuple_numbers = values.size() / 5;
            raw_dihedral_a.reserve(raw_dihedral_a.size() + tuple_numbers);
            raw_dihedral_b.reserve(raw_dihedral_b.size() + tuple_numbers);
            raw_dihedral_c.reserve(raw_dihedral_c.size() + tuple_numbers);
            raw_dihedral_d.reserve(raw_dihedral_d.size() + tuple_numbers);
            raw_dihedral_type.reserve(raw_dihedral_type.size() + tuple_numbers);
            for (std::size_t j = 0; j + 4 < values.size(); j += 5)
            {
                raw_dihedral_a.push_back(Amber_Parse_Int(values[j]));
                raw_dihedral_b.push_back(Amber_Parse_Int(values[j + 1]));
                raw_dihedral_c.push_back(Amber_Parse_Int(values[j + 2]));
                raw_dihedral_d.push_back(Amber_Parse_Int(values[j + 3]));
                raw_dihedral_type.push_back(Amber_Parse_Int(values[j + 4]) - 1);
            }
        }
        else if (current_flag == "ATOM_TYPE_INDEX")
        {
            Amber_Require_Section_Size(
                values, static_cast<std::size_t>(atom_numbers), controller,
                "Xponge::Amber_Load_Classical_Force_Field");
            ff->lj.atom_type.resize(atom_numbers);
            for (int j = 0; j < atom_numbers; j++)
            {
                ff->lj.atom_type[j] = Amber_Parse_Int(values[j]) - 1;
            }
        }
        else if (current_flag == "LENNARD_JONES_ACOEF")
        {
            int pair_type_numbers =
                atom_type_numbers * (atom_type_numbers + 1) / 2;
            Amber_Require_Section_Size(
                values, static_cast<std::size_t>(pair_type_numbers), controller,
                "Xponge::Amber_Load_Classical_Force_Field");
            ff->lj.pair_A.resize(pair_type_numbers);
            for (int j = 0; j < pair_type_numbers; j++)
            {
                ff->lj.pair_A[j] = 12.0f * Amber_Parse_Float(values[j]);
            }
        }
        else if (current_flag == "LENNARD_JONES_BCOEF")
        {
            int pair_type_numbers =
                atom_type_numbers * (atom_type_numbers + 1) / 2;
            Amber_Require_Section_Size(
                values, static_cast<std::size_t>(pair_type_numbers), controller,
                "Xponge::Amber_Load_Classical_Force_Field");
            ff->lj.pair_B.resize(pair_type_numbers);
            for (int j = 0; j < pair_type_numbers; j++)
            {
                ff->lj.pair_B[j] = 6.0f * Amber_Parse_Float(values[j]);
            }
        }
        i--;
    }

    ff->lj.atom_type_numbers = atom_type_numbers;

    ff->dihedrals.atom_a.reserve(raw_dihedral_a.size());
    ff->dihedrals.atom_b.reserve(raw_dihedral_a.size());
    ff->dihedrals.atom_c.reserve(raw_dihedral_a.size());
    ff->dihedrals.atom_d.reserve(raw_dihedral_a.size());
    ff->dihedrals.pk.reserve(raw_dihedral_a.size());
    ff->dihedrals.pn.reserve(raw_dihedral_a.size());
    ff->dihedrals.ipn.reserve(raw_dihedral_a.size());
    ff->dihedrals.gamc.reserve(raw_dihedral_a.size());
    ff->dihedrals.gams.reserve(raw_dihedral_a.size());

    ff->nb14.atom_a.reserve(raw_dihedral_a.size());
    ff->nb14.atom_b.reserve(raw_dihedral_a.size());
    ff->nb14.A.reserve(raw_dihedral_a.size());
    ff->nb14.B.reserve(raw_dihedral_a.size());
    ff->nb14.cf_scale_factor.reserve(raw_dihedral_a.size());

    for (std::size_t i = 0; i < raw_dihedral_a.size(); i++)
    {
        int type_index = raw_dihedral_type[i];
        int atom_a = raw_dihedral_a[i] / 3;
        int atom_b = raw_dihedral_b[i] / 3;
        int atom_c = std::abs(raw_dihedral_c[i] / 3);
        int atom_d = std::abs(raw_dihedral_d[i] / 3);
        float pk = dihedral_type_pk[type_index];
        float phase = dihedral_type_phase[type_index];
        float pn = std::fabs(dihedral_type_periodicity[type_index]);
        if (std::fabs(phase - CONSTANT_Pi) <= 0.001f)
        {
            phase = CONSTANT_Pi;
        }
        float gamc = std::cos(phase) * pk;
        float gams = std::sin(phase) * pk;
        if (std::fabs(gamc) < 1e-6f)
        {
            gamc = 0.0f;
        }
        if (std::fabs(gams) < 1e-6f)
        {
            gams = 0.0f;
        }

        ff->dihedrals.atom_a.push_back(atom_a);
        ff->dihedrals.atom_b.push_back(atom_b);
        ff->dihedrals.atom_c.push_back(atom_c);
        ff->dihedrals.atom_d.push_back(atom_d);
        ff->dihedrals.pk.push_back(pk);
        ff->dihedrals.pn.push_back(pn);
        ff->dihedrals.ipn.push_back(static_cast<int>(pn + 0.001f));
        ff->dihedrals.gamc.push_back(gamc);
        ff->dihedrals.gams.push_back(gams);

        if (raw_dihedral_c[i] > 0 && !ff->lj.atom_type.empty() &&
            !ff->lj.pair_A.empty() && !ff->lj.pair_B.empty())
        {
            int type_a = ff->lj.atom_type[atom_a];
            int type_b = ff->lj.atom_type[atom_d];
            if (type_a > type_b)
            {
                int temp = type_a;
                type_a = type_b;
                type_b = temp;
            }
            int pair_type = type_b * (type_b + 1) / 2 + type_a;
            float lj_scale = 0.0f;
            float cf_scale = 0.0f;
            if (type_index < static_cast<int>(scnb_scale_factor.size()) &&
                scnb_scale_factor[type_index] != 0.0f)
            {
                lj_scale = 1.0f / scnb_scale_factor[type_index];
            }
            if (type_index < static_cast<int>(scee_scale_factor.size()) &&
                scee_scale_factor[type_index] != 0.0f)
            {
                cf_scale = 1.0f / scee_scale_factor[type_index];
            }
            ff->nb14.atom_a.push_back(atom_a);
            ff->nb14.atom_b.push_back(atom_d);
            ff->nb14.A.push_back(lj_scale * ff->lj.pair_A[pair_type]);
            ff->nb14.B.push_back(lj_scale * ff->lj.pair_B[pair_type]);
            ff->nb14.cf_scale_factor.push_back(cf_scale);
        }
    }
}

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
        controller->Throw_SPONGE_Error(spongeErrorConflictingCommand, error_by,
                                       "Reason:\n\t'atom_numbers' is different "
                                       "in different input files\n");
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
        std::string current_flag = Amber_Trim(line.substr(6));
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
        else if (current_flag == "RADII")
        {
            system->generalized_born.radius.resize(values.size());
            for (std::size_t i = 0; i < values.size(); i++)
            {
                system->generalized_born.radius[i] = std::stof(values[i]);
            }
        }
        else if (current_flag == "SCREEN")
        {
            system->generalized_born.scale_factor.resize(values.size());
            for (std::size_t i = 0; i < values.size(); i++)
            {
                system->generalized_born.scale_factor[i] = std::stof(values[i]);
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
    if (!system->generalized_born.radius.empty() &&
        system->generalized_born.radius.size() !=
            static_cast<std::size_t>(atom_numbers))
    {
        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                       "Xponge::Amber_Load_Parm7",
                                       "Reason:\n\tRADII length in amber_parm7 "
                                       "does not match atom_numbers\n");
    }
    if (!system->generalized_born.scale_factor.empty() &&
        system->generalized_born.scale_factor.size() !=
            static_cast<std::size_t>(atom_numbers))
    {
        controller->Throw_SPONGE_Error(
            spongeErrorBadFileFormat, "Xponge::Amber_Load_Parm7",
            "Reason:\n\tSCREEN length in amber_parm7 does not match "
            "atom_numbers\n");
    }

    system->exclusions.excluded_atoms.assign(atom_numbers, {});
    int count = 0;
    for (int i = 0;
         i < atom_numbers && i < static_cast<int>(excluded_numbers.size()); i++)
    {
        for (int j = 0; j < excluded_numbers[i] &&
                        count < static_cast<int>(excluded_list.size());
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
        system->start_time = start_time;
    }
    else
    {
        system->start_time = 0.0;
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
    if (!has_vel)
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
    system->source = InputSource::kAmber;
    Amber_Load_Parm7(system, controller);
    Amber_Load_Rst7(system, controller);
    Amber_Load_Classical_Force_Field(system, controller);
}

}  // namespace Xponge

#pragma once

#include "../xponge.h"
#include "./common.hpp"

namespace Xponge
{

namespace fs = std::filesystem;

static constexpr float Gromacs_Pi = 3.14159265358979323846f;

struct Gromacs_Defaults
{
    int nbfunc = 1;
    int comb_rule = 2;
    bool gen_pairs = true;
    float fudge_lj = 1.0f;
    float fudge_qq = 1.0f;
};

struct Gromacs_Atom_Type
{
    std::string name;
    float mass = 0.0f;
    float charge = 0.0f;
    std::string ptype;
    float v = 0.0f;
    float w = 0.0f;
};

struct Gromacs_Bond_Type
{
    std::string ai;
    std::string aj;
    int funct = 0;
    float b0 = 0.0f;
    float kb = 0.0f;
};

struct Gromacs_Angle_Type
{
    std::string ai;
    std::string aj;
    std::string ak;
    int funct = 0;
    float theta0 = 0.0f;
    float k = 0.0f;
    float ub0 = 0.0f;
    float kub = 0.0f;
};

struct Gromacs_Dihedral_Type
{
    std::string ai;
    std::string aj;
    std::string ak;
    std::string al;
    int funct = 0;
    std::vector<float> parameters;
};

struct Gromacs_Pair_Type
{
    std::string ai;
    std::string aj;
    int funct = 0;
    std::vector<float> parameters;
};

struct Gromacs_CMap_Type
{
    std::string ai;
    std::string aj;
    std::string ak;
    std::string al;
    std::string am;
    int funct = 0;
    int resolution = 0;
    std::vector<float> grid;
};

struct Gromacs_Molecule_Atom
{
    int nr = 0;
    std::string type;
    int resnr = 0;
    std::string residue;
    std::string atom;
    int cgnr = 0;
    float charge = 0.0f;
    float mass = 0.0f;
};

struct Gromacs_Bond
{
    int ai = 0;
    int aj = 0;
    int funct = 0;
    std::vector<float> parameters;
};

struct Gromacs_Pair
{
    int ai = 0;
    int aj = 0;
    int funct = 0;
    std::vector<float> parameters;
};

struct Gromacs_Angle
{
    int ai = 0;
    int aj = 0;
    int ak = 0;
    int funct = 0;
    std::vector<float> parameters;
};

struct Gromacs_Dihedral
{
    int ai = 0;
    int aj = 0;
    int ak = 0;
    int al = 0;
    int funct = 0;
    std::vector<float> parameters;
};

struct Gromacs_Settle
{
    int ow = 0;
    int funct = 0;
    float doh = 0.0f;
    float dhh = 0.0f;
};

struct Gromacs_Constraint
{
    int ai = 0;
    int aj = 0;
    int funct = 0;
    std::vector<float> parameters;
};

struct Gromacs_CMap
{
    int ai = 0;
    int aj = 0;
    int ak = 0;
    int al = 0;
    int am = 0;
    int funct = 0;
};

struct Gromacs_Molecule
{
    std::string name;
    int nrexcl = 0;
    std::vector<Gromacs_Molecule_Atom> atoms;
    std::vector<Gromacs_Bond> bonds;
    std::vector<Gromacs_Pair> pairs;
    std::vector<Gromacs_Angle> angles;
    std::vector<Gromacs_Dihedral> dihedrals;
    std::vector<Gromacs_Settle> settles;
    std::vector<Gromacs_Constraint> constraints;
    std::vector<Gromacs_CMap> cmaps;
};

struct Gromacs_Residue_Info
{
    int atom_numbers = 0;
};

struct Gromacs_Topology
{
    Gromacs_Defaults defaults;
    std::unordered_map<std::string, Gromacs_Atom_Type> atom_types;
    std::vector<Gromacs_Bond_Type> bond_types;
    std::vector<Gromacs_Angle_Type> angle_types;
    std::vector<Gromacs_Dihedral_Type> dihedral_types;
    std::vector<Gromacs_Pair_Type> pair_types;
    std::vector<Gromacs_CMap_Type> cmap_types;
    std::unordered_map<std::string, Gromacs_Molecule> molecules;
    std::vector<std::pair<std::string, int>> system_molecules;
};

static std::string Gromacs_Trim(const std::string& value)
{
    std::size_t begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos)
    {
        return "";
    }
    std::size_t end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

static std::string Gromacs_Strip_Comment(const std::string& line)
{
    std::size_t comment = line.find(';');
    if (comment == std::string::npos)
    {
        return Gromacs_Trim(line);
    }
    return Gromacs_Trim(line.substr(0, comment));
}

static std::vector<std::string> Gromacs_Split(const std::string& line)
{
    std::vector<std::string> tokens;
    std::istringstream stream(line);
    std::string token;
    while (stream >> token)
    {
        tokens.push_back(token);
    }
    return tokens;
}

static std::vector<std::string> Gromacs_Split_List(const std::string& value)
{
    std::vector<std::string> tokens;
    std::string current;
    for (char ch : value)
    {
        if (ch == ',' || ch == ';' || ch == ':' || ch == ' ' || ch == '\t')
        {
            if (!current.empty())
            {
                tokens.push_back(current);
                current.clear();
            }
        }
        else
        {
            current.push_back(ch);
        }
    }
    if (!current.empty())
    {
        tokens.push_back(current);
    }
    return tokens;
}

static bool Gromacs_Is_True(const std::string& value)
{
    return value == "yes" || value == "Yes" || value == "YES" || value == "1" ||
           value == "true" || value == "TRUE";
}

static fs::path Gromacs_Resolve_Include(
    const fs::path& parent_dir, const std::string& include_name,
    const std::vector<fs::path>& include_dirs, CONTROLLER* controller,
    const char* error_by)
{
    fs::path candidate = parent_dir / include_name;
    if (fs::exists(candidate))
    {
        return candidate;
    }
    for (const fs::path& include_dir : include_dirs)
    {
        candidate = include_dir / include_name;
        if (fs::exists(candidate))
        {
            return candidate;
        }
    }
    std::string reason = "Reason:\n\tfailed to resolve GROMACS include file '" +
                         include_name + "'\n";
    controller->Throw_SPONGE_Error(spongeErrorBadFileFormat, error_by,
                                   reason.c_str());
    return {};
}

static void Gromacs_Preprocess_File(const fs::path& file_path,
                                    std::set<std::string>* macros,
                                    const std::vector<fs::path>& include_dirs,
                                    std::vector<std::string>* lines,
                                    CONTROLLER* controller,
                                    const char* error_by)
{
    std::ifstream fin(file_path);
    if (!fin.is_open())
    {
        std::string reason =
            "Reason:\n\tfailed to open GROMACS topology file '" +
            file_path.string() + "'\n";
        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat, error_by,
                                       reason.c_str());
    }

    struct Conditional_State
    {
        bool parent_active = true;
        bool branch_active = true;
        bool else_seen = false;
    };
    std::vector<Conditional_State> stack;
    auto is_active = [&stack]() -> bool
    {
        if (stack.empty())
        {
            return true;
        }
        return stack.back().parent_active && stack.back().branch_active;
    };

    std::string raw_line;
    std::string continued_line;
    while (std::getline(fin, raw_line))
    {
        std::string line = Gromacs_Trim(raw_line);
        if (!line.empty() && line[0] == '#')
        {
            std::vector<std::string> tokens = Gromacs_Split(line);
            if (tokens.empty())
            {
                continue;
            }
            if (tokens[0] == "#include")
            {
                if (!is_active())
                {
                    continue;
                }
                std::size_t begin = line.find('"');
                std::size_t end = line.find_last_of('"');
                if (begin == std::string::npos || end == std::string::npos ||
                    end <= begin)
                {
                    controller->Throw_SPONGE_Error(
                        spongeErrorBadFileFormat, error_by,
                        "Reason:\n\tinvalid GROMACS #include directive\n");
                }
                std::string include_name =
                    line.substr(begin + 1, end - begin - 1);
                fs::path include_path = Gromacs_Resolve_Include(
                    file_path.parent_path(), include_name, include_dirs,
                    controller, error_by);
                Gromacs_Preprocess_File(include_path, macros, include_dirs,
                                        lines, controller, error_by);
                continue;
            }
            if (tokens[0] == "#define")
            {
                if (is_active() && tokens.size() >= 2)
                {
                    macros->insert(tokens[1]);
                }
                continue;
            }
            if (tokens[0] == "#undef")
            {
                if (is_active() && tokens.size() >= 2)
                {
                    macros->erase(tokens[1]);
                }
                continue;
            }
            if (tokens[0] == "#ifdef" || tokens[0] == "#ifndef")
            {
                bool parent_active = is_active();
                bool defined =
                    (tokens.size() >= 2 && macros->count(tokens[1]) > 0);
                bool branch_active =
                    (tokens[0] == "#ifdef") ? defined : !defined;
                stack.push_back({parent_active, branch_active, false});
                continue;
            }
            if (tokens[0] == "#else")
            {
                if (stack.empty() || stack.back().else_seen)
                {
                    controller->Throw_SPONGE_Error(
                        spongeErrorBadFileFormat, error_by,
                        "Reason:\n\tinvalid GROMACS #else directive\n");
                }
                stack.back().branch_active = !stack.back().branch_active;
                stack.back().else_seen = true;
                continue;
            }
            if (tokens[0] == "#endif")
            {
                if (stack.empty())
                {
                    controller->Throw_SPONGE_Error(
                        spongeErrorBadFileFormat, error_by,
                        "Reason:\n\tinvalid GROMACS #endif directive\n");
                }
                stack.pop_back();
                continue;
            }
            continue;
        }

        if (!is_active())
        {
            continue;
        }
        line = Gromacs_Strip_Comment(line);
        if (line.empty())
        {
            continue;
        }
        if (!continued_line.empty())
        {
            continued_line += " ";
            continued_line += line;
        }
        else
        {
            continued_line = line;
        }
        while (!continued_line.empty() && continued_line.back() == '\\')
        {
            continued_line.pop_back();
            continued_line = Gromacs_Trim(continued_line);
            if (!std::getline(fin, raw_line))
            {
                controller->Throw_SPONGE_Error(
                    spongeErrorBadFileFormat, error_by,
                    "Reason:\n\tunterminated GROMACS line continuation\n");
            }
            std::string next_line =
                Gromacs_Strip_Comment(Gromacs_Trim(raw_line));
            if (!next_line.empty())
            {
                continued_line += " ";
                continued_line += next_line;
            }
        }
        if (!continued_line.empty())
        {
            lines->push_back(continued_line);
            continued_line.clear();
        }
    }

    if (!stack.empty())
    {
        controller->Throw_SPONGE_Error(
            spongeErrorBadFileFormat, error_by,
            "Reason:\n\tunterminated GROMACS preprocessor conditional\n");
    }
}

static float Gromacs_To_Kcal(float value_in_kj) { return value_in_kj / 4.184f; }

static float Gromacs_To_Angstrom(float value_in_nm)
{
    return value_in_nm * 10.0f;
}

static float Gromacs_To_Radian(float value_in_degree)
{
    return value_in_degree * Gromacs_Pi / 180.0f;
}

static std::pair<float, float> Gromacs_Get_C6_C12(
    const Gromacs_Defaults& defaults, const Gromacs_Atom_Type& atom_i,
    const Gromacs_Atom_Type& atom_j)
{
    if (defaults.comb_rule == 1)
    {
        return {std::sqrt(atom_i.v * atom_j.v) * 1000000.0f / 4.184f,
                std::sqrt(atom_i.w * atom_j.w) * 1000000000000.0f / 4.184f};
    }

    float sigma = Gromacs_To_Angstrom(0.5f * (atom_i.v + atom_j.v));
    float epsilon = Gromacs_To_Kcal(std::sqrt(atom_i.w * atom_j.w));
    float sigma6 = std::pow(sigma, 6.0f);
    return {4.0f * epsilon * sigma6, 4.0f * epsilon * sigma6 * sigma6};
}

static std::pair<float, float> Gromacs_Get_C6_C12_From_Pair_Parameters(
    const Gromacs_Defaults& defaults, const std::vector<float>& parameters)
{
    if (parameters.size() < 2)
    {
        return {0.0f, 0.0f};
    }
    if (defaults.comb_rule == 1)
    {
        return {parameters[0] * 1000000.0f / 4.184f,
                parameters[1] * 1000000000000.0f / 4.184f};
    }
    float sigma = Gromacs_To_Angstrom(parameters[0]);
    float epsilon = Gromacs_To_Kcal(parameters[1]);
    float sigma6 = std::pow(sigma, 6.0f);
    return {4.0f * epsilon * sigma6, 4.0f * epsilon * sigma6 * sigma6};
}

static bool Gromacs_Type_Match(const std::string& pattern,
                               const std::string& value)
{
    return pattern == "X" || pattern == value;
}

static int Gromacs_Count_Wildcards(const std::vector<std::string>& values)
{
    int count = 0;
    for (const std::string& value : values)
    {
        if (value == "X")
        {
            count++;
        }
    }
    return count;
}

static const Gromacs_Bond_Type* Gromacs_Find_Bond_Type(
    const std::vector<Gromacs_Bond_Type>& bond_types, const std::string& ai,
    const std::string& aj, int funct)
{
    for (const Gromacs_Bond_Type& type : bond_types)
    {
        if (type.funct != funct)
        {
            continue;
        }
        if ((type.ai == ai && type.aj == aj) ||
            (type.ai == aj && type.aj == ai))
        {
            return &type;
        }
    }
    return NULL;
}

static const Gromacs_Angle_Type* Gromacs_Find_Angle_Type(
    const std::vector<Gromacs_Angle_Type>& angle_types, const std::string& ai,
    const std::string& aj, const std::string& ak, int funct)
{
    for (const Gromacs_Angle_Type& type : angle_types)
    {
        if (type.funct != funct)
        {
            continue;
        }
        if ((type.ai == ai && type.aj == aj && type.ak == ak) ||
            (type.ai == ak && type.aj == aj && type.ak == ai))
        {
            return &type;
        }
    }
    return NULL;
}

static std::vector<const Gromacs_Dihedral_Type*> Gromacs_Find_Dihedral_Types(
    const std::vector<Gromacs_Dihedral_Type>& dihedral_types,
    const std::string& ai, const std::string& aj, const std::string& ak,
    const std::string& al, int funct)
{
    std::vector<const Gromacs_Dihedral_Type*> matches;
    int best_wildcards = 1000;
    for (const Gromacs_Dihedral_Type& type : dihedral_types)
    {
        if (type.funct != funct)
        {
            continue;
        }
        bool forward = Gromacs_Type_Match(type.ai, ai) &&
                       Gromacs_Type_Match(type.aj, aj) &&
                       Gromacs_Type_Match(type.ak, ak) &&
                       Gromacs_Type_Match(type.al, al);
        bool backward = Gromacs_Type_Match(type.ai, al) &&
                        Gromacs_Type_Match(type.aj, ak) &&
                        Gromacs_Type_Match(type.ak, aj) &&
                        Gromacs_Type_Match(type.al, ai);
        if (!forward && !backward)
        {
            continue;
        }
        int wildcards =
            Gromacs_Count_Wildcards({type.ai, type.aj, type.ak, type.al});
        if (wildcards < best_wildcards)
        {
            matches.clear();
            best_wildcards = wildcards;
        }
        if (wildcards == best_wildcards)
        {
            matches.push_back(&type);
        }
    }
    return matches;
}

static const Gromacs_Pair_Type* Gromacs_Find_Pair_Type(
    const std::vector<Gromacs_Pair_Type>& pair_types, const std::string& ai,
    const std::string& aj, int funct)
{
    for (const Gromacs_Pair_Type& type : pair_types)
    {
        if (type.funct != funct)
        {
            continue;
        }
        if ((type.ai == ai && type.aj == aj) ||
            (type.ai == aj && type.aj == ai))
        {
            return &type;
        }
    }
    return NULL;
}

static int Gromacs_Find_CMap_Type(
    const std::vector<Gromacs_CMap_Type>& cmap_types, const std::string& ai,
    const std::string& aj, const std::string& ak, const std::string& al,
    const std::string& am, int funct)
{
    int match = -1;
    int best_wildcards = 1000;
    for (std::size_t i = 0; i < cmap_types.size(); i++)
    {
        const Gromacs_CMap_Type& type = cmap_types[i];
        if (type.funct != funct)
        {
            continue;
        }
        bool forward = Gromacs_Type_Match(type.ai, ai) &&
                       Gromacs_Type_Match(type.aj, aj) &&
                       Gromacs_Type_Match(type.ak, ak) &&
                       Gromacs_Type_Match(type.al, al) &&
                       Gromacs_Type_Match(type.am, am);
        if (!forward)
        {
            continue;
        }
        int wildcards = Gromacs_Count_Wildcards(
            {type.ai, type.aj, type.ak, type.al, type.am});
        if (wildcards < best_wildcards)
        {
            best_wildcards = wildcards;
            match = static_cast<int>(i);
        }
    }
    return match;
}

static Gromacs_Topology Gromacs_Parse_Topology(CONTROLLER* controller)
{
    const char* error_by = "Xponge::Load_Gromacs_Inputs";
    if (!controller->Command_Exist("gromacs_top"))
    {
        controller->Throw_SPONGE_Error(
            spongeErrorMissingCommand, error_by,
            "Reason:\n\tgromacs_top is required for GROMACS input\n");
    }
    if (!controller->Command_Exist("gromacs_gro"))
    {
        controller->Throw_SPONGE_Error(
            spongeErrorMissingCommand, error_by,
            "Reason:\n\tgromacs_gro is required for GROMACS input\n");
    }

    fs::path top_path = controller->Command("gromacs_top");
    std::vector<fs::path> include_dirs;
    include_dirs.push_back(top_path.parent_path());
    if (controller->Command_Exist("gromacs_include_dir"))
    {
        for (const std::string& token :
             Gromacs_Split_List(controller->Command("gromacs_include_dir")))
        {
            include_dirs.push_back(token);
        }
    }
    std::set<std::string> macros;
    if (controller->Command_Exist("gromacs_define"))
    {
        for (const std::string& token :
             Gromacs_Split_List(controller->Command("gromacs_define")))
        {
            macros.insert(token);
        }
    }

    std::vector<std::string> lines;
    Gromacs_Preprocess_File(top_path, &macros, include_dirs, &lines, controller,
                            error_by);

    Gromacs_Topology topology;
    std::string current_section;
    Gromacs_Molecule* current_molecule = NULL;

    for (const std::string& line : lines)
    {
        if (line.front() == '[' && line.back() == ']')
        {
            current_section = Gromacs_Trim(line.substr(1, line.size() - 2));
            continue;
        }

        std::vector<std::string> tokens = Gromacs_Split(line);
        if (tokens.empty())
        {
            continue;
        }

        if (current_section == "defaults")
        {
            if (tokens.size() < 5)
            {
                controller->Throw_SPONGE_Error(
                    spongeErrorBadFileFormat, error_by,
                    "Reason:\n\tinvalid [ defaults ] section in GROMACS "
                    "topology\n");
            }
            topology.defaults.nbfunc = std::stoi(tokens[0]);
            topology.defaults.comb_rule = std::stoi(tokens[1]);
            topology.defaults.gen_pairs = Gromacs_Is_True(tokens[2]);
            topology.defaults.fudge_lj = std::stof(tokens[3]);
            topology.defaults.fudge_qq = std::stof(tokens[4]);
        }
        else if (current_section == "atomtypes")
        {
            if (tokens.size() < 7)
            {
                controller->Throw_SPONGE_Error(
                    spongeErrorBadFileFormat, error_by,
                    "Reason:\n\tinvalid [ atomtypes ] section in GROMACS "
                    "topology\n");
            }
            Gromacs_Atom_Type atom_type;
            atom_type.name = tokens[0];
            atom_type.mass = std::stof(tokens[tokens.size() - 5]);
            atom_type.charge = std::stof(tokens[tokens.size() - 4]);
            atom_type.ptype = tokens[tokens.size() - 3];
            atom_type.v = std::stof(tokens[tokens.size() - 2]);
            atom_type.w = std::stof(tokens[tokens.size() - 1]);
            topology.atom_types[atom_type.name] = atom_type;
        }
        else if (current_section == "bondtypes")
        {
            if (tokens.size() < 5)
            {
                controller->Throw_SPONGE_Error(
                    spongeErrorBadFileFormat, error_by,
                    "Reason:\n\tinvalid [ bondtypes ] section in GROMACS "
                    "topology\n");
            }
            Gromacs_Bond_Type bond_type;
            bond_type.ai = tokens[0];
            bond_type.aj = tokens[1];
            bond_type.funct = std::stoi(tokens[2]);
            bond_type.b0 = std::stof(tokens[3]);
            bond_type.kb = std::stof(tokens[4]);
            topology.bond_types.push_back(bond_type);
        }
        else if (current_section == "angletypes")
        {
            if (tokens.size() < 6)
            {
                controller->Throw_SPONGE_Error(
                    spongeErrorBadFileFormat, error_by,
                    "Reason:\n\tinvalid [ angletypes ] section in GROMACS "
                    "topology\n");
            }
            Gromacs_Angle_Type angle_type;
            angle_type.ai = tokens[0];
            angle_type.aj = tokens[1];
            angle_type.ak = tokens[2];
            angle_type.funct = std::stoi(tokens[3]);
            angle_type.theta0 = std::stof(tokens[4]);
            angle_type.k = std::stof(tokens[5]);
            if (tokens.size() >= 8)
            {
                angle_type.ub0 = std::stof(tokens[6]);
                angle_type.kub = std::stof(tokens[7]);
            }
            topology.angle_types.push_back(angle_type);
        }
        else if (current_section == "dihedraltypes")
        {
            if (tokens.size() < 6)
            {
                controller->Throw_SPONGE_Error(
                    spongeErrorBadFileFormat, error_by,
                    "Reason:\n\tinvalid [ dihedraltypes ] section in GROMACS "
                    "topology\n");
            }
            Gromacs_Dihedral_Type dihedral_type;
            dihedral_type.ai = tokens[0];
            dihedral_type.aj = tokens[1];
            dihedral_type.ak = tokens[2];
            dihedral_type.al = tokens[3];
            dihedral_type.funct = std::stoi(tokens[4]);
            for (std::size_t i = 5; i < tokens.size(); i++)
            {
                dihedral_type.parameters.push_back(std::stof(tokens[i]));
            }
            topology.dihedral_types.push_back(dihedral_type);
        }
        else if (current_section == "pairtypes")
        {
            if (tokens.size() < 5)
            {
                controller->Throw_SPONGE_Error(
                    spongeErrorBadFileFormat, error_by,
                    "Reason:\n\tinvalid [ pairtypes ] section in GROMACS "
                    "topology\n");
            }
            Gromacs_Pair_Type pair_type;
            pair_type.ai = tokens[0];
            pair_type.aj = tokens[1];
            pair_type.funct = std::stoi(tokens[2]);
            for (std::size_t i = 3; i < tokens.size(); i++)
            {
                pair_type.parameters.push_back(std::stof(tokens[i]));
            }
            topology.pair_types.push_back(pair_type);
        }
        else if (current_section == "cmaptypes")
        {
            if (tokens.size() < 8)
            {
                controller->Throw_SPONGE_Error(
                    spongeErrorBadFileFormat, error_by,
                    "Reason:\n\tinvalid [ cmaptypes ] section in GROMACS "
                    "topology\n");
            }
            Gromacs_CMap_Type cmap_type;
            cmap_type.ai = tokens[0];
            cmap_type.aj = tokens[1];
            cmap_type.ak = tokens[2];
            cmap_type.al = tokens[3];
            cmap_type.am = tokens[4];
            cmap_type.funct = std::stoi(tokens[5]);
            int resolution_phi = std::stoi(tokens[6]);
            int resolution_psi = std::stoi(tokens[7]);
            if (resolution_phi != resolution_psi)
            {
                controller->Throw_SPONGE_Error(
                    spongeErrorBadFileFormat, error_by,
                    "Reason:\n\tnon-square GROMACS CMAP grids are not "
                    "supported yet\n");
            }
            cmap_type.resolution = resolution_phi;
            std::size_t expected_grid_size =
                static_cast<std::size_t>(cmap_type.resolution) *
                static_cast<std::size_t>(cmap_type.resolution);
            if (tokens.size() != 8 + expected_grid_size)
            {
                controller->Throw_SPONGE_Error(
                    spongeErrorBadFileFormat, error_by,
                    "Reason:\n\tinvalid GROMACS CMAP grid size\n");
            }
            cmap_type.grid.reserve(expected_grid_size);
            for (std::size_t i = 8; i < tokens.size(); i++)
            {
                cmap_type.grid.push_back(std::stof(tokens[i]));
            }
            topology.cmap_types.push_back(cmap_type);
        }
        else if (current_section == "moleculetype")
        {
            if (tokens.size() < 2)
            {
                controller->Throw_SPONGE_Error(
                    spongeErrorBadFileFormat, error_by,
                    "Reason:\n\tinvalid [ moleculetype ] section in GROMACS "
                    "topology\n");
            }
            Gromacs_Molecule molecule;
            molecule.name = tokens[0];
            molecule.nrexcl = std::stoi(tokens[1]);
            topology.molecules[molecule.name] = molecule;
            current_molecule = &topology.molecules[molecule.name];
        }
        else if (current_section == "atoms")
        {
            if (current_molecule == NULL || tokens.size() < 7)
            {
                controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                               error_by,
                                               "Reason:\n\tinvalid [ atoms ] "
                                               "section in GROMACS topology\n");
            }
            Gromacs_Molecule_Atom atom;
            atom.nr = std::stoi(tokens[0]);
            atom.type = tokens[1];
            atom.resnr = std::stoi(tokens[2]);
            atom.residue = tokens[3];
            atom.atom = tokens[4];
            atom.cgnr = std::stoi(tokens[5]);
            atom.charge = std::stof(tokens[6]);
            if (tokens.size() >= 8)
            {
                atom.mass = std::stof(tokens[7]);
            }
            current_molecule->atoms.push_back(atom);
        }
        else if (current_section == "bonds")
        {
            if (current_molecule == NULL || tokens.size() < 3)
            {
                controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                               error_by,
                                               "Reason:\n\tinvalid [ bonds ] "
                                               "section in GROMACS topology\n");
            }
            Gromacs_Bond bond;
            bond.ai = std::stoi(tokens[0]);
            bond.aj = std::stoi(tokens[1]);
            bond.funct = std::stoi(tokens[2]);
            for (std::size_t i = 3; i < tokens.size(); i++)
            {
                bond.parameters.push_back(std::stof(tokens[i]));
            }
            current_molecule->bonds.push_back(bond);
        }
        else if (current_section == "pairs")
        {
            if (current_molecule == NULL || tokens.size() < 3)
            {
                controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                               error_by,
                                               "Reason:\n\tinvalid [ pairs ] "
                                               "section in GROMACS topology\n");
            }
            Gromacs_Pair pair;
            pair.ai = std::stoi(tokens[0]);
            pair.aj = std::stoi(tokens[1]);
            pair.funct = std::stoi(tokens[2]);
            for (std::size_t i = 3; i < tokens.size(); i++)
            {
                pair.parameters.push_back(std::stof(tokens[i]));
            }
            current_molecule->pairs.push_back(pair);
        }
        else if (current_section == "angles")
        {
            if (current_molecule == NULL || tokens.size() < 4)
            {
                controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                               error_by,
                                               "Reason:\n\tinvalid [ angles ] "
                                               "section in GROMACS topology\n");
            }
            Gromacs_Angle angle;
            angle.ai = std::stoi(tokens[0]);
            angle.aj = std::stoi(tokens[1]);
            angle.ak = std::stoi(tokens[2]);
            angle.funct = std::stoi(tokens[3]);
            for (std::size_t i = 4; i < tokens.size(); i++)
            {
                angle.parameters.push_back(std::stof(tokens[i]));
            }
            current_molecule->angles.push_back(angle);
        }
        else if (current_section == "dihedrals")
        {
            if (current_molecule == NULL || tokens.size() < 5)
            {
                controller->Throw_SPONGE_Error(
                    spongeErrorBadFileFormat, error_by,
                    "Reason:\n\tinvalid [ dihedrals ] section in GROMACS "
                    "topology\n");
            }
            Gromacs_Dihedral dihedral;
            dihedral.ai = std::stoi(tokens[0]);
            dihedral.aj = std::stoi(tokens[1]);
            dihedral.ak = std::stoi(tokens[2]);
            dihedral.al = std::stoi(tokens[3]);
            dihedral.funct = std::stoi(tokens[4]);
            for (std::size_t i = 5; i < tokens.size(); i++)
            {
                dihedral.parameters.push_back(std::stof(tokens[i]));
            }
            current_molecule->dihedrals.push_back(dihedral);
        }
        else if (current_section == "settles")
        {
            if (current_molecule == NULL || tokens.size() < 4)
            {
                controller->Throw_SPONGE_Error(
                    spongeErrorBadFileFormat, error_by,
                    "Reason:\n\tinvalid [ settles ] section in GROMACS "
                    "topology\n");
            }
            Gromacs_Settle settle;
            settle.ow = std::stoi(tokens[0]);
            settle.funct = std::stoi(tokens[1]);
            settle.doh = std::stof(tokens[2]);
            settle.dhh = std::stof(tokens[3]);
            current_molecule->settles.push_back(settle);
        }
        else if (current_section == "constraints")
        {
            if (current_molecule == NULL || tokens.size() < 3)
            {
                controller->Throw_SPONGE_Error(
                    spongeErrorBadFileFormat, error_by,
                    "Reason:\n\tinvalid [ constraints ] section in GROMACS "
                    "topology\n");
            }
            Gromacs_Constraint constraint;
            constraint.ai = std::stoi(tokens[0]);
            constraint.aj = std::stoi(tokens[1]);
            constraint.funct = std::stoi(tokens[2]);
            for (std::size_t i = 3; i < tokens.size(); i++)
            {
                constraint.parameters.push_back(std::stof(tokens[i]));
            }
            current_molecule->constraints.push_back(constraint);
        }
        else if (current_section == "cmap")
        {
            if (current_molecule == NULL || tokens.size() < 6)
            {
                controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                               error_by,
                                               "Reason:\n\tinvalid [ cmap ] "
                                               "section in GROMACS topology\n");
            }
            Gromacs_CMap cmap;
            cmap.ai = std::stoi(tokens[0]);
            cmap.aj = std::stoi(tokens[1]);
            cmap.ak = std::stoi(tokens[2]);
            cmap.al = std::stoi(tokens[3]);
            cmap.am = std::stoi(tokens[4]);
            cmap.funct = std::stoi(tokens[5]);
            current_molecule->cmaps.push_back(cmap);
        }
        else if (current_section == "molecules")
        {
            if (tokens.size() < 2)
            {
                controller->Throw_SPONGE_Error(
                    spongeErrorBadFileFormat, error_by,
                    "Reason:\n\tinvalid [ molecules ] section in GROMACS "
                    "topology\n");
            }
            topology.system_molecules.push_back(
                {tokens[0], std::stoi(tokens[1])});
        }
    }

    return topology;
}

static void Gromacs_Load_Gro(System* system, CONTROLLER* controller)
{
    const char* error_by = "Xponge::Load_Gromacs_Inputs";
    std::ifstream fin(controller->Command("gromacs_gro"));
    if (!fin.is_open())
    {
        controller->Throw_SPONGE_Error(
            spongeErrorBadFileFormat, error_by,
            "Reason:\n\tfailed to open GROMACS gro file\n");
    }

    std::string line;
    std::getline(fin, line);
    if (!std::getline(fin, line))
    {
        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat, error_by,
                                       "Reason:\n\tinvalid GROMACS gro file\n");
    }
    int atom_numbers = std::stoi(Gromacs_Trim(line));
    Load_Ensure_Atom_Numbers(system, atom_numbers, controller, error_by);
    system->atoms.coordinate.resize(3 * atom_numbers);
    system->atoms.velocity.assign(3 * atom_numbers, 0.0f);

    for (int i = 0; i < atom_numbers; i++)
    {
        if (!std::getline(fin, line) || line.size() < 44)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat, error_by,
                "Reason:\n\tinvalid atom line in GROMACS gro file\n");
        }
        try
        {
            system->atoms.coordinate[3 * i] =
                Gromacs_To_Angstrom(std::stof(line.substr(20, 8)));
            system->atoms.coordinate[3 * i + 1] =
                Gromacs_To_Angstrom(std::stof(line.substr(28, 8)));
            system->atoms.coordinate[3 * i + 2] =
                Gromacs_To_Angstrom(std::stof(line.substr(36, 8)));
        }
        catch (const std::exception&)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat, error_by,
                "Reason:\n\tinvalid coordinate field in GROMACS gro file\n");
        }
    }

    if (!std::getline(fin, line))
    {
        controller->Throw_SPONGE_Error(
            spongeErrorBadFileFormat, error_by,
            "Reason:\n\tmissing box line in GROMACS gro file\n");
    }
    std::vector<std::string> tokens = Gromacs_Split(line);
    if (tokens.size() != 3 && tokens.size() != 9)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorBadFileFormat, error_by,
            "Reason:\n\tunsupported box line in GROMACS gro file\n");
    }
    if (tokens.size() == 9 && (std::fabs(std::stof(tokens[3])) > 1e-6f ||
                               std::fabs(std::stof(tokens[4])) > 1e-6f ||
                               std::fabs(std::stof(tokens[5])) > 1e-6f ||
                               std::fabs(std::stof(tokens[6])) > 1e-6f ||
                               std::fabs(std::stof(tokens[7])) > 1e-6f ||
                               std::fabs(std::stof(tokens[8])) > 1e-6f))
    {
        controller->Throw_SPONGE_Error(
            spongeErrorBadFileFormat, error_by,
            "Reason:\n\ttriclinic GROMACS gro boxes are not supported yet\n");
    }
    system->box.box_length = {Gromacs_To_Angstrom(std::stof(tokens[0])),
                              Gromacs_To_Angstrom(std::stof(tokens[1])),
                              Gromacs_To_Angstrom(std::stof(tokens[2]))};
    system->box.box_angle = {90.0f, 90.0f, 90.0f};
}

static void Gromacs_Instantiate_System(const Gromacs_Topology& topology,
                                       System* system, CONTROLLER* controller)
{
    const char* error_by = "Xponge::Load_Gromacs_Inputs";
    system->source = InputSource::kGromacs;
    system->start_time = 0.0;
    system->atoms.mass.clear();
    system->atoms.charge.clear();
    system->residues.atom_numbers.clear();
    system->exclusions.excluded_atoms.clear();
    system->generalized_born = GeneralizedBorn{};
    system->virtual_atoms = VirtualAtoms{};
    Load_Reset_Classical_Force_Field(&system->classical_force_field);

    std::vector<std::string> global_atom_types;
    std::vector<std::vector<int>> molecule_local_to_global;

    for (const auto& item : topology.system_molecules)
    {
        auto iter = topology.molecules.find(item.first);
        if (iter == topology.molecules.end())
        {
            std::string reason =
                "Reason:\n\tmolecule '" + item.first +
                "' referenced in [ molecules ] is not defined\n";
            controller->Throw_SPONGE_Error(spongeErrorBadFileFormat, error_by,
                                           reason.c_str());
        }
        const Gromacs_Molecule& molecule = iter->second;
        for (int copy = 0; copy < item.second; copy++)
        {
            std::vector<int> local_to_global(molecule.atoms.size());
            int current_resnr = std::numeric_limits<int>::min();
            std::string current_residue;
            for (std::size_t i = 0; i < molecule.atoms.size(); i++)
            {
                const Gromacs_Molecule_Atom& atom = molecule.atoms[i];
                auto atom_type_iter = topology.atom_types.find(atom.type);
                if (atom_type_iter == topology.atom_types.end())
                {
                    std::string reason =
                        "Reason:\n\tundefined GROMACS atom type '" + atom.type +
                        "'\n";
                    controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                                   error_by, reason.c_str());
                }
                const Gromacs_Atom_Type& atom_type = atom_type_iter->second;
                local_to_global[i] =
                    static_cast<int>(system->atoms.mass.size());
                system->atoms.mass.push_back(atom.mass > 0.0f ? atom.mass
                                                              : atom_type.mass);
                system->atoms.charge.push_back(atom.charge *
                                               CONSTANT_SPONGE_CHARGE_SCALE);
                global_atom_types.push_back(atom.type);
                if (atom.resnr != current_resnr ||
                    atom.residue != current_residue)
                {
                    system->residues.atom_numbers.push_back(0);
                    current_resnr = atom.resnr;
                    current_residue = atom.residue;
                }
                system->residues.atom_numbers.back() += 1;
            }
            molecule_local_to_global.push_back(local_to_global);
        }
    }

    int atom_numbers = static_cast<int>(system->atoms.mass.size());
    system->exclusions.excluded_atoms.assign(atom_numbers, {});
    Xponge::Bonds& bonds = system->classical_force_field.bonds;
    Xponge::DistanceConstraints& constraints =
        system->classical_force_field.constraints;
    Xponge::Angles& angles = system->classical_force_field.angles;
    Xponge::UreyBradley& urey = system->classical_force_field.urey_bradley;
    Xponge::Torsions& dihedrals = system->classical_force_field.dihedrals;
    Xponge::Torsions& impropers = system->classical_force_field.impropers;
    Xponge::NB14& nb14 = system->classical_force_field.nb14;
    Xponge::CMap& cmap = system->classical_force_field.cmap;

    cmap.unique_type_numbers = static_cast<int>(topology.cmap_types.size());
    cmap.resolution.resize(cmap.unique_type_numbers);
    cmap.type_offset.resize(cmap.unique_type_numbers);
    cmap.unique_gridpoint_numbers = 0;
    for (int i = 0; i < cmap.unique_type_numbers; i++)
    {
        const Gromacs_CMap_Type& cmap_type = topology.cmap_types[i];
        cmap.resolution[i] = cmap_type.resolution;
        cmap.type_offset[i] = 16 * cmap.unique_gridpoint_numbers;
        cmap.unique_gridpoint_numbers +=
            cmap_type.resolution * cmap_type.resolution;
        for (float value : cmap_type.grid)
        {
            cmap.grid_value.push_back(Gromacs_To_Kcal(value));
        }
    }

    std::unordered_map<std::string, int> atom_type_id;
    std::vector<std::string> ordered_types;
    for (const std::string& atom_type_name : global_atom_types)
    {
        const Gromacs_Atom_Type& atom_type =
            topology.atom_types.at(atom_type_name);
        std::pair<float, float> self_c6_c12 =
            Gromacs_Get_C6_C12(topology.defaults, atom_type, atom_type);
        char lj_key[CHAR_LENGTH_MAX];
        snprintf(lj_key, sizeof(lj_key), "%.9g|%.9g", self_c6_c12.first,
                 self_c6_c12.second);
        auto iter = atom_type_id.find(lj_key);
        if (iter == atom_type_id.end())
        {
            int next_id = static_cast<int>(atom_type_id.size());
            atom_type_id[lj_key] = next_id;
            ordered_types.push_back(atom_type_name);
            iter = atom_type_id.find(lj_key);
        }
        system->classical_force_field.lj.atom_type.push_back(iter->second);
    }
    system->classical_force_field.lj.atom_type_numbers =
        static_cast<int>(atom_type_id.size());
    int pair_type_numbers =
        system->classical_force_field.lj.atom_type_numbers *
        (system->classical_force_field.lj.atom_type_numbers + 1) / 2;
    system->classical_force_field.lj.pair_A.resize(pair_type_numbers);
    system->classical_force_field.lj.pair_B.resize(pair_type_numbers);

    for (int type_j_index = 0;
         type_j_index < static_cast<int>(ordered_types.size()); type_j_index++)
    {
        for (int type_i_index = 0; type_i_index <= type_j_index; type_i_index++)
        {
            const Gromacs_Atom_Type& type_i =
                topology.atom_types.at(ordered_types[type_i_index]);
            const Gromacs_Atom_Type& type_j =
                topology.atom_types.at(ordered_types[type_j_index]);
            std::pair<float, float> c6_c12 =
                Gromacs_Get_C6_C12(topology.defaults, type_i, type_j);
            int pair_id = type_j_index * (type_j_index + 1) / 2 + type_i_index;
            system->classical_force_field.lj.pair_A[pair_id] =
                12.0f * c6_c12.second;
            system->classical_force_field.lj.pair_B[pair_id] =
                6.0f * c6_c12.first;
        }
    }

    int molecule_index = 0;
    for (const auto& item : topology.system_molecules)
    {
        const Gromacs_Molecule& molecule = topology.molecules.at(item.first);
        for (int copy = 0; copy < item.second; copy++, molecule_index++)
        {
            const std::vector<int>& local_to_global =
                molecule_local_to_global[molecule_index];
            std::vector<std::vector<int>> adjacency(molecule.atoms.size());

            auto require_local_atom = [&](int local_index)
            {
                if (local_index < 0 ||
                    local_index >= static_cast<int>(molecule.atoms.size()))
                {
                    controller->Throw_SPONGE_Error(
                        spongeErrorBadFileFormat, error_by,
                        "Reason:\n\tGROMACS molecule atom index is out of "
                        "range\n");
                }
            };

            auto append_bond =
                [&](int ai_local, int aj_local, float k, float r0)
            {
                require_local_atom(ai_local);
                require_local_atom(aj_local);
                bonds.atom_a.push_back(local_to_global[ai_local]);
                bonds.atom_b.push_back(local_to_global[aj_local]);
                bonds.k.push_back(k);
                bonds.r0.push_back(r0);
                adjacency[ai_local].push_back(aj_local);
                adjacency[aj_local].push_back(ai_local);
            };

            auto append_constraint = [&](int ai_local, int aj_local, float r0)
            {
                require_local_atom(ai_local);
                require_local_atom(aj_local);
                constraints.atom_a.push_back(local_to_global[ai_local]);
                constraints.atom_b.push_back(local_to_global[aj_local]);
                constraints.r0.push_back(r0);
            };

            for (const Gromacs_Bond& bond : molecule.bonds)
            {
                int ai_local = bond.ai - 1;
                int aj_local = bond.aj - 1;
                require_local_atom(ai_local);
                require_local_atom(aj_local);
                const Gromacs_Molecule_Atom& atom_i = molecule.atoms[ai_local];
                const Gromacs_Molecule_Atom& atom_j = molecule.atoms[aj_local];
                const Gromacs_Bond_Type* type = NULL;
                if (bond.parameters.size() >= 2)
                {
                    append_bond(ai_local, aj_local,
                                Gromacs_To_Kcal(bond.parameters[1]) / 200.0f,
                                Gromacs_To_Angstrom(bond.parameters[0]));
                }
                else
                {
                    type =
                        Gromacs_Find_Bond_Type(topology.bond_types, atom_i.type,
                                               atom_j.type, bond.funct);
                    if (type == NULL)
                    {
                        controller->Throw_SPONGE_Error(
                            spongeErrorBadFileFormat, error_by,
                            "Reason:\n\tfailed to find GROMACS bond type\n");
                    }
                    append_bond(ai_local, aj_local,
                                Gromacs_To_Kcal(type->kb) / 200.0f,
                                Gromacs_To_Angstrom(type->b0));
                }
            }

            for (const Gromacs_Settle& settle : molecule.settles)
            {
                int oxygen_local = settle.ow - 1;
                int hydrogen_1_local = oxygen_local + 1;
                int hydrogen_2_local = oxygen_local + 2;
                require_local_atom(oxygen_local);
                require_local_atom(hydrogen_1_local);
                require_local_atom(hydrogen_2_local);
                append_bond(oxygen_local, hydrogen_1_local, 0.0f,
                            Gromacs_To_Angstrom(settle.doh));
                append_constraint(oxygen_local, hydrogen_1_local,
                                  Gromacs_To_Angstrom(settle.doh));
                append_bond(oxygen_local, hydrogen_2_local, 0.0f,
                            Gromacs_To_Angstrom(settle.doh));
                append_constraint(oxygen_local, hydrogen_2_local,
                                  Gromacs_To_Angstrom(settle.doh));
                append_bond(hydrogen_1_local, hydrogen_2_local, 0.0f,
                            Gromacs_To_Angstrom(settle.dhh));
                append_constraint(hydrogen_1_local, hydrogen_2_local,
                                  Gromacs_To_Angstrom(settle.dhh));
            }

            for (const Gromacs_Constraint& constraint : molecule.constraints)
            {
                if (constraint.parameters.empty())
                {
                    controller->Throw_SPONGE_Error(
                        spongeErrorBadFileFormat, error_by,
                        "Reason:\n\tfailed to resolve GROMACS constraint "
                        "distance\n");
                }
                append_bond(constraint.ai - 1, constraint.aj - 1, 0.0f,
                            Gromacs_To_Angstrom(constraint.parameters[0]));
                append_constraint(
                    constraint.ai - 1, constraint.aj - 1,
                    Gromacs_To_Angstrom(constraint.parameters[0]));
            }

            for (int i = 0; i < static_cast<int>(molecule.atoms.size()); i++)
            {
                std::vector<int> distance(molecule.atoms.size(), -1);
                std::queue<int> queue;
                distance[i] = 0;
                queue.push(i);
                while (!queue.empty())
                {
                    int current = queue.front();
                    queue.pop();
                    if (distance[current] >= molecule.nrexcl)
                    {
                        continue;
                    }
                    for (int next : adjacency[current])
                    {
                        if (distance[next] >= 0)
                        {
                            continue;
                        }
                        distance[next] = distance[current] + 1;
                        queue.push(next);
                    }
                }
                for (int j = i + 1; j < static_cast<int>(molecule.atoms.size());
                     j++)
                {
                    if (distance[j] > 0 && distance[j] <= molecule.nrexcl)
                    {
                        system->exclusions.excluded_atoms[local_to_global[i]]
                            .push_back(local_to_global[j]);
                    }
                }
            }

            for (const Gromacs_CMap& cmap_item : molecule.cmaps)
            {
                int ai_local = cmap_item.ai - 1;
                int aj_local = cmap_item.aj - 1;
                int ak_local = cmap_item.ak - 1;
                int al_local = cmap_item.al - 1;
                int am_local = cmap_item.am - 1;
                require_local_atom(ai_local);
                require_local_atom(aj_local);
                require_local_atom(ak_local);
                require_local_atom(al_local);
                require_local_atom(am_local);
                const Gromacs_Molecule_Atom& atom_i = molecule.atoms[ai_local];
                const Gromacs_Molecule_Atom& atom_j = molecule.atoms[aj_local];
                const Gromacs_Molecule_Atom& atom_k = molecule.atoms[ak_local];
                const Gromacs_Molecule_Atom& atom_l = molecule.atoms[al_local];
                const Gromacs_Molecule_Atom& atom_m = molecule.atoms[am_local];
                int cmap_type = Gromacs_Find_CMap_Type(
                    topology.cmap_types, atom_i.type, atom_j.type, atom_k.type,
                    atom_l.type, atom_m.type, cmap_item.funct);
                if (cmap_type < 0)
                {
                    controller->Throw_SPONGE_Error(
                        spongeErrorBadFileFormat, error_by,
                        "Reason:\n\tfailed to find GROMACS CMAP type\n");
                }
                cmap.atom_a.push_back(local_to_global[ai_local]);
                cmap.atom_b.push_back(local_to_global[aj_local]);
                cmap.atom_c.push_back(local_to_global[ak_local]);
                cmap.atom_d.push_back(local_to_global[al_local]);
                cmap.atom_e.push_back(local_to_global[am_local]);
                cmap.cmap_type.push_back(cmap_type);
            }

            for (const Gromacs_Angle& angle : molecule.angles)
            {
                int ai_local = angle.ai - 1;
                int aj_local = angle.aj - 1;
                int ak_local = angle.ak - 1;
                const Gromacs_Molecule_Atom& atom_i = molecule.atoms[ai_local];
                const Gromacs_Molecule_Atom& atom_j = molecule.atoms[aj_local];
                const Gromacs_Molecule_Atom& atom_k = molecule.atoms[ak_local];
                float theta0 = 0.0f;
                float angle_k = 0.0f;
                float ub0 = 0.0f;
                float kub = 0.0f;
                if (angle.parameters.size() >= 2)
                {
                    theta0 = Gromacs_To_Radian(angle.parameters[0]);
                    angle_k = Gromacs_To_Kcal(angle.parameters[1]) / 2.0f;
                    if (angle.funct == 5 && angle.parameters.size() >= 4)
                    {
                        ub0 = Gromacs_To_Angstrom(angle.parameters[2]);
                        kub = Gromacs_To_Kcal(angle.parameters[3]) / 200.0f;
                    }
                }
                else
                {
                    const Gromacs_Angle_Type* type = Gromacs_Find_Angle_Type(
                        topology.angle_types, atom_i.type, atom_j.type,
                        atom_k.type, angle.funct);
                    if (type == NULL)
                    {
                        controller->Throw_SPONGE_Error(
                            spongeErrorBadFileFormat, error_by,
                            "Reason:\n\tfailed to find GROMACS angle type\n");
                    }
                    theta0 = Gromacs_To_Radian(type->theta0);
                    angle_k = Gromacs_To_Kcal(type->k) / 2.0f;
                    if (angle.funct == 5)
                    {
                        ub0 = Gromacs_To_Angstrom(type->ub0);
                        kub = Gromacs_To_Kcal(type->kub) / 200.0f;
                    }
                }
                urey.atom_a.push_back(local_to_global[ai_local]);
                urey.atom_b.push_back(local_to_global[aj_local]);
                urey.atom_c.push_back(local_to_global[ak_local]);
                urey.angle_k.push_back(angle_k);
                urey.angle_theta0.push_back(theta0);
                urey.bond_k.push_back(kub);
                urey.bond_r0.push_back(ub0);
            }

            for (const Gromacs_Dihedral& dihedral : molecule.dihedrals)
            {
                int ai_local = dihedral.ai - 1;
                int aj_local = dihedral.aj - 1;
                int ak_local = dihedral.ak - 1;
                int al_local = dihedral.al - 1;
                const Gromacs_Molecule_Atom& atom_i = molecule.atoms[ai_local];
                const Gromacs_Molecule_Atom& atom_j = molecule.atoms[aj_local];
                const Gromacs_Molecule_Atom& atom_k = molecule.atoms[ak_local];
                const Gromacs_Molecule_Atom& atom_l = molecule.atoms[al_local];

                auto append_proper =
                    [&](float phase_deg, float k_kj, int multiplicity)
                {
                    dihedrals.atom_a.push_back(local_to_global[ai_local]);
                    dihedrals.atom_b.push_back(local_to_global[aj_local]);
                    dihedrals.atom_c.push_back(local_to_global[ak_local]);
                    dihedrals.atom_d.push_back(local_to_global[al_local]);
                    dihedrals.ipn.push_back(multiplicity);
                    dihedrals.pn.push_back(static_cast<float>(multiplicity));
                    dihedrals.pk.push_back(Gromacs_To_Kcal(k_kj));
                    float phase = Gromacs_To_Radian(phase_deg);
                    dihedrals.gamc.push_back(cosf(phase) * dihedrals.pk.back());
                    dihedrals.gams.push_back(sinf(phase) * dihedrals.pk.back());
                };
                auto append_improper = [&](float phase_deg, float k_kj)
                {
                    impropers.atom_a.push_back(local_to_global[ai_local]);
                    impropers.atom_b.push_back(local_to_global[aj_local]);
                    impropers.atom_c.push_back(local_to_global[ak_local]);
                    impropers.atom_d.push_back(local_to_global[al_local]);
                    impropers.pk.push_back(Gromacs_To_Kcal(k_kj));
                    impropers.pn.push_back(0.0f);
                    impropers.ipn.push_back(0);
                    impropers.gamc.push_back(Gromacs_To_Radian(phase_deg));
                    impropers.gams.push_back(0.0f);
                };

                if (!dihedral.parameters.empty())
                {
                    if (dihedral.funct == 2)
                    {
                        append_improper(dihedral.parameters[0],
                                        dihedral.parameters[1]);
                    }
                    else
                    {
                        append_proper(
                            dihedral.parameters[0], dihedral.parameters[1],
                            static_cast<int>(
                                std::lround(dihedral.parameters.size() >= 3
                                                ? dihedral.parameters[2]
                                                : 1.0f)));
                    }
                    continue;
                }

                std::vector<const Gromacs_Dihedral_Type*> types =
                    Gromacs_Find_Dihedral_Types(
                        topology.dihedral_types, atom_i.type, atom_j.type,
                        atom_k.type, atom_l.type, dihedral.funct);
                if (types.empty())
                {
                    controller->Throw_SPONGE_Error(
                        spongeErrorBadFileFormat, error_by,
                        "Reason:\n\tfailed to find GROMACS dihedral type\n");
                }
                for (const Gromacs_Dihedral_Type* type : types)
                {
                    if (dihedral.funct == 2)
                    {
                        append_improper(type->parameters[0],
                                        type->parameters[1]);
                    }
                    else
                    {
                        append_proper(type->parameters[0], type->parameters[1],
                                      static_cast<int>(std::lround(
                                          type->parameters.size() >= 3
                                              ? type->parameters[2]
                                              : 1.0f)));
                    }
                }
            }

            for (const Gromacs_Pair& pair : molecule.pairs)
            {
                int ai_local = pair.ai - 1;
                int aj_local = pair.aj - 1;
                const Gromacs_Molecule_Atom& atom_i = molecule.atoms[ai_local];
                const Gromacs_Molecule_Atom& atom_j = molecule.atoms[aj_local];
                std::pair<float, float> c6_c12{0.0f, 0.0f};
                if (pair.parameters.size() >= 2)
                {
                    c6_c12 = Gromacs_Get_C6_C12_From_Pair_Parameters(
                        topology.defaults, pair.parameters);
                }
                else
                {
                    const Gromacs_Pair_Type* pair_type =
                        Gromacs_Find_Pair_Type(topology.pair_types, atom_i.type,
                                               atom_j.type, pair.funct);
                    if (pair_type != NULL)
                    {
                        c6_c12 = Gromacs_Get_C6_C12_From_Pair_Parameters(
                            topology.defaults, pair_type->parameters);
                    }
                    else if (topology.defaults.gen_pairs)
                    {
                        c6_c12 = Gromacs_Get_C6_C12(
                            topology.defaults,
                            topology.atom_types.at(atom_i.type),
                            topology.atom_types.at(atom_j.type));
                        c6_c12.first *= topology.defaults.fudge_lj;
                        c6_c12.second *= topology.defaults.fudge_lj;
                    }
                    else
                    {
                        controller->Throw_SPONGE_Error(
                            spongeErrorBadFileFormat, error_by,
                            "Reason:\n\tfailed to resolve GROMACS pair "
                            "interaction\n");
                    }
                }
                nb14.atom_a.push_back(local_to_global[ai_local]);
                nb14.atom_b.push_back(local_to_global[aj_local]);
                nb14.A.push_back(12.0f * c6_c12.second);
                nb14.B.push_back(6.0f * c6_c12.first);
                nb14.cf_scale_factor.push_back(topology.defaults.fudge_qq);
            }
        }
    }
}

void Load_Gromacs_Inputs(System* system, CONTROLLER* controller)
{
    Gromacs_Topology topology = Gromacs_Parse_Topology(controller);
    Gromacs_Instantiate_System(topology, system, controller);
    Gromacs_Load_Gro(system, controller);
}

}  // namespace Xponge

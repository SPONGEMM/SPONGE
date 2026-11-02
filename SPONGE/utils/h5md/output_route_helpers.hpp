#pragma once

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace SpongeH5OutputRoute
{
inline std::string Sanitize_Output_Name(const std::string& name)
{
    std::string sanitized;
    sanitized.reserve(name.size());
    for (unsigned char c : name)
    {
        if (std::isalnum(c) || c == '_')
        {
            sanitized.push_back(static_cast<char>(c));
        }
        else
        {
            sanitized.push_back('_');
        }
    }
    if (sanitized.empty() ||
        std::isdigit(static_cast<unsigned char>(sanitized[0])))
    {
        sanitized.insert(sanitized.begin(), '_');
    }
    return sanitized;
}

inline std::vector<std::string> Make_Unique_Output_Names(
    const std::vector<std::string>& names)
{
    std::map<std::string, int> next_suffix;
    std::map<std::string, bool> used_names;
    std::vector<std::string> ret;
    ret.reserve(names.size());
    for (const std::string& name : names)
    {
        std::string base_name = Sanitize_Output_Name(name);
        if (base_name == "step" || base_name == "time")
        {
            base_name = "mdout_" + base_name;
        }
        std::string candidate = base_name;
        int& suffix = next_suffix[base_name];
        while (used_names[candidate])
        {
            ++suffix;
            candidate = base_name + "_" + std::to_string(suffix);
        }
        used_names[candidate] = true;
        ret.push_back(candidate);
    }
    return ret;
}

inline bool Parse_Output_Double(const std::string& text, double* value)
{
    if (value == NULL) return false;
    char* end = NULL;
    const double parsed = std::strtod(text.c_str(), &end);
    if (end == text.c_str()) return false;
    while (end != NULL && *end != '\0')
    {
        if (!std::isspace(static_cast<unsigned char>(*end))) return false;
        ++end;
    }
    *value = parsed;
    return true;
}

inline bool Read_Text_File_If_Present(const char* file_name, std::string* text)
{
    if (file_name == NULL || text == NULL || file_name[0] == '\0')
    {
        return false;
    }
    std::ifstream input(file_name, std::ios::in | std::ios::binary);
    if (!input.good())
    {
        return false;
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    *text = buffer.str();
    return true;
}

inline bool Is_Reaxff_Output_Key(const std::string& name)
{
    return name == "REAXFF" || name.rfind("REAXFF_", 0) == 0;
}

inline bool Output_Key_Exists(const std::vector<std::string>& keys,
                              const char* name)
{
    if (name == NULL) return false;
    for (const std::string& key : keys)
    {
        if (key == name) return true;
    }
    return false;
}
}  // namespace SpongeH5OutputRoute

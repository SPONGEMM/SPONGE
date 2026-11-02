#include "toml.h"

#include <sstream>

#include "toml_decode.hpp"

namespace sponge::toml_wrap
{
namespace
{
using DecodeNode = sponge::toml_decode::node;
using DecodeArray = sponge::toml_decode::array;
using DecodeTable = sponge::toml_decode::table;

std::string CanonicalCommandKey(const std::string& full_key)
{
    struct Alias
    {
        const char* alias;
        const char* canonical;
    };
    static const Alias aliases[] = {
        {"write_interval_information", "write_information_interval"},
        {"write_interval_trajectory", "write_trajectory_interval"},
        {"write_interval_mdout", "write_mdout_interval"},
        {"write_interval_restart", "write_restart_file_interval"},
        {"write_interval_restart_file", "write_restart_file_interval"},
        // Flat shorthand for [xtc] path; the table form already flattens to
        // xtc_path and needs no alias.
        {"xtc", "xtc_path"},
    };

    for (const auto& alias : aliases)
    {
        if (full_key == alias.alias)
        {
            return alias.canonical;
        }
    }
    const std::string mc_barostat_prefix = "barostat_monte_carlo_";
    if (full_key.rfind(mc_barostat_prefix, 0) == 0)
    {
        return "monte_carlo_barostat_" +
               full_key.substr(mc_barostat_prefix.size());
    }
    return full_key;
}

bool RequiresStructuredSerialization(const DecodeNode& node)
{
    if (node.as_table() != nullptr)
    {
        return true;
    }
    const auto* arr = node.as_array();
    if (arr == nullptr)
    {
        return false;
    }
    for (const auto& item : *arr)
    {
        if (item.as_table() != nullptr || item.as_array() != nullptr)
        {
            return true;
        }
    }
    return false;
}

std::string EscapeTomlBasicString(const std::string& input)
{
    std::ostringstream oss;
    for (char c : input)
    {
        switch (c)
        {
            case '\\':
                oss << "\\\\";
                break;
            case '"':
                oss << "\\\"";
                break;
            case '\b':
                oss << "\\b";
                break;
            case '\f':
                oss << "\\f";
                break;
            case '\n':
                oss << "\\n";
                break;
            case '\r':
                oss << "\\r";
                break;
            case '\t':
                oss << "\\t";
                break;
            default:
                oss << c;
                break;
        }
    }
    return oss.str();
}

std::string SerializeNodeCompact(const DecodeNode& node);

std::string SerializeArrayCompact(const DecodeArray& arr)
{
    std::ostringstream oss;
    oss << '[';
    bool first = true;
    for (const auto& item : arr)
    {
        if (!first)
        {
            oss << ',';
        }
        oss << SerializeNodeCompact(item);
        first = false;
    }
    oss << ']';
    return oss.str();
}

std::string SerializeTableCompact(const DecodeTable& table)
{
    std::ostringstream oss;
    oss << '{';
    bool first = true;
    for (const auto& [key, value] : table)
    {
        if (!first)
        {
            oss << ',';
        }
        oss << key << '=' << SerializeNodeCompact(value);
        first = false;
    }
    oss << '}';
    return oss.str();
}

std::string SerializeNodeCompact(const DecodeNode& node)
{
    if (const auto* val = node.as_string())
    {
        return "\"" + EscapeTomlBasicString(*val) + "\"";
    }
    if (const auto* val = node.as_integer())
    {
        return std::to_string(*val);
    }
    if (const auto* val = node.as_floating())
    {
        std::ostringstream oss;
        oss << *val;
        return oss.str();
    }
    if (const auto* val = node.as_bool())
    {
        return *val ? "true" : "false";
    }
    if (const auto* arr = node.as_array())
    {
        return SerializeArrayCompact(*arr);
    }
    if (const auto* table = node.as_table())
    {
        return SerializeTableCompact(*table);
    }
    throw std::runtime_error("unsupported TOML node type for serialization");
}

std::string NodeValueToString(const DecodeNode& node,
                              const std::string& full_key,
                              std::string* error_message)
{
    if (const auto* val = node.as_string())
    {
        return *val;
    }
    if (const auto* val = node.as_integer())
    {
        return std::to_string(*val);
    }
    if (const auto* val = node.as_floating())
    {
        std::ostringstream oss;
        oss << *val;
        return oss.str();
    }
    if (const auto* val = node.as_bool())
    {
        return *val ? "true" : "false";
    }
    if (const auto* arr = node.as_array())
    {
        if (RequiresStructuredSerialization(node))
        {
            return SerializeArrayCompact(*arr);
        }
        std::ostringstream oss;
        bool first = true;
        for (const auto& item : *arr)
        {
            if (!first)
            {
                oss << ' ';
            }
            std::string item_value =
                NodeValueToString(item, full_key, error_message);
            if (!error_message->empty())
            {
                return "";
            }
            oss << item_value;
            first = false;
        }
        return oss.str();
    }
    if (const auto* table = node.as_table())
    {
        return SerializeTableCompact(*table);
    }
    *error_message = "unsupported TOML value type for '" + full_key + "'";
    return "";
}

bool FlattenTable(const DecodeTable& table, const std::string& prefix,
                  std::map<std::string, std::string>* parsed_commands,
                  std::string* error_message)
{
    for (const auto& [key, value] : table)
    {
        if (const auto* nested = value.as_table())
        {
            const std::string next_prefix =
                prefix.empty() ? key : prefix + "_" + key;
            if (!FlattenTable(*nested, next_prefix, parsed_commands,
                              error_message))
            {
                return false;
            }
            continue;
        }

        const std::string full_key = prefix.empty() ? key : prefix + "_" + key;
        std::string value_str =
            NodeValueToString(value, full_key, error_message);
        if (!error_message->empty())
        {
            return false;
        }
        const std::string canonical_key = CanonicalCommandKey(full_key);
        const auto insert_result =
            parsed_commands->emplace(canonical_key, value_str);
        if (!insert_result.second)
        {
            *error_message = "TOML key '" + full_key +
                             "' conflicts with an existing command for '" +
                             canonical_key + "'";
            return false;
        }
    }
    return true;
}
}  // namespace

bool ParseAndFlatten(const std::string& content, const std::string& source_path,
                     std::map<std::string, std::string>* parsed_commands,
                     std::string* error_message)
{
    if (parsed_commands == nullptr || error_message == nullptr)
    {
        return false;
    }
    parsed_commands->clear();
    error_message->clear();

    try
    {
        const auto config = sponge::toml_decode::detail::parse_toml_string(
            content, source_path);
        return FlattenTable(config, "", parsed_commands, error_message);
    }
    catch (const std::exception& err)
    {
        *error_message = err.what();
        return false;
    }
}
}  // namespace sponge::toml_wrap

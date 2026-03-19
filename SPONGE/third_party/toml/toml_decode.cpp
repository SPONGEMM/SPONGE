#define TOML_HEADER_ONLY 0
#define TOML_IMPLEMENTATION
#include <toml++/toml.h>

#include "toml_decode.hpp"

namespace sponge::toml_decode::detail
{

namespace
{

auto convert_node(const toml::node& input) -> sponge::toml_decode::node;

auto convert_array(const toml::array& input) -> sponge::toml_decode::array
{
    sponge::toml_decode::array out;
    out.reserve(input.size());
    for (const auto& item : input)
    {
        out.push_back(convert_node(item));
    }
    return out;
}

auto convert_table(const toml::table& input) -> sponge::toml_decode::table
{
    sponge::toml_decode::table out;
    out.reserve(input.size());
    for (const auto& [key, value] : input)
    {
        out.emplace(std::string(key.str()), convert_node(value));
    }
    return out;
}

auto convert_node(const toml::node& input) -> sponge::toml_decode::node
{
    if (const auto* value = input.as_integer())
    {
        return sponge::toml_decode::node{value->get()};
    }
    if (const auto* value = input.as_floating_point())
    {
        return sponge::toml_decode::node{value->get()};
    }
    if (const auto* value = input.as_boolean())
    {
        return sponge::toml_decode::node{value->get()};
    }
    if (const auto* value = input.as_string())
    {
        return sponge::toml_decode::node{std::string(value->get())};
    }
    if (const auto* value = input.as_array())
    {
        return sponge::toml_decode::node{convert_array(*value)};
    }
    if (const auto* value = input.as_table())
    {
        return sponge::toml_decode::node{convert_table(*value)};
    }

    throw std::runtime_error("unsupported TOML node type");
}

}  // namespace

auto parse_toml_file(std::string_view path) -> table
{
    const auto parsed = toml::parse_file(std::string(path));
    return convert_table(parsed);
}

}  // namespace sponge::toml_decode::detail

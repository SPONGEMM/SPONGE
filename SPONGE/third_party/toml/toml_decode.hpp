#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace sponge::toml_decode
{

struct node;
using array = std::vector<node>;
using table = std::unordered_map<std::string, node>;

struct node
{
    using array_ptr = std::shared_ptr<array>;
    using table_ptr = std::shared_ptr<table>;
    using storage_type = std::variant<std::int64_t, double, bool, std::string,
                                      array_ptr, table_ptr>;

    storage_type storage;

    node() = default;
    explicit node(std::int64_t value) : storage(value) {}
    explicit node(double value) : storage(value) {}
    explicit node(bool value) : storage(value) {}
    explicit node(std::string value) : storage(std::move(value)) {}
    explicit node(array value)
        : storage(std::make_shared<array>(std::move(value)))
    {
    }
    explicit node(table value)
        : storage(std::make_shared<table>(std::move(value)))
    {
    }

    const std::int64_t* as_integer() const
    {
        return std::get_if<std::int64_t>(&storage);
    }

    const double* as_floating() const { return std::get_if<double>(&storage); }

    const bool* as_bool() const { return std::get_if<bool>(&storage); }

    const std::string* as_string() const
    {
        return std::get_if<std::string>(&storage);
    }

    const array* as_array() const
    {
        if (const auto* ptr = std::get_if<array_ptr>(&storage))
        {
            return ptr->get();
        }
        return nullptr;
    }

    const table* as_table() const
    {
        if (const auto* ptr = std::get_if<table_ptr>(&storage))
        {
            return ptr->get();
        }
        return nullptr;
    }
};

template <class Owner, class Value>
struct field_descriptor
{
    using owner_type = Owner;
    using value_type = Value;

    std::string_view name;
    Value Owner::* member;
};

template <class Owner, class Value>
constexpr auto field(std::string_view name, Value Owner::* member)
    -> field_descriptor<Owner, Value>
{
    return {name, member};
}

template <class T>
struct reflect;

template <class T, class = void>
struct has_reflect : std::false_type
{
};

template <class T>
struct has_reflect<T, std::void_t<decltype(reflect<T>::fields())>>
    : std::true_type
{
};

template <class T>
inline constexpr bool has_reflect_v = has_reflect<T>::value;

template <class T>
struct is_std_vector : std::false_type
{
};

template <class T, class Allocator>
struct is_std_vector<std::vector<T, Allocator>> : std::true_type
{
};

template <class T>
inline constexpr bool is_std_vector_v = is_std_vector<T>::value;

template <class T>
struct is_std_optional : std::false_type
{
};

template <class T>
struct is_std_optional<std::optional<T>> : std::true_type
{
};

template <class T>
inline constexpr bool is_std_optional_v = is_std_optional<T>::value;

template <class T>
T decode_node(const node& value);

template <class T>
T decode_table(const table& value);

template <class T>
T parse_file(std::string_view path);

template <class T>
T parse_string(std::string_view content, std::string_view source_path = {});

namespace detail
{

template <class T>
using decay_t = std::remove_cv_t<std::remove_reference_t<T>>;

template <class T>
struct vector_value_type;

template <class T, class Allocator>
struct vector_value_type<std::vector<T, Allocator>>
{
    using type = T;
};

template <class T>
struct optional_value_type;

template <class T>
struct optional_value_type<std::optional<T>>
{
    using type = T;
};

auto parse_toml_file(std::string_view path) -> table;
auto parse_toml_string(std::string_view content,
                       std::string_view source_path = {}) -> table;

inline const node* find_node(const table& value, std::string_view key)
{
    auto it = value.find(std::string(key));
    if (it == value.end())
    {
        return nullptr;
    }
    return &it->second;
}

[[noreturn]] inline void throw_type_error(std::string_view target)
{
    throw std::runtime_error("type mismatch: expected " + std::string(target));
}

template <class T, class Desc>
void assign_field(const table& input, T& out, const Desc& desc)
{
    using field_type = decay_t<decltype(out.*(desc.member))>;

    const auto* child = find_node(input, desc.name);
    if (!child)
    {
        return;
    }

    out.*(desc.member) = decode_node<field_type>(*child);
}

template <class T, class Tuple, std::size_t... Indices>
void assign_all_fields(const table& input, T& out, const Tuple& fields,
                       std::index_sequence<Indices...>)
{
    (assign_field(input, out, std::get<Indices>(fields)), ...);
}

}  // namespace detail

template <class T>
T decode_node(const node& value)
{
    if constexpr (std::is_same_v<T, int>)
    {
        if (const auto* integer = value.as_integer())
        {
            return static_cast<int>(*integer);
        }
        detail::throw_type_error("int");
    }
    else if constexpr (std::is_same_v<T, float>)
    {
        if (const auto* floating = value.as_floating())
        {
            return static_cast<float>(*floating);
        }
        if (const auto* integer = value.as_integer())
        {
            return static_cast<float>(*integer);
        }
        detail::throw_type_error("float");
    }
    else if constexpr (std::is_same_v<T, double>)
    {
        if (const auto* floating = value.as_floating())
        {
            return *floating;
        }
        if (const auto* integer = value.as_integer())
        {
            return static_cast<double>(*integer);
        }
        detail::throw_type_error("double");
    }
    else if constexpr (std::is_same_v<T, bool>)
    {
        if (const auto* boolean = value.as_bool())
        {
            return *boolean;
        }
        detail::throw_type_error("bool");
    }
    else if constexpr (std::is_same_v<T, std::string>)
    {
        if (const auto* string = value.as_string())
        {
            return *string;
        }
        detail::throw_type_error("string");
    }
    else if constexpr (is_std_optional_v<T>)
    {
        using value_type = typename detail::optional_value_type<T>::type;
        return decode_node<value_type>(value);
    }
    else if constexpr (is_std_vector_v<T>)
    {
        using element_type = typename detail::vector_value_type<T>::type;

        if (const auto* items = value.as_array())
        {
            T out;
            out.reserve(items->size());
            for (const auto& item : *items)
            {
                out.push_back(decode_node<element_type>(item));
            }
            return out;
        }
        detail::throw_type_error("array");
    }
    else if constexpr (has_reflect_v<T>)
    {
        if (const auto* object = value.as_table())
        {
            return decode_table<T>(*object);
        }
        detail::throw_type_error("table");
    }
    else
    {
        static_assert(sizeof(T) == 0,
                      "decode_node<T> does not support this type yet");
    }
}

template <class T>
T decode_table(const table& value)
{
    static_assert(has_reflect_v<T>,
                  "decode_table<T> requires reflect<T>::fields()");

    T out{};
    constexpr auto fields = reflect<T>::fields();
    detail::assign_all_fields(
        value, out, fields,
        std::make_index_sequence<std::tuple_size_v<decltype(fields)>>{});
    return out;
}

template <class T>
T parse_file(std::string_view path)
{
    return decode_table<T>(detail::parse_toml_file(path));
}

template <class T>
T parse_string(std::string_view content, std::string_view source_path)
{
    return decode_table<T>(detail::parse_toml_string(content, source_path));
}

}  // namespace sponge::toml_decode

#define SPONGE_TOML_DECODE_FIELD_2(Type, member) \
    ::sponge::toml_decode::field(#member, &Type::member)
#define SPONGE_TOML_DECODE_FIELD_3(Type, key, member) \
    ::sponge::toml_decode::field(key, &Type::member)
#define SPONGE_TOML_DECODE_MEMBER(Type, member) \
    SPONGE_TOML_DECODE_FIELD_2(Type, member)
#define SPONGE_TOML_DECODE_NAMED_MEMBER(Type, key, member) \
    SPONGE_TOML_DECODE_FIELD_3(Type, key, member)

#define SPONGE_TOML_DECODE_REFLECT(Type, ...)    \
    namespace sponge::toml_decode                \
    {                                            \
    template <>                                  \
    struct reflect<Type>                         \
    {                                            \
        static constexpr auto fields()           \
        {                                        \
            return std::make_tuple(__VA_ARGS__); \
        }                                        \
    };                                           \
    }

#include "drive_config.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>

#include "../third_party/toml/toml_decode.hpp"
#include "../utils/float_classification.hpp"

namespace SpongeDynamics
{
namespace
{
using Node = sponge::toml_decode::node;
using Table = sponge::toml_decode::table;
constexpr double pi = 3.14159265358979323846;

[[noreturn]] void Bad(const std::string& key, const std::string& reason)
{
    throw std::runtime_error(key + ": " + reason);
}

void Keys(const Table& table, const std::set<std::string>& allowed,
          const std::string& key)
{
    for (const auto& item : table)
        if (!allowed.count(item.first))
            Bad(key + "." + item.first, "unknown parameter");
}

const Node& Required(const Table& table, const char* name,
                     const std::string& key)
{
    auto it = table.find(name);
    if (it == table.end())
        Bad(key + "." + name, "required parameter is missing");
    return it->second;
}

std::string String(const Node& node, const std::string& key)
{
    if (auto value = node.as_string()) return *value;
    Bad(key, "expected a string");
}

double Number(const Node& node, const std::string& key)
{
    double value;
    if (auto floating = node.as_floating())
        value = *floating;
    else if (auto integer = node.as_integer())
        value = static_cast<double>(*integer);
    else
        Bad(key, "expected a number");
    if (!SpongeFloat::Is_Finite(value)) Bad(key, "must be finite");
    return value;
}

Vec3 Vector(const Node& node, const std::string& key)
{
    auto array = node.as_array();
    if (!array || array->size() != 3)
        Bad(key, "expected exactly three numbers");
    return {Number((*array)[0], key), Number((*array)[1], key),
            Number((*array)[2], key)};
}

Schedule Parse_Schedule(const Node& node, const std::string& key)
{
    auto table = node.as_table();
    if (!table) Bad(key, "expected a table");
    Schedule out;
    auto type = String(Required(*table, "type", key), key + ".type");
    if (type == "constant")
    {
        Keys(*table, {"type"}, key);
    }
    else if (type == "ramp" || type == "sine")
    {
        const bool ramp = type == "ramp";
        Keys(*table,
             ramp ? std::set<std::string>{"type", "start", "duration"}
                  : std::set<std::string>{"type", "start", "period", "phase"},
             key);
        out.kind = ramp ? ScheduleKind::Ramp : ScheduleKind::Sine;
        if (table->count("start"))
            out.start = Number(table->at("start"), key + ".start");
        const char* length = ramp ? "duration" : "period";
        double value =
            Number(Required(*table, length, key), key + "." + length);
        if (value <= 0) Bad(key + "." + length, "must be positive (ps)");
        if (ramp)
            out.duration = value;
        else
        {
            out.period = value;
            if (table->count("phase"))
                out.phase = Number(table->at("phase"), key + ".phase");
        }
    }
    else if (type == "table")
    {
        Keys(*table, {"type", "points"}, key);
        out.kind = ScheduleKind::Table;
        auto array = Required(*table, "points", key).as_array();
        if (!array || array->empty())
            Bad(key + ".points", "expected nonempty [time_ps, scale] pairs");
        for (const auto& row : *array)
        {
            auto pair = row.as_array();
            if (!pair || pair->size() != 2)
                Bad(key + ".points", "expected [time_ps, scale] pairs");
            const double time = Number((*pair)[0], key + ".points.time");
            const double scale = Number((*pair)[1], key + ".points.scale");
            if (!out.points.empty() && time <= out.points.back()[0])
                Bad(key + ".points", "times must strictly increase");
            out.points.push_back({time, scale});
        }
    }
    else
        Bad(key + ".type", "expected constant, ramp, sine or table");
    return out;
}
}  // namespace

double Schedule::Scale(double time_ps) const
{
    if (!SpongeFloat::Is_Finite(time_ps))
        Bad("external_force time", "must be finite");
    switch (kind)
    {
        case ScheduleKind::Constant:
            return 1.0;
        case ScheduleKind::Ramp:
            if (time_ps <= start) return 0.0;
            return std::min(1.0, (time_ps - start) / duration);
        case ScheduleKind::Sine:
            return std::sin(2.0 * pi * ((time_ps - start) / period) + phase);
        case ScheduleKind::Table:
            if (time_ps <= points.front()[0]) return points.front()[1];
            if (time_ps >= points.back()[0]) return points.back()[1];
            auto next = std::upper_bound(points.begin(), points.end(), time_ps,
                                         [](double time, const auto& point)
                                         { return time < point[0]; });
            const auto& prev = *(next - 1);
            const double fraction =
                (time_ps - prev[0]) / ((*next)[0] - prev[0]);
            return (1.0 - fraction) * prev[1] + fraction * (*next)[1];
    }
    throw std::logic_error("invalid schedule kind");
}

std::vector<Drive> Parse_Drives(const std::string& toml,
                                const std::string& source)
{
    const auto root =
        sponge::toml_decode::detail::parse_toml_string(toml, source);
    const auto entry = root.find("external_force");
    if (entry == root.end()) return {};
    auto array = entry->second.as_array();
    if (!array)
        Bad("external_force", "use [[external_force]] array-of-tables syntax");
    std::vector<Drive> result;
    for (std::size_t index = 0; index < array->size(); ++index)
    {
        const std::string key = "external_force[" + std::to_string(index) + "]";
        auto table = (*array)[index].as_table();
        if (!table) Bad(key, "expected a table");
        Drive drive;
        auto type = String(Required(*table, "type", key), key + ".type");
        const char* value_key = nullptr;
        std::set<std::string> allowed{"type", "atoms", "schedule", "profile"};
        if (type == "force")
        {
            drive.kind = DriveKind::Force;
            value_key = "force";
        }
        else if (type == "acceleration")
        {
            drive.kind = DriveKind::Acceleration;
            value_key = "acceleration";
        }
        else if (type == "electric_field")
        {
            drive.kind = DriveKind::ElectricField;
            value_key = "field";
        }
        else if (type == "group_load")
        {
            drive.kind = DriveKind::GroupLoad;
            allowed.insert("force");
            allowed.insert("torque");
            if (!table->count("atoms"))
                Bad(key + ".atoms",
                    "group_load requires an explicit selection");
            if (!table->count("force") && !table->count("torque"))
                Bad(key, "group_load requires force and/or torque");
            if (table->count("force"))
                drive.value = Vector(table->at("force"), key + ".force");
            if (table->count("torque"))
                drive.torque = Vector(table->at("torque"), key + ".torque");
        }
        else
            Bad(key + ".type",
                "expected force, acceleration, electric_field or group_load");
        if (value_key)
        {
            allowed.insert(value_key);
            drive.value =
                Vector(Required(*table, value_key, key), key + "." + value_key);
        }
        if (table->count("profile"))
        {
            const auto* p = table->at("profile").as_table();
            if (!p) Bad(key + ".profile", "expected a table");
            Keys(*p, {"type", "axis", "mode"}, key + ".profile");
            if (String(Required(*p, "type", key), key + ".profile.type") !=
                "cosine")
                Bad(key, "unsupported spatial profile");
            auto axis =
                String(Required(*p, "axis", key), key + ".profile.axis");
            if (axis != "x" && axis != "y" && axis != "z")
                Bad(key, "profile axis must be x/y/z");
            drive.profile.axis = axis == "x" ? 0 : axis == "y" ? 1 : 2;
            if (p->count("mode"))
            {
                const auto* n = p->at("mode").as_integer();
                if (!n || *n < 1 || *n > 1000)
                    Bad(key, "profile mode must be an integer in [1,1000]");
                drive.profile.mode = int(*n);
            }
            if (drive.kind == DriveKind::GroupLoad)
                Bad(key, "group_load cannot have a spatial profile");
            drive.profile.enabled = true;
        }
        Keys(*table, allowed, key);
        if (table->count("atoms"))
        {
            auto ids = table->at("atoms").as_array();
            if (!ids || ids->empty())
                Bad(key + ".atoms", "expected nonempty global atom indices");
            std::set<int> seen;
            for (const auto& id_node : *ids)
            {
                auto id = id_node.as_integer();
                if (!id || *id < 0 || *id > std::numeric_limits<int>::max())
                    Bad(key + ".atoms", "expected nonnegative integer indices");
                if (!seen.insert(static_cast<int>(*id)).second)
                    Bad(key + ".atoms", "duplicate atom index");
                drive.atoms.push_back(static_cast<int>(*id));
            }
        }
        if (table->count("schedule"))
            drive.schedule =
                Parse_Schedule(table->at("schedule"), key + ".schedule");
        result.push_back(std::move(drive));
    }
    return result;
}

void Validate_Selections(const std::vector<Drive>& drives, int atom_count)
{
    if (atom_count <= 0) Bad("external_force", "system must contain atoms");
    for (std::size_t index = 0; index < drives.size(); ++index)
        for (int atom : drives[index].atoms)
            if (atom >= atom_count)
                Bad("external_force[" + std::to_string(index) + "].atoms",
                    "index outside system");
}
}  // namespace SpongeDynamics

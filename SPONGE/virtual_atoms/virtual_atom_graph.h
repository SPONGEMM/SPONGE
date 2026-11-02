#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

#include "../xponge/ir/forcefield.h"

struct VirtualAtomGraph
{
    std::vector<int> atom_levels;
    std::vector<std::size_t> record_order;
    int max_level = 0;
};

inline bool Virtual_Atom_Parameter_Is_Finite(float value)
{
    std::uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "float must be 32 bit");
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x7f800000u) != 0x7f800000u;
}

inline bool Build_Virtual_Atom_Graph(
    const std::vector<Xponge::VirtualAtomRecord>& records, int atom_numbers,
    VirtualAtomGraph* graph, std::string* error)
{
    if (graph == nullptr || error == nullptr)
    {
        return false;
    }
    graph->atom_levels.assign(std::max(atom_numbers, 0), 0);
    graph->record_order.clear();
    graph->max_level = 0;
    error->clear();
    if (atom_numbers < 0)
    {
        *error = "atom count must not be negative";
        return false;
    }

    std::vector<int> target_record(atom_numbers, -1);
    for (std::size_t i = 0; i < records.size(); ++i)
    {
        const auto& record = records[i];
        std::size_t expected_sources = 0;
        std::size_t expected_parameters = 0;
        switch (record.type)
        {
            case 0:
                expected_sources = 1;
                expected_parameters = 1;
                break;
            case 1:
                expected_sources = 2;
                expected_parameters = 1;
                break;
            case 2:
            case 3:
                expected_sources = 3;
                expected_parameters = 2;
                break;
            default:
            {
                std::ostringstream out;
                out << "virtual atom record " << i << " has unsupported type "
                    << record.type;
                *error = out.str();
                return false;
            }
        }
        if (record.from.size() != expected_sources ||
            record.parameter.size() != expected_parameters)
        {
            std::ostringstream out;
            out << "virtual atom " << record.virtual_atom << " (type "
                << record.type << ") has invalid source/parameter arity";
            *error = out.str();
            return false;
        }
        if (record.virtual_atom < 0 || record.virtual_atom >= atom_numbers)
        {
            std::ostringstream out;
            out << "virtual atom target " << record.virtual_atom
                << " is outside [0, " << atom_numbers << ")";
            *error = out.str();
            return false;
        }
        if (target_record[record.virtual_atom] >= 0)
        {
            std::ostringstream out;
            out << "virtual atom target " << record.virtual_atom
                << " is defined more than once";
            *error = out.str();
            return false;
        }
        target_record[record.virtual_atom] = static_cast<int>(i);
        for (int source : record.from)
        {
            if (source < 0 || source >= atom_numbers)
            {
                std::ostringstream out;
                out << "virtual atom " << record.virtual_atom << " source "
                    << source << " is outside [0, " << atom_numbers << ")";
                *error = out.str();
                return false;
            }
            if (source == record.virtual_atom)
            {
                std::ostringstream out;
                out << "virtual atom " << record.virtual_atom
                    << " depends on itself";
                *error = out.str();
                return false;
            }
        }
        for (float parameter : record.parameter)
        {
            if (!Virtual_Atom_Parameter_Is_Finite(parameter))
            {
                std::ostringstream out;
                out << "virtual atom " << record.virtual_atom
                    << " has a non-finite parameter";
                *error = out.str();
                return false;
            }
        }
    }

    std::vector<unsigned char> visit(records.size(), 0);
    std::function<bool(std::size_t)> resolve_level = [&](std::size_t index)
    {
        if (visit[index] == 2) return true;
        if (visit[index] == 1)
        {
            std::ostringstream out;
            out << "virtual atom dependency cycle reaches target "
                << records[index].virtual_atom;
            *error = out.str();
            return false;
        }
        visit[index] = 1;
        int level = 1;
        for (int source : records[index].from)
        {
            const int source_record = target_record[source];
            if (source_record >= 0)
            {
                if (!resolve_level(static_cast<std::size_t>(source_record)))
                    return false;
                level = std::max(level, graph->atom_levels[source] + 1);
            }
        }
        graph->atom_levels[records[index].virtual_atom] = level;
        graph->max_level = std::max(graph->max_level, level);
        visit[index] = 2;
        return true;
    };

    for (std::size_t i = 0; i < records.size(); ++i)
    {
        if (!resolve_level(i)) return false;
    }
    graph->record_order.resize(records.size());
    for (std::size_t i = 0; i < records.size(); ++i) graph->record_order[i] = i;
    std::stable_sort(graph->record_order.begin(), graph->record_order.end(),
                     [&](std::size_t lhs, std::size_t rhs)
                     {
                         return graph->atom_levels[records[lhs].virtual_atom] <
                                graph->atom_levels[records[rhs].virtual_atom];
                     });
    return true;
}

#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "../float_classification.hpp"
#include "utils/h5md/h5_legacy_sidecar_contract.hpp"
#include "utils/h5md/h5_structural_state.hpp"

namespace SpongeH5MD
{
inline float Dot3(const float* lhs, const float* rhs)
{
    return lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2];
}

inline float Norm3(const float* value) { return std::sqrt(Dot3(value, value)); }

inline float Angle_Degrees(const float* lhs, const float* rhs)
{
    const float denom = Norm3(lhs) * Norm3(rhs);
    if (denom == 0.0f)
    {
        return 0.0f;
    }
    const float cosine =
        std::max(-1.0f, std::min(1.0f, Dot3(lhs, rhs) / denom));
    return std::acos(cosine) * 57.295779513082320876f;
}

template <typename Vector>
inline void Box_Edges_To_Lengths_And_Angles(const std::array<float, 9>& edges,
                                            Vector* box_length,
                                            Vector* box_angle)
{
    const float* a = &edges[0];
    const float* b = &edges[3];
    const float* c = &edges[6];
    box_length->x = Norm3(a);
    box_length->y = Norm3(b);
    box_length->z = Norm3(c);
    box_angle->x = Angle_Degrees(b, c);
    box_angle->y = Angle_Degrees(a, c);
    box_angle->z = Angle_Degrees(a, b);
}

template <typename Vector>
inline bool Apply_Restart_Structural_State(
    const RestartStructuralState& state, int expected_atom_count,
    Vector* coordinate, Vector* velocity, Vector* box_length, Vector* box_angle,
    double* start_time, std::string* error_message)
{
    auto fail = [error_message](const std::string& message)
    {
        if (error_message != nullptr)
        {
            *error_message = message;
        }
        return false;
    };

    if (expected_atom_count <= 0)
    {
        return fail(
            "expected atom count must be positive before applying H5 restart "
            "structural state");
    }
    if (state.atom_count != static_cast<std::size_t>(expected_atom_count))
    {
        std::ostringstream out;
        out << "H5 restart atom count mismatch: expected "
            << expected_atom_count << ", got " << state.atom_count;
        return fail(out.str());
    }
    if (state.position_xyz.size() != state.atom_count * 3)
    {
        return fail("H5 restart position array size does not match atom count");
    }
    if (state.has_velocity && state.velocity_xyz.size() != state.atom_count * 3)
    {
        return fail("H5 restart velocity array size does not match atom count");
    }
    if (coordinate == nullptr || velocity == nullptr || box_length == nullptr ||
        box_angle == nullptr || start_time == nullptr)
    {
        return fail("H5 restart structural output pointer is null");
    }

    for (std::size_t i = 0; i < state.atom_count; ++i)
    {
        coordinate[i].x = state.position_xyz[3 * i];
        coordinate[i].y = state.position_xyz[3 * i + 1];
        coordinate[i].z = state.position_xyz[3 * i + 2];
        if (state.has_velocity)
        {
            velocity[i].x = state.velocity_xyz[3 * i];
            velocity[i].y = state.velocity_xyz[3 * i + 1];
            velocity[i].z = state.velocity_xyz[3 * i + 2];
        }
        else
        {
            velocity[i].x = 0.0f;
            velocity[i].y = 0.0f;
            velocity[i].z = 0.0f;
        }
    }
    Box_Edges_To_Lengths_And_Angles(state.box_edges, box_length, box_angle);
    *start_time = state.time;
    return true;
}

inline bool Apply_Nose_Hoover_Chain_Dynamic_State(
    const RestartDynamicState& state, int expected_chain_length,
    float* chain_coordinate, float* chain_velocity, std::string* error_message)
{
    auto fail = [error_message](const std::string& message)
    {
        if (error_message != nullptr)
        {
            *error_message = message;
        }
        return false;
    };

    if (!state.has_nose_hoover_chain)
    {
        return fail("H5 restart does not contain Nose-Hoover chain state");
    }
    if (expected_chain_length <= 0)
    {
        return fail(
            "Nose-Hoover chain length must be positive before applying H5 "
            "restart state");
    }
    if (state.nose_hoover_chain_pair_count !=
        static_cast<std::size_t>(expected_chain_length))
    {
        std::ostringstream out;
        out << "H5 Nose-Hoover chain length mismatch: expected "
            << expected_chain_length << ", got "
            << state.nose_hoover_chain_pair_count;
        return fail(out.str());
    }
    if (state.nose_hoover_chain_coordinate_velocity_pairs.size() !=
        static_cast<std::size_t>(expected_chain_length) * 2)
    {
        return fail(
            "H5 Nose-Hoover chain state value count does not match chain "
            "length");
    }
    if (chain_coordinate == nullptr || chain_velocity == nullptr)
    {
        return fail("Nose-Hoover chain output pointer is null");
    }

    for (int i = 0; i < expected_chain_length; ++i)
    {
        chain_coordinate[i] =
            state.nose_hoover_chain_coordinate_velocity_pairs[2 * i];
        chain_velocity[i] =
            state.nose_hoover_chain_coordinate_velocity_pairs[2 * i + 1];
    }
    chain_velocity[expected_chain_length] = 0.0f;
    return true;
}

inline bool Extract_Sits_Nk_Protocol_State(const RestartProtocolState& state,
                                           const std::string& module_name,
                                           int expected_k_count,
                                           std::vector<float>* nk_values,
                                           std::string* error_message)
{
    auto fail = [error_message](const std::string& message)
    {
        if (error_message != nullptr)
        {
            *error_message = message;
        }
        return false;
    };

    if (module_name.empty())
    {
        return fail("SITS module name is empty");
    }
    if (expected_k_count <= 0)
    {
        return fail(
            "SITS k count must be positive before applying H5 restart state");
    }
    if (nk_values == nullptr)
    {
        return fail("SITS Nk output pointer is null");
    }
    const RestartSitsState* module_state = nullptr;
    for (const auto& candidate : state.sits_states)
    {
        if (candidate.module_name == module_name)
        {
            module_state = &candidate;
            break;
        }
    }
    if (module_state == nullptr)
    {
        return fail("H5 restart does not contain SITS state for module " +
                    module_name);
    }
    const auto values_iter = module_state->float_states.find("nk");
    if (values_iter == module_state->float_states.end())
    {
        return fail(
            "H5 restart SITS state does not contain nk values for module " +
            module_name);
    }
    if (values_iter->second.size() !=
        static_cast<std::size_t>(expected_k_count))
    {
        std::ostringstream out;
        out << "H5 SITS nk count mismatch for module " << module_name
            << ": expected " << expected_k_count << ", got "
            << values_iter->second.size();
        return fail(out.str());
    }
    for (const float value : values_iter->second)
    {
        if (!(value > 0.0f) || !SpongeFloat::Is_Finite(value))
        {
            return fail("H5 SITS nk values must be positive finite numbers");
        }
    }

    *nk_values = values_iter->second;
    return true;
}

inline bool Write_Text_File(const std::string& file_path,
                            const std::string& contents,
                            std::string* error_message)
{
    std::ofstream output(file_path.c_str(), std::ios::out | std::ios::binary);
    if (!output.is_open())
    {
        if (error_message != nullptr)
        {
            *error_message =
                "failed to open text file for H5 restart materialization: " +
                file_path;
        }
        return false;
    }
    output << contents;
    if (!output.good())
    {
        if (error_message != nullptr)
        {
            *error_message =
                "failed to write text file for H5 restart materialization: " +
                file_path;
        }
        return false;
    }
    return true;
}

inline bool Materialize_Metadynamics_Text_State(
    const RestartProtocolState& state, const std::string& module_name,
    const std::string& hills_file_name, const std::string& history_file_name,
    const std::string& edge_file_name, const std::string& potential_file_name,
    const std::string& direct_file_name, bool* materialized_any,
    std::string* error_message)
{
    auto fail = [error_message](const std::string& message)
    {
        if (error_message != nullptr)
        {
            *error_message = message;
        }
        return false;
    };

    if (materialized_any != nullptr)
    {
        *materialized_any = false;
    }
    if (module_name.empty())
    {
        return fail("metadynamics module name is empty");
    }
    const RestartMetadynamicsState* metad_state = nullptr;
    for (const auto& candidate : state.metadynamics_states)
    {
        if (candidate.name == module_name ||
            (module_name == "meta" && candidate.name == "default"))
        {
            metad_state = &candidate;
            break;
        }
    }
    if (metad_state == nullptr)
    {
        return fail(
            "H5 restart does not contain metadynamics state for module " +
            module_name);
    }

    struct Mapping
    {
        const char* component;
        const std::string* file_name;
    };
    const Mapping mappings[] = {
        {"hills", &hills_file_name},
        {"history", &history_file_name},
        {"edge", &edge_file_name},
        {"potential_export", &potential_file_name},
        {"direct_export", &direct_file_name},
    };

    bool wrote_any = false;
    for (const auto& mapping : mappings)
    {
        const auto value_iter =
            metad_state->text_states.find(mapping.component);
        if (value_iter == metad_state->text_states.end())
        {
            continue;
        }
        if (mapping.file_name == nullptr || mapping.file_name->empty())
        {
            return fail(
                std::string(
                    "metadynamics restart component has no target file: ") +
                mapping.component);
        }
        if (!Write_Text_File(*mapping.file_name, value_iter->second,
                             error_message))
        {
            return false;
        }
        wrote_any = true;
    }

    if (!wrote_any)
    {
        return fail(
            "H5 restart metadynamics state contains no supported text "
            "components for module " +
            module_name);
    }
    if (materialized_any != nullptr)
    {
        *materialized_any = true;
    }
    return true;
}

inline std::string Sanitize_Protocol_Sidecar_File_Name(const std::string& key)
{
    std::string sanitized;
    sanitized.reserve(key.size());
    for (unsigned char c : key)
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
    if (sanitized.empty())
    {
        sanitized = "protocol_sidecar";
    }
    return sanitized + ".txt";
}

inline bool Materialize_Protocol_Sidecar_Text_State(
    const RestartProtocolState& state, const std::string& output_dir,
    std::vector<LegacySidecarBinding>* sidecars, std::string* error_message)
{
    auto fail = [error_message](const std::string& message)
    {
        if (error_message != nullptr)
        {
            *error_message = message;
        }
        return false;
    };

    if (sidecars == nullptr)
    {
        return fail("protocol sidecar output pointer is null");
    }
    sidecars->clear();
    if (state.sidecar_text_states.empty())
    {
        return true;
    }
    if (output_dir.empty())
    {
        return fail("protocol sidecar materialization directory is empty");
    }

    try
    {
        std::filesystem::create_directories(output_dir);
    }
    catch (const std::exception& err)
    {
        return fail(
            std::string("failed to create protocol sidecar directory: ") +
            err.what());
    }

    std::set<std::string> seen_keys;
    for (const auto& state_entry : state.sidecar_text_states)
    {
        if (!Command_Key_Allowed(H5_Protocol_Sidecar_Command_Keys(),
                                 state_entry.key))
        {
            return fail("unsupported H5 protocol restart sidecar key: " +
                        state_entry.key);
        }
        if (!seen_keys.insert(state_entry.key).second)
        {
            return fail("duplicate H5 protocol restart sidecar key: " +
                        state_entry.key);
        }
        const std::filesystem::path file_path =
            std::filesystem::path(output_dir) /
            Sanitize_Protocol_Sidecar_File_Name(state_entry.key);
        if (!Write_Text_File(file_path.string(), state_entry.text,
                             error_message))
        {
            return false;
        }
        sidecars->push_back({state_entry.key, file_path.string()});
    }
    return true;
}
}  // namespace SpongeH5MD

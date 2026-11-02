#pragma once

#include <cctype>
#include <cstdint>
#include <string>

namespace SpongeH5InputMetadata
{
struct CompatibilityResult
{
    bool compatible = true;
    std::string error_message;
};

struct TopologyMetadata
{
    std::string schema_name;
    std::string schema_version;
    std::string identity_uuid;
    std::int64_t atom_count = 0;
    std::string atom_ordering_hash;
    std::string topology_hash;
    std::string force_field_hash;
};

struct ProtocolMetadata
{
    std::string schema_name;
    std::string schema_version;
    std::string identity_uuid;
    std::string topology_hash;
    std::string protocol_hash;
    bool has_protocol_owned_state = false;
};

struct RestartMetadata
{
    std::string schema_name;
    std::string schema_version;
    std::string identity_uuid;
    std::int64_t atom_count = 0;
    std::string atom_ordering_hash;
    std::string producer_topology_hash;
    std::string producer_protocol_hash;
    std::string state_hash;
    std::string computed_state_hash;
    bool has_structural_state = false;
    bool has_velocity = false;
    bool has_dynamic_state = false;
    bool has_protocol_state = false;
};

struct TrajectoryMetadata
{
    std::string schema_name;
    std::string schema_version;
    std::string identity_uuid;
    std::string particle_stream = "all";
    std::int64_t atom_count = 0;
    std::int64_t frame_count = 0;
    std::string atom_ordering_hash;
    bool has_position = false;
    bool has_box = false;
    bool has_velocity = false;
    bool has_force = false;
    bool has_vds_manifest = false;
};

inline CompatibilityResult Compatible() { return {}; }

inline CompatibilityResult Incompatible(const std::string& message)
{
    CompatibilityResult result;
    result.compatible = false;
    result.error_message = message;
    return result;
}

inline bool Both_Set(const std::string& lhs, const std::string& rhs)
{
    return !lhs.empty() && !rhs.empty();
}

inline bool Is_Canonical_Uuid(const std::string& value)
{
    if (value.size() != 36) return false;
    for (std::size_t index = 0; index < value.size(); ++index)
    {
        const bool separator =
            index == 8 || index == 13 || index == 18 || index == 23;
        if (separator)
        {
            if (value[index] != '-') return false;
        }
        else if (!std::isxdigit(static_cast<unsigned char>(value[index])))
        {
            return false;
        }
    }
    return true;
}

inline CompatibilityResult Check_Artifact_Identity(
    const std::string& label, const std::string& schema_name,
    const std::string& expected_schema_name, const std::string& schema_version,
    const std::string& expected_schema_version,
    const std::string& identity_uuid)
{
    if (schema_name != expected_schema_name)
    {
        return Incompatible(label + " schema name must be " +
                            expected_schema_name + ", got " + schema_name);
    }
    if (schema_version != expected_schema_version)
    {
        return Incompatible(label + " schema version must be " +
                            expected_schema_version + ", got " +
                            schema_version);
    }
    if (!Is_Canonical_Uuid(identity_uuid))
    {
        return Incompatible(label +
                            " identity UUID must use canonical UUID syntax");
    }
    return Compatible();
}

inline CompatibilityResult Check_Topology_Metadata(
    const TopologyMetadata& topology)
{
    const auto artifact_check = Check_Artifact_Identity(
        "topology", topology.schema_name, "sponge.topology.h5",
        topology.schema_version, "sponge.input.v2", topology.identity_uuid);
    if (!artifact_check.compatible) return artifact_check;
    if (topology.atom_count <= 0)
    {
        return Incompatible("topology atom_count must be positive");
    }
    if (topology.topology_hash.empty())
    {
        return Incompatible("topology_hash is required");
    }
    if (topology.atom_ordering_hash.empty())
    {
        return Incompatible("topology atom_order_hash is required");
    }
    if (topology.force_field_hash.empty())
    {
        return Incompatible("topology forcefield_hash is required");
    }
    return Compatible();
}

inline CompatibilityResult Check_Restart_Against_Topology(
    const RestartMetadata& restart, const TopologyMetadata& topology)
{
    const auto topology_check = Check_Topology_Metadata(topology);
    if (!topology_check.compatible)
    {
        return topology_check;
    }
    if (!restart.has_structural_state)
    {
        return Incompatible("restart structural state is required");
    }
    const auto artifact_check = Check_Artifact_Identity(
        "restart", restart.schema_name, "sponge.restart.h5",
        restart.schema_version, "sponge.input.v2", restart.identity_uuid);
    if (!artifact_check.compatible) return artifact_check;
    if (restart.atom_count != topology.atom_count)
    {
        return Incompatible("restart atom_count does not match topology");
    }
    if (restart.atom_ordering_hash.empty())
    {
        return Incompatible("restart atom_order_hash is required");
    }
    if (restart.atom_ordering_hash != topology.atom_ordering_hash)
    {
        return Incompatible(
            "restart atom_ordering_hash does not match topology");
    }
    if (restart.producer_topology_hash.empty())
    {
        return Incompatible("restart topology_hash is required");
    }
    if (restart.producer_topology_hash != topology.topology_hash)
    {
        return Incompatible("restart topology hash does not match topology");
    }
    if (restart.state_hash.empty())
    {
        return Incompatible("restart state_hash is required");
    }
    if (restart.computed_state_hash.empty())
    {
        return Incompatible("restart state_hash could not be computed");
    }
    if (restart.state_hash != restart.computed_state_hash)
    {
        return Incompatible("restart state_hash does not match state data");
    }
    return Compatible();
}

inline CompatibilityResult Check_Trajectory_Against_Topology(
    const TrajectoryMetadata& trajectory, const TopologyMetadata& topology)
{
    const auto topology_check = Check_Topology_Metadata(topology);
    if (!topology_check.compatible)
    {
        return topology_check;
    }
    const auto artifact_check =
        Check_Artifact_Identity("trajectory", trajectory.schema_name,
                                "sponge.output.h5md", trajectory.schema_version,
                                "sponge.output.v2", trajectory.identity_uuid);
    if (!artifact_check.compatible) return artifact_check;
    if (!trajectory.has_position)
    {
        return Incompatible("trajectory position dataset is required");
    }
    if (!trajectory.has_box)
    {
        return Incompatible("trajectory box dataset is required");
    }
    if (trajectory.atom_count != topology.atom_count)
    {
        return Incompatible("trajectory atom_count does not match topology");
    }
    if (trajectory.frame_count <= 0)
    {
        return Incompatible("trajectory frame_count must be positive");
    }
    if (Both_Set(trajectory.atom_ordering_hash, topology.atom_ordering_hash) &&
        trajectory.atom_ordering_hash != topology.atom_ordering_hash)
    {
        return Incompatible(
            "trajectory atom_ordering_hash does not match topology");
    }
    return Compatible();
}

inline CompatibilityResult Check_Protocol_Against_Topology(
    const ProtocolMetadata& protocol, const TopologyMetadata& topology)
{
    const auto topology_check = Check_Topology_Metadata(topology);
    if (!topology_check.compatible)
    {
        return topology_check;
    }
    const auto artifact_check = Check_Artifact_Identity(
        "protocol", protocol.schema_name, "sponge.protocol.h5",
        protocol.schema_version, "sponge.input.v2", protocol.identity_uuid);
    if (!artifact_check.compatible) return artifact_check;
    if (protocol.topology_hash.empty())
    {
        return Incompatible("protocol topology_hash is required");
    }
    if (protocol.topology_hash != topology.topology_hash)
    {
        return Incompatible("protocol topology hash does not match topology");
    }
    if (protocol.protocol_hash.empty())
    {
        return Incompatible("protocol content_hash is required");
    }
    return Compatible();
}

inline CompatibilityResult Check_Protocol_State_Against_Protocol(
    const RestartMetadata& restart, const ProtocolMetadata& protocol)
{
    if (!restart.has_protocol_state)
    {
        return Compatible();
    }
    if (restart.producer_protocol_hash.empty())
    {
        return Incompatible(
            "restart protocol state requires producer_protocol_hash");
    }
    if (restart.producer_protocol_hash != protocol.protocol_hash)
    {
        return Incompatible("restart protocol state does not match protocol");
    }
    return Compatible();
}

}  // namespace SpongeH5InputMetadata

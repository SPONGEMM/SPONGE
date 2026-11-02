#include <cstdlib>
#include <iostream>

#include "utils/h5md/h5_input_metadata.hpp"

namespace
{
void Expect(bool condition, const char* message)
{
    if (!condition)
    {
        std::cerr << message << '\n';
        std::exit(1);
    }
}
void Expect_Incompatible(
    const SpongeH5InputMetadata::CompatibilityResult& result,
    const char* message)
{
    Expect(!result.compatible, "Expected metadata compatibility failure");
    if (result.error_message.find(message) == std::string::npos)
    {
        std::cerr << "Expected error containing '" << message << "', got '"
                  << result.error_message << "'\n";
        std::exit(1);
    }
}

SpongeH5InputMetadata::TopologyMetadata Topology()
{
    SpongeH5InputMetadata::TopologyMetadata topology;
    topology.schema_name = "sponge.topology.h5";
    topology.schema_version = "sponge.input.v2";
    topology.identity_uuid = "123e4567-e89b-12d3-a456-426614174000";
    topology.atom_count = 42;
    topology.atom_ordering_hash = "atoms";
    topology.topology_hash = "top";
    topology.force_field_hash = "ff";
    return topology;
}
}  // namespace

int main()
{
    {
        auto topology = Topology();
        auto result = SpongeH5InputMetadata::Check_Topology_Metadata(topology);
        Expect(result.compatible, "Valid topology metadata should pass");
    }

    {
        auto topology = Topology();
        topology.atom_count = 0;
        Expect_Incompatible(
            SpongeH5InputMetadata::Check_Topology_Metadata(topology),
            "atom_count");
    }

    {
        auto topology = Topology();
        SpongeH5InputMetadata::RestartMetadata restart;
        restart.schema_name = "sponge.restart.h5";
        restart.schema_version = "sponge.input.v2";
        restart.identity_uuid = "123e4567-e89b-12d3-a456-426614174001";
        restart.atom_count = topology.atom_count;
        restart.atom_ordering_hash = topology.atom_ordering_hash;
        restart.producer_topology_hash = topology.topology_hash;
        restart.state_hash = "sha256:state";
        restart.computed_state_hash = restart.state_hash;
        restart.has_structural_state = true;
        auto result = SpongeH5InputMetadata::Check_Restart_Against_Topology(
            restart, topology);
        Expect(result.compatible, "Compatible restart metadata should pass");
    }

    {
        auto topology = Topology();
        SpongeH5InputMetadata::RestartMetadata restart;
        restart.schema_name = "sponge.restart.h5";
        restart.schema_version = "sponge.input.v2";
        restart.identity_uuid = "123e4567-e89b-12d3-a456-426614174001";
        restart.atom_count = topology.atom_count + 1;
        restart.has_structural_state = true;
        Expect_Incompatible(
            SpongeH5InputMetadata::Check_Restart_Against_Topology(restart,
                                                                  topology),
            "atom_count");
    }

    {
        auto topology = Topology();
        SpongeH5InputMetadata::TrajectoryMetadata trajectory;
        trajectory.schema_name = "sponge.output.h5md";
        trajectory.schema_version = "sponge.output.v2";
        trajectory.identity_uuid = "123e4567-e89b-12d3-a456-426614174002";
        trajectory.atom_count = topology.atom_count;
        trajectory.atom_ordering_hash = topology.atom_ordering_hash;
        trajectory.frame_count = 10;
        trajectory.has_position = true;
        trajectory.has_box = true;
        auto result = SpongeH5InputMetadata::Check_Trajectory_Against_Topology(
            trajectory, topology);
        Expect(result.compatible, "Compatible trajectory metadata should pass");
    }

    {
        auto topology = Topology();
        SpongeH5InputMetadata::TrajectoryMetadata trajectory;
        trajectory.schema_name = "sponge.output.h5md";
        trajectory.schema_version = "sponge.output.v2";
        trajectory.identity_uuid = "123e4567-e89b-12d3-a456-426614174002";
        trajectory.atom_count = topology.atom_count;
        trajectory.frame_count = 10;
        trajectory.has_position = true;
        Expect_Incompatible(
            SpongeH5InputMetadata::Check_Trajectory_Against_Topology(trajectory,
                                                                     topology),
            "box");
    }

    {
        auto topology = Topology();
        SpongeH5InputMetadata::ProtocolMetadata protocol;
        protocol.schema_name = "sponge.protocol.h5";
        protocol.schema_version = "sponge.input.v2";
        protocol.identity_uuid = "123e4567-e89b-12d3-a456-426614174003";
        protocol.topology_hash = topology.topology_hash;
        protocol.protocol_hash = "protocol";
        auto result = SpongeH5InputMetadata::Check_Protocol_Against_Topology(
            protocol, topology);
        Expect(result.compatible, "Compatible protocol metadata should pass");
    }

    {
        SpongeH5InputMetadata::RestartMetadata restart;
        restart.has_protocol_state = true;
        restart.producer_protocol_hash = "old";
        SpongeH5InputMetadata::ProtocolMetadata protocol;
        protocol.protocol_hash = "new";
        Expect_Incompatible(
            SpongeH5InputMetadata::Check_Protocol_State_Against_Protocol(
                restart, protocol),
            "protocol state");
    }

    return 0;
}

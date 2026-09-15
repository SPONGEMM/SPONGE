#pragma once

#include <optional>

#include "../utils/h5md/h5_structural_state.hpp"

namespace SpongeIO
{
// Owns module payloads. Structural coordinates are still captured synchronously
// by trajectory_output at export time; this is not an asynchronous work item.
struct RestartOutputState
{
    std::vector<float> nhc_coordinates;
    std::vector<float> nhc_velocities;
    std::optional<SpongeH5MD::RestartSitsState> sits;
    std::optional<SpongeH5MD::RestartMetadynamicsState> metadynamics;
    std::optional<SpongeH5MD::RestartDynamicState> dynamic;
    std::string metadynamics_name;
    std::string metad_hills_file;
    std::string metad_history_file;
    std::string metad_edge_file;
    std::string metad_potential_file;
    std::string metad_direct_file;
    std::string restraint_name;
    std::vector<float> restraint_reference;
    std::map<std::string, std::vector<float>> cv_references;
};
}  // namespace SpongeIO

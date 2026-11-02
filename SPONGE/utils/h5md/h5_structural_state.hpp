#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace SpongeH5MD
{
struct RestartRngState
{
    std::string engine;
    std::int64_t state_schema_version = 1;
    std::size_t stream_count = 0;
    std::size_t words_per_stream = 0;
    std::vector<std::int64_t> state_words;
};

struct RestartStructuralState
{
    std::int64_t step = 0;
    double time = 0.0;
    std::size_t atom_count = 0;
    std::vector<float> position_xyz;
    std::vector<float> velocity_xyz;
    std::array<float, 9> box_edges{};
    bool has_velocity = false;
};

struct RestartDynamicState
{
    bool has_nose_hoover_chain = false;
    std::size_t nose_hoover_chain_pair_count = 0;
    std::vector<float> nose_hoover_chain_coordinate_velocity_pairs;
    std::map<std::string, std::string> rng_state_text;
    std::map<std::string, RestartRngState> rng_states;
    std::map<std::string, std::string> integrator_state_text;
    std::map<std::string, std::map<std::string, std::string>>
        thermostat_text_states;
    std::map<std::string, std::map<std::string, std::vector<float>>>
        thermostat_float_states;
    std::map<std::string, std::map<std::string, std::vector<std::int64_t>>>
        thermostat_integer_states;
    std::map<std::string, std::map<std::string, std::string>>
        barostat_text_states;
    std::map<std::string, std::map<std::string, std::vector<float>>>
        barostat_float_states;
    std::map<std::string, std::map<std::string, std::vector<std::int64_t>>>
        barostat_integer_states;
};

struct RestartSitsState
{
    std::string module_name;
    std::int64_t state_schema_version = 1;
    std::map<std::string, std::vector<float>> float_states;
};

struct RestartMetadynamicsState
{
    std::string name;
    std::map<std::string, std::string> text_states;
    bool has_typed_state = false;
    std::int64_t state_schema_version = 1;
    std::size_t ndim = 0;

    std::vector<std::int64_t> grid_count;
    std::vector<float> grid_min;
    std::vector<float> grid_max;
    std::vector<float> potential_value;
    std::vector<float> potential_force;
    std::vector<float> edge_log_normalization;
    std::vector<float> edge_normal_force;

    std::size_t scatter_count = 0;
    std::vector<float> scatter_position;
    std::vector<float> scatter_potential;
    std::vector<float> scatter_force;

    std::size_t hill_count = 0;
    std::vector<float> hill_center;
    std::vector<float> hill_height;
    std::vector<float> hill_inverse_width;
    std::vector<float> hill_period;
    std::vector<float> hill_sink;

    bool has_runtime_state = false;
    float potential_max = 0.0f;
    float sum_max = 0.0f;
    float new_max = 0.0f;
    std::int64_t max_index = 0;
    float exit_tag = 0.0f;
    float rct = 0.0f;
    float rbias = 0.0f;
    float bias = 0.0f;
    float minus_beta_f = 1.0f;
    float minus_beta_f_plus_v = 0.0f;
};

struct RestartRestraintState
{
    std::string name;
    std::size_t atom_count = 0;
    std::vector<float> reference_coordinates;
};

struct RestartCVReferenceState
{
    std::string name;
    std::size_t atom_count = 0;
    std::vector<float> reference_coordinates;
};

struct RestartProtocolSidecarTextState
{
    std::string key;
    std::string text;
};

struct RestartProtocolState
{
    std::vector<RestartSitsState> sits_states;
    std::vector<RestartMetadynamicsState> metadynamics_states;
    std::vector<RestartRestraintState> restraint_states;
    std::vector<RestartCVReferenceState> cv_reference_states;
    std::vector<RestartProtocolSidecarTextState> sidecar_text_states;
};
}  // namespace SpongeH5MD

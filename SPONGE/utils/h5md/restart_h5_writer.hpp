#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "utils/h5md/canonical_dataset_hash.hpp"
#include "utils/h5md/h5_structural_state.hpp"
#include "utils/h5md/h5md_writer.hpp"
#include "utils/h5md/output_plan.hpp"

namespace SpongeH5MD
{
class RestartH5Writer
{
   public:
    explicit RestartH5Writer(WriterBackend* backend) : writer_(backend) {}

    bool Open(const SpongeH5OutputPlan::ResolvedOutputPlan& plan,
              const std::string& schema_version = kInputSchemaVersion,
              const std::string& identity_uuid = Generate_Uuid_V4())
    {
        if (!plan.restart.enabled)
        {
            last_error_ = "RestartH5Writer requires enabled restart plan";
            return false;
        }
        WriterOptions options;
        options.path = plan.restart.path;
        options.schema_name = "sponge.restart.h5";
        options.schema_version = schema_version;
        options.identity_uuid = identity_uuid;
        options.output_mode = "checkpoint";
        options.observable_only = false;
        if (!writer_.Open(options))
        {
            last_error_ = writer_.Last_Error();
            return false;
        }
        return writer_.Ensure_Group("/schema") &&
               writer_.Write_String("/schema/name", "sponge.restart.h5") &&
               writer_.Write_String("/schema/version", schema_version) &&
               Ensure_Base_Layout();
    }

    bool Ensure_Base_Layout()
    {
        if (!writer_.Ensure_Group(path::run)) return false;
        if (!writer_.Ensure_Group(path::particles_all)) return false;
        if (!writer_.Ensure_Group(path::particles_all_position)) return false;
        if (!writer_.Ensure_Group(path::particles_all_velocity)) return false;
        if (!writer_.Ensure_Group(path::particles_all_box)) return false;
        if (!writer_.Ensure_Group(path::particles_all_box_edges)) return false;
        if (!writer_.Ensure_Group(path::parameters_restart)) return false;
        if (!writer_.Ensure_Group(path::restart_rng_state)) return false;
        if (!writer_.Ensure_Group(path::restart_integrator_state)) return false;
        if (!writer_.Ensure_Group(path::restart_thermostat)) return false;
        if (!writer_.Ensure_Group(path::restart_barostat)) return false;
        if (!writer_.Ensure_Group(path::restart_references)) return false;
        if (!writer_.Ensure_Group(path::restart_restraint_references))
            return false;
        if (!writer_.Ensure_Group(path::restart_cv_references)) return false;
        if (!writer_.Ensure_Group(path::restart_protocol_sidecars))
            return false;
        if (!writer_.Ensure_Group(path::restart_bias)) return false;
        if (!writer_.Ensure_Group(path::restart_sits)) return false;
        if (!writer_.Ensure_Group(path::restart_meta)) return false;
        return true;
    }

    bool Define_Structural_State(const std::size_t atom_count,
                                 bool include_velocity)
    {
        atom_count_ = atom_count;
        if (!writer_.Create_Dataset({path::particles_all_step,
                                     DataType::int64,
                                     {{0}, {1}, {1}},
                                     true}))
        {
            return false;
        }
        if (!writer_.Create_Dataset({path::particles_all_time,
                                     DataType::float64,
                                     {{0}, {1}, {1}},
                                     true}))
        {
            return false;
        }
        if (!writer_.Set_String_Attribute(path::particles_all_time, "unit",
                                          "ps"))
        {
            return false;
        }
        if (!writer_.Create_Dataset(
                {path::position_value,
                 DataType::float32,
                 {{0, atom_count, 3}, {1, atom_count, 3}, {1, atom_count, 3}},
                 true}))
        {
            return false;
        }
        if (!writer_.Set_String_Attribute(path::position_value, "unit",
                                          "Angstrom"))
        {
            return false;
        }
        if (!writer_.Create_Hard_Link(path::particles_all_step,
                                      path::position_step))
        {
            return false;
        }
        if (!writer_.Create_Hard_Link(path::particles_all_time,
                                      path::position_time))
        {
            return false;
        }
        if (!writer_.Create_Dataset({path::box_edges_value,
                                     DataType::float32,
                                     {{0, 3, 3}, {1, 3, 3}, {1, 3, 3}},
                                     true}))
        {
            return false;
        }
        if (!writer_.Set_String_Attribute(path::box_edges_value, "unit",
                                          "Angstrom"))
        {
            return false;
        }
        if (!writer_.Create_Hard_Link(path::particles_all_step,
                                      path::box_edges_step))
        {
            return false;
        }
        if (!writer_.Create_Hard_Link(path::particles_all_time,
                                      path::box_edges_time))
        {
            return false;
        }
        if (include_velocity)
        {
            if (!writer_.Create_Dataset({path::velocity_value,
                                         DataType::float32,
                                         {{0, atom_count, 3},
                                          {1, atom_count, 3},
                                          {1, atom_count, 3}},
                                         true}))
            {
                return false;
            }
            if (!writer_.Set_String_Attribute(path::velocity_value, "unit",
                                              "Angstrom ps-1"))
            {
                return false;
            }
            if (!writer_.Create_Hard_Link(path::particles_all_step,
                                          path::velocity_step))
            {
                return false;
            }
            if (!writer_.Create_Hard_Link(path::particles_all_time,
                                          path::velocity_time))
            {
                return false;
            }
        }
        include_velocity_ = include_velocity;
        return true;
    }

    bool Write_Structural_State(const int64_t step, const double time,
                                const float* position_xyz,
                                const float* box_edges_3x3,
                                const float* velocity_xyz = nullptr)
    {
        if (state_written_)
        {
            last_error_ = "restart H5 already contains one structural state";
            return Mark_Failed();
        }
        if (!writer_.Append_Int64(path::particles_all_step, &step, 1))
        {
            return Mark_Failed();
        }
        if (!writer_.Append_Float64(path::particles_all_time, &time, 1))
        {
            return Mark_Failed();
        }
        if (!writer_.Append_Float32(path::position_value, position_xyz,
                                    atom_count_ * 3))
        {
            return Mark_Failed();
        }
        if (!writer_.Append_Float32(path::box_edges_value, box_edges_3x3, 9))
        {
            return Mark_Failed();
        }
        if (include_velocity_ && velocity_xyz != nullptr)
        {
            if (!writer_.Append_Float32(path::velocity_value, velocity_xyz,
                                        atom_count_ * 3))
            {
                return Mark_Failed();
            }
        }
        state_written_ = true;
        state_hash_.Add_Numeric(path::particles_all_step, {1}, &step, 1);
        state_hash_.Add_Numeric(path::particles_all_time, {1}, &time, 1);
        state_hash_.Add_Numeric(path::position_value, {1, atom_count_, 3},
                                position_xyz, atom_count_ * 3);
        state_hash_.Add_Numeric(path::box_edges_value, {1, 3, 3}, box_edges_3x3,
                                9);
        if (include_velocity_ && velocity_xyz != nullptr)
        {
            state_hash_.Add_Numeric(path::velocity_value, {1, atom_count_, 3},
                                    velocity_xyz, atom_count_ * 3);
        }
        if (!Write_Run_Metadata(step, time) ||
            !writer_.Write_Output_Completion(1, step, time))
        {
            return Mark_Failed();
        }
        return true;
    }

    bool Write_Run_Metadata(const int64_t step, const double time,
                            const std::string& state_type = "restart")
    {
        if (!writer_.Create_Dataset({path::run_current_step,
                                     DataType::int64,
                                     {{0}, {1}, {1}},
                                     true}))
        {
            return false;
        }
        if (!writer_.Create_Dataset({path::run_current_time,
                                     DataType::float64,
                                     {{0}, {1}, {1}},
                                     true}))
        {
            return false;
        }
        if (!writer_.Append_Int64(path::run_current_step, &step, 1))
        {
            return false;
        }
        if (!writer_.Append_Float64(path::run_current_time, &time, 1))
        {
            return false;
        }
        return writer_.Write_String(path::run_state_type, state_type);
    }

    bool Write_Lineage(const std::string& topology_hash,
                       const std::string& atom_order_hash,
                       const std::string& producer_protocol_hash = "")
    {
        if (topology_hash.empty() || atom_order_hash.empty())
        {
            last_error_ =
                "restart lineage requires topology_hash and atom_order_hash";
            return Mark_Failed();
        }
        if (!writer_.Write_String(path::run_topology_hash, topology_hash) ||
            !writer_.Write_String(path::run_atom_order_hash, atom_order_hash))
        {
            return Mark_Failed();
        }
        if (!producer_protocol_hash.empty() &&
            !writer_.Write_String(path::run_producer_protocol_hash,
                                  producer_protocol_hash))
        {
            return Mark_Failed();
        }
        return true;
    }

    bool Write_Restart_Generation(const int64_t generation)
    {
        if (generation <= 0)
        {
            last_error_ = "restart generation must be positive";
            return Mark_Failed();
        }
        if (generation_written_)
        {
            last_error_ = "restart H5 already contains a generation";
            return Mark_Failed();
        }
        if (!writer_.Create_Dataset({path::output_restart_generation,
                                     DataType::int64,
                                     {{0}, {1}, {1}},
                                     true}) ||
            !writer_.Append_Int64(path::output_restart_generation, &generation,
                                  1))
        {
            return Mark_Failed();
        }
        generation_written_ = true;
        return true;
    }

    bool Write_Nose_Hoover_Chain_State(const float* coordinate_velocity_pairs,
                                       std::size_t pair_count)
    {
        if (!writer_.Create_Dataset({path::restart_nhc,
                                     DataType::float32,
                                     {{0, 2}, {pair_count, 2}, {pair_count, 2}},
                                     true}))
        {
            return false;
        }
        if (!writer_.Append_Float32(path::restart_nhc,
                                    coordinate_velocity_pairs, pair_count * 2))
        {
            return false;
        }
        state_hash_.Add_Numeric(path::restart_nhc, {pair_count, 2},
                                coordinate_velocity_pairs, pair_count * 2);
        return true;
    }

    bool Write_Rng_State_Text(const std::string& module_name,
                              const std::string& value)
    {
        if (!Validate_State_Component_Name(module_name, "rng state module"))
        {
            return Mark_Failed();
        }
        if (!writer_.Ensure_Group(path::restart_rng_state)) return false;
        const std::string dataset_path = Restart_Rng_State_Path(module_name);
        if (!writer_.Write_String(dataset_path, value)) return false;
        state_hash_.Add_String(dataset_path, value);
        return true;
    }

    bool Write_Rng_State(const std::string& module_name,
                         const RestartRngState& state)
    {
        if (!Validate_State_Component_Name(module_name, "rng state module"))
        {
            return Mark_Failed();
        }
        if (state.engine.empty() || state.state_schema_version <= 0 ||
            state.stream_count == 0 || state.words_per_stream == 0 ||
            state.state_words.size() !=
                state.stream_count * state.words_per_stream)
        {
            last_error_ = "invalid typed RNG state for module " + module_name;
            return Mark_Failed();
        }
        const std::string root = Restart_Rng_State_Path(module_name);
        const std::string schema_path =
            Restart_Rng_State_Component_Path(module_name, "schema_version");
        const std::string engine_path =
            Restart_Rng_State_Component_Path(module_name, "engine");
        const std::string words_path =
            Restart_Rng_State_Component_Path(module_name, "state_words");
        if (!writer_.Ensure_Group(root) ||
            !writer_.Write_String(engine_path, state.engine) ||
            !writer_.Create_Dataset(
                {schema_path, DataType::int64, {{0}, {1}, {1}}, true}) ||
            !writer_.Append_Int64(schema_path, &state.state_schema_version,
                                  1) ||
            !writer_.Create_Dataset(
                {words_path,
                 DataType::int64,
                 {{0, state.words_per_stream},
                  {state.stream_count, state.words_per_stream},
                  {state.stream_count, state.words_per_stream}},
                 true}) ||
            !writer_.Append_Int64(words_path, state.state_words.data(),
                                  state.state_words.size()))
        {
            return Mark_Failed();
        }
        state_hash_.Add_String(engine_path, state.engine);
        state_hash_.Add_Numeric(schema_path, {1}, &state.state_schema_version,
                                1);
        state_hash_.Add_Numeric(
            words_path, {state.stream_count, state.words_per_stream},
            state.state_words.data(), state.state_words.size());
        return true;
    }

    bool Write_Integrator_State_Text(const std::string& key,
                                     const std::string& value)
    {
        if (!Validate_State_Component_Name(key, "integrator state key"))
        {
            return Mark_Failed();
        }
        if (!writer_.Ensure_Group(path::restart_integrator_state)) return false;
        const std::string dataset_path = Restart_Integrator_State_Path(key);
        if (!writer_.Write_String(dataset_path, value)) return false;
        state_hash_.Add_String(dataset_path, value);
        return true;
    }

    bool Write_Thermostat_State_Text(const std::string& module_name,
                                     const std::string& state_name,
                                     const std::string& value)
    {
        if (!Validate_State_Component_Name(module_name, "thermostat module") ||
            !Validate_State_Component_Name(state_name, "thermostat state"))
        {
            return Mark_Failed();
        }
        if (!writer_.Ensure_Group(Restart_Thermostat_State_Root(module_name)))
        {
            return false;
        }
        const std::string dataset_path =
            Restart_Thermostat_State_Path(module_name, state_name);
        if (!writer_.Write_String(dataset_path, value)) return false;
        state_hash_.Add_String(dataset_path, value);
        return true;
    }

    bool Write_Thermostat_State_Float(const std::string& module_name,
                                      const std::string& state_name,
                                      const float* values, std::size_t count)
    {
        if (!Validate_State_Component_Name(module_name, "thermostat module") ||
            !Validate_State_Component_Name(state_name, "thermostat state"))
        {
            return Mark_Failed();
        }
        if (!writer_.Ensure_Group(Restart_Thermostat_State_Root(module_name)))
        {
            return false;
        }
        const std::string dataset_path =
            Restart_Thermostat_State_Path(module_name, state_name);
        if (!writer_.Create_Dataset({dataset_path,
                                     DataType::float32,
                                     {{0}, {count}, {count}},
                                     true}))
        {
            return false;
        }
        if (!writer_.Append_Float32(dataset_path, values, count)) return false;
        state_hash_.Add_Numeric(dataset_path, {count}, values, count);
        return true;
    }

    bool Write_Thermostat_State_Int64(const std::string& module_name,
                                      const std::string& state_name,
                                      const std::int64_t* values,
                                      std::size_t count)
    {
        if (!Validate_State_Component_Name(module_name, "thermostat module") ||
            !Validate_State_Component_Name(state_name, "thermostat state") ||
            values == nullptr || count == 0)
        {
            return Mark_Failed();
        }
        if (!writer_.Ensure_Group(Restart_Thermostat_State_Root(module_name)))
        {
            return false;
        }
        const std::string dataset_path =
            Restart_Thermostat_State_Path(module_name, state_name);
        if (!writer_.Create_Dataset({dataset_path,
                                     DataType::int64,
                                     {{0}, {count}, {count}},
                                     true}) ||
            !writer_.Append_Int64(dataset_path, values, count))
        {
            return false;
        }
        state_hash_.Add_Numeric(dataset_path, {count}, values, count);
        return true;
    }

    bool Write_Barostat_State_Text(const std::string& module_name,
                                   const std::string& state_name,
                                   const std::string& value)
    {
        if (!Validate_State_Component_Name(module_name, "barostat module") ||
            !Validate_State_Component_Name(state_name, "barostat state"))
        {
            return Mark_Failed();
        }
        if (!writer_.Ensure_Group(Restart_Barostat_State_Root(module_name)))
        {
            return false;
        }
        const std::string dataset_path =
            Restart_Barostat_State_Path(module_name, state_name);
        if (!writer_.Write_String(dataset_path, value)) return false;
        state_hash_.Add_String(dataset_path, value);
        return true;
    }

    bool Write_Barostat_State_Float(const std::string& module_name,
                                    const std::string& state_name,
                                    const float* values, std::size_t count)
    {
        if (!Validate_State_Component_Name(module_name, "barostat module") ||
            !Validate_State_Component_Name(state_name, "barostat state"))
        {
            return Mark_Failed();
        }
        if (!writer_.Ensure_Group(Restart_Barostat_State_Root(module_name)))
        {
            return false;
        }
        const std::string dataset_path =
            Restart_Barostat_State_Path(module_name, state_name);
        if (!writer_.Create_Dataset({dataset_path,
                                     DataType::float32,
                                     {{0}, {count}, {count}},
                                     true}))
        {
            return false;
        }
        if (!writer_.Append_Float32(dataset_path, values, count)) return false;
        state_hash_.Add_Numeric(dataset_path, {count}, values, count);
        return true;
    }

    bool Write_Barostat_State_Int64(const std::string& module_name,
                                    const std::string& state_name,
                                    const std::int64_t* values,
                                    std::size_t count)
    {
        if (!Validate_State_Component_Name(module_name, "barostat module") ||
            !Validate_State_Component_Name(state_name, "barostat state") ||
            values == nullptr || count == 0)
        {
            return Mark_Failed();
        }
        if (!writer_.Ensure_Group(Restart_Barostat_State_Root(module_name)))
        {
            return false;
        }
        const std::string dataset_path =
            Restart_Barostat_State_Path(module_name, state_name);
        if (!writer_.Create_Dataset({dataset_path,
                                     DataType::int64,
                                     {{0}, {count}, {count}},
                                     true}) ||
            !writer_.Append_Int64(dataset_path, values, count))
        {
            return false;
        }
        state_hash_.Add_Numeric(dataset_path, {count}, values, count);
        return true;
    }

    bool Write_Sits_State(const std::string& module_name,
                          const std::string& state_name, const float* values,
                          std::size_t count)
    {
        const std::string module_path = Restart_Sits_State_Root(module_name);
        const std::string dataset_path =
            Restart_Sits_State_Path(module_name, state_name);
        if (!writer_.Ensure_Group(module_path)) return false;
        if (!writer_.Create_Dataset({dataset_path,
                                     DataType::float32,
                                     {{0}, {count}, {count}},
                                     true}))
        {
            return false;
        }
        if (!writer_.Append_Float32(dataset_path, values, count)) return false;
        state_hash_.Add_Numeric(dataset_path, {count}, values, count);
        return true;
    }

    bool Write_Sits_State_Schema_Version(const std::string& module_name,
                                         std::int64_t schema_version)
    {
        const std::string module_path = Restart_Sits_State_Root(module_name);
        if (!writer_.Ensure_Group(module_path)) return false;
        const std::string dataset_path =
            Restart_Sits_State_Path(module_name, "schema_version");
        const std::string value = std::to_string(schema_version);
        if (!writer_.Write_String(dataset_path, value)) return false;
        state_hash_.Add_String(dataset_path, value);
        return true;
    }

    bool Write_Restraint_Reference(const std::string& restraint_name,
                                   const float* coordinates,
                                   std::size_t atom_count)
    {
        if (!Validate_State_Component_Name(restraint_name,
                                           "restraint state name"))
        {
            return Mark_Failed();
        }
        if (coordinates == nullptr || atom_count == 0)
        {
            last_error_ =
                "restraint reference requires coordinates and a positive "
                "atom count";
            return Mark_Failed();
        }
        const std::string root =
            Restart_Restraint_Reference_Root(restraint_name);
        const std::string dataset_path =
            Restart_Restraint_Reference_Coordinate_Path(restraint_name);
        if (!writer_.Ensure_Group(root)) return false;
        if (!writer_.Create_Dataset({dataset_path,
                                     DataType::float32,
                                     {{0, 3}, {atom_count, 3}, {atom_count, 3}},
                                     true}))
        {
            return false;
        }
        if (!writer_.Append_Float32(dataset_path, coordinates, atom_count * 3))
        {
            return false;
        }
        state_hash_.Add_Numeric(dataset_path, {atom_count, 3}, coordinates,
                                atom_count * 3);
        return true;
    }

    bool Write_CV_Reference(const std::string& cv_name,
                            const float* coordinates, std::size_t atom_count)
    {
        if (!Validate_State_Component_Name(cv_name, "CV reference name"))
        {
            return Mark_Failed();
        }
        if (coordinates == nullptr || atom_count == 0)
        {
            last_error_ =
                "CV reference requires coordinates and a positive atom count";
            return Mark_Failed();
        }
        const std::string root = Restart_CV_Reference_Root(cv_name);
        const std::string dataset_path =
            Restart_CV_Reference_Coordinate_Path(cv_name);
        if (!writer_.Ensure_Group(root)) return false;
        if (!writer_.Create_Dataset({dataset_path,
                                     DataType::float32,
                                     {{0, 3}, {atom_count, 3}, {atom_count, 3}},
                                     true}))
        {
            return false;
        }
        if (!writer_.Append_Float32(dataset_path, coordinates, atom_count * 3))
        {
            return false;
        }
        state_hash_.Add_Numeric(dataset_path, {atom_count, 3}, coordinates,
                                atom_count * 3);
        return true;
    }

    bool Write_Metad_State_Text(const std::string& name,
                                const std::string& component,
                                const std::string& value)
    {
        const std::string meta_path = Restart_Metad_State_Root(name);
        if (!writer_.Ensure_Group(meta_path)) return false;
        const std::string dataset_path =
            Restart_Metad_State_Path(name, component);
        if (!writer_.Write_String(dataset_path, value)) return false;
        state_hash_.Add_String(dataset_path, value);
        return true;
    }

    bool Write_Metadynamics_State(const RestartMetadynamicsState& state)
    {
        if (!state.has_typed_state)
        {
            last_error_ = "metadynamics typed state is not populated";
            return Mark_Failed();
        }
        if (!Validate_State_Component_Name(state.name,
                                           "metadynamics state name") ||
            state.ndim == 0 || state.grid_count.size() != state.ndim ||
            state.grid_min.size() != state.ndim ||
            state.grid_max.size() != state.ndim)
        {
            if (last_error_.empty())
            {
                last_error_ =
                    "metadynamics typed state grid metadata is invalid";
            }
            return Mark_Failed();
        }
        const std::string root = Restart_Metad_State_Root(state.name);
        std::size_t expected_grid_size = 1;
        for (const std::int64_t count : state.grid_count)
        {
            if (count <= 0 || static_cast<std::uint64_t>(count) >
                                  std::numeric_limits<std::size_t>::max() /
                                      expected_grid_size)
            {
                last_error_ = "metadynamics typed state grid count is invalid";
                return Mark_Failed();
            }
            expected_grid_size *= static_cast<std::size_t>(count);
        }
        const bool invalid_grid_state =
            state.potential_value.size() != expected_grid_size ||
            (!state.potential_force.empty() &&
             state.potential_force.size() != expected_grid_size * state.ndim) ||
            (!state.edge_log_normalization.empty() &&
             state.edge_log_normalization.size() != expected_grid_size) ||
            (!state.edge_normal_force.empty() &&
             state.edge_normal_force.size() != expected_grid_size * state.ndim);
        const bool invalid_scatter_state =
            (state.scatter_count == 0 && (!state.scatter_position.empty() ||
                                          !state.scatter_potential.empty() ||
                                          !state.scatter_force.empty())) ||
            (state.scatter_count != 0 &&
             (state.scatter_position.size() !=
                  state.scatter_count * state.ndim ||
              state.scatter_potential.size() != state.scatter_count ||
              state.scatter_force.size() != state.scatter_count * state.ndim));
        const bool invalid_hill_state =
            state.hill_center.size() != state.hill_count * state.ndim ||
            state.hill_height.size() != state.hill_count ||
            state.hill_inverse_width.size() != state.hill_count * state.ndim ||
            state.hill_period.size() != state.hill_count * state.ndim;
        if (invalid_grid_state || invalid_scatter_state || invalid_hill_state)
        {
            last_error_ =
                "metadynamics typed state array dimensions are invalid";
            return Mark_Failed();
        }
        if (!writer_.Ensure_Group(root)) return false;
        const std::int64_t ndim = static_cast<std::int64_t>(state.ndim);
        if (!Write_Metad_Int64(root + "/state_schema_version",
                               &state.state_schema_version, {1}) ||
            !Write_Metad_Int64(root + "/ndim", &ndim, {1}) ||
            !Write_Metad_Int64(root + "/grid/count", state.grid_count.data(),
                               {state.ndim}) ||
            !Write_Metad_Float(root + "/grid/min", state.grid_min.data(),
                               {state.ndim}) ||
            !Write_Metad_Float(root + "/grid/max", state.grid_max.data(),
                               {state.ndim}))
        {
            return Mark_Failed();
        }
        const std::size_t grid_size = state.potential_value.size();
        if (grid_size != 0 &&
            (!Write_Metad_Float(root + "/potential/value",
                                state.potential_value.data(), {grid_size}) ||
             (!state.potential_force.empty() &&
              !Write_Metad_Float(root + "/potential/force",
                                 state.potential_force.data(),
                                 {grid_size, state.ndim}))))
        {
            return Mark_Failed();
        }
        if ((!state.edge_log_normalization.empty() &&
             !Write_Metad_Float(root + "/edge/log_normalization",
                                state.edge_log_normalization.data(),
                                {state.edge_log_normalization.size()})) ||
            (!state.edge_normal_force.empty() &&
             !Write_Metad_Float(
                 root + "/edge/normal_force", state.edge_normal_force.data(),
                 {state.edge_log_normalization.size(), state.ndim})))
        {
            return Mark_Failed();
        }
        if (state.scatter_count != 0 &&
            (!Write_Metad_Float(root + "/scatter/position",
                                state.scatter_position.data(),
                                {state.scatter_count, state.ndim}) ||
             !Write_Metad_Float(root + "/scatter/potential",
                                state.scatter_potential.data(),
                                {state.scatter_count}) ||
             !Write_Metad_Float(root + "/scatter/force",
                                state.scatter_force.data(),
                                {state.scatter_count, state.ndim})))
        {
            return Mark_Failed();
        }
        if (state.hill_count != 0 &&
            (!Write_Metad_Float(root + "/hills/center",
                                state.hill_center.data(),
                                {state.hill_count, state.ndim}) ||
             !Write_Metad_Float(root + "/hills/height",
                                state.hill_height.data(), {state.hill_count}) ||
             !Write_Metad_Float(root + "/hills/inverse_width",
                                state.hill_inverse_width.data(),
                                {state.hill_count, state.ndim}) ||
             !Write_Metad_Float(root + "/hills/period",
                                state.hill_period.data(),
                                {state.hill_count, state.ndim})))
        {
            return Mark_Failed();
        }
        if (!state.hill_sink.empty() &&
            !Write_Metad_Float(root + "/hills/sink", state.hill_sink.data(),
                               {state.hill_sink.size()}))
        {
            return Mark_Failed();
        }
        if (state.has_runtime_state)
        {
            const float runtime_values[] = {state.potential_max,
                                            state.sum_max,
                                            state.new_max,
                                            state.exit_tag,
                                            state.rct,
                                            state.rbias,
                                            state.bias,
                                            state.minus_beta_f,
                                            state.minus_beta_f_plus_v};
            const char* runtime_names[] = {
                "potential_max", "sum_max",      "new_max",
                "exit_tag",      "rct",          "rbias",
                "bias",          "minus_beta_f", "minus_beta_f_plus_v"};
            for (std::size_t i = 0; i < 9; ++i)
            {
                if (!Write_Metad_Float(root + "/runtime/" + runtime_names[i],
                                       &runtime_values[i], {1}))
                {
                    return Mark_Failed();
                }
            }
            if (!Write_Metad_Int64(root + "/runtime/max_index",
                                   &state.max_index, {1}))
            {
                return Mark_Failed();
            }
        }
        return true;
    }

    bool Write_Protocol_Sidecar_Text(const std::string& key,
                                     const std::string& value)
    {
        if (key.empty() || key.find('/') != std::string::npos)
        {
            last_error_ =
                "protocol sidecar restart key must be non-empty and must not "
                "contain '/'";
            return Mark_Failed();
        }
        if (!writer_.Ensure_Group(path::restart_protocol_sidecars))
            return false;
        const std::string dataset_path = Restart_Protocol_Sidecar_Path(key);
        if (!writer_.Write_String(dataset_path, value)) return false;
        state_hash_.Add_String(dataset_path, value);
        return true;
    }

    bool Write_Legacy_Sidecar_Paths(const std::vector<std::string>& keys,
                                    const std::vector<std::string>& paths)
    {
        if (!writer_.Ensure_Group(path::sponge_files)) return false;
        if (!writer_.Ensure_Group(path::legacy_sidecars)) return false;
        return writer_.Write_String_Array(path::legacy_sidecar_keys, keys) &&
               writer_.Write_String_Array(path::legacy_sidecar_paths, paths);
    }

    bool Finalize()
    {
        const std::string state_hash = state_hash_.Digest("restart.spgr.h5");
        if (!writer_.Write_String(path::run_state_hash, state_hash))
        {
            return Mark_Failed();
        }
        return writer_.Finalize(1);
    }
    bool Flush() { return writer_.Flush(); }
    bool Close() { return writer_.Close(); }

    bool State_Written() const { return state_written_; }

    std::string Last_Error() const
    {
        if (!last_error_.empty()) return last_error_;
        return writer_.Last_Error();
    }

   private:
    bool Write_Metad_Float(const std::string& dataset_path, const float* values,
                           const std::vector<std::size_t>& dimensions)
    {
        if (values == nullptr || dimensions.empty())
        {
            last_error_ =
                "metadynamics float dataset is empty: " + dataset_path;
            return false;
        }
        std::vector<std::size_t> initial = dimensions;
        initial[0] = 0;
        if (!writer_.Create_Dataset({dataset_path,
                                     DataType::float32,
                                     {initial, dimensions, dimensions},
                                     true}))
        {
            return false;
        }
        std::size_t count = 1;
        for (const std::size_t dimension : dimensions) count *= dimension;
        if (!writer_.Append_Float32(dataset_path, values, count)) return false;
        state_hash_.Add_Numeric(dataset_path, dimensions, values, count);
        return true;
    }

    bool Write_Metad_Int64(const std::string& dataset_path,
                           const std::int64_t* values,
                           const std::vector<std::size_t>& dimensions)
    {
        if (values == nullptr || dimensions.empty())
        {
            last_error_ =
                "metadynamics integer dataset is empty: " + dataset_path;
            return false;
        }
        std::vector<std::size_t> initial = dimensions;
        initial[0] = 0;
        if (!writer_.Create_Dataset({dataset_path,
                                     DataType::int64,
                                     {initial, dimensions, dimensions},
                                     true}))
        {
            return false;
        }
        std::size_t count = 1;
        for (const std::size_t dimension : dimensions) count *= dimension;
        if (!writer_.Append_Int64(dataset_path, values, count)) return false;
        state_hash_.Add_Numeric(dataset_path, dimensions, values, count);
        return true;
    }

    bool Validate_State_Component_Name(const std::string& name,
                                       const char* label)
    {
        if (name.empty() || name.find('/') != std::string::npos)
        {
            last_error_ = std::string(label) +
                          " must be non-empty and must not contain '/'";
            return false;
        }
        return true;
    }

    bool Mark_Failed()
    {
        const std::string reason = Last_Error();
        writer_.Mark_Failed(reason);
        return false;
    }

    H5MDWriter writer_;
    std::size_t atom_count_ = 0;
    bool include_velocity_ = false;
    bool state_written_ = false;
    bool generation_written_ = false;
    CanonicalDatasetHash state_hash_;
    std::string last_error_;
};
}  // namespace SpongeH5MD

#pragma once

#include <hdf5.h>

#include <algorithm>
#include <cstdint>
#include <highfive/highfive.hpp>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "utils/h5md/canonical_dataset_hash.hpp"
#include "utils/h5md/h5_input_metadata.hpp"
#include "utils/h5md/h5_structural_state.hpp"
#include "utils/h5md/h5md_writer.hpp"

namespace SpongeH5MD
{
class RestartH5Reader
{
   public:
    bool Open(const std::string& file_path)
    {
        last_error_.clear();
        try
        {
            file_.reset(
                new HighFive::File(file_path, HighFive::File::ReadOnly));
            return true;
        }
        catch (const std::exception& err)
        {
            return Fail(std::string("failed to open restart H5 file: ") +
                        err.what());
        }
    }

    bool Read_Metadata(SpongeH5InputMetadata::RestartMetadata* metadata)
    {
        if (metadata == nullptr)
        {
            return Fail("restart metadata output pointer is null");
        }
        if (!Ensure_File()) return false;

        SpongeH5InputMetadata::RestartMetadata result;
        try
        {
            if (Exists("/schema/name"))
            {
                result.schema_name = Read_String("/schema/name");
            }
            else if (Exists(path::sponge_schema_name))
            {
                result.schema_name = Read_String(path::sponge_schema_name);
            }
            if (Exists(path::sponge_schema_version))
            {
                result.schema_version =
                    Read_String(path::sponge_schema_version);
            }
            else if (Exists("/schema/version"))
            {
                result.schema_version = Read_String("/schema/version");
            }
            if (Exists(path::identity_uuid))
            {
                result.identity_uuid = Read_String(path::identity_uuid);
            }
            if (Exists(path::run_atom_order_hash))
            {
                result.atom_ordering_hash =
                    Read_String(path::run_atom_order_hash);
            }
            if (Exists(path::run_topology_hash))
            {
                result.producer_topology_hash =
                    Read_String(path::run_topology_hash);
            }
            if (Exists(path::run_producer_protocol_hash))
            {
                result.producer_protocol_hash =
                    Read_String(path::run_producer_protocol_hash);
            }
            if (Exists(path::run_state_hash))
            {
                result.state_hash = Read_String(path::run_state_hash);
            }

            result.has_structural_state = Has_Structural_State();
            result.has_velocity = Exists(path::velocity_value);
            if (Exists(path::position_value))
            {
                const auto dims = Dimensions(path::position_value);
                if (dims.size() == 3)
                {
                    result.atom_count = static_cast<std::int64_t>(dims[1]);
                }
            }
            result.has_dynamic_state =
                Group_Has_Dataset(path::restart_rng_state) ||
                Group_Has_Dataset(path::restart_integrator_state) ||
                Group_Has_Dataset(path::restart_thermostat) ||
                Group_Has_Dataset(path::restart_barostat);
            result.has_protocol_state =
                Group_Has_Dataset(path::restart_bias) ||
                Group_Has_Dataset(path::restart_restraint_references) ||
                Group_Has_Dataset(path::restart_cv_references) ||
                Group_Has_Dataset(path::restart_protocol_sidecars);
            result.computed_state_hash = Compute_State_Hash_Impl();

            *metadata = result;
            return true;
        }
        catch (const std::exception& err)
        {
            return Fail(std::string("failed to read restart metadata: ") +
                        err.what());
        }
    }

    bool Read_Structural_State(RestartStructuralState* state)
    {
        if (state == nullptr)
        {
            return Fail("restart structural state output pointer is null");
        }
        if (!Ensure_File()) return false;

        try
        {
            const auto position_dims = Require_Dimensions(
                path::position_value, {1, 0, 3}, "restart position");
            const std::size_t atom_count = position_dims[1];
            if (atom_count == 0)
            {
                return Fail("restart position atom dimension must be positive");
            }

            Require_Dimensions(path::particles_all_step, {1}, "restart step");
            Require_Dimensions(path::particles_all_time, {1}, "restart time");
            Require_Dimensions(path::box_edges_value, {1, 3, 3},
                               "restart box edges");

            RestartStructuralState result;
            result.atom_count = atom_count;
            result.step = Read_Required_Single<std::int64_t>(
                path::particles_all_step, "restart step");
            result.time = Read_Required_Single<double>(path::particles_all_time,
                                                       "restart time");
            result.position_xyz = Read_Required_Vector<float>(
                path::position_value, atom_count * 3, "restart position");

            const auto box = Read_Required_Vector<float>(
                path::box_edges_value, 9, "restart box edges");
            for (std::size_t i = 0; i < result.box_edges.size(); ++i)
            {
                result.box_edges[i] = box[i];
            }

            if (Exists(path::velocity_value))
            {
                Require_Dimensions(path::velocity_value, {1, atom_count, 3},
                                   "restart velocity");
                result.velocity_xyz = Read_Required_Vector<float>(
                    path::velocity_value, atom_count * 3, "restart velocity");
                result.has_velocity = true;
            }

            *state = result;
            return true;
        }
        catch (const std::exception& err)
        {
            return Fail(
                std::string("failed to read restart structural state: ") +
                err.what());
        }
    }

    bool Read_Dynamic_State(RestartDynamicState* state)
    {
        if (state == nullptr)
        {
            return Fail("restart dynamic state output pointer is null");
        }
        if (!Ensure_File()) return false;

        try
        {
            RestartDynamicState result;
            if (Exists(path::restart_nhc))
            {
                const auto dims =
                    Require_Dimensions(path::restart_nhc, {0, 2},
                                       "restart Nose-Hoover chain state");
                result.has_nose_hoover_chain = true;
                result.nose_hoover_chain_pair_count = dims[0];
                result.nose_hoover_chain_coordinate_velocity_pairs =
                    Read_Required_Vector<float>(
                        path::restart_nhc, dims[0] * dims[1],
                        "restart Nose-Hoover chain state");
            }
            if (Exists(path::restart_rng_state))
            {
                for (const auto& module_name :
                     List_Group_Children(path::restart_rng_state,
                                         HighFive::ObjectType::Dataset))
                {
                    result.rng_state_text[module_name] =
                        Read_String(Restart_Rng_State_Path(module_name));
                }
                for (const auto& module_name : List_Group_Children(
                         path::restart_rng_state, HighFive::ObjectType::Group))
                {
                    const std::string schema_path =
                        Restart_Rng_State_Component_Path(module_name,
                                                         "schema_version");
                    const std::string engine_path =
                        Restart_Rng_State_Component_Path(module_name, "engine");
                    const std::string words_path =
                        Restart_Rng_State_Component_Path(module_name,
                                                         "state_words");
                    RestartRngState rng;
                    rng.state_schema_version =
                        Read_Numeric_Single<std::int64_t>(
                            schema_path, "typed RNG state schema version");
                    rng.engine = Read_String(engine_path);
                    const auto dimensions = Require_Dimensions(
                        words_path, {0, 0}, "typed RNG state words");
                    rng.stream_count = dimensions[0];
                    rng.words_per_stream = dimensions[1];
                    rng.state_words = Read_Required_Vector<std::int64_t>(
                        words_path, rng.stream_count * rng.words_per_stream,
                        "typed RNG state words");
                    result.rng_states[module_name] = rng;
                }
            }
            if (Exists(path::restart_integrator_state))
            {
                for (const auto& key :
                     List_Group_Children(path::restart_integrator_state,
                                         HighFive::ObjectType::Dataset))
                {
                    result.integrator_state_text[key] =
                        Read_String(Restart_Integrator_State_Path(key));
                }
            }
            if (Exists(path::restart_thermostat))
            {
                for (const auto& module_name : List_Group_Children(
                         path::restart_thermostat, HighFive::ObjectType::Group))
                {
                    std::map<std::string, std::string> text_states;
                    std::map<std::string, std::vector<float>> float_states;
                    std::map<std::string, std::vector<std::int64_t>>
                        integer_states;
                    Read_Named_Dynamic_State_Group(
                        Restart_Thermostat_State_Root(module_name),
                        &text_states, &float_states, &integer_states);
                    if (!text_states.empty())
                        result.thermostat_text_states[module_name] =
                            text_states;
                    if (!float_states.empty())
                        result.thermostat_float_states[module_name] =
                            float_states;
                    if (!integer_states.empty())
                        result.thermostat_integer_states[module_name] =
                            integer_states;
                }
            }
            if (Exists(path::restart_barostat))
            {
                for (const auto& module_name : List_Group_Children(
                         path::restart_barostat, HighFive::ObjectType::Group))
                {
                    std::map<std::string, std::string> text_states;
                    std::map<std::string, std::vector<float>> float_states;
                    std::map<std::string, std::vector<std::int64_t>>
                        integer_states;
                    Read_Named_Dynamic_State_Group(
                        Restart_Barostat_State_Root(module_name), &text_states,
                        &float_states, &integer_states);
                    if (!text_states.empty())
                        result.barostat_text_states[module_name] = text_states;
                    if (!float_states.empty())
                        result.barostat_float_states[module_name] =
                            float_states;
                    if (!integer_states.empty())
                        result.barostat_integer_states[module_name] =
                            integer_states;
                }
            }

            *state = result;
            return true;
        }
        catch (const std::exception& err)
        {
            return Fail(std::string("failed to read restart dynamic state: ") +
                        err.what());
        }
    }

    bool Read_Protocol_State(RestartProtocolState* state)
    {
        if (state == nullptr)
        {
            return Fail("restart protocol state output pointer is null");
        }
        if (!Ensure_File()) return false;

        try
        {
            RestartProtocolState result;
            if (Exists(path::restart_sits))
            {
                for (const auto& module_name : List_Group_Children(
                         path::restart_sits, HighFive::ObjectType::Group))
                {
                    RestartSitsState module;
                    module.module_name = module_name;
                    const std::string module_path =
                        Restart_Sits_State_Root(module_name);
                    for (const auto& state_name : List_Group_Children(
                             module_path, HighFive::ObjectType::Dataset))
                    {
                        const std::string dataset_path =
                            Restart_Sits_State_Path(module_name, state_name);
                        if (state_name == "schema_version")
                        {
                            if (Dataset_Is_String(dataset_path))
                            {
                                module.state_schema_version =
                                    std::stoll(Read_String(dataset_path));
                            }
                            else
                            {
                                module.state_schema_version =
                                    Read_Numeric_Single<std::int64_t>(
                                        dataset_path,
                                        "restart SITS schema version");
                            }
                            if (module.state_schema_version != 1)
                            {
                                throw std::runtime_error(
                                    "unsupported restart SITS schema version");
                            }
                            continue;
                        }
                        module.float_states[state_name] = Read_Float_Dataset(
                            dataset_path, "restart SITS state");
                    }
                    result.sits_states.push_back(module);
                }
            }

            if (Exists(path::restart_meta))
            {
                for (const auto& metad_name : List_Group_Children(
                         path::restart_meta, HighFive::ObjectType::Group))
                {
                    RestartMetadynamicsState metadynamics;
                    metadynamics.name = metad_name;
                    const std::string metad_path =
                        Restart_Metad_State_Root(metad_name);
                    for (const auto& component_name : List_Group_Children(
                             metad_path, HighFive::ObjectType::Dataset))
                    {
                        const std::string component_path =
                            Restart_Metad_State_Path(metad_name,
                                                     component_name);
                        if (Dataset_Is_String(component_path))
                        {
                            metadynamics.text_states[component_name] =
                                Read_String(component_path);
                        }
                    }
                    if (Exists(metad_path + "/state_schema_version"))
                    {
                        Read_Canonical_Metadynamics_State(metad_path,
                                                          &metadynamics);
                    }
                    else if (Exists(metad_path + "/potential/axis/min") ||
                             Exists(metad_path + "/scatter/axis/min") ||
                             Exists(metad_path + "/hills_typed/value"))
                    {
                        Read_Xponge_Metadynamics_State(metad_path,
                                                       &metadynamics);
                    }
                    result.metadynamics_states.push_back(metadynamics);
                }
            }

            if (Exists(path::restart_restraint_references))
            {
                for (const auto& restraint_name :
                     List_Group_Children(path::restart_restraint_references,
                                         HighFive::ObjectType::Group))
                {
                    RestartRestraintState restraint;
                    restraint.name = restraint_name;
                    const std::string coordinate_path =
                        Restart_Restraint_Reference_Coordinate_Path(
                            restraint_name);
                    const auto dims = Require_Dimensions(
                        coordinate_path, {0, 3},
                        "restart restraint reference coordinate");
                    if (dims[0] == 0)
                    {
                        throw std::runtime_error(
                            "restart restraint reference coordinate atom "
                            "dimension must be positive at " +
                            coordinate_path);
                    }
                    restraint.atom_count = dims[0];
                    restraint.reference_coordinates =
                        Read_Required_Vector<float>(
                            coordinate_path, dims[0] * dims[1],
                            "restart restraint reference coordinate");
                    result.restraint_states.push_back(std::move(restraint));
                }
            }

            if (Exists(path::restart_cv_references))
            {
                for (const auto& cv_name :
                     List_Group_Children(path::restart_cv_references,
                                         HighFive::ObjectType::Group))
                {
                    RestartCVReferenceState reference;
                    reference.name = cv_name;
                    const std::string coordinate_path =
                        Restart_CV_Reference_Coordinate_Path(cv_name);
                    const auto dims =
                        Require_Dimensions(coordinate_path, {0, 3},
                                           "restart CV reference coordinate");
                    if (dims[0] == 0)
                    {
                        throw std::runtime_error(
                            "restart CV reference coordinate atom dimension "
                            "must be positive at " +
                            coordinate_path);
                    }
                    reference.atom_count = dims[0];
                    reference.reference_coordinates =
                        Read_Required_Vector<float>(
                            coordinate_path, dims[0] * dims[1],
                            "restart CV reference coordinate");
                    result.cv_reference_states.push_back(std::move(reference));
                }
            }

            if (Exists(path::restart_protocol_sidecars))
            {
                for (const auto& key :
                     List_Group_Children(path::restart_protocol_sidecars,
                                         HighFive::ObjectType::Dataset))
                {
                    RestartProtocolSidecarTextState sidecar;
                    sidecar.key = key;
                    sidecar.text =
                        Read_String(Restart_Protocol_Sidecar_Path(key));
                    result.sidecar_text_states.push_back(sidecar);
                }
            }

            *state = result;
            return true;
        }
        catch (const std::exception& err)
        {
            return Fail(std::string("failed to read restart protocol state: ") +
                        err.what());
        }
    }

    bool Compute_State_Hash(std::string* state_hash)
    {
        if (state_hash == nullptr)
        {
            return Fail("restart state-hash output pointer is null");
        }
        if (!Ensure_File()) return false;
        try
        {
            *state_hash = Compute_State_Hash_Impl();
            return true;
        }
        catch (const std::exception& err)
        {
            return Fail(std::string("failed to compute restart state hash: ") +
                        err.what());
        }
    }

    std::string Last_Error() const { return last_error_; }

   private:
    bool Ensure_File()
    {
        if (file_ == nullptr)
        {
            return Fail("restart H5 reader is not open");
        }
        return true;
    }

    bool Exists(const std::string& object_path) const
    {
        return file_ != nullptr && file_->exist(object_path);
    }

    bool Has_Structural_State() const
    {
        return Exists(path::particles_all_step) &&
               Exists(path::particles_all_time) &&
               Exists(path::position_value) && Exists(path::box_edges_value);
    }

    bool Group_Has_Dataset(const std::string& group_path)
    {
        if (!Exists(group_path))
        {
            return false;
        }
        const auto group = file_->getGroup(group_path);
        for (const auto& object_name : group.listObjectNames())
        {
            const std::string child_path = group_path + "/" + object_name;
            const auto type = file_->getObjectType(child_path);
            if (type == HighFive::ObjectType::Dataset)
            {
                return true;
            }
            if (type == HighFive::ObjectType::Group &&
                Group_Has_Dataset(child_path))
            {
                return true;
            }
        }
        return false;
    }

    std::vector<std::string> List_Group_Children(
        const std::string& group_path, HighFive::ObjectType object_type)
    {
        std::vector<std::string> names;
        if (!Exists(group_path))
        {
            return names;
        }
        const auto group = file_->getGroup(group_path);
        for (const auto& object_name : group.listObjectNames())
        {
            const std::string child_path = group_path + "/" + object_name;
            if (file_->getObjectType(child_path) == object_type)
            {
                names.push_back(object_name);
            }
        }
        return names;
    }

    std::string Read_String(const std::string& dataset_path)
    {
        std::string value;
        file_->getDataSet(dataset_path).read(value);
        return value;
    }

    template <typename T>
    std::vector<T> Read_All_Numeric(const std::string& dataset_path,
                                    const std::vector<std::size_t>& dimensions)
    {
        const std::size_t count = dimensions.empty() ? 1 : Product(dimensions);
        std::vector<T> values(count);
        const auto dataset = file_->getDataSet(dataset_path);
        if (H5Dread(dataset.getId(), Native_H5_Type<T>(), H5S_ALL, H5S_ALL,
                    H5P_DEFAULT, values.data()) < 0)
        {
            throw std::runtime_error(
                "failed to read restart state dataset at " + dataset_path);
        }
        return values;
    }

    void Collect_Dataset_Paths(const std::string& group_path,
                               std::vector<std::string>* paths)
    {
        if (!Exists(group_path)) return;
        const auto group = file_->getGroup(group_path);
        for (const std::string& name : group.listObjectNames())
        {
            const std::string child = group_path + "/" + name;
            const auto type = file_->getObjectType(child);
            if (type == HighFive::ObjectType::Dataset)
            {
                paths->push_back(child);
            }
            else if (type == HighFive::ObjectType::Group)
            {
                Collect_Dataset_Paths(child, paths);
            }
        }
    }

    void Add_Dataset_To_State_Hash(const std::string& dataset_path,
                                   CanonicalDatasetHash* hash)
    {
        const auto dataset = file_->getDataSet(dataset_path);
        const auto dimensions = dataset.getSpace().getDimensions();
        const HighFive::DataType data_type = dataset.getDataType();
        const H5T_class_t type_class = H5Tget_class(data_type.getId());
        if (type_class == H5T_STRING)
        {
            if (dimensions.empty())
            {
                std::string value;
                dataset.read(value);
                hash->Add_String(dataset_path, value);
            }
            else
            {
                std::vector<std::string> values;
                dataset.read(values);
                hash->Add_Strings(dataset_path, dimensions, values);
            }
            return;
        }
        if (type_class == H5T_FLOAT)
        {
            if (H5Tget_size(data_type.getId()) == sizeof(float))
            {
                const auto values =
                    Read_All_Numeric<float>(dataset_path, dimensions);
                hash->Add_Numeric(dataset_path, dimensions, values.data(),
                                  values.size());
                return;
            }
            if (H5Tget_size(data_type.getId()) == sizeof(double))
            {
                const auto values =
                    Read_All_Numeric<double>(dataset_path, dimensions);
                hash->Add_Numeric(dataset_path, dimensions, values.data(),
                                  values.size());
                return;
            }
        }
        if (type_class == H5T_INTEGER)
        {
            const bool is_unsigned =
                H5Tget_sign(data_type.getId()) == H5T_SGN_NONE;
            const std::size_t size = H5Tget_size(data_type.getId());
            if (size == sizeof(std::int64_t))
            {
                if (is_unsigned)
                {
                    const auto values = Read_All_Numeric<std::uint64_t>(
                        dataset_path, dimensions);
                    hash->Add_Numeric(dataset_path, dimensions, values.data(),
                                      values.size());
                }
                else
                {
                    const auto values = Read_All_Numeric<std::int64_t>(
                        dataset_path, dimensions);
                    hash->Add_Numeric(dataset_path, dimensions, values.data(),
                                      values.size());
                }
                return;
            }
            if (size == sizeof(std::int32_t))
            {
                if (is_unsigned)
                {
                    const auto values = Read_All_Numeric<std::uint32_t>(
                        dataset_path, dimensions);
                    hash->Add_Numeric(dataset_path, dimensions, values.data(),
                                      values.size());
                }
                else
                {
                    const auto values = Read_All_Numeric<std::int32_t>(
                        dataset_path, dimensions);
                    hash->Add_Numeric(dataset_path, dimensions, values.data(),
                                      values.size());
                }
                return;
            }
            if (size == sizeof(std::int8_t))
            {
                if (is_unsigned)
                {
                    const auto values = Read_All_Numeric<std::uint8_t>(
                        dataset_path, dimensions);
                    hash->Add_Numeric(dataset_path, dimensions, values.data(),
                                      values.size());
                }
                else
                {
                    const auto values =
                        Read_All_Numeric<std::int8_t>(dataset_path, dimensions);
                    hash->Add_Numeric(dataset_path, dimensions, values.data(),
                                      values.size());
                }
                return;
            }
        }
        throw std::runtime_error("unsupported restart state dataset type at " +
                                 dataset_path);
    }

    std::string Compute_State_Hash_Impl()
    {
        CanonicalDatasetHash hash;
        std::vector<std::string> paths;
        for (const char* particle_path :
             {path::particles_all_step, path::particles_all_time,
              path::position_value, path::velocity_value, path::box_edges_value,
              path::force_value})
        {
            if (Exists(particle_path)) paths.push_back(particle_path);
        }
        Collect_Dataset_Paths(path::parameters_restart, &paths);
        std::sort(paths.begin(), paths.end());
        paths.erase(std::unique(paths.begin(), paths.end()), paths.end());
        for (const std::string& dataset_path : paths)
        {
            Add_Dataset_To_State_Hash(dataset_path, &hash);
        }
        return hash.Digest("restart.spgr.h5");
    }

    std::vector<std::size_t> Dimensions(const std::string& dataset_path)
    {
        return file_->getDataSet(dataset_path).getSpace().getDimensions();
    }

    std::vector<std::size_t> Require_Dimensions(
        const std::string& dataset_path,
        const std::vector<std::size_t>& expected, const std::string& label)
    {
        if (!Exists(dataset_path))
        {
            throw std::runtime_error(label +
                                     " dataset is missing: " + dataset_path);
        }
        const auto dims = Dimensions(dataset_path);
        if (dims.size() != expected.size())
        {
            std::ostringstream out;
            out << label << " rank mismatch at " << dataset_path
                << ": expected " << expected.size() << ", got " << dims.size();
            throw std::runtime_error(out.str());
        }
        for (std::size_t i = 0; i < expected.size(); ++i)
        {
            if (expected[i] != 0 && dims[i] != expected[i])
            {
                std::ostringstream out;
                out << label << " shape mismatch at " << dataset_path
                    << ": dimension " << i << " expected " << expected[i]
                    << ", got " << dims[i];
                throw std::runtime_error(out.str());
            }
        }
        return dims;
    }

    template <typename T>
    std::vector<T> Read_Required_Vector(const std::string& dataset_path,
                                        std::size_t expected_size,
                                        const std::string& label)
    {
        const auto dims = Dimensions(dataset_path);
        const std::vector<std::size_t> offsets(dims.size(), 0);
        std::vector<T> values =
            Read_Required_Selection<T>(dataset_path, offsets, dims, label);
        if (values.size() != expected_size)
        {
            std::ostringstream out;
            out << label << " value count mismatch at " << dataset_path
                << ": expected " << expected_size << ", got " << values.size();
            throw std::runtime_error(out.str());
        }
        return values;
    }

    std::vector<float> Read_Float_Dataset(const std::string& dataset_path,
                                          const std::string& label)
    {
        if (!Exists(dataset_path))
        {
            throw std::runtime_error(label +
                                     " dataset is missing: " + dataset_path);
        }
        const auto dims = Dimensions(dataset_path);
        const std::vector<std::size_t> offsets(dims.size(), 0);
        return Read_Required_Selection<float>(dataset_path, offsets, dims,
                                              label);
    }

    bool Dataset_Is_String(const std::string& dataset_path)
    {
        const HighFive::DataType data_type =
            file_->getDataSet(dataset_path).getDataType();
        return H5Tget_class(data_type.getId()) == H5T_STRING;
    }

    template <typename T>
    T Read_Numeric_Single(const std::string& dataset_path,
                          const std::string& label)
    {
        if (!Exists(dataset_path))
        {
            throw std::runtime_error(label +
                                     " dataset is missing: " + dataset_path);
        }
        const auto dims = Dimensions(dataset_path);
        if (dims.empty())
        {
            T value{};
            file_->getDataSet(dataset_path).read(value);
            return value;
        }
        if (Product(dims) != 1)
        {
            throw std::runtime_error(label + " must contain one value at " +
                                     dataset_path);
        }
        return Read_Required_Single<T>(dataset_path, label);
    }

    static std::size_t Checked_Grid_Size(
        const std::vector<std::int64_t>& counts, const std::string& label)
    {
        std::size_t result = 1;
        for (const std::int64_t count : counts)
        {
            if (count <= 0 ||
                static_cast<std::uint64_t>(count) >
                    std::numeric_limits<std::size_t>::max() / result)
            {
                throw std::runtime_error(label +
                                         " contains invalid grid counts");
            }
            result *= static_cast<std::size_t>(count);
        }
        return result;
    }

    void Read_Canonical_Metadynamics_State(const std::string& root,
                                           RestartMetadynamicsState* state)
    {
        state->state_schema_version = Read_Numeric_Single<std::int64_t>(
            root + "/state_schema_version",
            "metadynamics state schema version");
        if (state->state_schema_version != 1)
        {
            throw std::runtime_error(
                "unsupported metadynamics state schema version at " + root);
        }
        const std::int64_t ndim = Read_Numeric_Single<std::int64_t>(
            root + "/ndim", "metadynamics state dimension");
        if (ndim <= 0)
        {
            throw std::runtime_error(
                "metadynamics state dimension must be positive at " + root);
        }
        state->ndim = static_cast<std::size_t>(ndim);
        Require_Dimensions(root + "/grid/count", {state->ndim},
                           "metadynamics grid count");
        Require_Dimensions(root + "/grid/min", {state->ndim},
                           "metadynamics grid minimum");
        Require_Dimensions(root + "/grid/max", {state->ndim},
                           "metadynamics grid maximum");
        state->grid_count = Read_Required_Vector<std::int64_t>(
            root + "/grid/count", state->ndim, "metadynamics grid count");
        state->grid_min = Read_Required_Vector<float>(
            root + "/grid/min", state->ndim, "metadynamics grid minimum");
        state->grid_max = Read_Required_Vector<float>(
            root + "/grid/max", state->ndim, "metadynamics grid maximum");
        const std::size_t grid_size =
            Checked_Grid_Size(state->grid_count, root);
        if (Exists(root + "/potential/value"))
        {
            Require_Dimensions(root + "/potential/value", {grid_size},
                               "metadynamics grid potential");
            state->potential_value = Read_Required_Vector<float>(
                root + "/potential/value", grid_size,
                "metadynamics grid potential");
        }
        if (Exists(root + "/potential/force"))
        {
            Require_Dimensions(root + "/potential/force",
                               {grid_size, state->ndim},
                               "metadynamics grid force");
            state->potential_force = Read_Required_Vector<float>(
                root + "/potential/force", grid_size * state->ndim,
                "metadynamics grid force");
        }
        if (Exists(root + "/edge/log_normalization"))
        {
            Require_Dimensions(root + "/edge/log_normalization", {grid_size},
                               "metadynamics edge normalization");
            state->edge_log_normalization = Read_Required_Vector<float>(
                root + "/edge/log_normalization", grid_size,
                "metadynamics edge normalization");
        }
        if (Exists(root + "/edge/normal_force"))
        {
            Require_Dimensions(root + "/edge/normal_force",
                               {grid_size, state->ndim},
                               "metadynamics edge normal force");
            state->edge_normal_force = Read_Required_Vector<float>(
                root + "/edge/normal_force", grid_size * state->ndim,
                "metadynamics edge normal force");
        }
        if (Exists(root + "/scatter/position"))
        {
            const auto dims =
                Require_Dimensions(root + "/scatter/position", {0, state->ndim},
                                   "metadynamics scatter position");
            if (dims[0] == 0)
            {
                throw std::runtime_error(
                    "metadynamics scatter point count must be positive at " +
                    root);
            }
            state->scatter_count = dims[0];
            state->scatter_position = Read_Required_Vector<float>(
                root + "/scatter/position", state->scatter_count * state->ndim,
                "metadynamics scatter position");
            Require_Dimensions(root + "/scatter/potential",
                               {state->scatter_count},
                               "metadynamics scatter potential");
            Require_Dimensions(root + "/scatter/force",
                               {state->scatter_count, state->ndim},
                               "metadynamics scatter force");
            state->scatter_potential = Read_Required_Vector<float>(
                root + "/scatter/potential", state->scatter_count,
                "metadynamics scatter potential");
            state->scatter_force = Read_Required_Vector<float>(
                root + "/scatter/force", state->scatter_count * state->ndim,
                "metadynamics scatter force");
        }
        if (Exists(root + "/hills/center"))
        {
            const auto dims =
                Require_Dimensions(root + "/hills/center", {0, state->ndim},
                                   "metadynamics hill center");
            state->hill_count = dims[0];
            state->hill_center = Read_Required_Vector<float>(
                root + "/hills/center", state->hill_count * state->ndim,
                "metadynamics hill center");
            Require_Dimensions(root + "/hills/height", {state->hill_count},
                               "metadynamics hill height");
            Require_Dimensions(root + "/hills/inverse_width",
                               {state->hill_count, state->ndim},
                               "metadynamics hill inverse width");
            Require_Dimensions(root + "/hills/period",
                               {state->hill_count, state->ndim},
                               "metadynamics hill period");
            state->hill_height = Read_Required_Vector<float>(
                root + "/hills/height", state->hill_count,
                "metadynamics hill height");
            state->hill_inverse_width = Read_Required_Vector<float>(
                root + "/hills/inverse_width", state->hill_count * state->ndim,
                "metadynamics hill inverse width");
            state->hill_period = Read_Required_Vector<float>(
                root + "/hills/period", state->hill_count * state->ndim,
                "metadynamics hill period");
        }
        if (Exists(root + "/hills/sink"))
        {
            state->hill_sink = Read_Float_Dataset(
                root + "/hills/sink", "metadynamics hill sink state");
        }
        if (Exists(root + "/runtime"))
        {
            state->potential_max = Read_Numeric_Single<float>(
                root + "/runtime/potential_max",
                "metadynamics runtime potential maximum");
            state->sum_max = Read_Numeric_Single<float>(
                root + "/runtime/sum_max",
                "metadynamics runtime normalization maximum");
            state->new_max = Read_Numeric_Single<float>(
                root + "/runtime/new_max",
                "metadynamics runtime shifted maximum");
            state->exit_tag = Read_Numeric_Single<float>(
                root + "/runtime/exit_tag", "metadynamics runtime exit tag");
            state->rct = Read_Numeric_Single<float>(root + "/runtime/rct",
                                                    "metadynamics runtime rct");
            state->rbias = Read_Numeric_Single<float>(
                root + "/runtime/rbias", "metadynamics runtime rbias");
            state->bias = Read_Numeric_Single<float>(
                root + "/runtime/bias", "metadynamics runtime bias");
            state->minus_beta_f =
                Read_Numeric_Single<float>(root + "/runtime/minus_beta_f",
                                           "metadynamics runtime minus beta f");
            state->minus_beta_f_plus_v = Read_Numeric_Single<float>(
                root + "/runtime/minus_beta_f_plus_v",
                "metadynamics runtime minus beta f plus v");
            state->max_index = Read_Numeric_Single<std::int64_t>(
                root + "/runtime/max_index",
                "metadynamics runtime maximum index");
            state->has_runtime_state = true;
        }
        state->has_typed_state = true;
    }

    void Read_Xponge_Metadynamics_State(const std::string& root,
                                        RestartMetadynamicsState* state)
    {
        const std::string field_root = Exists(root + "/potential/axis/min")
                                           ? root + "/potential"
                                           : root + "/scatter";
        if (Exists(field_root + "/axis/min"))
        {
            const std::int64_t ndim = Read_Numeric_Single<std::int64_t>(
                field_root + "/ndim", "XPONGE metadynamics dimension");
            if (ndim <= 0)
            {
                throw std::runtime_error(
                    "XPONGE metadynamics dimension must be positive at " +
                    field_root);
            }
            state->ndim = static_cast<std::size_t>(ndim);
            state->grid_count = Read_Required_Vector<std::int64_t>(
                field_root + "/grid", state->ndim,
                "XPONGE metadynamics grid count");
            state->grid_min = Read_Required_Vector<float>(
                field_root + "/axis/min", state->ndim,
                "XPONGE metadynamics grid minimum");
            state->grid_max = Read_Required_Vector<float>(
                field_root + "/axis/max", state->ndim,
                "XPONGE metadynamics grid maximum");
            if (Exists(root + "/potential/value"))
            {
                state->potential_value = Read_Float_Dataset(
                    root + "/potential/value", "XPONGE metadynamics potential");
                if (Exists(root + "/potential/force"))
                {
                    state->potential_force = Read_Float_Dataset(
                        root + "/potential/force", "XPONGE metadynamics force");
                }
            }
            if (Exists(root + "/scatter/coordinate"))
            {
                const std::int64_t scatter_ndim =
                    Read_Numeric_Single<std::int64_t>(
                        root + "/scatter/ndim",
                        "XPONGE metadynamics scatter dimension");
                if (scatter_ndim == ndim)
                {
                    const auto dims = Require_Dimensions(
                        root + "/scatter/coordinate", {0, state->ndim},
                        "XPONGE metadynamics scatter coordinate");
                    state->scatter_count = dims[0];
                    state->scatter_position = Read_Required_Vector<float>(
                        root + "/scatter/coordinate",
                        state->scatter_count * state->ndim,
                        "XPONGE metadynamics scatter coordinate");
                    state->scatter_potential = Read_Float_Dataset(
                        root + "/scatter/value",
                        "XPONGE metadynamics scatter potential");
                    state->scatter_force =
                        Read_Float_Dataset(root + "/scatter/force",
                                           "XPONGE metadynamics scatter force");
                }
            }
            else if (Exists(root + "/potential/coordinate") &&
                     Exists(root + "/potential/scatter_size") &&
                     Read_Numeric_Single<std::int64_t>(
                         root + "/potential/scatter_size",
                         "XPONGE metadynamics scatter size") > 0)
            {
                const auto dims = Require_Dimensions(
                    root + "/potential/coordinate", {0, state->ndim},
                    "XPONGE metadynamics potential coordinate");
                state->scatter_count = dims[0];
                state->scatter_position = Read_Required_Vector<float>(
                    root + "/potential/coordinate",
                    state->scatter_count * state->ndim,
                    "XPONGE metadynamics potential coordinate");
                state->scatter_potential = state->potential_value;
                state->scatter_force = state->potential_force;
            }
        }
        if (Exists(root + "/hills_typed/value"))
        {
            if (state->ndim == 0)
            {
                throw std::runtime_error(
                    "XPONGE typed hills require metadynamics dimensions at " +
                    root);
            }
            const auto dims =
                Require_Dimensions(root + "/hills_typed/value", {0, 0},
                                   "XPONGE metadynamics hills");
            if (dims[1] < state->ndim + 1)
            {
                throw std::runtime_error(
                    "XPONGE metadynamics hill column count is invalid at " +
                    root);
            }
            const auto rows = Read_Required_Vector<float>(
                root + "/hills_typed/value", dims[0] * dims[1],
                "XPONGE metadynamics hills");
            state->hill_count = dims[0];
            state->hill_center.reserve(state->hill_count * state->ndim);
            state->hill_height.reserve(state->hill_count);
            for (std::size_t row = 0; row < state->hill_count; ++row)
            {
                for (std::size_t dim = 0; dim < state->ndim; ++dim)
                {
                    state->hill_center.push_back(rows[row * dims[1] + dim]);
                }
                state->hill_height.push_back(rows[row * dims[1] + state->ndim]);
            }
        }
        state->state_schema_version = 0;
        state->has_typed_state = state->ndim != 0;
    }

    void Read_Named_Dynamic_State_Group(
        const std::string& group_path,
        std::map<std::string, std::string>* text_states,
        std::map<std::string, std::vector<float>>* float_states,
        std::map<std::string, std::vector<std::int64_t>>* integer_states)
    {
        for (const auto& state_name :
             List_Group_Children(group_path, HighFive::ObjectType::Dataset))
        {
            const std::string dataset_path = group_path + "/" + state_name;
            if (Dynamic_State_Name_Is_Integer(state_name))
            {
                const auto dimensions = Require_Dimensions(
                    dataset_path, {0}, "restart integer dynamic state");
                (*integer_states)[state_name] =
                    Read_Required_Vector<std::int64_t>(
                        dataset_path, dimensions[0],
                        "restart integer dynamic state");
            }
            else if (Dynamic_State_Name_Is_Float(state_name))
            {
                (*float_states)[state_name] =
                    Read_Float_Dataset(dataset_path, "restart dynamic state");
            }
            else
            {
                (*text_states)[state_name] = Read_String(dataset_path);
            }
        }
    }

    static bool Dynamic_State_Name_Is_Float(const std::string& state_name)
    {
        return state_name == "g" || state_name == "lambda" ||
               state_name == "delta_box_length_max" ||
               state_name == "total_count" || state_name == "accept_count" ||
               state_name == "accept_rate";
    }

    static bool Dynamic_State_Name_Is_Integer(const std::string& state_name)
    {
        return state_name == "total_count_int64" ||
               state_name == "accept_count_int64";
    }

    template <typename T>
    std::vector<T> Read_Required_Selection(
        const std::string& dataset_path,
        const std::vector<std::size_t>& offsets,
        const std::vector<std::size_t>& counts, const std::string& label)
    {
        if (offsets.size() != counts.size())
        {
            throw std::runtime_error(label + " selection rank mismatch at " +
                                     dataset_path);
        }
        const std::size_t value_count = Product(counts);
        std::vector<T> values(value_count);
        HighFive::DataSet dataset = file_->getDataSet(dataset_path);
        const std::vector<hsize_t> h_offsets = To_HSize(offsets);
        const std::vector<hsize_t> h_counts = To_HSize(counts);
        hid_t file_space = H5Dget_space(dataset.getId());
        if (file_space < 0)
        {
            throw std::runtime_error(label + " failed to get dataspace at " +
                                     dataset_path);
        }
        const herr_t select_rc =
            H5Sselect_hyperslab(file_space, H5S_SELECT_SET, h_offsets.data(),
                                nullptr, h_counts.data(), nullptr);
        if (select_rc < 0)
        {
            H5Sclose(file_space);
            throw std::runtime_error(label + " failed to select hyperslab at " +
                                     dataset_path);
        }
        hid_t mem_space = H5Screate_simple(static_cast<int>(h_counts.size()),
                                           h_counts.data(), nullptr);
        if (mem_space < 0)
        {
            H5Sclose(file_space);
            throw std::runtime_error(label +
                                     " failed to create memory dataspace at " +
                                     dataset_path);
        }
        const herr_t read_rc =
            H5Dread(dataset.getId(), Native_H5_Type<T>(), mem_space, file_space,
                    H5P_DEFAULT, values.data());
        H5Sclose(mem_space);
        H5Sclose(file_space);
        if (read_rc < 0)
        {
            throw std::runtime_error(label + " failed to read hyperslab at " +
                                     dataset_path);
        }
        return values;
    }

    template <typename T>
    T Read_Required_Single(const std::string& dataset_path,
                           const std::string& label)
    {
        const auto values = Read_Required_Vector<T>(dataset_path, 1, label);
        return values[0];
    }

    static std::vector<hsize_t> To_HSize(const std::vector<std::size_t>& values)
    {
        std::vector<hsize_t> converted;
        converted.reserve(values.size());
        for (const std::size_t value : values)
        {
            converted.push_back(static_cast<hsize_t>(value));
        }
        return converted;
    }

    static std::size_t Product(const std::vector<std::size_t>& values)
    {
        return std::accumulate(
            values.begin(), values.end(), static_cast<std::size_t>(1),
            [](std::size_t lhs, std::size_t rhs) { return lhs * rhs; });
    }

    template <typename T>
    static hid_t Native_H5_Type()
    {
        if constexpr (std::is_same<T, float>::value)
        {
            return H5T_NATIVE_FLOAT;
        }
        else if constexpr (std::is_same<T, double>::value)
        {
            return H5T_NATIVE_DOUBLE;
        }
        else if constexpr (std::is_same<T, std::int64_t>::value)
        {
            return H5T_NATIVE_INT64;
        }
        else if constexpr (std::is_same<T, std::uint64_t>::value)
        {
            return H5T_NATIVE_UINT64;
        }
        else if constexpr (std::is_same<T, std::int32_t>::value)
        {
            return H5T_NATIVE_INT32;
        }
        else if constexpr (std::is_same<T, std::uint32_t>::value)
        {
            return H5T_NATIVE_UINT32;
        }
        else if constexpr (std::is_same<T, std::int8_t>::value)
        {
            return H5T_NATIVE_INT8;
        }
        else if constexpr (std::is_same<T, std::uint8_t>::value)
        {
            return H5T_NATIVE_UINT8;
        }
        else
        {
            static_assert(std::is_same<T, float>::value ||
                              std::is_same<T, double>::value ||
                              std::is_same<T, std::int64_t>::value ||
                              std::is_same<T, std::uint64_t>::value ||
                              std::is_same<T, std::int32_t>::value ||
                              std::is_same<T, std::uint32_t>::value ||
                              std::is_same<T, std::int8_t>::value ||
                              std::is_same<T, std::uint8_t>::value,
                          "unsupported HDF5 numeric read type");
        }
    }

    bool Fail(const std::string& message)
    {
        last_error_ = message;
        return false;
    }

    std::unique_ptr<HighFive::File> file_;
    std::string last_error_;
};
}  // namespace SpongeH5MD

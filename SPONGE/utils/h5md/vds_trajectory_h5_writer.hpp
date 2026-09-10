#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

#include "utils/h5md/atomic_file_publish.hpp"
#include "utils/h5md/h5md_writer.hpp"
#include "utils/h5md/module_h5_mappings.hpp"
#include "utils/h5md/output_plan.hpp"
#include "utils/h5md/trajectory_h5_writer.hpp"

namespace SpongeH5MD
{
struct VdsShardManifestEntry
{
    int64_t index = 0;
    std::string path;
    int64_t frame_start = 0;
    int64_t frame_count = 0;
    int64_t byte_size = 0;
    int64_t observable_frame_count = 0;
    int64_t nhc_frame_count = 0;
    int64_t sits_nk_frame_count = 0;
    int64_t metadynamics_scalar_frame_count = 0;
    int64_t reaxff_frame_count = 0;
    int64_t step_start = 0;
    int64_t step_end = 0;
    double time_start = 0.0;
    double time_end = 0.0;
    std::string status = "open";
};

inline bool Manifest_Entry_Has_Stream_Frames(const VdsShardManifestEntry& entry)
{
    return entry.frame_count > 0 || entry.observable_frame_count > 0 ||
           entry.nhc_frame_count > 0 || entry.sits_nk_frame_count > 0 ||
           entry.metadynamics_scalar_frame_count > 0 ||
           entry.reaxff_frame_count > 0;
}

inline bool Manifest_Entry_Has_Negative_Stream_Count(
    const VdsShardManifestEntry& entry)
{
    return entry.frame_count < 0 || entry.observable_frame_count < 0 ||
           entry.nhc_frame_count < 0 || entry.sits_nk_frame_count < 0 ||
           entry.metadynamics_scalar_frame_count < 0 ||
           entry.reaxff_frame_count < 0;
}

class VdsTrajectoryH5Writer
{
   public:
    explicit VdsTrajectoryH5Writer(WriterBackendFactory* backend_factory)
        : backend_factory_(backend_factory)
    {
    }

    ~VdsTrajectoryH5Writer()
    {
        if (wrapper_writer_ != nullptr) wrapper_writer_->Close();
        if (!wrapper_path_.empty())
        {
            Remove_File_If_Exists(wrapper_path_ + ".working");
        }
    }

    bool Open(const SpongeH5OutputPlan::ResolvedOutputPlan& plan,
              const std::string& schema_version = kOutputSchemaVersion,
              const std::string& identity_uuid = Generate_Uuid_V4())
    {
        if (backend_factory_ == nullptr)
        {
            last_error_ = "VDS trajectory writer requires a backend factory";
            return false;
        }
        if (!plan.trajectory.enabled || !plan.trajectory.vds)
        {
            last_error_ =
                "VdsTrajectoryH5Writer requires enabled VDS trajectory plan";
            return false;
        }
        plan_ = plan;
        schema_version_ = schema_version;
        chunk_size_ = plan.trajectory.chunk_size;
        shard_root_ = plan.trajectory.derived_shard_root;
        wrapper_path_ = plan.trajectory.path;
        identity_uuid_ = identity_uuid;

        wrapper_backend_ = backend_factory_->Create_Backend();
        if (wrapper_backend_ == nullptr)
        {
            last_error_ = "failed to create VDS wrapper backend";
            return false;
        }
        wrapper_writer_ = std::make_unique<H5MDWriter>(wrapper_backend_.get());
        WriterOptions options;
        options.path = plan.trajectory.path;
        options.schema_name = "sponge.output.h5md";
        options.schema_version = schema_version;
        options.identity_uuid = identity_uuid_;
        options.output_mode = "vds";
        options.observable_only = false;
        options.atomic_snapshot = true;
        if (!wrapper_writer_->Open(options))
        {
            last_error_ = wrapper_writer_->Last_Error();
            return false;
        }
        if (!wrapper_writer_->Ensure_Group(path::shard_manifest))
        {
            last_error_ = wrapper_writer_->Last_Error();
            return false;
        }
        if (!wrapper_writer_->Ensure_Group(path::shard_manifest_stream_counts))
        {
            last_error_ = wrapper_writer_->Last_Error();
            return false;
        }
        const std::string chunk_size_string = std::to_string(chunk_size_);
        if (!wrapper_writer_->Write_String(path::output_trajectory_chunk_size,
                                           chunk_size_string) ||
            !wrapper_writer_->Flush() ||
            !wrapper_writer_->Publish_Snapshot(wrapper_path_))
        {
            last_error_ = wrapper_writer_->Last_Error();
            return false;
        }
        return true;
    }

    bool Define_Particle_Datasets(const std::size_t atom_count,
                                  bool include_velocity, bool include_force)
    {
        atom_count_ = atom_count;
        include_velocity_ = include_velocity;
        include_force_ = include_force;
        particle_layout_defined_ = true;
        if (current_shard_writer_ != nullptr)
        {
            return current_shard_writer_->Define_Particle_Datasets(
                atom_count_, include_velocity_, include_force_);
        }
        return true;
    }

    bool Write_Topology_Compatibility(const std::string& topology_hash,
                                      const std::string& atom_order_hash)
    {
        topology_hash_ = topology_hash;
        atom_order_hash_ = atom_order_hash;
        if (wrapper_writer_ == nullptr ||
            !wrapper_writer_->Write_Topology_Compatibility(topology_hash_,
                                                           atom_order_hash_))
        {
            last_error_ = wrapper_writer_ == nullptr
                              ? "VDS wrapper writer is not open"
                              : wrapper_writer_->Last_Error();
            return false;
        }
        if (current_shard_writer_ != nullptr &&
            !current_shard_writer_->Write_Topology_Compatibility(
                topology_hash_, atom_order_hash_))
        {
            last_error_ = current_shard_writer_->Last_Error();
            return false;
        }
        return true;
    }

    bool Ensure_Sits_Nk_Observable(const std::string& module_name,
                                   std::size_t k_count)
    {
        sits_module_name_ = module_name;
        sits_k_count_ = k_count;
        sits_nk_layout_defined_ = true;
        if (current_shard_writer_ != nullptr)
        {
            return current_shard_writer_->Ensure_Sits_Nk_Observable(
                sits_module_name_, sits_k_count_);
        }
        return true;
    }

    bool Ensure_Nose_Hoover_Chain_Observables(std::size_t chain_length)
    {
        nhc_chain_length_ = chain_length;
        nhc_layout_defined_ = true;
        if (current_shard_writer_ != nullptr)
        {
            return current_shard_writer_->Ensure_Nose_Hoover_Chain_Observables(
                nhc_chain_length_);
        }
        return true;
    }

    bool Ensure_Metadynamics_Scalars()
    {
        metadynamics_scalar_layout_defined_ = true;
        if (current_shard_writer_ != nullptr)
        {
            return current_shard_writer_->Ensure_Metadynamics_Scalars();
        }
        return true;
    }

    bool Ensure_Reaxff_Energy_Terms(const std::vector<std::string>& terms)
    {
        reaxff_terms_ = terms;
        reaxff_layout_defined_ = true;
        if (current_shard_writer_ != nullptr)
        {
            return current_shard_writer_->Ensure_Reaxff_Energy_Terms(
                reaxff_terms_);
        }
        return true;
    }

    bool Define_Observable_Stream(
        const std::vector<std::string>& hdf5_names,
        const std::vector<std::string>& original_names)
    {
        observable_hdf5_names_ = hdf5_names;
        observable_original_names_ = original_names;
        observable_layout_defined_ = true;
        if (current_shard_writer_ != nullptr)
        {
            return current_shard_writer_->Define_Observable_Stream(
                observable_hdf5_names_, observable_original_names_);
        }
        return true;
    }

    bool Append_Particle_Frame(const int64_t step, const double time,
                               const float* position_xyz,
                               const float* box_edges_3x3,
                               const float* velocity_xyz = nullptr,
                               const float* force_xyz = nullptr)
    {
        if (!particle_layout_defined_)
        {
            last_error_ = "particle layout must be defined before appending";
            return false;
        }
        if (current_shard_writer_ == nullptr ||
            current_shard_frame_count_ >= chunk_size_)
        {
            if (!Rotate_To_New_Shard(step, time))
            {
                return false;
            }
        }
        if (!current_shard_writer_->Append_Particle_Frame(
                step, time, position_xyz, box_edges_3x3, velocity_xyz,
                force_xyz))
        {
            last_error_ = current_shard_writer_->Last_Error();
            return false;
        }
        current_manifest_entry_.frame_count += 1;
        current_manifest_entry_.step_end = step;
        current_manifest_entry_.time_end = time;
        current_shard_frame_count_ += 1;
        total_trajectory_frame_count_ += 1;
        return true;
    }

    bool Append_Observable_Frame(
        const int64_t step, const double time,
        const std::map<std::string, double>& values_by_hdf5_name)
    {
        if (wrapper_writer_ == nullptr)
        {
            last_error_ = "VDS wrapper is not open";
            return false;
        }
        if (!observable_layout_defined_)
        {
            last_error_ = "observable layout must be defined before appending";
            return false;
        }
        if (current_shard_writer_ == nullptr)
        {
            if (!Ensure_Current_Shard(step, time)) return false;
        }
        if (!current_shard_writer_->Append_Observable_Frame(
                step, time, values_by_hdf5_name))
        {
            last_error_ = current_shard_writer_->Last_Error();
            return false;
        }
        current_manifest_entry_.observable_frame_count += 1;
        Extend_Nonparticle_Shard_Range(step, time);
        total_observable_frame_count_ += 1;
        return true;
    }

    bool Append_Sits_Nk_Frame(const int64_t step, const double time,
                              const std::string& module_name,
                              const float* values, std::size_t k_count)
    {
        if (!sits_nk_layout_defined_)
        {
            last_error_ = "SITS nk layout must be defined before appending";
            return false;
        }
        if (module_name != sits_module_name_ || k_count != sits_k_count_)
        {
            last_error_ = "SITS nk shape changed within VDS trajectory";
            return false;
        }
        if (!Ensure_Current_Shard(step, time)) return false;
        if (!current_shard_writer_->Append_Sits_Nk_Frame(
                step, time, module_name, values, k_count))
        {
            last_error_ = current_shard_writer_->Last_Error();
            return false;
        }
        current_manifest_entry_.sits_nk_frame_count += 1;
        Extend_Nonparticle_Shard_Range(step, time);
        return true;
    }

    bool Append_Nose_Hoover_Chain_Frame(const int64_t step, const double time,
                                        const float* coordinates,
                                        const float* velocities,
                                        std::size_t chain_length)
    {
        if (!nhc_layout_defined_)
        {
            last_error_ = "NHC layout must be defined before appending";
            return false;
        }
        if (chain_length != nhc_chain_length_)
        {
            last_error_ = "NHC chain length changed within VDS trajectory";
            return false;
        }
        if (!Ensure_Current_Shard(step, time)) return false;
        if (!current_shard_writer_->Append_Nose_Hoover_Chain_Frame(
                step, time, coordinates, velocities, chain_length))
        {
            last_error_ = current_shard_writer_->Last_Error();
            return false;
        }
        current_manifest_entry_.nhc_frame_count += 1;
        Extend_Nonparticle_Shard_Range(step, time);
        return true;
    }

    bool Append_Metadynamics_Scalar_Frame(const int64_t step, const double time,
                                          double meta, double rbias, double rct)
    {
        if (!metadynamics_scalar_layout_defined_)
        {
            last_error_ =
                "metadynamics scalar layout must be defined before appending";
            return false;
        }
        if (!Ensure_Current_Shard(step, time)) return false;
        if (!current_shard_writer_->Append_Metadynamics_Scalar_Frame(
                step, time, meta, rbias, rct))
        {
            last_error_ = current_shard_writer_->Last_Error();
            return false;
        }
        current_manifest_entry_.metadynamics_scalar_frame_count += 1;
        Extend_Nonparticle_Shard_Range(step, time);
        return true;
    }

    bool Append_Reaxff_Frame(
        const int64_t step, const double time,
        const std::map<std::string, double>& values_by_term)
    {
        if (!reaxff_layout_defined_)
        {
            last_error_ = "ReaxFF layout must be defined before appending";
            return false;
        }
        if (!Ensure_Current_Shard(step, time)) return false;
        if (!current_shard_writer_->Append_Reaxff_Frame(step, time,
                                                        values_by_term))
        {
            last_error_ = current_shard_writer_->Last_Error();
            return false;
        }
        current_manifest_entry_.reaxff_frame_count += 1;
        Extend_Nonparticle_Shard_Range(step, time);
        return true;
    }

    bool Ensure_Reaxff_Eeq_Charge_Snapshot(std::size_t atom_count)
    {
        if (wrapper_writer_ == nullptr)
        {
            last_error_ = "VDS wrapper is not open";
            return false;
        }
        ModuleH5MappingWriter module_writer(wrapper_writer_.get());
        if (!module_writer.Ensure_Reaxff_Eeq_Charge_Snapshot(atom_count))
        {
            last_error_ = module_writer.Last_Error();
            return false;
        }
        return true;
    }

    bool Write_Reaxff_Eeq_Charge_Snapshot(const float* values,
                                          std::size_t atom_count)
    {
        if (wrapper_writer_ == nullptr)
        {
            last_error_ = "VDS wrapper is not open";
            return false;
        }
        ModuleH5MappingWriter module_writer(wrapper_writer_.get());
        if (!module_writer.Write_Reaxff_Eeq_Charge_Snapshot(values, atom_count))
        {
            last_error_ = module_writer.Last_Error();
            return false;
        }
        return true;
    }

    bool Write_Metadynamics_Diagnostic(const std::string& name,
                                       const std::string& component,
                                       const std::string& text)
    {
        if (wrapper_writer_ == nullptr)
        {
            last_error_ = "VDS wrapper is not open";
            return false;
        }
        ModuleH5MappingWriter module_writer(wrapper_writer_.get());
        if (!module_writer.Write_Metadynamics_Diagnostic(name, component, text))
        {
            last_error_ = module_writer.Last_Error();
            return false;
        }
        return true;
    }

    bool Write_Legacy_Sidecar_Paths(const std::vector<std::string>& keys,
                                    const std::vector<std::string>& paths)
    {
        if (wrapper_writer_ == nullptr)
        {
            last_error_ = "VDS wrapper is not open";
            return false;
        }
        if (!wrapper_writer_->Ensure_Group(path::sponge_files))
        {
            return false;
        }
        if (!wrapper_writer_->Ensure_Group(path::legacy_sidecars))
        {
            return false;
        }
        if (!wrapper_writer_->Write_String_Array(path::legacy_sidecar_keys,
                                                 keys))
        {
            return false;
        }
        return wrapper_writer_->Write_String_Array(path::legacy_sidecar_paths,
                                                   paths);
    }

    bool Finalize() { return Finalize_Internal(false); }

    bool Finalize_With_Repair() { return Finalize_Internal(true); }

    bool Finalize_Internal(bool allow_repair)
    {
        repair_applied_ = false;
        repaired_shard_count_ = 0;
        if (!Complete_Current_Shard(allow_repair))
        {
            if (wrapper_writer_ != nullptr)
            {
                wrapper_writer_->Mark_Failed(last_error_);
            }
            return false;
        }
        if (!Validate_Complete_Manifest(allow_repair))
        {
            if (wrapper_writer_ != nullptr)
            {
                wrapper_writer_->Mark_Failed(last_error_);
            }
            return false;
        }
        if (!Publish_Completed_Prefix(allow_repair, true))
        {
            if (wrapper_writer_ != nullptr)
            {
                wrapper_writer_->Mark_Failed(last_error_);
            }
            return false;
        }
        return true;
    }

    const std::vector<VdsShardManifestEntry>& Manifest() const
    {
        return manifest_;
    }

    std::size_t Total_Trajectory_Frame_Count() const
    {
        return total_trajectory_frame_count_;
    }

    std::size_t Total_Observable_Frame_Count() const
    {
        return total_observable_frame_count_;
    }

    std::string Last_Error() const { return last_error_; }

   private:
    bool Ensure_Current_Shard(const int64_t step, const double time)
    {
        return current_shard_writer_ != nullptr ||
               Rotate_To_New_Shard(step, time);
    }

    void Extend_Nonparticle_Shard_Range(const int64_t step, const double time)
    {
        if (current_manifest_entry_.frame_count != 0) return;
        current_manifest_entry_.step_end =
            std::max(current_manifest_entry_.step_end, step);
        current_manifest_entry_.time_end =
            std::max(current_manifest_entry_.time_end, time);
    }

    bool Rotate_To_New_Shard(const int64_t step, const double time)
    {
        const std::size_t manifest_size_before = manifest_.size();
        if (!Complete_Current_Shard(false))
        {
            return false;
        }
        if (manifest_.size() != manifest_size_before &&
            !Publish_Completed_Prefix(false, false))
        {
            return false;
        }
        const int64_t shard_index = static_cast<int64_t>(manifest_.size());
        const std::string shard_path = Shard_Path(shard_index);

        current_shard_backend_ = backend_factory_->Create_Backend();
        if (current_shard_backend_ == nullptr)
        {
            last_error_ = "failed to create shard backend";
            return false;
        }
        current_shard_writer_ =
            std::make_unique<TrajectoryH5Writer>(current_shard_backend_.get());

        SpongeH5OutputPlan::ResolvedOutputPlan shard_plan;
        shard_plan.trajectory.enabled = true;
        shard_plan.trajectory.vds = false;
        shard_plan.trajectory.path = shard_path;
        if (!current_shard_writer_->Open_Single_File(
                shard_plan, schema_version_, identity_uuid_, "vds_shard"))
        {
            last_error_ = current_shard_writer_->Last_Error();
            return false;
        }
        if ((!topology_hash_.empty() || !atom_order_hash_.empty()) &&
            !current_shard_writer_->Write_Topology_Compatibility(
                topology_hash_, atom_order_hash_))
        {
            last_error_ = current_shard_writer_->Last_Error();
            return false;
        }
        if (particle_layout_defined_ &&
            !current_shard_writer_->Define_Particle_Datasets(
                atom_count_, include_velocity_, include_force_))
        {
            last_error_ = current_shard_writer_->Last_Error();
            return false;
        }
        if (observable_layout_defined_ &&
            !current_shard_writer_->Define_Observable_Stream(
                observable_hdf5_names_, observable_original_names_))
        {
            last_error_ = current_shard_writer_->Last_Error();
            return false;
        }
        if (nhc_layout_defined_ &&
            !current_shard_writer_->Ensure_Nose_Hoover_Chain_Observables(
                nhc_chain_length_))
        {
            last_error_ = current_shard_writer_->Last_Error();
            return false;
        }
        if (sits_nk_layout_defined_ &&
            !current_shard_writer_->Ensure_Sits_Nk_Observable(sits_module_name_,
                                                              sits_k_count_))
        {
            last_error_ = current_shard_writer_->Last_Error();
            return false;
        }
        if (metadynamics_scalar_layout_defined_ &&
            !current_shard_writer_->Ensure_Metadynamics_Scalars())
        {
            last_error_ = current_shard_writer_->Last_Error();
            return false;
        }
        if (reaxff_layout_defined_ &&
            !current_shard_writer_->Ensure_Reaxff_Energy_Terms(reaxff_terms_))
        {
            last_error_ = current_shard_writer_->Last_Error();
            return false;
        }

        current_manifest_entry_ = {};
        current_manifest_entry_.index = shard_index;
        current_manifest_entry_.path = shard_path;
        current_manifest_entry_.frame_start =
            static_cast<int64_t>(total_trajectory_frame_count_);
        current_manifest_entry_.frame_count = 0;
        current_manifest_entry_.observable_frame_count = 0;
        current_manifest_entry_.nhc_frame_count = 0;
        current_manifest_entry_.sits_nk_frame_count = 0;
        current_manifest_entry_.metadynamics_scalar_frame_count = 0;
        current_manifest_entry_.reaxff_frame_count = 0;
        current_manifest_entry_.step_start = step;
        current_manifest_entry_.step_end = step;
        current_manifest_entry_.time_start = time;
        current_manifest_entry_.time_end = time;
        current_manifest_entry_.status = "open";
        current_shard_frame_count_ = 0;
        return true;
    }

    bool Complete_Current_Shard(bool allow_repair)
    {
        if (current_shard_writer_ == nullptr)
        {
            return true;
        }
        if (Manifest_Entry_Has_Stream_Frames(current_manifest_entry_))
        {
            current_manifest_entry_.status = "complete";
            if (!current_shard_writer_->Finalize())
            {
                last_error_ = current_shard_writer_->Last_Error();
                if (allow_repair)
                {
                    repair_applied_ = true;
                    repaired_shard_count_ += 1;
                    current_shard_writer_.reset();
                    current_shard_backend_.reset();
                    current_shard_frame_count_ = 0;
                    Recompute_Totals_From_Manifest();
                    return true;
                }
                return false;
            }
            current_shard_writer_.reset();
            current_shard_backend_.reset();
            std::error_code size_error;
            const auto size = std::filesystem::file_size(
                current_manifest_entry_.path, size_error);
            if (!size_error)
            {
                current_manifest_entry_.byte_size = static_cast<int64_t>(size);
            }
            manifest_.push_back(current_manifest_entry_);
        }
        current_shard_writer_.reset();
        current_shard_backend_.reset();
        current_shard_frame_count_ = 0;
        return true;
    }

    std::string Shard_Path(const int64_t index) const
    {
        char buffer[64];
        std::snprintf(buffer, sizeof(buffer), "segment_%06lld.spg.h5md",
                      static_cast<long long>(index));
        return shard_root_ + "/" + buffer;
    }

    bool Validate_Complete_Manifest(bool allow_repair)
    {
        int64_t expected_frame_start = 0;
        int64_t previous_index = -1;
        std::size_t valid_prefix_count = 0;
        for (const auto& entry : manifest_)
        {
            if (entry.status != "complete")
            {
                if (allow_repair)
                {
                    return Repair_Manifest_To_Prefix(valid_prefix_count);
                }
                last_error_ = "manifest contains incomplete shard";
                return false;
            }
            if (entry.index != previous_index + 1)
            {
                if (allow_repair)
                {
                    return Repair_Manifest_To_Prefix(valid_prefix_count);
                }
                last_error_ = "manifest shard indices are not contiguous";
                return false;
            }
            if (entry.frame_start != expected_frame_start)
            {
                if (allow_repair)
                {
                    return Repair_Manifest_To_Prefix(valid_prefix_count);
                }
                last_error_ = "manifest frame ranges are not contiguous";
                return false;
            }
            if (Manifest_Entry_Has_Negative_Stream_Count(entry) ||
                !Manifest_Entry_Has_Stream_Frames(entry))
            {
                if (allow_repair)
                {
                    return Repair_Manifest_To_Prefix(valid_prefix_count);
                }
                last_error_ =
                    "manifest shard must contain at least one stream frame";
                return false;
            }
            expected_frame_start += entry.frame_count;
            previous_index = entry.index;
            valid_prefix_count += 1;
        }
        return true;
    }

    bool Repair_Manifest_To_Prefix(std::size_t valid_prefix_count)
    {
        if (valid_prefix_count == manifest_.size())
        {
            return true;
        }
        repair_applied_ = true;
        repaired_shard_count_ += manifest_.size() - valid_prefix_count;
        manifest_.resize(valid_prefix_count);
        Recompute_Totals_From_Manifest();
        return true;
    }

    void Recompute_Totals_From_Manifest()
    {
        total_trajectory_frame_count_ = 0;
        total_observable_frame_count_ = 0;
        for (const auto& entry : manifest_)
        {
            total_trajectory_frame_count_ +=
                static_cast<std::size_t>(entry.frame_count);
            total_observable_frame_count_ +=
                static_cast<std::size_t>(entry.observable_frame_count);
        }
    }

    bool Write_Repair_Metadata(bool allow_repair)
    {
        if (wrapper_writer_ == nullptr)
        {
            last_error_ = "VDS wrapper is not open";
            return false;
        }
        if (!wrapper_writer_->Write_String(
                path::output_repair_policy,
                allow_repair ? "complete_prefix" : "strict"))
        {
            last_error_ = wrapper_writer_->Last_Error();
            return false;
        }
        if (!wrapper_writer_->Write_String(
                path::output_repair_status,
                repair_applied_ ? "applied" : "not_applied"))
        {
            last_error_ = wrapper_writer_->Last_Error();
            return false;
        }
        const int64_t repaired_count =
            static_cast<int64_t>(repaired_shard_count_);
        if (!wrapper_writer_->Create_Dataset({path::output_repaired_shard_count,
                                              DataType::int64,
                                              {{0}, {1}, {1}},
                                              true}))
        {
            last_error_ = wrapper_writer_->Last_Error();
            return false;
        }
        if (!wrapper_writer_->Append_Int64(path::output_repaired_shard_count,
                                           &repaired_count, 1))
        {
            last_error_ = wrapper_writer_->Last_Error();
            return false;
        }
        return true;
    }

    bool Write_Published_Output_Completion()
    {
        if (wrapper_writer_ == nullptr)
        {
            last_error_ = "VDS wrapper is not open";
            return false;
        }
        int64_t frame_count = 0;
        int64_t last_step = -1;
        double last_time = 0.0;
        if (manifest_.empty() ||
            published_completion_manifest_count_ == manifest_.size())
        {
            return true;
        }
        for (const auto& entry : manifest_)
        {
            frame_count += entry.frame_count;
            if (entry.frame_count > 0)
            {
                last_step = entry.step_end;
                last_time = entry.time_end;
            }
        }
        if (!wrapper_writer_->Write_Output_Completion(frame_count, last_step,
                                                      last_time))
        {
            last_error_ = wrapper_writer_->Last_Error();
            return false;
        }
        published_completion_manifest_count_ = manifest_.size();
        return true;
    }

    bool Write_Manifest_To_Wrapper()
    {
        if (wrapper_writer_ == nullptr)
        {
            last_error_ = "VDS wrapper is not open";
            return false;
        }
        if (!wrapper_writer_->Ensure_Group(path::shard_manifest))
        {
            return false;
        }
        if (!wrapper_writer_->Ensure_Group(path::shard_manifest_stream_counts))
        {
            return false;
        }
        if (manifest_.empty())
        {
            return true;
        }
        if (published_manifest_count_ > manifest_.size())
        {
            last_error_ = "cannot retract an already published VDS shard";
            return false;
        }

        std::vector<std::string> paths;
        std::vector<std::string> statuses;
        std::vector<int64_t> indices;
        std::vector<int64_t> frame_starts;
        std::vector<int64_t> frame_counts;
        std::vector<int64_t> byte_sizes;
        std::vector<int64_t> observable_frame_counts;
        std::vector<int64_t> nhc_frame_counts;
        std::vector<int64_t> sits_frame_counts;
        std::vector<int64_t> metadynamics_frame_counts;
        std::vector<int64_t> reaxff_frame_counts;
        std::vector<int64_t> step_starts;
        std::vector<int64_t> step_ends;
        std::vector<double> time_starts;
        std::vector<double> time_ends;
        for (const auto& entry : manifest_)
        {
            paths.push_back(Vds_Source_Path(entry.path));
            statuses.push_back(entry.status);
            indices.push_back(entry.index);
            frame_starts.push_back(entry.frame_start);
            frame_counts.push_back(entry.frame_count);
            byte_sizes.push_back(entry.byte_size);
            observable_frame_counts.push_back(entry.observable_frame_count);
            nhc_frame_counts.push_back(entry.nhc_frame_count);
            sits_frame_counts.push_back(entry.sits_nk_frame_count);
            metadynamics_frame_counts.push_back(
                entry.metadynamics_scalar_frame_count);
            reaxff_frame_counts.push_back(entry.reaxff_frame_count);
            step_starts.push_back(entry.step_start);
            step_ends.push_back(entry.step_end);
            time_starts.push_back(entry.time_start);
            time_ends.push_back(entry.time_end);
        }

        if (!wrapper_writer_->Write_String_Array(path::shard_manifest_path,
                                                 paths))
        {
            last_error_ = wrapper_writer_->Last_Error();
            return false;
        }
        if (!wrapper_writer_->Write_String_Array(path::shard_manifest_status,
                                                 statuses))
        {
            last_error_ = wrapper_writer_->Last_Error();
            return false;
        }

        const std::size_t append_start = published_manifest_count_;
        const auto write_i64 =
            [&](const char* dataset_path, const std::vector<int64_t>& values)
        {
            if (!wrapper_writer_->Create_Dataset(
                    {dataset_path, DataType::int64, {{0}, {0}, {1}}, true}))
            {
                last_error_ = wrapper_writer_->Last_Error();
                return false;
            }
            if (append_start >= values.size())
            {
                return true;
            }
            if (!wrapper_writer_->Append_Int64(dataset_path,
                                               values.data() + append_start,
                                               values.size() - append_start))
            {
                last_error_ = wrapper_writer_->Last_Error();
                return false;
            }
            return true;
        };
        const auto write_f64 =
            [&](const char* dataset_path, const std::vector<double>& values)
        {
            if (!wrapper_writer_->Create_Dataset(
                    {dataset_path, DataType::float64, {{0}, {0}, {1}}, true}))
            {
                last_error_ = wrapper_writer_->Last_Error();
                return false;
            }
            if (append_start >= values.size())
            {
                return true;
            }
            if (!wrapper_writer_->Append_Float64(dataset_path,
                                                 values.data() + append_start,
                                                 values.size() - append_start))
            {
                last_error_ = wrapper_writer_->Last_Error();
                return false;
            }
            return true;
        };

        if (!write_i64(path::shard_manifest_index, indices) ||
            !write_i64(path::shard_manifest_frame_start, frame_starts) ||
            !write_i64(path::shard_manifest_frame_count, frame_counts) ||
            !write_i64(path::shard_manifest_byte_size, byte_sizes) ||
            !write_i64(path::shard_manifest_particles_count, frame_counts) ||
            !write_i64(path::shard_manifest_observables_count,
                       observable_frame_counts) ||
            !write_i64(path::shard_manifest_nhc_count, nhc_frame_counts) ||
            !write_i64(path::shard_manifest_sits_count, sits_frame_counts) ||
            !write_i64(path::shard_manifest_metadynamics_count,
                       metadynamics_frame_counts) ||
            !write_i64(path::shard_manifest_reaxff_count,
                       reaxff_frame_counts) ||
            !write_i64(path::shard_manifest_step_start, step_starts) ||
            !write_i64(path::shard_manifest_step_end, step_ends) ||
            !write_f64(path::shard_manifest_time_start, time_starts) ||
            !write_f64(path::shard_manifest_time_end, time_ends))
        {
            return false;
        }
        published_manifest_count_ = manifest_.size();
        return true;
    }

    std::vector<VirtualDatasetSource> Make_Particle_Vds_Sources(
        const char* dataset_path,
        const std::vector<std::size_t>& trailing_dims) const
    {
        std::vector<VirtualDatasetSource> sources;
        sources.reserve(manifest_.size());
        for (const auto& entry : manifest_)
        {
            if (entry.frame_count <= 0) continue;
            VirtualDatasetSource source;
            source.file_path = Vds_Source_Path(entry.path);
            source.dataset_path = dataset_path;
            source.source_dims.push_back(
                static_cast<std::size_t>(entry.frame_count));
            source.virtual_start.push_back(
                static_cast<std::size_t>(entry.frame_start));
            for (const std::size_t dim : trailing_dims)
            {
                source.source_dims.push_back(dim);
                source.virtual_start.push_back(0);
            }
            sources.push_back(source);
        }
        return sources;
    }

    bool Materialize_Particle_Virtual_Datasets()
    {
        if (wrapper_writer_ == nullptr)
        {
            last_error_ = "VDS wrapper is not open";
            return false;
        }
        if (!particle_layout_defined_ || total_trajectory_frame_count_ == 0)
        {
            return true;
        }

        const std::size_t total_frames = total_trajectory_frame_count_;
        const auto write_vds =
            [&](const char* dataset_path, DataType type,
                const std::vector<std::size_t>& trailing_dims)
        {
            std::vector<std::size_t> dims;
            dims.push_back(total_frames);
            for (const std::size_t dim : trailing_dims)
            {
                dims.push_back(dim);
            }
            DatasetSpec spec;
            spec.path = dataset_path;
            spec.type = type;
            spec.shape.dims = dims;
            spec.shape.max_dims = dims;
            spec.shape.chunk_dims = dims;
            spec.appendable = false;
            return wrapper_writer_->Create_Virtual_Dataset(
                spec, Make_Particle_Vds_Sources(dataset_path, trailing_dims));
        };

        if (!write_vds(path::particles_all_step, DataType::int64, {}))
        {
            last_error_ = wrapper_writer_->Last_Error();
            return false;
        }
        if (!write_vds(path::particles_all_time, DataType::float64, {}))
        {
            last_error_ = wrapper_writer_->Last_Error();
            return false;
        }
        if (!wrapper_writer_->Set_String_Attribute(path::particles_all_time,
                                                   "unit", "ps"))
        {
            last_error_ = wrapper_writer_->Last_Error();
            return false;
        }
        if (!write_vds(path::position_value, DataType::float32,
                       {atom_count_, 3}))
        {
            last_error_ = wrapper_writer_->Last_Error();
            return false;
        }
        if (!wrapper_writer_->Set_String_Attribute(path::position_value, "unit",
                                                   "Angstrom"))
        {
            last_error_ = wrapper_writer_->Last_Error();
            return false;
        }
        if (!wrapper_writer_->Create_Hard_Link(path::particles_all_step,
                                               path::position_step))
        {
            last_error_ = wrapper_writer_->Last_Error();
            return false;
        }
        if (!wrapper_writer_->Create_Hard_Link(path::particles_all_time,
                                               path::position_time))
        {
            last_error_ = wrapper_writer_->Last_Error();
            return false;
        }
        if (!write_vds(path::box_edges_value, DataType::float32, {3, 3}))
        {
            last_error_ = wrapper_writer_->Last_Error();
            return false;
        }
        if (!wrapper_writer_->Set_String_Attribute(path::box_edges_value,
                                                   "unit", "Angstrom"))
        {
            last_error_ = wrapper_writer_->Last_Error();
            return false;
        }
        if (!wrapper_writer_->Create_Hard_Link(path::particles_all_step,
                                               path::box_edges_step))
        {
            last_error_ = wrapper_writer_->Last_Error();
            return false;
        }
        if (!wrapper_writer_->Create_Hard_Link(path::particles_all_time,
                                               path::box_edges_time))
        {
            last_error_ = wrapper_writer_->Last_Error();
            return false;
        }
        if (include_velocity_)
        {
            if (!write_vds(path::velocity_value, DataType::float32,
                           {atom_count_, 3}))
            {
                last_error_ = wrapper_writer_->Last_Error();
                return false;
            }
            if (!wrapper_writer_->Set_String_Attribute(path::velocity_value,
                                                       "unit", "Angstrom ps-1"))
            {
                last_error_ = wrapper_writer_->Last_Error();
                return false;
            }
            if (!wrapper_writer_->Create_Hard_Link(path::particles_all_step,
                                                   path::velocity_step))
            {
                last_error_ = wrapper_writer_->Last_Error();
                return false;
            }
            if (!wrapper_writer_->Create_Hard_Link(path::particles_all_time,
                                                   path::velocity_time))
            {
                last_error_ = wrapper_writer_->Last_Error();
                return false;
            }
        }
        if (include_force_)
        {
            if (!write_vds(path::force_value, DataType::float32,
                           {atom_count_, 3}))
            {
                last_error_ = wrapper_writer_->Last_Error();
                return false;
            }
            if (!wrapper_writer_->Set_String_Attribute(
                    path::force_value, "unit", "kcal mol-1 Angstrom-1"))
            {
                last_error_ = wrapper_writer_->Last_Error();
                return false;
            }
            if (!wrapper_writer_->Create_Hard_Link(path::particles_all_step,
                                                   path::force_step))
            {
                last_error_ = wrapper_writer_->Last_Error();
                return false;
            }
            if (!wrapper_writer_->Create_Hard_Link(path::particles_all_time,
                                                   path::force_time))
            {
                last_error_ = wrapper_writer_->Last_Error();
                return false;
            }
        }
        return true;
    }

    std::vector<VirtualDatasetSource> Make_Observable_Vds_Sources(
        const char* dataset_path) const
    {
        std::vector<VirtualDatasetSource> sources;
        sources.reserve(manifest_.size());
        std::size_t virtual_start = 0;
        for (const auto& entry : manifest_)
        {
            if (entry.observable_frame_count <= 0)
            {
                continue;
            }
            VirtualDatasetSource source;
            source.file_path = Vds_Source_Path(entry.path);
            source.dataset_path = dataset_path;
            source.source_dims.push_back(
                static_cast<std::size_t>(entry.observable_frame_count));
            source.virtual_start.push_back(virtual_start);
            virtual_start +=
                static_cast<std::size_t>(entry.observable_frame_count);
            sources.push_back(source);
        }
        return sources;
    }

    bool Materialize_Observable_Virtual_Datasets()
    {
        if (wrapper_writer_ == nullptr)
        {
            last_error_ = "VDS wrapper is not open";
            return false;
        }
        if (!observable_layout_defined_ || total_observable_frame_count_ == 0)
        {
            return true;
        }

        const std::size_t total_frames = total_observable_frame_count_;
        const auto write_vds =
            [&](const std::string& dataset_path, DataType type)
        {
            DatasetSpec spec;
            spec.path = dataset_path;
            spec.type = type;
            spec.shape.dims = {total_frames};
            spec.shape.max_dims = {total_frames};
            spec.shape.chunk_dims = {total_frames};
            spec.appendable = false;
            return wrapper_writer_->Create_Virtual_Dataset(
                spec, Make_Observable_Vds_Sources(dataset_path.c_str()));
        };

        if (!write_vds(path::observables_all_step, DataType::int64))
        {
            last_error_ = wrapper_writer_->Last_Error();
            return false;
        }
        if (!write_vds(path::observables_all_time, DataType::float64))
        {
            last_error_ = wrapper_writer_->Last_Error();
            return false;
        }
        for (const std::string& name : observable_hdf5_names_)
        {
            const std::string group = Observable_Root(name);
            const std::string value_path = Observable_Value_Path(name);
            if (!wrapper_writer_->Ensure_Group(group))
            {
                last_error_ = wrapper_writer_->Last_Error();
                return false;
            }
            if (!write_vds(value_path, DataType::float64))
            {
                last_error_ = wrapper_writer_->Last_Error();
                return false;
            }
            if (!wrapper_writer_->Create_Hard_Link(path::observables_all_step,
                                                   Observable_Step_Path(name)))
            {
                last_error_ = wrapper_writer_->Last_Error();
                return false;
            }
            if (!wrapper_writer_->Create_Hard_Link(path::observables_all_time,
                                                   Observable_Time_Path(name)))
            {
                last_error_ = wrapper_writer_->Last_Error();
                return false;
            }
        }
        if (!wrapper_writer_->Write_String_Array(
                path::mdout_columns_original_name, observable_original_names_))
        {
            last_error_ = wrapper_writer_->Last_Error();
            return false;
        }
        if (!wrapper_writer_->Write_String_Array(path::mdout_columns_hdf5_name,
                                                 observable_hdf5_names_))
        {
            last_error_ = wrapper_writer_->Last_Error();
            return false;
        }
        return true;
    }

    std::size_t Total_Module_Frame_Count(
        int64_t VdsShardManifestEntry::* frame_count_member) const
    {
        std::size_t total = 0;
        for (const auto& entry : manifest_)
        {
            total += static_cast<std::size_t>(entry.*frame_count_member);
        }
        return total;
    }

    std::vector<VirtualDatasetSource> Make_Module_Vds_Sources(
        const std::string& dataset_path,
        int64_t VdsShardManifestEntry::* frame_count_member,
        const std::vector<std::size_t>& trailing_dims) const
    {
        std::vector<VirtualDatasetSource> sources;
        sources.reserve(manifest_.size());
        std::size_t virtual_start = 0;
        for (const auto& entry : manifest_)
        {
            const int64_t frame_count = entry.*frame_count_member;
            if (frame_count <= 0)
            {
                continue;
            }
            VirtualDatasetSource source;
            source.file_path = Vds_Source_Path(entry.path);
            source.dataset_path = dataset_path;
            source.source_dims.push_back(static_cast<std::size_t>(frame_count));
            source.virtual_start.push_back(virtual_start);
            virtual_start += static_cast<std::size_t>(frame_count);
            for (const std::size_t dim : trailing_dims)
            {
                source.source_dims.push_back(dim);
                source.virtual_start.push_back(0);
            }
            sources.push_back(source);
        }
        return sources;
    }

    bool Create_Module_Vds(const std::string& dataset_path, DataType type,
                           int64_t VdsShardManifestEntry::* frame_count_member,
                           const std::vector<std::size_t>& trailing_dims)
    {
        const std::size_t total_frames =
            Total_Module_Frame_Count(frame_count_member);
        if (total_frames == 0)
        {
            return true;
        }
        std::vector<std::size_t> dims;
        dims.push_back(total_frames);
        for (const std::size_t dim : trailing_dims)
        {
            dims.push_back(dim);
        }
        DatasetSpec spec;
        spec.path = dataset_path;
        spec.type = type;
        spec.shape.dims = dims;
        spec.shape.max_dims = dims;
        spec.shape.chunk_dims = dims;
        spec.appendable = false;
        return wrapper_writer_->Create_Virtual_Dataset(
            spec, Make_Module_Vds_Sources(dataset_path, frame_count_member,
                                          trailing_dims));
    }

    bool Create_Module_Scalar_With_Axis(
        const std::string& group, const std::string& step_path,
        const std::string& time_path,
        int64_t VdsShardManifestEntry::* frame_count_member)
    {
        if (!wrapper_writer_->Ensure_Group(group))
        {
            last_error_ = wrapper_writer_->Last_Error();
            return false;
        }
        if (!Create_Module_Vds(Scalar_Observable_Value_Path(group),
                               DataType::float64, frame_count_member, {}))
        {
            last_error_ = wrapper_writer_->Last_Error();
            return false;
        }
        if (!wrapper_writer_->Create_Hard_Link(
                step_path, Scalar_Observable_Step_Path(group)))
        {
            last_error_ = wrapper_writer_->Last_Error();
            return false;
        }
        if (!wrapper_writer_->Create_Hard_Link(
                time_path, Scalar_Observable_Time_Path(group)))
        {
            last_error_ = wrapper_writer_->Last_Error();
            return false;
        }
        return true;
    }

    bool Materialize_Nhc_Virtual_Datasets()
    {
        if (!nhc_layout_defined_ ||
            Total_Module_Frame_Count(&VdsShardManifestEntry::nhc_frame_count) ==
                0)
        {
            return true;
        }
        if (!wrapper_writer_->Ensure_Group(module_path::nhc_root))
        {
            last_error_ = wrapper_writer_->Last_Error();
            return false;
        }
        if (!wrapper_writer_->Ensure_Group(Nose_Hoover_Chain_Coordinate_Root()))
        {
            last_error_ = wrapper_writer_->Last_Error();
            return false;
        }
        if (!wrapper_writer_->Ensure_Group(Nose_Hoover_Chain_Velocity_Root()))
        {
            last_error_ = wrapper_writer_->Last_Error();
            return false;
        }
        if (!Create_Module_Vds(module_path::nhc_step, DataType::int64,
                               &VdsShardManifestEntry::nhc_frame_count, {}))
        {
            last_error_ = wrapper_writer_->Last_Error();
            return false;
        }
        if (!Create_Module_Vds(module_path::nhc_time, DataType::float64,
                               &VdsShardManifestEntry::nhc_frame_count, {}))
        {
            last_error_ = wrapper_writer_->Last_Error();
            return false;
        }
        if (!Create_Module_Vds(
                module_path::nhc_coordinate_value, DataType::float32,
                &VdsShardManifestEntry::nhc_frame_count, {nhc_chain_length_}))
        {
            last_error_ = wrapper_writer_->Last_Error();
            return false;
        }
        if (!wrapper_writer_->Create_Hard_Link(
                module_path::nhc_step, module_path::nhc_coordinate_step))
        {
            last_error_ = wrapper_writer_->Last_Error();
            return false;
        }
        if (!wrapper_writer_->Create_Hard_Link(
                module_path::nhc_time, module_path::nhc_coordinate_time))
        {
            last_error_ = wrapper_writer_->Last_Error();
            return false;
        }
        if (!Create_Module_Vds(
                module_path::nhc_velocity_value, DataType::float32,
                &VdsShardManifestEntry::nhc_frame_count, {nhc_chain_length_}))
        {
            last_error_ = wrapper_writer_->Last_Error();
            return false;
        }
        if (!wrapper_writer_->Create_Hard_Link(module_path::nhc_step,
                                               module_path::nhc_velocity_step))
        {
            last_error_ = wrapper_writer_->Last_Error();
            return false;
        }
        if (!wrapper_writer_->Create_Hard_Link(module_path::nhc_time,
                                               module_path::nhc_velocity_time))
        {
            last_error_ = wrapper_writer_->Last_Error();
            return false;
        }
        return true;
    }

    bool Materialize_Sits_Nk_Virtual_Datasets()
    {
        if (!sits_nk_layout_defined_ ||
            Total_Module_Frame_Count(
                &VdsShardManifestEntry::sits_nk_frame_count) == 0)
        {
            return true;
        }
        const std::string root = Sits_Module_Root(sits_module_name_);
        const std::string nk_root = Sits_Nk_Root(sits_module_name_);
        if (!wrapper_writer_->Ensure_Group(module_path::sits_root) ||
            !wrapper_writer_->Ensure_Group(root) ||
            !wrapper_writer_->Ensure_Group(nk_root))
        {
            last_error_ = wrapper_writer_->Last_Error();
            return false;
        }
        if (!Create_Module_Vds(Sits_Nk_Step_Path(sits_module_name_),
                               DataType::int64,
                               &VdsShardManifestEntry::sits_nk_frame_count, {}))
        {
            last_error_ = wrapper_writer_->Last_Error();
            return false;
        }
        if (!Create_Module_Vds(Sits_Nk_Time_Path(sits_module_name_),
                               DataType::float64,
                               &VdsShardManifestEntry::sits_nk_frame_count, {}))
        {
            last_error_ = wrapper_writer_->Last_Error();
            return false;
        }
        if (!Create_Module_Vds(
                Sits_Nk_Value_Path(sits_module_name_), DataType::float32,
                &VdsShardManifestEntry::sits_nk_frame_count, {sits_k_count_}))
        {
            last_error_ = wrapper_writer_->Last_Error();
            return false;
        }
        return true;
    }

    bool Materialize_Metadynamics_Virtual_Datasets()
    {
        if (!metadynamics_scalar_layout_defined_ ||
            Total_Module_Frame_Count(
                &VdsShardManifestEntry::metadynamics_scalar_frame_count) == 0)
        {
            return true;
        }
        if (!wrapper_writer_->Ensure_Group(module_path::metad_root))
        {
            last_error_ = wrapper_writer_->Last_Error();
            return false;
        }
        if (!Create_Module_Vds(
                module_path::metad_step, DataType::int64,
                &VdsShardManifestEntry::metadynamics_scalar_frame_count, {}))
        {
            last_error_ = wrapper_writer_->Last_Error();
            return false;
        }
        if (!Create_Module_Vds(
                module_path::metad_time, DataType::float64,
                &VdsShardManifestEntry::metadynamics_scalar_frame_count, {}))
        {
            last_error_ = wrapper_writer_->Last_Error();
            return false;
        }
        return Create_Module_Scalar_With_Axis(
                   Metadynamics_Scalar_Root("meta"), module_path::metad_step,
                   module_path::metad_time,
                   &VdsShardManifestEntry::metadynamics_scalar_frame_count) &&
               Create_Module_Scalar_With_Axis(
                   Metadynamics_Scalar_Root("rbias"), module_path::metad_step,
                   module_path::metad_time,
                   &VdsShardManifestEntry::metadynamics_scalar_frame_count) &&
               Create_Module_Scalar_With_Axis(
                   Metadynamics_Scalar_Root("rct"), module_path::metad_step,
                   module_path::metad_time,
                   &VdsShardManifestEntry::metadynamics_scalar_frame_count);
    }

    bool Materialize_Reaxff_Virtual_Datasets()
    {
        if (!reaxff_layout_defined_ ||
            Total_Module_Frame_Count(
                &VdsShardManifestEntry::reaxff_frame_count) == 0)
        {
            return true;
        }
        if (!wrapper_writer_->Ensure_Group(module_path::reaxff_root))
        {
            last_error_ = wrapper_writer_->Last_Error();
            return false;
        }
        if (!Create_Module_Vds(module_path::reaxff_step, DataType::int64,
                               &VdsShardManifestEntry::reaxff_frame_count, {}))
        {
            last_error_ = wrapper_writer_->Last_Error();
            return false;
        }
        if (!Create_Module_Vds(module_path::reaxff_time, DataType::float64,
                               &VdsShardManifestEntry::reaxff_frame_count, {}))
        {
            last_error_ = wrapper_writer_->Last_Error();
            return false;
        }
        for (const std::string& term : reaxff_terms_)
        {
            if (!Create_Module_Scalar_With_Axis(
                    Reaxff_Term_Root(term), module_path::reaxff_step,
                    module_path::reaxff_time,
                    &VdsShardManifestEntry::reaxff_frame_count))
            {
                return false;
            }
        }
        return true;
    }

    bool Materialize_Module_Virtual_Datasets()
    {
        if (wrapper_writer_ == nullptr)
        {
            last_error_ = "VDS wrapper is not open";
            return false;
        }
        return Materialize_Nhc_Virtual_Datasets() &&
               Materialize_Sits_Nk_Virtual_Datasets() &&
               Materialize_Metadynamics_Virtual_Datasets() &&
               Materialize_Reaxff_Virtual_Datasets();
    }

    bool Publish_Completed_Prefix(bool allow_repair, bool finalize)
    {
        if (wrapper_writer_ == nullptr)
        {
            last_error_ = "VDS wrapper is not open";
            return false;
        }
        if (!Materialize_Particle_Virtual_Datasets() ||
            !Materialize_Observable_Virtual_Datasets() ||
            !Materialize_Module_Virtual_Datasets() ||
            !Write_Manifest_To_Wrapper() ||
            !Write_Published_Output_Completion())
        {
            return false;
        }
        if (finalize && !Write_Repair_Metadata(allow_repair))
        {
            return false;
        }
        const std::string status =
            finalize ? "particle, observable, and module virtual datasets "
                       "materialized"
                     : "complete shard prefix published";
        if (!wrapper_writer_->Write_String(path::output_vds_status, status))
        {
            last_error_ = wrapper_writer_->Last_Error();
            return false;
        }
        if (!Write_Published_Stream_Completions()) return false;
        publication_epoch_ += 1;
        if (finalize)
        {
            if (!wrapper_writer_->Finalize(publication_epoch_))
            {
                last_error_ = wrapper_writer_->Last_Error();
                return false;
            }
            if (!wrapper_writer_->Publish_Snapshot(wrapper_path_))
            {
                last_error_ = wrapper_writer_->Last_Error();
                return false;
            }
            return true;
        }
        if (!wrapper_writer_->Write_Publication_Epoch(publication_epoch_))
        {
            last_error_ = wrapper_writer_->Last_Error();
            return false;
        }
        if (!wrapper_writer_->Flush())
        {
            last_error_ = wrapper_writer_->Last_Error();
            return false;
        }
        // A long-lived Windows reader may prevent replacement of the published
        // wrapper. Keep the last complete prefix and retry on the next publish.
        if (!wrapper_writer_->Publish_Snapshot(wrapper_path_, true))
        {
            last_error_ = wrapper_writer_->Last_Error();
            return false;
        }
        return true;
    }

    bool Write_Published_Stream_Completions()
    {
        if (wrapper_writer_ == nullptr)
        {
            last_error_ = "VDS wrapper is not open";
            return false;
        }
        const auto total = [&](int64_t VdsShardManifestEntry::* member)
        {
            int64_t count = 0;
            for (const auto& entry : manifest_) count += entry.*member;
            return count;
        };
        const auto write = [&](const char* stream_name, int64_t count,
                               const char* logical_kind,
                               const std::string& step_path,
                               const std::string& time_path,
                               const std::vector<std::string>& value_paths,
                               bool experimental = false)
        {
            if (!wrapper_writer_->Write_Output_Stream_Descriptor(
                    stream_name, logical_kind, step_path, time_path,
                    value_paths, experimental) ||
                !wrapper_writer_->Write_Output_Stream_Completion(stream_name,
                                                                 count))
            {
                last_error_ = wrapper_writer_->Last_Error();
                return false;
            }
            return true;
        };
        std::vector<std::string> particle_values = {path::position_value,
                                                    path::box_edges_value};
        if (include_velocity_) particle_values.push_back(path::velocity_value);
        if (include_force_) particle_values.push_back(path::force_value);
        if (!write("particles", total(&VdsShardManifestEntry::frame_count),
                   "trajectory_frames", path::particles_all_step,
                   path::particles_all_time, particle_values))
        {
            return false;
        }
        std::vector<std::string> observable_values;
        for (const auto& name : observable_hdf5_names_)
        {
            observable_values.push_back(Observable_Value_Path(name));
        }
        if (observable_layout_defined_ &&
            !write("observables",
                   total(&VdsShardManifestEntry::observable_frame_count),
                   "thermo_frames", path::observables_all_step,
                   path::observables_all_time, observable_values))
        {
            return false;
        }
        if (nhc_layout_defined_ &&
            !write("nose_hoover_chain",
                   total(&VdsShardManifestEntry::nhc_frame_count),
                   "module_frames", module_path::nhc_step,
                   module_path::nhc_time,
                   {module_path::nhc_coordinate_value,
                    module_path::nhc_velocity_value}))
        {
            return false;
        }
        if (sits_nk_layout_defined_ &&
            !write("sits", total(&VdsShardManifestEntry::sits_nk_frame_count),
                   "module_frames", Sits_Nk_Step_Path(sits_module_name_),
                   Sits_Nk_Time_Path(sits_module_name_),
                   {Sits_Nk_Value_Path(sits_module_name_)}))
        {
            return false;
        }
        if (metadynamics_scalar_layout_defined_ &&
            !write(
                "metadynamics",
                total(&VdsShardManifestEntry::metadynamics_scalar_frame_count),
                "module_frames", module_path::metad_step,
                module_path::metad_time,
                {Metadynamics_Scalar_Value_Path("meta"),
                 Metadynamics_Scalar_Value_Path("rbias"),
                 Metadynamics_Scalar_Value_Path("rct")}))
        {
            return false;
        }
        std::vector<std::string> reaxff_values;
        for (const auto& term : reaxff_terms_)
        {
            reaxff_values.push_back(Reaxff_Term_Value_Path(term));
        }
        return !reaxff_layout_defined_ ||
               write("reaxff",
                     total(&VdsShardManifestEntry::reaxff_frame_count),
                     "module_frames", module_path::reaxff_step,
                     module_path::reaxff_time, reaxff_values);
    }

    std::string Vds_Source_Path(const std::string& shard_path) const
    {
        const std::filesystem::path wrapper(wrapper_path_);
        const std::filesystem::path wrapper_parent = wrapper.parent_path();
        if (wrapper_parent.empty())
        {
            return shard_path;
        }
        std::error_code error;
        const std::filesystem::path relative = std::filesystem::relative(
            std::filesystem::path(shard_path), wrapper_parent, error);
        if (error || relative.empty())
        {
            return shard_path;
        }
        return relative.generic_string();
    }

    WriterBackendFactory* backend_factory_ = nullptr;
    SpongeH5OutputPlan::ResolvedOutputPlan plan_;
    std::string schema_version_;
    int chunk_size_ = SpongeH5OutputContract::kDefaultTrajectoryChunkSize;
    std::string shard_root_;
    std::string wrapper_path_;
    std::string identity_uuid_;
    std::string topology_hash_;
    std::string atom_order_hash_;

    std::unique_ptr<WriterBackend> wrapper_backend_;
    std::unique_ptr<H5MDWriter> wrapper_writer_;
    std::unique_ptr<WriterBackend> current_shard_backend_;
    std::unique_ptr<TrajectoryH5Writer> current_shard_writer_;

    std::size_t atom_count_ = 0;
    bool include_velocity_ = false;
    bool include_force_ = false;
    bool particle_layout_defined_ = false;
    bool observable_layout_defined_ = false;
    bool nhc_layout_defined_ = false;
    std::size_t nhc_chain_length_ = 0;
    bool sits_nk_layout_defined_ = false;
    std::string sits_module_name_;
    std::size_t sits_k_count_ = 0;
    bool metadynamics_scalar_layout_defined_ = false;
    bool reaxff_layout_defined_ = false;
    std::vector<std::string> reaxff_terms_;
    std::vector<std::string> observable_hdf5_names_;
    std::vector<std::string> observable_original_names_;
    VdsShardManifestEntry current_manifest_entry_;
    std::vector<VdsShardManifestEntry> manifest_;
    int current_shard_frame_count_ = 0;
    std::size_t total_trajectory_frame_count_ = 0;
    std::size_t total_observable_frame_count_ = 0;
    bool repair_applied_ = false;
    std::size_t repaired_shard_count_ = 0;
    std::size_t published_manifest_count_ = 0;
    std::size_t published_completion_manifest_count_ = 0;
    int64_t publication_epoch_ = 0;
    std::string last_error_;
};
}  // namespace SpongeH5MD

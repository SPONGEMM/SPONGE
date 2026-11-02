#include "h5_bundle_test_common.hpp"
#include "utils/h5md/vds_trajectory_h5_writer.hpp"

using namespace SpongeH5Test;
using namespace SpongeH5MD;

static SpongeH5OutputPlan::ResolvedOutputPlan Make_Vds_Plan()
{
    SpongeH5OutputPlan::ResolvedOutputPlan plan;
    plan.trajectory.enabled = true;
    plan.trajectory.path = "/tmp/sponge_h5_vds_case/prod.spg.h5md";
    plan.trajectory.vds = true;
    plan.trajectory.chunk_size = 2;
    plan.trajectory.derived_shard_root =
        "/tmp/sponge_h5_vds_case/prod.spg.shards";
    return plan;
}

static void Append_Full_Frame(VdsTrajectoryH5Writer& writer, int64_t step)
{
    float position[3] = {static_cast<float>(step), 0.0f, 0.0f};
    float velocity[3] = {1.0f, 2.0f, 3.0f};
    float box[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    float nhc[2] = {0.1f, 0.2f};
    float sits[3] = {1.0f, 2.0f, 3.0f};
    const double time = static_cast<double>(step) * 0.01;

    REQUIRE_TRUE(
        writer.Append_Particle_Frame(step, time, position, box, velocity));
    REQUIRE_TRUE(writer.Append_Observable_Frame(
        step, time, {{"temperature", 300.0 + step}}));
    REQUIRE_TRUE(
        writer.Append_Nose_Hoover_Chain_Frame(step, time, nhc, nhc, 2));
    REQUIRE_TRUE(writer.Append_Sits_Nk_Frame(step, time, "sits_a", sits, 3));
    REQUIRE_TRUE(
        writer.Append_Metadynamics_Scalar_Frame(step, time, 1.0, 2.0, 3.0));
    REQUIRE_TRUE(writer.Append_Reaxff_Frame(step, time,
                                            {{"bond", 1.25}, {"angle", 2.25}}));
}

static void Require_Dataset_Spec(
    const BackendLog& log, const std::string& dataset_path,
    DataType expected_type, const std::vector<std::size_t>& expected_dims,
    const std::vector<std::size_t>& expected_max_dims,
    const std::vector<std::size_t>& expected_chunk_dims,
    bool expected_appendable)
{
    const auto iter = log.datasets.find(dataset_path);
    REQUIRE_TRUE(iter != log.datasets.end());
    REQUIRE_EQ(iter->second.type, expected_type);
    REQUIRE_EQ(iter->second.shape.dims, expected_dims);
    REQUIRE_EQ(iter->second.shape.max_dims, expected_max_dims);
    REQUIRE_EQ(iter->second.shape.chunk_dims, expected_chunk_dims);
    REQUIRE_EQ(iter->second.appendable, expected_appendable);
}

static void Test_Vds_Wrapper_And_Module_Virtual_Datasets()
{
    MockBackendFactory factory;
    VdsTrajectoryH5Writer writer(&factory);
    auto plan = Make_Vds_Plan();

    REQUIRE_TRUE(writer.Open(plan, "test"));
    REQUIRE_TRUE(writer.Write_Topology_Compatibility("top-hash", "order-hash"));
    REQUIRE_TRUE(writer.Define_Particle_Datasets(1, true, false));
    REQUIRE_TRUE(writer.Define_Observable_Stream({"temperature"}, {"TEMP"}));
    REQUIRE_TRUE(writer.Ensure_Nose_Hoover_Chain_Observables(2));
    REQUIRE_TRUE(writer.Ensure_Sits_Nk_Observable("sits_a", 3));
    REQUIRE_TRUE(writer.Ensure_Metadynamics_Scalars());
    REQUIRE_TRUE(writer.Ensure_Reaxff_Energy_Terms({"bond", "angle"}));
    REQUIRE_TRUE(
        writer.Write_Metadynamics_Diagnostic("meta0", "hills", "HILLS"));
    REQUIRE_TRUE(writer.Write_Legacy_Sidecar_Paths(
        {"crd", "mdout"}, {"legacy.crd", "legacy.out"}));

    Append_Full_Frame(writer, 1);
    Append_Full_Frame(writer, 2);
    Append_Full_Frame(writer, 3);

    REQUIRE_EQ(writer.Manifest().size(), static_cast<std::size_t>(1));
    const auto& live_wrapper = *factory.logs[0];
    REQUIRE_EQ(live_wrapper.datasets.at(path::position_value).shape.dims,
               std::vector<std::size_t>({2, 1, 3}));
    REQUIRE_EQ(live_wrapper.virtual_datasets.at(path::position_value).size(),
               static_cast<std::size_t>(1));
    REQUIRE_EQ(live_wrapper.string_arrays.at(path::shard_manifest_path).size(),
               static_cast<std::size_t>(1));
    REQUIRE_EQ(live_wrapper.strings.at(path::output_vds_status),
               std::string("complete shard prefix published"));
    REQUIRE_EQ(live_wrapper.append_counts.at(path::shard_manifest_index),
               static_cast<int64_t>(1));

    REQUIRE_TRUE(writer.Finalize());
    REQUIRE_EQ(writer.Manifest().size(), static_cast<std::size_t>(2));
    REQUIRE_EQ(writer.Manifest()[0].frame_count, 2);
    REQUIRE_EQ(writer.Manifest()[1].frame_count, 1);
    REQUIRE_EQ(writer.Manifest()[0].observable_frame_count, 2);
    REQUIRE_EQ(writer.Manifest()[1].observable_frame_count, 1);
    REQUIRE_EQ(writer.Manifest()[0].nhc_frame_count, 2);
    REQUIRE_EQ(writer.Manifest()[1].nhc_frame_count, 1);
    REQUIRE_EQ(writer.Manifest()[0].sits_nk_frame_count, 2);
    REQUIRE_EQ(writer.Manifest()[1].sits_nk_frame_count, 1);
    REQUIRE_EQ(writer.Manifest()[0].metadynamics_scalar_frame_count, 2);
    REQUIRE_EQ(writer.Manifest()[1].metadynamics_scalar_frame_count, 1);
    REQUIRE_EQ(writer.Manifest()[0].reaxff_frame_count, 2);
    REQUIRE_EQ(writer.Manifest()[1].reaxff_frame_count, 1);
    REQUIRE_EQ(writer.Total_Trajectory_Frame_Count(),
               static_cast<std::size_t>(3));
    REQUIRE_EQ(writer.Total_Observable_Frame_Count(),
               static_cast<std::size_t>(3));

    REQUIRE_TRUE(factory.logs.size() >= 3);
    const auto& wrapper = *factory.logs[0];
    REQUIRE_TRUE(wrapper.groups.count(path::shard_manifest) != 0);
    REQUIRE_EQ(wrapper.strings.at(path::sponge_schema_name),
               std::string("sponge.output.h5md"));
    REQUIRE_EQ(wrapper.strings.at(path::sponge_schema_version),
               std::string("test"));
    REQUIRE_EQ(wrapper.strings.at(path::sponge_topology_hash),
               std::string("top-hash"));
    REQUIRE_EQ(wrapper.strings.at(path::sponge_atom_order_hash),
               std::string("order-hash"));
    REQUIRE_TRUE(wrapper.virtual_datasets.count(path::position_value) != 0);
    REQUIRE_TRUE(wrapper.virtual_datasets.count(path::velocity_value) != 0);
    REQUIRE_TRUE(wrapper.virtual_datasets.count(path::force_value) == 0);
    REQUIRE_TRUE(wrapper.virtual_datasets.count(path::particles_all_step) != 0);
    REQUIRE_TRUE(wrapper.virtual_datasets.count(path::particles_all_time) != 0);
    REQUIRE_TRUE(wrapper.virtual_datasets.count(path::box_edges_value) != 0);
    REQUIRE_TRUE(wrapper.virtual_datasets.count(path::observables_all_step) !=
                 0);
    REQUIRE_TRUE(wrapper.virtual_datasets.count(path::observables_all_time) !=
                 0);
    REQUIRE_TRUE(wrapper.virtual_datasets.count(
                     Observable_Value_Path("temperature")) != 0);
    REQUIRE_TRUE(wrapper.groups.count(Nose_Hoover_Chain_Coordinate_Root()) !=
                 0);
    REQUIRE_TRUE(wrapper.groups.count(Nose_Hoover_Chain_Velocity_Root()) != 0);
    REQUIRE_TRUE(wrapper.groups.count(Sits_Module_Root("sits_a")) != 0);
    REQUIRE_TRUE(wrapper.virtual_datasets.count(module_path::nhc_step) != 0);
    REQUIRE_TRUE(wrapper.virtual_datasets.count(module_path::nhc_time) != 0);
    REQUIRE_TRUE(
        wrapper.virtual_datasets.count(module_path::nhc_coordinate_value) != 0);
    REQUIRE_TRUE(
        wrapper.virtual_datasets.count(module_path::nhc_velocity_value) != 0);
    REQUIRE_TRUE(wrapper.virtual_datasets.count(Sits_Nk_Step_Path("sits_a")) !=
                 0);
    REQUIRE_TRUE(wrapper.virtual_datasets.count(Sits_Nk_Time_Path("sits_a")) !=
                 0);
    REQUIRE_TRUE(wrapper.virtual_datasets.count(Sits_Nk_Value_Path("sits_a")) !=
                 0);
    REQUIRE_TRUE(wrapper.virtual_datasets.count(module_path::metad_step) != 0);
    REQUIRE_TRUE(wrapper.virtual_datasets.count(module_path::metad_time) != 0);
    REQUIRE_TRUE(wrapper.virtual_datasets.count(
                     Metadynamics_Scalar_Value_Path("meta")) != 0);
    REQUIRE_TRUE(wrapper.virtual_datasets.count(
                     Metadynamics_Scalar_Value_Path("rbias")) != 0);
    REQUIRE_TRUE(wrapper.virtual_datasets.count(
                     Metadynamics_Scalar_Value_Path("rct")) != 0);
    REQUIRE_TRUE(wrapper.virtual_datasets.count(module_path::reaxff_step) != 0);
    REQUIRE_TRUE(wrapper.virtual_datasets.count(module_path::reaxff_time) != 0);
    REQUIRE_TRUE(
        wrapper.virtual_datasets.count(Reaxff_Term_Value_Path("bond")) != 0);
    REQUIRE_TRUE(
        wrapper.virtual_datasets.count(Reaxff_Term_Value_Path("angle")) != 0);

    Require_Dataset_Spec(wrapper, path::particles_all_step, DataType::int64,
                         {3}, {3}, {3}, false);
    Require_Dataset_Spec(wrapper, path::particles_all_time, DataType::float64,
                         {3}, {3}, {3}, false);
    Require_Dataset_Spec(wrapper, path::position_value, DataType::float32,
                         {3, 1, 3}, {3, 1, 3}, {3, 1, 3}, false);
    Require_Dataset_Spec(wrapper, path::velocity_value, DataType::float32,
                         {3, 1, 3}, {3, 1, 3}, {3, 1, 3}, false);
    Require_Dataset_Spec(wrapper, path::box_edges_value, DataType::float32,
                         {3, 3, 3}, {3, 3, 3}, {3, 3, 3}, false);
    Require_Dataset_Spec(wrapper, path::observables_all_step, DataType::int64,
                         {3}, {3}, {3}, false);
    Require_Dataset_Spec(wrapper, path::observables_all_time, DataType::float64,
                         {3}, {3}, {3}, false);
    Require_Dataset_Spec(wrapper, Observable_Value_Path("temperature"),
                         DataType::float64, {3}, {3}, {3}, false);
    Require_Dataset_Spec(wrapper, module_path::nhc_step, DataType::int64, {3},
                         {3}, {3}, false);
    Require_Dataset_Spec(wrapper, module_path::nhc_time, DataType::float64, {3},
                         {3}, {3}, false);
    Require_Dataset_Spec(wrapper, module_path::nhc_coordinate_value,
                         DataType::float32, {3, 2}, {3, 2}, {3, 2}, false);
    Require_Dataset_Spec(wrapper, module_path::nhc_velocity_value,
                         DataType::float32, {3, 2}, {3, 2}, {3, 2}, false);
    Require_Dataset_Spec(wrapper, Sits_Nk_Step_Path("sits_a"), DataType::int64,
                         {3}, {3}, {3}, false);
    Require_Dataset_Spec(wrapper, Sits_Nk_Time_Path("sits_a"),
                         DataType::float64, {3}, {3}, {3}, false);
    Require_Dataset_Spec(wrapper, Sits_Nk_Value_Path("sits_a"),
                         DataType::float32, {3, 3}, {3, 3}, {3, 3}, false);
    Require_Dataset_Spec(wrapper, module_path::metad_step, DataType::int64, {3},
                         {3}, {3}, false);
    Require_Dataset_Spec(wrapper, module_path::metad_time, DataType::float64,
                         {3}, {3}, {3}, false);
    Require_Dataset_Spec(wrapper, Metadynamics_Scalar_Value_Path("meta"),
                         DataType::float64, {3}, {3}, {3}, false);
    Require_Dataset_Spec(wrapper, Metadynamics_Scalar_Value_Path("rbias"),
                         DataType::float64, {3}, {3}, {3}, false);
    Require_Dataset_Spec(wrapper, Metadynamics_Scalar_Value_Path("rct"),
                         DataType::float64, {3}, {3}, {3}, false);
    Require_Dataset_Spec(wrapper, module_path::reaxff_step, DataType::int64,
                         {3}, {3}, {3}, false);
    Require_Dataset_Spec(wrapper, module_path::reaxff_time, DataType::float64,
                         {3}, {3}, {3}, false);
    Require_Dataset_Spec(wrapper, Reaxff_Term_Value_Path("bond"),
                         DataType::float64, {3}, {3}, {3}, false);
    Require_Dataset_Spec(wrapper, Reaxff_Term_Value_Path("angle"),
                         DataType::float64, {3}, {3}, {3}, false);

    const auto& position_sources =
        wrapper.virtual_datasets.at(path::position_value);
    REQUIRE_EQ(position_sources.size(), static_cast<std::size_t>(2));
    REQUIRE_EQ(position_sources[0].file_path,
               std::string("prod.spg.shards/segment_000000.spg.h5md"));
    REQUIRE_EQ(position_sources[1].file_path,
               std::string("prod.spg.shards/segment_000001.spg.h5md"));
    REQUIRE_EQ(position_sources[0].dataset_path,
               std::string(path::position_value));
    REQUIRE_EQ(position_sources[1].dataset_path,
               std::string(path::position_value));
    REQUIRE_EQ(position_sources[0].source_dims[0], static_cast<std::size_t>(2));
    REQUIRE_EQ(position_sources[0].source_dims[1], static_cast<std::size_t>(1));
    REQUIRE_EQ(position_sources[0].source_dims[2], static_cast<std::size_t>(3));
    REQUIRE_EQ(position_sources[1].virtual_start[0],
               static_cast<std::size_t>(2));
    REQUIRE_EQ(position_sources[1].source_dims[0], static_cast<std::size_t>(1));
    REQUIRE_TRUE(Has_Hard_Link(*factory.logs[0], path::particles_all_step,
                               path::position_step));
    REQUIRE_TRUE(Has_Hard_Link(*factory.logs[0], path::particles_all_time,
                               path::position_time));
    REQUIRE_TRUE(Has_Hard_Link(*factory.logs[0], path::particles_all_step,
                               path::velocity_step));
    REQUIRE_TRUE(Has_Hard_Link(*factory.logs[0], path::particles_all_time,
                               path::velocity_time));
    REQUIRE_TRUE(!Has_Hard_Link(*factory.logs[0], path::particles_all_step,
                                path::force_step));
    REQUIRE_TRUE(!Has_Hard_Link(*factory.logs[0], path::particles_all_time,
                                path::force_time));
    REQUIRE_TRUE(Has_Hard_Link(*factory.logs[0], path::particles_all_step,
                               path::box_edges_step));
    REQUIRE_TRUE(Has_Hard_Link(*factory.logs[0], path::particles_all_time,
                               path::box_edges_time));

    const auto& observable_sources =
        wrapper.virtual_datasets.at(Observable_Value_Path("temperature"));
    REQUIRE_EQ(observable_sources.size(), static_cast<std::size_t>(2));
    REQUIRE_EQ(observable_sources[0].file_path,
               std::string("prod.spg.shards/segment_000000.spg.h5md"));
    REQUIRE_EQ(observable_sources[1].file_path,
               std::string("prod.spg.shards/segment_000001.spg.h5md"));
    REQUIRE_EQ(observable_sources[0].dataset_path,
               Observable_Value_Path("temperature"));
    REQUIRE_EQ(observable_sources[1].dataset_path,
               Observable_Value_Path("temperature"));
    REQUIRE_EQ(observable_sources[0].source_dims[0],
               static_cast<std::size_t>(2));
    REQUIRE_EQ(observable_sources[1].virtual_start[0],
               static_cast<std::size_t>(2));
    REQUIRE_TRUE(Has_Hard_Link(*factory.logs[0], path::observables_all_step,
                               Observable_Step_Path("temperature")));
    REQUIRE_TRUE(Has_Hard_Link(*factory.logs[0], path::observables_all_time,
                               Observable_Time_Path("temperature")));
    REQUIRE_EQ(wrapper.string_arrays.at(path::mdout_columns_original_name)[0],
               std::string("TEMP"));
    REQUIRE_EQ(wrapper.string_arrays.at(path::mdout_columns_hdf5_name)[0],
               std::string("temperature"));

    const auto& nhc_sources =
        wrapper.virtual_datasets.at(module_path::nhc_coordinate_value);
    REQUIRE_EQ(nhc_sources.size(), static_cast<std::size_t>(2));
    REQUIRE_EQ(nhc_sources[0].file_path,
               std::string("prod.spg.shards/segment_000000.spg.h5md"));
    REQUIRE_EQ(nhc_sources[1].file_path,
               std::string("prod.spg.shards/segment_000001.spg.h5md"));
    REQUIRE_EQ(nhc_sources[0].dataset_path,
               std::string(module_path::nhc_coordinate_value));
    REQUIRE_EQ(nhc_sources[0].source_dims[0], static_cast<std::size_t>(2));
    REQUIRE_EQ(nhc_sources[0].source_dims[1], static_cast<std::size_t>(2));
    REQUIRE_EQ(nhc_sources[1].virtual_start[0], static_cast<std::size_t>(2));
    REQUIRE_TRUE(Has_Hard_Link(*factory.logs[0], module_path::nhc_step,
                               module_path::nhc_coordinate_step));
    REQUIRE_TRUE(Has_Hard_Link(*factory.logs[0], module_path::nhc_time,
                               module_path::nhc_coordinate_time));
    REQUIRE_TRUE(Has_Hard_Link(*factory.logs[0], module_path::nhc_step,
                               module_path::nhc_velocity_step));
    REQUIRE_TRUE(Has_Hard_Link(*factory.logs[0], module_path::nhc_time,
                               module_path::nhc_velocity_time));

    const auto& sits_sources =
        wrapper.virtual_datasets.at(Sits_Nk_Value_Path("sits_a"));
    REQUIRE_EQ(sits_sources.size(), static_cast<std::size_t>(2));
    REQUIRE_EQ(sits_sources[0].file_path,
               std::string("prod.spg.shards/segment_000000.spg.h5md"));
    REQUIRE_EQ(sits_sources[1].file_path,
               std::string("prod.spg.shards/segment_000001.spg.h5md"));
    REQUIRE_EQ(sits_sources[0].dataset_path, Sits_Nk_Value_Path("sits_a"));
    REQUIRE_EQ(sits_sources[0].source_dims[1], static_cast<std::size_t>(3));
    REQUIRE_EQ(sits_sources[1].virtual_start[0], static_cast<std::size_t>(2));

    const auto& metad_sources =
        wrapper.virtual_datasets.at(Metadynamics_Scalar_Value_Path("meta"));
    REQUIRE_EQ(metad_sources.size(), static_cast<std::size_t>(2));
    REQUIRE_EQ(metad_sources[0].file_path,
               std::string("prod.spg.shards/segment_000000.spg.h5md"));
    REQUIRE_EQ(metad_sources[1].file_path,
               std::string("prod.spg.shards/segment_000001.spg.h5md"));
    REQUIRE_EQ(metad_sources[0].dataset_path,
               Metadynamics_Scalar_Value_Path("meta"));
    REQUIRE_EQ(metad_sources[1].virtual_start[0], static_cast<std::size_t>(2));
    const auto& reaxff_sources =
        wrapper.virtual_datasets.at(Reaxff_Term_Value_Path("bond"));
    REQUIRE_EQ(reaxff_sources.size(), static_cast<std::size_t>(2));
    REQUIRE_EQ(reaxff_sources[0].file_path,
               std::string("prod.spg.shards/segment_000000.spg.h5md"));
    REQUIRE_EQ(reaxff_sources[1].file_path,
               std::string("prod.spg.shards/segment_000001.spg.h5md"));
    REQUIRE_EQ(reaxff_sources[0].dataset_path, Reaxff_Term_Value_Path("bond"));
    REQUIRE_EQ(reaxff_sources[1].virtual_start[0], static_cast<std::size_t>(2));

    REQUIRE_TRUE(Has_Hard_Link(*factory.logs[0], module_path::metad_step,
                               Metadynamics_Scalar_Step_Path("meta")));
    REQUIRE_TRUE(Has_Hard_Link(*factory.logs[0], module_path::metad_time,
                               Metadynamics_Scalar_Time_Path("meta")));
    REQUIRE_TRUE(Has_Hard_Link(*factory.logs[0], module_path::metad_step,
                               Metadynamics_Scalar_Step_Path("rbias")));
    REQUIRE_TRUE(Has_Hard_Link(*factory.logs[0], module_path::metad_time,
                               Metadynamics_Scalar_Time_Path("rbias")));
    REQUIRE_TRUE(Has_Hard_Link(*factory.logs[0], module_path::metad_step,
                               Metadynamics_Scalar_Step_Path("rct")));
    REQUIRE_TRUE(Has_Hard_Link(*factory.logs[0], module_path::metad_time,
                               Metadynamics_Scalar_Time_Path("rct")));
    REQUIRE_TRUE(Has_Hard_Link(*factory.logs[0], module_path::reaxff_step,
                               Reaxff_Term_Step_Path("bond")));
    REQUIRE_TRUE(Has_Hard_Link(*factory.logs[0], module_path::reaxff_time,
                               Reaxff_Term_Time_Path("bond")));
    REQUIRE_TRUE(Has_Hard_Link(*factory.logs[0], module_path::reaxff_step,
                               Reaxff_Term_Step_Path("angle")));
    REQUIRE_TRUE(Has_Hard_Link(*factory.logs[0], module_path::reaxff_time,
                               Reaxff_Term_Time_Path("angle")));

    REQUIRE_EQ(wrapper.strings.at(path::output_repair_policy),
               std::string("strict"));
    REQUIRE_EQ(wrapper.strings.at(path::output_repair_status),
               std::string("not_applied"));
    REQUIRE_TRUE(wrapper.datasets.count(path::output_repaired_shard_count) !=
                 0);
    REQUIRE_EQ(wrapper.append_counts.at(path::output_repaired_shard_count), 1);
    REQUIRE_EQ(wrapper.strings.at(path::output_trajectory_chunk_size),
               std::string("2"));
    REQUIRE_EQ(
        wrapper.strings.at(Metadynamics_Diagnostic_Path("meta0", "hills")),
        std::string("HILLS"));
    REQUIRE_EQ(wrapper.string_arrays.at(path::legacy_sidecar_keys)[1],
               std::string("mdout"));
    REQUIRE_EQ(wrapper.string_arrays.at(path::legacy_sidecar_paths)[0],
               std::string("legacy.crd"));
    REQUIRE_TRUE(wrapper.datasets.count(path::shard_manifest_index) != 0);
    REQUIRE_TRUE(wrapper.datasets.count(path::shard_manifest_frame_start) != 0);
    REQUIRE_TRUE(wrapper.datasets.count(path::shard_manifest_frame_count) != 0);
    REQUIRE_TRUE(wrapper.datasets.count(path::shard_manifest_particles_count) !=
                 0);
    REQUIRE_TRUE(
        wrapper.datasets.count(path::shard_manifest_observables_count) != 0);
    REQUIRE_TRUE(wrapper.datasets.count(path::shard_manifest_nhc_count) != 0);
    REQUIRE_TRUE(wrapper.datasets.count(path::shard_manifest_sits_count) != 0);
    REQUIRE_TRUE(
        wrapper.datasets.count(path::shard_manifest_metadynamics_count) != 0);
    REQUIRE_TRUE(wrapper.datasets.count(path::shard_manifest_reaxff_count) !=
                 0);
    REQUIRE_TRUE(wrapper.datasets.count(path::shard_manifest_step_start) != 0);
    REQUIRE_TRUE(wrapper.datasets.count(path::shard_manifest_step_end) != 0);
    REQUIRE_TRUE(wrapper.datasets.count(path::shard_manifest_time_start) != 0);
    REQUIRE_TRUE(wrapper.datasets.count(path::shard_manifest_time_end) != 0);
    REQUIRE_EQ(wrapper.string_arrays.at(path::shard_manifest_path).size(),
               static_cast<std::size_t>(2));
    REQUIRE_EQ(wrapper.string_arrays.at(path::shard_manifest_status).size(),
               static_cast<std::size_t>(2));
    REQUIRE_EQ(wrapper.string_arrays.at(path::shard_manifest_status)[0],
               std::string("complete"));
    REQUIRE_EQ(wrapper.string_arrays.at(path::shard_manifest_status)[1],
               std::string("complete"));
    REQUIRE_EQ(wrapper.append_counts.at(path::shard_manifest_index),
               static_cast<int64_t>(2));
    REQUIRE_EQ(wrapper.append_counts.at(path::shard_manifest_frame_start),
               static_cast<int64_t>(2));
    REQUIRE_EQ(wrapper.append_counts.at(path::shard_manifest_frame_count),
               static_cast<int64_t>(2));
    REQUIRE_EQ(wrapper.append_counts.at(path::shard_manifest_particles_count),
               static_cast<int64_t>(2));
    REQUIRE_EQ(wrapper.append_counts.at(path::shard_manifest_observables_count),
               static_cast<int64_t>(2));
    REQUIRE_EQ(wrapper.append_counts.at(path::shard_manifest_nhc_count),
               static_cast<int64_t>(2));
    REQUIRE_EQ(wrapper.append_counts.at(path::shard_manifest_sits_count),
               static_cast<int64_t>(2));
    REQUIRE_EQ(
        wrapper.append_counts.at(path::shard_manifest_metadynamics_count),
        static_cast<int64_t>(2));
    REQUIRE_EQ(wrapper.append_counts.at(path::shard_manifest_reaxff_count),
               static_cast<int64_t>(2));
    REQUIRE_EQ(wrapper.append_counts.at(path::shard_manifest_step_start),
               static_cast<int64_t>(2));
    REQUIRE_EQ(wrapper.append_counts.at(path::shard_manifest_step_end),
               static_cast<int64_t>(2));
    REQUIRE_EQ(wrapper.append_counts.at(path::shard_manifest_time_start),
               static_cast<int64_t>(2));
    REQUIRE_EQ(wrapper.append_counts.at(path::shard_manifest_time_end),
               static_cast<int64_t>(2));
    REQUIRE_EQ(wrapper.append_counts.at(
                   Output_Stream_Committed_Count_Path("particles")),
               static_cast<int64_t>(2));
    REQUIRE_EQ(wrapper.append_counts.at(
                   Output_Stream_Committed_Count_Path("observables")),
               static_cast<int64_t>(2));
    REQUIRE_EQ(wrapper.append_counts.at(
                   Output_Stream_Committed_Count_Path("metadynamics")),
               static_cast<int64_t>(2));
    REQUIRE_EQ(wrapper.append_counts.at(path::output_publication_epoch),
               static_cast<int64_t>(3));

    const auto& shard0 = *factory.logs[1];
    const auto& shard1 = *factory.logs[2];
    REQUIRE_EQ(shard0.opened_path,
               std::string("/tmp/sponge_h5_vds_case/prod.spg.shards/"
                           "segment_000000.spg.h5md"));
    REQUIRE_EQ(shard1.opened_path,
               std::string("/tmp/sponge_h5_vds_case/prod.spg.shards/"
                           "segment_000001.spg.h5md"));
    REQUIRE_EQ(shard0.strings.at(path::sponge_schema_name),
               std::string("sponge.output.h5md"));
    REQUIRE_EQ(shard1.strings.at(path::sponge_schema_name),
               std::string("sponge.output.h5md"));
    REQUIRE_EQ(shard0.strings.at(path::sponge_schema_version),
               std::string("test"));
    REQUIRE_EQ(shard1.strings.at(path::sponge_schema_version),
               std::string("test"));
    REQUIRE_EQ(shard0.strings.at(path::sponge_topology_hash),
               std::string("top-hash"));
    REQUIRE_EQ(shard1.strings.at(path::sponge_atom_order_hash),
               std::string("order-hash"));
    REQUIRE_TRUE(shard0.datasets.count(path::force_value) == 0);
    REQUIRE_TRUE(shard1.datasets.count(path::force_value) == 0);
}

static void Test_Vds_Optional_Particle_Fields_Disabled()
{
    MockBackendFactory factory;
    VdsTrajectoryH5Writer writer(&factory);
    auto plan = Make_Vds_Plan();
    plan.trajectory.chunk_size = 1;

    REQUIRE_TRUE(writer.Open(plan, "test"));
    REQUIRE_TRUE(writer.Define_Particle_Datasets(1, false, false));

    float position_0[3] = {1.0f, 0.0f, 0.0f};
    float position_1[3] = {2.0f, 0.0f, 0.0f};
    float box[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    REQUIRE_TRUE(writer.Append_Particle_Frame(10, 0.1, position_0, box));
    REQUIRE_TRUE(writer.Append_Particle_Frame(20, 0.2, position_1, box));
    REQUIRE_TRUE(writer.Finalize());

    REQUIRE_EQ(writer.Manifest().size(), static_cast<std::size_t>(2));
    REQUIRE_EQ(writer.Total_Trajectory_Frame_Count(),
               static_cast<std::size_t>(2));
    REQUIRE_TRUE(factory.logs.size() >= 3);

    const auto& wrapper = *factory.logs[0];
    REQUIRE_TRUE(wrapper.virtual_datasets.count(path::particles_all_step) != 0);
    REQUIRE_TRUE(wrapper.virtual_datasets.count(path::particles_all_time) != 0);
    REQUIRE_TRUE(wrapper.virtual_datasets.count(path::position_value) != 0);
    REQUIRE_TRUE(wrapper.virtual_datasets.count(path::box_edges_value) != 0);
    REQUIRE_TRUE(wrapper.virtual_datasets.count(path::velocity_value) == 0);
    REQUIRE_TRUE(wrapper.virtual_datasets.count(path::force_value) == 0);
    Require_Dataset_Spec(wrapper, path::particles_all_step, DataType::int64,
                         {2}, {2}, {2}, false);
    Require_Dataset_Spec(wrapper, path::particles_all_time, DataType::float64,
                         {2}, {2}, {2}, false);
    Require_Dataset_Spec(wrapper, path::position_value, DataType::float32,
                         {2, 1, 3}, {2, 1, 3}, {2, 1, 3}, false);
    Require_Dataset_Spec(wrapper, path::box_edges_value, DataType::float32,
                         {2, 3, 3}, {2, 3, 3}, {2, 3, 3}, false);
    REQUIRE_TRUE(Has_Hard_Link(*factory.logs[0], path::particles_all_step,
                               path::position_step));
    REQUIRE_TRUE(Has_Hard_Link(*factory.logs[0], path::particles_all_time,
                               path::position_time));
    REQUIRE_TRUE(Has_Hard_Link(*factory.logs[0], path::particles_all_step,
                               path::box_edges_step));
    REQUIRE_TRUE(Has_Hard_Link(*factory.logs[0], path::particles_all_time,
                               path::box_edges_time));
    REQUIRE_TRUE(!Has_Hard_Link(*factory.logs[0], path::particles_all_step,
                                path::velocity_step));
    REQUIRE_TRUE(!Has_Hard_Link(*factory.logs[0], path::particles_all_time,
                                path::velocity_time));
    REQUIRE_TRUE(!Has_Hard_Link(*factory.logs[0], path::particles_all_step,
                                path::force_step));
    REQUIRE_TRUE(!Has_Hard_Link(*factory.logs[0], path::particles_all_time,
                                path::force_time));

    const auto& position_sources =
        wrapper.virtual_datasets.at(path::position_value);
    REQUIRE_EQ(position_sources.size(), static_cast<std::size_t>(2));
    REQUIRE_EQ(position_sources[0].source_dims[0], static_cast<std::size_t>(1));
    REQUIRE_EQ(position_sources[1].virtual_start[0],
               static_cast<std::size_t>(1));

    const auto& shard0 = *factory.logs[1];
    const auto& shard1 = *factory.logs[2];
    REQUIRE_TRUE(shard0.datasets.count(path::position_value) != 0);
    REQUIRE_TRUE(shard1.datasets.count(path::position_value) != 0);
    REQUIRE_TRUE(shard0.datasets.count(path::box_edges_value) != 0);
    REQUIRE_TRUE(shard1.datasets.count(path::box_edges_value) != 0);
    REQUIRE_TRUE(shard0.datasets.count(path::velocity_value) == 0);
    REQUIRE_TRUE(shard1.datasets.count(path::velocity_value) == 0);
    REQUIRE_TRUE(shard0.datasets.count(path::force_value) == 0);
    REQUIRE_TRUE(shard1.datasets.count(path::force_value) == 0);
    REQUIRE_EQ(wrapper.string_arrays.at(path::shard_manifest_status).size(),
               static_cast<std::size_t>(2));
    REQUIRE_EQ(wrapper.string_arrays.at(path::shard_manifest_status)[0],
               std::string("complete"));
    REQUIRE_EQ(wrapper.string_arrays.at(path::shard_manifest_status)[1],
               std::string("complete"));
}

static void Test_Vds_Shard_Filename_Sequence_Uses_Six_Digit_Padding()
{
    MockBackendFactory factory;
    VdsTrajectoryH5Writer writer(&factory);
    auto plan = Make_Vds_Plan();
    plan.trajectory.chunk_size = 1;

    REQUIRE_TRUE(writer.Open(plan, "test"));
    REQUIRE_TRUE(writer.Define_Particle_Datasets(1, false, false));

    float position[3] = {0, 0, 0};
    float box[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    REQUIRE_TRUE(writer.Append_Particle_Frame(1, 0.0, position, box));
    REQUIRE_TRUE(writer.Append_Particle_Frame(2, 0.1, position, box));
    REQUIRE_TRUE(writer.Append_Particle_Frame(3, 0.2, position, box));
    REQUIRE_TRUE(writer.Finalize());

    REQUIRE_EQ(writer.Manifest().size(), static_cast<std::size_t>(3));
    REQUIRE_EQ(writer.Manifest()[2].path,
               std::string("/tmp/sponge_h5_vds_case/prod.spg.shards/"
                           "segment_000002.spg.h5md"));
    REQUIRE_TRUE(factory.logs.size() >= 4);
    REQUIRE_EQ(factory.logs[3]->opened_path,
               std::string("/tmp/sponge_h5_vds_case/prod.spg.shards/"
                           "segment_000002.spg.h5md"));
}

static void Test_Strict_Finalize_Fails_On_Shard_Finalize_Error()
{
    MockBackendFactory factory;
    factory.fail_finalize = {false, true};
    VdsTrajectoryH5Writer writer(&factory);
    auto plan = Make_Vds_Plan();

    REQUIRE_TRUE(writer.Open(plan, "test"));
    REQUIRE_TRUE(writer.Define_Particle_Datasets(1, false, false));
    float position[3] = {0, 0, 0};
    float box[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    REQUIRE_TRUE(writer.Append_Particle_Frame(1, 0.0, position, box));
    REQUIRE_TRUE(!writer.Finalize());
    REQUIRE_TRUE(factory.logs.size() >= 2);
    const auto& wrapper = *factory.logs[0];
    REQUIRE_EQ(wrapper.status, FileStatus::failed);
    REQUIRE_EQ(wrapper.strings.at(path::output_status), std::string("failed"));
    REQUIRE_EQ(wrapper.strings.at(path::output_error),
               std::string("mock finalize failure"));
}

static void Test_Vds_Materialize_Failure_Marks_Wrapper_Failed()
{
    MockBackendFactory factory;
    VdsTrajectoryH5Writer writer(&factory);
    auto plan = Make_Vds_Plan();

    REQUIRE_TRUE(writer.Open(plan, "test"));
    REQUIRE_TRUE(writer.Define_Particle_Datasets(1, false, false));
    float position[3] = {0, 0, 0};
    float box[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    REQUIRE_TRUE(writer.Append_Particle_Frame(1, 0.0, position, box));

    REQUIRE_TRUE(!factory.logs.empty());
    factory.logs[0]->fail_next_virtual_dataset = true;
    REQUIRE_TRUE(!writer.Finalize());

    const auto& wrapper = *factory.logs[0];
    REQUIRE_EQ(writer.Last_Error(),
               std::string("mock virtual dataset failure: ") +
                   path::particles_all_step);
    REQUIRE_EQ(wrapper.status, FileStatus::failed);
    REQUIRE_EQ(wrapper.strings.at(path::output_status), std::string("failed"));
    REQUIRE_EQ(wrapper.strings.at(path::output_error),
               std::string("mock virtual dataset failure: ") +
                   path::particles_all_step);
}

static void Test_Vds_Manifest_Write_Failure_Marks_Wrapper_Failed()
{
    MockBackendFactory factory;
    VdsTrajectoryH5Writer writer(&factory);
    auto plan = Make_Vds_Plan();

    REQUIRE_TRUE(writer.Open(plan, "test"));
    REQUIRE_TRUE(writer.Define_Particle_Datasets(1, false, false));
    float position[3] = {0, 0, 0};
    float box[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    REQUIRE_TRUE(writer.Append_Particle_Frame(1, 0.0, position, box));

    REQUIRE_TRUE(!factory.logs.empty());
    factory.logs[0]->fail_next_string_array = true;
    REQUIRE_TRUE(!writer.Finalize());

    const auto& wrapper = *factory.logs[0];
    REQUIRE_EQ(writer.Last_Error(), std::string("mock string-array failure: ") +
                                        path::shard_manifest_path);
    REQUIRE_EQ(wrapper.status, FileStatus::failed);
    REQUIRE_EQ(wrapper.strings.at(path::output_status), std::string("failed"));
    REQUIRE_EQ(
        wrapper.strings.at(path::output_error),
        std::string("mock string-array failure: ") + path::shard_manifest_path);
}

static void Test_Vds_Repair_Metadata_Failure_Marks_Wrapper_Failed()
{
    MockBackendFactory factory;
    VdsTrajectoryH5Writer writer(&factory);
    auto plan = Make_Vds_Plan();

    REQUIRE_TRUE(writer.Open(plan, "test"));
    REQUIRE_TRUE(writer.Define_Particle_Datasets(1, false, false));
    float position[3] = {0, 0, 0};
    float box[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    REQUIRE_TRUE(writer.Append_Particle_Frame(1, 0.0, position, box));

    REQUIRE_TRUE(!factory.logs.empty());
    factory.logs[0]->fail_string_path = path::output_repair_policy;
    REQUIRE_TRUE(!writer.Finalize());

    const auto& wrapper = *factory.logs[0];
    REQUIRE_EQ(writer.Last_Error(), std::string("mock string failure: ") +
                                        path::output_repair_policy);
    REQUIRE_EQ(wrapper.status, FileStatus::failed);
    REQUIRE_EQ(wrapper.strings.at(path::output_status), std::string("failed"));
    REQUIRE_EQ(
        wrapper.strings.at(path::output_error),
        std::string("mock string failure: ") + path::output_repair_policy);
}

static void Test_Vds_Status_Write_Failure_Marks_Wrapper_Failed()
{
    MockBackendFactory factory;
    VdsTrajectoryH5Writer writer(&factory);
    auto plan = Make_Vds_Plan();

    REQUIRE_TRUE(writer.Open(plan, "test"));
    REQUIRE_TRUE(writer.Define_Particle_Datasets(1, false, false));
    float position[3] = {0, 0, 0};
    float box[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    REQUIRE_TRUE(writer.Append_Particle_Frame(1, 0.0, position, box));

    REQUIRE_TRUE(!factory.logs.empty());
    factory.logs[0]->fail_string_path = path::output_vds_status;
    REQUIRE_TRUE(!writer.Finalize());

    const auto& wrapper = *factory.logs[0];
    REQUIRE_EQ(writer.Last_Error(),
               std::string("mock string failure: ") + path::output_vds_status);
    REQUIRE_EQ(wrapper.status, FileStatus::failed);
    REQUIRE_EQ(wrapper.strings.at(path::output_status), std::string("failed"));
    REQUIRE_EQ(wrapper.strings.at(path::output_error),
               std::string("mock string failure: ") + path::output_vds_status);
}

static void Test_Vds_Wrapper_Finalize_Failure_Marks_Wrapper_Failed()
{
    MockBackendFactory factory;
    factory.fail_finalize = {true, false};
    VdsTrajectoryH5Writer writer(&factory);
    auto plan = Make_Vds_Plan();

    REQUIRE_TRUE(writer.Open(plan, "test"));
    REQUIRE_TRUE(writer.Define_Particle_Datasets(1, false, false));
    float position[3] = {0, 0, 0};
    float box[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    REQUIRE_TRUE(writer.Append_Particle_Frame(1, 0.0, position, box));

    REQUIRE_TRUE(!writer.Finalize());
    REQUIRE_TRUE(factory.logs.size() >= 2);
    const auto& wrapper = *factory.logs[0];
    REQUIRE_EQ(writer.Last_Error(), std::string("mock finalize failure"));
    REQUIRE_EQ(wrapper.status, FileStatus::failed);
    REQUIRE_EQ(wrapper.strings.at(path::output_status), std::string("failed"));
    REQUIRE_EQ(wrapper.strings.at(path::output_error),
               std::string("mock finalize failure"));
}

static void Test_Vds_Precondition_Errors()
{
    MockBackendFactory factory;
    VdsTrajectoryH5Writer writer(&factory);
    auto plan = Make_Vds_Plan();

    float position[3] = {0, 0, 0};
    float box[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    REQUIRE_TRUE(!writer.Append_Particle_Frame(1, 0.0, position, box));
    REQUIRE_EQ(writer.Last_Error(),
               std::string("particle layout must be defined before appending"));

    REQUIRE_TRUE(writer.Open(plan, "test"));
    REQUIRE_TRUE(writer.Define_Particle_Datasets(1, false, false));
    REQUIRE_TRUE(
        !writer.Append_Observable_Frame(1, 0.0, {{"temperature", 1.0}}));
    REQUIRE_EQ(writer.Last_Error(),
               std::string("observable layout must be defined before "
                           "appending"));

    REQUIRE_TRUE(writer.Append_Particle_Frame(1, 0.0, position, box));
    REQUIRE_TRUE(
        !writer.Append_Nose_Hoover_Chain_Frame(1, 0.0, position, position, 3));
    REQUIRE_EQ(writer.Last_Error(),
               std::string("NHC layout must be defined before appending"));
    REQUIRE_TRUE(writer.Ensure_Nose_Hoover_Chain_Observables(2));
    REQUIRE_TRUE(
        !writer.Append_Nose_Hoover_Chain_Frame(1, 0.0, position, position, 3));
    REQUIRE_EQ(writer.Last_Error(),
               std::string("NHC chain length changed within VDS trajectory"));
    REQUIRE_TRUE(
        !writer.Append_Metadynamics_Scalar_Frame(1, 0.0, 1.0, 2.0, 3.0));
    REQUIRE_EQ(
        writer.Last_Error(),
        std::string(
            "metadynamics scalar layout must be defined before appending"));
    REQUIRE_TRUE(!writer.Append_Reaxff_Frame(1, 0.0, {{"bond", 1.0}}));
    REQUIRE_EQ(writer.Last_Error(),
               std::string("ReaxFF layout must be defined before appending"));
    REQUIRE_TRUE(!writer.Append_Sits_Nk_Frame(1, 0.0, "sits_a", position, 3));
    REQUIRE_EQ(writer.Last_Error(),
               std::string("SITS nk layout must be defined before appending"));
    REQUIRE_TRUE(writer.Ensure_Sits_Nk_Observable("sits_a", 3));
    REQUIRE_TRUE(!writer.Append_Sits_Nk_Frame(1, 0.0, "sits_b", position, 3));
    REQUIRE_EQ(writer.Last_Error(),
               std::string("SITS nk shape changed within VDS trajectory"));
}

static void Test_Vds_Preserves_Observables_Before_First_Particle_Frame()
{
    MockBackendFactory factory;
    VdsTrajectoryH5Writer writer(&factory);
    auto plan = Make_Vds_Plan();

    REQUIRE_TRUE(writer.Open(plan, "test"));
    REQUIRE_TRUE(writer.Define_Particle_Datasets(1, false, false));
    REQUIRE_TRUE(writer.Define_Observable_Stream({"temperature"}, {"TEMP"}));

    REQUIRE_TRUE(
        writer.Append_Observable_Frame(0, 0.0, {{"temperature", 299.0}}));
    REQUIRE_TRUE(
        writer.Append_Observable_Frame(1, 0.01, {{"temperature", 300.0}}));
    REQUIRE_EQ(writer.Total_Observable_Frame_Count(),
               static_cast<std::size_t>(2));

    float position[3] = {1.0f, 0.0f, 0.0f};
    float box[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    REQUIRE_TRUE(writer.Append_Particle_Frame(1, 0.01, position, box));
    REQUIRE_TRUE(
        writer.Append_Observable_Frame(2, 0.02, {{"temperature", 301.0}}));
    REQUIRE_EQ(writer.Total_Observable_Frame_Count(),
               static_cast<std::size_t>(3));
    REQUIRE_TRUE(writer.Finalize());

    REQUIRE_EQ(writer.Manifest().size(), static_cast<std::size_t>(1));
    REQUIRE_EQ(writer.Manifest()[0].frame_count, static_cast<int64_t>(1));
    REQUIRE_EQ(writer.Manifest()[0].observable_frame_count,
               static_cast<int64_t>(3));
    REQUIRE_EQ(writer.Manifest()[0].step_start, static_cast<int64_t>(0));
    REQUIRE_EQ(writer.Manifest()[0].step_end, static_cast<int64_t>(1));
    REQUIRE_EQ(factory.logs[1]->append_counts.at(path::observables_all_step),
               static_cast<int64_t>(3));
}

static void Test_Vds_Finalizes_Module_Frames_Before_First_Particle_Frame()
{
    MockBackendFactory factory;
    VdsTrajectoryH5Writer writer(&factory);
    auto plan = Make_Vds_Plan();

    REQUIRE_TRUE(writer.Open(plan, "test"));
    REQUIRE_TRUE(writer.Define_Particle_Datasets(1, false, false));
    REQUIRE_TRUE(writer.Define_Observable_Stream({"temperature"}, {"TEMP"}));
    REQUIRE_TRUE(writer.Ensure_Nose_Hoover_Chain_Observables(2));
    REQUIRE_TRUE(writer.Ensure_Sits_Nk_Observable("sits_a", 3));
    REQUIRE_TRUE(writer.Ensure_Metadynamics_Scalars());
    REQUIRE_TRUE(writer.Ensure_Reaxff_Energy_Terms({"bond", "angle"}));

    float nhc[2] = {0.1f, 0.2f};
    float sits[3] = {1.0f, 2.0f, 3.0f};
    for (int64_t step = 0; step < 2; ++step)
    {
        const double time = static_cast<double>(step) * 0.01;
        REQUIRE_TRUE(writer.Append_Reaxff_Frame(
            step, time, {{"bond", 1.25}, {"angle", 2.25}}));
        REQUIRE_TRUE(
            writer.Append_Metadynamics_Scalar_Frame(step, time, 1.0, 2.0, 3.0));
        REQUIRE_TRUE(
            writer.Append_Sits_Nk_Frame(step, time, "sits_a", sits, 3));
        REQUIRE_TRUE(
            writer.Append_Nose_Hoover_Chain_Frame(step, time, nhc, nhc, 2));
        REQUIRE_TRUE(writer.Append_Observable_Frame(
            step, time, {{"temperature", 300.0 + step}}));
    }

    REQUIRE_TRUE(writer.Finalize());
    REQUIRE_EQ(writer.Total_Trajectory_Frame_Count(),
               static_cast<std::size_t>(0));
    REQUIRE_EQ(writer.Manifest().size(), static_cast<std::size_t>(1));
    const auto& entry = writer.Manifest()[0];
    REQUIRE_EQ(entry.frame_count, static_cast<int64_t>(0));
    REQUIRE_EQ(entry.observable_frame_count, static_cast<int64_t>(2));
    REQUIRE_EQ(entry.nhc_frame_count, static_cast<int64_t>(2));
    REQUIRE_EQ(entry.sits_nk_frame_count, static_cast<int64_t>(2));
    REQUIRE_EQ(entry.metadynamics_scalar_frame_count, static_cast<int64_t>(2));
    REQUIRE_EQ(entry.reaxff_frame_count, static_cast<int64_t>(2));
    REQUIRE_EQ(entry.step_start, static_cast<int64_t>(0));
    REQUIRE_EQ(entry.step_end, static_cast<int64_t>(1));
    REQUIRE_EQ(entry.time_end, 0.01);

    const auto& wrapper = *factory.logs[0];
    REQUIRE_TRUE(wrapper.virtual_datasets.count(path::particles_all_step) == 0);
    REQUIRE_TRUE(wrapper.virtual_datasets.count(module_path::nhc_step) != 0);
    REQUIRE_TRUE(wrapper.virtual_datasets.count(Sits_Nk_Step_Path("sits_a")) !=
                 0);
    REQUIRE_TRUE(wrapper.virtual_datasets.count(
                     Metadynamics_Scalar_Value_Path("meta")) != 0);
    REQUIRE_TRUE(
        wrapper.virtual_datasets.count(Reaxff_Term_Value_Path("bond")) != 0);
}

static void Test_Vds_Finalizes_Observable_Only_Shard()
{
    MockBackendFactory factory;
    VdsTrajectoryH5Writer writer(&factory);
    auto plan = Make_Vds_Plan();

    REQUIRE_TRUE(writer.Open(plan, "test"));
    REQUIRE_TRUE(writer.Define_Particle_Datasets(1, false, false));
    REQUIRE_TRUE(writer.Define_Observable_Stream({"temperature"}, {"TEMP"}));
    REQUIRE_TRUE(
        writer.Append_Observable_Frame(0, 0.0, {{"temperature", 299.0}}));

    REQUIRE_TRUE(writer.Finalize());
    REQUIRE_EQ(writer.Total_Trajectory_Frame_Count(),
               static_cast<std::size_t>(0));
    REQUIRE_EQ(writer.Total_Observable_Frame_Count(),
               static_cast<std::size_t>(1));
    REQUIRE_EQ(writer.Manifest().size(), static_cast<std::size_t>(1));
    REQUIRE_EQ(writer.Manifest()[0].frame_count, static_cast<int64_t>(0));
    REQUIRE_EQ(writer.Manifest()[0].observable_frame_count,
               static_cast<int64_t>(1));
    REQUIRE_EQ(factory.logs[1]->append_counts.at(path::observables_all_step),
               static_cast<int64_t>(1));
    REQUIRE_EQ(
        factory.logs[0]->virtual_datasets.at(path::observables_all_step).size(),
        static_cast<std::size_t>(1));
    REQUIRE_TRUE(
        factory.logs[0]->virtual_datasets.count(path::particles_all_step) == 0);
}

static void Test_Vds_Open_Precondition_Errors()
{
    auto plan = Make_Vds_Plan();
    {
        VdsTrajectoryH5Writer writer(nullptr);
        REQUIRE_TRUE(!writer.Open(plan, "test"));
        REQUIRE_EQ(
            writer.Last_Error(),
            std::string("VDS trajectory writer requires a backend factory"));
    }
    {
        MockBackendFactory factory;
        VdsTrajectoryH5Writer writer(&factory);
        auto invalid_plan = plan;
        invalid_plan.trajectory.enabled = false;
        REQUIRE_TRUE(!writer.Open(invalid_plan, "test"));
        REQUIRE_EQ(
            writer.Last_Error(),
            std::string(
                "VdsTrajectoryH5Writer requires enabled VDS trajectory plan"));
    }
    {
        MockBackendFactory factory;
        VdsTrajectoryH5Writer writer(&factory);
        auto invalid_plan = plan;
        invalid_plan.trajectory.vds = false;
        REQUIRE_TRUE(!writer.Open(invalid_plan, "test"));
        REQUIRE_EQ(
            writer.Last_Error(),
            std::string(
                "VdsTrajectoryH5Writer requires enabled VDS trajectory plan"));
    }
    {
        MockBackendFactory factory;
        VdsTrajectoryH5Writer writer(&factory);
        REQUIRE_TRUE(
            !writer.Write_Metadynamics_Diagnostic("meta0", "hills", "HILLS"));
        REQUIRE_EQ(writer.Last_Error(), std::string("VDS wrapper is not open"));
        REQUIRE_TRUE(
            !writer.Write_Legacy_Sidecar_Paths({"mdout"}, {"legacy.out"}));
        REQUIRE_EQ(writer.Last_Error(), std::string("VDS wrapper is not open"));
    }
}

static void Test_Vds_Finalize_Without_Frames()
{
    MockBackendFactory factory;
    VdsTrajectoryH5Writer writer(&factory);
    auto plan = Make_Vds_Plan();

    REQUIRE_TRUE(writer.Open(plan, "test"));
    REQUIRE_TRUE(writer.Finalize());
    REQUIRE_EQ(writer.Manifest().size(), static_cast<std::size_t>(0));
    REQUIRE_EQ(writer.Total_Trajectory_Frame_Count(),
               static_cast<std::size_t>(0));
    REQUIRE_EQ(writer.Total_Observable_Frame_Count(),
               static_cast<std::size_t>(0));

    const auto& wrapper = *factory.logs[0];
    REQUIRE_EQ(wrapper.status, FileStatus::finalized);
    REQUIRE_EQ(wrapper.strings.at(path::output_repair_policy),
               std::string("strict"));
    REQUIRE_EQ(wrapper.strings.at(path::output_repair_status),
               std::string("not_applied"));
    REQUIRE_TRUE(wrapper.datasets.count(path::output_repaired_shard_count) !=
                 0);
    REQUIRE_EQ(wrapper.append_counts.at(path::output_repaired_shard_count), 1);
    REQUIRE_EQ(
        wrapper.strings.at(path::output_vds_status),
        std::string(
            "particle, observable, and module virtual datasets materialized"));
    REQUIRE_TRUE(wrapper.virtual_datasets.empty());
}

static void Test_Complete_Prefix_Repair_Finalize()
{
    MockBackendFactory factory;
    factory.fail_finalize = {false, true};
    VdsTrajectoryH5Writer writer(&factory);
    auto plan = Make_Vds_Plan();

    REQUIRE_TRUE(writer.Open(plan, "test"));
    REQUIRE_TRUE(writer.Define_Particle_Datasets(1, false, false));
    float position[3] = {0, 0, 0};
    float box[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    REQUIRE_TRUE(writer.Append_Particle_Frame(1, 0.0, position, box));
    REQUIRE_TRUE(writer.Finalize_With_Repair());
    REQUIRE_EQ(writer.Manifest().size(), static_cast<std::size_t>(0));

    const auto& wrapper = *factory.logs[0];
    REQUIRE_EQ(wrapper.strings.at(path::output_repair_policy),
               std::string("complete_prefix"));
    REQUIRE_EQ(wrapper.strings.at(path::output_repair_status),
               std::string("applied"));
    REQUIRE_TRUE(wrapper.datasets.count(path::output_repaired_shard_count) !=
                 0);
    REQUIRE_EQ(wrapper.append_counts.at(path::output_repaired_shard_count), 1);
}

static void Test_Complete_Prefix_Repair_Retains_Valid_Prefix()
{
    MockBackendFactory factory;
    factory.fail_finalize = {false, false, true};
    VdsTrajectoryH5Writer writer(&factory);
    auto plan = Make_Vds_Plan();
    plan.trajectory.chunk_size = 1;

    REQUIRE_TRUE(writer.Open(plan, "test"));
    REQUIRE_TRUE(writer.Define_Particle_Datasets(1, false, false));

    float position_a[3] = {1.0f, 0.0f, 0.0f};
    float position_b[3] = {2.0f, 0.0f, 0.0f};
    float box[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    REQUIRE_TRUE(writer.Append_Particle_Frame(1, 0.1, position_a, box));
    REQUIRE_TRUE(writer.Append_Particle_Frame(2, 0.2, position_b, box));
    REQUIRE_TRUE(writer.Finalize_With_Repair());

    REQUIRE_EQ(writer.Manifest().size(), static_cast<std::size_t>(1));
    REQUIRE_EQ(writer.Manifest()[0].index, static_cast<int64_t>(0));
    REQUIRE_EQ(writer.Manifest()[0].frame_start, static_cast<int64_t>(0));
    REQUIRE_EQ(writer.Manifest()[0].frame_count, static_cast<int64_t>(1));
    REQUIRE_EQ(writer.Manifest()[0].step_start, static_cast<int64_t>(1));
    REQUIRE_EQ(writer.Manifest()[0].step_end, static_cast<int64_t>(1));
    REQUIRE_EQ(writer.Total_Trajectory_Frame_Count(),
               static_cast<std::size_t>(1));

    const auto& wrapper = *factory.logs[0];
    REQUIRE_EQ(wrapper.strings.at(path::output_repair_policy),
               std::string("complete_prefix"));
    REQUIRE_EQ(wrapper.strings.at(path::output_repair_status),
               std::string("applied"));
    REQUIRE_TRUE(wrapper.datasets.count(path::output_repaired_shard_count) !=
                 0);
    REQUIRE_EQ(wrapper.append_counts.at(path::output_repaired_shard_count), 1);
    REQUIRE_TRUE(wrapper.virtual_datasets.count(path::position_value) != 0);
    const auto& position_sources =
        wrapper.virtual_datasets.at(path::position_value);
    REQUIRE_EQ(position_sources.size(), static_cast<std::size_t>(1));
    REQUIRE_EQ(position_sources[0].file_path,
               std::string("prod.spg.shards/segment_000000.spg.h5md"));
    REQUIRE_EQ(position_sources[0].source_dims[0], static_cast<std::size_t>(1));
    REQUIRE_EQ(position_sources[0].virtual_start[0],
               static_cast<std::size_t>(0));
    REQUIRE_EQ(wrapper.string_arrays.at(path::shard_manifest_status).size(),
               static_cast<std::size_t>(1));
    REQUIRE_EQ(wrapper.string_arrays.at(path::shard_manifest_status)[0],
               std::string("complete"));
}

static void Test_Vds_Source_Path_Relativization()
{
    MockBackendFactory factory;
    VdsTrajectoryH5Writer writer(&factory);
    auto plan = Make_Vds_Plan();
    plan.trajectory.path = "/tmp/sponge_h5_vds_case/wrappers/prod.spg.h5md";
    plan.trajectory.derived_shard_root =
        "/tmp/sponge_h5_vds_case/shards/prod.spg.shards";
    plan.trajectory.chunk_size = 1;

    REQUIRE_TRUE(writer.Open(plan, "test"));
    REQUIRE_TRUE(writer.Define_Particle_Datasets(1, false, false));

    float position[3] = {1.0f, 0.0f, 0.0f};
    float box[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    REQUIRE_TRUE(writer.Append_Particle_Frame(1, 0.1, position, box));
    REQUIRE_TRUE(writer.Finalize());

    const auto& wrapper = *factory.logs[0];
    REQUIRE_TRUE(wrapper.virtual_datasets.count(path::position_value) != 0);
    const auto& sources = wrapper.virtual_datasets.at(path::position_value);
    REQUIRE_EQ(sources.size(), static_cast<std::size_t>(1));
    REQUIRE_EQ(
        sources[0].file_path,
        std::string("../shards/prod.spg.shards/segment_000000.spg.h5md"));
}

static void Test_Vds_Source_Path_Without_Wrapper_Parent()
{
    MockBackendFactory factory;
    VdsTrajectoryH5Writer writer(&factory);
    auto plan = Make_Vds_Plan();
    plan.trajectory.path = "prod.spg.h5md";
    plan.trajectory.derived_shard_root = "prod.spg.shards";
    plan.trajectory.chunk_size = 1;

    REQUIRE_TRUE(writer.Open(plan, "test"));
    REQUIRE_TRUE(writer.Define_Particle_Datasets(1, false, false));

    float position[3] = {1.0f, 0.0f, 0.0f};
    float box[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    REQUIRE_TRUE(writer.Append_Particle_Frame(1, 0.1, position, box));
    REQUIRE_TRUE(writer.Finalize());

    const auto& wrapper = *factory.logs[0];
    REQUIRE_TRUE(wrapper.virtual_datasets.count(path::position_value) != 0);
    const auto& sources = wrapper.virtual_datasets.at(path::position_value);
    REQUIRE_EQ(sources.size(), static_cast<std::size_t>(1));
    REQUIRE_EQ(sources[0].file_path,
               std::string("prod.spg.shards/segment_000000.spg.h5md"));
}

static void Test_Vds_Chunk_Boundary_Frame_Counts()
{
    struct BoundaryCase
    {
        std::size_t frame_count;
        std::vector<std::size_t> shard_frame_counts;
    };
    const std::vector<BoundaryCase> cases = {
        {1, {1}},
        {2, {2}},
        {3, {2, 1}},
        {5, {2, 2, 1}},
    };

    for (const auto& boundary : cases)
    {
        MockBackendFactory factory;
        VdsTrajectoryH5Writer writer(&factory);
        auto plan = Make_Vds_Plan();
        plan.trajectory.chunk_size = 2;

        REQUIRE_TRUE(writer.Open(plan, "test"));
        REQUIRE_TRUE(writer.Define_Particle_Datasets(1, false, false));
        float box[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        for (std::size_t frame = 0; frame < boundary.frame_count; ++frame)
        {
            float position[3] = {static_cast<float>(frame), 0.0f, 0.0f};
            REQUIRE_TRUE(writer.Append_Particle_Frame(
                static_cast<std::int64_t>(frame), 0.1 * frame, position, box));
        }
        REQUIRE_TRUE(writer.Finalize());

        REQUIRE_EQ(writer.Total_Trajectory_Frame_Count(), boundary.frame_count);
        REQUIRE_EQ(writer.Manifest().size(),
                   boundary.shard_frame_counts.size());
        for (std::size_t shard = 0; shard < boundary.shard_frame_counts.size();
             ++shard)
        {
            REQUIRE_EQ(writer.Manifest()[shard].frame_count,
                       boundary.shard_frame_counts[shard]);
        }
    }
}

int main()
{
    return Run_Test(
        []
        {
            Test_Vds_Wrapper_And_Module_Virtual_Datasets();
            Test_Vds_Optional_Particle_Fields_Disabled();
            Test_Vds_Shard_Filename_Sequence_Uses_Six_Digit_Padding();
            Test_Strict_Finalize_Fails_On_Shard_Finalize_Error();
            Test_Vds_Materialize_Failure_Marks_Wrapper_Failed();
            Test_Vds_Manifest_Write_Failure_Marks_Wrapper_Failed();
            Test_Vds_Repair_Metadata_Failure_Marks_Wrapper_Failed();
            Test_Vds_Status_Write_Failure_Marks_Wrapper_Failed();
            Test_Vds_Wrapper_Finalize_Failure_Marks_Wrapper_Failed();
            Test_Vds_Precondition_Errors();
            Test_Vds_Preserves_Observables_Before_First_Particle_Frame();
            Test_Vds_Finalizes_Module_Frames_Before_First_Particle_Frame();
            Test_Vds_Finalizes_Observable_Only_Shard();
            Test_Vds_Open_Precondition_Errors();
            Test_Vds_Finalize_Without_Frames();
            Test_Complete_Prefix_Repair_Finalize();
            Test_Complete_Prefix_Repair_Retains_Valid_Prefix();
            Test_Vds_Chunk_Boundary_Frame_Counts();
            Test_Vds_Source_Path_Relativization();
            Test_Vds_Source_Path_Without_Wrapper_Parent();
        });
}

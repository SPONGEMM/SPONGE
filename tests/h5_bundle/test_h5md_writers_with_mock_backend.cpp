#include "h5_bundle_test_common.hpp"
#include "utils/control/legacy_output_flush.hpp"
#include "utils/h5md/module_h5_mappings.hpp"
#include "utils/h5md/observable_h5_writer.hpp"
#include "utils/h5md/restart_h5_writer.hpp"
#include "utils/h5md/trajectory_h5_writer.hpp"

using namespace SpongeH5Test;
using namespace SpongeH5MD;

static SpongeH5OutputPlan::ResolvedOutputPlan Make_Plan()
{
    SpongeH5OutputPlan::ResolvedOutputPlan plan;
    plan.trajectory.enabled = true;
    plan.trajectory.path = "trajectory.spg.h5md";
    plan.restart.enabled = true;
    plan.restart.path = "restart.spgr.h5";
    plan.observable.enabled = true;
    plan.observable.path = "observable.obs.spg.h5md";
    return plan;
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

static void Test_Public_H5MD_Path_Constants()
{
    REQUIRE_EQ(std::string(path::h5md), std::string("/h5md"));
    REQUIRE_EQ(std::string(path::particles), std::string("/particles"));
    REQUIRE_EQ(std::string(path::observables), std::string("/observables"));
    REQUIRE_EQ(std::string(path::parameters), std::string("/parameters"));
    REQUIRE_EQ(std::string(path::sponge), std::string("/parameters/sponge"));
    REQUIRE_EQ(std::string(path::sponge_schema),
               std::string("/parameters/sponge/schema"));
    REQUIRE_EQ(std::string(path::sponge_schema_name),
               std::string("/parameters/sponge/schema/name"));
    REQUIRE_EQ(std::string(path::sponge_schema_version),
               std::string("/parameters/sponge/schema/version"));
    REQUIRE_EQ(std::string(path::sponge_output),
               std::string("/parameters/sponge/output"));
    REQUIRE_EQ(std::string(path::sponge_mdout),
               std::string("/parameters/sponge/mdout"));
    REQUIRE_EQ(std::string(path::sponge_log),
               std::string("/parameters/sponge/log"));
    REQUIRE_EQ(std::string(path::sponge_files),
               std::string("/parameters/sponge/files"));
    REQUIRE_EQ(std::string(path::sponge_provenance),
               std::string("/parameters/sponge/provenance"));
    REQUIRE_EQ(Sponge_Provenance_Path("launch_id"),
               std::string("/parameters/sponge/provenance/launch_id"));

    REQUIRE_EQ(std::string(path::output_status),
               std::string("/parameters/sponge/output/status"));
    REQUIRE_EQ(std::string(path::output_frame_count),
               std::string("/parameters/sponge/output/frame_count"));
    REQUIRE_EQ(std::string(path::output_last_complete_step),
               std::string("/parameters/sponge/output/last_complete_step"));
    REQUIRE_EQ(std::string(path::output_last_complete_time),
               std::string("/parameters/sponge/output/last_complete_time"));
    REQUIRE_EQ(std::string(path::output_error),
               std::string("/parameters/sponge/output/error"));
    REQUIRE_EQ(std::string(path::output_publication_epoch),
               std::string("/parameters/sponge/output/publication_epoch"));
    REQUIRE_EQ(std::string(path::output_restart_generation),
               std::string("/parameters/sponge/output/restart_generation"));
    REQUIRE_EQ(std::string(path::output_streams),
               std::string("/parameters/sponge/output/streams"));
    REQUIRE_EQ(Output_Stream_Committed_Count_Path("particles"),
               std::string("/parameters/sponge/output/streams/particles/"
                           "committed_count"));
    REQUIRE_EQ(std::string(path::output_trajectory_chunk_size),
               std::string("/parameters/sponge/output/trajectory_chunk_size"));
    REQUIRE_EQ(std::string(path::output_vds_status),
               std::string("/parameters/sponge/output/vds_status"));
    REQUIRE_EQ(std::string(path::output_repair_policy),
               std::string("/parameters/sponge/output/repair_policy"));
    REQUIRE_EQ(std::string(path::output_repair_status),
               std::string("/parameters/sponge/output/repair_status"));
    REQUIRE_EQ(std::string(path::output_repaired_shard_count),
               std::string("/parameters/sponge/output/repaired_shard_count"));

    REQUIRE_EQ(std::string(path::shard_status),
               std::string("/parameters/sponge/shard/status"));
    REQUIRE_EQ(std::string(path::shard_frame_start),
               std::string("/parameters/sponge/shard/frame_start"));
    REQUIRE_EQ(std::string(path::shard_frame_count),
               std::string("/parameters/sponge/shard/frame_count"));
    REQUIRE_EQ(std::string(path::shard_last_complete_step),
               std::string("/parameters/sponge/shard/last_complete_step"));
    REQUIRE_EQ(std::string(path::shard_last_complete_time),
               std::string("/parameters/sponge/shard/last_complete_time"));

    REQUIRE_EQ(std::string(path::shard_manifest),
               std::string("/parameters/sponge/output/shard_manifest"));
    REQUIRE_EQ(std::string(path::shard_manifest_index),
               std::string("/parameters/sponge/output/shard_manifest/index"));
    REQUIRE_EQ(std::string(path::shard_manifest_path),
               std::string("/parameters/sponge/output/shard_manifest/path"));
    REQUIRE_EQ(
        std::string(path::shard_manifest_frame_start),
        std::string("/parameters/sponge/output/shard_manifest/frame_start"));
    REQUIRE_EQ(
        std::string(path::shard_manifest_frame_count),
        std::string("/parameters/sponge/output/shard_manifest/frame_count"));
    REQUIRE_EQ(std::string(path::shard_manifest_stream_counts),
               std::string("/parameters/sponge/output/shard_manifest/"
                           "stream_counts"));
    REQUIRE_EQ(std::string(path::shard_manifest_observables_count),
               std::string("/parameters/sponge/output/shard_manifest/"
                           "stream_counts/observables"));
    REQUIRE_EQ(
        std::string(path::shard_manifest_step_start),
        std::string("/parameters/sponge/output/shard_manifest/step_start"));
    REQUIRE_EQ(
        std::string(path::shard_manifest_step_end),
        std::string("/parameters/sponge/output/shard_manifest/step_end"));
    REQUIRE_EQ(
        std::string(path::shard_manifest_time_start),
        std::string("/parameters/sponge/output/shard_manifest/time_start"));
    REQUIRE_EQ(
        std::string(path::shard_manifest_time_end),
        std::string("/parameters/sponge/output/shard_manifest/time_end"));
    REQUIRE_EQ(std::string(path::shard_manifest_status),
               std::string("/parameters/sponge/output/shard_manifest/status"));

    REQUIRE_EQ(std::string(path::mdout_columns),
               std::string("/parameters/sponge/mdout/columns"));
    REQUIRE_EQ(std::string(path::mdout_columns_original_name),
               std::string("/parameters/sponge/mdout/columns/original_name"));
    REQUIRE_EQ(std::string(path::mdout_columns_hdf5_name),
               std::string("/parameters/sponge/mdout/columns/hdf5_name"));
    REQUIRE_EQ(std::string(path::mdinfo_text),
               std::string("/parameters/sponge/log/mdinfo_text"));
    REQUIRE_EQ(std::string(path::legacy_sidecars),
               std::string("/parameters/sponge/files/legacy_sidecars"));
    REQUIRE_EQ(std::string(path::legacy_sidecar_keys),
               std::string("/parameters/sponge/files/legacy_sidecars/key"));
    REQUIRE_EQ(std::string(path::legacy_sidecar_paths),
               std::string("/parameters/sponge/files/legacy_sidecars/path"));

    REQUIRE_EQ(std::string(path::particles_all), std::string("/particles/all"));
    REQUIRE_EQ(std::string(path::particles_all_position),
               std::string("/particles/all/position"));
    REQUIRE_EQ(std::string(path::particles_all_velocity),
               std::string("/particles/all/velocity"));
    REQUIRE_EQ(std::string(path::particles_all_force),
               std::string("/particles/all/force"));
    REQUIRE_EQ(std::string(path::particles_all_box),
               std::string("/particles/all/box"));
    REQUIRE_EQ(std::string(path::particles_all_box_edges),
               std::string("/particles/all/box/edges"));
    REQUIRE_EQ(std::string(path::particles_all_step),
               std::string("/particles/all/step"));
    REQUIRE_EQ(std::string(path::particles_all_time),
               std::string("/particles/all/time"));
    REQUIRE_EQ(std::string(path::position_value),
               std::string("/particles/all/position/value"));
    REQUIRE_EQ(std::string(path::position_step),
               std::string("/particles/all/position/step"));
    REQUIRE_EQ(std::string(path::position_time),
               std::string("/particles/all/position/time"));
    REQUIRE_EQ(std::string(path::velocity_value),
               std::string("/particles/all/velocity/value"));
    REQUIRE_EQ(std::string(path::velocity_step),
               std::string("/particles/all/velocity/step"));
    REQUIRE_EQ(std::string(path::velocity_time),
               std::string("/particles/all/velocity/time"));
    REQUIRE_EQ(std::string(path::force_value),
               std::string("/particles/all/force/value"));
    REQUIRE_EQ(std::string(path::force_step),
               std::string("/particles/all/force/step"));
    REQUIRE_EQ(std::string(path::force_time),
               std::string("/particles/all/force/time"));
    REQUIRE_EQ(std::string(path::box_edges_value),
               std::string("/particles/all/box/edges/value"));
    REQUIRE_EQ(std::string(path::box_edges_step),
               std::string("/particles/all/box/edges/step"));
    REQUIRE_EQ(std::string(path::box_edges_time),
               std::string("/particles/all/box/edges/time"));

    REQUIRE_EQ(std::string(path::observables_all),
               std::string("/observables/all"));
    REQUIRE_EQ(std::string(path::observables_all_step),
               std::string("/observables/all/step"));
    REQUIRE_EQ(std::string(path::observables_all_time),
               std::string("/observables/all/time"));
    REQUIRE_EQ(Observable_Root("temperature"),
               std::string("/observables/all/temperature"));
    REQUIRE_EQ(Observable_Value_Path("temperature"),
               std::string("/observables/all/temperature/value"));
    REQUIRE_EQ(Observable_Step_Path("temperature"),
               std::string("/observables/all/temperature/step"));
    REQUIRE_EQ(Observable_Time_Path("temperature"),
               std::string("/observables/all/temperature/time"));

    REQUIRE_EQ(std::string(path::run), std::string("/run"));
    REQUIRE_EQ(std::string(path::run_current_step),
               std::string("/run/current_step"));
    REQUIRE_EQ(std::string(path::run_current_time),
               std::string("/run/current_time"));
    REQUIRE_EQ(std::string(path::run_state_type),
               std::string("/run/state_type"));
    REQUIRE_EQ(std::string(path::parameters_restart),
               std::string("/parameters/restart"));
    REQUIRE_EQ(std::string(path::restart_rng_state),
               std::string("/parameters/restart/rng_state"));
    REQUIRE_EQ(std::string(path::restart_integrator_state),
               std::string("/parameters/restart/integrator_state"));
    REQUIRE_EQ(std::string(path::restart_thermostat),
               std::string("/parameters/restart/thermostat"));
    REQUIRE_EQ(std::string(path::restart_nhc),
               std::string("/parameters/restart/thermostat/nose_hoover_chain"));
    REQUIRE_EQ(std::string(path::restart_barostat),
               std::string("/parameters/restart/barostat"));
    REQUIRE_EQ(std::string(path::restart_bias),
               std::string("/parameters/restart/bias"));
    REQUIRE_EQ(std::string(path::restart_sits),
               std::string("/parameters/restart/bias/sits"));
    REQUIRE_EQ(std::string(path::restart_meta),
               std::string("/parameters/restart/bias/meta"));
    REQUIRE_EQ(Restart_Sits_State_Root("sits_a"),
               std::string("/parameters/restart/bias/sits/sits_a"));
    REQUIRE_EQ(Restart_Sits_State_Path("sits_a", "nk"),
               std::string("/parameters/restart/bias/sits/sits_a/nk"));
    REQUIRE_EQ(Restart_Metad_State_Root("meta0"),
               std::string("/parameters/restart/bias/meta/meta0"));
    REQUIRE_EQ(Restart_Metad_State_Path("meta0", "hills"),
               std::string("/parameters/restart/bias/meta/meta0/hills"));
}

static void Test_Writer_Open_Preconditions_Reject_Unbound_Bundles()
{
    {
        auto log = std::make_shared<BackendLog>();
        MockBackend backend(log);
        TrajectoryH5Writer writer(&backend);
        SpongeH5OutputPlan::ResolvedOutputPlan plan;

        REQUIRE_TRUE(!writer.Open_Single_File(plan, "test"));
        REQUIRE_EQ(
            writer.Last_Error(),
            std::string(
                "TrajectoryH5Writer requires enabled non-VDS trajectory plan"));

        plan.trajectory.enabled = true;
        plan.trajectory.vds = true;
        plan.trajectory.path = "prod.spg.h5md";
        REQUIRE_TRUE(!writer.Open_Single_File(plan, "test"));
        REQUIRE_EQ(
            writer.Last_Error(),
            std::string(
                "TrajectoryH5Writer requires enabled non-VDS trajectory plan"));
    }

    {
        auto log = std::make_shared<BackendLog>();
        MockBackend backend(log);
        ObservableH5Writer writer(&backend);
        SpongeH5OutputPlan::ResolvedOutputPlan plan;
        plan.trajectory.enabled = true;
        plan.trajectory.path = "prod.spg.h5md";

        REQUIRE_TRUE(!writer.Open(plan, "test"));
        REQUIRE_EQ(
            writer.Last_Error(),
            std::string("ObservableH5Writer requires enabled observable plan"));
    }

    {
        auto log = std::make_shared<BackendLog>();
        MockBackend backend(log);
        RestartH5Writer writer(&backend);
        SpongeH5OutputPlan::ResolvedOutputPlan plan;
        plan.observable.enabled = true;
        plan.observable.path = "prod.obs.spg.h5md";

        REQUIRE_TRUE(!writer.Open(plan, "test"));
        REQUIRE_EQ(
            writer.Last_Error(),
            std::string("RestartH5Writer requires enabled restart plan"));
    }
}

static void Test_H5MD_Writer_No_Backend_Is_Safe()
{
    H5MDWriter writer(nullptr);
    WriterOptions options;
    options.path = "missing-backend.spg.h5md";
    options.schema_name = "sponge.output.h5md";
    options.schema_version = "test";

    int64_t step = 1;
    double time = 0.1;

    REQUIRE_TRUE(!writer.Is_Attached());
    REQUIRE_TRUE(!writer.Open(options));
    REQUIRE_TRUE(!writer.Flush());
    REQUIRE_TRUE(!writer.Close());
    REQUIRE_TRUE(!writer.Finalize());
    REQUIRE_TRUE(!writer.Ensure_Group(path::observables_all));
    REQUIRE_TRUE(!writer.Create_Dataset(
        {path::observables_all_step, DataType::int64, {{0}, {0}, {0}}, true}));
    REQUIRE_TRUE(!writer.Create_Virtual_Dataset(
        {path::observables_all_step, DataType::int64, {{0}, {0}, {0}}, false},
        {}));
    REQUIRE_TRUE(!writer.Create_Hard_Link(path::observables_all_step,
                                          path::position_step));
    REQUIRE_TRUE(!writer.Append_Int64(path::observables_all_step, &step, 1));
    REQUIRE_TRUE(!writer.Append_Float64(path::observables_all_time, &time, 1));
    REQUIRE_TRUE(!writer.Write_String(path::sponge_schema_name, "schema"));
    REQUIRE_TRUE(!writer.Write_String_Array(path::mdout_columns_hdf5_name,
                                            {"temperature"}));
    REQUIRE_TRUE(!writer.Set_Status(FileStatus::open));
    REQUIRE_TRUE(!writer.Mark_Failed("failure"));
    REQUIRE_TRUE(!writer.Write_Output_Completion(1, step, time));
    REQUIRE_EQ(writer.Status(), FileStatus::closed);
    REQUIRE_EQ(writer.Last_Error(),
               std::string("H5MD writer backend is not attached"));
}

static void Test_Common_Layout_Roots_And_Output_Metadata()
{
    {
        auto log = std::make_shared<BackendLog>();
        MockBackend backend(log);
        H5MDWriter writer(&backend);

        WriterOptions options;
        options.path = "trajectory.spg.h5md";
        options.schema_name = "sponge.output.h5md";
        options.schema_version = "test";
        options.observable_only = false;
        REQUIRE_TRUE(writer.Open(options));

        REQUIRE_TRUE(log->groups.count(path::h5md) != 0);
        REQUIRE_TRUE(log->groups.count(path::particles) != 0);
        REQUIRE_TRUE(log->groups.count(path::observables) != 0);
        REQUIRE_TRUE(log->groups.count(path::parameters) != 0);
        REQUIRE_TRUE(log->groups.count(path::sponge) != 0);
        REQUIRE_TRUE(log->groups.count(path::sponge_schema) != 0);
        REQUIRE_TRUE(log->groups.count(path::sponge_output) != 0);
        REQUIRE_EQ(log->strings[path::sponge_schema_name],
                   std::string("sponge.output.h5md"));
        REQUIRE_EQ(log->strings[path::sponge_schema_version],
                   std::string("test"));
        REQUIRE_EQ(log->strings[path::output_status], std::string("open"));
        REQUIRE_EQ(log->append_counts[path::output_frame_count],
                   static_cast<int64_t>(1));
        REQUIRE_EQ(log->append_counts[path::output_last_complete_step],
                   static_cast<int64_t>(1));
        REQUIRE_EQ(log->append_counts[path::output_last_complete_time],
                   static_cast<int64_t>(1));
    }
    {
        auto log = std::make_shared<BackendLog>();
        MockBackend backend(log);
        H5MDWriter writer(&backend);

        WriterOptions options;
        options.path = "observable.obs.spg.h5md";
        options.schema_name = "sponge.output.h5md";
        options.schema_version = "test";
        options.observable_only = true;
        REQUIRE_TRUE(writer.Open(options));

        REQUIRE_TRUE(log->groups.count(path::h5md) != 0);
        REQUIRE_TRUE(log->groups.count(path::particles) == 0);
        REQUIRE_TRUE(log->groups.count(path::observables) != 0);
        REQUIRE_TRUE(log->groups.count(path::parameters) != 0);
        REQUIRE_TRUE(log->groups.count(path::sponge) != 0);
        REQUIRE_TRUE(log->groups.count(path::sponge_schema) != 0);
        REQUIRE_TRUE(log->groups.count(path::sponge_output) != 0);
        REQUIRE_EQ(log->strings[path::sponge_schema_name],
                   std::string("sponge.output.h5md"));
        REQUIRE_EQ(log->strings[path::sponge_schema_version],
                   std::string("test"));
        REQUIRE_EQ(log->strings[path::output_status], std::string("open"));
        REQUIRE_EQ(log->append_counts[path::output_frame_count],
                   static_cast<int64_t>(1));
        REQUIRE_EQ(log->append_counts[path::output_last_complete_step],
                   static_cast<int64_t>(1));
        REQUIRE_EQ(log->append_counts[path::output_last_complete_time],
                   static_cast<int64_t>(1));
    }
}

static void Test_Trajectory_Writer_Paths_And_Completion()
{
    auto log = std::make_shared<BackendLog>();
    MockBackend backend(log);
    TrajectoryH5Writer writer(&backend);
    auto plan = Make_Plan();

    REQUIRE_TRUE(writer.Open_Single_File(plan, "test"));
    REQUIRE_TRUE(writer.Define_Particle_Datasets(2, true, true));
    REQUIRE_TRUE(writer.Define_Observable_Stream({"temperature", "pressure"},
                                                 {"TEMP", "PRESS"}));

    float position[6] = {0, 1, 2, 3, 4, 5};
    float velocity[6] = {1, 1, 1, 2, 2, 2};
    float force[6] = {3, 3, 3, 4, 4, 4};
    float box[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    REQUIRE_TRUE(
        writer.Append_Particle_Frame(10, 0.5, position, box, velocity, force));
    REQUIRE_TRUE(writer.Append_Observable_Frame(
        10, 0.5, {{"temperature", 300.0}, {"pressure", 1.0}}));
    REQUIRE_TRUE(writer.Finalize());

    REQUIRE_TRUE(log->datasets.count(path::position_value) != 0);
    REQUIRE_TRUE(log->datasets.count(path::velocity_value) != 0);
    REQUIRE_TRUE(log->datasets.count(path::force_value) != 0);
    REQUIRE_TRUE(log->datasets.count(path::box_edges_value) != 0);
    REQUIRE_TRUE(log->datasets.count(Observable_Value_Path("temperature")) !=
                 0);
    Require_Dataset_Spec(*log, path::particles_all_step, DataType::int64, {0},
                         {0}, {0}, true);
    Require_Dataset_Spec(*log, path::particles_all_time, DataType::float64, {0},
                         {0}, {0}, true);
    Require_Dataset_Spec(*log, path::position_value, DataType::float32,
                         {0, 2, 3}, {0, 0, 0}, {0, 2, 3}, true);
    Require_Dataset_Spec(*log, path::velocity_value, DataType::float32,
                         {0, 2, 3}, {0, 0, 0}, {0, 2, 3}, true);
    Require_Dataset_Spec(*log, path::force_value, DataType::float32, {0, 2, 3},
                         {0, 0, 0}, {0, 2, 3}, true);
    Require_Dataset_Spec(*log, path::box_edges_value, DataType::float32,
                         {0, 3, 3}, {0, 0, 0}, {0, 3, 3}, true);
    Require_Dataset_Spec(*log, path::observables_all_step, DataType::int64, {0},
                         {0}, {0}, true);
    Require_Dataset_Spec(*log, path::observables_all_time, DataType::float64,
                         {0}, {0}, {0}, true);
    Require_Dataset_Spec(*log, Observable_Value_Path("temperature"),
                         DataType::float64, {0}, {0}, {0}, true);
    REQUIRE_TRUE(
        Has_Hard_Link(*log, path::particles_all_step, path::position_step));
    REQUIRE_TRUE(
        Has_Hard_Link(*log, path::particles_all_time, path::position_time));
    REQUIRE_TRUE(
        Has_Hard_Link(*log, path::particles_all_step, path::velocity_step));
    REQUIRE_TRUE(
        Has_Hard_Link(*log, path::particles_all_time, path::velocity_time));
    REQUIRE_TRUE(
        Has_Hard_Link(*log, path::particles_all_step, path::force_step));
    REQUIRE_TRUE(
        Has_Hard_Link(*log, path::particles_all_time, path::force_time));
    REQUIRE_TRUE(
        Has_Hard_Link(*log, path::particles_all_step, path::box_edges_step));
    REQUIRE_TRUE(
        Has_Hard_Link(*log, path::particles_all_time, path::box_edges_time));
    REQUIRE_TRUE(Has_Hard_Link(*log, path::observables_all_step,
                               Observable_Step_Path("temperature")));
    REQUIRE_TRUE(Has_Hard_Link(*log, path::observables_all_time,
                               Observable_Time_Path("temperature")));
    REQUIRE_EQ(log->append_counts[path::position_value], 6);
    REQUIRE_EQ(log->append_counts[path::box_edges_value], 9);
    REQUIRE_EQ(log->append_counts[Observable_Value_Path("temperature")], 1);
    REQUIRE_EQ(
        log->append_counts[Output_Stream_Committed_Count_Path("particles")],
        static_cast<int64_t>(1));
    REQUIRE_EQ(
        log->append_counts[Output_Stream_Committed_Count_Path("observables")],
        static_cast<int64_t>(1));
    REQUIRE_EQ(log->append_counts[path::output_publication_epoch],
               static_cast<int64_t>(3));
    REQUIRE_EQ(log->status, FileStatus::finalized);
}

static void Test_Trajectory_Optional_Velocity_And_Force_Paths()
{
    auto log = std::make_shared<BackendLog>();
    MockBackend backend(log);
    TrajectoryH5Writer writer(&backend);
    auto plan = Make_Plan();

    REQUIRE_TRUE(writer.Open_Single_File(plan, "test"));
    REQUIRE_TRUE(writer.Define_Particle_Datasets(1, false, false));

    float position[3] = {0, 1, 2};
    float box[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    REQUIRE_TRUE(writer.Append_Particle_Frame(10, 0.5, position, box));

    REQUIRE_TRUE(log->datasets.count(path::position_value) != 0);
    REQUIRE_TRUE(log->datasets.count(path::box_edges_value) != 0);
    REQUIRE_TRUE(log->datasets.count(path::velocity_value) == 0);
    REQUIRE_TRUE(log->datasets.count(path::force_value) == 0);
    REQUIRE_TRUE(
        Has_Hard_Link(*log, path::particles_all_step, path::position_step));
    REQUIRE_TRUE(
        Has_Hard_Link(*log, path::particles_all_time, path::position_time));
    REQUIRE_TRUE(
        Has_Hard_Link(*log, path::particles_all_step, path::box_edges_step));
    REQUIRE_TRUE(
        Has_Hard_Link(*log, path::particles_all_time, path::box_edges_time));
    REQUIRE_TRUE(
        !Has_Hard_Link(*log, path::particles_all_step, path::velocity_step));
    REQUIRE_TRUE(
        !Has_Hard_Link(*log, path::particles_all_time, path::velocity_time));
    REQUIRE_TRUE(
        !Has_Hard_Link(*log, path::particles_all_step, path::force_step));
    REQUIRE_TRUE(
        !Has_Hard_Link(*log, path::particles_all_time, path::force_time));
    REQUIRE_EQ(log->append_counts[path::position_value],
               static_cast<int64_t>(3));
    REQUIRE_EQ(log->append_counts[path::box_edges_value],
               static_cast<int64_t>(9));
    REQUIRE_EQ(log->append_counts[path::velocity_value],
               static_cast<int64_t>(0));
    REQUIRE_EQ(log->append_counts[path::force_value], static_cast<int64_t>(0));
}

static void Test_Bundled_Writers_Publish_With_Unified_Flushes()
{
    {
        auto log = std::make_shared<BackendLog>();
        MockBackend backend(log);
        TrajectoryH5Writer writer(&backend);
        auto plan = Make_Plan();
        REQUIRE_TRUE(writer.Open_Single_File(plan, "test"));
        REQUIRE_TRUE(writer.Define_Particle_Datasets(1, false, false));
        REQUIRE_TRUE(writer.Define_Observable_Stream({"energy"}, {"E"}));

        float position[3] = {0, 1, 2};
        float box[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        REQUIRE_TRUE(
            writer.Append_Observable_Frame(10, 0.5, {{"energy", -1.0}}));
        REQUIRE_TRUE(writer.Append_Particle_Frame(10, 0.5, position, box));
        REQUIRE_EQ(log->flush_calls, static_cast<int64_t>(0));
        REQUIRE_EQ(log->append_counts[path::output_frame_count],
                   static_cast<int64_t>(1));

        REQUIRE_TRUE(writer.Publish());
        REQUIRE_EQ(log->flush_calls, static_cast<int64_t>(2));
        REQUIRE_EQ(log->append_counts[path::output_frame_count],
                   static_cast<int64_t>(2));
        REQUIRE_TRUE(writer.Publish());
        REQUIRE_EQ(log->flush_calls, static_cast<int64_t>(2));
    }
    {
        auto log = std::make_shared<BackendLog>();
        MockBackend backend(log);
        ObservableH5Writer writer(&backend);
        auto plan = Make_Plan();
        REQUIRE_TRUE(writer.Open(plan, "test"));
        REQUIRE_TRUE(writer.Define_Observable_Stream({"energy"}, {"E"}));
        REQUIRE_TRUE(
            writer.Append_Observable_Frame(10, 0.5, {{"energy", -1.0}}));
        REQUIRE_EQ(log->flush_calls, static_cast<int64_t>(0));

        REQUIRE_TRUE(writer.Publish());
        REQUIRE_EQ(log->flush_calls, static_cast<int64_t>(2));
        REQUIRE_EQ(log->append_counts[path::output_frame_count],
                   static_cast<int64_t>(2));
    }
}

static void Test_Legacy_Output_Flush_Coordinator()
{
    FILE* file = std::tmpfile();
    REQUIRE_TRUE(file != nullptr);
    REQUIRE_TRUE(std::fputs("legacy frame\n", file) >= 0);
    SpongeLegacyIO::OutputFlushCoordinator::Mark_Dirty(file, "test stream");
    SpongeLegacyIO::OutputFlushCoordinator::Mark_Dirty(file, "test stream");
    REQUIRE_EQ(SpongeLegacyIO::OutputFlushCoordinator::Dirty_Count(),
               static_cast<std::size_t>(1));
    std::string error_message;
    REQUIRE_TRUE(
        SpongeLegacyIO::OutputFlushCoordinator::Flush_Dirty(&error_message));
    REQUIRE_TRUE(error_message.empty());
    REQUIRE_EQ(SpongeLegacyIO::OutputFlushCoordinator::Dirty_Count(),
               static_cast<std::size_t>(0));
    REQUIRE_EQ(std::fclose(file), 0);
}

static void Test_Trajectory_And_Observable_Base_Layout_Paths()
{
    {
        auto log = std::make_shared<BackendLog>();
        MockBackend backend(log);
        TrajectoryH5Writer writer(&backend);
        auto plan = Make_Plan();

        REQUIRE_TRUE(writer.Open_Single_File(plan, "test"));

        REQUIRE_TRUE(log->groups.count(path::particles_all) != 0);
        REQUIRE_TRUE(log->groups.count(path::particles_all_position) != 0);
        REQUIRE_TRUE(log->groups.count(path::particles_all_velocity) != 0);
        REQUIRE_TRUE(log->groups.count(path::particles_all_force) != 0);
        REQUIRE_TRUE(log->groups.count(path::particles_all_box) != 0);
        REQUIRE_TRUE(log->groups.count(path::particles_all_box_edges) != 0);
        REQUIRE_TRUE(log->groups.count(path::observables_all) != 0);
        REQUIRE_TRUE(log->groups.count(path::sponge_mdout) != 0);
        REQUIRE_TRUE(log->groups.count(path::mdout_columns) != 0);
        REQUIRE_TRUE(log->groups.count(path::sponge_log) != 0);
        REQUIRE_EQ(log->strings[path::sponge_schema_name],
                   std::string("sponge.output.h5md"));
        REQUIRE_EQ(log->strings[path::sponge_schema_version],
                   std::string("test"));
    }
    {
        auto log = std::make_shared<BackendLog>();
        MockBackend backend(log);
        ObservableH5Writer writer(&backend);
        auto plan = Make_Plan();

        REQUIRE_TRUE(writer.Open(plan, "test"));

        REQUIRE_TRUE(log->groups.count("/particles") == 0);
        REQUIRE_TRUE(log->groups.count(path::observables_all) != 0);
        REQUIRE_TRUE(log->groups.count(path::sponge_mdout) != 0);
        REQUIRE_TRUE(log->groups.count(path::mdout_columns) != 0);
        REQUIRE_TRUE(log->groups.count(path::sponge_log) != 0);
        REQUIRE_EQ(log->strings[path::sponge_schema_name],
                   std::string("sponge.output.h5md"));
        REQUIRE_EQ(log->strings[path::sponge_schema_version],
                   std::string("test"));
    }
}

static void Test_Observable_Only_Writer()
{
    auto log = std::make_shared<BackendLog>();
    MockBackend backend(log);
    ObservableH5Writer writer(&backend);
    auto plan = Make_Plan();

    REQUIRE_TRUE(writer.Open(plan, "test"));
    REQUIRE_TRUE(writer.Define_Observable_Stream({"energy", "temperature"},
                                                 {"E", "TEMP"}));
    REQUIRE_TRUE(writer.Append_Observable_Frame(
        2, 0.1, {{"energy", -3.0}, {"temperature", 300.0}}));
    REQUIRE_TRUE(writer.Write_Mdinfo_Text("OBS MDINFO"));
    REQUIRE_TRUE(writer.Write_Legacy_Sidecar_Paths(
        {"mdout", "metad_hills"}, {"legacy.mdout", "legacy.myhill"}));
    REQUIRE_TRUE(writer.Write_Provenance_String("launch_id", "obs-launch"));
    REQUIRE_TRUE(writer.Finalize());

    REQUIRE_TRUE(log->groups.count("/particles") == 0);
    REQUIRE_TRUE(log->datasets.count(Observable_Value_Path("energy")) != 0);
    REQUIRE_TRUE(log->datasets.count(Observable_Value_Path("temperature")) !=
                 0);
    Require_Dataset_Spec(*log, path::observables_all_step, DataType::int64, {0},
                         {0}, {0}, true);
    Require_Dataset_Spec(*log, path::observables_all_time, DataType::float64,
                         {0}, {0}, {0}, true);
    Require_Dataset_Spec(*log, Observable_Value_Path("energy"),
                         DataType::float64, {0}, {0}, {0}, true);
    Require_Dataset_Spec(*log, Observable_Value_Path("temperature"),
                         DataType::float64, {0}, {0}, {0}, true);
    REQUIRE_TRUE(Has_Hard_Link(*log, path::observables_all_step,
                               Observable_Step_Path("energy")));
    REQUIRE_TRUE(Has_Hard_Link(*log, path::observables_all_time,
                               Observable_Time_Path("energy")));
    REQUIRE_TRUE(Has_Hard_Link(*log, path::observables_all_step,
                               Observable_Step_Path("temperature")));
    REQUIRE_TRUE(Has_Hard_Link(*log, path::observables_all_time,
                               Observable_Time_Path("temperature")));
    REQUIRE_EQ(log->append_counts[path::observables_all_step],
               static_cast<int64_t>(1));
    REQUIRE_EQ(log->append_counts[path::observables_all_time],
               static_cast<int64_t>(1));
    REQUIRE_EQ(log->append_counts[Observable_Value_Path("energy")], 1);
    REQUIRE_EQ(log->append_counts[Observable_Value_Path("temperature")], 1);
    REQUIRE_EQ(log->append_counts[path::output_frame_count], 2);
    REQUIRE_EQ(
        log->append_counts[Output_Stream_Committed_Count_Path("observables")],
        static_cast<int64_t>(1));
    REQUIRE_EQ(log->append_counts[path::output_publication_epoch],
               static_cast<int64_t>(3));
    REQUIRE_EQ(log->string_arrays[path::mdout_columns_original_name][1],
               std::string("TEMP"));
    REQUIRE_EQ(log->string_arrays[path::mdout_columns_hdf5_name][0],
               std::string("energy"));
    REQUIRE_EQ(log->strings[path::mdinfo_text], std::string("OBS MDINFO"));
    REQUIRE_EQ(log->string_arrays[path::legacy_sidecar_keys][1],
               std::string("metad_hills"));
    REQUIRE_EQ(log->string_arrays[path::legacy_sidecar_paths][0],
               std::string("legacy.mdout"));
    REQUIRE_TRUE(log->groups.count(path::sponge_provenance) != 0);
    REQUIRE_EQ(log->strings[Sponge_Provenance_Path("launch_id")],
               std::string("obs-launch"));
    REQUIRE_EQ(log->status, FileStatus::finalized);
}

static void Test_Observable_Only_Module_Proxy_Paths()
{
    auto log = std::make_shared<BackendLog>();
    MockBackend backend(log);
    ObservableH5Writer writer(&backend);
    auto plan = Make_Plan();

    REQUIRE_TRUE(writer.Open(plan, "test"));
    REQUIRE_TRUE(writer.Ensure_Nose_Hoover_Chain_Observables(2));
    REQUIRE_TRUE(writer.Ensure_Sits_Nk_Observable("obs_sits", 3));
    REQUIRE_TRUE(writer.Ensure_Metadynamics_Scalars());
    REQUIRE_TRUE(writer.Ensure_Reaxff_Energy_Terms({"bond"}));

    float nhc[2] = {0.1f, 0.2f};
    float sits[3] = {1.0f, 2.0f, 3.0f};
    REQUIRE_TRUE(writer.Append_Nose_Hoover_Chain_Frame(3, 0.15, nhc, nhc, 2));
    REQUIRE_TRUE(writer.Append_Sits_Nk_Frame(3, 0.15, "obs_sits", sits, 3));
    REQUIRE_TRUE(
        writer.Append_Metadynamics_Scalar_Frame(3, 0.15, 1.0, 2.0, 3.0));
    REQUIRE_TRUE(writer.Append_Reaxff_Frame(3, 0.15, {{"bond", 5.0}}));
    REQUIRE_TRUE(
        writer.Write_Metadynamics_Diagnostic("meta0", "hills", "HILLS"));

    REQUIRE_TRUE(log->groups.count("/particles") == 0);
    REQUIRE_TRUE(log->datasets.count(module_path::nhc_coordinate_value) != 0);
    REQUIRE_TRUE(log->datasets.count(module_path::nhc_velocity_value) != 0);
    REQUIRE_TRUE(log->datasets.count(Sits_Nk_Value_Path("obs_sits")) != 0);
    REQUIRE_TRUE(log->datasets.count(Sits_Nk_Step_Path("obs_sits")) != 0);
    REQUIRE_TRUE(log->datasets.count(Sits_Nk_Time_Path("obs_sits")) != 0);
    REQUIRE_TRUE(log->datasets.count(Metadynamics_Scalar_Value_Path("meta")) !=
                 0);
    REQUIRE_TRUE(log->datasets.count(Metadynamics_Scalar_Value_Path("rbias")) !=
                 0);
    REQUIRE_TRUE(log->datasets.count(Metadynamics_Scalar_Value_Path("rct")) !=
                 0);
    REQUIRE_TRUE(log->datasets.count(Reaxff_Term_Value_Path("bond")) != 0);
    REQUIRE_TRUE(Has_Hard_Link(*log, module_path::nhc_step,
                               module_path::nhc_coordinate_step));
    REQUIRE_TRUE(Has_Hard_Link(*log, module_path::nhc_time,
                               module_path::nhc_coordinate_time));
    REQUIRE_TRUE(Has_Hard_Link(*log, module_path::metad_step,
                               Metadynamics_Scalar_Step_Path("rbias")));
    REQUIRE_TRUE(Has_Hard_Link(*log, module_path::reaxff_step,
                               Reaxff_Term_Step_Path("bond")));
    REQUIRE_EQ(log->append_counts[module_path::nhc_coordinate_value],
               static_cast<int64_t>(2));
    REQUIRE_EQ(log->append_counts[Sits_Nk_Value_Path("obs_sits")],
               static_cast<int64_t>(3));
    REQUIRE_EQ(log->append_counts[Sits_Nk_Step_Path("obs_sits")],
               static_cast<int64_t>(1));
    REQUIRE_EQ(log->append_counts[Sits_Nk_Time_Path("obs_sits")],
               static_cast<int64_t>(1));
    REQUIRE_EQ(log->strings[Metadynamics_Diagnostic_Path("meta0", "hills")],
               std::string("HILLS"));
}

static void Test_Trajectory_Append_Failure_Marks_Failed()
{
    auto log = std::make_shared<BackendLog>();
    MockBackend backend(log);
    TrajectoryH5Writer writer(&backend);
    auto plan = Make_Plan();

    REQUIRE_TRUE(writer.Open_Single_File(plan, "test"));
    REQUIRE_TRUE(writer.Define_Particle_Datasets(1, false, false));

    float position[3] = {0, 1, 2};
    float box[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    log->fail_next_append = true;
    REQUIRE_TRUE(!writer.Append_Particle_Frame(1, 0.0, position, box));
    REQUIRE_EQ(log->status, FileStatus::failed);
    REQUIRE_EQ(log->strings[path::output_status], std::string("failed"));
    REQUIRE_EQ(log->strings[path::output_error],
               std::string("mock append failure"));
}

static void Test_Trajectory_Observable_Missing_Value_Does_Not_Advance()
{
    auto log = std::make_shared<BackendLog>();
    MockBackend backend(log);
    TrajectoryH5Writer writer(&backend);
    auto plan = Make_Plan();

    REQUIRE_TRUE(writer.Open_Single_File(plan, "test"));
    REQUIRE_TRUE(writer.Define_Observable_Stream({"energy", "temperature"},
                                                 {"E", "TEMP"}));
    REQUIRE_TRUE(!writer.Append_Observable_Frame(1, 0.0, {{"energy", 1.0}}));
    REQUIRE_EQ(writer.Observable_Frame_Count(), static_cast<std::size_t>(0));
    REQUIRE_EQ(writer.Last_Error(),
               std::string("observable value is missing: temperature"));
    REQUIRE_EQ(log->status, FileStatus::open);
    REQUIRE_EQ(log->strings[path::output_error], std::string(""));
}

static void Test_Observable_Missing_Value_Marks_Failed()
{
    auto log = std::make_shared<BackendLog>();
    MockBackend backend(log);
    ObservableH5Writer writer(&backend);
    auto plan = Make_Plan();

    REQUIRE_TRUE(writer.Open(plan, "test"));
    REQUIRE_TRUE(writer.Define_Observable_Stream({"energy", "temperature"},
                                                 {"E", "TEMP"}));
    REQUIRE_TRUE(!writer.Append_Observable_Frame(1, 0.0, {{"energy", 1.0}}));
    REQUIRE_EQ(log->status, FileStatus::failed);
    REQUIRE_EQ(log->strings[path::output_status], std::string("failed"));
    REQUIRE_EQ(log->strings[path::output_error],
               std::string("observable value is missing: temperature"));
}

static void Test_Restart_Writer_Is_Single_State()
{
    auto log = std::make_shared<BackendLog>();
    MockBackend backend(log);
    RestartH5Writer writer(&backend);
    auto plan = Make_Plan();

    REQUIRE_TRUE(writer.Open(plan, "test"));
    REQUIRE_TRUE(writer.Write_Restart_Generation(7));
    REQUIRE_TRUE(writer.Define_Structural_State(2, true));

    float position[6] = {0, 1, 2, 3, 4, 5};
    float velocity[6] = {1, 1, 1, 2, 2, 2};
    float box[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    REQUIRE_TRUE(
        writer.Write_Structural_State(20, 1.0, position, box, velocity));
    REQUIRE_TRUE(writer.State_Written());
    REQUIRE_TRUE(
        !writer.Write_Structural_State(21, 1.1, position, box, velocity));

    REQUIRE_TRUE(log->datasets.count(path::position_value) != 0);
    REQUIRE_TRUE(log->datasets.count(path::velocity_value) != 0);
    REQUIRE_TRUE(log->datasets.count(path::run_current_step) != 0);
    Require_Dataset_Spec(*log, path::output_restart_generation, DataType::int64,
                         {0}, {1}, {1}, true);
    REQUIRE_EQ(log->append_counts[path::output_restart_generation],
               static_cast<int64_t>(1));
    Require_Dataset_Spec(*log, path::particles_all_step, DataType::int64, {0},
                         {1}, {1}, true);
    Require_Dataset_Spec(*log, path::particles_all_time, DataType::float64, {0},
                         {1}, {1}, true);
    Require_Dataset_Spec(*log, path::position_value, DataType::float32,
                         {0, 2, 3}, {1, 2, 3}, {1, 2, 3}, true);
    Require_Dataset_Spec(*log, path::velocity_value, DataType::float32,
                         {0, 2, 3}, {1, 2, 3}, {1, 2, 3}, true);
    Require_Dataset_Spec(*log, path::box_edges_value, DataType::float32,
                         {0, 3, 3}, {1, 3, 3}, {1, 3, 3}, true);
    Require_Dataset_Spec(*log, path::run_current_step, DataType::int64, {0},
                         {1}, {1}, true);
    Require_Dataset_Spec(*log, path::run_current_time, DataType::float64, {0},
                         {1}, {1}, true);
    REQUIRE_TRUE(
        Has_Hard_Link(*log, path::particles_all_step, path::position_step));
    REQUIRE_TRUE(
        Has_Hard_Link(*log, path::particles_all_time, path::position_time));
    REQUIRE_TRUE(
        Has_Hard_Link(*log, path::particles_all_step, path::velocity_step));
    REQUIRE_TRUE(
        Has_Hard_Link(*log, path::particles_all_time, path::velocity_time));
    REQUIRE_TRUE(
        Has_Hard_Link(*log, path::particles_all_step, path::box_edges_step));
    REQUIRE_TRUE(
        Has_Hard_Link(*log, path::particles_all_time, path::box_edges_time));
    REQUIRE_EQ(log->strings[path::run_state_type], std::string("restart"));
    REQUIRE_EQ(log->status, FileStatus::failed);
    REQUIRE_EQ(log->strings[path::output_status], std::string("failed"));
    REQUIRE_EQ(log->strings[path::output_error],
               std::string("restart H5 already contains one structural state"));
}

static void Test_Restart_Optional_Velocity_Path()
{
    auto log = std::make_shared<BackendLog>();
    MockBackend backend(log);
    RestartH5Writer writer(&backend);
    auto plan = Make_Plan();

    REQUIRE_TRUE(writer.Open(plan, "test"));
    REQUIRE_TRUE(writer.Define_Structural_State(1, false));

    float position[3] = {0, 1, 2};
    float box[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    REQUIRE_TRUE(writer.Write_Structural_State(20, 1.0, position, box));

    REQUIRE_TRUE(writer.State_Written());
    REQUIRE_TRUE(log->datasets.count(path::position_value) != 0);
    REQUIRE_TRUE(log->datasets.count(path::box_edges_value) != 0);
    REQUIRE_TRUE(log->datasets.count(path::velocity_value) == 0);
    REQUIRE_TRUE(
        Has_Hard_Link(*log, path::particles_all_step, path::position_step));
    REQUIRE_TRUE(
        Has_Hard_Link(*log, path::particles_all_time, path::position_time));
    REQUIRE_TRUE(
        Has_Hard_Link(*log, path::particles_all_step, path::box_edges_step));
    REQUIRE_TRUE(
        Has_Hard_Link(*log, path::particles_all_time, path::box_edges_time));
    REQUIRE_TRUE(
        !Has_Hard_Link(*log, path::particles_all_step, path::velocity_step));
    REQUIRE_TRUE(
        !Has_Hard_Link(*log, path::particles_all_time, path::velocity_time));
    REQUIRE_EQ(log->append_counts[path::position_value],
               static_cast<int64_t>(3));
    REQUIRE_EQ(log->append_counts[path::box_edges_value],
               static_cast<int64_t>(9));
    REQUIRE_EQ(log->append_counts[path::velocity_value],
               static_cast<int64_t>(0));
    REQUIRE_EQ(log->strings[path::run_state_type], std::string("restart"));
}

static void Test_Restart_Writer_Base_Layout_Paths()
{
    auto log = std::make_shared<BackendLog>();
    MockBackend backend(log);
    RestartH5Writer writer(&backend);
    auto plan = Make_Plan();

    REQUIRE_TRUE(writer.Open(plan, "test"));

    REQUIRE_TRUE(log->groups.count(path::run) != 0);
    REQUIRE_TRUE(log->groups.count(path::particles_all) != 0);
    REQUIRE_TRUE(log->groups.count(path::particles_all_position) != 0);
    REQUIRE_TRUE(log->groups.count(path::particles_all_velocity) != 0);
    REQUIRE_TRUE(log->groups.count(path::particles_all_box) != 0);
    REQUIRE_TRUE(log->groups.count(path::particles_all_box_edges) != 0);
    REQUIRE_TRUE(log->groups.count(path::parameters_restart) != 0);
    REQUIRE_TRUE(log->groups.count(path::restart_thermostat) != 0);
    REQUIRE_TRUE(log->groups.count(path::restart_barostat) != 0);
    REQUIRE_TRUE(log->groups.count(path::restart_bias) != 0);
    REQUIRE_TRUE(log->groups.count(path::restart_sits) != 0);
    REQUIRE_TRUE(log->groups.count(path::restart_meta) != 0);
    REQUIRE_EQ(log->strings[path::sponge_schema_name],
               std::string("sponge.restart.h5"));
    REQUIRE_EQ(log->strings[path::sponge_schema_version], std::string("test"));
}

static void Test_Restart_Module_State_And_Legacy_Provenance()
{
    auto log = std::make_shared<BackendLog>();
    MockBackend backend(log);
    RestartH5Writer writer(&backend);
    auto plan = Make_Plan();

    REQUIRE_TRUE(writer.Open(plan, "test"));

    float nhc_pairs[4] = {0.1f, 0.2f, 0.3f, 0.4f};
    float sits_state[3] = {1.0f, 2.0f, 3.0f};
    float sits_weight[2] = {4.0f, 5.0f};
    REQUIRE_TRUE(writer.Write_Nose_Hoover_Chain_State(nhc_pairs, 2));
    REQUIRE_TRUE(writer.Write_Sits_State("sits_a", "nk", sits_state, 3));
    REQUIRE_TRUE(writer.Write_Sits_State("sits_a", "weight", sits_weight, 2));
    REQUIRE_TRUE(writer.Write_Metad_State_Text("meta0", "hills", "HILLS"));
    REQUIRE_TRUE(writer.Write_Metad_State_Text("meta0", "history", "HISTORY"));
    REQUIRE_TRUE(writer.Write_Metad_State_Text("meta0", "edge", "EDGE"));
    REQUIRE_TRUE(writer.Write_Metad_State_Text("meta0", "potential_export",
                                               "POTENTIAL"));
    REQUIRE_TRUE(
        writer.Write_Metad_State_Text("meta0", "direct_export", "DIRECT"));
    REQUIRE_TRUE(writer.Write_Legacy_Sidecar_Paths(
        {"rst", "mdout"}, {"legacy.rst", "legacy.out"}));

    REQUIRE_TRUE(log->datasets.count(path::restart_nhc) != 0);
    REQUIRE_TRUE(log->datasets.count(Restart_Sits_State_Path("sits_a", "nk")) !=
                 0);
    REQUIRE_TRUE(
        log->datasets.count(Restart_Sits_State_Path("sits_a", "weight")) != 0);
    Require_Dataset_Spec(*log, path::restart_nhc, DataType::float32, {0, 2},
                         {2, 2}, {2, 2}, true);
    Require_Dataset_Spec(*log, Restart_Sits_State_Path("sits_a", "nk"),
                         DataType::float32, {0}, {3}, {3}, true);
    Require_Dataset_Spec(*log, Restart_Sits_State_Path("sits_a", "weight"),
                         DataType::float32, {0}, {2}, {2}, true);
    REQUIRE_TRUE(log->groups.count(Restart_Sits_State_Root("sits_a")) != 0);
    REQUIRE_TRUE(log->groups.count(Restart_Metad_State_Root("meta0")) != 0);
    REQUIRE_EQ(log->append_counts[path::restart_nhc], static_cast<int64_t>(4));
    REQUIRE_EQ(log->append_counts[Restart_Sits_State_Path("sits_a", "nk")],
               static_cast<int64_t>(3));
    REQUIRE_EQ(log->append_counts[Restart_Sits_State_Path("sits_a", "weight")],
               static_cast<int64_t>(2));
    REQUIRE_EQ(log->strings[Restart_Metad_State_Path("meta0", "hills")],
               std::string("HILLS"));
    REQUIRE_EQ(log->strings[Restart_Metad_State_Path("meta0", "history")],
               std::string("HISTORY"));
    REQUIRE_EQ(log->strings[Restart_Metad_State_Path("meta0", "edge")],
               std::string("EDGE"));
    REQUIRE_EQ(
        log->strings[Restart_Metad_State_Path("meta0", "potential_export")],
        std::string("POTENTIAL"));
    REQUIRE_EQ(log->strings[Restart_Metad_State_Path("meta0", "direct_export")],
               std::string("DIRECT"));
    REQUIRE_EQ(log->string_arrays[path::legacy_sidecar_keys].size(),
               static_cast<std::size_t>(2));
    REQUIRE_EQ(log->string_arrays[path::legacy_sidecar_paths][1],
               std::string("legacy.out"));
    REQUIRE_TRUE(log->groups.count(path::sponge_files) != 0);
    REQUIRE_TRUE(log->groups.count(path::legacy_sidecars) != 0);
}

static void Test_Legacy_Provenance_On_Trajectory_And_Observable()
{
    {
        auto log = std::make_shared<BackendLog>();
        MockBackend backend(log);
        TrajectoryH5Writer writer(&backend);
        auto plan = Make_Plan();

        REQUIRE_TRUE(writer.Open_Single_File(plan, "test"));
        REQUIRE_TRUE(writer.Write_Legacy_Sidecar_Paths(
            {"crd", "box"}, {"legacy.crd", "legacy.box"}));
        REQUIRE_EQ(log->string_arrays[path::legacy_sidecar_keys][0],
                   std::string("crd"));
        REQUIRE_EQ(log->string_arrays[path::legacy_sidecar_paths][1],
                   std::string("legacy.box"));
        REQUIRE_TRUE(log->groups.count(path::sponge_files) != 0);
        REQUIRE_TRUE(log->groups.count(path::legacy_sidecars) != 0);
    }
    {
        auto log = std::make_shared<BackendLog>();
        MockBackend backend(log);
        ObservableH5Writer writer(&backend);
        auto plan = Make_Plan();

        REQUIRE_TRUE(writer.Open(plan, "test"));
        REQUIRE_TRUE(
            writer.Write_Legacy_Sidecar_Paths({"mdout"}, {"legacy.out"}));
        REQUIRE_EQ(log->string_arrays[path::legacy_sidecar_keys][0],
                   std::string("mdout"));
        REQUIRE_EQ(log->string_arrays[path::legacy_sidecar_paths][0],
                   std::string("legacy.out"));
        REQUIRE_TRUE(log->groups.count(path::sponge_files) != 0);
        REQUIRE_TRUE(log->groups.count(path::legacy_sidecars) != 0);
    }
}

static void Test_Writer_Open_Preconditions()
{
    {
        auto log = std::make_shared<BackendLog>();
        MockBackend backend(log);
        TrajectoryH5Writer writer(&backend);
        auto plan = Make_Plan();
        plan.trajectory.enabled = false;
        REQUIRE_TRUE(!writer.Open_Single_File(plan, "test"));
        REQUIRE_TRUE(writer.Last_Error().find("enabled non-VDS trajectory") !=
                     std::string::npos);
    }
    {
        auto log = std::make_shared<BackendLog>();
        MockBackend backend(log);
        TrajectoryH5Writer writer(&backend);
        auto plan = Make_Plan();
        plan.trajectory.vds = true;
        REQUIRE_TRUE(!writer.Open_Single_File(plan, "test"));
        REQUIRE_TRUE(writer.Last_Error().find("enabled non-VDS trajectory") !=
                     std::string::npos);
    }
    {
        auto log = std::make_shared<BackendLog>();
        MockBackend backend(log);
        RestartH5Writer writer(&backend);
        auto plan = Make_Plan();
        plan.restart.enabled = false;
        REQUIRE_TRUE(!writer.Open(plan, "test"));
        REQUIRE_TRUE(writer.Last_Error().find("enabled restart plan") !=
                     std::string::npos);
    }
    {
        auto log = std::make_shared<BackendLog>();
        MockBackend backend(log);
        ObservableH5Writer writer(&backend);
        auto plan = Make_Plan();
        plan.observable.enabled = false;
        REQUIRE_TRUE(!writer.Open(plan, "test"));
        REQUIRE_TRUE(writer.Last_Error().find("enabled observable plan") !=
                     std::string::npos);
    }
}

static void Test_Canonical_Schema_Units_And_Topology_Compatibility()
{
    auto log = std::make_shared<BackendLog>();
    MockBackend backend(log);
    TrajectoryH5Writer writer(&backend);
    auto plan = Make_Plan();

    REQUIRE_TRUE(writer.Open_Single_File(plan));
    REQUIRE_EQ(log->string_attributes.at(
                   std::make_pair(std::string("/h5md/creator"), "name")),
               std::string("SPONGE"));
    REQUIRE_EQ(log->string_attributes.at(
                   std::make_pair(std::string("/h5md/creator"), "version")),
               std::string(kSpongeWriterVersion));
    REQUIRE_EQ(log->strings[path::sponge_schema_version],
               std::string(kCanonicalSchemaVersion));
    REQUIRE_TRUE(writer.Define_Particle_Datasets(2, true, true));
    REQUIRE_EQ(log->string_attributes.at(
                   std::make_pair(path::particles_all_time, "unit")),
               std::string("ps"));
    REQUIRE_EQ(
        log->string_attributes.at(std::make_pair(path::position_value, "unit")),
        std::string("Angstrom"));
    REQUIRE_EQ(log->string_attributes.at(
                   std::make_pair(path::box_edges_value, "unit")),
               std::string("Angstrom"));
    REQUIRE_EQ(
        log->string_attributes.at(std::make_pair(path::velocity_value, "unit")),
        std::string("Angstrom ps-1"));
    REQUIRE_EQ(
        log->string_attributes.at(std::make_pair(path::force_value, "unit")),
        std::string("kcal mol-1 Angstrom-1"));
    REQUIRE_TRUE(writer.Write_Topology_Compatibility("top-hash", "order-hash"));
    REQUIRE_EQ(log->strings[path::sponge_topology_hash],
               std::string("top-hash"));
    REQUIRE_EQ(log->strings[path::sponge_atom_order_hash],
               std::string("order-hash"));
}

int main()
{
    return Run_Test(
        []
        {
            Test_Public_H5MD_Path_Constants();
            Test_Writer_Open_Preconditions_Reject_Unbound_Bundles();
            Test_H5MD_Writer_No_Backend_Is_Safe();
            Test_Common_Layout_Roots_And_Output_Metadata();
            Test_Trajectory_Writer_Paths_And_Completion();
            Test_Trajectory_Optional_Velocity_And_Force_Paths();
            Test_Bundled_Writers_Publish_With_Unified_Flushes();
            Test_Legacy_Output_Flush_Coordinator();
            Test_Trajectory_And_Observable_Base_Layout_Paths();
            Test_Observable_Only_Writer();
            Test_Observable_Only_Module_Proxy_Paths();
            Test_Trajectory_Append_Failure_Marks_Failed();
            Test_Trajectory_Observable_Missing_Value_Does_Not_Advance();
            Test_Observable_Missing_Value_Marks_Failed();
            Test_Restart_Writer_Is_Single_State();
            Test_Restart_Optional_Velocity_Path();
            Test_Restart_Writer_Base_Layout_Paths();
            Test_Restart_Module_State_And_Legacy_Provenance();
            Test_Legacy_Provenance_On_Trajectory_And_Observable();
            Test_Canonical_Schema_Units_And_Topology_Compatibility();
            Test_Writer_Open_Preconditions();
        });
}

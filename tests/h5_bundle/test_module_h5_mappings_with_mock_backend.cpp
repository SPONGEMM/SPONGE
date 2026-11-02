#include "h5_bundle_test_common.hpp"
#include "utils/h5md/module_h5_mappings.hpp"

using namespace SpongeH5Test;
using namespace SpongeH5MD;

static std::unique_ptr<H5MDWriter> Open_Writer(std::shared_ptr<BackendLog> log)
{
    auto backend = std::unique_ptr<MockBackend>(new MockBackend(log));
    auto writer = std::unique_ptr<H5MDWriter>(new H5MDWriter(backend.get()));
    static std::vector<std::unique_ptr<MockBackend>> kept_backends;
    kept_backends.push_back(std::move(backend));
    WriterOptions options;
    options.path = "module.spg.h5md";
    REQUIRE_TRUE(writer->Open(options));
    return writer;
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

static void Test_Module_Path_Constants()
{
    REQUIRE_EQ(std::string(module_path::nhc_root),
               std::string("/observables/all/thermostat/nose_hoover_chain"));
    REQUIRE_EQ(
        std::string(module_path::nhc_step),
        std::string("/observables/all/thermostat/nose_hoover_chain/step"));
    REQUIRE_EQ(
        std::string(module_path::nhc_time),
        std::string("/observables/all/thermostat/nose_hoover_chain/time"));
    REQUIRE_EQ(
        std::string(module_path::nhc_coordinate_value),
        std::string(
            "/observables/all/thermostat/nose_hoover_chain/coordinate/value"));
    REQUIRE_EQ(Nose_Hoover_Chain_Coordinate_Root(),
               std::string(
                   "/observables/all/thermostat/nose_hoover_chain/coordinate"));
    REQUIRE_EQ(
        std::string(module_path::nhc_coordinate_step),
        std::string(
            "/observables/all/thermostat/nose_hoover_chain/coordinate/step"));
    REQUIRE_EQ(
        std::string(module_path::nhc_coordinate_time),
        std::string(
            "/observables/all/thermostat/nose_hoover_chain/coordinate/time"));
    REQUIRE_EQ(
        std::string(module_path::nhc_velocity_value),
        std::string(
            "/observables/all/thermostat/nose_hoover_chain/velocity/value"));
    REQUIRE_EQ(
        Nose_Hoover_Chain_Velocity_Root(),
        std::string("/observables/all/thermostat/nose_hoover_chain/velocity"));
    REQUIRE_EQ(
        std::string(module_path::nhc_velocity_step),
        std::string(
            "/observables/all/thermostat/nose_hoover_chain/velocity/step"));
    REQUIRE_EQ(
        std::string(module_path::nhc_velocity_time),
        std::string(
            "/observables/all/thermostat/nose_hoover_chain/velocity/time"));

    REQUIRE_EQ(std::string(module_path::sits_root),
               std::string("/observables/all/sits"));
    REQUIRE_EQ(Sits_Module_Root("sits_a"),
               std::string("/observables/all/sits/sits_a"));
    REQUIRE_EQ(Sits_Nk_Root("sits_a"),
               std::string("/observables/all/sits/sits_a/nk"));
    REQUIRE_EQ(Scalar_Observable_Value_Path(Sits_Nk_Root("sits_a")),
               std::string("/observables/all/sits/sits_a/nk/value"));
    REQUIRE_EQ(Scalar_Observable_Step_Path(Sits_Nk_Root("sits_a")),
               std::string("/observables/all/sits/sits_a/nk/step"));
    REQUIRE_EQ(Scalar_Observable_Time_Path(Sits_Nk_Root("sits_a")),
               std::string("/observables/all/sits/sits_a/nk/time"));
    REQUIRE_EQ(Sits_Nk_Value_Path("sits_a"),
               std::string("/observables/all/sits/sits_a/nk/value"));
    REQUIRE_EQ(Sits_Nk_Step_Path("sits_a"),
               std::string("/observables/all/sits/sits_a/nk/step"));
    REQUIRE_EQ(Sits_Nk_Time_Path("sits_a"),
               std::string("/observables/all/sits/sits_a/nk/time"));

    REQUIRE_EQ(std::string(module_path::metad_root),
               std::string("/observables/all/metadynamics"));
    REQUIRE_EQ(std::string(module_path::metad_step),
               std::string("/observables/all/metadynamics/step"));
    REQUIRE_EQ(std::string(module_path::metad_time),
               std::string("/observables/all/metadynamics/time"));
    REQUIRE_EQ(std::string(module_path::metad_parameter_root),
               std::string("/parameters/sponge/metadynamics"));
    REQUIRE_EQ(Metadynamics_Scalar_Root("meta"),
               std::string("/observables/all/metadynamics/meta"));
    REQUIRE_EQ(Metadynamics_Scalar_Value_Path("meta"),
               std::string("/observables/all/metadynamics/meta/value"));
    REQUIRE_EQ(Metadynamics_Scalar_Step_Path("meta"),
               std::string("/observables/all/metadynamics/meta/step"));
    REQUIRE_EQ(Metadynamics_Scalar_Time_Path("meta"),
               std::string("/observables/all/metadynamics/meta/time"));
    REQUIRE_EQ(Metadynamics_Diagnostic_Root("meta0"),
               std::string("/parameters/sponge/metadynamics/meta0"));
    REQUIRE_EQ(Metadynamics_Diagnostic_Path("meta0", "hills"),
               std::string("/parameters/sponge/metadynamics/meta0/hills"));

    REQUIRE_EQ(std::string(module_path::reaxff_root),
               std::string("/observables/all/reaxff"));
    REQUIRE_EQ(std::string(module_path::reaxff_step),
               std::string("/observables/all/reaxff/step"));
    REQUIRE_EQ(std::string(module_path::reaxff_time),
               std::string("/observables/all/reaxff/time"));
    REQUIRE_EQ(std::string(module_path::reaxff_parameter_root),
               std::string("/parameters/sponge/reaxff"));
    REQUIRE_EQ(std::string(module_path::reaxff_eeq_charges_value),
               std::string("/parameters/sponge/reaxff/eeq_charges/value"));
    REQUIRE_EQ(Reaxff_Term_Root("bond"),
               std::string("/observables/all/reaxff/bond"));
    REQUIRE_EQ(Reaxff_Term_Value_Path("bond"),
               std::string("/observables/all/reaxff/bond/value"));
    REQUIRE_EQ(Reaxff_Term_Step_Path("bond"),
               std::string("/observables/all/reaxff/bond/step"));
    REQUIRE_EQ(Reaxff_Term_Time_Path("bond"),
               std::string("/observables/all/reaxff/bond/time"));
}

static void Test_Nhc_And_Sits_Mappings()
{
    auto log = std::make_shared<BackendLog>();
    auto writer = Open_Writer(log);
    ModuleH5MappingWriter module(writer.get());

    REQUIRE_TRUE(module.Ensure_Nose_Hoover_Chain_Observables(4));
    REQUIRE_TRUE(module.Ensure_Sits_Nk_Observable("sits_a", 3));
    float nhc[4] = {0, 1, 2, 3};
    float sits[3] = {4, 5, 6};
    REQUIRE_TRUE(module.Append_Nose_Hoover_Chain_Frame(1, 0.1, nhc, nhc, 4));
    REQUIRE_TRUE(module.Append_Sits_Nk_Frame(1, 0.1, "sits_a", sits, 3));

    REQUIRE_TRUE(log->datasets.count(module_path::nhc_coordinate_value) != 0);
    REQUIRE_TRUE(log->datasets.count(module_path::nhc_velocity_value) != 0);
    REQUIRE_TRUE(log->datasets.count(Sits_Nk_Value_Path("sits_a")) != 0);
    REQUIRE_TRUE(log->groups.count(Nose_Hoover_Chain_Coordinate_Root()) != 0);
    REQUIRE_TRUE(log->groups.count(Nose_Hoover_Chain_Velocity_Root()) != 0);
    REQUIRE_TRUE(log->groups.count(Sits_Module_Root("sits_a")) != 0);
    Require_Dataset_Spec(*log, module_path::nhc_step, DataType::int64, {0}, {0},
                         {0}, true);
    Require_Dataset_Spec(*log, module_path::nhc_time, DataType::float64, {0},
                         {0}, {0}, true);
    Require_Dataset_Spec(*log, module_path::nhc_coordinate_value,
                         DataType::float32, {0, 4}, {0, 0}, {0, 4}, true);
    Require_Dataset_Spec(*log, module_path::nhc_velocity_value,
                         DataType::float32, {0, 4}, {0, 0}, {0, 4}, true);
    Require_Dataset_Spec(*log, Sits_Nk_Step_Path("sits_a"), DataType::int64,
                         {0}, {0}, {0}, true);
    Require_Dataset_Spec(*log, Sits_Nk_Time_Path("sits_a"), DataType::float64,
                         {0}, {0}, {0}, true);
    Require_Dataset_Spec(*log, Sits_Nk_Value_Path("sits_a"), DataType::float32,
                         {0, 3}, {0, 0}, {0, 3}, true);
    REQUIRE_TRUE(Has_Hard_Link(*log, module_path::nhc_step,
                               module_path::nhc_coordinate_step));
    REQUIRE_TRUE(Has_Hard_Link(*log, module_path::nhc_time,
                               module_path::nhc_coordinate_time));
    REQUIRE_TRUE(Has_Hard_Link(*log, module_path::nhc_step,
                               module_path::nhc_velocity_step));
    REQUIRE_TRUE(Has_Hard_Link(*log, module_path::nhc_time,
                               module_path::nhc_velocity_time));
    REQUIRE_EQ(log->append_counts[module_path::nhc_coordinate_value],
               static_cast<int64_t>(4));
    REQUIRE_EQ(log->append_counts[Sits_Nk_Value_Path("sits_a")],
               static_cast<int64_t>(3));
}

static void Test_Metadynamics_And_Diagnostics()
{
    auto log = std::make_shared<BackendLog>();
    auto writer = Open_Writer(log);
    ModuleH5MappingWriter module(writer.get());

    REQUIRE_TRUE(module.Ensure_Metadynamics_Scalars());
    REQUIRE_TRUE(
        module.Append_Metadynamics_Scalar_Frame(2, 0.2, 1.0, 2.0, 3.0));
    REQUIRE_TRUE(module.Write_Metadynamics_Hills("meta0", "HILLS"));
    REQUIRE_TRUE(module.Write_Metadynamics_History("meta0", "HISTORY"));
    REQUIRE_TRUE(module.Write_Metadynamics_Edge("meta0", "EDGE"));
    REQUIRE_TRUE(module.Write_Metadynamics_Potential_Export("meta0", "GRID"));
    REQUIRE_TRUE(module.Write_Metadynamics_Direct_Export("meta0", "DIRECT"));

    REQUIRE_TRUE(log->datasets.count(Metadynamics_Scalar_Value_Path("meta")) !=
                 0);
    REQUIRE_TRUE(log->datasets.count(Metadynamics_Scalar_Value_Path("rbias")) !=
                 0);
    REQUIRE_TRUE(log->datasets.count(Metadynamics_Scalar_Value_Path("rct")) !=
                 0);
    Require_Dataset_Spec(*log, module_path::metad_step, DataType::int64, {0},
                         {0}, {0}, true);
    Require_Dataset_Spec(*log, module_path::metad_time, DataType::float64, {0},
                         {0}, {0}, true);
    Require_Dataset_Spec(*log, Metadynamics_Scalar_Value_Path("meta"),
                         DataType::float64, {0}, {0}, {0}, true);
    Require_Dataset_Spec(*log, Metadynamics_Scalar_Value_Path("rbias"),
                         DataType::float64, {0}, {0}, {0}, true);
    Require_Dataset_Spec(*log, Metadynamics_Scalar_Value_Path("rct"),
                         DataType::float64, {0}, {0}, {0}, true);
    REQUIRE_TRUE(Has_Hard_Link(*log, module_path::metad_step,
                               Metadynamics_Scalar_Step_Path("meta")));
    REQUIRE_TRUE(Has_Hard_Link(*log, module_path::metad_time,
                               Metadynamics_Scalar_Time_Path("meta")));
    REQUIRE_TRUE(Has_Hard_Link(*log, module_path::metad_step,
                               Metadynamics_Scalar_Step_Path("rbias")));
    REQUIRE_TRUE(Has_Hard_Link(*log, module_path::metad_time,
                               Metadynamics_Scalar_Time_Path("rbias")));
    REQUIRE_TRUE(Has_Hard_Link(*log, module_path::metad_step,
                               Metadynamics_Scalar_Step_Path("rct")));
    REQUIRE_TRUE(Has_Hard_Link(*log, module_path::metad_time,
                               Metadynamics_Scalar_Time_Path("rct")));
    REQUIRE_EQ(log->append_counts[Metadynamics_Scalar_Value_Path("meta")],
               static_cast<int64_t>(1));
    REQUIRE_EQ(log->strings[Metadynamics_Diagnostic_Path("meta0", "hills")],
               std::string("HILLS"));
    REQUIRE_EQ(log->strings[Metadynamics_Diagnostic_Path("meta0", "history")],
               std::string("HISTORY"));
    REQUIRE_EQ(log->strings[Metadynamics_Diagnostic_Path("meta0", "edge")],
               std::string("EDGE"));
    REQUIRE_EQ(
        log->strings[Metadynamics_Diagnostic_Path("meta0", "potential_export")],
        std::string("GRID"));
    REQUIRE_EQ(
        log->strings[Metadynamics_Diagnostic_Path("meta0", "direct_export")],
        std::string("DIRECT"));
    REQUIRE_TRUE(log->groups.count(module_path::metad_parameter_root) != 0);
    REQUIRE_TRUE(log->groups.count(Metadynamics_Diagnostic_Root("meta0")) != 0);
}

static void Test_Reaxff_Mappings()
{
    auto log = std::make_shared<BackendLog>();
    auto writer = Open_Writer(log);
    ModuleH5MappingWriter module(writer.get());

    REQUIRE_TRUE(module.Ensure_Reaxff_Energy_Terms({"bond", "angle"}));
    REQUIRE_TRUE(
        module.Append_Reaxff_Frame(3, 0.3, {{"bond", 1.0}, {"angle", 2.0}}));
    REQUIRE_TRUE(!module.Append_Reaxff_Frame(4, 0.4, {{"bond", 1.0}}));
    REQUIRE_TRUE(module.Ensure_Reaxff_Eeq_Charge_Snapshot(3));
    const std::vector<float> charges = {-0.25f, 0.5f, -0.25f};
    REQUIRE_TRUE(module.Write_Reaxff_Eeq_Charge_Snapshot(charges.data(),
                                                         charges.size()));

    REQUIRE_TRUE(log->datasets.count(Reaxff_Term_Value_Path("bond")) != 0);
    REQUIRE_TRUE(log->datasets.count(Reaxff_Term_Value_Path("angle")) != 0);
    Require_Dataset_Spec(*log, module_path::reaxff_step, DataType::int64, {0},
                         {0}, {0}, true);
    Require_Dataset_Spec(*log, module_path::reaxff_time, DataType::float64, {0},
                         {0}, {0}, true);
    Require_Dataset_Spec(*log, Reaxff_Term_Value_Path("bond"),
                         DataType::float64, {0}, {0}, {0}, true);
    Require_Dataset_Spec(*log, Reaxff_Term_Value_Path("angle"),
                         DataType::float64, {0}, {0}, {0}, true);
    REQUIRE_TRUE(Has_Hard_Link(*log, module_path::reaxff_step,
                               Reaxff_Term_Step_Path("bond")));
    REQUIRE_TRUE(Has_Hard_Link(*log, module_path::reaxff_time,
                               Reaxff_Term_Time_Path("bond")));
    REQUIRE_TRUE(Has_Hard_Link(*log, module_path::reaxff_step,
                               Reaxff_Term_Step_Path("angle")));
    REQUIRE_TRUE(Has_Hard_Link(*log, module_path::reaxff_time,
                               Reaxff_Term_Time_Path("angle")));
    Require_Dataset_Spec(*log, module_path::reaxff_eeq_charges_value,
                         DataType::float32, {3}, {3}, {3}, false);
    REQUIRE_EQ(log->float32_snapshots[module_path::reaxff_eeq_charges_value],
               charges);
    const auto eeq_unit_key =
        std::make_pair(std::string(module_path::reaxff_eeq_charges_value),
                       std::string("unit"));
    REQUIRE_EQ(log->string_attributes[eeq_unit_key], std::string("e"));
    REQUIRE_EQ(module.Last_Error(), std::string("missing ReaxFF term: angle"));
}

int main()
{
    return Run_Test(
        []
        {
            Test_Module_Path_Constants();
            Test_Nhc_And_Sits_Mappings();
            Test_Metadynamics_And_Diagnostics();
            Test_Reaxff_Mappings();
        });
}

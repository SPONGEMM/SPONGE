#include <highfive/highfive.hpp>
#include <numeric>
#include <type_traits>

#include "h5_bundle_test_common.hpp"
#include "utils/h5md/atomic_file_publish.hpp"
#include "utils/h5md/completion_tracker.hpp"
#include "utils/h5md/highfive_backend.hpp"
#include "utils/h5md/module_h5_mappings.hpp"
#include "utils/h5md/observable_h5_writer.hpp"
#include "utils/h5md/restart_h5_writer.hpp"
#include "utils/h5md/trajectory_h5_writer.hpp"
#include "utils/h5md/vds_trajectory_h5_writer.hpp"

using namespace SpongeH5Test;
using namespace SpongeH5MD;

class MaybeFailFinalizeHighFiveBackend : public HighFiveBackend
{
   public:
    explicit MaybeFailFinalizeHighFiveBackend(bool fail_finalize)
        : fail_finalize_(fail_finalize)
    {
    }

    bool Finalize() override
    {
        if (fail_finalize_)
        {
            return false;
        }
        return HighFiveBackend::Finalize();
    }

   private:
    bool fail_finalize_ = false;
};

class SelectiveFailHighFiveBackendFactory : public WriterBackendFactory
{
   public:
    explicit SelectiveFailHighFiveBackendFactory(
        const std::vector<bool>& fail_finalize_by_creation_index)
        : fail_finalize_by_creation_index_(fail_finalize_by_creation_index)
    {
    }

    std::unique_ptr<WriterBackend> Create_Backend() override
    {
        const bool fail_finalize =
            creation_index_ < fail_finalize_by_creation_index_.size() &&
            fail_finalize_by_creation_index_[creation_index_];
        ++creation_index_;
        return std::unique_ptr<WriterBackend>(
            new MaybeFailFinalizeHighFiveBackend(fail_finalize));
    }

   private:
    std::size_t creation_index_ = 0;
    std::vector<bool> fail_finalize_by_creation_index_;
};

static SpongeH5OutputPlan::ResolvedOutputPlan Make_File_Plan(
    const std::filesystem::path& trajectory_path,
    const std::filesystem::path& restart_path)
{
    SpongeH5OutputPlan::ResolvedOutputPlan plan;
    plan.trajectory.enabled = true;
    plan.trajectory.path = trajectory_path.string();
    plan.restart.enabled = true;
    plan.restart.path = restart_path.string();
    return plan;
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
    else if constexpr (std::is_same<T, int64_t>::value)
    {
        return H5T_NATIVE_INT64;
    }
    else
    {
        static_assert(std::is_same<T, float>::value ||
                          std::is_same<T, double>::value ||
                          std::is_same<T, int64_t>::value,
                      "unsupported test HDF5 read type");
    }
}

template <typename T>
static std::vector<T> Read_Flat_Dataset(HighFive::DataSet dataset)
{
    const auto dims = dataset.getSpace().getDimensions();
    const std::size_t value_count = std::accumulate(
        dims.begin(), dims.end(), static_cast<std::size_t>(1),
        [](std::size_t lhs, std::size_t rhs) { return lhs * rhs; });
    std::vector<T> values(value_count);
    hsize_t mem_dims[1] = {static_cast<hsize_t>(value_count)};
    hid_t mem_space = H5Screate_simple(1, mem_dims, nullptr);
    REQUIRE_TRUE(mem_space >= 0);
    const herr_t rc = H5Dread(dataset.getId(), Native_H5_Type<T>(), mem_space,
                              H5S_ALL, H5P_DEFAULT, values.data());
    H5Sclose(mem_space);
    REQUIRE_TRUE(rc >= 0);
    return values;
}

static std::vector<int64_t> Read_Int64_Vector(HighFive::File& file,
                                              const std::string& path_name)
{
    return Read_Flat_Dataset<int64_t>(file.getDataSet(path_name));
}

static std::vector<double> Read_Float64_Vector(HighFive::File& file,
                                               const std::string& path_name)
{
    return Read_Flat_Dataset<double>(file.getDataSet(path_name));
}

static std::vector<hsize_t> Dataset_Max_Dims(HighFive::DataSet& dataset)
{
    hid_t space = H5Dget_space(dataset.getId());
    REQUIRE_TRUE(space >= 0);
    const int rank = H5Sget_simple_extent_ndims(space);
    REQUIRE_TRUE(rank > 0);
    std::vector<hsize_t> dims(static_cast<std::size_t>(rank), 0);
    std::vector<hsize_t> max_dims(static_cast<std::size_t>(rank), 0);
    REQUIRE_TRUE(
        H5Sget_simple_extent_dims(space, dims.data(), max_dims.data()) >= 0);
    H5Sclose(space);
    return max_dims;
}

static std::vector<hsize_t> Dataset_Chunk_Dims(HighFive::DataSet& dataset)
{
    hid_t plist = H5Dget_create_plist(dataset.getId());
    REQUIRE_TRUE(plist >= 0);
    hid_t space = H5Dget_space(dataset.getId());
    REQUIRE_TRUE(space >= 0);
    const int rank = H5Sget_simple_extent_ndims(space);
    REQUIRE_TRUE(rank > 0);
    H5Sclose(space);
    std::vector<hsize_t> chunk_dims(static_cast<std::size_t>(rank), 0);
    REQUIRE_TRUE(H5Pget_chunk(plist, rank, chunk_dims.data()) >= 0);
    H5Pclose(plist);
    return chunk_dims;
}

static std::string Read_String(HighFive::File& file,
                               const std::string& path_name)
{
    std::string value;
    file.getDataSet(path_name).read(value);
    return value;
}

static std::vector<std::string> Read_String_Vector(HighFive::File& file,
                                                   const std::string& path_name)
{
    std::vector<std::string> values;
    file.getDataSet(path_name).read(values);
    return values;
}

static void Test_H5MD_Writer_Detached_Backend_Semantics()
{
    H5MDWriter writer(nullptr);
    WriterOptions options;
    options.path = "unused.spg.h5md";

    REQUIRE_TRUE(!writer.Is_Attached());
    REQUIRE_TRUE(!writer.Open(options));
    REQUIRE_TRUE(!writer.Flush());
    REQUIRE_TRUE(!writer.Close());
    REQUIRE_TRUE(!writer.Finalize());
    REQUIRE_TRUE(!writer.Mark_Failed("detached"));
    REQUIRE_TRUE(writer.Status() == FileStatus::closed);
    REQUIRE_EQ(writer.Last_Error(),
               std::string("H5MD writer backend is not attached"));
}

static void Test_H5MD_Writer_Repeated_Output_Completion()
{
    const auto dir = Unique_Temp_Path("repeated_completion");
    std::filesystem::create_directories(dir);
    const auto file_path = dir / "completion.spg.h5md";

    {
        HighFiveBackend backend;
        H5MDWriter writer(&backend);
        WriterOptions options;
        options.path = file_path.string();
        REQUIRE_TRUE(writer.Open(options));
        REQUIRE_TRUE(writer.Write_Output_Completion(1, 10, 0.5));
        REQUIRE_TRUE(writer.Write_Output_Completion(2, 20, 1.0));
        REQUIRE_TRUE(writer.Finalize());
        REQUIRE_TRUE(writer.Close());
    }

    {
        HighFive::File file(file_path.string(), HighFive::File::ReadOnly);
        const auto frame_count =
            Read_Int64_Vector(file, path::output_frame_count);
        const auto last_step =
            Read_Int64_Vector(file, path::output_last_complete_step);
        const auto last_time =
            Read_Float64_Vector(file, path::output_last_complete_time);
        REQUIRE_EQ(frame_count.size(), static_cast<std::size_t>(3));
        REQUIRE_EQ(frame_count[0], static_cast<int64_t>(0));
        REQUIRE_EQ(frame_count[1], static_cast<int64_t>(1));
        REQUIRE_EQ(frame_count[2], static_cast<int64_t>(2));
        REQUIRE_EQ(last_step[0], static_cast<int64_t>(-1));
        REQUIRE_EQ(last_step[1], static_cast<int64_t>(10));
        REQUIRE_EQ(last_step[2], static_cast<int64_t>(20));
        REQUIRE_EQ(last_time[0], 0.0);
        REQUIRE_EQ(last_time[1], 0.5);
        REQUIRE_EQ(last_time[2], 1.0);
    }

    std::filesystem::remove_all(dir);
}

static void Test_HighFive_Backend_Basic_File_Layout()
{
    const auto dir = Unique_Temp_Path("basic");
    std::filesystem::create_directories(dir);
    const auto file_path = dir / "basic.spg.h5md";

    HighFiveBackend backend;
    H5MDWriter writer(&backend);
    WriterOptions options;
    options.path = file_path.string();
    options.schema_name = "test.schema";
    options.schema_version = "1";
    REQUIRE_TRUE(writer.Open(options));
    REQUIRE_TRUE(writer.Create_Dataset({Observable_Value_Path("temperature"),
                                        DataType::float64,
                                        {{0}, {0}, {0}},
                                        true}));
    const double temperature = 300.0;
    REQUIRE_TRUE(writer.Append_Float64(Observable_Value_Path("temperature"),
                                       &temperature, 1));
    REQUIRE_TRUE(writer.Create_Hard_Link(Observable_Value_Path("temperature"),
                                         "/observables/all/temperature_alias"));
    REQUIRE_TRUE(writer.Create_Hard_Link(Observable_Value_Path("temperature"),
                                         "/observables/all/temperature_alias"));
    REQUIRE_TRUE(writer.Write_String_Array(
        "/parameters/sponge/test/string_array", {"alpha", "beta"}));
    REQUIRE_TRUE(writer.Finalize());
    REQUIRE_TRUE(writer.Close());

    {
        HighFive::File file(file_path.string(), HighFive::File::ReadOnly);
        REQUIRE_TRUE(file.exist("/h5md"));
        REQUIRE_TRUE(file.exist("/h5md/creator"));
        auto creator = file.getGroup("/h5md/creator");
        std::string creator_name;
        std::string creator_version;
        creator.getAttribute("name").read(creator_name);
        creator.getAttribute("version").read(creator_version);
        REQUIRE_EQ(creator_name, std::string("SPONGE"));
        REQUIRE_EQ(creator_version, std::string(kSpongeWriterVersion));
        REQUIRE_TRUE(file.exist(path::sponge_schema_name));
        REQUIRE_TRUE(file.exist(path::sponge_schema_version));
        REQUIRE_EQ(Read_String(file, path::sponge_schema_name),
                   std::string("test.schema"));
        REQUIRE_EQ(Read_String(file, path::sponge_schema_version),
                   std::string("1"));
        REQUIRE_EQ(Read_String(file, path::output_status),
                   std::string("finalized"));
        const auto initial_frame_count =
            Read_Int64_Vector(file, path::output_frame_count);
        const auto initial_last_step =
            Read_Int64_Vector(file, path::output_last_complete_step);
        const auto initial_last_time =
            Read_Float64_Vector(file, path::output_last_complete_time);
        REQUIRE_EQ(initial_frame_count.size(), static_cast<std::size_t>(1));
        REQUIRE_EQ(initial_frame_count[0], static_cast<int64_t>(0));
        REQUIRE_EQ(initial_last_step[0], static_cast<int64_t>(-1));
        REQUIRE_EQ(initial_last_time[0], 0.0);
        REQUIRE_TRUE(file.exist(Observable_Value_Path("temperature")));
        REQUIRE_TRUE(file.exist("/observables/all/temperature_alias"));
        auto dataset = file.getDataSet(Observable_Value_Path("temperature"));
        auto alias_dataset =
            file.getDataSet("/observables/all/temperature_alias");
        REQUIRE_EQ(dataset.getSpace().getDimensions()[0],
                   static_cast<std::size_t>(1));
        REQUIRE_EQ(alias_dataset.getSpace().getDimensions()[0],
                   static_cast<std::size_t>(1));
        std::vector<double> alias_values;
        alias_values = Read_Flat_Dataset<double>(alias_dataset);
        REQUIRE_EQ(alias_values.size(), static_cast<std::size_t>(1));
        REQUIRE_EQ(alias_values[0], 300.0);
        const auto strings =
            Read_String_Vector(file, "/parameters/sponge/test/string_array");
        REQUIRE_EQ(strings.size(), static_cast<std::size_t>(2));
        REQUIRE_EQ(strings[0], std::string("alpha"));
        REQUIRE_EQ(strings[1], std::string("beta"));
    }

    std::filesystem::remove_all(dir);
}

static void Test_HighFive_Backend_Nested_Group_Idempotence()
{
    const auto dir = Unique_Temp_Path("nested_groups");
    std::filesystem::create_directories(dir);
    const auto file_path = dir / "nested_groups.spg.h5md";

    HighFiveBackend backend;
    H5MDWriter writer(&backend);
    WriterOptions options;
    options.path = file_path.string();
    options.schema_name = "test.schema";
    options.schema_version = "nested-groups";
    REQUIRE_TRUE(writer.Open(options));
    REQUIRE_TRUE(writer.Ensure_Group("/parameters/sponge/test/deep/group"));
    REQUIRE_TRUE(writer.Ensure_Group("/parameters/sponge/test/deep/group"));
    REQUIRE_TRUE(writer.Write_String("/parameters/sponge/test/deep/group/value",
                                     "nested"));
    REQUIRE_TRUE(writer.Finalize());
    REQUIRE_TRUE(writer.Close());

    {
        HighFive::File file(file_path.string(), HighFive::File::ReadOnly);
        REQUIRE_TRUE(file.exist("/parameters/sponge/test"));
        REQUIRE_TRUE(file.exist("/parameters/sponge/test/deep"));
        REQUIRE_TRUE(file.exist("/parameters/sponge/test/deep/group"));
        REQUIRE_TRUE(file.exist("/parameters/sponge/test/deep/group/value"));
        REQUIRE_EQ(
            Read_String(file, "/parameters/sponge/test/deep/group/value"),
            std::string("nested"));
    }

    std::filesystem::remove_all(dir);
}

static void Test_HighFive_Backend_Factory_And_Dataset_Reopen_Semantics()
{
    const auto dir = Unique_Temp_Path("factory_dataset_reopen");
    const auto file_path = dir / "nested" / "output" / "factory.spg.h5md";

    {
        HighFiveBackendFactory factory;
        auto backend = factory.Create_Backend();
        REQUIRE_TRUE(backend.get() != nullptr);
        H5MDWriter writer(backend.get());

        WriterOptions options;
        options.path = file_path.string();
        options.schema_name = "factory.schema";
        options.schema_version = "2";
        REQUIRE_TRUE(writer.Open(options));
        REQUIRE_TRUE(writer.Create_Dataset({"/observables/all/vector/value",
                                            DataType::float64,
                                            {{0, 2}, {0, 0}, {0, 2}},
                                            true}));
        REQUIRE_TRUE(
            writer.Create_Dataset({"/observables/all/fixed_matrix/value",
                                   DataType::float64,
                                   {{2, 2}, {2, 2}, {2, 2}},
                                   false}));
        double first[2] = {1.0, 2.0};
        double second[2] = {3.0, 4.0};
        double third[2] = {5.0, 6.0};
        REQUIRE_TRUE(
            writer.Append_Float64("/observables/all/vector/value", first, 2));
        REQUIRE_TRUE(writer.Create_Dataset({"/observables/all/vector/value",
                                            DataType::float64,
                                            {{0, 2}, {0, 0}, {0, 2}},
                                            true}));
        REQUIRE_TRUE(
            writer.Append_Float64("/observables/all/vector/value", second, 2));
        REQUIRE_TRUE(
            writer.Append_Float64("/observables/all/vector/value", third, 2));
        REQUIRE_TRUE(writer.Finalize());
        REQUIRE_TRUE(writer.Close());
    }

    {
        REQUIRE_TRUE(std::filesystem::exists(file_path));
        HighFive::File file(file_path.string(), HighFive::File::ReadOnly);
        REQUIRE_EQ(Read_String(file, path::sponge_schema_name),
                   std::string("factory.schema"));
        REQUIRE_EQ(Read_String(file, path::sponge_schema_version),
                   std::string("2"));
        REQUIRE_TRUE(file.exist("/observables/all/vector/value"));
        auto dataset = file.getDataSet("/observables/all/vector/value");
        const auto dims = dataset.getSpace().getDimensions();
        REQUIRE_EQ(dims[0], static_cast<std::size_t>(3));
        REQUIRE_EQ(dims[1], static_cast<std::size_t>(2));
        std::vector<double> values;
        values = Read_Flat_Dataset<double>(dataset);
        REQUIRE_EQ(values.size(), static_cast<std::size_t>(6));
        REQUIRE_EQ(values[0], 1.0);
        REQUIRE_EQ(values[1], 2.0);
        REQUIRE_EQ(values[2], 3.0);
        REQUIRE_EQ(values[3], 4.0);
        REQUIRE_EQ(values[4], 5.0);
        REQUIRE_EQ(values[5], 6.0);
        const auto vector_max_dims = Dataset_Max_Dims(dataset);
        const auto vector_chunk_dims = Dataset_Chunk_Dims(dataset);
        REQUIRE_EQ(vector_max_dims[0], H5S_UNLIMITED);
        REQUIRE_EQ(vector_max_dims[1], static_cast<hsize_t>(2));
        REQUIRE_EQ(vector_chunk_dims[0], static_cast<hsize_t>(1));
        REQUIRE_EQ(vector_chunk_dims[1], static_cast<hsize_t>(2));
        auto fixed_dataset =
            file.getDataSet("/observables/all/fixed_matrix/value");
        const auto fixed_dims = fixed_dataset.getSpace().getDimensions();
        const auto fixed_max_dims = Dataset_Max_Dims(fixed_dataset);
        const auto fixed_chunk_dims = Dataset_Chunk_Dims(fixed_dataset);
        REQUIRE_EQ(fixed_dims[0], static_cast<std::size_t>(2));
        REQUIRE_EQ(fixed_dims[1], static_cast<std::size_t>(2));
        REQUIRE_EQ(fixed_max_dims[0], static_cast<hsize_t>(2));
        REQUIRE_EQ(fixed_max_dims[1], static_cast<hsize_t>(2));
        REQUIRE_EQ(fixed_chunk_dims[0], static_cast<hsize_t>(2));
        REQUIRE_EQ(fixed_chunk_dims[1], static_cast<hsize_t>(2));
        REQUIRE_EQ(Read_String(file, path::output_status),
                   std::string("finalized"));
    }

    std::filesystem::remove_all(dir);
}

static void Test_HighFive_Backend_String_Overwrite()
{
    const auto dir = Unique_Temp_Path("string_overwrite");
    std::filesystem::create_directories(dir);
    const auto file_path = dir / "overwrite.spg.h5md";

    {
        HighFiveBackend backend;
        H5MDWriter writer(&backend);
        WriterOptions options;
        options.path = file_path.string();
        REQUIRE_TRUE(writer.Open(options));
        REQUIRE_TRUE(writer.Write_String(path::mdinfo_text, "old mdinfo"));
        REQUIRE_TRUE(writer.Write_String(path::mdinfo_text, "new mdinfo"));
        REQUIRE_TRUE(
            writer.Write_String_Array(path::legacy_sidecar_keys, {"old_key"}));
        REQUIRE_TRUE(writer.Write_String_Array(path::legacy_sidecar_keys,
                                               {"new_key", "second_key"}));
        REQUIRE_TRUE(writer.Write_String_Array(
            "/parameters/sponge/test/nested/string_array",
            {"", "with spaces", "relative/path.spg.h5md"}));
        REQUIRE_TRUE(writer.Write_String_Array(
            "/parameters/sponge/test/nested/string_array",
            {"replacement", ""}));
        REQUIRE_TRUE(writer.Finalize());
        REQUIRE_TRUE(writer.Close());
    }

    {
        HighFive::File file(file_path.string(), HighFive::File::ReadOnly);
        REQUIRE_EQ(Read_String(file, path::mdinfo_text),
                   std::string("new mdinfo"));
        const auto keys = Read_String_Vector(file, path::legacy_sidecar_keys);
        REQUIRE_EQ(keys.size(), static_cast<std::size_t>(2));
        REQUIRE_EQ(keys[0], std::string("new_key"));
        REQUIRE_EQ(keys[1], std::string("second_key"));
        const auto nested_strings = Read_String_Vector(
            file, "/parameters/sponge/test/nested/string_array");
        REQUIRE_EQ(nested_strings.size(), static_cast<std::size_t>(2));
        REQUIRE_EQ(nested_strings[0], std::string("replacement"));
        REQUIRE_EQ(nested_strings[1], std::string(""));
    }

    std::filesystem::remove_all(dir);
}

static void Test_HighFive_Backend_Observable_Only_Layout()
{
    const auto dir = Unique_Temp_Path("observable_only");
    std::filesystem::create_directories(dir);
    const auto file_path = dir / "observable.obs.spg.h5md";

    HighFiveBackend backend;
    H5MDWriter writer(&backend);
    WriterOptions options;
    options.path = file_path.string();
    options.observable_only = true;
    REQUIRE_TRUE(writer.Open(options));
    REQUIRE_TRUE(writer.Finalize());
    REQUIRE_TRUE(writer.Close());

    {
        HighFive::File file(file_path.string(), HighFive::File::ReadOnly);
        REQUIRE_TRUE(file.exist("/h5md"));
        REQUIRE_TRUE(file.exist("/observables"));
        REQUIRE_TRUE(file.exist("/parameters"));
        REQUIRE_TRUE(!file.exist("/particles"));
        REQUIRE_EQ(Read_String(file, path::output_status),
                   std::string("finalized"));
    }

    std::filesystem::remove_all(dir);
}

static void Test_HighFive_Backend_Rejects_Invalid_Operations()
{
    const auto dir = Unique_Temp_Path("invalid");
    std::filesystem::create_directories(dir);
    const auto file_path = dir / "invalid.spg.h5md";

    HighFiveBackend backend;
    H5MDWriter writer(&backend);
    WriterOptions options;
    options.path = file_path.string();
    REQUIRE_TRUE(writer.Open(options));

    REQUIRE_TRUE(
        !writer.Create_Dataset({"", DataType::float64, {{0}, {0}, {0}}, true}));
    REQUIRE_TRUE(!writer.Create_Dataset({"/observables/all/empty_dims/value",
                                         DataType::float64,
                                         {{}, {}, {}},
                                         true}));

    REQUIRE_TRUE(writer.Create_Dataset({Observable_Value_Path("temperature"),
                                        DataType::float64,
                                        {{0}, {0}, {0}},
                                        true}));
    double scalar_values[2] = {300.0, 301.0};
    REQUIRE_TRUE(!writer.Append_Float64(Observable_Value_Path("temperature"),
                                        scalar_values, 2));
    REQUIRE_TRUE(!writer.Append_Float64(Observable_Value_Path("temperature"),
                                        nullptr, 1));

    int64_t value = 1;
    REQUIRE_TRUE(!writer.Append_Int64("/missing/dataset", &value, 1));
    REQUIRE_TRUE(!writer.Create_Virtual_Dataset(
        {"/virtual/string", DataType::string, {{1}, {1}, {1}}, false}, {}));
    REQUIRE_TRUE(!writer.Create_Hard_Link("/missing/source", "/missing/link"));

    VirtualDatasetSource rank_mismatch_source;
    rank_mismatch_source.file_path = "missing_source.h5";
    rank_mismatch_source.dataset_path = "/source";
    rank_mismatch_source.source_dims = {1, 2};
    rank_mismatch_source.virtual_start = {0, 0};
    REQUIRE_TRUE(!writer.Create_Virtual_Dataset(
        {"/virtual/rank_mismatch", DataType::float64, {{1}, {1}, {1}}, false},
        {rank_mismatch_source}));

    VirtualDatasetSource start_mismatch_source;
    start_mismatch_source.file_path = "missing_source.h5";
    start_mismatch_source.dataset_path = "/source";
    start_mismatch_source.source_dims = {1};
    start_mismatch_source.virtual_start = {0, 0};
    REQUIRE_TRUE(!writer.Create_Virtual_Dataset(
        {"/virtual/start_mismatch", DataType::float64, {{1}, {1}, {1}}, false},
        {start_mismatch_source}));

    REQUIRE_TRUE(writer.Close());

    std::filesystem::remove_all(dir);
}

static void Test_HighFive_Backend_Failed_Metadata()
{
    const auto dir = Unique_Temp_Path("failed_metadata");
    std::filesystem::create_directories(dir);
    const auto file_path = dir / "failed.spg.h5md";

    {
        HighFiveBackend backend;
        H5MDWriter writer(&backend);
        WriterOptions options;
        options.path = file_path.string();
        REQUIRE_TRUE(writer.Open(options));
        REQUIRE_TRUE(writer.Mark_Failed("intentional failure"));
        REQUIRE_TRUE(writer.Close());
    }

    {
        HighFive::File file(file_path.string(), HighFive::File::ReadOnly);
        REQUIRE_EQ(Read_String(file, path::output_status),
                   std::string("failed"));
        REQUIRE_EQ(Read_String(file, path::output_error),
                   std::string("intentional failure"));
    }

    std::filesystem::remove_all(dir);
}

static void Test_HighFive_Backend_Status_State()
{
    const auto dir = Unique_Temp_Path("backend_status");
    std::filesystem::create_directories(dir);
    const auto file_path = dir / "status.spg.h5md";

    {
        HighFiveBackend backend;
        int64_t value = 1;
        REQUIRE_TRUE(!backend.Append_Int64("/missing/dataset", &value, 1));
        REQUIRE_TRUE(backend.Status() == FileStatus::failed);
        REQUIRE_TRUE(!backend.Last_Error().empty());
    }

    {
        HighFiveBackend backend;
        WriterOptions options;
        options.path = file_path.string();
        REQUIRE_TRUE(backend.Open(options));
        REQUIRE_TRUE(backend.Status() == FileStatus::open);
        REQUIRE_TRUE(backend.Set_Status(FileStatus::closing));
        REQUIRE_TRUE(backend.Status() == FileStatus::closing);
        REQUIRE_TRUE(backend.Close());
        REQUIRE_TRUE(backend.Status() == FileStatus::closed);
        REQUIRE_TRUE(!backend.Write_String("/parameters/sponge/after_close",
                                           "should fail"));
        REQUIRE_TRUE(backend.Status() == FileStatus::failed);
        REQUIRE_TRUE(backend.Last_Error().find("not open") !=
                     std::string::npos);
    }

    {
        HighFive::File file(file_path.string(), HighFive::File::ReadOnly);
        REQUIRE_EQ(Read_String(file, path::output_status),
                   std::string("closing"));
    }

    std::filesystem::remove_all(dir);
}

static void Test_Output_Completion_Tracker_With_Real_Backend()
{
    const auto dir = Unique_Temp_Path("completion_tracker");
    std::filesystem::create_directories(dir);
    const auto file_path = dir / "completion.spg.h5md";

    {
        HighFiveBackend backend;
        H5MDWriter writer(&backend);
        WriterOptions options;
        options.path = file_path.string();
        REQUIRE_TRUE(writer.Open(options));

        OutputCompletionTracker tracker(&writer);
        REQUIRE_TRUE(tracker.Mark_Open());
        REQUIRE_TRUE(tracker.Begin_Frame(100, 1.5));
        REQUIRE_TRUE(tracker.Complete_Frame());
        REQUIRE_TRUE(tracker.Mark_Closing());
        REQUIRE_TRUE(tracker.Mark_Finalized());
        REQUIRE_TRUE(writer.Close());
    }

    {
        HighFive::File file(file_path.string(), HighFive::File::ReadOnly);
        REQUIRE_EQ(Read_String(file, path::output_status),
                   std::string("finalized"));
        const auto frame_count =
            Read_Int64_Vector(file, path::output_frame_count);
        const auto last_step =
            Read_Int64_Vector(file, path::output_last_complete_step);
        const auto last_time =
            Read_Float64_Vector(file, path::output_last_complete_time);
        REQUIRE_EQ(frame_count.back(), static_cast<int64_t>(1));
        REQUIRE_EQ(last_step.back(), static_cast<int64_t>(100));
        REQUIRE_EQ(last_time.back(), 1.5);
    }

    std::filesystem::remove_all(dir);
}

static void Test_HighFive_Backend_Virtual_Dataset()
{
    const auto dir = Unique_Temp_Path("vds");
    std::filesystem::create_directories(dir);
    const auto source_path = dir / "source.spg.h5md";
    const auto wrapper_path = dir / "wrapper.spg.h5md";

    {
        HighFiveBackend source_backend;
        H5MDWriter source_writer(&source_backend);
        WriterOptions options;
        options.path = source_path.string();
        REQUIRE_TRUE(source_writer.Open(options));
        REQUIRE_TRUE(source_writer.Create_Dataset(
            {"/particles/all/step", DataType::int64, {{0}, {0}, {0}}, true}));
        int64_t steps[2] = {10, 20};
        REQUIRE_TRUE(
            source_writer.Append_Int64("/particles/all/step", &steps[0], 1));
        REQUIRE_TRUE(
            source_writer.Append_Int64("/particles/all/step", &steps[1], 1));
        REQUIRE_TRUE(source_writer.Finalize());
        REQUIRE_TRUE(source_writer.Close());
    }

    {
        HighFiveBackend wrapper_backend;
        H5MDWriter wrapper_writer(&wrapper_backend);
        WriterOptions options;
        options.path = wrapper_path.string();
        REQUIRE_TRUE(wrapper_writer.Open(options));

        VirtualDatasetSource source;
        source.file_path = source_path.string();
        source.dataset_path = "/particles/all/step";
        source.source_dims = {2};
        source.virtual_start = {0};
        REQUIRE_TRUE(wrapper_writer.Create_Virtual_Dataset(
            {"/particles/all/step", DataType::int64, {{2}, {2}, {2}}, false},
            {source}));
        REQUIRE_TRUE(wrapper_writer.Finalize());
        REQUIRE_TRUE(wrapper_writer.Close());
    }

    {
        HighFive::File file(wrapper_path.string(), HighFive::File::ReadOnly);
        REQUIRE_TRUE(file.exist("/particles/all/step"));
        auto dataset = file.getDataSet("/particles/all/step");
        REQUIRE_EQ(dataset.getSpace().getDimensions()[0],
                   static_cast<std::size_t>(2));
        std::vector<int64_t> values;
        values = Read_Flat_Dataset<int64_t>(dataset);
        REQUIRE_EQ(values.size(), static_cast<std::size_t>(2));
        REQUIRE_EQ(values[0], static_cast<int64_t>(10));
        REQUIRE_EQ(values[1], static_cast<int64_t>(20));
    }

    std::filesystem::remove_all(dir);
}

static void Test_Trajectory_Writer_With_Real_Backend()
{
    const auto dir = Unique_Temp_Path("trajectory_writer");
    std::filesystem::create_directories(dir);
    const auto trajectory_path = dir / "trajectory.spg.h5md";
    const auto restart_path = dir / "unused.spgr.h5";

    {
        HighFiveBackend backend;
        TrajectoryH5Writer writer(&backend);
        auto plan = Make_File_Plan(trajectory_path, restart_path);
        REQUIRE_TRUE(writer.Open_Single_File(plan, "test"));
        REQUIRE_TRUE(writer.Define_Particle_Datasets(2, true, true));
        REQUIRE_TRUE(
            writer.Define_Observable_Stream({"temperature"}, {"TEMP"}));
        float position[6] = {0, 1, 2, 3, 4, 5};
        float velocity[6] = {1, 1, 1, 2, 2, 2};
        float force[6] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
        float box[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        REQUIRE_TRUE(writer.Append_Particle_Frame(10, 0.5, position, box,
                                                  velocity, force));
        REQUIRE_TRUE(
            writer.Append_Observable_Frame(10, 0.5, {{"temperature", 300.0}}));
        REQUIRE_TRUE(writer.Ensure_Nose_Hoover_Chain_Observables(2));
        float nhc_coordinates[2] = {0.5f, 0.6f};
        float nhc_velocities[2] = {0.7f, 0.8f};
        REQUIRE_TRUE(writer.Append_Nose_Hoover_Chain_Frame(
            10, 0.5, nhc_coordinates, nhc_velocities, 2));
        REQUIRE_TRUE(writer.Ensure_Sits_Nk_Observable("traj_sits", 2));
        float sits_values[2] = {2.0f, 3.0f};
        REQUIRE_TRUE(
            writer.Append_Sits_Nk_Frame(10, 0.5, "traj_sits", sits_values, 2));
        REQUIRE_TRUE(writer.Ensure_Metadynamics_Scalars());
        REQUIRE_TRUE(
            writer.Append_Metadynamics_Scalar_Frame(10, 0.5, 6.0, 7.0, 8.0));
        REQUIRE_TRUE(writer.Ensure_Reaxff_Energy_Terms({"bond", "angle"}));
        REQUIRE_TRUE(writer.Append_Reaxff_Frame(
            10, 0.5, {{"bond", 9.0}, {"angle", 10.0}}));
        REQUIRE_TRUE(writer.Write_Mdinfo_Text("MDINFO TEXT"));
        REQUIRE_TRUE(writer.Write_Legacy_Sidecar_Paths(
            {"mdout", "trajectory_file"},
            {"legacy.mdout", "legacy_coordinate.dat"}));
        REQUIRE_TRUE(writer.Finalize());
        REQUIRE_TRUE(writer.Close());
    }

    {
        HighFive::File file(trajectory_path.string(), HighFive::File::ReadOnly);
        REQUIRE_TRUE(file.exist(path::position_value));
        REQUIRE_TRUE(file.exist(path::position_step));
        REQUIRE_TRUE(file.exist(path::position_time));
        REQUIRE_TRUE(file.exist(path::velocity_value));
        REQUIRE_TRUE(file.exist(path::velocity_step));
        REQUIRE_TRUE(file.exist(path::velocity_time));
        REQUIRE_TRUE(file.exist(path::force_value));
        REQUIRE_TRUE(file.exist(path::force_step));
        REQUIRE_TRUE(file.exist(path::force_time));
        REQUIRE_TRUE(file.exist(path::box_edges_value));
        REQUIRE_TRUE(file.exist(path::box_edges_step));
        REQUIRE_TRUE(file.exist(path::box_edges_time));
        REQUIRE_TRUE(file.exist(Observable_Value_Path("temperature")));
        REQUIRE_TRUE(file.exist(Observable_Step_Path("temperature")));
        REQUIRE_TRUE(file.exist(Observable_Time_Path("temperature")));
        REQUIRE_TRUE(file.exist(module_path::nhc_coordinate_value));
        REQUIRE_TRUE(file.exist(module_path::nhc_velocity_value));
        REQUIRE_TRUE(file.exist(Sits_Nk_Value_Path("traj_sits")));
        REQUIRE_TRUE(file.exist(Metadynamics_Scalar_Value_Path("meta")));
        REQUIRE_TRUE(file.exist(Reaxff_Term_Value_Path("bond")));
        REQUIRE_TRUE(file.exist(Reaxff_Term_Value_Path("angle")));
        REQUIRE_TRUE(file.exist(path::mdinfo_text));
        REQUIRE_TRUE(file.exist(path::legacy_sidecar_keys));
        REQUIRE_TRUE(file.exist(path::legacy_sidecar_paths));
        REQUIRE_EQ(Read_String(file, path::sponge_schema_name),
                   std::string("sponge.output.h5md"));
        REQUIRE_EQ(Read_String(file, path::sponge_schema_version),
                   std::string("test"));
        REQUIRE_TRUE(file.exist(path::sponge_log));
        REQUIRE_EQ(
            file.getDataSet(path::position_value).getSpace().getDimensions()[0],
            static_cast<std::size_t>(1));
        REQUIRE_EQ(
            file.getDataSet(path::position_value).getSpace().getDimensions()[1],
            static_cast<std::size_t>(2));
        REQUIRE_EQ(
            file.getDataSet(path::force_value).getSpace().getDimensions()[0],
            static_cast<std::size_t>(1));
        REQUIRE_EQ(
            file.getDataSet(path::force_value).getSpace().getDimensions()[1],
            static_cast<std::size_t>(2));
        REQUIRE_EQ(file.getDataSet(Observable_Value_Path("temperature"))
                       .getSpace()
                       .getDimensions()[0],
                   static_cast<std::size_t>(1));
        std::vector<float> forces;
        forces = Read_Flat_Dataset<float>(file.getDataSet(path::force_value));
        REQUIRE_EQ(forces.size(), static_cast<std::size_t>(6));
        REQUIRE_EQ(forces[0], 0.1f);
        REQUIRE_EQ(forces[5], 0.6f);
        std::vector<float> nhc_coordinates_read;
        std::vector<float> nhc_velocities_read;
        nhc_coordinates_read = Read_Flat_Dataset<float>(
            file.getDataSet(module_path::nhc_coordinate_value));
        nhc_velocities_read = Read_Flat_Dataset<float>(
            file.getDataSet(module_path::nhc_velocity_value));
        REQUIRE_EQ(nhc_coordinates_read[0], 0.5f);
        REQUIRE_EQ(nhc_velocities_read[1], 0.8f);
        std::vector<float> sits_read;
        sits_read = Read_Flat_Dataset<float>(
            file.getDataSet(Sits_Nk_Value_Path("traj_sits")));
        REQUIRE_EQ(sits_read[0], 2.0f);
        REQUIRE_EQ(sits_read[1], 3.0f);
        std::vector<double> metad_meta;
        metad_meta = Read_Flat_Dataset<double>(
            file.getDataSet(Metadynamics_Scalar_Value_Path("meta")));
        REQUIRE_EQ(metad_meta[0], 6.0);
        std::vector<double> reaxff_bond;
        std::vector<double> reaxff_angle;
        reaxff_bond = Read_Flat_Dataset<double>(
            file.getDataSet(Reaxff_Term_Value_Path("bond")));
        reaxff_angle = Read_Flat_Dataset<double>(
            file.getDataSet(Reaxff_Term_Value_Path("angle")));
        REQUIRE_EQ(reaxff_bond[0], 9.0);
        REQUIRE_EQ(reaxff_angle[0], 10.0);
        REQUIRE_EQ(Read_String(file, path::mdinfo_text),
                   std::string("MDINFO TEXT"));
        const auto original_columns =
            Read_String_Vector(file, path::mdout_columns_original_name);
        const auto hdf5_columns =
            Read_String_Vector(file, path::mdout_columns_hdf5_name);
        REQUIRE_EQ(original_columns[0], std::string("TEMP"));
        REQUIRE_EQ(hdf5_columns[0], std::string("temperature"));
        const auto legacy_keys =
            Read_String_Vector(file, path::legacy_sidecar_keys);
        const auto legacy_paths =
            Read_String_Vector(file, path::legacy_sidecar_paths);
        REQUIRE_EQ(legacy_keys.size(), static_cast<std::size_t>(2));
        REQUIRE_EQ(legacy_paths.size(), static_cast<std::size_t>(2));
        REQUIRE_EQ(legacy_keys[0], std::string("mdout"));
        REQUIRE_EQ(legacy_paths[1], std::string("legacy_coordinate.dat"));
        const auto frame_count =
            Read_Int64_Vector(file, path::output_frame_count);
        const auto publication_epochs =
            Read_Int64_Vector(file, path::output_publication_epoch);
        const auto particle_committed = Read_Int64_Vector(
            file, Output_Stream_Committed_Count_Path("particles"));
        const auto observable_committed = Read_Int64_Vector(
            file, Output_Stream_Committed_Count_Path("observables"));
        const auto last_step =
            Read_Int64_Vector(file, path::output_last_complete_step);
        const auto last_time =
            Read_Float64_Vector(file, path::output_last_complete_time);
        const auto position_step = Read_Int64_Vector(file, path::position_step);
        const auto velocity_step = Read_Int64_Vector(file, path::velocity_step);
        const auto force_step = Read_Int64_Vector(file, path::force_step);
        const auto box_step = Read_Int64_Vector(file, path::box_edges_step);
        const auto temperature_step =
            Read_Int64_Vector(file, Observable_Step_Path("temperature"));
        const auto position_time =
            Read_Float64_Vector(file, path::position_time);
        const auto velocity_time =
            Read_Float64_Vector(file, path::velocity_time);
        const auto force_time = Read_Float64_Vector(file, path::force_time);
        const auto box_time = Read_Float64_Vector(file, path::box_edges_time);
        const auto temperature_time =
            Read_Float64_Vector(file, Observable_Time_Path("temperature"));
        REQUIRE_EQ(frame_count.back(), static_cast<int64_t>(1));
        REQUIRE_EQ(publication_epochs.size(), static_cast<std::size_t>(3));
        REQUIRE_EQ(publication_epochs[0], static_cast<int64_t>(0));
        REQUIRE_EQ(publication_epochs[1], static_cast<int64_t>(1));
        REQUIRE_EQ(publication_epochs[2], static_cast<int64_t>(2));
        REQUIRE_EQ(particle_committed.back(), static_cast<int64_t>(1));
        REQUIRE_EQ(observable_committed.back(), static_cast<int64_t>(1));
        REQUIRE_EQ(last_step.back(), static_cast<int64_t>(10));
        REQUIRE_EQ(last_time.back(), 0.5);
        REQUIRE_EQ(position_step.back(), static_cast<int64_t>(10));
        REQUIRE_EQ(velocity_step.back(), static_cast<int64_t>(10));
        REQUIRE_EQ(force_step.back(), static_cast<int64_t>(10));
        REQUIRE_EQ(box_step.back(), static_cast<int64_t>(10));
        REQUIRE_EQ(temperature_step.back(), static_cast<int64_t>(10));
        REQUIRE_EQ(position_time.back(), 0.5);
        REQUIRE_EQ(velocity_time.back(), 0.5);
        REQUIRE_EQ(force_time.back(), 0.5);
        REQUIRE_EQ(box_time.back(), 0.5);
        REQUIRE_EQ(temperature_time.back(), 0.5);
        REQUIRE_EQ(Read_String(file, path::output_status),
                   std::string("finalized"));
    }

    std::filesystem::remove_all(dir);
}

static void Test_Trajectory_Optional_Particle_Fields_With_Real_Backend()
{
    const auto dir = Unique_Temp_Path("trajectory_optional_particle_fields");
    std::filesystem::create_directories(dir);
    const auto trajectory_path = dir / "trajectory.spg.h5md";
    const auto restart_path = dir / "unused.spgr.h5";

    {
        HighFiveBackend backend;
        TrajectoryH5Writer writer(&backend);
        auto plan = Make_File_Plan(trajectory_path, restart_path);
        REQUIRE_TRUE(writer.Open_Single_File(plan, "test"));
        REQUIRE_TRUE(writer.Define_Particle_Datasets(1, false, false));
        float position[3] = {0, 1, 2};
        float box[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        REQUIRE_TRUE(writer.Append_Particle_Frame(30, 1.5, position, box));
        REQUIRE_TRUE(writer.Finalize());
        REQUIRE_TRUE(writer.Close());
    }

    {
        HighFive::File file(trajectory_path.string(), HighFive::File::ReadOnly);
        REQUIRE_TRUE(file.exist(path::position_value));
        REQUIRE_TRUE(file.exist(path::position_step));
        REQUIRE_TRUE(file.exist(path::position_time));
        REQUIRE_TRUE(file.exist(path::box_edges_value));
        REQUIRE_TRUE(file.exist(path::box_edges_step));
        REQUIRE_TRUE(file.exist(path::box_edges_time));
        REQUIRE_TRUE(!file.exist(path::velocity_value));
        REQUIRE_TRUE(!file.exist(path::velocity_step));
        REQUIRE_TRUE(!file.exist(path::velocity_time));
        REQUIRE_TRUE(!file.exist(path::force_value));
        REQUIRE_TRUE(!file.exist(path::force_step));
        REQUIRE_TRUE(!file.exist(path::force_time));
        REQUIRE_EQ(
            file.getDataSet(path::position_value).getSpace().getDimensions()[0],
            static_cast<std::size_t>(1));
        REQUIRE_EQ(
            file.getDataSet(path::position_value).getSpace().getDimensions()[1],
            static_cast<std::size_t>(1));
        REQUIRE_EQ(
            file.getDataSet(path::position_value).getSpace().getDimensions()[2],
            static_cast<std::size_t>(3));
        std::vector<float> positions;
        positions =
            Read_Flat_Dataset<float>(file.getDataSet(path::position_value));
        REQUIRE_EQ(positions.size(), static_cast<std::size_t>(3));
        REQUIRE_EQ(positions[2], 2.0f);
        const auto frame_count =
            Read_Int64_Vector(file, path::output_frame_count);
        const auto last_step =
            Read_Int64_Vector(file, path::output_last_complete_step);
        const auto position_step = Read_Int64_Vector(file, path::position_step);
        const auto box_step = Read_Int64_Vector(file, path::box_edges_step);
        const auto position_time =
            Read_Float64_Vector(file, path::position_time);
        const auto box_time = Read_Float64_Vector(file, path::box_edges_time);
        REQUIRE_EQ(frame_count.back(), static_cast<int64_t>(1));
        REQUIRE_EQ(last_step.back(), static_cast<int64_t>(30));
        REQUIRE_EQ(position_step.back(), static_cast<int64_t>(30));
        REQUIRE_EQ(box_step.back(), static_cast<int64_t>(30));
        REQUIRE_EQ(position_time.back(), 1.5);
        REQUIRE_EQ(box_time.back(), 1.5);
        REQUIRE_EQ(Read_String(file, path::output_status),
                   std::string("finalized"));
    }

    std::filesystem::remove_all(dir);
}

static void Test_Restart_Writer_With_Real_Backend()
{
    const auto dir = Unique_Temp_Path("restart_writer");
    std::filesystem::create_directories(dir);
    const auto trajectory_path = dir / "unused.spg.h5md";
    const auto restart_path = dir / "restart.spgr.h5";

    {
        HighFiveBackend backend;
        RestartH5Writer writer(&backend);
        auto plan = Make_File_Plan(trajectory_path, restart_path);
        REQUIRE_TRUE(writer.Open(plan, "test"));
        REQUIRE_TRUE(writer.Write_Restart_Generation(7));
        REQUIRE_TRUE(writer.Define_Structural_State(2, true));
        float position[6] = {0, 1, 2, 3, 4, 5};
        float velocity[6] = {1, 1, 1, 2, 2, 2};
        float box[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        float nhc[4] = {0.1f, 0.2f, 0.3f, 0.4f};
        float sits[3] = {1.0f, 2.0f, 3.0f};
        float sits_weight[2] = {4.0f, 5.0f};
        REQUIRE_TRUE(
            writer.Write_Structural_State(20, 1.0, position, box, velocity));
        REQUIRE_TRUE(writer.Write_Nose_Hoover_Chain_State(nhc, 2));
        REQUIRE_TRUE(writer.Write_Sits_State("sits_a", "nk", sits, 3));
        REQUIRE_TRUE(
            writer.Write_Sits_State("sits_a", "weight", sits_weight, 2));
        REQUIRE_TRUE(writer.Write_Metad_State_Text("meta0", "hills", "HILLS"));
        REQUIRE_TRUE(
            writer.Write_Metad_State_Text("meta0", "history", "HISTORY"));
        REQUIRE_TRUE(writer.Write_Metad_State_Text("meta0", "edge", "EDGE"));
        REQUIRE_TRUE(writer.Write_Metad_State_Text("meta0", "potential_export",
                                                   "POTENTIAL"));
        REQUIRE_TRUE(
            writer.Write_Metad_State_Text("meta0", "direct_export", "DIRECT"));
        REQUIRE_TRUE(writer.Write_Legacy_Sidecar_Paths(
            {"restrain", "metad_hills"}, {"restrain.out", "myhill.dat"}));
        REQUIRE_TRUE(writer.Finalize());
        REQUIRE_TRUE(writer.Close());
    }

    {
        HighFive::File file(restart_path.string(), HighFive::File::ReadOnly);
        REQUIRE_TRUE(file.exist(path::position_value));
        REQUIRE_TRUE(file.exist(path::position_step));
        REQUIRE_TRUE(file.exist(path::position_time));
        REQUIRE_TRUE(file.exist(path::velocity_value));
        REQUIRE_TRUE(file.exist(path::velocity_step));
        REQUIRE_TRUE(file.exist(path::velocity_time));
        REQUIRE_TRUE(file.exist(path::box_edges_step));
        REQUIRE_TRUE(file.exist(path::box_edges_time));
        REQUIRE_TRUE(file.exist(path::run_current_step));
        REQUIRE_TRUE(file.exist(path::run_current_time));
        REQUIRE_TRUE(file.exist(path::run_state_type));
        REQUIRE_TRUE(file.exist(path::output_restart_generation));
        REQUIRE_TRUE(file.exist(path::parameters_restart));
        REQUIRE_TRUE(file.exist(path::restart_thermostat));
        REQUIRE_TRUE(file.exist(path::restart_barostat));
        REQUIRE_TRUE(file.exist(path::restart_bias));
        REQUIRE_TRUE(file.exist(path::restart_sits));
        REQUIRE_TRUE(file.exist(path::restart_meta));
        REQUIRE_TRUE(file.exist(path::restart_nhc));
        REQUIRE_TRUE(file.exist(Restart_Sits_State_Path("sits_a", "nk")));
        REQUIRE_TRUE(file.exist(Restart_Sits_State_Path("sits_a", "weight")));
        REQUIRE_TRUE(file.exist(Restart_Metad_State_Path("meta0", "hills")));
        REQUIRE_TRUE(file.exist(Restart_Metad_State_Path("meta0", "history")));
        REQUIRE_TRUE(file.exist(Restart_Metad_State_Path("meta0", "edge")));
        REQUIRE_TRUE(
            file.exist(Restart_Metad_State_Path("meta0", "potential_export")));
        REQUIRE_TRUE(
            file.exist(Restart_Metad_State_Path("meta0", "direct_export")));
        REQUIRE_TRUE(file.exist(path::legacy_sidecars));
        REQUIRE_TRUE(file.exist(path::legacy_sidecar_keys));
        REQUIRE_TRUE(file.exist(path::legacy_sidecar_paths));
        REQUIRE_EQ(Read_String(file, path::sponge_schema_name),
                   std::string("sponge.restart.h5"));
        REQUIRE_EQ(Read_String(file, path::sponge_schema_version),
                   std::string("test"));
        REQUIRE_TRUE(!Read_String(file, path::identity_uuid).empty());
        REQUIRE_EQ(Read_String(file, path::output_mode),
                   std::string("checkpoint"));
        REQUIRE_EQ(
            file.getDataSet(path::position_value).getSpace().getDimensions()[0],
            static_cast<std::size_t>(1));
        REQUIRE_EQ(
            file.getDataSet(path::restart_nhc).getSpace().getDimensions()[0],
            static_cast<std::size_t>(2));
        REQUIRE_EQ(file.getDataSet(Restart_Sits_State_Path("sits_a", "nk"))
                       .getSpace()
                       .getDimensions()[0],
                   static_cast<std::size_t>(3));
        REQUIRE_EQ(file.getDataSet(Restart_Sits_State_Path("sits_a", "weight"))
                       .getSpace()
                       .getDimensions()[0],
                   static_cast<std::size_t>(2));
        const auto frame_count =
            Read_Int64_Vector(file, path::output_frame_count);
        const auto restart_generation =
            Read_Int64_Vector(file, path::output_restart_generation);
        const auto last_step =
            Read_Int64_Vector(file, path::output_last_complete_step);
        const auto current_time =
            Read_Float64_Vector(file, path::run_current_time);
        const auto position_step = Read_Int64_Vector(file, path::position_step);
        const auto velocity_step = Read_Int64_Vector(file, path::velocity_step);
        const auto box_step = Read_Int64_Vector(file, path::box_edges_step);
        const auto position_time =
            Read_Float64_Vector(file, path::position_time);
        const auto velocity_time =
            Read_Float64_Vector(file, path::velocity_time);
        const auto box_time = Read_Float64_Vector(file, path::box_edges_time);
        REQUIRE_EQ(frame_count.back(), static_cast<int64_t>(1));
        REQUIRE_EQ(restart_generation.back(), static_cast<int64_t>(7));
        REQUIRE_EQ(last_step.back(), static_cast<int64_t>(20));
        REQUIRE_EQ(current_time.back(), 1.0);
        REQUIRE_EQ(position_step.back(), static_cast<int64_t>(20));
        REQUIRE_EQ(velocity_step.back(), static_cast<int64_t>(20));
        REQUIRE_EQ(box_step.back(), static_cast<int64_t>(20));
        REQUIRE_EQ(position_time.back(), 1.0);
        REQUIRE_EQ(velocity_time.back(), 1.0);
        REQUIRE_EQ(box_time.back(), 1.0);
        std::vector<float> sits_weight_read;
        sits_weight_read = Read_Flat_Dataset<float>(
            file.getDataSet(Restart_Sits_State_Path("sits_a", "weight")));
        REQUIRE_EQ(sits_weight_read.size(), static_cast<std::size_t>(2));
        REQUIRE_EQ(sits_weight_read[0], 4.0f);
        REQUIRE_EQ(sits_weight_read[1], 5.0f);
        REQUIRE_EQ(
            Read_String(file, Restart_Metad_State_Path("meta0", "hills")),
            std::string("HILLS"));
        REQUIRE_EQ(
            Read_String(file, Restart_Metad_State_Path("meta0", "history")),
            std::string("HISTORY"));
        REQUIRE_EQ(Read_String(file, Restart_Metad_State_Path("meta0", "edge")),
                   std::string("EDGE"));
        REQUIRE_EQ(Read_String(file, Restart_Metad_State_Path(
                                         "meta0", "potential_export")),
                   std::string("POTENTIAL"));
        REQUIRE_EQ(Read_String(file, Restart_Metad_State_Path("meta0",
                                                              "direct_export")),
                   std::string("DIRECT"));
        REQUIRE_EQ(Read_String(file, path::run_state_type),
                   std::string("restart"));
        const auto legacy_keys =
            Read_String_Vector(file, path::legacy_sidecar_keys);
        const auto legacy_paths =
            Read_String_Vector(file, path::legacy_sidecar_paths);
        REQUIRE_EQ(legacy_keys[0], std::string("restrain"));
        REQUIRE_EQ(legacy_paths[1], std::string("myhill.dat"));
        REQUIRE_EQ(Read_String(file, path::output_status),
                   std::string("finalized"));
    }

    std::filesystem::remove_all(dir);
}

static void Test_Atomic_File_Replacement_Preserves_Last_Published_File()
{
    const auto dir = Unique_Temp_Path("atomic_restart_publish");
    std::filesystem::create_directories(dir);
    const auto destination_path = dir / "restart.spgr.h5";
    const auto temporary_path = dir / "restart.spgr.h5.tmp";
    const auto trajectory_path = dir / "unused.spg.h5md";

    auto write_generation = [&](const int64_t generation)
    {
        HighFiveBackend backend;
        RestartH5Writer writer(&backend);
        auto plan = Make_File_Plan(trajectory_path, temporary_path);
        REQUIRE_TRUE(writer.Open(plan, "test"));
        REQUIRE_TRUE(writer.Write_Restart_Generation(generation));
        REQUIRE_TRUE(writer.Write_Run_Metadata(
            generation, static_cast<double>(generation)));
        REQUIRE_TRUE(writer.Finalize());
        REQUIRE_TRUE(writer.Close());
    };

    std::string error_message;
    write_generation(1);
    REQUIRE_TRUE(Atomic_Replace_File(
        temporary_path.string(), destination_path.string(), &error_message));
    write_generation(2);
    REQUIRE_TRUE(Atomic_Replace_File(
        temporary_path.string(), destination_path.string(), &error_message));
    REQUIRE_TRUE(!std::filesystem::exists(temporary_path));
    {
        HighFive::File destination(destination_path.string(),
                                   HighFive::File::ReadOnly);
        REQUIRE_EQ(
            Read_Int64_Vector(destination, path::output_restart_generation)
                .back(),
            static_cast<int64_t>(2));
    }

    int64_t published_generation = 2;
#ifdef _WIN32
    write_generation(3);
    HANDLE held_destination = CreateFileA(
        destination_path.string().c_str(), GENERIC_READ, FILE_SHARE_READ, NULL,
        OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    REQUIRE_TRUE(held_destination != INVALID_HANDLE_VALUE);
    std::thread release_reader(
        [held_destination]()
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            CloseHandle(held_destination);
        });
    const bool replaced_after_retry = Atomic_Replace_File(
        temporary_path.string(), destination_path.string(), &error_message);
    release_reader.join();
    REQUIRE_TRUE(replaced_after_retry);
    published_generation = 3;
    {
        HighFive::File destination(destination_path.string(),
                                   HighFive::File::ReadOnly);
        REQUIRE_EQ(
            Read_Int64_Vector(destination, path::output_restart_generation)
                .back(),
            static_cast<int64_t>(3));
    }

    const auto snapshot_path = dir / "live_wrapper.spg.h5md";
    HighFiveBackend snapshot_backend;
    H5MDWriter snapshot_writer(&snapshot_backend);
    WriterOptions snapshot_options;
    snapshot_options.path = snapshot_path.string();
    snapshot_options.schema_name = "sponge.output.h5md";
    snapshot_options.schema_version = "test";
    snapshot_options.identity_uuid = "first-live-wrapper";
    snapshot_options.output_mode = "vds";
    snapshot_options.atomic_snapshot = true;
    REQUIRE_TRUE(snapshot_writer.Open(snapshot_options));
    REQUIRE_TRUE(snapshot_writer.Write_Publication_Epoch(1));
    REQUIRE_TRUE(snapshot_writer.Publish_Snapshot(snapshot_path.string()));

    HANDLE persistent_reader = CreateFileA(
        snapshot_path.string().c_str(), GENERIC_READ, FILE_SHARE_READ, NULL,
        OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    REQUIRE_TRUE(persistent_reader != INVALID_HANDLE_VALUE);
    REQUIRE_TRUE(snapshot_writer.Write_Publication_Epoch(2));
    REQUIRE_TRUE(
        snapshot_writer.Publish_Snapshot(snapshot_path.string(), true));
    {
        HighFive::File published(snapshot_path.string(),
                                 HighFive::File::ReadOnly);
        REQUIRE_EQ(
            Read_Int64_Vector(published, path::output_publication_epoch).back(),
            static_cast<int64_t>(1));
    }
    REQUIRE_TRUE(snapshot_writer.Write_Publication_Epoch(3));
    REQUIRE_TRUE(!snapshot_writer.Publish_Snapshot(snapshot_path.string()));
    const auto pending_snapshot_path = std::filesystem::path(
        snapshot_path.string() + ".pending.first-live-wrapper");
    REQUIRE_TRUE(std::filesystem::exists(pending_snapshot_path));
    {
        HighFive::File pending(pending_snapshot_path.string(),
                               HighFive::File::ReadOnly);
        REQUIRE_EQ(
            Read_Int64_Vector(pending, path::output_publication_epoch).back(),
            static_cast<int64_t>(3));
    }
    REQUIRE_TRUE(snapshot_writer.Close());

    HighFiveBackend successor_backend;
    H5MDWriter successor_writer(&successor_backend);
    snapshot_options.identity_uuid = "second-live-wrapper";
    REQUIRE_TRUE(successor_writer.Open(snapshot_options));
    REQUIRE_TRUE(successor_writer.Write_Publication_Epoch(4));
    REQUIRE_TRUE(!successor_writer.Publish_Snapshot(snapshot_path.string()));
    const auto successor_pending_path = std::filesystem::path(
        snapshot_path.string() + ".pending.second-live-wrapper");
    REQUIRE_TRUE(std::filesystem::exists(pending_snapshot_path));
    REQUIRE_TRUE(std::filesystem::exists(successor_pending_path));
    {
        HighFive::File first_pending(pending_snapshot_path.string(),
                                     HighFive::File::ReadOnly);
        REQUIRE_EQ(
            Read_Int64_Vector(first_pending, path::output_publication_epoch)
                .back(),
            static_cast<int64_t>(3));
        HighFive::File second_pending(successor_pending_path.string(),
                                      HighFive::File::ReadOnly);
        REQUIRE_EQ(
            Read_Int64_Vector(second_pending, path::output_publication_epoch)
                .back(),
            static_cast<int64_t>(4));
    }
    REQUIRE_TRUE(successor_writer.Close());
    CloseHandle(persistent_reader);
    REQUIRE_TRUE(Atomic_Replace_File(successor_pending_path.string(),
                                     snapshot_path.string()));
    {
        HighFive::File published(snapshot_path.string(),
                                 HighFive::File::ReadOnly);
        REQUIRE_EQ(
            Read_Int64_Vector(published, path::output_publication_epoch).back(),
            static_cast<int64_t>(4));
    }
    REQUIRE_TRUE(std::filesystem::exists(pending_snapshot_path));
#endif

    REQUIRE_TRUE(!Atomic_Replace_File(
        temporary_path.string(), destination_path.string(), &error_message));
    REQUIRE_TRUE(!error_message.empty());
    {
        HighFive::File destination(destination_path.string(),
                                   HighFive::File::ReadOnly);
        REQUIRE_EQ(
            Read_Int64_Vector(destination, path::output_restart_generation)
                .back(),
            published_generation);
    }

    write_generation(published_generation + 1);
    {
        TemporaryFileGuard guard(temporary_path.string());
    }
    REQUIRE_TRUE(!std::filesystem::exists(temporary_path));
    {
        HighFive::File destination(destination_path.string(),
                                   HighFive::File::ReadOnly);
        REQUIRE_EQ(
            Read_Int64_Vector(destination, path::output_restart_generation)
                .back(),
            published_generation);
    }

    std::filesystem::remove_all(dir);
}

static void Test_Restart_Optional_Velocity_With_Real_Backend()
{
    const auto dir = Unique_Temp_Path("restart_optional_velocity");
    std::filesystem::create_directories(dir);
    const auto trajectory_path = dir / "unused.spg.h5md";
    const auto restart_path = dir / "restart.spgr.h5";

    {
        HighFiveBackend backend;
        RestartH5Writer writer(&backend);
        auto plan = Make_File_Plan(trajectory_path, restart_path);
        REQUIRE_TRUE(writer.Open(plan, "test"));
        REQUIRE_TRUE(writer.Define_Structural_State(1, false));
        float position[3] = {0, 1, 2};
        float box[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        REQUIRE_TRUE(writer.Write_Structural_State(40, 2.0, position, box));
        REQUIRE_TRUE(writer.Finalize());
        REQUIRE_TRUE(writer.Close());
    }

    {
        HighFive::File file(restart_path.string(), HighFive::File::ReadOnly);
        REQUIRE_TRUE(file.exist(path::position_value));
        REQUIRE_TRUE(file.exist(path::position_step));
        REQUIRE_TRUE(file.exist(path::position_time));
        REQUIRE_TRUE(file.exist(path::box_edges_value));
        REQUIRE_TRUE(file.exist(path::box_edges_step));
        REQUIRE_TRUE(file.exist(path::box_edges_time));
        REQUIRE_TRUE(!file.exist(path::velocity_value));
        REQUIRE_TRUE(!file.exist(path::velocity_step));
        REQUIRE_TRUE(!file.exist(path::velocity_time));
        REQUIRE_TRUE(file.exist(path::run_current_step));
        REQUIRE_TRUE(file.exist(path::run_current_time));
        REQUIRE_TRUE(file.exist(path::run_state_type));
        REQUIRE_EQ(
            file.getDataSet(path::position_value).getSpace().getDimensions()[0],
            static_cast<std::size_t>(1));
        REQUIRE_EQ(
            file.getDataSet(path::position_value).getSpace().getDimensions()[1],
            static_cast<std::size_t>(1));
        REQUIRE_EQ(
            file.getDataSet(path::position_value).getSpace().getDimensions()[2],
            static_cast<std::size_t>(3));
        std::vector<float> positions;
        positions =
            Read_Flat_Dataset<float>(file.getDataSet(path::position_value));
        REQUIRE_EQ(positions.size(), static_cast<std::size_t>(3));
        REQUIRE_EQ(positions[2], 2.0f);
        const auto frame_count =
            Read_Int64_Vector(file, path::output_frame_count);
        const auto last_step =
            Read_Int64_Vector(file, path::output_last_complete_step);
        const auto current_step =
            Read_Int64_Vector(file, path::run_current_step);
        const auto current_time =
            Read_Float64_Vector(file, path::run_current_time);
        const auto position_step = Read_Int64_Vector(file, path::position_step);
        const auto box_step = Read_Int64_Vector(file, path::box_edges_step);
        const auto position_time =
            Read_Float64_Vector(file, path::position_time);
        const auto box_time = Read_Float64_Vector(file, path::box_edges_time);
        REQUIRE_EQ(frame_count.back(), static_cast<int64_t>(1));
        REQUIRE_EQ(last_step.back(), static_cast<int64_t>(40));
        REQUIRE_EQ(current_step.back(), static_cast<int64_t>(40));
        REQUIRE_EQ(current_time.back(), 2.0);
        REQUIRE_EQ(position_step.back(), static_cast<int64_t>(40));
        REQUIRE_EQ(box_step.back(), static_cast<int64_t>(40));
        REQUIRE_EQ(position_time.back(), 2.0);
        REQUIRE_EQ(box_time.back(), 2.0);
        REQUIRE_EQ(Read_String(file, path::run_state_type),
                   std::string("restart"));
        REQUIRE_EQ(Read_String(file, path::output_status),
                   std::string("finalized"));
    }

    std::filesystem::remove_all(dir);
}

static void Test_Restart_Writer_Custom_Run_Metadata_With_Real_Backend()
{
    const auto dir = Unique_Temp_Path("restart_writer_custom_metadata");
    std::filesystem::create_directories(dir);
    const auto trajectory_path = dir / "unused.spg.h5md";
    const auto restart_path = dir / "restart.spgr.h5";

    {
        HighFiveBackend backend;
        RestartH5Writer writer(&backend);
        auto plan = Make_File_Plan(trajectory_path, restart_path);
        REQUIRE_TRUE(writer.Open(plan, "test"));
        REQUIRE_TRUE(writer.Write_Run_Metadata(42, 2.5, "checkpoint"));
        REQUIRE_TRUE(writer.Finalize());
        REQUIRE_TRUE(writer.Close());
    }

    {
        HighFive::File file(restart_path.string(), HighFive::File::ReadOnly);
        const auto current_step =
            Read_Int64_Vector(file, path::run_current_step);
        const auto current_time =
            Read_Float64_Vector(file, path::run_current_time);
        REQUIRE_EQ(current_step.back(), static_cast<int64_t>(42));
        REQUIRE_EQ(current_time.back(), 2.5);
        REQUIRE_EQ(Read_String(file, path::run_state_type),
                   std::string("checkpoint"));
        REQUIRE_EQ(Read_String(file, path::output_status),
                   std::string("finalized"));
    }

    std::filesystem::remove_all(dir);
}

static void Test_Restart_Writer_Rejects_Second_State_With_Real_Backend()
{
    const auto dir = Unique_Temp_Path("restart_writer_second_state");
    std::filesystem::create_directories(dir);
    const auto trajectory_path = dir / "unused.spg.h5md";
    const auto restart_path = dir / "restart.spgr.h5";

    {
        HighFiveBackend backend;
        RestartH5Writer writer(&backend);
        auto plan = Make_File_Plan(trajectory_path, restart_path);
        REQUIRE_TRUE(writer.Open(plan, "test"));
        REQUIRE_TRUE(writer.Define_Structural_State(1, false));
        float position_a[3] = {0, 0, 0};
        float position_b[3] = {1, 1, 1};
        float box[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        REQUIRE_TRUE(writer.Write_Structural_State(1, 0.1, position_a, box));
        REQUIRE_TRUE(!writer.Write_Structural_State(2, 0.2, position_b, box));
        REQUIRE_TRUE(writer.Close());
    }

    {
        HighFive::File file(restart_path.string(), HighFive::File::ReadOnly);
        REQUIRE_EQ(Read_String(file, path::output_status),
                   std::string("failed"));
        REQUIRE_TRUE(Read_String(file, path::output_error)
                         .find("already contains one structural state") !=
                     std::string::npos);
        REQUIRE_TRUE(!file.exist(path::velocity_value));
        REQUIRE_EQ(
            file.getDataSet(path::position_value).getSpace().getDimensions()[0],
            static_cast<std::size_t>(1));
    }

    std::filesystem::remove_all(dir);
}

static void Test_Observable_Writer_With_Real_Backend()
{
    const auto dir = Unique_Temp_Path("observable_writer");
    std::filesystem::create_directories(dir);
    const auto observable_path = dir / "observable.obs.spg.h5md";

    {
        HighFiveBackend backend;
        ObservableH5Writer writer(&backend);
        SpongeH5OutputPlan::ResolvedOutputPlan plan;
        plan.observable.enabled = true;
        plan.observable.path = observable_path.string();

        REQUIRE_TRUE(writer.Open(plan, "test"));
        REQUIRE_TRUE(
            writer.Define_Observable_Stream({"temperature"}, {"TEMP"}));
        REQUIRE_TRUE(
            writer.Append_Observable_Frame(10, 0.1, {{"temperature", 300.0}}));
        REQUIRE_TRUE(writer.Ensure_Nose_Hoover_Chain_Observables(2));
        float nhc_coordinates[2] = {0.1f, 0.2f};
        float nhc_velocities[2] = {0.3f, 0.4f};
        REQUIRE_TRUE(writer.Append_Nose_Hoover_Chain_Frame(
            11, 0.2, nhc_coordinates, nhc_velocities, 2));
        REQUIRE_TRUE(writer.Ensure_Sits_Nk_Observable("obs_sits", 2));
        float sits_values[2] = {1.0f, 2.0f};
        REQUIRE_TRUE(
            writer.Append_Sits_Nk_Frame(12, 0.3, "obs_sits", sits_values, 2));
        REQUIRE_TRUE(writer.Ensure_Metadynamics_Scalars());
        REQUIRE_TRUE(
            writer.Append_Metadynamics_Scalar_Frame(13, 0.4, 1.0, 2.0, 3.0));
        REQUIRE_TRUE(writer.Ensure_Reaxff_Energy_Terms({"bond", "angle"}));
        REQUIRE_TRUE(writer.Append_Reaxff_Frame(
            14, 0.5, {{"bond", 4.0}, {"angle", 5.0}}));
        REQUIRE_TRUE(writer.Write_Mdinfo_Text("OBSERVABLE MDINFO"));
        REQUIRE_TRUE(writer.Write_Legacy_Sidecar_Paths(
            {"mdout", "mdinfo"}, {"legacy.mdout", "mdinfo.log"}));
        REQUIRE_TRUE(
            writer.Write_Provenance_String("launch_id", "observable-launch"));
        REQUIRE_TRUE(writer.Finalize());
        REQUIRE_TRUE(writer.Close());
    }

    {
        HighFive::File file(observable_path.string(), HighFive::File::ReadOnly);
        REQUIRE_TRUE(file.exist("/h5md"));
        REQUIRE_TRUE(file.exist("/observables"));
        REQUIRE_TRUE(file.exist("/parameters"));
        REQUIRE_TRUE(!file.exist("/particles"));
        REQUIRE_TRUE(file.exist(Observable_Value_Path("temperature")));
        REQUIRE_TRUE(file.exist(module_path::nhc_coordinate_value));
        REQUIRE_TRUE(file.exist(module_path::nhc_velocity_value));
        REQUIRE_TRUE(file.exist(Sits_Nk_Value_Path("obs_sits")));
        REQUIRE_TRUE(file.exist(Metadynamics_Scalar_Value_Path("meta")));
        REQUIRE_TRUE(file.exist(Metadynamics_Scalar_Value_Path("rbias")));
        REQUIRE_TRUE(file.exist(Metadynamics_Scalar_Value_Path("rct")));
        REQUIRE_TRUE(file.exist(Reaxff_Term_Value_Path("bond")));
        REQUIRE_TRUE(file.exist(Reaxff_Term_Value_Path("angle")));
        REQUIRE_TRUE(file.exist(path::mdinfo_text));
        REQUIRE_TRUE(file.exist(path::legacy_sidecar_keys));
        REQUIRE_TRUE(file.exist(Sponge_Provenance_Path("launch_id")));
        REQUIRE_EQ(file.getDataSet(Observable_Value_Path("temperature"))
                       .getSpace()
                       .getDimensions()[0],
                   static_cast<std::size_t>(1));
        std::vector<float> nhc_coordinates_read;
        std::vector<float> nhc_velocities_read;
        nhc_coordinates_read = Read_Flat_Dataset<float>(
            file.getDataSet(module_path::nhc_coordinate_value));
        nhc_velocities_read = Read_Flat_Dataset<float>(
            file.getDataSet(module_path::nhc_velocity_value));
        REQUIRE_EQ(nhc_coordinates_read[0], 0.1f);
        REQUIRE_EQ(nhc_velocities_read[1], 0.4f);
        std::vector<float> sits_read;
        sits_read = Read_Flat_Dataset<float>(
            file.getDataSet(Sits_Nk_Value_Path("obs_sits")));
        REQUIRE_EQ(sits_read[0], 1.0f);
        REQUIRE_EQ(sits_read[1], 2.0f);
        std::vector<double> metad_meta;
        std::vector<double> metad_rbias;
        std::vector<double> metad_rct;
        metad_meta = Read_Flat_Dataset<double>(
            file.getDataSet(Metadynamics_Scalar_Value_Path("meta")));
        metad_rbias = Read_Flat_Dataset<double>(
            file.getDataSet(Metadynamics_Scalar_Value_Path("rbias")));
        metad_rct = Read_Flat_Dataset<double>(
            file.getDataSet(Metadynamics_Scalar_Value_Path("rct")));
        REQUIRE_EQ(metad_meta[0], 1.0);
        REQUIRE_EQ(metad_rbias[0], 2.0);
        REQUIRE_EQ(metad_rct[0], 3.0);
        std::vector<double> reaxff_bond;
        std::vector<double> reaxff_angle;
        reaxff_bond = Read_Flat_Dataset<double>(
            file.getDataSet(Reaxff_Term_Value_Path("bond")));
        reaxff_angle = Read_Flat_Dataset<double>(
            file.getDataSet(Reaxff_Term_Value_Path("angle")));
        REQUIRE_EQ(reaxff_bond[0], 4.0);
        REQUIRE_EQ(reaxff_angle[0], 5.0);
        REQUIRE_TRUE(file.exist(Observable_Step_Path("temperature")));
        REQUIRE_TRUE(file.exist(Observable_Time_Path("temperature")));
        REQUIRE_EQ(
            Read_Int64_Vector(file, Observable_Step_Path("temperature"))[0],
            static_cast<int64_t>(10));
        REQUIRE_EQ(
            Read_Float64_Vector(file, Observable_Time_Path("temperature"))[0],
            0.1);
        REQUIRE_EQ(Read_Int64_Vector(file, module_path::nhc_coordinate_step)[0],
                   static_cast<int64_t>(11));
        REQUIRE_EQ(
            Read_Float64_Vector(file, module_path::nhc_coordinate_time)[0],
            0.2);
        REQUIRE_EQ(Read_Int64_Vector(file, module_path::nhc_velocity_step)[0],
                   static_cast<int64_t>(11));
        REQUIRE_EQ(Read_Float64_Vector(file, module_path::nhc_velocity_time)[0],
                   0.2);
        REQUIRE_EQ(Read_Int64_Vector(file, Sits_Nk_Step_Path("obs_sits"))[0],
                   static_cast<int64_t>(12));
        REQUIRE_EQ(Read_Float64_Vector(file, Sits_Nk_Time_Path("obs_sits"))[0],
                   0.3);
        REQUIRE_EQ(
            Read_Int64_Vector(file, Metadynamics_Scalar_Step_Path("meta"))[0],
            static_cast<int64_t>(13));
        REQUIRE_EQ(
            Read_Float64_Vector(file, Metadynamics_Scalar_Time_Path("meta"))[0],
            0.4);
        REQUIRE_EQ(
            Read_Int64_Vector(file, Metadynamics_Scalar_Step_Path("rbias"))[0],
            static_cast<int64_t>(13));
        REQUIRE_EQ(Read_Float64_Vector(
                       file, Metadynamics_Scalar_Time_Path("rbias"))[0],
                   0.4);
        REQUIRE_EQ(
            Read_Int64_Vector(file, Metadynamics_Scalar_Step_Path("rct"))[0],
            static_cast<int64_t>(13));
        REQUIRE_EQ(
            Read_Float64_Vector(file, Metadynamics_Scalar_Time_Path("rct"))[0],
            0.4);
        REQUIRE_EQ(Read_Int64_Vector(file, Reaxff_Term_Step_Path("bond"))[0],
                   static_cast<int64_t>(14));
        REQUIRE_EQ(Read_Float64_Vector(file, Reaxff_Term_Time_Path("bond"))[0],
                   0.5);
        REQUIRE_EQ(Read_Int64_Vector(file, Reaxff_Term_Step_Path("angle"))[0],
                   static_cast<int64_t>(14));
        REQUIRE_EQ(Read_Float64_Vector(file, Reaxff_Term_Time_Path("angle"))[0],
                   0.5);
        const auto frame_count =
            Read_Int64_Vector(file, path::output_frame_count);
        REQUIRE_EQ(frame_count.back(), static_cast<int64_t>(1));
        const auto original_columns =
            Read_String_Vector(file, path::mdout_columns_original_name);
        const auto hdf5_columns =
            Read_String_Vector(file, path::mdout_columns_hdf5_name);
        REQUIRE_EQ(original_columns[0], std::string("TEMP"));
        REQUIRE_EQ(hdf5_columns[0], std::string("temperature"));
        REQUIRE_EQ(Read_String(file, path::mdinfo_text),
                   std::string("OBSERVABLE MDINFO"));
        const auto legacy_keys =
            Read_String_Vector(file, path::legacy_sidecar_keys);
        const auto legacy_paths =
            Read_String_Vector(file, path::legacy_sidecar_paths);
        REQUIRE_EQ(legacy_keys[1], std::string("mdinfo"));
        REQUIRE_EQ(legacy_paths[1], std::string("mdinfo.log"));
        REQUIRE_EQ(Read_String(file, Sponge_Provenance_Path("launch_id")),
                   std::string("observable-launch"));
        REQUIRE_EQ(Read_String(file, path::output_status),
                   std::string("finalized"));
    }

    std::filesystem::remove_all(dir);
}

static void Test_Observable_Writer_Missing_Value_With_Real_Backend()
{
    const auto dir = Unique_Temp_Path("observable_writer_missing_value");
    std::filesystem::create_directories(dir);
    const auto observable_path = dir / "observable_missing.obs.spg.h5md";

    {
        HighFiveBackend backend;
        ObservableH5Writer writer(&backend);
        SpongeH5OutputPlan::ResolvedOutputPlan plan;
        plan.observable.enabled = true;
        plan.observable.path = observable_path.string();

        REQUIRE_TRUE(writer.Open(plan, "test"));
        REQUIRE_TRUE(writer.Define_Observable_Stream({"energy", "temperature"},
                                                     {"E", "TEMP"}));
        REQUIRE_TRUE(
            !writer.Append_Observable_Frame(10, 0.1, {{"energy", -1.0}}));
        REQUIRE_TRUE(writer.Close());
    }

    {
        HighFive::File file(observable_path.string(), HighFive::File::ReadOnly);
        REQUIRE_EQ(Read_String(file, path::output_status),
                   std::string("failed"));
        REQUIRE_EQ(Read_String(file, path::output_error),
                   std::string("observable value is missing: temperature"));
        const auto frame_count =
            Read_Int64_Vector(file, path::output_frame_count);
        const auto last_step =
            Read_Int64_Vector(file, path::output_last_complete_step);
        REQUIRE_EQ(frame_count.back(), static_cast<int64_t>(0));
        REQUIRE_EQ(last_step.back(), static_cast<int64_t>(-1));
    }

    std::filesystem::remove_all(dir);
}

static void Test_Module_Metad_And_Reaxff_With_Real_Backend()
{
    const auto dir = Unique_Temp_Path("module_metad_reaxff");
    std::filesystem::create_directories(dir);
    const auto file_path = dir / "module.spg.h5md";

    {
        HighFiveBackend backend;
        H5MDWriter writer(&backend);
        WriterOptions options;
        options.path = file_path.string();
        REQUIRE_TRUE(writer.Open(options));

        ModuleH5MappingWriter module_writer(&writer);
        REQUIRE_TRUE(module_writer.Is_Attached());
        REQUIRE_TRUE(module_writer.Write_Metadynamics_Potential_Export(
            "meta0", "POTENTIAL EXPORT"));
        REQUIRE_TRUE(module_writer.Write_Metadynamics_Direct_Export(
            "meta0", "DIRECT EXPORT"));
        REQUIRE_TRUE(
            module_writer.Write_Metadynamics_Hills("meta0", "HILLS TEXT"));
        REQUIRE_TRUE(
            module_writer.Write_Metadynamics_History("meta0", "HISTORY TEXT"));
        REQUIRE_TRUE(
            module_writer.Write_Metadynamics_Edge("meta0", "EDGE TEXT"));

        REQUIRE_TRUE(module_writer.Ensure_Reaxff_Energy_Terms(
            {"bond", "angle", "over"}));
        REQUIRE_TRUE(module_writer.Append_Reaxff_Frame(
            10, 0.1, {{"bond", 1.0}, {"angle", 2.0}, {"over", 3.0}}));
        REQUIRE_TRUE(module_writer.Ensure_Reaxff_Eeq_Charge_Snapshot(3));
        const std::vector<float> first_charges = {-0.1f, 0.2f, -0.1f};
        const std::vector<float> final_charges = {-0.25f, 0.5f, -0.25f};
        REQUIRE_TRUE(module_writer.Write_Reaxff_Eeq_Charge_Snapshot(
            first_charges.data(), first_charges.size()));
        REQUIRE_TRUE(module_writer.Write_Reaxff_Eeq_Charge_Snapshot(
            final_charges.data(), final_charges.size()));
        REQUIRE_TRUE(writer.Finalize());
        REQUIRE_TRUE(writer.Close());
    }

    {
        HighFive::File file(file_path.string(), HighFive::File::ReadOnly);
        REQUIRE_EQ(Read_String(file, Metadynamics_Diagnostic_Path(
                                         "meta0", "potential_export")),
                   std::string("POTENTIAL EXPORT"));
        REQUIRE_EQ(Read_String(file, Metadynamics_Diagnostic_Path(
                                         "meta0", "direct_export")),
                   std::string("DIRECT EXPORT"));
        REQUIRE_EQ(
            Read_String(file, Metadynamics_Diagnostic_Path("meta0", "hills")),
            std::string("HILLS TEXT"));
        REQUIRE_EQ(
            Read_String(file, Metadynamics_Diagnostic_Path("meta0", "history")),
            std::string("HISTORY TEXT"));
        REQUIRE_EQ(
            Read_String(file, Metadynamics_Diagnostic_Path("meta0", "edge")),
            std::string("EDGE TEXT"));

        std::vector<double> bond;
        std::vector<double> angle;
        std::vector<double> over;
        bond = Read_Flat_Dataset<double>(
            file.getDataSet(Reaxff_Term_Value_Path("bond")));
        angle = Read_Flat_Dataset<double>(
            file.getDataSet(Reaxff_Term_Value_Path("angle")));
        over = Read_Flat_Dataset<double>(
            file.getDataSet(Reaxff_Term_Value_Path("over")));
        REQUIRE_EQ(bond[0], 1.0);
        REQUIRE_EQ(angle[0], 2.0);
        REQUIRE_EQ(over[0], 3.0);
        const auto charges = Read_Flat_Dataset<float>(
            file.getDataSet(module_path::reaxff_eeq_charges_value));
        REQUIRE_EQ(charges, std::vector<float>({-0.25f, 0.5f, -0.25f}));
        REQUIRE_EQ(Read_String(file, path::output_status),
                   std::string("finalized"));
    }

    std::filesystem::remove_all(dir);
}

static void Test_Vds_Finalize_Without_Frames_With_Real_Backend()
{
    const auto dir = Unique_Temp_Path("vds_zero_frames");
    std::filesystem::create_directories(dir);
    const auto wrapper_path = dir / "zero.spg.h5md";
    const auto shard_root = dir / "zero.spg.shards";

    {
        HighFiveBackendFactory factory;
        VdsTrajectoryH5Writer writer(&factory);
        SpongeH5OutputPlan::ResolvedOutputPlan plan;
        plan.trajectory.enabled = true;
        plan.trajectory.path = wrapper_path.string();
        plan.trajectory.vds = true;
        plan.trajectory.chunk_size = 2;
        plan.trajectory.derived_shard_root = shard_root.string();

        REQUIRE_TRUE(writer.Open(plan, "test"));
        REQUIRE_TRUE(writer.Finalize());
        REQUIRE_EQ(writer.Manifest().size(), static_cast<std::size_t>(0));
        REQUIRE_EQ(writer.Total_Trajectory_Frame_Count(),
                   static_cast<std::size_t>(0));
        REQUIRE_EQ(writer.Total_Observable_Frame_Count(),
                   static_cast<std::size_t>(0));
    }

    {
        HighFive::File file(wrapper_path.string(), HighFive::File::ReadOnly);
        REQUIRE_TRUE(file.exist("/h5md"));
        REQUIRE_TRUE(file.exist("/particles"));
        REQUIRE_TRUE(file.exist("/observables"));
        REQUIRE_TRUE(file.exist(path::output_trajectory_chunk_size));
        REQUIRE_TRUE(file.exist(path::output_repair_policy));
        REQUIRE_TRUE(file.exist(path::output_repair_status));
        REQUIRE_TRUE(file.exist(path::output_repaired_shard_count));
        REQUIRE_TRUE(file.exist(path::output_vds_status));
        REQUIRE_TRUE(!file.exist(path::particles_all_step));
        REQUIRE_TRUE(!file.exist(path::particles_all_time));
        REQUIRE_TRUE(!file.exist(path::position_value));
        REQUIRE_TRUE(!file.exist(path::box_edges_value));
        REQUIRE_TRUE(!file.exist(path::observables_all_step));
        REQUIRE_TRUE(!file.exist(path::observables_all_time));
        REQUIRE_TRUE(!file.exist(path::shard_manifest_index));
        REQUIRE_TRUE(!file.exist(path::shard_manifest_path));
        REQUIRE_TRUE(!std::filesystem::exists(shard_root));

        const auto frame_count =
            Read_Int64_Vector(file, path::output_frame_count);
        const auto last_step =
            Read_Int64_Vector(file, path::output_last_complete_step);
        const auto last_time =
            Read_Float64_Vector(file, path::output_last_complete_time);
        const auto repaired_count =
            Read_Int64_Vector(file, path::output_repaired_shard_count);
        REQUIRE_EQ(frame_count.size(), static_cast<std::size_t>(1));
        REQUIRE_EQ(frame_count[0], static_cast<int64_t>(0));
        REQUIRE_EQ(last_step[0], static_cast<int64_t>(-1));
        REQUIRE_EQ(last_time[0], 0.0);
        REQUIRE_EQ(repaired_count[0], static_cast<int64_t>(0));
        REQUIRE_EQ(Read_String(file, path::output_trajectory_chunk_size),
                   std::string("2"));
        REQUIRE_EQ(Read_String(file, path::output_repair_policy),
                   std::string("strict"));
        REQUIRE_EQ(Read_String(file, path::output_repair_status),
                   std::string("not_applied"));
        REQUIRE_EQ(Read_String(file, path::output_vds_status),
                   std::string("particle, observable, and module virtual "
                               "datasets materialized"));
        REQUIRE_EQ(Read_String(file, path::output_status),
                   std::string("finalized"));
    }

    std::filesystem::remove_all(dir);
}

static void Test_Vds_Finalize_With_Only_Observables_With_Real_Backend()
{
    const auto dir = Unique_Temp_Path("vds_observable_only");
    std::filesystem::create_directories(dir);
    const auto wrapper_path = dir / "observable_only.spg.h5md";
    const auto shard_root = dir / "observable_only.spg.shards";

    {
        HighFiveBackendFactory factory;
        VdsTrajectoryH5Writer writer(&factory);
        SpongeH5OutputPlan::ResolvedOutputPlan plan;
        plan.trajectory.enabled = true;
        plan.trajectory.path = wrapper_path.string();
        plan.trajectory.vds = true;
        plan.trajectory.chunk_size = 1000;
        plan.trajectory.derived_shard_root = shard_root.string();

        REQUIRE_TRUE(writer.Open(plan, "test"));
        REQUIRE_TRUE(writer.Define_Particle_Datasets(1, false, false));
        REQUIRE_TRUE(
            writer.Define_Observable_Stream({"temperature"}, {"TEMP"}));
        REQUIRE_TRUE(
            writer.Append_Observable_Frame(10, 0.1, {{"temperature", 299.0}}));
        REQUIRE_TRUE(
            writer.Append_Observable_Frame(20, 0.2, {{"temperature", 300.0}}));
        REQUIRE_TRUE(writer.Finalize());
        REQUIRE_EQ(writer.Total_Trajectory_Frame_Count(),
                   static_cast<std::size_t>(0));
        REQUIRE_EQ(writer.Total_Observable_Frame_Count(),
                   static_cast<std::size_t>(2));
        REQUIRE_EQ(writer.Manifest().size(), static_cast<std::size_t>(1));
        REQUIRE_EQ(writer.Manifest()[0].frame_count, static_cast<int64_t>(0));
        REQUIRE_EQ(writer.Manifest()[0].observable_frame_count,
                   static_cast<int64_t>(2));
    }

    {
        HighFive::File file(wrapper_path.string(), HighFive::File::ReadOnly);
        REQUIRE_TRUE(!file.exist(path::particles_all_step));
        REQUIRE_TRUE(!file.exist(path::position_value));
        REQUIRE_TRUE(file.exist(path::observables_all_step));
        REQUIRE_TRUE(file.exist(Observable_Value_Path("temperature")));
        REQUIRE_EQ(Read_Int64_Vector(file, path::observables_all_step),
                   std::vector<int64_t>({10, 20}));
        REQUIRE_EQ(
            Read_Float64_Vector(file, Observable_Value_Path("temperature")),
            std::vector<double>({299.0, 300.0}));
        REQUIRE_EQ(Read_Int64_Vector(file, path::shard_manifest_frame_count)[0],
                   static_cast<int64_t>(0));
        REQUIRE_EQ(
            Read_Int64_Vector(file, path::shard_manifest_observables_count)[0],
            static_cast<int64_t>(2));
        REQUIRE_EQ(Read_Int64_Vector(file, path::output_frame_count).back(),
                   static_cast<int64_t>(0));
        REQUIRE_EQ(
            Read_Int64_Vector(file, path::output_last_complete_step).back(),
            static_cast<int64_t>(-1));
        REQUIRE_EQ(
            Read_Float64_Vector(file, path::output_last_complete_time).back(),
            0.0);
    }

    REQUIRE_TRUE(
        std::filesystem::exists(shard_root / "segment_000000.spg.h5md"));
    std::filesystem::remove_all(dir);
}

#ifdef _WIN32
static void Test_Atomic_Snapshot_Rejects_Forced_HDF5_File_Locking()
{
    const char* previous_value = std::getenv("HDF5_USE_FILE_LOCKING");
    const bool had_previous_value = previous_value != nullptr;
    const std::string saved_value =
        had_previous_value ? previous_value : std::string();

    REQUIRE_TRUE(_putenv_s("HDF5_USE_FILE_LOCKING", "TRUE") == 0);
    HighFiveBackend backend;
    WriterOptions options;
    options.path =
        Unique_Temp_Path("forced_hdf5_locking").string() + ".spg.h5md";
    options.atomic_snapshot = true;
    const bool opened = backend.Open(options);
    const std::string error = backend.Last_Error();

    REQUIRE_TRUE(_putenv_s("HDF5_USE_FILE_LOCKING", "BEST_EFFORT") == 0);
    HighFiveBackend best_effort_backend;
    options.path =
        Unique_Temp_Path("best_effort_hdf5_locking").string() + ".spg.h5md";
    const bool best_effort_opened = best_effort_backend.Open(options);
    const std::string best_effort_error = best_effort_backend.Last_Error();

    REQUIRE_TRUE(_putenv_s("HDF5_USE_FILE_LOCKING",
                           had_previous_value ? saved_value.c_str() : "") == 0);
    REQUIRE_TRUE(!opened);
    REQUIRE_TRUE(error.find("HDF5_USE_FILE_LOCKING") != std::string::npos);
    REQUIRE_TRUE(error.find("FALSE") != std::string::npos);
    REQUIRE_TRUE(!best_effort_opened);
    REQUIRE_TRUE(best_effort_error.find("HDF5_USE_FILE_LOCKING") !=
                 std::string::npos);
    REQUIRE_TRUE(best_effort_error.find("FALSE") != std::string::npos);
}
#endif

static void Test_Vds_Complete_Prefix_Repair_With_Real_Backend()
{
    const auto dir = Unique_Temp_Path("vds_complete_prefix_repair");
    std::filesystem::create_directories(dir);
    const auto wrapper_path = dir / "repair.spg.h5md";
    const auto shard_root = dir / "repair.spg.shards";

    {
        SelectiveFailHighFiveBackendFactory factory({false, false, true});
        VdsTrajectoryH5Writer writer(&factory);
        SpongeH5OutputPlan::ResolvedOutputPlan plan;
        plan.trajectory.enabled = true;
        plan.trajectory.path = wrapper_path.string();
        plan.trajectory.vds = true;
        plan.trajectory.chunk_size = 1;
        plan.trajectory.derived_shard_root = shard_root.string();

        REQUIRE_TRUE(writer.Open(plan, "test"));
        REQUIRE_TRUE(writer.Define_Particle_Datasets(1, false, false));

        float box[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        float position_0[3] = {1, 0, 0};
        float position_1[3] = {2, 0, 0};
        REQUIRE_TRUE(writer.Append_Particle_Frame(10, 0.1, position_0, box));
        REQUIRE_TRUE(writer.Append_Particle_Frame(20, 0.2, position_1, box));
        REQUIRE_TRUE(writer.Finalize_With_Repair());
        REQUIRE_EQ(writer.Manifest().size(), static_cast<std::size_t>(1));
        REQUIRE_EQ(writer.Total_Trajectory_Frame_Count(),
                   static_cast<std::size_t>(1));
        REQUIRE_EQ(writer.Total_Observable_Frame_Count(),
                   static_cast<std::size_t>(0));
    }

    {
        HighFive::File file(wrapper_path.string(), HighFive::File::ReadOnly);
        REQUIRE_TRUE(file.exist(path::position_value));
        REQUIRE_TRUE(file.exist(path::box_edges_value));
        REQUIRE_TRUE(!file.exist(path::velocity_value));
        REQUIRE_TRUE(!file.exist(path::force_value));
        REQUIRE_TRUE(file.exist(path::shard_manifest_index));
        REQUIRE_TRUE(file.exist(path::shard_manifest_path));
        REQUIRE_TRUE(file.exist(path::shard_manifest_status));

        auto position_dataset = file.getDataSet(path::position_value);
        const auto position_dims = position_dataset.getSpace().getDimensions();
        REQUIRE_EQ(position_dims[0], static_cast<std::size_t>(1));
        REQUIRE_EQ(position_dims[1], static_cast<std::size_t>(1));
        REQUIRE_EQ(position_dims[2], static_cast<std::size_t>(3));
        std::vector<float> positions;
        positions = Read_Flat_Dataset<float>(position_dataset);
        REQUIRE_EQ(positions.size(), static_cast<std::size_t>(3));
        REQUIRE_EQ(positions[0], 1.0f);
        REQUIRE_EQ(positions[1], 0.0f);
        REQUIRE_EQ(positions[2], 0.0f);

        const auto position_steps =
            Read_Int64_Vector(file, path::position_step);
        const auto position_times =
            Read_Float64_Vector(file, path::position_time);
        REQUIRE_EQ(position_steps.size(), static_cast<std::size_t>(1));
        REQUIRE_EQ(position_times.size(), static_cast<std::size_t>(1));
        REQUIRE_EQ(position_steps[0], static_cast<int64_t>(10));
        REQUIRE_EQ(position_times[0], 0.1);

        const auto manifest_indices =
            Read_Int64_Vector(file, path::shard_manifest_index);
        const auto manifest_frame_counts =
            Read_Int64_Vector(file, path::shard_manifest_frame_count);
        const auto manifest_frame_starts =
            Read_Int64_Vector(file, path::shard_manifest_frame_start);
        const auto manifest_step_starts =
            Read_Int64_Vector(file, path::shard_manifest_step_start);
        const auto manifest_step_ends =
            Read_Int64_Vector(file, path::shard_manifest_step_end);
        const auto manifest_time_starts =
            Read_Float64_Vector(file, path::shard_manifest_time_start);
        const auto manifest_time_ends =
            Read_Float64_Vector(file, path::shard_manifest_time_end);
        const auto manifest_paths =
            Read_String_Vector(file, path::shard_manifest_path);
        const auto manifest_statuses =
            Read_String_Vector(file, path::shard_manifest_status);
        REQUIRE_EQ(manifest_indices.size(), static_cast<std::size_t>(1));
        REQUIRE_EQ(manifest_indices[0], static_cast<int64_t>(0));
        REQUIRE_EQ(manifest_frame_counts[0], static_cast<int64_t>(1));
        REQUIRE_EQ(manifest_frame_starts[0], static_cast<int64_t>(0));
        REQUIRE_EQ(manifest_step_starts[0], static_cast<int64_t>(10));
        REQUIRE_EQ(manifest_step_ends[0], static_cast<int64_t>(10));
        REQUIRE_EQ(manifest_time_starts[0], 0.1);
        REQUIRE_EQ(manifest_time_ends[0], 0.1);
        REQUIRE_EQ(manifest_paths.size(), static_cast<std::size_t>(1));
        REQUIRE_EQ(manifest_statuses.size(), static_cast<std::size_t>(1));
        REQUIRE_TRUE(manifest_paths[0].find("segment_000000.spg.h5md") !=
                     std::string::npos);
        REQUIRE_EQ(manifest_statuses[0], std::string("complete"));

        const auto output_frame_count =
            Read_Int64_Vector(file, path::output_frame_count);
        const auto last_step =
            Read_Int64_Vector(file, path::output_last_complete_step);
        const auto last_time =
            Read_Float64_Vector(file, path::output_last_complete_time);
        REQUIRE_EQ(output_frame_count.back(), static_cast<int64_t>(1));
        REQUIRE_EQ(last_step.back(), static_cast<int64_t>(10));
        REQUIRE_EQ(last_time.back(), 0.1);
        REQUIRE_EQ(Read_String(file, path::output_status),
                   std::string("finalized"));
        REQUIRE_EQ(Read_String(file, path::output_trajectory_chunk_size),
                   std::string("1"));
        REQUIRE_EQ(Read_String(file, path::output_repair_policy),
                   std::string("complete_prefix"));
        REQUIRE_EQ(Read_String(file, path::output_repair_status),
                   std::string("applied"));
        const auto repaired_shard_count =
            Read_Int64_Vector(file, path::output_repaired_shard_count);
        REQUIRE_EQ(repaired_shard_count[0], static_cast<int64_t>(1));
    }

    {
        const auto shard0_path = shard_root / "segment_000000.spg.h5md";
        REQUIRE_TRUE(std::filesystem::exists(shard0_path));
        HighFive::File shard0(shard0_path.string(), HighFive::File::ReadOnly);
        REQUIRE_EQ(Read_String(shard0, path::sponge_schema_name),
                   std::string("sponge.output.h5md"));
        REQUIRE_EQ(Read_String(shard0, path::sponge_schema_version),
                   std::string("test"));
        REQUIRE_TRUE(shard0.exist(path::position_value));
        REQUIRE_TRUE(!shard0.exist(path::velocity_value));
        REQUIRE_TRUE(!shard0.exist(path::force_value));
    }

    std::filesystem::remove_all(dir);
}

static void Test_Vds_Trajectory_Writer_With_Real_Backend()
{
    const auto dir = Unique_Temp_Path("vds_writer");
    std::filesystem::create_directories(dir);
    const auto wrapper_path = dir / "trajectory.spg.h5md";
    const auto shard_root = dir / "trajectory.spg.shards";

    {
        HighFiveBackendFactory factory;
        VdsTrajectoryH5Writer writer(&factory);
        SpongeH5OutputPlan::ResolvedOutputPlan plan;
        plan.trajectory.enabled = true;
        plan.trajectory.path = wrapper_path.string();
        plan.trajectory.vds = true;
        plan.trajectory.chunk_size = 1;
        plan.trajectory.derived_shard_root = shard_root.string();

        REQUIRE_TRUE(writer.Open(plan, kOutputSchemaVersion,
                                 "123e4567-e89b-42d3-a456-426614174000"));
        REQUIRE_TRUE(writer.Define_Particle_Datasets(1, false, false));
        REQUIRE_TRUE(
            writer.Define_Observable_Stream({"temperature"}, {"TEMP"}));
        REQUIRE_TRUE(writer.Ensure_Nose_Hoover_Chain_Observables(2));
        REQUIRE_TRUE(writer.Ensure_Sits_Nk_Observable("sits_a", 3));
        REQUIRE_TRUE(writer.Ensure_Metadynamics_Scalars());
        REQUIRE_TRUE(writer.Ensure_Reaxff_Energy_Terms({"bond", "angle"}));
        REQUIRE_TRUE(writer.Write_Metadynamics_Diagnostic("meta0", "hills",
                                                          "HILLS TEXT"));
        REQUIRE_TRUE(writer.Write_Legacy_Sidecar_Paths(
            {"myhill", "mdinfo"}, {"myhill.dat", "mdinfo.log"}));

        float box[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        float position_0[3] = {1, 0, 0};
        float position_1[3] = {2, 0, 0};
        float nhc_0[2] = {0.1f, 0.2f};
        float nhc_1[2] = {0.3f, 0.4f};
        float sits_0[3] = {1.0f, 2.0f, 3.0f};
        float sits_1[3] = {4.0f, 5.0f, 6.0f};
        REQUIRE_TRUE(writer.Append_Reaxff_Frame(
            10, 0.1, {{"bond", 1.5}, {"angle", 3.5}}));
        REQUIRE_TRUE(writer.Append_Particle_Frame(10, 0.1, position_0, box));
        REQUIRE_TRUE(
            writer.Append_Observable_Frame(10, 0.1, {{"temperature", 300.0}}));
        REQUIRE_TRUE(
            writer.Append_Nose_Hoover_Chain_Frame(10, 0.1, nhc_0, nhc_0, 2));
        REQUIRE_TRUE(writer.Append_Sits_Nk_Frame(10, 0.1, "sits_a", sits_0, 3));
        REQUIRE_TRUE(
            writer.Append_Metadynamics_Scalar_Frame(10, 0.1, 1.0, 2.0, 3.0));
        REQUIRE_TRUE(writer.Append_Particle_Frame(20, 0.2, position_1, box));
        REQUIRE_TRUE(
            writer.Append_Observable_Frame(20, 0.2, {{"temperature", 301.0}}));
        REQUIRE_TRUE(
            writer.Append_Nose_Hoover_Chain_Frame(20, 0.2, nhc_1, nhc_1, 2));
        REQUIRE_TRUE(writer.Append_Sits_Nk_Frame(20, 0.2, "sits_a", sits_1, 3));
        REQUIRE_TRUE(
            writer.Append_Metadynamics_Scalar_Frame(20, 0.2, 4.0, 5.0, 6.0));
        REQUIRE_TRUE(writer.Append_Reaxff_Frame(
            20, 0.2, {{"bond", 2.5}, {"angle", 4.5}}));

        {
            HighFive::File live_wrapper(wrapper_path.string(),
                                        HighFive::File::ReadOnly);
            const auto live_dims = live_wrapper.getDataSet(path::position_value)
                                       .getSpace()
                                       .getDimensions();
            REQUIRE_EQ(live_dims[0], static_cast<std::size_t>(1));
            const auto live_positions = Read_Flat_Dataset<float>(
                live_wrapper.getDataSet(path::position_value));
            REQUIRE_EQ(live_positions[0], 1.0f);
            const auto live_manifest = Read_Int64_Vector(
                live_wrapper, path::shard_manifest_frame_count);
            REQUIRE_EQ(live_manifest.size(), static_cast<std::size_t>(1));
            REQUIRE_EQ(live_manifest[0], static_cast<int64_t>(1));
            const auto live_completion =
                Read_Int64_Vector(live_wrapper, path::output_frame_count);
            REQUIRE_EQ(live_completion.back(), static_cast<int64_t>(1));
            const auto live_epochs =
                Read_Int64_Vector(live_wrapper, path::output_publication_epoch);
            REQUIRE_EQ(live_epochs.back(), static_cast<int64_t>(1));
            REQUIRE_EQ(Read_Int64_Vector(
                           live_wrapper,
                           Output_Stream_Committed_Count_Path("particles"))
                           .back(),
                       static_cast<int64_t>(1));
            REQUIRE_EQ(
                Read_Int64_Vector(live_wrapper,
                                  path::shard_manifest_observables_count)[0],
                static_cast<int64_t>(1));
            REQUIRE_EQ(Read_String(live_wrapper, path::output_vds_status),
                       std::string("complete shard prefix published"));
            REQUIRE_EQ(Read_String(live_wrapper, path::output_status),
                       std::string("open"));
            REQUIRE_EQ(Read_String(live_wrapper, path::sponge_schema_version),
                       std::string(kOutputSchemaVersion));
            REQUIRE_TRUE(
                !Read_String(live_wrapper, path::identity_uuid).empty());
            REQUIRE_EQ(Read_String(live_wrapper, path::output_mode),
                       std::string("vds"));
        }
        REQUIRE_TRUE(writer.Finalize());
    }

    {
        HighFive::File file(wrapper_path.string(), HighFive::File::ReadOnly);
        REQUIRE_TRUE(file.exist(path::position_value));
        REQUIRE_TRUE(file.exist(path::position_step));
        REQUIRE_TRUE(file.exist(path::position_time));
        REQUIRE_TRUE(file.exist(path::box_edges_value));
        REQUIRE_TRUE(file.exist(path::box_edges_step));
        REQUIRE_TRUE(file.exist(path::box_edges_time));
        REQUIRE_TRUE(!file.exist(path::velocity_value));
        REQUIRE_TRUE(!file.exist(path::velocity_step));
        REQUIRE_TRUE(!file.exist(path::velocity_time));
        REQUIRE_TRUE(!file.exist(path::force_value));
        REQUIRE_TRUE(!file.exist(path::force_step));
        REQUIRE_TRUE(!file.exist(path::force_time));
        REQUIRE_TRUE(file.exist(Observable_Value_Path("temperature")));
        REQUIRE_TRUE(file.exist(Observable_Step_Path("temperature")));
        REQUIRE_TRUE(file.exist(Observable_Time_Path("temperature")));
        REQUIRE_TRUE(file.exist(module_path::nhc_coordinate_value));
        REQUIRE_TRUE(file.exist(module_path::nhc_coordinate_step));
        REQUIRE_TRUE(file.exist(module_path::nhc_coordinate_time));
        REQUIRE_TRUE(file.exist(module_path::nhc_velocity_value));
        REQUIRE_TRUE(file.exist(module_path::nhc_velocity_step));
        REQUIRE_TRUE(file.exist(module_path::nhc_velocity_time));
        REQUIRE_TRUE(file.exist(Sits_Nk_Value_Path("sits_a")));
        REQUIRE_TRUE(file.exist(Sits_Nk_Step_Path("sits_a")));
        REQUIRE_TRUE(file.exist(Sits_Nk_Time_Path("sits_a")));
        REQUIRE_TRUE(file.exist(Metadynamics_Scalar_Value_Path("meta")));
        REQUIRE_TRUE(file.exist(Metadynamics_Scalar_Step_Path("meta")));
        REQUIRE_TRUE(file.exist(Metadynamics_Scalar_Time_Path("meta")));
        REQUIRE_TRUE(file.exist(Metadynamics_Scalar_Value_Path("rbias")));
        REQUIRE_TRUE(file.exist(Metadynamics_Scalar_Step_Path("rbias")));
        REQUIRE_TRUE(file.exist(Metadynamics_Scalar_Time_Path("rbias")));
        REQUIRE_TRUE(file.exist(Metadynamics_Scalar_Value_Path("rct")));
        REQUIRE_TRUE(file.exist(Metadynamics_Scalar_Step_Path("rct")));
        REQUIRE_TRUE(file.exist(Metadynamics_Scalar_Time_Path("rct")));
        REQUIRE_TRUE(
            file.exist(Metadynamics_Diagnostic_Path("meta0", "hills")));
        REQUIRE_TRUE(file.exist(Reaxff_Term_Value_Path("bond")));
        REQUIRE_TRUE(file.exist(Reaxff_Term_Step_Path("bond")));
        REQUIRE_TRUE(file.exist(Reaxff_Term_Time_Path("bond")));
        REQUIRE_TRUE(file.exist(Reaxff_Term_Value_Path("angle")));
        REQUIRE_TRUE(file.exist(Reaxff_Term_Step_Path("angle")));
        REQUIRE_TRUE(file.exist(Reaxff_Term_Time_Path("angle")));
        REQUIRE_TRUE(file.exist(path::shard_manifest_path));
        REQUIRE_TRUE(file.exist(path::shard_manifest_status));
        REQUIRE_TRUE(file.exist(path::shard_manifest_frame_count));
        REQUIRE_TRUE(file.exist(path::legacy_sidecar_keys));
        REQUIRE_TRUE(file.exist(path::legacy_sidecar_paths));
        REQUIRE_TRUE(file.exist(path::output_trajectory_chunk_size));
        REQUIRE_TRUE(file.exist(path::output_vds_status));
        REQUIRE_TRUE(file.exist(path::output_repaired_shard_count));

        auto position_dataset = file.getDataSet(path::position_value);
        const auto position_dims = position_dataset.getSpace().getDimensions();
        REQUIRE_EQ(position_dims[0], static_cast<std::size_t>(2));
        REQUIRE_EQ(position_dims[1], static_cast<std::size_t>(1));
        REQUIRE_EQ(position_dims[2], static_cast<std::size_t>(3));
        std::vector<float> positions;
        positions = Read_Flat_Dataset<float>(position_dataset);
        REQUIRE_EQ(positions.size(), static_cast<std::size_t>(6));
        REQUIRE_EQ(positions[0], 1.0f);
        REQUIRE_EQ(positions[3], 2.0f);
        const auto vds_position_steps =
            Read_Int64_Vector(file, path::position_step);
        const auto vds_position_times =
            Read_Float64_Vector(file, path::position_time);
        const auto vds_box_steps =
            Read_Int64_Vector(file, path::box_edges_step);
        const auto vds_box_times =
            Read_Float64_Vector(file, path::box_edges_time);
        REQUIRE_EQ(vds_position_steps[0], static_cast<int64_t>(10));
        REQUIRE_EQ(vds_position_steps[1], static_cast<int64_t>(20));
        REQUIRE_EQ(vds_position_times[0], 0.1);
        REQUIRE_EQ(vds_position_times[1], 0.2);
        REQUIRE_EQ(vds_box_steps[0], static_cast<int64_t>(10));
        REQUIRE_EQ(vds_box_steps[1], static_cast<int64_t>(20));
        REQUIRE_EQ(vds_box_times[0], 0.1);
        REQUIRE_EQ(vds_box_times[1], 0.2);

        auto temperature_dataset =
            file.getDataSet(Observable_Value_Path("temperature"));
        REQUIRE_EQ(temperature_dataset.getSpace().getDimensions()[0],
                   static_cast<std::size_t>(2));
        std::vector<double> temperatures;
        temperatures = Read_Flat_Dataset<double>(temperature_dataset);
        REQUIRE_EQ(temperatures.size(), static_cast<std::size_t>(2));
        REQUIRE_EQ(temperatures[0], 300.0);
        REQUIRE_EQ(temperatures[1], 301.0);
        const auto temperature_steps =
            Read_Int64_Vector(file, Observable_Step_Path("temperature"));
        const auto temperature_times =
            Read_Float64_Vector(file, Observable_Time_Path("temperature"));
        REQUIRE_EQ(temperature_steps[0], static_cast<int64_t>(10));
        REQUIRE_EQ(temperature_steps[1], static_cast<int64_t>(20));
        REQUIRE_EQ(temperature_times[0], 0.1);
        REQUIRE_EQ(temperature_times[1], 0.2);
        const auto vds_original_columns = Read_String_Vector(
            file, "/parameters/sponge/mdout/columns/original_name");
        const auto vds_hdf5_columns = Read_String_Vector(
            file, "/parameters/sponge/mdout/columns/hdf5_name");
        REQUIRE_EQ(vds_original_columns[0], std::string("TEMP"));
        REQUIRE_EQ(vds_hdf5_columns[0], std::string("temperature"));

        auto nhc_dataset = file.getDataSet(module_path::nhc_coordinate_value);
        REQUIRE_EQ(nhc_dataset.getSpace().getDimensions()[0],
                   static_cast<std::size_t>(2));
        REQUIRE_EQ(nhc_dataset.getSpace().getDimensions()[1],
                   static_cast<std::size_t>(2));
        std::vector<float> nhc_values;
        nhc_values = Read_Flat_Dataset<float>(nhc_dataset);
        REQUIRE_EQ(nhc_values[0], 0.1f);
        REQUIRE_EQ(nhc_values[2], 0.3f);
        const auto nhc_coordinate_steps =
            Read_Int64_Vector(file, module_path::nhc_coordinate_step);
        const auto nhc_coordinate_times =
            Read_Float64_Vector(file, module_path::nhc_coordinate_time);
        const auto nhc_velocity_steps =
            Read_Int64_Vector(file, module_path::nhc_velocity_step);
        const auto nhc_velocity_times =
            Read_Float64_Vector(file, module_path::nhc_velocity_time);
        REQUIRE_EQ(nhc_coordinate_steps[0], static_cast<int64_t>(10));
        REQUIRE_EQ(nhc_coordinate_steps[1], static_cast<int64_t>(20));
        REQUIRE_EQ(nhc_coordinate_times[0], 0.1);
        REQUIRE_EQ(nhc_coordinate_times[1], 0.2);
        REQUIRE_EQ(nhc_velocity_steps[0], static_cast<int64_t>(10));
        REQUIRE_EQ(nhc_velocity_steps[1], static_cast<int64_t>(20));
        REQUIRE_EQ(nhc_velocity_times[0], 0.1);
        REQUIRE_EQ(nhc_velocity_times[1], 0.2);

        auto sits_dataset = file.getDataSet(Sits_Nk_Value_Path("sits_a"));
        REQUIRE_EQ(sits_dataset.getSpace().getDimensions()[0],
                   static_cast<std::size_t>(2));
        REQUIRE_EQ(sits_dataset.getSpace().getDimensions()[1],
                   static_cast<std::size_t>(3));
        std::vector<float> sits_values;
        sits_values = Read_Flat_Dataset<float>(sits_dataset);
        REQUIRE_EQ(sits_values[0], 1.0f);
        REQUIRE_EQ(sits_values[3], 4.0f);
        const auto sits_steps =
            Read_Int64_Vector(file, Sits_Nk_Step_Path("sits_a"));
        const auto sits_times =
            Read_Float64_Vector(file, Sits_Nk_Time_Path("sits_a"));
        REQUIRE_EQ(sits_steps[0], static_cast<int64_t>(10));
        REQUIRE_EQ(sits_steps[1], static_cast<int64_t>(20));
        REQUIRE_EQ(sits_times[0], 0.1);
        REQUIRE_EQ(sits_times[1], 0.2);

        std::vector<double> metad_meta;
        std::vector<double> metad_rbias;
        std::vector<double> metad_rct;
        metad_meta = Read_Flat_Dataset<double>(
            file.getDataSet(Metadynamics_Scalar_Value_Path("meta")));
        metad_rbias = Read_Flat_Dataset<double>(
            file.getDataSet(Metadynamics_Scalar_Value_Path("rbias")));
        metad_rct = Read_Flat_Dataset<double>(
            file.getDataSet(Metadynamics_Scalar_Value_Path("rct")));
        REQUIRE_EQ(metad_meta[0], 1.0);
        REQUIRE_EQ(metad_meta[1], 4.0);
        REQUIRE_EQ(metad_rbias[0], 2.0);
        REQUIRE_EQ(metad_rbias[1], 5.0);
        REQUIRE_EQ(metad_rct[0], 3.0);
        REQUIRE_EQ(metad_rct[1], 6.0);
        const auto metad_steps =
            Read_Int64_Vector(file, Metadynamics_Scalar_Step_Path("meta"));
        const auto metad_times =
            Read_Float64_Vector(file, Metadynamics_Scalar_Time_Path("meta"));
        REQUIRE_EQ(metad_steps[0], static_cast<int64_t>(10));
        REQUIRE_EQ(metad_steps[1], static_cast<int64_t>(20));
        REQUIRE_EQ(metad_times[0], 0.1);
        REQUIRE_EQ(metad_times[1], 0.2);
        REQUIRE_EQ(
            Read_Int64_Vector(file, Metadynamics_Scalar_Step_Path("rbias"))[1],
            static_cast<int64_t>(20));
        REQUIRE_EQ(Read_Float64_Vector(
                       file, Metadynamics_Scalar_Time_Path("rbias"))[1],
                   0.2);
        REQUIRE_EQ(
            Read_Int64_Vector(file, Metadynamics_Scalar_Step_Path("rct"))[1],
            static_cast<int64_t>(20));
        REQUIRE_EQ(
            Read_Float64_Vector(file, Metadynamics_Scalar_Time_Path("rct"))[1],
            0.2);
        REQUIRE_EQ(
            Read_String(file, Metadynamics_Diagnostic_Path("meta0", "hills")),
            std::string("HILLS TEXT"));

        std::vector<double> reaxff_bond;
        std::vector<double> reaxff_angle;
        reaxff_bond = Read_Flat_Dataset<double>(
            file.getDataSet(Reaxff_Term_Value_Path("bond")));
        reaxff_angle = Read_Flat_Dataset<double>(
            file.getDataSet(Reaxff_Term_Value_Path("angle")));
        REQUIRE_EQ(reaxff_bond[0], 1.5);
        REQUIRE_EQ(reaxff_bond[1], 2.5);
        REQUIRE_EQ(reaxff_angle[0], 3.5);
        REQUIRE_EQ(reaxff_angle[1], 4.5);
        const auto reaxff_steps =
            Read_Int64_Vector(file, Reaxff_Term_Step_Path("bond"));
        const auto reaxff_times =
            Read_Float64_Vector(file, Reaxff_Term_Time_Path("bond"));
        REQUIRE_EQ(reaxff_steps[0], static_cast<int64_t>(10));
        REQUIRE_EQ(reaxff_steps[1], static_cast<int64_t>(20));
        REQUIRE_EQ(reaxff_times[0], 0.1);
        REQUIRE_EQ(reaxff_times[1], 0.2);
        REQUIRE_EQ(Read_Int64_Vector(file, Reaxff_Term_Step_Path("angle"))[1],
                   static_cast<int64_t>(20));
        REQUIRE_EQ(Read_Float64_Vector(file, Reaxff_Term_Time_Path("angle"))[1],
                   0.2);

        auto manifest_count = file.getDataSet(path::shard_manifest_frame_count);
        std::vector<int64_t> frame_counts;
        frame_counts = Read_Flat_Dataset<int64_t>(manifest_count);
        REQUIRE_EQ(frame_counts.size(), static_cast<std::size_t>(2));
        REQUIRE_EQ(frame_counts[0], static_cast<int64_t>(1));
        REQUIRE_EQ(frame_counts[1], static_cast<int64_t>(1));
        const auto manifest_particle_counts =
            Read_Int64_Vector(file, path::shard_manifest_particles_count);
        const auto manifest_observable_counts =
            Read_Int64_Vector(file, path::shard_manifest_observables_count);
        const auto manifest_metadynamics_counts =
            Read_Int64_Vector(file, path::shard_manifest_metadynamics_count);
        REQUIRE_EQ(manifest_particle_counts[0], static_cast<int64_t>(1));
        REQUIRE_EQ(manifest_particle_counts[1], static_cast<int64_t>(1));
        REQUIRE_EQ(manifest_observable_counts[0], static_cast<int64_t>(1));
        REQUIRE_EQ(manifest_observable_counts[1], static_cast<int64_t>(1));
        REQUIRE_EQ(manifest_metadynamics_counts[0], static_cast<int64_t>(1));
        REQUIRE_EQ(manifest_metadynamics_counts[1], static_cast<int64_t>(1));
        const auto manifest_indices =
            Read_Int64_Vector(file, path::shard_manifest_index);
        const auto manifest_frame_starts =
            Read_Int64_Vector(file, path::shard_manifest_frame_start);
        const auto manifest_step_starts =
            Read_Int64_Vector(file, path::shard_manifest_step_start);
        const auto manifest_step_ends =
            Read_Int64_Vector(file, path::shard_manifest_step_end);
        const auto manifest_time_starts =
            Read_Float64_Vector(file, path::shard_manifest_time_start);
        const auto manifest_time_ends =
            Read_Float64_Vector(file, path::shard_manifest_time_end);
        REQUIRE_EQ(manifest_indices[0], static_cast<int64_t>(0));
        REQUIRE_EQ(manifest_indices[1], static_cast<int64_t>(1));
        REQUIRE_EQ(manifest_frame_starts[0], static_cast<int64_t>(0));
        REQUIRE_EQ(manifest_frame_starts[1], static_cast<int64_t>(1));
        REQUIRE_EQ(manifest_step_starts[0], static_cast<int64_t>(10));
        REQUIRE_EQ(manifest_step_starts[1], static_cast<int64_t>(20));
        REQUIRE_EQ(manifest_step_ends[0], static_cast<int64_t>(10));
        REQUIRE_EQ(manifest_step_ends[1], static_cast<int64_t>(20));
        REQUIRE_EQ(manifest_time_starts[0], 0.1);
        REQUIRE_EQ(manifest_time_starts[1], 0.2);
        REQUIRE_EQ(manifest_time_ends[0], 0.1);
        REQUIRE_EQ(manifest_time_ends[1], 0.2);
        const auto manifest_paths =
            Read_String_Vector(file, path::shard_manifest_path);
        const auto manifest_byte_sizes =
            Read_Int64_Vector(file, path::shard_manifest_byte_size);
        const auto manifest_statuses =
            Read_String_Vector(file, path::shard_manifest_status);
        REQUIRE_EQ(manifest_paths.size(), static_cast<std::size_t>(2));
        REQUIRE_EQ(manifest_byte_sizes.size(), static_cast<std::size_t>(2));
        REQUIRE_EQ(manifest_statuses.size(), static_cast<std::size_t>(2));
        REQUIRE_EQ(manifest_statuses[0], std::string("complete"));
        REQUIRE_EQ(manifest_statuses[1], std::string("complete"));
        REQUIRE_EQ(
            manifest_paths[0],
            std::string("trajectory.spg.shards/segment_000000.spg.h5md"));
        REQUIRE_EQ(
            manifest_paths[1],
            std::string("trajectory.spg.shards/segment_000001.spg.h5md"));
        REQUIRE_TRUE(manifest_byte_sizes[0] > 0);
        REQUIRE_TRUE(manifest_byte_sizes[1] > 0);
        const auto legacy_keys =
            Read_String_Vector(file, path::legacy_sidecar_keys);
        const auto legacy_paths =
            Read_String_Vector(file, path::legacy_sidecar_paths);
        REQUIRE_EQ(legacy_keys[0], std::string("myhill"));
        REQUIRE_EQ(legacy_keys[1], std::string("mdinfo"));
        REQUIRE_EQ(legacy_paths[0], std::string("myhill.dat"));
        REQUIRE_EQ(legacy_paths[1], std::string("mdinfo.log"));
        const auto output_frame_count =
            Read_Int64_Vector(file, path::output_frame_count);
        const auto last_step =
            Read_Int64_Vector(file, path::output_last_complete_step);
        REQUIRE_EQ(output_frame_count.back(), static_cast<int64_t>(2));
        const auto publication_epochs =
            Read_Int64_Vector(file, path::output_publication_epoch);
        REQUIRE_EQ(publication_epochs.size(), static_cast<std::size_t>(3));
        REQUIRE_EQ(publication_epochs[0], static_cast<int64_t>(0));
        REQUIRE_EQ(publication_epochs[1], static_cast<int64_t>(1));
        REQUIRE_EQ(publication_epochs[2], static_cast<int64_t>(2));
        REQUIRE_EQ(Read_Int64_Vector(
                       file, Output_Stream_Committed_Count_Path("particles"))
                       .back(),
                   static_cast<int64_t>(2));
        REQUIRE_EQ(Read_Int64_Vector(
                       file, Output_Stream_Committed_Count_Path("observables"))
                       .back(),
                   static_cast<int64_t>(2));
        REQUIRE_EQ(Read_Int64_Vector(
                       file, Output_Stream_Committed_Count_Path("metadynamics"))
                       .back(),
                   static_cast<int64_t>(2));
        REQUIRE_EQ(last_step.back(), static_cast<int64_t>(20));
        REQUIRE_EQ(Read_String(file, path::output_status),
                   std::string("finalized"));
        REQUIRE_TRUE(!std::filesystem::exists(
            std::filesystem::path(wrapper_path.string() + ".working")));
        REQUIRE_EQ(Read_String(file, path::output_trajectory_chunk_size),
                   std::string("1"));
        REQUIRE_EQ(Read_String(file, path::output_vds_status),
                   std::string("particle, observable, and module virtual "
                               "datasets materialized"));
        REQUIRE_EQ(Read_String(file, path::output_repair_policy),
                   std::string("strict"));
        REQUIRE_EQ(Read_String(file, path::output_repair_status),
                   std::string("not_applied"));
        const auto repaired_shard_count =
            Read_Int64_Vector(file, path::output_repaired_shard_count);
        REQUIRE_EQ(repaired_shard_count[0], static_cast<int64_t>(0));
    }

    {
        const auto shard0_path = shard_root / "segment_000000.spg.h5md";
        const auto shard1_path = shard_root / "segment_000001.spg.h5md";
        REQUIRE_TRUE(std::filesystem::exists(shard0_path));
        REQUIRE_TRUE(std::filesystem::exists(shard1_path));

        HighFive::File shard0(shard0_path.string(), HighFive::File::ReadOnly);
        HighFive::File shard1(shard1_path.string(), HighFive::File::ReadOnly);
        REQUIRE_EQ(Read_String(shard0, path::sponge_schema_name),
                   std::string("sponge.output.h5md"));
        REQUIRE_EQ(Read_String(shard1, path::sponge_schema_name),
                   std::string("sponge.output.h5md"));
        REQUIRE_EQ(Read_String(shard0, path::sponge_schema_version),
                   std::string(kOutputSchemaVersion));
        REQUIRE_EQ(Read_String(shard1, path::sponge_schema_version),
                   std::string(kOutputSchemaVersion));
        REQUIRE_EQ(Read_String(shard0, path::identity_uuid),
                   std::string("123e4567-e89b-42d3-a456-426614174000"));
        REQUIRE_EQ(Read_String(shard1, path::identity_uuid),
                   std::string("123e4567-e89b-42d3-a456-426614174000"));
        REQUIRE_TRUE(shard0.exist(path::position_value));
        REQUIRE_TRUE(shard1.exist(path::position_value));
        REQUIRE_TRUE(shard0.exist(path::box_edges_value));
        REQUIRE_TRUE(shard1.exist(path::box_edges_value));
        REQUIRE_TRUE(!shard0.exist(path::velocity_value));
        REQUIRE_TRUE(!shard1.exist(path::velocity_value));
        REQUIRE_TRUE(!shard0.exist(path::velocity_step));
        REQUIRE_TRUE(!shard1.exist(path::velocity_step));
        REQUIRE_TRUE(!shard0.exist(path::velocity_time));
        REQUIRE_TRUE(!shard1.exist(path::velocity_time));
        REQUIRE_TRUE(!shard0.exist(path::force_value));
        REQUIRE_TRUE(!shard1.exist(path::force_value));
        REQUIRE_TRUE(!shard0.exist(path::force_step));
        REQUIRE_TRUE(!shard1.exist(path::force_step));
        REQUIRE_TRUE(!shard0.exist(path::force_time));
        REQUIRE_TRUE(!shard1.exist(path::force_time));
    }

    std::filesystem::remove_all(dir);
}

int main()
{
    return Run_Test(
        []
        {
            Test_H5MD_Writer_Detached_Backend_Semantics();
            Test_H5MD_Writer_Repeated_Output_Completion();
            Test_HighFive_Backend_Basic_File_Layout();
            Test_HighFive_Backend_Nested_Group_Idempotence();
            Test_HighFive_Backend_Factory_And_Dataset_Reopen_Semantics();
            Test_HighFive_Backend_String_Overwrite();
            Test_HighFive_Backend_Observable_Only_Layout();
            Test_HighFive_Backend_Rejects_Invalid_Operations();
            Test_HighFive_Backend_Failed_Metadata();
            Test_HighFive_Backend_Status_State();
            Test_Output_Completion_Tracker_With_Real_Backend();
            Test_HighFive_Backend_Virtual_Dataset();
            Test_Trajectory_Writer_With_Real_Backend();
            Test_Trajectory_Optional_Particle_Fields_With_Real_Backend();
            Test_Restart_Writer_With_Real_Backend();
            Test_Atomic_File_Replacement_Preserves_Last_Published_File();
            Test_Restart_Optional_Velocity_With_Real_Backend();
            Test_Restart_Writer_Custom_Run_Metadata_With_Real_Backend();
            Test_Restart_Writer_Rejects_Second_State_With_Real_Backend();
            Test_Observable_Writer_With_Real_Backend();
            Test_Observable_Writer_Missing_Value_With_Real_Backend();
            Test_Module_Metad_And_Reaxff_With_Real_Backend();
            Test_Vds_Finalize_Without_Frames_With_Real_Backend();
            Test_Vds_Finalize_With_Only_Observables_With_Real_Backend();
            Test_Vds_Complete_Prefix_Repair_With_Real_Backend();
            Test_Vds_Trajectory_Writer_With_Real_Backend();
#ifdef _WIN32
            Test_Atomic_Snapshot_Rejects_Forced_HDF5_File_Locking();
#endif
        });
}

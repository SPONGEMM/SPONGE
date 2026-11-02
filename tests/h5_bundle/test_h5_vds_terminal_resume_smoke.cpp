#include <cstdio>
#include <cstdlib>
#include <highfive/highfive.hpp>
#include <iostream>
#include <numeric>
#include <type_traits>

#include "h5_bundle_test_common.hpp"
#include "utils/h5md/highfive_backend.hpp"
#include "utils/h5md/vds_trajectory_h5_writer.hpp"

using namespace SpongeH5Test;
using namespace SpongeH5MD;

namespace
{
constexpr int kSkipReturnCode = 77;

void Require_Runtime_Smoke_Enabled()
{
    const char* enabled = std::getenv("SPONGE_H5_ENABLE_RUNTIME_SMOKE");
    if (enabled == nullptr || std::string(enabled) != "1")
    {
        std::cerr << "Skipping VDS terminal/resume smoke; set "
                     "SPONGE_H5_ENABLE_RUNTIME_SMOKE=1 to run it.\n";
        std::exit(kSkipReturnCode);
    }
}

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

SpongeH5OutputPlan::ResolvedOutputPlan Make_Vds_Plan(
    const std::filesystem::path& wrapper_path,
    const std::filesystem::path& shard_root)
{
    SpongeH5OutputPlan::ResolvedOutputPlan plan;
    plan.trajectory.enabled = true;
    plan.trajectory.path = wrapper_path.string();
    plan.trajectory.vds = true;
    plan.trajectory.chunk_size = 1;
    plan.trajectory.derived_shard_root = shard_root.string();
    return plan;
}

template <typename T>
hid_t Native_H5_Type()
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
std::vector<T> Read_Flat_Dataset(HighFive::DataSet dataset)
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

std::vector<int64_t> Read_Int64_Vector(HighFive::File& file,
                                       const std::string& path_name)
{
    return Read_Flat_Dataset<int64_t>(file.getDataSet(path_name));
}

std::vector<double> Read_Float64_Vector(HighFive::File& file,
                                        const std::string& path_name)
{
    return Read_Flat_Dataset<double>(file.getDataSet(path_name));
}

std::string Read_String(HighFive::File& file, const std::string& path_name)
{
    std::string value;
    file.getDataSet(path_name).read(value);
    return value;
}

std::vector<std::string> Read_String_Vector(HighFive::File& file,
                                            const std::string& path_name)
{
    std::vector<std::string> values;
    file.getDataSet(path_name).read(values);
    return values;
}

void Require_Common_Wrapper_Metadata(HighFive::File& file,
                                     const std::string& repair_status,
                                     int64_t repaired_shard_count)
{
    REQUIRE_EQ(Read_String(file, path::output_status),
               std::string("finalized"));
    REQUIRE_EQ(Read_String(file, path::output_trajectory_chunk_size),
               std::string("1"));
    REQUIRE_EQ(Read_String(file, path::output_repair_policy),
               std::string("complete_prefix"));
    REQUIRE_EQ(Read_String(file, path::output_repair_status), repair_status);
    REQUIRE_EQ(
        Read_String(file, path::output_vds_status),
        std::string(
            "particle, observable, and module virtual datasets materialized"));
    const auto repaired_count =
        Read_Int64_Vector(file, path::output_repaired_shard_count);
    REQUIRE_EQ(repaired_count.size(), static_cast<std::size_t>(1));
    REQUIRE_EQ(repaired_count[0], repaired_shard_count);
}

void Require_Position_Vds(HighFive::File& file,
                          const std::vector<float>& expected_positions)
{
    auto position_dataset = file.getDataSet(path::position_value);
    const auto position_dims = position_dataset.getSpace().getDimensions();
    REQUIRE_EQ(position_dims[0], expected_positions.size() / 3);
    REQUIRE_EQ(position_dims[1], static_cast<std::size_t>(1));
    REQUIRE_EQ(position_dims[2], static_cast<std::size_t>(3));
    const auto positions = Read_Flat_Dataset<float>(position_dataset);
    REQUIRE_EQ(positions.size(), expected_positions.size());
    for (std::size_t i = 0; i < expected_positions.size(); ++i)
    {
        REQUIRE_EQ(positions[i], expected_positions[i]);
    }
}

void Require_Manifest_Prefix(HighFive::File& file,
                             const std::size_t expected_count)
{
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
    const auto manifest_paths =
        Read_String_Vector(file, path::shard_manifest_path);
    const auto manifest_statuses =
        Read_String_Vector(file, path::shard_manifest_status);

    REQUIRE_EQ(manifest_indices.size(), expected_count);
    REQUIRE_EQ(manifest_frame_counts.size(), expected_count);
    REQUIRE_EQ(manifest_frame_starts.size(), expected_count);
    REQUIRE_EQ(manifest_step_starts.size(), expected_count);
    REQUIRE_EQ(manifest_step_ends.size(), expected_count);
    REQUIRE_EQ(manifest_paths.size(), expected_count);
    REQUIRE_EQ(manifest_statuses.size(), expected_count);

    for (std::size_t i = 0; i < expected_count; ++i)
    {
        const int64_t index = static_cast<int64_t>(i);
        const int64_t step = static_cast<int64_t>((i + 1) * 10);
        char shard_name[64];
        std::snprintf(shard_name, sizeof(shard_name), "segment_%06lld.spg.h5md",
                      static_cast<long long>(index));

        REQUIRE_EQ(manifest_indices[i], index);
        REQUIRE_EQ(manifest_frame_counts[i], static_cast<int64_t>(1));
        REQUIRE_EQ(manifest_frame_starts[i], index);
        REQUIRE_EQ(manifest_step_starts[i], step);
        REQUIRE_EQ(manifest_step_ends[i], step);
        REQUIRE_TRUE(manifest_paths[i].find(shard_name) != std::string::npos);
        REQUIRE_EQ(manifest_statuses[i], std::string("complete"));
    }
}

void Append_Two_One_Atom_Frames(VdsTrajectoryH5Writer& writer)
{
    float box[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    float position_0[3] = {1, 0, 0};
    float position_1[3] = {2, 0, 0};
    REQUIRE_TRUE(writer.Append_Particle_Frame(10, 0.1, position_0, box));
    REQUIRE_TRUE(writer.Append_Particle_Frame(20, 0.2, position_1, box));
}

void Test_Vds_Terminal_Tail_Is_Repaired_To_Complete_Prefix()
{
    const auto dir = Unique_Temp_Path("vds_terminal_tail_repair_smoke");
    std::filesystem::create_directories(dir);
    const auto wrapper_path = dir / "repair.spg.h5md";
    const auto shard_root = dir / "repair.spg.shards";

    {
        SelectiveFailHighFiveBackendFactory factory({false, false, true});
        VdsTrajectoryH5Writer writer(&factory);
        REQUIRE_TRUE(
            writer.Open(Make_Vds_Plan(wrapper_path, shard_root), "smoke"));
        REQUIRE_TRUE(writer.Define_Particle_Datasets(1, false, false));
        Append_Two_One_Atom_Frames(writer);
        REQUIRE_TRUE(writer.Finalize_With_Repair());
        REQUIRE_EQ(writer.Manifest().size(), static_cast<std::size_t>(1));
        REQUIRE_EQ(writer.Total_Trajectory_Frame_Count(),
                   static_cast<std::size_t>(1));
    }

    {
        HighFive::File file(wrapper_path.string(), HighFive::File::ReadOnly);
        REQUIRE_TRUE(file.exist(path::position_value));
        REQUIRE_TRUE(file.exist(path::box_edges_value));
        REQUIRE_TRUE(!file.exist(path::velocity_value));
        REQUIRE_TRUE(!file.exist(path::force_value));
        Require_Position_Vds(file, {1.0f, 0.0f, 0.0f});
        Require_Manifest_Prefix(file, 1);

        const auto frame_count =
            Read_Int64_Vector(file, path::output_frame_count);
        const auto last_step =
            Read_Int64_Vector(file, path::output_last_complete_step);
        const auto last_time =
            Read_Float64_Vector(file, path::output_last_complete_time);
        REQUIRE_EQ(frame_count.back(), static_cast<int64_t>(1));
        REQUIRE_EQ(last_step.back(), static_cast<int64_t>(10));
        REQUIRE_EQ(last_time.back(), 0.1);
        Require_Common_Wrapper_Metadata(file, "applied", 1);
    }

    {
        const auto shard0_path = shard_root / "segment_000000.spg.h5md";
        REQUIRE_TRUE(std::filesystem::exists(shard0_path));
        HighFive::File shard0(shard0_path.string(), HighFive::File::ReadOnly);
        REQUIRE_EQ(Read_String(shard0, path::sponge_schema_name),
                   std::string("sponge.output.h5md"));
        REQUIRE_EQ(Read_String(shard0, path::sponge_schema_version),
                   std::string("smoke"));
        REQUIRE_TRUE(shard0.exist(path::position_value));
    }

    std::filesystem::remove_all(dir);
}

void Test_Vds_Resume_Policy_Noops_When_Terminal_Shards_Are_Complete()
{
    const auto dir = Unique_Temp_Path("vds_resume_policy_noop_smoke");
    std::filesystem::create_directories(dir);
    const auto wrapper_path = dir / "resume.spg.h5md";
    const auto shard_root = dir / "resume.spg.shards";

    {
        HighFiveBackendFactory factory;
        VdsTrajectoryH5Writer writer(&factory);
        REQUIRE_TRUE(
            writer.Open(Make_Vds_Plan(wrapper_path, shard_root), "smoke"));
        REQUIRE_TRUE(writer.Define_Particle_Datasets(1, false, false));
        Append_Two_One_Atom_Frames(writer);
        REQUIRE_TRUE(writer.Finalize_With_Repair());
        REQUIRE_EQ(writer.Manifest().size(), static_cast<std::size_t>(2));
        REQUIRE_EQ(writer.Total_Trajectory_Frame_Count(),
                   static_cast<std::size_t>(2));
    }

    {
        HighFive::File file(wrapper_path.string(), HighFive::File::ReadOnly);
        Require_Position_Vds(file, {1.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f});
        Require_Manifest_Prefix(file, 2);

        const auto frame_count =
            Read_Int64_Vector(file, path::output_frame_count);
        const auto last_step =
            Read_Int64_Vector(file, path::output_last_complete_step);
        const auto last_time =
            Read_Float64_Vector(file, path::output_last_complete_time);
        REQUIRE_EQ(frame_count.back(), static_cast<int64_t>(2));
        REQUIRE_EQ(last_step.back(), static_cast<int64_t>(20));
        REQUIRE_EQ(last_time.back(), 0.2);
        Require_Common_Wrapper_Metadata(file, "not_applied", 0);
    }

    REQUIRE_TRUE(
        std::filesystem::exists(shard_root / "segment_000000.spg.h5md"));
    REQUIRE_TRUE(
        std::filesystem::exists(shard_root / "segment_000001.spg.h5md"));
    std::filesystem::remove_all(dir);
}
}  // namespace

int main()
{
    Require_Runtime_Smoke_Enabled();
    return Run_Test(
        []
        {
            Test_Vds_Terminal_Tail_Is_Repaired_To_Complete_Prefix();
            Test_Vds_Resume_Policy_Noops_When_Terminal_Shards_Are_Complete();
        });
}

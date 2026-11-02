#include <array>
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

#include "h5_bundle_test_common.hpp"
#include "utils/h5md/highfive_backend.hpp"
#include "utils/h5md/trajectory_h5_reader.hpp"
#include "utils/h5md/trajectory_h5_writer.hpp"
#include "utils/h5md/vds_trajectory_h5_writer.hpp"

using namespace SpongeH5Test;
using namespace SpongeH5MD;

static void Require_Writer(bool ok, const TrajectoryH5Writer& writer,
                           const char* operation)
{
    if (!ok)
    {
        throw TestFailure(std::string(operation) + ": " + writer.Last_Error());
    }
}

static void Require_Writer(bool ok, const VdsTrajectoryH5Writer& writer,
                           const char* operation)
{
    if (!ok)
    {
        throw TestFailure(std::string(operation) + ": " + writer.Last_Error());
    }
}

static void Require_Reader(bool ok, const TrajectoryH5Reader& reader,
                           const char* operation)
{
    if (!ok)
    {
        throw TestFailure(std::string(operation) + ": " + reader.Last_Error());
    }
}

static SpongeH5OutputPlan::ResolvedOutputPlan Make_Trajectory_Plan(
    const std::filesystem::path& trajectory_path)
{
    SpongeH5OutputPlan::ResolvedOutputPlan plan;
    plan.trajectory.enabled = true;
    plan.trajectory.path = trajectory_path.string();
    plan.trajectory.vds = false;
    return plan;
}

static SpongeH5OutputPlan::ResolvedOutputPlan Make_Vds_Trajectory_Plan(
    const std::filesystem::path& trajectory_path,
    const std::filesystem::path& shard_root)
{
    SpongeH5OutputPlan::ResolvedOutputPlan plan =
        Make_Trajectory_Plan(trajectory_path);
    plan.trajectory.vds = true;
    plan.trajectory.chunk_size = 1;
    plan.trajectory.derived_shard_root = shard_root.string();
    return plan;
}

static void Write_Trajectory_File(const std::filesystem::path& file_path)
{
    const std::array<float, 9> box0 = {
        10.0f, 0.0f, 0.0f, 0.0f, 20.0f, 0.0f, 0.0f, 0.0f, 30.0f,
    };
    const std::array<float, 9> box1 = {
        11.0f, 0.0f, 0.0f, 0.0f, 21.0f, 0.0f, 0.0f, 0.0f, 31.0f,
    };
    const std::vector<float> pos0 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const std::vector<float> pos1 = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    const std::vector<float> vel0 = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
    const std::vector<float> vel1 = {0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f};

    HighFiveBackend backend;
    TrajectoryH5Writer writer(&backend);
    Require_Writer(
        writer.Open_Single_File(Make_Trajectory_Plan(file_path), "1"), writer,
        "open trajectory writer");
    Require_Writer(writer.Define_Particle_Datasets(2, true, false), writer,
                   "define trajectory particle datasets");
    Require_Writer(writer.Append_Particle_Frame(10, 0.02, pos0.data(),
                                                box0.data(), vel0.data()),
                   writer, "append first trajectory frame");
    Require_Writer(writer.Append_Particle_Frame(20, 0.04, pos1.data(),
                                                box1.data(), vel1.data()),
                   writer, "append second trajectory frame");
    Require_Writer(writer.Finalize(), writer, "finalize trajectory writer");
    Require_Writer(writer.Close(), writer, "close trajectory writer");
}

static void Write_Vds_Trajectory_File(const std::filesystem::path& file_path,
                                      const std::filesystem::path& shard_root)
{
    const std::array<float, 9> box0 = {
        10.0f, 0.0f, 0.0f, 0.0f, 20.0f, 0.0f, 0.0f, 0.0f, 30.0f,
    };
    const std::array<float, 9> box1 = {
        12.0f, 0.0f, 0.0f, 0.0f, 22.0f, 0.0f, 0.0f, 0.0f, 32.0f,
    };
    const std::vector<float> pos0 = {1.0f, 2.0f, 3.0f};
    const std::vector<float> pos1 = {4.0f, 5.0f, 6.0f};

    HighFiveBackendFactory factory;
    VdsTrajectoryH5Writer writer(&factory);
    Require_Writer(
        writer.Open(Make_Vds_Trajectory_Plan(file_path, shard_root), "1"),
        writer, "open VDS trajectory writer");
    Require_Writer(writer.Define_Particle_Datasets(1, false, false), writer,
                   "define VDS trajectory particle datasets");
    Require_Writer(
        writer.Append_Particle_Frame(10, 0.02, pos0.data(), box0.data()),
        writer, "append first VDS trajectory frame");
    Require_Writer(
        writer.Append_Particle_Frame(20, 0.04, pos1.data(), box1.data()),
        writer, "append second VDS trajectory frame");
    Require_Writer(writer.Finalize(), writer, "finalize VDS trajectory writer");
}

static void Require_Float_Vector_Close(const std::vector<float>& actual,
                                       const std::vector<float>& expected)
{
    REQUIRE_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i)
    {
        REQUIRE_TRUE(std::fabs(actual[i] - expected[i]) < 1.0e-6f);
    }
}

static void Test_Trajectory_Reader_Reads_Metadata_And_Frame()
{
    const auto dir = Unique_Temp_Path("trajectory_reader");
    std::filesystem::create_directories(dir);
    const auto file_path = dir / "prod.spg.h5md";

    Write_Trajectory_File(file_path);

    TrajectoryH5Reader reader;
    Require_Reader(reader.Open(file_path.string()), reader,
                   "open trajectory reader");

    SpongeH5InputMetadata::TrajectoryMetadata metadata;
    Require_Reader(reader.Read_Metadata(&metadata), reader,
                   "read trajectory metadata");
    REQUIRE_EQ(metadata.schema_version, std::string("1"));
    REQUIRE_EQ(metadata.particle_stream, std::string("all"));
    REQUIRE_EQ(metadata.atom_count, static_cast<std::int64_t>(2));
    REQUIRE_EQ(metadata.frame_count, static_cast<std::int64_t>(2));
    REQUIRE_TRUE(metadata.has_position);
    REQUIRE_TRUE(metadata.has_box);
    REQUIRE_TRUE(metadata.has_velocity);

    RestartStructuralState frame;
    Require_Reader(reader.Read_Frame(1, &frame), reader,
                   "read trajectory frame 1");
    REQUIRE_EQ(frame.step, static_cast<std::int64_t>(20));
    REQUIRE_TRUE(std::fabs(frame.time - 0.04) < 1.0e-12);
    REQUIRE_EQ(frame.atom_count, static_cast<std::size_t>(2));
    REQUIRE_TRUE(frame.has_velocity);
    Require_Float_Vector_Close(frame.position_xyz,
                               {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
    Require_Float_Vector_Close(frame.velocity_xyz,
                               {0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f});
    REQUIRE_TRUE(frame.box_edges[0] == 11.0f);
    REQUIRE_TRUE(frame.box_edges[4] == 21.0f);
    REQUIRE_TRUE(frame.box_edges[8] == 31.0f);

    REQUIRE_TRUE(!reader.Read_Frame(2, &frame));
    REQUIRE_TRUE(reader.Last_Error().find("out of range") != std::string::npos);

    reader = TrajectoryH5Reader{};  // Release the file before deleting it on
                                    // Windows.
    std::filesystem::remove_all(dir);
}

static void Test_Trajectory_Reader_Reads_Vds_Wrapper()
{
    const auto dir = Unique_Temp_Path("trajectory_reader_vds");
    std::filesystem::create_directories(dir);
    const auto file_path = dir / "prod.spg.h5md";
    const auto shard_root = dir / "prod.spg.shards";

    Write_Vds_Trajectory_File(file_path, shard_root);

    TrajectoryH5Reader reader;
    Require_Reader(reader.Open(file_path.string()), reader,
                   "open VDS trajectory reader");

    SpongeH5InputMetadata::TrajectoryMetadata metadata;
    Require_Reader(reader.Read_Metadata(&metadata), reader,
                   "read VDS trajectory metadata");
    REQUIRE_EQ(metadata.frame_count, static_cast<std::int64_t>(2));
    REQUIRE_EQ(metadata.atom_count, static_cast<std::int64_t>(1));
    REQUIRE_TRUE(metadata.has_vds_manifest);

    RestartStructuralState frame;
    Require_Reader(reader.Read_Frame(1, &frame), reader,
                   "read VDS trajectory frame 1");
    REQUIRE_EQ(frame.step, static_cast<std::int64_t>(20));
    REQUIRE_TRUE(std::fabs(frame.time - 0.04) < 1.0e-12);
    Require_Float_Vector_Close(frame.position_xyz, {4.0f, 5.0f, 6.0f});
    REQUIRE_TRUE(frame.box_edges[0] == 12.0f);
    REQUIRE_TRUE(frame.box_edges[4] == 22.0f);
    REQUIRE_TRUE(frame.box_edges[8] == 32.0f);

    reader = TrajectoryH5Reader{};  // Release the file before deleting it on
                                    // Windows.
    std::filesystem::remove_all(dir);
}

static void Test_Trajectory_Reader_Refreshes_Swmr_Frames()
{
    const auto dir = Unique_Temp_Path("trajectory_reader_swmr");
    std::filesystem::create_directories(dir);
    const auto file_path = dir / "live.spg.h5md";
    const std::array<float, 9> box = {
        10.0f, 0.0f, 0.0f, 0.0f, 20.0f, 0.0f, 0.0f, 0.0f, 30.0f,
    };
    const std::vector<float> position = {1.0f, 2.0f, 3.0f};

    HighFiveBackend backend;
    TrajectoryH5Writer writer(&backend);
    Require_Writer(
        writer.Open_Single_File(Make_Trajectory_Plan(file_path), "1"), writer,
        "open SWMR trajectory writer");
    Require_Writer(writer.Define_Particle_Datasets(1, false, false), writer,
                   "define SWMR trajectory datasets");
    Require_Writer(writer.Start_Swmr_Write(), writer, "start SWMR writer");

    TrajectoryH5Reader reader;
    Require_Reader(reader.Open_Swmr(file_path.string()), reader,
                   "open SWMR trajectory reader");
    std::int64_t committed = -1;
    Require_Reader(reader.Read_Committed_Frame_Count(&committed), reader,
                   "read initial committed frame count");
    REQUIRE_EQ(committed, static_cast<std::int64_t>(0));

    Require_Writer(
        writer.Append_Particle_Frame(10, 0.02, position.data(), box.data()),
        writer, "publish SWMR trajectory frame");
    Require_Writer(writer.Publish(), writer,
                   "flush SWMR trajectory publication");
    Require_Reader(reader.Refresh(), reader, "refresh SWMR trajectory reader");
    Require_Reader(reader.Read_Committed_Frame_Count(&committed), reader,
                   "read refreshed committed frame count");
    REQUIRE_EQ(committed, static_cast<std::int64_t>(1));

    SpongeH5InputMetadata::TrajectoryMetadata metadata;
    Require_Reader(reader.Read_Metadata(&metadata), reader,
                   "read refreshed SWMR metadata");
    REQUIRE_EQ(metadata.frame_count, static_cast<std::int64_t>(1));
    RestartStructuralState frame;
    Require_Reader(reader.Read_Frame(0, &frame), reader,
                   "read refreshed SWMR frame");
    REQUIRE_EQ(frame.step, static_cast<std::int64_t>(10));
    Require_Float_Vector_Close(frame.position_xyz, position);

    Require_Writer(writer.Finalize(), writer, "finalize SWMR writer");
    Require_Writer(writer.Close(), writer, "close SWMR writer");
    reader = TrajectoryH5Reader{};
    std::filesystem::remove_all(dir);
}

int main()
{
    return Run_Test(
        []
        {
            Test_Trajectory_Reader_Reads_Metadata_And_Frame();
            Test_Trajectory_Reader_Reads_Vds_Wrapper();
            Test_Trajectory_Reader_Refreshes_Swmr_Frames();
        });
}

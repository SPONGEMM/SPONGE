#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

#include "h5_bundle_test_common.hpp"
#include "utils/h5md/highfive_backend.hpp"
#include "utils/h5md/output_plan.hpp"
#include "utils/h5md/trajectory_h5_writer.hpp"
#include "utils/h5md/vds_trajectory_h5_writer.hpp"

namespace
{
using SpongeH5MD::HighFiveBackend;
using SpongeH5MD::HighFiveBackendFactory;
using SpongeH5MD::TrajectoryH5Writer;
using SpongeH5MD::VdsTrajectoryH5Writer;

constexpr const char* kBundleId = "123e4567-e89b-42d3-a456-426614174000";

template <typename Writer>
bool Require(bool ok, const Writer& writer, const char* operation)
{
    if (ok) return true;
    std::cerr << operation << " failed: " << writer.Last_Error() << "\n";
    return false;
}

SpongeH5OutputPlan::ResolvedOutputPlan Make_Plan(
    const std::filesystem::path& prefix, bool vds)
{
    SpongeH5OutputPlan::ResolvedOutputPlan plan;
    plan.trajectory.enabled = true;
    plan.trajectory.path = prefix.string() + ".spg.h5md";
    plan.trajectory.vds = vds;
    if (vds)
    {
        plan.trajectory.chunk_size = 1;
        plan.trajectory.derived_shard_root = prefix.string() + ".spg.shards";
    }
    return plan;
}

template <typename Writer>
bool Define_Streams(Writer& writer)
{
    return Require(writer.Define_Particle_Datasets(1, false, false), writer,
                   "define particles") &&
           Require(writer.Define_Observable_Stream({"temperature"}, {"TEMP"}),
                   writer, "define observables") &&
           Require(writer.Ensure_Metadynamics_Scalars(), writer,
                   "define metadynamics");
}

template <typename Writer>
bool Append_Frame(Writer& writer, int64_t step, double time, float position_x,
                  double temperature)
{
    float position[3] = {position_x, 0.0f, 0.0f};
    float box[9] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    return Require(writer.Append_Particle_Frame(step, time, position, box),
                   writer, "append particles") &&
           Require(writer.Append_Observable_Frame(
                       step, time, {{"temperature", temperature}}),
                   writer, "append observables") &&
           Require(
               writer.Append_Metadynamics_Scalar_Frame(
                   step, time, position_x, position_x + 1.0, position_x + 2.0),
               writer, "append metadynamics");
}

bool Write_Single(const std::filesystem::path& prefix, bool live)
{
    HighFiveBackend backend;
    TrajectoryH5Writer writer(&backend);
    const auto plan = Make_Plan(prefix, false);
    if (!Require(writer.Open_Single_File(plan, SpongeH5MD::kOutputSchemaVersion,
                                         kBundleId, "single"),
                 writer, "open single file") ||
        !Define_Streams(writer) ||
        !Require(writer.Start_Swmr_Write(), writer, "start SWMR") ||
        !Append_Frame(writer, 10, 0.1, 1.0f, 300.0) ||
        !Require(writer.Publish(), writer, "publish first frame"))
    {
        return false;
    }

    if (live)
    {
        const auto ready_path =
            std::filesystem::path(prefix.string() + ".ready");
        const auto continue_path =
            std::filesystem::path(prefix.string() + ".continue");
        std::ofstream(ready_path.string()).close();
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (!std::filesystem::exists(continue_path))
        {
            if (std::chrono::steady_clock::now() >= deadline)
            {
                std::cerr << "timed out waiting for " << continue_path << "\n";
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    return Append_Frame(writer, 20, 0.2, 2.0f, 301.0) &&
           Require(writer.Finalize(), writer, "finalize single file") &&
           Require(writer.Close(), writer, "close single file");
}

bool Write_Vds(const std::filesystem::path& prefix)
{
    HighFiveBackendFactory factory;
    VdsTrajectoryH5Writer writer(&factory);
    const auto plan = Make_Plan(prefix, true);
    return Require(
               writer.Open(plan, SpongeH5MD::kOutputSchemaVersion, kBundleId),
               writer, "open VDS") &&
           Define_Streams(writer) &&
           Append_Frame(writer, 10, 0.1, 1.0f, 300.0) &&
           Append_Frame(writer, 20, 0.2, 2.0f, 301.0) &&
           Require(writer.Finalize(), writer, "finalize VDS");
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "usage: h5_bundle_writer_probe "
                     "<single|single_live|vds> <output-prefix>\n";
        return 2;
    }

    const std::string mode = argv[1];
    const std::filesystem::path prefix = argv[2];
    std::error_code error;
    std::filesystem::create_directories(prefix.parent_path(), error);
    if (error)
    {
        std::cerr << "failed to create output directory: " << error.message()
                  << "\n";
        return 1;
    }

    bool ok = false;
    if (mode == "single")
        ok = Write_Single(prefix, false);
    else if (mode == "single_live")
        ok = Write_Single(prefix, true);
    else if (mode == "vds")
        ok = Write_Vds(prefix);
    else
    {
        std::cerr << "unsupported mode: " << mode << "\n";
        return 2;
    }
    return ok ? 0 : 1;
}

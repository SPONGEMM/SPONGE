#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "../h5_bundle/h5_bundle_test_common.hpp"
#include "../h5_bundle/test_shell.hpp"
#include "utils/float_classification.hpp"
#include "utils/h5md/highfive_backend.hpp"
#include "utils/h5md/trajectory_h5_reader.hpp"
#include "utils/h5md/trajectory_h5_writer.hpp"

namespace
{
// Scaled LJ coefficients make tiny cell changes visible at mdout precision.
// Rerun performs no integration, so this is a numerical fixture, not a fluid.
constexpr double kA = 42515280000.0;
constexpr double kB = 58320000.0;
constexpr double kCutoff = 10.0;
struct Frame
{
    float x, y, z, gamma, second_x;
};

std::string Read_Text(const std::filesystem::path& path)
{
    std::ifstream in(path);
    REQUIRE_TRUE(in.good());
    return {std::istreambuf_iterator<char>(in), {}};
}

void Write_Text(const std::filesystem::path& path, const std::string& text)
{
    std::ofstream out(path);
    out << text;
    REQUIRE_TRUE(out.good());
}

std::array<float, 9> Cell(const Frame& frame)
{
    const double angle = frame.gamma * std::acos(-1.0) / 180.0;
    return {frame.x,
            0,
            0,
            frame.gamma == 90 ? 0.0f : float(frame.y * std::cos(angle)),
            frame.gamma == 90 ? frame.y : float(frame.y * std::sin(angle)),
            0,
            0,
            0,
            frame.z};
}

void Near(double actual, double expected, double tolerance)
{
    if (!SpongeFloat::Is_Finite(actual) ||
        std::fabs(actual - expected) > tolerance)
    {
        std::ostringstream error;
        error << std::setprecision(12) << "actual=" << actual
              << " expected=" << expected << " tolerance=" << tolerance;
        throw TestFailure(error.str());
    }
}

using Columns = std::map<std::string, std::vector<double>>;
Columns Read_Mdout(const std::filesystem::path& path)
{
    std::istringstream input(Read_Text(path));
    std::string line, key;
    std::getline(input, line);
    std::istringstream header(line);
    std::vector<std::string> keys;
    while (header >> key) keys.push_back(key);
    Columns columns;
    while (std::getline(input, line))
    {
        if (line.empty()) continue;
        std::istringstream row(line);
        for (const auto& name : keys)
        {
            double value;
            REQUIRE_TRUE(bool(row >> value));
            columns[name].push_back(value);
        }
    }
    return columns;
}

template <typename T>
void Write_H5_Value(HighFive::File& file, const std::string& path,
                    const T& value)
{
    std::size_t slash = 1;
    while ((slash = path.find('/', slash)) != std::string::npos)
    {
        const auto group = path.substr(0, slash++);
        if (!file.exist(group)) file.createGroup(group);
    }
    file.createDataSet(path, value);
}

void Write_H5_Metadata(const std::filesystem::path& root)
{
    HighFive::File topology((root / "topology.h5").string(),
                            HighFive::File::Overwrite);
    Write_H5_Value(topology, "/schema/name", std::string("sponge.topology.h5"));
    Write_H5_Value(topology, "/schema/version", std::string("sponge.input.v2"));
    Write_H5_Value(topology, "/identity/uuid",
                   std::string("123e4567-e89b-42d3-a456-426614174000"));
    Write_H5_Value(topology, "/topology/atom_count", std::int64_t(2));
    Write_H5_Value(topology, "/topology/atom_order_hash",
                   std::string("sha256:") + std::string(64, 'b'));
    Write_H5_Value(topology, "/topology/topology_hash",
                   std::string("sha256:") + std::string(64, 'a'));
    Write_H5_Value(topology, "/topology/forcefield_hash",
                   std::string("sha256:") + std::string(64, 'c'));
    Write_H5_Value(topology, SpongeH5MD::path::legacy_sidecar_keys,
                   std::vector<std::string>{"mass_in_file", "charge_in_file",
                                            "LJ_in_file"});
    Write_H5_Value(topology, SpongeH5MD::path::legacy_sidecar_paths,
                   std::vector<std::string>{(root / "mass.txt").string(),
                                            (root / "charge.txt").string(),
                                            (root / "lj.txt").string()});
    HighFive::File protocol((root / "protocol.h5").string(),
                            HighFive::File::Overwrite);
    Write_H5_Value(protocol, "/schema/name", std::string("sponge.protocol.h5"));
    Write_H5_Value(protocol, "/schema/version", std::string("sponge.input.v2"));
    Write_H5_Value(protocol, "/identity/uuid",
                   std::string("123e4567-e89b-42d3-a456-426614174001"));
    Write_H5_Value(protocol, "/protocol/topology_compatibility/topology_hash",
                   std::string("sha256:") + std::string(64, 'a'));
    Write_H5_Value(protocol, "/identity/content_hash",
                   std::string("sha256:") + std::string(64, 'd'));
}

void Run_Case(const std::filesystem::path& executable,
              const std::filesystem::path& root,
              const std::vector<Frame>& frames, bool h5,
              const std::string& compatibility = "", bool reject = false,
              int start = 0, int strip = 0, int limit = 0)
{
    std::filesystem::create_directories(root);
    Write_Text(root / "mass.txt", "2\n12\n12\n");
    Write_Text(root / "charge.txt", "2\n0\n0\n");
    Write_Text(root / "coordinate.txt",
               "2\n1 5 5\n37 5 5\n40 40 40 90 90 90\n");
    std::ostringstream lj;
    lj << std::setprecision(17) << "2 1\n" << kA << "\n" << kB << "\n0\n0\n";
    Write_Text(root / "lj.txt", lj.str());
    SpongeH5OutputPlan::ResolvedOutputPlan plan;
    plan.trajectory.enabled = true;
    plan.trajectory.path = (root / "input.spg.h5md").generic_string();
    SpongeH5MD::HighFiveBackend backend;
    SpongeH5MD::TrajectoryH5Writer writer(&backend);
    REQUIRE_TRUE(writer.Open_Single_File(plan));
    REQUIRE_TRUE(writer.Define_Particle_Datasets(2, false, false));
    std::ofstream crd(root / "input.crd", std::ios::binary);
    std::ofstream boxes(root / "input.box");
    boxes << std::setprecision(9);
    for (std::size_t i = 0; i < frames.size(); ++i)
    {
        const auto& frame = frames[i];
        const std::array<float, 6> positions = {1, 5, 5, frame.second_x, 5, 5};
        const auto cell = Cell(frame);
        REQUIRE_TRUE(writer.Append_Particle_Frame(
            i, 5.0 + i * 0.2, positions.data(), cell.data()));
        crd.write(reinterpret_cast<const char*>(positions.data()),
                  sizeof(positions));
        boxes << frame.x << " " << frame.y << " " << frame.z << " 90 90 "
              << frame.gamma << "\n";
    }
    REQUIRE_TRUE(writer.Finalize());
    REQUIRE_TRUE(writer.Close());
    crd.close();
    boxes.close();
    std::ostringstream mdin;
    mdin << "mode = \"rerun\"\ncutoff = 10\nPM.MPI_size = 0\n"
         << "mass_in_file = \"" << (root / "mass.txt").generic_string()
         << "\"\n"
         << "charge_in_file = \"" << (root / "charge.txt").generic_string()
         << "\"\n"
         << "coordinate_in_file = \""
         << (root / "coordinate.txt").generic_string() << "\"\n"
         << "LJ_in_file = \"" << (root / "lj.txt").generic_string() << "\"\n"
         << "mdout = \"" << (root / "mdout.txt").generic_string() << "\"\n"
         << "mdinfo = \"" << (root / "mdinfo.txt").generic_string() << "\"\n"
         << "frc = \"" << (root / "force.dat").generic_string() << "\"\n"
         << "rerun_start = " << start << "\nrerun_strip = " << strip << "\n"
         << "write_trajectory_interval = 1\nwrite_restart_file_interval = 0\n"
         << "output_h5_trajectory_path = \""
         << (root / "output.spg.h5md").generic_string() << "\"\n";
    if (h5)
    {
        Write_H5_Metadata(root);
        mdin << "input_h5_trajectory_path = \"" << plan.trajectory.path
             << "\"\n"
             << "input_h5_topology_path = \""
             << (root / "topology.h5").generic_string() << "\"\n"
             << "input_h5_protocol_path = \""
             << (root / "protocol.h5").generic_string() << "\"\n";
    }
    else
        mdin << "crd = \"" << (root / "input.crd").generic_string() << "\"\n"
             << "box = \"" << (root / "input.box").generic_string() << "\"\n";
    if (!compatibility.empty())
        mdin << "rerun_need_box_update = " << compatibility << "\n";
    if (limit) mdin << "frame_limit = " << limit << "\n";
    Write_Text(root / "mdin.toml", mdin.str());
    const auto log = root / "stdout.txt";
    const int ret = Run_Test_Shell_Command(Shell_Quote(executable) + " -mdin " +
                                           Shell_Quote(root / "mdin.toml") +
                                           " > " + Shell_Quote(log) + " 2>&1");
    const auto output = Read_Text(log);
    if (reject)
    {
        REQUIRE_TRUE(ret != 0);
        REQUIRE_TRUE(output.find("only 1 (or true) is accepted") !=
                     std::string::npos);
        return;
    }
    if (ret != 0) throw TestFailure(output);
    REQUIRE_TRUE((output.find("Warning: rerun_need_box_update is deprecated") !=
                  std::string::npos) == !compatibility.empty());
    const auto values = Read_Mdout(root / "mdout.txt");
    SpongeH5MD::TrajectoryH5Reader reader;
    REQUIRE_TRUE(reader.Open((root / "output.spg.h5md").string()));
    std::ifstream forces(root / "force.dat", std::ios::binary);
    REQUIRE_TRUE(forces.good());
    std::size_t count = 0;
    for (std::size_t i = start; i < frames.size(); i += strip + 1)
    {
        // frame_limit bounds the input frame index, including skipped frames.
        if (limit && i >= static_cast<std::size_t>(limit)) break;
        const auto& frame = frames[i];
        SpongeH5MD::RestartStructuralState actual;
        REQUIRE_TRUE(reader.Read_Frame(count, &actual));
        const auto expected_cell = Cell(frame);
        for (int j = 0; j < 9; ++j)
            Near(actual.box_edges[j], expected_cell[j], 2e-5);
        if (h5) Near(actual.time, 5.0 + i * 0.2, 1e-12);
        const std::array<float, 6> expected_position = {1, 5, 5, frame.second_x,
                                                        5, 5};
        for (int j = 0; j < 6; ++j)
            Near(actual.position_xyz.at(j), expected_position[j], 2e-5);
        const double distance = frame.x - (frame.second_x - 1.0);
        const double r6 = std::pow(distance, 6);
        const double energy = distance < kCutoff ? kA / (r6 * r6) - kB / r6 : 0;
        // mdout rounds to 0.01 kcal/mol; allow the float force-field
        // arithmetic.
        Near(values.at("LJ_short").at(count), energy,
             0.006 + std::fabs(energy) * 2e-6);
        // This independently checks the effective volume used by the force
        // code.
        const double volume =
            expected_cell[0] * double(expected_cell[4]) * expected_cell[8];
        const double tail = -4.0 * 2.0 / 3.0 * std::acos(-1.0) * kB /
                            std::pow(kCutoff, 3) / volume;
        Near(values.at("LJ_long").at(count), tail, 0.006);
        const double force =
            distance < kCutoff ? (12 * kA / (r6 * r6) - 6 * kB / r6) / distance
                               : 0;
        std::array<float, 6> force_frame;
        forces.read(reinterpret_cast<char*>(force_frame.data()),
                    sizeof(force_frame));
        REQUIRE_EQ(forces.gcount(),
                   static_cast<std::streamsize>(sizeof(force_frame)));
        Near(force_frame[0], force, 0.005 + std::fabs(force) * 3e-5);
        Near(force_frame[3], -force, 0.005 + std::fabs(force) * 3e-5);
        for (int j : {1, 2, 4, 5}) Near(force_frame[j], 0.0, 0.01);
        ++count;
    }
    REQUIRE_EQ(values.at("LJ_short").size(), count);
    SpongeH5InputMetadata::TrajectoryMetadata metadata;
    REQUIRE_TRUE(reader.Read_Metadata(&metadata));
    REQUIRE_EQ(metadata.frame_count, static_cast<std::int64_t>(count));
}
}  // namespace

int main(int argc, char** argv)
{
    const char* enabled = std::getenv("SPONGE_H5_ENABLE_RUNTIME_SMOKE");
    if (!enabled || std::string(enabled) != "1") return 77;
    if (argc != 2) return 2;
    const auto root = SpongeH5Test::Unique_Temp_Path("rerun_cells");
    try
    {
        const std::vector<Frame> varied = {
            {40, 40, 40, 90, 37}, {40.00003f, 40, 40, 90, 37},
            {48, 48, 48, 90, 43}, {48, 48, 48, 90, 43},
            {40, 40, 40, 90, 37}, {40, 42, 44, 80, 38}};
        const std::vector<Frame> fixed(3, varied.front());
        for (bool h5 : {false, true})
        {
            const auto prefix = h5 ? "h5_" : "legacy_";
            Run_Case(argv[1], root / (std::string(prefix) + "varied"), varied,
                     h5);
            Run_Case(argv[1], root / (std::string(prefix) + "fixed"), fixed,
                     h5);
            Run_Case(argv[1], root / (std::string(prefix) + "compat"), varied,
                     h5, "1");
            Run_Case(argv[1], root / (std::string(prefix) + "true"), varied, h5,
                     "true");
            Run_Case(argv[1], root / (std::string(prefix) + "selected"), varied,
                     h5, "", false, 1, 1);
            Run_Case(argv[1], root / (std::string(prefix) + "limit"), varied,
                     h5, "", false, 0, 0, 2);
            int invalid = 0;
            for (const auto& value :
                 {"0", "false", "2", "-1", "0.5", "\"invalid\""})
                Run_Case(argv[1],
                         root / (std::string(prefix) + "invalid_" +
                                 std::to_string(invalid++)),
                         varied, h5, value, true);
        }
        std::filesystem::remove_all(root);
    }
    catch (const std::exception& error)
    {
        std::cerr << "rerun cell regression: " << error.what()
                  << "\nevidence: " << root << '\n';
        return 1;
    }
    return 0;
}

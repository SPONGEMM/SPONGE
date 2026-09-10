#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <highfive/highfive.hpp>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "h5_input_matrix_fixture.hpp"
#include "utils/h5md/h5_legacy_sidecar.hpp"
#include "utils/h5md/rerun_frame_selection.hpp"
#include "utils/h5md/trajectory_h5_reader.hpp"

namespace
{
constexpr float kTolerance = 1.0e-5f;

enum class Mode
{
    normal,
    rerun,
};

enum class InputFamily
{
    legacy,
    bundled,
    bundled_with_sidecar,
};

enum class OutputFamily
{
    legacy,
    bundled,
};

enum class EvidenceType
{
    static_contract,
    prepared_mdin_guard,
    runtime_smoke,
    semantic_fixture_comparison,
};

struct MatrixCase
{
    const char* name;
    Mode mode;
    InputFamily input;
    OutputFamily output;
    bool vds_applicable;
    bool vds;
    EvidenceType evidence;
};

std::string Read_Text_File(const std::filesystem::path& path);

const char* Mode_Name(const Mode mode)
{
    return mode == Mode::normal ? "normal" : "rerun";
}

const char* Input_Name(const InputFamily input)
{
    if (input == InputFamily::legacy)
    {
        return "legacy";
    }
    if (input == InputFamily::bundled)
    {
        return "bundled";
    }
    return "bundled_with_sidecar";
}

const char* Output_Name(const OutputFamily output)
{
    return output == OutputFamily::legacy ? "legacy" : "bundled";
}

std::string Combination_Key(const MatrixCase& spec)
{
    return std::string(Mode_Name(spec.mode)) + "|" + Input_Name(spec.input) +
           "|" + Output_Name(spec.output);
}

std::vector<SpongeH5MD::LegacySidecarBinding> Read_Sidecars(
    const std::filesystem::path& path)
{
    std::vector<SpongeH5MD::LegacySidecarBinding> sidecars;
    std::string error;
    REQUIRE_TRUE(SpongeH5MD::Read_Legacy_Sidecars_From_H5(path.string(),
                                                          &sidecars, &error));
    return sidecars;
}

std::string Path_For_Key(
    const std::vector<SpongeH5MD::LegacySidecarBinding>& sidecars,
    const std::string& key)
{
    for (const auto& sidecar : sidecars)
    {
        if (sidecar.key == key)
        {
            return sidecar.path;
        }
    }
    throw TestFailure("missing sidecar key: " + key);
}

std::set<std::string> Sidecar_Key_Set(const std::filesystem::path& path)
{
    const auto sidecars = Read_Sidecars(path);
    std::set<std::string> keys;
    for (const auto& sidecar : sidecars)
    {
        REQUIRE_TRUE(keys.insert(sidecar.key).second);
    }
    return keys;
}

std::string Relative_Sidecar_File(const std::filesystem::path& sidecar_root,
                                  const std::filesystem::path& path)
{
    const auto relative = std::filesystem::relative(path, sidecar_root);
    return relative.generic_string();
}

std::set<std::string> Materialized_Sidecar_File_Set(
    const std::filesystem::path& bundle_root)
{
    const auto sidecar_root = bundle_root / "legacy_sidecars";
    std::set<std::string> files;
    REQUIRE_TRUE(std::filesystem::is_directory(sidecar_root));
    for (const auto& entry :
         std::filesystem::recursive_directory_iterator(sidecar_root))
    {
        if (entry.is_regular_file())
        {
            REQUIRE_TRUE(
                files.insert(Relative_Sidecar_File(sidecar_root, entry.path()))
                    .second);
        }
    }
    REQUIRE_TRUE(!files.empty());
    return files;
}

void Insert_H5_Sidecar_References(std::set<std::string>* references,
                                  const std::filesystem::path& bundle_root,
                                  const std::filesystem::path& h5_file)
{
    REQUIRE_TRUE(references != nullptr);
    const auto sidecar_root = bundle_root / "legacy_sidecars";
    for (const auto& sidecar : Read_Sidecars(h5_file))
    {
        const std::filesystem::path sidecar_path(sidecar.path);
        REQUIRE_TRUE(sidecar_path.is_absolute());
        REQUIRE_TRUE(std::filesystem::exists(sidecar_path));
        references->insert(Relative_Sidecar_File(sidecar_root, sidecar_path));
    }
}

void Insert_Mdin_Sidecar_References(std::set<std::string>* references,
                                    const std::string& mdin)
{
    REQUIRE_TRUE(references != nullptr);
    const std::string prefix = "\"legacy_sidecars/";
    std::size_t pos = 0;
    while ((pos = mdin.find(prefix, pos)) != std::string::npos)
    {
        const std::size_t start = pos + 1;
        const std::size_t end = mdin.find('"', start);
        REQUIRE_TRUE(end != std::string::npos);
        references->insert(
            mdin.substr(start + std::string("legacy_sidecars/").size(),
                        end - start - std::string("legacy_sidecars/").size()));
        pos = end + 1;
    }
}

std::set<std::string> Referenced_Sidecar_File_Set(
    const std::filesystem::path& bundle_root)
{
    std::set<std::string> references;
    Insert_H5_Sidecar_References(&references, bundle_root,
                                 bundle_root / "topology.spgt.h5");
    Insert_H5_Sidecar_References(&references, bundle_root,
                                 bundle_root / "protocol.spgp.h5");
    Insert_H5_Sidecar_References(&references, bundle_root,
                                 bundle_root / "restart.spgr.h5");
    Insert_Mdin_Sidecar_References(
        &references, Read_Text_File(bundle_root / "mdin.bundled.spg.toml"));
    return references;
}

void Require_Float_Close(const float actual, const float expected,
                         const std::string& label)
{
    if (std::fabs(actual - expected) > kTolerance)
    {
        throw TestFailure(label + " mismatch");
    }
}

std::vector<float> Read_Float_Flat_Dataset(HighFive::DataSet dataset)
{
    const auto dims = dataset.getSpace().getDimensions();
    std::size_t value_count = 1;
    for (const auto dim : dims)
    {
        value_count *= dim;
    }

    std::vector<float> values(value_count);
    hsize_t mem_dims[1] = {static_cast<hsize_t>(value_count)};
    hid_t mem_space = H5Screate_simple(1, mem_dims, nullptr);
    REQUIRE_TRUE(mem_space >= 0);
    const herr_t rc = H5Dread(dataset.getId(), H5T_NATIVE_FLOAT, mem_space,
                              H5S_ALL, H5P_DEFAULT, values.data());
    H5Sclose(mem_space);
    REQUIRE_TRUE(rc >= 0);
    return values;
}

void Require_H5_Has_Legacy_Sidecars(const std::filesystem::path& path)
{
    const auto sidecars = Read_Sidecars(path);
    REQUIRE_TRUE(!sidecars.empty());
}

void Require_H5_Sidecar_Paths_Are_Materialized(
    const std::filesystem::path& path)
{
    const auto sidecars = Read_Sidecars(path);
    REQUIRE_TRUE(!sidecars.empty());
    for (const auto& sidecar : sidecars)
    {
        const std::filesystem::path sidecar_path(sidecar.path);
        REQUIRE_TRUE(sidecar_path.is_absolute());
        REQUIRE_TRUE(sidecar_path.string().find("legacy_sidecars") !=
                     std::string::npos);
        SpongeH5InputMatrix::Require_Path_Exists(sidecar_path);
    }
}

void Require_H5_Has_No_Legacy_Sidecars(const std::filesystem::path& path)
{
    const auto sidecars = Read_Sidecars(path);
    REQUIRE_TRUE(sidecars.empty());
}

std::filesystem::path Source_Dir_For(const Mode mode, const InputFamily input)
{
    if (mode == Mode::normal)
    {
        const auto core = SpongeH5InputMatrix::Core_Structural_Path();
        if (input == InputFamily::legacy)
        {
            return core / "legacy_input";
        }
        if (input == InputFamily::bundled)
        {
            return core / "bundled_input" / "bundle";
        }
        return core / "bundled_input_with_legacy_sidecar" / "bundle";
    }

    const auto full = SpongeH5InputMatrix::Full_Contract_Rerun_Path();
    if (input == InputFamily::legacy)
    {
        return full / "legacy_input";
    }
    if (input == InputFamily::bundled)
    {
        return full / "bundled_input" / "bundle";
    }
    return full / "bundled_input_with_legacy_sidecar" / "bundle";
}

std::filesystem::path Mdin_For(const Mode mode, const InputFamily input)
{
    const auto source = Source_Dir_For(mode, input);
    if (input == InputFamily::legacy)
    {
        return source / "mdin.spg.toml";
    }
    return source / "mdin.bundled.spg.toml";
}

std::string Read_Text_File(const std::filesystem::path& path)
{
    std::ifstream in(path);
    REQUIRE_TRUE(in.good());
    return std::string(std::istreambuf_iterator<char>(in),
                       std::istreambuf_iterator<char>());
}

bool Has_Key_Line(const std::string& mdin, const std::string& key)
{
    std::istringstream stream(mdin);
    std::string line;
    while (std::getline(stream, line))
    {
        const auto stripped = line.substr(0, line.find('#'));
        std::size_t pos = 0;
        while (pos < stripped.size() &&
               std::isspace(static_cast<unsigned char>(stripped[pos])))
        {
            ++pos;
        }
        if (stripped.compare(pos, key.size(), key) != 0)
        {
            continue;
        }
        pos += key.size();
        while (pos < stripped.size() &&
               std::isspace(static_cast<unsigned char>(stripped[pos])))
        {
            ++pos;
        }
        if (pos < stripped.size() && stripped[pos] == '=')
        {
            return true;
        }
    }
    return false;
}

void Require_Key_Value_Line(const std::string& mdin, const std::string& key,
                            const std::string& value)
{
    std::istringstream stream(mdin);
    std::string line;
    const std::string expected = key + " = " + value;
    while (std::getline(stream, line))
    {
        const auto stripped = line.substr(0, line.find('#'));
        std::size_t begin = 0;
        while (begin < stripped.size() &&
               std::isspace(static_cast<unsigned char>(stripped[begin])))
        {
            ++begin;
        }
        std::size_t end = stripped.size();
        while (end > begin &&
               std::isspace(static_cast<unsigned char>(stripped[end - 1])))
        {
            --end;
        }
        if (stripped.substr(begin, end - begin) == expected)
        {
            return;
        }
    }
    throw TestFailure("missing mdin key/value: " + expected);
}

bool Path_Tree_Contains(const std::filesystem::path& root,
                        const std::string& needle)
{
    REQUIRE_TRUE(std::filesystem::exists(root));
    for (const auto& entry :
         std::filesystem::recursive_directory_iterator(root))
    {
        if (entry.path().string().find(needle) != std::string::npos)
        {
            return true;
        }
    }
    return false;
}

void Require_Input_Fixture_Exists(const Mode mode, const InputFamily input)
{
    const auto source = Source_Dir_For(mode, input);
    SpongeH5InputMatrix::Require_Path_Exists(source);
    SpongeH5InputMatrix::Require_Path_Exists(Mdin_For(mode, input));

    if (input != InputFamily::legacy)
    {
        SpongeH5InputMatrix::Require_Path_Exists(source / "topology.spgt.h5");
        SpongeH5InputMatrix::Require_Path_Exists(source / "protocol.spgp.h5");
        SpongeH5InputMatrix::Require_Path_Exists(source / "restart.spgr.h5");
    }
    if (mode == Mode::rerun && input != InputFamily::legacy)
    {
        SpongeH5InputMatrix::Require_Path_Exists(source /
                                                 "trajectory.spg.h5md");
    }
}

void Require_Rerun_Trajectory_Frames(const std::filesystem::path& path,
                                     const std::size_t expected_frames)
{
    HighFive::File file(path.string(), HighFive::File::ReadOnly);
    std::vector<std::int64_t> steps;
    std::vector<double> times;
    std::vector<float> positions;
    std::vector<float> boxes;
    file.getDataSet("/particles/all/step").read(steps);
    file.getDataSet("/particles/all/time").read(times);
    positions = Read_Float_Flat_Dataset(
        file.getDataSet("/particles/all/position/value"));
    boxes = Read_Float_Flat_Dataset(
        file.getDataSet("/particles/all/box/edges/value"));

    REQUIRE_EQ(steps.size(), expected_frames);
    REQUIRE_EQ(times.size(), expected_frames);
    REQUIRE_EQ(positions.size(), expected_frames * 2 * 3);
    REQUIRE_EQ(boxes.size(), expected_frames * 3 * 3);
    for (std::size_t i = 0; i < expected_frames; ++i)
    {
        REQUIRE_EQ(steps[i], static_cast<std::int64_t>(i));
        REQUIRE_TRUE(std::fabs(times[i] - static_cast<double>(i)) < 1.0e-12);
    }

    const std::vector<float> first_position = {1.0f, 2.0f, 3.0f,
                                               4.0f, 5.0f, 6.0f};
    for (std::size_t i = 0; i < first_position.size(); ++i)
    {
        Require_Float_Close(positions[i], first_position[i],
                            "rerun trajectory first position");
    }
    const std::vector<float> first_box = {10.0f, 0.0f, 0.0f, 0.0f, 20.0f,
                                          0.0f,  0.0f, 0.0f, 30.0f};
    for (std::size_t i = 0; i < first_box.size(); ++i)
    {
        Require_Float_Close(boxes[i], first_box[i],
                            "rerun trajectory first box");
    }
}

void Require_Rerun_Selection_Reads_Expected_Frame(
    const std::filesystem::path& path, const int rerun_start,
    const int rerun_strip, const int rerun_frame_limit)
{
    SpongeH5MD::TrajectoryH5Reader reader;
    REQUIRE_TRUE(reader.Open(path.string()));

    SpongeH5InputMetadata::TrajectoryMetadata metadata;
    REQUIRE_TRUE(reader.Read_Metadata(&metadata));
    REQUIRE_EQ(metadata.frame_count, static_cast<std::int64_t>(2));
    REQUIRE_EQ(metadata.atom_count, static_cast<std::int64_t>(2));

    int next_frame_index = rerun_start;
    int selected_frames = 0;
    while (selected_frames < rerun_frame_limit)
    {
        const auto selection = SpongeH5MD::Select_Next_H5_Rerun_Frame(
            next_frame_index, rerun_strip,
            static_cast<int>(metadata.frame_count));
        REQUIRE_TRUE(selection.has_frame);
        REQUIRE_EQ(selection.frame_index, 1);
        REQUIRE_EQ(selection.skipped_frames, rerun_strip);

        SpongeH5MD::RestartStructuralState frame;
        REQUIRE_TRUE(reader.Read_Frame(
            static_cast<std::size_t>(selection.frame_index), &frame));
        REQUIRE_EQ(frame.step, static_cast<std::int64_t>(1));
        REQUIRE_TRUE(std::fabs(frame.time - 1.0) < 1.0e-12);
        REQUIRE_EQ(frame.atom_count, static_cast<std::size_t>(2));
        REQUIRE_EQ(frame.position_xyz.size(), static_cast<std::size_t>(6));
        Require_Float_Close(frame.position_xyz[0], 1.5f,
                            "selected rerun frame position");
        Require_Float_Close(frame.position_xyz[5], 6.5f,
                            "selected rerun frame position");
        Require_Float_Close(frame.box_edges[0], 11.0f,
                            "selected rerun frame box");

        next_frame_index = selection.next_frame_index;
        ++selected_frames;
    }

    REQUIRE_EQ(selected_frames, rerun_frame_limit);
    const auto exhausted = SpongeH5MD::Select_Next_H5_Rerun_Frame(
        next_frame_index, rerun_strip, static_cast<int>(metadata.frame_count));
    REQUIRE_TRUE(!exhausted.has_frame);
}

std::vector<MatrixCase> Legal_Matrix_Cases()
{
    return {
        {"normal_legacy_in_legacy_out", Mode::normal, InputFamily::legacy,
         OutputFamily::legacy, false, false, EvidenceType::runtime_smoke},
        {"normal_legacy_in_bundled_out", Mode::normal, InputFamily::legacy,
         OutputFamily::bundled, false, false, EvidenceType::runtime_smoke},
        {"normal_bundled_in_legacy_out", Mode::normal, InputFamily::bundled,
         OutputFamily::legacy, false, false, EvidenceType::runtime_smoke},
        {"normal_sidecar_in_legacy_out", Mode::normal,
         InputFamily::bundled_with_sidecar, OutputFamily::legacy, false, false,
         EvidenceType::runtime_smoke},
        {"normal_bundled_in_bundled_out", Mode::normal, InputFamily::bundled,
         OutputFamily::bundled, false, false, EvidenceType::runtime_smoke},
        {"normal_sidecar_in_bundled_out", Mode::normal,
         InputFamily::bundled_with_sidecar, OutputFamily::bundled, false, false,
         EvidenceType::runtime_smoke},
        {"rerun_legacy_in_legacy_out", Mode::rerun, InputFamily::legacy,
         OutputFamily::legacy, false, false, EvidenceType::runtime_smoke},
        {"rerun_legacy_in_bundled_out_vds_off", Mode::rerun,
         InputFamily::legacy, OutputFamily::bundled, true, false,
         EvidenceType::runtime_smoke},
        {"rerun_legacy_in_bundled_out_vds_on", Mode::rerun, InputFamily::legacy,
         OutputFamily::bundled, true, true, EvidenceType::runtime_smoke},
        {"rerun_bundled_in_legacy_out", Mode::rerun, InputFamily::bundled,
         OutputFamily::legacy, false, false, EvidenceType::runtime_smoke},
        {"rerun_sidecar_in_legacy_out", Mode::rerun,
         InputFamily::bundled_with_sidecar, OutputFamily::legacy, false, false,
         EvidenceType::runtime_smoke},
        {"rerun_bundled_in_bundled_out_vds_off", Mode::rerun,
         InputFamily::bundled, OutputFamily::bundled, true, false,
         EvidenceType::runtime_smoke},
        {"rerun_bundled_in_bundled_out_vds_on", Mode::rerun,
         InputFamily::bundled, OutputFamily::bundled, true, true,
         EvidenceType::runtime_smoke},
        {"rerun_sidecar_in_bundled_out_vds_off", Mode::rerun,
         InputFamily::bundled_with_sidecar, OutputFamily::bundled, true, false,
         EvidenceType::runtime_smoke},
        {"rerun_sidecar_in_bundled_out_vds_on", Mode::rerun,
         InputFamily::bundled_with_sidecar, OutputFamily::bundled, true, true,
         EvidenceType::runtime_smoke},
    };
}

void Test_Legal_Matrix_Cases_Are_Explicit_And_Fixture_Backed()
{
    const auto cases = Legal_Matrix_Cases();
    REQUIRE_EQ(cases.size(), static_cast<std::size_t>(15));

    const std::set<std::string> expected_names = {
        "normal_legacy_in_legacy_out",
        "normal_legacy_in_bundled_out",
        "normal_bundled_in_legacy_out",
        "normal_bundled_in_bundled_out",
        "normal_sidecar_in_legacy_out",
        "normal_sidecar_in_bundled_out",
        "rerun_legacy_in_legacy_out",
        "rerun_bundled_in_legacy_out",
        "rerun_sidecar_in_legacy_out",
        "rerun_legacy_in_bundled_out_vds_off",
        "rerun_legacy_in_bundled_out_vds_on",
        "rerun_bundled_in_bundled_out_vds_off",
        "rerun_bundled_in_bundled_out_vds_on",
        "rerun_sidecar_in_bundled_out_vds_off",
        "rerun_sidecar_in_bundled_out_vds_on"};
    std::set<std::string> actual_names;
    std::map<std::string, int> combination_counts;
    int normal_cases = 0;
    int rerun_cases = 0;
    int vds_applicable_cases = 0;
    int vds_on_cases = 0;
    int vds_off_cases = 0;
    int runtime_smoke_evidence_cases = 0;

    for (const auto& spec : cases)
    {
        const std::string name(spec.name);
        REQUIRE_TRUE(name.find(" ") == std::string::npos);
        REQUIRE_TRUE(actual_names.insert(name).second);
        ++combination_counts[Combination_Key(spec)];
        if (spec.mode == Mode::normal)
        {
            ++normal_cases;
        }
        else
        {
            ++rerun_cases;
        }
        if (spec.vds_applicable)
        {
            ++vds_applicable_cases;
            if (spec.vds)
            {
                ++vds_on_cases;
            }
            else
            {
                ++vds_off_cases;
            }
        }
        Require_Input_Fixture_Exists(spec.mode, spec.input);

        if (spec.output == OutputFamily::legacy)
        {
            REQUIRE_TRUE(!spec.vds_applicable);
            REQUIRE_TRUE(!spec.vds);
        }
        if (spec.vds_applicable)
        {
            REQUIRE_TRUE(spec.mode == Mode::rerun);
            REQUIRE_TRUE(spec.output == OutputFamily::bundled);
        }
        if (spec.mode == Mode::normal && spec.output == OutputFamily::bundled)
        {
            REQUIRE_TRUE(!spec.vds_applicable);
            REQUIRE_TRUE(!spec.vds);
        }
        if (spec.mode == Mode::rerun && spec.output == OutputFamily::bundled)
        {
            REQUIRE_TRUE(spec.vds_applicable);
        }
        if (spec.evidence == EvidenceType::runtime_smoke)
        {
            ++runtime_smoke_evidence_cases;
        }
    }

    REQUIRE_TRUE(actual_names == expected_names);
    REQUIRE_EQ(normal_cases, 6);
    REQUIRE_EQ(rerun_cases, 9);
    REQUIRE_EQ(vds_applicable_cases, 6);
    REQUIRE_EQ(vds_on_cases, 3);
    REQUIRE_EQ(vds_off_cases, 3);
    REQUIRE_EQ(runtime_smoke_evidence_cases, 15);

    const std::map<std::string, int> expected_combination_counts = {
        {"normal|legacy|legacy", 1},
        {"normal|legacy|bundled", 1},
        {"normal|bundled|legacy", 1},
        {"normal|bundled|bundled", 1},
        {"normal|bundled_with_sidecar|legacy", 1},
        {"normal|bundled_with_sidecar|bundled", 1},
        {"rerun|legacy|legacy", 1},
        {"rerun|legacy|bundled", 2},
        {"rerun|bundled|legacy", 1},
        {"rerun|bundled|bundled", 2},
        {"rerun|bundled_with_sidecar|legacy", 1},
        {"rerun|bundled_with_sidecar|bundled", 2}};
    REQUIRE_TRUE(combination_counts == expected_combination_counts);
}

void Test_Fixture_Helper_Copies_Case_To_Temp()
{
    const auto source = SpongeH5InputMatrix::Core_Structural_Path() /
                        "bundled_input_with_legacy_sidecar" / "bundle";
    const auto copied = SpongeH5InputMatrix::Copy_Case_To_Temp(
        source, "h5_input_matrix_fixture_copy");

    REQUIRE_TRUE(copied != source);
    SpongeH5InputMatrix::Require_Path_Exists(copied / "mdin.bundled.spg.toml");
    SpongeH5InputMatrix::Require_Path_Exists(copied / "topology.spgt.h5");
    SpongeH5InputMatrix::Require_Path_Exists(copied / "legacy_sidecars" /
                                             "mass_in_file" / "mass.txt");

    const auto sidecars = Read_Sidecars(copied / "topology.spgt.h5");
    REQUIRE_EQ(Path_For_Key(sidecars, "mass_in_file"),
               std::filesystem::absolute(copied / "legacy_sidecars" /
                                         "mass_in_file" / "mass.txt")
                   .lexically_normal()
                   .string());

    std::filesystem::remove_all(copied);
}

void Test_Fixture_Helper_Describes_Case_Paths()
{
    const auto source = SpongeH5InputMatrix::Full_Contract_Rerun_Path() /
                        "bundled_input" / "bundle";
    const auto source_paths =
        SpongeH5InputMatrix::Describe_Case(source, "mdin.bundled.spg.toml");

    REQUIRE_TRUE(source_paths.root == source);
    REQUIRE_TRUE(source_paths.mdin == source / "mdin.bundled.spg.toml");
    SpongeH5InputMatrix::Require_Path_Exists(source_paths.mdin);
    SpongeH5InputMatrix::Require_Path_Exists(source_paths.topology_h5);
    SpongeH5InputMatrix::Require_Path_Exists(source_paths.protocol_h5);
    SpongeH5InputMatrix::Require_Path_Exists(source_paths.restart_h5);
    SpongeH5InputMatrix::Require_Path_Exists(source_paths.trajectory_h5);
    REQUIRE_TRUE(source_paths.normal_output.mdout == source / "mdout.txt");
    REQUIRE_TRUE(source_paths.normal_output.mdinfo == source / "mdinfo.txt");
    REQUIRE_TRUE(source_paths.normal_output.h5_restart ==
                 source / "output" / "restart_out.spgr.h5");
    REQUIRE_TRUE(source_paths.normal_output.h5_trajectory ==
                 source / "output" / "traj_out.spg.h5md");
    REQUIRE_TRUE(source_paths.normal_output.h5_observable ==
                 source / "output" / "observable_out.obs.spg.h5md");
    REQUIRE_TRUE(source_paths.rerun_output.mdout == source / "mdout.txt");
    REQUIRE_TRUE(source_paths.rerun_output.mdinfo == source / "mdinfo.txt");
    REQUIRE_TRUE(source_paths.rerun_output.h5_restart.empty());
    REQUIRE_TRUE(source_paths.rerun_output.h5_trajectory ==
                 source / "output" / "rerun_traj_out.spg.h5md");
    REQUIRE_TRUE(source_paths.rerun_output.h5_observable ==
                 source / "output" / "rerun_observable_out.obs.spg.h5md");

    const auto copied_paths = SpongeH5InputMatrix::Copy_Case_To_Temp(
        source, "h5_input_matrix_fixture_paths", "mdin.bundled.spg.toml");
    REQUIRE_TRUE(copied_paths.root != source);
    REQUIRE_TRUE(copied_paths.mdin ==
                 copied_paths.root / "mdin.bundled.spg.toml");
    SpongeH5InputMatrix::Require_Path_Exists(copied_paths.mdin);
    SpongeH5InputMatrix::Require_Path_Exists(copied_paths.trajectory_h5);
    REQUIRE_TRUE(copied_paths.rerun_output.h5_trajectory ==
                 copied_paths.root / "output" / "rerun_traj_out.spg.h5md");

    std::filesystem::remove_all(copied_paths.root);
}

void Test_Fixture_Helper_Missing_Path_Has_Explicit_Message()
{
    const auto missing_path =
        SpongeH5Test::Unique_Temp_Path("missing_fixture_path") /
        "missing_restart.spgr.h5";

    bool threw = false;
    try
    {
        SpongeH5InputMatrix::Require_Path_Exists(missing_path);
    }
    catch (const TestFailure& error)
    {
        threw = true;
        const std::string message = error.what();
        REQUIRE_TRUE(message.find("missing H5 input matrix fixture path") !=
                     std::string::npos);
        REQUIRE_TRUE(message.find("missing_restart.spgr.h5") !=
                     std::string::npos);
    }
    REQUIRE_TRUE(threw);

    const auto incomplete_case =
        SpongeH5Test::Unique_Temp_Path("missing_fixture_describe_case");
    std::filesystem::create_directories(incomplete_case);
    threw = false;
    try
    {
        (void)SpongeH5InputMatrix::Describe_Case(incomplete_case,
                                                 "mdin.bundled.spg.toml");
    }
    catch (const TestFailure& error)
    {
        threw = true;
        const std::string message = error.what();
        REQUIRE_TRUE(message.find("missing H5 input matrix fixture path") !=
                     std::string::npos);
        REQUIRE_TRUE(message.find("mdin.bundled.spg.toml") !=
                     std::string::npos);
    }
    REQUIRE_TRUE(threw);
    std::filesystem::remove_all(incomplete_case);
}

void Test_Pure_Bundled_Fixtures_Are_Sidecar_Free()
{
    for (const Mode mode : {Mode::normal, Mode::rerun})
    {
        const auto source = Source_Dir_For(mode, InputFamily::bundled);
        REQUIRE_TRUE(!Path_Tree_Contains(source, "legacy_sidecars"));
        REQUIRE_TRUE(Read_Text_File(Mdin_For(mode, InputFamily::bundled))
                         .find("legacy_sidecars") == std::string::npos);
        Require_H5_Has_No_Legacy_Sidecars(source / "topology.spgt.h5");
        Require_H5_Has_No_Legacy_Sidecars(source / "protocol.spgp.h5");
        Require_H5_Has_No_Legacy_Sidecars(source / "restart.spgr.h5");
        if (mode == Mode::rerun)
        {
            Require_H5_Has_No_Legacy_Sidecars(source / "trajectory.spg.h5md");
        }
    }
}

void Test_Bundled_With_Sidecar_Fixtures_Carry_Sidecar_Tables()
{
    for (const Mode mode : {Mode::normal, Mode::rerun})
    {
        const auto source =
            Source_Dir_For(mode, InputFamily::bundled_with_sidecar);
        Require_H5_Sidecar_Paths_Are_Materialized(source / "topology.spgt.h5");
        Require_H5_Sidecar_Paths_Are_Materialized(source / "protocol.spgp.h5");
        Require_H5_Sidecar_Paths_Are_Materialized(source / "restart.spgr.h5");
        if (mode == Mode::rerun)
        {
            Require_H5_Has_No_Legacy_Sidecars(source / "trajectory.spg.h5md");
            REQUIRE_TRUE(Sidecar_Key_Set(source / "topology.spgt.h5") ==
                         std::set<std::string>({
                             "pairwise_force_in_file",
                             "listed_forces_in_file",
                             "mass_in_file",
                             "charge_in_file",
                             "exclude_in_file",
                             "bond_in_file",
                             "angle_in_file",
                             "dihedral_in_file",
                             "LJ_in_file",
                             "nb14_extra_in_file",
                             "urey_bradley_in_file",
                             "cmap_in_file",
                             "gb_in_file",
                             "virtual_atom_in_file",
                             "LJ_soft_core_in_file",
                             "subsys_division_in_file",
                             "EAM_in_file",
                             "EAM_atom_type_in_file",
                             "SW_in_file",
                             "EDIP_in_file",
                             "TERSOFF_in_file",
                             "REAXFF_in_file",
                             "REAXFF_type_in_file",
                         }));
            REQUIRE_TRUE(Sidecar_Key_Set(source / "protocol.spgp.h5") ==
                         std::set<std::string>({
                             "cv_in_file",
                             "constrain_in_file",
                             "restrain_in_file",
                             "soft_walls_in_file",
                             "SITS_in_file",
                             "SITS_atom_in_file",
                             "restrain_atom_id",
                             "restrain_weight_in_file",
                             "meta_edge_in_file",
                             "restrain_cv_in_file",
                             "steer_cv_in_file",
                         }));
            REQUIRE_TRUE(Sidecar_Key_Set(source / "restart.spgr.h5") ==
                         std::set<std::string>({
                             "SITS_nk_in_file",
                             "restrain_coordinate_in_file",
                             "meta_potential_in_file",
                             "meta_scatter_in_file",
                             "nose_hoover_chain_restart_input",
                             "hills_in_file",
                             "cv_in_file",
                             "constrain_in_file",
                             "restrain_in_file",
                             "pairwise_force_in_file",
                             "listed_forces_in_file",
                             "soft_walls_in_file",
                             "SITS_in_file",
                             "SITS_atom_in_file",
                             "restrain_atom_id",
                             "restrain_weight_in_file",
                             "meta_edge_in_file",
                             "restrain_cv_in_file",
                             "steer_cv_in_file",
                         }));
        }
        else
        {
            REQUIRE_TRUE(
                Sidecar_Key_Set(source / "topology.spgt.h5") ==
                std::set<std::string>({"mass_in_file", "charge_in_file"}));
            REQUIRE_TRUE(
                Sidecar_Key_Set(source / "protocol.spgp.h5") ==
                std::set<std::string>(
                    {"cv_in_file", "restrain_in_file", "SITS_in_file"}));
            REQUIRE_TRUE(
                Sidecar_Key_Set(source / "restart.spgr.h5") ==
                std::set<std::string>({"SITS_nk_in_file", "cv_in_file",
                                       "restrain_in_file", "SITS_in_file"}));
        }
        REQUIRE_TRUE(Materialized_Sidecar_File_Set(source) ==
                     Referenced_Sidecar_File_Set(source));
    }
}

void Test_Rerun_Bundled_Fixtures_Contain_Trajectory_Input()
{
    for (const InputFamily input :
         {InputFamily::bundled, InputFamily::bundled_with_sidecar})
    {
        const auto source = Source_Dir_For(Mode::rerun, input);
        Require_Rerun_Trajectory_Frames(source / "trajectory.spg.h5md", 2);
    }
}

void Test_Full_Contract_Bundled_Mdin_Locks_H5_Input_Output_Contract()
{
    for (const InputFamily input :
         {InputFamily::bundled, InputFamily::bundled_with_sidecar})
    {
        const auto mdin = Read_Text_File(Mdin_For(Mode::rerun, input));
        Require_Key_Value_Line(mdin, "mode", "\"rerun\"");
        Require_Key_Value_Line(mdin, "step_limit", "10");
        Require_Key_Value_Line(mdin, "input_h5_topology_path",
                               "\"topology.spgt.h5\"");
        Require_Key_Value_Line(mdin, "input_h5_protocol_path",
                               "\"protocol.spgp.h5\"");
        Require_Key_Value_Line(mdin, "input_h5_restart_path",
                               "\"restart.spgr.h5\"");
        Require_Key_Value_Line(mdin, "input_h5_restart_load", "\"full\"");
        Require_Key_Value_Line(mdin, "input_h5_trajectory_path",
                               "\"trajectory.spg.h5md\"");
        Require_Key_Value_Line(mdin, "input_h5_trajectory_particle_stream",
                               "\"all\"");
        Require_Key_Value_Line(mdin, "output_h5_trajectory_path",
                               "\"prod.spg.h5md\"");
        Require_Key_Value_Line(mdin, "output_h5_trajectory_vds", "true");
        Require_Key_Value_Line(mdin, "output_h5_restart_path",
                               "\"prod.spgr.h5\"");
        Require_Key_Value_Line(mdin, "output_h5_observable_path",
                               "\"prod.obs.spg.h5md\"");
        Require_Key_Value_Line(mdin, "mdout", "\"mdout.txt\"");

        REQUIRE_TRUE(!Has_Key_Line(mdin, "coordinate_in_file"));
        REQUIRE_TRUE(!Has_Key_Line(mdin, "velocity_in_file"));
        REQUIRE_TRUE(!Has_Key_Line(mdin, "rst7"));
        REQUIRE_TRUE(!Has_Key_Line(mdin, "crd"));
        REQUIRE_TRUE(!Has_Key_Line(mdin, "box"));
        REQUIRE_TRUE(!Has_Key_Line(mdin, "vel"));

        if (input == InputFamily::bundled)
        {
            REQUIRE_TRUE(mdin.find("legacy_sidecars") == std::string::npos);
            REQUIRE_TRUE(!Has_Key_Line(mdin, "custom_pair_in_file"));
            REQUIRE_TRUE(!Has_Key_Line(mdin, "custom_bond_in_file"));
        }
        else
        {
            Require_Key_Value_Line(
                mdin, "custom_pair_in_file",
                "\"legacy_sidecars/custom_pair_in_file/custom_pair.txt\"");
            Require_Key_Value_Line(
                mdin, "custom_bond_in_file",
                "\"legacy_sidecars/custom_bond_in_file/custom_bond.txt\"");
        }
    }
}

void Test_Rerun_Frame_Selection_Uses_Fixture_Trajectory()
{
    for (const InputFamily input :
         {InputFamily::bundled, InputFamily::bundled_with_sidecar})
    {
        const auto source = Source_Dir_For(Mode::rerun, input);
        Require_Rerun_Selection_Reads_Expected_Frame(
            source / "trajectory.spg.h5md", 0, 1, 1);
        Require_Rerun_Selection_Reads_Expected_Frame(
            source / "trajectory.spg.h5md", 1, 0, 1);
    }
}
}  // namespace

int main()
{
    return SpongeH5Test::Run_Test(
        []
        {
            Test_Legal_Matrix_Cases_Are_Explicit_And_Fixture_Backed();
            Test_Fixture_Helper_Copies_Case_To_Temp();
            Test_Fixture_Helper_Describes_Case_Paths();
            Test_Fixture_Helper_Missing_Path_Has_Explicit_Message();
            Test_Pure_Bundled_Fixtures_Are_Sidecar_Free();
            Test_Bundled_With_Sidecar_Fixtures_Carry_Sidecar_Tables();
            Test_Rerun_Bundled_Fixtures_Contain_Trajectory_Input();
            Test_Full_Contract_Bundled_Mdin_Locks_H5_Input_Output_Contract();
            Test_Rerun_Frame_Selection_Uses_Fixture_Trajectory();
        });
}

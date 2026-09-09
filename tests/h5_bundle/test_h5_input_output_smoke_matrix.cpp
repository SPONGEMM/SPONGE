#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <highfive/highfive.hpp>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "h5_input_matrix_fixture.hpp"
#include "test_shell.hpp"
#include "utils/h5md/h5_legacy_sidecar.hpp"
#include "utils/h5md/output_route_helpers.hpp"

namespace
{
constexpr int kSkipReturnCode = 77;
constexpr float kTolerance = 1.0e-5f;

std::string Raw_Restart_Topology_Hash()
{
    return "sha256:" + std::string(64, 'a');
}

std::string Raw_Restart_Atom_Order_Hash()
{
    return "sha256:" + std::string(64, 'b');
}

struct CoreState
{
    std::vector<float> position = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
    };
    std::vector<float> velocity = {
        0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
    };
    std::vector<float> box = {
        10.0f, 0.0f, 0.0f, 0.0f, 20.0f, 0.0f, 0.0f, 0.0f, 30.0f,
    };
};

struct MdoutTable
{
    std::vector<std::string> columns;
    std::map<std::string, std::vector<double>> values_by_column;
};

std::string Read_Text(const std::filesystem::path& path)
{
    std::ifstream in(path.c_str());
    if (!in.good())
    {
        throw TestFailure("failed to read " + path.string());
    }
    std::ostringstream out;
    out << in.rdbuf();
    return out.str();
}

void Write_Text(const std::filesystem::path& path, const std::string& text)
{
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path.c_str());
    out << text;
    if (!out.good())
    {
        throw TestFailure("failed to write " + path.string());
    }
}

void Append_Text(const std::filesystem::path& path, const std::string& text)
{
    std::ofstream out(path.c_str(), std::ios::app);
    out << text;
    if (!out.good())
    {
        throw TestFailure("failed to append " + path.string());
    }
}

std::string Trim_Right(std::string text)
{
    while (!text.empty() && (text.back() == ' ' || text.back() == '\t' ||
                             text.back() == '\r' || text.back() == '\n'))
    {
        text.pop_back();
    }
    return text;
}

std::vector<std::string> Normalized_Nonempty_Lines(
    const std::filesystem::path& path)
{
    std::ifstream in(path.c_str());
    if (!in.good())
    {
        throw TestFailure("failed to read " + path.string());
    }
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line))
    {
        line = Trim_Right(line);
        if (!line.empty())
        {
            lines.push_back(line);
        }
    }
    return lines;
}

void Require_Text_Equivalent(const std::filesystem::path& lhs,
                             const std::filesystem::path& rhs)
{
    const auto lhs_lines = Normalized_Nonempty_Lines(lhs);
    const auto rhs_lines = Normalized_Nonempty_Lines(rhs);
    REQUIRE_EQ(lhs_lines.size(), rhs_lines.size());
    std::vector<std::string> header_tokens;
    for (std::size_t i = 0; i < lhs_lines.size(); ++i)
    {
        std::istringstream lhs_row(lhs_lines[i]);
        std::istringstream rhs_row(rhs_lines[i]);
        std::vector<std::string> lhs_tokens;
        std::vector<std::string> rhs_tokens;
        std::string token;
        while (lhs_row >> token)
        {
            lhs_tokens.push_back(token);
        }
        while (rhs_row >> token)
        {
            rhs_tokens.push_back(token);
        }
        REQUIRE_EQ(lhs_tokens.size(), rhs_tokens.size());
        if (i == 0)
        {
            REQUIRE_EQ(lhs_lines[i], rhs_lines[i]);
            header_tokens = lhs_tokens;
            continue;
        }
        for (std::size_t j = 0; j < lhs_tokens.size(); ++j)
        {
            const std::string column =
                j < header_tokens.size() ? header_tokens[j] : "";
            double lhs_value = 0.0;
            double rhs_value = 0.0;
            const bool lhs_numeric = SpongeH5OutputRoute::Parse_Output_Double(
                lhs_tokens[j], &lhs_value);
            const bool rhs_numeric = SpongeH5OutputRoute::Parse_Output_Double(
                rhs_tokens[j], &rhs_value);
            if (!lhs_numeric || !rhs_numeric)
            {
                REQUIRE_EQ(lhs_tokens[j], rhs_tokens[j]);
                continue;
            }
            const bool cmap_column = column == "cmap";
            const double relative_tolerance = 1.0e-4;
            const double tolerance =
                (cmap_column ? 1.0e-2 : 1.0e-6) +
                relative_tolerance *
                    std::max(std::fabs(lhs_value), std::fabs(rhs_value));
            if (std::fabs(lhs_value - rhs_value) > tolerance)
            {
                throw TestFailure("mdout value mismatch in column " + column);
            }
        }
    }
}

void Require_Contains(const std::string& text, const std::string& needle)
{
    REQUIRE_TRUE(text.find(needle) != std::string::npos);
}

bool Starts_With(const std::string& text, const std::string& prefix)
{
    return text.size() >= prefix.size() &&
           text.compare(0, prefix.size(), prefix) == 0;
}

void Require_Float_Vector_Close(const std::vector<float>& actual,
                                const std::vector<float>& expected,
                                const std::string& label)
{
    if (actual.size() != expected.size())
    {
        throw TestFailure(label + " size mismatch");
    }
    for (std::size_t i = 0; i < actual.size(); ++i)
    {
        if (std::fabs(actual[i] - expected[i]) > kTolerance)
        {
            std::ostringstream message;
            message << label << " mismatch at " << i << ": actual=" << actual[i]
                    << " expected=" << expected[i];
            throw TestFailure(message.str());
        }
    }
}

void Require_Double_Vector_Close(const std::vector<double>& actual,
                                 const std::vector<double>& expected,
                                 const std::string& label)
{
    if (actual.size() != expected.size())
    {
        throw TestFailure(label + " size mismatch");
    }
    for (std::size_t i = 0; i < actual.size(); ++i)
    {
        if (std::fabs(actual[i] - expected[i]) > 1.0e-10)
        {
            std::ostringstream message;
            message << label << " mismatch at " << i << ": actual=" << actual[i]
                    << " expected=" << expected[i];
            throw TestFailure(message.str());
        }
    }
}

std::vector<float> Read_Float_Vector(HighFive::File& file,
                                     const std::string& path)
{
    const auto dataset = file.getDataSet(path);
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

std::vector<std::int64_t> Read_Int64_Vector(HighFive::File& file,
                                            const std::string& path)
{
    std::vector<std::int64_t> values;
    file.getDataSet(path).read(values);
    return values;
}

std::vector<double> Read_Float64_Vector(HighFive::File& file,
                                        const std::string& path)
{
    std::vector<double> values;
    file.getDataSet(path).read(values);
    return values;
}

std::vector<std::string> Read_String_Vector(HighFive::File& file,
                                            const std::string& path)
{
    std::vector<std::string> values;
    file.getDataSet(path).read(values);
    return values;
}

std::string Read_String_Scalar(HighFive::File& file, const std::string& path)
{
    std::string value;
    file.getDataSet(path).read(value);
    return value;
}

MdoutTable Read_Mdout_Table(const std::filesystem::path& path)
{
    std::ifstream in(path.c_str());
    if (!in.good())
    {
        throw TestFailure("failed to read mdout table " + path.string());
    }

    MdoutTable table;
    std::string line;
    if (!std::getline(in, line))
    {
        throw TestFailure("empty mdout table " + path.string());
    }
    {
        std::istringstream header(line);
        std::string column;
        while (header >> column)
        {
            table.columns.push_back(column);
            table.values_by_column[column] = {};
        }
    }
    REQUIRE_TRUE(!table.columns.empty());

    while (std::getline(in, line))
    {
        line = Trim_Right(line);
        if (line.empty())
        {
            continue;
        }

        std::istringstream row(line);
        std::vector<std::string> tokens;
        std::string token;
        while (row >> token)
        {
            tokens.push_back(token);
        }
        REQUIRE_EQ(tokens.size(), table.columns.size());
        for (std::size_t i = 0; i < tokens.size(); ++i)
        {
            double value = 0.0;
            if (!SpongeH5OutputRoute::Parse_Output_Double(tokens[i], &value))
            {
                value = std::numeric_limits<double>::quiet_NaN();
            }
            table.values_by_column[table.columns[i]].push_back(value);
        }
    }

    for (const auto& column : table.columns)
    {
        REQUIRE_TRUE(!table.values_by_column[column].empty());
    }
    return table;
}

void Require_Mdout_Columns_Equivalent(const std::filesystem::path& lhs,
                                      const std::filesystem::path& rhs,
                                      const std::vector<std::string>& columns)
{
    const auto lhs_table = Read_Mdout_Table(lhs);
    const auto rhs_table = Read_Mdout_Table(rhs);
    for (const auto& column : columns)
    {
        const auto lhs_iter = lhs_table.values_by_column.find(column);
        const auto rhs_iter = rhs_table.values_by_column.find(column);
        REQUIRE_TRUE(lhs_iter != lhs_table.values_by_column.end());
        REQUIRE_TRUE(rhs_iter != rhs_table.values_by_column.end());
        REQUIRE_TRUE(lhs_iter->second.size() >= rhs_iter->second.size());
        const std::size_t lhs_offset =
            lhs_iter->second.size() - rhs_iter->second.size();
        for (std::size_t i = 0; i < rhs_iter->second.size(); ++i)
        {
            const double lhs_value = lhs_iter->second[lhs_offset + i];
            const double rhs_value = rhs_iter->second[i];
            const double relative_tolerance = 1.0e-4;
            const double tolerance =
                1.0e-6 + relative_tolerance * std::max(std::fabs(lhs_value),
                                                       std::fabs(rhs_value));
            if (std::fabs(lhs_value - rhs_value) > tolerance)
            {
                throw TestFailure("mdout " + column + " mismatch");
            }
        }
    }
}

void Require_Mdout_Row_Count(const std::filesystem::path& path,
                             const std::size_t expected_count)
{
    const auto table = Read_Mdout_Table(path);
    for (const auto& column : table.columns)
    {
        REQUIRE_EQ(table.values_by_column.at(column).size(), expected_count);
    }
}

void Require_Core_Mdout_Equivalent(const std::filesystem::path& lhs,
                                   const std::filesystem::path& rhs)
{
    Require_Mdout_Columns_Equivalent(
        lhs, rhs,
        {"step", "time", "temperature", "potential", "eff_pot", "PM"});
}

void Require_Rerun_Mdout_Equivalent(const std::filesystem::path& lhs,
                                    const std::filesystem::path& rhs)
{
    Require_Mdout_Columns_Equivalent(
        lhs, rhs,
        {"temperature",           "LJ_short",      "LJ_long",
         "LJ",          "LJ_soft",     "LJ_soft_short", "LJ_soft_long",
         "PM",          "custom_pair", "nb14_LJ",       "nb14_EE",
         "bond",        "angle",       "urey_bradley",  "dihedral",
         "custom_bond", "SW",          "EAM",           "restrain",
         "z_wall",      "distance"});
}

void Require_Rerun_Selection_Mdout_Equivalent(const std::filesystem::path& lhs,
                                              const std::filesystem::path& rhs)
{
    Require_Mdout_Columns_Equivalent(
        lhs, rhs,
        {"frame",        "temperature",           "LJ_short",
         "LJ_long",      "LJ",          "LJ_soft",     "LJ_soft_short",
         "LJ_soft_long", "PM",          "custom_pair", "nb14_LJ",
         "nb14_EE",      "bond",        "angle",       "urey_bradley",
         "dihedral",     "custom_bond", "SW",          "EAM",
         "z_wall",       "distance"});
}

void Require_Pure_Bundled_Rerun_Mdout_Core_Equivalent(
    const std::filesystem::path& lhs, const std::filesystem::path& rhs)
{
    Require_Mdout_Columns_Equivalent(
        lhs, rhs,
        {"temperature", "LJ_short", "LJ_long", "LJ", "LJ_soft", "LJ_soft_short",
         "LJ_soft_long", "PM", "bond", "angle", "urey_bradley", "dihedral"});
}

void Require_H5_Restart_Matches_Core_State(const std::filesystem::path& path)
{
    const CoreState expected;
    HighFive::File file(path.string(), HighFive::File::ReadOnly);
    Require_Float_Vector_Close(
        Read_Float_Vector(file, "/particles/all/position/value"),
        expected.position, "restart position");
    Require_Float_Vector_Close(
        Read_Float_Vector(file, "/particles/all/velocity/value"),
        expected.velocity, "restart velocity");
    Require_Float_Vector_Close(
        Read_Float_Vector(file, "/particles/all/box/edges/value"), expected.box,
        "restart box");

    std::vector<std::int64_t> steps;
    file.getDataSet("/particles/all/step").read(steps);
    REQUIRE_EQ(steps.size(), static_cast<std::size_t>(1));
    REQUIRE_EQ(steps[0], static_cast<std::int64_t>(0));

    std::vector<double> times;
    file.getDataSet("/particles/all/time").read(times);
    REQUIRE_EQ(times.size(), static_cast<std::size_t>(1));
    REQUIRE_TRUE(std::fabs(times[0]) < 1.0e-12);
    REQUIRE_TRUE(file.exist(SpongeH5MD::path::run_topology_hash));
    REQUIRE_TRUE(file.exist(SpongeH5MD::path::run_atom_order_hash));
    REQUIRE_TRUE(file.exist(SpongeH5MD::path::run_state_hash));
    REQUIRE_TRUE(SpongeH5OutputContract::Is_Canonical_Sha256(
        Read_String_Scalar(file, SpongeH5MD::path::run_topology_hash)));
    REQUIRE_TRUE(SpongeH5OutputContract::Is_Canonical_Sha256(
        Read_String_Scalar(file, SpongeH5MD::path::run_atom_order_hash)));
    REQUIRE_TRUE(SpongeH5OutputContract::Is_Canonical_Sha256(
        Read_String_Scalar(file, SpongeH5MD::path::run_state_hash)));
}

void Require_Frame_Sequence(const std::vector<std::int64_t>& actual_steps,
                            const std::vector<double>& actual_times,
                            const std::vector<std::int64_t>& expected_steps,
                            const std::vector<double>& expected_times)
{
    REQUIRE_EQ(expected_steps.size(), expected_times.size());
    REQUIRE_EQ(actual_steps.size(), expected_steps.size());
    REQUIRE_EQ(actual_times.size(), expected_times.size());
    for (std::size_t i = 0; i < expected_steps.size(); ++i)
    {
        REQUIRE_EQ(actual_steps[i], expected_steps[i]);
        REQUIRE_TRUE(std::fabs(actual_times[i] - expected_times[i]) < 1.0e-12);
    }
}

void Require_H5_Trajectory_Has_Frames(const std::filesystem::path& path,
                                      const std::vector<std::int64_t>& steps,
                                      const std::vector<double>& times)
{
    HighFive::File file(path.string(), HighFive::File::ReadOnly);
    const auto actual_steps = Read_Int64_Vector(file, "/particles/all/step");
    const auto actual_times = Read_Float64_Vector(file, "/particles/all/time");
    const auto positions =
        Read_Float_Vector(file, "/particles/all/position/value");
    const auto boxes =
        Read_Float_Vector(file, "/particles/all/box/edges/value");
    const auto expected_frames = steps.size();

    Require_Frame_Sequence(actual_steps, actual_times, steps, times);
    REQUIRE_EQ(positions.size(), expected_frames * 2 * 3);
    REQUIRE_EQ(boxes.size(), expected_frames * 3 * 3);

    if (file.exist("/parameters/sponge/output/frame_count"))
    {
        const auto frame_count =
            Read_Int64_Vector(file, "/parameters/sponge/output/frame_count");
        REQUIRE_TRUE(!frame_count.empty());
        REQUIRE_EQ(frame_count.back(),
                   static_cast<std::int64_t>(expected_frames));
    }
}

std::filesystem::path Resolve_Vds_Shard_Path(
    const std::filesystem::path& wrapper_path, const std::string& manifest_path)
{
    const std::filesystem::path shard_path(manifest_path);
    if (shard_path.is_absolute())
    {
        return shard_path;
    }
    const auto wrapper_relative = wrapper_path.parent_path() / shard_path;
    if (std::filesystem::exists(wrapper_relative))
    {
        return wrapper_relative;
    }
    const auto case_relative =
        wrapper_path.parent_path().parent_path() / shard_path;
    if (std::filesystem::exists(case_relative))
    {
        return case_relative;
    }
    return wrapper_relative;
}

void Require_VDS_Shards_Are_Complete(
    const std::filesystem::path& wrapper_path,
    const std::vector<std::int64_t>& expected_steps,
    const std::vector<double>& expected_times,
    const std::vector<std::int64_t>& expected_shard_step_starts,
    const std::vector<double>& expected_shard_time_starts)
{
    SpongeH5InputMatrix::Require_Path_Exists(wrapper_path);
    HighFive::File file(wrapper_path.string(), HighFive::File::ReadOnly);

    REQUIRE_TRUE(file.exist(SpongeH5MD::path::output_status));
    REQUIRE_TRUE(file.exist(SpongeH5MD::path::output_vds_status));
    REQUIRE_TRUE(file.exist(SpongeH5MD::path::output_repair_policy));
    REQUIRE_TRUE(file.exist(SpongeH5MD::path::output_repair_status));
    REQUIRE_TRUE(file.exist(SpongeH5MD::path::output_repaired_shard_count));
    REQUIRE_EQ(Read_String_Scalar(file, SpongeH5MD::path::output_status),
               std::string("finalized"));
    REQUIRE_EQ(Read_String_Scalar(file, SpongeH5MD::path::output_vds_status),
               std::string("particle, observable, and module virtual datasets "
                           "materialized"));
    REQUIRE_EQ(Read_String_Scalar(file, SpongeH5MD::path::output_repair_policy),
               std::string("strict"));
    REQUIRE_EQ(Read_String_Scalar(file, SpongeH5MD::path::output_repair_status),
               std::string("not_applied"));
    const auto repaired_shard_count =
        Read_Int64_Vector(file, SpongeH5MD::path::output_repaired_shard_count);
    REQUIRE_TRUE(!repaired_shard_count.empty());
    REQUIRE_EQ(repaired_shard_count.back(), static_cast<std::int64_t>(0));

    REQUIRE_TRUE(file.exist(SpongeH5MD::path::output_frame_count));
    REQUIRE_TRUE(file.exist(SpongeH5MD::path::output_last_complete_step));
    REQUIRE_TRUE(file.exist(SpongeH5MD::path::output_last_complete_time));
    const auto output_frame_count =
        Read_Int64_Vector(file, SpongeH5MD::path::output_frame_count);
    const auto output_last_step =
        Read_Int64_Vector(file, SpongeH5MD::path::output_last_complete_step);
    const auto output_last_time =
        Read_Float64_Vector(file, SpongeH5MD::path::output_last_complete_time);
    REQUIRE_TRUE(!output_frame_count.empty());
    REQUIRE_EQ(output_frame_count.back(),
               static_cast<std::int64_t>(expected_steps.size()));
    REQUIRE_EQ(output_last_step.back(), expected_steps.back());
    REQUIRE_TRUE(std::fabs(output_last_time.back() - expected_times.back()) <
                 1.0e-12);

    REQUIRE_TRUE(file.exist(SpongeH5MD::path::shard_manifest_index));
    REQUIRE_TRUE(file.exist(SpongeH5MD::path::shard_manifest_path));
    REQUIRE_TRUE(file.exist(SpongeH5MD::path::shard_manifest_status));
    REQUIRE_TRUE(file.exist(SpongeH5MD::path::shard_manifest_frame_start));
    REQUIRE_TRUE(file.exist(SpongeH5MD::path::shard_manifest_frame_count));
    REQUIRE_TRUE(file.exist(SpongeH5MD::path::shard_manifest_step_start));
    REQUIRE_TRUE(file.exist(SpongeH5MD::path::shard_manifest_step_end));
    REQUIRE_TRUE(file.exist(SpongeH5MD::path::shard_manifest_time_start));
    REQUIRE_TRUE(file.exist(SpongeH5MD::path::shard_manifest_time_end));

    const auto indices =
        Read_Int64_Vector(file, SpongeH5MD::path::shard_manifest_index);
    const auto paths =
        Read_String_Vector(file, SpongeH5MD::path::shard_manifest_path);
    const auto statuses =
        Read_String_Vector(file, SpongeH5MD::path::shard_manifest_status);
    const auto frame_starts =
        Read_Int64_Vector(file, SpongeH5MD::path::shard_manifest_frame_start);
    const auto frame_counts =
        Read_Int64_Vector(file, SpongeH5MD::path::shard_manifest_frame_count);
    const auto step_starts =
        Read_Int64_Vector(file, SpongeH5MD::path::shard_manifest_step_start);
    const auto step_ends =
        Read_Int64_Vector(file, SpongeH5MD::path::shard_manifest_step_end);
    const auto time_starts =
        Read_Float64_Vector(file, SpongeH5MD::path::shard_manifest_time_start);
    const auto time_ends =
        Read_Float64_Vector(file, SpongeH5MD::path::shard_manifest_time_end);
    const auto wrapper_positions =
        Read_Float_Vector(file, "/particles/all/position/value");
    const auto wrapper_boxes =
        Read_Float_Vector(file, "/particles/all/box/edges/value");

    const std::size_t shard_count = indices.size();
    REQUIRE_TRUE(shard_count > 0);
    REQUIRE_EQ(expected_shard_step_starts.size(), shard_count);
    REQUIRE_EQ(expected_shard_time_starts.size(), shard_count);
    REQUIRE_EQ(paths.size(), shard_count);
    REQUIRE_EQ(statuses.size(), shard_count);
    REQUIRE_EQ(frame_starts.size(), shard_count);
    REQUIRE_EQ(frame_counts.size(), shard_count);
    REQUIRE_EQ(step_starts.size(), shard_count);
    REQUIRE_EQ(step_ends.size(), shard_count);
    REQUIRE_EQ(time_starts.size(), shard_count);
    REQUIRE_EQ(time_ends.size(), shard_count);
    REQUIRE_TRUE(!expected_steps.empty());
    REQUIRE_TRUE(wrapper_positions.size() % expected_steps.size() == 0);
    REQUIRE_TRUE(wrapper_boxes.size() % expected_steps.size() == 0);
    const std::size_t position_values_per_frame =
        wrapper_positions.size() / expected_steps.size();
    const std::size_t box_values_per_frame =
        wrapper_boxes.size() / expected_steps.size();

    std::int64_t next_frame_start = 0;
    for (std::size_t i = 0; i < shard_count; ++i)
    {
        REQUIRE_EQ(indices[i], static_cast<std::int64_t>(i));
        REQUIRE_EQ(statuses[i], std::string("complete"));
        REQUIRE_EQ(frame_starts[i], next_frame_start);
        REQUIRE_TRUE(frame_counts[i] > 0);
        const std::size_t first_frame =
            static_cast<std::size_t>(frame_starts[i]);
        const std::size_t last_frame =
            first_frame + static_cast<std::size_t>(frame_counts[i]) - 1;
        REQUIRE_TRUE(last_frame < expected_steps.size());
        // A shard can open on an observable before its first particle frame.
        REQUIRE_EQ(step_starts[i], expected_shard_step_starts[i]);
        REQUIRE_EQ(step_ends[i], expected_steps[last_frame]);
        REQUIRE_TRUE(std::fabs(time_starts[i] - expected_shard_time_starts[i]) <
                     1.0e-12);
        REQUIRE_TRUE(std::fabs(time_ends[i] - expected_times[last_frame]) <
                     1.0e-12);

        const auto shard_path = Resolve_Vds_Shard_Path(wrapper_path, paths[i]);
        SpongeH5InputMatrix::Require_Path_Exists(shard_path);
        REQUIRE_TRUE(shard_path.parent_path().filename().string().find(
                         ".spg.shards") != std::string::npos);
        REQUIRE_TRUE(shard_path.filename().string().find("segment_") == 0);

        HighFive::File shard_file(shard_path.string(),
                                  HighFive::File::ReadOnly);
        const auto shard_steps =
            Read_Int64_Vector(shard_file, "/particles/all/step");
        const auto shard_times =
            Read_Float64_Vector(shard_file, "/particles/all/time");
        REQUIRE_EQ(shard_steps.size(),
                   static_cast<std::size_t>(frame_counts[i]));
        REQUIRE_EQ(shard_times.size(),
                   static_cast<std::size_t>(frame_counts[i]));
        const auto shard_positions =
            Read_Float_Vector(shard_file, "/particles/all/position/value");
        const auto shard_boxes =
            Read_Float_Vector(shard_file, "/particles/all/box/edges/value");
        REQUIRE_EQ(shard_positions.size(),
                   static_cast<std::size_t>(frame_counts[i]) *
                       position_values_per_frame);
        REQUIRE_EQ(
            shard_boxes.size(),
            static_cast<std::size_t>(frame_counts[i]) * box_values_per_frame);
        for (std::size_t frame = 0; frame < shard_steps.size(); ++frame)
        {
            const std::size_t expected_index = first_frame + frame;
            REQUIRE_EQ(shard_steps[frame], expected_steps[expected_index]);
            REQUIRE_TRUE(std::fabs(shard_times[frame] -
                                   expected_times[expected_index]) < 1.0e-12);
            const std::size_t shard_position_offset =
                frame * position_values_per_frame;
            const std::size_t wrapper_position_offset =
                expected_index * position_values_per_frame;
            Require_Float_Vector_Close(
                std::vector<float>(
                    shard_positions.begin() + shard_position_offset,
                    shard_positions.begin() + shard_position_offset +
                        position_values_per_frame),
                std::vector<float>(
                    wrapper_positions.begin() + wrapper_position_offset,
                    wrapper_positions.begin() + wrapper_position_offset +
                        position_values_per_frame),
                "VDS shard position");

            const std::size_t shard_box_offset = frame * box_values_per_frame;
            const std::size_t wrapper_box_offset =
                expected_index * box_values_per_frame;
            Require_Float_Vector_Close(
                std::vector<float>(shard_boxes.begin() + shard_box_offset,
                                   shard_boxes.begin() + shard_box_offset +
                                       box_values_per_frame),
                std::vector<float>(wrapper_boxes.begin() + wrapper_box_offset,
                                   wrapper_boxes.begin() + wrapper_box_offset +
                                       box_values_per_frame),
                "VDS shard box");
        }

        next_frame_start += frame_counts[i];
    }
    REQUIRE_EQ(next_frame_start,
               static_cast<std::int64_t>(expected_steps.size()));
}

void Require_H5_Trajectory_First_Frame_Matches_Core_State(
    const std::filesystem::path& path, const std::int64_t expected_step)
{
    const CoreState expected;
    HighFive::File file(path.string(), HighFive::File::ReadOnly);
    const auto steps = Read_Int64_Vector(file, "/particles/all/step");
    const auto times = Read_Float64_Vector(file, "/particles/all/time");
    const auto positions =
        Read_Float_Vector(file, "/particles/all/position/value");
    const auto boxes =
        Read_Float_Vector(file, "/particles/all/box/edges/value");

    REQUIRE_TRUE(!steps.empty());
    REQUIRE_TRUE(!times.empty());
    REQUIRE_EQ(steps[0], expected_step);
    REQUIRE_TRUE(std::fabs(times[0]) < 1.0e-12);

    REQUIRE_TRUE(positions.size() >= expected.position.size());
    REQUIRE_TRUE(boxes.size() >= expected.box.size());
    Require_Float_Vector_Close(
        std::vector<float>(positions.begin(),
                           positions.begin() + expected.position.size()),
        expected.position, "trajectory first-frame position");
    Require_Float_Vector_Close(
        std::vector<float>(boxes.begin(), boxes.begin() + expected.box.size()),
        expected.box, "trajectory first-frame box");
}

void Require_H5_Trajectory_Frame_Matches_Rerun_Runtime_State(
    const std::filesystem::path& path, const std::size_t output_frame_index)
{
    HighFive::File file(path.string(), HighFive::File::ReadOnly);
    const auto positions =
        Read_Float_Vector(file, "/particles/all/position/value");
    const auto boxes =
        Read_Float_Vector(file, "/particles/all/box/edges/value");

    // The two-atom I/O fixture uses type 2 with sources {0, 0, 0}.
    // Its target (atom 1) coincides with atom 0 after coordinate refresh.
    const std::vector<float> expected_position = {
        1.5f, 2.5f, 3.5f, 1.5f, 2.5f, 3.5f,
    };
    const std::vector<float> expected_box = {
        10.0f, 0.0f, 0.0f, 0.0f, 20.0f, 0.0f, 0.0f, 0.0f, 30.0f,
    };

    const std::size_t position_offset =
        output_frame_index * expected_position.size();
    const std::size_t box_offset = output_frame_index * expected_box.size();
    REQUIRE_TRUE(positions.size() >=
                 position_offset + expected_position.size());
    REQUIRE_TRUE(boxes.size() >= box_offset + expected_box.size());
    Require_Float_Vector_Close(
        std::vector<float>(
            positions.begin() + position_offset,
            positions.begin() + position_offset + expected_position.size()),
        expected_position, "trajectory rerun output position");
    Require_Float_Vector_Close(
        std::vector<float>(boxes.begin() + box_offset,
                           boxes.begin() + box_offset + expected_box.size()),
        expected_box, "trajectory rerun output box");
}

void Require_H5_Observable_Stream_Matches_Mdout(
    const std::filesystem::path& path, const std::filesystem::path& mdout_path,
    const std::vector<std::int64_t>& expected_steps,
    const std::vector<double>& expected_times)
{
    HighFive::File file(path.string(), HighFive::File::ReadOnly);
    REQUIRE_TRUE(file.exist(SpongeH5MD::path::observables_all_step));
    REQUIRE_TRUE(file.exist(SpongeH5MD::path::observables_all_time));
    REQUIRE_TRUE(file.exist(SpongeH5MD::path::mdout_columns_original_name));
    REQUIRE_TRUE(file.exist(SpongeH5MD::path::mdout_columns_hdf5_name));

    const auto steps =
        Read_Int64_Vector(file, SpongeH5MD::path::observables_all_step);
    const auto times =
        Read_Float64_Vector(file, SpongeH5MD::path::observables_all_time);
    Require_Frame_Sequence(steps, times, expected_steps, expected_times);
    const auto expected_frames = expected_steps.size();

    std::vector<std::string> original_columns;
    std::vector<std::string> hdf5_columns;
    file.getDataSet(SpongeH5MD::path::mdout_columns_original_name)
        .read(original_columns);
    file.getDataSet(SpongeH5MD::path::mdout_columns_hdf5_name)
        .read(hdf5_columns);
    REQUIRE_TRUE(!original_columns.empty());
    REQUIRE_EQ(original_columns.size(), hdf5_columns.size());

    const std::string first_value_path =
        SpongeH5MD::Observable_Value_Path(hdf5_columns.front());
    REQUIRE_TRUE(file.exist(first_value_path));
    const auto first_values = Read_Float64_Vector(file, first_value_path);
    REQUIRE_EQ(first_values.size(), expected_frames);

    const auto mdout = Read_Mdout_Table(mdout_path);
    for (std::size_t i = 0; i < original_columns.size(); ++i)
    {
        const auto mdout_column =
            mdout.values_by_column.find(original_columns[i]);
        REQUIRE_TRUE(mdout_column != mdout.values_by_column.end());
        REQUIRE_EQ(mdout_column->second.size(), expected_frames);

        const std::string value_path =
            SpongeH5MD::Observable_Value_Path(hdf5_columns[i]);
        REQUIRE_TRUE(file.exist(value_path));
        const auto h5_values = Read_Float64_Vector(file, value_path);
        Require_Double_Vector_Close(h5_values, mdout_column->second,
                                    "observable " + original_columns[i]);
    }
}

void Copy_Directory_Contents(const std::filesystem::path& source,
                             const std::filesystem::path& destination)
{
    SpongeH5InputMatrix::Require_Path_Exists(source);
    std::filesystem::create_directories(destination);
    for (const auto& entry : std::filesystem::directory_iterator(source))
    {
        std::filesystem::copy(
            entry.path(), destination / entry.path().filename(),
            std::filesystem::copy_options::recursive |
                std::filesystem::copy_options::overwrite_existing);
    }
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

std::string Remove_Key_Lines(const std::string& mdin,
                             const std::vector<std::string>& keys)
{
    std::istringstream input(mdin);
    std::ostringstream output;
    std::string line;
    while (std::getline(input, line))
    {
        bool remove = false;
        const auto stripped = line.substr(0, line.find('#'));
        for (const auto& key : keys)
        {
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
                remove = true;
                break;
            }
        }
        if (!remove)
        {
            output << line << "\n";
        }
    }
    return output.str();
}

std::string Remove_Toml_Section(const std::string& mdin,
                                const std::string& section_name)
{
    std::istringstream input(mdin);
    std::ostringstream output;
    std::string line;
    bool skipping = false;
    const std::string section_header = "[" + section_name + "]";
    while (std::getline(input, line))
    {
        const std::string stripped = Trim_Right(line);
        if (stripped == section_header)
        {
            skipping = true;
            continue;
        }
        if (skipping && stripped.size() >= 2 && stripped.front() == '[' &&
            stripped.back() == ']')
        {
            skipping = false;
        }
        if (!skipping)
        {
            output << line << "\n";
        }
    }
    return output.str();
}

void Append_If_Missing(std::string* mdin, const std::string& key,
                       const std::string& line)
{
    if (!Has_Key_Line(*mdin, key))
    {
        std::size_t insert_pos = mdin->size();
        std::size_t line_begin = 0;
        while (line_begin < mdin->size())
        {
            const std::size_t line_end = mdin->find('\n', line_begin);
            const std::size_t raw_line_end =
                line_end == std::string::npos ? mdin->size() : line_end;
            const std::string current =
                mdin->substr(line_begin, raw_line_end - line_begin);
            std::size_t pos = 0;
            while (pos < current.size() &&
                   std::isspace(static_cast<unsigned char>(current[pos])))
            {
                ++pos;
            }
            if (pos < current.size() && current[pos] == '[')
            {
                insert_pos = line_begin;
                break;
            }
            if (line_end == std::string::npos)
            {
                break;
            }
            line_begin = line_end + 1;
        }

        std::string insertion = line + "\n";
        if (insert_pos > 0 && (*mdin)[insert_pos - 1] != '\n')
        {
            insertion = "\n" + insertion;
        }
        mdin->insert(insert_pos, insertion);
    }
}

std::string Toml_Quoted_String(const std::string& raw)
{
    std::string quoted = "\"";
    for (const char c : raw)
    {
        if (c == '\\' || c == '"')
        {
            quoted += '\\';
        }
        quoted += c;
    }
    quoted += "\"";
    return quoted;
}

std::string Toml_Relative_Path(const std::filesystem::path& root,
                               const std::filesystem::path& path)
{
    return Toml_Quoted_String(path.lexically_relative(root).generic_string());
}

struct PreparedCase
{
    std::filesystem::path root;
    std::filesystem::path mdin;
    std::filesystem::path mdout;
    std::filesystem::path mdinfo;
    std::filesystem::path h5_restart;
    std::filesystem::path h5_trajectory;
    std::filesystem::path h5_observable;
};

void Install_Valid_EAM_Atom_Types(const std::filesystem::path& root)
{
    // The synthetic funcfl fixture has one element. Its serialized type 1
    // is out of range during simulation.
    // Normalize every runtime representation, including legacy and sidecar
    // input.
    for (const auto& relative :
         {"eam_atom_type.txt",
          "legacy_sidecars/EAM_atom_type_in_file/eam_atom_type.txt"})
    {
        const auto path = root / relative;
        if (std::filesystem::exists(path)) Write_Text(path, "0\n0\n");
    }
    const auto topology_h5 = root / "topology.spgt.h5";
    if (!std::filesystem::exists(topology_h5)) return;
    HighFive::File file(topology_h5.string(), HighFive::File::ReadWrite);
    if (!file.exist("/manybody/eam/atom_type")) return;
    const std::vector<int> atom_type = {0, 0};
    file.getDataSet("/manybody/eam/atom_type").write(atom_type);
}

struct RerunSelection
{
    int start = 0;
    int strip = 0;
    int frame_limit = 2;
};

struct NormalSmokeCase
{
    const char* name;
    std::filesystem::path source_dir;
    const char* mdin_name;
    bool bundled_output;
};

struct RerunLegacyOutputCase
{
    const char* name;
    std::filesystem::path source_dir;
    const char* mdin_name;
};

struct RerunBundledOutputCase
{
    const char* name;
    std::filesystem::path source_dir;
    const char* mdin_name;
    bool vds;
};

enum class SidecarSmokeKind
{
    injected_without_explicit_keys,
    same_key_same_path,
    same_key_different_path,
    pure_bundled_without_sidecar_files,
};

struct SidecarSmokeCase
{
    const char* name;
    std::filesystem::path source_dir;
    const char* mdin_name;
    SidecarSmokeKind kind;
};

template <typename CaseType>
void Require_Case_Name_Set(const std::vector<CaseType>& cases,
                           const std::set<std::string>& expected_names)
{
    std::set<std::string> actual_names;
    for (const auto& test_case : cases)
    {
        REQUIRE_TRUE(actual_names.insert(test_case.name).second);
    }
    REQUIRE_EQ(actual_names.size(), expected_names.size());
    for (const auto& expected_name : expected_names)
    {
        REQUIRE_TRUE(actual_names.count(expected_name) == 1);
    }
}

template <typename CaseType>
void Insert_Case_Names(std::set<std::string>* names,
                       const std::vector<CaseType>& cases)
{
    REQUIRE_TRUE(names != nullptr);
    for (const auto& test_case : cases)
    {
        REQUIRE_TRUE(names->insert(test_case.name).second);
    }
}

std::vector<NormalSmokeCase> Normal_Smoke_Cases(
    const std::filesystem::path& legacy_source,
    const std::filesystem::path& bundled_source,
    const std::filesystem::path& sidecar_source)
{
    return {
        {"normal_legacy_in_bundled_out", legacy_source, "mdin.spg.toml", true},
        {"normal_bundled_in_legacy_out", bundled_source,
         "mdin.bundled.spg.toml", false},
        {"normal_sidecar_in_legacy_out", sidecar_source,
         "mdin.bundled.spg.toml", false},
        {"normal_bundled_in_bundled_out", bundled_source,
         "mdin.bundled.spg.toml", true},
        {"normal_sidecar_in_bundled_out", sidecar_source,
         "mdin.bundled.spg.toml", true},
    };
}

std::vector<SidecarSmokeCase> Sidecar_Smoke_Cases(
    const std::filesystem::path& bundled_source,
    const std::filesystem::path& sidecar_source)
{
    return {
        {"sidecar_injected_without_explicit_legacy_keys", sidecar_source,
         "mdin.bundled.spg.toml",
         SidecarSmokeKind::injected_without_explicit_keys},
        {"sidecar_same_key_same_path", sidecar_source,
         "mdin.override_same_path.spg.toml",
         SidecarSmokeKind::same_key_same_path},
        {"sidecar_same_key_different_path", sidecar_source,
         "mdin.override_conflict.spg.toml",
         SidecarSmokeKind::same_key_different_path},
        {"pure_bundled_without_sidecar_files", bundled_source,
         "mdin.bundled.spg.toml",
         SidecarSmokeKind::pure_bundled_without_sidecar_files},
    };
}

std::vector<RerunLegacyOutputCase> Rerun_Legacy_Output_Cases(
    const std::filesystem::path& bundled_source,
    const std::filesystem::path& sidecar_source)
{
    return {
        {"rerun_bundled_in_legacy_out", bundled_source,
         "mdin.bundled.spg.toml"},
        {"rerun_sidecar_in_legacy_out", sidecar_source,
         "mdin.bundled.spg.toml"},
    };
}

std::vector<RerunBundledOutputCase> Rerun_Bundled_Output_Cases(
    const std::filesystem::path& legacy_source,
    const std::filesystem::path& bundled_source,
    const std::filesystem::path& sidecar_source)
{
    return {
        {"rerun_legacy_in_bundled_out_vds_off", legacy_source, "mdin.spg.toml",
         false},
        {"rerun_legacy_in_bundled_out_vds_on", legacy_source, "mdin.spg.toml",
         true},
        {"rerun_bundled_in_bundled_out_vds_off", bundled_source,
         "mdin.bundled.spg.toml", false},
        {"rerun_bundled_in_bundled_out_vds_on", bundled_source,
         "mdin.bundled.spg.toml", true},
        {"rerun_sidecar_in_bundled_out_vds_off", sidecar_source,
         "mdin.bundled.spg.toml", false},
        {"rerun_sidecar_in_bundled_out_vds_on", sidecar_source,
         "mdin.bundled.spg.toml", true},
    };
}

std::vector<RerunLegacyOutputCase> Rerun_Selection_Cases(
    const std::filesystem::path& bundled_source,
    const std::filesystem::path& sidecar_source)
{
    (void)bundled_source;
    return {
        {"rerun_sidecar_second_frame_only_legacy_out", sidecar_source,
         "mdin.bundled.spg.toml"},
    };
}

PreparedCase Prepare_Case(const std::filesystem::path& temp_root,
                          const std::string& name,
                          const std::filesystem::path& source_dir,
                          const std::string& source_mdin,
                          const bool bundled_output)
{
    PreparedCase prepared;
    prepared.root = temp_root / name;
    Copy_Directory_Contents(source_dir, prepared.root);
    Install_Valid_EAM_Atom_Types(prepared.root);
    const auto output_paths =
        SpongeH5InputMatrix::Normal_Output_Paths(prepared.root);
    std::filesystem::create_directories(output_paths.output_dir);

    std::string mdin = Read_Text(prepared.root / source_mdin);
    mdin = Remove_Key_Lines(
        mdin, {"output_h5_trajectory_path", "output_h5_trajectory_vds",
               "output_h5_restart_path", "output_h5_restart_topology_hash",
               "output_h5_restart_atom_order_hash",
               "output_h5_restart_protocol_hash", "output_h5_observable_path"});
    Append_If_Missing(&mdin, "mdinfo", "mdinfo = \"mdinfo.txt\"");
    Append_If_Missing(&mdin, "mdout", "mdout = \"mdout.txt\"");
    if (bundled_output)
    {
        mdin = Remove_Key_Lines(mdin, {"write_trajectory_interval"});
        Append_If_Missing(&mdin, "write_trajectory_interval",
                          "write_trajectory_interval = 1");
        Append_If_Missing(
            &mdin, "output_h5_restart_path",
            "output_h5_restart_path = " +
                Toml_Relative_Path(prepared.root, output_paths.h5_restart));
        if (!Has_Key_Line(mdin, "input_h5_topology_path"))
        {
            Append_If_Missing(&mdin, "output_h5_restart_topology_hash",
                              "output_h5_restart_topology_hash = \"" +
                                  Raw_Restart_Topology_Hash() + "\"");
            Append_If_Missing(&mdin, "output_h5_restart_atom_order_hash",
                              "output_h5_restart_atom_order_hash = \"" +
                                  Raw_Restart_Atom_Order_Hash() + "\"");
        }
        Append_If_Missing(
            &mdin, "output_h5_trajectory_path",
            "output_h5_trajectory_path = " +
                Toml_Relative_Path(prepared.root, output_paths.h5_trajectory));
        Append_If_Missing(&mdin, "output_h5_trajectory_vds",
                          "output_h5_trajectory_vds = false");
        Append_If_Missing(
            &mdin, "output_h5_observable_path",
            "output_h5_observable_path = " +
                Toml_Relative_Path(prepared.root, output_paths.h5_observable));
    }

    prepared.mdin = prepared.root / "mdin.smoke.spg.toml";
    prepared.mdout = output_paths.mdout;
    prepared.mdinfo = output_paths.mdinfo;
    prepared.h5_restart = output_paths.h5_restart;
    prepared.h5_trajectory = output_paths.h5_trajectory;
    prepared.h5_observable = output_paths.h5_observable;
    Write_Text(prepared.mdin, mdin);
    return prepared;
}

PreparedCase Prepare_Rerun_Case(const std::filesystem::path& temp_root,
                                const std::string& name,
                                const std::filesystem::path& source_dir,
                                const std::string& source_mdin,
                                const bool bundled_output,
                                const bool vds_output = false,
                                const RerunSelection selection = {})
{
    PreparedCase prepared;
    prepared.root = temp_root / name;
    Copy_Directory_Contents(source_dir, prepared.root);
    Install_Valid_EAM_Atom_Types(prepared.root);
    const auto output_paths =
        SpongeH5InputMatrix::Rerun_Output_Paths(prepared.root);
    std::filesystem::create_directories(output_paths.output_dir);

    std::string mdin = Read_Text(prepared.root / source_mdin);
    mdin = Remove_Key_Lines(
        mdin,
        {"output_h5_trajectory_path", "output_h5_trajectory_vds",
         "output_h5_restart_path", "output_h5_observable_path", "rerun_start",
         "rerun_strip", "rerun_frame_limit", "rerun_need_box_update"});
    Append_If_Missing(&mdin, "mdinfo", "mdinfo = \"mdinfo.txt\"");
    Append_If_Missing(&mdin, "mdout", "mdout = \"mdout.txt\"");
    Append_If_Missing(&mdin, "rerun_start",
                      "rerun_start = " + std::to_string(selection.start));
    Append_If_Missing(&mdin, "rerun_strip",
                      "rerun_strip = " + std::to_string(selection.strip));
    Append_If_Missing(
        &mdin, "rerun_frame_limit",
        "rerun_frame_limit = " + std::to_string(selection.frame_limit));
    Append_If_Missing(&mdin, "rerun_need_box_update",
                      "rerun_need_box_update = 0");
    Append_If_Missing(&mdin, "write_mdout_interval",
                      "write_mdout_interval = 1");
    if (bundled_output)
    {
        Append_If_Missing(&mdin, "write_trajectory_interval",
                          "write_trajectory_interval = 1");
        Append_If_Missing(
            &mdin, "output_h5_trajectory_path",
            "output_h5_trajectory_path = " +
                Toml_Relative_Path(prepared.root, output_paths.h5_trajectory));
        Append_If_Missing(&mdin, "output_h5_trajectory_vds",
                          std::string("output_h5_trajectory_vds = ") +
                              (vds_output ? "true" : "false"));
        Append_If_Missing(
            &mdin, "output_h5_observable_path",
            "output_h5_observable_path = " +
                Toml_Relative_Path(prepared.root, output_paths.h5_observable));
    }

    prepared.mdin = prepared.root / "mdin.rerun.smoke.spg.toml";
    prepared.mdout = output_paths.mdout;
    prepared.mdinfo = output_paths.mdinfo;
    prepared.h5_restart = output_paths.h5_restart;
    prepared.h5_trajectory = output_paths.h5_trajectory;
    prepared.h5_observable = output_paths.h5_observable;
    Write_Text(prepared.mdin, mdin);
    return prepared;
}

void Require_Normal_Prepared_Mdin(const PreparedCase& prepared,
                                  const bool bundled_output)
{
    const auto mdin = Read_Text(prepared.mdin);
    const auto output_paths =
        SpongeH5InputMatrix::Normal_Output_Paths(prepared.root);
    REQUIRE_TRUE(prepared.mdout == output_paths.mdout);
    REQUIRE_TRUE(prepared.mdinfo == output_paths.mdinfo);
    REQUIRE_TRUE(prepared.h5_restart == output_paths.h5_restart);
    REQUIRE_TRUE(prepared.h5_trajectory == output_paths.h5_trajectory);
    REQUIRE_TRUE(prepared.h5_observable == output_paths.h5_observable);
    Require_Contains(mdin, "mdout = \"mdout.txt\"");
    Require_Contains(mdin, "mdinfo = \"mdinfo.txt\"");
    if (bundled_output)
    {
        Require_Contains(mdin, "write_trajectory_interval = 1");
        REQUIRE_TRUE(!Has_Key_Line(mdin, "crd"));
        REQUIRE_TRUE(!Has_Key_Line(mdin, "box"));
        REQUIRE_TRUE(!Has_Key_Line(mdin, "vel"));
        REQUIRE_TRUE(!Has_Key_Line(mdin, "frc"));
        REQUIRE_TRUE(!Has_Key_Line(mdin, "rst"));
        Require_Contains(mdin, "output_h5_restart_path = " +
                                   Toml_Relative_Path(prepared.root,
                                                      output_paths.h5_restart));
        if (!Has_Key_Line(mdin, "input_h5_topology_path"))
        {
            Require_Contains(mdin, "output_h5_restart_topology_hash = \"" +
                                       Raw_Restart_Topology_Hash() + "\"");
            Require_Contains(mdin, "output_h5_restart_atom_order_hash = \"" +
                                       Raw_Restart_Atom_Order_Hash() + "\"");
        }
        Require_Contains(
            mdin,
            "output_h5_trajectory_path = " +
                Toml_Relative_Path(prepared.root, output_paths.h5_trajectory));
        Require_Contains(mdin, "output_h5_trajectory_vds = false");
        Require_Contains(
            mdin,
            "output_h5_observable_path = " +
                Toml_Relative_Path(prepared.root, output_paths.h5_observable));
    }
    else
    {
        REQUIRE_TRUE(!Has_Key_Line(mdin, "output_h5_restart_path"));
        REQUIRE_TRUE(!Has_Key_Line(mdin, "output_h5_trajectory_path"));
        REQUIRE_TRUE(!Has_Key_Line(mdin, "output_h5_trajectory_vds"));
        REQUIRE_TRUE(!Has_Key_Line(mdin, "output_h5_observable_path"));
    }
}

void Require_Rerun_Prepared_Mdin(const PreparedCase& prepared,
                                 const bool bundled_output,
                                 const bool vds_output,
                                 const RerunSelection& selection)
{
    const auto mdin = Read_Text(prepared.mdin);
    const auto output_paths =
        SpongeH5InputMatrix::Rerun_Output_Paths(prepared.root);
    REQUIRE_TRUE(prepared.mdout == output_paths.mdout);
    REQUIRE_TRUE(prepared.mdinfo == output_paths.mdinfo);
    REQUIRE_TRUE(prepared.h5_restart == output_paths.h5_restart);
    REQUIRE_TRUE(prepared.h5_trajectory == output_paths.h5_trajectory);
    REQUIRE_TRUE(prepared.h5_observable == output_paths.h5_observable);
    Require_Contains(mdin, "mdout = \"mdout.txt\"");
    Require_Contains(mdin, "mdinfo = \"mdinfo.txt\"");
    Require_Contains(mdin, "rerun_start = " + std::to_string(selection.start));
    Require_Contains(mdin, "rerun_strip = " + std::to_string(selection.strip));
    Require_Contains(
        mdin, "rerun_frame_limit = " + std::to_string(selection.frame_limit));
    Require_Contains(mdin, "rerun_need_box_update = 0");
    if (bundled_output)
    {
        Require_Contains(mdin, "write_trajectory_interval = 1");
        Require_Contains(
            mdin,
            "output_h5_trajectory_path = " +
                Toml_Relative_Path(prepared.root, output_paths.h5_trajectory));
        Require_Contains(mdin, std::string("output_h5_trajectory_vds = ") +
                                   (vds_output ? "true" : "false"));
        REQUIRE_TRUE(!Has_Key_Line(mdin, "output_h5_restart_path"));
        Require_Contains(
            mdin,
            "output_h5_observable_path = " +
                Toml_Relative_Path(prepared.root, output_paths.h5_observable));
    }
    else
    {
        REQUIRE_TRUE(!Has_Key_Line(mdin, "output_h5_trajectory_path"));
        REQUIRE_TRUE(!Has_Key_Line(mdin, "output_h5_trajectory_vds"));
        REQUIRE_TRUE(!Has_Key_Line(mdin, "output_h5_restart_path"));
        REQUIRE_TRUE(!Has_Key_Line(mdin, "output_h5_observable_path"));
    }
}

void Require_No_H5_Input_Bindings(const std::string& mdin)
{
    static const std::vector<std::string> keys = {
        "input_h5_topology_path", "input_h5_protocol_path",
        "input_h5_restart_path", "input_h5_restart_load",
        "input_h5_trajectory_path"};
    for (const auto& key : keys)
    {
        REQUIRE_TRUE(!Has_Key_Line(mdin, key));
    }
}

void Require_No_Legacy_Restart_Input_Keys(const std::string& mdin)
{
    static const std::vector<std::string> keys = {"coordinate_in_file",
                                                  "velocity_in_file", "rst7"};
    for (const auto& key : keys)
    {
        REQUIRE_TRUE(!Has_Key_Line(mdin, key));
    }
}

void Require_No_Legacy_Rerun_Input_Keys(const std::string& mdin)
{
    static const std::vector<std::string> keys = {"crd", "box", "vel"};
    for (const auto& key : keys)
    {
        REQUIRE_TRUE(!Has_Key_Line(mdin, key));
    }
}

void Require_Normal_Legacy_Input_Mdin(const PreparedCase& prepared)
{
    const auto mdin = Read_Text(prepared.mdin);
    Require_No_H5_Input_Bindings(mdin);
    REQUIRE_TRUE(!Has_Key_Line(mdin, "input_h5_trajectory_particle_stream"));
    REQUIRE_TRUE(Has_Key_Line(mdin, "coordinate_in_file"));
    REQUIRE_TRUE(Has_Key_Line(mdin, "velocity_in_file"));
    REQUIRE_TRUE(Has_Key_Line(mdin, "mass_in_file"));
    REQUIRE_TRUE(Has_Key_Line(mdin, "charge_in_file"));
    REQUIRE_TRUE(Has_Key_Line(mdin, "cv_in_file"));
    REQUIRE_TRUE(Has_Key_Line(mdin, "restrain_in_file"));
}

void Require_Rerun_Legacy_Input_Mdin(const PreparedCase& prepared)
{
    const auto mdin = Read_Text(prepared.mdin);
    Require_No_H5_Input_Bindings(mdin);
    REQUIRE_TRUE(Has_Key_Line(mdin, "input_h5_trajectory_particle_stream"));
    REQUIRE_TRUE(Has_Key_Line(mdin, "coordinate_in_file"));
    REQUIRE_TRUE(Has_Key_Line(mdin, "velocity_in_file"));
    REQUIRE_TRUE(Has_Key_Line(mdin, "mass_in_file"));
    REQUIRE_TRUE(Has_Key_Line(mdin, "charge_in_file"));
    REQUIRE_TRUE(Has_Key_Line(mdin, "cv_in_file"));
    REQUIRE_TRUE(Has_Key_Line(mdin, "restrain_in_file"));
    REQUIRE_TRUE(Has_Key_Line(mdin, "crd"));
    REQUIRE_TRUE(Has_Key_Line(mdin, "box"));
    REQUIRE_TRUE(Has_Key_Line(mdin, "vel"));
}

void Require_Normal_Bundled_Input_Mdin(const PreparedCase& prepared)
{
    const auto mdin = Read_Text(prepared.mdin);
    Require_No_Legacy_Restart_Input_Keys(mdin);
    Require_No_Legacy_Rerun_Input_Keys(mdin);
    REQUIRE_TRUE(Has_Key_Line(mdin, "input_h5_topology_path"));
    REQUIRE_TRUE(Has_Key_Line(mdin, "input_h5_protocol_path"));
    REQUIRE_TRUE(Has_Key_Line(mdin, "input_h5_restart_path"));
    REQUIRE_TRUE(Has_Key_Line(mdin, "input_h5_restart_load"));
    REQUIRE_TRUE(!Has_Key_Line(mdin, "input_h5_trajectory_path"));
    REQUIRE_TRUE(!Has_Key_Line(mdin, "input_h5_trajectory_particle_stream"));
}

void Require_Rerun_Bundled_Input_Mdin(const PreparedCase& prepared)
{
    const auto mdin = Read_Text(prepared.mdin);
    Require_No_Legacy_Restart_Input_Keys(mdin);
    Require_No_Legacy_Rerun_Input_Keys(mdin);
    REQUIRE_TRUE(Has_Key_Line(mdin, "input_h5_topology_path"));
    REQUIRE_TRUE(Has_Key_Line(mdin, "input_h5_protocol_path"));
    REQUIRE_TRUE(Has_Key_Line(mdin, "input_h5_restart_path"));
    REQUIRE_TRUE(Has_Key_Line(mdin, "input_h5_restart_load"));
    REQUIRE_TRUE(Has_Key_Line(mdin, "input_h5_trajectory_path"));
    REQUIRE_TRUE(Has_Key_Line(mdin, "input_h5_trajectory_particle_stream"));
}

void Run_SPONGE(const std::filesystem::path& executable,
                const PreparedCase& test_case)
{
    const auto log_path = test_case.root / "sponge.stdout.txt";
    const std::string command = Shell_Quote(executable) + " -mdin " +
                                Shell_Quote(test_case.mdin) + " > " +
                                Shell_Quote(log_path) + " 2>&1";
    const int ret = Run_Test_Shell_Command(command);
    if (ret != 0)
    {
        throw TestFailure("SPONGE smoke failed for " +
                          test_case.root.filename().string() + "\n" +
                          Read_Text(log_path));
    }
    SpongeH5InputMatrix::Require_Path_Exists(test_case.mdout);
    SpongeH5InputMatrix::Require_Path_Exists(test_case.mdinfo);
}

void Require_No_Default_Legacy_Trajectory_Or_Restart_Outputs(
    const PreparedCase& test_case)
{
    static const std::vector<std::string> forbidden = {"mdcrd.dat", "mdbox.txt",
                                                       "restart"};
    for (const auto& filename : forbidden)
    {
        REQUIRE_TRUE(!std::filesystem::exists(test_case.root / filename));
    }
}

void Require_Normal_Legacy_Restart_Output(const PreparedCase& test_case)
{
    SpongeH5InputMatrix::Require_Path_Exists(test_case.root /
                                             "restart_coordinate.txt");
    SpongeH5InputMatrix::Require_Path_Exists(test_case.root /
                                             "restart_velocity.txt");
}

void Run_SPONGE_Expect_Failure(const std::filesystem::path& executable,
                               const PreparedCase& test_case,
                               const std::string& expected_error)
{
    const auto log_path = test_case.root / "sponge.stdout.txt";
    const std::string command = Shell_Quote(executable) + " -mdin " +
                                Shell_Quote(test_case.mdin) + " > " +
                                Shell_Quote(log_path) + " 2>&1";
    const int ret = Run_Test_Shell_Command(command);
    if (ret == 0)
    {
        throw TestFailure("SPONGE smoke unexpectedly succeeded for " +
                          test_case.root.filename().string());
    }
    Require_Contains(Read_Text(log_path), expected_error);
}

std::string Legacy_Sidecar_Path_For_Key(const std::filesystem::path& h5_path,
                                        const std::string& key)
{
    std::vector<SpongeH5MD::LegacySidecarBinding> sidecars;
    std::string error;
    REQUIRE_TRUE(SpongeH5MD::Read_Legacy_Sidecars_From_H5(h5_path.string(),
                                                          &sidecars, &error));
    for (const auto& sidecar : sidecars)
    {
        if (sidecar.key == key)
        {
            return sidecar.path;
        }
    }
    throw TestFailure("missing legacy sidecar key: " + key);
}

void Require_H5_Legacy_Sidecar_Keys_Absent(
    const std::filesystem::path& h5_path,
    const std::set<std::string>& absent_keys)
{
    std::vector<SpongeH5MD::LegacySidecarBinding> sidecars;
    std::string error;
    REQUIRE_TRUE(SpongeH5MD::Read_Legacy_Sidecars_From_H5(h5_path.string(),
                                                          &sidecars, &error));
    for (const auto& sidecar : sidecars)
    {
        REQUIRE_TRUE(absent_keys.count(sidecar.key) == 0);
    }
}

void Require_H5_Legacy_Sidecar_Keys_Present(
    const std::filesystem::path& h5_path,
    const std::set<std::string>& expected_keys)
{
    std::vector<SpongeH5MD::LegacySidecarBinding> sidecars;
    std::string error;
    REQUIRE_TRUE(SpongeH5MD::Read_Legacy_Sidecars_From_H5(h5_path.string(),
                                                          &sidecars, &error));
    std::set<std::string> actual_keys;
    for (const auto& sidecar : sidecars)
    {
        actual_keys.insert(sidecar.key);
    }
    for (const auto& key : expected_keys)
    {
        REQUIRE_TRUE(actual_keys.count(key) == 1);
    }
}

std::string Relative_Materialized_Sidecar_Path(
    const std::filesystem::path& sidecar_root,
    const std::filesystem::path& path)
{
    return path.lexically_relative(sidecar_root).generic_string();
}

std::set<std::string> H5_Referenced_Sidecar_File_Set(
    const std::filesystem::path& bundle_root)
{
    const auto sidecar_root = bundle_root / "legacy_sidecars";
    REQUIRE_TRUE(std::filesystem::is_directory(sidecar_root));

    std::set<std::string> referenced;
    for (const auto& h5_name :
         {"topology.spgt.h5", "protocol.spgp.h5", "restart.spgr.h5"})
    {
        std::vector<SpongeH5MD::LegacySidecarBinding> sidecars;
        std::string error;
        REQUIRE_TRUE(SpongeH5MD::Read_Legacy_Sidecars_From_H5(
            (bundle_root / h5_name).string(), &sidecars, &error));
        for (const auto& sidecar : sidecars)
        {
            const std::filesystem::path sidecar_path(sidecar.path);
            REQUIRE_TRUE(sidecar_path.is_absolute());
            REQUIRE_TRUE(std::filesystem::exists(sidecar_path));
            REQUIRE_TRUE(sidecar_path.parent_path().filename() == sidecar.key);
            referenced.insert(
                Relative_Materialized_Sidecar_Path(sidecar_root, sidecar_path));
        }
    }
    REQUIRE_TRUE(!referenced.empty());
    return referenced;
}

std::set<std::string> Materialized_Sidecar_File_Set(
    const std::filesystem::path& bundle_root)
{
    const auto sidecar_root = bundle_root / "legacy_sidecars";
    REQUIRE_TRUE(std::filesystem::is_directory(sidecar_root));

    std::set<std::string> materialized;
    for (const auto& entry :
         std::filesystem::recursive_directory_iterator(sidecar_root))
    {
        if (entry.is_regular_file())
        {
            REQUIRE_TRUE(materialized
                             .insert(Relative_Materialized_Sidecar_Path(
                                 sidecar_root, entry.path()))
                             .second);
        }
    }
    REQUIRE_TRUE(!materialized.empty());
    return materialized;
}

void Require_Materialized_Sidecars_Are_Exactly_H5_Referenced(
    const std::filesystem::path& bundle_root)
{
    REQUIRE_TRUE(Materialized_Sidecar_File_Set(bundle_root) ==
                 H5_Referenced_Sidecar_File_Set(bundle_root));
}

void Remove_Legacy_Sidecar_Directories(const std::filesystem::path& root)
{
    std::vector<std::filesystem::path> sidecar_dirs;
    for (const auto& entry :
         std::filesystem::recursive_directory_iterator(root))
    {
        if (entry.is_directory() &&
            entry.path().filename() == "legacy_sidecars")
        {
            sidecar_dirs.push_back(entry.path());
        }
    }
    for (const auto& path : sidecar_dirs)
    {
        std::filesystem::remove_all(path);
    }
}

void Remove_H5_Legacy_Sidecar_Keys(const std::filesystem::path& h5_path,
                                   const std::set<std::string>& keys_to_remove)
{
    HighFive::File file(h5_path.string(), HighFive::File::ReadWrite);
    if (!file.exist(SpongeH5MD::path::legacy_sidecar_keys) &&
        !file.exist(SpongeH5MD::path::legacy_sidecar_paths))
    {
        return;
    }
    REQUIRE_TRUE(file.exist(SpongeH5MD::path::legacy_sidecar_keys));
    REQUIRE_TRUE(file.exist(SpongeH5MD::path::legacy_sidecar_paths));

    std::vector<std::string> keys;
    std::vector<std::string> paths;
    file.getDataSet(SpongeH5MD::path::legacy_sidecar_keys).read(keys);
    file.getDataSet(SpongeH5MD::path::legacy_sidecar_paths).read(paths);
    REQUIRE_EQ(keys.size(), paths.size());

    std::vector<std::string> filtered_keys;
    std::vector<std::string> filtered_paths;
    for (std::size_t i = 0; i < keys.size(); ++i)
    {
        if (keys_to_remove.count(keys[i]) == 0)
        {
            filtered_keys.push_back(keys[i]);
            filtered_paths.push_back(paths[i]);
        }
    }
    if (filtered_keys.size() == keys.size())
    {
        return;
    }

    H5Ldelete(file.getId(), SpongeH5MD::path::legacy_sidecar_keys, H5P_DEFAULT);
    H5Ldelete(file.getId(), SpongeH5MD::path::legacy_sidecar_paths,
              H5P_DEFAULT);
    file.createDataSet<std::string>(SpongeH5MD::path::legacy_sidecar_keys,
                                    HighFive::DataSpace({filtered_keys.size()}))
        .write(filtered_keys);
    file.createDataSet<std::string>(
            SpongeH5MD::path::legacy_sidecar_paths,
            HighFive::DataSpace({filtered_paths.size()}))
        .write(filtered_paths);
}

void Delete_H5_Object_If_Exists(HighFive::File* file, const char* path)
{
    REQUIRE_TRUE(file != nullptr);
    if (file->exist(path))
    {
        H5Ldelete(file->getId(), path, H5P_DEFAULT);
    }
}

void Scrub_Runtime_Unstable_Rerun_Features(const PreparedCase& test_case)
{
    std::string mdin = Read_Text(test_case.mdin);
    const bool had_h5_trajectory_output =
        Has_Key_Line(mdin, "output_h5_trajectory_path");
    const bool had_h5_observable_output =
        Has_Key_Line(mdin, "output_h5_observable_path");
    const bool had_vds_enabled =
        mdin.find("output_h5_trajectory_vds = true") != std::string::npos;
    if (Has_Key_Line(mdin, "input_h5_restart_path"))
    {
        mdin = Remove_Key_Lines(mdin, {"input_h5_restart_load"});
        Append_If_Missing(&mdin, "input_h5_restart_load",
                          "input_h5_restart_load = \"structural\"");
    }
    if (had_h5_trajectory_output)
    {
        mdin = Remove_Key_Lines(
            mdin, {"output_h5_trajectory_path", "output_h5_trajectory_vds",
                   "output_h5_observable_path", "write_trajectory_interval"});
        Append_If_Missing(&mdin, "write_trajectory_interval",
                          "write_trajectory_interval = 1");
        Append_If_Missing(
            &mdin, "output_h5_trajectory_path",
            "output_h5_trajectory_path = " +
                Toml_Relative_Path(test_case.root, test_case.h5_trajectory));
        Append_If_Missing(&mdin, "output_h5_trajectory_vds",
                          std::string("output_h5_trajectory_vds = ") +
                              (had_vds_enabled ? "true" : "false"));
        if (had_h5_observable_output)
        {
            Append_If_Missing(&mdin, "output_h5_observable_path",
                              "output_h5_observable_path = " +
                                  Toml_Relative_Path(test_case.root,
                                                     test_case.h5_observable));
        }
    }
    Write_Text(test_case.mdin, mdin);
}

void Require_Runtime_Unstable_Rerun_Features_Scrubbed(
    const PreparedCase& test_case)
{
    const auto mdin = Read_Text(test_case.mdin);
    Require_Contains(mdin, "input_h5_restart_load = \"structural\"");
}

void Require_No_Legacy_Sidecar_Directories(const std::filesystem::path& root)
{
    for (const auto& entry :
         std::filesystem::recursive_directory_iterator(root))
    {
        REQUIRE_TRUE(entry.path().filename() != "legacy_sidecars");
    }
}

void Test_Mdin_Key_Line_Helper()
{
    const std::string mdin =
        "output_h5_trajectory_path_extra = \"keep\"\n"
        "  output_h5_trajectory_path = \"remove\" # comment\n"
        "# output_h5_trajectory_path = \"commented\"\n"
        "other_output_h5_trajectory_path = \"keep\"\n";
    REQUIRE_TRUE(Has_Key_Line(mdin, "output_h5_trajectory_path"));
    REQUIRE_TRUE(
        !Has_Key_Line(mdin, "output_h5_trajectory_path_extra_missing"));

    const auto stripped = Remove_Key_Lines(mdin, {"output_h5_trajectory_path"});
    REQUIRE_TRUE(!Has_Key_Line(stripped, "output_h5_trajectory_path"));
    REQUIRE_TRUE(Has_Key_Line(stripped, "output_h5_trajectory_path_extra"));
    REQUIRE_TRUE(Has_Key_Line(stripped, "other_output_h5_trajectory_path"));
}

void Require_Runtime_Smoke_Enabled()
{
    const char* enabled = std::getenv("SPONGE_H5_ENABLE_RUNTIME_SMOKE");
    if (enabled == nullptr || std::string(enabled) != "1")
    {
        std::cerr << "Skipping runtime smoke matrix; set "
                     "SPONGE_H5_ENABLE_RUNTIME_SMOKE=1 in a runnable SPONGE "
                     "CPU/GPU environment to enable it.\n";
        std::exit(kSkipReturnCode);
    }
}

void Validate_Runtime_Smoke_Preparation()
{
    Test_Mdin_Key_Line_Helper();

    const auto temp_root =
        SpongeH5Test::Unique_Temp_Path("h5_runtime_smoke_prepare_check");
    std::filesystem::create_directories(temp_root);

    const auto core = SpongeH5InputMatrix::Core_Structural_Path();
    const auto core_legacy_source = core / "legacy_input";
    const auto core_bundled_source = core / "bundled_input" / "bundle";
    const auto core_sidecar_source =
        core / "bundled_input_with_legacy_sidecar" / "bundle";
    const auto normal_cases = Normal_Smoke_Cases(
        core_legacy_source, core_bundled_source, core_sidecar_source);
    REQUIRE_EQ(normal_cases.size(), static_cast<std::size_t>(5));
    Require_Case_Name_Set(
        normal_cases,
        {"normal_legacy_in_bundled_out", "normal_bundled_in_legacy_out",
         "normal_sidecar_in_legacy_out", "normal_bundled_in_bundled_out",
         "normal_sidecar_in_bundled_out"});
    const auto sidecar_cases =
        Sidecar_Smoke_Cases(core_bundled_source, core_sidecar_source);
    REQUIRE_EQ(sidecar_cases.size(), static_cast<std::size_t>(4));
    Require_Case_Name_Set(
        sidecar_cases,
        {"sidecar_injected_without_explicit_legacy_keys",
         "sidecar_same_key_same_path", "sidecar_same_key_different_path",
         "pure_bundled_without_sidecar_files"});
    REQUIRE_TRUE(std::string(sidecar_cases[2].mdin_name) ==
                 "mdin.override_conflict.spg.toml");
    REQUIRE_TRUE(std::string(sidecar_cases[1].mdin_name) ==
                 "mdin.override_same_path.spg.toml");
    SpongeH5InputMatrix::Require_Path_Exists(core_sidecar_source /
                                             "mdin.override_conflict.spg.toml");
    SpongeH5InputMatrix::Require_Path_Exists(
        core_sidecar_source / "mdin.override_same_path.spg.toml");

    for (const auto& spec : normal_cases)
    {
        const auto prepared =
            Prepare_Case(temp_root, std::string("prepare_") + spec.name,
                         spec.source_dir, spec.mdin_name, spec.bundled_output);
        Require_Normal_Prepared_Mdin(prepared, spec.bundled_output);
        if (Starts_With(std::string(spec.name), "normal_legacy_in_"))
        {
            Require_Normal_Legacy_Input_Mdin(prepared);
        }
        else
        {
            Require_Normal_Bundled_Input_Mdin(prepared);
        }
    }

    for (const auto& spec : sidecar_cases)
    {
        const auto prepared =
            Prepare_Case(temp_root, std::string("prepare_") + spec.name,
                         spec.source_dir, spec.mdin_name, false);
        Require_Normal_Prepared_Mdin(prepared, false);
        Require_Normal_Bundled_Input_Mdin(prepared);
        if (spec.kind != SidecarSmokeKind::pure_bundled_without_sidecar_files)
        {
            Require_Materialized_Sidecars_Are_Exactly_H5_Referenced(
                prepared.root);
        }
        if (spec.kind == SidecarSmokeKind::injected_without_explicit_keys)
        {
            const auto mdin = Read_Text(prepared.mdin);
            for (const auto& key :
                 {"mass_in_file", "charge_in_file",
                  "cv_in_file", "restrain_in_file", "SITS_in_file"})
            {
                REQUIRE_TRUE(!Has_Key_Line(mdin, key));
            }
            for (const auto& key :
                 {"mass_in_file", "charge_in_file"})
            {
                const auto sidecar_path = Legacy_Sidecar_Path_For_Key(
                    prepared.root / "topology.spgt.h5", key);
                REQUIRE_TRUE(std::filesystem::exists(sidecar_path));
            }
            for (const auto& key :
                 {"cv_in_file", "restrain_in_file", "SITS_in_file"})
            {
                const auto sidecar_path = Legacy_Sidecar_Path_For_Key(
                    prepared.root / "protocol.spgp.h5", key);
                REQUIRE_TRUE(std::filesystem::exists(sidecar_path));
            }
        }
        if (spec.kind == SidecarSmokeKind::same_key_different_path)
        {
            Require_Contains(Read_Text(prepared.mdin),
                             "mass_in_file = \"override_mass.txt\"");
            SpongeH5InputMatrix::Require_Path_Exists(prepared.root /
                                                     "override_mass.txt");
            const auto mass_sidecar_path = Legacy_Sidecar_Path_For_Key(
                prepared.root / "topology.spgt.h5", "mass_in_file");
            REQUIRE_TRUE(std::filesystem::exists(mass_sidecar_path));
            HighFive::File topology(
                (prepared.root / "topology.spgt.h5").string(),
                HighFive::File::ReadOnly);
            REQUIRE_TRUE(topology.exist("/atoms/mass"));
        }
        if (spec.kind == SidecarSmokeKind::same_key_same_path)
        {
            Require_Contains(Read_Text(prepared.mdin),
                             "SITS_in_file = "
                             "\"legacy_sidecars/SITS_in_file/sits.txt\"");
            const auto sits_sidecar_path = Legacy_Sidecar_Path_For_Key(
                prepared.root / "protocol.spgp.h5", "SITS_in_file");
            REQUIRE_TRUE(std::filesystem::exists(sits_sidecar_path));
        }
        if (spec.kind == SidecarSmokeKind::pure_bundled_without_sidecar_files)
        {
            Remove_Legacy_Sidecar_Directories(prepared.root);
            Require_No_Legacy_Sidecar_Directories(prepared.root);
        }
    }

    const auto normal_baseline =
        Prepare_Case(temp_root, "normal_legacy_in_legacy_out",
                     core_legacy_source, "mdin.spg.toml", false);
    Require_Normal_Prepared_Mdin(normal_baseline, false);
    Require_Normal_Legacy_Input_Mdin(normal_baseline);

    const auto normal_bundled = Prepare_Case(
        temp_root, "normal_bundled_prepare_check",
        core / "bundled_input" / "bundle", "mdin.bundled.spg.toml", true);
    Require_Normal_Prepared_Mdin(normal_bundled, true);
    Require_Normal_Bundled_Input_Mdin(normal_bundled);

    const auto full = SpongeH5InputMatrix::Full_Contract_Rerun_Path();
    const auto full_legacy_source = full / "legacy_input";
    const auto full_bundled_source = full / "bundled_input" / "bundle";
    const auto full_sidecar_source =
        full / "bundled_input_with_legacy_sidecar" / "bundle";
    const auto rerun_legacy_output_cases =
        Rerun_Legacy_Output_Cases(full_bundled_source, full_sidecar_source);
    REQUIRE_EQ(rerun_legacy_output_cases.size(), static_cast<std::size_t>(2));
    Require_Case_Name_Set(
        rerun_legacy_output_cases,
        {"rerun_bundled_in_legacy_out", "rerun_sidecar_in_legacy_out"});
    const auto rerun_bundled_output_cases = Rerun_Bundled_Output_Cases(
        full_legacy_source, full_bundled_source, full_sidecar_source);
    REQUIRE_EQ(rerun_bundled_output_cases.size(), static_cast<std::size_t>(6));
    Require_Case_Name_Set(rerun_bundled_output_cases,
                          {"rerun_legacy_in_bundled_out_vds_off",
                           "rerun_legacy_in_bundled_out_vds_on",
                           "rerun_bundled_in_bundled_out_vds_off",
                           "rerun_bundled_in_bundled_out_vds_on",
                           "rerun_sidecar_in_bundled_out_vds_off",
                           "rerun_sidecar_in_bundled_out_vds_on"});
    const auto rerun_selection_cases =
        Rerun_Selection_Cases(full_bundled_source, full_sidecar_source);
    REQUIRE_EQ(rerun_selection_cases.size(), static_cast<std::size_t>(1));
    Require_Case_Name_Set(rerun_selection_cases,
                          {"rerun_sidecar_second_frame_only_legacy_out"});

    std::set<std::string> planned_runtime_case_names = {
        "normal_legacy_in_legacy_out", "rerun_legacy_in_legacy_out",
        "rerun_legacy_second_frame_only_legacy_out"};
    Insert_Case_Names(&planned_runtime_case_names, normal_cases);
    Insert_Case_Names(&planned_runtime_case_names, rerun_legacy_output_cases);
    Insert_Case_Names(&planned_runtime_case_names, rerun_bundled_output_cases);
    Insert_Case_Names(&planned_runtime_case_names, rerun_selection_cases);
    const std::set<std::string> expected_runtime_case_names = {
        "normal_legacy_in_legacy_out",
        "normal_legacy_in_bundled_out",
        "normal_bundled_in_legacy_out",
        "normal_sidecar_in_legacy_out",
        "normal_bundled_in_bundled_out",
        "normal_sidecar_in_bundled_out",
        "rerun_legacy_in_legacy_out",
        "rerun_bundled_in_legacy_out",
        "rerun_legacy_in_bundled_out_vds_off",
        "rerun_legacy_in_bundled_out_vds_on",
        "rerun_bundled_in_bundled_out_vds_off",
        "rerun_bundled_in_bundled_out_vds_on",
        "rerun_sidecar_in_bundled_out_vds_off",
        "rerun_sidecar_in_bundled_out_vds_on",
        "rerun_sidecar_in_legacy_out",
        "rerun_legacy_second_frame_only_legacy_out",
        "rerun_sidecar_second_frame_only_legacy_out"};
    REQUIRE_TRUE(planned_runtime_case_names == expected_runtime_case_names);

    for (const auto& spec : rerun_legacy_output_cases)
    {
        const auto prepared =
            Prepare_Rerun_Case(temp_root, std::string("prepare_") + spec.name,
                               spec.source_dir, spec.mdin_name, false);
        Require_Rerun_Prepared_Mdin(prepared, false, false, {});
        Require_Rerun_Bundled_Input_Mdin(prepared);
    }

    for (const auto& spec : rerun_bundled_output_cases)
    {
        const auto prepared =
            Prepare_Rerun_Case(temp_root, std::string("prepare_") + spec.name,
                               spec.source_dir, spec.mdin_name, true, spec.vds);
        Require_Rerun_Prepared_Mdin(prepared, true, spec.vds, {});
        if (Starts_With(std::string(spec.name), "rerun_legacy_"))
        {
            Require_Rerun_Legacy_Input_Mdin(prepared);
        }
        else
        {
            Require_Rerun_Bundled_Input_Mdin(prepared);
        }
    }

    const auto rerun_baseline =
        Prepare_Rerun_Case(temp_root, "rerun_legacy_in_legacy_out",
                           full_legacy_source, "mdin.spg.toml", false);
    Require_Rerun_Prepared_Mdin(rerun_baseline, false, false, {});
    Require_Rerun_Legacy_Input_Mdin(rerun_baseline);

    const auto rerun_bundled = Prepare_Rerun_Case(
        temp_root, "rerun_bundled_prepare_check",
        full / "bundled_input" / "bundle", "mdin.bundled.spg.toml", true, true);
    Require_Rerun_Prepared_Mdin(rerun_bundled, true, true, {});
    Require_Rerun_Bundled_Input_Mdin(rerun_bundled);

    auto rerun_scrub_check = Prepare_Rerun_Case(
        temp_root, "rerun_runtime_scrub_prepare_check",
        full / "bundled_input_with_legacy_sidecar" / "bundle",
        "mdin.bundled.spg.toml", true, true);
    Require_Rerun_Prepared_Mdin(rerun_scrub_check, true, true, {});
    Require_Rerun_Bundled_Input_Mdin(rerun_scrub_check);
    Require_H5_Legacy_Sidecar_Keys_Present(
        rerun_scrub_check.root / "topology.spgt.h5",
        {"EDIP_in_file", "REAXFF_in_file", "REAXFF_type_in_file"});
    Scrub_Runtime_Unstable_Rerun_Features(rerun_scrub_check);
    Require_Runtime_Unstable_Rerun_Features_Scrubbed(rerun_scrub_check);

    const RerunSelection second_frame_only = {1, 0, 1};
    for (const auto& spec : rerun_selection_cases)
    {
        const auto prepared = Prepare_Rerun_Case(
            temp_root, std::string("prepare_") + spec.name, spec.source_dir,
            spec.mdin_name, false, false, second_frame_only);
        Require_Rerun_Prepared_Mdin(prepared, false, false, second_frame_only);
        Require_Rerun_Bundled_Input_Mdin(prepared);
    }

    const auto selected_rerun = Prepare_Rerun_Case(
        temp_root, "rerun_second_frame_prepare_check",
        full / "bundled_input_with_legacy_sidecar" / "bundle",
        "mdin.bundled.spg.toml", false, false, second_frame_only);
    Require_Rerun_Prepared_Mdin(selected_rerun, false, false,
                                second_frame_only);
    Require_Rerun_Bundled_Input_Mdin(selected_rerun);

    std::filesystem::remove_all(temp_root);
}

void Run_Legacy_Sidecar_Smoke_Cases(
    const std::filesystem::path& sponge_executable,
    const std::filesystem::path& temp_root,
    const std::filesystem::path& baseline_mdout,
    const std::filesystem::path& bundled_source,
    const std::filesystem::path& sidecar_source)
{
    for (const auto& spec : Sidecar_Smoke_Cases(bundled_source, sidecar_source))
    {
        auto test_case = Prepare_Case(temp_root, spec.name, spec.source_dir,
                                      spec.mdin_name, false);
        switch (spec.kind)
        {
            case SidecarSmokeKind::injected_without_explicit_keys:
                Run_SPONGE(sponge_executable, test_case);
                Require_Normal_Legacy_Restart_Output(test_case);
                Require_Text_Equivalent(baseline_mdout, test_case.mdout);
                break;
            case SidecarSmokeKind::same_key_same_path:
                Run_SPONGE(sponge_executable, test_case);
                Require_Normal_Legacy_Restart_Output(test_case);
                Require_Text_Equivalent(baseline_mdout, test_case.mdout);
                break;
            case SidecarSmokeKind::same_key_different_path:
                Run_SPONGE_Expect_Failure(
                    sponge_executable, test_case,
                    "mass_in_file is also set. Native H5 topology data and "
                    "legacy text topology input cannot both own atom masses");
                break;
            case SidecarSmokeKind::pure_bundled_without_sidecar_files:
                Remove_Legacy_Sidecar_Directories(test_case.root);
                Run_SPONGE(sponge_executable, test_case);
                Require_Normal_Legacy_Restart_Output(test_case);
                Require_Core_Mdout_Equivalent(baseline_mdout, test_case.mdout);
                break;
        }
    }
}

void Run_Normal_Mode_Matrix(const std::filesystem::path& sponge_executable)
{
    const auto temp_root =
        SpongeH5Test::Unique_Temp_Path("h5_input_output_smoke_matrix");
    std::filesystem::create_directories(temp_root);

    const auto core = SpongeH5InputMatrix::Core_Structural_Path();
    const auto legacy_source = core / "legacy_input";
    const auto bundled_source = core / "bundled_input" / "bundle";
    const auto sidecar_source =
        core / "bundled_input_with_legacy_sidecar" / "bundle";

    const auto baseline = Prepare_Case(temp_root, "normal_legacy_in_legacy_out",
                                       legacy_source, "mdin.spg.toml", false);
    Run_SPONGE(sponge_executable, baseline);
    Require_Normal_Legacy_Restart_Output(baseline);

    const auto cases =
        Normal_Smoke_Cases(legacy_source, bundled_source, sidecar_source);
    auto missing_lineage =
        Prepare_Case(temp_root, "normal_legacy_in_bundled_out_missing_lineage",
                     legacy_source, "mdin.spg.toml", true);
    auto missing_lineage_mdin = Read_Text(missing_lineage.mdin);
    missing_lineage_mdin = Remove_Key_Lines(
        missing_lineage_mdin, {"output_h5_restart_topology_hash",
                               "output_h5_restart_atom_order_hash"});
    Write_Text(missing_lineage.mdin, missing_lineage_mdin);
    Run_SPONGE_Expect_Failure(
        sponge_executable, missing_lineage,
        "H5 restart output requires canonical topology_hash and "
        "atom_order_hash lineage");
    for (const auto& spec : cases)
    {
        const auto test_case =
            Prepare_Case(temp_root, spec.name, spec.source_dir, spec.mdin_name,
                         spec.bundled_output);
        Run_SPONGE(sponge_executable, test_case);
        if (Starts_With(std::string(spec.name), "normal_bundled_in_"))
        {
            Require_Core_Mdout_Equivalent(baseline.mdout, test_case.mdout);
        }
        else
        {
            Require_Text_Equivalent(baseline.mdout, test_case.mdout);
        }
        if (spec.bundled_output)
        {
            Require_No_Default_Legacy_Trajectory_Or_Restart_Outputs(test_case);
            SpongeH5InputMatrix::Require_Path_Exists(test_case.h5_restart);
            Require_H5_Restart_Matches_Core_State(test_case.h5_restart);
            SpongeH5InputMatrix::Require_Path_Exists(test_case.h5_trajectory);
            Require_H5_Trajectory_Has_Frames(test_case.h5_trajectory, {1},
                                             {0.0});
            Require_H5_Trajectory_First_Frame_Matches_Core_State(
                test_case.h5_trajectory, 1);
            Require_H5_Observable_Stream_Matches_Mdout(
                test_case.h5_trajectory, test_case.mdout, {0, 1}, {0.0, 0.0});
            SpongeH5InputMatrix::Require_Path_Exists(test_case.h5_observable);
            Require_H5_Observable_Stream_Matches_Mdout(
                test_case.h5_observable, test_case.mdout, {0, 1}, {0.0, 0.0});
        }
        else
        {
            Require_Normal_Legacy_Restart_Output(test_case);
        }
    }

    Run_Legacy_Sidecar_Smoke_Cases(sponge_executable, temp_root, baseline.mdout,
                                   bundled_source, sidecar_source);

    std::filesystem::remove_all(temp_root);
}

void Run_Rerun_Frame_Selection_Smoke_Cases(
    const std::filesystem::path& sponge_executable,
    const std::filesystem::path& temp_root,
    const std::filesystem::path& legacy_source,
    const std::filesystem::path& bundled_source,
    const std::filesystem::path& sidecar_source)
{
    const RerunSelection second_frame_only = {1, 0, 1};
    const auto baseline = Prepare_Rerun_Case(
        temp_root, "rerun_legacy_second_frame_only_legacy_out", legacy_source,
        "mdin.spg.toml", false, false, second_frame_only);
    Scrub_Runtime_Unstable_Rerun_Features(baseline);
    Run_SPONGE(sponge_executable, baseline);
    Require_Mdout_Row_Count(baseline.mdout, 1);

    const auto cases = Rerun_Selection_Cases(bundled_source, sidecar_source);
    for (const auto& spec : cases)
    {
        const auto test_case =
            Prepare_Rerun_Case(temp_root, spec.name, spec.source_dir,
                               spec.mdin_name, false, false, second_frame_only);
        Scrub_Runtime_Unstable_Rerun_Features(test_case);
        Run_SPONGE(sponge_executable, test_case);
        Require_Mdout_Row_Count(test_case.mdout, 1);
        Require_Rerun_Selection_Mdout_Equivalent(baseline.mdout,
                                                 test_case.mdout);
    }
}

void Run_Rerun_Mode_Matrix(const std::filesystem::path& sponge_executable)
{
    const auto temp_root =
        SpongeH5Test::Unique_Temp_Path("h5_rerun_smoke_matrix");
    std::filesystem::create_directories(temp_root);

    const auto full = SpongeH5InputMatrix::Full_Contract_Rerun_Path();
    const auto legacy_source = full / "legacy_input";
    const auto bundled_source = full / "bundled_input" / "bundle";
    const auto sidecar_source =
        full / "bundled_input_with_legacy_sidecar" / "bundle";

    const auto baseline =
        Prepare_Rerun_Case(temp_root, "rerun_legacy_in_legacy_out",
                           legacy_source, "mdin.spg.toml", false);
    Scrub_Runtime_Unstable_Rerun_Features(baseline);
    Run_SPONGE(sponge_executable, baseline);

    const auto legacy_output_cases =
        Rerun_Legacy_Output_Cases(bundled_source, sidecar_source);
    for (const auto& spec : legacy_output_cases)
    {
        const auto test_case = Prepare_Rerun_Case(
            temp_root, spec.name, spec.source_dir, spec.mdin_name, false);
        Scrub_Runtime_Unstable_Rerun_Features(test_case);
        Run_SPONGE(sponge_executable, test_case);
        if (Starts_With(std::string(spec.name), "rerun_bundled_"))
        {
            Require_Pure_Bundled_Rerun_Mdout_Core_Equivalent(baseline.mdout,
                                                             test_case.mdout);
        }
        else
        {
            Require_Rerun_Mdout_Equivalent(baseline.mdout, test_case.mdout);
        }
    }

    const auto bundled_output_cases = Rerun_Bundled_Output_Cases(
        legacy_source, bundled_source, sidecar_source);
    for (const auto& spec : bundled_output_cases)
    {
        const auto sidecar_bundled =
            Prepare_Rerun_Case(temp_root, spec.name, spec.source_dir,
                               spec.mdin_name, true, spec.vds);
        Scrub_Runtime_Unstable_Rerun_Features(sidecar_bundled);
        Run_SPONGE(sponge_executable, sidecar_bundled);
        Require_No_Default_Legacy_Trajectory_Or_Restart_Outputs(
            sidecar_bundled);
        const bool pure_bundled_input =
            Starts_With(std::string(spec.name), "rerun_bundled_");
        if (pure_bundled_input)
        {
            Require_Pure_Bundled_Rerun_Mdout_Core_Equivalent(
                baseline.mdout, sidecar_bundled.mdout);
        }
        else
        {
            Require_Rerun_Mdout_Equivalent(baseline.mdout,
                                           sidecar_bundled.mdout);
        }
        const bool h5_rerun_input =
            pure_bundled_input ||
            Starts_With(std::string(spec.name), "rerun_sidecar_");
        const std::vector<double> trajectory_times =
            h5_rerun_input ? std::vector<double>{1.0}
                           : std::vector<double>{0.0};
        const std::vector<double> observable_times =
            h5_rerun_input ? std::vector<double>{1.0, 1.001}
                           : std::vector<double>{0.0, 0.001};
        SpongeH5InputMatrix::Require_Path_Exists(sidecar_bundled.h5_trajectory);
        Require_H5_Trajectory_Has_Frames(sidecar_bundled.h5_trajectory, {1},
                                         trajectory_times);
        Require_H5_Trajectory_Frame_Matches_Rerun_Runtime_State(
            sidecar_bundled.h5_trajectory, 0);
        if (spec.vds)
        {
            // Step 0 observables open the shard before particle step 1.
            Require_VDS_Shards_Are_Complete(sidecar_bundled.h5_trajectory, {1},
                                            trajectory_times, {0},
                                            {observable_times.front()});
        }
        // VDS and single-file outputs both retain the initial observable frame,
        // including its values, even though no particle frame exists at step 0.
        Require_H5_Observable_Stream_Matches_Mdout(
            sidecar_bundled.h5_trajectory, sidecar_bundled.mdout, {0, 1},
            observable_times);
        SpongeH5InputMatrix::Require_Path_Exists(sidecar_bundled.h5_observable);
        Require_H5_Observable_Stream_Matches_Mdout(
            sidecar_bundled.h5_observable, sidecar_bundled.mdout, {0, 1},
            observable_times);
    }

    Run_Rerun_Frame_Selection_Smoke_Cases(sponge_executable, temp_root,
                                          legacy_source, bundled_source,
                                          sidecar_source);

    std::filesystem::remove_all(temp_root);
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::cerr << "usage: test_h5_input_output_smoke_matrix "
                     "<SPONGE executable>\n";
        return 2;
    }

    try
    {
        Validate_Runtime_Smoke_Preparation();
        Require_Runtime_Smoke_Enabled();
        Run_Normal_Mode_Matrix(argv[1]);
        Run_Rerun_Mode_Matrix(argv[1]);
    }
    catch (const std::exception& err)
    {
        std::cerr << err.what() << '\n';
        return 1;
    }
    return 0;
}

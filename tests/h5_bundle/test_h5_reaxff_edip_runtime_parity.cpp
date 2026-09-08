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

#include "../../SPONGE/utils/float_classification.hpp"
#include "h5_input_matrix_fixture.hpp"
#include "test_shell.hpp"
#include "utils/h5md/h5md_writer.hpp"
#include "utils/h5md/module_h5_mappings.hpp"
#include "utils/h5md/output_route_helpers.hpp"

namespace
{
constexpr int kSkipReturnCode = 77;

struct PreparedCase
{
    std::filesystem::path root;
    std::filesystem::path mdin;
    std::filesystem::path mdout;
    std::filesystem::path mdinfo;
    std::filesystem::path h5_trajectory;
    std::filesystem::path h5_observable;
};

struct MdoutTable
{
    std::vector<std::string> columns;
    std::map<std::string, std::vector<double>> values_by_column;
};

const std::vector<std::string>& Manybody_Columns()
{
    static const std::vector<std::string> columns = {
        "EDIP",        "REAXFF_BOND", "REAXFF_VDW", "REAXFF_EEQ", "REAXFF_ELP",
        "REAXFF_OVUN", "REAXFF_ANG",  "REAXFF_PEN", "REAXFF_COA", "REAXFF_TOR",
        "REAXFF_CONJ", "REAXFF_HB",   "REAXFF"};
    return columns;
}

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

void Install_Valid_Native_EAM_Atom_Types(
    const std::filesystem::path& topology_h5)
{
    HighFive::File file(topology_h5.string(), HighFive::File::ReadWrite);
    const std::vector<int> atom_type = {0, 0};
    file.getDataSet("/manybody/eam/atom_type").write(atom_type);
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

void Remove_Legacy_Sidecar_Directories(const std::filesystem::path& root)
{
    if (!std::filesystem::exists(root))
    {
        return;
    }
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

const std::vector<float>& Focused_EDIP_Positions()
{
    static const std::vector<float> positions = {
        0.0f, 0.0f, 0.0f, 1.5f, 0.0f, 0.0f, 0.1f, 0.0f, 0.0f, 1.6f, 0.0f, 0.0f,
    };
    return positions;
}

void Write_Focused_EDIP_Legacy_Trajectory(const std::filesystem::path& path)
{
    if (!std::filesystem::exists(path))
    {
        return;
    }
    const auto& positions = Focused_EDIP_Positions();
    std::ofstream out(path.c_str(), std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(positions.data()),
              static_cast<std::streamsize>(positions.size() * sizeof(float)));
    if (!out.good())
    {
        throw TestFailure("failed to write focused EDIP trajectory " +
                          path.string());
    }
}

void Write_Focused_EDIP_H5_Trajectory(const std::filesystem::path& path)
{
    if (!std::filesystem::exists(path))
    {
        return;
    }
    HighFive::File file(path.string(), HighFive::File::ReadWrite);
    auto dataset = file.getDataSet(SpongeH5MD::path::position_value);
    const auto dimensions = dataset.getSpace().getDimensions();
    REQUIRE_EQ(dimensions, std::vector<std::size_t>({2, 2, 3}));
    const auto& positions = Focused_EDIP_Positions();
    REQUIRE_TRUE(H5Dwrite(dataset.getId(), H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                          H5P_DEFAULT, positions.data()) >= 0);
}

void Remove_H5_EDIP_Blockers(const std::filesystem::path& path)
{
    if (!std::filesystem::exists(path))
    {
        return;
    }
    HighFive::File file(path.string(), HighFive::File::ReadWrite);
    for (const char* name :
         {"/topology/exclusions", "/forcefield/virtual_atom"})
    {
        if (file.exist(name))
        {
            REQUIRE_TRUE(H5Ldelete(file.getId(), name, H5P_DEFAULT) >= 0);
        }
    }
}

void Activate_Focused_EDIP_Interaction(const std::filesystem::path& root)
{
    Write_Focused_EDIP_Legacy_Trajectory(root / "traj.dat");
    Write_Focused_EDIP_H5_Trajectory(root / "trajectory.spg.h5md");
    Remove_H5_EDIP_Blockers(root / "topology.spgt.h5");

    // The generic two-atom I/O fixture has a coincident virtual site.
    // Keep both atoms independent in this nonzero EDIP interaction test.
    const auto sidecar_virtual_atoms =
        root / "legacy_sidecars" / "virtual_atom_in_file" / "virtual_atom.txt";
    if (std::filesystem::exists(sidecar_virtual_atoms))
    {
        Write_Text(sidecar_virtual_atoms, "");
    }

    const auto sidecar_exclusions =
        root / "legacy_sidecars" / "exclude_in_file" / "exclude.txt";
    if (std::filesystem::exists(sidecar_exclusions))
    {
        Write_Text(sidecar_exclusions, "2 0\n0\n0\n");
    }
}

void Require_No_Legacy_Sidecar_Directories(const std::filesystem::path& root)
{
    for (const auto& entry :
         std::filesystem::recursive_directory_iterator(root))
    {
        REQUIRE_TRUE(entry.path().filename() != "legacy_sidecars");
    }
}

void Require_No_ReaxFF_Text_Bridge(const std::filesystem::path& root)
{
    const auto bridge_root = root / ".sponge_h5_native_manybody";
    REQUIRE_TRUE(!std::filesystem::exists(bridge_root / "reaxff.txt"));
    REQUIRE_TRUE(!std::filesystem::exists(bridge_root / "reaxff_type.txt"));
}

PreparedCase Prepare_Rerun_Case(const std::filesystem::path& temp_root,
                                const std::string& name,
                                const std::filesystem::path& source_dir,
                                const std::string& source_mdin,
                                const bool bundled_output,
                                const bool vds_output)
{
    PreparedCase prepared;
    prepared.root = temp_root / name;
    Copy_Directory_Contents(source_dir, prepared.root);
    const auto output_paths =
        SpongeH5InputMatrix::Rerun_Output_Paths(prepared.root);
    std::filesystem::create_directories(output_paths.output_dir);

    std::string mdin = Read_Text(prepared.root / source_mdin);
    mdin = Remove_Key_Lines(
        mdin,
        {"output_h5_trajectory_path", "output_h5_trajectory_vds",
         "output_h5_restart_path", "output_h5_observable_path", "rerun_start",
         "rerun_strip", "rerun_frame_limit", "rerun_need_box_update",
         "input_h5_restart_load", "exclude_in_file", "virtual_atom_in_file"});
    Append_If_Missing(&mdin, "mdinfo", "mdinfo = \"mdinfo.txt\"");
    Append_If_Missing(&mdin, "mdout", "mdout = \"mdout.txt\"");
    Append_If_Missing(&mdin, "rerun_start", "rerun_start = 0");
    Append_If_Missing(&mdin, "rerun_strip", "rerun_strip = 0");
    Append_If_Missing(&mdin, "rerun_frame_limit", "rerun_frame_limit = 2");
    Append_If_Missing(&mdin, "rerun_need_box_update",
                      "rerun_need_box_update = 0");
    Append_If_Missing(&mdin, "write_mdout_interval",
                      "write_mdout_interval = 1");
    if (Has_Key_Line(mdin, "input_h5_restart_path"))
    {
        Append_If_Missing(&mdin, "input_h5_restart_load",
                          "input_h5_restart_load = \"structural\"");
    }
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

    prepared.mdin = prepared.root / "mdin.manybody.parity.spg.toml";
    prepared.mdout = output_paths.mdout;
    prepared.mdinfo = output_paths.mdinfo;
    prepared.h5_trajectory = output_paths.h5_trajectory;
    prepared.h5_observable = output_paths.h5_observable;
    Write_Text(prepared.mdin, mdin);
    Activate_Focused_EDIP_Interaction(prepared.root);
    return prepared;
}

void Require_Focused_EDIP_Interaction(const PreparedCase& test_case)
{
    REQUIRE_TRUE(!Has_Key_Line(Read_Text(test_case.mdin), "exclude_in_file"));
    REQUIRE_TRUE(
        !Has_Key_Line(Read_Text(test_case.mdin), "virtual_atom_in_file"));
    const auto& expected = Focused_EDIP_Positions();

    const auto legacy_trajectory = test_case.root / "traj.dat";
    if (std::filesystem::exists(legacy_trajectory))
    {
        std::ifstream in(legacy_trajectory.c_str(), std::ios::binary);
        std::vector<float> actual(expected.size());
        in.read(reinterpret_cast<char*>(actual.data()),
                static_cast<std::streamsize>(actual.size() * sizeof(float)));
        REQUIRE_TRUE(in.good());
        REQUIRE_EQ(actual, expected);
    }

    const auto h5_trajectory = test_case.root / "trajectory.spg.h5md";
    if (std::filesystem::exists(h5_trajectory))
    {
        HighFive::File file(h5_trajectory.string(), HighFive::File::ReadOnly);
        const auto dataset = file.getDataSet(SpongeH5MD::path::position_value);
        std::vector<float> actual(expected.size());
        REQUIRE_TRUE(H5Dread(dataset.getId(), H5T_NATIVE_FLOAT, H5S_ALL,
                             H5S_ALL, H5P_DEFAULT, actual.data()) >= 0);
        REQUIRE_EQ(actual, expected);
    }

    const auto topology = test_case.root / "topology.spgt.h5";
    if (std::filesystem::exists(topology))
    {
        HighFive::File file(topology.string(), HighFive::File::ReadOnly);
        REQUIRE_TRUE(!file.exist("/topology/exclusions"));
        REQUIRE_TRUE(!file.exist("/forcefield/virtual_atom"));
    }

    const auto sidecar_exclusions =
        test_case.root / "legacy_sidecars" / "exclude_in_file" / "exclude.txt";
    if (std::filesystem::exists(sidecar_exclusions))
    {
        REQUIRE_EQ(Read_Text(sidecar_exclusions), std::string("2 0\n0\n0\n"));
    }
}

void Require_Manybody_Not_Scrubbed(const PreparedCase& test_case)
{
    const auto mdin = Read_Text(test_case.mdin);
    REQUIRE_TRUE(mdin.find("[REAXFF]") != std::string::npos ||
                 Has_Key_Line(mdin, "input_h5_topology_path"));
    REQUIRE_TRUE(Has_Key_Line(mdin, "EDIP_in_file") ||
                 Has_Key_Line(mdin, "input_h5_topology_path"));
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
        throw TestFailure("SPONGE manybody parity smoke failed for " +
                          test_case.root.filename().string() + "\n" +
                          Read_Text(log_path));
    }
    SpongeH5InputMatrix::Require_Path_Exists(test_case.mdout);
    SpongeH5InputMatrix::Require_Path_Exists(test_case.mdinfo);
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

std::vector<float> Read_Float32_Vector(HighFive::File& file,
                                       const std::string& path)
{
    std::vector<float> values;
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
            const double tolerance =
                1.0e-6 +
                1.0e-4 * std::max(std::fabs(lhs_value), std::fabs(rhs_value));
            if (std::fabs(lhs_value - rhs_value) > tolerance)
            {
                throw TestFailure("mdout " + column + " mismatch");
            }
        }
    }
}

void Require_Manybody_Results_Nontrivial(
    const std::filesystem::path& mdout_path)
{
    const auto table = Read_Mdout_Table(mdout_path);
    for (const auto& column : {std::string("EDIP"), std::string("REAXFF")})
    {
        const auto iter = table.values_by_column.find(column);
        REQUIRE_TRUE(iter != table.values_by_column.end());
        bool nontrivial = false;
        for (const double value : iter->second)
        {
            if (SpongeFloat::Is_Finite(value) && std::fabs(value) > 1.0e-8)
            {
                nontrivial = true;
                break;
            }
        }
        if (!nontrivial)
        {
            throw TestFailure(column +
                              " module-owned runtime result is all trivial");
        }
    }
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

void Require_H5_Observable_Columns_Match_Mdout(
    const std::filesystem::path& path, const std::filesystem::path& mdout_path,
    const std::vector<std::int64_t>& expected_steps,
    const std::vector<double>& expected_times,
    const std::vector<std::string>& required_columns)
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

    const auto original_columns =
        Read_String_Vector(file, SpongeH5MD::path::mdout_columns_original_name);
    const auto hdf5_columns =
        Read_String_Vector(file, SpongeH5MD::path::mdout_columns_hdf5_name);
    REQUIRE_EQ(original_columns.size(), hdf5_columns.size());
    std::map<std::string, std::string> h5_name_by_original_name;
    for (std::size_t i = 0; i < original_columns.size(); ++i)
    {
        h5_name_by_original_name[original_columns[i]] = hdf5_columns[i];
    }

    const auto mdout = Read_Mdout_Table(mdout_path);
    for (const auto& column : required_columns)
    {
        const auto h5_name = h5_name_by_original_name.find(column);
        REQUIRE_TRUE(h5_name != h5_name_by_original_name.end());
        const auto mdout_column = mdout.values_by_column.find(column);
        REQUIRE_TRUE(mdout_column != mdout.values_by_column.end());
        REQUIRE_EQ(mdout_column->second.size(), expected_steps.size());

        const std::string value_path =
            SpongeH5MD::Observable_Value_Path(h5_name->second);
        REQUIRE_TRUE(file.exist(value_path));
        const auto h5_values = Read_Float64_Vector(file, value_path);
        REQUIRE_EQ(h5_values.size(), expected_steps.size());
        for (std::size_t i = 0; i < h5_values.size(); ++i)
        {
            const double expected = mdout_column->second[i];
            const double actual = h5_values[i];
            const double tolerance =
                1.0e-10 +
                1.0e-8 * std::max(std::fabs(actual), std::fabs(expected));
            if (std::fabs(actual - expected) > tolerance)
            {
                throw TestFailure("observable " + column + " mismatch");
            }
        }
    }
}

void Require_VDS_Wrapper_Finalized(const std::filesystem::path& wrapper_path)
{
    SpongeH5InputMatrix::Require_Path_Exists(wrapper_path);
    HighFive::File file(wrapper_path.string(), HighFive::File::ReadOnly);
    REQUIRE_TRUE(file.exist(SpongeH5MD::path::output_status));
    REQUIRE_TRUE(file.exist(SpongeH5MD::path::output_vds_status));
    REQUIRE_TRUE(file.exist(SpongeH5MD::path::shard_manifest_path));
    REQUIRE_EQ(Read_String_Scalar(file, SpongeH5MD::path::output_status),
               std::string("finalized"));
    REQUIRE_EQ(Read_String_Scalar(file, SpongeH5MD::path::output_vds_status),
               std::string("particle, observable, and module virtual datasets "
                           "materialized"));
    const auto shard_paths =
        Read_String_Vector(file, SpongeH5MD::path::shard_manifest_path);
    REQUIRE_TRUE(!shard_paths.empty());
}

void Require_Reaxff_Eeq_Charge_Snapshot(const std::filesystem::path& h5_path)
{
    HighFive::File file(h5_path.string(), HighFive::File::ReadOnly);
    REQUIRE_TRUE(file.exist(SpongeH5MD::module_path::reaxff_eeq_charges_value));
    const auto charges = Read_Float32_Vector(
        file, SpongeH5MD::module_path::reaxff_eeq_charges_value);
    REQUIRE_TRUE(!charges.empty());
    for (const float charge : charges)
    {
        REQUIRE_TRUE(SpongeFloat::Is_Finite(charge));
    }
}

void Require_Runtime_Smoke_Enabled()
{
    const char* enabled = std::getenv("SPONGE_H5_ENABLE_RUNTIME_SMOKE");
    if (enabled == nullptr || std::string(enabled) != "1")
    {
        std::cerr << "Skipping REAXFF/EDIP runtime parity smoke; set "
                     "SPONGE_H5_ENABLE_RUNTIME_SMOKE=1 in a runnable SPONGE "
                     "CPU/GPU environment to enable it.\n";
        std::exit(kSkipReturnCode);
    }
}

void Validate_Preparation()
{
    const auto temp_root =
        SpongeH5Test::Unique_Temp_Path("h5_reaxff_edip_prepare_check");
    std::filesystem::create_directories(temp_root);
    const auto full = SpongeH5InputMatrix::Full_Contract_Rerun_Path();
    const auto legacy = Prepare_Rerun_Case(temp_root, "manybody_legacy_prepare",
                                           full / "legacy_input",
                                           "mdin.spg.toml", false, false);
    const auto sidecar = Prepare_Rerun_Case(
        temp_root, "manybody_sidecar_prepare",
        full / "bundled_input_with_legacy_sidecar" / "bundle",
        "mdin.bundled.spg.toml", true, true);
    auto pure = Prepare_Rerun_Case(temp_root, "manybody_pure_bundled_prepare",
                                   full / "bundled_input" / "bundle",
                                   "mdin.bundled.spg.toml", true, true);
    Remove_Legacy_Sidecar_Directories(pure.root);

    Require_Manybody_Not_Scrubbed(legacy);
    Require_Manybody_Not_Scrubbed(sidecar);
    Require_Manybody_Not_Scrubbed(pure);
    Require_Focused_EDIP_Interaction(legacy);
    Require_Focused_EDIP_Interaction(sidecar);
    Require_Focused_EDIP_Interaction(pure);
    Require_No_Legacy_Sidecar_Directories(pure.root);
    REQUIRE_TRUE(Read_Text(legacy.mdin).find("[REAXFF]") != std::string::npos);
    REQUIRE_TRUE(Has_Key_Line(Read_Text(legacy.mdin), "EDIP_in_file"));
    REQUIRE_TRUE(
        Has_Key_Line(Read_Text(sidecar.mdin), "input_h5_topology_path"));
    REQUIRE_TRUE(
        Has_Key_Line(Read_Text(sidecar.mdin), "input_h5_restart_load"));
    REQUIRE_TRUE(Has_Key_Line(Read_Text(pure.mdin), "input_h5_topology_path"));
    REQUIRE_TRUE(!Has_Key_Line(Read_Text(pure.mdin), "EDIP_in_file"));
    REQUIRE_TRUE(Read_Text(pure.mdin).find("[REAXFF]") == std::string::npos);
    std::filesystem::remove_all(temp_root);
}

void Run_Manybody_Runtime_Parity(const std::filesystem::path& sponge_executable)
{
    const auto temp_root =
        SpongeH5Test::Unique_Temp_Path("h5_reaxff_edip_runtime_parity");
    std::filesystem::create_directories(temp_root);

    const auto full = SpongeH5InputMatrix::Full_Contract_Rerun_Path();
    const auto legacy_source = full / "legacy_input";
    const auto pure_source = full / "bundled_input" / "bundle";
    const auto sidecar_source =
        full / "bundled_input_with_legacy_sidecar" / "bundle";

    const auto baseline = Prepare_Rerun_Case(
        temp_root, "manybody_legacy_in_legacy_out_no_manybody_scrub",
        legacy_source, "mdin.spg.toml", false, false);
    Require_Manybody_Not_Scrubbed(baseline);
    Run_SPONGE(sponge_executable, baseline);
    SpongeH5InputMatrix::Require_Path_Exists(baseline.root / "eeq_charges.txt");
    Require_Mdout_Columns_Equivalent(baseline.mdout, baseline.mdout,
                                     Manybody_Columns());
    Require_Manybody_Results_Nontrivial(baseline.mdout);

    const auto legacy_out = Prepare_Rerun_Case(
        temp_root, "manybody_sidecar_in_legacy_out_no_manybody_scrub",
        sidecar_source, "mdin.bundled.spg.toml", false, false);
    Require_Manybody_Not_Scrubbed(legacy_out);
    Run_SPONGE(sponge_executable, legacy_out);
    SpongeH5InputMatrix::Require_Path_Exists(legacy_out.root /
                                             "eeq_charges.txt");
    Require_Mdout_Columns_Equivalent(baseline.mdout, legacy_out.mdout,
                                     Manybody_Columns());
    Require_Manybody_Results_Nontrivial(legacy_out.mdout);

    const auto pure_legacy_out = Prepare_Rerun_Case(
        temp_root, "manybody_pure_bundled_in_legacy_out_no_sidecar",
        pure_source, "mdin.bundled.spg.toml", false, false);
    Install_Valid_Native_EAM_Atom_Types(pure_legacy_out.root /
                                        "topology.spgt.h5");
    Remove_Legacy_Sidecar_Directories(pure_legacy_out.root);
    Require_Manybody_Not_Scrubbed(pure_legacy_out);
    Require_No_Legacy_Sidecar_Directories(pure_legacy_out.root);
    Run_SPONGE(sponge_executable, pure_legacy_out);
    SpongeH5InputMatrix::Require_Path_Exists(pure_legacy_out.root /
                                             "eeq_charges.txt");
    REQUIRE_TRUE(!std::filesystem::exists(
        pure_legacy_out.root / ".sponge_h5_native_manybody" / "edip.txt"));
    Require_No_ReaxFF_Text_Bridge(pure_legacy_out.root);
    Require_Mdout_Columns_Equivalent(baseline.mdout, pure_legacy_out.mdout,
                                     Manybody_Columns());
    Require_Manybody_Results_Nontrivial(pure_legacy_out.mdout);

    for (const bool vds : {false, true})
    {
        const auto bundled_out = Prepare_Rerun_Case(
            temp_root,
            std::string("manybody_sidecar_in_bundled_out_vds_") +
                (vds ? "on" : "off") + "_no_manybody_scrub",
            sidecar_source, "mdin.bundled.spg.toml", true, vds);
        Require_Manybody_Not_Scrubbed(bundled_out);
        Run_SPONGE(sponge_executable, bundled_out);
        REQUIRE_TRUE(
            !std::filesystem::exists(bundled_out.root / "eeq_charges.txt"));
        Require_Mdout_Columns_Equivalent(baseline.mdout, bundled_out.mdout,
                                         Manybody_Columns());
        Require_Manybody_Results_Nontrivial(bundled_out.mdout);
        SpongeH5InputMatrix::Require_Path_Exists(bundled_out.h5_observable);
        Require_H5_Observable_Columns_Match_Mdout(
            bundled_out.h5_observable, bundled_out.mdout, {0, 1}, {1.0, 1.001},
            Manybody_Columns());
        Require_Reaxff_Eeq_Charge_Snapshot(bundled_out.h5_observable);
        SpongeH5InputMatrix::Require_Path_Exists(bundled_out.h5_trajectory);
        if (vds)
        {
            Require_VDS_Wrapper_Finalized(bundled_out.h5_trajectory);
            Require_Reaxff_Eeq_Charge_Snapshot(bundled_out.h5_trajectory);
        }
        else
        {
            Require_H5_Observable_Columns_Match_Mdout(
                bundled_out.h5_trajectory, bundled_out.mdout, {0, 1},
                {1.0, 1.001}, Manybody_Columns());
            Require_Reaxff_Eeq_Charge_Snapshot(bundled_out.h5_trajectory);
        }

        const auto pure_bundled_out = Prepare_Rerun_Case(
            temp_root,
            std::string("manybody_pure_bundled_in_bundled_out_vds_") +
                (vds ? "on" : "off") + "_no_sidecar",
            pure_source, "mdin.bundled.spg.toml", true, vds);
        Install_Valid_Native_EAM_Atom_Types(pure_bundled_out.root /
                                            "topology.spgt.h5");
        Remove_Legacy_Sidecar_Directories(pure_bundled_out.root);
        Require_Manybody_Not_Scrubbed(pure_bundled_out);
        Require_No_Legacy_Sidecar_Directories(pure_bundled_out.root);
        Run_SPONGE(sponge_executable, pure_bundled_out);
        REQUIRE_TRUE(!std::filesystem::exists(pure_bundled_out.root /
                                              "eeq_charges.txt"));
        REQUIRE_TRUE(!std::filesystem::exists(
            pure_bundled_out.root / ".sponge_h5_native_manybody" / "edip.txt"));
        Require_No_ReaxFF_Text_Bridge(pure_bundled_out.root);
        Require_Mdout_Columns_Equivalent(baseline.mdout, pure_bundled_out.mdout,
                                         Manybody_Columns());
        Require_Manybody_Results_Nontrivial(pure_bundled_out.mdout);
        SpongeH5InputMatrix::Require_Path_Exists(
            pure_bundled_out.h5_observable);
        Require_H5_Observable_Columns_Match_Mdout(
            pure_bundled_out.h5_observable, pure_bundled_out.mdout, {0, 1},
            {1.0, 1.001}, Manybody_Columns());
        Require_Reaxff_Eeq_Charge_Snapshot(pure_bundled_out.h5_observable);
        SpongeH5InputMatrix::Require_Path_Exists(
            pure_bundled_out.h5_trajectory);
        if (vds)
        {
            Require_VDS_Wrapper_Finalized(pure_bundled_out.h5_trajectory);
            Require_Reaxff_Eeq_Charge_Snapshot(pure_bundled_out.h5_trajectory);
        }
        else
        {
            Require_H5_Observable_Columns_Match_Mdout(
                pure_bundled_out.h5_trajectory, pure_bundled_out.mdout, {0, 1},
                {1.0, 1.001}, Manybody_Columns());
            Require_Reaxff_Eeq_Charge_Snapshot(pure_bundled_out.h5_trajectory);
        }
    }

    std::filesystem::remove_all(temp_root);
}
}  // namespace

int main(int argc, char** argv)
{
    try
    {
        Validate_Preparation();
        Require_Runtime_Smoke_Enabled();
        REQUIRE_TRUE(argc >= 2);
        Run_Manybody_Runtime_Parity(argv[1]);
    }
    catch (const std::exception& err)
    {
        std::cerr << err.what() << "\n";
        return 1;
    }
    return 0;
}

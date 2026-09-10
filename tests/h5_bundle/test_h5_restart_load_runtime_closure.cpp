#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <highfive/highfive.hpp>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../../SPONGE/utils/float_classification.hpp"
#include "h5_input_matrix_fixture.hpp"
#include "test_shell.hpp"
#include "utils/h5md/h5md_writer.hpp"
#include "utils/h5md/restart_h5_reader.hpp"
#include "utils/random/restart_rng_state.hpp"

namespace
{
constexpr int kSkipReturnCode = 77;

struct PreparedCase
{
    std::filesystem::path root;
    std::filesystem::path mdin;
    std::filesystem::path mdout;
    std::filesystem::path mdinfo;
    std::filesystem::path h5_restart;
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

void Require_Contains(const std::string& text, const std::string& needle)
{
    std::string normalized = text;
#ifdef _WIN32
    std::replace(normalized.begin(), normalized.end(), '\\', '/');
#endif
    if (normalized.find(needle) == std::string::npos)
    {
        throw TestFailure("expected text to contain: " + needle);
    }
}

template <typename Container>
void Require_Float_Container_Close(const Container& lhs, const Container& rhs,
                                   const float tolerance = 1.0e-6f)
{
    REQUIRE_EQ(lhs.size(), rhs.size());
    for (std::size_t index = 0; index < lhs.size(); ++index)
    {
        const float scale = std::max(
            1.0f, std::max(std::fabs(lhs[index]), std::fabs(rhs[index])));
        if (std::fabs(lhs[index] - rhs[index]) > tolerance * scale)
        {
            std::ostringstream message;
            message << "float container mismatch at index " << index
                    << ": lhs=" << lhs[index] << ", rhs=" << rhs[index]
                    << ", tolerance=" << tolerance
                    << ", scaled_limit=" << tolerance * scale;
            throw TestFailure(message.str());
        }
    }
}

void Replace_All(std::string* text, const std::string& from,
                 const std::string& to)
{
    std::size_t pos = 0;
    while ((pos = text->find(from, pos)) != std::string::npos)
    {
        text->replace(pos, from.size(), to);
        pos += to.size();
    }
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

void Delete_H5_Object_If_Exists(const std::filesystem::path& h5_path,
                                const char* object_path)
{
    HighFive::File file(h5_path.string(), HighFive::File::ReadWrite);
    if (file.exist(object_path))
    {
        H5Ldelete(file.getId(), object_path, H5P_DEFAULT);
    }
}

void Write_H5_String_Overwrite(const std::filesystem::path& h5_path,
                               const std::string& object_path,
                               const std::string& value)
{
    HighFive::File file(h5_path.string(), HighFive::File::ReadWrite);
    if (file.exist(object_path))
    {
        H5Ldelete(file.getId(), object_path.c_str(), H5P_DEFAULT);
    }
    auto dataset = file.createDataSet<std::string>(
        object_path, HighFive::DataSpace::From(value));
    dataset.write(value);
}

void Install_Valid_Native_EAM_Atom_Types(
    const std::filesystem::path& topology_h5)
{
    HighFive::File file(topology_h5.string(), HighFive::File::ReadWrite);
    const std::vector<int> atom_type = {0, 0};
    file.getDataSet("/manybody/eam/atom_type").write(atom_type);
}

void Enable_Meta_In_Restart_Protocol_Sidecar(
    const std::filesystem::path& restart_h5_path)
{
    const std::string cv_text =
        "print\n"
        "{\n"
        "    CV = distance\n"
        "}\n"
        "distance\n"
        "{\n"
        "    CV_type = distance\n"
        "    atom = 0 1\n"
        "}\n"
        "meta\n"
        "{\n"
        "    CV = distance\n"
        "    CV_period = 0\n"
        "    CV_sigma = 0.5\n"
        "    CV_minimal = 0\n"
        "    CV_maximum = 10\n"
        "    CV_grid = 4\n"
        "    edge_in_file = sumhill.log\n"
        "    potential_update_interval = 1000\n"
        "}\n";
    Write_H5_String_Overwrite(
        restart_h5_path,
        SpongeH5MD::Restart_Protocol_Sidecar_Path("cv_in_file"), cv_text);
}

void Isolate_NHC_Dynamic_Runtime_Inputs(const std::filesystem::path& root)
{
    const auto topology = root / "topology.spgt.h5";
    for (const char* object_path : {"/forcefield", "/manybody"})
    {
        Delete_H5_Object_If_Exists(topology, object_path);
    }

    const auto protocol = root / "protocol.spgp.h5";
    for (const char* object_path : {"/constraint", "/cv", "/meta", "/restraint",
                                    "/sits", "/steer", "/wall"})
    {
        Delete_H5_Object_If_Exists(protocol, object_path);
    }
}

void Isolate_Sits_Protocol_Runtime_Inputs(const std::filesystem::path& root)
{
    const auto topology = root / "topology.spgt.h5";
    for (const char* object_path : {"/forcefield", "/manybody"})
    {
        Delete_H5_Object_If_Exists(topology, object_path);
    }

    const auto protocol = root / "protocol.spgp.h5";
    for (const char* object_path :
         {"/constraint", "/cv", "/meta", "/restraint", "/steer", "/wall"})
    {
        Delete_H5_Object_If_Exists(protocol, object_path);
    }
}

void Install_Full_Sits_Restart_State(const std::filesystem::path& restart_path,
                                     const std::vector<float>& nk,
                                     const std::vector<float>& log_norm,
                                     const std::vector<float>& log_nk)
{
    REQUIRE_EQ(nk.size(), log_norm.size());
    REQUIRE_EQ(nk.size(), log_nk.size());
    HighFive::File restart(restart_path.string(), HighFive::File::ReadWrite);
    const std::string root = SpongeH5MD::Restart_Sits_State_Root("SITS");
    if (!restart.exist(root)) restart.createGroup(root);
    const auto write = [&restart, &root](const std::string& name,
                                         const std::vector<float>& values)
    {
        const std::string path = root + "/" + name;
        if (restart.exist(path))
        {
            restart.getDataSet(path).write(values);
        }
        else
        {
            restart
                .createDataSet<float>(path, HighFive::DataSpace::From(values))
                .write(values);
        }
    };
    write("nk", nk);
    write("log_norm", log_norm);
    write("log_nk", log_nk);
}

void Install_Portable_Rng_Dynamic_State(
    const std::filesystem::path& restart_path, const std::string& module,
    const SpongeH5MD::RestartRngState& rng_state,
    const std::string& integrator_mode)
{
    SpongeH5MD::RestartStructuralState structural_state;
    {
        SpongeH5MD::RestartH5Reader structural_reader;
        REQUIRE_TRUE(structural_reader.Open(restart_path.string()));
        REQUIRE_TRUE(
            structural_reader.Read_Structural_State(&structural_state));
    }
    Delete_H5_Object_If_Exists(restart_path, SpongeH5MD::path::restart_nhc);
    Delete_H5_Object_If_Exists(restart_path,
                               SpongeH5MD::path::restart_rng_state);
    Delete_H5_Object_If_Exists(restart_path,
                               SpongeH5MD::path::restart_thermostat);
    Delete_H5_Object_If_Exists(restart_path,
                               SpongeH5MD::path::restart_barostat);

    HighFive::File restart(restart_path.string(), HighFive::File::ReadWrite);
    const std::string root = SpongeH5MD::Restart_Rng_State_Path(module);
    restart.createGroup(root);
    const std::vector<std::int64_t> schema = {rng_state.state_schema_version};
    auto schema_dataset = restart.createDataSet<std::int64_t>(
        SpongeH5MD::Restart_Rng_State_Component_Path(module, "schema_version"),
        HighFive::DataSpace::From(schema));
    schema_dataset.write(schema);
    auto engine_dataset = restart.createDataSet<std::string>(
        SpongeH5MD::Restart_Rng_State_Component_Path(module, "engine"),
        HighFive::DataSpace::From(rng_state.engine));
    engine_dataset.write(rng_state.engine);
    auto words_dataset = restart.createDataSet<std::int64_t>(
        SpongeH5MD::Restart_Rng_State_Component_Path(module, "state_words"),
        HighFive::DataSpace(
            {rng_state.stream_count, rng_state.words_per_stream}));
    std::vector<std::vector<std::int64_t>> words(
        rng_state.stream_count,
        std::vector<std::int64_t>(rng_state.words_per_stream));
    for (std::size_t stream = 0; stream < rng_state.stream_count; ++stream)
    {
        std::copy_n(
            rng_state.state_words.begin() + stream * rng_state.words_per_stream,
            rng_state.words_per_stream, words[stream].begin());
    }
    words_dataset.write(words);
    if (!restart.exist(SpongeH5MD::path::restart_integrator_state))
    {
        restart.createGroup(SpongeH5MD::path::restart_integrator_state);
    }
    for (const auto& item : std::vector<std::pair<std::string, std::string>>{
             {"mode", integrator_mode},
             {"step", std::to_string(structural_state.step)},
             {"time", std::to_string(structural_state.time)}})
    {
        const std::string path =
            SpongeH5MD::Restart_Integrator_State_Path(item.first);
        if (restart.exist(path))
        {
            H5Ldelete(restart.getId(), path.c_str(), H5P_DEFAULT);
        }
        restart
            .createDataSet<std::string>(path,
                                        HighFive::DataSpace::From(item.second))
            .write(item.second);
    }
}

void Install_Monte_Carlo_Adaptive_State(
    const std::filesystem::path& restart_path)
{
    HighFive::File restart(restart_path.string(), HighFive::File::ReadWrite);
    const std::string root =
        SpongeH5MD::Restart_Barostat_State_Root("monte_carlo_barostat");
    restart.createGroup(root);
    const std::vector<float> delta = {0.01f, 0.02f, 0.03f};
    const std::vector<float> rate = {31.0f, 32.0f, 33.0f};
    const std::vector<std::int64_t> total = {10, 20, 30};
    const std::vector<std::int64_t> accepted = {3, 6, 9};
    restart
        .createDataSet<float>(root + "/delta_box_length_max",
                              HighFive::DataSpace::From(delta))
        .write(delta);
    restart
        .createDataSet<float>(root + "/accept_rate",
                              HighFive::DataSpace::From(rate))
        .write(rate);
    restart
        .createDataSet<std::int64_t>(root + "/total_count_int64",
                                     HighFive::DataSpace::From(total))
        .write(total);
    restart
        .createDataSet<std::int64_t>(root + "/accept_count_int64",
                                     HighFive::DataSpace::From(accepted))
        .write(accepted);
}

void Install_Portable_Rng_Module_Float_State(
    const std::filesystem::path& restart_path, const std::string& module,
    const std::string& state_name, const std::vector<float>& values,
    const bool thermostat_state)
{
    HighFive::File restart(restart_path.string(), HighFive::File::ReadWrite);
    const std::string root =
        thermostat_state ? SpongeH5MD::Restart_Thermostat_State_Root(module)
                         : SpongeH5MD::Restart_Barostat_State_Root(module);
    restart.createGroup(root);
    restart
        .createDataSet<float>(root + "/" + state_name,
                              HighFive::DataSpace::From(values))
        .write(values);
}

void Isolate_Positional_Restraint_Runtime_Inputs(
    const std::filesystem::path& root)
{
    const auto topology = root / "topology.spgt.h5";
    for (const char* object_path : {"/forcefield", "/manybody"})
    {
        Delete_H5_Object_If_Exists(topology, object_path);
    }

    const auto protocol = root / "protocol.spgp.h5";
    for (const char* object_path :
         {"/constraint", "/cv", "/meta", "/restraint/config", "/restraint/cv",
          "/sits", "/steer", "/wall"})
    {
        Delete_H5_Object_If_Exists(protocol, object_path);
    }
}

void Isolate_And_Install_Native_CV_Runtime_Inputs(
    const std::filesystem::path& root)
{
    const auto topology = root / "topology.spgt.h5";
    for (const char* object_path : {"/forcefield", "/manybody"})
    {
        Delete_H5_Object_If_Exists(topology, object_path);
    }

    const auto protocol_path = root / "protocol.spgp.h5";
    for (const char* object_path : {"/constraint", "/cv", "/meta", "/restraint",
                                    "/sits", "/steer", "/wall"})
    {
        Delete_H5_Object_If_Exists(protocol_path, object_path);
    }
    HighFive::File protocol(protocol_path.string(), HighFive::File::ReadWrite);
    protocol.createGroup("/cv");
    protocol.createGroup("/cv/backbone");
    auto cv_type = protocol.createDataSet<std::string>(
        "/cv/backbone/type", HighFive::DataSpace::From(std::string("rmsd")));
    cv_type.write(std::string("rmsd"));
    const std::vector<int> atom_indices = {0, 1};
    auto atoms = protocol.createDataSet<int>(
        "/cv/backbone/atom_indices", HighFive::DataSpace::From(atom_indices));
    atoms.write(atom_indices);

    protocol.createGroup("/restraint");
    protocol.createGroup("/restraint/umbrella");
    auto restraint_type = protocol.createDataSet<std::string>(
        "/restraint/umbrella/type",
        HighFive::DataSpace::From(std::string("cv_harmonic")));
    restraint_type.write(std::string("cv_harmonic"));
    const std::vector<std::string> cv_refs = {"backbone"};
    auto refs = protocol.createDataSet<std::string>(
        "/restraint/umbrella/cv_refs", HighFive::DataSpace::From(cv_refs));
    refs.write(cv_refs);
    const std::vector<float> weight = {1.0f};
    auto weights = protocol.createDataSet<float>(
        "/restraint/umbrella/weight", HighFive::DataSpace::From(weight));
    weights.write(weight);
    const std::vector<float> reference = {1.0f};
    auto references = protocol.createDataSet<float>(
        "/restraint/umbrella/reference", HighFive::DataSpace::From(reference));
    references.write(reference);

    const auto restart_path = root / "restart.spgr.h5";
    HighFive::File restart(restart_path.string(), HighFive::File::ReadWrite);
    for (const char* object_path :
         {SpongeH5MD::path::restart_restraint_references,
          SpongeH5MD::path::restart_protocol_sidecars})
    {
        if (restart.exist(object_path))
        {
            H5Ldelete(restart.getId(), object_path, H5P_DEFAULT);
        }
    }
    if (!restart.exist("/parameters/restart/references/cv"))
    {
        restart.createGroup("/parameters/restart/references/cv");
    }
    restart.createGroup("/parameters/restart/references/cv/backbone");
    const std::vector<float> reference_coordinates = {0.0f, 0.0f, 0.0f,
                                                      1.0f, 0.0f, 0.0f};
    auto coordinate = restart.createDataSet<float>(
        "/parameters/restart/references/cv/backbone/coordinate",
        HighFive::DataSpace({2, 3}));
    REQUIRE_TRUE(H5Dwrite(coordinate.getId(), H5T_NATIVE_FLOAT, H5S_ALL,
                          H5S_ALL, H5P_DEFAULT,
                          reference_coordinates.data()) >= 0);
}

void Isolate_And_Install_Native_Metadynamics_Runtime_Inputs(
    const std::filesystem::path& root)
{
    const auto topology = root / "topology.spgt.h5";
    for (const char* object_path : {"/forcefield", "/manybody"})
    {
        Delete_H5_Object_If_Exists(topology, object_path);
    }

    const auto protocol_path = root / "protocol.spgp.h5";
    for (const char* object_path : {"/constraint", "/cv", "/meta", "/restraint",
                                    "/sits", "/steer", "/wall"})
    {
        Delete_H5_Object_If_Exists(protocol_path, object_path);
    }
    HighFive::File protocol(protocol_path.string(), HighFive::File::ReadWrite);
    protocol.createGroup("/cv");
    protocol.createGroup("/cv/distance");
    auto cv_type = protocol.createDataSet<std::string>(
        "/cv/distance/type",
        HighFive::DataSpace::From(std::string("distance")));
    cv_type.write(std::string("distance"));
    const std::vector<int> atom_indices = {0, 1};
    auto atoms = protocol.createDataSet<int>(
        "/cv/distance/atom_indices", HighFive::DataSpace::From(atom_indices));
    atoms.write(atom_indices);
    const std::vector<float> sigma = {0.5f};
    auto cv_sigma = protocol.createDataSet<float>(
        "/cv/distance/sigma", HighFive::DataSpace::From(sigma));
    cv_sigma.write(sigma);
    const std::vector<float> period = {0.0f};
    auto cv_period = protocol.createDataSet<float>(
        "/cv/distance/period", HighFive::DataSpace::From(period));
    cv_period.write(period);

    protocol.createGroup("/meta");
    protocol.createGroup("/meta/native_bias");
    protocol.createGroup("/meta/native_bias/grid");
    const std::vector<std::string> cv_refs = {"distance"};
    auto refs = protocol.createDataSet<std::string>(
        "/meta/native_bias/cv_refs", HighFive::DataSpace::From(cv_refs));
    refs.write(cv_refs);
    const std::vector<float> grid_min = {0.0f};
    auto minimum = protocol.createDataSet<float>(
        "/meta/native_bias/grid/min", HighFive::DataSpace::From(grid_min));
    minimum.write(grid_min);
    const std::vector<float> grid_max = {10.0f};
    auto maximum = protocol.createDataSet<float>(
        "/meta/native_bias/grid/max", HighFive::DataSpace::From(grid_max));
    maximum.write(grid_max);
    const std::vector<long long> grid_count = {4};
    auto count = protocol.createDataSet<long long>(
        "/meta/native_bias/grid/count", HighFive::DataSpace::From(grid_count));
    count.write(grid_count);
    const long long interval = 1000;
    auto update_interval = protocol.createDataSet<long long>(
        "/meta/native_bias/potential_update_interval_default",
        HighFive::DataSpace::From(interval));
    update_interval.write(interval);

    Delete_H5_Object_If_Exists(root / "restart.spgr.h5",
                               SpongeH5MD::path::restart_protocol_sidecars);
    Delete_H5_Object_If_Exists(root / "restart.spgr.h5",
                               SpongeH5MD::path::restart_restraint_references);
    Delete_H5_Object_If_Exists(root / "restart.spgr.h5",
                               SpongeH5MD::path::restart_cv_references);
    HighFive::File restart((root / "restart.spgr.h5").string(),
                           HighFive::File::ReadWrite);
    const std::string named_state =
        std::string(SpongeH5MD::path::restart_meta) + "/native_bias";
    if (restart.exist(SpongeH5MD::path::restart_meta))
    {
        REQUIRE_TRUE(H5Ldelete(restart.getId(), SpongeH5MD::path::restart_meta,
                               H5P_DEFAULT) >= 0);
    }
    restart.createGroup(SpongeH5MD::path::restart_meta);
    restart.createGroup(named_state);
    restart.createGroup(named_state + "/grid");
    restart.createGroup(named_state + "/potential");
    restart.createGroup(named_state + "/edge");
    restart.createGroup(named_state + "/hills");
    restart.createGroup(named_state + "/runtime");

    const auto write_int64 = [&restart](const std::string& path,
                                        const std::vector<long long>& values)
    {
        auto dataset = restart.createDataSet<long long>(
            path, HighFive::DataSpace::From(values));
        dataset.write(values);
    };
    const auto write_float =
        [&restart](const std::string& path, const std::vector<float>& values)
    {
        auto dataset = restart.createDataSet<float>(
            path, HighFive::DataSpace::From(values));
        dataset.write(values);
    };
    const auto write_float_matrix =
        [&restart](const std::string& path, const std::vector<float>& values,
                   const std::vector<std::size_t>& dimensions)
    {
        auto dataset =
            restart.createDataSet<float>(path, HighFive::DataSpace(dimensions));
        REQUIRE_TRUE(H5Dwrite(dataset.getId(), H5T_NATIVE_FLOAT, H5S_ALL,
                              H5S_ALL, H5P_DEFAULT, values.data()) >= 0);
    };

    write_int64(named_state + "/state_schema_version", {1});
    write_int64(named_state + "/ndim", {1});
    write_int64(named_state + "/grid/count", {4});
    write_float(named_state + "/grid/min", {0.0f});
    write_float(named_state + "/grid/max", {10.0f});
    write_float(named_state + "/potential/value", {1.0f, 2.0f, 3.0f, 4.0f});
    write_float_matrix(named_state + "/potential/force",
                       {0.1f, 0.2f, 0.3f, 0.4f}, {4, 1});
    write_float(named_state + "/edge/log_normalization",
                {0.0f, 0.0f, 0.0f, 0.0f});
    write_float_matrix(named_state + "/edge/normal_force",
                       {0.0f, 0.0f, 0.0f, 0.0f}, {4, 1});
    write_float_matrix(named_state + "/hills/center", {1.0f}, {1, 1});
    write_float(named_state + "/hills/height", {0.25f});
    write_float_matrix(named_state + "/hills/inverse_width", {2.0f}, {1, 1});
    write_float_matrix(named_state + "/hills/period", {0.0f}, {1, 1});
    write_float(named_state + "/runtime/potential_max", {4.0f});
    write_float(named_state + "/runtime/sum_max", {0.0f});
    write_float(named_state + "/runtime/new_max", {0.0f});
    write_float(named_state + "/runtime/exit_tag", {0.0f});
    write_float(named_state + "/runtime/rct", {0.1f});
    write_float(named_state + "/runtime/rbias", {0.2f});
    write_float(named_state + "/runtime/bias", {0.3f});
    write_float(named_state + "/runtime/minus_beta_f", {0.4f});
    write_float(named_state + "/runtime/minus_beta_f_plus_v", {0.5f});
    write_int64(named_state + "/runtime/max_index", {0});
}

void Convert_Metadynamics_Runtime_Inputs_To_Scatter_Sink(
    const std::filesystem::path& root)
{
    const auto protocol_path = root / "protocol.spgp.h5";
    Delete_H5_Object_If_Exists(protocol_path, "/cv");
    Delete_H5_Object_If_Exists(protocol_path, "/meta");

    const auto restart_path = root / "restart.spgr.h5";
    HighFive::File restart(restart_path.string(), HighFive::File::ReadWrite);
    const std::string native_state =
        std::string(SpongeH5MD::path::restart_meta) + "/native_bias";
    const std::string scatter_sink_state =
        std::string(SpongeH5MD::path::restart_meta) + "/meta";
    REQUIRE_TRUE(restart.exist(native_state));
    REQUIRE_TRUE(H5Lmove(restart.getId(), native_state.c_str(), restart.getId(),
                         scatter_sink_state.c_str(), H5P_DEFAULT,
                         H5P_DEFAULT) >= 0);
    restart.createGroup(scatter_sink_state + "/scatter");

    const std::vector<float> scatter_position = {1.0f, 4.0f, 8.0f};
    auto position = restart.createDataSet<float>(
        scatter_sink_state + "/scatter/position", HighFive::DataSpace({3, 1}));
    REQUIRE_TRUE(H5Dwrite(position.getId(), H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                          H5P_DEFAULT, scatter_position.data()) >= 0);
    const std::vector<float> scatter_potential = {1.0f, 2.0f, 3.0f};
    auto potential = restart.createDataSet<float>(
        scatter_sink_state + "/scatter/potential",
        HighFive::DataSpace::From(scatter_potential));
    potential.write(scatter_potential);
    const std::vector<float> scatter_force = {0.1f, 0.2f, 0.3f};
    auto force = restart.createDataSet<float>(
        scatter_sink_state + "/scatter/force", HighFive::DataSpace({3, 1}));
    REQUIRE_TRUE(H5Dwrite(force.getId(), H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                          H5P_DEFAULT, scatter_force.data()) >= 0);
    const std::vector<float> hill_sink = {0.75f};
    auto sink =
        restart.createDataSet<float>(scatter_sink_state + "/hills/sink",
                                     HighFive::DataSpace::From(hill_sink));
    sink.write(hill_sink);

    if (!restart.exist(SpongeH5MD::path::restart_protocol_sidecars))
    {
        restart.createGroup(SpongeH5MD::path::restart_protocol_sidecars);
    }
    const std::string cv_text =
        "print\n"
        "{\n"
        "    CV = distance\n"
        "}\n"
        "distance\n"
        "{\n"
        "    CV_type = distance\n"
        "    atom = 0 1\n"
        "    CV_point = 1 4 8\n"
        "}\n"
        "meta\n"
        "{\n"
        "    CV = distance\n"
        "    CV_period = 0\n"
        "    CV_sigma = 0.5\n"
        "    CV_minimal = 0\n"
        "    CV_maximum = 10\n"
        "    CV_grid = 4\n"
        "    scatter = 3\n"
        "    sink = 1\n"
        "    potential_update_interval = 1000\n"
        "}\n";
    auto sidecar = restart.createDataSet<std::string>(
        SpongeH5MD::Restart_Protocol_Sidecar_Path("cv_in_file"),
        HighFive::DataSpace::From(cv_text));
    sidecar.write(cv_text);
}

PreparedCase Prepare_Restart_Load_Case(
    const std::filesystem::path& temp_root, const std::string& name,
    const std::filesystem::path& source_dir, const std::string& load_policy,
    const bool nvt_with_nhc, const bool remove_metadynamics_restart_state,
    const bool remove_sits_restart_state, const bool enable_sits,
    const bool enable_meta)
{
    PreparedCase prepared;
    prepared.root = temp_root / name;
    Copy_Directory_Contents(source_dir, prepared.root);
    std::filesystem::create_directories(prepared.root / "out");

    if (remove_metadynamics_restart_state)
    {
        Delete_H5_Object_If_Exists(prepared.root / "restart.spgr.h5",
                                   SpongeH5MD::path::restart_meta);
    }
    if (remove_sits_restart_state)
    {
        Delete_H5_Object_If_Exists(prepared.root / "restart.spgr.h5",
                                   SpongeH5MD::path::restart_sits);
    }
    if (enable_meta)
    {
        Enable_Meta_In_Restart_Protocol_Sidecar(prepared.root /
                                                "restart.spgr.h5");
    }
    if (nvt_with_nhc && load_policy == "dynamic" && !enable_sits)
    {
        Isolate_NHC_Dynamic_Runtime_Inputs(prepared.root);
    }

    std::string mdin = Read_Text(prepared.root / "mdin.bundled.spg.toml");
    Replace_All(&mdin, "input_h5_restart_load = \"full\"",
                "input_h5_restart_load = \"" + load_policy + "\"");
    Replace_All(&mdin, "output_h5_trajectory_path = \"prod.spg.h5md\"",
                "output_h5_trajectory_path = \"out/traj.spg.h5md\"");
    Replace_All(&mdin, "output_h5_restart_path = \"prod.spgr.h5\"",
                "output_h5_restart_path = \"out/restart.spgr.h5\"");
    Replace_All(&mdin, "output_h5_observable_path = \"prod.obs.spg.h5md\"",
                "output_h5_observable_path = \"out/obs.spg.h5md\"");
    Replace_All(&mdin, "mdout = \"mdout.txt\"", "mdout = \"out/mdout.txt\"");
    Replace_All(&mdin, "mdinfo = \"mdinfo.txt\"",
                "mdinfo = \"out/mdinfo.txt\"");

    if (nvt_with_nhc)
    {
        Replace_All(&mdin, "mode = \"rerun\"", "mode = \"nvt\"");
        const std::string step_limit = enable_sits ? "0" : "1";
        Replace_All(&mdin, "step_limit = 10",
                    "step_limit = " + step_limit +
                        "\n"
                        "dt = 0\n"
                        "thermostat = \"nose_hoover_chain\"\n"
                        "target_temperature = 300\n"
                        "write_mdout_interval = 1\n"
                        "write_trajectory_interval = 0\n"
                        "write_restart_file_interval = 1");
        mdin = Remove_Key_Lines(mdin, {"input_h5_trajectory_path",
                                       "input_h5_trajectory_particle_stream"});
        mdin += "\n[nose_hoover_chain]\nlength = 2\n";
    }
    else
    {
        Replace_All(&mdin, "step_limit = 10", "step_limit = 0");
    }
    if (enable_sits)
    {
        mdin +=
            "\n[SITS]\n"
            "mode = \"production\"\n"
            "k_numbers = 2\n"
            "T = \"300/310\"\n";
    }

    prepared.mdin = prepared.root / "mdin.restart_load.spg.toml";
    prepared.mdout = prepared.root / "out" / "mdout.txt";
    prepared.mdinfo = prepared.root / "out" / "mdinfo.txt";
    prepared.h5_restart = prepared.root / "out" / "restart.spgr.h5";
    Write_Text(prepared.mdin, mdin);
    return prepared;
}

PreparedCase Prepare_Portable_Rng_Runtime_Case(
    const std::filesystem::path& temp_root, const std::string& name,
    const std::filesystem::path& source_dir, const std::string& module,
    const SpongeH5MD::RestartRngState& rng_state, const int step_limit = 1)
{
    auto prepared =
        Prepare_Restart_Load_Case(temp_root, name, source_dir, "dynamic", true,
                                  false, false, false, false);
    std::string mdin = Read_Text(prepared.mdin);
    Replace_All(&mdin, "dt = 0", "dt = 0.001");
    Replace_All(&mdin, "step_limit = 1",
                "step_limit = " + std::to_string(step_limit));
    if (module == "middle_langevin")
    {
        Replace_All(&mdin, "thermostat = \"nose_hoover_chain\"",
                    "thermostat = \"middle_langevin\"\n"
                    "thermostat_tau = 0.001\n"
                    "thermostat_seed = 999");
    }
    else if (module == "andersen")
    {
        Replace_All(&mdin, "thermostat = \"nose_hoover_chain\"",
                    "thermostat = \"andersen\"\n"
                    "thermostat_tau = 0.001\n"
                    "thermostat_seed = 999");
    }
    else if (module == "bussi_thermostat")
    {
        Replace_All(&mdin, "thermostat = \"nose_hoover_chain\"",
                    "thermostat = \"bussi_thermostat\"\n"
                    "thermostat_tau = 0.1\n"
                    "thermostat_seed = 999");
    }
    else if (module == "pressure_based_barostat")
    {
        Replace_All(&mdin, "mode = \"nvt\"", "mode = \"npt\"");
        Replace_All(&mdin, "thermostat = \"nose_hoover_chain\"",
                    "thermostat = \"berendsen_thermostat\"\n"
                    "thermostat_tau = 0.1\n"
                    "target_pressure = 1000\n"
                    "barostat = \"andersen_barostat\"\n"
                    "barostat_tau = 0.1\n"
                    "barostat_update_interval = 1\n"
                    "barostat_seed = 999");
    }
    else if (module == "monte_carlo_barostat")
    {
        Replace_All(&mdin, "mode = \"nvt\"", "mode = \"npt\"");
        Replace_All(&mdin, "thermostat = \"nose_hoover_chain\"",
                    "thermostat = \"berendsen_thermostat\"\n"
                    "thermostat_tau = 1\n"
                    "target_pressure = 1\n"
                    "barostat = \"monte_carlo_barostat\"");
        mdin +=
            "\n[monte_carlo_barostat]\n"
            "seed = 999\n"
            "update_interval = 1\n"
            "check_interval = 10\n"
            "couple_dimension = \"XYZ\"\n";
    }
    Write_Text(prepared.mdin, mdin);

    const auto restart_path = prepared.root / "restart.spgr.h5";
    Install_Portable_Rng_Dynamic_State(
        restart_path, module, rng_state,
        module == "monte_carlo_barostat" || module == "pressure_based_barostat"
            ? "npt"
            : "nvt");
    if (module == "monte_carlo_barostat")
    {
        Install_Monte_Carlo_Adaptive_State(restart_path);
    }
    else if (module == "bussi_thermostat")
    {
        Install_Portable_Rng_Module_Float_State(restart_path, module, "lambda",
                                                {1.0f}, true);
    }
    else if (module == "pressure_based_barostat")
    {
        Install_Portable_Rng_Module_Float_State(
            restart_path, module, "g", {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
            false);
    }
    return prepared;
}

PreparedCase Prepare_Continuation_From_Checkpoint(
    const std::filesystem::path& temp_root, const std::string& name,
    const PreparedCase& producer)
{
    PreparedCase continuation;
    continuation.root = temp_root / name;
    Copy_Directory_Contents(producer.root, continuation.root);
    std::filesystem::remove_all(continuation.root / "out");
    std::filesystem::create_directories(continuation.root / "out");
    std::filesystem::copy_file(
        producer.h5_restart, continuation.root / "restart.spgr.h5",
        std::filesystem::copy_options::overwrite_existing);
    continuation.mdin = continuation.root / "mdin.restart_load.spg.toml";
    continuation.mdout = continuation.root / "out" / "mdout.txt";
    continuation.mdinfo = continuation.root / "out" / "mdinfo.txt";
    continuation.h5_restart = continuation.root / "out" / "restart.spgr.h5";
    return continuation;
}

SpongeH5MD::RestartDynamicState Read_Restart_Dynamic_State(
    const std::filesystem::path& restart_path)
{
    SpongeH5MD::RestartH5Reader reader;
    REQUIRE_TRUE(reader.Open(restart_path.string()));
    SpongeH5MD::RestartDynamicState state;
    REQUIRE_TRUE(reader.Read_Dynamic_State(&state));
    return state;
}

void Require_Portable_Rng_Continuation_Equivalent(
    const std::filesystem::path& continuous_restart,
    const std::filesystem::path& resumed_restart, const std::string& module)
{
    SpongeH5MD::RestartH5Reader continuous_reader;
    SpongeH5MD::RestartH5Reader resumed_reader;
    REQUIRE_TRUE(continuous_reader.Open(continuous_restart.string()));
    REQUIRE_TRUE(resumed_reader.Open(resumed_restart.string()));
    SpongeH5MD::RestartStructuralState continuous_structural;
    SpongeH5MD::RestartStructuralState resumed_structural;
    REQUIRE_TRUE(
        continuous_reader.Read_Structural_State(&continuous_structural));
    REQUIRE_TRUE(resumed_reader.Read_Structural_State(&resumed_structural));
    REQUIRE_EQ(continuous_structural.step, resumed_structural.step);
    REQUIRE_TRUE(std::fabs(continuous_structural.time -
                           resumed_structural.time) <= 1.0e-12);
    Require_Float_Container_Close(continuous_structural.position_xyz,
                                  resumed_structural.position_xyz);
    Require_Float_Container_Close(continuous_structural.velocity_xyz,
                                  resumed_structural.velocity_xyz);
    Require_Float_Container_Close(continuous_structural.box_edges,
                                  resumed_structural.box_edges);

    SpongeH5MD::RestartDynamicState continuous_dynamic;
    SpongeH5MD::RestartDynamicState resumed_dynamic;
    REQUIRE_TRUE(continuous_reader.Read_Dynamic_State(&continuous_dynamic));
    REQUIRE_TRUE(resumed_reader.Read_Dynamic_State(&resumed_dynamic));
    const auto& continuous_rng = continuous_dynamic.rng_states.at(module);
    const auto& resumed_rng = resumed_dynamic.rng_states.at(module);
    REQUIRE_EQ(continuous_rng.engine, resumed_rng.engine);
    REQUIRE_EQ(continuous_rng.state_schema_version,
               resumed_rng.state_schema_version);
    REQUIRE_EQ(continuous_rng.stream_count, resumed_rng.stream_count);
    REQUIRE_EQ(continuous_rng.words_per_stream, resumed_rng.words_per_stream);
    REQUIRE_EQ(continuous_rng.state_words, resumed_rng.state_words);
    if (module == "monte_carlo_barostat")
    {
        const auto& continuous_floats =
            continuous_dynamic.barostat_float_states.at(module);
        const auto& resumed_floats =
            resumed_dynamic.barostat_float_states.at(module);
        REQUIRE_EQ(continuous_floats.size(), resumed_floats.size());
        for (const auto& item : continuous_floats)
        {
            Require_Float_Container_Close(item.second,
                                          resumed_floats.at(item.first));
        }
        REQUIRE_EQ(continuous_dynamic.barostat_integer_states.at(module),
                   resumed_dynamic.barostat_integer_states.at(module));
    }
    else if (module == "bussi_thermostat")
    {
        Require_Float_Container_Close(
            continuous_dynamic.thermostat_float_states.at(module).at("lambda"),
            resumed_dynamic.thermostat_float_states.at(module).at("lambda"),
            1.0e-4f);
    }
    else if (module == "pressure_based_barostat")
    {
        Require_Float_Container_Close(
            continuous_dynamic.barostat_float_states.at(module).at("g"),
            resumed_dynamic.barostat_float_states.at(module).at("g"), 5.0e-4f);
    }
}

void Refresh_Restart_State_Hash(const std::filesystem::path& restart_path)
{
#ifdef _WIN32
    // Legacy sidecar text requires native line endings. HDF5 payloads do
    // not receive the text-mode conversion that Write_Text provides.
    {
        HighFive::File file(restart_path.string(), HighFive::File::ReadWrite);
        const auto path = SpongeH5MD::path::restart_protocol_sidecars;
        if (file.exist(path))
        {
            const auto group = file.getGroup(path);
            for (const auto& name : group.listObjectNames())
            {
                auto dataset = group.getDataSet(name);
                std::string text;
                dataset.read(text);
                Replace_All(&text, "\r\n", "\n");
                Replace_All(&text, "\n", "\r\n");
                dataset.write(text);
            }
        }
    }
#endif
    std::string state_hash;
    {
        SpongeH5MD::RestartH5Reader reader;
        REQUIRE_TRUE(reader.Open(restart_path.string()));
        REQUIRE_TRUE(reader.Compute_State_Hash(&state_hash));
    }
    Write_H5_String_Overwrite(restart_path, SpongeH5MD::path::run_state_hash,
                              state_hash);
}

std::filesystem::path Run_SPONGE(const std::filesystem::path& executable,
                                 const PreparedCase& test_case)
{
    Refresh_Restart_State_Hash(test_case.root / "restart.spgr.h5");
    const auto log_path = test_case.root / "sponge.stdout.txt";
    const std::string command = Shell_Quote(executable) + " -mdin " +
                                Shell_Quote(test_case.mdin) + " > " +
                                Shell_Quote(log_path) + " 2>&1";
    const int ret = Run_Test_Shell_Command(command);
    if (ret != 0)
    {
        throw TestFailure("SPONGE restart-load smoke failed for " +
                          test_case.root.filename().string() + " (exit " +
                          std::to_string(ret) + ")\n" + Read_Text(log_path));
    }
    SpongeH5InputMatrix::Require_Path_Exists(test_case.mdout);
    SpongeH5InputMatrix::Require_Path_Exists(test_case.mdinfo);
    return log_path;
}

void Run_SPONGE_Expect_Failure(const std::filesystem::path& executable,
                               const PreparedCase& test_case,
                               const std::string& expected_error)
{
    Refresh_Restart_State_Hash(test_case.root / "restart.spgr.h5");
    const auto log_path = test_case.root / "sponge.stdout.txt";
    const std::string command = Shell_Quote(executable) + " -mdin " +
                                Shell_Quote(test_case.mdin) + " > " +
                                Shell_Quote(log_path) + " 2>&1";
    const int ret = Run_Test_Shell_Command(command);
    REQUIRE_TRUE(ret != 0);
    Require_Contains(Read_Text(log_path), expected_error);
}

void Require_Restart_Contains(const std::filesystem::path& h5_path,
                              const std::string& object_path)
{
    SpongeH5InputMatrix::Require_Path_Exists(h5_path);
    HighFive::File file(h5_path.string(), HighFive::File::ReadOnly);
    REQUIRE_TRUE(file.exist(object_path));
}

std::vector<float> Read_H5_Float_Dataset(const std::filesystem::path& h5_path,
                                         const std::string& object_path)
{
    HighFive::File file(h5_path.string(), HighFive::File::ReadOnly);
    const auto dataset = file.getDataSet(object_path);
    const auto dimensions = dataset.getSpace().getDimensions();
    std::size_t value_count = 1;
    for (const auto dimension : dimensions)
    {
        value_count *= dimension;
    }
    std::vector<float> values(value_count);
    REQUIRE_TRUE(H5Dread(dataset.getId(), H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                         H5P_DEFAULT, values.data()) >= 0);
    return values;
}

std::vector<std::size_t> Read_H5_Dataset_Dimensions(
    const std::filesystem::path& h5_path, const std::string& object_path)
{
    HighFive::File file(h5_path.string(), HighFive::File::ReadOnly);
    return file.getDataSet(object_path).getSpace().getDimensions();
}

void Require_Finite_Values(const std::vector<float>& values)
{
    REQUIRE_TRUE(!values.empty());
    for (const float value : values)
    {
        REQUIRE_TRUE(SpongeFloat::Is_Finite(value));
    }
}

void Require_No_Nonfinite_Text(const std::filesystem::path& path)
{
    std::istringstream stream(Read_Text(path));
    std::string token;
    while (stream >> token)
    {
        std::transform(token.begin(), token.end(), token.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        REQUIRE_TRUE(token.find("nan") == std::string::npos);
        REQUIRE_TRUE(token.find("inf") == std::string::npos);
    }
}

void Require_Runtime_Smoke_Enabled()
{
    const char* enabled = std::getenv("SPONGE_H5_ENABLE_RUNTIME_SMOKE");
    if (enabled == nullptr || std::string(enabled) != "1")
    {
        std::cerr << "Skipping restart-load runtime closure smoke; set "
                     "SPONGE_H5_ENABLE_RUNTIME_SMOKE=1 in a runnable SPONGE "
                     "CPU/GPU environment to enable it.\n";
        std::exit(kSkipReturnCode);
    }
}

void Run_Restart_Load_Runtime_Closure(
    const std::filesystem::path& sponge_executable)
{
    const auto temp_root =
        SpongeH5Test::Unique_Temp_Path("h5_restart_load_runtime_closure");
    std::filesystem::create_directories(temp_root);

    const auto full = SpongeH5InputMatrix::Full_Contract_Rerun_Path();
    const auto pure_source = full / "bundled_input" / "bundle";
    const auto sidecar_source =
        full / "bundled_input_with_legacy_sidecar" / "bundle";

    const auto protocol_sits = Prepare_Restart_Load_Case(
        temp_root, "restart_load_protocol_sidecar_sits_only", sidecar_source,
        "protocol", false, true, false, true, false);
    const auto protocol_sits_log = Run_SPONGE(sponge_executable, protocol_sits);
    SpongeH5InputMatrix::Require_Path_Exists(protocol_sits.root /
                                             ".sponge_h5_restart_protocol" /
                                             "SITS_nk_in_file.txt");
    Require_Contains(Read_Text(protocol_sits_log), "START INITIALIZING SITS");
    Require_Contains(Read_Text(protocol_sits_log),
                     "Read Nk from .sponge_h5_restart_protocol/"
                     "SITS_nk_in_file.txt");

    const std::vector<float> native_sits_nk = {1.25f, 2.5f};
    const std::vector<float> native_sits_log_norm = {-2.5f, -1.25f};
    // Deliberately differ from log(nk): equality after the run proves the
    // module-level apply hook restored the typed device buffer rather than
    // merely rebuilding it from nk during initialization.
    const std::vector<float> native_sits_log_nk = {0.125f, 0.875f};
    const auto protocol_sits_native = Prepare_Restart_Load_Case(
        temp_root, "restart_load_protocol_native_sits_full_state", pure_source,
        "protocol", true, true, false, true, false);
    {
        std::string mdin = Read_Text(protocol_sits_native.mdin);
        Replace_All(&mdin, "step_limit = 0", "step_limit = 1");
        Write_Text(protocol_sits_native.mdin, mdin);
    }
    Isolate_Sits_Protocol_Runtime_Inputs(protocol_sits_native.root);
    Delete_H5_Object_If_Exists(protocol_sits_native.root / "restart.spgr.h5",
                               SpongeH5MD::path::restart_protocol_sidecars);
    Delete_H5_Object_If_Exists(protocol_sits_native.root / "restart.spgr.h5",
                               SpongeH5MD::path::restart_references);
    Install_Full_Sits_Restart_State(
        protocol_sits_native.root / "restart.spgr.h5", native_sits_nk,
        native_sits_log_norm, native_sits_log_nk);
    const auto protocol_sits_native_log =
        Run_SPONGE(sponge_executable, protocol_sits_native);
    Require_Contains(Read_Text(protocol_sits_native_log),
                     "Read Nk from native H5 restart");
    REQUIRE_TRUE(!std::filesystem::exists(protocol_sits_native.root /
                                          ".sponge_h5_restart_protocol"));
    REQUIRE_EQ(Read_H5_Float_Dataset(
                   protocol_sits_native.h5_restart,
                   SpongeH5MD::Restart_Sits_State_Path("SITS", "nk")),
               native_sits_nk);
    REQUIRE_EQ(Read_H5_Float_Dataset(
                   protocol_sits_native.h5_restart,
                   SpongeH5MD::Restart_Sits_State_Path("SITS", "log_norm")),
               native_sits_log_norm);
    REQUIRE_EQ(Read_H5_Float_Dataset(
                   protocol_sits_native.h5_restart,
                   SpongeH5MD::Restart_Sits_State_Path("SITS", "log_nk")),
               native_sits_log_nk);

    const auto dynamic_nhc = Prepare_Restart_Load_Case(
        temp_root, "restart_load_dynamic_supported_nhc_nvt", pure_source,
        "dynamic", true, false, false, false, false);
    const auto dynamic_nhc_log = Run_SPONGE(sponge_executable, dynamic_nhc);
    Require_Contains(Read_Text(dynamic_nhc_log),
                     "START INITIALIZING NOSE HOOVER CHAIN");
    Require_No_Nonfinite_Text(dynamic_nhc.mdout);
    Require_Restart_Contains(dynamic_nhc.h5_restart,
                             SpongeH5MD::path::restart_nhc);
    const auto input_nhc = Read_H5_Float_Dataset(
        dynamic_nhc.root / "restart.spgr.h5", SpongeH5MD::path::restart_nhc);
    const auto output_nhc = Read_H5_Float_Dataset(
        dynamic_nhc.h5_restart, SpongeH5MD::path::restart_nhc);
    Require_Finite_Values(input_nhc);
    Require_Finite_Values(output_nhc);
    REQUIRE_TRUE(std::any_of(input_nhc.begin(), input_nhc.end(),
                             [](const float value) { return value != 0.0f; }));
    REQUIRE_EQ(input_nhc, output_nhc);

    const auto middle_langevin = Prepare_Portable_Rng_Runtime_Case(
        temp_root, "restart_load_dynamic_middle_langevin", pure_source,
        "middle_langevin", SpongeRestartRng::Counter_Philox_State(1234, 7));
    const auto middle_langevin_log =
        Run_SPONGE(sponge_executable, middle_langevin);
    Require_Contains(Read_Text(middle_langevin_log),
                     "START INITIALIZING MIDDLE LANGEVIN DYNAMICS");
    const auto middle_output =
        Read_Restart_Dynamic_State(middle_langevin.h5_restart);
    std::uint64_t restored_seed = 0;
    std::uint64_t restored_invocation = 0;
    std::string rng_error;
    REQUIRE_TRUE(SpongeRestartRng::Decode_Counter_Philox_State(
        middle_output.rng_states.at("middle_langevin"), &restored_seed,
        &restored_invocation, &rng_error));
    REQUIRE_EQ(restored_seed, static_cast<std::uint64_t>(1234));
    REQUIRE_EQ(restored_invocation, static_cast<std::uint64_t>(8));
    const auto middle_continuous = Prepare_Portable_Rng_Runtime_Case(
        temp_root, "restart_load_dynamic_middle_langevin_continuous",
        pure_source, "middle_langevin",
        SpongeRestartRng::Counter_Philox_State(1234, 7), 2);
    Run_SPONGE(sponge_executable, middle_continuous);
    const auto middle_resumed = Prepare_Continuation_From_Checkpoint(
        temp_root, "restart_load_dynamic_middle_langevin_resumed",
        middle_langevin);
    Run_SPONGE(sponge_executable, middle_resumed);
    Require_Portable_Rng_Continuation_Equivalent(middle_continuous.h5_restart,
                                                 middle_resumed.h5_restart,
                                                 "middle_langevin");

    const auto andersen = Prepare_Portable_Rng_Runtime_Case(
        temp_root, "restart_load_dynamic_andersen", pure_source, "andersen",
        SpongeRestartRng::Counter_Philox_State(4321, 11));
    const auto andersen_log = Run_SPONGE(sponge_executable, andersen);
    Require_Contains(Read_Text(andersen_log),
                     "START INITIALIZING ANDERSEN THERMOSTAT");
    const auto andersen_output =
        Read_Restart_Dynamic_State(andersen.h5_restart);
    REQUIRE_TRUE(SpongeRestartRng::Decode_Counter_Philox_State(
        andersen_output.rng_states.at("andersen"), &restored_seed,
        &restored_invocation, &rng_error));
    REQUIRE_EQ(restored_seed, static_cast<std::uint64_t>(4321));
    REQUIRE_EQ(restored_invocation, static_cast<std::uint64_t>(12));
    const auto andersen_continuous = Prepare_Portable_Rng_Runtime_Case(
        temp_root, "restart_load_dynamic_andersen_continuous", pure_source,
        "andersen", SpongeRestartRng::Counter_Philox_State(4321, 11), 2);
    Run_SPONGE(sponge_executable, andersen_continuous);
    const auto andersen_resumed = Prepare_Continuation_From_Checkpoint(
        temp_root, "restart_load_dynamic_andersen_resumed", andersen);
    Run_SPONGE(sponge_executable, andersen_resumed);
    Require_Portable_Rng_Continuation_Equivalent(andersen_continuous.h5_restart,
                                                 andersen_resumed.h5_restart,
                                                 "andersen");

    const auto bussi = Prepare_Portable_Rng_Runtime_Case(
        temp_root, "restart_load_dynamic_bussi_thermostat", pure_source,
        "bussi_thermostat", SpongeRestartRng::Counter_Philox_State(2468, 13));
    const auto bussi_log = Run_SPONGE(sponge_executable, bussi);
    Require_Contains(Read_Text(bussi_log),
                     "START INITIALIZING BUSSI THERMOSTAT");
    const auto bussi_output = Read_Restart_Dynamic_State(bussi.h5_restart);
    REQUIRE_TRUE(SpongeRestartRng::Decode_Counter_Philox_State(
        bussi_output.rng_states.at("bussi_thermostat"), &restored_seed,
        &restored_invocation, &rng_error));
    REQUIRE_EQ(restored_seed, static_cast<std::uint64_t>(2468));
    REQUIRE_EQ(restored_invocation, static_cast<std::uint64_t>(14));
    const auto bussi_continuous = Prepare_Portable_Rng_Runtime_Case(
        temp_root, "restart_load_dynamic_bussi_thermostat_continuous",
        pure_source, "bussi_thermostat",
        SpongeRestartRng::Counter_Philox_State(2468, 13), 2);
    Run_SPONGE(sponge_executable, bussi_continuous);
    const auto bussi_resumed = Prepare_Continuation_From_Checkpoint(
        temp_root, "restart_load_dynamic_bussi_thermostat_resumed", bussi);
    Run_SPONGE(sponge_executable, bussi_resumed);
    Require_Portable_Rng_Continuation_Equivalent(bussi_continuous.h5_restart,
                                                 bussi_resumed.h5_restart,
                                                 "bussi_thermostat");

    const auto pressure_barostat = Prepare_Portable_Rng_Runtime_Case(
        temp_root, "restart_load_dynamic_pressure_based_barostat", pure_source,
        "pressure_based_barostat",
        SpongeRestartRng::Counter_Philox_State(8642, 17));
    const auto pressure_barostat_log =
        Run_SPONGE(sponge_executable, pressure_barostat);
    Require_Contains(Read_Text(pressure_barostat_log),
                     "START INITIALIZING PRESSURE BASED BAROSTAT");
    const auto pressure_barostat_output =
        Read_Restart_Dynamic_State(pressure_barostat.h5_restart);
    REQUIRE_TRUE(SpongeRestartRng::Decode_Counter_Philox_State(
        pressure_barostat_output.rng_states.at("pressure_based_barostat"),
        &restored_seed, &restored_invocation, &rng_error));
    REQUIRE_EQ(restored_seed, static_cast<std::uint64_t>(8642));
    REQUIRE_EQ(restored_invocation, static_cast<std::uint64_t>(18));
    const auto pressure_barostat_continuous = Prepare_Portable_Rng_Runtime_Case(
        temp_root, "restart_load_dynamic_pressure_based_barostat_continuous",
        pure_source, "pressure_based_barostat",
        SpongeRestartRng::Counter_Philox_State(8642, 17), 2);
    Run_SPONGE(sponge_executable, pressure_barostat_continuous);
    const auto pressure_barostat_resumed = Prepare_Continuation_From_Checkpoint(
        temp_root, "restart_load_dynamic_pressure_based_barostat_resumed",
        pressure_barostat);
    Run_SPONGE(sponge_executable, pressure_barostat_resumed);
    Require_Portable_Rng_Continuation_Equivalent(
        pressure_barostat_continuous.h5_restart,
        pressure_barostat_resumed.h5_restart, "pressure_based_barostat");

    const std::uint64_t mc_initial_state = 0x123456789abcdef0ULL;
    const auto monte_carlo = Prepare_Portable_Rng_Runtime_Case(
        temp_root, "restart_load_dynamic_monte_carlo_barostat", pure_source,
        "monte_carlo_barostat",
        SpongeRestartRng::Splitmix64_State(mc_initial_state));
    const auto monte_carlo_log = Run_SPONGE(sponge_executable, monte_carlo);
    Require_Contains(Read_Text(monte_carlo_log),
                     "START INITIALIZING MC BAROSTAT");
    const auto monte_carlo_output =
        Read_Restart_Dynamic_State(monte_carlo.h5_restart);
    std::uint64_t mc_output_state = 0;
    REQUIRE_TRUE(SpongeRestartRng::Decode_Splitmix64_State(
        monte_carlo_output.rng_states.at("monte_carlo_barostat"),
        &mc_output_state, &rng_error));
    REQUIRE_TRUE(mc_output_state != mc_initial_state);
    REQUIRE_EQ(
        monte_carlo_output.barostat_integer_states.at("monte_carlo_barostat")
            .at("total_count_int64")[0],
        static_cast<std::int64_t>(11));
    REQUIRE_EQ(
        monte_carlo_output.barostat_integer_states.at("monte_carlo_barostat")
            .at("total_count_int64")[1],
        static_cast<std::int64_t>(20));
    REQUIRE_EQ(
        monte_carlo_output.barostat_integer_states.at("monte_carlo_barostat")
            .at("total_count_int64")[2],
        static_cast<std::int64_t>(30));
    const auto monte_carlo_continuous = Prepare_Portable_Rng_Runtime_Case(
        temp_root, "restart_load_dynamic_monte_carlo_barostat_continuous",
        pure_source, "monte_carlo_barostat",
        SpongeRestartRng::Splitmix64_State(mc_initial_state), 2);
    Run_SPONGE(sponge_executable, monte_carlo_continuous);
    const auto monte_carlo_resumed = Prepare_Continuation_From_Checkpoint(
        temp_root, "restart_load_dynamic_monte_carlo_barostat_resumed",
        monte_carlo);
    Run_SPONGE(sponge_executable, monte_carlo_resumed);
    Require_Portable_Rng_Continuation_Equivalent(
        monte_carlo_continuous.h5_restart, monte_carlo_resumed.h5_restart,
        "monte_carlo_barostat");

    const auto full_supported = Prepare_Restart_Load_Case(
        temp_root, "restart_load_full_supported_nhc_sits", sidecar_source,
        "full", true, true, false, true, false);
    const auto full_supported_log =
        Run_SPONGE(sponge_executable, full_supported);
    Require_Contains(Read_Text(full_supported_log),
                     "START INITIALIZING NOSE HOOVER CHAIN");
    Require_Contains(Read_Text(full_supported_log), "START INITIALIZING SITS");
    Require_Contains(Read_Text(full_supported_log),
                     "Read Nk from .sponge_h5_restart_protocol/"
                     "SITS_nk_in_file.txt");

    const auto protocol_restraint = Prepare_Restart_Load_Case(
        temp_root, "restart_load_protocol_native_restraint", pure_source,
        "protocol", true, true, true, false, false);
    Isolate_Positional_Restraint_Runtime_Inputs(protocol_restraint.root);
    Delete_H5_Object_If_Exists(protocol_restraint.root / "restart.spgr.h5",
                               SpongeH5MD::path::restart_protocol_sidecars);
    Write_Text(
        protocol_restraint.mdin,
        Remove_Key_Lines(Read_Text(protocol_restraint.mdin),
                         {"custom_pair_in_file", "custom_bond_in_file"}));
    const auto protocol_restraint_log =
        Run_SPONGE(sponge_executable, protocol_restraint);
    Require_Contains(Read_Text(protocol_restraint_log),
                     "START INITIALIZING RESTRAIN");
    const std::string restraint_reference_path =
        SpongeH5MD::Restart_Restraint_Reference_Coordinate_Path("default");
    Require_Restart_Contains(protocol_restraint.h5_restart,
                             restraint_reference_path);
    const auto input_restraint_reference = Read_H5_Float_Dataset(
        protocol_restraint.root / "restart.spgr.h5", restraint_reference_path);
    const auto output_restraint_reference = Read_H5_Float_Dataset(
        protocol_restraint.h5_restart, restraint_reference_path);
    REQUIRE_EQ(input_restraint_reference, output_restraint_reference);

    const auto native_cv = Prepare_Restart_Load_Case(
        temp_root, "restart_load_protocol_native_cv", pure_source, "protocol",
        true, true, true, false, false);
    Isolate_And_Install_Native_CV_Runtime_Inputs(native_cv.root);
    Write_Text(native_cv.mdin, Remove_Key_Lines(Read_Text(native_cv.mdin),
                                                {"custom_pair_in_file",
                                                 "custom_bond_in_file"}));
    const auto native_cv_log = Run_SPONGE(sponge_executable, native_cv);
    Require_Contains(Read_Text(native_cv_log), "1 CV defined");
    Require_Contains(Read_Text(native_cv_log),
                     "START INITIALIZING RESTRAIN CV");
    REQUIRE_TRUE(!std::filesystem::exists(
        native_cv.root / ".sponge_h5_native_protocol" / "cv.txt"));
    const std::string cv_reference_path =
        SpongeH5MD::Restart_CV_Reference_Coordinate_Path("backbone");
    Require_Restart_Contains(native_cv.h5_restart, cv_reference_path);
    REQUIRE_EQ(Read_H5_Float_Dataset(native_cv.root / "restart.spgr.h5",
                                     cv_reference_path),
               Read_H5_Float_Dataset(native_cv.h5_restart, cv_reference_path));

    const auto native_metadynamics = Prepare_Restart_Load_Case(
        temp_root, "restart_load_protocol_native_metadynamics", pure_source,
        "protocol", true, false, true, false, false);
    Isolate_And_Install_Native_Metadynamics_Runtime_Inputs(
        native_metadynamics.root);
    Write_Text(
        native_metadynamics.mdin,
        Remove_Key_Lines(Read_Text(native_metadynamics.mdin),
                         {"custom_pair_in_file", "custom_bond_in_file"}));
    const auto native_metadynamics_log =
        Run_SPONGE(sponge_executable, native_metadynamics);
    Require_Contains(Read_Text(native_metadynamics_log),
                     "START INITIALIZING 1D-META");
    Require_Contains(Read_Text(native_metadynamics_log),
                     "Applied typed H5 metadynamics state: native_bias");
    Require_Restart_Contains(
        native_metadynamics.root / "out" / "obs.spg.h5md",
        "/parameters/sponge/metadynamics/native_bias/hills");
    Require_Restart_Contains(native_metadynamics.h5_restart,
                             SpongeH5MD::Restart_Metad_State_Path(
                                 "native_bias", "state_schema_version"));
    Require_Restart_Contains(
        native_metadynamics.h5_restart,
        SpongeH5MD::Restart_Metad_State_Path("native_bias", "potential/value"));
    Require_Restart_Contains(
        native_metadynamics.h5_restart,
        SpongeH5MD::Restart_Metad_State_Path("native_bias", "hills/center"));
    Require_Finite_Values(Read_H5_Float_Dataset(
        native_metadynamics.h5_restart, SpongeH5MD::Restart_Metad_State_Path(
                                            "native_bias", "potential/value")));
    REQUIRE_TRUE(!std::filesystem::exists(
        native_metadynamics.root / ".sponge_h5_native_protocol" / "cv.txt"));

    const auto scatter_sink_metadynamics = Prepare_Restart_Load_Case(
        temp_root, "restart_load_protocol_typed_metadynamics_scatter_sink",
        pure_source, "protocol", true, false, true, false, false);
    Isolate_And_Install_Native_Metadynamics_Runtime_Inputs(
        scatter_sink_metadynamics.root);
    Convert_Metadynamics_Runtime_Inputs_To_Scatter_Sink(
        scatter_sink_metadynamics.root);
    Write_Text(
        scatter_sink_metadynamics.mdin,
        Remove_Key_Lines(Read_Text(scatter_sink_metadynamics.mdin),
                         {"custom_pair_in_file", "custom_bond_in_file"}));
    const auto scatter_sink_log =
        Run_SPONGE(sponge_executable, scatter_sink_metadynamics);
    Require_Contains(Read_Text(scatter_sink_log), "Use 3 scatter point for CV");
    Require_Contains(Read_Text(scatter_sink_log),
                     "reading sink/submarine dimension for meta: 1");
    Require_Contains(Read_Text(scatter_sink_log),
                     "Applied typed H5 metadynamics state: meta");
    Require_No_Nonfinite_Text(scatter_sink_metadynamics.mdout);

    const std::string scatter_root =
        SpongeH5MD::Restart_Metad_State_Root("meta") + "/scatter";
    const std::string edge_root =
        SpongeH5MD::Restart_Metad_State_Root("meta") + "/edge";
    const std::string hills_root =
        SpongeH5MD::Restart_Metad_State_Root("meta") + "/hills";
    Require_Restart_Contains(scatter_sink_metadynamics.h5_restart,
                             scatter_root + "/position");
    Require_Restart_Contains(scatter_sink_metadynamics.h5_restart,
                             scatter_root + "/potential");
    Require_Restart_Contains(scatter_sink_metadynamics.h5_restart,
                             scatter_root + "/force");
    Require_Restart_Contains(scatter_sink_metadynamics.h5_restart,
                             edge_root + "/log_normalization");
    Require_Restart_Contains(scatter_sink_metadynamics.h5_restart,
                             edge_root + "/normal_force");
    Require_Restart_Contains(scatter_sink_metadynamics.h5_restart,
                             hills_root + "/sink");
    REQUIRE_EQ(Read_H5_Float_Dataset(scatter_sink_metadynamics.h5_restart,
                                     scatter_root + "/position"),
               std::vector<float>({1.0f, 4.0f, 8.0f}));
    REQUIRE_EQ(Read_H5_Float_Dataset(scatter_sink_metadynamics.h5_restart,
                                     edge_root + "/log_normalization"),
               std::vector<float>({0.0f, 0.0f, 0.0f, 0.0f}));
    REQUIRE_EQ(Read_H5_Float_Dataset(scatter_sink_metadynamics.h5_restart,
                                     edge_root + "/normal_force"),
               std::vector<float>({0.0f, 0.0f, 0.0f, 0.0f}));
    REQUIRE_EQ(Read_H5_Float_Dataset(scatter_sink_metadynamics.h5_restart,
                                     hills_root + "/sink"),
               std::vector<float>({0.75f}));
    const auto scatter_potential = Read_H5_Float_Dataset(
        scatter_sink_metadynamics.h5_restart, scatter_root + "/potential");
    const auto scatter_force = Read_H5_Float_Dataset(
        scatter_sink_metadynamics.h5_restart, scatter_root + "/force");
    Require_Finite_Values(scatter_potential);
    Require_Finite_Values(scatter_force);
    REQUIRE_EQ(scatter_potential.size(), static_cast<std::size_t>(3));
    REQUIRE_EQ(scatter_force.size(), static_cast<std::size_t>(3));
    const auto hill_dimensions = Read_H5_Dataset_Dimensions(
        scatter_sink_metadynamics.h5_restart, hills_root + "/center");
    REQUIRE_EQ(hill_dimensions.size(), static_cast<std::size_t>(2));
    REQUIRE_TRUE(hill_dimensions[0] >= 1);
    REQUIRE_EQ(hill_dimensions[1], static_cast<std::size_t>(1));
    Require_Finite_Values(Read_H5_Float_Dataset(
        scatter_sink_metadynamics.h5_restart, hills_root + "/center"));
    Require_Finite_Values(
        Read_H5_Float_Dataset(scatter_sink_metadynamics.h5_restart,
                              SpongeH5MD::Restart_Metad_State_Root("meta") +
                                  "/runtime/potential_max"));

    const auto native_metadynamics_mismatch = Prepare_Restart_Load_Case(
        temp_root, "restart_load_protocol_native_metadynamics_grid_reject",
        pure_source, "protocol", true, false, true, false, false);
    Isolate_And_Install_Native_Metadynamics_Runtime_Inputs(
        native_metadynamics_mismatch.root);
    Write_Text(
        native_metadynamics_mismatch.mdin,
        Remove_Key_Lines(Read_Text(native_metadynamics_mismatch.mdin),
                         {"custom_pair_in_file", "custom_bond_in_file"}));
    {
        HighFive::File restart(
            (native_metadynamics_mismatch.root / "restart.spgr.h5").string(),
            HighFive::File::ReadWrite);
        const std::vector<float> mismatched_maximum = {9.0f};
        restart
            .getDataSet(
                SpongeH5MD::Restart_Metad_State_Path("native_bias", "grid/max"))
            .write(mismatched_maximum);
    }
    Run_SPONGE_Expect_Failure(
        sponge_executable, native_metadynamics_mismatch,
        "metadynamics restart grid does not match the active protocol");

    const auto protocol_meta = Prepare_Restart_Load_Case(
        temp_root, "restart_load_protocol_sidecar_meta_initialized",
        sidecar_source, "protocol", false, false, true, false, true);
    const auto protocol_meta_log = Run_SPONGE(sponge_executable, protocol_meta);
    Require_Contains(Read_Text(protocol_meta_log),
                     "START INITIALIZING 1D-META");
    SpongeH5InputMatrix::Require_Path_Exists(protocol_meta.root / "myhill.log");
    SpongeH5InputMatrix::Require_Path_Exists(protocol_meta.root /
                                             "Meta_Potential.txt");

    const auto dynamic_rerun_reject = Prepare_Restart_Load_Case(
        temp_root, "restart_load_dynamic_rerun_without_nhc_rejects",
        pure_source, "dynamic", false, false, false, false, false);
    Run_SPONGE_Expect_Failure(
        sponge_executable, dynamic_rerun_reject,
        "Restart contains Nose-Hoover chain state, but the "
        "nose_hoover_chain thermostat is not initialized");

    const auto protocol_meta_reject = Prepare_Restart_Load_Case(
        temp_root, "restart_load_protocol_meta_without_module_rejects",
        sidecar_source, "protocol", false, false, false, false, false);
    Run_SPONGE_Expect_Failure(
        sponge_executable, protocol_meta_reject,
        "Restart contains metadynamics state, but the meta module is not "
        "initialized");

    const auto pure_protocol_custom_force = Prepare_Restart_Load_Case(
        temp_root, "restart_load_protocol_pure_bundled_custom_force_native",
        pure_source, "protocol", false, true, false, true, false);
    Install_Valid_Native_EAM_Atom_Types(pure_protocol_custom_force.root /
                                        "topology.spgt.h5");
    const auto pure_protocol_custom_force_log =
        Run_SPONGE(sponge_executable, pure_protocol_custom_force);
    REQUIRE_TRUE(!std::filesystem::exists(pure_protocol_custom_force.root /
                                          ".sponge_h5_native_custom_force"));
    REQUIRE_TRUE(!std::filesystem::exists(pure_protocol_custom_force.root /
                                          ".sponge_h5_native_qc"));
    REQUIRE_TRUE(!std::filesystem::exists(pure_protocol_custom_force.root /
                                          ".sponge_h5_native_manybody" /
                                          "sw.txt"));
    REQUIRE_TRUE(!std::filesystem::exists(pure_protocol_custom_force.root /
                                          ".sponge_h5_native_manybody" /
                                          "tersoff.txt"));
    REQUIRE_TRUE(!std::filesystem::exists(pure_protocol_custom_force.root /
                                          ".sponge_h5_native_manybody" /
                                          "eam.txt"));
    REQUIRE_TRUE(!std::filesystem::exists(pure_protocol_custom_force.root /
                                          ".sponge_h5_native_manybody" /
                                          "eam_atom_type.txt"));
    REQUIRE_TRUE(!std::filesystem::exists(pure_protocol_custom_force.root /
                                          ".sponge_h5_native_manybody" /
                                          "reaxff.txt"));
    REQUIRE_TRUE(!std::filesystem::exists(pure_protocol_custom_force.root /
                                          ".sponge_h5_native_manybody" /
                                          "reaxff_type.txt"));
    Require_Contains(Read_Text(pure_protocol_custom_force_log),
                     "START INITIALIZING PAIRWISE FORCE FROM NATIVE H5");
    Require_Contains(Read_Text(pure_protocol_custom_force_log),
                     "START INITIALIZING LISTED FORCES FROM NATIVE H5");
    Require_Contains(
        Read_Text(pure_protocol_custom_force_log),
        "START INITIALIZING STILLINGER WEBER FORCE FROM NATIVE H5");
    Require_Contains(Read_Text(pure_protocol_custom_force_log),
                     "START INITIALIZING TERSOFF FORCE FROM NATIVE H5");
    Require_Contains(Read_Text(pure_protocol_custom_force_log),
                     "START INITIALIZING REAXFF_EEQ (native H5)");
    Require_Contains(Read_Text(pure_protocol_custom_force_log),
                     "START INITIALIZING EAM FORCE FROM NATIVE H5");
    Require_Contains(Read_Text(pure_protocol_custom_force_log),
                     "START INITIALIZING SITS");

    std::filesystem::remove_all(temp_root);
}
}  // namespace

int main(int argc, char** argv)
{
    try
    {
        Require_Runtime_Smoke_Enabled();
        REQUIRE_TRUE(argc >= 2);
        Run_Restart_Load_Runtime_Closure(argv[1]);
    }
    catch (const std::exception& err)
    {
        std::cerr << err.what() << "\n";
        return 1;
    }
    return 0;
}

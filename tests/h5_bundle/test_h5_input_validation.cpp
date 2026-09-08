#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <highfive/highfive.hpp>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "h5_bundle_test_common.hpp"
#include "h5_input_matrix_fixture.hpp"
#include "utils/h5md/highfive_backend.hpp"
#include "utils/h5md/input_validation.hpp"
#include "utils/h5md/protocol_cv_h5.hpp"
#include "utils/h5md/protocol_metadynamics_h5.hpp"
#include "utils/h5md/protocol_restraint_h5.hpp"
#include "utils/h5md/protocol_steer_h5.hpp"
#include "utils/h5md/restart_h5_writer.hpp"
#include "utils/h5md/topology_custom_force_h5_materializer.hpp"
#include "utils/h5md/topology_manybody_h5_materializer.hpp"
#include "utils/h5md/trajectory_h5_writer.hpp"
#include "xponge/load/native/hard_wall_h5.hpp"
#include "xponge/load/native/restraint_h5.hpp"

using namespace SpongeH5Test;
using namespace SpongeH5MD;

static void Ensure_Group(HighFive::File& file, const std::string& group_path)
{
    if (group_path.empty() || group_path == "/")
    {
        return;
    }
    std::string current;
    std::size_t begin = 1;
    while (begin < group_path.size())
    {
        const std::size_t slash = group_path.find('/', begin);
        const std::string component = group_path.substr(
            begin,
            slash == std::string::npos ? std::string::npos : slash - begin);
        current += "/" + component;
        if (!file.exist(current))
        {
            file.createGroup(current);
        }
        if (slash == std::string::npos)
        {
            break;
        }
        begin = slash + 1;
    }
}

static void Ensure_Parent_Group(HighFive::File& file,
                                const std::string& dataset_path)
{
    const std::size_t slash = dataset_path.find_last_of('/');
    if (slash == std::string::npos || slash == 0)
    {
        return;
    }
    Ensure_Group(file, dataset_path.substr(0, slash));
}

template <typename T>
static void Write_Scalar(HighFive::File& file, const std::string& path,
                         const T& value)
{
    Ensure_Parent_Group(file, path);
    auto dataset =
        file.createDataSet<T>(path, HighFive::DataSpace::From(value));
    dataset.write(value);
}

static void Write_Float_Vector(HighFive::File& file, const std::string& path,
                               const std::vector<float>& values)
{
    Ensure_Parent_Group(file, path);
    auto dataset =
        file.createDataSet<float>(path, HighFive::DataSpace::From(values));
    dataset.write(values);
}

static void Write_Float_XYZ_Matrix(HighFive::File& file,
                                   const std::string& path,
                                   const std::vector<float>& values)
{
    REQUIRE_TRUE(!values.empty() && values.size() % 3 == 0);
    Ensure_Parent_Group(file, path);
    auto dataset = file.createDataSet<float>(
        path, HighFive::DataSpace({values.size() / 3, 3}));
    REQUIRE_TRUE(H5Dwrite(dataset.getId(), H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                          H5P_DEFAULT, values.data()) >= 0);
}

template <typename T>
static void Write_Matrix(HighFive::File& file, const std::string& path,
                         const std::vector<T>& values, std::size_t rows,
                         std::size_t columns)
{
    REQUIRE_EQ(values.size(), rows * columns);
    Ensure_Parent_Group(file, path);
    auto dataset =
        file.createDataSet<T>(path, HighFive::DataSpace({rows, columns}));
    REQUIRE_TRUE(H5Dwrite(dataset.getId(),
                          std::is_same<T, int>::value ? H5T_NATIVE_INT
                                                      : H5T_NATIVE_FLOAT,
                          H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()) >= 0);
}

static void Write_Int_Vector(HighFive::File& file, const std::string& path,
                             const std::vector<int>& values)
{
    Ensure_Parent_Group(file, path);
    auto dataset =
        file.createDataSet<int>(path, HighFive::DataSpace::From(values));
    dataset.write(values);
}

static void Write_Int64_Vector(HighFive::File& file, const std::string& path,
                               const std::vector<std::int64_t>& values)
{
    Ensure_Parent_Group(file, path);
    auto dataset = file.createDataSet<std::int64_t>(
        path, HighFive::DataSpace::From(values));
    dataset.write(values);
}

static void Write_String_Vector(HighFive::File& file, const std::string& path,
                                const std::vector<std::string>& values)
{
    Ensure_Parent_Group(file, path);
    auto dataset = file.createDataSet<std::string>(
        path, HighFive::DataSpace::From(values));
    dataset.write(values);
}

static void Test_Protocol_Reader_Loads_Typed_CV_Restraint()
{
    const auto dir = Unique_Temp_Path("protocol_cv_restraint");
    std::filesystem::create_directories(dir);
    const auto path = dir / "protocol.spgp.h5";
    {
        HighFive::File file(path.string(), HighFive::File::Overwrite);
        Write_Scalar(file, "/restraint/umbrella/type",
                     std::string("cv_harmonic"));
        Write_String_Vector(file, "/restraint/umbrella/cv_refs",
                            {"torsion", "distance"});
        Write_Float_Vector(file, "/restraint/umbrella/weight", {100.0f, 50.0f});
        Write_Float_Vector(file, "/restraint/umbrella/reference", {1.5f, 3.0f});
        Write_Float_Vector(file, "/restraint/umbrella/period",
                           {6.283185f, 0.0f});
        Write_Int64_Vector(file, "/restraint/umbrella/schedule/start_step",
                           {10, 20});
        Write_Scalar(file, "/restraint/disabled_cv/type",
                     std::string("cv_harmonic"));
        Write_String_Vector(file, "/restraint/disabled_cv/cv_refs",
                            {"ignored"});
        Write_Float_Vector(file, "/restraint/disabled_cv/weight", {1.0f});
        Write_Float_Vector(file, "/restraint/disabled_cv/reference", {2.0f});
        Write_Scalar(file, "/restraint/disabled_cv/enabled_default", 0);
    }
    {
        ProtocolRestraintH5Reader reader;
        REQUIRE_TRUE(reader.Open(path.string()));
        std::vector<ProtocolCVRestraint> restraints;
        REQUIRE_TRUE(reader.Read_CV_Restraints(&restraints));
        REQUIRE_EQ(restraints.size(), static_cast<std::size_t>(1));
        REQUIRE_EQ(restraints[0].name, std::string("umbrella"));
        REQUIRE_EQ(restraints[0].cv_refs[0], std::string("torsion"));
        REQUIRE_EQ(restraints[0].weight[1], 50.0f);
        REQUIRE_EQ(restraints[0].start_step[1], static_cast<std::int64_t>(20));
        REQUIRE_EQ(restraints[0].max_step[0], static_cast<std::int64_t>(0));
    }
    std::filesystem::remove_all(dir);
}

static void Test_Protocol_Reader_Loads_Native_CV_Objects()
{
    const auto dir = Unique_Temp_Path("protocol_native_cv");
    std::filesystem::create_directories(dir);
    const auto protocol_path = dir / "protocol.spgp.h5";
    const auto restart_path = dir / "restart.spgr.h5";
    {
        HighFive::File file(protocol_path.string(), HighFive::File::Overwrite);
        Write_Scalar(file, "/cv/distance/type", std::string("distance"));
        Write_Scalar(file, "/cv/distance/dimension", 1LL);
        Write_Int64_Vector(file, "/cv/distance/atom_indices", {0, 1});
        Write_Float_Vector(file, "/cv/distance/period", {0.0f});
        Write_Float_Vector(file, "/cv/distance/sigma", {0.5f});

        Write_Scalar(file, "/cv/backbone/type", std::string("rmsd"));
        Write_Int64_Vector(file, "/cv/backbone/atom_indices", {0, 1});
        Write_Scalar(file, "/cv/backbone/rotate", 0);
        Write_Scalar(file, "/cv/disabled/type", std::string("distance"));
        Write_Int64_Vector(file, "/cv/disabled/atom_indices", {0, 1});
        Write_Scalar(file, "/cv/disabled/enabled_default", 0);
    }
    {
        HighFive::File file(restart_path.string(), HighFive::File::Overwrite);
        Write_Float_XYZ_Matrix(
            file, "/parameters/restart/references/cv/backbone/coordinate",
            {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    }
    {
        ProtocolCVH5Reader reader;
        REQUIRE_TRUE(reader.Open_Protocol(protocol_path.string()));
        REQUIRE_TRUE(reader.Open_Restart(restart_path.string()));
        std::vector<ProtocolCVDefinition> definitions;
        REQUIRE_TRUE(reader.Read_Definitions(2, &definitions));
        REQUIRE_EQ(definitions.size(), static_cast<std::size_t>(2));
        REQUIRE_EQ(definitions[0].name, std::string("backbone"));
        REQUIRE_EQ(definitions[0].type, std::string("rmsd"));
        REQUIRE_EQ(definitions[0].reference_coordinates,
                   std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
        REQUIRE_TRUE(std::find(definitions[0].runtime_parameters.begin(),
                               definitions[0].runtime_parameters.end(),
                               std::make_pair(std::string("coordinate"),
                                              std::string("1 2 3 4 5 6"))) !=
                     definitions[0].runtime_parameters.end());
        REQUIRE_EQ(definitions[1].name, std::string("distance"));
        REQUIRE_EQ(definitions[1].period[0], 0.0f);
        REQUIRE_EQ(definitions[1].sigma[0], 0.5f);
    }
    std::filesystem::remove_all(dir);
}

static void Test_Protocol_Reader_Loads_Typed_Virtual_Atoms()
{
    const auto dir = Unique_Temp_Path("protocol_virtual_atom");
    std::filesystem::create_directories(dir);
    const auto protocol_path = dir / "protocol.spgp.h5";
    {
        HighFive::File file(protocol_path.string(), HighFive::File::Overwrite);
        Write_Scalar(file, "/cv/virtual_atom/dt/type", std::string("center"));
        Write_Int64_Vector(file, "/cv/virtual_atom/dt/atom_indices", {0, 1});
        Write_Float_Vector(file, "/cv/virtual_atom/dt/weight", {0.25f, 0.75f});
        Write_Scalar(file, "/cv/distance/type", std::string("distance"));
        Write_String_Vector(file, "/cv/distance/atom_refs", {"dt", "2"});
    }
    {
        ProtocolCVH5Reader reader;
        REQUIRE_TRUE(reader.Open_Protocol(protocol_path.string()));
        std::vector<ProtocolVirtualAtomDefinition> virtual_atoms;
        REQUIRE_TRUE(reader.Read_Virtual_Atoms(3, &virtual_atoms));
        REQUIRE_EQ(virtual_atoms.size(), static_cast<std::size_t>(1));
        REQUIRE_EQ(virtual_atoms[0].name, std::string("dt"));
        REQUIRE_EQ(virtual_atoms[0].atom_indices, std::vector<int>({0, 1}));
        REQUIRE_EQ(virtual_atoms[0].weight, std::vector<float>({0.25f, 0.75f}));
        std::vector<ProtocolCVDefinition> definitions;
        REQUIRE_TRUE(reader.Read_Definitions(3, &definitions));
        REQUIRE_EQ(definitions.size(), static_cast<std::size_t>(1));
        REQUIRE_TRUE(std::find(definitions[0].runtime_parameters.begin(),
                               definitions[0].runtime_parameters.end(),
                               std::make_pair(std::string("atom"),
                                              std::string("dt 2"))) !=
                     definitions[0].runtime_parameters.end());
    }
    std::filesystem::remove_all(dir);
}

static void Test_Protocol_Reader_Loads_Native_Metadynamics_Object()
{
    const auto dir = Unique_Temp_Path("protocol_native_metadynamics");
    std::filesystem::create_directories(dir);
    const auto protocol_path = dir / "protocol.spgp.h5";
    {
        HighFive::File file(protocol_path.string(), HighFive::File::Overwrite);
        Write_String_Vector(file, "/meta/bias/cv_refs", {"distance"});
        Write_Scalar(file, "/meta/bias/ndim", 1LL);
        Write_Float_Vector(file, "/meta/bias/grid/min", {0.0f});
        Write_Float_Vector(file, "/meta/bias/grid/max", {10.0f});
        Write_Int64_Vector(file, "/meta/bias/grid/count", {64});
        Write_Scalar(file, "/meta/bias/hill_height_default", 1.5f);
        Write_Scalar(file, "/meta/bias/sumhill_freq_default", 20LL);
        Write_Scalar(file, "/meta/bias/method_flags/subhill", 1);
        Write_Scalar(file, "/meta/bias/method_flags/dip", 0.25f);
        Write_String_Vector(file, "/meta/disabled/cv_refs", {"distance"});
        Write_Scalar(file, "/meta/disabled/enabled_default", 0);
    }
    ProtocolCVDefinition distance;
    distance.name = "distance";
    distance.dimension = 1;
    distance.period = {0.0f};
    distance.sigma = {0.5f};
    ProtocolMetadynamicsDefinition definition;
    bool found = false;
    {
        ProtocolMetadynamicsH5Reader reader;
        REQUIRE_TRUE(reader.Open(protocol_path.string()));
        REQUIRE_TRUE(reader.Read_Definition({distance}, &definition, &found));
        REQUIRE_TRUE(found);
        REQUIRE_EQ(definition.name, std::string("bias"));
        REQUIRE_EQ(definition.cv_refs, std::vector<std::string>({"distance"}));
        REQUIRE_TRUE(std::find(definition.runtime_parameters.begin(),
                               definition.runtime_parameters.end(),
                               std::make_pair(std::string("CV_sigma"),
                                              std::string("0.5"))) !=
                     definition.runtime_parameters.end());
        REQUIRE_TRUE(std::find(definition.runtime_parameters.begin(),
                               definition.runtime_parameters.end(),
                               std::make_pair(std::string("subhill"),
                                              std::string("1"))) !=
                     definition.runtime_parameters.end());
    }
    {
        HighFive::File file(protocol_path.string(), HighFive::File::ReadWrite);
        const int enabled = 1;
        file.getDataSet("/meta/disabled/enabled_default").write(enabled);
    }
    {
        ProtocolMetadynamicsH5Reader duplicate_reader;
        REQUIRE_TRUE(duplicate_reader.Open(protocol_path.string()));
        REQUIRE_TRUE(
            !duplicate_reader.Read_Definition({distance}, &definition, &found));
        REQUIRE_TRUE(duplicate_reader.Last_Error().find(
                         "at most one enabled metadynamics object") !=
                     std::string::npos);
    }
    std::filesystem::remove_all(dir);
}

static void Test_Protocol_Reader_Loads_Native_Steering_Object()
{
    const auto dir = Unique_Temp_Path("protocol_native_steering");
    std::filesystem::create_directories(dir);
    const auto protocol_path = dir / "protocol.spgp.h5";
    {
        HighFive::File file(protocol_path.string(), HighFive::File::Overwrite);
        Write_String_Vector(file, "/steer/cv_refs", {"distance"});
        Write_Float_Vector(file, "/steer/weight", {2.0f});
    }
    ProtocolCVDefinition distance;
    distance.name = "distance";
    distance.dimension = 1;
    ProtocolSteeringDefinition definition;
    bool found = false;
    {
        ProtocolSteeringH5Reader reader;
        REQUIRE_TRUE(reader.Open(protocol_path.string()));
        REQUIRE_TRUE(reader.Read_Definition({distance}, &definition, &found));
        REQUIRE_TRUE(found);
        REQUIRE_EQ(definition.cv_refs, std::vector<std::string>({"distance"}));
        REQUIRE_EQ(definition.weight, std::vector<float>({2.0f}));
        REQUIRE_TRUE(std::find(definition.runtime_parameters.begin(),
                               definition.runtime_parameters.end(),
                               std::make_pair(std::string("CV"),
                                              std::string("distance"))) !=
                     definition.runtime_parameters.end());

        ProtocolSteeringH5Reader invalid_reader;
        REQUIRE_TRUE(invalid_reader.Open(protocol_path.string()));
        REQUIRE_TRUE(!invalid_reader.Read_Definition({}, &definition, &found));
        REQUIRE_TRUE(invalid_reader.Last_Error().find(
                         "references missing or disabled /cv/distance") !=
                     std::string::npos);
    }
    std::filesystem::remove_all(dir);
}

static void Test_Positional_Restraint_Reader_Loads_Named_Object()
{
    const auto dir = Unique_Temp_Path("protocol_positional_restraint");
    std::filesystem::create_directories(dir);
    const auto protocol_path = dir / "protocol.spgp.h5";
    const auto restart_path = dir / "restart.spgr.h5";
    {
        HighFive::File file(protocol_path.string(), HighFive::File::Overwrite);
        Write_Scalar(file, "/restraint/backbone/type",
                     std::string("harmonic_positional"));
        Write_Int64_Vector(file, "/restraint/backbone/atom_indices", {0, 1});
        Write_Float_XYZ_Matrix(file, "/restraint/backbone/weight",
                               {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
        Write_Scalar(file, "/restraint/backbone/single_weight_default", 8.0f);
        Write_Scalar(file, "/restraint/backbone/refcoord_scaling_default",
                     std::string("all"));
        Write_Scalar(file, "/restraint/backbone/calc_virial_default", 1);
        Write_Scalar(file, "/restraint/disabled/type",
                     std::string("harmonic_positional"));
        Write_Int64_Vector(file, "/restraint/disabled/atom_indices", {0});
        Write_Scalar(file, "/restraint/disabled/enabled_default", 0);
    }
    {
        HighFive::File file(restart_path.string(), HighFive::File::Overwrite);
        Write_Float_XYZ_Matrix(
            file,
            "/parameters/restart/references/restraint/backbone/coordinate",
            {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    }
    NativeRestraintH5Reader reader;
    REQUIRE_TRUE(reader.Open_Protocol(protocol_path.string()));
    REQUIRE_TRUE(reader.Open_Restart(restart_path.string()));
    REQUIRE_TRUE(reader.Has_Positional_Restraint());
    Xponge::PositionalRestraint state;
    REQUIRE_TRUE(reader.Read(2, &state));
    REQUIRE_TRUE(state.present);
    REQUIRE_EQ(state.name, std::string("backbone"));
    REQUIRE_EQ(state.atom_indices, std::vector<int>({0, 1}));
    REQUIRE_EQ(state.weight,
               std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
    REQUIRE_EQ(state.reference_coordinates,
               std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
    REQUIRE_TRUE(state.has_single_weight_default);
    REQUIRE_EQ(state.single_weight_default, 8.0f);
    REQUIRE_TRUE(state.has_refcoord_scaling_default);
    REQUIRE_EQ(state.refcoord_scaling_default, std::string("all"));
    REQUIRE_TRUE(state.has_calc_virial_default);
    REQUIRE_TRUE(state.calc_virial_default);
    REQUIRE_TRUE(!std::filesystem::exists(dir / "materialized"));
    reader = {};
    std::filesystem::remove_all(dir);
}

static void Write_Topology_Metadata(const std::filesystem::path& path,
                                    std::int64_t atom_count)
{
    HighFive::File file(path.string(), HighFive::File::Overwrite);
    Write_Scalar(file, "/schema/name", std::string("sponge.topology.h5"));
    Write_Scalar(file, "/schema/version", std::string("sponge.input.v2"));
    Write_Scalar(file, "/identity/uuid",
                 std::string("123e4567-e89b-12d3-a456-426614174000"));
    Write_Scalar(file, "/topology/atom_count", atom_count);
    Write_Scalar(file, "/topology/atom_order_hash", std::string("atoms"));
    Write_Scalar(file, "/topology/topology_hash", std::string("top"));
    Write_Scalar(file, "/topology/forcefield_hash", std::string("ff"));
}

static void Write_Protocol_Metadata(const std::filesystem::path& path,
                                    const std::string& topology_hash)
{
    HighFive::File file(path.string(), HighFive::File::Overwrite);
    Write_Scalar(file, "/schema/name", std::string("sponge.protocol.h5"));
    Write_Scalar(file, "/schema/version", std::string("sponge.input.v2"));
    Write_Scalar(file, "/identity/uuid",
                 std::string("123e4567-e89b-12d3-a456-426614174001"));
    Write_Scalar(file, "/protocol/topology_compatibility/topology_hash",
                 topology_hash);
    Write_Scalar(file, "/identity/content_hash", std::string("protocol"));
}

static SpongeH5OutputPlan::ResolvedOutputPlan Make_Restart_Output_Plan(
    const std::filesystem::path& restart_path)
{
    SpongeH5OutputPlan::ResolvedOutputPlan plan;
    plan.restart.enabled = true;
    plan.restart.path = restart_path.string();
    return plan;
}

static SpongeH5OutputPlan::ResolvedOutputPlan Make_Trajectory_Output_Plan(
    const std::filesystem::path& trajectory_path)
{
    SpongeH5OutputPlan::ResolvedOutputPlan plan;
    plan.trajectory.enabled = true;
    plan.trajectory.path = trajectory_path.string();
    plan.trajectory.vds = false;
    return plan;
}

static void Write_Restart_File(const std::filesystem::path& path,
                               std::size_t atom_count,
                               bool include_dynamic_state,
                               bool include_sits_state = false,
                               bool include_metad_state = false,
                               bool include_protocol_sidecar_state = false)
{
    std::vector<float> position(atom_count * 3, 0.0f);
    std::vector<float> velocity(atom_count * 3, 0.0f);
    for (std::size_t i = 0; i < position.size(); ++i)
    {
        position[i] = static_cast<float>(i + 1);
        velocity[i] = static_cast<float>(i) * 0.1f;
    }
    const std::array<float, 9> box = {
        10.0f, 0.0f, 0.0f, 0.0f, 20.0f, 0.0f, 0.0f, 0.0f, 30.0f,
    };
    const std::vector<float> nhc_state = {0.25f, 0.50f};
    const std::vector<float> sits_nk_state = {1.0f, 2.0f};
    const std::vector<float> sits_log_norm_state = {-2.0f, -1.0f};
    const std::vector<float> sits_log_nk_state = {0.0f, 0.693147f};

    HighFiveBackend backend;
    RestartH5Writer writer(&backend);
    REQUIRE_TRUE(writer.Open(Make_Restart_Output_Plan(path),
                             SpongeH5MD::kInputSchemaVersion));
    REQUIRE_TRUE(writer.Write_Lineage("top", "atoms", "protocol"));
    REQUIRE_TRUE(writer.Define_Structural_State(atom_count, true));
    REQUIRE_TRUE(writer.Write_Structural_State(10, 0.02, position.data(),
                                               box.data(), velocity.data()));
    if (include_dynamic_state)
    {
        REQUIRE_TRUE(writer.Write_Nose_Hoover_Chain_State(nhc_state.data(), 1));
    }
    if (include_sits_state)
    {
        REQUIRE_TRUE(writer.Write_Sits_State("SITS", "nk", sits_nk_state.data(),
                                             sits_nk_state.size()));
        REQUIRE_TRUE(writer.Write_Sits_State("SITS", "log_norm",
                                             sits_log_norm_state.data(),
                                             sits_log_norm_state.size()));
        REQUIRE_TRUE(writer.Write_Sits_State("SITS", "log_nk",
                                             sits_log_nk_state.data(),
                                             sits_log_nk_state.size()));
    }
    if (include_metad_state)
    {
        REQUIRE_TRUE(writer.Write_Metad_State_Text("meta", "hills", "HILLS\n"));
        REQUIRE_TRUE(
            writer.Write_Metad_State_Text("meta", "history", "HISTORY\n"));
    }
    if (include_protocol_sidecar_state)
    {
        REQUIRE_TRUE(
            writer.Write_Protocol_Sidecar_Text("cv_in_file", "CV_PAYLOAD\n"));
    }
    REQUIRE_TRUE(writer.Finalize());
    REQUIRE_TRUE(writer.Close());
}

static void Refresh_Restart_State_Hash(const std::filesystem::path& path)
{
    std::string state_hash;
    {
        SpongeH5MD::RestartH5Reader reader;
        REQUIRE_TRUE(reader.Open(path.string()));
        REQUIRE_TRUE(reader.Compute_State_Hash(&state_hash));
    }
    HighFive::File file(path.string(), HighFive::File::ReadWrite);
    REQUIRE_TRUE(H5Ldelete(file.getId(), SpongeH5MD::path::run_state_hash,
                           H5P_DEFAULT) >= 0);
    Write_Scalar(file, SpongeH5MD::path::run_state_hash, state_hash);
}

static void Add_Unsupported_Dynamic_State(const std::filesystem::path& path,
                                          const std::string& module_name)
{
    {
        HighFive::File file(path.string(), HighFive::File::ReadWrite);
        Write_Scalar(file, SpongeH5MD::Restart_Rng_State_Path(module_name),
                     std::string("unsupported:philox_device_state"));
    }
    Refresh_Restart_State_Hash(path);
}

static void Add_Supported_Dynamic_State(const std::filesystem::path& path,
                                        const std::string& module_name)
{
    {
        HighFive::File file(path.string(), HighFive::File::ReadWrite);
        if (module_name == "integrator_state")
        {
            Write_Scalar(file,
                         SpongeH5MD::Restart_Integrator_State_Path("mode"),
                         std::string("nvt"));
            Write_Scalar(file,
                         SpongeH5MD::Restart_Integrator_State_Path("step"),
                         std::string("10"));
            Write_Scalar(file,
                         SpongeH5MD::Restart_Integrator_State_Path("time"),
                         std::string("0.02"));
        }
        else
        {
            Write_Scalar(file, SpongeH5MD::Restart_Rng_State_Path(module_name),
                         std::string("serialized_rng_state"));
            if (module_name == "bussi_thermostat")
            {
                Write_Float_Vector(file,
                                   SpongeH5MD::Restart_Thermostat_State_Path(
                                       module_name, "lambda"),
                                   {0.87f});
            }
            else if (module_name == "pressure_based_barostat")
            {
                Write_Float_Vector(
                    file,
                    SpongeH5MD::Restart_Barostat_State_Path(module_name, "g"),
                    {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
            }
            else
            {
                throw TestFailure("unknown supported dynamic state module: " +
                                  module_name);
            }
        }
    }
    Refresh_Restart_State_Hash(path);
}

static void Add_Unsupported_Protocol_State(const std::filesystem::path& path)
{
    {
        HighFive::File file(path.string(), HighFive::File::ReadWrite);
        Write_Float_Vector(
            file,
            std::string(SpongeH5MD::path::restart_bias) + "/unsupported/state",
            {1.0f});
    }
    Refresh_Restart_State_Hash(path);
}

static void Write_Trajectory_File(const std::filesystem::path& path,
                                  std::size_t atom_count)
{
    std::vector<float> position(atom_count * 3, 0.0f);
    for (std::size_t i = 0; i < position.size(); ++i)
    {
        position[i] = static_cast<float>(i + 1);
    }
    const std::array<float, 9> box = {
        10.0f, 0.0f, 0.0f, 0.0f, 20.0f, 0.0f, 0.0f, 0.0f, 30.0f,
    };

    HighFiveBackend backend;
    TrajectoryH5Writer writer(&backend);
    REQUIRE_TRUE(writer.Open_Single_File(Make_Trajectory_Output_Plan(path),
                                         SpongeH5MD::kOutputSchemaVersion));
    REQUIRE_TRUE(writer.Define_Particle_Datasets(atom_count, false, false));
    REQUIRE_TRUE(
        writer.Append_Particle_Frame(10, 0.02, position.data(), box.data()));
    REQUIRE_TRUE(writer.Finalize());
    REQUIRE_TRUE(writer.Close());
}

static SpongeH5InputPlan::ResolvedInputPlan Make_Input_Plan(
    const std::filesystem::path& topology_path,
    const std::filesystem::path& protocol_path,
    const std::filesystem::path& restart_path)
{
    SpongeH5InputPlan::ResolvedInputPlan plan;
    plan.any_h5_input_enabled = true;
    plan.legacy_input_allowed = false;
    plan.topology.enabled = true;
    plan.topology.path = topology_path.string();
    plan.protocol.enabled = true;
    plan.protocol.path = protocol_path.string();
    plan.restart.binding.enabled = true;
    plan.restart.binding.path = restart_path.string();
    return plan;
}

static void Require_Valid(
    const SpongeH5InputValidation::ValidationResult& result)
{
    if (!result.valid)
    {
        throw TestFailure(result.error_message);
    }
}

static void Require_Invalid_Contains(
    const SpongeH5InputValidation::ValidationResult& result,
    const std::string& needle)
{
    REQUIRE_TRUE(!result.valid);
    REQUIRE_TRUE(result.error_message.find(needle) != std::string::npos);
}

static void Replace_String_Dataset(const std::filesystem::path& file_path,
                                   const std::string& dataset_path,
                                   const std::string& value)
{
    HighFive::File file(file_path.string(), HighFive::File::ReadWrite);
    REQUIRE_TRUE(file.exist(dataset_path));
    REQUIRE_TRUE(H5Ldelete(file.getId(), dataset_path.c_str(), H5P_DEFAULT) >=
                 0);
    Write_Scalar(file, dataset_path, value);
}

static void Test_Validates_Structural_Bundle_Metadata()
{
    const auto dir = Unique_Temp_Path("h5_input_validation_valid");
    std::filesystem::create_directories(dir);
    const auto topology = dir / "system.spgt.h5";
    const auto protocol = dir / "protocol.spgp.h5";
    const auto restart = dir / "restart.spgr.h5";

    Write_Topology_Metadata(topology, 2);
    Write_Protocol_Metadata(protocol, "top");
    Write_Restart_File(restart, 2, false);

    Require_Valid(SpongeH5InputValidation::Validate_Resolved_Input_Plan(
        Make_Input_Plan(topology, protocol, restart)));

    std::filesystem::remove_all(dir);
}

static void Test_Rejects_Invalid_Artifact_Schema_And_Identity()
{
    const auto dir = Unique_Temp_Path("h5_input_validation_identity");
    std::filesystem::create_directories(dir);
    const auto topology = dir / "system.spgt.h5";
    const auto protocol = dir / "protocol.spgp.h5";
    const auto restart = dir / "restart.spgr.h5";

    Write_Topology_Metadata(topology, 2);
    Write_Protocol_Metadata(protocol, "top");
    Write_Restart_File(restart, 2, false);
    const auto plan = Make_Input_Plan(topology, protocol, restart);

    Replace_String_Dataset(topology, "/schema/version", "1");
    Require_Invalid_Contains(
        SpongeH5InputValidation::Validate_Resolved_Input_Plan(plan),
        "topology schema version");

    Write_Topology_Metadata(topology, 2);
    Replace_String_Dataset(protocol, "/identity/uuid", "not-a-uuid");
    Require_Invalid_Contains(
        SpongeH5InputValidation::Validate_Resolved_Input_Plan(plan),
        "protocol identity UUID");

    std::filesystem::remove_all(dir);
}

static void Test_Rejects_Restart_Lineage_And_State_Hash_Mismatch()
{
    const auto dir = Unique_Temp_Path("h5_input_validation_lineage");
    std::filesystem::create_directories(dir);
    const auto topology = dir / "system.spgt.h5";
    const auto protocol = dir / "protocol.spgp.h5";
    const auto restart = dir / "restart.spgr.h5";

    Write_Topology_Metadata(topology, 2);
    Write_Protocol_Metadata(protocol, "top");
    Write_Restart_File(restart, 2, false);
    const auto plan = Make_Input_Plan(topology, protocol, restart);

    Replace_String_Dataset(restart, SpongeH5MD::path::run_topology_hash,
                           "other-topology");
    Require_Invalid_Contains(
        SpongeH5InputValidation::Validate_Resolved_Input_Plan(plan),
        "restart topology hash");

    Write_Restart_File(restart, 2, false);
    {
        HighFive::File file(restart.string(), HighFive::File::ReadWrite);
        const float modified_position[6] = {99.0f, 2.0f, 3.0f,
                                            4.0f,  5.0f, 6.0f};
        const auto dataset = file.getDataSet(SpongeH5MD::path::position_value);
        REQUIRE_TRUE(H5Dwrite(dataset.getId(), H5T_NATIVE_FLOAT, H5S_ALL,
                              H5S_ALL, H5P_DEFAULT, modified_position) >= 0);
    }
    Require_Invalid_Contains(
        SpongeH5InputValidation::Validate_Resolved_Input_Plan(plan),
        "restart state_hash does not match");

    std::filesystem::remove_all(dir);
}

static void Test_Validates_Controller_Input_Bindings()
{
    const auto dir = Unique_Temp_Path("h5_input_validation_controller");
    std::filesystem::create_directories(dir);
    const auto topology = dir / "system.spgt.h5";
    const auto protocol = dir / "protocol.spgp.h5";
    const auto restart = dir / "restart.spgr.h5";
    const auto trajectory = dir / "prod.spg.h5md";

    Write_Topology_Metadata(topology, 2);
    Write_Protocol_Metadata(protocol, "top");
    Write_Restart_File(restart, 2, true, true);
    Write_Trajectory_File(trajectory, 2);

    CONTROLLER controller;
    controller.Set("mode", "rerun");
    controller.Set("input_h5_topology_path", topology.string());
    controller.Set("input_h5_protocol_path", protocol.string());
    controller.Set("input_h5_restart_path", restart.string());
    controller.Set("input_h5_restart_load", "full");
    controller.Set("input_h5_trajectory_path", trajectory.string());
    controller.Set("input_h5_trajectory_particle_stream", "all");

    Require_Valid(
        SpongeH5InputValidation::Validate_Input_Bindings(&controller));

    std::filesystem::remove_all(dir);
}

static void Test_Controller_Input_Bindings_Propagate_Resolver_Errors()
{
    {
        CONTROLLER controller;
        controller.Set("mode", "nve");
        controller.Set("input_h5_protocol_path", "protocol.spgp.h5");
        controller.Set("input_h5_restart_path", "restart.spgr.h5");

        Require_Invalid_Contains(
            SpongeH5InputValidation::Validate_Input_Bindings(&controller),
            "input_h5_topology_path");
    }

    {
        CONTROLLER controller;
        controller.Set("mode", "nve");
        controller.Set("input_h5_topology_path", "topology.spgt.h5");
        controller.Set("input_h5_protocol_path", "protocol.spgp.h5");
        controller.Set("input_h5_restart_path", "restart.spgr.h5");
        controller.Set("input_h5_restart_load", "custom");

        Require_Invalid_Contains(
            SpongeH5InputValidation::Validate_Input_Bindings(&controller),
            "input_h5_restart_load = custom is reserved");
    }

    {
        CONTROLLER controller;
        controller.Set("input_h5_topology_path", "topology.spgt.h5");
        controller.Set("input_h5_protocol_path", "protocol.spgp.h5");
        controller.Set("input_h5_restart_path", "restart.spgr.h5");
        controller.Set("gromacs_top", "system.top");

        Require_Invalid_Contains(
            SpongeH5InputValidation::Validate_Input_Bindings(&controller),
            "support only SPONGE native inputs; GROMACS");
    }

    {
        CONTROLLER controller;
        controller.Set("input_h5_topology_path", "topology.spgt.h5");
        controller.Set("input_h5_protocol_path", "protocol.spgp.h5");
        controller.Set("input_h5_restart_path", "restart.spgr.h5");
        controller.Set("amber_parm7", "system.parm7");

        Require_Invalid_Contains(
            SpongeH5InputValidation::Validate_Input_Bindings(&controller),
            "support only SPONGE native inputs; AMBER");
    }
}

static void Test_Rejects_Restart_Atom_Count_Mismatch()
{
    const auto dir = Unique_Temp_Path("h5_input_validation_restart_mismatch");
    std::filesystem::create_directories(dir);
    const auto topology = dir / "system.spgt.h5";
    const auto protocol = dir / "protocol.spgp.h5";
    const auto restart = dir / "restart.spgr.h5";

    Write_Topology_Metadata(topology, 3);
    Write_Protocol_Metadata(protocol, "top");
    Write_Restart_File(restart, 2, false);

    Require_Invalid_Contains(
        SpongeH5InputValidation::Validate_Resolved_Input_Plan(
            Make_Input_Plan(topology, protocol, restart)),
        "restart atom_count");

    std::filesystem::remove_all(dir);
}

static void Test_Validates_H5md_Rerun_Trajectory_Metadata()
{
    const auto dir = Unique_Temp_Path("h5_input_validation_trajectory");
    std::filesystem::create_directories(dir);
    const auto topology = dir / "system.spgt.h5";
    const auto protocol = dir / "protocol.spgp.h5";
    const auto restart = dir / "restart.spgr.h5";
    const auto trajectory = dir / "prod.spg.h5md";

    Write_Topology_Metadata(topology, 2);
    Write_Protocol_Metadata(protocol, "top");
    Write_Restart_File(restart, 2, false);
    Write_Trajectory_File(trajectory, 2);

    auto plan = Make_Input_Plan(topology, protocol, restart);
    plan.trajectory.binding.enabled = true;
    plan.trajectory.binding.path = trajectory.string();
    Require_Valid(SpongeH5InputValidation::Validate_Resolved_Input_Plan(plan));

    std::filesystem::remove_all(dir);
}

static void Test_Rejects_Missing_Requested_Trajectory_Stream()
{
    const auto dir = Unique_Temp_Path("h5_input_validation_stream");
    std::filesystem::create_directories(dir);
    const auto topology = dir / "system.spgt.h5";
    const auto protocol = dir / "protocol.spgp.h5";
    const auto restart = dir / "restart.spgr.h5";
    const auto trajectory = dir / "prod.spg.h5md";

    Write_Topology_Metadata(topology, 2);
    Write_Protocol_Metadata(protocol, "top");
    Write_Restart_File(restart, 2, false);
    Write_Trajectory_File(trajectory, 2);

    auto plan = Make_Input_Plan(topology, protocol, restart);
    plan.trajectory.binding.enabled = true;
    plan.trajectory.binding.path = trajectory.string();
    plan.trajectory.particle_stream = "solute";
    Require_Invalid_Contains(
        SpongeH5InputValidation::Validate_Resolved_Input_Plan(plan),
        "position");

    std::filesystem::remove_all(dir);
}

static void Test_Validates_Dynamic_Load_When_Dynamic_State_Is_Present()
{
    const auto dir = Unique_Temp_Path("h5_input_validation_dynamic");
    std::filesystem::create_directories(dir);
    const auto topology = dir / "system.spgt.h5";
    const auto protocol = dir / "protocol.spgp.h5";
    const auto restart = dir / "restart.spgr.h5";

    Write_Topology_Metadata(topology, 2);
    Write_Protocol_Metadata(protocol, "top");
    Write_Restart_File(restart, 2, true);

    auto plan = Make_Input_Plan(topology, protocol, restart);
    plan.restart.load_policy =
        SpongeH5InputContract::RestartLoadPolicy::dynamic;
    Require_Valid(SpongeH5InputValidation::Validate_Resolved_Input_Plan(plan));

    std::filesystem::remove_all(dir);
}

static void Test_Recognizes_Each_Supported_Dynamic_State_Module()
{
    for (const std::string& module_name :
         {"integrator_state", "bussi_thermostat", "pressure_based_barostat"})
    {
        const auto dir = Unique_Temp_Path("h5_input_validation_" + module_name);
        std::filesystem::create_directories(dir);
        const auto topology = dir / "system.spgt.h5";
        const auto protocol = dir / "protocol.spgp.h5";
        const auto restart = dir / "restart.spgr.h5";

        Write_Topology_Metadata(topology, 2);
        Write_Protocol_Metadata(protocol, "top");
        Write_Restart_File(restart, 2, false);
        Add_Supported_Dynamic_State(restart, module_name);

        auto plan = Make_Input_Plan(topology, protocol, restart);
        plan.restart.load_policy =
            SpongeH5InputContract::RestartLoadPolicy::dynamic;
        Require_Valid(
            SpongeH5InputValidation::Validate_Resolved_Input_Plan(plan));

        std::filesystem::remove_all(dir);
    }
}

static void Test_Rejects_Dynamic_Load_When_Dynamic_State_Is_Absent()
{
    const auto dir = Unique_Temp_Path("h5_input_validation_dynamic_absent");
    std::filesystem::create_directories(dir);
    const auto topology = dir / "system.spgt.h5";
    const auto protocol = dir / "protocol.spgp.h5";
    const auto restart = dir / "restart.spgr.h5";

    Write_Topology_Metadata(topology, 2);
    Write_Protocol_Metadata(protocol, "top");
    Write_Restart_File(restart, 2, false);

    auto plan = Make_Input_Plan(topology, protocol, restart);
    plan.restart.load_policy =
        SpongeH5InputContract::RestartLoadPolicy::dynamic;
    Require_Invalid_Contains(
        SpongeH5InputValidation::Validate_Resolved_Input_Plan(plan),
        "dynamic state is absent");

    std::filesystem::remove_all(dir);
}

static void Test_Rejects_Dynamic_Load_When_Only_Unsupported_State_Is_Present()
{
    const std::array<std::string, 3> modules = {"middle_langevin", "andersen",
                                                "monte_carlo_barostat"};
    const std::array<std::string, 3> diagnostics = {
        "Middle Langevin Philox RNG state", "Andersen thermostat Philox RNG",
        "Monte Carlo barostat C rand state"};
    for (std::size_t index = 0; index < modules.size(); ++index)
    {
        const auto dir = Unique_Temp_Path(
            "h5_input_validation_dynamic_unsupported_" + modules[index]);
        std::filesystem::create_directories(dir);
        const auto topology = dir / "system.spgt.h5";
        const auto protocol = dir / "protocol.spgp.h5";
        const auto restart = dir / "restart.spgr.h5";

        Write_Topology_Metadata(topology, 2);
        Write_Protocol_Metadata(protocol, "top");
        Write_Restart_File(restart, 2, false);
        Add_Unsupported_Dynamic_State(restart, modules[index]);

        auto plan = Make_Input_Plan(topology, protocol, restart);
        plan.restart.load_policy =
            SpongeH5InputContract::RestartLoadPolicy::dynamic;
        const auto result =
            SpongeH5InputValidation::Validate_Resolved_Input_Plan(plan);
        Require_Invalid_Contains(result, "unsupported payloads");
        Require_Invalid_Contains(result, modules[index]);
        Require_Invalid_Contains(result, diagnostics[index]);

        std::filesystem::remove_all(dir);
    }
}

static void Test_Recognizes_Portable_Stochastic_Dynamic_State()
{
    for (const char* module : {"middle_langevin", "andersen"})
    {
        RestartDynamicState state;
        state.rng_states[module] =
            SpongeRestartRng::Counter_Philox_State(1234, 5678);
        REQUIRE_TRUE(
            SpongeH5InputValidation::Has_Supported_Dynamic_State(state));
        REQUIRE_TRUE(
            !SpongeH5InputValidation::Has_Unsupported_Dynamic_State(state));
    }

    RestartDynamicState bussi;
    bussi.rng_states["bussi_thermostat"] =
        SpongeRestartRng::Counter_Philox_State(1234, 5678);
    bussi.thermostat_float_states["bussi_thermostat"]["lambda"] = {0.95f};
    REQUIRE_TRUE(SpongeH5InputValidation::Has_Supported_Dynamic_State(bussi));
    REQUIRE_TRUE(
        !SpongeH5InputValidation::Has_Unsupported_Dynamic_State(bussi));

    RestartDynamicState pressure;
    pressure.rng_states["pressure_based_barostat"] =
        SpongeRestartRng::Counter_Philox_State(2468, 1357);
    pressure.barostat_float_states["pressure_based_barostat"]["g"] = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    REQUIRE_TRUE(
        SpongeH5InputValidation::Has_Supported_Dynamic_State(pressure));
    REQUIRE_TRUE(
        !SpongeH5InputValidation::Has_Unsupported_Dynamic_State(pressure));

    bussi.rng_state_text["bussi_thermostat"] = "legacy";
    REQUIRE_TRUE(SpongeH5InputValidation::Has_Unsupported_Dynamic_State(bussi));
    REQUIRE_TRUE(
        SpongeH5InputValidation::Unsupported_Dynamic_State_Reason(bussi).find(
            "both typed and legacy") != std::string::npos);

    RestartDynamicState monte_carlo;
    monte_carlo.rng_states["monte_carlo_barostat"] =
        SpongeRestartRng::Splitmix64_State(0x123456789abcdef0ULL);
    REQUIRE_TRUE(
        !SpongeH5InputValidation::Has_Supported_Dynamic_State(monte_carlo));
    monte_carlo.barostat_float_states["monte_carlo_barostat"]
                                     ["delta_box_length_max"] = {0.1f, 0.2f,
                                                                 0.3f};
    monte_carlo.barostat_float_states["monte_carlo_barostat"]["accept_rate"] = {
        31.0f, 32.0f, 33.0f};
    monte_carlo.barostat_integer_states["monte_carlo_barostat"]
                                       ["total_count_int64"] = {10, 20, 30};
    monte_carlo.barostat_integer_states["monte_carlo_barostat"]
                                       ["accept_count_int64"] = {3, 6, 9};
    REQUIRE_TRUE(
        SpongeH5InputValidation::Has_Supported_Dynamic_State(monte_carlo));
    REQUIRE_TRUE(
        !SpongeH5InputValidation::Has_Unsupported_Dynamic_State(monte_carlo));

    RestartDynamicState malformed;
    malformed.rng_states["middle_langevin"] =
        SpongeRestartRng::Counter_Philox_State(1, 2);
    malformed.rng_states["middle_langevin"].engine = "unknown.engine";
    REQUIRE_TRUE(
        SpongeH5InputValidation::Has_Unsupported_Dynamic_State(malformed));
    REQUIRE_TRUE(
        SpongeH5InputValidation::Unsupported_Dynamic_State_Reason(malformed)
            .find("incompatible schema") != std::string::npos);
}

static void Test_Validates_Protocol_Load_When_Sits_State_Is_Present()
{
    const auto dir = Unique_Temp_Path("h5_input_validation_protocol");
    std::filesystem::create_directories(dir);
    const auto topology = dir / "system.spgt.h5";
    const auto protocol = dir / "protocol.spgp.h5";
    const auto restart = dir / "restart.spgr.h5";

    Write_Topology_Metadata(topology, 2);
    Write_Protocol_Metadata(protocol, "top");
    Write_Restart_File(restart, 2, false, true);

    auto plan = Make_Input_Plan(topology, protocol, restart);
    plan.restart.load_policy =
        SpongeH5InputContract::RestartLoadPolicy::protocol;
    Require_Valid(SpongeH5InputValidation::Validate_Resolved_Input_Plan(plan));

    std::filesystem::remove_all(dir);
}

static void Test_Validates_Protocol_Load_When_Metad_State_Is_Present()
{
    const auto dir = Unique_Temp_Path("h5_input_validation_protocol_metad");
    std::filesystem::create_directories(dir);
    const auto topology = dir / "system.spgt.h5";
    const auto protocol = dir / "protocol.spgp.h5";
    const auto restart = dir / "restart.spgr.h5";

    Write_Topology_Metadata(topology, 2);
    Write_Protocol_Metadata(protocol, "top");
    Write_Restart_File(restart, 2, false, false, true);

    auto plan = Make_Input_Plan(topology, protocol, restart);
    plan.restart.load_policy =
        SpongeH5InputContract::RestartLoadPolicy::protocol;
    Require_Valid(SpongeH5InputValidation::Validate_Resolved_Input_Plan(plan));

    std::filesystem::remove_all(dir);
}

static void Test_Validates_Protocol_Load_When_Sidecar_State_Is_Present()
{
    const auto dir = Unique_Temp_Path("h5_input_validation_protocol_sidecar");
    std::filesystem::create_directories(dir);
    const auto topology = dir / "system.spgt.h5";
    const auto protocol = dir / "protocol.spgp.h5";
    const auto restart = dir / "restart.spgr.h5";

    Write_Topology_Metadata(topology, 2);
    Write_Protocol_Metadata(protocol, "top");
    Write_Restart_File(restart, 2, false, false, false, true);

    auto plan = Make_Input_Plan(topology, protocol, restart);
    plan.restart.load_policy =
        SpongeH5InputContract::RestartLoadPolicy::protocol;
    Require_Valid(SpongeH5InputValidation::Validate_Resolved_Input_Plan(plan));

    std::filesystem::remove_all(dir);
}

static void Test_Validates_Full_Load_When_Supported_States_Are_Present()
{
    const auto dir = Unique_Temp_Path("h5_input_validation_full");
    std::filesystem::create_directories(dir);
    const auto topology = dir / "system.spgt.h5";
    const auto protocol = dir / "protocol.spgp.h5";
    const auto restart = dir / "restart.spgr.h5";

    Write_Topology_Metadata(topology, 2);
    Write_Protocol_Metadata(protocol, "top");
    Write_Restart_File(restart, 2, true, true);

    auto plan = Make_Input_Plan(topology, protocol, restart);
    plan.restart.load_policy = SpongeH5InputContract::RestartLoadPolicy::full;
    Require_Valid(SpongeH5InputValidation::Validate_Resolved_Input_Plan(plan));

    std::filesystem::remove_all(dir);
}

static void Test_Rejects_Protocol_Load_When_Only_Unsupported_State_Is_Present()
{
    const auto dir =
        Unique_Temp_Path("h5_input_validation_protocol_unsupported");
    std::filesystem::create_directories(dir);
    const auto topology = dir / "system.spgt.h5";
    const auto protocol = dir / "protocol.spgp.h5";
    const auto restart = dir / "restart.spgr.h5";

    Write_Topology_Metadata(topology, 2);
    Write_Protocol_Metadata(protocol, "top");
    Write_Restart_File(restart, 2, false);
    Add_Unsupported_Protocol_State(restart);

    auto plan = Make_Input_Plan(topology, protocol, restart);
    plan.restart.load_policy =
        SpongeH5InputContract::RestartLoadPolicy::protocol;
    Require_Invalid_Contains(
        SpongeH5InputValidation::Validate_Resolved_Input_Plan(plan),
        "no currently supported payload");

    std::filesystem::remove_all(dir);
}

static void Test_Rejects_Protocol_Load_When_Protocol_State_Is_Absent()
{
    const auto dir = Unique_Temp_Path("h5_input_validation_protocol_absent");
    std::filesystem::create_directories(dir);
    const auto topology = dir / "system.spgt.h5";
    const auto protocol = dir / "protocol.spgp.h5";
    const auto restart = dir / "restart.spgr.h5";

    Write_Topology_Metadata(topology, 2);
    Write_Protocol_Metadata(protocol, "top");
    Write_Restart_File(restart, 2, false);

    auto plan = Make_Input_Plan(topology, protocol, restart);
    plan.restart.load_policy =
        SpongeH5InputContract::RestartLoadPolicy::protocol;
    Require_Invalid_Contains(
        SpongeH5InputValidation::Validate_Resolved_Input_Plan(plan),
        "protocol state is absent");

    std::filesystem::remove_all(dir);
}

static void Test_Custom_Force_Reader_Loads_Native_Definitions()
{
    const auto dir = Unique_Temp_Path("h5_input_custom_force_native");
    std::filesystem::create_directories(dir);
    const auto topology = dir / "system.spgt.h5";
    {
        HighFive::File file(topology.string(), HighFive::File::Overwrite);
        const std::string pair_root = "/forcefield/custom_force/pairwise";
        Write_Scalar(file, pair_root + "/name", std::string("custom_pair"));
        Write_Scalar(file, pair_root + "/potential",
                     std::string("E = epsilon_ij / r_ij;"));
        Write_String_Vector(file, pair_root + "/parameters/type",
                            {"float", "float"});
        Write_String_Vector(file, pair_root + "/parameters/name",
                            {"epsilon_ij", "sigma_ij"});
        Write_Scalar(file, pair_root + "/parameters/ij_count",
                     static_cast<std::int64_t>(2));
        Write_Scalar(file, pair_root + "/with_ele", 0);
        Write_Scalar(file, pair_root + "/electrostatic_potential",
                     std::string(""));
        const std::string pair_data = pair_root + "/data/custom_pair";
        Write_Scalar(file, pair_data + "/atom_count", 2);
        Write_Scalar(file, pair_data + "/type_count", 1);
        Write_Scalar(file, pair_data + "/pair_count", 1);
        Write_Matrix<float>(file, pair_data + "/parameter/value", {1.0f, 2.0f},
                            2, 1);
        Write_Int_Vector(file, pair_data + "/atom_type", {0, 0});

        const std::string listed_root = "/forcefield/custom_force/listed";
        Write_String_Vector(file, listed_root + "/name", {"custom_bond"});
        Write_String_Vector(file, listed_root + "/potential",
                            {"E = k * (r_ab - r0) * (r_ab - r0);"});
        Write_String_Vector(file, listed_root + "/connected_atoms", {"ab"});
        Write_String_Vector(file, listed_root + "/constrain_distance", {"r0"});
        Write_String_Vector(file, listed_root + "/parameters/type",
                            {"int", "int", "float", "float"});
        Write_String_Vector(file, listed_root + "/parameters/name",
                            {"atom_a", "atom_b", "k", "r0"});
        Write_Int64_Vector(file, listed_root + "/parameters/offset", {0, 4});
        const std::string listed_data = listed_root + "/data/custom_bond";
        Write_Scalar(file, listed_data + "/item_count", 1);
        Write_Matrix<float>(file, listed_data + "/parameter/value",
                            {0.0f, 1.0f, 3.0f, 1.5f}, 1, 4);

        Write_Matrix<int>(file, "/forcefield/bond_soft/atoms", {0, 1}, 1, 2);
        Write_Float_Vector(file, "/forcefield/bond_soft/k", {4.0f});
        Write_Float_Vector(file, "/forcefield/bond_soft/r0", {1.25f});
        Write_Int_Vector(file, "/forcefield/bond_soft/from_a_or_b", {1});
    }

    TopologyCustomForceH5Materializer reader;
    REQUIRE_TRUE(reader.Open(topology.string()));
    NativePairwiseForceDefinition pairwise;
    REQUIRE_TRUE(reader.Read_Pairwise(&pairwise));
    REQUIRE_EQ(pairwise.name, std::string("custom_pair"));
    REQUIRE_EQ(pairwise.ij_parameter_count, 2);
    REQUIRE_EQ(pairwise.parameter_values.size(), static_cast<std::size_t>(2));
    REQUIRE_EQ(pairwise.atom_type.size(), static_cast<std::size_t>(2));

    std::vector<NativeListedForceDefinition> listed;
    REQUIRE_TRUE(reader.Read_Listed(&listed));
    REQUIRE_EQ(listed.size(), static_cast<std::size_t>(1));
    REQUIRE_EQ(listed[0].name, std::string("custom_bond"));
    REQUIRE_EQ(listed[0].item_count, 1);
    REQUIRE_EQ(listed[0].parameter_values.size(), static_cast<std::size_t>(4));

    NativeListedForceDefinition bond_soft;
    REQUIRE_TRUE(reader.Read_Bond_Soft(0.25f, 0.5f, &bond_soft));
    REQUIRE_EQ(bond_soft.name, std::string("bond_soft"));
    REQUIRE_EQ(bond_soft.item_count, 1);
    REQUIRE_EQ(bond_soft.parameter_values.size(), static_cast<std::size_t>(7));
    REQUIRE_TRUE(std::fabs(bond_soft.parameter_values[5] - 0.25f) < 1e-6f);
    REQUIRE_TRUE(std::fabs(bond_soft.parameter_values[6] - 0.5f) < 1e-6f);

    reader = {};
    std::filesystem::remove_all(dir);
}

static void Test_EDIP_Reader_Loads_Dense_Runtime_Definition()
{
    const auto dir = Unique_Temp_Path("h5_input_edip_native");
    std::filesystem::create_directories(dir);
    const auto topology = dir / "system.spgt.h5";
    {
        HighFive::File file(topology.string(), HighFive::File::Overwrite);
        Write_Scalar(file, "/manybody/edip/atom_type_count", 1);
        Write_Int_Vector(file, "/manybody/edip/atom_type", {0, 0});
        Write_Matrix<int>(file, "/manybody/edip/pair/type", {0, 0}, 1, 2);
        Write_Matrix<float>(file, "/manybody/edip/pair/parameters",
                            {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, 1,
                            8);
        Write_Matrix<int>(file, "/manybody/edip/triple/type", {0, 0, 0}, 1, 3);
        Write_Matrix<float>(
            file, "/manybody/edip/triple/parameters",
            {9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f}, 1,
            9);
    }

    TopologyManybodyH5Materializer reader;
    REQUIRE_TRUE(reader.Open(topology.string()));
    NativeEDIPDefinition definition;
    REQUIRE_TRUE(reader.Read_EDIP(&definition));
    REQUIRE_EQ(definition.atom_type_count, 1);
    REQUIRE_EQ(definition.atom_type, std::vector<int>({0, 0}));
    REQUIRE_EQ(definition.pair_parameters.size(), static_cast<std::size_t>(8));
    REQUIRE_EQ(definition.triple_parameters.size(),
               static_cast<std::size_t>(9));
    REQUIRE_TRUE(std::fabs(definition.pair_parameters.front() - 1.0f) < 1e-6f);
    REQUIRE_TRUE(std::fabs(definition.triple_parameters.back() - 17.0f) <
                 1e-6f);

    reader = {};
    std::filesystem::remove_all(dir);
}

static void Test_ReaxFF_Reader_Loads_Typed_Runtime_Definition()
{
    const auto topology = SpongeH5InputMatrix::Full_Contract_Rerun_Path() /
                          "bundled_input" / "bundle" / "topology.spgt.h5";
    TopologyManybodyH5Materializer reader;
    REQUIRE_TRUE(reader.Open(topology.string()));
    NativeReaxFFDefinition definition;
    REQUIRE_TRUE(reader.Read_ReaxFF(&definition));
    REQUIRE_TRUE(definition.general.size() >= static_cast<std::size_t>(39));
    REQUIRE_EQ(definition.atoms.size(), static_cast<std::size_t>(2));
    REQUIRE_EQ(definition.atoms[0].name, std::string("O"));
    REQUIRE_EQ(definition.bonds.size(), static_cast<std::size_t>(1));
    REQUIRE_EQ(definition.bonds[0].type[0], 0);
    REQUIRE_EQ(definition.bonds[0].type[1], 1);
    REQUIRE_EQ(definition.off_diagonal.size(), static_cast<std::size_t>(1));
    REQUIRE_EQ(definition.angles.size(), static_cast<std::size_t>(1));
    REQUIRE_EQ(definition.torsions.size(), static_cast<std::size_t>(1));
    REQUIRE_EQ(definition.hydrogen_bonds.size(), static_cast<std::size_t>(1));
    REQUIRE_EQ(definition.atom_type, std::vector<int>({0, 1}));
}

static void Test_Protocol_Reader_Loads_Native_Hard_Wall()
{
    const auto dir = Unique_Temp_Path("protocol_native_hard_wall");
    std::filesystem::create_directories(dir);
    const auto protocol = dir / "protocol.spgp.h5";
    {
        HighFive::File file(protocol.string(), HighFive::File::Overwrite);
        Write_Float_Vector(file, "/wall/hard/bounds_low",
                           {-INFINITY, -INFINITY, 1.5f});
        Write_Float_Vector(file, "/wall/hard/bounds_high",
                           {INFINITY, INFINITY, 12.0f});
        Write_Scalar(file, "/wall/hard/allow_npt", std::int32_t{1});
    }

    NativeHardWallH5Reader reader;
    REQUIRE_TRUE(reader.Open(protocol.string()));
    REQUIRE_TRUE(reader.Has_Hard_Wall());
    NativeHardWallDefinition definition;
    REQUIRE_TRUE(reader.Read(&definition));
    std::uint32_t low_bits = 0;
    std::memcpy(&low_bits, &definition.bounds_low[0], sizeof(low_bits));
    REQUIRE_EQ(low_bits, std::uint32_t{0xff800000U});
    REQUIRE_TRUE(std::fabs(definition.bounds_low[2] - 1.5f) < 1e-6f);
    REQUIRE_TRUE(std::fabs(definition.bounds_high[2] - 12.0f) < 1e-6f);
    REQUIRE_TRUE(definition.allow_npt);

    const auto invalid_protocol = dir / "invalid_protocol.spgp.h5";
    {
        HighFive::File file(invalid_protocol.string(),
                            HighFive::File::Overwrite);
        Write_Float_Vector(file, "/wall/hard/bounds_low",
                           {-INFINITY, -INFINITY, 1.5f});
        Write_Float_Vector(file, "/wall/hard/bounds_high",
                           {INFINITY, INFINITY, 1.0f});
    }
    NativeHardWallH5Reader invalid_reader;
    REQUIRE_TRUE(invalid_reader.Open(invalid_protocol.string()));
    REQUIRE_TRUE(!invalid_reader.Read(&definition));
    REQUIRE_TRUE(invalid_reader.Last_Error().find(
                     "low bound must be smaller") != std::string::npos);

    reader = {};
    invalid_reader = {};
    std::filesystem::remove_all(dir);
}

static void Test_Protocol_Readers_Share_One_File_Lifetime()
{
    const auto dir = Unique_Temp_Path("protocol_shared_reader_lifetime");
    std::filesystem::create_directories(dir);
    const auto protocol = dir / "protocol.spgp.h5";
    {
        HighFive::File file(protocol.string(), HighFive::File::Overwrite);
        file.createGroup("/cv");
    }

    auto file = std::make_shared<HighFive::File>(protocol.string(),
                                                 HighFive::File::ReadOnly);
    const auto baseline_object_count =
        H5Fget_obj_count(file->getId(), H5F_OBJ_ALL);
    REQUIRE_TRUE(baseline_object_count >= 1);
    {
        ProtocolCVH5Reader cv_reader;
        ProtocolRestraintH5Reader restraint_reader;
        ProtocolMetadynamicsH5Reader metadynamics_reader;
        ProtocolSteeringH5Reader steering_reader;
        REQUIRE_TRUE(cv_reader.Open_Protocol(file));
        REQUIRE_TRUE(restraint_reader.Open(file));
        REQUIRE_TRUE(metadynamics_reader.Open(file));
        REQUIRE_TRUE(steering_reader.Open(file));
        REQUIRE_EQ(file.use_count(), static_cast<long>(5));

        std::vector<ProtocolCVDefinition> cvs;
        std::vector<ProtocolVirtualAtomDefinition> virtual_atoms;
        std::vector<ProtocolCVRestraint> restraints;
        ProtocolMetadynamicsDefinition metadynamics;
        ProtocolSteeringDefinition steering;
        bool has_metadynamics = false;
        bool has_steering = false;
        REQUIRE_TRUE(cv_reader.Read_Definitions(1, &cvs));
        REQUIRE_TRUE(cv_reader.Read_Virtual_Atoms(1, &virtual_atoms));
        REQUIRE_TRUE(restraint_reader.Read_CV_Restraints(&restraints));
        REQUIRE_TRUE(metadynamics_reader.Read_Definition(cvs, &metadynamics,
                                                         &has_metadynamics));
        REQUIRE_TRUE(
            steering_reader.Read_Definition(cvs, &steering, &has_steering));
        REQUIRE_EQ(H5Fget_obj_count(file->getId(), H5F_OBJ_ALL),
                   baseline_object_count);
    }
    REQUIRE_EQ(file.use_count(), static_cast<long>(1));
    file.reset();
    REQUIRE_TRUE(std::filesystem::remove(protocol));
    std::filesystem::remove_all(dir);
}

int main()
{
    return Run_Test(
        []
        {
            Test_Protocol_Reader_Loads_Typed_CV_Restraint();
            Test_Protocol_Reader_Loads_Native_CV_Objects();
            Test_Protocol_Reader_Loads_Typed_Virtual_Atoms();
            Test_Protocol_Reader_Loads_Native_Metadynamics_Object();
            Test_Protocol_Reader_Loads_Native_Steering_Object();
            Test_Positional_Restraint_Reader_Loads_Named_Object();
            Test_Validates_Structural_Bundle_Metadata();
            Test_Rejects_Invalid_Artifact_Schema_And_Identity();
            Test_Rejects_Restart_Lineage_And_State_Hash_Mismatch();
            Test_Validates_Controller_Input_Bindings();
            Test_Controller_Input_Bindings_Propagate_Resolver_Errors();
            Test_Rejects_Restart_Atom_Count_Mismatch();
            Test_Validates_H5md_Rerun_Trajectory_Metadata();
            Test_Rejects_Missing_Requested_Trajectory_Stream();
            Test_Validates_Dynamic_Load_When_Dynamic_State_Is_Present();
            Test_Recognizes_Each_Supported_Dynamic_State_Module();
            Test_Rejects_Dynamic_Load_When_Dynamic_State_Is_Absent();
            Test_Rejects_Dynamic_Load_When_Only_Unsupported_State_Is_Present();
            Test_Recognizes_Portable_Stochastic_Dynamic_State();
            Test_Validates_Protocol_Load_When_Sits_State_Is_Present();
            Test_Validates_Protocol_Load_When_Metad_State_Is_Present();
            Test_Validates_Protocol_Load_When_Sidecar_State_Is_Present();
            Test_Validates_Full_Load_When_Supported_States_Are_Present();
            Test_Rejects_Protocol_Load_When_Only_Unsupported_State_Is_Present();
            Test_Rejects_Protocol_Load_When_Protocol_State_Is_Absent();
            Test_Custom_Force_Reader_Loads_Native_Definitions();
            Test_EDIP_Reader_Loads_Dense_Runtime_Definition();
            Test_ReaxFF_Reader_Loads_Typed_Runtime_Definition();
            Test_Protocol_Reader_Loads_Native_Hard_Wall();
            Test_Protocol_Readers_Share_One_File_Lifetime();
        });
}

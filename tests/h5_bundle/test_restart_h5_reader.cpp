#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <highfive/highfive.hpp>
#include <string>
#include <vector>

#include "h5_bundle_test_common.hpp"
#include "utils/h5md/highfive_backend.hpp"
#include "utils/h5md/restart_h5_reader.hpp"
#include "utils/h5md/restart_h5_writer.hpp"
#include "utils/random/restart_rng_state.hpp"

using namespace SpongeH5Test;
using namespace SpongeH5MD;

static SpongeH5OutputPlan::ResolvedOutputPlan Make_Restart_Plan(
    const std::filesystem::path& restart_path)
{
    SpongeH5OutputPlan::ResolvedOutputPlan plan;
    plan.restart.enabled = true;
    plan.restart.path = restart_path.string();
    return plan;
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

static void Require_Box_Close(const std::array<float, 9>& actual,
                              const std::array<float, 9>& expected)
{
    for (std::size_t i = 0; i < expected.size(); ++i)
    {
        REQUIRE_TRUE(std::fabs(actual[i] - expected[i]) < 1.0e-6f);
    }
}

static void Require_Writer(bool ok, const RestartH5Writer& writer,
                           const char* operation)
{
    if (!ok)
    {
        throw TestFailure(std::string(operation) + ": " + writer.Last_Error());
    }
}

static void Require_Reader(bool ok, const RestartH5Reader& reader,
                           const char* operation)
{
    if (!ok)
    {
        throw TestFailure(std::string(operation) + ": " + reader.Last_Error());
    }
}

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

static void Add_Float_Vector_Dataset(const std::filesystem::path& file_path,
                                     const std::string& dataset_path,
                                     const std::vector<float>& values)
{
    HighFive::File file(file_path.string(), HighFive::File::ReadWrite);
    Ensure_Parent_Group(file, dataset_path);
    auto dataset = file.createDataSet<float>(dataset_path,
                                             HighFive::DataSpace::From(values));
    dataset.write(values);
}

static void Write_Restart_File(const std::filesystem::path& file_path,
                               bool include_velocity)
{
    const std::vector<float> position = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
    };
    const std::vector<float> velocity = {
        0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f,
    };
    const std::array<float, 9> box = {
        10.0f, 0.0f, 0.0f, 0.0f, 20.0f, 0.0f, 0.0f, 0.0f, 30.0f,
    };

    HighFiveBackend backend;
    RestartH5Writer writer(&backend);
    Require_Writer(writer.Open(Make_Restart_Plan(file_path), "1"), writer,
                   "open restart writer");
    Require_Writer(writer.Define_Structural_State(3, include_velocity), writer,
                   "define structural restart state");
    Require_Writer(writer.Write_Structural_State(
                       42, 0.084, position.data(), box.data(),
                       include_velocity ? velocity.data() : nullptr),
                   writer, "write structural restart state");
    Require_Writer(writer.Finalize(), writer, "finalize restart writer");
    Require_Writer(writer.Close(), writer, "close restart writer");
}

static void Write_Restart_File_With_Module_State(
    const std::filesystem::path& file_path)
{
    const std::vector<float> position = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
    };
    const std::vector<float> velocity = {
        0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
    };
    const std::array<float, 9> box = {
        10.0f, 0.0f, 0.0f, 0.0f, 20.0f, 0.0f, 0.0f, 0.0f, 30.0f,
    };
    const std::vector<float> nhc_state = {
        0.25f,
        0.50f,
        0.75f,
        1.00f,
    };
    const std::vector<float> sits_nk = {2.0f, 3.0f, 5.0f};
    const std::vector<float> sits_log_norm = {-4.0f, -3.0f, -2.0f};
    const std::vector<float> sits_log_nk = {0.693147f, 1.098612f, 1.609438f};
    const std::vector<float> restraint_reference = {
        1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f,
    };
    const std::vector<float> cv_reference = {
        0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f,
    };

    HighFiveBackend backend;
    RestartH5Writer writer(&backend);
    Require_Writer(writer.Open(Make_Restart_Plan(file_path), "1"), writer,
                   "open restart writer with module state");
    Require_Writer(writer.Define_Structural_State(2, true), writer,
                   "define structural restart state with module state");
    Require_Writer(writer.Write_Structural_State(7, 0.014, position.data(),
                                                 box.data(), velocity.data()),
                   writer, "write structural restart state with module state");
    Require_Writer(writer.Write_Nose_Hoover_Chain_State(nhc_state.data(), 2),
                   writer, "write NHC restart state");
    Require_Writer(writer.Write_Integrator_State_Text("mode", "npt"), writer,
                   "write integrator restart mode");
    Require_Writer(writer.Write_Rng_State(
                       "bussi_thermostat",
                       SpongeRestartRng::Counter_Philox_State(12345, 67890)),
                   writer, "write Bussi typed RNG restart state");
    const float bussi_lambda[1] = {0.95f};
    Require_Writer(writer.Write_Thermostat_State_Float(
                       "bussi_thermostat", "lambda", bussi_lambda, 1),
                   writer, "write Bussi lambda restart state");
    const float pressure_g[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Require_Writer(writer.Write_Rng_State(
                       "pressure_based_barostat",
                       SpongeRestartRng::Counter_Philox_State(24680, 13579)),
                   writer, "write pressure barostat typed RNG restart state");
    Require_Writer(writer.Write_Barostat_State_Float("pressure_based_barostat",
                                                     "g", pressure_g, 6),
                   writer, "write pressure barostat g restart state");
    Require_Writer(writer.Write_Rng_State(
                       "middle_langevin",
                       SpongeRestartRng::Counter_Philox_State(1234, 5678)),
                   writer, "write Middle Langevin typed RNG restart state");
    Require_Writer(
        writer.Write_Rng_State(
            "andersen", SpongeRestartRng::Counter_Philox_State(4321, 8765)),
        writer, "write Andersen typed RNG restart state");
    Require_Writer(
        writer.Write_Rng_State(
            "monte_carlo_barostat",
            SpongeRestartRng::Splitmix64_State(0x123456789abcdef0ULL)),
        writer, "write Monte Carlo barostat typed RNG restart state");
    const float mc_delta[3] = {0.10f, 0.20f, 0.30f};
    const float mc_rate[3] = {31.0f, 32.0f, 33.0f};
    const std::int64_t mc_total[3] = {10, 20, 30};
    const std::int64_t mc_accept[3] = {3, 6, 9};
    Require_Writer(
        writer.Write_Barostat_State_Float("monte_carlo_barostat",
                                          "delta_box_length_max", mc_delta, 3),
        writer, "write Monte Carlo barostat delta state");
    Require_Writer(writer.Write_Barostat_State_Float("monte_carlo_barostat",
                                                     "accept_rate", mc_rate, 3),
                   writer, "write Monte Carlo barostat rate state");
    Require_Writer(
        writer.Write_Barostat_State_Int64("monte_carlo_barostat",
                                          "total_count_int64", mc_total, 3),
        writer, "write Monte Carlo barostat total count state");
    Require_Writer(
        writer.Write_Barostat_State_Int64("monte_carlo_barostat",
                                          "accept_count_int64", mc_accept, 3),
        writer, "write Monte Carlo barostat accept count state");
    Require_Writer(writer.Write_Sits_State_Schema_Version("sits_bias", 1),
                   writer, "write SITS restart schema version");
    Require_Writer(writer.Write_Sits_State("sits_bias", "nk", sits_nk.data(),
                                           sits_nk.size()),
                   writer, "write SITS nk restart state");
    Require_Writer(
        writer.Write_Sits_State("sits_bias", "log_norm", sits_log_norm.data(),
                                sits_log_norm.size()),
        writer, "write SITS log_norm restart state");
    Require_Writer(
        writer.Write_Sits_State("sits_bias", "log_nk", sits_log_nk.data(),
                                sits_log_nk.size()),
        writer, "write SITS log_nk restart state");
    Require_Writer(
        writer.Write_Metad_State_Text("metad_bias", "hills", "HILLS_PAYLOAD"),
        writer, "write metad hills restart state");
    Require_Writer(writer.Write_Metad_State_Text("metad_bias", "history",
                                                 "HISTORY_PAYLOAD"),
                   writer, "write metad history restart state");
    RestartMetadynamicsState typed_metadynamics;
    typed_metadynamics.name = "typed_bias";
    typed_metadynamics.has_typed_state = true;
    typed_metadynamics.ndim = 2;
    typed_metadynamics.grid_count = {2, 2};
    typed_metadynamics.grid_min = {0.0f, -1.0f};
    typed_metadynamics.grid_max = {1.0f, 1.0f};
    typed_metadynamics.potential_value = {1.0f, 2.0f, 3.0f, 4.0f};
    typed_metadynamics.potential_force = {0.1f, 0.2f, 0.3f, 0.4f,
                                          0.5f, 0.6f, 0.7f, 0.8f};
    typed_metadynamics.edge_log_normalization = {-2.0f, -1.0f, 0.0f, 1.0f};
    typed_metadynamics.edge_normal_force = {-0.1f, -0.2f, -0.3f, -0.4f,
                                            -0.5f, -0.6f, -0.7f, -0.8f};
    typed_metadynamics.scatter_count = 2;
    typed_metadynamics.scatter_position = {0.25f, -0.5f, 0.75f, 0.5f};
    typed_metadynamics.scatter_potential = {5.0f, 6.0f};
    typed_metadynamics.scatter_force = {0.9f, 1.0f, 1.1f, 1.2f};
    typed_metadynamics.hill_count = 2;
    typed_metadynamics.hill_center = {0.1f, -0.2f, 0.8f, 0.4f};
    typed_metadynamics.hill_height = {1.5f, 1.6f};
    typed_metadynamics.hill_inverse_width = {2.0f, 4.0f, 2.0f, 4.0f};
    typed_metadynamics.hill_period = {0.0f, 6.28f, 0.0f, 6.28f};
    typed_metadynamics.hill_sink = {0.25f, 0.5f};
    typed_metadynamics.has_runtime_state = true;
    typed_metadynamics.potential_max = 6.0f;
    typed_metadynamics.sum_max = 1.0f;
    typed_metadynamics.new_max = 0.75f;
    typed_metadynamics.max_index = 1;
    typed_metadynamics.exit_tag = 2.0f;
    typed_metadynamics.rct = 0.2f;
    typed_metadynamics.rbias = 0.3f;
    typed_metadynamics.bias = 0.4f;
    typed_metadynamics.minus_beta_f = 0.5f;
    typed_metadynamics.minus_beta_f_plus_v = 0.6f;
    Require_Writer(writer.Write_Metadynamics_State(typed_metadynamics), writer,
                   "write typed metadynamics restart state");
    Require_Writer(writer.Write_Restraint_Reference(
                       "backbone", restraint_reference.data(), 2),
                   writer, "write restraint reference state");
    Require_Writer(writer.Write_CV_Reference("rmsd", cv_reference.data(), 2),
                   writer, "write CV reference state");
    Require_Writer(
        writer.Write_Protocol_Sidecar_Text("cv_in_file", "CV_PAYLOAD"), writer,
        "write protocol sidecar restart state");
    Require_Writer(writer.Finalize(), writer,
                   "finalize restart writer with module state");
    Require_Writer(writer.Close(), writer,
                   "close restart writer with module state");
}

static void Test_Restart_Reader_Round_Trips_Structural_State()
{
    const auto dir = Unique_Temp_Path("restart_reader");
    std::filesystem::create_directories(dir);
    const auto file_path = dir / "prod.spgr.h5";

    Write_Restart_File(file_path, true);

    RestartH5Reader reader;
    Require_Reader(reader.Open(file_path.string()), reader,
                   "open restart reader");

    SpongeH5InputMetadata::RestartMetadata metadata;
    Require_Reader(reader.Read_Metadata(&metadata), reader,
                   "read restart metadata");
    REQUIRE_EQ(metadata.schema_version, std::string("1"));
    REQUIRE_EQ(metadata.atom_count, static_cast<std::int64_t>(3));
    REQUIRE_TRUE(metadata.has_structural_state);
    REQUIRE_TRUE(metadata.has_velocity);
    REQUIRE_TRUE(!metadata.has_protocol_state);

    RestartStructuralState state;
    Require_Reader(reader.Read_Structural_State(&state), reader,
                   "read restart structural state");
    REQUIRE_EQ(state.step, static_cast<std::int64_t>(42));
    REQUIRE_TRUE(std::fabs(state.time - 0.084) < 1.0e-12);
    REQUIRE_EQ(state.atom_count, static_cast<std::size_t>(3));
    REQUIRE_TRUE(state.has_velocity);

    Require_Float_Vector_Close(
        state.position_xyz,
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
    Require_Float_Vector_Close(
        state.velocity_xyz,
        {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f});
    Require_Box_Close(state.box_edges, {10.0f, 0.0f, 0.0f, 0.0f, 20.0f, 0.0f,
                                        0.0f, 0.0f, 30.0f});

    reader =
        RestartH5Reader{};  // Release the file before deleting it on Windows.
    std::filesystem::remove_all(dir);
}

static void Test_Restart_Reader_Allows_No_Velocity_State()
{
    const auto dir = Unique_Temp_Path("restart_reader_no_velocity");
    std::filesystem::create_directories(dir);
    const auto file_path = dir / "prod.spgr.h5";

    Write_Restart_File(file_path, false);

    RestartH5Reader reader;
    Require_Reader(reader.Open(file_path.string()), reader,
                   "open restart reader without velocity");

    SpongeH5InputMetadata::RestartMetadata metadata;
    Require_Reader(reader.Read_Metadata(&metadata), reader,
                   "read restart metadata without velocity");
    REQUIRE_TRUE(metadata.has_structural_state);
    REQUIRE_TRUE(!metadata.has_velocity);

    RestartStructuralState state;
    Require_Reader(reader.Read_Structural_State(&state), reader,
                   "read restart structural state without velocity");
    REQUIRE_TRUE(!state.has_velocity);
    REQUIRE_TRUE(state.velocity_xyz.empty());

    reader =
        RestartH5Reader{};  // Release the file before deleting it on Windows.
    std::filesystem::remove_all(dir);
}

static void Test_Restart_Reader_Reports_Unsupported_Dynamic_State_As_Metadata()
{
    const auto dir = Unique_Temp_Path("restart_reader_unsupported_dynamic");
    std::filesystem::create_directories(dir);
    const auto file_path = dir / "prod.spgr.h5";

    Write_Restart_File(file_path, true);
    {
        HighFive::File file(file_path.string(), HighFive::File::ReadWrite);
        Ensure_Parent_Group(
            file, SpongeH5MD::Restart_Rng_State_Path("middle_langevin"));
        auto dataset = file.createDataSet<std::string>(
            SpongeH5MD::Restart_Rng_State_Path("middle_langevin"),
            HighFive::DataSpace::From(std::string("unsupported")));
        dataset.write(std::string("unsupported"));
    }

    RestartH5Reader reader;
    Require_Reader(reader.Open(file_path.string()), reader,
                   "open restart reader with unsupported dynamic state");

    SpongeH5InputMetadata::RestartMetadata metadata;
    Require_Reader(reader.Read_Metadata(&metadata), reader,
                   "read restart metadata with unsupported dynamic state");
    REQUIRE_TRUE(metadata.has_dynamic_state);

    RestartDynamicState dynamic_state;
    Require_Reader(reader.Read_Dynamic_State(&dynamic_state), reader,
                   "read unsupported dynamic state");
    REQUIRE_TRUE(!dynamic_state.has_nose_hoover_chain);

    reader =
        RestartH5Reader{};  // Release the file before deleting it on Windows.
    std::filesystem::remove_all(dir);
}

static void Test_Restart_Reader_Round_Trips_Dynamic_And_Protocol_State()
{
    const auto dir = Unique_Temp_Path("restart_reader_module_state");
    std::filesystem::create_directories(dir);
    const auto file_path = dir / "prod.spgr.h5";

    Write_Restart_File_With_Module_State(file_path);

    RestartH5Reader reader;
    Require_Reader(reader.Open(file_path.string()), reader,
                   "open restart reader with module state");

    SpongeH5InputMetadata::RestartMetadata metadata;
    Require_Reader(reader.Read_Metadata(&metadata), reader,
                   "read restart metadata with module state");
    REQUIRE_TRUE(metadata.has_dynamic_state);
    REQUIRE_TRUE(metadata.has_protocol_state);

    RestartDynamicState dynamic_state;
    Require_Reader(reader.Read_Dynamic_State(&dynamic_state), reader,
                   "read restart dynamic state");
    REQUIRE_TRUE(dynamic_state.has_nose_hoover_chain);
    REQUIRE_EQ(dynamic_state.nose_hoover_chain_pair_count,
               static_cast<std::size_t>(2));
    Require_Float_Vector_Close(
        dynamic_state.nose_hoover_chain_coordinate_velocity_pairs,
        {0.25f, 0.50f, 0.75f, 1.00f});
    REQUIRE_EQ(dynamic_state.integrator_state_text["mode"], std::string("npt"));
    std::uint64_t seed = 0;
    std::uint64_t invocation_count = 0;
    std::string rng_error;
    REQUIRE_TRUE(SpongeRestartRng::Decode_Counter_Philox_State(
        dynamic_state.rng_states["bussi_thermostat"], &seed, &invocation_count,
        &rng_error));
    REQUIRE_EQ(seed, static_cast<std::uint64_t>(12345));
    REQUIRE_EQ(invocation_count, static_cast<std::uint64_t>(67890));
    Require_Float_Vector_Close(
        dynamic_state.thermostat_float_states["bussi_thermostat"]["lambda"],
        {0.95f});
    REQUIRE_TRUE(SpongeRestartRng::Decode_Counter_Philox_State(
        dynamic_state.rng_states["pressure_based_barostat"], &seed,
        &invocation_count, &rng_error));
    REQUIRE_EQ(seed, static_cast<std::uint64_t>(24680));
    REQUIRE_EQ(invocation_count, static_cast<std::uint64_t>(13579));
    Require_Float_Vector_Close(
        dynamic_state.barostat_float_states["pressure_based_barostat"]["g"],
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    REQUIRE_TRUE(SpongeRestartRng::Decode_Counter_Philox_State(
        dynamic_state.rng_states["middle_langevin"], &seed, &invocation_count,
        &rng_error));
    REQUIRE_EQ(seed, static_cast<std::uint64_t>(1234));
    REQUIRE_EQ(invocation_count, static_cast<std::uint64_t>(5678));
    REQUIRE_TRUE(SpongeRestartRng::Decode_Counter_Philox_State(
        dynamic_state.rng_states["andersen"], &seed, &invocation_count,
        &rng_error));
    REQUIRE_EQ(seed, static_cast<std::uint64_t>(4321));
    REQUIRE_EQ(invocation_count, static_cast<std::uint64_t>(8765));
    std::uint64_t mc_rng_state = 0;
    REQUIRE_TRUE(SpongeRestartRng::Decode_Splitmix64_State(
        dynamic_state.rng_states["monte_carlo_barostat"], &mc_rng_state,
        &rng_error));
    REQUIRE_EQ(mc_rng_state, 0x123456789abcdef0ULL);
    Require_Float_Vector_Close(
        dynamic_state.barostat_float_states["monte_carlo_barostat"]
                                           ["delta_box_length_max"],
        {0.10f, 0.20f, 0.30f});
    REQUIRE_EQ(dynamic_state.barostat_integer_states["monte_carlo_barostat"]
                                                    ["total_count_int64"],
               std::vector<std::int64_t>({10, 20, 30}));
    REQUIRE_EQ(dynamic_state.barostat_integer_states["monte_carlo_barostat"]
                                                    ["accept_count_int64"],
               std::vector<std::int64_t>({3, 6, 9}));

    RestartProtocolState protocol_state;
    Require_Reader(reader.Read_Protocol_State(&protocol_state), reader,
                   "read restart protocol state");
    REQUIRE_EQ(protocol_state.sits_states.size(), static_cast<std::size_t>(1));
    REQUIRE_EQ(protocol_state.sits_states[0].module_name,
               std::string("sits_bias"));
    REQUIRE_EQ(protocol_state.sits_states[0].state_schema_version,
               static_cast<std::int64_t>(1));
    REQUIRE_EQ(protocol_state.sits_states[0].float_states.count("nk"),
               static_cast<std::size_t>(1));
    REQUIRE_EQ(protocol_state.sits_states[0].float_states.count("log_norm"),
               static_cast<std::size_t>(1));
    REQUIRE_EQ(protocol_state.sits_states[0].float_states.count("log_nk"),
               static_cast<std::size_t>(1));
    Require_Float_Vector_Close(protocol_state.sits_states[0].float_states["nk"],
                               {2.0f, 3.0f, 5.0f});
    Require_Float_Vector_Close(
        protocol_state.sits_states[0].float_states["log_norm"],
        {-4.0f, -3.0f, -2.0f});
    Require_Float_Vector_Close(
        protocol_state.sits_states[0].float_states["log_nk"],
        {0.693147f, 1.098612f, 1.609438f});

    REQUIRE_EQ(protocol_state.metadynamics_states.size(),
               static_cast<std::size_t>(2));
    const auto text_metadynamics =
        std::find_if(protocol_state.metadynamics_states.begin(),
                     protocol_state.metadynamics_states.end(),
                     [](const RestartMetadynamicsState& value)
                     { return value.name == "metad_bias"; });
    REQUIRE_TRUE(text_metadynamics != protocol_state.metadynamics_states.end());
    REQUIRE_EQ(text_metadynamics->text_states["hills"],
               std::string("HILLS_PAYLOAD"));
    REQUIRE_EQ(text_metadynamics->text_states["history"],
               std::string("HISTORY_PAYLOAD"));
    const auto typed_metadynamics =
        std::find_if(protocol_state.metadynamics_states.begin(),
                     protocol_state.metadynamics_states.end(),
                     [](const RestartMetadynamicsState& value)
                     { return value.name == "typed_bias"; });
    REQUIRE_TRUE(typed_metadynamics !=
                 protocol_state.metadynamics_states.end());
    REQUIRE_TRUE(typed_metadynamics->has_typed_state);
    REQUIRE_EQ(typed_metadynamics->state_schema_version,
               static_cast<std::int64_t>(1));
    REQUIRE_EQ(typed_metadynamics->ndim, static_cast<std::size_t>(2));
    REQUIRE_EQ(typed_metadynamics->grid_count,
               std::vector<std::int64_t>({2, 2}));
    Require_Float_Vector_Close(typed_metadynamics->grid_min, {0.0f, -1.0f});
    Require_Float_Vector_Close(typed_metadynamics->grid_max, {1.0f, 1.0f});
    Require_Float_Vector_Close(typed_metadynamics->potential_value,
                               {1.0f, 2.0f, 3.0f, 4.0f});
    Require_Float_Vector_Close(
        typed_metadynamics->potential_force,
        {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f});
    Require_Float_Vector_Close(typed_metadynamics->edge_log_normalization,
                               {-2.0f, -1.0f, 0.0f, 1.0f});
    REQUIRE_EQ(typed_metadynamics->scatter_count, static_cast<std::size_t>(2));
    Require_Float_Vector_Close(typed_metadynamics->scatter_position,
                               {0.25f, -0.5f, 0.75f, 0.5f});
    REQUIRE_EQ(typed_metadynamics->hill_count, static_cast<std::size_t>(2));
    Require_Float_Vector_Close(typed_metadynamics->hill_center,
                               {0.1f, -0.2f, 0.8f, 0.4f});
    Require_Float_Vector_Close(typed_metadynamics->hill_height, {1.5f, 1.6f});
    Require_Float_Vector_Close(typed_metadynamics->hill_sink, {0.25f, 0.5f});
    REQUIRE_TRUE(typed_metadynamics->has_runtime_state);
    REQUIRE_TRUE(std::fabs(typed_metadynamics->potential_max - 6.0f) < 1.0e-6f);
    REQUIRE_EQ(typed_metadynamics->max_index, static_cast<std::int64_t>(1));
    REQUIRE_TRUE(std::fabs(typed_metadynamics->minus_beta_f_plus_v - 0.6f) <
                 1.0e-6f);
    REQUIRE_EQ(protocol_state.restraint_states.size(),
               static_cast<std::size_t>(1));
    REQUIRE_EQ(protocol_state.restraint_states[0].name,
               std::string("backbone"));
    REQUIRE_EQ(protocol_state.restraint_states[0].atom_count,
               static_cast<std::size_t>(2));
    Require_Float_Vector_Close(
        protocol_state.restraint_states[0].reference_coordinates,
        {1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f});
    REQUIRE_EQ(protocol_state.cv_reference_states.size(),
               static_cast<std::size_t>(1));
    REQUIRE_EQ(protocol_state.cv_reference_states[0].name, std::string("rmsd"));
    REQUIRE_EQ(protocol_state.cv_reference_states[0].atom_count,
               static_cast<std::size_t>(2));
    Require_Float_Vector_Close(
        protocol_state.cv_reference_states[0].reference_coordinates,
        {0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f});
    REQUIRE_EQ(protocol_state.sidecar_text_states.size(),
               static_cast<std::size_t>(1));
    REQUIRE_EQ(protocol_state.sidecar_text_states[0].key,
               std::string("cv_in_file"));
    REQUIRE_EQ(protocol_state.sidecar_text_states[0].text,
               std::string("CV_PAYLOAD"));

    reader =
        RestartH5Reader{};  // Release the file before deleting it on Windows.
    std::filesystem::remove_all(dir);
}

static void Test_Restart_Reader_Reports_Open_Failure()
{
    RestartH5Reader reader;
    REQUIRE_TRUE(!reader.Open("/tmp/sponge_restart_reader_missing.spgr.h5"));
    REQUIRE_TRUE(reader.Last_Error().find("failed to open restart H5 file") !=
                 std::string::npos);
}

int main()
{
    return Run_Test(
        []
        {
            Test_Restart_Reader_Round_Trips_Structural_State();
            Test_Restart_Reader_Allows_No_Velocity_State();
            Test_Restart_Reader_Reports_Unsupported_Dynamic_State_As_Metadata();
            Test_Restart_Reader_Round_Trips_Dynamic_And_Protocol_State();
            Test_Restart_Reader_Reports_Open_Failure();
        });
}

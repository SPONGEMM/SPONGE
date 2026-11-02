#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "utils/h5md/input_assembler.hpp"

struct TestVector
{
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

static void Require(bool value)
{
    if (!value)
    {
        throw std::runtime_error("requirement failed");
    }
}

static void Test_Apply_Structural_State_With_Velocity()
{
    SpongeH5MD::RestartStructuralState state;
    state.atom_count = 2;
    state.time = 1.25;
    state.position_xyz = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    state.velocity_xyz = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
    state.has_velocity = true;
    state.box_edges = {
        10.0f, 0.0f, 0.0f, 0.0f, 20.0f, 0.0f, 0.0f, 0.0f, 30.0f,
    };

    TestVector coordinate[2];
    TestVector velocity[2];
    TestVector box_length;
    TestVector box_angle;
    double start_time = 0.0;
    std::string error;
    Require(SpongeH5MD::Apply_Restart_Structural_State(
        state, 2, coordinate, velocity, &box_length, &box_angle, &start_time,
        &error));

    Require(coordinate[1].x == 4.0f);
    Require(coordinate[1].y == 5.0f);
    Require(coordinate[1].z == 6.0f);
    Require(velocity[0].x == 0.1f);
    Require(velocity[1].z == 0.6f);
    Require(box_length.x == 10.0f);
    Require(box_length.y == 20.0f);
    Require(box_length.z == 30.0f);
    Require(std::fabs(box_angle.x - 90.0f) < 1.0e-5f);
    Require(std::fabs(box_angle.y - 90.0f) < 1.0e-5f);
    Require(std::fabs(box_angle.z - 90.0f) < 1.0e-5f);
    Require(std::fabs(start_time - 1.25) < 1.0e-12);
}

static void Test_Apply_Structural_State_Zero_Fills_Missing_Velocity()
{
    SpongeH5MD::RestartStructuralState state;
    state.atom_count = 1;
    state.position_xyz = {1.0f, 2.0f, 3.0f};
    state.box_edges = {
        1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
    };

    TestVector coordinate[1];
    TestVector velocity[1];
    TestVector box_length;
    TestVector box_angle;
    double start_time = 7.0;
    std::string error;
    Require(SpongeH5MD::Apply_Restart_Structural_State(
        state, 1, coordinate, velocity, &box_length, &box_angle, &start_time,
        &error));
    Require(velocity[0].x == 0.0f);
    Require(velocity[0].y == 0.0f);
    Require(velocity[0].z == 0.0f);
}

static void Test_Apply_Structural_State_Rejects_Atom_Count_Mismatch()
{
    SpongeH5MD::RestartStructuralState state;
    state.atom_count = 2;
    state.position_xyz = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    TestVector coordinate[1];
    TestVector velocity[1];
    TestVector box_length;
    TestVector box_angle;
    double start_time = 0.0;
    std::string error;
    Require(!SpongeH5MD::Apply_Restart_Structural_State(
        state, 1, coordinate, velocity, &box_length, &box_angle, &start_time,
        &error));
    Require(error.find("atom count mismatch") != std::string::npos);
}

static void Test_Apply_Nose_Hoover_Chain_Dynamic_State()
{
    SpongeH5MD::RestartDynamicState state;
    state.has_nose_hoover_chain = true;
    state.nose_hoover_chain_pair_count = 2;
    state.nose_hoover_chain_coordinate_velocity_pairs = {
        1.0f,
        0.1f,
        2.0f,
        0.2f,
    };

    float coordinate[2] = {};
    float velocity[3] = {-1.0f, -1.0f, -1.0f};
    std::string error;
    Require(SpongeH5MD::Apply_Nose_Hoover_Chain_Dynamic_State(
        state, 2, coordinate, velocity, &error));
    Require(coordinate[0] == 1.0f);
    Require(coordinate[1] == 2.0f);
    Require(velocity[0] == 0.1f);
    Require(velocity[1] == 0.2f);
    Require(velocity[2] == 0.0f);
}

static void Test_Apply_Nose_Hoover_Chain_Dynamic_State_Rejects_Length_Mismatch()
{
    SpongeH5MD::RestartDynamicState state;
    state.has_nose_hoover_chain = true;
    state.nose_hoover_chain_pair_count = 1;
    state.nose_hoover_chain_coordinate_velocity_pairs = {1.0f, 0.1f};

    float coordinate[2] = {};
    float velocity[3] = {};
    std::string error;
    Require(!SpongeH5MD::Apply_Nose_Hoover_Chain_Dynamic_State(
        state, 2, coordinate, velocity, &error));
    Require(error.find("chain length mismatch") != std::string::npos);
}

static void Test_Extract_Sits_Nk_Protocol_State()
{
    SpongeH5MD::RestartProtocolState state;
    SpongeH5MD::RestartSitsState sits_state;
    sits_state.module_name = "SITS";
    sits_state.float_states["nk"] = {1.0f, 2.0f, 4.0f};
    state.sits_states.push_back(sits_state);

    std::vector<float> nk_values;
    std::string error;
    Require(SpongeH5MD::Extract_Sits_Nk_Protocol_State(state, "SITS", 3,
                                                       &nk_values, &error));
    Require(nk_values.size() == 3);
    Require(nk_values[0] == 1.0f);
    Require(nk_values[2] == 4.0f);
}

static void Test_Extract_Sits_Nk_Protocol_State_Rejects_Bad_State()
{
    SpongeH5MD::RestartProtocolState state;
    SpongeH5MD::RestartSitsState sits_state;
    sits_state.module_name = "SITS";
    sits_state.float_states["nk"] = {1.0f, 0.0f};
    state.sits_states.push_back(sits_state);

    std::vector<float> nk_values;
    std::string error;
    Require(!SpongeH5MD::Extract_Sits_Nk_Protocol_State(state, "SITS", 2,
                                                        &nk_values, &error));
    Require(error.find("positive finite") != std::string::npos);

    // Parse at runtime so the regression exercises Release/fast-math checks.
    for (const char* text : {"inf", "-inf", "nan"})
    {
        state.sits_states[0].float_states["nk"] = {1.0f,
                                                   std::strtof(text, nullptr)};
        error.clear();
        Require(!SpongeH5MD::Extract_Sits_Nk_Protocol_State(
            state, "SITS", 2, &nk_values, &error));
        Require(error.find("positive finite") != std::string::npos);
    }

    sits_state.float_states["nk"] = {1.0f};
    state.sits_states.clear();
    state.sits_states.push_back(sits_state);
    Require(!SpongeH5MD::Extract_Sits_Nk_Protocol_State(state, "SITS", 2,
                                                        &nk_values, &error));
    Require(error.find("nk count mismatch") != std::string::npos);
}

static std::filesystem::path Unique_Temp_Dir(const std::string& name)
{
    const auto stamp =
        std::chrono::high_resolution_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path() /
           ("sponge_h5_input_assembler_" + std::to_string(stamp) + "_" + name);
}

static std::string Read_Text_File(const std::filesystem::path& path)
{
    std::ifstream input(path.c_str(), std::ios::in | std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(input)),
                       std::istreambuf_iterator<char>());
}

static void Test_Materialize_Metadynamics_Text_State()
{
    const auto dir = Unique_Temp_Dir("metad");
    std::filesystem::create_directories(dir);

    SpongeH5MD::RestartProtocolState state;
    SpongeH5MD::RestartMetadynamicsState metad_state;
    metad_state.name = "meta";
    metad_state.text_states["hills"] = "HILLS\n";
    metad_state.text_states["history"] = "HISTORY\n";
    metad_state.text_states["edge"] = "EDGE\n";
    state.metadynamics_states.push_back(metad_state);

    bool materialized = false;
    std::string error;
    Require(SpongeH5MD::Materialize_Metadynamics_Text_State(
        state, "meta", (dir / "myhill.log").string(),
        (dir / "history.log").string(), (dir / "sumhill.log").string(),
        (dir / "Meta_Potential.txt").string(),
        (dir / "Meta_directly.txt").string(), &materialized, &error));
    Require(materialized);
    Require(Read_Text_File(dir / "myhill.log") == "HILLS\n");
    Require(Read_Text_File(dir / "history.log") == "HISTORY\n");
    Require(Read_Text_File(dir / "sumhill.log") == "EDGE\n");

    std::filesystem::remove_all(dir);
}

static void Test_Materialize_Metadynamics_Text_State_Rejects_Missing_Module()
{
    SpongeH5MD::RestartProtocolState state;
    SpongeH5MD::RestartMetadynamicsState metad_state;
    metad_state.name = "other";
    metad_state.text_states["hills"] = "HILLS\n";
    state.metadynamics_states.push_back(metad_state);

    bool materialized = false;
    std::string error;
    Require(!SpongeH5MD::Materialize_Metadynamics_Text_State(
        state, "meta", "myhill.log", "history.log", "sumhill.log",
        "Meta_Potential.txt", "Meta_directly.txt", &materialized, &error));
    Require(!materialized);
    Require(error.find("does not contain metadynamics state") !=
            std::string::npos);
}

static void Test_Materialize_Protocol_Sidecar_Text_State()
{
    const auto dir = Unique_Temp_Dir("protocol_sidecar");
    SpongeH5MD::RestartProtocolState state;
    SpongeH5MD::RestartProtocolSidecarTextState sidecar;
    sidecar.key = "cv_in_file";
    sidecar.text = "CV_PAYLOAD\n";
    state.sidecar_text_states.push_back(sidecar);

    std::vector<SpongeH5MD::LegacySidecarBinding> sidecars;
    std::string error;
    Require(SpongeH5MD::Materialize_Protocol_Sidecar_Text_State(
        state, dir.string(), &sidecars, &error));
    Require(sidecars.size() == 1);
    Require(sidecars[0].key == "cv_in_file");
    Require(Read_Text_File(sidecars[0].path) == "CV_PAYLOAD\n");

    std::filesystem::remove_all(dir);
}

static void Test_Materialize_Protocol_Sidecar_Text_State_Rejects_Bad_Key()
{
    const auto dir = Unique_Temp_Dir("protocol_sidecar_bad_key");
    SpongeH5MD::RestartProtocolState state;
    SpongeH5MD::RestartProtocolSidecarTextState sidecar;
    sidecar.key = "coordinate_in_file";
    sidecar.text = "COORDINATE\n";
    state.sidecar_text_states.push_back(sidecar);

    std::vector<SpongeH5MD::LegacySidecarBinding> sidecars;
    std::string error;
    Require(!SpongeH5MD::Materialize_Protocol_Sidecar_Text_State(
        state, dir.string(), &sidecars, &error));
    Require(error.find("unsupported H5 protocol restart sidecar key") !=
            std::string::npos);

    std::filesystem::remove_all(dir);
}

int main()
{
    try
    {
        Test_Apply_Structural_State_With_Velocity();
        Test_Apply_Structural_State_Zero_Fills_Missing_Velocity();
        Test_Apply_Structural_State_Rejects_Atom_Count_Mismatch();
        Test_Apply_Nose_Hoover_Chain_Dynamic_State();
        Test_Apply_Nose_Hoover_Chain_Dynamic_State_Rejects_Length_Mismatch();
        Test_Extract_Sits_Nk_Protocol_State();
        Test_Extract_Sits_Nk_Protocol_State_Rejects_Bad_State();
        Test_Materialize_Metadynamics_Text_State();
        Test_Materialize_Metadynamics_Text_State_Rejects_Missing_Module();
        Test_Materialize_Protocol_Sidecar_Text_State();
        Test_Materialize_Protocol_Sidecar_Text_State_Rejects_Bad_Key();
    }
    catch (const std::exception&)
    {
        return 1;
    }
    return 0;
}

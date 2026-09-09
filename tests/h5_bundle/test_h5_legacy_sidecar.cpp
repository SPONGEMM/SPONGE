#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <highfive/highfive.hpp>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "utils/h5md/h5_legacy_sidecar.hpp"
#include "utils/h5md/topology_custom_force_h5_materializer.hpp"

struct TestFailure : std::runtime_error
{
    using std::runtime_error::runtime_error;
};

#define REQUIRE_TRUE(expr)                                    \
    do                                                        \
    {                                                         \
        if (!(expr))                                          \
        {                                                     \
            std::ostringstream require_message;               \
            require_message << __FILE__ << ":" << __LINE__    \
                            << " requirement failed: " #expr; \
            throw TestFailure(require_message.str());         \
        }                                                     \
    } while (false)

#define REQUIRE_EQ(lhs, rhs)                                          \
    do                                                                \
    {                                                                 \
        const auto require_lhs = (lhs);                               \
        const auto require_rhs = (rhs);                               \
        if (!(require_lhs == require_rhs))                            \
        {                                                             \
            std::ostringstream require_message;                       \
            require_message << __FILE__ << ":" << __LINE__            \
                            << " equality failed: " #lhs " == " #rhs; \
            throw TestFailure(require_message.str());                 \
        }                                                             \
    } while (false)

class FakeController
{
   public:
    bool Command_Exist(const char* key)
    {
        return key != nullptr && commands_.count(key) != 0;
    }

    const char* Command(const char* key)
    {
        const auto iter = commands_.find(key == nullptr ? "" : key);
        if (iter == commands_.end())
        {
            return "";
        }
        return iter->second.c_str();
    }

    void Set_Command(const char* key, const char* value, int check = 1)
    {
        commands_[key == nullptr ? "" : key] = value == nullptr ? "" : value;
        checks_[key == nullptr ? "" : key] = check;
    }

    int Check_Value(const std::string& key) const
    {
        const auto iter = checks_.find(key);
        return iter == checks_.end() ? -1 : iter->second;
    }

   private:
    std::map<std::string, std::string> commands_;
    std::map<std::string, int> checks_;
};

static std::filesystem::path Unique_Temp_Dir(const std::string& name)
{
    const auto stamp =
        std::chrono::high_resolution_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path() /
           ("sponge_h5_legacy_sidecar_" + std::to_string(stamp) + "_" + name);
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

static void Write_String_Vector(HighFive::File& file,
                                const std::string& dataset_path,
                                const std::vector<std::string>& values)
{
    Ensure_Parent_Group(file, dataset_path);
    auto dataset = file.createDataSet<std::string>(
        dataset_path, HighFive::DataSpace({values.size()}));
    dataset.write(values);
}

template <typename T>
static void Write_Vector(HighFive::File& file, const std::string& dataset_path,
                         const std::vector<T>& values)
{
    Ensure_Parent_Group(file, dataset_path);
    auto dataset =
        file.createDataSet<T>(dataset_path, HighFive::DataSpace::From(values));
    dataset.write(values);
}

template <typename T>
static void Write_Scalar(HighFive::File& file, const std::string& dataset_path,
                         const T value)
{
    Ensure_Parent_Group(file, dataset_path);
    auto dataset =
        file.createDataSet<T>(dataset_path, HighFive::DataSpace::From(value));
    dataset.write(value);
}

template <typename T>
static void Write_Matrix(HighFive::File& file, const std::string& dataset_path,
                         const std::vector<std::vector<T>>& values)
{
    Ensure_Parent_Group(file, dataset_path);
    auto dataset =
        file.createDataSet<T>(dataset_path, HighFive::DataSpace::From(values));
    dataset.write(values);
}

static std::string Read_Text(const std::filesystem::path& path)
{
    std::ifstream in(path);
    return std::string(std::istreambuf_iterator<char>(in),
                       std::istreambuf_iterator<char>());
}

static void Write_Sidecar_File(const std::filesystem::path& path,
                               const std::vector<std::string>& keys,
                               const std::vector<std::string>& sidecar_paths)
{
    HighFive::File file(path.string(), HighFive::File::Overwrite);
    Write_String_Vector(file, SpongeH5MD::path::legacy_sidecar_keys, keys);
    Write_String_Vector(file, SpongeH5MD::path::legacy_sidecar_paths,
                        sidecar_paths);
}

static void Test_Reads_And_Resolves_Relative_Sidecar_Paths()
{
    const auto dir = Unique_Temp_Dir("read");
    std::filesystem::create_directories(dir / "containers");
    const auto container = dir / "containers" / "topology.spgt.h5";

    Write_Sidecar_File(container, {"mass_in_file", "charge_in_file"},
                       {"sidecars/mass.txt", (dir / "charge.txt").string()});

    std::vector<SpongeH5MD::LegacySidecarBinding> sidecars;
    std::string error;
    REQUIRE_TRUE(SpongeH5MD::Read_Legacy_Sidecars_From_H5(container.string(),
                                                          &sidecars, &error));
    REQUIRE_EQ(sidecars.size(), static_cast<std::size_t>(2));
    REQUIRE_EQ(sidecars[0].key, std::string("mass_in_file"));
    REQUIRE_EQ(sidecars[0].path, (dir / "containers" / "sidecars" / "mass.txt")
                                     .lexically_normal()
                                     .string());
    REQUIRE_EQ(sidecars[1].path,
               (dir / "charge.txt").lexically_normal().string());

    std::filesystem::remove_all(dir);
}

static void Test_Injects_Allowed_Sidecar_Command()
{
    FakeController controller;
    std::vector<SpongeH5MD::LegacySidecarBinding> sidecars = {
        {"mass_in_file", "/tmp/mass.txt"},
    };
    std::string error;

    REQUIRE_TRUE(SpongeH5MD::Inject_Legacy_Sidecar_Commands(
        &controller, sidecars, SpongeH5MD::H5_Topology_Sidecar_Command_Keys(),
        "topology", &error));
    REQUIRE_TRUE(controller.Command_Exist("mass_in_file"));
    REQUIRE_EQ(std::string(controller.Command("mass_in_file")),
               std::string("/tmp/mass.txt"));
    REQUIRE_EQ(controller.Check_Value("mass_in_file"), 0);
}

static void Test_Rejects_Unsupported_Sidecar_Command()
{
    FakeController controller;
    std::vector<SpongeH5MD::LegacySidecarBinding> sidecars = {
        {"coordinate_in_file", "/tmp/coordinate.txt"},
    };
    std::string error;

    REQUIRE_TRUE(!SpongeH5MD::Inject_Legacy_Sidecar_Commands(
        &controller, sidecars, SpongeH5MD::H5_Topology_Sidecar_Command_Keys(),
        "topology", &error));
    REQUIRE_TRUE(error.find("unsupported H5 legacy sidecar key") !=
                 std::string::npos);
}

static void Test_Rejects_Conflicting_Sidecar_Command()
{
    FakeController controller;
    controller.Set_Command("mass_in_file", "/tmp/legacy_mass.txt", 1);
    std::vector<SpongeH5MD::LegacySidecarBinding> sidecars = {
        {"mass_in_file", "/tmp/h5_mass.txt"},
    };
    std::string error;

    REQUIRE_TRUE(!SpongeH5MD::Inject_Legacy_Sidecar_Commands(
        &controller, sidecars, SpongeH5MD::H5_Topology_Sidecar_Command_Keys(),
        "topology", &error));
    REQUIRE_TRUE(error.find("conflicts with existing command") !=
                 std::string::npos);
}

static void Test_Accepts_Relative_Existing_Path_For_Same_Sidecar()
{
    const auto dir = Unique_Temp_Dir("same_path");
    std::filesystem::create_directories(dir / "legacy_sidecars" /
                                        "charge_in_file");
    const auto sidecar_path =
        dir / "legacy_sidecars" / "charge_in_file" / "charge.txt";
    {
        std::ofstream out(sidecar_path);
        out << "1\n0 H\n";
    }

    const auto previous_cwd = std::filesystem::current_path();
    std::filesystem::current_path(dir);
    try
    {
        FakeController controller;
        controller.Set_Command("charge_in_file",
                               "legacy_sidecars/charge_in_file/charge.txt",
                               1);
        std::vector<SpongeH5MD::LegacySidecarBinding> sidecars = {
            {"charge_in_file", sidecar_path.string()},
        };
        std::string error;

        REQUIRE_TRUE(SpongeH5MD::Inject_Legacy_Sidecar_Commands(
            &controller, sidecars,
            SpongeH5MD::H5_Topology_Sidecar_Command_Keys(), "topology",
            &error));
        REQUIRE_EQ(std::string(controller.Command("charge_in_file")),
                   std::string("legacy_sidecars/charge_in_file/charge.txt"));
    }
    catch (...)
    {
        std::filesystem::current_path(previous_cwd);
        std::filesystem::remove_all(dir);
        throw;
    }
    std::filesystem::current_path(previous_cwd);
    std::filesystem::remove_all(dir);
}

static void Test_Materializes_RB_And_Bond_Soft_From_Typed_H5()
{
    const auto dir = Unique_Temp_Dir("custom_force");
    std::filesystem::create_directories(dir);
    const auto topology = dir / "topology.spgt.h5";
    {
        HighFive::File file(topology.string(), HighFive::File::Overwrite);
        const std::string listed = "/forcefield/custom_force/listed";
        Write_String_Vector(file, listed + "/name",
                            {"custom_bond", "Ryckaert_Bellemans"});
        Write_String_Vector(file, listed + "/potential",
                            {"E = k * (r_ab - r0) * (r_ab - r0);",
                             "E = c0 + c1 * cosf(phi_abcd - CONSTANT_Pi);"});
        Write_String_Vector(file, listed + "/connected_atoms", {"ab", ""});
        Write_String_Vector(file, listed + "/constrain_distance", {"r0", ""});
        Write_String_Vector(
            file, listed + "/parameters/type",
            {"int", "int", "float", "float", "int", "int", "int", "int",
             "float", "float", "float", "float", "float", "float"});
        Write_String_Vector(
            file, listed + "/parameters/name",
            {"atom_a", "atom_b", "k", "r0", "atom_a", "atom_b", "atom_c",
             "atom_d", "c0", "c1", "c2", "c3", "c4", "c5"});
        Write_Vector<std::int64_t>(file, listed + "/parameters/offset",
                                   {0, 4, 14});
        Write_Scalar<int>(file, listed + "/data/custom_bond/item_count", 1);
        Write_Matrix<float>(file, listed + "/data/custom_bond/parameter/value",
                            {{0.0f, 1.0f, 12.5f, 1.25f}});
        Write_Scalar<int>(file, listed + "/data/Ryckaert_Bellemans/item_count",
                          1);
        Write_Matrix<float>(
            file, listed + "/data/Ryckaert_Bellemans/parameter/value",
            {{0.0f, 1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f}});

        const std::string soft = "/forcefield/bond_soft";
        Write_Matrix<int>(file, soft + "/atoms", {{1, 2}});
        Write_Vector<float>(file, soft + "/k", {20.0f});
        Write_Vector<float>(file, soft + "/r0", {1.4f});
        Write_Vector<int>(file, soft + "/from_a_or_b", {1});
    }

    FakeController controller;
    controller.Set_Command("lambda_bond", "0.25", 1);
    controller.Set_Command("soft_bond_alpha", "0.5", 1);
    std::string error;
    const auto materialized = dir / "materialized";
    const bool materialized_ok =
        SpongeH5MD::Materialize_Native_Custom_Force_Text_Inputs_From_H5(
            &controller, topology.string(), materialized, &error);
    if (!materialized_ok)
    {
        throw TestFailure("custom force materialization failed: " + error);
    }

    for (const std::string key :
         {"listed_forces_in_file", "custom_bond_in_file",
          "Ryckaert_Bellemans_in_file", "bond_soft_in_file"})
    {
        REQUIRE_TRUE(controller.Command_Exist(key.c_str()));
        REQUIRE_TRUE(
            std::filesystem::is_regular_file(controller.Command(key.c_str())));
    }
    const std::string descriptor =
        Read_Text(controller.Command("listed_forces_in_file"));
    REQUIRE_TRUE(descriptor.find("[[[ Ryckaert_Bellemans ]]]") !=
                 std::string::npos);
    REQUIRE_TRUE(descriptor.find("int atom_d, float c0") != std::string::npos);
    REQUIRE_TRUE(descriptor.find("[[[ bond_soft ]]]") != std::string::npos);
    std::istringstream soft_data(
        Read_Text(controller.Command("bond_soft_in_file")));
    int soft_count = 0;
    int atom_a = -1;
    int atom_b = -1;
    int from_a_or_b = -1;
    float k = 0.0f;
    float r0 = 0.0f;
    float lambda = 0.0f;
    float alpha = 0.0f;
    soft_data >> soft_count >> atom_a >> atom_b >> k >> r0 >> from_a_or_b >>
        lambda >> alpha;
    REQUIRE_EQ(soft_count, 1);
    REQUIRE_EQ(atom_a, 1);
    REQUIRE_EQ(atom_b, 2);
    REQUIRE_EQ(from_a_or_b, 1);
    REQUIRE_TRUE(std::fabs(k - 20.0f) < 1.0e-6f);
    REQUIRE_TRUE(std::fabs(r0 - 1.4f) < 1.0e-6f);
    REQUIRE_TRUE(std::fabs(lambda - 0.25f) < 1.0e-6f);
    REQUIRE_TRUE(std::fabs(alpha - 0.5f) < 1.0e-6f);
    std::istringstream rb_data(
        Read_Text(controller.Command("Ryckaert_Bellemans_in_file")));
    int rb_count = 0;
    std::vector<float> rb_values(10, 0.0f);
    rb_data >> rb_count;
    for (float& value : rb_values) rb_data >> value;
    REQUIRE_EQ(rb_count, 1);
    const std::vector<float> expected_rb = {0.0f, 1.0f, 2.0f, 3.0f, 0.1f,
                                            0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
    for (std::size_t i = 0; i < expected_rb.size(); ++i)
    {
        REQUIRE_TRUE(std::fabs(rb_values[i] - expected_rb[i]) < 1.0e-6f);
    }

    std::filesystem::remove_all(dir);
}

int main()
{
    try
    {
        Test_Reads_And_Resolves_Relative_Sidecar_Paths();
        Test_Injects_Allowed_Sidecar_Command();
        Test_Rejects_Unsupported_Sidecar_Command();
        Test_Rejects_Conflicting_Sidecar_Command();
        Test_Accepts_Relative_Existing_Path_For_Same_Sidecar();
        Test_Materializes_RB_And_Bond_Soft_From_Typed_H5();
    }
    catch (const std::exception& err)
    {
        std::cerr << err.what() << std::endl;
        return 1;
    }
    return 0;
}

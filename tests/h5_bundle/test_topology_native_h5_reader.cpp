#include <cmath>
#include <filesystem>
#include <highfive/highfive.hpp>
#include <limits>
#include <string>
#include <vector>

#include "h5_bundle_test_common.hpp"
#include "utils/h5md/topology_native_h5_reader.hpp"

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

static void Write_Int_Matrix(HighFive::File& file, const std::string& path,
                             const std::vector<int>& values, std::size_t rows,
                             std::size_t columns)
{
    REQUIRE_EQ(values.size(), rows * columns);
    std::vector<std::vector<int>> matrix(rows, std::vector<int>(columns));
    for (std::size_t row = 0; row < rows; ++row)
    {
        for (std::size_t column = 0; column < columns; ++column)
        {
            matrix[row][column] = values[row * columns + column];
        }
    }
    Ensure_Parent_Group(file, path);
    auto dataset =
        file.createDataSet<int>(path, HighFive::DataSpace::From(matrix));
    dataset.write(matrix);
}

static void Write_Float_Matrix(HighFive::File& file, const std::string& path,
                               const std::vector<float>& values,
                               std::size_t rows, std::size_t columns)
{
    REQUIRE_EQ(values.size(), rows * columns);
    std::vector<std::vector<float>> matrix(rows, std::vector<float>(columns));
    for (std::size_t row = 0; row < rows; ++row)
    {
        for (std::size_t column = 0; column < columns; ++column)
        {
            matrix[row][column] = values[row * columns + column];
        }
    }
    Ensure_Parent_Group(file, path);
    auto dataset =
        file.createDataSet<float>(path, HighFive::DataSpace::From(matrix));
    dataset.write(matrix);
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

static void Require_Int_Vector_Eq(const std::vector<int>& actual,
                                  const std::vector<int>& expected)
{
    REQUIRE_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i)
    {
        REQUIRE_EQ(actual[i], expected[i]);
    }
}

static void Require_Reader(bool ok, const TopologyNativeH5Reader& reader,
                           const char* operation)
{
    if (!ok)
    {
        throw TestFailure(std::string(operation) + ": " + reader.Last_Error());
    }
}

static void Test_Reads_Native_Mass_And_Charge()
{
    const auto dir = Unique_Temp_Path("topology_native_reader");
    std::filesystem::create_directories(dir);
    const auto path = dir / "topology.spgt.h5";

    {
        HighFive::File file(path.string(), HighFive::File::Overwrite);
        Write_Scalar(file, "/schema/version", std::string("1"));
        Write_Scalar(file, "/topology/atom_count",
                     static_cast<std::int64_t>(3));
        Write_Float_Vector(file, "/atoms/mass", {12.0f, 1.0f, 16.0f});
        Write_Float_Vector(file, "/atoms/charge", {-0.2f, 0.1f, 0.1f});
        Write_Int64_Vector(file, "/topology/exclusions/offset", {0, 2, 3, 3});
        Write_Int_Vector(file, "/topology/exclusions/list", {1, 2, 0});
        Write_Int_Vector(file, "/forcefield/lj/type", {0, 1, 1});
        Write_Scalar(file, "/forcefield/lj/atom_type_count", 2);
        Write_Float_Matrix(file, "/forcefield/lj/params",
                           {12.0f, 6.0f, 24.0f, 12.0f, 36.0f, 18.0f}, 3, 2);
        Write_Int_Matrix(file, "/forcefield/bond/atoms", {0, 1, 1, 2}, 2, 2);
        Write_Float_Vector(file, "/forcefield/bond/k", {100.0f, 200.0f});
        Write_Float_Vector(file, "/forcefield/bond/r0", {1.0f, 2.0f});
        Write_Int_Matrix(file, "/forcefield/angle/atoms", {0, 1, 2}, 1, 3);
        Write_Float_Vector(file, "/forcefield/angle/k", {50.0f});
        Write_Float_Vector(file, "/forcefield/angle/theta0", {1.57f});
        Write_Int_Matrix(file, "/forcefield/dihedral/atoms", {0, 1, 2, 0}, 1,
                         4);
        Write_Float_Vector(file, "/forcefield/dihedral/pk", {0.25f});
        Write_Float_Vector(file, "/forcefield/dihedral/pn", {3.0f});
        Write_Int_Vector(file, "/forcefield/dihedral/ipn", {3});
        Write_Float_Vector(file, "/forcefield/dihedral/gamc", {0.10f});
        Write_Float_Vector(file, "/forcefield/dihedral/gams", {0.20f});
        Write_Int_Matrix(file, "/forcefield/improper/atoms", {0, 2, 1, 0}, 1,
                         4);
        Write_Float_Vector(file, "/forcefield/improper/pk", {0.75f});
        Write_Float_Vector(file, "/forcefield/improper/phi0", {1.25f});
        Write_Int_Matrix(file, "/forcefield/nb14/atoms", {0, 2}, 1, 2);
        Write_Float_Matrix(file, "/forcefield/nb14/params",
                           {120.0f, 60.0f, 0.833333f}, 1, 3);
        Write_Float_Matrix(file, "/forcefield/gb/params",
                           {1.20f, 0.80f, 1.30f, 0.85f, 1.40f, 0.90f}, 3, 2);
        Write_Int_Vector(file, "/forcefield/virtual_atom/type", {0, 2});
        Write_Int_Vector(file, "/forcefield/virtual_atom/atom", {2, 1});
        Write_Int64_Vector(file, "/forcefield/virtual_atom/from_offset",
                           {0, 1, 4});
        Write_Int_Vector(file, "/forcefield/virtual_atom/from", {0, 0, 1, 2});
        Write_Int64_Vector(file, "/forcefield/virtual_atom/parameter_offset",
                           {0, 1, 3});
        Write_Float_Vector(file, "/forcefield/virtual_atom/parameter",
                           {0.5f, 0.25f, 0.75f});
        Write_Int_Matrix(file, "/forcefield/urey_bradley/atoms", {0, 1, 2}, 1,
                         3);
        Write_Float_Vector(file, "/forcefield/urey_bradley/angle_k", {11.0f});
        Write_Float_Vector(file, "/forcefield/urey_bradley/angle_theta0",
                           {1.23f});
        Write_Float_Vector(file, "/forcefield/urey_bradley/bond_k", {22.0f});
        Write_Float_Vector(file, "/forcefield/urey_bradley/bond_r0", {2.34f});
        Write_Int_Matrix(file, "/forcefield/cmap/atoms", {0, 1, 2, 0, 1}, 1, 5);
        Write_Int_Vector(file, "/forcefield/cmap/type", {0});
        Write_Int_Vector(file, "/forcefield/cmap/resolution", {2});
        Write_Float_Vector(file, "/forcefield/cmap/grid_value",
                           {0.1f, 0.2f, 0.3f, 0.4f});
        Write_Int_Vector(file, "/forcefield/lj_soft_core/atom_type_A",
                         {0, 1, 1});
        Write_Int_Vector(file, "/forcefield/lj_soft_core/atom_type_B",
                         {0, 0, 1});
        Write_Scalar(file, "/forcefield/lj_soft_core/atom_type_count_A", 2);
        Write_Scalar(file, "/forcefield/lj_soft_core/atom_type_count_B", 2);
        Write_Float_Vector(file, "/forcefield/lj_soft_core/pair_AA",
                           {1.0f, 2.0f, 3.0f});
        Write_Float_Vector(file, "/forcefield/lj_soft_core/pair_AB",
                           {4.0f, 5.0f, 6.0f});
        Write_Float_Vector(file, "/forcefield/lj_soft_core/pair_BA",
                           {7.0f, 8.0f, 9.0f});
        Write_Float_Vector(file, "/forcefield/lj_soft_core/pair_BB",
                           {10.0f, 11.0f, 12.0f});
        Write_Int_Vector(file, "/forcefield/subsys_division", {0, 1, 1});
    }

    TopologyNativeH5Reader reader;
    Require_Reader(reader.Open(path.string()), reader, "open topology");
    NativeTopologyCoreState state;
    Require_Reader(reader.Read_Core_State(&state), reader,
                   "read native core state");

    REQUIRE_EQ(state.atom_count, 3);
    REQUIRE_TRUE(state.has_mass);
    REQUIRE_TRUE(state.has_charge);
    REQUIRE_TRUE(state.has_exclusions);
    REQUIRE_TRUE(state.has_lj);
    REQUIRE_TRUE(state.has_bonds);
    REQUIRE_TRUE(state.has_angles);
    REQUIRE_TRUE(state.has_dihedrals);
    REQUIRE_TRUE(state.has_impropers);
    REQUIRE_TRUE(state.has_nb14);
    REQUIRE_TRUE(state.has_gb);
    REQUIRE_TRUE(state.has_virtual_atoms);
    REQUIRE_TRUE(state.has_urey_bradley);
    REQUIRE_TRUE(state.has_cmap);
    REQUIRE_TRUE(state.has_lj_soft_core);
    Require_Float_Vector_Close(state.mass, {12.0f, 1.0f, 16.0f});
    Require_Float_Vector_Close(state.charge, {-0.2f, 0.1f, 0.1f});
    REQUIRE_EQ(state.exclusions.excluded_atoms.size(), 3);
    Require_Int_Vector_Eq(state.exclusions.excluded_atoms[0], {1, 2});
    Require_Int_Vector_Eq(state.exclusions.excluded_atoms[1], {0});
    Require_Int_Vector_Eq(state.exclusions.excluded_atoms[2], {});
    REQUIRE_EQ(state.lj.atom_type_numbers, 2);
    Require_Int_Vector_Eq(state.lj.atom_type, {0, 1, 1});
    Require_Float_Vector_Close(state.lj.pair_A, {12.0f, 24.0f, 36.0f});
    Require_Float_Vector_Close(state.lj.pair_B, {6.0f, 12.0f, 18.0f});
    Require_Int_Vector_Eq(state.bonds.atom_a, {0, 1});
    Require_Int_Vector_Eq(state.bonds.atom_b, {1, 2});
    Require_Float_Vector_Close(state.bonds.k, {100.0f, 200.0f});
    Require_Float_Vector_Close(state.bonds.r0, {1.0f, 2.0f});
    Require_Int_Vector_Eq(state.angles.atom_a, {0});
    Require_Int_Vector_Eq(state.angles.atom_b, {1});
    Require_Int_Vector_Eq(state.angles.atom_c, {2});
    Require_Float_Vector_Close(state.angles.k, {50.0f});
    Require_Float_Vector_Close(state.angles.theta0, {1.57f});
    Require_Int_Vector_Eq(state.dihedrals.atom_a, {0});
    Require_Int_Vector_Eq(state.dihedrals.atom_b, {1});
    Require_Int_Vector_Eq(state.dihedrals.atom_c, {2});
    Require_Int_Vector_Eq(state.dihedrals.atom_d, {0});
    Require_Float_Vector_Close(state.dihedrals.pk, {0.25f});
    Require_Float_Vector_Close(state.dihedrals.pn, {3.0f});
    Require_Int_Vector_Eq(state.dihedrals.ipn, {3});
    Require_Float_Vector_Close(state.dihedrals.gamc, {0.10f});
    Require_Float_Vector_Close(state.dihedrals.gams, {0.20f});
    Require_Int_Vector_Eq(state.impropers.atom_a, {0});
    Require_Int_Vector_Eq(state.impropers.atom_b, {2});
    Require_Int_Vector_Eq(state.impropers.atom_c, {1});
    Require_Int_Vector_Eq(state.impropers.atom_d, {0});
    Require_Float_Vector_Close(state.impropers.pk, {0.75f});
    Require_Float_Vector_Close(state.impropers.gamc, {1.25f});
    Require_Float_Vector_Close(state.impropers.gams, {0.0f});
    Require_Int_Vector_Eq(state.nb14.atom_a, {0});
    Require_Int_Vector_Eq(state.nb14.atom_b, {2});
    Require_Float_Vector_Close(state.nb14.A, {120.0f});
    Require_Float_Vector_Close(state.nb14.B, {60.0f});
    Require_Float_Vector_Close(state.nb14.cf_scale_factor, {0.833333f});
    Require_Float_Vector_Close(state.gb.radius, {1.20f, 1.30f, 1.40f});
    Require_Float_Vector_Close(state.gb.scale_factor, {0.80f, 0.85f, 0.90f});
    REQUIRE_EQ(state.virtual_atoms.records.size(), 2);
    REQUIRE_EQ(state.virtual_atoms.records[0].type, 0);
    REQUIRE_EQ(state.virtual_atoms.records[0].virtual_atom, 2);
    Require_Int_Vector_Eq(state.virtual_atoms.records[0].from, {0});
    Require_Float_Vector_Close(state.virtual_atoms.records[0].parameter,
                               {0.5f});
    REQUIRE_EQ(state.virtual_atoms.records[1].type, 2);
    REQUIRE_EQ(state.virtual_atoms.records[1].virtual_atom, 1);
    Require_Int_Vector_Eq(state.virtual_atoms.records[1].from, {0, 1, 2});
    Require_Float_Vector_Close(state.virtual_atoms.records[1].parameter,
                               {0.25f, 0.75f});
    Require_Int_Vector_Eq(state.urey_bradley.atom_a, {0});
    Require_Int_Vector_Eq(state.urey_bradley.atom_b, {1});
    Require_Int_Vector_Eq(state.urey_bradley.atom_c, {2});
    Require_Float_Vector_Close(state.urey_bradley.angle_k, {11.0f});
    Require_Float_Vector_Close(state.urey_bradley.angle_theta0, {1.23f});
    Require_Float_Vector_Close(state.urey_bradley.bond_k, {22.0f});
    Require_Float_Vector_Close(state.urey_bradley.bond_r0, {2.34f});
    Require_Int_Vector_Eq(state.cmap.atom_a, {0});
    Require_Int_Vector_Eq(state.cmap.atom_b, {1});
    Require_Int_Vector_Eq(state.cmap.atom_c, {2});
    Require_Int_Vector_Eq(state.cmap.atom_d, {0});
    Require_Int_Vector_Eq(state.cmap.atom_e, {1});
    Require_Int_Vector_Eq(state.cmap.cmap_type, {0});
    Require_Int_Vector_Eq(state.cmap.resolution, {2});
    REQUIRE_EQ(state.cmap.unique_type_numbers, 1);
    REQUIRE_EQ(state.cmap.unique_gridpoint_numbers, 4);
    Require_Int_Vector_Eq(state.cmap.type_offset, {0});
    Require_Float_Vector_Close(state.cmap.grid_value, {0.1f, 0.2f, 0.3f, 0.4f});
    REQUIRE_EQ(state.lj_soft_core.atom_numbers, 3);
    REQUIRE_EQ(state.lj_soft_core.atom_type_numbers_A, 2);
    REQUIRE_EQ(state.lj_soft_core.atom_type_numbers_B, 2);
    Require_Int_Vector_Eq(state.lj_soft_core.atom_LJ_type_A, {0, 1, 1});
    Require_Int_Vector_Eq(state.lj_soft_core.atom_LJ_type_B, {0, 0, 1});
    Require_Float_Vector_Close(state.lj_soft_core.LJ_AA, {1.0f, 2.0f, 3.0f});
    Require_Float_Vector_Close(state.lj_soft_core.LJ_AB, {4.0f, 5.0f, 6.0f});
    Require_Float_Vector_Close(state.lj_soft_core.LJ_BA, {7.0f, 8.0f, 9.0f});
    Require_Float_Vector_Close(state.lj_soft_core.LJ_BB, {10.0f, 11.0f, 12.0f});
    Require_Int_Vector_Eq(state.lj_soft_core.subsystem_division, {0, 1, 1});
}

static void Test_Rejects_Length_Mismatch()
{
    const auto dir = Unique_Temp_Path("topology_native_reader_mismatch");
    std::filesystem::create_directories(dir);
    const auto path = dir / "topology.spgt.h5";

    {
        HighFive::File file(path.string(), HighFive::File::Overwrite);
        Write_Scalar(file, "/topology/atom_count",
                     static_cast<std::int64_t>(2));
        Write_Float_Vector(file, "/atoms/mass", {12.0f, 1.0f, 16.0f});
    }

    TopologyNativeH5Reader reader;
    Require_Reader(reader.Open(path.string()), reader, "open mismatch");
    NativeTopologyCoreState state;
    REQUIRE_TRUE(!reader.Read_Core_State(&state));
    REQUIRE_TRUE(reader.Last_Error().find("does not match") !=
                 std::string::npos);
}

static void Test_Rejects_Non_Positive_Mass()
{
    const auto dir = Unique_Temp_Path("topology_native_reader_bad_mass");
    std::filesystem::create_directories(dir);
    const auto path = dir / "topology.spgt.h5";

    {
        HighFive::File file(path.string(), HighFive::File::Overwrite);
        Write_Scalar(file, "/topology/atom_count",
                     static_cast<std::int64_t>(2));
        Write_Float_Vector(file, "/atoms/mass", {12.0f, 0.0f});
    }

    TopologyNativeH5Reader reader;
    Require_Reader(reader.Open(path.string()), reader, "open bad mass");
    NativeTopologyCoreState state;
    REQUIRE_TRUE(!reader.Read_Core_State(&state));
    REQUIRE_TRUE(reader.Last_Error().find("non-positive") != std::string::npos);
}

static void Test_Derives_Atom_Count_From_Charge_When_Metadata_Is_Absent()
{
    const auto dir = Unique_Temp_Path("topology_native_reader_charge_only");
    std::filesystem::create_directories(dir);
    const auto path = dir / "topology.spgt.h5";

    {
        HighFive::File file(path.string(), HighFive::File::Overwrite);
        Write_Float_Vector(file, "/atoms/charge", {-0.5f, 0.5f});
    }

    TopologyNativeH5Reader reader;
    Require_Reader(reader.Open(path.string()), reader, "open charge only");
    NativeTopologyCoreState state;
    Require_Reader(reader.Read_Core_State(&state), reader, "read charge only");

    REQUIRE_EQ(state.atom_count, 2);
    REQUIRE_TRUE(!state.has_mass);
    REQUIRE_TRUE(state.has_charge);
    Require_Float_Vector_Close(state.charge, {-0.5f, 0.5f});
}

static void Test_Rejects_Invalid_Exclusion_Offset()
{
    const auto dir = Unique_Temp_Path("topology_native_reader_bad_exclusion");
    std::filesystem::create_directories(dir);
    const auto path = dir / "topology.spgt.h5";

    {
        HighFive::File file(path.string(), HighFive::File::Overwrite);
        Write_Scalar(file, "/topology/atom_count",
                     static_cast<std::int64_t>(2));
        Write_Int64_Vector(file, "/topology/exclusions/offset", {0, 2, 1});
        Write_Int_Vector(file, "/topology/exclusions/list", {1});
    }

    TopologyNativeH5Reader reader;
    Require_Reader(reader.Open(path.string()), reader, "open bad exclusion");
    NativeTopologyCoreState state;
    REQUIRE_TRUE(!reader.Read_Core_State(&state));
    REQUIRE_TRUE(reader.Last_Error().find("final value") != std::string::npos ||
                 reader.Last_Error().find("monotonic") != std::string::npos);
}

static void Test_Rejects_LJ_Param_Count_Mismatch()
{
    const auto dir = Unique_Temp_Path("topology_native_reader_bad_lj");
    std::filesystem::create_directories(dir);
    const auto path = dir / "topology.spgt.h5";

    {
        HighFive::File file(path.string(), HighFive::File::Overwrite);
        Write_Scalar(file, "/topology/atom_count",
                     static_cast<std::int64_t>(2));
        Write_Int_Vector(file, "/forcefield/lj/type", {0, 1});
        Write_Scalar(file, "/forcefield/lj/atom_type_count", 2);
        Write_Float_Matrix(file, "/forcefield/lj/params",
                           {12.0f, 6.0f, 24.0f, 12.0f}, 2, 2);
    }

    TopologyNativeH5Reader reader;
    Require_Reader(reader.Open(path.string()), reader, "open bad lj");
    NativeTopologyCoreState state;
    REQUIRE_TRUE(!reader.Read_Core_State(&state));
    REQUIRE_TRUE(reader.Last_Error().find("row count") != std::string::npos);
}

static void Test_Reads_Split_LJ_Pair_Arrays()
{
    const auto dir = Unique_Temp_Path("topology_native_reader_split_lj");
    std::filesystem::create_directories(dir);
    const auto path = dir / "topology.spgt.h5";

    {
        HighFive::File file(path.string(), HighFive::File::Overwrite);
        Write_Scalar(file, "/topology/atom_count",
                     static_cast<std::int64_t>(3));
        Write_Int_Vector(file, "/forcefield/lj/type", {0, 1, 1});
        Write_Scalar(file, "/forcefield/lj/atom_type_count", 2);
        Write_Float_Vector(file, "/forcefield/lj/pair_A_12",
                           {12.0f, 24.0f, 36.0f});
        Write_Float_Vector(file, "/forcefield/lj/pair_B_6",
                           {6.0f, 12.0f, 18.0f});
    }

    TopologyNativeH5Reader reader;
    Require_Reader(reader.Open(path.string()), reader, "open split lj");
    NativeTopologyCoreState state;
    Require_Reader(reader.Read_Core_State(&state), reader, "read split lj");
    REQUIRE_TRUE(state.has_lj);
    REQUIRE_EQ(state.lj.atom_type_numbers, 2);
    Require_Int_Vector_Eq(state.lj.atom_type, {0, 1, 1});
    Require_Float_Vector_Close(state.lj.pair_A, {12.0f, 24.0f, 36.0f});
    Require_Float_Vector_Close(state.lj.pair_B, {6.0f, 12.0f, 18.0f});
}

static void Test_Rejects_Dihedral_Runtime_Length_Mismatch()
{
    const auto dir = Unique_Temp_Path("topology_native_reader_bad_dihedral");
    std::filesystem::create_directories(dir);
    const auto path = dir / "topology.spgt.h5";

    {
        HighFive::File file(path.string(), HighFive::File::Overwrite);
        Write_Scalar(file, "/topology/atom_count",
                     static_cast<std::int64_t>(4));
        Write_Int_Matrix(file, "/forcefield/dihedral/atoms", {0, 1, 2, 3}, 1,
                         4);
        Write_Float_Vector(file, "/forcefield/dihedral/pk", {0.25f, 0.50f});
        Write_Float_Vector(file, "/forcefield/dihedral/pn", {3.0f});
        Write_Int_Vector(file, "/forcefield/dihedral/ipn", {3});
        Write_Float_Vector(file, "/forcefield/dihedral/gamc", {0.10f});
        Write_Float_Vector(file, "/forcefield/dihedral/gams", {0.20f});
    }

    TopologyNativeH5Reader reader;
    Require_Reader(reader.Open(path.string()), reader, "open bad dihedral");
    NativeTopologyCoreState state;
    REQUIRE_TRUE(!reader.Read_Core_State(&state));
    REQUIRE_TRUE(reader.Last_Error().find("runtime parameter lengths") !=
                 std::string::npos);
}

static void Test_Reads_Legacy_Dihedral_Parameter_Names()
{
    const auto dir = Unique_Temp_Path("topology_native_reader_legacy_dihedral");
    std::filesystem::create_directories(dir);
    const auto path = dir / "topology.spgt.h5";

    {
        HighFive::File file(path.string(), HighFive::File::Overwrite);
        Write_Scalar(file, "/topology/atom_count",
                     static_cast<std::int64_t>(3));
        Write_Int_Matrix(file, "/forcefield/dihedral/atoms", {0, 1, 2, 0}, 1,
                         4);
        Write_Float_Vector(file, "/forcefield/dihedral/k", {2.0f});
        Write_Int_Vector(file, "/forcefield/dihedral/periodicity", {3});
        Write_Float_Vector(file, "/forcefield/dihedral/phi0", {3.14f});
    }

    TopologyNativeH5Reader reader;
    Require_Reader(reader.Open(path.string()), reader, "open legacy dihedral");
    NativeTopologyCoreState state;
    Require_Reader(reader.Read_Core_State(&state), reader,
                   "read legacy dihedral");
    REQUIRE_TRUE(state.has_dihedrals);
    Require_Int_Vector_Eq(state.dihedrals.atom_a, {0});
    Require_Int_Vector_Eq(state.dihedrals.atom_b, {1});
    Require_Int_Vector_Eq(state.dihedrals.atom_c, {2});
    Require_Int_Vector_Eq(state.dihedrals.atom_d, {0});
    Require_Float_Vector_Close(state.dihedrals.pk, {2.0f});
    Require_Float_Vector_Close(state.dihedrals.pn, {3.0f});
    Require_Int_Vector_Eq(state.dihedrals.ipn, {3});
    Require_Float_Vector_Close(state.dihedrals.gamc, {std::cos(3.14f) * 2.0f});
    Require_Float_Vector_Close(state.dihedrals.gams, {std::sin(3.14f) * 2.0f});
}

static void Test_Reads_Legacy_Scale_NB14_Params()
{
    const auto dir = Unique_Temp_Path("topology_native_reader_nb14_scale");
    std::filesystem::create_directories(dir);
    const auto path = dir / "topology.spgt.h5";

    {
        HighFive::File file(path.string(), HighFive::File::Overwrite);
        Write_Scalar(file, "/topology/atom_count",
                     static_cast<std::int64_t>(3));
        Write_Int_Vector(file, "/forcefield/lj/type", {0, 1, 1});
        Write_Scalar(file, "/forcefield/lj/atom_type_count", 2);
        Write_Float_Matrix(file, "/forcefield/lj/params",
                           {12.0f, 6.0f, 24.0f, 12.0f, 36.0f, 18.0f}, 3, 2);
        Write_Int_Matrix(file, "/forcefield/nb14/atoms", {0, 2}, 1, 2);
        Write_Float_Matrix(file, "/forcefield/nb14/params", {0.5f, 0.833333f},
                           1, 2);
    }

    TopologyNativeH5Reader reader;
    Require_Reader(reader.Open(path.string()), reader, "open nb14 scale");
    NativeTopologyCoreState state;
    Require_Reader(reader.Read_Core_State(&state), reader, "read nb14 scale");
    REQUIRE_TRUE(state.has_nb14);
    Require_Int_Vector_Eq(state.nb14.atom_a, {0});
    Require_Int_Vector_Eq(state.nb14.atom_b, {2});
    Require_Float_Vector_Close(state.nb14.A, {144.0f});
    Require_Float_Vector_Close(state.nb14.B, {36.0f});
    Require_Float_Vector_Close(state.nb14.cf_scale_factor, {0.833333f});
}

static void Test_Rejects_Legacy_Scale_NB14_Without_LJ()
{
    const auto dir = Unique_Temp_Path("topology_native_reader_nb14_no_lj");
    std::filesystem::create_directories(dir);
    const auto path = dir / "topology.spgt.h5";

    {
        HighFive::File file(path.string(), HighFive::File::Overwrite);
        Write_Scalar(file, "/topology/atom_count",
                     static_cast<std::int64_t>(3));
        Write_Int_Matrix(file, "/forcefield/nb14/atoms", {0, 2}, 1, 2);
        Write_Float_Matrix(file, "/forcefield/nb14/params", {0.5f, 0.833333f},
                           1, 2);
    }

    TopologyNativeH5Reader reader;
    Require_Reader(reader.Open(path.string()), reader, "open nb14 no lj");
    NativeTopologyCoreState state;
    REQUIRE_TRUE(!reader.Read_Core_State(&state));
    REQUIRE_TRUE(reader.Last_Error().find("require initialized native LJ") !=
                 std::string::npos);
}

static void Test_Rejects_NB14_Param_Count_Mismatch()
{
    const auto dir = Unique_Temp_Path("topology_native_reader_bad_nb14");
    std::filesystem::create_directories(dir);
    const auto path = dir / "topology.spgt.h5";

    {
        HighFive::File file(path.string(), HighFive::File::Overwrite);
        Write_Scalar(file, "/topology/atom_count",
                     static_cast<std::int64_t>(3));
        Write_Int_Matrix(file, "/forcefield/nb14/atoms", {0, 2}, 1, 2);
        Write_Float_Matrix(file, "/forcefield/nb14/params",
                           {120.0f, 60.0f, 0.833333f, 240.0f, 120.0f, 0.5f}, 2,
                           3);
    }

    TopologyNativeH5Reader reader;
    Require_Reader(reader.Open(path.string()), reader, "open bad nb14");
    NativeTopologyCoreState state;
    REQUIRE_TRUE(!reader.Read_Core_State(&state));
    REQUIRE_TRUE(reader.Last_Error().find("nb14 params row count") !=
                 std::string::npos);
}

static void Test_Rejects_Virtual_Atom_Arity_Mismatch()
{
    const auto dir = Unique_Temp_Path("topology_native_reader_bad_vatom");
    std::filesystem::create_directories(dir);
    const auto path = dir / "topology.spgt.h5";

    {
        HighFive::File file(path.string(), HighFive::File::Overwrite);
        Write_Scalar(file, "/topology/atom_count",
                     static_cast<std::int64_t>(3));
        Write_Int_Vector(file, "/forcefield/virtual_atom/type", {2});
        Write_Int_Vector(file, "/forcefield/virtual_atom/atom", {2});
        Write_Int64_Vector(file, "/forcefield/virtual_atom/from_offset",
                           {0, 2});
        Write_Int_Vector(file, "/forcefield/virtual_atom/from", {0, 1});
        Write_Int64_Vector(file, "/forcefield/virtual_atom/parameter_offset",
                           {0, 1});
        Write_Float_Vector(file, "/forcefield/virtual_atom/parameter", {0.5f});
    }

    TopologyNativeH5Reader reader;
    Require_Reader(reader.Open(path.string()), reader, "open bad vatom");
    NativeTopologyCoreState state;
    REQUIRE_TRUE(!reader.Read_Core_State(&state));
    REQUIRE_TRUE(reader.Last_Error().find("arity") != std::string::npos);
}

static void Test_Rejects_Non_Finite_Virtual_Atom_Parameter()
{
    const auto dir = Unique_Temp_Path("topology_native_reader_nonfinite_vatom");
    std::filesystem::create_directories(dir);
    const auto path = dir / "topology.spgt.h5";

    {
        HighFive::File file(path.string(), HighFive::File::Overwrite);
        Write_Scalar(file, "/topology/atom_count",
                     static_cast<std::int64_t>(3));
        Write_Int_Vector(file, "/forcefield/virtual_atom/type", {1});
        Write_Int_Vector(file, "/forcefield/virtual_atom/atom", {2});
        Write_Int64_Vector(file, "/forcefield/virtual_atom/from_offset",
                           {0, 2});
        Write_Int_Vector(file, "/forcefield/virtual_atom/from", {0, 1});
        Write_Int64_Vector(file, "/forcefield/virtual_atom/parameter_offset",
                           {0, 1});
        Write_Float_Vector(file, "/forcefield/virtual_atom/parameter",
                           {std::numeric_limits<float>::infinity()});
    }

    TopologyNativeH5Reader reader;
    Require_Reader(reader.Open(path.string()), reader, "open non-finite vatom");
    NativeTopologyCoreState state;
    REQUIRE_TRUE(!reader.Read_Core_State(&state));
    REQUIRE_TRUE(reader.Last_Error().find("non-finite") != std::string::npos);
}

static void Test_Rejects_GB_Atom_Count_Mismatch()
{
    const auto dir = Unique_Temp_Path("topology_native_reader_bad_gb");
    std::filesystem::create_directories(dir);
    const auto path = dir / "topology.spgt.h5";

    {
        HighFive::File file(path.string(), HighFive::File::Overwrite);
        Write_Scalar(file, "/topology/atom_count",
                     static_cast<std::int64_t>(3));
        Write_Float_Matrix(file, "/forcefield/gb/params",
                           {1.20f, 0.80f, 1.30f, 0.85f}, 2, 2);
    }

    TopologyNativeH5Reader reader;
    Require_Reader(reader.Open(path.string()), reader, "open bad gb");
    NativeTopologyCoreState state;
    REQUIRE_TRUE(!reader.Read_Core_State(&state));
    REQUIRE_TRUE(reader.Last_Error().find("/forcefield/gb/params") !=
                 std::string::npos);
}

static void Test_Rejects_Urey_Bradley_Length_Mismatch()
{
    const auto dir = Unique_Temp_Path("topology_native_reader_bad_ub");
    std::filesystem::create_directories(dir);
    const auto path = dir / "topology.spgt.h5";

    {
        HighFive::File file(path.string(), HighFive::File::Overwrite);
        Write_Scalar(file, "/topology/atom_count",
                     static_cast<std::int64_t>(3));
        Write_Int_Matrix(file, "/forcefield/urey_bradley/atoms", {0, 1, 2}, 1,
                         3);
        Write_Float_Vector(file, "/forcefield/urey_bradley/angle_k",
                           {11.0f, 12.0f});
        Write_Float_Vector(file, "/forcefield/urey_bradley/angle_theta0",
                           {1.23f});
        Write_Float_Vector(file, "/forcefield/urey_bradley/bond_k", {22.0f});
        Write_Float_Vector(file, "/forcefield/urey_bradley/bond_r0", {2.34f});
    }

    TopologyNativeH5Reader reader;
    Require_Reader(reader.Open(path.string()), reader, "open bad ub");
    NativeTopologyCoreState state;
    REQUIRE_TRUE(!reader.Read_Core_State(&state));
    REQUIRE_TRUE(reader.Last_Error().find("urey-bradley parameter lengths") !=
                 std::string::npos);
}

static void Test_Rejects_CMap_Grid_Length_Mismatch()
{
    const auto dir = Unique_Temp_Path("topology_native_reader_bad_cmap");
    std::filesystem::create_directories(dir);
    const auto path = dir / "topology.spgt.h5";

    {
        HighFive::File file(path.string(), HighFive::File::Overwrite);
        Write_Scalar(file, "/topology/atom_count",
                     static_cast<std::int64_t>(3));
        Write_Int_Matrix(file, "/forcefield/cmap/atoms", {0, 1, 2, 0, 1}, 1, 5);
        Write_Int_Vector(file, "/forcefield/cmap/type", {0});
        Write_Int_Vector(file, "/forcefield/cmap/resolution", {2});
        Write_Float_Vector(file, "/forcefield/cmap/grid_value",
                           {0.1f, 0.2f, 0.3f});
    }

    TopologyNativeH5Reader reader;
    Require_Reader(reader.Open(path.string()), reader, "open bad cmap");
    NativeTopologyCoreState state;
    REQUIRE_TRUE(!reader.Read_Core_State(&state));
    REQUIRE_TRUE(reader.Last_Error().find("grid_value length") !=
                 std::string::npos);
}

static void Test_Rejects_LJ_Soft_Core_Pair_Count_Mismatch()
{
    const auto dir = Unique_Temp_Path("topology_native_reader_bad_lj_soft");
    std::filesystem::create_directories(dir);
    const auto path = dir / "topology.spgt.h5";

    {
        HighFive::File file(path.string(), HighFive::File::Overwrite);
        Write_Scalar(file, "/topology/atom_count",
                     static_cast<std::int64_t>(3));
        Write_Int_Vector(file, "/forcefield/lj_soft_core/atom_type_A",
                         {0, 1, 1});
        Write_Int_Vector(file, "/forcefield/lj_soft_core/atom_type_B",
                         {0, 0, 1});
        Write_Scalar(file, "/forcefield/lj_soft_core/atom_type_count_A", 2);
        Write_Scalar(file, "/forcefield/lj_soft_core/atom_type_count_B", 2);
        Write_Float_Vector(file, "/forcefield/lj_soft_core/pair_AA",
                           {1.0f, 2.0f});
        Write_Float_Vector(file, "/forcefield/lj_soft_core/pair_AB",
                           {4.0f, 5.0f, 6.0f});
        Write_Float_Vector(file, "/forcefield/lj_soft_core/pair_BA",
                           {7.0f, 8.0f, 9.0f});
        Write_Float_Vector(file, "/forcefield/lj_soft_core/pair_BB",
                           {10.0f, 11.0f, 12.0f});
    }

    TopologyNativeH5Reader reader;
    Require_Reader(reader.Open(path.string()), reader, "open bad lj soft");
    NativeTopologyCoreState state;
    REQUIRE_TRUE(!reader.Read_Core_State(&state));
    REQUIRE_TRUE(reader.Last_Error().find("pair parameter lengths") !=
                 std::string::npos);
}

int main()
{
    try
    {
        Test_Reads_Native_Mass_And_Charge();
        Test_Rejects_Length_Mismatch();
        Test_Rejects_Non_Positive_Mass();
        Test_Derives_Atom_Count_From_Charge_When_Metadata_Is_Absent();
        Test_Rejects_Invalid_Exclusion_Offset();
        Test_Rejects_LJ_Param_Count_Mismatch();
        Test_Reads_Split_LJ_Pair_Arrays();
        Test_Rejects_Dihedral_Runtime_Length_Mismatch();
        Test_Reads_Legacy_Dihedral_Parameter_Names();
        Test_Reads_Legacy_Scale_NB14_Params();
        Test_Rejects_Legacy_Scale_NB14_Without_LJ();
        Test_Rejects_NB14_Param_Count_Mismatch();
        Test_Rejects_Virtual_Atom_Arity_Mismatch();
        Test_Rejects_Non_Finite_Virtual_Atom_Parameter();
        Test_Rejects_GB_Atom_Count_Mismatch();
        Test_Rejects_Urey_Bradley_Length_Mismatch();
        Test_Rejects_CMap_Grid_Length_Mismatch();
        Test_Rejects_LJ_Soft_Core_Pair_Count_Mismatch();
    }
    catch (const std::exception& err)
    {
        std::cerr << err.what() << std::endl;
        return 1;
    }
    return 0;
}

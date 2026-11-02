#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <highfive/highfive.hpp>
#include <iomanip>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace SpongeH5MD
{
namespace detail
{
template <typename T>
inline hid_t Native_H5_Type();

template <>
inline hid_t Native_H5_Type<int>()
{
    return H5T_NATIVE_INT;
}

template <>
inline hid_t Native_H5_Type<float>()
{
    return H5T_NATIVE_FLOAT;
}

inline std::string With_Label(const std::string& data, const std::string& label)
{
    if (label.empty())
    {
        return data;
    }
    return data + " !" + label;
}

template <typename T>
inline std::string Number_String(const T value)
{
    std::ostringstream out;
    out << std::setprecision(9) << value;
    return out.str();
}
}  // namespace detail

struct NativeEDIPDefinition
{
    int atom_type_count = 0;
    std::vector<int> atom_type;
    std::vector<float> pair_parameters;
    std::vector<float> triple_parameters;
};

struct NativeReaxFFAtomParameter
{
    std::string name;
    std::array<float, 32> values{};
};

struct NativeReaxFFBondParameter
{
    std::array<int, 2> type{};
    std::array<float, 16> values{};
};

template <std::size_t TypeColumns, std::size_t ValueColumns>
struct NativeReaxFFRow
{
    std::array<int, TypeColumns> type{};
    std::array<float, ValueColumns> values{};
};

struct NativeReaxFFDefinition
{
    std::vector<float> general;
    std::vector<NativeReaxFFAtomParameter> atoms;
    std::vector<NativeReaxFFBondParameter> bonds;
    std::vector<NativeReaxFFRow<2, 6>> off_diagonal;
    std::vector<NativeReaxFFRow<3, 7>> angles;
    std::vector<NativeReaxFFRow<4, 7>> torsions;
    std::vector<NativeReaxFFRow<3, 4>> hydrogen_bonds;
    std::vector<int> atom_type;
};

class TopologyManybodyH5Materializer
{
   public:
    bool Open(const std::string& file_path)
    {
        last_error_.clear();
        try
        {
            file_.reset(
                new HighFive::File(file_path, HighFive::File::ReadOnly));
            return true;
        }
        catch (const std::exception& err)
        {
            return Fail(std::string("failed to open topology H5 file: ") +
                        err.what());
        }
    }

    bool Has_EDIP() const { return Exists("/manybody/edip"); }

    bool Has_ReaxFF() const
    {
        return Exists("/manybody/reaxff/parameters") &&
               Exists("/manybody/reaxff/type");
    }

    bool Read_ReaxFF(NativeReaxFFDefinition* definition)
    {
        if (definition == nullptr)
        {
            return Fail("ReaxFF definition output is null");
        }
        if (!Ensure_File()) return false;
        try
        {
            NativeReaxFFDefinition result;
            const std::string root = "/manybody/reaxff/parameters";
            result.general = Read_Vector<float>(root + "/general/value",
                                                "ReaxFF general values");
            const auto general_count =
                Read_Scalar<std::int64_t>(root + "/general/count");
            if (general_count !=
                    static_cast<std::int64_t>(result.general.size()) ||
                result.general.size() < 39)
            {
                return Fail(
                    "ReaxFF general count/value mismatch or fewer than 39 "
                    "runtime parameters");
            }

            const auto atom_count =
                Read_Scalar<std::int64_t>(root + "/atom/count");
            const auto atom_names = Read_Vector<std::string>(
                root + "/atom/type_name", "ReaxFF atom type names");
            const auto atom_values =
                Read_Vector<float>(root + "/atom/value", "ReaxFF atom values");
            const auto atom_offsets = Read_Vector<std::int64_t>(
                root + "/atom/value_offset", "ReaxFF atom value offsets");
            const auto atom_line_offsets =
                Read_Vector<std::int64_t>(root + "/atom/line_value_offset",
                                          "ReaxFF atom line value offsets");
            if (atom_count <= 0 ||
                atom_names.size() != static_cast<std::size_t>(atom_count) ||
                atom_offsets.size() != atom_names.size() + 1 ||
                atom_line_offsets.size() != atom_names.size() * 4 + 1 ||
                atom_offsets.front() != 0 || atom_line_offsets.front() != 0 ||
                atom_offsets.back() !=
                    static_cast<std::int64_t>(atom_values.size()) ||
                atom_line_offsets.back() !=
                    static_cast<std::int64_t>(atom_values.size()))
            {
                return Fail("ReaxFF atom section shape is invalid");
            }
            std::map<std::string, int> type_map;
            result.atoms.resize(atom_names.size());
            for (std::size_t atom = 0; atom < atom_names.size(); ++atom)
            {
                if (atom_names[atom].empty() ||
                    type_map.count(atom_names[atom]) != 0)
                {
                    return Fail(
                        "ReaxFF atom type names must be non-empty and "
                        "unique");
                }
                if (atom_offsets[atom] !=
                        static_cast<std::int64_t>(atom * 32) ||
                    atom_offsets[atom + 1] !=
                        static_cast<std::int64_t>((atom + 1) * 32))
                {
                    return Fail(
                        "ReaxFF atom rows must contain four groups of eight "
                        "values");
                }
                for (std::size_t line = 0; line < 4; ++line)
                {
                    const std::size_t index = atom * 4 + line;
                    if (atom_line_offsets[index] !=
                            static_cast<std::int64_t>(index * 8) ||
                        atom_line_offsets[index + 1] !=
                            static_cast<std::int64_t>((index + 1) * 8))
                    {
                        return Fail(
                            "ReaxFF atom line rows must contain eight values");
                    }
                }
                type_map[atom_names[atom]] = static_cast<int>(atom);
                result.atoms[atom].name = atom_names[atom];
                std::copy_n(atom_values.begin() + atom * 32, 32,
                            result.atoms[atom].values.begin());
            }

            const auto bond_count =
                Read_Scalar<std::int64_t>(root + "/bond/count");
            const auto bond_types =
                Read_Matrix<int>(root + "/bond/type", 2, "ReaxFF bond type");
            const auto bond_values =
                Read_Vector<float>(root + "/bond/value", "ReaxFF bond values");
            const auto bond_offsets = Read_Vector<std::int64_t>(
                root + "/bond/value_offset", "ReaxFF bond value offsets");
            const auto bond_line_offsets =
                Read_Vector<std::int64_t>(root + "/bond/line_value_offset",
                                          "ReaxFF bond line value offsets");
            if (bond_count < 0 ||
                bond_types.size() != static_cast<std::size_t>(bond_count) * 2 ||
                bond_values.size() !=
                    static_cast<std::size_t>(bond_count) * 16 ||
                bond_offsets.size() !=
                    static_cast<std::size_t>(bond_count) + 1 ||
                bond_line_offsets.size() !=
                    static_cast<std::size_t>(bond_count) * 2 + 1)
            {
                return Fail("ReaxFF bond section shape is invalid");
            }
            result.bonds.resize(static_cast<std::size_t>(bond_count));
            for (std::size_t bond = 0; bond < result.bonds.size(); ++bond)
            {
                if (bond_offsets[bond] !=
                        static_cast<std::int64_t>(bond * 16) ||
                    bond_offsets[bond + 1] !=
                        static_cast<std::int64_t>((bond + 1) * 16) ||
                    bond_line_offsets[2 * bond] !=
                        static_cast<std::int64_t>(bond * 16) ||
                    bond_line_offsets[2 * bond + 1] !=
                        static_cast<std::int64_t>(bond * 16 + 8) ||
                    bond_line_offsets[2 * bond + 2] !=
                        static_cast<std::int64_t>((bond + 1) * 16))
                {
                    return Fail(
                        "ReaxFF bond rows must contain two groups of eight "
                        "values");
                }
                for (std::size_t col = 0; col < 2; ++col)
                {
                    const int type = bond_types[2 * bond + col];
                    if (type <= 0 || type > atom_count)
                    {
                        return Fail("ReaxFF bond type index is out of range");
                    }
                    result.bonds[bond].type[col] = type - 1;
                }
                std::copy_n(bond_values.begin() + bond * 16, 16,
                            result.bonds[bond].values.begin());
            }

            if (!Read_ReaxFF_Rows<2, 6>(root + "/off_diagonal", atom_count,
                                        true, &result.off_diagonal) ||
                !Read_ReaxFF_Rows<3, 7>(root + "/angle", atom_count, true,
                                        &result.angles) ||
                !Read_ReaxFF_Rows<4, 7>(root + "/torsion", atom_count, false,
                                        &result.torsions) ||
                !Read_ReaxFF_Rows<3, 4>(root + "/hydrogen_bond", atom_count,
                                        true, &result.hydrogen_bonds))
            {
                return false;
            }

            const auto assigned_names = Read_Vector<std::string>(
                "/manybody/reaxff/type/name", "ReaxFF assigned atom types");
            const auto assigned_count =
                Read_Scalar<std::int64_t>("/manybody/reaxff/type/count");
            if (assigned_count !=
                static_cast<std::int64_t>(assigned_names.size()))
            {
                return Fail(
                    "ReaxFF assigned atom type count/name length mismatch");
            }
            result.atom_type.reserve(assigned_names.size());
            for (const auto& name : assigned_names)
            {
                const auto found = type_map.find(name);
                if (found == type_map.end())
                {
                    return Fail("ReaxFF assigned atom type is not declared: " +
                                name);
                }
                result.atom_type.push_back(found->second);
            }
            *definition = std::move(result);
            return true;
        }
        catch (const std::exception& err)
        {
            return Fail(std::string("failed to read ReaxFF from native H5 "
                                    "payload: ") +
                        err.what());
        }
    }

    bool Read_EDIP(NativeEDIPDefinition* definition)
    {
        if (definition == nullptr)
        {
            return Fail("EDIP definition output is null");
        }
        if (!Ensure_File()) return false;
        try
        {
            NativeEDIPDefinition result;
            result.atom_type_count =
                Read_Scalar<int>("/manybody/edip/atom_type_count");
            result.atom_type =
                Read_Vector<int>("/manybody/edip/atom_type", "EDIP atom type");
            const auto pair_type = Read_Matrix<int>("/manybody/edip/pair/type",
                                                    2, "EDIP pair type");
            const auto pair_parameter = Read_Matrix<float>(
                "/manybody/edip/pair/parameters", 8, "EDIP pair parameters");
            const auto triple_type = Read_Matrix<int>(
                "/manybody/edip/triple/type", 3, "EDIP triple type");
            const auto triple_parameter =
                Read_Matrix<float>("/manybody/edip/triple/parameters", 9,
                                   "EDIP triple parameters");

            if (result.atom_type_count <= 0)
            {
                return Fail("/manybody/edip/atom_type_count must be positive");
            }
            const std::size_t type_count =
                static_cast<std::size_t>(result.atom_type_count);
            const std::size_t full_pair_count = type_count * type_count;
            const std::size_t full_triple_count = full_pair_count * type_count;
            if (pair_parameter.size() / 8 != pair_type.size() / 2)
            {
                return Fail(
                    "/manybody/edip/pair type/parameter row count mismatch");
            }
            const std::size_t pair_count = pair_type.size() / 2;
            if (pair_count != full_pair_count &&
                pair_count != type_count * (type_count + 1) / 2)
            {
                return Fail(
                    "/manybody/edip/pair row count must be either "
                    "atom_type_count^2 or triangular");
            }
            if (triple_type.size() / 3 != full_triple_count ||
                triple_parameter.size() / 9 != full_triple_count)
            {
                return Fail(
                    "/manybody/edip/triple payload does not match "
                    "atom_type_count^3");
            }

            result.pair_parameters.assign(full_pair_count * 8, 0.0f);
            std::vector<bool> pair_seen(full_pair_count, false);
            for (std::size_t row = 0; row < pair_count; ++row)
            {
                const int a = pair_type[2 * row];
                const int b = pair_type[2 * row + 1];
                if (a < 0 || b < 0 || a >= result.atom_type_count ||
                    b >= result.atom_type_count)
                {
                    return Fail("EDIP pair type index is out of range");
                }
                for (const auto pair :
                     {std::pair<int, int>{a, b}, std::pair<int, int>{b, a}})
                {
                    const std::size_t index =
                        static_cast<std::size_t>(pair.first) * type_count +
                        static_cast<std::size_t>(pair.second);
                    std::copy(pair_parameter.begin() + 8 * row,
                              pair_parameter.begin() + 8 * row + 8,
                              result.pair_parameters.begin() + 8 * index);
                    pair_seen[index] = true;
                }
            }
            if (std::find(pair_seen.begin(), pair_seen.end(), false) !=
                pair_seen.end())
            {
                return Fail("EDIP pair payload does not cover all type pairs");
            }

            result.triple_parameters.assign(full_triple_count * 9, 0.0f);
            std::vector<bool> triple_seen(full_triple_count, false);
            for (std::size_t row = 0; row < full_triple_count; ++row)
            {
                const int a = triple_type[3 * row];
                const int b = triple_type[3 * row + 1];
                const int c = triple_type[3 * row + 2];
                if (a < 0 || b < 0 || c < 0 || a >= result.atom_type_count ||
                    b >= result.atom_type_count || c >= result.atom_type_count)
                {
                    return Fail("EDIP triple type index is out of range");
                }
                const std::size_t index =
                    (static_cast<std::size_t>(a) * type_count +
                     static_cast<std::size_t>(b)) *
                        type_count +
                    static_cast<std::size_t>(c);
                if (triple_seen[index])
                {
                    return Fail("EDIP triple payload contains duplicate types");
                }
                std::copy(triple_parameter.begin() + 9 * row,
                          triple_parameter.begin() + 9 * row + 9,
                          result.triple_parameters.begin() + 9 * index);
                triple_seen[index] = true;
            }
            for (const int type : result.atom_type)
            {
                if (type < 0 || type >= result.atom_type_count)
                {
                    return Fail("EDIP atom type index is out of range");
                }
            }
            *definition = std::move(result);
            return true;
        }
        catch (const std::exception& err)
        {
            return Fail(std::string("failed to read EDIP from native H5 "
                                    "payload: ") +
                        err.what());
        }
    }

    bool Materialize_EDIP(const std::filesystem::path& path)
    {
        NativeEDIPDefinition definition;
        if (!Read_EDIP(&definition)) return false;
        try
        {
            std::filesystem::create_directories(path.parent_path());
            std::ofstream out(path);
            if (!out)
            {
                return Fail("failed to create materialized EDIP input: " +
                            path.string());
            }
            out << definition.atom_type.size() << ' '
                << definition.atom_type_count << "\n";
            out << "# pair\n";
            for (int a = 0; a < definition.atom_type_count; ++a)
            {
                for (int b = 0; b < definition.atom_type_count; ++b)
                {
                    out << a << ' ' << b;
                    const std::size_t index = static_cast<std::size_t>(
                        a * definition.atom_type_count + b);
                    for (std::size_t col = 0; col < 8; ++col)
                    {
                        out << ' '
                            << detail::Number_String(
                                   definition.pair_parameters[8 * index + col]);
                    }
                    out << "\n";
                }
            }
            out << "# triple\n";
            for (int a = 0; a < definition.atom_type_count; ++a)
            {
                for (int b = 0; b < definition.atom_type_count; ++b)
                {
                    for (int c = 0; c < definition.atom_type_count; ++c)
                    {
                        const std::size_t index =
                            (static_cast<std::size_t>(a) *
                                 definition.atom_type_count +
                             b) *
                                definition.atom_type_count +
                            c;
                        out << a << ' ' << b << ' ' << c;
                        for (std::size_t col = 0; col < 9; ++col)
                        {
                            out << ' '
                                << detail::Number_String(
                                       definition
                                           .triple_parameters[9 * index + col]);
                        }
                        out << "\n";
                    }
                }
            }
            out << "# atom types\n";
            for (int value : definition.atom_type)
            {
                out << value << "\n";
            }
            return true;
        }
        catch (const std::exception& err)
        {
            return Fail(std::string("failed to materialize EDIP from native "
                                    "H5 payload: ") +
                        err.what());
        }
    }

    bool Materialize_ReaxFF(const std::filesystem::path& parameter_path,
                            const std::filesystem::path& type_path)
    {
        if (!Ensure_File()) return false;
        try
        {
            std::filesystem::create_directories(parameter_path.parent_path());
            std::filesystem::create_directories(type_path.parent_path());
            if (!Write_ReaxFF_Parameter_File(parameter_path))
            {
                return false;
            }
            if (!Write_ReaxFF_Type_File(type_path))
            {
                return false;
            }
            return true;
        }
        catch (const std::exception& err)
        {
            return Fail(std::string("failed to materialize ReaxFF from native "
                                    "H5 payload: ") +
                        err.what());
        }
    }

    std::string Last_Error() const { return last_error_; }

   private:
    template <std::size_t TypeColumns, std::size_t ValueColumns>
    bool Read_ReaxFF_Rows(
        const std::string& root, const std::int64_t atom_type_count,
        const bool require_positive_types,
        std::vector<NativeReaxFFRow<TypeColumns, ValueColumns>>* rows)
    {
        const auto count = Read_Scalar<std::int64_t>(root + "/count");
        const auto types = Read_Matrix<int>(root + "/type", TypeColumns,
                                            "ReaxFF section type");
        const auto values = Read_Matrix<float>(root + "/value", ValueColumns,
                                               "ReaxFF section values");
        if (count < 0 ||
            types.size() != static_cast<std::size_t>(count) * TypeColumns ||
            values.size() != static_cast<std::size_t>(count) * ValueColumns)
        {
            return Fail("ReaxFF counted section shape is invalid at " + root);
        }
        rows->resize(static_cast<std::size_t>(count));
        for (std::size_t row = 0; row < rows->size(); ++row)
        {
            for (std::size_t col = 0; col < TypeColumns; ++col)
            {
                const int type = types[row * TypeColumns + col];
                const int minimum = require_positive_types ? 1 : 0;
                if (type < minimum || type > atom_type_count)
                {
                    return Fail(
                        "ReaxFF section type index is out of range at " + root);
                }
                (*rows)[row].type[col] = type == 0 ? -1 : type - 1;
            }
            std::copy_n(values.begin() + row * ValueColumns, ValueColumns,
                        (*rows)[row].values.begin());
        }
        return true;
    }

    bool Ensure_File()
    {
        if (file_ == nullptr)
        {
            return Fail("topology H5 materializer is not open");
        }
        return true;
    }

    bool Exists(const std::string& object_path) const
    {
        return file_ != nullptr && file_->exist(object_path);
    }

    template <typename T>
    T Read_Scalar(const std::string& dataset_path)
    {
        if (!Exists(dataset_path))
        {
            throw std::runtime_error("dataset is missing: " + dataset_path);
        }
        T value{};
        file_->getDataSet(dataset_path).read(value);
        return value;
    }

    template <typename T>
    std::vector<T> Read_Vector(const std::string& dataset_path,
                               const std::string& label)
    {
        const auto dims = Dimensions(dataset_path);
        if (dims.size() != 1)
        {
            throw std::runtime_error(label + " dataset " + dataset_path +
                                     " must be one-dimensional");
        }
        std::vector<T> values;
        file_->getDataSet(dataset_path).read(values);
        return values;
    }

    template <typename T>
    std::vector<T> Read_Matrix(const std::string& dataset_path,
                               const std::size_t columns,
                               const std::string& label)
    {
        const auto dims = Dimensions(dataset_path);
        if (dims.size() != 2 || dims[1] != columns)
        {
            std::ostringstream out;
            out << label << " dataset " << dataset_path << " must have shape "
                << "[n," << columns << "]";
            throw std::runtime_error(out.str());
        }
        std::vector<T> values(dims[0] * dims[1]);
        HighFive::DataSet dataset = file_->getDataSet(dataset_path);
        const hsize_t h_dims[2] = {static_cast<hsize_t>(dims[0]),
                                   static_cast<hsize_t>(dims[1])};
        hid_t mem_space = H5Screate_simple(2, h_dims, nullptr);
        if (mem_space < 0)
        {
            throw std::runtime_error(label +
                                     " failed to create memory dataspace at " +
                                     dataset_path);
        }
        const herr_t read_rc =
            H5Dread(dataset.getId(), detail::Native_H5_Type<T>(), mem_space,
                    H5S_ALL, H5P_DEFAULT, values.data());
        H5Sclose(mem_space);
        if (read_rc < 0)
        {
            throw std::runtime_error(label + " failed to read dataset at " +
                                     dataset_path);
        }
        return values;
    }

    std::vector<std::size_t> Dimensions(const std::string& dataset_path)
    {
        if (!Exists(dataset_path))
        {
            throw std::runtime_error("dataset is missing: " + dataset_path);
        }
        return file_->getDataSet(dataset_path).getSpace().getDimensions();
    }

    bool Write_ReaxFF_Type_File(const std::filesystem::path& path)
    {
        const auto names = Read_Vector<std::string>(
            "/manybody/reaxff/type/name", "ReaxFF atom type names");
        const std::int64_t count =
            Read_Scalar<std::int64_t>("/manybody/reaxff/type/count");
        if (count != static_cast<std::int64_t>(names.size()))
        {
            return Fail(
                "/manybody/reaxff/type/count does not match name "
                "length");
        }
        std::ofstream out(path);
        if (!out)
        {
            return Fail("failed to create materialized ReaxFF type input: " +
                        path.string());
        }
        out << names.size() << "\n";
        for (const auto& name : names)
        {
            out << name << "\n";
        }
        return true;
    }

    bool Write_ReaxFF_Parameter_File(const std::filesystem::path& path)
    {
        std::ofstream out(path);
        if (!out)
        {
            return Fail(
                "failed to create materialized ReaxFF parameter input: " +
                path.string());
        }

        const std::string root = "/manybody/reaxff/parameters";
        out << Read_Scalar<std::string>(root + "/header") << "\n";
        if (!Write_ReaxFF_General(out, root + "/general")) return false;
        if (!Write_ReaxFF_Atom(out, root + "/atom")) return false;
        if (!Write_ReaxFF_Bond(out, root + "/bond")) return false;
        if (!Write_ReaxFF_Simple_Count_Section(out, root + "/off_diagonal", 2,
                                               6))
        {
            return false;
        }
        if (!Write_ReaxFF_Simple_Count_Section(out, root + "/angle", 3, 7))
        {
            return false;
        }
        if (!Write_ReaxFF_Simple_Count_Section(out, root + "/torsion", 4, 7))
        {
            return false;
        }
        if (!Write_ReaxFF_Simple_Count_Section(out, root + "/hydrogen_bond", 3,
                                               4))
        {
            return false;
        }
        return true;
    }

    bool Write_ReaxFF_General(std::ofstream& out, const std::string& root)
    {
        const auto values =
            Read_Vector<float>(root + "/value", "ReaxFF general values");
        const auto labels =
            Read_Vector<std::string>(root + "/label", "ReaxFF general labels");
        const auto count = Read_Scalar<std::int64_t>(root + "/count");
        const auto count_label =
            Read_Scalar<std::string>(root + "/count_label");
        if (count != static_cast<std::int64_t>(values.size()) ||
            values.size() != labels.size())
        {
            return Fail("ReaxFF general count/value/label length mismatch");
        }
        if (count <= 38)
        {
            return Fail(
                "ReaxFF native H5 general parameter payload is "
                "incomplete for runtime initialization");
        }
        out << detail::With_Label(std::to_string(count), count_label) << "\n";
        for (std::size_t i = 0; i < values.size(); ++i)
        {
            out << detail::With_Label(detail::Number_String(values[i]),
                                      labels[i])
                << "\n";
        }
        return true;
    }

    bool Write_ReaxFF_Atom(std::ofstream& out, const std::string& root)
    {
        const auto count = Read_Scalar<std::int64_t>(root + "/count");
        const auto count_label =
            Read_Scalar<std::string>(root + "/count_label");
        const auto headers =
            Read_Vector<std::string>(root + "/header", "ReaxFF atom header");
        const auto type_names = Read_Vector<std::string>(
            root + "/type_name", "ReaxFF atom type name");
        const auto values =
            Read_Vector<float>(root + "/value", "ReaxFF atom values");
        const auto line_labels = Read_Vector<std::string>(
            root + "/line_label", "ReaxFF atom line labels");
        const auto line_offsets = Read_Vector<std::int64_t>(
            root + "/line_value_offset", "ReaxFF atom line value offsets");
        if (headers.size() != 3 ||
            type_names.size() != static_cast<std::size_t>(count) ||
            line_labels.size() != static_cast<std::size_t>(4 * count) ||
            line_offsets.size() != line_labels.size() + 1 ||
            line_offsets.back() != static_cast<std::int64_t>(values.size()))
        {
            return Fail("ReaxFF atom section shape is invalid");
        }
        out << detail::With_Label(std::to_string(count), count_label) << "\n";
        for (const auto& header : headers)
        {
            out << header << "\n";
        }
        for (std::size_t atom = 0; atom < type_names.size(); ++atom)
        {
            for (std::size_t local_line = 0; local_line < 4; ++local_line)
            {
                const std::size_t line = atom * 4 + local_line;
                std::ostringstream row;
                if (local_line == 0)
                {
                    row << type_names[atom];
                }
                const auto begin = static_cast<std::size_t>(line_offsets[line]);
                const auto end =
                    static_cast<std::size_t>(line_offsets[line + 1]);
                for (std::size_t value = begin; value < end; ++value)
                {
                    if (row.tellp() > 0) row << ' ';
                    row << detail::Number_String(values[value]);
                }
                out << detail::With_Label(row.str(), line_labels[line]) << "\n";
            }
        }
        return true;
    }

    bool Write_ReaxFF_Bond(std::ofstream& out, const std::string& root)
    {
        const auto count = Read_Scalar<std::int64_t>(root + "/count");
        const auto count_label =
            Read_Scalar<std::string>(root + "/count_label");
        const auto headers =
            Read_Vector<std::string>(root + "/header", "ReaxFF bond header");
        const auto types =
            Read_Matrix<int>(root + "/type", 2, "ReaxFF bond type");
        const auto values =
            Read_Vector<float>(root + "/value", "ReaxFF bond values");
        const auto line_labels = Read_Vector<std::string>(
            root + "/line_label", "ReaxFF bond line labels");
        const auto line_offsets = Read_Vector<std::int64_t>(
            root + "/line_value_offset", "ReaxFF bond line value offsets");
        if (headers.size() != 1 ||
            types.size() / 2 != static_cast<std::size_t>(count) ||
            line_labels.size() != static_cast<std::size_t>(2 * count) ||
            line_offsets.size() != line_labels.size() + 1 ||
            line_offsets.back() != static_cast<std::int64_t>(values.size()))
        {
            return Fail("ReaxFF bond section shape is invalid");
        }
        out << detail::With_Label(std::to_string(count), count_label) << "\n";
        out << headers[0] << "\n";
        for (std::size_t bond = 0; bond < static_cast<std::size_t>(count);
             ++bond)
        {
            for (std::size_t local_line = 0; local_line < 2; ++local_line)
            {
                const std::size_t line = bond * 2 + local_line;
                std::ostringstream row;
                if (local_line == 0)
                {
                    row << types[2 * bond] << ' ' << types[2 * bond + 1];
                }
                const auto begin = static_cast<std::size_t>(line_offsets[line]);
                const auto end =
                    static_cast<std::size_t>(line_offsets[line + 1]);
                for (std::size_t value = begin; value < end; ++value)
                {
                    if (row.tellp() > 0) row << ' ';
                    row << detail::Number_String(values[value]);
                }
                out << detail::With_Label(row.str(), line_labels[line]) << "\n";
            }
        }
        return true;
    }

    bool Write_ReaxFF_Simple_Count_Section(std::ofstream& out,
                                           const std::string& root,
                                           const std::size_t type_columns,
                                           const std::size_t value_columns)
    {
        const auto count = Read_Scalar<std::int64_t>(root + "/count");
        const auto count_label =
            Read_Scalar<std::string>(root + "/count_label");
        const auto types = Read_Matrix<int>(root + "/type", type_columns,
                                            "ReaxFF counted section type");
        const auto values = Read_Matrix<float>(root + "/value", value_columns,
                                               "ReaxFF counted section values");
        if (types.size() / type_columns != static_cast<std::size_t>(count) ||
            values.size() / value_columns != static_cast<std::size_t>(count))
        {
            return Fail("ReaxFF counted section row count mismatch at " + root);
        }
        out << detail::With_Label(std::to_string(count), count_label) << "\n";
        for (std::size_t row = 0; row < static_cast<std::size_t>(count); ++row)
        {
            for (std::size_t col = 0; col < type_columns; ++col)
            {
                if (col > 0) out << ' ';
                out << types[type_columns * row + col];
            }
            for (std::size_t col = 0; col < value_columns; ++col)
            {
                out << ' '
                    << detail::Number_String(values[value_columns * row + col]);
            }
            out << "\n";
        }
        return true;
    }

    bool Fail(const std::string& message)
    {
        last_error_ = message;
        return false;
    }

    std::unique_ptr<HighFive::File> file_;
    std::string last_error_;
};

template <typename ControllerType>
inline bool Materialize_Native_ReaxFF_Text_Input_From_H5(
    ControllerType* controller, const std::string& topology_h5_path,
    const std::filesystem::path& output_dir, std::string* error_message)
{
    auto fail = [error_message](const std::string& message)
    {
        if (error_message != nullptr)
        {
            *error_message = message;
        }
        return false;
    };

    if (controller == nullptr)
    {
        return fail("controller pointer is null");
    }

    TopologyManybodyH5Materializer materializer;
    if (!materializer.Open(topology_h5_path))
    {
        return fail(materializer.Last_Error());
    }

    const bool has_reaxff_in_file =
        controller->Command_Exist("REAXFF", "in_file");
    const bool has_reaxff_type_in_file =
        controller->Command_Exist("REAXFF", "type_in_file");
    if (materializer.Has_ReaxFF() && !has_reaxff_in_file &&
        !has_reaxff_type_in_file)
    {
        const auto parameter_path =
            std::filesystem::absolute(output_dir / "reaxff.txt")
                .lexically_normal();
        const auto type_path =
            std::filesystem::absolute(output_dir / "reaxff_type.txt")
                .lexically_normal();
        if (!materializer.Materialize_ReaxFF(parameter_path, type_path))
        {
            return fail(materializer.Last_Error());
        }
        controller->Set_Command("REAXFF_in_file",
                                parameter_path.string().c_str(), 0);
        controller->Set_Command("REAXFF_type_in_file",
                                type_path.string().c_str(), 0);
    }
    else if (materializer.Has_ReaxFF() &&
             has_reaxff_in_file != has_reaxff_type_in_file)
    {
        return fail(
            "partial ReaxFF legacy override is invalid: both "
            "REAXFF_in_file and REAXFF_type_in_file are required");
    }
    return true;
}
}  // namespace SpongeH5MD

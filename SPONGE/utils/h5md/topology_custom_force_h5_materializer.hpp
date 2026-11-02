#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <highfive/highfive.hpp>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "../float_classification.hpp"

namespace SpongeH5MD
{
struct ListedForceBinding
{
    std::string name;
    std::filesystem::path data_path;
};

struct NativeListedForceDefinition
{
    std::string name;
    std::string potential;
    std::vector<std::string> parameter_types;
    std::vector<std::string> parameter_names;
    std::string connected_atoms;
    std::string constrain_distance;
    int item_count = 0;
    std::vector<float> parameter_values;
};

struct NativePairwiseForceDefinition
{
    std::string name;
    std::string potential;
    std::vector<std::string> parameter_types;
    std::vector<std::string> parameter_names;
    bool with_ele = true;
    std::string electrostatic_potential;
    int atom_count = 0;
    int type_count = 0;
    int pair_count = 0;
    int ij_parameter_count = 0;
    std::vector<float> parameter_values;
    std::vector<int> atom_type;
};

class TopologyCustomForceH5Materializer
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

    bool Has_Pairwise() const
    {
        return Exists("/forcefield/custom_force/pairwise/name") &&
               Exists("/forcefield/custom_force/pairwise/data");
    }

    bool Has_Listed() const
    {
        return Exists("/forcefield/custom_force/listed/name") &&
               Exists("/forcefield/custom_force/listed/data");
    }

    bool Has_Bond_Soft() const { return Exists("/forcefield/bond_soft/atoms"); }

    bool Read_Pairwise(NativePairwiseForceDefinition* definition)
    {
        if (!Ensure_File()) return false;
        if (definition == nullptr)
        {
            return Fail("custom pairwise definition output is null");
        }
        try
        {
            const std::string root = "/forcefield/custom_force/pairwise";
            NativePairwiseForceDefinition result;
            result.name = Read_String(root + "/name");
            result.potential = Read_String(root + "/potential");
            result.parameter_types =
                Read_String_Vector(root + "/parameters/type");
            result.parameter_names =
                Read_String_Vector(root + "/parameters/name");
            result.ij_parameter_count =
                Read_Int_Scalar(root + "/parameters/ij_count");
            result.with_ele = Read_Bool(root + "/with_ele");
            if (Exists(root + "/electrostatic_potential"))
            {
                result.electrostatic_potential =
                    Read_String(root + "/electrostatic_potential");
            }
            const std::string data_root =
                root + "/data/" + Safe_Name(result.name);
            result.atom_count = Read_Int_Scalar(data_root + "/atom_count");
            result.type_count = Read_Int_Scalar(data_root + "/type_count");
            result.pair_count = Read_Int_Scalar(data_root + "/pair_count");
            result.parameter_values =
                Read_Float_Matrix(data_root + "/parameter/value");
            result.atom_type = Read_Int_Vector(data_root + "/atom_type");
            if (!Validate_Native_Pairwise_Definition(result)) return false;
            *definition = std::move(result);
            return true;
        }
        catch (const std::exception& err)
        {
            return Fail(std::string("failed to read custom pairwise force "
                                    "from native H5 payload: ") +
                        err.what());
        }
    }

    bool Read_Listed(std::vector<NativeListedForceDefinition>* definitions)
    {
        if (!Ensure_File()) return false;
        if (definitions == nullptr)
        {
            return Fail("custom listed definition output is null");
        }
        try
        {
            const std::string root = "/forcefield/custom_force/listed";
            const auto names = Read_String_Vector(root + "/name");
            const auto potentials = Read_String_Vector(root + "/potential");
            const auto connected_atoms =
                Read_String_Vector(root + "/connected_atoms");
            const auto constrain_distance =
                Read_String_Vector(root + "/constrain_distance");
            const auto parameter_types =
                Read_String_Vector(root + "/parameters/type");
            const auto parameter_names =
                Read_String_Vector(root + "/parameters/name");
            const auto parameter_offsets =
                Read_Int64_Vector(root + "/parameters/offset");
            if (names.empty() || potentials.size() != names.size() ||
                connected_atoms.size() != names.size() ||
                constrain_distance.size() != names.size())
            {
                return Fail("custom listed descriptor array size mismatch");
            }
            if (parameter_types.size() != parameter_names.size() ||
                parameter_offsets.size() != names.size() + 1 ||
                parameter_offsets.front() != 0 ||
                parameter_offsets.back() !=
                    static_cast<std::int64_t>(parameter_names.size()) ||
                !std::is_sorted(parameter_offsets.begin(),
                                parameter_offsets.end()))
            {
                return Fail("custom listed parameter metadata is invalid");
            }

            definitions->clear();
            definitions->reserve(names.size());
            for (std::size_t force = 0; force < names.size(); ++force)
            {
                NativeListedForceDefinition definition;
                definition.name = names[force];
                definition.potential = potentials[force];
                definition.connected_atoms = connected_atoms[force];
                definition.constrain_distance = constrain_distance[force];
                const auto begin =
                    static_cast<std::size_t>(parameter_offsets[force]);
                const auto end =
                    static_cast<std::size_t>(parameter_offsets[force + 1]);
                definition.parameter_types.assign(
                    parameter_types.begin() + begin,
                    parameter_types.begin() + end);
                definition.parameter_names.assign(
                    parameter_names.begin() + begin,
                    parameter_names.begin() + end);
                const std::string data_root =
                    root + "/data/" + Safe_Name(definition.name);
                definition.item_count =
                    Read_Int_Scalar(data_root + "/item_count");
                definition.parameter_values =
                    Read_Float_Matrix(data_root + "/parameter/value");
                if (!Validate_Native_Listed_Definition(definition))
                {
                    return false;
                }
                definitions->push_back(std::move(definition));
            }
            return true;
        }
        catch (const std::exception& err)
        {
            return Fail(std::string("failed to read custom listed force from "
                                    "native H5 payload: ") +
                        err.what());
        }
    }

    bool Read_Bond_Soft(const float lambda_bond, const float alpha,
                        NativeListedForceDefinition* definition)
    {
        if (!Ensure_File()) return false;
        if (definition == nullptr)
        {
            return Fail("bond_soft definition output is null");
        }
        try
        {
            if (!SpongeFloat::Is_Finite(lambda_bond) ||
                !SpongeFloat::Is_Finite(alpha))
            {
                return Fail("bond_soft lambda and alpha must be finite");
            }
            const std::string root = "/forcefield/bond_soft";
            const auto atoms = Read_Int_Matrix(root + "/atoms", 2);
            const auto k = Read_Float_Vector(root + "/k");
            const auto r0 = Read_Float_Vector(root + "/r0");
            const auto from_a_or_b = Read_Int_Vector(root + "/from_a_or_b");
            const std::size_t count = atoms.size() / 2;
            if (k.size() != count || r0.size() != count ||
                from_a_or_b.size() != count)
            {
                return Fail("bond_soft typed dataset lengths differ");
            }
            NativeListedForceDefinition result;
            result.name = "bond_soft";
            result.potential =
                "float lambda_eff = from_a_or_b == 0 ? 1.0f - lambda_bond "
                ": lambda_bond;\n"
                "SADfloat<9> delta = r_ab - r0;\n"
                "E = lambda_eff * k * delta * delta / (1.0f + alpha * "
                "(1.0f - lambda_eff) * delta * delta);";
            result.parameter_types = {"int", "int",   "float", "float",
                                      "int", "float", "float"};
            result.parameter_names = {"atom_a", "atom_b",      "k",
                                      "r0",     "from_a_or_b", "lambda_bond",
                                      "alpha"};
            result.item_count = static_cast<int>(count);
            result.parameter_values.reserve(count * 7);
            for (std::size_t index = 0; index < count; ++index)
            {
                if (atoms[2 * index] < 0 || atoms[2 * index + 1] < 0 ||
                    atoms[2 * index] == atoms[2 * index + 1])
                {
                    return Fail("bond_soft atom indices are invalid");
                }
                if (!SpongeFloat::Is_Finite(k[index]) ||
                    !SpongeFloat::Is_Finite(r0[index]) ||
                    (from_a_or_b[index] != 0 && from_a_or_b[index] != 1))
                {
                    return Fail("bond_soft typed values are invalid");
                }
                result.parameter_values.insert(
                    result.parameter_values.end(),
                    {static_cast<float>(atoms[2 * index]),
                     static_cast<float>(atoms[2 * index + 1]), k[index],
                     r0[index], static_cast<float>(from_a_or_b[index]),
                     lambda_bond, alpha});
            }
            *definition = std::move(result);
            return true;
        }
        catch (const std::exception& err)
        {
            return Fail(std::string("failed to read bond_soft from native H5 "
                                    "payload: ") +
                        err.what());
        }
    }

    bool Materialize_Pairwise(const std::filesystem::path& descriptor_path,
                              const std::filesystem::path& data_path)
    {
        if (!Ensure_File()) return false;
        try
        {
            const std::string root = "/forcefield/custom_force/pairwise";
            const auto name = Read_String(root + "/name");
            const auto potential = Read_String(root + "/potential");
            const auto parameter_types =
                Read_String_Vector(root + "/parameters/type");
            const auto parameter_names =
                Read_String_Vector(root + "/parameters/name");
            const bool with_ele = Read_Bool(root + "/with_ele");
            const std::string data_root = root + "/data/" + name;
            const auto parameter_values =
                Read_Float_Matrix(data_root + "/parameter/value");
            const auto atom_type = Read_Int_Vector(data_root + "/atom_type");
            const auto atom_count = Read_Int_Scalar(data_root + "/atom_count");
            const auto type_count = Read_Int_Scalar(data_root + "/type_count");
            const auto pair_count = Read_Int_Scalar(data_root + "/pair_count");

            if (parameter_types.size() != parameter_names.size())
            {
                return Fail(
                    "custom pairwise parameter type/name count mismatch");
            }
            if (atom_count != static_cast<int>(atom_type.size()))
            {
                return Fail(
                    "custom pairwise atom_count does not match atom_type");
            }
            if (pair_count * static_cast<int>(parameter_types.size()) !=
                static_cast<int>(parameter_values.size()))
            {
                return Fail("custom pairwise parameter matrix size mismatch");
            }

            Write_Pairwise_Descriptor(descriptor_path, name, potential,
                                      parameter_types, parameter_names,
                                      with_ele);
            Write_Pairwise_Data(data_path, atom_count, type_count,
                                parameter_values, atom_type,
                                parameter_types.size());
            return true;
        }
        catch (const std::exception& err)
        {
            return Fail(std::string("failed to materialize custom pairwise "
                                    "force from native H5 payload: ") +
                        err.what());
        }
    }

    bool Materialize_Listed(const std::filesystem::path& descriptor_path,
                            const std::filesystem::path& output_dir,
                            std::vector<ListedForceBinding>* bindings)
    {
        if (!Ensure_File()) return false;
        try
        {
            const std::string root = "/forcefield/custom_force/listed";
            const auto names = Read_String_Vector(root + "/name");
            const auto potentials = Read_String_Vector(root + "/potential");
            const auto connected_atoms =
                Read_String_Vector(root + "/connected_atoms");
            const auto constrain_distance =
                Read_String_Vector(root + "/constrain_distance");
            const auto parameter_types =
                Read_String_Vector(root + "/parameters/type");
            const auto parameter_names =
                Read_String_Vector(root + "/parameters/name");
            const auto parameter_offsets =
                Read_Int64_Vector(root + "/parameters/offset");
            if (names.empty())
            {
                return Fail("custom listed force name list is empty");
            }
            if (potentials.size() != names.size() ||
                connected_atoms.size() != names.size() ||
                constrain_distance.size() != names.size())
            {
                return Fail("custom listed descriptor array size mismatch");
            }
            if (parameter_types.size() != parameter_names.size())
            {
                return Fail("custom listed parameter type/name count mismatch");
            }
            if (parameter_offsets.size() != names.size() + 1 ||
                parameter_offsets.front() != 0 ||
                parameter_offsets.back() !=
                    static_cast<std::int64_t>(parameter_names.size()) ||
                !std::is_sorted(parameter_offsets.begin(),
                                parameter_offsets.end()))
            {
                return Fail("custom listed parameter offsets are invalid");
            }

            Write_Listed_Descriptor(descriptor_path, names, potentials,
                                    parameter_types, parameter_names,
                                    parameter_offsets, connected_atoms,
                                    constrain_distance);
            if (bindings == nullptr)
            {
                return Fail("custom listed binding output is null");
            }
            bindings->clear();
            for (std::size_t force = 0; force < names.size(); ++force)
            {
                const auto data_path =
                    output_dir / (Safe_Name(names[force]) + ".txt");
                const std::size_t parameter_count = static_cast<std::size_t>(
                    parameter_offsets[force + 1] - parameter_offsets[force]);
                Write_Listed_Data(data_path, root, names[force],
                                  parameter_count);
                bindings->push_back({names[force], data_path});
            }
            return true;
        }
        catch (const std::exception& err)
        {
            return Fail(std::string("failed to materialize custom listed "
                                    "force from native H5 payload: ") +
                        err.what());
        }
    }

    bool Materialize_Bond_Soft(const std::filesystem::path& descriptor_path,
                               const std::filesystem::path& data_path,
                               const float lambda_bond, const float alpha,
                               const bool append_descriptor)
    {
        if (!Ensure_File()) return false;
        try
        {
            if (!SpongeFloat::Is_Finite(lambda_bond) ||
                !SpongeFloat::Is_Finite(alpha))
            {
                return Fail("bond_soft lambda and alpha must be finite");
            }
            const std::string root = "/forcefield/bond_soft";
            const auto atoms = Read_Int_Matrix(root + "/atoms", 2);
            const auto k = Read_Float_Vector(root + "/k");
            const auto r0 = Read_Float_Vector(root + "/r0");
            const auto from_a_or_b = Read_Int_Vector(root + "/from_a_or_b");
            const std::size_t count = atoms.size() / 2;
            if (k.size() != count || r0.size() != count ||
                from_a_or_b.size() != count)
            {
                return Fail("bond_soft typed dataset lengths differ");
            }
            for (const int value : from_a_or_b)
            {
                if (value != 0 && value != 1)
                {
                    return Fail("bond_soft from_a_or_b values must be 0 or 1");
                }
            }
            Write_Bond_Soft_Descriptor(descriptor_path, append_descriptor);
            Write_Bond_Soft_Data(data_path, atoms, k, r0, from_a_or_b,
                                 lambda_bond, alpha);
            return true;
        }
        catch (const std::exception& err)
        {
            return Fail(std::string("failed to materialize bond_soft from "
                                    "native H5 payload: ") +
                        err.what());
        }
    }

    std::string Last_Error() const { return last_error_; }

   private:
    bool Validate_Native_Pairwise_Definition(
        const NativePairwiseForceDefinition& definition)
    {
        if (definition.name.empty() || definition.name.size() >= 512 ||
            definition.name.find_first_of("[]\r\n\t ") != std::string::npos ||
            definition.potential.empty())
        {
            return Fail("custom pairwise descriptor is invalid");
        }
        if (definition.parameter_types.size() !=
                definition.parameter_names.size() ||
            definition.ij_parameter_count <= 0 ||
            definition.ij_parameter_count !=
                static_cast<int>(definition.parameter_names.size()))
        {
            return Fail(
                "custom pairwise currently requires every parameter "
                "to be an *_ij pair parameter");
        }
        for (std::size_t index = 0; index < definition.parameter_names.size();
             ++index)
        {
            if (definition.parameter_types[index] != "int" &&
                definition.parameter_types[index] != "float")
            {
                return Fail(
                    "custom pairwise parameter type must be int or float");
            }
            if (index <
                    static_cast<std::size_t>(definition.ij_parameter_count) &&
                (definition.parameter_names[index].size() < 3 ||
                 definition.parameter_names[index].compare(
                     definition.parameter_names[index].size() - 3, 3, "_ij") !=
                     0))
            {
                return Fail("custom pairwise *_ij parameters must be first");
            }
        }
        if (definition.atom_count <= 0 || definition.type_count <= 0 ||
            definition.pair_count !=
                definition.type_count * (definition.type_count + 1) / 2 ||
            definition.atom_type.size() !=
                static_cast<std::size_t>(definition.atom_count) ||
            definition.parameter_values.size() !=
                static_cast<std::size_t>(definition.ij_parameter_count) *
                    definition.pair_count)
        {
            return Fail("custom pairwise data dimensions are invalid");
        }
        for (const int type : definition.atom_type)
        {
            if (type < 0 || type >= definition.type_count)
            {
                return Fail("custom pairwise atom type is out of range");
            }
        }
        for (std::size_t parameter = 0;
             parameter <
             static_cast<std::size_t>(definition.ij_parameter_count);
             ++parameter)
        {
            for (int pair = 0; pair < definition.pair_count; ++pair)
            {
                const float value =
                    definition
                        .parameter_values[parameter * definition.pair_count +
                                          pair];
                if (!SpongeFloat::Is_Finite(value) ||
                    (definition.parameter_types[parameter] == "int" &&
                     std::trunc(value) != value))
                {
                    return Fail("custom pairwise parameter value is invalid");
                }
            }
        }
        return true;
    }

    bool Validate_Native_Listed_Definition(
        const NativeListedForceDefinition& definition)
    {
        if (definition.name.empty() || definition.name.size() >= 512 ||
            definition.name.find_first_of("[]\r\n\t ") != std::string::npos)
        {
            return Fail("custom listed force name is invalid");
        }
        if (definition.potential.empty() ||
            definition.parameter_names.empty() || definition.item_count < 0 ||
            definition.parameter_names.size() !=
                definition.parameter_types.size())
        {
            return Fail("custom listed force definition is incomplete");
        }
        const std::size_t parameter_count = definition.parameter_names.size();
        if (definition.parameter_values.size() !=
            static_cast<std::size_t>(definition.item_count) * parameter_count)
        {
            return Fail("custom listed parameter matrix size mismatch");
        }
        for (std::size_t parameter = 0; parameter < parameter_count;
             ++parameter)
        {
            if (definition.parameter_types[parameter] != "int" &&
                definition.parameter_types[parameter] != "float")
            {
                return Fail(
                    "custom listed parameter type must be int or float");
            }
            for (int item = 0; item < definition.item_count; ++item)
            {
                const float value =
                    definition.parameter_values[static_cast<std::size_t>(item) *
                                                    parameter_count +
                                                parameter];
                if (!SpongeFloat::Is_Finite(value) ||
                    (definition.parameter_types[parameter] == "int" &&
                     std::trunc(value) != value))
                {
                    return Fail("custom listed parameter value is invalid");
                }
            }
        }
        return true;
    }

    static std::string Number_String(const float value)
    {
        std::ostringstream out;
        out << std::setprecision(9) << value;
        return out.str();
    }

    bool Ensure_File()
    {
        if (file_ == nullptr)
        {
            return Fail("topology H5 custom force materializer is not open");
        }
        return true;
    }

    bool Exists(const std::string& object_path) const
    {
        return file_ != nullptr && file_->exist(object_path);
    }

    std::vector<std::size_t> Dimensions(const std::string& dataset_path)
    {
        if (!Exists(dataset_path))
        {
            throw std::runtime_error("dataset is missing: " + dataset_path);
        }
        return file_->getDataSet(dataset_path).getSpace().getDimensions();
    }

    std::string Read_String(const std::string& dataset_path)
    {
        std::string value;
        file_->getDataSet(dataset_path).read(value);
        return value;
    }

    std::vector<std::string> Read_String_Vector(const std::string& dataset_path)
    {
        std::vector<std::string> values;
        file_->getDataSet(dataset_path).read(values);
        return values;
    }

    int Read_Int_Scalar(const std::string& dataset_path)
    {
        int value = 0;
        file_->getDataSet(dataset_path).read(value);
        return value;
    }

    bool Read_Bool(const std::string& dataset_path)
    {
        signed char value = 0;
        const auto dataset = file_->getDataSet(dataset_path);
        if (H5Dread(dataset.getId(), H5T_NATIVE_SCHAR, H5S_ALL, H5S_ALL,
                    H5P_DEFAULT, &value) < 0)
        {
            throw std::runtime_error("failed to read enum bool dataset: " +
                                     dataset_path);
        }
        return value != 0;
    }

    std::vector<int> Read_Int_Vector(const std::string& dataset_path)
    {
        const auto dims = Dimensions(dataset_path);
        std::size_t count = 1;
        for (const auto dim : dims) count *= dim;
        std::vector<int> values(count);
        file_->getDataSet(dataset_path).read(values);
        return values;
    }

    std::vector<std::int64_t> Read_Int64_Vector(const std::string& dataset_path)
    {
        const auto dims = Dimensions(dataset_path);
        std::size_t count = 1;
        for (const auto dim : dims) count *= dim;
        std::vector<std::int64_t> values(count);
        file_->getDataSet(dataset_path).read(values);
        return values;
    }

    std::vector<float> Read_Float_Vector(const std::string& dataset_path)
    {
        const auto dims = Dimensions(dataset_path);
        if (dims.size() != 1)
        {
            throw std::runtime_error("dataset must be a vector: " +
                                     dataset_path);
        }
        std::vector<float> values(dims[0]);
        file_->getDataSet(dataset_path).read(values);
        return values;
    }

    std::vector<int> Read_Int_Matrix(const std::string& dataset_path,
                                     const std::size_t columns)
    {
        const auto dims = Dimensions(dataset_path);
        if (dims.size() != 2 || dims[1] != columns)
        {
            throw std::runtime_error("dataset has invalid matrix shape: " +
                                     dataset_path);
        }
        std::vector<int> values(dims[0] * dims[1]);
        const auto dataset = file_->getDataSet(dataset_path);
        const hsize_t h_dims[2] = {static_cast<hsize_t>(dims[0]),
                                   static_cast<hsize_t>(dims[1])};
        hid_t mem_space = H5Screate_simple(2, h_dims, nullptr);
        if (mem_space < 0)
        {
            throw std::runtime_error("failed to create memory dataspace for " +
                                     dataset_path);
        }
        const herr_t read_rc =
            H5Dread(dataset.getId(), H5T_NATIVE_INT, mem_space, H5S_ALL,
                    H5P_DEFAULT, values.data());
        H5Sclose(mem_space);
        if (read_rc < 0)
        {
            throw std::runtime_error("failed to read int matrix: " +
                                     dataset_path);
        }
        return values;
    }

    std::vector<float> Read_Float_Matrix(const std::string& dataset_path)
    {
        const auto dims = Dimensions(dataset_path);
        if (dims.size() != 2)
        {
            throw std::runtime_error("dataset must be a matrix: " +
                                     dataset_path);
        }
        std::vector<float> values(dims[0] * dims[1]);
        const auto dataset = file_->getDataSet(dataset_path);
        const hsize_t h_dims[2] = {static_cast<hsize_t>(dims[0]),
                                   static_cast<hsize_t>(dims[1])};
        hid_t mem_space = H5Screate_simple(2, h_dims, nullptr);
        if (mem_space < 0)
        {
            throw std::runtime_error("failed to create memory dataspace for " +
                                     dataset_path);
        }
        const herr_t read_rc =
            H5Dread(dataset.getId(), H5T_NATIVE_FLOAT, mem_space, H5S_ALL,
                    H5P_DEFAULT, values.data());
        H5Sclose(mem_space);
        if (read_rc < 0)
        {
            throw std::runtime_error("failed to read float matrix: " +
                                     dataset_path);
        }
        return values;
    }

    void Write_Pairwise_Descriptor(
        const std::filesystem::path& path, const std::string& name,
        const std::string& potential,
        const std::vector<std::string>& parameter_types,
        const std::vector<std::string>& parameter_names, const bool with_ele)
    {
        std::filesystem::create_directories(path.parent_path());
        std::ofstream out(path);
        out << "[[[ " << name << " ]]]\n"
            << "[[ potential ]]\n"
            << potential << "\n"
            << "[[ parameters ]]\n";
        for (std::size_t i = 0; i < parameter_names.size(); ++i)
        {
            if (i != 0) out << ", ";
            out << parameter_types[i] << ' ' << parameter_names[i];
        }
        out << "\n[[ with_ele ]]\n"
            << (with_ele ? "true" : "false") << "\n"
            << "[[ end ]]\n";
    }

    void Write_Pairwise_Data(const std::filesystem::path& path,
                             const int atom_count, const int type_count,
                             const std::vector<float>& parameter_values,
                             const std::vector<int>& atom_type,
                             const std::size_t parameter_count)
    {
        std::filesystem::create_directories(path.parent_path());
        std::ofstream out(path);
        out << atom_count << ' ' << type_count << "\n";
        for (std::size_t row = 0;
             row < parameter_values.size() / parameter_count; ++row)
        {
            for (std::size_t col = 0; col < parameter_count; ++col)
            {
                if (col != 0) out << ' ';
                out << Number_String(
                    parameter_values[row * parameter_count + col]);
            }
            out << "\n";
        }
        for (const int value : atom_type)
        {
            out << value << "\n";
        }
    }

    void Write_Listed_Descriptor(
        const std::filesystem::path& path,
        const std::vector<std::string>& names,
        const std::vector<std::string>& potentials,
        const std::vector<std::string>& parameter_types,
        const std::vector<std::string>& parameter_names,
        const std::vector<std::int64_t>& parameter_offsets,
        const std::vector<std::string>& connected_atoms,
        const std::vector<std::string>& constrain_distance)
    {
        std::filesystem::create_directories(path.parent_path());
        std::ofstream out(path);
        for (std::size_t force = 0; force < names.size(); ++force)
        {
            out << "[[[ " << names[force] << " ]]]\n"
                << "[[ potential ]]\n"
                << potentials[force] << "\n"
                << "[[ parameters ]]\n";
            const auto begin =
                static_cast<std::size_t>(parameter_offsets[force]);
            const auto end =
                static_cast<std::size_t>(parameter_offsets[force + 1]);
            for (std::size_t i = begin; i < end; ++i)
            {
                if (i != begin) out << ", ";
                out << parameter_types[i] << ' ' << parameter_names[i];
            }
            out << "\n";
            if (!connected_atoms[force].empty())
            {
                out << "[[ connected_atoms ]]\n"
                    << connected_atoms[force] << "\n";
            }
            if (!constrain_distance[force].empty())
            {
                out << "[[ constrain_distance ]]\n"
                    << constrain_distance[force] << "\n";
            }
            out << "[[ end ]]\n";
        }
    }

    void Write_Listed_Data(const std::filesystem::path& path,
                           const std::string& root, const std::string& name,
                           const std::size_t parameter_count)
    {
        std::filesystem::create_directories(path.parent_path());
        std::ofstream out(path);
        const std::string data_root = root + "/data/" + Safe_Name(name);
        const int item_count = Read_Int_Scalar(data_root + "/item_count");
        const auto values = Read_Float_Matrix(data_root + "/parameter/value");
        if (values.size() !=
            static_cast<std::size_t>(item_count) * parameter_count)
        {
            throw std::runtime_error("custom listed data size mismatch for " +
                                     name);
        }
        out << item_count << "\n";
        for (int item = 0; item < item_count; ++item)
        {
            for (std::size_t param = 0; param < parameter_count; ++param)
            {
                if (param != 0) out << ' ';
                out << Number_String(values[item * parameter_count + param]);
            }
            out << "\n";
        }
    }

    static std::string Safe_Name(const std::string& name)
    {
        std::string safe = name;
        for (char& value : safe)
        {
            const unsigned char code = static_cast<unsigned char>(value);
            if (!(std::isalnum(code) || value == '_')) value = '_';
        }
        return safe.empty() ? "unnamed" : safe;
    }

    void Write_Bond_Soft_Descriptor(const std::filesystem::path& path,
                                    const bool append)
    {
        std::filesystem::create_directories(path.parent_path());
        std::ofstream out(path, append ? std::ios::app : std::ios::out);
        out << "[[[ bond_soft ]]]\n"
            << "[[ potential ]]\n"
            << "float lambda_eff = from_a_or_b == 0 ? 1.0f - lambda_bond : "
               "lambda_bond;\n"
            << "SADfloat<9> delta = r_ab - r0;\n"
            << "E = lambda_eff * k * delta * delta / (1.0f + alpha * (1.0f - "
               "lambda_eff) * delta * delta);\n"
            << "[[ parameters ]]\n"
            << "int atom_a, int atom_b, float k, float r0, int from_a_or_b, "
               "float lambda_bond, float alpha\n"
            << "[[ end ]]\n";
    }

    void Write_Bond_Soft_Data(const std::filesystem::path& path,
                              const std::vector<int>& atoms,
                              const std::vector<float>& k,
                              const std::vector<float>& r0,
                              const std::vector<int>& from_a_or_b,
                              const float lambda_bond, const float alpha)
    {
        std::filesystem::create_directories(path.parent_path());
        std::ofstream out(path);
        out << k.size() << "\n";
        for (std::size_t i = 0; i < k.size(); ++i)
        {
            out << atoms[2 * i] << ' ' << atoms[2 * i + 1] << ' '
                << Number_String(k[i]) << ' ' << Number_String(r0[i]) << ' '
                << from_a_or_b[i] << ' ' << Number_String(lambda_bond) << ' '
                << Number_String(alpha) << "\n";
        }
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
inline bool Materialize_Native_Custom_Force_Text_Inputs_From_H5(
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

    TopologyCustomForceH5Materializer materializer;
    if (!materializer.Open(topology_h5_path))
    {
        return fail(materializer.Last_Error());
    }

    if (materializer.Has_Pairwise() &&
        !controller->Command_Exist("custom_pair_in_file"))
    {
        const auto descriptor_path =
            std::filesystem::absolute(output_dir / "pairwise_force.txt")
                .lexically_normal();
        const auto data_path =
            std::filesystem::absolute(output_dir / "custom_pair.txt")
                .lexically_normal();
        if (!materializer.Materialize_Pairwise(descriptor_path, data_path))
        {
            return fail(materializer.Last_Error());
        }
        if (!controller->Command_Exist("pairwise_force_in_file"))
        {
            controller->Set_Command("pairwise_force_in_file",
                                    descriptor_path.string().c_str(), 0);
        }
        controller->Set_Command("custom_pair_in_file",
                                data_path.string().c_str(), 0);
    }

    const bool has_listed = materializer.Has_Listed();
    const bool has_bond_soft = materializer.Has_Bond_Soft();
    if (has_bond_soft && controller->Command_Exist("listed_forces_in_file"))
    {
        return fail(
            "bundled bond_soft data conflicts with existing "
            "listed_forces_in_file");
    }
    if ((has_listed || has_bond_soft) &&
        !controller->Command_Exist("listed_forces_in_file"))
    {
        const auto descriptor_path =
            std::filesystem::absolute(output_dir / "listed_forces.txt")
                .lexically_normal();
        std::vector<ListedForceBinding> bindings;
        if (has_listed &&
            !materializer.Materialize_Listed(
                descriptor_path, descriptor_path.parent_path(), &bindings))
        {
            return fail(materializer.Last_Error());
        }
        if (has_bond_soft)
        {
            if (!controller->Command_Exist("lambda_bond"))
            {
                return fail(
                    "lambda_bond is required when bundled bond_soft data is "
                    "present");
            }
            float lambda_bond = 0.0f;
            float alpha = 0.0f;
            try
            {
                lambda_bond = std::stof(controller->Command("lambda_bond"));
                if (controller->Command_Exist("soft_bond_alpha"))
                {
                    alpha = std::stof(controller->Command("soft_bond_alpha"));
                }
            }
            catch (const std::exception& err)
            {
                return fail(std::string("invalid bundled bond_soft scalar: ") +
                            err.what());
            }
            const auto data_path =
                std::filesystem::absolute(output_dir / "bond_soft.txt")
                    .lexically_normal();
            if (!materializer.Materialize_Bond_Soft(
                    descriptor_path, data_path, lambda_bond, alpha, has_listed))
            {
                return fail(materializer.Last_Error());
            }
            bindings.push_back({"bond_soft", data_path});
        }
        controller->Set_Command("listed_forces_in_file",
                                descriptor_path.string().c_str(), 0);
        for (const auto& binding : bindings)
        {
            const std::string command = binding.name + "_in_file";
            if (controller->Command_Exist(command.c_str()))
            {
                return fail("bundled listed force " + binding.name +
                            " conflicts with existing command " + command);
            }
            controller->Set_Command(command.c_str(),
                                    binding.data_path.string().c_str(), 0);
        }
    }
    return true;
}
}  // namespace SpongeH5MD

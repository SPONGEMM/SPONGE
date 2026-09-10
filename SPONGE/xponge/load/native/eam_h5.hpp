#pragma once

#include <hdf5.h>

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <highfive/highfive.hpp>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../../utils/float_classification.hpp"

namespace SpongeH5MD
{
class NativeEAMH5Materializer
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
        catch (const std::exception& error)
        {
            return Fail(std::string("failed to open topology H5 file: ") +
                        error.what());
        }
    }

    bool Has_EAM() const
    {
        return file_ != nullptr && file_->exist("/manybody/eam");
    }

    bool Materialize(const std::filesystem::path& parameter_path,
                     const std::filesystem::path& atom_type_path,
                     bool* has_atom_type)
    {
        if (has_atom_type == nullptr)
        {
            return Fail("native EAM atom-type presence pointer is null");
        }
        *has_atom_type = false;
        if (file_ == nullptr)
        {
            return Fail("native EAM H5 materializer is not open");
        }

        try
        {
            const std::string root = "/manybody/eam";
            const std::string format = Read_Text_Scalar(root + "/format");
            const int type_count = Read_Scalar<int>(root + "/atom_type_count");
            const int nrho = Read_Scalar<int>(root + "/nrho");
            const float drho = Read_Scalar<float>(root + "/drho");
            const int nr = Read_Scalar<int>(root + "/nr");
            const float dr = Read_Scalar<float>(root + "/dr");
            const float cut = Read_Scalar<float>(root + "/cut");
            if (type_count <= 0 || nrho <= 0 || nr <= 0)
            {
                return Fail(
                    "native EAM atom_type_count, nrho, and nr must be "
                    "positive");
            }
            if (!SpongeFloat::Is_Finite(drho) || drho <= 0.0f ||
                !SpongeFloat::Is_Finite(dr) || dr <= 0.0f ||
                !SpongeFloat::Is_Finite(cut) || cut <= 0.0f)
            {
                return Fail(
                    "native EAM drho, dr, and cut must be positive "
                    "finite values");
            }

            const auto embed =
                Read_Flat<float>(root + "/embed/raw_ev",
                                 {static_cast<std::size_t>(type_count),
                                  static_cast<std::size_t>(nrho)},
                                 "EAM raw embedding table");
            const auto density =
                Read_Flat<float>(root + "/electron_density/value",
                                 {static_cast<std::size_t>(type_count),
                                  static_cast<std::size_t>(nr)},
                                 "EAM electron-density table");
            Validate_Finite(embed, root + "/embed/raw_ev");
            Validate_Finite(density, root + "/electron_density/value");

            std::filesystem::create_directories(parameter_path.parent_path());
            std::ofstream output(parameter_path);
            if (!output)
            {
                return Fail("failed to create materialized EAM input: " +
                            parameter_path.string());
            }
            output << std::setprecision(9);
            if (format == "funcfl")
            {
                if (!Write_Funcfl(output, root, type_count, nrho, drho, nr, dr,
                                  cut, embed, density))
                {
                    return false;
                }
            }
            else if (format == "setfl")
            {
                if (!Write_Setfl(output, root, type_count, nrho, drho, nr, dr,
                                 cut, embed, density))
                {
                    return false;
                }
            }
            else
            {
                return Fail("unsupported native EAM format: " + format);
            }
            output.close();
            if (!output)
            {
                return Fail("failed to write materialized EAM input: " +
                            parameter_path.string());
            }

            if (Exists(root + "/atom_type"))
            {
                const auto atom_type =
                    Read_Vector<int>(root + "/atom_type", "EAM atom type");
                std::ofstream type_output(atom_type_path);
                if (!type_output)
                {
                    return Fail(
                        "failed to create materialized EAM atom-type input: " +
                        atom_type_path.string());
                }
                for (int value : atom_type)
                {
                    type_output << value << '\n';
                }
                type_output.close();
                if (!type_output)
                {
                    return Fail(
                        "failed to write materialized EAM atom-type input: " +
                        atom_type_path.string());
                }
                *has_atom_type = true;
            }
            return true;
        }
        catch (const std::exception& error)
        {
            return Fail(std::string("failed to materialize native EAM: ") +
                        error.what());
        }
    }

    const std::string& Last_Error() const { return last_error_; }

   private:
    bool Exists(const std::string& path) const
    {
        return file_ != nullptr && file_->exist(path);
    }

    template <typename T>
    T Read_Scalar(const std::string& path) const
    {
        if (!Exists(path))
        {
            throw std::runtime_error("dataset is missing: " + path);
        }
        const auto dimensions =
            file_->getDataSet(path).getSpace().getDimensions();
        if (!dimensions.empty())
        {
            throw std::runtime_error(path + " must be a scalar dataset");
        }
        T value{};
        file_->getDataSet(path).read(value);
        return value;
    }

    std::string Read_Text_Scalar(const std::string& path) const
    {
        if (!Exists(path))
        {
            throw std::runtime_error("dataset is missing: " + path);
        }
        const auto dimensions =
            file_->getDataSet(path).getSpace().getDimensions();
        if (!dimensions.empty())
        {
            throw std::runtime_error(path + " must be a scalar dataset");
        }
        std::string value;
        file_->getDataSet(path).read(value);
        return value;
    }

    std::vector<std::string> Read_Text_Vector(const std::string& path,
                                              std::size_t expected) const
    {
        if (!Exists(path))
        {
            throw std::runtime_error("dataset is missing: " + path);
        }
        const auto dimensions =
            file_->getDataSet(path).getSpace().getDimensions();
        if (dimensions != std::vector<std::size_t>{expected})
        {
            throw std::runtime_error(path + " has an invalid length");
        }
        std::vector<std::string> values;
        file_->getDataSet(path).read(values);
        return values;
    }

    template <typename T>
    std::vector<T> Read_Vector(const std::string& path,
                               const std::string& label) const
    {
        if (!Exists(path))
        {
            throw std::runtime_error("dataset is missing: " + path);
        }
        const auto dimensions =
            file_->getDataSet(path).getSpace().getDimensions();
        if (dimensions.size() != 1)
        {
            throw std::runtime_error(label + " dataset " + path +
                                     " must be one-dimensional");
        }
        return Read_Flat<T>(path, dimensions, label);
    }

    template <typename T>
    std::vector<T> Read_Flat(const std::string& path,
                             const std::vector<std::size_t>& expected,
                             const std::string& label) const
    {
        if (!Exists(path))
        {
            throw std::runtime_error("dataset is missing: " + path);
        }
        const auto dimensions =
            file_->getDataSet(path).getSpace().getDimensions();
        if (dimensions != expected)
        {
            std::ostringstream message;
            message << label << " dataset " << path << " must have shape [";
            for (std::size_t index = 0; index < expected.size(); ++index)
            {
                if (index != 0) message << ',';
                message << expected[index];
            }
            message << ']';
            throw std::runtime_error(message.str());
        }
        std::size_t count = 1;
        for (std::size_t value : dimensions) count *= value;
        std::vector<T> values(count);
        HighFive::DataSet dataset = file_->getDataSet(path);
        if (H5Dread(dataset.getId(), Native_H5_Type<T>(), H5S_ALL, H5S_ALL,
                    H5P_DEFAULT, values.data()) < 0)
        {
            throw std::runtime_error(label + " failed to read dataset " + path);
        }
        return values;
    }

    bool Write_Funcfl(std::ofstream& output, const std::string& root,
                      int type_count, int nrho, float drho, int nr, float dr,
                      float cut, const std::vector<float>& embed,
                      const std::vector<float>& density)
    {
        if (type_count != 1)
        {
            return Fail(
                "native funcfl EAM must contain exactly one atom "
                "type");
        }
        const auto atomic_number =
            Read_Flat<int>(root + "/atomic_number", {1}, "EAM atomic number");
        const auto mass = Read_Flat<float>(root + "/mass", {1}, "EAM mass");
        const auto lattice_constant = Read_Flat<float>(
            root + "/lattice_constant", {1}, "EAM lattice constant");
        const auto lattice_type = Read_Text_Vector(root + "/lattice_type", 1);
        const auto z =
            Read_Flat<float>(root + "/funcfl/z", {static_cast<std::size_t>(nr)},
                             "EAM funcfl z table");
        Validate_Finite(mass, root + "/mass");
        Validate_Finite(lattice_constant, root + "/lattice_constant");
        Validate_Finite(z, root + "/funcfl/z");
        output << "Generated from bundled native EAM funcfl\n";
        output << atomic_number[0] << ' ' << mass[0] << ' '
               << lattice_constant[0] << ' ' << lattice_type[0] << '\n';
        output << nrho << ' ' << drho << ' ' << nr << ' ' << dr << ' ' << cut
               << '\n';
        Write_Values(output, embed);
        Write_Values(output, z);
        Write_Values(output, density);
        return true;
    }

    bool Write_Setfl(std::ofstream& output, const std::string& root,
                     int type_count, int nrho, float drho, int nr, float dr,
                     float cut, const std::vector<float>& embed,
                     const std::vector<float>& density)
    {
        const std::size_t types = static_cast<std::size_t>(type_count);
        const auto type_name = Read_Text_Vector(root + "/type_name", types);
        const auto atomic_number = Read_Flat<int>(root + "/atomic_number",
                                                  {types}, "EAM atomic number");
        const auto mass = Read_Flat<float>(root + "/mass", {types}, "EAM mass");
        const auto lattice_constant = Read_Flat<float>(
            root + "/lattice_constant", {types}, "EAM lattice constant");
        const auto lattice_type =
            Read_Text_Vector(root + "/lattice_type", types);
        const auto pair =
            Read_Flat<float>(root + "/pair_potential/value",
                             {types, types, static_cast<std::size_t>(nr)},
                             "EAM pair-potential table");
        Validate_Finite(mass, root + "/mass");
        Validate_Finite(lattice_constant, root + "/lattice_constant");
        Validate_Finite(pair, root + "/pair_potential/value");

        output << "Generated from bundled native EAM setfl\n\n\n";
        output << type_count;
        for (const std::string& value : type_name) output << ' ' << value;
        output << '\n';
        output << nrho << ' ' << drho << ' ' << nr << ' ' << dr << ' ' << cut
               << '\n';
        for (std::size_t type = 0; type < types; ++type)
        {
            output << atomic_number[type] << ' ' << mass[type] << ' '
                   << lattice_constant[type] << ' ' << lattice_type[type]
                   << '\n';
            Write_Values(output, embed, type * nrho, nrho);
            Write_Values(output, density, type * nr, nr);
        }
        for (std::size_t left = 0; left < types; ++left)
        {
            for (std::size_t right = 0; right <= left; ++right)
            {
                const std::size_t offset = (left * types + right) * nr;
                for (int index = 0; index < nr; ++index)
                {
                    const float radius = index == 0 ? 1.0e-8f : index * dr;
                    output << pair[offset + index] * radius / kEvToKcalMol
                           << '\n';
                }
            }
        }
        return true;
    }

    static void Write_Values(std::ofstream& output,
                             const std::vector<float>& values,
                             std::size_t offset = 0, std::size_t count = 0)
    {
        if (count == 0) count = values.size() - offset;
        for (std::size_t index = 0; index < count; ++index)
        {
            output << values[offset + index] << '\n';
        }
    }

    static void Validate_Finite(const std::vector<float>& values,
                                const std::string& path)
    {
        for (float value : values)
        {
            if (!SpongeFloat::Is_Finite(value))
            {
                throw std::runtime_error(path + " contains a non-finite value");
            }
        }
    }

    template <typename T>
    static hid_t Native_H5_Type();

    bool Fail(const std::string& message)
    {
        last_error_ = message;
        return false;
    }

    static constexpr float kEvToKcalMol = 23.0605480f;
    std::unique_ptr<HighFive::File> file_;
    std::string last_error_;
};

template <>
inline hid_t NativeEAMH5Materializer::Native_H5_Type<int>()
{
    return H5T_NATIVE_INT;
}

template <>
inline hid_t NativeEAMH5Materializer::Native_H5_Type<float>()
{
    return H5T_NATIVE_FLOAT;
}
}  // namespace SpongeH5MD

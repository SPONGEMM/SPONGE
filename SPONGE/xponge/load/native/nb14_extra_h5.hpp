#pragma once

#include <hdf5.h>

#include <cstddef>
#include <highfive/highfive.hpp>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace SpongeH5MD
{
struct NativeNB14ExtraState
{
    bool present = false;
    std::vector<int> atom_a;
    std::vector<int> atom_b;
    std::vector<float> A;
    std::vector<float> B;
    std::vector<float> cf_scale_factor;
};

class NativeNB14ExtraH5Reader
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

    bool Read(int atom_count, NativeNB14ExtraState* state)
    {
        if (state == nullptr)
        {
            return Fail("native nb14_extra state output pointer is null");
        }
        if (file_ == nullptr)
        {
            return Fail("native nb14_extra H5 reader is not open");
        }

        NativeNB14ExtraState result;
        try
        {
            const bool has_canonical = Exists("/forcefield/nb14") ||
                                       Exists("/forcefield/nb14/atoms") ||
                                       Exists("/forcefield/nb14/params");
            const bool has_extra = Exists("/forcefield/nb14_extra") ||
                                   Exists("/forcefield/nb14_extra/atoms") ||
                                   Exists("/forcefield/nb14_extra/params");
            if (!has_extra)
            {
                *state = result;
                return true;
            }
            if (has_canonical)
            {
                return Fail(
                    "native NB14 input cannot define both /forcefield/nb14 "
                    "and /forcefield/nb14_extra");
            }
            if (!Exists("/forcefield/nb14_extra/atoms") ||
                !Exists("/forcefield/nb14_extra/params"))
            {
                return Fail(
                    "native nb14_extra parameters require both "
                    "/forcefield/nb14_extra/atoms and "
                    "/forcefield/nb14_extra/params");
            }

            const auto atom_dimensions =
                Dimensions("/forcefield/nb14_extra/atoms");
            const auto param_dimensions =
                Dimensions("/forcefield/nb14_extra/params");
            if (atom_dimensions.size() != 2 || atom_dimensions[1] != 2)
            {
                return Fail(
                    "/forcefield/nb14_extra/atoms must have shape [n,2]");
            }
            if (param_dimensions.size() != 2 || param_dimensions[1] != 3)
            {
                return Fail(
                    "/forcefield/nb14_extra/params must have shape [n,3]");
            }
            if (atom_dimensions[0] != param_dimensions[0])
            {
                return Fail(
                    "native nb14_extra params row count must match atoms row "
                    "count");
            }

            const auto atoms =
                Read_Matrix<int>("/forcefield/nb14_extra/atoms",
                                 atom_dimensions, "nb14_extra atoms");
            const auto params =
                Read_Matrix<float>("/forcefield/nb14_extra/params",
                                   param_dimensions, "nb14_extra params");
            const std::size_t record_count = atom_dimensions[0];
            result.atom_a.resize(record_count);
            result.atom_b.resize(record_count);
            result.A.resize(record_count);
            result.B.resize(record_count);
            result.cf_scale_factor.resize(record_count);
            for (std::size_t index = 0; index < record_count; ++index)
            {
                const int atom_a = atoms[2 * index];
                const int atom_b = atoms[2 * index + 1];
                if (atom_a < 0 || atom_b < 0 ||
                    (atom_count > 0 &&
                     (atom_a >= atom_count || atom_b >= atom_count)))
                {
                    std::ostringstream message;
                    message << "/forcefield/nb14_extra/atoms contains atom "
                               "index outside [0, "
                            << atom_count << ") at row " << index;
                    return Fail(message.str());
                }
                result.atom_a[index] = atom_a;
                result.atom_b[index] = atom_b;
                result.A[index] = 12.0f * params[3 * index];
                result.B[index] = 6.0f * params[3 * index + 1];
                result.cf_scale_factor[index] = params[3 * index + 2];
            }
            result.present = true;
            *state = result;
            return true;
        }
        catch (const std::exception& error)
        {
            return Fail(std::string("failed to read native nb14_extra: ") +
                        error.what());
        }
    }

    const std::string& Last_Error() const { return last_error_; }

   private:
    bool Exists(const std::string& path) const
    {
        return file_ != nullptr && file_->exist(path);
    }

    std::vector<std::size_t> Dimensions(const std::string& path) const
    {
        return file_->getDataSet(path).getSpace().getDimensions();
    }

    template <typename T>
    std::vector<T> Read_Matrix(const std::string& path,
                               const std::vector<std::size_t>& dimensions,
                               const std::string& label) const
    {
        std::vector<T> values(dimensions[0] * dimensions[1]);
        HighFive::DataSet dataset = file_->getDataSet(path);
        const herr_t read_result =
            H5Dread(dataset.getId(), Native_H5_Type<T>(), H5S_ALL, H5S_ALL,
                    H5P_DEFAULT, values.data());
        if (read_result < 0)
        {
            throw std::runtime_error(label + " failed to read dataset at " +
                                     path);
        }
        return values;
    }

    template <typename T>
    static hid_t Native_H5_Type();

    bool Fail(const std::string& message)
    {
        last_error_ = message;
        return false;
    }

    std::unique_ptr<HighFive::File> file_;
    std::string last_error_;
};

template <>
inline hid_t NativeNB14ExtraH5Reader::Native_H5_Type<int>()
{
    return H5T_NATIVE_INT;
}

template <>
inline hid_t NativeNB14ExtraH5Reader::Native_H5_Type<float>()
{
    return H5T_NATIVE_FLOAT;
}
}  // namespace SpongeH5MD

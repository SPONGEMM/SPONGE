#pragma once

#include <array>
#include <cmath>
#include <highfive/highfive.hpp>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../../utils/float_classification.hpp"

namespace SpongeH5MD
{
struct NativeHardWallDefinition
{
    std::array<float, 3> bounds_low{{-INFINITY, -INFINITY, -INFINITY}};
    std::array<float, 3> bounds_high{{INFINITY, INFINITY, INFINITY}};
    bool allow_npt = false;
};

class NativeHardWallH5Reader
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
            return Fail(std::string("failed to open protocol H5 file: ") +
                        error.what());
        }
    }

    bool Has_Hard_Wall() const
    {
        return file_ != nullptr && file_->exist("/wall/hard");
    }

    bool Read(NativeHardWallDefinition* definition)
    {
        if (file_ == nullptr)
        {
            return Fail("native hard-wall protocol H5 reader is not open");
        }
        if (definition == nullptr)
        {
            return Fail("native hard-wall definition output is null");
        }
        try
        {
            NativeHardWallDefinition result;
            result.bounds_low =
                Read_Bounds("/wall/hard/bounds_low", "hard-wall low bounds");
            result.bounds_high =
                Read_Bounds("/wall/hard/bounds_high", "hard-wall high bounds");
            result.allow_npt = Read_Allow_Npt();
            Validate(result);
            *definition = result;
            return true;
        }
        catch (const std::exception& error)
        {
            return Fail(std::string("failed to read native hard wall: ") +
                        error.what());
        }
    }

    const std::string& Last_Error() const { return last_error_; }

   private:
    std::array<float, 3> Read_Bounds(const std::string& path,
                                     const std::string& label) const
    {
        if (!file_->exist(path))
        {
            throw std::runtime_error("dataset is missing: " + path);
        }
        const auto dataset = file_->getDataSet(path);
        const auto dimensions = dataset.getSpace().getDimensions();
        if (dimensions != std::vector<std::size_t>{3})
        {
            throw std::runtime_error(label + " dataset " + path +
                                     " must have shape [3]");
        }
        std::vector<float> values;
        dataset.read(values);
        if (values.size() != 3)
        {
            throw std::runtime_error(label + " value count must be three");
        }
        return {{values[0], values[1], values[2]}};
    }

    bool Read_Allow_Npt() const
    {
        const bool has_dataset = file_->exist("/wall/hard/allow_npt");
        bool has_attribute = false;
        std::int32_t dataset_value = 0;
        std::int32_t attribute_value = 0;
        if (has_dataset)
        {
            const auto dataset = file_->getDataSet("/wall/hard/allow_npt");
            if (!dataset.getSpace().getDimensions().empty())
            {
                throw std::runtime_error(
                    "/wall/hard/allow_npt dataset must be scalar");
            }
            dataset.read(dataset_value);
            Validate_Bool(dataset_value, "/wall/hard/allow_npt dataset");
        }
        const auto group = file_->getGroup("/wall/hard");
        if (group.hasAttribute("allow_npt"))
        {
            has_attribute = true;
            group.getAttribute("allow_npt").read(attribute_value);
            Validate_Bool(attribute_value, "/wall/hard allow_npt attribute");
        }
        if (has_dataset && has_attribute && dataset_value != attribute_value)
        {
            throw std::runtime_error(
                "/wall/hard allow_npt dataset and attribute disagree");
        }
        return (has_dataset ? dataset_value : attribute_value) != 0;
    }

    static void Validate_Bool(std::int32_t value, const std::string& label)
    {
        if (value != 0 && value != 1)
        {
            throw std::runtime_error(label + " must be 0 or 1");
        }
    }

    static void Validate(const NativeHardWallDefinition& definition)
    {
        bool has_finite_bound = false;
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            const float low = definition.bounds_low[axis];
            const float high = definition.bounds_high[axis];
            if (SpongeFloat::Is_Nan(low) || SpongeFloat::Is_Nan(high))
            {
                throw std::runtime_error(
                    "hard-wall bounds must not contain NaN");
            }
            if (!(low < high))
            {
                throw std::runtime_error(
                    "each hard-wall low bound must be smaller than its high "
                    "bound");
            }
            has_finite_bound = has_finite_bound ||
                               SpongeFloat::Is_Finite(low) ||
                               SpongeFloat::Is_Finite(high);
        }
        if (!has_finite_bound)
        {
            throw std::runtime_error(
                "hard-wall protocol requires at least one finite bound");
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
}  // namespace SpongeH5MD

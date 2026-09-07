#pragma once

#include <hdf5.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <highfive/highfive.hpp>
#include <iomanip>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../float_classification.hpp"
#include "protocol_cv_h5.hpp"

namespace SpongeH5MD
{
struct ProtocolSteeringDefinition
{
    std::vector<std::string> cv_refs;
    std::vector<float> weight;
    std::vector<std::pair<std::string, std::string>> runtime_parameters;
};

class ProtocolSteeringH5Reader
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

    bool Open(const std::shared_ptr<HighFive::File>& file)
    {
        last_error_.clear();
        if (file == nullptr) return Fail("protocol H5 file is null");
        file_ = file;
        return true;
    }

    bool Read_Definition(
        const std::vector<ProtocolCVDefinition>& cv_definitions,
        ProtocolSteeringDefinition* definition, bool* found)
    {
        if (definition == nullptr || found == nullptr)
        {
            return Fail("steering definition output pointer is null");
        }
        *definition = {};
        *found = false;
        if (file_ == nullptr) return Fail("protocol H5 reader is not open");
        if (!file_->exist("/steer")) return true;
        try
        {
            if (!Read_Enabled()) return true;
            const bool has_refs = file_->exist("/steer/cv_refs");
            const bool has_weight = file_->exist("/steer/weight");
            if (!has_refs && !has_weight) return true;
            if (!has_refs || !has_weight)
            {
                throw std::runtime_error(
                    "/steer requires both cv_refs and weight datasets");
            }
            definition->cv_refs = Read_String_Vector("/steer/cv_refs");
            if (definition->cv_refs.empty())
            {
                throw std::runtime_error(
                    "/steer/cv_refs must contain at least one CV");
            }
            std::set<std::string> unique_refs;
            for (const auto& cv_ref : definition->cv_refs)
            {
                if (cv_ref.empty() ||
                    cv_ref.find_first_of(" \t\r\n,{}=/") != std::string::npos)
                {
                    throw std::runtime_error(
                        "/steer/cv_refs contains an invalid CV name");
                }
                if (!unique_refs.insert(cv_ref).second)
                {
                    throw std::runtime_error(
                        "/steer/cv_refs contains a duplicate CV");
                }
                const auto cv =
                    std::find_if(cv_definitions.begin(), cv_definitions.end(),
                                 [&](const ProtocolCVDefinition& value)
                                 { return value.name == cv_ref; });
                if (cv == cv_definitions.end())
                {
                    throw std::runtime_error(
                        "/steer/cv_refs references missing or disabled /cv/" +
                        cv_ref);
                }
                if (cv->dimension != 1)
                {
                    throw std::runtime_error(
                        "/steer/cv_refs requires scalar CV " + cv_ref);
                }
            }
            definition->weight =
                Read_Float_Vector("/steer/weight", definition->cv_refs.size());
            definition->runtime_parameters = {
                {"CV", Join_Strings(definition->cv_refs)},
                {"weight", Join_Floats(definition->weight)}};
            *found = true;
            return true;
        }
        catch (const std::exception& error)
        {
            return Fail(
                std::string("failed to read typed steering definition: ") +
                error.what());
        }
    }

    const std::string& Last_Error() const { return last_error_; }

   private:
    bool Read_Enabled()
    {
        int enabled = 1;
        if (file_->exist("/steer/enabled_default"))
        {
            auto dataset = file_->getDataSet("/steer/enabled_default");
            if (!dataset.getSpace().getDimensions().empty())
            {
                throw std::runtime_error(
                    "/steer/enabled_default must be a scalar");
            }
            dataset.read(enabled);
        }
        else
        {
            auto group = file_->getGroup("/steer");
            if (group.hasAttribute("enabled_default"))
            {
                group.getAttribute("enabled_default").read(enabled);
            }
        }
        return enabled != 0;
    }

    std::vector<std::string> Read_String_Vector(const std::string& path)
    {
        auto dataset = file_->getDataSet(path);
        const auto dimensions = dataset.getSpace().getDimensions();
        if (dimensions.size() != 1)
        {
            throw std::runtime_error(path + " must have shape [n]");
        }
        std::vector<std::string> values;
        dataset.read(values);
        return values;
    }

    std::vector<float> Read_Float_Vector(const std::string& path,
                                         std::size_t count)
    {
        auto dataset = file_->getDataSet(path);
        if (dataset.getSpace().getDimensions() !=
            std::vector<std::size_t>{count})
        {
            throw std::runtime_error(path + " must match cv_refs length");
        }
        std::vector<float> values(count);
        if (H5Dread(dataset.getId(), H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                    H5P_DEFAULT, values.data()) < 0)
        {
            throw std::runtime_error("failed to read " + path);
        }
        for (float value : values)
        {
            if (!SpongeFloat::Is_Finite(value))
            {
                throw std::runtime_error(path + " contains a non-finite value");
            }
        }
        return values;
    }

    std::string Join_Strings(const std::vector<std::string>& values)
    {
        std::ostringstream out;
        for (std::size_t i = 0; i < values.size(); ++i)
        {
            if (i != 0) out << ' ';
            out << values[i];
        }
        return out.str();
    }

    std::string Join_Floats(const std::vector<float>& values)
    {
        std::ostringstream out;
        out << std::setprecision(std::numeric_limits<float>::max_digits10);
        for (std::size_t i = 0; i < values.size(); ++i)
        {
            if (i != 0) out << ' ';
            out << values[i];
        }
        return out.str();
    }

    bool Fail(const std::string& message)
    {
        last_error_ = message;
        return false;
    }

    std::shared_ptr<HighFive::File> file_;
    std::string last_error_;
};
}  // namespace SpongeH5MD

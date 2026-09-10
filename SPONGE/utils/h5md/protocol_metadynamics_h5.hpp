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
struct ProtocolMetadynamicsDefinition
{
    std::string name;
    std::vector<std::string> cv_refs;
    std::vector<std::pair<std::string, std::string>> runtime_parameters;
};

class ProtocolMetadynamicsH5Reader
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
        ProtocolMetadynamicsDefinition* definition, bool* found)
    {
        if (definition == nullptr || found == nullptr)
        {
            return Fail("metadynamics definition output pointer is null");
        }
        *definition = {};
        *found = false;
        if (file_ == nullptr) return Fail("protocol H5 reader is not open");
        if (!file_->exist("/meta")) return true;
        try
        {
            std::vector<std::string> names =
                file_->getGroup("/meta").listObjectNames();
            std::sort(names.begin(), names.end());
            for (const auto& name : names)
            {
                const std::string root = "/meta/" + name;
                if (name == "config" ||
                    file_->getObjectType(root) != HighFive::ObjectType::Group)
                {
                    continue;
                }
                // Older converter output may retain a restart-derived
                // /meta/<name>/grid snapshot without defining a protocol
                // bias object. cv_refs is the native object discriminator.
                if (!file_->exist(root + "/cv_refs")) continue;
                Validate_Name(name, root);
                if (!Read_Enabled(root)) continue;
                if (*found)
                {
                    throw std::runtime_error(
                        "/meta permits at most one enabled metadynamics "
                        "object in the current SPONGE runtime");
                }
                Read_One(root, name, cv_definitions, definition);
                *found = true;
            }
            return true;
        }
        catch (const std::exception& error)
        {
            return Fail(
                std::string("failed to read typed metadynamics definition: ") +
                error.what());
        }
    }

    const std::string& Last_Error() const { return last_error_; }

   private:
    void Read_One(const std::string& root, const std::string& name,
                  const std::vector<ProtocolCVDefinition>& cv_definitions,
                  ProtocolMetadynamicsDefinition* definition)
    {
        definition->name = name;
        definition->cv_refs = Read_String_Vector(root + "/cv_refs");
        if (definition->cv_refs.empty())
        {
            throw std::runtime_error(root +
                                     "/cv_refs must contain at least one CV");
        }
        std::set<std::string> unique_refs;
        std::vector<const ProtocolCVDefinition*> referenced_cvs;
        for (const auto& cv_ref : definition->cv_refs)
        {
            Validate_Reference(cv_ref, root + "/cv_refs");
            if (!unique_refs.insert(cv_ref).second)
            {
                throw std::runtime_error(root +
                                         "/cv_refs contains a duplicate CV");
            }
            const auto cv =
                std::find_if(cv_definitions.begin(), cv_definitions.end(),
                             [&](const ProtocolCVDefinition& value)
                             { return value.name == cv_ref; });
            if (cv == cv_definitions.end())
            {
                throw std::runtime_error(root +
                                         "/cv_refs references missing "
                                         "or disabled /cv/" +
                                         cv_ref);
            }
            if (cv->dimension != 1)
            {
                throw std::runtime_error(root + "/cv_refs requires scalar CV " +
                                         cv_ref);
            }
            referenced_cvs.push_back(&*cv);
        }

        const std::size_t ndim = definition->cv_refs.size();
        long long declared_ndim = static_cast<long long>(ndim);
        if (Read_Optional_Scalar(root, "ndim", &declared_ndim) &&
            declared_ndim != static_cast<long long>(ndim))
        {
            throw std::runtime_error(root +
                                     "/ndim must match the cv_refs length");
        }
        Add_Runtime_Parameter(definition, "CV",
                              Join_Values(definition->cv_refs));
        Add_Runtime_Parameter(definition, "Ndim",
                              std::to_string(declared_ndim));

        std::vector<float> sigma = Read_Optional_Float_Vector(
            root + "/sigma", ndim, "metadynamics sigma");
        if (sigma.empty())
        {
            for (const auto* cv : referenced_cvs)
            {
                if (cv->sigma.size() != 1)
                {
                    throw std::runtime_error(
                        root + "/sigma is missing and /cv/" + cv->name +
                        "/sigma cannot provide one scalar value");
                }
                sigma.push_back(cv->sigma[0]);
            }
        }
        for (float value : sigma)
        {
            if (!(value > 0.0f))
                throw std::runtime_error(root +
                                         "/sigma values must be positive");
        }
        Add_Runtime_Parameter(definition, "CV_sigma", Join_Values(sigma));

        std::vector<float> period = Read_Optional_Float_Vector(
            root + "/period", ndim, "metadynamics period");
        if (period.empty())
        {
            period.reserve(ndim);
            for (const auto* cv : referenced_cvs)
            {
                period.push_back(cv->period.size() == 1 ? cv->period[0] : 0.0f);
            }
        }
        Add_Runtime_Parameter(definition, "CV_period", Join_Values(period));

        const std::vector<float> grid_min = Read_Required_Float_Vector(
            root + "/grid/min", ndim, "metadynamics grid minimum");
        const std::vector<float> grid_max = Read_Required_Float_Vector(
            root + "/grid/max", ndim, "metadynamics grid maximum");
        const std::vector<long long> grid_count = Read_Required_Int_Vector(
            root + "/grid/count", ndim, "metadynamics grid count");
        for (std::size_t i = 0; i < ndim; ++i)
        {
            if (!(grid_max[i] > grid_min[i]))
                throw std::runtime_error(
                    root + "/grid/max must be greater than grid/min");
            if (grid_count[i] <= 1 ||
                grid_count[i] > std::numeric_limits<int>::max())
                throw std::runtime_error(
                    root + "/grid/count values must be in [2, INT_MAX]");
        }
        Add_Runtime_Parameter(definition, "CV_minimal", Join_Values(grid_min));
        Add_Runtime_Parameter(definition, "CV_maximum", Join_Values(grid_max));
        Add_Runtime_Parameter(definition, "CV_grid", Join_Values(grid_count));

        const std::vector<float> cutoff = Read_Optional_Float_Vector(
            root + "/cutoff", ndim, "metadynamics cutoff");
        if (!cutoff.empty())
        {
            for (float value : cutoff)
                if (!(value > 0.0f))
                    throw std::runtime_error(root +
                                             "/cutoff values must be positive");
            Add_Runtime_Parameter(definition, "cutoff", Join_Values(cutoff));
        }

        Read_Optional_Float_Default(root, "hill_height_default", "height",
                                    definition, false);
        Read_Optional_Int_Default(root, "sumhill_freq_default", "sumhill_freq",
                                  definition, 0);
        Read_Optional_Int_Default(root, "potential_update_interval_default",
                                  "potential_update_interval", definition, 1);
        Read_Optional_Float_Default(root, "well_tempered_factor_default",
                                    "welltemp_factor", definition, true, 1.0);
        Read_Optional_Float_Default(root, "wall_height_default", "wall_height",
                                    definition, false);
        Read_Optional_Float_Default(root, "max_force_default", "max_force",
                                    definition, false);
        Read_Method_Flags(root, definition);
    }

    void Read_Method_Flags(const std::string& root,
                           ProtocolMetadynamicsDefinition* definition)
    {
        const std::string flags_root = root + "/method_flags";
        if (!file_->exist(flags_root)) return;
        if (file_->getObjectType(flags_root) != HighFive::ObjectType::Group)
            throw std::runtime_error(flags_root + " must be a group");

        int subhill = 0;
        if (Read_Optional_Scalar(flags_root, "subhill", &subhill) && subhill)
            Add_Runtime_Parameter(definition, "subhill", "1");
        for (const char* flag : {"kde", "mask", "sink", "convmeta", "grw"})
        {
            long long value = 0;
            if (Read_Optional_Scalar(flags_root, flag, &value))
                Add_Runtime_Parameter(definition, flag, std::to_string(value));
        }
        double dip = 0.0;
        if (Read_Optional_Scalar(flags_root, "dip", &dip))
        {
            Require_Finite(dip, flags_root + "/dip");
            Add_Runtime_Parameter(definition, "dip", Format_Float(dip));
        }
    }

    void Read_Optional_Float_Default(const std::string& root, const char* field,
                                     const char* runtime_name,
                                     ProtocolMetadynamicsDefinition* definition,
                                     bool require_positive,
                                     double exclusive_minimum = 0.0)
    {
        double value = 0.0;
        if (!Read_Optional_Scalar(root, field, &value)) return;
        Require_Finite(value, root + "/" + field);
        if (require_positive && !(value > exclusive_minimum))
            throw std::runtime_error(root + "/" + field +
                                     " is outside its valid range");
        Add_Runtime_Parameter(definition, runtime_name, Format_Float(value));
    }

    void Read_Optional_Int_Default(const std::string& root, const char* field,
                                   const char* runtime_name,
                                   ProtocolMetadynamicsDefinition* definition,
                                   long long minimum)
    {
        long long value = 0;
        if (!Read_Optional_Scalar(root, field, &value)) return;
        if (value < minimum || value > std::numeric_limits<int>::max())
            throw std::runtime_error(root + "/" + field +
                                     " is outside its valid range");
        Add_Runtime_Parameter(definition, runtime_name, std::to_string(value));
    }

    std::vector<std::string> Read_String_Vector(const std::string& path)
    {
        if (!file_->exist(path))
            throw std::runtime_error("field is missing: " + path);
        auto dataset = file_->getDataSet(path);
        const auto dims = dataset.getSpace().getDimensions();
        std::vector<std::string> values;
        if (dims.empty())
        {
            std::string value;
            dataset.read(value);
            values.push_back(value);
        }
        else if (dims.size() == 1)
        {
            dataset.read(values);
        }
        else
        {
            throw std::runtime_error(path + " must be a scalar or 1D string");
        }
        return values;
    }

    std::vector<float> Read_Required_Float_Vector(const std::string& path,
                                                  std::size_t count,
                                                  const char* label)
    {
        const auto values = Read_Optional_Float_Vector(path, count, label);
        if (values.empty())
            throw std::runtime_error("field is missing: " + path);
        return values;
    }

    std::vector<float> Read_Optional_Float_Vector(const std::string& path,
                                                  std::size_t count,
                                                  const char* label)
    {
        if (!file_->exist(path)) return {};
        const auto dims = file_->getDataSet(path).getSpace().getDimensions();
        if (dims != std::vector<std::size_t>{count})
            throw std::runtime_error(path + " must have shape [ndim]");
        std::vector<float> values(count);
        auto dataset = file_->getDataSet(path);
        if (H5Dread(dataset.getId(), H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                    H5P_DEFAULT, values.data()) < 0)
            throw std::runtime_error(std::string("failed to read ") + label);
        for (float value : values) Require_Finite(value, path);
        return values;
    }

    std::vector<long long> Read_Required_Int_Vector(const std::string& path,
                                                    std::size_t count,
                                                    const char* label)
    {
        if (!file_->exist(path))
            throw std::runtime_error("field is missing: " + path);
        const auto dims = file_->getDataSet(path).getSpace().getDimensions();
        if (dims != std::vector<std::size_t>{count})
            throw std::runtime_error(path + " must have shape [ndim]");
        std::vector<long long> values(count);
        auto dataset = file_->getDataSet(path);
        if (H5Dread(dataset.getId(), H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL,
                    H5P_DEFAULT, values.data()) < 0)
            throw std::runtime_error(std::string("failed to read ") + label);
        return values;
    }

    bool Read_Enabled(const std::string& root)
    {
        int enabled = 1;
        Read_Optional_Scalar(root, "enabled_default", &enabled);
        return enabled != 0;
    }

    template <typename T>
    bool Read_Optional_Scalar(const std::string& root, const std::string& field,
                              T* value)
    {
        const std::string path = root + "/" + field;
        if (file_->exist(path))
        {
            if (!file_->getDataSet(path).getSpace().getDimensions().empty())
                throw std::runtime_error(path + " must be a scalar");
            file_->getDataSet(path).read(*value);
            return true;
        }
        auto group = file_->getGroup(root);
        if (group.hasAttribute(field))
        {
            group.getAttribute(field).read(*value);
            return true;
        }
        return false;
    }

    void Validate_Name(const std::string& name, const std::string& path)
    {
        if (name.empty() ||
            name.find_first_of(" \t\r\n,{}=/") != std::string::npos)
            throw std::runtime_error(
                path + " has an invalid metadynamics object name");
    }

    void Validate_Reference(const std::string& value, const std::string& path)
    {
        if (value.empty() ||
            value.find_first_of(" \t\r\n,{}=/") != std::string::npos)
            throw std::runtime_error(path +
                                     " contains an invalid CV reference");
    }

    void Add_Runtime_Parameter(ProtocolMetadynamicsDefinition* definition,
                               const std::string& name,
                               const std::string& value)
    {
        definition->runtime_parameters.push_back({name, value});
    }

    template <typename T>
    static std::string Join_Values(const std::vector<T>& values)
    {
        std::ostringstream output;
        output << std::setprecision(std::numeric_limits<double>::max_digits10);
        for (std::size_t i = 0; i < values.size(); ++i)
        {
            if (i != 0) output << ' ';
            output << values[i];
        }
        return output.str();
    }

    static std::string Format_Float(double value)
    {
        std::ostringstream output;
        output << std::setprecision(std::numeric_limits<double>::max_digits10)
               << value;
        return output.str();
    }

    static void Require_Finite(double value, const std::string& path)
    {
        if (!SpongeFloat::Is_Finite(value))
            throw std::runtime_error(path + " contains a non-finite value");
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

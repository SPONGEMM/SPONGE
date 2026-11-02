#pragma once

#include <hdf5.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <highfive/highfive.hpp>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../float_classification.hpp"

namespace SpongeH5MD
{
struct ProtocolCVRestraint
{
    std::string name;
    std::vector<std::string> cv_refs;
    std::vector<float> weight;
    std::vector<float> reference;
    std::vector<float> period;
    std::vector<std::int64_t> start_step;
    std::vector<std::int64_t> max_step;
    std::vector<std::int64_t> reduce_step;
    std::vector<std::int64_t> stop_step;
};

class ProtocolRestraintH5Reader
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

    bool Read_CV_Restraints(std::vector<ProtocolCVRestraint>* restraints)
    {
        if (restraints == nullptr)
        {
            return Fail("CV restraint output pointer is null");
        }
        restraints->clear();
        if (file_ == nullptr) return Fail("protocol H5 reader is not open");
        if (!file_->exist("/restraint")) return true;
        try
        {
            std::vector<std::string> names =
                file_->getGroup("/restraint").listObjectNames();
            std::sort(names.begin(), names.end());
            for (const auto& name : names)
            {
                const std::string root = "/restraint/" + name;
                const std::string refs_path = root + "/cv_refs";
                if (!file_->exist(refs_path)) continue;
                if (!Read_Optional_Enabled(root)) continue;
                const std::string type = Read_Required_String(root, "type");
                if (type != "cv_harmonic")
                {
                    return Fail(root +
                                "/type must be 'cv_harmonic' when cv_refs "
                                "is present");
                }
                ProtocolCVRestraint restraint;
                restraint.name = name;
                restraint.cv_refs = Read_String_Vector(refs_path);
                const std::size_t count = restraint.cv_refs.size();
                if (count == 0)
                {
                    return Fail(refs_path + " must not be empty");
                }
                for (const auto& cv_ref : restraint.cv_refs)
                {
                    if (cv_ref.empty() || cv_ref.find_first_of(" \t\r\n,{}=") !=
                                              std::string::npos)
                    {
                        return Fail(refs_path +
                                    " contains an invalid CV object name");
                    }
                }
                restraint.weight =
                    Read_Float_Vector(root + "/weight", count, true, 0.0f);
                restraint.reference =
                    Read_Float_Vector(root + "/reference", count, true, 0.0f);
                restraint.period =
                    Read_Float_Vector(root + "/period", count, false, 0.0f);
                restraint.start_step = Read_Int64_Vector(
                    root + "/schedule/start_step", count, false, 0);
                restraint.max_step = Read_Int64_Vector(
                    root + "/schedule/max_step", count, false, 0);
                restraint.reduce_step = Read_Int64_Vector(
                    root + "/schedule/reduce_step", count, false, 0);
                restraint.stop_step = Read_Int64_Vector(
                    root + "/schedule/stop_step", count, false, 0);
                restraints->push_back(std::move(restraint));
            }
            return true;
        }
        catch (const std::exception& error)
        {
            return Fail(std::string("failed to read typed CV restraints: ") +
                        error.what());
        }
    }

    const std::string& Last_Error() const { return last_error_; }

   private:
    std::string Read_Required_String(const std::string& root,
                                     const std::string& field)
    {
        const std::string path = root + "/" + field;
        std::string value;
        if (file_->exist(path))
        {
            file_->getDataSet(path).read(value);
            return value;
        }
        auto group = file_->getGroup(root);
        if (group.hasAttribute(field))
        {
            group.getAttribute(field).read(value);
            return value;
        }
        throw std::runtime_error("field is missing: " + path);
    }

    bool Read_Optional_Enabled(const std::string& root)
    {
        constexpr const char* field = "enabled_default";
        const std::string path = root + "/" + field;
        int enabled = 1;
        if (file_->exist(path))
        {
            file_->getDataSet(path).read(enabled);
            return enabled != 0;
        }
        auto group = file_->getGroup(root);
        if (group.hasAttribute(field))
        {
            group.getAttribute(field).read(enabled);
        }
        return enabled != 0;
    }

    std::vector<std::string> Read_String_Vector(const std::string& path)
    {
        const auto dims = file_->getDataSet(path).getSpace().getDimensions();
        if (dims.size() != 1)
            throw std::runtime_error(path + " must have shape [n]");
        std::vector<std::string> values;
        file_->getDataSet(path).read(values);
        return values;
    }

    std::vector<float> Read_Float_Vector(const std::string& path,
                                         std::size_t count, bool required,
                                         float default_value)
    {
        if (!file_->exist(path))
        {
            if (required)
                throw std::runtime_error("dataset is missing: " + path);
            return std::vector<float>(count, default_value);
        }
        const auto dims = file_->getDataSet(path).getSpace().getDimensions();
        if (dims != std::vector<std::size_t>{count})
            throw std::runtime_error(path + " must match cv_refs length");
        std::vector<float> values(count);
        auto dataset = file_->getDataSet(path);
        if (H5Dread(dataset.getId(), H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                    H5P_DEFAULT, values.data()) < 0)
            throw std::runtime_error("failed to read " + path);
        for (float value : values)
            if (!SpongeFloat::Is_Finite(value))
                throw std::runtime_error(path + " contains a non-finite value");
        return values;
    }

    std::vector<std::int64_t> Read_Int64_Vector(const std::string& path,
                                                std::size_t count,
                                                bool required,
                                                std::int64_t default_value)
    {
        if (!file_->exist(path))
        {
            if (required)
                throw std::runtime_error("dataset is missing: " + path);
            return std::vector<std::int64_t>(count, default_value);
        }
        const auto dims = file_->getDataSet(path).getSpace().getDimensions();
        if (dims != std::vector<std::size_t>{count})
            throw std::runtime_error(path + " must match cv_refs length");
        std::vector<std::int64_t> values(count);
        auto dataset = file_->getDataSet(path);
        if (H5Dread(dataset.getId(), H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL,
                    H5P_DEFAULT, values.data()) < 0)
            throw std::runtime_error("failed to read " + path);
        for (std::int64_t value : values)
        {
            if (value < 0 || value > static_cast<std::int64_t>(
                                         std::numeric_limits<int>::max()))
            {
                throw std::runtime_error(
                    path + " contains a step outside the runtime int range");
            }
        }
        return values;
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

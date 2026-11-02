#pragma once

#include <cstdint>
#include <highfive/highfive.hpp>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace SpongeH5MD
{
struct NativeSoftWallDefinition
{
    std::string name;
    std::string potential;
};

class NativeSoftWallH5Reader
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

    bool Has_Soft_Wall() const
    {
        return file_ != nullptr && file_->exist("/wall/soft");
    }

    bool Read(std::vector<NativeSoftWallDefinition>* definitions)
    {
        if (file_ == nullptr)
        {
            return Fail("native soft-wall protocol H5 reader is not open");
        }
        if (definitions == nullptr)
        {
            return Fail("native soft-wall definition output is null");
        }

        try
        {
            const std::int64_t count = Read_Scalar<std::int64_t>(
                "/wall/soft/count", "soft-wall count");
            if (count <= 0)
            {
                return Fail("/wall/soft/count must be greater than zero");
            }
            const auto expected = static_cast<std::size_t>(count);
            const auto names = Read_Text_Vector("/wall/soft/name", expected);
            const auto potentials =
                Read_Text_Vector("/wall/soft/potential", expected);
            Validate_Entries(names, potentials);
            definitions->clear();
            definitions->reserve(expected);
            for (std::size_t index = 0; index < expected; ++index)
            {
                definitions->push_back({names[index], potentials[index]});
            }
            return true;
        }
        catch (const std::exception& error)
        {
            return Fail(std::string("failed to read native soft wall: ") +
                        error.what());
        }
    }

    const std::string& Last_Error() const { return last_error_; }

   private:
    template <typename T>
    T Read_Scalar(const std::string& path, const std::string& label) const
    {
        if (!file_->exist(path))
        {
            throw std::runtime_error("dataset is missing: " + path);
        }
        const auto dimensions =
            file_->getDataSet(path).getSpace().getDimensions();
        if (!dimensions.empty())
        {
            throw std::runtime_error(label + " dataset " + path +
                                     " must be scalar");
        }
        T value{};
        file_->getDataSet(path).read(value);
        return value;
    }

    std::vector<std::string> Read_Text_Vector(const std::string& path,
                                              std::size_t expected) const
    {
        if (!file_->exist(path))
        {
            throw std::runtime_error("dataset is missing: " + path);
        }
        const auto dimensions =
            file_->getDataSet(path).getSpace().getDimensions();
        if (dimensions != std::vector<std::size_t>{expected})
        {
            std::ostringstream message;
            message << path << " must have shape [" << expected << ']';
            throw std::runtime_error(message.str());
        }
        std::vector<std::string> values;
        file_->getDataSet(path).read(values);
        return values;
    }

    static void Validate_Entries(const std::vector<std::string>& names,
                                 const std::vector<std::string>& potentials)
    {
        constexpr std::size_t module_name_capacity = 512;
        std::set<std::string> unique_names;
        for (std::size_t index = 0; index < names.size(); ++index)
        {
            const std::string& name = names[index];
            if (name.empty())
            {
                throw std::runtime_error(
                    "/wall/soft/name contains an empty name");
            }
            if (name.size() >= module_name_capacity)
            {
                throw std::runtime_error(
                    "/wall/soft/name exceeds the soft-wall module-name "
                    "capacity");
            }
            if (name.find_first_of("[]\r\n\t ") != std::string::npos)
            {
                throw std::runtime_error(
                    "/wall/soft/name contains a configuration delimiter");
            }
            if (!unique_names.insert(name).second)
            {
                throw std::runtime_error(
                    "/wall/soft/name contains a duplicate name");
            }

            const std::string& potential = potentials[index];
            if (potential.empty())
            {
                throw std::runtime_error(
                    "/wall/soft/potential contains an empty potential");
            }
            if (potential.find('\0') != std::string::npos)
            {
                throw std::runtime_error(
                    "/wall/soft/potential contains an embedded null byte");
            }
            std::istringstream lines(potential);
            std::string line;
            while (std::getline(lines, line))
            {
                const auto first = line.find_first_not_of(" \t");
                if (first != std::string::npos &&
                    line.compare(first, 2, "[[") == 0)
                {
                    throw std::runtime_error(
                        "/wall/soft/potential contains a configuration "
                        "delimiter");
                }
            }
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

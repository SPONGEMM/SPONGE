#pragma once

#include <cstdint>
#include <highfive/highfive.hpp>
#include <memory>
#include <sstream>
#include <string>

#include "utils/h5md/h5_input_metadata.hpp"
#include "utils/h5md/h5md_writer.hpp"

namespace SpongeH5MD
{
class TopologyH5Reader
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

    bool Read_Metadata(SpongeH5InputMetadata::TopologyMetadata* metadata)
    {
        if (metadata == nullptr)
        {
            return Fail("topology metadata output pointer is null");
        }
        if (!Ensure_File()) return false;

        SpongeH5InputMetadata::TopologyMetadata result;
        try
        {
            result.schema_name = Read_Optional_String("/schema/name");
            if (result.schema_name.empty())
            {
                result.schema_name =
                    Read_Optional_String(path::sponge_schema_name);
            }
            result.schema_version = Read_Optional_String("/schema/version");
            if (result.schema_version.empty())
            {
                result.schema_version =
                    Read_Optional_String(path::sponge_schema_version);
            }
            result.identity_uuid = Read_Optional_String(path::identity_uuid);
            result.atom_count = Read_Optional_Int64("/topology/atom_count");
            result.atom_ordering_hash =
                Read_Optional_String("/topology/atom_order_hash");
            result.topology_hash =
                Read_Optional_String("/topology/topology_hash");
            result.force_field_hash =
                Read_Optional_String("/topology/forcefield_hash");
            *metadata = result;
            return true;
        }
        catch (const std::exception& err)
        {
            return Fail(std::string("failed to read topology metadata: ") +
                        err.what());
        }
    }

    std::string Last_Error() const { return last_error_; }

   private:
    bool Ensure_File()
    {
        if (file_ == nullptr)
        {
            return Fail("topology H5 reader is not open");
        }
        return true;
    }

    bool Exists(const std::string& object_path) const
    {
        return file_ != nullptr && file_->exist(object_path);
    }

    std::string Read_Optional_String(const std::string& dataset_path)
    {
        if (!Exists(dataset_path))
        {
            return "";
        }
        try
        {
            std::string value;
            file_->getDataSet(dataset_path).read(value);
            return value;
        }
        catch (const std::exception&)
        {
            const auto value = Read_Optional_Int64(dataset_path);
            return value == 0 ? "" : std::to_string(value);
        }
    }

    std::int64_t Read_Optional_Int64(const std::string& dataset_path)
    {
        if (!Exists(dataset_path))
        {
            return 0;
        }
        std::int64_t value = 0;
        file_->getDataSet(dataset_path).read(value);
        return value;
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

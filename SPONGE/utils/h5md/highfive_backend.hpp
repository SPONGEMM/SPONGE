#pragma once

#include <hdf5.h>

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <highfive/highfive.hpp>
#include <limits>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include "utils/h5md/atomic_file_publish.hpp"
#include "utils/h5md/h5md_writer.hpp"

#if H5_VERSION_GE(1, 10, 7) && \
    !(H5_VERS_MAJOR == 1 && H5_VERS_MINOR == 12 && H5_VERS_RELEASE == 0)
#define SPONGE_H5_HAS_FILE_LOCKING_API 1
#else
#define SPONGE_H5_HAS_FILE_LOCKING_API 0
#endif

namespace SpongeH5MD
{
inline bool HDF5_File_Locking_Is_Forced_On_By_Environment()
{
    const char* value = std::getenv("HDF5_USE_FILE_LOCKING");
    if (value == nullptr) return false;
    std::string normalized(value);
    for (char& character : normalized)
    {
        character = static_cast<char>(
            std::toupper(static_cast<unsigned char>(character)));
    }
    return normalized == "TRUE" || normalized == "1" ||
           normalized == "BEST_EFFORT";
}

inline std::string Snapshot_Recovery_Path(const std::string& destination_path,
                                          const std::string& identity_uuid)
{
    std::string safe_identity = identity_uuid;
    for (char& character : safe_identity)
    {
        const unsigned char value = static_cast<unsigned char>(character);
        if (!std::isalnum(value) && character != '-' && character != '_')
        {
            character = '_';
        }
    }
    if (safe_identity.empty()) safe_identity = "unknown";
    return destination_path + ".pending." + safe_identity;
}

class HighFiveBackend : public WriterBackend
{
   public:
    bool Open(const WriterOptions& options) override
    {
        options_ = options;
        last_error_.clear();
        swmr_write_started_ = false;
#ifdef SPONGE_H5_TEST_FAULT_INJECTION
        test_fault_consumed_ = false;
#endif
        try
        {
            actual_path_ = options.atomic_snapshot ? options.path + ".working"
                                                   : options.path;
            const std::filesystem::path file_path(actual_path_);
            const std::filesystem::path parent = file_path.parent_path();
            if (!parent.empty())
            {
                std::filesystem::create_directories(parent);
            }
            bool use_custom_access_props = options.swmr_compatible;
#if defined(_WIN32) && SPONGE_H5_HAS_FILE_LOCKING_API
            use_custom_access_props =
                use_custom_access_props || options.atomic_snapshot;
#elif defined(_WIN32)
            if (options.atomic_snapshot)
            {
                return Fail(
                    "Windows atomic snapshots require HDF5 "
                    "1.10.7-1.10.x or 1.12.1+; HDF5 1.12.0 lacks "
                    "H5Pset_file_locking");
            }
#endif
            if (use_custom_access_props)
            {
                HighFive::FileAccessProps access_props =
                    HighFive::FileAccessProps::Empty();
#if defined(_WIN32) && SPONGE_H5_HAS_FILE_LOCKING_API
                if (options.atomic_snapshot &&
                    HDF5_File_Locking_Is_Forced_On_By_Environment())
                {
                    return Fail(
                        "HDF5_USE_FILE_LOCKING forces file locking on, which "
                        "prevents Windows atomic snapshot publication; unset "
                        "it or set it to FALSE");
                }
                if (options.atomic_snapshot &&
                    H5Pset_file_locking(access_props.getId(), false, true) < 0)
                {
                    return Fail(
                        "failed to disable HDF5 locking for an atomic "
                        "snapshot");
                }
#endif
                if (options.swmr_compatible)
                {
                    access_props.add(HighFive::FileVersionBounds(
                        H5F_LIBVER_LATEST, H5F_LIBVER_LATEST));
                }
                file_.reset(new HighFive::File(
                    actual_path_, HighFive::File::Overwrite,
                    HighFive::FileCreateProps::Default(), access_props));
            }
            else
            {
                file_.reset(new HighFive::File(actual_path_,
                                               HighFive::File::Overwrite));
            }
            status_ = FileStatus::open;
            return true;
        }
        catch (const std::exception& err)
        {
            return Fail(std::string("failed to open HDF5 file: ") + err.what());
        }
    }

    bool Start_Swmr_Write() override
    {
        if (!Ensure_File()) return false;
        if (swmr_write_started_) return true;
        if (!options_.swmr_compatible)
        {
            return Fail("HDF5 file was not opened with SWMR compatibility");
        }
        try
        {
            file_->flush();
            file_->startSWMRWrite();
            swmr_write_started_ = true;
            return true;
        }
        catch (const std::exception& err)
        {
            return Fail(std::string("failed to start HDF5 SWMR write: ") +
                        err.what());
        }
    }

    bool Flush() override
    {
        if (!Ensure_File()) return false;
        try
        {
            file_->flush();
            return true;
        }
        catch (const std::exception& err)
        {
            return Fail(std::string("failed to flush HDF5 file: ") +
                        err.what());
        }
    }

    bool Close() override
    {
        if (file_ == nullptr)
        {
            status_ = FileStatus::closed;
            return true;
        }
        try
        {
            file_->flush();
            file_.reset();
            dataset_specs_.clear();
            swmr_write_started_ = false;
            status_ = FileStatus::closed;
            return true;
        }
        catch (const std::exception& err)
        {
            return Fail(std::string("failed to close HDF5 file: ") +
                        err.what());
        }
    }

    bool Finalize() override
    {
        if (!Ensure_File()) return false;
#ifdef SPONGE_H5_TEST_FAULT_INJECTION
        if (Test_Fault_Matches("finalize"))
        {
            return Inject_Test_Failure("finalize", "");
        }
#endif
        if (!Write_String(path::output_status, "finalized")) return false;
        status_ = FileStatus::finalized;
        return Flush();
    }

    bool Publish_Snapshot(const std::string& destination_path,
                          bool allow_deferred = false) override
    {
        if (!Ensure_File() || !Flush()) return false;
        const std::string temporary_path = destination_path + ".tmp";
        const std::string pending_path =
            Snapshot_Recovery_Path(destination_path, options_.identity_uuid);
        std::string error;
        if (!Remove_File_If_Exists(temporary_path, &error))
        {
            return Fail(error);
        }
        std::error_code copy_error;
        std::filesystem::copy_file(
            actual_path_, temporary_path,
            std::filesystem::copy_options::overwrite_existing, copy_error);
        if (copy_error)
        {
            return Fail("failed to snapshot HDF5 file " + actual_path_ +
                        " to " + temporary_path + ": " + copy_error.message());
        }
        bool destination_busy = false;
        if (!Atomic_Replace_File(temporary_path, destination_path, &error,
                                 &destination_busy, !allow_deferred))
        {
            if (allow_deferred && destination_busy)
            {
                if (!Remove_File_If_Exists(temporary_path, &error))
                {
                    return Fail(error);
                }
                return true;
            }
            if (destination_busy)
            {
                std::string pending_error;
                if (Atomic_Replace_File(temporary_path, pending_path,
                                        &pending_error))
                {
                    return Fail(error + "; latest snapshot retained at " +
                                pending_path);
                }
                return Fail(error +
                            "; additionally failed to retain latest "
                            "snapshot at " +
                            pending_path + ": " + pending_error);
            }
            Remove_File_If_Exists(temporary_path);
            return Fail(error);
        }
        Remove_File_If_Exists(pending_path);
        return true;
    }

    bool Ensure_Group(const std::string& group_path) override
    {
        if (!Ensure_File()) return false;
        if (group_path.empty() || group_path == "/") return true;
        try
        {
            std::string current;
            for (const std::string& component : Split_Path(group_path))
            {
                current += "/" + component;
                if (!file_->exist(current))
                {
                    if (swmr_write_started_)
                    {
                        return Fail(
                            "cannot create group after SWMR write "
                            "started: " +
                            current);
                    }
                    file_->createGroup(current);
                }
            }
            return true;
        }
        catch (const std::exception& err)
        {
            return Fail(std::string("failed to ensure group ") + group_path +
                        ": " + err.what());
        }
    }

    bool Create_Dataset(const DatasetSpec& spec) override
    {
        if (!Ensure_File()) return false;
        if (spec.path.empty() || spec.shape.dims.empty())
        {
            return Fail("dataset path and dimensions must be non-empty");
        }
        try
        {
            if (!Ensure_Parent_Group(spec.path)) return false;
            if (file_->exist(spec.path))
            {
                dataset_specs_[spec.path] = Normalize_Spec(spec);
                return true;
            }
            if (swmr_write_started_)
            {
                return Fail("cannot create dataset after SWMR write started: " +
                            spec.path);
            }
            const DatasetSpec normalized = Normalize_Spec(spec);
            // DataSpace takes size_t dimensions.  Chunking takes hsize_t
            // dimensions, so keep the two representations separate on macOS.
            const std::vector<std::size_t>& dims = normalized.shape.dims;
            const std::vector<std::size_t> max_dims = To_Size_Max(normalized);
            const std::vector<hsize_t> chunk_dims =
                To_HSize(normalized.shape.chunk_dims);
            HighFive::DataSpace space(dims, max_dims);
            HighFive::DataSetCreateProps props;
            props.add(HighFive::Chunking(chunk_dims));
            switch (normalized.type)
            {
                case DataType::int64:
                    file_->createDataSet<int64_t>(spec.path, space, props);
                    break;
                case DataType::float32:
                    file_->createDataSet<float>(spec.path, space, props);
                    break;
                case DataType::float64:
                    file_->createDataSet<double>(spec.path, space, props);
                    break;
                case DataType::string:
                    file_->createDataSet<std::string>(spec.path, space, props);
                    break;
            }
            dataset_specs_[spec.path] = normalized;
            return true;
        }
        catch (const std::exception& err)
        {
            return Fail(std::string("failed to create dataset ") + spec.path +
                        ": " + err.what());
        }
    }

    bool Create_Virtual_Dataset(
        const DatasetSpec& spec,
        const std::vector<VirtualDatasetSource>& sources) override
    {
        if (!Ensure_File()) return false;
        if (spec.path.empty() || spec.shape.dims.empty())
        {
            return Fail(
                "virtual dataset path and dimensions must be non-empty");
        }
        if (sources.empty())
        {
            return Fail("virtual dataset sources must be non-empty: " +
                        spec.path);
        }
        if (spec.type == DataType::string)
        {
            return Fail("string virtual datasets are not supported: " +
                        spec.path);
        }
        try
        {
            if (swmr_write_started_)
            {
                return Fail(
                    "cannot create virtual dataset after SWMR write "
                    "started: " +
                    spec.path);
            }
            if (!Ensure_Parent_Group(spec.path)) return false;
            Delete_If_Exists(spec.path);

            const DatasetSpec normalized = Normalize_Spec(spec);
            const std::vector<hsize_t> dims = To_HSize(normalized.shape.dims);
            hid_t virtual_space = H5Screate_simple(
                static_cast<int>(dims.size()), dims.data(), nullptr);
            if (virtual_space < 0)
            {
                return Fail("failed to create virtual dataspace: " + spec.path);
            }

            hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
            if (dcpl < 0)
            {
                H5Sclose(virtual_space);
                return Fail("failed to create virtual dataset property list: " +
                            spec.path);
            }

            for (const auto& source : sources)
            {
                if (source.source_dims.size() != normalized.shape.dims.size() ||
                    source.virtual_start.size() != normalized.shape.dims.size())
                {
                    H5Pclose(dcpl);
                    H5Sclose(virtual_space);
                    return Fail("virtual dataset source rank mismatch: " +
                                spec.path);
                }
                const std::vector<hsize_t> source_dims =
                    To_HSize(source.source_dims);
                const std::vector<hsize_t> virtual_start =
                    To_HSize(source.virtual_start);
                hid_t source_space =
                    H5Screate_simple(static_cast<int>(source_dims.size()),
                                     source_dims.data(), nullptr);
                if (source_space < 0)
                {
                    H5Pclose(dcpl);
                    H5Sclose(virtual_space);
                    return Fail("failed to create source dataspace for VDS: " +
                                spec.path);
                }
                const herr_t select_rc = H5Sselect_hyperslab(
                    virtual_space, H5S_SELECT_SET, virtual_start.data(),
                    nullptr, source_dims.data(), nullptr);
                if (select_rc < 0)
                {
                    H5Sclose(source_space);
                    H5Pclose(dcpl);
                    H5Sclose(virtual_space);
                    return Fail("failed to select VDS hyperslab: " + spec.path);
                }
                const herr_t virtual_rc = H5Pset_virtual(
                    dcpl, virtual_space, source.file_path.c_str(),
                    source.dataset_path.c_str(), source_space);
                H5Sclose(source_space);
                if (virtual_rc < 0)
                {
                    H5Pclose(dcpl);
                    H5Sclose(virtual_space);
                    return Fail("failed to map VDS source " + source.file_path +
                                ":" + source.dataset_path + " into " +
                                spec.path);
                }
            }

            const hid_t dataset = H5Dcreate2(
                file_->getId(), spec.path.c_str(), H5_Type(normalized.type),
                virtual_space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
            H5Pclose(dcpl);
            H5Sclose(virtual_space);
            if (dataset < 0)
            {
                return Fail("failed to create virtual dataset: " + spec.path);
            }
            H5Dclose(dataset);
            dataset_specs_[spec.path] = normalized;
            return true;
        }
        catch (const std::exception& err)
        {
            return Fail(std::string("failed to create virtual dataset ") +
                        spec.path + ": " + err.what());
        }
    }

    bool Create_Hard_Link(const std::string& target,
                          const std::string& link_path) override
    {
        if (!Ensure_File()) return false;
        if (target == link_path) return true;
        try
        {
            if (!Ensure_Parent_Group(link_path)) return false;
            if (!file_->exist(target))
            {
                return Fail("hard-link target does not exist: " + target);
            }
            if (file_->exist(link_path))
            {
                if (swmr_write_started_) return true;
                if (H5Ldelete(file_->getId(), link_path.c_str(), H5P_DEFAULT) <
                    0)
                {
                    return Fail("failed to replace hard link: " + link_path);
                }
            }
            if (swmr_write_started_)
            {
                return Fail(
                    "cannot create hard link after SWMR write started: " +
                    link_path);
            }
            const herr_t rc =
                H5Lcreate_hard(file_->getId(), target.c_str(), file_->getId(),
                               link_path.c_str(), H5P_DEFAULT, H5P_DEFAULT);
            if (rc < 0)
            {
                return Fail("failed to create hard link " + link_path + " -> " +
                            target);
            }
            return true;
        }
        catch (const std::exception& err)
        {
            return Fail(std::string("failed to create hard link ") + link_path +
                        ": " + err.what());
        }
    }

    bool Append_Int64(const std::string& dataset_path, const int64_t* data,
                      std::size_t count) override
    {
        return Append_Data(dataset_path, data, count);
    }

    bool Append_Float32(const std::string& dataset_path, const float* data,
                        std::size_t count) override
    {
        return Append_Data(dataset_path, data, count);
    }

    bool Append_Float64(const std::string& dataset_path, const double* data,
                        std::size_t count) override
    {
        return Append_Data(dataset_path, data, count);
    }

    bool Write_Float32(const std::string& dataset_path, const float* data,
                       std::size_t count) override
    {
        if (!Ensure_File()) return false;
        if (data == nullptr || count == 0)
        {
            return Fail("float32 snapshot data is empty: " + dataset_path);
        }
        try
        {
            if (!Ensure_Parent_Group(dataset_path)) return false;
            if (!file_->exist(dataset_path))
            {
                if (swmr_write_started_)
                {
                    return Fail(
                        "cannot create float32 snapshot dataset after SWMR "
                        "write started: " +
                        dataset_path);
                }
                DatasetSpec spec;
                spec.path = dataset_path;
                spec.type = DataType::float32;
                spec.shape = {{count}, {count}, {count}};
                spec.appendable = false;
                if (!Create_Dataset(spec)) return false;
            }
            HighFive::DataSet dataset = file_->getDataSet(dataset_path);
            const auto dimensions = dataset.getSpace().getDimensions();
            if (dimensions != std::vector<std::size_t>{count})
            {
                return Fail("float32 snapshot dataset shape mismatch: " +
                            dataset_path);
            }
            const std::vector<float> values(data, data + count);
            dataset.write(values);
            return true;
        }
        catch (const std::exception& err)
        {
            return Fail(std::string("failed to write float32 snapshot ") +
                        dataset_path + ": " + err.what());
        }
    }

    bool Write_String(const std::string& dataset_path,
                      const std::string& value) override
    {
        if (!Ensure_File()) return false;
        try
        {
            if (!Ensure_Parent_Group(dataset_path)) return false;
            if (file_->exist(dataset_path))
            {
                file_->getDataSet(dataset_path).write(value);
                return true;
            }
            if (swmr_write_started_)
            {
                return Fail(
                    "cannot create string dataset after SWMR write "
                    "started: " +
                    dataset_path);
            }
            HighFive::DataSet dataset = file_->createDataSet<std::string>(
                dataset_path, HighFive::DataSpace::From(value));
            dataset.write(value);
            return true;
        }
        catch (const std::exception& err)
        {
            return Fail(std::string("failed to write string dataset ") +
                        dataset_path + ": " + err.what());
        }
    }

    bool Write_String_Array(const std::string& dataset_path,
                            const std::vector<std::string>& values) override
    {
        if (!Ensure_File()) return false;
        try
        {
            if (!Ensure_Parent_Group(dataset_path)) return false;
            if (file_->exist(dataset_path))
            {
                HighFive::DataSet dataset = file_->getDataSet(dataset_path);
                if (dataset.getSpace().getDimensions() ==
                    std::vector<std::size_t>{values.size()})
                {
                    dataset.write(values);
                    return true;
                }
                if (swmr_write_started_)
                {
                    return Fail(
                        "cannot resize string-array dataset after SWMR "
                        "write started: " +
                        dataset_path);
                }
            }
            else if (swmr_write_started_)
            {
                return Fail(
                    "cannot create string-array dataset after SWMR "
                    "write started: " +
                    dataset_path);
            }
            Delete_If_Exists(dataset_path);
            HighFive::DataSpace space({values.size()});
            HighFive::DataSet dataset =
                file_->createDataSet<std::string>(dataset_path, space);
            dataset.write(values);
            return true;
        }
        catch (const std::exception& err)
        {
            return Fail(std::string("failed to write string-array dataset ") +
                        dataset_path + ": " + err.what());
        }
    }

    bool Set_String_Attribute(const std::string& object_path,
                              const std::string& name,
                              const std::string& value) override
    {
        if (!Ensure_File()) return false;
        try
        {
            if (!file_->exist(object_path))
            {
                return Fail("failed to set attribute on missing object " +
                            object_path);
            }
            if (swmr_write_started_)
            {
                return Fail(
                    "cannot modify attributes after SWMR write started: " +
                    object_path + "@" + name);
            }
            const hid_t object_id =
                H5Oopen(file_->getId(), object_path.c_str(), H5P_DEFAULT);
            if (object_id < 0)
            {
                return Fail("failed to inspect attribute target " +
                            object_path);
            }
            const H5I_type_t object_type = H5Iget_type(object_id);
            H5Oclose(object_id);
            if (object_type == H5I_DATASET)
            {
                HighFive::DataSet dataset = file_->getDataSet(object_path);
                if (dataset.hasAttribute(name)) dataset.deleteAttribute(name);
                auto attribute = dataset.createAttribute<std::string>(
                    name, HighFive::DataSpace::From(value));
                attribute.write(value);
            }
            else if (object_type == H5I_GROUP)
            {
                HighFive::Group group = file_->getGroup(object_path);
                if (group.hasAttribute(name)) group.deleteAttribute(name);
                auto attribute = group.createAttribute<std::string>(
                    name, HighFive::DataSpace::From(value));
                attribute.write(value);
            }
            else
            {
                return Fail("unsupported attribute target " + object_path);
            }
            return true;
        }
        catch (const std::exception& err)
        {
            return Fail(std::string("failed to set string attribute ") + name +
                        " on " + object_path + ": " + err.what());
        }
    }

    bool Set_Status(FileStatus status) override
    {
        status_ = status;
        return Write_String(path::output_status, Status_String(status));
    }

    FileStatus Status() const override { return status_; }
    std::string Last_Error() const override { return last_error_; }

   private:
    static hid_t H5_Type(const DataType type)
    {
        switch (type)
        {
            case DataType::int64:
                return H5T_NATIVE_LLONG;
            case DataType::float32:
                return H5T_NATIVE_FLOAT;
            case DataType::float64:
                return H5T_NATIVE_DOUBLE;
            case DataType::string:
                return H5T_C_S1;
        }
        return H5T_NATIVE_DOUBLE;
    }

    template <typename T>
    bool Append_Data(const std::string& dataset_path, const T* data,
                     std::size_t count)
    {
        if (!Ensure_File()) return false;
#ifdef SPONGE_H5_TEST_FAULT_INJECTION
        if (dataset_path.rfind("/parameters/sponge/output/", 0) != 0 &&
            Test_Fault_Matches("append"))
        {
            return Inject_Test_Failure("append", dataset_path);
        }
#endif
        if (data == nullptr)
        {
            return Fail("append data pointer is null for dataset " +
                        dataset_path);
        }
        try
        {
            const auto spec_iter = dataset_specs_.find(dataset_path);
            if (spec_iter == dataset_specs_.end() ||
                !file_->exist(dataset_path))
            {
                return Fail("dataset was not created before append: " +
                            dataset_path);
            }
            HighFive::DataSet dataset = file_->getDataSet(dataset_path);
            std::vector<std::size_t> dims = dataset.getSpace().getDimensions();
            if (dims.empty())
            {
                return Fail("append target has scalar dataspace: " +
                            dataset_path);
            }
            const std::size_t record_size = Record_Size(dims);
            std::size_t append_records = 1;
            std::size_t expected_count = record_size;
            const bool fixed_first_dim_bulk_append =
                spec_iter->second.appendable &&
                !spec_iter->second.shape.max_dims.empty() &&
                spec_iter->second.shape.max_dims[0] != 0;
            if (fixed_first_dim_bulk_append && record_size > 0 &&
                count % record_size == 0)
            {
                append_records = count / record_size;
                expected_count = count;
            }
            if (count != expected_count || append_records == 0)
            {
                return Fail(
                    "append count does not match one dataset record for " +
                    dataset_path);
            }
            std::vector<std::size_t> new_dims = dims;
            new_dims[0] += append_records;
            dataset.resize(new_dims);
            std::vector<std::size_t> offset(dims.size(), 0);
            offset[0] = dims[0];
            std::vector<std::size_t> selection_count = new_dims;
            selection_count[0] = append_records;
            for (std::size_t i = 1; i < selection_count.size(); ++i)
            {
                selection_count[i] = dims[i];
            }
            const std::vector<hsize_t> h_offset = To_HSize(offset);
            const std::vector<hsize_t> h_count = To_HSize(selection_count);
            hid_t file_space = H5Dget_space(dataset.getId());
            if (file_space < 0)
            {
                return Fail("failed to get append dataspace: " + dataset_path);
            }
            const herr_t select_rc =
                H5Sselect_hyperslab(file_space, H5S_SELECT_SET, h_offset.data(),
                                    nullptr, h_count.data(), nullptr);
            if (select_rc < 0)
            {
                H5Sclose(file_space);
                return Fail("failed to select append hyperslab: " +
                            dataset_path);
            }
            hid_t mem_space = H5Screate_simple(static_cast<int>(h_count.size()),
                                               h_count.data(), nullptr);
            if (mem_space < 0)
            {
                H5Sclose(file_space);
                return Fail("failed to create append memory dataspace: " +
                            dataset_path);
            }
            const herr_t write_rc =
                H5Dwrite(dataset.getId(), H5_Type(spec_iter->second.type),
                         mem_space, file_space, H5P_DEFAULT, data);
            H5Sclose(mem_space);
            H5Sclose(file_space);
            if (write_rc < 0)
            {
                return Fail("failed to write append hyperslab: " +
                            dataset_path);
            }
            return true;
        }
        catch (const std::exception& err)
        {
            return Fail(std::string("failed to append dataset ") +
                        dataset_path + ": " + err.what());
        }
    }

    bool Ensure_File()
    {
        if (file_ != nullptr) return true;
        return Fail("HDF5 file is not open");
    }

    bool Ensure_Parent_Group(const std::string& object_path)
    {
        const std::size_t pos = object_path.find_last_of('/');
        if (pos == std::string::npos || pos == 0) return true;
        return Ensure_Group(object_path.substr(0, pos));
    }

    void Delete_If_Exists(const std::string& object_path)
    {
        if (file_->exist(object_path))
        {
            H5Ldelete(file_->getId(), object_path.c_str(), H5P_DEFAULT);
            dataset_specs_.erase(object_path);
        }
    }

    DatasetSpec Normalize_Spec(const DatasetSpec& spec) const
    {
        DatasetSpec normalized = spec;
        if (normalized.shape.max_dims.size() != normalized.shape.dims.size())
        {
            normalized.shape.max_dims.assign(normalized.shape.dims.size(), 0);
        }
        if (normalized.shape.chunk_dims.size() != normalized.shape.dims.size())
        {
            normalized.shape.chunk_dims = normalized.shape.dims;
        }
        for (std::size_t i = 0; i < normalized.shape.chunk_dims.size(); ++i)
        {
            if (normalized.shape.chunk_dims[i] == 0)
            {
                normalized.shape.chunk_dims[i] = 1;
            }
        }
        return normalized;
    }

    std::vector<hsize_t> To_HSize(const std::vector<std::size_t>& values) const
    {
        std::vector<hsize_t> converted;
        converted.reserve(values.size());
        for (std::size_t value : values)
        {
            converted.push_back(static_cast<hsize_t>(value));
        }
        return converted;
    }

    std::vector<std::size_t> To_Size_Max(const DatasetSpec& spec) const
    {
        std::vector<std::size_t> converted;
        converted.reserve(spec.shape.dims.size());
        for (std::size_t i = 0; i < spec.shape.dims.size(); ++i)
        {
            const std::size_t configured = spec.shape.max_dims[i];
            if (spec.appendable && i == 0 && configured == 0)
            {
                converted.push_back(std::numeric_limits<std::size_t>::max());
            }
            else if (configured == 0)
            {
                converted.push_back(spec.shape.dims[i]);
            }
            else
            {
                converted.push_back(configured);
            }
        }
        return converted;
    }

    static std::size_t Record_Size(const std::vector<std::size_t>& dims)
    {
        if (dims.size() == 1) return 1;
        return std::accumulate(
            dims.begin() + 1, dims.end(), static_cast<std::size_t>(1),
            [](std::size_t lhs, std::size_t rhs) { return lhs * rhs; });
    }

    static std::vector<std::string> Split_Path(const std::string& path)
    {
        std::vector<std::string> components;
        std::size_t start = 0;
        while (start < path.size())
        {
            while (start < path.size() && path[start] == '/') ++start;
            if (start >= path.size()) break;
            const std::size_t end = path.find('/', start);
            components.push_back(path.substr(start, end == std::string::npos
                                                        ? std::string::npos
                                                        : end - start));
            if (end == std::string::npos) break;
            start = end + 1;
        }
        return components;
    }

    static const char* Status_String(FileStatus status)
    {
        switch (status)
        {
            case FileStatus::closed:
                return "closed";
            case FileStatus::open:
                return "open";
            case FileStatus::closing:
                return "closing";
            case FileStatus::finalized:
                return "finalized";
            case FileStatus::failed:
                return "failed";
        }
        return "unknown";
    }

    bool Fail(const std::string& message)
    {
        last_error_ = message;
        status_ = FileStatus::failed;
        return false;
    }

#ifdef SPONGE_H5_TEST_FAULT_INJECTION
    std::string Test_Fault_Family() const
    {
        if (options_.schema_name == "sponge.restart.h5") return "restart";
        if (options_.observable_only) return "observable";
        return "trajectory";
    }

    bool Test_Fault_Matches(const char* phase) const
    {
        if (test_fault_consumed_) return false;
        const char* configured = std::getenv("SPONGE_H5_TEST_FAULT");
        if (configured == nullptr) return false;
        return std::string(configured) == Test_Fault_Family() + ":" + phase;
    }

    bool Inject_Test_Failure(const char* phase, const std::string& dataset_path)
    {
        test_fault_consumed_ = true;
        last_error_ =
            "injected H5 " + Test_Fault_Family() + " " + phase + " failure";
        if (!dataset_path.empty()) last_error_ += " at " + dataset_path;
        status_ = FileStatus::failed;
        const std::string reason = last_error_;
        Write_String(path::output_status, "failed");
        Write_String(path::output_error, reason);
        Flush();
        last_error_ = reason;
        status_ = FileStatus::failed;
        return false;
    }
#endif

    WriterOptions options_;
    std::string actual_path_;
    std::unique_ptr<HighFive::File> file_;
    std::unordered_map<std::string, DatasetSpec> dataset_specs_;
    FileStatus status_ = FileStatus::closed;
    std::string last_error_;
    bool swmr_write_started_ = false;
#ifdef SPONGE_H5_TEST_FAULT_INJECTION
    bool test_fault_consumed_ = false;
#endif
};

class HighFiveBackendFactory : public WriterBackendFactory
{
   public:
    std::unique_ptr<WriterBackend> Create_Backend() override
    {
        return std::unique_ptr<WriterBackend>(new HighFiveBackend());
    }
};
}  // namespace SpongeH5MD

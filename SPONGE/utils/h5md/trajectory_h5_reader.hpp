#pragma once

#include <hdf5.h>

#include <algorithm>
#include <cstdint>
#include <highfive/highfive.hpp>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "utils/h5md/h5_input_metadata.hpp"
#include "utils/h5md/h5_structural_state.hpp"
#include "utils/h5md/h5md_writer.hpp"

namespace SpongeH5MD
{
class TrajectoryH5Reader
{
   public:
    bool Open(const std::string& file_path) { return Open(file_path, "all"); }

    bool Open(const std::string& file_path, const std::string& particle_stream)
    {
        return Open_Impl(file_path, particle_stream, false);
    }

    bool Open_Swmr(const std::string& file_path)
    {
        return Open_Swmr(file_path, "all");
    }

    bool Open_Swmr(const std::string& file_path,
                   const std::string& particle_stream)
    {
        return Open_Impl(file_path, particle_stream, true);
    }

    bool Refresh()
    {
        if (!Ensure_File()) return false;
        try
        {
            for (const std::string& dataset_path : Refreshable_Datasets())
            {
                if (!Exists(dataset_path)) continue;
                HighFive::DataSet dataset = file_->getDataSet(dataset_path);
                if (H5Drefresh(dataset.getId()) < 0)
                {
                    return Fail("failed to refresh HDF5 dataset: " +
                                dataset_path);
                }
            }
            return true;
        }
        catch (const std::exception& err)
        {
            return Fail(std::string("failed to refresh H5MD trajectory: ") +
                        err.what());
        }
    }

    bool Read_Committed_Frame_Count(std::int64_t* frame_count)
    {
        if (frame_count == nullptr)
        {
            return Fail("committed frame-count output pointer is null");
        }
        if (!Ensure_File()) return false;
        try
        {
            if (!Exists(path::output_frame_count))
            {
                return Fail("committed frame-count dataset is missing");
            }
            const auto dims = Dimensions(path::output_frame_count);
            if (dims.size() != 1 || dims[0] == 0)
            {
                return Fail(
                    "committed frame-count dataset is empty or invalid");
            }
            *frame_count = Read_Required_Single<std::int64_t>(
                path::output_frame_count, {dims[0] - 1},
                "committed frame count");
            return true;
        }
        catch (const std::exception& err)
        {
            return Fail(std::string("failed to read committed frame count: ") +
                        err.what());
        }
    }

    bool Read_Metadata(SpongeH5InputMetadata::TrajectoryMetadata* metadata)
    {
        if (metadata == nullptr)
        {
            return Fail("trajectory metadata output pointer is null");
        }
        if (!Ensure_File()) return false;

        SpongeH5InputMetadata::TrajectoryMetadata result;
        try
        {
            if (Exists(path::sponge_schema_name))
            {
                result.schema_name = Read_String(path::sponge_schema_name);
            }
            if (Exists(path::sponge_schema_version))
            {
                result.schema_version =
                    Read_String(path::sponge_schema_version);
            }
            if (Exists(path::identity_uuid))
            {
                result.identity_uuid = Read_String(path::identity_uuid);
            }
            if (Exists(path::sponge_atom_order_hash))
            {
                result.atom_ordering_hash =
                    Read_String(path::sponge_atom_order_hash);
            }
            result.particle_stream = particle_stream_;
            result.has_position = Exists(Position_Value_Path());
            result.has_box = Exists(Box_Edges_Value_Path());
            result.has_velocity = Exists(Velocity_Value_Path());
            result.has_force = Exists(Force_Value_Path());
            result.has_vds_manifest = Exists(path::shard_manifest);
            if (result.has_position)
            {
                const auto dims = Dimensions(Position_Value_Path());
                if (dims.size() == 3)
                {
                    result.frame_count = static_cast<std::int64_t>(dims[0]);
                    result.atom_count = static_cast<std::int64_t>(dims[1]);
                }
            }
            if (swmr_read_)
            {
                std::int64_t committed = 0;
                if (!Read_Committed_Frame_Count(&committed)) return false;
                result.frame_count = std::min(result.frame_count, committed);
            }
            *metadata = result;
            return true;
        }
        catch (const std::exception& err)
        {
            return Fail(std::string("failed to read trajectory metadata: ") +
                        err.what());
        }
    }

    bool Read_Frame(std::size_t frame_index, RestartStructuralState* frame)
    {
        if (frame == nullptr)
        {
            return Fail("trajectory frame output pointer is null");
        }
        if (!Ensure_File()) return false;

        try
        {
            const auto position_dims = Require_Dimensions(
                Position_Value_Path(), {0, 0, 3}, "trajectory position");
            const std::size_t frame_count = position_dims[0];
            const std::size_t atom_count = position_dims[1];
            if (frame_count == 0)
            {
                return Fail("trajectory contains no frames");
            }
            if (swmr_read_)
            {
                std::int64_t committed = 0;
                if (!Read_Committed_Frame_Count(&committed)) return false;
                if (committed < 0 ||
                    frame_index >= static_cast<std::size_t>(committed))
                {
                    std::ostringstream out;
                    out << "trajectory frame index is not committed: "
                        << frame_index << " >= " << committed;
                    return Fail(out.str());
                }
            }
            if (atom_count == 0)
            {
                return Fail(
                    "trajectory position atom dimension must be positive");
            }
            if (frame_index >= frame_count)
            {
                std::ostringstream out;
                out << "trajectory frame index out of range: " << frame_index
                    << " >= " << frame_count;
                return Fail(out.str());
            }

            Require_Dimensions(Particle_Step_Path(), {frame_count},
                               "trajectory step");
            Require_Dimensions(Particle_Time_Path(), {frame_count},
                               "trajectory time");
            Require_Dimensions(Box_Edges_Value_Path(), {frame_count, 3, 3},
                               "trajectory box edges");

            RestartStructuralState result;
            result.atom_count = atom_count;
            result.step = Read_Required_Single<std::int64_t>(
                Particle_Step_Path(), {frame_index}, "trajectory step");
            result.time = Read_Required_Single<double>(
                Particle_Time_Path(), {frame_index}, "trajectory time");
            result.position_xyz = Read_Required_Selection<float>(
                Position_Value_Path(), {frame_index, 0, 0}, {1, atom_count, 3},
                atom_count * 3, "trajectory position");

            const auto box = Read_Required_Selection<float>(
                Box_Edges_Value_Path(), {frame_index, 0, 0}, {1, 3, 3}, 9,
                "trajectory box edges");
            for (std::size_t i = 0; i < result.box_edges.size(); ++i)
            {
                result.box_edges[i] = box[i];
            }

            if (Exists(Velocity_Value_Path()))
            {
                Require_Dimensions(Velocity_Value_Path(),
                                   {frame_count, atom_count, 3},
                                   "trajectory velocity");
                result.velocity_xyz = Read_Required_Selection<float>(
                    Velocity_Value_Path(), {frame_index, 0, 0},
                    {1, atom_count, 3}, atom_count * 3, "trajectory velocity");
                result.has_velocity = true;
            }

            *frame = result;
            return true;
        }
        catch (const std::exception& err)
        {
            return Fail(std::string("failed to read trajectory frame: ") +
                        err.what());
        }
    }

    std::string Last_Error() const { return last_error_; }

   private:
    bool Open_Impl(const std::string& file_path,
                   const std::string& particle_stream, bool swmr_read)
    {
        last_error_.clear();
        particle_stream_ = particle_stream.empty() ? "all" : particle_stream;
        swmr_read_ = swmr_read;
        try
        {
            const auto mode =
                swmr_read ? HighFive::File::ReadOnly | HighFive::File::ReadSWMR
                          : HighFive::File::ReadOnly;
            file_.reset(new HighFive::File(file_path, mode));
            return true;
        }
        catch (const std::exception& err)
        {
            return Fail(std::string("failed to open H5MD trajectory file: ") +
                        err.what());
        }
    }

    std::vector<std::string> Refreshable_Datasets() const
    {
        return {path::output_frame_count,
                path::output_last_complete_step,
                path::output_last_complete_time,
                Particle_Step_Path(),
                Particle_Time_Path(),
                Position_Value_Path(),
                Box_Edges_Value_Path(),
                Velocity_Value_Path(),
                Force_Value_Path()};
    }

    bool Ensure_File()
    {
        if (file_ == nullptr)
        {
            return Fail("H5MD trajectory reader is not open");
        }
        return true;
    }

    bool Exists(const std::string& object_path) const
    {
        return file_ != nullptr && file_->exist(object_path);
    }

    std::string Particle_Root_Path() const
    {
        return std::string("/particles/") + particle_stream_;
    }

    std::string Particle_Step_Path() const
    {
        return Particle_Root_Path() + "/step";
    }

    std::string Particle_Time_Path() const
    {
        return Particle_Root_Path() + "/time";
    }

    std::string Position_Value_Path() const
    {
        return Particle_Root_Path() + "/position/value";
    }

    std::string Velocity_Value_Path() const
    {
        return Particle_Root_Path() + "/velocity/value";
    }

    std::string Force_Value_Path() const
    {
        return Particle_Root_Path() + "/force/value";
    }

    std::string Box_Edges_Value_Path() const
    {
        return Particle_Root_Path() + "/box/edges/value";
    }

    std::string Read_String(const std::string& dataset_path)
    {
        std::string value;
        file_->getDataSet(dataset_path).read(value);
        return value;
    }

    std::vector<std::size_t> Dimensions(const std::string& dataset_path)
    {
        return file_->getDataSet(dataset_path).getSpace().getDimensions();
    }

    std::vector<std::size_t> Require_Dimensions(
        const std::string& dataset_path,
        const std::vector<std::size_t>& expected, const std::string& label)
    {
        if (!Exists(dataset_path))
        {
            throw std::runtime_error(label +
                                     " dataset is missing: " + dataset_path);
        }
        const auto dims = Dimensions(dataset_path);
        if (dims.size() != expected.size())
        {
            std::ostringstream out;
            out << label << " rank mismatch at " << dataset_path
                << ": expected " << expected.size() << ", got " << dims.size();
            throw std::runtime_error(out.str());
        }
        for (std::size_t i = 0; i < expected.size(); ++i)
        {
            if (expected[i] != 0 && dims[i] != expected[i])
            {
                std::ostringstream out;
                out << label << " shape mismatch at " << dataset_path
                    << ": dimension " << i << " expected " << expected[i]
                    << ", got " << dims[i];
                throw std::runtime_error(out.str());
            }
        }
        return dims;
    }

    template <typename T>
    std::vector<T> Read_Required_Selection(
        const std::string& dataset_path,
        const std::vector<std::size_t>& offsets,
        const std::vector<std::size_t>& counts, std::size_t expected_size,
        const std::string& label)
    {
        std::vector<T> values =
            Read_Hyperslab<T>(dataset_path, offsets, counts, label);
        if (values.size() != expected_size)
        {
            std::ostringstream out;
            out << label << " value count mismatch at " << dataset_path
                << ": expected " << expected_size << ", got " << values.size();
            throw std::runtime_error(out.str());
        }
        return values;
    }

    template <typename T>
    std::vector<T> Read_Hyperslab(const std::string& dataset_path,
                                  const std::vector<std::size_t>& offsets,
                                  const std::vector<std::size_t>& counts,
                                  const std::string& label)
    {
        if (offsets.size() != counts.size())
        {
            throw std::runtime_error(label + " selection rank mismatch at " +
                                     dataset_path);
        }
        const std::size_t value_count = Product(counts);
        std::vector<T> values(value_count);
        HighFive::DataSet dataset = file_->getDataSet(dataset_path);
        const std::vector<hsize_t> h_offsets = To_HSize(offsets);
        const std::vector<hsize_t> h_counts = To_HSize(counts);
        hid_t file_space = H5Dget_space(dataset.getId());
        if (file_space < 0)
        {
            throw std::runtime_error(label + " failed to get dataspace at " +
                                     dataset_path);
        }
        const herr_t select_rc =
            H5Sselect_hyperslab(file_space, H5S_SELECT_SET, h_offsets.data(),
                                nullptr, h_counts.data(), nullptr);
        if (select_rc < 0)
        {
            H5Sclose(file_space);
            throw std::runtime_error(label + " failed to select hyperslab at " +
                                     dataset_path);
        }
        hid_t mem_space = H5Screate_simple(static_cast<int>(h_counts.size()),
                                           h_counts.data(), nullptr);
        if (mem_space < 0)
        {
            H5Sclose(file_space);
            throw std::runtime_error(label +
                                     " failed to create memory dataspace at " +
                                     dataset_path);
        }
        const herr_t read_rc =
            H5Dread(dataset.getId(), Native_H5_Type<T>(), mem_space, file_space,
                    H5P_DEFAULT, values.data());
        H5Sclose(mem_space);
        H5Sclose(file_space);
        if (read_rc < 0)
        {
            throw std::runtime_error(label + " failed to read hyperslab at " +
                                     dataset_path);
        }
        return values;
    }

    template <typename T>
    T Read_Required_Single(const std::string& dataset_path,
                           const std::vector<std::size_t>& offsets,
                           const std::string& label)
    {
        const auto values =
            Read_Required_Selection<T>(dataset_path, offsets, {1}, 1, label);
        return values[0];
    }

    static std::vector<hsize_t> To_HSize(const std::vector<std::size_t>& values)
    {
        std::vector<hsize_t> converted;
        converted.reserve(values.size());
        for (const std::size_t value : values)
        {
            converted.push_back(static_cast<hsize_t>(value));
        }
        return converted;
    }

    static std::size_t Product(const std::vector<std::size_t>& values)
    {
        return std::accumulate(
            values.begin(), values.end(), static_cast<std::size_t>(1),
            [](std::size_t lhs, std::size_t rhs) { return lhs * rhs; });
    }

    template <typename T>
    static hid_t Native_H5_Type()
    {
        if constexpr (std::is_same<T, float>::value)
        {
            return H5T_NATIVE_FLOAT;
        }
        else if constexpr (std::is_same<T, double>::value)
        {
            return H5T_NATIVE_DOUBLE;
        }
        else if constexpr (std::is_same<T, std::int64_t>::value)
        {
            return H5T_NATIVE_INT64;
        }
        else
        {
            static_assert(std::is_same<T, float>::value ||
                              std::is_same<T, double>::value ||
                              std::is_same<T, std::int64_t>::value,
                          "unsupported HDF5 numeric read type");
        }
    }

    bool Fail(const std::string& message)
    {
        last_error_ = message;
        return false;
    }

    std::unique_ptr<HighFive::File> file_;
    std::string last_error_;
    std::string particle_stream_ = "all";
    bool swmr_read_ = false;
};
}  // namespace SpongeH5MD

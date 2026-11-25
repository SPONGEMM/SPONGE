#pragma once

#include <limits>
#include <map>
#include <string>
#include <vector>

#include "utils/h5md/module_h5_mappings.hpp"

namespace SpongeH5MD
{
struct MetadynamicsGrid
{
    // Values use C order, with the value channel as the last dimension.
    std::vector<std::size_t> shape;
    std::vector<std::vector<float>> axes;
    std::vector<std::string> cv_labels;
    std::vector<std::string> value_labels;
    bool scattered = false;
    std::vector<float> values;

    std::vector<std::size_t> Value_Shape() const
    {
        auto dims = shape;
        dims.push_back(value_labels.size());
        return dims;
    }
    std::size_t Value_Count() const
    {
        std::size_t count = value_labels.size();
        for (auto n : shape)
        {
            if (!n || count > std::numeric_limits<std::size_t>::max() / n)
                return 0;
            count *= n;
        }
        return count;
    }
    bool Valid() const
    {
        if (shape.empty() || axes.empty() || axes.size() != cv_labels.size() ||
            !Value_Count() ||
            (scattered ? shape.size() != 1 : shape.size() != axes.size()))
            return false;
        for (std::size_t i = 0; i < axes.size(); ++i)
            if (axes[i].size() != shape[scattered ? 0 : i]) return false;
        return true;
    }
    bool Same_Layout(const MetadynamicsGrid& other) const
    {
        return shape == other.shape && axes == other.axes &&
               cv_labels == other.cv_labels &&
               value_labels == other.value_labels &&
               scattered == other.scattered;
    }
};

class MetadynamicsPotentialSnapshots
{
   public:
    static std::string Root(const std::string& name)
    {
        return Metadynamics_Diagnostic_Root(name) + "/potential_snapshots";
    }
    bool Define(H5MDWriter* writer, const std::string& name,
                const MetadynamicsGrid& grid, bool virtual_values = false)
    {
        if (!grid.Valid()) return false;
        auto& state = states_[name];
        if (state.failed) return false;
        if (state.defined) return state.layout.Same_Layout(grid);
        const auto root = Root(name);
        state.failed = true;
        if (!writer->Ensure_Group(root + "/axes") ||
            !writer->Write_String(root + "/layout",
                                  grid.scattered ? "points" : "rectilinear") ||
            !writer->Write_String_Array(root + "/cv_labels", grid.cv_labels) ||
            !writer->Write_String_Array(root + "/value_labels",
                                        grid.value_labels))
            return false;
        for (std::size_t i = 0; i < grid.axes.size(); ++i)
        {
            const auto path = root + "/axes/" + std::to_string(i);
            const auto n = grid.axes[i].size();
            if (!writer->Create_Dataset(
                    {path, DataType::float32, {{0}, {n}, {n}}, true}) ||
                !writer->Append_Float32(path, grid.axes[i].data(), n))
                return false;
        }
        if (!writer->Create_Dataset({root + "/committed_count",
                                     DataType::int64,
                                     {{0}, {0}, {1}},
                                     true}))
            return false;
        if (!virtual_values)
        {
            auto dims = grid.Value_Shape();
            dims.insert(dims.begin(), 0);
            auto chunk = dims;
            chunk[0] = 1;
            if (!writer->Create_Dataset({root + "/values",
                                         DataType::float32,
                                         {dims, dims, chunk},
                                         true}) ||
                !writer->Create_Dataset(
                    {root + "/step", DataType::int64, {{0}, {0}, {1}}, true}) ||
                !writer->Create_Dataset(
                    {root + "/time", DataType::float64, {{0}, {0}, {1}}, true}))
                return false;
        }
        state.layout = grid;
        state.layout.values.clear();
        state.defined = true;
        state.failed = false;
        return true;
    }
    bool Write(H5MDWriter* writer, const std::string& name,
               const MetadynamicsGrid& grid, int64_t step, double time)
    {
        if (!Define(writer, name, grid) ||
            grid.values.size() != grid.Value_Count())
            return false;
        auto& state = states_.at(name);
        const auto root = Root(name);
        const int64_t count = state.count + 1;
        state.failed = true;
        if (!writer->Append_Float32(root + "/values", grid.values.data(),
                                    grid.values.size()) ||
            !writer->Append_Int64(root + "/step", &step, 1) ||
            !writer->Append_Float64(root + "/time", &time, 1) ||
            !writer->Flush() ||
            !writer->Append_Int64(root + "/committed_count", &count, 1) ||
            !writer->Flush())
            return false;
        state.count = count;
        state.failed = false;
        return true;
    }

   private:
    struct State
    {
        bool defined = false;
        bool failed = false;
        int64_t count = 0;
        MetadynamicsGrid layout;
    };
    std::map<std::string, State> states_;
};
}  // namespace SpongeH5MD

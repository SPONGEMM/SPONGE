#pragma once

#include <limits>
#include <map>
#include <string>

#include "utils/h5md/module_h5_mappings.hpp"

namespace SpongeH5MD
{
// UTF-8 exports use an append-only byte stream, never the HDF5 VLEN heap.
// A committed record becomes visible only after its payload and index flush.
class MetadynamicsPotentialSnapshots
{
   public:
    bool Write(H5MDWriter* writer, const std::string& name,
               const std::string& text, int64_t step, double time)
    {
        const std::string root =
            Metadynamics_Diagnostic_Root(name) + "/potential_snapshots";
        auto& state = states_[name];
        if (state.failed) return false;
        if (!state.defined)
        {
            if (!writer->Ensure_Group(root) ||
                !writer->Create_Dataset({root + "/data",
                                         DataType::uint8,
                                         {{0}, {0}, {65536}},
                                         true}))
                return false;
            for (const auto* field :
                 {"offset", "length", "step", "committed_count"})
            {
                if (!writer->Create_Dataset({root + "/" + field,
                                             DataType::int64,
                                             {{0}, {0}, {1}},
                                             true}))
                    return false;
            }
            if (!writer->Create_Dataset(
                    {root + "/time", DataType::float64, {{0}, {0}, {1}}, true}))
                return false;
            state.defined = true;
        }
        if (text.empty()) return true;
        if (text.size() >
            static_cast<std::size_t>(std::numeric_limits<int64_t>::max() -
                                     state.bytes))
            return false;
        const int64_t length = static_cast<int64_t>(text.size());
        const int64_t count = state.count + 1;
        state.failed = true;
        if (!writer->Append_UInt8(root + "/data",
                                  reinterpret_cast<const uint8_t*>(text.data()),
                                  text.size()) ||
            !writer->Append_Int64(root + "/offset", &state.bytes, 1) ||
            !writer->Append_Int64(root + "/length", &length, 1) ||
            !writer->Append_Int64(root + "/step", &step, 1) ||
            !writer->Append_Float64(root + "/time", &time, 1) ||
            !writer->Flush() ||
            !writer->Append_Int64(root + "/committed_count", &count, 1) ||
            !writer->Flush())
            return false;
        state.bytes += length;
        state.count = count;
        state.failed = false;
        return true;
    }

   private:
    struct State
    {
        bool defined = false;
        bool failed = false;
        int64_t bytes = 0;
        int64_t count = 0;
    };
    std::map<std::string, State> states_;
};
}  // namespace SpongeH5MD

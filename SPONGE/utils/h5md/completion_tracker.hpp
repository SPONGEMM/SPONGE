#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "utils/h5md/h5md_writer.hpp"
#include "utils/h5md/vds_trajectory_h5_writer.hpp"

namespace SpongeH5MD
{
struct CompletionState
{
    FileStatus status = FileStatus::closed;
    int64_t frame_count = 0;
    int64_t last_complete_step = -1;
    double last_complete_time = 0.0;
    bool frame_open = false;
    int64_t pending_step = -1;
    double pending_time = 0.0;
};

class OutputCompletionTracker
{
   public:
    explicit OutputCompletionTracker(H5MDWriter* writer) : writer_(writer) {}

    bool Mark_Open()
    {
        state_.status = FileStatus::open;
        if (writer_ == nullptr)
        {
            return true;
        }
        return writer_->Set_Status(FileStatus::open) &&
               writer_->Write_String(path::output_status, "open") &&
               Write_Counts();
    }

    bool Begin_Frame(int64_t step, double time)
    {
        if (state_.frame_open)
        {
            last_error_ = "cannot begin a new frame before completing current";
            return false;
        }
        state_.frame_open = true;
        state_.pending_step = step;
        state_.pending_time = time;
        return true;
    }

    bool Complete_Frame()
    {
        if (!state_.frame_open)
        {
            last_error_ = "cannot complete frame because no frame is open";
            return false;
        }
        state_.frame_open = false;
        state_.frame_count += 1;
        state_.last_complete_step = state_.pending_step;
        state_.last_complete_time = state_.pending_time;
        state_.pending_step = -1;
        state_.pending_time = 0.0;
        return Write_Counts();
    }

    bool Mark_Closing()
    {
        if (state_.frame_open)
        {
            last_error_ = "cannot close while a frame is incomplete";
            return false;
        }
        state_.status = FileStatus::closing;
        if (writer_ == nullptr)
        {
            return true;
        }
        return writer_->Set_Status(FileStatus::closing) &&
               writer_->Write_String(path::output_status, "closing");
    }

    bool Mark_Finalized()
    {
        if (state_.frame_open)
        {
            last_error_ = "cannot finalize while a frame is incomplete";
            return false;
        }
        state_.status = FileStatus::finalized;
        if (writer_ == nullptr)
        {
            return true;
        }
        return writer_->Set_Status(FileStatus::finalized) &&
               writer_->Write_String(path::output_status, "finalized") &&
               Write_Counts();
    }

    bool Mark_Failed(const std::string& reason)
    {
        state_.status = FileStatus::failed;
        last_error_ = reason;
        if (writer_ == nullptr)
        {
            return true;
        }
        return writer_->Set_Status(FileStatus::failed) &&
               writer_->Write_String(path::output_status, "failed") &&
               writer_->Write_String(path::output_error, reason);
    }

    bool Has_Incomplete_Frame() const { return state_.frame_open; }

    const CompletionState& State() const { return state_; }
    std::string Last_Error() const { return last_error_; }

   private:
    bool Write_Counts()
    {
        if (writer_ == nullptr)
        {
            return true;
        }
        if (!writer_->Create_Dataset({path::output_frame_count,
                                      DataType::int64,
                                      {{0}, {1}, {1}},
                                      true}))
        {
            return false;
        }
        if (!writer_->Create_Dataset({path::output_last_complete_step,
                                      DataType::int64,
                                      {{0}, {1}, {1}},
                                      true}))
        {
            return false;
        }
        if (!writer_->Create_Dataset({path::output_last_complete_time,
                                      DataType::float64,
                                      {{0}, {1}, {1}},
                                      true}))
        {
            return false;
        }
        const int64_t frame_count = state_.frame_count;
        const int64_t step = state_.last_complete_step;
        const double time = state_.last_complete_time;
        return writer_->Append_Int64(path::output_frame_count, &frame_count,
                                     1) &&
               writer_->Append_Int64(path::output_last_complete_step, &step,
                                     1) &&
               writer_->Append_Float64(path::output_last_complete_time, &time,
                                       1);
    }

    H5MDWriter* writer_ = nullptr;
    CompletionState state_;
    std::string last_error_;
};

struct ManifestValidationReport
{
    bool valid = true;
    std::string error_message;
    int64_t complete_shard_count = 0;
    int64_t complete_frame_count = 0;
};

inline ManifestValidationReport Validate_Complete_Manifest(
    const std::vector<VdsShardManifestEntry>& manifest,
    bool allow_repair = false)
{
    ManifestValidationReport report;
    int64_t expected_frame_start = 0;
    int64_t previous_index = -1;
    for (const auto& entry : manifest)
    {
        if (entry.status != "complete")
        {
            if (allow_repair)
            {
                break;
            }
            report.valid = false;
            report.error_message = "manifest contains incomplete shard";
            return report;
        }
        if (entry.index != previous_index + 1)
        {
            report.valid = false;
            report.error_message = "manifest shard indices are not contiguous";
            return report;
        }
        if (entry.frame_start != expected_frame_start)
        {
            report.valid = false;
            report.error_message = "manifest frame ranges are not contiguous";
            return report;
        }
        if (Manifest_Entry_Has_Negative_Stream_Count(entry) ||
            !Manifest_Entry_Has_Stream_Frames(entry))
        {
            report.valid = false;
            report.error_message =
                "manifest shard must contain at least one stream frame";
            return report;
        }
        expected_frame_start += entry.frame_count;
        previous_index = entry.index;
        report.complete_shard_count += 1;
        report.complete_frame_count += entry.frame_count;
    }
    return report;
}
}  // namespace SpongeH5MD

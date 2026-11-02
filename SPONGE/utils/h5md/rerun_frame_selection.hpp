#pragma once

#include <algorithm>

namespace SpongeH5MD
{
struct H5RerunFrameSelection
{
    bool has_frame = false;
    int frame_index = -1;
    int next_frame_index = 0;
    int skipped_frames = 0;
};

inline H5RerunFrameSelection Select_Next_H5_Rerun_Frame(int next_frame_index,
                                                        int strip,
                                                        int frame_count)
{
    H5RerunFrameSelection selection;
    selection.next_frame_index = std::max(0, next_frame_index);
    if (frame_count <= 0)
    {
        return selection;
    }
    const int skip_count = std::max(0, strip);
    const int frame_index = selection.next_frame_index + skip_count;
    selection.skipped_frames = std::min(
        skip_count, std::max(0, frame_count - selection.next_frame_index));
    if (frame_index >= frame_count)
    {
        selection.next_frame_index = frame_count;
        return selection;
    }
    selection.has_frame = true;
    selection.frame_index = frame_index;
    selection.next_frame_index = frame_index + 1;
    return selection;
}
}  // namespace SpongeH5MD

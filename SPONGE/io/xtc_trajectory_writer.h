#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

// The vendored xdrfile headers are deliberately NOT included here: xdrfile.h
// defines `DIM`, `matrix` and `rvec` in the global namespace, which collide
// with existing SPONGE identifiers. Keeping the C dependency opaque confines it
// to xtc_trajectory_writer.cpp.
struct XDRFILE;

namespace SpongeXTC
{
// Streaming writer for GROMACS XTC trajectories.
//
// Units: SPONGE works in Angstrom, XTC is specified in nanometre. Every
// coordinate and box edge is scaled by 0.1 on the way out; this is not
// configurable because the file carries no unit marker and a wrong factor
// would silently produce a trajectory that every other tool misreads.
//
// Durability: an XTC frame is self-describing (magic + natoms + step + time +
// box + natoms + compressed coordinates) and the container has no trailer, so
// a file ending on a frame boundary is always readable. Flush() pushes
// completed frames to the OS without closing, and the frame-aligned stdio
// buffer installed in Open() keeps a single frame from straddling a buffer
// boundary, so an interrupt drops a whole frame instead of leaving a fragment.
class XtcTrajectoryWriter
{
   public:
    XtcTrajectoryWriter() = default;
    ~XtcTrajectoryWriter();
    XtcTrajectoryWriter(const XtcTrajectoryWriter&) = delete;
    XtcTrajectoryWriter& operator=(const XtcTrajectoryWriter&) = delete;

    // `buffer_frames` sizes the stdio buffer in whole frames; <=0 keeps the
    // stdio default. `append` continues an existing trajectory (resumed runs);
    // the existing atom count is verified before the first frame is written.
    bool Open(const std::string& path, int atom_numbers, float precision,
              int buffer_frames, bool append, bool frame_index,
              std::string* error_message);

    // `coordinates` holds 3 * atom_numbers Angstrom values.
    // `box_row_major` holds 9 Angstrom values in row-major lower-triangular
    // order (a11, 0, 0, a21, a22, 0, a31, a32, a33).
    bool Append(const float* coordinates, int step, float time_in_ps,
                const float box_row_major[9], std::string* error_message);

    bool Flush(std::string* error_message);

    void Close();

    bool Is_Open() const { return handle_ != nullptr; }
    const std::string& Path() const { return path_; }
    const std::string& Index_Path() const { return index_path_; }
    std::int64_t Frame_Count() const { return frame_count_; }

   private:
    void Close_Index_File();
    bool Write_Index_Entry(std::int64_t end_offset, std::string* error_message);

    XDRFILE* handle_ = nullptr;
    std::string path_;
    std::FILE* index_file_ = nullptr;
    std::string index_path_;
    std::vector<float> nanometre_buffer_;
    int atom_numbers_ = 0;
    float precision_ = 1000.0f;
    bool dirty_ = false;
    bool frame_index_enabled_ = false;
    std::int64_t frame_count_ = 0;
};
}  // namespace SpongeXTC

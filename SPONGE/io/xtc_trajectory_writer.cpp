#include "xtc_trajectory_writer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

// Confined to this translation unit on purpose: xdrfile.h leaks DIM, matrix and
// rvec into the global namespace.
#include "xdrfile.h"
#include "xdrfile_xtc.h"
#undef DIM

namespace
{
// SPONGE stores Angstrom, XTC is defined in nanometre.
constexpr float ANGSTROM_TO_NANOMETRE = 0.1f;

// Upper bound of an encoded frame, used to size the stdio buffer so a single
// frame never straddles a buffer boundary. xdrfile packs at most 9 atoms per
// compressed block with a 4-byte length prefix and at most 3 bytes per atom,
// so 40 bytes per 9 atoms leaves ample headroom.
std::size_t Frame_Bytes_Upper_Bound(int atom_numbers)
{
    const std::size_t blocks =
        static_cast<std::size_t>(atom_numbers) / 9 + 1;
    const std::size_t estimate = 128 + blocks * 40;
    return std::max<std::size_t>(estimate, 512);
}

// xdrfile scales coordinates by `precision` and stores the result in an int,
// so |coordinate_nm * precision| must stay well inside the int range.
constexpr float MAX_SCALED_COORDINATE = 2.0e6f;
}  // namespace

namespace SpongeXTC
{
XtcTrajectoryWriter::~XtcTrajectoryWriter() { Close(); }

bool XtcTrajectoryWriter::Open(const std::string& path, int atom_numbers,
                               float precision, int buffer_frames, bool append,
                               bool frame_index, std::string* error_message)
{
    Close();

    if (atom_numbers <= 0)
    {
        *error_message = "xtc writer: 'atom_numbers' must be positive";
        return false;
    }
    if (!(precision > 0.0f))
    {
        *error_message =
            "xtc writer: 'precision' must be positive (1000.0 means 0.001 nm)";
        return false;
    }

    if (append)
    {
        // Resume runs must not concatenate frames of a different topology.
        std::FILE* probe = std::fopen(path.c_str(), "rb");
        if (probe != nullptr)
        {
            std::fclose(probe);
            int existing_atoms = 0;
            if (read_xtc_natoms(path.c_str(), &existing_atoms) == exdrOK &&
                existing_atoms > 0 && existing_atoms != atom_numbers)
            {
                *error_message = "xtc writer: cannot append to '" + path +
                                 "': the existing trajectory has " +
                                 std::to_string(existing_atoms) +
                                 " atoms but the current system has " +
                                 std::to_string(atom_numbers);
                return false;
            }
        }
    }

    handle_ = xdrfile_open(path.c_str(), append ? "a" : "w");
    if (handle_ == nullptr)
    {
        *error_message =
            "xtc writer: failed to open '" + path + "' for writing";
        return false;
    }

    path_ = path;
    atom_numbers_ = atom_numbers;
    precision_ = precision;
    dirty_ = false;
    frame_count_ = 0;
    nanometre_buffer_.assign(
        static_cast<std::size_t>(atom_numbers) * 3, 0.0f);

    const std::size_t frame_bytes = Frame_Bytes_Upper_Bound(atom_numbers);
    const int frames = buffer_frames > 0 ? buffer_frames : 1;
    const int buffer_status =
        xdrfile_setvbuf(handle_, frame_bytes, frames);
    if (buffer_status != exdrOK)
    {
        // A missing buffer only costs throughput and frame atomicity, never
        // correctness, so this is reported rather than fatal.
        std::fprintf(stderr,
                     "Warning: failed to install an XTC stdio buffer of %d "
                     "frames; falling back to the default buffer\n",
                     frames);
    }

    if (frame_index)
    {
        index_path_ = path + ".idx";
        index_file_ = std::fopen(index_path_.c_str(), append ? "ab" : "wb");
        if (index_file_ == nullptr)
        {
            xdrfile_close(handle_);
            handle_ = nullptr;
            *error_message = "xtc writer: failed to open '" + index_path_ +
                             "' for writing";
            return false;
        }
    }
    frame_index_enabled_ = frame_index;
    return true;
}

bool XtcTrajectoryWriter::Append(const float* coordinates, int step,
                                 float time_in_ps,
                                 const float box_row_major[9],
                                 std::string* error_message)
{
    if (handle_ == nullptr)
    {
        *error_message = "xtc writer: Append() called before Open()";
        return false;
    }

    // First frame only: catch an unreachable precision before spending I/O on a
    // trajectory that can never be compressed.
    if (frame_count_ == 0)
    {
        float max_abs = 0.0f;
        for (int i = 0; i < atom_numbers_ * 3; ++i)
        {
            max_abs = std::max(max_abs, std::fabs(coordinates[i]));
        }
        const float scaled =
            max_abs * ANGSTROM_TO_NANOMETRE * precision_;
        if (!(scaled < MAX_SCALED_COORDINATE))
        {
            *error_message =
                "xtc writer: xtc precision " + std::to_string(precision_) +
                " overflows the XTC integer grid for a coordinate extent of " +
                std::to_string(max_abs) +
                " Angstrom; lower the [xtc] precision value";
            return false;
        }
    }

    for (int i = 0; i < atom_numbers_ * 3; ++i)
    {
        nanometre_buffer_[static_cast<std::size_t>(i)] =
            coordinates[i] * ANGSTROM_TO_NANOMETRE;
    }

    matrix box;
    std::memset(box, 0, sizeof(box));
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            box[row][column] = box_row_major[row * 3 + column] *
                               ANGSTROM_TO_NANOMETRE;
        }
    }

    const int status = write_xtc(
        handle_, atom_numbers_, step, time_in_ps, box,
        reinterpret_cast<rvec*>(nanometre_buffer_.data()), precision_);
    if (status != exdrOK)
    {
        *error_message = std::string("xtc writer: write_xtc failed (") +
                         exdr_message[status] + ") at step " +
                         std::to_string(step);
        return false;
    }

    ++frame_count_;
    dirty_ = true;

    if (frame_index_enabled_)
    {
        // xdr_tell() reports the end of the frame just written and, per the C
        // standard, flushes the pending write buffer first, so the recorded
        // offset is always backed by bytes that reached the OS.
        const std::int64_t end_offset = xdr_tell(handle_);
        if (end_offset < 0)
        {
            *error_message = "xtc writer: xdr_tell failed while indexing";
            return false;
        }
        if (!Write_Index_Entry(end_offset, error_message))
        {
            return false;
        }
    }
    return true;
}

bool XtcTrajectoryWriter::Flush(std::string* error_message)
{
    if (!dirty_ || handle_ == nullptr)
    {
        return true;
    }
    dirty_ = false;
    if (xdrfile_flush(handle_) != exdrOK)
    {
        *error_message = "xtc writer: failed to flush '" + path_ + "'";
        return false;
    }
    return true;
}

void XtcTrajectoryWriter::Close()
{
    if (handle_ != nullptr)
    {
        xdrfile_close(handle_);
        handle_ = nullptr;
    }
    Close_Index_File();
    dirty_ = false;
}

void XtcTrajectoryWriter::Close_Index_File()
{
    if (index_file_ != nullptr)
    {
        std::fclose(index_file_);
        index_file_ = nullptr;
    }
}

bool XtcTrajectoryWriter::Write_Index_Entry(std::int64_t end_offset,
                                            std::string* error_message)
{
    if (index_file_ == nullptr)
    {
        return true;
    }
    if (std::fwrite(&end_offset, sizeof(end_offset), 1, index_file_) != 1)
    {
        *error_message =
            "xtc writer: failed to append to '" + index_path_ + "'";
        return false;
    }
    if (std::fflush(index_file_) != 0)
    {
        *error_message =
            "xtc writer: failed to flush '" + index_path_ + "'";
        return false;
    }
    return true;
}
}  // namespace SpongeXTC

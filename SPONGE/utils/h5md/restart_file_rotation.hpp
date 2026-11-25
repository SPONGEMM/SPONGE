#pragma once

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <highfive/highfive.hpp>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "utils/h5md/atomic_file_publish.hpp"
#include "utils/h5md/h5md_writer.hpp"

namespace SpongeH5MD
{
// Each history file is an immutable, self-contained checkpoint. The number in
// its name orders archival operations, independently of the simulation step.
struct RestartHistoryEntry
{
    int64_t sequence;
    std::filesystem::path path;
};

inline std::filesystem::path Restart_History_Directory(
    const std::string& restart_path)
{
    const auto parent = std::filesystem::path(restart_path).parent_path();
    return parent.empty() ? std::filesystem::path(".") : parent;
}

inline std::vector<RestartHistoryEntry> List_Restart_History(
    const std::string& restart_path)
{
    const auto directory = Restart_History_Directory(restart_path);
    std::vector<RestartHistoryEntry> entries;
    if (!std::filesystem::exists(directory)) return entries;
    const std::string prefix = "history-";
    const std::string suffix =
        "_" + std::filesystem::path(restart_path).filename().string();
    for (const auto& entry : std::filesystem::directory_iterator(directory))
    {
        if (!std::filesystem::is_regular_file(entry.symlink_status())) continue;
        const auto name = entry.path().filename().string();
        if (name.size() <= prefix.size() + suffix.size() ||
            name.compare(0, prefix.size(), prefix) != 0 ||
            name.compare(name.size() - suffix.size(), suffix.size(), suffix) !=
                0)
            continue;
        const auto digits = name.substr(
            prefix.size(), name.size() - prefix.size() - suffix.size());
        int64_t sequence = 0;
        const auto parsed = std::from_chars(
            digits.data(), digits.data() + digits.size(), sequence);
        if (parsed.ec == std::errc() &&
            parsed.ptr == digits.data() + digits.size() && sequence > 0 &&
            digits == std::to_string(sequence))
            entries.push_back({sequence, entry.path()});
    }
    std::sort(entries.begin(), entries.end(),
              [](const RestartHistoryEntry& lhs, const RestartHistoryEntry& rhs)
              { return lhs.sequence < rhs.sequence; });
    return entries;
}

inline int64_t Read_Restart_File_Generation(
    const std::filesystem::path& checkpoint_path)
{
    HighFive::File file(checkpoint_path.string(), HighFive::File::ReadOnly);
    // Older checkpoints may not carry a publication generation.
    if (!file.exist(path::output_restart_generation)) return 0;
    const auto dataset = file.getDataSet(path::output_restart_generation);
    const auto dimensions = dataset.getSpace().getDimensions();
    if (dimensions != std::vector<std::size_t>{1})
        throw std::runtime_error("invalid restart generation shape");
    std::vector<int64_t> generations;
    dataset.read(generations);
    if (generations[0] <= 0)
        throw std::runtime_error("restart generation must be positive");
    return generations[0];
}

inline bool Read_Last_Restart_Generation(const std::string& restart_path,
                                         int64_t* generation,
                                         std::string* error)
{
    try
    {
        *generation = 0;
        if (std::filesystem::exists(restart_path))
            *generation = Read_Restart_File_Generation(restart_path);
        // A user may restore an older checkpoint at the latest path. Continue
        // above all retained generations instead of reusing publication IDs.
        for (const auto& entry : List_Restart_History(restart_path))
        {
            *generation =
                std::max(*generation, Read_Restart_File_Generation(entry.path));
        }
        if (*generation == std::numeric_limits<int64_t>::max())
            throw std::runtime_error("restart generation is exhausted");
        return true;
    }
    catch (const std::exception& exception)
    {
        *error = std::string("failed to read previous restart generation: ") +
                 exception.what();
        return false;
    }
}

inline bool Restart_Files_Have_Identical_Contents(
    const std::filesystem::path& lhs, const std::filesystem::path& rhs)
{
    if (std::filesystem::file_size(lhs) != std::filesystem::file_size(rhs))
        return false;
    std::ifstream left(lhs, std::ios::binary), right(rhs, std::ios::binary);
    if (!left.is_open() || !right.is_open())
        throw std::runtime_error("failed to compare restart archive contents");
    std::array<char, 65536> left_buffer, right_buffer;
    do
    {
        left.read(left_buffer.data(), left_buffer.size());
        right.read(right_buffer.data(), right_buffer.size());
        if ((left.fail() && !left.eof()) || (right.fail() && !right.eof()))
            throw std::runtime_error("failed to read restart archive contents");
        if (left.gcount() != right.gcount() ||
            !std::equal(left_buffer.begin(),
                        left_buffer.begin() + left.gcount(),
                        right_buffer.begin()))
            return false;
    } while (!left.eof() && !right.eof());
    return left.eof() && right.eof();
}

inline bool Is_Same_Restart_Checkpoint(const std::filesystem::path& lhs,
                                       const std::filesystem::path& rhs)
{
    if (std::filesystem::equivalent(lhs, rhs)) return true;
    // A copy fallback can remain after a crash between archiving and
    // publishing. Prefer publication identity to avoid reading large files;
    // old checkpoints without that identity need an exact byte comparison.
    {
        HighFive::File left(lhs.string(), HighFive::File::ReadOnly);
        HighFive::File right(rhs.string(), HighFive::File::ReadOnly);
        if (left.exist(path::output_restart_generation) &&
            right.exist(path::output_restart_generation) &&
            left.exist(path::identity_uuid) && right.exist(path::identity_uuid))
        {
            std::string left_uuid, right_uuid;
            left.getDataSet(path::identity_uuid).read(left_uuid);
            right.getDataSet(path::identity_uuid).read(right_uuid);
            if (!left_uuid.empty() && !right_uuid.empty())
                return left_uuid == right_uuid &&
                       Read_Restart_File_Generation(lhs) ==
                           Read_Restart_File_Generation(rhs);
        }
    }
    // Release HDF5 handles before reopening via streams (also on Windows).
    return Restart_Files_Have_Identical_Contents(lhs, rhs);
}

// Return value describes publication, not pruning. A pruning error must never
// roll back the newly published checkpoint or remove additional history.
inline bool Publish_Restart_With_History(const std::string& temporary_path,
                                         const std::string& restart_path,
                                         int max_count, std::string* error,
                                         std::string* cleanup_error)
{
    error->clear();
    cleanup_error->clear();
    try
    {
        if (max_count <= 0)
            throw std::runtime_error(
                "max_restart_export_count must be positive");
        auto history = List_Restart_History(restart_path);
        std::string new_archive;
        // The guard removes only an archive made by this attempt on failure.
        // Existing archives, including one left by an interrupted attempt,
        // remain untouched until the new latest file has been published.
        if (max_count > 1 && std::filesystem::exists(restart_path))
        {
            if (history.empty() ||
                !Is_Same_Restart_Checkpoint(restart_path, history.back().path))
            {
                if (!history.empty() && history.back().sequence ==
                                            std::numeric_limits<int64_t>::max())
                    throw std::runtime_error(
                        "restart history sequence is exhausted");
                const int64_t sequence =
                    history.empty() ? 1 : history.back().sequence + 1;
                const auto directory = Restart_History_Directory(restart_path);
                std::filesystem::create_directories(directory);
                new_archive =
                    (directory /
                     ("history-" + std::to_string(sequence) + "_" +
                      std::filesystem::path(restart_path).filename().string()))
                        .string();
                const auto staged_archive = new_archive + ".tmp";
                if (!Remove_File_If_Exists(staged_archive, error)) return false;
                TemporaryFileGuard staged_guard(staged_archive);
                // Linking a symlink can preserve the symlink itself on POSIX.
                // Copy its referent instead: the archive must be a regular,
                // independent file even if the external target is modified.
                const bool source_is_symlink = std::filesystem::is_symlink(
                    std::filesystem::symlink_status(restart_path));
                std::error_code link_error;
                if (!source_is_symlink)
                    std::filesystem::create_hard_link(
                        restart_path, staged_archive, link_error);
                if ((source_is_symlink || link_error) &&
                    !std::filesystem::copy_file(restart_path, staged_archive))
                    throw std::runtime_error("failed to copy previous restart");
                if (!Atomic_Replace_File(staged_archive, new_archive, error))
                    return false;
                staged_guard.Release();
                history.push_back({sequence, new_archive});
            }
        }
        TemporaryFileGuard archive_guard(new_archive);
        if (!Atomic_Replace_File(temporary_path, restart_path, error))
            return false;
        archive_guard.Release();

        // Publication is committed. Failure to prune leaves extra valid files.
        try
        {
            const auto keep = static_cast<std::size_t>(max_count - 1);
            for (std::size_t index = 0; index + keep < history.size(); ++index)
                std::filesystem::remove(history[index].path);
        }
        catch (const std::exception& exception)
        {
            *cleanup_error = std::string("failed to prune restart history: ") +
                             exception.what();
        }
        return true;
    }
    catch (const std::exception& exception)
    {
        *error = std::string("failed to preserve restart history: ") +
                 exception.what();
        return false;
    }
}
}  // namespace SpongeH5MD

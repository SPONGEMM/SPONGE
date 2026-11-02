#pragma once

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace SpongeH5MD
{
#ifdef _WIN32
inline bool Destination_Replace_Is_Blocked_By_Open_Handle(
    const std::string& path)
{
    HANDLE probe =
        CreateFileA(path.c_str(), DELETE,
                    FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                    NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (probe != INVALID_HANDLE_VALUE)
    {
        CloseHandle(probe);
        return false;
    }
    const DWORD probe_error = GetLastError();
    return probe_error == ERROR_SHARING_VIOLATION ||
           probe_error == ERROR_LOCK_VIOLATION;
}
#endif

inline bool Remove_File_If_Exists(const std::string& path,
                                  std::string* error_message = nullptr)
{
    if (std::remove(path.c_str()) == 0 || errno == ENOENT) return true;
    if (error_message != nullptr)
    {
        *error_message = "failed to remove temporary file " + path + ": " +
                         std::strerror(errno);
    }
    return false;
}

inline bool Atomic_Replace_File(const std::string& temporary_path,
                                const std::string& destination_path,
                                std::string* error_message = nullptr,
                                bool* destination_busy = nullptr,
                                bool retry_destination_busy = true)
{
    if (destination_busy != nullptr) *destination_busy = false;
#ifdef _WIN32
    constexpr int kSharingRetryCount = 50;
    constexpr auto kSharingRetryDelay = std::chrono::milliseconds(10);
    const int retry_count = retry_destination_busy ? kSharingRetryCount : 0;
    DWORD last_error = ERROR_SUCCESS;
    for (int attempt = 0; attempt <= retry_count; ++attempt)
    {
        if (MoveFileExA(temporary_path.c_str(), destination_path.c_str(),
                        MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        {
            return true;
        }
        last_error = GetLastError();
        const bool retryable = last_error == ERROR_SHARING_VIOLATION ||
                               last_error == ERROR_LOCK_VIOLATION ||
                               last_error == ERROR_ACCESS_DENIED;
        if (!retryable || attempt == retry_count) break;
        std::this_thread::sleep_for(kSharingRetryDelay);
    }
    if (error_message != nullptr)
    {
        *error_message = "failed to atomically replace " + destination_path +
                         " with " + temporary_path + ": Windows error " +
                         std::to_string(last_error);
    }
    if (destination_busy != nullptr)
    {
        *destination_busy =
            last_error == ERROR_SHARING_VIOLATION ||
            last_error == ERROR_LOCK_VIOLATION ||
            (last_error == ERROR_ACCESS_DENIED &&
             Destination_Replace_Is_Blocked_By_Open_Handle(destination_path));
    }
    return false;
#else
    if (std::rename(temporary_path.c_str(), destination_path.c_str()) == 0)
    {
        return true;
    }
    if (error_message != nullptr)
    {
        *error_message = "failed to atomically replace " + destination_path +
                         " with " + temporary_path + ": " +
                         std::strerror(errno);
    }
    return false;
#endif
}

class TemporaryFileGuard
{
   public:
    explicit TemporaryFileGuard(std::string path) : path_(std::move(path)) {}
    ~TemporaryFileGuard()
    {
        if (armed_) Remove_File_If_Exists(path_);
    }

    TemporaryFileGuard(const TemporaryFileGuard&) = delete;
    TemporaryFileGuard& operator=(const TemporaryFileGuard&) = delete;

    void Release() { armed_ = false; }

   private:
    std::string path_;
    bool armed_ = true;
};
}  // namespace SpongeH5MD

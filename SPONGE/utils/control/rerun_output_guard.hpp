#pragma once

#include <filesystem>
#include <string>
#include <system_error>

namespace SpongeRerunOutput
{
inline bool Same_File(const std::string& output, const std::string& input)
{
    if (output == input) return true;
    std::error_code error;
    // equivalent also detects symbolic links and hard links to an existing input.
    return std::filesystem::equivalent(output, input, error) && !error;
}
}

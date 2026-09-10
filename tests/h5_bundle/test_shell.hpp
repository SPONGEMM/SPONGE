#pragma once

#include <cstdlib>
#include <filesystem>
#include <string>

inline std::string Shell_Quote(const std::filesystem::path& path)
{
#ifdef _WIN32
    return "\"" + path.string() + "\"";
#else
    std::string quoted = "'";
    for (const char c : path.string())
    {
        quoted += c == '\'' ? "'\\''" : std::string(1, c);
    }
    return quoted + "'";
#endif
}

inline int Run_Test_Shell_Command(const std::string& command)
{
#ifdef _WIN32
    // cmd.exe strips the outer quotes when the executable path is quoted.
    return std::system(("\"" + command + "\"").c_str());
#else
    return std::system(command.c_str());
#endif
}

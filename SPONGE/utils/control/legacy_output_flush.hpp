#pragma once

#include <cstdio>
#include <map>
#include <sstream>
#include <string>

namespace SpongeLegacyIO
{
class OutputFlushCoordinator
{
   public:
    static void Mark_Dirty(FILE* file, const char* stream_name)
    {
        if (file == nullptr) return;
        Dirty_Files()[file] =
            stream_name == nullptr ? "legacy output" : stream_name;
    }

    static bool Flush_Dirty(std::string* error_message = nullptr)
    {
        bool ok = true;
        std::ostringstream errors;
        for (const auto& entry : Dirty_Files())
        {
            if (std::fflush(entry.first) == 0) continue;
            if (!ok) errors << ", ";
            errors << entry.second;
            ok = false;
        }
        Dirty_Files().clear();
        if (!ok && error_message != nullptr)
        {
            *error_message =
                "failed to flush legacy output streams: " + errors.str();
        }
        return ok;
    }

    static std::size_t Dirty_Count() { return Dirty_Files().size(); }

   private:
    static std::map<FILE*, std::string>& Dirty_Files()
    {
        static std::map<FILE*, std::string> files;
        return files;
    }
};
}  // namespace SpongeLegacyIO

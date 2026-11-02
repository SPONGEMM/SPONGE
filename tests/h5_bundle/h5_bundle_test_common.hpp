#pragma once

#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

static constexpr int spongeErrorValueErrorCommand = 1;

class CONTROLLER
{
   public:
    bool Command_Exist(const char* key) const
    {
        return key != nullptr && commands_.count(key) != 0;
    }

    const char* Command(const char* key) const
    {
        const auto iter = commands_.find(key == nullptr ? "" : key);
        if (iter == commands_.end())
        {
            return "";
        }
        return iter->second.c_str();
    }

    void Throw_SPONGE_Error(int, const char* location, const char* message)
    {
        std::ostringstream out;
        out << location << ": " << message;
        throw std::runtime_error(out.str());
    }

    void Set(const std::string& key, const std::string& value)
    {
        commands_[key] = value;
        checks_[key] = 1;
    }

    void Set_Command(const char* key, const char* value, int check = 1)
    {
        const std::string normalized_key = key == nullptr ? "" : key;
        commands_[normalized_key] = value == nullptr ? "" : value;
        checks_[normalized_key] = check;
    }

    int Check_Value(const std::string& key) const
    {
        const auto iter = checks_.find(key);
        return iter == checks_.end() ? -1 : iter->second;
    }

   private:
    std::map<std::string, std::string> commands_;
    std::map<std::string, int> checks_;
};

#include "utils/control/h5_output_contract.hpp"
#include "utils/h5md/h5md_writer.hpp"

struct TestFailure : std::runtime_error
{
    using std::runtime_error::runtime_error;
};

#define REQUIRE_TRUE(expr)                                    \
    do                                                        \
    {                                                         \
        if (!(expr))                                          \
        {                                                     \
            std::ostringstream require_message;               \
            require_message << __FILE__ << ":" << __LINE__    \
                            << " requirement failed: " #expr; \
            throw TestFailure(require_message.str());         \
        }                                                     \
    } while (false)

#define REQUIRE_EQ(lhs, rhs)                                          \
    do                                                                \
    {                                                                 \
        const auto require_lhs = (lhs);                               \
        const auto require_rhs = (rhs);                               \
        if (!(require_lhs == require_rhs))                            \
        {                                                             \
            std::ostringstream require_message;                       \
            require_message << __FILE__ << ":" << __LINE__            \
                            << " equality failed: " #lhs " == " #rhs; \
            throw TestFailure(require_message.str());                 \
        }                                                             \
    } while (false)

namespace SpongeH5Test
{
using SpongeH5MD::DatasetSpec;
using SpongeH5MD::DataType;
using SpongeH5MD::FileStatus;
using SpongeH5MD::VirtualDatasetSource;
using SpongeH5MD::WriterBackend;
using SpongeH5MD::WriterBackendFactory;
using SpongeH5MD::WriterOptions;

struct BackendLog
{
    bool opened = false;
    bool closed = false;
    bool finalized = false;
    bool fail_finalize = false;
    bool fail_next_append = false;
    bool fail_next_virtual_dataset = false;
    bool fail_next_string_array = false;
    std::int64_t flush_calls = 0;
    std::string fail_string_path;
    FileStatus status = FileStatus::closed;
    std::string opened_path;
    std::string last_error;
    std::set<std::string> groups;
    std::map<std::string, DatasetSpec> datasets;
    std::map<std::string, std::vector<VirtualDatasetSource>> virtual_datasets;
    std::vector<std::pair<std::string, std::string>> hard_links;
    std::map<std::string, std::int64_t> append_counts;
    std::map<std::string, std::vector<float>> float32_snapshots;
    std::map<std::string, std::string> strings;
    std::map<std::string, std::vector<std::string>> string_arrays;
    std::map<std::pair<std::string, std::string>, std::string>
        string_attributes;
};

class MockBackend : public WriterBackend
{
   public:
    explicit MockBackend(std::shared_ptr<BackendLog> log) : log_(std::move(log))
    {
    }

    bool Open(const WriterOptions& options) override
    {
        log_->opened = true;
        log_->opened_path = options.path;
        log_->status = FileStatus::open;
        return true;
    }

    bool Flush() override
    {
        ++log_->flush_calls;
        return true;
    }

    bool Close() override
    {
        log_->closed = true;
        log_->status = FileStatus::closed;
        return true;
    }

    bool Finalize() override
    {
        if (log_->fail_finalize)
        {
            log_->last_error = "mock finalize failure";
            log_->status = FileStatus::failed;
            return false;
        }
        log_->finalized = true;
        log_->status = FileStatus::finalized;
        log_->strings[SpongeH5MD::path::output_status] = "finalized";
        return true;
    }

    bool Ensure_Group(const std::string& path) override
    {
        log_->groups.insert(path);
        return true;
    }

    bool Create_Dataset(const DatasetSpec& spec) override
    {
        log_->datasets[spec.path] = spec;
        return true;
    }

    bool Create_Virtual_Dataset(
        const DatasetSpec& spec,
        const std::vector<VirtualDatasetSource>& sources) override
    {
        if (log_->fail_next_virtual_dataset)
        {
            log_->fail_next_virtual_dataset = false;
            log_->last_error = "mock virtual dataset failure: " + spec.path;
            return false;
        }
        log_->datasets[spec.path] = spec;
        log_->virtual_datasets[spec.path] = sources;
        return true;
    }

    bool Create_Hard_Link(const std::string& target,
                          const std::string& link_path) override
    {
        log_->hard_links.push_back({target, link_path});
        return true;
    }

    bool Append_Int64(const std::string& path, const int64_t*,
                      std::size_t count) override
    {
        return Append(path, count);
    }

    bool Append_Float32(const std::string& path, const float*,
                        std::size_t count) override
    {
        return Append(path, count);
    }

    bool Append_Float64(const std::string& path, const double*,
                        std::size_t count) override
    {
        return Append(path, count);
    }

    bool Write_Float32(const std::string& path, const float* data,
                       std::size_t count) override
    {
        if (data == nullptr || count == 0)
        {
            log_->last_error = "mock float32 snapshot data is empty: " + path;
            return false;
        }
        log_->float32_snapshots[path] = std::vector<float>(data, data + count);
        return true;
    }

    bool Write_String(const std::string& path,
                      const std::string& value) override
    {
        if (!log_->fail_string_path.empty() && log_->fail_string_path == path)
        {
            log_->last_error = "mock string failure: " + path;
            log_->fail_string_path.clear();
            return false;
        }
        log_->strings[path] = value;
        return true;
    }

    bool Write_String_Array(const std::string& path,
                            const std::vector<std::string>& values) override
    {
        if (log_->fail_next_string_array)
        {
            log_->fail_next_string_array = false;
            log_->last_error = "mock string-array failure: " + path;
            return false;
        }
        log_->string_arrays[path] = values;
        return true;
    }

    bool Set_String_Attribute(const std::string& object_path,
                              const std::string& name,
                              const std::string& value) override
    {
        log_->string_attributes[{object_path, name}] = value;
        return true;
    }

    bool Set_Status(FileStatus status) override
    {
        log_->status = status;
        switch (status)
        {
            case FileStatus::closed:
                log_->strings[SpongeH5MD::path::output_status] = "closed";
                break;
            case FileStatus::open:
                log_->strings[SpongeH5MD::path::output_status] = "open";
                break;
            case FileStatus::closing:
                log_->strings[SpongeH5MD::path::output_status] = "closing";
                break;
            case FileStatus::finalized:
                log_->strings[SpongeH5MD::path::output_status] = "finalized";
                break;
            case FileStatus::failed:
                log_->strings[SpongeH5MD::path::output_status] = "failed";
                break;
        }
        return true;
    }

    FileStatus Status() const override { return log_->status; }

    std::string Last_Error() const override { return log_->last_error; }

   private:
    bool Append(const std::string& path, std::size_t count)
    {
        if (log_->fail_next_append)
        {
            log_->fail_next_append = false;
            log_->last_error = "mock append failure";
            return false;
        }
        log_->append_counts[path] += static_cast<std::int64_t>(count);
        return true;
    }

    std::shared_ptr<BackendLog> log_;
};

class MockBackendFactory : public WriterBackendFactory
{
   public:
    std::unique_ptr<WriterBackend> Create_Backend() override
    {
        auto log = std::make_shared<BackendLog>();
        if (logs.size() < fail_finalize.size() && fail_finalize[logs.size()])
        {
            log->fail_finalize = true;
        }
        logs.push_back(log);
        return std::unique_ptr<WriterBackend>(new MockBackend(log));
    }

    std::vector<std::shared_ptr<BackendLog>> logs;
    std::vector<bool> fail_finalize;
};

inline bool Has_Hard_Link(const BackendLog& log, const std::string& target,
                          const std::string& link)
{
    for (const auto& pair : log.hard_links)
    {
        if (pair.first == target && pair.second == link)
        {
            return true;
        }
    }
    return false;
}

inline std::filesystem::path Unique_Temp_Path(const std::string& name)
{
    const auto stamp =
        std::chrono::high_resolution_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path() /
           ("sponge_h5_bundle_" + std::to_string(stamp) + "_" + name);
}

template <typename Fn>
int Run_Test(Fn&& fn)
{
    try
    {
        fn();
        return 0;
    }
    catch (const std::exception& err)
    {
        std::cerr << err.what() << "\n";
        return 1;
    }
}
}  // namespace SpongeH5Test

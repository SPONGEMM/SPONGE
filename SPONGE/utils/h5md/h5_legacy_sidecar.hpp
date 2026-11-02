#pragma once

#include <algorithm>
#include <filesystem>
#include <highfive/highfive.hpp>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "utils/h5md/h5_legacy_sidecar_contract.hpp"
#include "utils/h5md/h5md_writer.hpp"

namespace SpongeH5MD
{
class LegacySidecarH5Reader
{
   public:
    bool Open(const std::string& file_path)
    {
        last_error_.clear();
        container_path_ = file_path;
        try
        {
            file_.reset(
                new HighFive::File(file_path, HighFive::File::ReadOnly));
            return true;
        }
        catch (const std::exception& err)
        {
            return Fail(std::string("failed to open H5 sidecar file: ") +
                        err.what());
        }
    }

    bool Read_Legacy_Sidecars(std::vector<LegacySidecarBinding>* sidecars)
    {
        if (sidecars == nullptr)
        {
            return Fail("legacy sidecar output pointer is null");
        }
        sidecars->clear();
        if (!Ensure_File()) return false;

        try
        {
            if (!Exists(path::legacy_sidecar_keys) &&
                !Exists(path::legacy_sidecar_paths))
            {
                return true;
            }
            if (!Exists(path::legacy_sidecar_keys) ||
                !Exists(path::legacy_sidecar_paths))
            {
                return Fail(
                    "legacy sidecar key/path datasets must be present "
                    "together");
            }
            std::vector<std::string> keys;
            std::vector<std::string> paths;
            file_->getDataSet(path::legacy_sidecar_keys).read(keys);
            file_->getDataSet(path::legacy_sidecar_paths).read(paths);
            if (keys.size() != paths.size())
            {
                return Fail("legacy sidecar key/path dataset length mismatch");
            }

            sidecars->reserve(keys.size());
            for (std::size_t i = 0; i < keys.size(); ++i)
            {
                if (keys[i].empty())
                {
                    return Fail("legacy sidecar key must not be empty");
                }
                if (paths[i].empty())
                {
                    return Fail("legacy sidecar path must not be empty");
                }
                sidecars->push_back(
                    {keys[i], Resolve_Container_Relative_Path(paths[i])});
            }
            return true;
        }
        catch (const std::exception& err)
        {
            return Fail(std::string("failed to read legacy sidecars: ") +
                        err.what());
        }
    }

    std::string Last_Error() const { return last_error_; }

   private:
    bool Ensure_File()
    {
        if (file_ == nullptr)
        {
            return Fail("H5 sidecar reader is not open");
        }
        return true;
    }

    bool Exists(const std::string& object_path) const
    {
        return file_ != nullptr && file_->exist(object_path);
    }

    std::string Resolve_Container_Relative_Path(
        const std::string& raw_path) const
    {
        const std::filesystem::path sidecar_path(raw_path);
        if (sidecar_path.is_absolute())
        {
            return sidecar_path.lexically_normal().string();
        }
        const auto base =
            std::filesystem::absolute(std::filesystem::path(container_path_))
                .parent_path();
        return (base / sidecar_path).lexically_normal().string();
    }

    bool Fail(const std::string& message)
    {
        last_error_ = message;
        return false;
    }

    std::unique_ptr<HighFive::File> file_;
    std::string container_path_;
    std::string last_error_;
};

inline bool Read_Legacy_Sidecars_From_H5(
    const std::string& file_path, std::vector<LegacySidecarBinding>* sidecars,
    std::string* error_message)
{
    LegacySidecarH5Reader reader;
    if (!reader.Open(file_path))
    {
        if (error_message != nullptr) *error_message = reader.Last_Error();
        return false;
    }
    if (!reader.Read_Legacy_Sidecars(sidecars))
    {
        if (error_message != nullptr) *error_message = reader.Last_Error();
        return false;
    }
    return true;
}

inline std::string Normalize_Sidecar_Path_For_Comparison(
    const std::string& path)
{
    return std::filesystem::absolute(std::filesystem::path(path))
        .lexically_normal()
        .string();
}

inline bool Sidecar_Paths_Equivalent(const std::string& lhs,
                                     const std::string& rhs)
{
    return lhs == rhs || Normalize_Sidecar_Path_For_Comparison(lhs) ==
                             Normalize_Sidecar_Path_For_Comparison(rhs);
}

template <typename ControllerType>
inline bool Inject_Legacy_Sidecar_Commands(
    ControllerType* controller,
    const std::vector<LegacySidecarBinding>& sidecars,
    const std::set<std::string>& allowed_keys, const std::string& source_label,
    std::string* error_message)
{
    auto fail = [error_message](const std::string& message)
    {
        if (error_message != nullptr)
        {
            *error_message = message;
        }
        return false;
    };

    if (controller == nullptr)
    {
        return fail("controller pointer is null");
    }

    for (const auto& sidecar : sidecars)
    {
        if (!Command_Key_Allowed(allowed_keys, sidecar.key))
        {
            return fail("unsupported H5 legacy sidecar key in " + source_label +
                        ": " + sidecar.key);
        }
        if (controller->Command_Exist(sidecar.key.c_str()))
        {
            const std::string existing =
                controller->Command(sidecar.key.c_str());
            if (Sidecar_Paths_Equivalent(existing, sidecar.path))
            {
                continue;
            }
            return fail(
                "H5 legacy sidecar key conflicts with existing command " +
                sidecar.key + ": existing=" + existing +
                ", h5=" + sidecar.path);
        }
        controller->Set_Command(sidecar.key.c_str(), sidecar.path.c_str(), 0);
    }
    return true;
}

template <typename ControllerType>
inline bool Inject_Legacy_Sidecar_Commands_From_H5(
    ControllerType* controller, const std::string& file_path,
    const std::set<std::string>& allowed_keys, const std::string& source_label,
    std::string* error_message)
{
    std::vector<LegacySidecarBinding> sidecars;
    if (!Read_Legacy_Sidecars_From_H5(file_path, &sidecars, error_message))
    {
        return false;
    }
    return Inject_Legacy_Sidecar_Commands(controller, sidecars, allowed_keys,
                                          source_label, error_message);
}
}  // namespace SpongeH5MD

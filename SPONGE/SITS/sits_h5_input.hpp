#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <highfive/highfive.hpp>
#include <iomanip>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../control.h"
#include "../utils/float_classification.hpp"
#include "../utils/h5md/input_plan.hpp"
#include "../utils/h5md/restart_h5_reader.hpp"

namespace SpongeH5MD
{
struct SitsH5Input
{
    bool has_atom_indices = false;
    std::vector<int> atom_indices;
    bool has_restart_state = false;
    RestartSitsState restart_state;
};

namespace SitsH5InputDetail
{
inline void Validate_Config_Token(const std::string& value,
                                  const std::string& path, bool is_key)
{
    if (value.empty() || value.find_first_of("\r\n{}") != std::string::npos ||
        (is_key && value.find('=') != std::string::npos))
    {
        throw std::runtime_error(path + " contains an invalid config token");
    }
}

template <typename T>
inline T Read_Scalar(HighFive::File* file, const std::string& path)
{
    auto dataset = file->getDataSet(path);
    if (!dataset.getSpace().getDimensions().empty())
    {
        throw std::runtime_error(path + " must be scalar");
    }
    T value{};
    dataset.read(value);
    return value;
}

template <typename T>
inline std::vector<T> Read_Vector(HighFive::File* file, const std::string& path)
{
    auto dataset = file->getDataSet(path);
    const auto dimensions = dataset.getSpace().getDimensions();
    if (dimensions.size() != 1)
    {
        throw std::runtime_error(path + " must be one-dimensional");
    }
    std::vector<T> values;
    dataset.read(values);
    return values;
}

inline void Set_If_Missing(CONTROLLER* controller,
                           const std::string& module_name,
                           const std::string& key, const std::string& value)
{
    if (!controller->Command_Exist(module_name.c_str(), key.c_str()))
    {
        controller->Set_Command(key.c_str(), value.c_str(), 0,
                                module_name.c_str());
    }
}

template <typename T>
inline std::string Numeric_String(T value)
{
    std::ostringstream out;
    out << std::setprecision(9) << value;
    return out.str();
}

inline void Load_Compatibility_Config(CONTROLLER* controller,
                                      HighFive::File* file,
                                      const std::string& module_name)
{
    constexpr const char* root = "/sits/config";
    if (!file->exist(root)) return;

    const auto declared_section_count =
        Read_Scalar<long long>(file, std::string(root) + "/section/count");
    const auto section_names =
        Read_Vector<std::string>(file, std::string(root) + "/section/name");
    const auto key_offsets = Read_Vector<std::int64_t>(
        file, std::string(root) + "/section/key_offset");
    const auto keys =
        Read_Vector<std::string>(file, std::string(root) + "/key");
    const auto values =
        Read_Vector<std::string>(file, std::string(root) + "/value");

    if (declared_section_count != 1 || section_names.size() != 1 ||
        section_names.front() != module_name)
    {
        throw std::runtime_error(
            "/sits/config must contain exactly the active SITS section");
    }
    if (key_offsets.size() != 2 || key_offsets.front() != 0 ||
        key_offsets.back() < 0 ||
        static_cast<std::size_t>(key_offsets.back()) != keys.size() ||
        keys.size() != values.size())
    {
        throw std::runtime_error(
            "/sits/config section offsets and key/value lengths are "
            "inconsistent");
    }
    Validate_Config_Token(section_names.front(), "/sits/config/section/name",
                          true);
    std::set<std::string> unique_keys;
    for (std::size_t item = 0; item < keys.size(); ++item)
    {
        Validate_Config_Token(keys[item], "/sits/config/key", true);
        Validate_Config_Token(values[item], "/sits/config/value", false);
        if (!unique_keys.insert(keys[item]).second)
        {
            throw std::runtime_error(
                "/sits/config/key contains a duplicate key");
        }
        Set_If_Missing(controller, module_name, keys[item], values[item]);
    }
    if (file->exist("/sits/atom_indices") &&
        (unique_keys.count("atom_in_file") ||
         unique_keys.count("atom_numbers")))
    {
        throw std::runtime_error(
            "/sits/config atom ownership conflicts with /sits/atom_indices");
    }
}

inline void Load_Typed_Method(CONTROLLER* controller, HighFive::File* file,
                              const std::string& module_name)
{
    constexpr const char* root = "/sits/method";
    if (!file->exist(root)) return;
    if (file->exist("/sits/enabled_default") &&
        Read_Scalar<int>(file, "/sits/enabled_default") == 0)
    {
        return;
    }
    if (file->exist(std::string(root) + "/schema_version") &&
        Read_Scalar<std::int64_t>(file,
                                  std::string(root) + "/schema_version") != 1)
    {
        throw std::runtime_error("unsupported /sits/method/schema_version");
    }
    if (!file->exist(std::string(root) + "/mode"))
    {
        throw std::runtime_error("/sits/method/mode is required");
    }
    const std::string mode =
        Read_Scalar<std::string>(file, std::string(root) + "/mode");
    const std::set<std::string> modes = {
        "observation", "iteration", "production", "empirical", "amd", "gamd"};
    if (!modes.count(mode))
    {
        throw std::runtime_error("/sits/method/mode is invalid");
    }
    Set_If_Missing(controller, module_name, "mode", mode);

    const auto inject_int =
        [&](const char* field, bool positive, const char* runtime_key = nullptr)
    {
        const std::string path = std::string(root) + "/" + field;
        if (!file->exist(path)) return;
        const auto value = Read_Scalar<std::int64_t>(file, path);
        if (positive && value <= 0)
        {
            throw std::runtime_error(path + " must be positive");
        }
        Set_If_Missing(controller, module_name,
                       runtime_key == nullptr ? field : runtime_key,
                       Numeric_String(value));
    };
    const auto inject_float =
        [&](const char* field, const char* runtime_key = nullptr)
    {
        const std::string path = std::string(root) + "/" + field;
        if (!file->exist(path)) return;
        const float value = Read_Scalar<float>(file, path);
        if (!SpongeFloat::Is_Finite(value))
        {
            throw std::runtime_error(path + " must be finite");
        }
        Set_If_Missing(controller, module_name,
                       runtime_key == nullptr ? field : runtime_key,
                       Numeric_String(value));
    };
    const auto inject_bool = [&](const char* field)
    {
        const std::string path = std::string(root) + "/" + field;
        if (!file->exist(path)) return;
        const auto value = Read_Scalar<std::int64_t>(file, path);
        if (value != 0 && value != 1)
        {
            throw std::runtime_error(path + " must be 0 or 1");
        }
        Set_If_Missing(controller, module_name, field, Numeric_String(value));
    };

    inject_int("k_numbers", true);
    inject_int("fb_interval", true);
    inject_int("record_interval", true);
    inject_int("update_interval", true);
    inject_bool("nk_rest");
    inject_bool("nk_fix");
    inject_float("temperature_low", "T_low");
    inject_float("temperature_high", "T_high");
    inject_float("pe_a");
    inject_float("pe_b");
    inject_float("fb_bias");
    inject_float("cross_enhance_factor");

    const std::string ladder_path = std::string(root) + "/temperature_ladder";
    if (file->exist(ladder_path))
    {
        const auto ladder = Read_Vector<float>(file, ladder_path);
        if (ladder.empty())
        {
            throw std::runtime_error(ladder_path + " must not be empty");
        }
        std::ostringstream value;
        value << std::setprecision(9);
        for (std::size_t index = 0; index < ladder.size(); ++index)
        {
            if (!(ladder[index] > 0.0f) ||
                !SpongeFloat::Is_Finite(ladder[index]))
            {
                throw std::runtime_error(
                    ladder_path + " must contain finite positive values");
            }
            if (index != 0) value << '/';
            value << ladder[index];
        }
        Set_If_Missing(controller, module_name, "T", value.str());
    }
}

inline void Load_Atom_Selection(CONTROLLER* controller, HighFive::File* file,
                                const std::string& module_name,
                                int atom_numbers, SitsH5Input* input)
{
    if (file->exist("/sits/enabled_default") &&
        Read_Scalar<int>(file, "/sits/enabled_default") == 0)
    {
        return;
    }
    if (controller->Command_Exist(module_name.c_str(), "atom_in_file") ||
        controller->Command_Exist(module_name.c_str(), "atom_numbers"))
    {
        return;
    }
    if (file->exist("/sits/atom_indices") &&
        file->exist("/sits/atom_numbers_policy"))
    {
        throw std::runtime_error(
            "/sits must not define both atom_indices and atom_numbers_policy");
    }
    if (file->exist("/sits/atom_indices"))
    {
        input->atom_indices = Read_Vector<int>(file, "/sits/atom_indices");
        if (input->atom_indices.empty())
        {
            throw std::runtime_error("/sits/atom_indices must not be empty");
        }
        std::set<int> unique_indices;
        for (std::size_t row = 0; row < input->atom_indices.size(); ++row)
        {
            const int atom = input->atom_indices[row];
            if (atom < 0 || atom >= atom_numbers)
            {
                throw std::runtime_error(
                    "/sits/atom_indices contains an out-of-range atom");
            }
            if (!unique_indices.insert(atom).second)
            {
                throw std::runtime_error(
                    "/sits/atom_indices contains a duplicate atom");
            }
        }
        input->has_atom_indices = true;
        return;
    }
    if (file->exist("/sits/atom_numbers_policy"))
    {
        const auto dataset = file->getDataSet("/sits/atom_numbers_policy");
        const auto type_class = H5Tget_class(dataset.getDataType().getId());
        std::string policy;
        if (type_class == H5T_STRING)
        {
            policy =
                Read_Scalar<std::string>(file, "/sits/atom_numbers_policy");
            if (policy != "ITS" && policy != "ALL")
            {
                throw std::runtime_error(
                    "/sits/atom_numbers_policy string must be ITS or ALL");
            }
        }
        else
        {
            const auto count =
                Read_Scalar<std::int64_t>(file, "/sits/atom_numbers_policy");
            if (count <= 0 || count > atom_numbers)
            {
                throw std::runtime_error(
                    "/sits/atom_numbers_policy integer is out of range");
            }
            policy = Numeric_String(count);
        }
        Set_If_Missing(controller, module_name, "atom_numbers", policy);
    }
}

inline void Load_Restart_State(CONTROLLER* controller,
                               const std::string& module_name,
                               SitsH5Input* input)
{
    if (controller->Command_Exist(module_name.c_str(), "nk_in_file")) return;
    if (!controller->Command_Exist("input_h5_restart_path")) return;
    const auto plan = SpongeH5InputPlan::Resolve_Input_Plan(controller);
    if (!plan.valid)
    {
        throw std::runtime_error(plan.error_message);
    }
    if (!plan.restart.binding.enabled ||
        (plan.restart.load_policy !=
             SpongeH5InputContract::RestartLoadPolicy::protocol &&
         plan.restart.load_policy !=
             SpongeH5InputContract::RestartLoadPolicy::full))
    {
        return;
    }
    RestartH5Reader reader;
    if (!reader.Open(plan.restart.binding.path))
    {
        throw std::runtime_error(reader.Last_Error());
    }
    RestartProtocolState state;
    if (!reader.Read_Protocol_State(&state))
    {
        throw std::runtime_error(reader.Last_Error());
    }
    const auto found =
        std::find_if(state.sits_states.begin(), state.sits_states.end(),
                     [&module_name](const RestartSitsState& candidate)
                     { return candidate.module_name == module_name; });
    if (found != state.sits_states.end())
    {
        input->restart_state = *found;
        input->has_restart_state = true;
    }
}
}  // namespace SitsH5InputDetail

inline SitsH5Input Load_H5_SITS_Input(CONTROLLER* controller,
                                      const char* module_name, int atom_numbers)
{
    SitsH5Input input;
    constexpr const char* input_key = "input_h5_protocol_path";
    if (!controller->Command_Exist(input_key)) return input;

    try
    {
        HighFive::File file(controller->Command(input_key),
                            HighFive::File::ReadOnly);
        if (!file.exist("/sits")) return input;
        if (file.exist("/sits/method"))
        {
            SitsH5InputDetail::Load_Typed_Method(controller, &file,
                                                 module_name);
        }
        else
        {
            SitsH5InputDetail::Load_Compatibility_Config(controller, &file,
                                                         module_name);
        }
        SitsH5InputDetail::Load_Atom_Selection(controller, &file, module_name,
                                               atom_numbers, &input);
        SitsH5InputDetail::Load_Restart_State(controller, module_name, &input);
        return input;
    }
    catch (const std::exception& error)
    {
        const std::string message =
            std::string("Reason:\n\tfailed to read typed SITS input: ") +
            error.what() + "\n";
        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                       "Load_H5_SITS_Input", message.c_str());
        return input;
    }
}
}  // namespace SpongeH5MD

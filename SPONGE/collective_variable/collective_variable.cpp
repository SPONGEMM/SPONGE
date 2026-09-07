#include "collective_variable.h"

#include <algorithm>
#include <cstdint>
#include <highfive/highfive.hpp>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "utils/h5md/protocol_cv_h5.hpp"
#include "utils/h5md/protocol_metadynamics_h5.hpp"
#include "utils/h5md/protocol_restraint_h5.hpp"
#include "utils/h5md/protocol_steer_h5.hpp"

namespace
{
struct CVConfigSection
{
    std::string name;
    std::vector<std::pair<std::string, std::string>> items;
};

std::vector<std::string> Read_H5_String_Vector(HighFive::File* file,
                                               const std::string& path)
{
    HighFive::DataSet dataset = file->getDataSet(path);
    const auto dimensions = dataset.getSpace().getDimensions();
    if (dimensions.size() != 1)
    {
        throw std::runtime_error(path + " must be a one-dimensional dataset");
    }
    std::vector<std::string> values;
    dataset.read(values);
    return values;
}

std::vector<std::int64_t> Read_H5_Int64_Vector(HighFive::File* file,
                                               const std::string& path)
{
    HighFive::DataSet dataset = file->getDataSet(path);
    const auto dimensions = dataset.getSpace().getDimensions();
    if (dimensions.size() != 1)
    {
        throw std::runtime_error(path + " must be a one-dimensional dataset");
    }
    std::vector<std::int64_t> values;
    dataset.read(values);
    return values;
}

void Validate_CV_Config_Token(const std::string& value, const std::string& path,
                              bool is_key)
{
    if (value.empty() || value.find_first_of("\r\n{}") != std::string::npos ||
        (is_key && value.find('=') != std::string::npos))
    {
        throw std::runtime_error(path + " contains an invalid config token");
    }
}

std::vector<CVConfigSection> Read_H5_CV_Config(HighFive::File* file,
                                               const std::string& root)
{
    const std::string count_path = root + "/section/count";
    HighFive::DataSet count_dataset = file->getDataSet(count_path);
    if (!count_dataset.getSpace().getDimensions().empty())
    {
        throw std::runtime_error(count_path + " must be a scalar dataset");
    }
    long long declared_section_count = 0;
    count_dataset.read(declared_section_count);
    const auto section_names =
        Read_H5_String_Vector(file, root + "/section/name");
    const auto key_offsets =
        Read_H5_Int64_Vector(file, root + "/section/key_offset");
    const auto keys = Read_H5_String_Vector(file, root + "/key");
    const auto values = Read_H5_String_Vector(file, root + "/value");

    if (declared_section_count <= 0 ||
        section_names.size() !=
            static_cast<std::size_t>(declared_section_count))
    {
        throw std::runtime_error(root +
                                 "/section/count must match a non-empty "
                                 "section/name vector");
    }
    if (key_offsets.size() != section_names.size() + 1 ||
        key_offsets.front() != 0 || key_offsets.back() < 0 ||
        static_cast<std::size_t>(key_offsets.back()) != keys.size() ||
        keys.size() != values.size())
    {
        throw std::runtime_error(
            root + " section offsets and key/value lengths are inconsistent");
    }

    std::vector<CVConfigSection> sections;
    std::set<std::string> unique_section_names;
    for (std::size_t section = 0; section < section_names.size(); ++section)
    {
        Validate_CV_Config_Token(section_names[section], root + "/section/name",
                                 true);
        if (!unique_section_names.insert(section_names[section]).second)
        {
            throw std::runtime_error(root +
                                     "/section/name contains a duplicate "
                                     "section");
        }
        if (key_offsets[section] < 0 ||
            key_offsets[section] > key_offsets[section + 1])
        {
            throw std::runtime_error(root +
                                     "/section/key_offset must be monotonic");
        }

        CVConfigSection parsed{section_names[section], {}};
        std::set<std::string> unique_keys;
        for (std::int64_t item = key_offsets[section];
             item < key_offsets[section + 1]; ++item)
        {
            const auto item_index = static_cast<std::size_t>(item);
            Validate_CV_Config_Token(keys[item_index], root + "/key", true);
            Validate_CV_Config_Token(values[item_index], root + "/value",
                                     false);
            if (!unique_keys.insert(keys[item_index]).second)
            {
                throw std::runtime_error(root +
                                         "/key contains a duplicate key in "
                                         "section " +
                                         section_names[section]);
            }
            parsed.items.push_back({keys[item_index], values[item_index]});
        }
        sections.push_back(std::move(parsed));
    }
    return sections;
}

void Merge_H5_CV_Config(const std::string& root,
                        const std::vector<CVConfigSection>& incoming,
                        std::vector<CVConfigSection>* merged)
{
    for (const auto& section : incoming)
    {
        auto existing = std::find_if(merged->begin(), merged->end(),
                                     [&](const CVConfigSection& value)
                                     { return value.name == section.name; });
        if (existing == merged->end())
        {
            merged->push_back(section);
        }
        else if (existing->items.size() != section.items.size() ||
                 !std::all_of(existing->items.begin(), existing->items.end(),
                              [&](const auto& item)
                              {
                                  return std::find(section.items.begin(),
                                                   section.items.end(),
                                                   item) != section.items.end();
                              }))
        {
            throw std::runtime_error(
                root + " conflicts with another typed CV definition for " +
                section.name);
        }
    }
}

void Set_CV_Config_Value(COLLECTIVE_VARIABLE_CONTROLLER* manager,
                         const std::string& section, const std::string& flag,
                         const std::string& value,
                         const std::string& source_path)
{
    const std::string full_key =
        section.empty() || section == "main" ? flag : section + "_" + flag;
    const auto existing = manager->original_commands.find(full_key);
    if (existing != manager->original_commands.end())
    {
        if (string_strip(existing->second) == value) return;
        manager->Throw_SPONGE_Error(
            spongeErrorConflictingCommand,
            "COLLECTIVE_VARIABLE_CONTROLLER::Load_H5_CV_Config",
            string_format(
                "Reason:\n\tCV command '%COMMAND%' has conflicting values "
                "while merging %SOURCE%\n",
                {{"COMMAND", full_key}, {"SOURCE", source_path}})
                .c_str());
    }
    manager->Set_Command(flag.c_str(), value.c_str(), 1,
                         section.empty() ? nullptr : section.c_str());
}

bool Load_H5_CV_Config(CONTROLLER* controller,
                       COLLECTIVE_VARIABLE_CONTROLLER* manager)
{
    constexpr const char* input_key = "input_h5_protocol_path";
    if (!controller->Command_Exist(input_key))
    {
        return false;
    }

    try
    {
        auto file = std::make_shared<HighFive::File>(
            controller->Command(input_key), HighFive::File::ReadOnly);
        constexpr const char* cv_root = "/cv/config";
        constexpr const char* restraint_root = "/restraint/config";
        constexpr const char* restraint_cv_root = "/restraint/cv/config";
        const bool has_cv = file->exist(cv_root);
        const bool has_restraint = file->exist(restraint_root);
        const bool has_restraint_cv = file->exist(restraint_cv_root);
        SpongeH5MD::ProtocolCVH5Reader cv_reader;
        std::vector<SpongeH5MD::ProtocolCVDefinition> typed_cvs;
        std::vector<SpongeH5MD::ProtocolVirtualAtomDefinition>
            typed_virtual_atoms;
        if (!cv_reader.Open_Protocol(file))
        {
            throw std::runtime_error(cv_reader.Last_Error());
        }
        if (controller->Command_Exist("input_h5_restart_path") &&
            !cv_reader.Open_Restart(
                controller->Command("input_h5_restart_path")))
        {
            throw std::runtime_error(cv_reader.Last_Error());
        }
        if (!cv_reader.Read_Definitions(
                static_cast<std::size_t>(manager->atom_numbers), &typed_cvs))
        {
            throw std::runtime_error(cv_reader.Last_Error());
        }
        if (!cv_reader.Read_Virtual_Atoms(
                static_cast<std::size_t>(manager->atom_numbers),
                &typed_virtual_atoms))
        {
            throw std::runtime_error(cv_reader.Last_Error());
        }
        SpongeH5MD::ProtocolRestraintH5Reader restraint_reader;
        std::vector<SpongeH5MD::ProtocolCVRestraint> typed_restraints;
        if (!restraint_reader.Open(file) ||
            !restraint_reader.Read_CV_Restraints(&typed_restraints))
        {
            throw std::runtime_error(restraint_reader.Last_Error());
        }
        SpongeH5MD::ProtocolMetadynamicsH5Reader metadynamics_reader;
        SpongeH5MD::ProtocolMetadynamicsDefinition typed_metadynamics;
        bool has_typed_metadynamics = false;
        if (!metadynamics_reader.Open(file) ||
            !metadynamics_reader.Read_Definition(typed_cvs, &typed_metadynamics,
                                                 &has_typed_metadynamics))
        {
            throw std::runtime_error(metadynamics_reader.Last_Error());
        }
        SpongeH5MD::ProtocolSteeringH5Reader steering_reader;
        SpongeH5MD::ProtocolSteeringDefinition typed_steering;
        bool has_typed_steering = false;
        if (!steering_reader.Open(file) ||
            !steering_reader.Read_Definition(typed_cvs, &typed_steering,
                                             &has_typed_steering))
        {
            throw std::runtime_error(steering_reader.Last_Error());
        }
        if (!has_cv && !has_restraint && !has_restraint_cv &&
            typed_cvs.empty() && typed_virtual_atoms.empty() &&
            typed_restraints.empty() && !has_typed_metadynamics &&
            !has_typed_steering)
        {
            return false;
        }
        const bool has_legacy_cv = controller->Command_Exist("cv_in_file");
        const bool has_legacy_restraint =
            controller->Command_Exist("restrain_in_file") ||
            controller->Command_Exist("restrain_cv_in_file");
        if (has_legacy_cv || has_legacy_restraint)
        {
            return false;
        }

        std::vector<CVConfigSection> sections;
        for (const auto& root : {cv_root, restraint_root, restraint_cv_root})
        {
            if (file->exist(root))
            {
                Merge_H5_CV_Config(root, Read_H5_CV_Config(file.get(), root),
                                   &sections);
            }
        }
        for (const auto& definition : typed_virtual_atoms)
        {
            CVConfigSection native{definition.name,
                                   {{"vatom_type", definition.type}}};
            Merge_H5_CV_Config("/cv/virtual_atom/<name>", {native}, &sections);
            manager->protocol_virtual_atom_indices[definition.name] =
                definition.atom_indices;
            if (!definition.weight.empty())
                manager->protocol_virtual_atom_weight[definition.name] =
                    definition.weight;
        }
        for (const auto& definition : typed_cvs)
        {
            CVConfigSection native{definition.name,
                                   definition.runtime_parameters};
            Merge_H5_CV_Config("/cv/<name>", {native}, &sections);
            if (!definition.period.empty())
            {
                manager->protocol_cv_period[definition.name] =
                    definition.period;
            }
            if (!definition.sigma.empty())
            {
                manager->protocol_cv_sigma[definition.name] = definition.sigma;
            }
            if (!definition.reference_coordinates.empty())
            {
                manager->protocol_cv_reference[definition.name] =
                    definition.reference_coordinates;
            }
        }
        if (!typed_restraints.empty())
        {
            CVConfigSection typed{"restrain", {}};
            std::vector<std::string> cv_refs;
            std::vector<float> weight, reference, period;
            std::vector<std::int64_t> start_step, max_step, reduce_step,
                stop_step;
            for (const auto& restraint : typed_restraints)
            {
                cv_refs.insert(cv_refs.end(), restraint.cv_refs.begin(),
                               restraint.cv_refs.end());
                weight.insert(weight.end(), restraint.weight.begin(),
                              restraint.weight.end());
                reference.insert(reference.end(), restraint.reference.begin(),
                                 restraint.reference.end());
                period.insert(period.end(), restraint.period.begin(),
                              restraint.period.end());
                start_step.insert(start_step.end(),
                                  restraint.start_step.begin(),
                                  restraint.start_step.end());
                max_step.insert(max_step.end(), restraint.max_step.begin(),
                                restraint.max_step.end());
                reduce_step.insert(reduce_step.end(),
                                   restraint.reduce_step.begin(),
                                   restraint.reduce_step.end());
                stop_step.insert(stop_step.end(), restraint.stop_step.begin(),
                                 restraint.stop_step.end());
            }
            auto join_strings = [](const auto& values)
            {
                std::ostringstream out;
                out << std::setprecision(
                    std::numeric_limits<float>::max_digits10);
                for (std::size_t i = 0; i < values.size(); ++i)
                {
                    if (i != 0) out << ' ';
                    out << values[i];
                }
                return out.str();
            };
            typed.items = {{"CV", join_strings(cv_refs)},
                           {"weight", join_strings(weight)},
                           {"reference", join_strings(reference)},
                           {"period", join_strings(period)},
                           {"start_step", join_strings(start_step)},
                           {"max_step", join_strings(max_step)},
                           {"reduce_step", join_strings(reduce_step)},
                           {"stop_step", join_strings(stop_step)}};
            Merge_H5_CV_Config("/restraint/<name>", {typed}, &sections);
        }
        if (has_typed_metadynamics)
        {
            CVConfigSection typed{"meta",
                                  typed_metadynamics.runtime_parameters};
            Merge_H5_CV_Config("/meta/<name>", {typed}, &sections);
            manager->protocol_metadynamics_name = typed_metadynamics.name;
        }
        if (has_typed_steering)
        {
            CVConfigSection typed{"steer", typed_steering.runtime_parameters};
            Merge_H5_CV_Config("/steer", {typed}, &sections);
        }

        for (const auto& section : sections)
        {
            for (const auto& item : section.items)
            {
                Set_CV_Config_Value(
                    manager, section.name, item.first, item.second,
                    std::string(controller->Command(input_key)));
            }
        }
        return !sections.empty();
    }
    catch (const std::exception& error)
    {
        const std::string message =
            std::string("Reason:\n\tfailed to load typed CV objects from ") +
            controller->Command(input_key) + ": " + error.what() + "\n";
        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat,
                                       "Load_H5_CV_Config", message.c_str());
    }
    return false;
}
}  // namespace

CV_MAP_TYPE* CV_MAP = new CV_MAP_TYPE;
CV_INSTANCE_TYPE* CV_INSTANCE_MAP = new CV_INSTANCE_TYPE;

void COLLECTIVE_VARIABLE_CONTROLLER::Initial(
    CONTROLLER* controller, int* no_direct_interaction_virtual_atom_numbers)
{
    controller->printf("START INITIALIZING CV CONTROLLER:\n");
    strcpy(module_name, "cv_controller");
    this->controller = controller;
    mdinfo = controller->mdinfo;
    const bool has_h5_cv = Load_H5_CV_Config(controller, this);
    if (has_h5_cv || controller->Command_Exist("cv_in_file") ||
        controller->Command_Exist("restrain_in_file") ||
        controller->Command_Exist("restrain_cv_in_file"))
    {
        int CV_numbers = 0;
        Commands_From_In_File(controller);
        int cv_vatom_count = 0;
        for (StringMap::iterator iter = this->commands.begin();
             iter != this->commands.end(); iter++)
        {
            int i = iter->first.rfind("vatom_type");
            if (i > 0 && i == (iter->first.length() - 10))
            {
                cv_vatom_name[iter->first.substr(0, i - 1)] = cv_vatom_count;
                cv_vatom_index[cv_vatom_count] = iter->first.substr(0, i - 1);
                cv_vatom_count += 1;
                no_direct_interaction_virtual_atom_numbers[0]++;
            }
            i = iter->first.rfind("CV_type");
            if (i > 0 && i == (iter->first.length() - 7))
            {
                CV_numbers++;
            }
        }
        printf("    %d CV defined\n", CV_numbers);
        printf("    %d cv virtual atoms\n",
               no_direct_interaction_virtual_atom_numbers[0]);
        is_initialized = 1;
        controller->printf("END INITIALIZING CV CONTROLLER\n\n");
    }
    else
    {
        controller->printf("CV CONTROLLER IS NOT INITIALIZING\n\n");
    }
}

static int read_one_line(FILE* In_File, char* line, char* ender)
{
    int line_count = 0;
    int ender_count = 0;
    int c;
    while ((c = getc(In_File)) != EOF)
    {
        if (line_count == 0 && (c == '\t' || c == ' '))
        {
            continue;
        }
        else if (c != '\n' && c != ',' && c != '{' && c != '}' && c != '\r')
        {
            line[line_count] = c;
            line_count += 1;
        }
        else
        {
            ender[ender_count] = c;
            ender_count += 1;
            break;
        }
    }
    while ((c = getc(In_File)) != EOF)
    {
        if (c == ' ' || c == '\t')
        {
            continue;
        }
        else if (c != '\n' && c != ',' && c != '{' && c != '}' && c != '\r')
        {
            fseek(In_File, -1, SEEK_CUR);
            break;
        }
        else
        {
            ender[ender_count] = c;
            ender_count += 1;
        }
    }
    line[line_count] = 0;
    ender[ender_count] = 0;
    if (line_count == 0 && ender_count == 0)
    {
        return EOF;
    }
    return 1;
}

static void Set_CV_Config_Command(COLLECTIVE_VARIABLE_CONTROLLER* manager,
                                  const char* line, const char* prefix,
                                  const std::string& source_path)
{
    const std::string section = string_strip(prefix == nullptr ? "" : prefix);
    if (section == "comments") return;

    const std::string command_line = line == nullptr ? "" : line;
    const auto separator = command_line.find('=');
    if (separator == std::string::npos) return;
    const std::string flag = string_strip(command_line.substr(0, separator));
    std::string value = command_line.substr(separator + 1);
    const auto comment = value.find('#');
    if (comment != std::string::npos) value.erase(comment);
    value = string_strip(value);
    if (flag.empty() || value.empty()) return;

    Set_CV_Config_Value(manager, section, flag, value, source_path);
}

void COLLECTIVE_VARIABLE_CONTROLLER::Commands_From_In_File(
    CONTROLLER* controller)
{
    for (const char* input_key :
         {"cv_in_file", "restrain_in_file", "restrain_cv_in_file"})
    {
        if (!controller->Command_Exist(input_key)) continue;
        const std::string cv_path = controller->Command(input_key);
        std::string ext = to_lower_copy(fs::path(cv_path).extension().string());
        if (ext == ".toml")
        {
            std::string toml_content = Read_File_To_String(cv_path, controller);
            Load_Toml_Commands(
                toml_content, cv_path, this,
                "COLLECTIVE_VARIABLE_CONTROLLER::Commands_From_In_File");
            continue;
        }
        FILE* In_File = NULL;
        Open_File_Safely(&In_File, cv_path.c_str(), "r", true);
        char line[CHAR_LENGTH_MAX];
        char prefix[CHAR_LENGTH_MAX] = {0};
        char ender[CHAR_LENGTH_MAX];
        while (true)
        {
            if (read_one_line(In_File, line, ender) == EOF)
            {
                break;
            }
            if (line[0] == '#')
            {
                if (line[1] == '#')
                {
                    if (strchr(ender, '{') != NULL)
                    {
                        int scanf_ret = sscanf(line, "%s", prefix);
                    }
                    if (strchr(ender, '}') != NULL)
                    {
                        prefix[0] = 0;
                    }
                }
                if (strchr(ender, '\n') == NULL)
                {
                    int scanf_ret = fscanf(In_File, "%*[^\n]%*[\n]");
                    fseek(In_File, -1, SEEK_CUR);
                }
            }
            else if (strchr(ender, '{') != NULL)
            {
                int scanf_ret = sscanf(line, "%s", prefix);
            }
            else
            {
                Set_CV_Config_Command(this, line, prefix, cv_path);
                line[0] = 0;
            }
            if (strchr(ender, '}') != NULL)
            {
                prefix[0] = 0;
            }
        }
        fclose(In_File);
    }
}

void COLLECTIVE_VARIABLE_CONTROLLER::Input_Check()
{
    if (!is_initialized) return;
    for (int i = 0; i < print_cv_list.size(); i++)
    {
        controller->Step_Print_Initial(print_cv_list[i]->module_name, "%.4f");
    }
    if (!(Command_Exist("dont_check_input") &&
          atoi(Command("dont_check_input"))))
    {
        int no_warning = 0;
        for (CheckMap::iterator iter = command_check.begin();
             iter != command_check.end(); iter++)
        {
            if (iter->second == 1)
            {
                controller->printf(
                    "Warning: CV command '%s' is set, but never used.\n",
                    iter->first.c_str());
                no_warning += 1;
            }
        }
        for (CheckMap::iterator iter = choice_check.begin();
             iter != choice_check.end(); iter++)
        {
            if (iter->second == 2)
            {
                controller->printf(
                    "Warning: the value '%s' of CV command '%s' matches none "
                    "of the choices.\n",
                    this->commands[iter->first].c_str(), iter->first.c_str());
                no_warning += 1;
            }
            else if (iter->second == 3)
            {
                controller->printf("Warning: CV command '%s' is not set.\n",
                                   iter->first.c_str());
                no_warning += 1;
            }
        }
        if (no_warning)
        {
            controller->printf(
                "\nWarning: CV inputs raised %d warning(s). Continuing "
                "execution. Set dont_check_input = 1 to disable this "
                "warning.\n",
                no_warning);
        }
    }
}

void COLLECTIVE_VARIABLE_CONTROLLER::Print_Initial()
{
    if (!is_initialized) return;
    controller->printf("START INITIALIZING CV PRINTER:\n");
    print_cv_list = Ask_For_CV("print", 0);
    for (int i = 0; i < print_cv_list.size(); i++)
    {
        if (controller->outputs_content.count(print_cv_list[i]->module_name))
        {
            std::string error_reason = "Reason:\n\tthe name of the CV '";
            error_reason += print_cv_list[i]->module_name;
            error_reason += "' to print is the same with a built-in output\n";
            controller->Throw_SPONGE_Error(
                spongeErrorConflictingCommand,
                "COLLECTIVE_VARIABLE_CONTROLLER::Print_Initial",
                error_reason.c_str());
        }
    }
    controller->printf("END INITIALIZING CV PRINTER\n\n");
}

void COLLECTIVE_VARIABLE_CONTROLLER::Compute_CV_For_Print(
    int atom_numbers, VECTOR* crd, Boundary boundary, int steps, int interval,
    bool print_zeroth_frame)
{
    if (!((print_zeroth_frame || steps) && (steps % interval == 0))) return;
    for (int i = 0; i < print_cv_list.size(); i++)
    {
        print_cv_list[i]->Compute(atom_numbers, crd, boundary,
                                  CV_NEED_CPU_VALUE, steps);
    }
}

void COLLECTIVE_VARIABLE_CONTROLLER::Step_Print()
{
    if (!is_initialized) return;
    if (CONTROLLER::MPI_size == 1)
    {
        for (int i = 0; i < print_cv_list.size(); i++)
        {
            controller->Step_Print(print_cv_list[i]->module_name,
                                   print_cv_list[i]->value);
        }
        return;
    }
    else  // 由拥有全局坐标的 CV rank 将结果发给0号进程
    {
#ifdef USE_MPI
        for (int i = 0; i < print_cv_list.size(); i++)
        {
            if (CONTROLLER::MPI_rank == 0)
            {
                MPI_Recv(&print_cv_list[i]->value, 1, MPI_FLOAT,
                         CONTROLLER::CV_MPI_rank, 0, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
                controller->Step_Print(print_cv_list[i]->module_name,
                                       print_cv_list[i]->value);
            }
            if (CONTROLLER::MPI_rank == CONTROLLER::CV_MPI_rank)
            {
                MPI_Send(&print_cv_list[i]->value, 1, MPI_FLOAT, 0, 0,
                         MPI_COMM_WORLD);
            }
        }
#endif
    }
}

COLLECTIVE_VARIABLE_PROTOTYPE* COLLECTIVE_VARIABLE_CONTROLLER::get_CV(
    const char* cv_name)
{
    if (!is_initialized)
    {
        this->Throw_SPONGE_Error(
            spongeErrorMissingCommand, "COLLECTIVE_VARIABLE_CONTROLLER::get_CV",
            "Reason:\n\tcommand 'cv_in_file' is not set\n");
    }
    if (CV_INSTANCE_MAP->count(cv_name))
    {
        return CV_INSTANCE_MAP[0][cv_name];
    }
    if (Command_Exist(cv_name, "CV_type"))
    {
        std::string cv_type = Command(cv_name, "CV_type");
        if (CV_MAP->count(cv_type))
        {
            COLLECTIVE_VARIABLE_PROTOTYPE* cv =
                CV_MAP[0][cv_type](this, cv_name);
            CV_INSTANCE_MAP[0][cv_name] = cv;
            return CV_INSTANCE_MAP[0][cv_name];
        }
        else
        {
            std::string error_reason = string_format(
                "Reason:\n\tthe type '%CV_TYPE%' of the CV '%CV_NAME%' is "
                "undefined",
                {{"CV_TYPE", cv_type}, {"CV_NAME", cv_name}});
            this->Throw_SPONGE_Error(spongeErrorTypeErrorCommand,
                                     "COLLECTIVE_VARIABLE_CONTROLLER::get_CV",
                                     error_reason.c_str());
        }
    }
    else
    {
        std::string error_reason = string_format(
            "Reason:\n\tthe type of the CV '%CV_NAME%' is required",
            {{"CV_NAME", cv_name}});
        this->Throw_SPONGE_Error(spongeErrorTypeErrorCommand,
                                 "COLLECTIVE_VARIABLE_CONTROLLER::get_CV",
                                 error_reason.c_str());
    }
    return 0;
}

CV_LIST COLLECTIVE_VARIABLE_CONTROLLER::Ask_For_CV(const char* name, int N,
                                                   float verbose_level,
                                                   int layout)
{
    CV_LIST cv_list;
    int CV_numbers = 0;
    std::vector<std::string> cv_names;
    std::string command = string_format("%NAME%_CV", {{"NAME", name}});
    if (Command_Exist(command.c_str()))
    {
        command = string_strip(Original_Command(command.c_str()));
        cv_names = string_split(command, " ");
        CV_numbers = cv_names.size();
    }
    if (N > 0 && CV_numbers != N)
    {
        std::string error_reason = string_format(
            "Reason:\n\t%N_NEED% CV(s) should be given to %NAME%, but %N_CV% "
            "found",
            {{"N_NEED", std::to_string(N)},
             {"NAME", name},
             {"N_CV", std::to_string(CV_numbers)}});
        Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                           "COLLECTIVE_VARIABLE_CONTROLLER::Ask_For_CV",
                           error_reason.c_str());
    }
    else if (N <= 0 && CV_numbers < -N)
    {
        std::string error_reason = string_format(
            "Reason:\n\tat least %N_NEED% CV(s) should be given to %NAME%, but "
            "only %N_CV% found",
            {{"N_NEED", std::to_string(N)},
             {"NAME", name},
             {"N_CV", std::to_string(CV_numbers)}});
        Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                           "COLLECTIVE_VARIABLE_CONTROLLER::Ask_For_CV",
                           error_reason.c_str());
    }
    if (verbose_level > -2)
    {
        for (int i = 0; i < layout; i++)
        {
            printf("    ");
        }
        printf("%d CV(s) found for %s\n", CV_numbers, name);
    }
    COLLECTIVE_VARIABLE_PROTOTYPE* cv;
    for (int i = 0; i < CV_numbers; i++)
    {
        command = cv_names[i];
        if (verbose_level > -1)
        {
            for (int ii = 0; ii < layout; ii++)
            {
                printf("    ");
            }
            printf("    CV %d: %s\n", i, command.c_str());
        }

        cv = get_CV(command.c_str());
        if (verbose_level > -1)
        {
            for (int ii = 0; ii < layout; ii++)
            {
                printf("    ");
            }
            printf("        type of '%s' is '%s'\n", command.c_str(),
                   cv->type_name);
        }
        cv_list.push_back(cv);
    }
    return cv_list;
}

int* COLLECTIVE_VARIABLE_CONTROLLER::Ask_For_Int_Parameter(
    const char* name, const char* parameter_name, int N, int layout,
    bool raise_error_when_missing, int default_value, float verbose_level,
    const char* unit)
{
    std::string unit_string;
    if (unit == NULL)
    {
        unit_string = "";
    }
    else
    {
        unit_string = " " + std::string(unit);
    }
    int* t;
    std::string command =
        string_format("%CV_NAME%_%PARAMETER_NAME%",
                      {{"CV_NAME", name}, {"PARAMETER_NAME", parameter_name}});
    if (!this->Command_Exist(command.c_str()))
    {
        if (raise_error_when_missing)
        {
            std::string error_reason = string_format(
                "Reason:\n\tno parameter '%PARAMETER_NAME%' found for CV "
                "'%CV_NAME%'",
                {{"CV_NAME", name}, {"PARAMETER_NAME", parameter_name}});
            Throw_SPONGE_Error(
                spongeErrorMissingCommand,
                "COLLECTIVE_VARIABLE_CONTROLLER::Ask_For_Int_Parameter",
                error_reason.c_str());
        }
        else
        {
            command = "";
        }
    }
    else
    {
        command = string_strip(this->Original_Command(command.c_str()));
    }
    if (verbose_level > -2)
    {
        for (int _lay = 0; _lay < layout; _lay++) this->printf("    ");
        this->printf("reading %d %s(s) for %s\n", N, parameter_name, name);
    }
    Malloc_Safely((void**)&t, sizeof(int) * N);
    std::vector<std::string> parameters = string_split(command, " ");
    if (parameters.size() < N && raise_error_when_missing)
    {
        std::string error_reason = string_format(
            "Reason:\n\tthe number of parameters '%PARAMETER_NAME%' \
for CV '%CV_NAME%' should be %N%, but only %SIZE% found",
            {{"CV_NAME", name},
             {"PARAMETER_NAME", parameter_name},
             {"N", std::to_string(N)},
             {"SIZE", std::to_string(parameters.size())}});
        Throw_SPONGE_Error(
            spongeErrorMissingCommand,
            "COLLECTIVE_VARIABLE_CONTROLLER::Ask_For_Int_Parameter",
            error_reason.c_str());
    }
    std::string hint;
    for (int i = 0; i < N; i++)
    {
        hint.clear();
        if (i >= parameters.size())
        {
            t[i] = default_value;
            hint = string_format(
                "%f%%s% (from default value)",
                {{"f", std::to_string(default_value)}, {"s", unit_string}});
        }
        else if (strcmp(parameter_name, "atom") == 0 &&
                 cv_vatom_name.count(parameters[i]))
        {
            t[i] = cv_vatom_name[parameters[i]] + atom_numbers;
            hint = string_format("%f%%s% (from cv virtual atom)",
                                 {{"f", parameters[i]}, {"s", unit_string}});
        }
        else if (controller->Command_Exist(parameters[i].c_str()))
        {
            controller->Check_Int(
                parameters[i].c_str(),
                "COLLECTIVE_VARIABLE_CONTROLLER::Ask_For_Int_Parameter");
            t[i] = atoi(controller->Command(parameters[i].c_str()));
            hint = string_format("%f%%s% (from command %C%)",
                                 {{"f", std::to_string(t[i])},
                                  {"s", unit_string},
                                  {"C", parameters[i]}});
        }
        else if (!is_str_int(parameters[i].c_str()))
        {
            std::string error_reason = string_format(
                "Reason:\n\t%VALUE% (the %i%-th value of '%PARAMETER_NAME%'\
for CV '%CV_NAME%') is not an int",
                {{"CV_NAME", name},
                 {"PARAMETER_NAME", parameter_name},
                 {"VALUE", parameters[i]},
                 {"i", std::to_string(i)}});
        }
        else
        {
            t[i] = atoi(parameters[i].c_str());
            hint = string_format(
                "%f%%s%", {{"f", std::to_string(t[i])}, {"s", unit_string}});
        }
        if (verbose_level > -1)
        {
            for (int _lay = 0; _lay < layout; _lay++) this->printf("    ");
            this->printf("    %s %d: %s\n", parameter_name, i, hint.c_str());
        }
    }
    return t;
}

float* COLLECTIVE_VARIABLE_CONTROLLER::Ask_For_Float_Parameter(
    const char* name, const char* parameter_name, int N, int layout,
    bool raise_error_when_missing, float default_value, float verbose_level,
    const char* unit)
{
    std::string unit_string;
    if (unit == NULL)
    {
        unit_string = "";
    }
    else
    {
        unit_string = " " + std::string(unit);
    }
    float* t;
    std::string command =
        string_format("%CV_NAME%_%PARAMETER_NAME%",
                      {{"CV_NAME", name}, {"PARAMETER_NAME", parameter_name}});
    if (!this->Command_Exist(command.c_str()))
    {
        if (raise_error_when_missing)
        {
            std::string error_reason = string_format(
                "Reason:\n\tno parameter '%PARAMETER_NAME%' found for CV "
                "'%CV_NAME%'",
                {{"CV_NAME", name}, {"PARAMETER_NAME", parameter_name}});
            Throw_SPONGE_Error(
                spongeErrorMissingCommand,
                "COLLECTIVE_VARIABLE_CONTROLLER::Ask_For_Int_Parameter",
                error_reason.c_str());
        }
        else
        {
            command = "";
        }
    }
    else
    {
        command = string_strip(this->Original_Command(command.c_str()));
    }
    if (verbose_level > -2)
    {
        for (int _lay = 0; _lay < layout; _lay++) this->printf("    ");
        this->printf("reading %d %s(s) for %s\n", N, parameter_name, name);
    }
    Malloc_Safely((void**)&t, sizeof(float) * N);
    std::vector<std::string> parameters = string_split(command, " ");
    if (parameters.size() < N && raise_error_when_missing)
    {
        std::string error_reason = string_format(
            "Reason:\n\tthe number of parameters '%PARAMETER_NAME%' \
for CV '%CV_NAME%' should be %N%, but only %SIZE% found",
            {{"CV_NAME", name},
             {"PARAMETER_NAME", parameter_name},
             {"N", std::to_string(N)},
             {"SIZE", std::to_string(parameters.size())}});
        Throw_SPONGE_Error(
            spongeErrorMissingCommand,
            "COLLECTIVE_VARIABLE_CONTROLLER::Ask_For_Float_Parameter",
            error_reason.c_str());
    }
    std::string hint;
    for (int i = 0; i < N; i++)
    {
        hint.clear();
        if (i >= parameters.size())
        {
            t[i] = default_value;
            hint = string_format(
                "%f%%s% (from default value)",
                {{"f", std::to_string(default_value)}, {"s", unit_string}});
        }
        else if (controller->Command_Exist(parameters[i].c_str()))
        {
            controller->Check_Float(
                parameters[i].c_str(),
                "COLLECTIVE_VARIABLE_CONTROLLER::Ask_For_Float_Parameter");
            t[i] = (float)atof(controller->Command(parameters[i].c_str()));
            hint = string_format("%f%%s% (from command %C%)",
                                 {{"f", std::to_string(t[i])},
                                  {"s", unit_string},
                                  {"C", parameters[i]}});
        }
        else if (!is_str_float(parameters[i].c_str()))
        {
            std::string error_reason = string_format(
                "Reason:\n\t%VALUE% (the %i%-th value of '%PARAMETER_NAME%'\
for CV '%CV_NAME%') is not a float",
                {{"CV_NAME", name},
                 {"PARAMETER_NAME", parameter_name},
                 {"VALUE", parameters[i]},
                 {"i", std::to_string(i)}});
        }
        else
        {
            t[i] = (float)atof(parameters[i].c_str());
            hint = string_format(
                "%f%%s%", {{"f", std::to_string(t[i])}, {"s", unit_string}});
        }
        if (verbose_level > -1)
        {
            for (int _lay = 0; _lay < layout; _lay++) this->printf("    ");
            this->printf("    %s %d: %s\n", parameter_name, i, hint.c_str());
        }
    }
    return t;
}

std::vector<std::string>
COLLECTIVE_VARIABLE_CONTROLLER::Ask_For_String_Parameter(
    const char* name, const char* parameter_name, int N, int layout,
    bool raise_error_when_missing, const char* default_value,
    float verbose_level, const char* unit)
{
    std::string unit_string;
    if (unit == NULL)
    {
        unit_string = "";
    }
    else
    {
        unit_string = " " + std::string(unit);
    }
    std::vector<std::string> t(N);
    std::string command =
        string_format("%CV_NAME%_%PARAMETER_NAME%",
                      {{"CV_NAME", name}, {"PARAMETER_NAME", parameter_name}});
    if (!this->Command_Exist(command.c_str()))
    {
        if (raise_error_when_missing)
        {
            std::string error_reason = string_format(
                "Reason:\n\tno parameter '%PARAMETER_NAME%' found for CV "
                "'%CV_NAME%'",
                {{"CV_NAME", name}, {"PARAMETER_NAME", parameter_name}});
            Throw_SPONGE_Error(
                spongeErrorMissingCommand,
                "COLLECTIVE_VARIABLE_CONTROLLER::Ask_For_Int_Parameter",
                error_reason.c_str());
        }
        else
        {
            command = "";
        }
    }
    else
    {
        command = string_strip(this->Original_Command(command.c_str()));
    }
    if (verbose_level > -2)
    {
        for (int _lay = 0; _lay < layout; _lay++) this->printf("    ");
        this->printf("reading %d %s(s) for %s\n", N, parameter_name, name);
    }
    std::vector<std::string> parameters = string_split(command, " ");
    if (parameters.size() < N && raise_error_when_missing)
    {
        std::string error_reason = string_format(
            "Reason:\n\t%N% number of parameter '%PARAMETER_NAME%' \
for CV '%CV_NAME%' should be %N%, but only %SIZE% found",
            {{"CV_NAME", name},
             {"PARAMETER_NAME", parameter_name},
             {"N", std::to_string(N)},
             {"%SIZE", std::to_string(parameters.size())}});
    }
    std::string hint;
    for (int i = 0; i < N; i++)
    {
        hint.clear();
        if (i >= parameters.size())
        {
            t[i] = default_value;
            hint = t[i] + unit_string + " (from default value)";
        }
        else
        {
            t[i] = parameters[i];
            hint = t[i] + unit_string;
        }
        if (verbose_level > -1)
        {
            for (int _lay = 0; _lay < layout; _lay++) this->printf("    ");
            this->printf("    %s %d: %s\n", parameter_name, i, hint.c_str());
        }
    }
    return t;
}

std::vector<int>
COLLECTIVE_VARIABLE_CONTROLLER::Ask_For_Indefinite_Length_Int_Parameter(
    const char* name, const char* parameter_name)
{
    if (strcmp(parameter_name, "atom") == 0)
    {
        const auto found = protocol_virtual_atom_indices.find(name);
        if (found != protocol_virtual_atom_indices.end()) return found->second;
    }
    std::vector<int> ints;
    std::string out;
    std::string file_name = parameter_name;
    file_name += "_in_file";
    std::istream* ss = NULL;
    if (Command_Exist(name, parameter_name))
    {
        std::string strs = Original_Command(name, parameter_name);
        ss = new std::istringstream(strs);
    }
    else if (Command_Exist(name, file_name.c_str()))
    {
        ss = new std::ifstream(Command(name, file_name.c_str()));
    }
    if (ss != NULL)
    {
        while (ss[0] >> out)
        {
            if (cv_vatom_name.count(out))
            {
                ints.push_back(cv_vatom_name[out] + atom_numbers);
            }
            else if (controller->Command_Exist(out.c_str()))
            {
                controller->Check_Int(out.c_str(),
                                      "COLLECTIVE_VARIABLE_CONTROLLER::Ask_For_"
                                      "Indefinite_Length_Int_Parameter");
                ints.push_back(atoi(controller->Command(out.c_str())));
            }
            else
            {
                if (!is_str_int(out.c_str()))
                {
                    Throw_SPONGE_Error(
                        spongeErrorTypeErrorCommand,
                        "COLLECTIVE_VARIABLE_CONTROLLER::Ask_For_Indefinite_"
                        "Length_Int_Parameter",
                        "Reason:\n\tone of the value is not an int\n");
                }
                ints.push_back(atoi(out.c_str()));
            }
        }
    }
    return ints;
}

std::vector<float>
COLLECTIVE_VARIABLE_CONTROLLER::Ask_For_Indefinite_Length_Float_Parameter(
    const char* name, const char* parameter_name)
{
    if (strcmp(parameter_name, "weight") == 0)
    {
        const auto found = protocol_virtual_atom_weight.find(name);
        if (found != protocol_virtual_atom_weight.end()) return found->second;
    }
    std::vector<float> floats;
    std::string out;
    std::string file_name = parameter_name;
    file_name += "_in_file";
    std::istream* ss = NULL;
    if (Command_Exist(name, parameter_name))
    {
        std::string strs = Original_Command(name, parameter_name);
        ss = new std::istringstream(strs);
    }
    else if (Command_Exist(name, file_name.c_str()))
    {
        ss = new std::ifstream(Command(name, file_name.c_str()));
    }
    if (ss != NULL)
    {
        while (ss[0] >> out)
        {
            if (cv_vatom_name.count(out))
            {
                floats.push_back(cv_vatom_name[out] + atom_numbers);
            }
            else if (controller->Command_Exist(out.c_str()))
            {
                controller->Check_Float(
                    out.c_str(),
                    "COLLECTIVE_VARIABLE_CONTROLLER::Ask_For_Indefinite_Length_"
                    "Float_Parameter");
                floats.push_back(atof(controller->Command(out.c_str())));
            }
            else
            {
                if (!is_str_float(out.c_str()))
                {
                    Throw_SPONGE_Error(
                        spongeErrorTypeErrorCommand,
                        "COLLECTIVE_VARIABLE_CONTROLLER::Ask_For_Indefinite_"
                        "Length_Float_Parameter",
                        "Reason:\n\tone of the value is not a float\n");
                }
                floats.push_back(atof(out.c_str()));
            }
        }
    }
    return floats;
}

int COLLECTIVE_VARIABLE_PROTOTYPE::Check_Whether_Computed_At_This_Step(int step,
                                                                       int need)
{
    if ((need & CV_NEED_CPU_VALUE) &&
        (last_update_step[CV_NEED_CPU_VALUE] == step))
        need &= ~CV_NEED_CPU_VALUE;
    if ((need & CV_NEED_GPU_VALUE) &&
        (last_update_step[CV_NEED_GPU_VALUE] == step))
        need &= ~CV_NEED_GPU_VALUE;
    if ((need & CV_NEED_CRD_GRADS) &&
        (last_update_step[CV_NEED_CRD_GRADS] == step))
        need &= ~CV_NEED_CRD_GRADS;
    if ((need & CV_NEED_VIRIAL) && (last_update_step[CV_NEED_VIRIAL] == step))
        need &= ~CV_NEED_VIRIAL;
    return need;
}

void COLLECTIVE_VARIABLE_PROTOTYPE::Record_Update_Step_Of_Slow_Computing_CV(
    int step, int need)
{
    if (need & CV_NEED_CPU_VALUE) last_update_step[CV_NEED_CPU_VALUE] = step;
    if (need & CV_NEED_CRD_GRADS) last_update_step[CV_NEED_CRD_GRADS] = step;
    if (need & CV_NEED_GPU_VALUE) last_update_step[CV_NEED_GPU_VALUE] = step;
    if (need & CV_NEED_VIRIAL) last_update_step[CV_NEED_VIRIAL] = step;
}

void COLLECTIVE_VARIABLE_PROTOTYPE::Record_Update_Step_Of_Fast_Computing_CV(
    int step, int need)
{
    last_update_step[CV_NEED_VIRIAL] = step;
    last_update_step[CV_NEED_CRD_GRADS] = step;
    last_update_step[CV_NEED_GPU_VALUE] = step;
    last_update_step[CV_NEED_CPU_VALUE] = step;
}

void COLLECTIVE_VARIABLE_PROTOTYPE::Super_Initial(
    COLLECTIVE_VARIABLE_CONTROLLER* manager, int atom_numbers,
    const char* module_name)
{
    this->manager = manager;
    strcpy(this->module_name, module_name);
    int total_atom_numbers = atom_numbers;
    if (manager != NULL)
    {
        total_atom_numbers += manager->cv_vatom_name.size();
    }
    Device_Malloc_Safely((void**)&crd_grads,
                         sizeof(VECTOR) * total_atom_numbers);
    Device_Malloc_Safely((void**)&virial, sizeof(LTMatrix3));
    Device_Malloc_Safely((void**)&d_value, sizeof(float));
    deviceMemset(crd_grads, 0, sizeof(VECTOR) * total_atom_numbers);
    deviceMemset(virial, 0, sizeof(LTMatrix3));
#ifdef GPU_ARCH_NAME
    deviceStreamCreate(&device_stream);
#endif
    last_update_step[CV_NEED_GPU_VALUE] = -1;
    last_update_step[CV_NEED_CPU_VALUE] = -1;
    last_update_step[CV_NEED_CRD_GRADS] = -1;
    last_update_step[CV_NEED_VIRIAL] = -1;
}

void COLLECTIVE_VARIABLE_PROTOTYPE::Validate_Boundary(
    const Boundary boundary) const
{
    if (boundary.policy != BoundaryPolicy::Open || supports_open_boundary)
        return;
    std::string reason = "Reason:\n\tCV type '";
    reason += type_name;
    reason += "' requires periodic boundary conditions\n";
    manager->Throw_SPONGE_Error(spongeErrorConflictingCommand,
                                "COLLECTIVE_VARIABLE_PROTOTYPE::Compute",
                                reason.c_str());
}

void COLLECTIVE_VARIABLE_PROTOTYPE::Initial(
    COLLECTIVE_VARIABLE_CONTROLLER* manager, int atom_numbers,
    const char* module_name)
{
    Super_Initial(manager, atom_numbers, module_name);
}

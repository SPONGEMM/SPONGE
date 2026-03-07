#pragma once

#include "../../third_party/toml/toml.h"

inline bool Malloc_Safely(void** address, size_t size)
{
    address[0] = NULL;
    address[0] = (void*)malloc(size);
    if (address[0] != NULL)
    {
        return true;
    }
    else
    {
#ifndef NO_GLOBAL_CONTROLLER
        extern CONTROLLER controller;
        controller.Throw_SPONGE_Error(spongeErrorMallocFailed, "Malloc_Safely");
#endif
        return false;
    }
}

inline bool Device_Malloc_Safely(void** address, size_t size)
{
#ifdef GPU_ARCH_NAME
    if (deviceMalloc(&address[0], size) == DEVICE_MALLOC_SUCCESS)
    {
        return true;
    }
    else
    {
#ifndef NO_GLOBAL_CONTROLLER
        extern CONTROLLER controller;
        controller.Throw_SPONGE_Error(spongeErrorMallocFailed,
                                      "Device_Malloc_Safely");
#endif
        return false;
    }
#else
    address[0] = malloc(size);
    return true;
#endif
}

inline bool Open_File_Safely(FILE** file, const char* file_name,
                             const char* open_type, bool check_os = false)
{
    file[0] = NULL;
    file[0] = fopen(file_name, open_type);
    if (file[0] == NULL)
    {
#ifndef NO_GLOBAL_CONTROLLER
        extern CONTROLLER controller;
        std::string output = "Open_File_Safely(";
        output += file_name;
        output += ")";
        controller.Throw_SPONGE_Error(spongeErrorOpenFileFailed,
                                      output.c_str());
#endif
        return false;
    }
    else
    {
        std::string open_type_str = open_type;
        if (check_os)
        {
            fclose(file[0]);
            file[0] = fopen(file_name, "rb");
            fseek(file[0], -2, SEEK_END);
            if (ftell(file[0]) < 2)
            {
#ifndef NO_GLOBAL_CONTROLLER
                extern CONTROLLER controller;
                std::string output = "Open_File_Safely(";
                output += file_name;
                output += ")";
                controller.Throw_SPONGE_Error(
                    spongeErrorOpenFileFailed, output.c_str(),
                    string_format("%FNAME% is an empty file",
                                  {{"FNAME", file_name}})
                        .c_str());
#endif
                return false;
            }
            bool hasCR = false;
            bool hasLF = false;
            fseek(file[0], 0, SEEK_SET);
            signed char ch;
            while ((ch = fgetc(file[0])) != EOF)
            {
                if (ch == '\r')
                {
                    hasCR = true;
                }
                else if (ch == '\n')
                {
                    hasLF = true;
                    break;
                }
            }
            fclose(file[0]);
            file[0] = fopen(file_name, open_type);
            if (hasCR && hasLF)
            {
#ifndef _WIN32
#ifndef NO_GLOBAL_CONTROLLER
                extern CONTROLLER controller;
                std::string output = "Open_File_Safely(";
                output += file_name;
                output += ")";
                controller.Throw_SPONGE_Error(
                    spongeErrorOpenFileFailed, output.c_str(),
                    string_format(
                        "%FNAME% is a file from Windows, but SPONGE you use is on Linux\n\
The shell commands like 'dos2unix %FNAME%' or 'sed -i 's/\\r$//' %FNAME%' may help you convert the file format",
                        {{"FNAME", file_name}})
                        .c_str());
#endif
                return false;
#endif
            }
            else if (hasLF)
            {
#ifdef _WIN32
#ifndef NO_GLOBAL_CONTROLLER
                extern CONTROLLER controller;
                std::string output = "Open_File_Safely(";
                output += file_name;
                output += ")";
                controller.Throw_SPONGE_Error(
                    spongeErrorOpenFileFailed, output.c_str(),
                    string_format("%FNAME% is a file from Linux, but SPONGE "
                                  "you use is on Windows\n",
                                  {{"FNAME", file_name}})
                        .c_str());
#endif
                return false;
#endif
            }
        }
        return true;
    }
}

inline bool Device_Malloc_And_Copy_Safely(void** d_address, void* h_address,
                                          size_t size,
                                          const char* var_name = NULL)
{
#ifndef CPU_ARCH_NAME
    deviceError_t device_error;
    if (var_name == NULL) var_name = "unnamed var";
    device_error = deviceMalloc(d_address, size);
    if (device_error != 0)
    {
#ifndef NO_GLOBAL_CONTROLLER
        extern CONTROLLER controller;
        controller.Throw_SPONGE_Error(spongeErrorMallocFailed,
                                      "Device_Malloc_And_Copy_Safely");
#endif
        return false;
    }
    device_error =
        deviceMemcpy(d_address[0], h_address, size, deviceMemcpyHostToDevice);
    if (device_error != 0)
    {
#ifndef NO_GLOBAL_CONTROLLER
        extern CONTROLLER controller;
        controller.Throw_SPONGE_Error(spongeErrorMallocFailed,
                                      "Device_Malloc_And_Copy_Safely");
#endif
        return false;
    }
    return true;
#else
    d_address[0] = h_address;
    return true;
#endif
}

inline FILE* CONTROLLER::Get_Output_File(bool binary, const char* command,
                                         const char* default_suffix,
                                         const char* default_filename)
{
    FILE* fp = NULL;
    std::string filename;
    if (this->Command_Exist(command))
    {
        filename = this->Command(command);
    }
    else if (default_suffix != NULL &&
             this->Command_Exist("default_out_file_prefix"))
    {
        filename = this->Command("default_out_file_prefix");
        filename += default_suffix;
    }
    else if (default_filename != NULL)
    {
        filename = default_filename;
    }
    if (!filename.empty())
    {
        std::string open_type = "w";
        if (binary)
        {
            open_type += "b";
        }
        Open_File_Safely(&fp, filename.c_str(), open_type.c_str());
    }
    return fp;
}

inline FILE* CONTROLLER::Get_Output_File(bool binary, const char* prefix,
                                         const char* command,
                                         const char* default_suffix,
                                         const char* default_filename)
{
    std::string full_command = prefix;
    full_command += "_";
    full_command += command;
    return this->Get_Output_File(binary, full_command.c_str(), default_suffix,
                                 default_filename);
}

inline void CONTROLLER::Set_File_Buffer(FILE* file, size_t one_frame_size)
{
    char* buffer;
    Malloc_Safely((void**)&buffer, one_frame_size * buffer_frame);
    if (setvbuf(file, buffer, _IOFBF, one_frame_size * buffer_frame) != 0)
    {
        std::string error_reason = string_format(
            "Reason:\n\tthe trajectory file will be written every %buffer_frame% and SPONGE failed to allocate a memory for this. \
Please use the command 'buffer_frame = xxx' to decrease the buffer size",
            {{"buffer_frame", std::to_string(buffer_frame)}});
        Throw_SPONGE_Error(spongeErrorMallocFailed,
                           "CONTROLLER::Set_File_Buffer", error_reason.c_str());
    }
}

namespace
{
enum class MdinInputFormat
{
    None,
    Text,
    Toml
};

std::string to_lower_copy(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    return value;
}

std::string Read_File_To_String(const std::string& path, CONTROLLER* controller)
{
    std::ifstream stream(path, std::ios::in | std::ios::binary);
    if (!stream.is_open())
    {
        std::string error_reason = string_format(
            "Reason:\n\tfail to open mdin file '%PATH%'", {{"PATH", path}});
        controller->Throw_SPONGE_Error(spongeErrorOpenFileFailed,
                                       "CONTROLLER::Commands_From_In_File",
                                       error_reason.c_str());
    }
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return buffer.str();
}

template <typename ControllerType>
void Load_Toml_Commands(const std::string& content,
                        const std::string& source_path,
                        ControllerType* controller, const char* error_by)
{
    std::map<std::string, std::string> parsed_commands;
    std::string parse_error;
    if (!sponge::toml_wrap::ParseAndFlatten(content, source_path,
                                            &parsed_commands, &parse_error))
    {
        std::string error_reason =
            string_format("Reason:\n\tTOML parse error in '%PATH%': %DESC%",
                          {{"PATH", source_path}, {"DESC", parse_error}});
        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat, error_by,
                                       error_reason.c_str());
        return;
    }
    for (const auto& [full_key, value] : parsed_commands)
    {
        controller->Set_Command(full_key.c_str(), value.c_str(), 1, NULL);
    }
}
}  // namespace

#pragma once

// SPONGE错误类型
enum spongeError
{
    spongeSuccess = 0,
    // 1000以下的错误留给 deviceError
    // 未实现的功能
    spongeErrorNotImplemented = 1001,
    // 文件格式（编码、换行符）问题 或 数据格式不正确
    spongeErrorBadFileFormat,
    // 冲突的命令
    spongeErrorConflictingCommand,
    // 缺失的命令
    spongeErrorMissingCommand,
    // 类型错误的命令
    spongeErrorTypeErrorCommand,
    // 值错误的命令
    spongeErrorValueErrorCommand,
    // 模拟崩溃
    spongeErrorSimulationBreakDown,
    // 内存分配失败
    spongeErrorMallocFailed,
    // 越界
    spongeErrorOverflow,
    // 打开文件失败
    spongeErrorOpenFileFailed,
};

inline void CONTROLLER::Throw_SPONGE_Error(const int error_number,
                                           const char* error_by,
                                           const char* extra_error_string)
{
    if (error_number == 0) return;
    std::string error_name;
    std::string error_reason;
    std::string error_by_;
    std::string extra_error_string_;
#ifdef USE_GPU
    if (error_number <= 1000)
    {
        error_name = deviceGetErrorName((deviceError_t)error_number);
        error_reason = deviceGetErrorName((deviceError_t)error_number);
    }
    else
#endif
    {
        switch (error_number)
        {
            case spongeErrorNotImplemented:
            {
                error_name = "spongeErrorNotImplemented";
                error_reason =
                    "The function has not been implemented in SPONGE yet";
                break;
            }
            case spongeErrorBadFileFormat:
            {
                error_name = "spongeErrorBadFileFormat";
                error_reason = "The format of the file is bad";
                break;
            }
            case spongeErrorConflictingCommand:
            {
                error_name = "spongeErrorConflictingCommand";
                error_reason = "Some commands are conflicting";
                break;
            }
            case spongeErrorMissingCommand:
            {
                error_name = "spongeErrorMissingCommand";
                error_reason = "Missing required command(s)";
                break;
            }
            case spongeErrorTypeErrorCommand:
            {
                error_name = "spongeErrorMissingCommand";
                error_reason = "The type of the command is wrong";
                break;
            }
            case spongeErrorValueErrorCommand:
            {
                error_name = "spongeErrorValueErrorCommand";
                error_reason = "The value of the command is wrong";
                break;
            }
            case spongeErrorSimulationBreakDown:
            {
                error_name = "spongeErrorSimulationBrokenDown";
                error_reason = "The system was broken down";
                break;
            }
            case spongeErrorMallocFailed:
            {
                error_name = "spongeErrorMallocFailed";
                error_reason = "Fail to allocate memory";
                break;
            }
            case spongeErrorOverflow:
            {
                error_name = "spongeErrorOverflow";
                error_reason = "Boundary was overflowed";
                break;
            }
            case spongeErrorOpenFileFailed:
            {
                error_name = "spongeErrorOpenFileFailed";
                error_reason = "Fail to open file";
                break;
            }
            default:
            {
                error_name = "spongeErrorUnclassified";
                error_reason = "Unclassified Error";
            }
        }
    }
    if (error_by != NULL)
    {
        error_by_ = std::string(" raised by ") + error_by;
    }
    else
    {
        error_by_ = std::string("");
    }
    if (extra_error_string != NULL)
    {
        extra_error_string_ = extra_error_string;
        if (extra_error_string_.back() != '\n')
        {
            extra_error_string_ += "\n";
        }
    }
    else
    {
        extra_error_string_ = "";
    }
    printf("\n%s%s\n%s\n%s", error_name.c_str(), error_by_.c_str(),
           error_reason.c_str(), extra_error_string_.c_str());
    fcloseall();
#ifdef USE_MPI
    MPI_Abort(MPI_COMM_WORLD, error_number);
#else
    exit(error_number);
#endif
}

inline void CONTROLLER::Throw_Formatted_SPONGE_Error(const int error_number,
                                                     const char* error_by,
                                                     const char* format, ...)
{
    char error_reason[CHAR_LENGTH_MAX];
    va_list args;
    va_start(args, format);
    vsnprintf(error_reason, sizeof(error_reason), format, args);
    va_end(args);
    Throw_SPONGE_Error(error_number, error_by, error_reason);
}

inline void CONTROLLER::Check_Error(float energy)
{
#ifdef GPU_ARCH_NAME
    deviceError_t device_error = deviceGetLastError();
    if (device_error == deviceErrorInvalidConfiguration ||
        device_error == deviceErrorInvalidValue ||
        device_error == deviceErrorLaunchOutOfResources)
    {
        Throw_SPONGE_Error(device_error, "CONTROLLER::Check_Error",
                           "Reasons:\n\tA device kernel function is launched "
                           "with wrong parameters, and this should be a bug. "
                           "Please report the issue to the developers.");
    }
    else if (device_error != 0)
    {
        Throw_SPONGE_Error(
            device_error, "CONTROLLER::Check_Error",
            "Possible reasons:\n\t1. the energy of the system is not fully "
            "minimized\n\t2. bad dt (too large)\n\t3. bad thermostat/barostat "
            "parameters\n\t4. bad force field parameters\n");
    }
#endif
    if (isnan(energy) || isinf(energy) || isnan(printf_sum) ||
        isinf(printf_sum))
    {
        Throw_SPONGE_Error(
            spongeErrorSimulationBreakDown, "CONTROLLER::Check_Error",
            "Possible reasons:\n\t1. the energy of the system is not fully "
            "minimized\n\t2. bad dt (too large)\n\t3. bad thermostat/barostat "
            "parameters\n\t4. bad force field parameters\n");
    }
    printf_sum = 0;
}

inline void CONTROLLER::Check_Int(const char* command, const char* error_by)
{
    const char* str = this->Command(command);
    if (!is_str_int(str))
    {
        char error_reason[CHAR_LENGTH_MAX];
        sprintf(
            error_reason,
            "Reason:\n\t the value '%s' of the command '%s' is not an int\n",
            str, command);
        this->Throw_SPONGE_Error(spongeErrorTypeErrorCommand, error_by,
                                 error_reason);
    }
}

inline void CONTROLLER::Check_Int(const char* prefix, const char* command,
                                  const char* error_by)
{
    const char* str = this->Command(prefix, command);
    if (!is_str_int(str))
    {
        char error_reason[CHAR_LENGTH_MAX];
        sprintf(
            error_reason,
            "Reason:\n\t the value '%s' of the command '%s' is not an int\n",
            str, command);
        this->Throw_SPONGE_Error(spongeErrorTypeErrorCommand, error_by,
                                 error_reason);
    }
}

inline void CONTROLLER::Check_Float(const char* command, const char* error_by)
{
    const char* str = this->Command(command);
    if (!is_str_float(str))
    {
        char error_reason[CHAR_LENGTH_MAX];
        sprintf(
            error_reason,
            "Reason:\n\t the value '%s' of the command '%s' is not a float\n",
            str, command);
        this->Throw_SPONGE_Error(spongeErrorTypeErrorCommand, error_by,
                                 error_reason);
    }
}

inline void CONTROLLER::Check_Float(const char* prefix, const char* command,
                                    const char* error_by)
{
    const char* str = this->Command(prefix, command);
    if (!is_str_float(str))
    {
        char error_reason[CHAR_LENGTH_MAX];
        sprintf(
            error_reason,
            "Reason:\n\t the value '%s' of the command '%s' is not a float\n",
            str, command);
        this->Throw_SPONGE_Error(spongeErrorTypeErrorCommand, error_by,
                                 error_reason);
    }
}

inline bool CONTROLLER::Get_Bool(const char* command, const char* error_by)
{
    const char* str = this->Command(command);
    if (is_str_equal(str, "true"))
    {
        return true;
    }
    else if (is_str_equal(str, "false"))
    {
        return false;
    }
    else
    {
        Check_Int(command, error_by);
        return atoi(str);
    }
}

inline bool CONTROLLER::Get_Bool(const char* prefix, const char* command,
                                 const char* error_by)
{
    const char* str = this->Command(prefix, command);
    if (is_str_equal(str, "true"))
    {
        return true;
    }
    else if (is_str_equal(str, "false"))
    {
        return false;
    }
    else
    {
        Check_Int(prefix, command, error_by);
        return atoi(str);
    }
}

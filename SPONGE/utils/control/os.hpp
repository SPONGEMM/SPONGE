#pragma once

#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

// 获取当前工作目录
inline std::string Get_Current_Working_Directory()
{
    char* buffer = NULL;
    buffer = getcwd(NULL, 0);
    if (buffer == NULL) return "Unknown";
    std::string path = buffer;
    free(buffer);
    return path;
}

// 获取SPONGE目录
inline std::string Get_SPONGE_Directory()
{
    char path[CHAR_LENGTH_MAX] = {0};
#if defined(__APPLE__)
    uint32_t size = sizeof(path);
    if (_NSGetExecutablePath(path, &size) != 0)
    {
        std::string buffer(size, '\0');
        if (_NSGetExecutablePath(buffer.data(), &size) == 0)
        {
            return buffer.c_str();
        }
        return Get_Current_Working_Directory();
    }
    return path;
#else
    int l = readlink("/proc/self/exe", path, CHAR_LENGTH_MAX - 1);
    if (l > 0 && l < CHAR_LENGTH_MAX)
    {
        path[l] = 0;
        return path;
    }
    return Get_Current_Working_Directory();
#endif
}

// 获取当前时间
inline std::string Get_Wall_Time()
{
    time_t timep;
    time(&timep);
    return asctime(localtime(&timep));
}

#ifdef USE_CUDA
static __global__ void device_get_built_arch(int* answer)
{
#ifdef __CUDA_ARCH__
    *answer = __CUDA_ARCH__ / 10;
#else
    *answer = 0;
#endif
}

// 获取当前GPU的架构
inline int Get_Built_Arch()
{
    int answer, *d_answer;
    Device_Malloc_Safely((void**)&d_answer, sizeof(int));
    device_get_built_arch<<<1, 1>>>(d_answer);
    deviceMemcpy(&answer, d_answer, sizeof(int), deviceMemcpyDeviceToHost);
    deviceFree(d_answer);
    d_answer = NULL;
    return answer;
}
#endif  // USE_CUDA

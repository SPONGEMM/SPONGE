#pragma once

#include <float.h>
#include <math.h>
#include <omp.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <deque>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
namespace fs = std::filesystem;

static auto quote_path(const fs::path& path)
{
    std::string value = path.string();
    if (value.find_first_of(" \t\"'") != std::string::npos)
    {
        value = "\"" + value + "\"";
    }
    return value;
};

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#include <direct.h>
#include <fcntl.h>
#include <io.h>
#define dlopen(filename, mode) LoadLibrary(filename)
#define dlsym(handle, function) GetProcAddress(handle, function)
#define dlerror() GetLastError()
#ifndef RTLD_LAZY
#define RTLD_LAZY 0
#endif
#ifndef RTLD_GLOBAL
#define RTLD_GLOBAL 0
#endif
#ifndef RTLD_NOW
#define RTLD_NOW 0
#endif
#define PLUGIN_API extern "C" __declspec(dllexport)
#define getcwd _getcwd
#define readlink(x, y, z) GetModuleFileName(NULL, y, z);
#elif defined(__linux__)
#include <dlfcn.h>
#include <libgen.h>
#include <unistd.h>
#define PLUGIN_API extern "C"
typedef void* HMODULE;
#elif defined(__APPLE__)
#define sincosf __sincosf
#include <dlfcn.h>
#include <libgen.h>
#include <unistd.h>
typedef void* HMODULE;
#define fcloseall()   \
    do                \
    {                 \
        fflush(NULL); \
    } while (0)
#endif

// device backend setup
#define FFT_BACKEND_H
#ifdef USE_HIP
#include "third_party/device_backend/hip_api.h"
#elif defined(USE_CUDA)
#include "third_party/device_backend/cuda_api.h"
#else
#include "third_party/device_backend/cpu_api.h"
#endif
#undef FFT_BACKEND_H

// MPI setup
#include "third_party/mpi.hpp"

// Tensor概念
#include "utils/tensor.hpp"

// omp参数设置
extern int max_omp_threads;
const int CACHE_LINE = 64;

// 释放无host对应的device指针
void Free_Single_Device_Pointer(void** device_ptr);
// 释放有host对应的device指针，如果host指针在栈中host_ptr输入为NULL
void Free_Host_And_Device_Pointer(void** host_ptr, void** device_ptr);

//-----------------------------------常数定义-----------------------------------

// 圆周率
#define CONSTANT_Pi 3.1415926535897932f
// 自然对数的底
#define CONSTANT_e 2.7182818284590452f
// 玻尔兹曼常量（kcal.mol^-1.K^ -1）
// 使用kcal为能量单位，因此kB=8.31441(J.mol^-1.K^-1)/4.18407(J/cal)/1000
#define CONSTANT_kB 0.00198716f
// 能量单位换算系数: eV -> kcal/mol
#define CONSTANT_EV_TO_KCAL_MOL 23.0605480f
// 能量单位换算系数: Hartree -> kcal/mol
#define CONSTANT_HARTREE_TO_KCAL_MOL 627.509474f
// 长度单位换算: Angstrom -> Bohr
#define CONSTANT_ANGSTROM_TO_BOHR 1.8897259886f
// 长度单位换算: Bohr -> Angstrom
#define CONSTANT_BOHR_TO_ANGSTROM 0.52917721092f
// SPONGE内部电荷缩放系数
#define CONSTANT_SPONGE_CHARGE_SCALE 18.2223f
// 力单位换算系数: Hartree/Bohr -> eV/Angstrom
#define CONSTANT_HARTREE_BOHR_TO_EV_ANGSTROM 14.3888000f
// 程序中使用的单位时间与物理时间的换算1/20.455*dt=1 ps
#define CONSTANT_TIME_CONVERTION 20.455f
// 程序中使用的单位压强与物理压强的换算
//  压强单位: bar -> kcal/mol/A^3
//  (1 kcal/mol) * (4.184074e3 J/kcal) / (6.023e23 mol^-1) * (1e30 m^3/A^3) *
//  (1e-5 bar/pa) 程序的压强/(kcal/mol/A^3 ) * CONSTANT_PRES_CONVERTION =
//  物理压强/bar
#define CONSTANT_PRES_CONVERTION 6.946827162543585e4f
// 物理压强/bar * CONSTANT_PRES_CONVERTION_INVERSE = 程序的压强/(kcal/mol/A^3 )
#define CONSTANT_PRES_CONVERTION_INVERSE 0.00001439506089041446f
// 角度制到弧度制的转换系数
#define CONSTANT_RAD_TO_DEG 57.2957795f
// 弧度制到角度制的转换系数
#define CONSTANT_DEG_TO_RAD 0.0174532925f
#define CONSTANT_DEG_TO_RAD_DOUBLE 0.017453292519943295769236907684886

// 用于确定盒子变化时坐标和速度如何变化的宏
#define SCALE_COORDINATES_NO 0
#define SCALE_COORDINATES_BY_ATOM 1
#define SCALE_COORDINATES_BY_MOLECULE 2
#define SCALE_VELOCITIES_NO 0
#define SCALE_VELOCITIES_BY_ATOM 1

#define CHAR_LENGTH_MAX 512

// 这一枚举类型定义了粒子传输的flag位(二进制)，该flag位置1说明需要在该方向进行信息传递，该flag位用于生成plan中的数据
enum ghost_dir_enum
{
    send_east = 1,
    send_west = 2,
    send_north = 4,
    send_south = 8,
    send_up = 16,
    send_down = 32
};

// 向量与三角矩阵定义
#include "utils/vector.hpp"

// 用于记录原子组
struct ATOM_GROUP
{
    int atom_numbers = 0;
    int ghost_numbers = 0;
    int* atom_serial;
};

typedef std::vector<std::vector<int>> CPP_ATOM_GROUP;
// 用于记录连接信息
typedef std::map<int, std::set<int>> CONECT;
typedef std::map<std::pair<int, int>, float> PAIR_DISTANCE;

// 求前缀和
struct Prefix_Sum
{
    int blockSize;
    int gridSize;
    int* in;
    int* temp;
    int* out;
    int padded_size;
    Prefix_Sum(int size);
    void Scan();
};

// 用来重置一个已经分配过显存的列表：list。使用CUDA一维block和thread启用
void Reset_List(int* list, const int replace_element, const int element_numbers,
                const int threads = 1024);
void Reset_List(float* list, const float replace_element,
                const int element_numbers, const int threads = 1024);
// 对一个列表的数值进行缩放
void Scale_List(float* list, const float scaler, const int element_numbers,
                const int threads = 1024);

// 对一个列表求和，并将和记录在sum中
void Sum_Of_List(const float* list, float* sum, const int end, int start = 0,
                 int threads = 1024);
void Sum_Of_List(const int* list, int* sum, const int end, int start = 0,
                 int threads = 1024);
void Sum_Of_List(const VECTOR* list, VECTOR* sum, const int end, int start = 0,
                 int threads = 1024);
void Sum_Of_List(const LTMatrix3* list, LTMatrix3* sum, const int end,
                 int start = 0, int threads = 1024);

// 用于生成高斯分布的随机数
// 用seed初始化制定长度的随机数生成器，每个生成器一次可以生成按高斯分布的四个独立的数
__global__ void Setup_Rand_Normal_Kernel(const int float4_numbers,
                                         Philox4_32_10_t* rand_state,
                                         const int seed);
// 用生成器生成一次随机数，将其存入数组中
__global__ void Rand_Normal(const int float4_numbers,
                            Philox4_32_10_t* rand_state, float4* rand_float4);

// 用于设备上的debug，将设备上的信息打印出来
void Device_Debug_Print(const int* x);
void Device_Debug_Print(const float* x);
void Device_Debug_Print(const VECTOR* x);
void Device_Debug_Print(const LTMatrix3* x);

// 用于做快速傅里叶变换前选择格点数目
int Get_Fft_Patameter(float length);
int Check_2357_Factor(int number);

// 自动微分算法相关
#include "utils/sad.hpp"
// 3x3矩阵奇异值分解
#include "utils/svd.hpp"

// for循环封装
#include "utils/device_iterator.hpp"

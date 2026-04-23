---
name: sponge-build-guide
description: >
  询问 SPONGE 构建、编译、环境配置、构建排错时使用。
  适用于 pixi 环境选择、CMake 配置、多后端编译、格式化、打包。
---

本技能适配 SPONGE 版本号：2.0.0-beta.1

用于指导 SPONGE 项目的构建流程，包括环境选择、CMake 配置、编译、格式化和打包。

## 构建的基本流程

所有构建通过 pixi 管理，标准流程为三步：

```bash
pixi install -e <环境名>             # 安装环境依赖
pixi run -e <环境名> configure       # CMake 配置
pixi run -e <环境名> compile         # 编译并安装
```

`pixi install` 会根据 `pixi.toml` 中的 feature 定义安装对应的 conda 依赖（编译器、CUDA toolkit、数学库等）。首次使用某个环境或依赖有变更时必须执行。pixi run 会自动触发 install，但显式执行可以提前发现依赖问题。

编译产物安装到 `$CONDA_PREFIX/bin/SPONGE`，构建目录为 `build-$PIXI_ENVIRONMENT_NAME`。

## 环境选择

根据目标硬件和用途选择环境：

| 环境 | 用途 | 并行后端 |
|------|------|----------|
| `dev-cuda13` | CUDA 13 开发（推荐 NVIDIA GPU 开发） | CUDA |
| `dev-cuda12` | CUDA 12 开发 | CUDA |
| `dev-hip` | AMD GPU / 海光 DCU 开发 | HIP |
| `dev-cpu` | CPU 开发 | 默认 `none`，可手动指定 SIMD |
| `dev-cpu-mpi` | CPU + MPI 开发 | 默认 `none` + MPI，可手动指定 SIMD |
| `cuda13` / `cuda12` / `hip` / `cpu` / `cpu-mpi` | 不带 dev 工具的纯构建环境 | 对应后端 |

`dev-*` 环境额外包含：git、clang-format、cmake-format、ruff、python、pytest、numpy、xponge、ase，用于开发、测试和 benchmark。

非 `dev-*` 环境仅包含编译依赖，不含测试工具。benchmark 和 format 任务只在 `dev-*` 环境下可用。

## HIP 环境特殊说明

`dev-hip` 和 `hip` 环境的 pixi 依赖中**不包含** HIP/ROCm SDK，只安装了 cmake、ninja、tomlplusplus。HIP/ROCm 需要用户在系统层面预先安装好，常见方式：

- 例如通过 `apt install` 安装 ROCm 包、从官方源安装、通过 `module load` 加载对应工具链等

CMake 配置时会通过系统路径查找 `hipcc` / `hipconfig` 来检测 HIP 环境。如果系统未安装或未正确加载，configure 阶段会报错。

## CMake 关键变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PARALLEL` | `auto` | 并行后端选择 |
| `CUDA_ARCH` | `native` | CUDA 架构（pixi task 中 CUDA 环境固定为 `all-major`） |
| `MPI` | `OFF` | 是否启用 MPI |
| `TARGETS` | `SPONGE` | 构建目标 |

`PARALLEL` 可选值：

- `auto`：自动检测，优先 GPU（CUDA > HIP），其次 CPU SIMD（AVX512 > AVX2 > AVX > SSE42 > SVE2 > SVE > NEON），最后 `none`
- `cuda`：CUDA 后端
- `hip`：HIP 后端
- `avx512` / `avx2` / `avx` / `sse42`：x86 SIMD 后端
- `sve2` / `sve` / `neon`：ARM SIMD 后端
- `none`：纯 CPU 无 SIMD

pixi task 中各环境的 `PARALLEL` 已预设好：CUDA 环境为 `cuda`，HIP 为 `hip`，CPU 环境默认为 `none`（可通过参数覆盖）。

## CPU 环境手动指定 SIMD

CPU 环境的 configure task 支持 `parallel` 参数：

```bash
pixi run -e dev-cpu configure avx2
```

这会传入 `-DPARALLEL='avx2'`，覆盖默认的 `none`。

## 编译并行度

compile task 支持 `jobs` 参数，默认为 4：

```bash
pixi run -e dev-cuda13 compile       # 默认 4 线程
pixi run -e dev-cuda13 compile 8     # 8 线程
```

pixi task args 的传参方式是按位置直接传值，不支持 `--arg value` 或 `arg=value` 的写法。

## 平台差异

### Linux x86_64

- 编译器：conda 提供的 `gcc/g++ 11.*`，通过 `CMAKE_CXX_COMPILER` 指定
- CPU 数学库：MKL
- CUDA + MPI 时需要 NCCL

### Linux aarch64

- 编译器：conda 提供的 `gcc/g++ 11.*`
- 数学库：OpenBLAS + FFTW + LAPACKE（无 MKL）
- 支持 SVE2/SVE/NEON SIMD

### Windows x64

- 编译器：MSVC (vs2022)
- 数学库：MKL
- 首次使用可能需要 `pixi run install-msvc`

### macOS ARM (osx-arm64)

- 编译器：Clang 22.1
- 数学库：OpenBLAS + FFTW + LAPACKE
- 仅支持 CPU 后端

## 全局编译设置

以下设置在 `cmake/utils/common.cmake` 中硬编码：

- C++ 标准：C++17
- 构建类型：Release
- 浮点优化：`-ffast-math`（GCC/Clang）、`/fp:fast`（MSVC）、`--use_fast_math`（CUDA）
- 警告：默认关闭（`-w` / `/W0`）
- OpenMP：始终启用，CUDA 环境通过 `-Xcompiler` 传递

## CUDA 特殊处理

- 自动清理 Release 构建中的 `-g -G` 调试标志
- 压制 `177-D` 未引用 kernel 警告
- 压制弃用 GPU 架构警告
- OpenMP 标志通过 `-Xcompiler` 转发给 nvcc

## MPI 构建

MPI 仅在 `cpu-mpi` / `dev-cpu-mpi` 环境下启用：

```bash
pixi run -e dev-cpu-mpi configure
pixi run -e dev-cpu-mpi compile
```

configure 会自动加 `-DMPI=ON` 和 `-DMPI_CXX_COMPILER="$CONDA_PREFIX/bin/mpicxx"`。

CUDA + MPI 场景（当前 pixi.toml 未定义此环境）需要额外链接 NCCL。

## 格式化

```bash
pixi run -e dev-cuda13 format        # 自动修复所有文件
pixi run -e dev-cuda13 format-check  # 仅检查，不修改
```

覆盖三类文件：

- C++（`.cpp` / `.h` / `.hpp`）：clang-format，要求 UTF-8 with BOM
- Python（`.py` / `.pyi` / `.ipynb`）：ruff format
- CMake（`CMakeLists.txt` / `.cmake`）：cmake-format

## 打包

```bash
pixi run -e dev-cuda13 package-conda
```

生成 conda v2 格式（`.conda`）包到 `packaging/outputs/`。打包脚本会：

- 根据环境名提取后端变体（如 `dev-cuda13` → `cuda13`）
- 自动探测平台和架构
- Linux 上用 patchelf 设置 RPATH
- macOS 上用 install_name_tool 设置路径
- 使用 zstd 压缩

## PRIPS 插件构建

```bash
pixi run -e dev-cuda13 prips-build   # 构建 sdist
pixi run -e dev-cuda13 pip install -e ./plugins/prips  # 可编辑安装
```

PRIPS 是 Python 插件接口，用于对接外部机器学习势场。

## 常见问题排查

### configure 找不到 CUDA

确认使用了 CUDA 环境（`dev-cuda12` 或 `dev-cuda13`），pixi 会自动安装 `cuda-nvcc`。不要在 CPU 环境里手动指定 `-DPARALLEL=cuda`。

### 编译器版本不匹配

CUDA 环境在 Linux x86_64 上指定 `CMAKE_CXX_COMPILER` 为 conda 提供的 `x86_64-conda-linux-gnu-c++`（GCC 11），不要使用系统编译器。

### 切换后端后编译出错

不同后端的构建目录是隔离的（`build-dev-cuda13`、`build-dev-cpu` 等），互不影响。如果同一环境切换了 CMake 选项，需要删除旧构建目录或重新 configure。

### nvcc 警告 "incompatible redefinition for option compiler-bindir"

这是 conda 环境下 nvcc 的已知警告，不影响编译结果，可以忽略。

### format 报错 BOM 缺失

C++ 源文件要求 UTF-8 with BOM。运行 `pixi run format` 会自动修复。

### 误用系统编译器或动态库

pixi 环境应提供完整的编译工具链，但有时 CMake 会意外找到系统自带的编译器或动态库。如果 pixi 环境内依赖已完备，可以用 `pixi install -e <环境名> --clean-envs` 重建干净环境来排除系统污染。

但部分场景本身就依赖 pixi 环境外部的工具链，此时不能用 `--clean-envs` 解决，需要自行处理好环境变量后再调用 pixi，例如：

- Windows 下依赖系统安装的 MSVC
- HIP 环境依赖系统安装的 ROCm / DTK

### 构建目录清理

```bash
rm -rf build-dev-cuda13   # 清理特定环境的构建
rm -rf build-*            # 清理所有构建
```

## CMake 文件结构

```
CMakeLists.txt                    # 根配置，串联以下模块
cmake/
  utils/
    common.cmake                  # 编译标准、浮点优化、OpenMP
    parallel.cmake                # 并行后端选择与检测
    checkBackend.cmake            # SIMD 特性检测
    mpi.cmake                     # MPI 配置
    targets.cmake                 # 构建目标加载
    warning.cmake                 # 警告压制
  parallel/
    cuda.cmake                    # CUDA 后端
    hip.cmake                     # HIP 后端
    avx512.cmake / avx2.cmake / avx.cmake / sse42.cmake  # x86 SIMD
    sve2.cmake / sve.cmake / neon.cmake                   # ARM SIMD
    none.cmake                    # 纯 CPU（含用于即时编译的 LLVM/Clang 依赖）
  math/
    cuda.cmake                    # CUDA 数学库（cuFFT、cuBLAS 等）
    hip.cmake                     # HIP 数学库（hipFFT、hipBLAS 等）
    mkl.cmake                     # Intel MKL（x86 CPU）
    open_source.cmake             # OpenBLAS + FFTW（ARM CPU）
  targets/
    SPONGE.cmake                  # SPONGE 可执行文件的源文件列表
  modules/
    Findamd_comgr.cmake           # AMD comgr 查找模块
```

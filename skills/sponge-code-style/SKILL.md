---
name: sponge-code-style
description: >
  询问 SPONGE 代码风格、命名规范、格式化工具时使用。
  适用于 C++、Python、CMake 的编码规范和格式化流程。
---

本技能适配 SPONGE 版本号：2.0.0-beta.1

## 格式化工具与版本

| 语言 | 工具 | 版本 | 配置文件 |
|------|------|------|----------|
| C++ (`.cpp` / `.h` / `.hpp`) | clang-format | 22.1.0 | `.clang-format` |
| Python (`.py` / `.pyi` / `.ipynb`) | ruff format | 0.15.1 | `ruff.toml` |
| CMake (`CMakeLists.txt` / `.cmake`) | cmake-format | 0.6.13 | 默认配置 |

## 格式化命令

```bash
pixi run -e dev-cuda13 format        # 自动修复所有文件
pixi run -e dev-cuda13 format-check  # 仅检查，不修改
```

两者都通过 `scripts/pre-commit` 脚本执行，覆盖所有 git 跟踪的文件。

## 文件编码

所有 C++ 源文件要求 **UTF-8 with BOM**（文件头 `0xEF 0xBB 0xBF`）。pre-commit 钩子会检查并在 `format` 模式下自动添加。

## C++ 代码风格

### clang-format 配置要点

- 基于 Google 风格，有修改
- 缩进：4 空格，不使用 tab
- 行宽限制：80 字符
- 大括号风格：Allman（左大括号独占一行）
- 允许短 if 和短循环写在一行
- 行尾：不强制，保留文件原有风格

### C++ 命名规范

| 元素 | 风格 | 示例 |
|------|------|------|
| 结构体名 | UPPER_CASE | `PAIRWISE_FORCE`、`BERENDSEN_THERMOSTAT_INFORMATION` |
| 成员方法 | PascalCase | `Initial()`、`Read_Configuration()`、`Compute_Force()` |
| 成员变量 | lower_case | `atom_numbers`、`is_initialized`、`module_name` |
| 常量/宏 | UPPER_CASE | `CHAR_LENGTH_MAX`、`TWO_DIVIDED_BY_SQRT_PI` |
| CUDA kernel | lower_case 或混合 | `pairwise_force_scatter_types()`、`Copy_LJ_Type_To_New_Crd()` |
| 文件名 | 混合风格 | `pairwise_force.cpp`、`Berendsen_thermostat.cpp`、`Middle_Langevin_MD.cpp` |
| 目录名 | 混合风格 | `custom_force/`、`Domain_decomposition/`、`Lennard_Jones_force/` |

文件名和目录名没有严格统一的规范，沿用已有的命名即可。新增模块时参考同级目录的命名风格。

### C++ 注释

- 允许中文注释
- 不要求为已有代码补充注释或文档
- 只在逻辑不自明的地方添加注释

## 核心抽象：device_api 与 LaneGroup

SPONGE 通过两层抽象实现跨后端（CUDA / HIP / CPU SIMD）的统一代码编写，这是项目最重要的设计模式。

### device_api

位于 `SPONGE/third_party/device_backend/`，通过宏将不同后端的 API 统一：

| 统一接口 | 作用 |
|----------|------|
| `Launch_Device_Kernel` | kernel 启动（GPU 的 `<<<>>>` / CPU 的函数调用） |
| `deviceMalloc` / `deviceFree` | 设备内存分配与释放 |
| `deviceMemcpy` / `deviceMemcpyAsync` | 内存拷贝 |
| `deviceStream_t` / `deviceStreamCreate` / `deviceStreamSync` | 流管理 |
| `device_mask_t` | 掩码类型（CUDA 32 位 / HIP 64 位） |
| `deviceShflDown` / `deviceBallot` / `deviceActiveMask` | warp 级原语 |
| `FFT_HANDLE` / `BLAS_HANDLE` / `SOLVER_HANDLE` | 数学库句柄 |

后端选择在 `common.h` 中通过编译宏决定：

```cpp
#ifdef USE_HIP
#include "third_party/device_backend/hip_api.h"
#elif defined(USE_CUDA)
#include "third_party/device_backend/cuda_api.h"
#else
#include "third_party/device_backend/cpu_api.h"
#endif
```

编写新代码时使用这些统一接口，不要直接调用 CUDA/HIP/CPU 的原生 API。

### LaneGroup

位于 `SPONGE/third_party/lane_group/`，抽象了 warp/SIMD lane 级并行操作。`LaneGroup` 是一个包含静态方法的结构体，在所有后端下提供一致的接口：

| 方法 | 作用 |
|------|------|
| `LaneGroup::Width()` | lane 数量（CUDA 32 / AVX 8 / SSE 4 / scalar 1） |
| `LaneGroup::Lane_Id()` | 当前 lane 的索引 |
| `LaneGroup::Active_Mask()` | 当前活跃 lane 的掩码 |
| `LaneGroup::Ballot(predicate)` | 收集各 lane 的谓词投票 |
| `LaneGroup::Broadcast(val, src_lane)` | 从指定 lane 广播值 |
| `LaneGroup::Shuffle_Down(val, delta)` | lane 间移位传值 |
| `LaneGroup::Reduce_Sum(val)` | lane 内归约求和 |
| `LaneGroup::First_Lane(mask)` | 掩码中最低活跃 lane |
| `LaneGroup::Count(mask)` | 掩码中活跃 lane 数量 |
| `LaneGroup::Prefix_Count(mask)` | 当前 lane 之前的活跃数量 |

后端实现通过 `lane_group/backend.h` 分发：

```
CUDA/HIP  → warp 原语（__shfl_sync 等）
AVX512    → 16 lane SIMD intrinsics
AVX/AVX2  → 8 lane SIMD intrinsics
SSE4.2    → 4 lane SIMD intrinsics
NEON      → 4 lane ARM SIMD
SVE/SVE2  → 动态宽度 ARM 向量
scalar    → 单 lane 标量回退
```

### 编写规范

- 涉及并行计算的代码应使用 `LaneGroup` 接口，不要直接写 CUDA warp intrinsics 或 SIMD intrinsics
- 涉及设备内存、流、kernel 启动的代码应使用 `device_api` 宏，不要直接调用 `cudaMalloc` 或 `hipMalloc`
- 如果某段逻辑确实只适用于特定后端，用 `#ifdef USE_GPU` / `#ifdef USE_CUDA` / `#ifdef USE_HIP` 等宏隔离

## Python 代码风格

### ruff 配置要点

```toml
line-length = 80
target-version = "py39"

[format]
quote-style = "double"
indent-style = "space"

[lint]
select = ["I"]    # 仅启用 isort（import 排序）
```

- 行宽限制：80 字符
- 使用双引号
- 空格缩进
- lint 只检查 import 排序

### Python 命名规范

| 元素 | 风格 | 示例 |
|------|------|------|
| 函数/方法 | snake_case | `parse_mdout_column()`、`write_thermostat_mdin()` |
| 测试函数 | `test_` + snake_case | `test_thermostat_tip3p_water()`、`test_reaxff_dimer()` |
| 变量 | snake_case | `statics_path`、`outputs_path`、`mpi_np` |
| 常量 | UPPER_CASE | `MINIMIZATION_STEP_LIMIT`、`TAIL_SAMPLES` |
| 类名 | PascalCase | `Outputer`、`Runner`、`Extractor` |

## CMake 代码风格

- 使用 cmake-format 格式化，`--line-ending auto`
- 变量名：UPPER_CASE（`SPONGE_SOURCES`、`PROJECT_ROOT_DIR`）
- 目标名：按实际命名（`SPONGE`、`sponge_toml`）

## pre-commit 钩子

`scripts/pre-commit` 是项目统一的格式检查入口，通过环境变量控制行为：

| 环境变量 | 值 | 说明 |
|----------|------|------|
| `GITHOOK_MODIFY` | `0` / `1` | 是否自动修复 |
| `GITHOOK_AUTO_ADD` | `0` / `1` | 修复后是否自动 `git add` |
| `GITHOOK_ALL` | `0` / `1` | 检查所有跟踪文件还是仅暂存文件 |
| `GITHOOK_FILES` | 文件列表 | 显式指定要检查的文件 |

`pixi run format` 对应 `MODIFY=1, ALL=1`，`pixi run format-check` 对应 `MODIFY=0, ALL=1`。

## 提交前检查流程

1. 写完代码后运行 `pixi run format`
2. 确认 `pixi run format-check` 无报错
3. 提交代码

如果 pre-commit 钩子已安装为 git hook，提交时会自动检查暂存文件的格式。

# HIP / dev-hip 实现计划

## 概要

- [ ] 目标：在 `linux-64` 上新增 `pixi` 环境 `hip` 和 `dev-hip`，让仓库具备原生 AMD/ROCm 6.3.x 的 HIP 构建能力，并接入现有本地构建与打包流程。
- [ ] 范围：本次不仅新增 `pixi` 环境，还要补齐 CMake HIP backend、设备抽象层、JIT、FFT/BLAS/Solver 映射、源码中的 CUDA 专用接口替换，以及 `dev-hip` 的打包接入。
- [ ] 约束：
  - [ ] 仅支持 `linux-64`
  - [ ] 采用原生 AMD/ROCm 路线，不支持 HIP-on-NVIDIA
  - [ ] 锁定 `ROCm 6.3.x`
  - [ ] 保留 GPU JIT 能力
  - [ ] 本次不实现 `hip-mpi` / `dev-hip-mpi`
- [ ] 交付结果：
  - [ ] `pixi install -e hip` / `pixi install -e dev-hip` 可用
  - [ ] `pixi run -e hip configure|compile` 可用
  - [ ] `pixi run -e dev-hip configure|compile|package` 可用
  - [ ] `PARALLEL=hip` 成为正式 CMake backend
  - [ ] HIP 构建下 `USE_GPU + USE_HIP` 生效，`USE_CUDA` 不生效

## 公共接口与行为变更

- [ ] 新增环境接口：
  - [ ] `hip`
  - [ ] `dev-hip`
- [ ] 新增构建接口：
  - [ ] `cmake -DPARALLEL=hip`
- [ ] 新增编译宏：
  - [ ] `USE_HIP`
  - [ ] 继续复用 `USE_GPU`
- [ ] 新增设备后端头文件：
  - [ ] `SPONGE/third_party/device_backend/hip_api.h`
- [ ] 新增后端中立常量/类型：
  - [ ] `DEVICE_BLAS_OP_N`
  - [ ] `DEVICE_BLAS_OP_T`
  - [ ] `DEVICE_BLAS_OP_C`
  - [ ] `DEVICE_FILL_MODE_UPPER`
  - [ ] `DEVICE_EIG_MODE_VECTOR`
- [ ] DLPack 设备类型规则：
  - [ ] `USE_CUDA` 时使用 `kDLCUDA`
  - [ ] `USE_HIP` 时使用 `kDLROCM`
- [ ] JIT 外部接口保持不变：
  - [ ] `JIT_Function` 名称不变
  - [ ] CUDA 继续走 NVRTC
  - [ ] HIP 改走 HIPRTC + HIP module API
- [ ] 打包接口保持不变：
  - [ ] `python conda/build_package.py --env dev-hip`
  - [ ] 但新增 HIP 运行库可移植性检查

## 阶段 1：构建与环境骨架落地

- [ ] 1.1 更新 `pixi` 环境定义
  - [ ] 修改 `pixi.toml`
  - [ ] 在 `[environments]` 中新增 `hip = ["hip"]`
  - [ ] 在 `[environments]` 中新增 `dev-hip = ["dev", "hip"]`
  - [ ] 新增 `feature.hip.target.linux-64.dependencies`
  - [ ] 依赖固定为：
    - [ ] `cmake >=3.24`
    - [ ] `ninja`
    - [ ] `hipcc 6.3.*`
    - [ ] `hip-devel 6.3.*`
    - [ ] `rocfft`
    - [ ] `gcc_linux-64 11.*`
    - [ ] `gxx_linux-64 11.*`
    - [ ] `tomlplusplus`
  - [ ] 不为 `win-64` / `osx-arm64` / `linux-aarch64` 增加 HIP 配置
  - [ ] 新增 `feature.hip.target.linux-64.tasks.configure`
  - [ ] 新增 `feature.hip.target.linux-64.tasks.compile`
  - [ ] `configure` 固定传入：
    - [ ] `-DPARALLEL='hip'`
    - [ ] `-DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX"`
    - [ ] `-DCMAKE_CXX_COMPILER="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"`
    - [ ] `-DCMAKE_HIP_COMPILER="$CONDA_PREFIX/bin/hipcc"`

- [ ] 1.2 新增 CMake HIP backend
  - [ ] 新增 `cmake/parallel/hip.cmake`
  - [ ] 在 HIP backend 中执行 `enable_language(HIP)`
  - [ ] 设置 `CPP_DIALECT = HIP`
  - [ ] 定义 `USE_GPU`
  - [ ] 定义 `USE_HIP`
  - [ ] 增加可选 cache 变量 `HIP_ARCH`
  - [ ] 当 `HIP_ARCH` 非空时设置 `CMAKE_HIP_ARCHITECTURES`
  - [ ] 查找 HIP runtime / HIPRTC / HIPFFT / HIPBLAS / HIPSOLVER / HIPRAND（注意可能ROCm自带find_package）
  - [ ] 将 HIP 库统一挂到 `common_libraries`

- [ ] 1.3 扩展 backend 自动检测与 CMake 公共逻辑
  - [ ] 修改 `cmake/utils/checkBackend.cmake`
  - [ ] 新增 `CheckHIP()`
  - [ ] 规则固定为：
    - [ ] `hipcc` 可执行存在
    - [ ] `hipconfig` 可执行存在
    - [ ] `hipconfig --platform` 为 `amd`
  - [ ] 让 GPU auto-detect 顺序改为 `CheckCuda -> CheckHIP`
  - [ ] 修改 `cmake/utils/parallel.cmake`，使 `hip` 进入合法 backend 列表
  - [ ] 修改 `cmake/utils/common.cmake`
  - [ ] 为 `HIP` 增加 `CMAKE_HIP_FLAGS`
  - [ ] 为 `HIP` 单独处理 OpenMP host flags 透传
  - [ ] 保留 CUDA 原有 OpenMP 透传逻辑
  - [ ] 保留 CPU 原有 OpenMP 逻辑
  - [ ] 修改 `cmake/utils/warning.cmake`
  - [ ] 保留 CUDA 专属 suppression
  - [ ] HIP 首版不新增 suppression

- [ ] 1.4 补齐 HIP 依赖查找模块
  - [ ] 使用ROCm包自带的cmake config

## 阶段 2：HIP 端到端后端接线与代码迁移

- [ ] 2.1 新增 HIP 设备抽象层
  - [ ] 新增 `SPONGE/third_party/device_backend/hip_api.h`
  - [ ] 以 `cuda_api.h` 为模板提供 HIP 对等映射
  - [ ] 映射内容至少覆盖：
    - [ ] 设备初始化
    - [ ] `deviceMalloc/deviceMemcpy/deviceMemset/deviceFree`
    - [ ] stream 类型与同步
    - [ ] error 类型与错误字符串
    - [ ] kernel launch 宏
    - [ ] FFT handle 与执行接口
    - [ ] BLAS handle 与执行接口
    - [ ] Solver handle 与执行接口
    - [ ] Random 状态与初始化接口
    - [ ] JIT 所需 runtime/compiler 类型
  - [ ] 在 `hip_api.h` 中定义 `GPU_ARCH_NAME "HIP"`

- [ ] 2.2 统一 backend 选择逻辑
  - [ ] 修改 `SPONGE/common.h`
  - [ ] 设备 backend 选择规则改为：
    - [ ] `USE_HIP -> hip_api.h`
    - [ ] `USE_CUDA -> cuda_api.h`
    - [ ] 其他 -> `cpu_api.h`
  - [ ] 修改 `SPONGE/third_party/device_backend/cuda_api.h`
  - [ ] 修改 `SPONGE/third_party/device_backend/cpu_api.h`
  - [ ] 三套 backend 都导出同一组中立常量：
    - [ ] `DEVICE_BLAS_OP_*`
    - [ ] `DEVICE_FILL_MODE_UPPER`
    - [ ] `DEVICE_EIG_MODE_VECTOR`

- [ ] 2.3 替换公共代码中的裸 CUDA 调用
  - [ ] 修改 `SPONGE/common.cpp`
  - [ ] 将直接出现的 `cudaMalloc/cudaMemcpy/cudaMemset` 替换为统一抽象调用
  - [ ] 全仓库规则固定为：
    - [ ] 公共路径不允许再出现裸 `cuda*` 调用
    - [ ] 必须经由 `device*` 抽象层

- [ ] 2.4 适配设备信息与启动信息输出
  - [ ] 修改 `SPONGE/control.cpp`
  - [ ] 修改 `SPONGE/utils/control/os.hpp`
  - [ ] `USE_HIP` 下输出 HIP/ROCm backend 信息
  - [ ] `USE_HIP` 下输出运行时架构名
  - [ ] 不复用 CUDA compute capability 检查
  - [ ] 保留 CUDA 路径原行为

- [ ] 2.5 修正 DLPack 设备类型
  - [ ] 修改 `SPONGE/utils/tensor.hpp`
  - [ ] `USE_HIP` 时将设备类型设置为 `kDLROCM`
  - [ ] `USE_CUDA` 时继续使用 `kDLCUDA`

- [ ] 2.6 修正 MPI 抽象中的 GPU 路径判断
  - [ ] 修改 `SPONGE/third_party/mpi.hpp`
  - [ ] 将对 `GPU_ARCH_NAME` 的公共判断改成显式 `USE_XCCL`
  - [ ] 防止 `USE_HIP` 误进入 NCCL 宏分支
  - [ ] 本次不为 HIP 实现 `USE_XCCL`

- [ ] 2.7 保持 FFT / BLAS / Solver 公共接口不变并映射到 HIP
  - [ ] 修改 `SPONGE/utils/fft.hpp`
  - [ ] `USE_HIP` 时使用 HIP FFT 类型与执行接口
  - [ ] 修改 `SPONGE/quantum_chemistry/scf/matrix.hpp`
  - [ ] 将直接写死的：
    - [ ] `CUBLAS_OP_*`
    - [ ] `CUSOLVER_EIG_MODE_VECTOR`
    - [ ] `CUBLAS_FILL_MODE_UPPER`
    - [ ] 替换为后端中立常量
  - [ ] 确保量化化学路径不再显式依赖 CUDA 枚举名

- [ ] 2.8 实现 HIP 版 GPU JIT
  - [ ] 修改 `SPONGE/third_party/jit/jit.hpp`
  - [ ] 保持 `JIT_Function` 外部接口不变
  - [ ] 新增 `USE_HIP` 路径：
    - [ ] 用 HIPRTC 编译源代码
    - [ ] 生成 code object
    - [ ] 用 HIP module API 加载模块
    - [ ] 获取函数句柄
    - [ ] 用 HIP module launch API 发射 kernel
  - [ ] 编译选项使用 `--offload-arch=<gfx*>`
  - [ ] 失败时将 HIPRTC / module / launch 错误写入 `error_reason`

- [ ] 2.9 修正运行时注入源码中的 CUDA 宏假设
  - [ ] 修改 `SPONGE/wall/soft_wall.cpp`
  - [ ] 修改 `SPONGE/collective_variable/combine.cpp`
  - [ ] 修改 `SPONGE/custom_force/listed_forces.cpp`
  - [ ] 修改 `SPONGE/custom_force/pairwise_force.cpp`
  - [ ] 规则固定为：
    - [ ] GPU 路径定义 `USE_GPU`
    - [ ] CUDA 路径定义 `USE_CUDA`
    - [ ] HIP 路径定义 `USE_HIP`
    - [ ] 不再只判断 `__CUDACC__`

- [ ] 2.10 接入 `dev-hip` 打包并增加坏包阻断
  - [ ] 修改 `conda/build_package.py`
  - [ ] 新增 `hip` 变体识别
  - [ ] 为 `hip/linux-64` 增加运行时依赖映射：
    - [ ] `hip-runtime-amd >=6.3,<7`
    - [ ] `rocfft >=1.0,<2`
    - [ ] `libstdcxx-ng`
    - [ ] `libgcc-ng`
  - [ ] 不在 metadata 中声明当前未确认的 `rocblas` / `rocsolver` / `rccl` 包名
  - [ ] 在打包前新增 `ldd` 校验
  - [ ] 若出现未解析依赖则直接失败
  - [ ] 若关键 HIP 依赖来自系统 ROCm 且不在 env 中则直接失败
  - [ ] 不实现自动 vendor `/opt/rocm` 库进包

## 测试与验收清单

- [ ] A. 环境层
  - [ ] `pixi install -e hip` 成功
  - [ ] `pixi install -e dev-hip` 成功
  - [ ] 仅 `linux-64` 提供 HIP 环境

- [ ] B. 配置层
  - [ ] `pixi run -e hip configure` 成功
  - [ ] `pixi run -e dev-hip configure` 成功
  - [ ] `PARALLEL=hip` 被识别为合法 backend
  - [ ] `PARALLEL=auto` 在 AMD ROCm 主机优先选 HIP

- [ ] C. 编译层
  - [ ] `pixi run -e hip compile` 成功生成并安装 `SPONGE`
  - [ ] HIP 构建下 `USE_HIP + USE_GPU` 生效
  - [ ] HIP 构建下 `USE_CUDA` 不生效

- [ ] D. 运行层
  - [ ] 程序启动输出 HIP/ROCm 信息
  - [ ] 设备枚举与设备选择可用
  - [ ] DLPack 设备类型为 `kDLROCM`
  - [ ] 无裸 CUDA API 符号错误

- [ ] E. JIT
  - [ ] HIP 下最小 JIT 示例可编译
  - [ ] HIP 下最小 JIT 示例可装载
  - [ ] 编译失败日志能进入 `error_reason`
  - [ ] launch 失败日志能进入 `error_reason`

- [ ] F. 数学库接口
  - [ ] FFT wrapper 在 HIP 下可构建
  - [ ] BLAS wrapper 在 HIP 下可构建
  - [ ] Solver wrapper 在 HIP 下可构建
  - [ ] 量化化学矩阵接口不再显式依赖 CUDA 枚举名

- [ ] G. MPI 防护
  - [ ] `MPI=ON + PARALLEL=hip` 配置阶段直接报错
  - [ ] `MPI=OFF + PARALLEL=hip` 正常配置

- [ ] H. 打包
  - [ ] `python conda/build_package.py --env dev-hip` 能识别 `hip`
  - [ ] 所有依赖来自 env 时打包成功
  - [ ] 有未解析库时打包失败
  - [ ] 有系统 ROCm 外部依赖时打包失败

## 默认假设

- [ ] 默认平台是 `linux-64`
- [ ] 默认后端路线是原生 AMD/ROCm
- [ ] 默认版本线是 `ROCm 6.3.x`
- [ ] 默认不支持 HIP+MPI
- [ ] 默认允许 HIP 自动推断架构，只有显式设置 `HIP_ARCH` 才覆盖
- [ ] 默认优先使用 `$CONDA_PREFIX` 内 HIP 组件
- [ ] 默认不自动打包系统 ROCm `.so`

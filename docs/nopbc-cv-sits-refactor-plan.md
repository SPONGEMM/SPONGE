# NOPBC、CV 与 SITS 解耦修改计划

## 1. 文档状态

- 目标分支：`lab/sidereus-ai`
- 源码基线：`3ca40fc5ddca069d42547467b45dc3f2e6347949`
- 文档性质：设计与实施计划，不代表功能已经修改或通过验收
- 讨论范围：NOPBC、CV/偏置调度、CV 边界语义、全体系 SITS、H5 SITS 配置时序

接口更新：四个 PR 现使用单一 Boundary；第 3、4 节是上述旧基线的诊断记录，
不能作为修改后源码状态。实际边界所有权和逐模块接口以
`pbc-nopbc-boundary-decoupling-plan.md` 为准，未实现下文示例中的独立 CV 上下文类。

## 2. 结论摘要

本次修改应完成四件事：

1. 将 CV、steer、CV restraint 和 metadynamics 的执行从 PME/PM 进程条件中拆出。
2. 为 CV 引入明确的边界条件策略；NOPBC 下不再隐式使用周期最小镜像。
3. 正式支持 NOPBC 与全体系 SITS（`atom_numbers = "ITS"` 或 `"ALL"`）组合。
4. 在 H5/legacy SITS 配置完全解析之后统一执行兼容性检查，保证 NOPBC 下的 selective SITS 始终明确拒绝。

本次不支持 NOPBC 与 selective SITS 的组合，也不实现 NOPBC selective energy/force decomposition。

目标支持矩阵如下：

| 功能 | PBC | NOPBC |
|---|---:|---:|
| 无 SITS | 支持 | 支持 |
| 全体系 SITS：`ITS`/`ALL` | 支持 | 支持 |
| Selective SITS：整数、文件或 H5 `atom_indices` | 支持 | 明确拒绝 |
| distance/displacement/angle/dihedral CV | 周期最小镜像 | 直接笛卡尔位移 |
| position/RMSD CV | 保持当前定义 | 保持当前非周期坐标定义 |
| scaled-position CV | 支持 | 默认拒绝，除非以后定义参考盒 |
| box-length CV | 支持 | 默认拒绝，除非以后定义参考盒 |

## 3. SPONGE 当前的 NOPBC 实现

### 3.1 配置入口和硬限制

`pbc` 默认为 `true`。当输入设置 `pbc = false` 时，
`periodic_box_condition_information::Initial` 仍依次执行：

1. `No_PBC_Check`；
2. `PBC_Check`；
3. 保存 `cell0`。

也就是说，NOPBC 并不会跳过 cell/rcell 的构造。当前实现仍使用输入坐标尾部的
`box_length` 和 `box_angle` 构造一个可逆的 `cell` 与 `rcell`。

`No_PBC_Check` 当前施加以下限制：

- 只允许单进程；`MPI_size > 1` 直接报错。
- `cutoff < 100 Å` 时给出警告，但不会终止。
- 三个方向的 box length 都必须至少为 `900 Å`。
- 不允许 NPT。
- SITS 检查直接读取 controller 中的 legacy 字符串；只有 `ITS`/`ALL` 被放行。

源码入口：

- `SPONGE/MD_core/pbc.hpp:3-101`
- `SPONGE/MD_core/MD_core.cpp:332-369`

### 3.2 NOPBC 不是“没有 box”，而是“大虚拟盒 + 非周期核心非键力”

当前实现保留一个至少 900 Å 的虚拟盒，主要是为了满足大量仍接收
`cell/rcell` 的公共接口。真正的 NOPBC 差异发生在核心非键力内核：

```cpp
VECTOR dr = crd[atom_j] - crd[atom_i];
```

LJ 和 Coulomb NOPBC 内核不调用 minimum-image，也不根据 cell 折返位移。
它们仍执行以下逻辑：

- 遍历 `i < j` 的原子对；
- 查询 exclusion list；
- 使用直接笛卡尔距离；
- 只计算 `r < cutoff` 的相互作用；
- 对两个原子累加大小相等、方向相反的力；
- 需要势能时同时累加 atom energy 和模块能量。

因此，大盒并不会取消 cutoff。若希望近似完整的真空全库仑相互作用，cutoff 本身必须足够大；
这也是当前代码在 cutoff 小于 100 Å 时发出警告的原因。

当前 NOPBC 非键实现是全原子对遍历，而不是 PBC neighbor-list 路径，计算复杂度近似为
`O(N²)`。

源码入口：

- `SPONGE/NO_PBC/Lennard_Jones_force_No_PBC.cpp:6-123`
- `SPONGE/NO_PBC/Coulomb_Force_No_PBC.cpp:3-96`

### 3.3 初始化分叉

`Main_Initial` 在 `md_info.pbc.pbc` 上分成两条力场初始化路径。

PBC 路径初始化：

- PBC LJ 和 soft-core LJ；
- PME/Particle Mesh；
- pairwise force；
- PBC NB14；
- SITS；
- selective SITS 的 dihedral、NB14 和 CMAP helper；
- 后续的 neighbor list 和 solvent LJ。

NOPBC 路径初始化：

- `LJ_NOPBC`；
- `CF_NOPBC`；
- 可选 GB；
- 使用 NOPBC LJ 参数表的普通 NB14；
- SITS 核心。

NOPBC 不初始化 PME、PBC LJ、selective SITS helper 和 PBC neighbor list。

bond、angle、Urey-Bradley、CMAP、dihedral、improper、listed force 等公共 bonded
模块在分叉之后统一初始化。它们的接口仍会收到 `cell/rcell`，因此是否使用周期位移由各模块
自己的实现决定；当前“大虚拟盒”在这里充当兼容层，而不是显式的边界策略。

源码入口：

- `SPONGE/main.cpp:1160-1205`
- `SPONGE/main.cpp:1249-1282`

### 3.4 每步力计算

单步力计算的主要顺序是：

1. 根据 SITS 需求设置势能计算标志。
2. 调用 `pm.Get_Atoms`。
3. 清零 domain-decomposition 力和 virial。
4. 更新 ghost 与 neighbor list。
5. 计算 ReaxFF（若启用）。
6. 调用 NOPBC LJ、NOPBC Coulomb 和可选 GB。
7. 调用 PME excluded-force 接口。
8. 计算 bonded、NB14、wall、plugin、restraint 等公共贡献。
9. 计算 PME reciprocal、CV 和 CV bias。
10. 调用 SITS 更新和增强。
11. 回传虚原子力。

其中多个 PBC/PM 公共调用在 NOPBC 下仍会出现，但依靠对象的 `is_initialized` 或
`PM_MPI_size` 守卫变为 no-op：

- `neighbor_list.Update` 在未初始化时立即返回；
- PME reciprocal 和 excluded-force 要求 `pm.is_initialized`；
- PM 坐标/力通信在 `PM_MPI_size == 0` 时返回。

这种“无条件调用 + 对象内部 no-op”让公共主循环比较统一，但也造成了模块能力不透明：
调用者无法从控制流直接判断功能是真正执行了还是静默跳过。当前 CV/PME 问题就是该模式的
一个具体后果。

源码入口：

- `SPONGE/main.cpp:1348-1635`
- `SPONGE/neighbor_list/neighbor_list.cpp:834-841`
- `SPONGE/PM_force/PM_force.cpp:1067-1074`
- `SPONGE/PM_force/PM_force.cpp:1816-1824`

### 3.5 PME/PM 与 NOPBC 进程状态

NOPBC 不调用 `pm.Initial`，全局 `pm` 对象保持未初始化，其 `PM_MPI_size` 为 0。
`Main_Process_Management` 随后把该值复制到 controller：

```text
MPI_size = 1
PP_MPI_size = 1
PM_MPI_size = 0
PM_MPI_rank = -1
```

NOPBC 又在更早阶段禁止多进程，因此当前合法 NOPBC 运行只有一个 PP 进程，没有 PM 进程。

这本身不妨碍 CV。单进程 NOPBC 的 `dd.crd` 已经是完整坐标，`dd.frc` 也是可直接累加的
完整力缓冲。真正的问题是主循环把 CV 和 PME reciprocal 放在同一个
`PM_MPI_size == 1` 条件中，导致 NOPBC 的 CV 被整体跳过。

源码入口：

- `SPONGE/main.cpp:1535-1582`
- `SPONGE/main.cpp:2147-2173`
- `SPONGE/PM_force/PM_force.cpp:353-374`

### 3.6 坐标、轨迹和 box 输出

NOPBC 输出时：

- 不执行 PBC molecule coordinate mapping；
- 直接输出未包裹的原始坐标；
- H5MD box 仍写入对角 `box_length`，因此输出中仍能看到大虚拟盒；
- molecule 模块在 NOPBC 下不建立用于周期重映射的状态。

所以当前 NOPBC 的坐标会自由漂移，不会被重新映射回虚拟盒。任何仍调用 periodic minimum-image
的上层模块，最终都可能在原子跨越虚拟盒半长时与 NOPBC 力场产生语义分歧。

源码入口：

- `SPONGE/main.cpp:1919-1932`
- `SPONGE/MD_core/output.hpp:38-68`
- `SPONGE/MD_core/mol.hpp:891-894`

### 3.7 当前 NOPBC 架构的本质

当前设计可以概括为：

```text
输入中的大 box
    ├── 构造 cell/rcell，供公共接口继续运行
    ├── 不用于 NOPBC LJ/Coulomb 的位移折返
    └── 仍可能被 CV、bonded、restraint 等上层模块使用

NOPBC 核心非键力
    ├── 直接笛卡尔位移
    ├── 全原子对遍历
    ├── exclusion list
    └── 有限 cutoff

运行条件
    ├── 单进程 PP
    ├── PM 未初始化
    ├── 无 neighbor list
    └── 非 NPT
```

因此，NOPBC 当前是一个独立非键后端，但还不是贯穿所有模块的统一边界条件策略。

## 4. 已确认的问题

### 4.1 CV 与 PME/PM 调度耦合

主循环当前只有在：

```cpp
CONTROLLER::MPI_size == 1 && CONTROLLER::PM_MPI_size == 1
```

时才执行 PME reciprocal、CV print、steer、CV restraint 和 metadynamics。

NOPBC 的 `PM_MPI_size` 为 0，因此 CV 可以成功解析和初始化，却不会在运行时执行。
PME reciprocal 函数内部已经具有 `is_initialized` 守卫，因而没有必要用 PM 进程条件保护整段 CV。

### 4.2 CV 边界语义与 NOPBC 力场不一致

distance、displacement、angle 和 dihedral CV 无条件使用 minimum-image displacement；
NOPBC LJ/Coulomb 使用直接位移。大虚拟盒只能延迟问题，不能定义正确语义。

position 和当前 RMSD 实现主要使用原始坐标；scaled-position 与 box-length 则直接依赖
`cell/rcell`，在 NOPBC 下没有明确物理定义。

### 4.3 全体系 SITS 支持未收尾

全体系 SITS 在每步更新中直接复制系统总能量、总力和总 virial 作为增强对象：

```text
U_enhanced = U_total
F_enhanced = F_total
V_enhanced = V_total
```

因此其核心算法不依赖 PBC、PME、neighbor list 或 minimum-image，架构上可以支持 NOPBC。

当前缺口包括：

- 无 NOPBC + `ITS/ALL` 的运行回归测试；
- NOPBC 下 SITS step output 路径不完整；
- H5 配置可能绕过当前兼容性检查；
- 尚未把支持矩阵写入用户文档。

### 4.4 Selective SITS 必须在 NOPBC 下稳定拒绝

Selective SITS 需要单独构造被增强部分的能量、力和 virial。当前 selected contribution 路径
依赖 PBC LJ、soft-core LJ、neighbor list、PME beta、minimum-image，以及只在 PBC 分支初始化的
dihedral/NB14/CMAP helper。

NOPBC 没有等价的 selected contribution provider。若绕过检查，运行可能使用不完整的
`U_enhanced/F_enhanced`，属于错误物理结果，而不仅仅是缺少一个功能。

### 4.5 H5 SITS 配置检查时序错误

当前 `pbc.Initial` 在 `sits.Initial` 之前执行，而 typed H5 SITS 的 mode 和 atom selection 是在
`sits.Initial` 内由 `sits_h5_input.hpp` 加载并注入 controller。

因此 `No_PBC_Check` 只能可靠看到 legacy controller 字段，不能可靠判断 H5 配置最终是：

- 全体系 `ITS/ALL`；还是
- selective `atom_indices`/整数选择。

边界兼容性检查必须发生在统一配置解析之后。

## 5. 目标设计

### 5.1 显式边界条件策略

沿用 PR1 的边界策略，不再定义第二套类型：

```cpp
enum class BoundaryPolicy : std::uint8_t
{
    Open = 0,
    Periodic = 1
};
```

`pbc=false` 产生 `BoundaryPolicy::Open`。统一的 `md_info.pbc.boundary` 持有
policy/cell/rcell，CV 接收它，不通过盒长推断边界类型，也不另存一套盒子。

### 5.2 独立的 CV 执行上下文

将 CV/bias 调用从 PME reciprocal 调用中拆出。实际实现保留直接调度，
使用 CV_MPI_rank 表达坐标所有权，并向计算函数传同一 Boundary；
不新增含独立盒子副本的 CVExecutionContext。

坐标所有权策略：

| 运行方式 | CV 坐标/力缓冲 |
|---|---|
| 单进程 PBC | `dd.crd` / `dd.frc` |
| 单进程 NOPBC | `dd.crd` / `dd.frc` |
| PP/PM 分离 | `pm.g_crd` / `pm.g_frc`，完成后回传 |

PME reciprocal 继续根据 PM/PME 自身初始化状态执行，不再决定 CV 是否执行。

### 5.3 CV 几何按策略计算

提供统一 displacement primitive：

```cpp
VECTOR Get_Displacement(a, b, boundary);
```

- `BoundaryPolicy::Periodic`：保持当前 minimum-image。
- `BoundaryPolicy::Open`：返回 `a - b`。

distance、displacement、angle、dihedral 统一使用该 primitive。禁止在各 CV 内自行根据 box 长度
猜测边界类型。

scaled-position 和 box-length CV 在 NOPBC 下先明确报 capability error；以后若要支持，应额外引入
“参考盒”概念，而不是复用 NOPBC 的虚拟大盒。

### 5.4 解析后的 SITS 能力模型

在 legacy/H5 配置统一解析后，得到稳定的 selection scope：

```cpp
enum class SitsSelectionScope
{
    AllSystem,
    Selective
};
```

兼容性规则：

```text
PBC + AllSystem   -> allow
PBC + Selective   -> allow
NOPBC + AllSystem -> allow
NOPBC + Selective -> reject with an explicit error
```

不要继续在 `pbc.hpp` 中直接检查 `SITS_atom_numbers` 字符串。推荐在 SITS 配置解析完成后调用：

```cpp
sits.Validate_Boundary_Compatibility(boundary_policy);
```

错误信息应包含最终解析来源和 scope，例如：

```text
Selective SITS is not supported with pbc=false.
Resolved selection source: H5 /sits/SITS/atom_indices.
Use atom_numbers = "ALL" or "ITS" for all-system SITS.
```

### 5.5 全体系 SITS 的 NOPBC 路径

全体系 SITS 继续复用现有 `Update_And_Enhance` 的 non-selective 分支，不新增 NOPBC 专用 SITS
kernel。SITS 只消费已经完成的 total state：

```cpp
sits.Update_And_Enhance(step, total_energy, need_pressure,
                        total_virial, total_force, beta0);
```

需要修复的是生命周期、输出和测试，而不是 SITS 数学公式。

## 6. 分阶段修改计划

### 阶段 0：建立失败基线

目标：在修改运行逻辑前，把当前缺陷固化为最小测试。

- 增加单进程 NOPBC + distance CV 测试，证明当前 CV 被 PM gate 跳过。
- 增加 NOPBC + steer CV 测试，检查修改前没有预期偏置力。
- 增加 H5 selective SITS + NOPBC 测试，证明当前配置时序可以绕过 legacy guard。
- 保存 PBC CV/SITS 基线，防止后续改变已有周期语义。

验收：测试应稳定复现问题，而不是只检查程序是否成功退出。

### 阶段 1：拆分 CV 与 PME 调度

主要修改位置：

- `SPONGE/main.cpp`
- 必要时新增 CV execution helper 文件

步骤：

1. 将 PME reciprocal 调用与 CV/bias 调用拆成两个独立函数或代码块。
2. 单进程时无论 PM 是否初始化，都使用 `dd.crd/dd.frc` 执行 CV。
3. 保留 PP/PM 分离下的全局坐标和力回传路径。
4. 保留 `vatom.Coordinate_Refresh_CV` 和 `Force_Redistribute_CV` 的调用顺序。
5. 确保 metadynamics H5 diagnostic 写出不再依赖 PME 是否执行。

验收：

- NOPBC CV 从 `****` 变为有限值；
- steer/restrain/meta 对力和能量产生可验证变化；
- PBC 单进程及 PP/PM 行为不回归；
- 无 CV 时不增加可测量的运行副作用。

### 阶段 2：引入 CV 边界策略

主要修改位置：

- `SPONGE/MD_core/pbc.h` / `pbc.hpp`
- `SPONGE/collective_variable/CV.h` 及实现
- `SPONGE/collective_variable/simple_cv.cpp`
- CV controller、steer、restrain_cv、metadynamics 的调用接口
- 可能复用或扩展 `third_party/jit/jit_matrix.h` 的 displacement helper

步骤：

1. 定义 `BoundaryPolicy`，由 MD 配置唯一解析。
2. 将 policy 传入 CV compute 路径。
3. distance/displacement/angle/dihedral 根据 policy 选择 displacement。
4. 为 scaled-position 和 box-length 增加 NOPBC capability error。
5. 检查虚原子 CV 坐标刷新及力回传是否需要同一策略。
6. 文档化各 CV 在 PBC/NOPBC 下的定义。

验收：

- 对同一对原子构造超过虚拟盒半长的位移：PBC CV 折返，NOPBC CV 不折返；
- angle/dihedral 在手工笛卡尔几何上匹配参考值；
- CV gradient 与有限差分一致；
- steer、CV restraint、meta 的偏置力方向与 boundary policy 一致；
- PBC 原有验证测试保持通过。

### 阶段 3：统一 H5/legacy SITS 配置与能力检查

主要修改位置：

- `SPONGE/SITS/sits_h5_input.hpp`
- `SPONGE/SITS/SITS.h` / `SITS.cpp`
- `SPONGE/MD_core/pbc.hpp`
- `SPONGE/main.cpp`

步骤：

1. 将最终 selection scope 作为 SITS 解析结果保存，而不是运行时反复读取字符串。
2. H5 和 legacy 输入映射到同一个 `SitsSelectionScope`。
3. 从 `No_PBC_Check` 删除基于 `SITS_atom_numbers` 字符串的判断。
4. 在 SITS 配置完全解析后执行 boundary capability validation。
5. selective + NOPBC 在初始化阶段稳定报错，不进入任何力计算。
6. 错误信息报告 selection source，便于诊断 H5/legacy 配置。

验收：

- legacy `ITS`、`ALL` 与 H5 policy `ITS`、`ALL` 均在 NOPBC 下通过；
- legacy 整数、`atom_in_file` 与 H5 `atom_indices` 均在 NOPBC 下以同类错误拒绝；
- PBC selective SITS 继续初始化全部 helper；
- disabled H5 SITS 不触发兼容性错误。

### 阶段 4：正式支持 NOPBC + 全体系 SITS

主要修改位置：

- `SPONGE/main.cpp`
- `SPONGE/SITS/SITS.cpp`
- output/H5 SITS 路径
- 输入参考文档

步骤：

1. 明确 non-selective SITS 使用 total energy/force/virial。
2. 检查 NOPBC 下 `need_potential` 始终满足 SITS 更新要求。
3. 将普通 `sits.Step_Print` 从 PBC-only 输出分支移到 SITS 生命周期公共位置。
4. 验证 legacy Nk trajectory/restart 输出。
5. 验证 H5 Nk observable 和 native restart state。
6. 在用户文档中写明支持矩阵和 selective 限制。

验收：

- NOPBC `observation`：`h_factor = 1`，力与无 SITS 基线一致；
- NOPBC `iteration`：Nk 在配置间隔更新，所有输出为有限值；
- NOPBC `production`：可从 legacy 和 native H5 restart 恢复并连续运行；
- `ITS` 与 `ALL` 行为一致；
- 修改前后的 PBC 全体系 SITS 在相同输入下保持数值一致。

### 阶段 5：文档、清理与完整回归

- 更新 `docs/input-reference/core.md` 中 NOPBC 的精确限制。
- 更新 `docs/input-reference/collective-variables.md` 中各 CV 的 boundary semantics。
- 更新 `docs/input-reference/enhanced-sampling.md` 中 SITS 支持矩阵。
- 删除不再使用的旧字符串 guard 和重复调度分支。
- 运行格式、编译、focused tests、H5 tests、PBC 回归及 NOPBC 组合测试。

## 7. 测试矩阵

### 7.1 CV 和偏置

| Boundary | CV/功能 | 核心断言 |
|---|---|---|
| PBC | distance/angle/dihedral | minimum-image 结果不回归 |
| NOPBC | distance/angle/dihedral | 直接位移结果与参考几何一致 |
| NOPBC | CV print | 输出有限值而非 `****` |
| NOPBC | steer | 力差与解析梯度一致 |
| NOPBC | CV restraint | 力、势能与参考公式一致 |
| NOPBC | metadynamics | hill/bias/force 正常更新 |
| NOPBC | scaled-position | 初始化阶段明确拒绝 |
| NOPBC | box-length | 初始化阶段明确拒绝 |

每个几何测试至少包含一个超过虚拟盒半长的坐标差，避免“大盒下碰巧相同”的假阳性。

### 7.2 SITS

| Boundary | Selection | 输入来源 | 预期 |
|---|---|---|---|
| PBC | `ITS`/`ALL` | legacy/H5 | 运行通过 |
| NOPBC | `ITS`/`ALL` | legacy/H5 | 运行通过 |
| PBC | selective integer | legacy/H5 | 运行通过 |
| PBC | selective atom file/indices | legacy/H5 | 运行通过 |
| NOPBC | selective integer | legacy/H5 | 初始化拒绝 |
| NOPBC | selective atom file/indices | legacy/H5 | 初始化拒绝 |

全体系 NOPBC 至少覆盖：

- observation；
- iteration；
- production；
- fresh run；
- legacy restart；
- native H5 restart；
- Nk/log state 连续性；
- mdout、legacy 文件和 H5 dataset 输出。

### 7.3 回归边界

- CPU 与 GPU 至少各运行一个 NOPBC CV 和全体系 SITS smoke case。
- PBC CV validation 全量通过。
- 现有 SITS iteration-to-production performance/functional case 通过。
- H5 input equivalence 和 runtime restart closure 通过。
- 不以“成功退出”代替数值断言。

## 8. 风险与非目标

### 8.1 本次非目标

- NOPBC 多进程支持。
- NOPBC NPT。
- Selective SITS 的 NOPBC contribution backend。
- 删除 NOPBC 的 900 Å 虚拟盒兼容层。
- 全面重构所有 bonded/restraint 模块的边界策略。
- 改变 SITS 的 Nk、feedback bias、AMD 或 GaMD 数学公式。

### 8.2 主要风险

1. **数值语义风险**：仅让 CV 成功执行而不修改 displacement 会产生更隐蔽的错误结果。
2. **调用顺序风险**：CV 必须在正确坐标刷新后执行，偏置后必须回传虚原子力。
3. **并行回归风险**：拆调度时不能破坏 PP/PM 的全局坐标与力通信。
4. **H5 生命周期风险**：不能只修 legacy guard；typed H5、compatibility config、restart 都要经过同一解析结果。
5. **输出回归风险**：全体系 SITS 在 NOPBC 下运行成功但不输出 Nk/bias，仍不能视为完整支持。
6. **假阳性测试风险**：小体系位于 900 Å 盒中心时，周期和非周期 displacement 可能恰好相同。

## 9. 完成定义

只有同时满足以下条件，才能宣称本计划完成：

- CV 执行不再由 PME/PM 是否初始化决定；
- CV 的 PBC/NOPBC 几何语义由显式 policy 决定；
- scaled-position/box-length 在 NOPBC 下不会静默使用虚拟盒；
- NOPBC + 全体系 SITS 的 observation/iteration/production 有数值测试；
- NOPBC + selective SITS 对 legacy 和 H5 输入都稳定拒绝；
- H5 restart 和输出连续性经过验证；
- PBC CV、PBC SITS 和 PP/PM 路径无回归；
- 用户文档准确描述最终支持矩阵。

## 10. 证据索引

- NOPBC 配置、box 构造与限制：`SPONGE/MD_core/pbc.hpp`
- MD 初始化顺序：`SPONGE/MD_core/MD_core.cpp`
- PBC/NOPBC 模块初始化与主循环：`SPONGE/main.cpp`
- NOPBC LJ：`SPONGE/NO_PBC/Lennard_Jones_force_No_PBC.cpp`
- NOPBC Coulomb：`SPONGE/NO_PBC/Coulomb_Force_No_PBC.cpp`
- PM/PME 守卫和进程状态：`SPONGE/PM_force/PM_force.cpp`
- CV 周期位移：`SPONGE/collective_variable/simple_cv.cpp`
- minimum-image primitive：`SPONGE/third_party/jit/jit_matrix.h`
- SITS 核心与 selection：`SPONGE/SITS/SITS.h`、`SPONGE/SITS/SITS.cpp`
- H5 SITS 配置：`SPONGE/SITS/sits_h5_input.hpp`
- NOPBC trajectory/box 输出：`SPONGE/MD_core/output.hpp`、`SPONGE/main.cpp`
- 现有输入文档：`docs/input-reference/core.md`、`collective-variables.md`、`enhanced-sampling.md`

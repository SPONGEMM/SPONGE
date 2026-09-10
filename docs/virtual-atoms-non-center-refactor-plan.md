# Virtual Atoms 非质心部分修复计划

## 1. 文档状态与范围

- 目标分支：`fix/nopbc-cv-sits`
- 当前基线：`b1017ae571b8d7abb4dbd1b3c0907f398928141e`
- 文档性质：实施计划，不表示相关问题已经修复或通过验收
- 主要源码：`SPONGE/virtual_atoms/virtual_atoms.cpp`、`SPONGE/virtual_atoms/virtual_atoms.h`
- 本计划覆盖：force-field virtual atom Type 0、Type 1、Type 2、Type 3，依赖层级、力回传、输入校验、局部化、MPI/update-group 和测试
- 本计划明确不覆盖：Type 4、`center`、`center_of_mass`、周期质心展开、质心权重语义。上述内容待单独讨论并形成后续方案

本文把“虚原子不参与积分”和“虚原子不需要力”区分开：虚原子不保存独立动力学自由度，
但作用在虚原子上的相互作用力必须按坐标映射的雅可比回传给来源原子，随后将虚原子力清零。

## 2. 当前模型

Type 0–3 都由真实原子或更低层虚原子构造：

| 类型 | 坐标定义 | 当前边界依赖 |
|---|---|---|
| Type 0 | `x_v=x_1, y_v=y_1, z_v=2h-z_1`，固定平面镜像 | 不使用边界条件 |
| Type 1 | `r_v=r_1+a(r_2-r_1)` | `Get_Displacement(..., boundary)` |
| Type 2 | `r_v=r_1+a(r_2-r_1)+b(r_3-r_1)` | `Get_Displacement(..., boundary)` |
| Type 3 | `r_v=r_1+d u/|u|`，`u=(r_2-r_1)+k(r_3-r_2)` | `Get_Displacement(..., boundary)` |

坐标按依赖层级正向刷新，力按相反层级逆向回传。单进程初始化阶段先使用全局记录，
进入 domain decomposition 后由 `Get_Local` 建立局部索引。Type 0–3 在 PP 进程上计算，
不绑定 CV owner 或 PM/PME 进程。

本次 `BoundaryPolicy` 重构已经使 Type 1–3 的坐标几何能够区分：

- PBC：使用当前 `cell/rcell` 的 minimum image；
- Open/NOPBC：使用直接笛卡尔位移。

下面的问题独立于上述边界接口，属于 virtual-atom 自身的历史正确性和生命周期缺陷。

## 3. 已确认问题

### 3.1 Type 1 力回传系数相反

当前坐标为：

```text
r_v = r_1 + a (r_2 - r_1)
    = (1-a) r_1 + a r_2
```

因此应有：

```text
F_1 += (1-a) F_v
F_2 += a F_v
F_v  = 0
```

当前实现却把 `a F_v` 加到 `atom_1`，把 `(1-a) F_v` 加到 `atom_2`。
四原子 PBC 运行检查中，`a=0.25` 得到的两个来源原子受力比为 `1:3`，正确结果应为 `3:1`。

影响：只要 Type 1 虚原子承受非零相互作用力，真实原子上的力矩和动力学都会错误；
虚原子最终力被正确清零并不能抵消该问题。

### 3.2 Type 2 非原子写竞争检测失效

初始化阶段使用 `v2_info.local_numbers` 检查同层 Type 2 是否共享来源原子，
但 `local_numbers` 要到 `Get_Local` 后才会填充，初始化时恒为 0。因此 `need_atomic`
实际上不会被置为 `true`，运行时总会选择 `v2_Force_Redistribute_No_Atomic`。

当同层多个 Type 2 虚原子共享来源原子时：

- CPU OpenMP 线程会并发执行非原子 `+=`；
- CUDA/HIP 线程会并发写入同一来源原子的力；
- 结果可能丢失部分力，并随线程调度产生非确定性。

### 3.3 层级推导依赖输入顺序

当前层级只按输入记录扫描一次。如果子虚原子先于父虚原子出现，父原子的层级仍为 0，
子原子会被错误放入过低层。后续同层并行刷新可能读取尚未更新的虚原子坐标。

当前还没有检测：

- 自依赖；
- 虚原子依赖环；
- 同一 `virtual_atom` 的重复定义；
- 记录顺序不是拓扑序；
- 来源原子和目标原子的负数或越界索引（legacy 路径）；
- 一个普通原子被意外覆盖为虚原子。

### 3.4 多层虚原子的 MPI/update-group 信息不完整

`update_ug_connectivity` 只遍历 `virtual_layer_info[0]`，即只完整连接第一层虚原子和来源原子。
`Get_Local` 则只判断虚原子是否在当前 rank，随后直接将所有来源索引转换成 `atom_local_id`，
没有验证来源原子是否也在本地。

对于单层 TIP4P 一类常见模型，第一层 connectivity 通常可以把虚原子和来源原子放入同一 update group。
对于嵌套虚原子，父子可能被 domain decomposition 分开，形成 `atom_local_id == -1`、陈旧坐标或越界访问。

### 3.5 Type 3 退化构型没有保护

Type 3 使用：

```text
u = (r_2-r_1) + k(r_3-r_2)
r_v = r_1 + d u/|u|
```

当 `|u|` 接近 0 时，坐标刷新中的归一化和力回传都会除零。`d=0` 时力回传还会通过
`rv1 * rv1` 再次产生零除。目前既没有初始化期参数检查，也没有运行期退化几何检查。

### 3.6 Type 0 参数契约与实现不一致

头文件注释定义 `z_v=2h-z_1`，字段名为 `h_double`。但初始化保存
`h_double=2*parameter`，坐标核又执行 `z_v=2*h_double-z_1`，实际结果是
`z_v=4*parameter-z_1`。

现有 bundled-I/O oracle 也复现了双重乘 2，因此旧测试会通过。这个问题不能在未确认历史文件格式前
直接修改，否则可能破坏旧输入。需要先确定 legacy 参数究竟表示平面高度 `h`、`h/2`，还是已有其他约定。

### 3.7 输入生命周期和资源管理缺少统一约束

legacy loader 主要校验字段数量，不统一校验索引、有限值和拓扑关系；H5 topology reader 的校验更完整，
导致两种输入路径在到达 `VIRTUAL_INFORMATION::Initial` 时不具备相同前置条件。

此外，`VIRTUAL_INFORMATION` 使用多组裸 host/device 指针，没有显式清理或重新初始化语义。
当前主流程通常只初始化一次，但测试、库调用或未来重建路径可能造成重复分配和残留状态。

## 4. 目标设计

### 4.1 建立统一的非质心虚原子中间表示

在分配任何 host/device 数组前，将 legacy、H5/Xponge 输入统一转换为内部定义：

```cpp
struct VirtualAtomDefinition
{
    int type;
    int target;
    std::vector<int> sources;
    std::vector<float> parameters;
};
```

统一校验完成后，再生成 Type 0–3 的紧凑运行结构。输入来源不再决定校验强弱。

### 4.2 用依赖图计算层级

处理步骤：

1. 建立 `target -> definition` 唯一映射；
2. 验证 target/source 索引范围和 arity；
3. 对来源中的虚原子建立有向依赖边；
4. 使用 Kahn 或 DFS 拓扑排序；
5. 明确报告自依赖、环和重复目标；
6. 根据最长依赖路径计算 level；
7. 按 `(level, type)` 生成运行数组。

输入文件顺序不得影响计算结果。错误信息应包含虚原子 target、type 和造成问题的 source。

### 4.3 坐标和力公式成对维护

每一种虚原子必须把坐标定义和力雅可比视为同一功能：

- Type 0：确认参数契约后固定单一镜像公式；
- Type 1：修正为 `(1-a)` 和 `a`；
- Type 2：保持 `(1-a-b), a, b`；
- Type 3：保持投影雅可比，同时增加退化保护；
- 所有类型在回传后清零目标虚原子力；
- 每一种类型都用有限差分验证 `F_i = -dE/dr_i`，不能只检查总力或虚原子力是否为 0。

### 4.4 正确性优先处理 Type 2 并发

第一阶段删除 `v2_Force_Redistribute_No_Atomic` 快速路径，Type 2 始终使用原子累加，先建立正确基线。

只有在性能数据证明该原子操作构成瓶颈后，才恢复安全优化。可选优化必须根据全局定义记录，
而不是尚未构建的局部表判断同层来源是否互斥；CPU 和 GPU 需要分别验证无数据竞争。

### 4.5 明确退化几何策略

Type 3 建议同时采用：

- 初始化期拒绝非有限参数和 `abs(d) <= epsilon`；
- 运行期检查 `u·u <= epsilon²`；
- 发生退化时明确终止并报告 target/source，而不是继续产生 NaN。

如果未来需要容忍瞬时退化，应另外定义连续化公式，不能简单将归一化分母截断后宣称物理等价。

### 4.6 MPI 局部化必须验证完整依赖闭包

`update_ug_connectivity` 应遍历所有层、所有 Type 0–3，并将每个虚原子与全部直接来源加入同一连通分量。
由传递闭包保证嵌套虚原子的整个依赖链属于同一 update group。

`Get_Local` 仍需防御性检查：

- target 在本地时，所有 source 的 `atom_local_id` 必须非负；
- 条件不满足时初始化或 domain refresh 立即报错；
- 不允许将 `-1` 写入局部虚原子记录；
- `local_atom_numbers` 参数应真正用于范围检查，或者从接口删除。

在该逻辑通过双 rank 验证前，应明确把“MPI + 多层虚原子”标记为未正式支持，而不是静默运行。

### 4.7 生命周期和资源所有权

将每层裸指针的分配、释放和状态复位集中到明确接口：

```cpp
void Reset();
void Build_From_Definitions(...);
```

要求：

- `Initial` 开始前状态为空；
- host/device 分配失败时不留下半初始化对象；
- `is_initialized` 只在所有层和局部辅助数组成功创建后设置；
- 如不准备支持重复初始化，则在第二次调用时明确报错；
- `need_atomic`、`local_state_ready`、`max_level` 和 layer vector 不得继承旧值。

## 5. 精确文件修改方案

### 5.1 `SPONGE/virtual_atoms/virtual_atoms.cpp`

- 修正 `v1_Force_Redistribute` 的两个权重；
- 删除或暂时停用 `v2_Force_Redistribute_No_Atomic`；
- 为 Type 3 坐标刷新和力回传加入统一退化检查；
- 将当前三遍但顺序相关的初始化改为“统一定义校验 → 拓扑排序 → 分层分配 → 数据上传”；
- 对 legacy 和 Xponge/H5 来源使用相同校验入口；
- `Get_Local` 验证 target/source 局部索引；
- `update_ug_connectivity` 遍历所有层，不再只读取第 0 层；
- 明确 Type 0 参数转换只发生一次；最终公式取决于第 7 节的兼容性决定；
- 保持 `Coordinate_Refresh` 正序逐层、`Force_Redistribute` 逆序逐层的生命周期；
- 不在本文件中修改 Type 4/质心逻辑，避免与后续质心设计混合。

### 5.2 `SPONGE/virtual_atoms/virtual_atoms.h`

- 修正文档公式，使参数名、存储值和运行公式一致；
- 将拼写错误的内部 `*_INFROMATION` 逐步更名为 `*_INFORMATION`；
- 增加统一 definition/validation/build 辅助类型或声明；
- 增加 Reset/析构或明确的一次性初始化约束；
- 如保留 Type 2 优化，使用每层冲突信息而不是全局模糊布尔值；
- Type 4 结构保持不动，等待独立方案。

### 5.3 `SPONGE/xponge/load/native/virtual_atoms.hpp`

- loader 继续负责解析文本，但不再把“成功读出字段”当成“定义有效”；
- 保留行号，为后续统一校验错误提供输入位置；
- 拒绝额外截断字段、非有限参数和无效整数；
- 索引、重复目标和依赖图校验交给统一验证层，避免 legacy/H5 重复实现不同规则。

### 5.4 `SPONGE/utils/h5md/topology_native_h5_reader.hpp`

- 保留现有 arity、offset 和索引范围检查；
- 补充非有限参数检查；
- 将重复目标、环和普通原子覆盖规则与 runtime 统一；
- H5 reader 可以提前失败，但 runtime 统一验证仍必须执行，不能依赖特定输入入口。

### 5.5 `SPONGE/main.cpp` 与 domain/update-group 路径

- 保持 Type 0–3 在 PP rank 上进行坐标刷新和力回传；
- 初始化顺序仍为 virtual atom 建图后，再建立 update group 和 domain decomposition；
- 必要时给 `VIRTUAL_INFORMATION::Get_Local` 传入 rank/controller 信息，以输出可定位的局部依赖错误；
- 不改变 CV owner/PME 调度；该部分属于 Type 4/质心后续方案。

### 5.6 测试与 CI

- 新增 `tests/virtual_atoms/`：纯几何、雅可比、图校验和非法输入单元测试；
- 更新 `tests/CMakeLists.txt` 纳入 virtual-atoms 测试；
- 新增 `benchmarks/validation/virtual_atoms/tests/`：CPU/CUDA 和 MPI 运行级测试；
- 更新 `pixi.toml`，增加 `vali-virtual-atoms` 和 `vali-virtual-atoms-mpi`；
- 更新 `.github/workflows/benchmark.yml`，在 CPU、CUDA、CPU-MPI 中运行相应测试；
- 更新 bundled-I/O virtual-atom oracle，使其验证精确来源原子受力，而不只是检查“真实原子力非零”；
- Type 0 oracle 在参数契约决定后再更新，并明确记录兼容行为。

## 6. 测试矩阵

### 6.1 每种类型的基本测试

| 测试 | Type 0 | Type 1 | Type 2 | Type 3 |
|---|---:|---:|---:|---:|
| 坐标解析解 | 必须 | 必须 | 必须 | 必须 |
| 虚原子最终力为 0 | 必须 | 必须 | 必须 | 必须 |
| 来源原子精确力系数 | 必须 | 必须 | 必须 | 必须 |
| 有限差分雅可比 | 必须 | 必须 | 必须 | 必须 |
| CPU/CUDA 一致性 | 必须 | 必须 | 必须 | 必须 |
| PBC 跨盒来源坐标 | 不适用/单独定义 | 必须 | 必须 | 必须 |
| NOPBC 大位移 | 不适用/绝对平面 | 必须 | 必须 | 必须 |

### 6.2 Type 2 并发测试

- 两个同层 Type 2 共享 `from_1`；
- 分别共享 `from_2`、`from_3` 和交叉来源；
- CPU 使用多个 OpenMP 线程重复运行；
- CUDA/HIP 重复运行并检查确定性；
- 与串行参考力逐分量比较；
- 检查来源总力和虚原子力清零。

### 6.3 Type 3 退化测试

- 正常非共线构型；
- `u` 很小但高于容差；
- `u=0`；
- `d=0`；
- 非有限参数；
- PBC minimum-image 后恰好退化；
- NOPBC 直接位移下不退化的对照。

### 6.4 图和输入测试

- 记录已按拓扑顺序；
- 父记录晚于子记录，但结果保持相同；
- 三层以上嵌套；
- 自依赖、两节点环、长环；
- 重复 target；
- 负数、等于 atom count、超过 atom count 的 source/target；
- 错误 arity、缺少参数、NaN/Inf 参数；
- legacy 与 native H5 对同一错误给出等价失败；
- 输入顺序随机打乱后坐标和力保持一致。

### 6.5 MPI 测试

- 1 rank 与 2 rank 的单层虚原子坐标/力一致；
- 1 rank 与 2 rank 的多层嵌套虚原子一致；
- 来源原子靠近 domain 边界；
- PBC 下虚原子依赖跨主盒边界；
- domain refresh/particle migration 前后结果一致；
- 人为破坏 update-group 完整性时必须明确报错，而不是继续访问 `-1` 索引。

## 7. Type 0 兼容性决定点

实施前需要单独确认：legacy `virtual_atom` Type 0 的参数是镜面高度 `h`，还是某个经过缩放的值。

验收证据至少包括：

- Xponge 当前生成器或历史生成器；
- 已有实际 Type 0 输入文件；
- 用户文档/论文/示例；
- 与其他引擎或解析公式的对照。

若参数就是 `h`，应将内部值保存为 `two_h=2*h`，核函数执行 `z_v=two_h-z_1`。
若历史格式确实定义为 `h/2`，应在文档中明确，并将字段命名改成不会重复乘 2 的物理含义。
不能继续保留“字段叫 `h_double`，核函数再次乘 2”这种隐式约定。

## 8. 分阶段实施顺序

### 阶段 A：建立失败测试

1. Type 1 精确来源力和有限差分测试；
2. Type 2 共享来源并发测试；
3. Type 3 退化失败测试；
4. 乱序依赖、环、重复和非法索引测试；
5. 双 rank 多层依赖测试。

### 阶段 B：修复局部数值正确性

1. 修正 Type 1 力系数；
2. Type 2 全部改用安全原子累加；
3. Type 3 增加退化检查；
4. 保持 PBC/NOPBC `Get_Displacement` 行为不变。

### 阶段 C：重构定义和层级

1. 引入统一 definition；
2. 统一 legacy/H5 runtime 校验；
3. 拓扑排序和循环检测；
4. 分层数组与资源生命周期重建。

### 阶段 D：修复 MPI 依赖闭包

1. 所有层加入 update-group connectivity；
2. `Get_Local` 完整性校验；
3. 单 rank/双 rank 对照；
4. domain migration 回归。

### 阶段 E：Type 0 契约落地

在第 7 节证据完成后独立修改 Type 0，避免与 Type 1–3 的确定性修复混在一起。

## 9. 验收标准

- Type 0–3 坐标解析解和有限差分力全部通过；
- Type 1 来源力权重符合 `(1-a), a`；
- Type 2 共享来源在 CPU 多线程和 GPU 上无竞争、可重复；
- Type 3 退化输入明确失败，不产生 NaN 后继续运行；
- 虚原子依赖图与输入顺序无关，环和重复定义明确失败；
- legacy/H5 对索引、arity、有限值和依赖图具有相同 runtime 约束；
- 所有层的 MPI update group 包含完整依赖闭包；
- 1 rank 与 2 rank 的坐标、能量和来源原子力在容差内一致；
- PBC 与 NOPBC 的 Type 1–3 分别使用 minimum-image 和直接笛卡尔位移；
- bundled-I/O 继续验证 legacy/H5 等价，同时新增独立科学 oracle；
- Type 4/质心代码在本任务中没有行为变化；
- CPU、CUDA、CPU-MPI 目标构建和相关旧回归通过。

## 10. 非目标与后续工作

本计划不决定以下问题：

- PBC 原子组质心应使用 anchor minimum-image、拓扑展开、连续轨迹还是其他定义；
- 跨半盒或非连通原子组的周期质心是否有唯一物理意义；
- `center` 权重是否必须归一化；
- Type 4 大原子组 GPU 归约实现；
- Type 4 在 CV owner、PM rank 和 PP rank 之间的坐标/力生命周期。

这些内容应在质心语义讨论完成后形成独立计划，随后再决定是否与本计划的实现合并到同一 PR。

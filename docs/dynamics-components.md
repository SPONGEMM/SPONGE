# 动力学能力与黏度工作流

本分支从 `lab/sidereus-ai` 的 `5e4d7a9` 整理。旧非平衡分支保留作对照；此处没有重新引入已经合入基线的 PME、dihedral、约束精度或 rerun 修复。

## 代码职责

| 位置 | 负责什么 | 不负责什么 |
| --- | --- | --- |
| `SPONGE/external_force/` | 原子力、加速度、qE、组总力/力矩；时间 schedule、空间 cosine profile | 黏度方法、动量交换 |
| `SPONGE/thermostat/velocity_bias/` | 质量加权 COM/余弦模式；热动能、热 DOF、保留速度 | 坐标推进、约束积分状态、采样文件 |
| `SPONGE/thermostat/thermal_step.hpp` | Berendsen/Bussi 的共同热速度缩放边界 | 包装整个 MD 步 |
| `SPONGE/constrain/tangent_frame.*` | 距离约束切空间、瞬时反力和联合模式投影的数学基元 | GK 工作流 |
| `SPONGE/MD_core/integrator/` | canonical q/s 的唯一 owner；受约束 Middle、RATTLE NVE 推进和重启 | 外力配置、文件写入 |
| `SPONGE/MD_core/energy_*` | 外力路径功、控温子步热量及补偿求和 | 内势能、外力施加 |
| `SPONGE/momentum_exchange/` | 原子交换、调度、实际写入动量与守恒诊断 | 剖面拟合、黏度估计 |
| `SPONGE/observables/` | 应力、速度剖面/模式与独立采样计划 | 选择 GK/Cosine/RNEMD 算法 |
| `SPONGE/io/` | 现有 Session、H5 writer、完整性日记和 restart | 解释黏度方法名 |
| `tools/viscosity/workflows/` | 将方法预设展开为显式引擎输入 | 逐步 MD |
| `tools/viscosity/analysis/` | ACF、积分窗、剖面/响应拟合、块统计、黏度与报告 | 自动认定收敛或实验精度 |
| `benchmarks/viscosity/` | LJ、水、己烷模型；独立参照/验收协议 | 通用引擎逻辑 |

原来的 `SPONGE/nonequilibrium/`、`FlowState`、`Measurement` 总控类已拆除。`scripts/analyze_*.py` 仅保留历史命令入口，数值实现在 `tools/viscosity/analysis/`。历史归档格式仍可读取。

## 简洁输入

GK 生产段只需普通 NVE 加通用应力采样；相关函数和黏度计算在运行后进行：

```toml
mode = "nve"
[sampling]
stress = true
interval = 10
output = "stress.h5"
```

周期扰动由三项独立能力组成，驱动幅度和测得流速幅度是不同的量：

```toml
[[external_force]]
type = "acceleration"
acceleration = [0.01, 0, 0]  # nm/ps^2
profile = {type="cosine", axis="z", mode=1}

[velocity_bias]
mode = "cosine"             # 联合保留三个平移分量及一个余弦模式
flow = "x"
gradient = "z"

[sampling]
profile = "cosine"
flow = "x"
gradient = "z"
bins = 20
interval = 10
output = "profile.h5"
```

只需保留整体平移时使用 `[velocity_bias] mode="com"`。不配置该表（或 `mode="none"`）保持既有控温路径。

原子动量交换与剖面采样也各自配置：

```toml
[momentum_exchange]
interval = 100
bins = 20
flow = "x"
gradient = "z"

[sampling]
profile = "slab"
bins = 20
flow = "x"
gradient = "z"
output = "profile.h5"
```

普通外力、外电场和组力矩沿用相同入口：

```toml
[[external_force]]
type = "electric_field"
field = [0, 0, 0.1]           # V/nm；内部静电仍由原有 PME 计算
schedule = {type="ramp", start=0, duration=10}  # ps
```

`force` 单位 pN/atom；`group_load` 的 `force` 是组总力 pN，`torque` 是绕质量中心的 pN nm。组载荷需显式零基原子编号 `atoms=[...]`，目前使用连续的非周期坐标。不可表示的力矩（例如共线组轴向力矩）明确报错。

引擎不再接收 `[viscosity] method=...` 或 `[[coupling]]`。历史预设可在引擎外展开，且输出文件必须与输入不同：

```bash
python -m tools.viscosity.workflows.config workflow.toml mdin.spg.toml
```

这是配置展开，不负责自动准备平衡态、判定稳态、扫描扰动强度或提交长程任务。已有 benchmark 负责这些模型相关编排。

## 热浴子步与时间层

令完整速度为 `v = b + c`，`b` 是质量度量下投影得到的保留模式，`c` 是热速度。

- Middle：完整速度 kick/半 drift → 在中点上扣除 b → 对 c 施加 OU → 投影噪声并恢复 b → 完整速度半 drift。
- Andersen：完整 kick/半 drift → 在碰撞子步重采样热子空间 → 恢复 b → 完整半 drift。非碰撞步热量为零，RNG 计数只在碰撞时递增。
- Berendsen/Bussi：完整 kick/drift 后，使用该几何处的热动能与热 DOF 计算缩放因子，仅缩放 c；Bussi 随机因子使用相同 DOF。
- NHC：在原有摩擦求值点估计 b、热动能与 DOF；链反馈使用热动能，摩擦项为 `-xi*(v-b)`，坐标推进仍用完整速度。链状态仍由 NHC 保存。

无约束 COM 模式热 DOF 为 `3N-3`，额外独立余弦模式为 `3N-4`；秩亏模式报错。受约束 Middle 使用联合切空间投影与实际秩，不把几次顺序投影当作等价算法。随机噪声投影到同一热子空间，因此不会把热浴噪声注入被保留模式。

外力路径功是梯形力与实际位移的内积；热量是热浴子步前后的动能差。NHC 单独扣除 force kick 后再测摩擦引起的动能变化。受约束 canonical 路径保留原有连续 double 坐标和漂移诊断，不将数值漂移冒充热量。

`sampling.velocity_time_layer` 默认 `force_centered`：对原有 staggered 速度只读地补半 kick，并按需做约束切空间观测。可选 `stored` 直接观察引擎存储层。NHC 当前只开放 `stored`，因为其摩擦推进不能宣称等价于力独立的半 kick。约束应力要求 `force_centered`。

采样阶段固定为 force 后、exchange 与 integrate 前。普通输出与采样拥有各自 cadence；应力需求固定每步请求相同力/能量/维里分支，改变采样间隔不改变积分轨迹。应力采用对称内部压力约定、去 COM 动能及瞬时约束反力；外部 body-force virial 不加入内部应力。热温度不会替换动量通量的动能项。

## IO 与重启

新增采样复用现有 `ObservableH5Writer` 和 OutputSession 生命周期，文件记录 `observable_contract=sponge.observables.physical_state.v1`、velocity time layer、sampling phase、单位和有效配置。剖面暂以每 bin 独立标量通道表示；尚未实现原生多维 profile dataset。结束状态与完整性日记必须匹配，分析拒绝未 finalized 数据。

状态分别归 integrator、thermostat、energy 和 exchange owner；只在既有 restart payload 中组合，不增加第二份时钟或状态副本。bias 的当前拟合无历史，不保存冗余速度；保存其配置身份以拒绝不兼容的动态继续。改 bins/采样间隔/输出路径可开始新的采样段；改变驱动、热浴、bias、有效质量或约束身份需 structural load 开始新动力学段。

保留最新基线 HMR 有效质量签名、原始质量拓扑选择规则、rerun 当前帧时序和数值故障检测。旧分支的动态 checkpoint 与本次组件身份不兼容，不能静默重置账本；旧轨迹和分析归档仍可读取。

## 第一版支持边界

| 组合 | 当前边界 |
| --- | --- |
| 无约束 COM/cosine bias | 五种控温器，固定盒 NVT；NVE 可只作观测 |
| 约束与 bias | cosine + Middle + SETTLE/SHAKE；canonical host 路径 |
| 约束应力 | SETTLE/SHAKE/RATTLE，正交固定盒；瞬时反力观察者 |
| RATTLE | 独立距离约束、canonical double、固定周期盒 NVE |
| 原子交换 | 无约束、无其他驱动的固定正交周期盒 NVE；偶数 bins |
| 压强耦合 | 新组件尚不开放 flow-aware barostat；普通既有 NPT 路径保留 |

新组件限定单 rank、固定拓扑/电荷、正质量，无虚拟原子、SITS、速度裁剪、墙、插件或其他未计入观测的 bias。带电应力/剖面要求中性固定盒，完整 direct/reciprocal PME 每步更新。不支持的组合明确拒绝，避免输出语义不清的黏度数据。

通用采样可用于 NVT，但 GK 分析工作流仍要求无驱动的平衡 NVE 段。Cosine 分析从声明的恒定全原子驱动读取加速度，检查轴/波数，不从测得响应反推外场。RNEMD 分析要求实际交换账本与 slab profile。

GK 分析保留旧分析器的 `--components shear3|isotropic5`；五分量是各向同性假设下的相关观测，不作为五条独立轨迹进行重采样。

短程回归检查的是解析力/力矩、热子空间不变量、噪声协方差、应力/约束诊断、交换守恒、采样 cadence、完整性与连续/分段重启。它们不等于长程黏度收敛、实验复现或所有 GPU/MPI 组合验收。常规 CI 未增加长程 MD。

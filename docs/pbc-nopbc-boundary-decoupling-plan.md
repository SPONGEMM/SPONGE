# PBC/NOPBC 边界接口与模块迁移

## 状态与范围

本文替代早期多结构和快照工厂方案，说明四个 PR 的接口边界。
PR1 是接口重构；PR2 是 CV/SITS 功能修复；PR3 是虚原子数学修复；PR4 是晶胞几何及 MC 回滚修复。
支持目标不等于所有配置都已通过科学验收。

不增加 NOPBC 近邻表、多体势 NOPBC、MPI NOPBC、选择性 SITS NOPBC；
不修改整分子成像、周期性聚合物识别或质心缩放，不增加插件 ABI 或 H5 schema。

## 单一数据结构

```cpp
enum class BoundaryPolicy : std::uint8_t { Open = 0, Periodic = 1 };
struct Boundary
{
    BoundaryPolicy policy = BoundaryPolicy::Open;
    LTMatrix3 cell;
    LTMatrix3 rcell;
};
```

MD_INFORMATION::periodic_box_condition_information::boundary 是运行时 policy/cell/rcell 的唯一持有者。
不再另外保存同名矩阵、独立可变的盒子副本或包装工厂。
原有 pbc 布尔配置入口保留，初始化时确定 policy；控压不改变 policy。
cell0 是网格重建的历史比较基准，不是另一份当前盒子状态。

sys.box_length/box_angle 保留为输入、输出和控压接口所需的几何元数据。
初始化先读取坐标/重启元数据，再构造 boundary；Update_Box 更新矩阵并反写元数据。
Rerun 的新帧元数据通过原有 Main_Box_Change 路径应用。不重写 I/O 协议，
也不把盒长数值当作物理边界模式。

主计算入口用 const Boundary& 引用持有者；盒子更新后，后续调用自然读取新值。
CPU/CUDA/HIP kernel 按值接收调用时的参数，不能把主机引用作为设备指针；
模块不跨步缓存边界副本。QC 的 Scale_Boundary 仅生成长度单位换算的局部值，不反写模拟盒子。

## 位移、映射和网格索引

```cpp
Get_Displacement(a, b, boundary);
Get_Displacement<BoundaryPolicy::Periodic>(a, b, boundary);
Get_Displacement<BoundaryPolicy::Open>(a, b, boundary);
Wrap_Coordinate(coordinate, boundary);
Get_Mesh_Index_Displacement(index_a, index_b, scaler);
```

通用位移在 Open 下返回 a-b，在 Periodic 下保持原 fractional rounding 公式。
固定策略模板不读取运行时 policy，仅用于已经确认支持该模式的计算路径。
这不是更换三斜角最小镜像算法，也不保证任意倾斜晶胞中的最短欧氏镜像。

Wrap_Coordinate 在 Open 下返回原坐标，在 Periodic 下映射回主盒。
整数网格位移仍是独立数学操作，不接收物理边界策略。
自动微分中带梯度的盒长参数属于导数输入，不是另一个运行时盒子持有者。

Host/JIT 使用同一 Boundary 定义和位移公式；listed/pairwise JIT ABI 直接传 Boundary。
Listed force 保留原正交盒长度梯度算法，本次不扩展其三斜角应力语义。

## 盒子更新顺序

保留 Main_Box_Change 的生命周期：

1. 压力控压、MC trial/reject 或 rerun 提交形变。
2. pbc.Update_Box 更新唯一 boundary 的矩阵以及盒长/盒角。
3. 按原有选项缩放坐标、速度。
4. 小变化更新 DD/PM 派生状态，大变化重建近邻网格、PM 和 DD。
5. 后续力、约束、虚原子和 CV 调用读取当前 boundary。

不增加 revision、Prepare/Commit 状态机或跨模块缓存。
MPI 按原有同步调度在各 rank 更新盒子，不引入跨进程 boundary 指针。

## 模块与文件覆盖

| 模块/文件 | 处理方式 |
| --- | --- |
| utils/boundary.h、vector.hpp、sad.hpp | 单一 Boundary；位移、映射、网格索引分离 |
| third_party/jit/jit_boundary.h、jit_sadvector.h 等 | Host/JIT 对齐；保留自动微分公式 |
| MD_core/pbc.h、pbc.hpp、main.cpp | 唯一持有者；初始化、控压、rerun、kernel 传参 |
| MD_core/mol.hpp、sys.hpp、output.hpp、rerun.hpp | 读取同一盒子；保留成像、质心和 I/O 语义 |
| bond、angle/Urey_Bradley、dihedral/improper、cmap、nb14 | 通用边界位移；不改能量/力公式 |
| constrain/shake、settle | 坐标约束及速度投影，保留上游 guard 和精度回退 |
| restrain、virtual_atoms | 通用边界参数；虚原子数学及依赖顺序修复在 PR3 |
| collective_variable、bias/steer、restrain_cv、sinkmeta | PR1 迁移参数并保留行为；PR2 修复 Open 策略和调度 |
| Lennard_Jones_force、solvent_LJ、LJ_soft_core | Boundary 贯穿调用；固定 Periodic 位移 |
| PM_force | Boundary 替代成对矩阵参数；网格/FFT 仍读取对应矩阵 |
| neighbor_list/full_neighbor_list | 构建、刷新、溢出重建统一参数；不加 Open 后端 |
| Domain_decomposition | 读取唯一矩阵；分数坐标、域分箱等数学不变 |
| SITS | 选择性周期分量传 Boundary；PR2 允许全体系 NOPBC |
| custom_force/listed_forces、pairwise_force | JIT ABI 同步迁移；pairwise 仍依赖周期近邻表 |
| manybody/SW、EDIP、EAM、Tersoff、reaxff 子模块 | 固定 Periodic 位移，Open 明确拒绝 |
| quantum_chemistry/scf/pre_scf、gradient/grad_nuclear | 实际策略及局部单位换算 |
| plugin | API v2 无边界约定，NOPBC 拒绝；不升级 ABI |
| NO_PBC/LJ、Coulomb、generalized_Born | 保留直接笛卡尔计算和专用后端 |
| barostat | 仍仅 PBC；几何和 MC 逆缩放修复在 PR4 |
| thermostat、nve、min、wall | 不需要新增边界对象，积分及绝对坐标外场不变 |

## 四个 PR 完成后的兼容性范围

| 功能 | PBC | NOPBC |
| --- | --- | --- |
| 通用成键力、约束、位置限制 | 支持 | 支持 |
| 虚原子 Type 0–3 | 支持 | 支持 |
| Type 4/加权中心 | 保留原定义 | 保留直接加权定义 |
| 距离、位移、角、二面角、位置/RMSD CV | 支持 | 支持各自定义 |
| scaled-position、box-length CV | 支持 | 拒绝 |
| steer、restrain CV、metadynamics | 随所用 CV | 随所用 CV |
| 全体系 ITS/ALL | 支持 | 支持，具体模式以专项测试为准 |
| 选择性 SITS | 支持 | 拒绝 |
| 周期 LJ/PME、周期近邻表、多体势 | 支持对应后端 | 不支持 |
| NOPBC LJ/Coulomb/GB | 对应 PBC 后端或不适用 | 保留专用实现 |
| listed custom force | 保留原盒型语义 | 支持 |
| pairwise custom force、插件 API v2 | 保留原支持 | 拒绝 |
| 控压、MPI 区域分解 | 保留原支持，MPI 非正交 DD 不支持 | 不支持 |

## 验证要求

- 四个 PR 独立编译，各自保留明确职责及回归。
- CPU/CUDA 动态/固定策略、坐标映射、盒子更新后再次调用的数值检查。
- 源码合约禁止旧类型/包装工厂，校验 Host/JIT 公共边界实现一致。
- 真正编译执行 listed PBC/Open 和 pairwise PBC JIT，与解析力对照。
- 保留约束探针、CV、十二肽、虚原子、H5 输入与控压回归。
- 最终栈检查 CPU、CUDA、CPU-MPI；np=3 控压验证区域分解，np=2 仅验证 CV owner。
- 正确性和性能测量分开报告，不能凭内联或测试通过宣称零性能回退。

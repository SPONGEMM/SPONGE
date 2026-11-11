# SPC/E 水余弦流正确性验证

本轮验证对象为 `feature/dynamics-components` 的 CPU 引擎（架构提交 `62a25d5`），参考为锁定的 LAMMPS 29 Aug 2024 CPU 程序。参考包通过项目内 `benchmarks/viscosity/reference-env/pixi.lock` 恢复；未修改系统库或驱动。当前 GPU 驱动不可用，未做 GPU 验收。

## 运行分层

1. **同帧模型与静电门禁**：逐原子力、势能、六应力分量分别比较 LJ、Coulomb、full，尾项单独计量。先后检查 PME 64/128 与 128/256 网格；64 最大力误差超过原有门槛，保留失败；128/256 在同一门槛下通过。
2. **受约束短程与观测定义**：三条水内距离约束；SPONGE projected Middle 与 LAMMPS native NHC+SHAKE；步长 0.125/0.25/0.5 fs，固定同样物理时长。独立 NumPy 逐水 Jacobian 投影重构物理速度、Fourier/拟合幅度与联合热温度，并检查 C++ canonical 约束残差。
3. **有界响应试验**：四组 A=+0.20,+0.10,+0.05,-0.20 nm/ps²，每引擎各 2 ps，0.25 fs，2 fs 采样；预先固定弃前 0.5 ps、0.5 ps 块长。每种幅度只有一条轨迹，且使用共同初态/随机种子；不是独立重复。
4. **正式输运验收尚待实施**：独立重复、更长轨迹、积分步长/网格/控温强度/有限波长与盒子尺寸检查、弱驱动线性响应区，以及与 GK、文献和实验的比较。本轮不自动外推零剪切黏度，不声明 ±5% 精度。

## 模型与初态

当前试验使用 216 水、648 原子，盒长 18.6382618 Å，密度约 998 kg/m³，8 Å cutoff。坐标复制自已归档的 10 ps NVT 末态；重新生成 300 K 切空间 Maxwell 速度，扣除 COM 和投影后的余弦模式。该结构及一个新速度抽样不能证明严格平衡。

它与 LAMMPS 示例 `in.cos.1000SPCE` 的 1000 水、10 Å cutoff、NPT 设置不同。本轮参考借用同一余弦驱动/测量方法，采用共同固定盒和有效 Hamiltonian 参数。有限波长约 1.86 nm，测得的是此有限体系的响应；不能将其直接等同于宏观实验黏度。

当前源位置：

- 模型：`notes/evidence/spce-cosine-replicas-periodic-v2-20261028/inputs/model`
- 结构：`notes/evidence/numerical-water-audit-20261103/dynamics/equil/final_coordinate.txt`
- 新证据：`notes/evidence/spce-cosine-validation-20261111`

旧档案不修改。新模型、速度、每次输入、引擎、执行源与分析源分别保留哈希。

SPONGE 输入速度处于交错时间层。参考引擎的初始物理速度用
`Pc(s + h F0 / 2m)` 构造；`F0` 必须来自相同 canonical 积分器和影子坐标路径。
新版入口先运行一个单步探针取第零帧力，再生成两引擎输入。独立审计重新使用
正式轨迹第零帧力核对初始速度，逐原子最大差门槛为 `5e-7` native。
早期普通力探针版本的短跑保留，实际交接误差另行量化，不冒称逐位相同。

## 定义与可比性

驱动为 `a_x(z)=A cos(2πz/Lz)`。固定盒和线性稳态假设下，`η=ρ A/(k² <V>)`；使用 g/cm³、nm/ps²、nm⁻¹、nm/ps 时结果数值单位为 mPa·s。

LAMMPS 的 `c_cos[7]` 对应原始质量 Fourier 幅度 `2<vx cos(kz)>_m`。SPONGE 同时给出原始、去 COM Fourier 和最小二乘拟合三种估计；它们在有限体系中不完全相同，全部报告，不按接近文献程度挑选。

热温度通过独立的约束切空间与 COM/余弦联合投影重算。LAMMPS native `viscosity/cos` 的温度定义与该投影不同，native NHC+SHAKE 也不是 SPONGE projected Middle 的同一有限步积分器。因此：

- 同帧力/单位/观测定义可作数值门禁；
- 长轨迹的统计流速响应可以对比；
- 不要求两种随机/确定性积分器逐帧相同；
- 同时设置 300 K 并不意味着实际采样温度已一致；
- LAMMPS 不是免检真值。

能量诊断比较同一力时间层的 `Δ(K+U_shifted)-W_external-Q_thermostat`，LJ 截断能量跳跃单独修正。约束 drift kinetic change 独立输出，不能冒充热量或用来掩盖积分残差。当前 LAMMPS 分支未提供等价的精确路径功账本，该量标记 unavailable，不以零替代。

## 实际入口与资源

`benchmarks/viscosity/water_cosine_validation.py` 提供 `run_case` 和独立重分析入口 `audit_case`。`--audit-only` 不重跑 MD，便于保留并修正后处理错误。

`audit_case` 读取轨迹并写分析结果；重分析之前把旧摘要、状态、剖面及能量诊断
存入 `analysis-history/<旧摘要哈希>/`。同时存储本次分析源码，分别报告执行与
分析源码哈希。完成归档写入 `closure.json` 后，入口拒绝原地修改。

```bash
python benchmarks/viscosity/water_cosine_validation.py \
  --campaign /path/to/campaign --name pilot-a020 \
  --steps 8000 --dt 0.00025 --stride 8 --amplitude 0.20
```

每次引擎有 1800 秒上限。实际控制器在单个 1 CPU、2 GiB、swap=0 的任务中串行运行两引擎。常规 CI 只增加独立投影/单位换算的小型数学测试，不加入这些 MD 采样。

上述命令使用已准备的 campaign：根目录需要经过哈希核验的 `model/`、
`physical-seed.npz`、包含引擎路径及哈希的 `plan.json`，以及通过的
`same-state-refined/summary.json`。本轮准备控制器和执行源码保存在证据的
`code/` 中；冻结的归档用于复核，新增运行应复制输入到新的 campaign。

## 结果解释规则

- 先核对 H5 finalized 日记、全部 step/time、输入哈希及输出有限性，再做估计；程序退出零或文件存在不算验收。
- 固定先平均响应、再求倒数。响应区间跨零时，黏度区间无界，报告 unresolved/null；不删除反号样本。
- 2 ps 在上述设计下只有三个块。bootstrap 重采样次数不是独立样本数，所得区间仅是条件性诊断。
- 至少检查实际温度、各块流速/温度、剖面相位/非余弦残差、三个幅度估计的敏感性和反向驱动；不能只比较一个最终黏度数值。
- 此前一次门禁的后处理误以为 LAMMPS 不输出第零帧；实际输出包括第零帧。保留失败记录，仅重分析完整轨迹，未重复该次 MD。

参考：[LAMMPS water cosine example](https://github.com/lammps/lammps/blob/develop/examples/VISCOSITY/in.cos.1000SPCE)、[viscosity/cos 定义](https://docs.lammps.org/compute_viscosity_cos.html)、[SPC/E 参数](https://docs.lammps.org/Howto_spc.html)。

## 2026-11-11 首轮结果

同帧 128/256 网格及三步长短程检查通过；相关回归 8/8 通过。
四组双引擎 2 ps 短跑全部完成，但目前不能判定黏度收敛或两引擎对齐。
例如 A=+0.20 时，Fourier 黏度点估计 SPONGE 为 0.420、LAMMPS 为
1.111 mPa·s，分析窗热温度分别为 308.71 和 301.84 K；LAMMPS 的响应
区间跨零。反向驱动估计也不稳定。每组只有三个时间块，温度、稳态、
独立性与有限尺寸系统误差均未验收。

完整表格、三种估计器、原始输入/输出、失败记录、PNG/PDF 图表及哈希清单
位于项目（仓库外）的 `notes/evidence/spce-cosine-validation-20261111/`。
项目说明为 `notes/2026-11-11-water-cosine-validation.md`。

## 2026-11-12 较长重复测试

三组独立随机准备，每组 3 ps 零驱动准备后做正反向各 10 ps 双引擎测试，
共 12 条生产轨迹；0.25 fs 步长、20 fs 采样，主窗 [2,10) ps。
正反向在 replica 内配对，三个 replica 的响应均值先汇总，再取倒数。
主 Fourier 估计 SPONGE 为 1.1263、LAMMPS 为 0.6569 mPa·s，条件性组间
95% 区间分别为 [0.6981,2.9126]、[0.4139,1.5917]；平均热温度为
301.102、300.530 K。区间来自 n=3、df=2 的条件性 t 分布假设，不能据重叠证明等价。

全部数值审计和 8 项相关回归通过；时间窗敏感性仍明显，黏度尚未验收收敛。
本轮仅增加独立的 `--analysis-seed` 及物理初态哈希记录，没有修改 MD 核心算法。
完整档案为项目（仓库外）的 `notes/evidence/spce-cosine-replicas-20261112/`，
项目说明为 `notes/2026-11-12-water-cosine-replicas.md`。

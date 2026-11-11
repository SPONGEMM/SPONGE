# 正己烷约束余弦运行与初态交接

`benchmarks/viscosity/hexane_cosine.py` 接受 `hexane_rattle.py prepare` 生成的
canonical 模型与该模型在当前二进制下通过的 `hexane_model.py static --rattle`
参考检查。模型包含原始 float 输入、投影后 double 坐标及目标物理速度。
同一模型可以交给 GK 和余弦运行器；两条轨迹后续的系综与温度仍须分别核验。

```bash
python benchmarks/viscosity/hexane_cosine.py \
  --source /path/to/canonical-model --static /path/to/static-gate \
  --sponge /path/to/SPONGE --probe /path/to/test_constrained_nve \
  --output /path/to/new-run --duration 1 --dt 0.0005 \
  --acceleration 0.2 --tau 0.1 --seed 79101 \
  --native-ps 0.01 --sample-interval 1
```

驱动固定为 x 方向、z 梯度、盒内一个余弦周期，acceleration 单位 nm/ps²；
时长、步长、温控时间常数和 native 输出间隔单位 ps。浴温来自模型。
使用 `SHAKE` 距离拓扑入口、约束 flow 的 double RATTLE 分裂与
`middle_langevin` 联合切向温控。输入键没有增加。输出目录必须不存在；失败记录保留。

初始外力按实际 native 发布坐标计算并转为 float，和内力相加后才执行物理速度到
incoming staggered velocity 的半步转换。准备探针重用引擎位置投影核心，
不是独立投影参考；另用 NumPy 约束线性代数核验物理速度。实际首帧总力必须
逐值等于预计算结果，不能只验证两条输入文件看起来相同。

首帧六应力以 LAMMPS 势场压力为基础，使用 LAMMPS 势场力加已知外力，独立计算
约束反力与动能压力。外力本身的 rF 不计入内部压力，但外力引起的约束反力必须计入。
整个运行按 native 帧重建物理动能和去流动热动能；double 末态另作严格核验。
检查点要求 flow v2 的固定力计算策略、完整时钟，以及功/热/漂移诊断与末帧精确一致。

`audit.csv` 报告扣除截断常数的能量、做功、热量和独立漂移诊断。
`E(t)-E(0)-W-Q` 只作有限性检查及诊断，尚未声明能量误差的科学接受阈值；
漂移动能没有混入热量。数值门通过不等于能量收敛、稳态、弱驱动极限或黏度验收。
本运行器不从短测试自动生成黏度值，也不自动选择排除时间或积分/拟合窗口。
正式多副本计划应在启动前声明两种驱动、丢弃时间、分块、估计器和误差分析，
与 GK 比较时必须保留实际温压及力场、尺寸、步长、温控差异。

旧 flow v1 动态检查点不能接到 v2 继续同一轨迹。旧实验保留冻结二进制；
新实验可经明确的物理状态转换重新准备，不能把重新准备标成连续续跑。

## 多副本聚合

`hexane_cosine_analysis.py` 消费 `sponge.hexane.cosine-replicas.v1` 计划和完整
campaign。生产计划先声明所有初态、驱动、时长、窗口、块长、估计器和随机种子。
分析器及其全部本地依赖另行冻结，不修改生产计划：

```bash
python benchmarks/viscosity/hexane_cosine_analysis.py --campaign /path/to/campaign --freeze
python /path/to/campaign/analysis-sources/benchmarks/viscosity/hexane_cosine_analysis.py \
  --campaign /path/to/campaign --output /path/to/campaign/analysis
```

冻结和输出目录均禁止覆盖。第二条只有全部 case 完成数值审计才接受输入；检查计划、
原始文件、实际初态、驱动/温控/采样条件及冻结源身份，不消费运行中的 HDF5。
每轨迹先求 signed amplitude 均值，再转换 fluidity；平均整副本 fluidity 后取倒数。
全部窗口、估计器和驱动共用整副本 bootstrap indices，同时传播温度和压力。
三块窗口用首末各一块作等长漂移比较；四块用首末各两块。

`summary.json` 保留全部条件估计、每块诊断、空间拟合、数值审计和时间成本。
`bootstrap.npz` 保存实际 indices、所有 signed fluidity、T/P、倒数和参考比值样本。
零/负响应不会筛除；零的 IEEE 倒数在 NPZ 中为 infinity，JSON 不写非有限数。
fluidity 区间触零时黏度区间为 null；参考比值/双驱动比值只有所有 draws 正且
fluidity 区间下限正才报告区间。不会通过删除难以解释的 draws 产生有限误差条。

实验参考沿同 draw 的实际温度、固定实际密度求 Michailidou 2013 关联式，保留
响应和温度的协方差。没有 EOS/压力/相态校验，也没有参考式不确定度传播；
不同温度 GK 的有限窗估计只能条件并列，不构成严格同状态方法等价性证明。

聚合完成后，可运行 `plot_hexane_cosine.py --campaign /path/to/campaign --output /path/to/new-plots`。
绘图器验证冻结分析身份并重放已保存 bootstrap，读取原计划指向的已闭合 GK 摘要；
输出响应/参考比值、实际温压、全部48个GK与12个余弦设置三组 PNG/PDF。
跨方法图采用相同黏度纵轴；未确定点/区间显式标记，不绘制虚假的有限误差条。
运行成本注明并行和共享主机差异，不能当作达到相同精度的成本比较。

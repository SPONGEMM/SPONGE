# 正己烷多副本 Green–Kubo 分析

`benchmarks/viscosity/hexane_gk_analysis.py` 接受冻结的
`sponge.hexane.gk-replicas.v1` 采样档案。它仅在整批完成、每个副本的数值审计通过后分析；
`execution.json` 是完成记录，不能用于判断仍在运行的进程是否存活。

```bash
python benchmarks/viscosity/hexane_gk_analysis.py \
  --campaign /absolute/path/to/campaign --output /absolute/path/to/new-analysis
```

输出目录必须不存在。分析核验预声明、冻结源、实际分析依赖、每个副本的运行计划、模型及
原始文件完整清单。checkpoint 必须标记 `rattle_nve_double_v3` 和
`energy_virial_every_step_v1`。应力文件须完整封口，时间与步数一致；对角压力也须形状一致、
全部有限、单位为 Pa。相同物理模型要求不含初始坐标/速度的拓扑与参数逐项匹配。

各轨迹独立分块求相关，不拼接轨迹，不把三个应力分量当作独立副本。每种排除时间、
decimation 和逐块去均值选择都保留完整积分曲线，并在预声明的所有相关窗取值。
块末不足一块的帧不参与相关积分，实际使用帧数和时间范围保存在每个结果中。
状态诊断另使用排除时间之后的全部帧，因此其均温与积分归一化所用均温可能略有不同。

每个副本先用其实际平均温度归一化。聚合诊断定义为

\[
\eta_{\mathrm{pool}} =
\frac{\operatorname{mean}_r(T_r\eta_r)}{\operatorname{mean}_r T_r}.
\]

所有设置共用同一份整副本重采样索引，保留它们之间的配对关系；`bootstrap.npz` 保存
索引与每次估计。负积分和负区间端点保留。另报告直接平均单副本黏度作为诊断，不能
通过选择两种聚合方式之一改善与实验的符合程度。

`summary.json` 保存全部单副本状态、48种预声明设置的结果、条件95%百分位区间、源哈希
及运行耗时。`r1/` 等目录保存每种设置的相关曲线（Pa²）和积分曲线（mPa·s）。
`cost_to_precision` 保持空值；总运行时间不代表达到指定科学精度的成本。

四个速度/热浴种子共享先前结构祖先，整副本 bootstrap 的样本数只有四。
不同 NVE 副本的实际温压也可能不同，聚合不能自动解释为同一平衡状态的物性。
有限相关窗、有限样本区间、完整性检查与数值审计都不是平衡、平台收敛、方法等价或实验验收。
本工具不选择最佳窗口、不根据区间宽度提前停机、不把本批结果标为正式黏度预测。

完成分析后，用 `plot_hexane_gk.py --analysis /path/to/analysis --output /path/to/new-plots`
生成 PNG/PDF。图中同时保留全部预声明窗口与条件区间，并分别展示实际温压和耗时。
完整积分曲线固定展示最早预声明排除时间和最细采样间隔，不按结果挑选。
绘图前逐副本核对 CSV 的窗口积分与 JSON 数值；输出 provenance 保存全部输入与图像哈希。
共享主机上的构建/测试重叠需另记录，运行时间不能解释为隔离硬件性能或达到指定精度的成本。

完成冻结分析后，可运行 `compare_gk_reference.py --campaign /path/to/campaign
--analysis /path/to/analysis --output /path/to/new-reference-diagnostic`。
工具核验冻结分析源、原计划、完成副本、所有设置的顺序，并从单副本积分和温度复算
全部 bootstrap 样本及区间。它不重新执行原始轨迹审计，原始数据的验收仍由前述分析器负责。

每次重采样用同一组副本索引同时计算池化黏度和池化温度，再按固定模型密度求参考值，
得到 `eta_MD / eta_reference(T,rho)`；输出保存全部配对样本。负积分和负比值不筛除。
不同温度副本池化的含义仍有限，因此结果仅为条件诊断。参考式来自
[Michailidou et al. (2013)](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=913970)，
本步骤没有 EOS、压力或相态匹配，也没有传播参考相关式不确定度。它不把实测温度偏离
293.15 K 的结果直接除以该温度的离散实验点，更不输出实验验收结论。

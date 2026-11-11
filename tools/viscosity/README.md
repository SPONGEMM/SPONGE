# 黏度工作流与分析

引擎能力和输入示例见 [动力学组件](../../docs/dynamics-components.md)。

- `workflows/config.py`：把 GK、Cosine、RNEMD 历史预设展开为显式 `external_force`、`velocity_bias`、`sampling`、`momentum_exchange` 输入；也可直接编写这些组件。
- `analysis/green_kubo.py`：应力 ACF 与指定积分窗、块长、bootstrap；禁止拼接独立能量初态作为同一平稳序列。
- `analysis/cosine.py`：指定稳态窗内的模式响应；使用平均响应之比，检查驱动与测量的轴/波数。
- `analysis/rnemd.py`：真实累计交换动量、两条剖面分支、通量与梯度。

```bash
python -m tools.viscosity.workflows.config workflow.toml mdin.spg.toml
python -m tools.viscosity.analysis.green_kubo stress.h5 --start-ps 0 --window-ps 1 --block-ps 25 --output analysis
```

这些命令示意接口；时间窗、块长、准备时长必须针对具体体系验证。三个分析器也接受旧版 finalized 归档，历史 `scripts/analyze_*.py` 命令入口继续保留。独立模型、文献/实验参照与长程任务仍在 `benchmarks/viscosity/`；短程 CI 不声明黏度收敛或实验精度。

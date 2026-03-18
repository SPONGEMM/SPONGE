---
name: sponge-prips-mlp
description: >
  询问使用 SPONGE 的 PRIPS 对接机器学习/神经网络势场时使用。
  适用于 Python plugin 编写、环境准备、力回填。
---

# SPONGE PRIPS MLP

本技能用于把外部机器学习势场通过 `PRIPS` 接到 `SPONGE`。

适用场景：

- 用 `PRIPS` 接入 `MACE`、`MolCT`、`NequIP`、`PaiNN`、自定义 `torch` 势到SPONGE中
- 写 Python plugin 的 `After_Initial` / `Calculate_Force` / `Mdout_Print`

## 推荐的接入路径

默认优先使用：

- `SPONGE` 负责步进、邻居表、输出与体系组织
- `PRIPS` Python plugin 负责调用外部 ML 势
- 外部模型在 plugin 的 `Calculate_Force()` 中读取坐标并回填力

这条路径的优点：

- 不需要先改 SPONGE C++ 核心
- 可以快速验证 ABI、坐标布局、设备与依赖
- benchmark 可直接复用 `benchmarks/utils.py`

## Python plugin 的基本结构

推荐把模型只初始化一次：

- `After_Initial()`
  读取静态原子类型、元素信息、初始坐标；构造 `Atoms`/图对象；加载模型
- `Calculate_Force()`
  从 `Sponge.dd.crd` 或 `Sponge.md_info.crd` 取当前坐标；调用模型；把力写回 `Sponge.dd.frc` 或 `Sponge.md_info.frc`
- `Mdout_Print()`
  输出性能指标、调试信息、额外 reference 文件

推荐策略：

- 只在 `After_Initial()` 构造模型，避免每步重复加载
- 在 `Calculate_Force()` 里只做坐标更新和推理
- 用 `numpy`、`cupy` 或 `pytorch` backend；`jax` 在 PRIPS 中默认只读，不适合原地改力

## 环境与安装

神经网络库通常很重且依赖于操作系统，通常不把环境依赖直接写进 `pixi.toml`。

优先手动安装，以MACE为例：

```bash
pixi run -e dev-cuda13 pip install --no-deps mace-torch
pixi run -e dev-cuda13 pip install --no-deps \
  e3nn==0.4.4 torch-ema prettytable matscipy h5py torchmetrics \
  python-hostlist configargparse GitPython tqdm lmdb orjson pandas \
  opt-einsum-fx lightning-utilities pytz tzdata wcwidth gitdb smmap
```

这样做的原因：

- 避免 `pip` 为 `torch` 再拉一遍整套 CUDA wheels
- 尽量不打乱 `pixi` 已经管理好的 `torch` / CUDA 组合

如果 `PRIPS` 与当前 SPONGE ABI 不匹配，重新安装本地插件：

```bash
pixi run -e dev-cuda13 pip install -e ./plugins/prips
```

## 解析 PRIPS plugin 路径

在 benchmark 或脚本里，优先直接从 `prips` 包位置解析 `_prips.so`：

```python
import prips
from pathlib import Path

plugin_path = Path(prips.__file__).resolve().parent / "_prips.so"
```

不要默认依赖 `python -m prips` 的输出，因为 editable 安装时可能缺额外辅助文件。

## MACE 的具体经验

已验证可行的最小链路：

- 小体系：`benchmarks/validation/misc/statics/tip3p`
- benchmark：`benchmarks/performance/mace/tests/test_tip3p_mace.py`
- 模型：`MACE-OFF23 small`
- 设备：先用 `cpu` 打通，再考虑 `cuda`

在当前仓库里，`perf-mace` 的 assert 来源是：

- plugin 每次调用 MACE 后记录推理耗时
- plugin 在 `Mdout_Print()` 额外保存最后一次纯 MACE 力
- 测试端离线重新调用同一 MACE 模型
- 比较两者误差，阈值当前使用 `max_abs_force_error <= 5e-4`

这类断言验证的是：

- PRIPS 坐标读取正确
- PRIPS 力写回前的模型推理结果正确
- benchmark 统计逻辑正确

它不直接验证：

- SPONGE 总力是否等于纯 MACE 力
- 长程动力学稳定性

---
name: sponge-benchmark-creator
description: >
  询问 SPONGE benchmark 新建 或 调整 时 使用
  - comparison validation performance 分类
  - benchmark 目录 组织
  - benchmark 测试 编写
  - utils 复用
  - pixi task 接入
---

本技能适配 SPONGE 版本号：2.0.0-beta.1

用于新建或调整 `benchmarks/` 下的测试时，目标是让 benchmark 分类清楚、结构统一、可复用、可维护。

## 先判断 benchmark 属于哪一类

- `comparison`
  用于和外部程序对比结果，例如 AMBER、GROMACS、LAMMPS、PySCF。
- `validation`
  用于验证 SPONGE 自身功能正确性、物理合理性、回归稳定性。
- `performance`
  用于验证长程稳定性、采样能力、复杂流程，或者真正的性能表现。

时间尺度也要匹配分类：

- `validation` 应尽量短，适合开发时频繁跑。
- `performance` 可以长，适合专项验证。
- 如果测试只有跑很久才有意义，优先放 `performance`，不要硬塞进 `validation`。

## 目录和组织规则

- 测试代码放在 `benchmarks/<category>/<suite>/tests/`
- 静态输入放在 `benchmarks/<category>/<suite>/statics/<case_name>/`
- reference 数据放在 `reference/` 子目录
- 测试运行输出走 `outputs/`
- 不要把运行结果写回 `statics/`

优先把“同一类能力”的 case 放进已有 suite。

- 如果只是已有测试的参数变体，优先参数化。
- 只有在能力边界明显不同的时候，才新建 suite。

## 新建 benchmark 时必须复用的公共能力

统一优先使用：

- [`benchmarks/utils.py`] 中的 `Outputer.prepare_output_case`
- [`benchmarks/utils.py`] 中的 `Runner.run_sponge`
- [`benchmarks/utils.py`] 中的 `Extractor`
- [`benchmarks/validation/utils.py`] 中已有的 validation 公共解析

避免重复造轮子：

- 不要在 suite 私有 `utils.py` 里再包一层通用 `run_sponge`
- 不要手写仓库里已经有的 `mdout` 解析
- 不要重复实现读原子数、力、势能、压强、应力的通用逻辑

suite 下的 `tests/utils.py` 只应放该 suite 专属逻辑，例如：

- 专用 `mdin` 生成
- 专用输出解析
- 专用 reference 加载或比对

## 测试入口和写法

新增 benchmark 默认用 pytest。若需兼容常见中文编码环境，优先使用 `python -X utf8 -m pytest` 的调用方式。

优先复用已有 fixture：

- `statics_path`
- `outputs_path`
- `mpi_np`

常见骨架：

```python
case_dir = Outputer.prepare_output_case(
    statics_path=statics_path,
    outputs_path=outputs_path,
    case_name=case_name,
    mpi_np=mpi_np,
    run_name=run_name,
)

Runner.run_sponge(case_dir, timeout=timeout, mpi_np=mpi_np)
```

## 断言原则

不要只验证“程序能跑完”，要验证有意义的指标。

- `comparison`
  应比较 energy、force、pressure、stress 等，并给出明确容差。
  如果是单点能/单点力测试，不要打开 `constrain` 或 `settle/shake` 一类约束；这类测试应比较未约束体系的原始势能与力。
- `validation`
  应验证功能行为，例如约束、生效范围、统计分布、边界条件、插件 hook。
- `performance`
  应验证功能行为，例如能量漂移、覆盖度、采样结果、长时间稳定性等。

阈值要显式写出，不要埋魔法数字。具体验证指标需要合适，不一定局限于上述列举指标。

## 统计帧过滤原则

涉及时间序列统计（如 `density`、`temperature`、`pressure`）时，默认排除 `step = 0` 的初始化帧，除非测试目标就是验证初始化输出。

- `step = 0` 常包含初始化路径或未充分平衡状态，直接纳入均值/区间断言会引入偏差。
- 推荐先解析 `step` 列，再按 `step != 0` 过滤后做统计断言。
- 如果必须包含 `step = 0`，在测试代码里显式说明原因，不要隐式依赖。

## 输出原则

测试结束前尽量用 `Outputer.print_table(...)` 输出摘要，至少包含：

- case（作为标题，而非table的一项）
- 关键指标
- 容差或目标值
- `PASS` / `FAIL`

保证失败时能快速定位问题。

## 命名规则

- 测试函数名使用 `test_<behavior>`
- `case_name`、目录名、`run_name` 保持一致或可追踪
- 文件名尽量体现能力，不要过泛

## 静态数据原则

- 不要为小变体复制整套大数据
- 能复用已有 case 就复用
- 没有测试代码承接的静态目录不要长期保留
- 同一个 benchmark 若已迁移，记得同步删掉旧目录和旧 task

## 新建后收尾检查

新增或调整 benchmark 后，至少检查以下内容：

1. 分类是否正确
2. 是否复用了已有公共 utils
3. 是否避免了重复 wrapper 和重复解析逻辑
4. 断言是否验证了真正的指标
5. 输出表格是否可读
6. 是否需要更新 `pixi.toml` 中的 task
7. 统计类断言是否正确处理了 `step = 0`

## 一句话原则

新增 benchmark 要做到：

- 分类清楚
- 时间尺度匹配
- 最大化复用
- 断言有意义
- 结构统一

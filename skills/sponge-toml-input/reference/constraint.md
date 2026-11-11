# 约束算法参数

## 约束模式

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `constrain_mode` | string | 无（不启用约束） | 约束算法 |

`constrain_mode` 可选值：

| 值 | 说明 |
|----|------|
| `"SETTLE"` | SETTLE 算法，专用于刚性水分子（三角形约束） |
| `"SHAKE"` | SHAKE 算法，通用键长约束 |
| `"RATTLE"` | 独立距离约束的 double 机械 NVE，限单 rank 固定正交周期盒 |

## RATTLE NVE

使用 `mode="nve"`、`pbc=true`，约束距离仍由 `[constrain]` 的 `in_file`
和 `mass` 选择；RATTLE 不先分派给 SETTLE，不增加精度、容差、迭代参数。
可添加 `[viscosity] method="green_kubo"`。坐标和切向交错速度保持 double，
原生力保持 float，支持中性固定电荷体系的逐步 direct/reciprocal PME。
初期不支持与 flow、coupling、RNEMD、控压、外加偏置或虚拟点组合。

输入速度沿用交错速度语义；物理速度还需加半步投影加速度。显式随机初速度
在 double 约束投影后归一化动能，此操作不是正则系综抽样。动态 H5 重启必须
有相同算法、约束、步长、盒及 double 状态；旧 SHAKE 重启应结构加载以开始
新实验。完整范围见 [机械 NVE](../../../docs/constrained-nve-core.md)。
每个约束连通分量最多 256 条独立约束；液体能量和黏度收敛尚需验证。

## SETTLE 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `settle_disable` | bool | `false` | 禁用 SETTLE |

## SHAKE 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `SHAKE_step_length` | float | - | SHAKE 迭代收敛参数 |

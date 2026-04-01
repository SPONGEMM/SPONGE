---
name: sponge-toml-input
description: >
  询问 SPONGE 输入文件（mdin.spg.toml）的参数配置时使用。
  适用于模拟参数设置、模块配置、输入输出文件指定等。
---

本技能适配 SPONGE 版本号：2.0.0-beta.1

SPONGE 的输入文件为 TOML 格式，默认文件名 `mdin.spg.toml`，通过 `-mdin` 参数指定：

```bash
SPONGE -mdin mdin.spg.toml
```

参数内容较多，按模块拆分为独立参考文件。本文件为索引，详细参数见各子文件。

## 参考文件索引

| 文件 | 内容 |
|------|------|
| [reference/core.md](reference/core.md) | 核心模拟参数：模式、步数、时间步长、截断距离等 |
| [reference/io.md](reference/io.md) | 输入输出文件与输出控制参数 |
| [reference/thermostat.md](reference/thermostat.md) | 恒温器参数 |
| [reference/barostat.md](reference/barostat.md) | 恒压器参数 |
| [reference/constraint.md](reference/constraint.md) | 约束算法参数（SETTLE / SHAKE） |
| [reference/neighbor_list.md](reference/neighbor_list.md) | 邻居表参数 |
| [reference/pme.md](reference/pme.md) | PME 静电参数 |
| [reference/enhanced_sampling.md](reference/enhanced_sampling.md) | 增强采样参数（CV、MetaD、SITS） |
| [reference/advanced.md](reference/advanced.md) | 高级参数：设备选择、墙约束、ReaxFF、插件等 |

## 最小示例

### NVT

```toml
md_name = "NVT water"
mode = "nvt"
step_limit = 50000
dt = 0.002
cutoff = 8.0
default_in_file_prefix = "WAT"
constrain_mode = "SHAKE"
thermostat = "middle_langevin"
thermostat_tau = 0.1
thermostat_seed = 2026
target_temperature = 300.0
write_information_interval = 1000
```

### NPT

```toml
md_name = "NPT water"
mode = "npt"
step_limit = 100000
dt = 0.002
cutoff = 8.0
default_in_file_prefix = "WAT"
constrain_mode = "SHAKE"
thermostat = "middle_langevin"
thermostat_tau = 0.1
thermostat_seed = 2026
target_temperature = 300.0
barostat = "andersen_barostat"
barostat_tau = 0.1
barostat_update_interval = 10
target_pressure = 1.0
write_information_interval = 1000
```

### 能量最小化

```toml
md_name = "minimization"
mode = "minimization"
step_limit = 1000
default_in_file_prefix = "system"
cutoff = 8.0
write_information_interval = 100
```

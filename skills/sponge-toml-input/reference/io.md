# 输入输出参数

## 输入文件

### 通用前缀

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `default_in_file_prefix` | string | - | 输入文件名前缀，自动匹配 `<prefix>_coordinate.txt` 等 |
| `default_out_file_prefix` | string | - | 输出文件名前缀 |

设置 `default_in_file_prefix = "WAT"` 后，SPONGE 会自动查找：
- `WAT_coordinate.txt` — 坐标
- `WAT_mass.txt` — 质量
- `WAT_charge.txt` — 电荷
- `WAT_LJ.txt` — LJ 参数
- `WAT_bond.txt` — 键
- `WAT_exclude.txt` — 排除列表
- 等等

### 单独指定输入文件

| 参数 | 类型 | 说明 |
|------|------|------|
| `coordinate_in_file` | string | 坐标文件路径 |
| `velocity_in_file` | string | 速度文件路径 |
| `mass_in_file` | string | 质量文件路径 |
| `charge_in_file` | string | 电荷文件路径 |
| `atom_in_file` | string | 原子信息文件 |
| `atom_type_in_file` | string | 原子类型文件 |
| `edge_in_file` | string | 键/边信息文件 |
| `angle_in_file` | string | 角度信息文件 |
| `dihedral_in_file` | string | 二面角信息文件 |
| `atom_numbers` | int | 原子总数（可从文件自动获取） |

### 外部格式导入

| 参数 | 类型 | 说明 |
|------|------|------|
| `amber_parm7` | string | AMBER parm7 力场/拓扑文件 |
| `amber_rst7` | string | AMBER rst7 坐标/速度文件 |
| `gromacs_gro` | string | GROMACS .gro 坐标文件 |
| `gromacs_top` | string | GROMACS .top 拓扑文件 |

## 输出文件

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `mdout` | string | `"mdout.txt"` | 标准输出文件（能量、温度等） |
| `mdinfo` | string | `"mdinfo.txt"` | 模拟信息/日志文件 |
| `crd` | string | - | 坐标轨迹文件（二进制） |
| `vel` | string | - | 速度轨迹文件（二进制） |
| `frc` | string | - | 力轨迹文件（二进制） |
| `box` | string | - | 盒子信息轨迹文件 |
| `rst` | string | - | 重启文件 |
| `trajectory_format` | string | `dat` | 坐标轨迹格式：`dat` 或 `xtc` |

### GROMACS XTC 轨迹

`trajectory_format = "xtc"` 时，坐标轨迹从 `mdcrd.dat`（裸二进制）改为 GROMACS
XTC 文件，取值不区分大小写（`"XTC"` 亦可）。两者互斥，一次运行只写一种坐标轨迹。
`mdbox.txt`、`vel`、`frc` 不受影响（XTC 只存坐标与盒子）。`trajectory_format`
不是 legacy 路径键，因此不会被 H5 输出启用后的侧车抑制规则关掉。

```toml
trajectory_format = "xtc"

[xtc]
path = "prod.xtc"
precision = 1000.0
flush = true
buffer_frames = 10
frame_index = true
append = false
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `path` | string | `mdcrd.xtc` 或 `<default_out_file_prefix>.xtc` | XTC 输出路径 |
| `precision` | float | `1000.0` | 有损精度，`1000.0` 表示 0.001 nm（0.01 Å） |
| `flush` | bool | `true` | 每写完一帧即刷入操作系统 |
| `buffer_frames` | int | `10` | 按整帧对齐的 stdio 缓冲帧数；`<= 0` 退化为单帧缓冲 |
| `frame_index` | bool | `false` | 额外写 `<path>.idx`，记录每帧结束偏移 |
| `append` | bool | `false` | 追加到已有轨迹而不是覆盖 |

`[xtc]` 表只是写法糖，展平后即 `xtc_path`、`xtc_precision`、`xtc_flush`、
`xtc_buffer_frames`、`xtc_frame_index`、`xtc_append`，扁平写法等价。

注意事项：

- **写入节奏**：XTC 帧与 legacy 坐标流共用 `write_trajectory_interval` 与同一个
  输出时机，切换格式不改变哪些步被记录；帧的 `step` 取 `sys.steps`，`time` 取
  `sys.Get_Current_Time()`（单位 ps）。
- **单位**：SPONGE 内部为 Å，XTC 规范为 nm，写出时统一乘 0.1。该系数不可配置
  —— 容器内没有单位标记，改动会产出其他工具全部误读的轨迹。
- **盒子**：每帧写入当前晶胞的下三角矩阵（nm），NPT 的形变三斜胞也会逐帧记录。
- **精度**：XTC 有损。`precision = 1000.0` 将坐标量化到 0.001 nm（0.01 Å），
  与 GROMACS 惯例一致。xdrfile 把 `坐标_nm × precision` 存成 32 位整数，缩放后
  的值必须落在该整数网格内；无法表示首帧的 precision 会在写首帧时报错，而不是
  静默写坏坐标。
- **中断安全**：XTC 每帧自描述、无文件尾标记，只要文件结束在帧边界就可读。
  `flush = true` 配合整帧对齐的 `buffer_frames`，单帧不会跨越缓冲边界，中断时
  已写完的帧全部落盘，最多丢掉正在写的那一帧。`frame_index = true` 额外把每帧
  结束偏移（8 字节、本机字节序）写入 `<path>.idx` 并逐帧刷盘，被 kill 后可精确
  截断到最后一个偏移。
- **续跑**：`append = true` 会先校验已有文件的原子数与当前体系一致，再续写，
  `.idx` 同步追加。
- **不支持的情形**：`rerun` 模式禁止 `trajectory_format = "xtc"`；同时设置
  `crd` 会被忽略并给出警告。
- **构建要求**：XTC writer 只在 `SPONGE_WITH_XTC=ON`（默认）时编译进来；关掉该
  选项的二进制会把 `trajectory_format = "xtc"` 当作初始化错误，而不是静默退回
  `dat`。
- **多副本**：XTC 是单流容器，无法表达多副本，每个副本需单独文件。
- **回读**：输出是标准 GROMACS XTC，VMD、PyMOL、MDAnalysis、`gmx trjconv`、
  `gmx check` 均可直接读取，无需转换。

## 输出频率控制

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `write_information_interval` | int | `1000` | mdinfo/mdout 写入间隔（步） |
| `write_mdout_interval` | int | `1000` | mdout 写入间隔 |
| `write_trajectory_interval` | int | 同 `write_information_interval` | 轨迹写入间隔 |
| `write_restart_file_interval` | int | `step_limit` | 重启文件写入间隔 |
| `max_restart_export_count` | int | - | 保留的重启文件最大数量 |
| `buffer_frame` | int | `10` | 文件缓冲帧数（影响 I/O 性能） |

## 输出内容控制

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `print_zeroth_frame` | int | `0` | 是否输出第 0 步的帧（1 = 是） |
| `print_pressure` | int | `0` | 是否输出压力 |
| `print_detail` | int | `0` | 详细输出级别 |

# Input File Reference

SPONGE input files use TOML format. The default filename is `mdin.spg.toml`, specified via the `-mdin` flag:

```bash
SPONGE -mdin mdin.spg.toml
```

## Table of Contents

- [Core Parameters](core.md) — mode, steps, time step, cutoff
- [Input/Output](io.md) — file specification and output control
- [Thermostat](thermostat.md) — thermostat algorithms and parameters
- [Barostat](barostat.md) — barostat algorithms and parameters
- [Restrain](restrain.md) — atom-coordinate restrain input
- [Constraints](constraint.md) — SETTLE / SHAKE
- [Neighbor List](neighbor-list.md) — neighbor list configuration
- [PME Electrostatics](pme.md) — Particle Mesh Ewald
- [Collective Variables](collective-variables.md) — CV definitions and examples
- [Enhanced Sampling](enhanced-sampling.md) — CV, MetaD, SITS
- [SinkMeta](sinkmeta.md) — detailed `meta` / SinkMeta parameter reference
- [Advanced](advanced.md) — device, wall constraints, ReaxFF, plugins

## Quick Examples

### NVT

```toml
md_name = "NVT water"
mode = "nvt"
step_limit = 50000
dt = 0.002
cutoff = 8.0
default_in_file_prefix = "WAT"
constrain_mode = "SHAKE"
target_temperature = 300.0
write_information_interval = 1000

[thermostat]
mode = "middle_langevin"
tau = 0.1
seed = 2026
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
target_temperature = 300.0
target_pressure = 1.0
write_information_interval = 1000

[thermostat]
mode = "middle_langevin"
tau = 0.1
seed = 2026

[barostat]
mode = "andersen_barostat"
tau = 0.1
update_interval = 10
```

### Energy Minimization

```toml
md_name = "minimization"
mode = "minimization"
step_limit = 1000
default_in_file_prefix = "system"
cutoff = 8.0
write_information_interval = 100
```

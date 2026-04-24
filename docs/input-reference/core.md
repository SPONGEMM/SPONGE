# Core Simulation Parameters

## Simulation Identity

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `md_name` | string | `"Default SPONGE MD Task Name"` | Simulation task name |

## Simulation Mode

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | string | **required** | Simulation mode |

`mode` options:

| Value | Description |
|-------|-------------|
| `"nve"` | Microcanonical ensemble (constant energy) |
| `"nvt"` | Canonical ensemble (constant temperature), requires thermostat |
| `"npt"` | Isothermal-isobaric ensemble, requires thermostat + barostat |
| `"minimization"` / `"min"` | Energy minimization |
| `"rerun"` | Trajectory reanalysis |

## Periodic Boundary Conditions

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pbc` | bool | `true` | Enable periodic boundary conditions |

If `pbc = false`, SPONGE switches to the no-PBC path. This cannot be used with
`npt`, is not supported in multi-process mode, and expects a very large box.

## Time and Steps

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `step_limit` | int | `1000` | Total simulation steps |
| `dt` | float | `0.001` | Time step (ps) |
| `frame_limit` | int | - | Frame limit for rerun mode |

## Force Cutoff

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cutoff` | float | `8.0` | Non-bonded interaction cutoff distance (A) |
| `skin` | float | `2.0` | Neighbor list skin distance (A) |

## Temperature and Pressure Targets

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_temperature` | float | `300.0` | Top-level target temperature (K), used in NVT/NPT |
| `target_pressure` | float | `1.0` | Top-level target pressure (bar), used in NPT |

These two keys stay at the top level even when thermostat and barostat options are
written in `[thermostat]` and `[barostat]`.

Temperature and pressure also support schedules through the top-level keys
`target_temperature_schedule_*` and `target_pressure_schedule_*`. See
[Thermostat](thermostat.md) and [Barostat](barostat.md).

## Working Directory

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `workspace` | string | directory of mdin file | Working directory path, can be absolute or relative |

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
| `target_temperature` | float | `300.0` | Target temperature (K), required for NVT/NPT |
| `target_pressure` | float | `1.0` | Target pressure (bar), required for NPT |

Temperature and pressure support schedules (step/linear changes). See [Thermostat](thermostat.md) and [Barostat](barostat.md).

## Working Directory

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `workspace` | string | directory of mdin file | Working directory path, can be absolute or relative |

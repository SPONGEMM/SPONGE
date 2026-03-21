# Thermostat Parameters

## Thermostat Selection

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `thermostat` | string | **required for NVT/NPT** | Thermostat algorithm |

`thermostat` options:

| Value | Description |
|-------|-------------|
| `"middle_langevin"` / `"langevin"` | Middle Langevin dynamics (recommended) |
| `"andersen"` | Andersen thermostat |
| `"berendsen_thermostat"` | Berendsen thermostat |
| `"bussi_thermostat"` | Bussi (velocity rescaling) thermostat |
| `"nose_hoover_chain"` | Nose-Hoover chain thermostat |

## Common Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_temperature` | float | `300.0` | Target temperature (K) |
| `thermostat_tau` / `tau` | float | - | Coupling time constant (ps) |
| `thermostat_seed` / `seed` | int | - | Random seed (required for stochastic thermostats) |
| `velocity_max` | float | - | Maximum allowed velocity |

## Temperature Schedule

Supports changing target temperature during simulation:

| Parameter | Type | Description |
|-----------|------|-------------|
| `target_temperature_schedule_mode` | string | `"step"` (step change) or `"linear"` (linear interpolation) |
| `target_temperature_schedule_steps` | array | Inline schedule, format: `[{step = N, value = T}, ...]` |
| `target_temperature_schedule_file` | string | External schedule file path (TOML format) |

Example:

```toml
target_temperature = 300.0
target_temperature_schedule_mode = "linear"
target_temperature_schedule_steps = [
    { step = 0, value = 100.0 },
    { step = 10000, value = 300.0 },
    { step = 50000, value = 300.0 },
]
```

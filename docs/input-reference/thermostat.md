# Thermostat Parameters

SPONGE requires a thermostat in `nvt` and `npt` modes.

For TOML input, prefer the grouped form:

```toml
mode = "nvt"
target_temperature = 300.0

[thermostat]
mode = "middle_langevin"
tau = 0.1
seed = 2026
```

The parser also accepts the flat compatibility keys `thermostat`,
`thermostat_tau`, and `thermostat_seed`, but new documentation should use the
grouped TOML form above.

## Thermostat Selection

| Parameter | Scope | Type | Default | Description |
|-----------|-------|------|---------|-------------|
| `mode` | `thermostat` | string | required for NVT/NPT | Thermostat algorithm |

Supported values:

| Value | Description |
|-------|-------------|
| `"middle_langevin"` | Middle Langevin dynamics |
| `"langevin"` | Deprecated alias of `middle_langevin` |
| `"andersen"` | Andersen thermostat |
| `"berendsen_thermostat"` | Berendsen thermostat |
| `"bussi_thermostat"` | Bussi velocity-rescaling thermostat |
| `"nose_hoover_chain"` | Nose-Hoover chain thermostat |

## Top-level Parameters

| Parameter | Scope | Type | Default | Description |
|-----------|-------|------|---------|-------------|
| `target_temperature` | `top-level` | float | `300.0` | Target temperature (K) |
| `velocity_max` | `top-level` | float | disabled | Velocity cap used by some thermostats |

`target_temperature` is not part of `[thermostat]`. It stays at the top level
and is also the quantity affected by temperature schedules.

## Common Thermostat Parameters

| Parameter | Scope | Type | Default | Description |
|-----------|-------|------|---------|-------------|
| `tau` | `thermostat` | float | `1.0` | Coupling time constant in ps |
| `seed` | `thermostat` | int | random | Random seed for stochastic thermostats |

Notes:

- `seed` is used by `middle_langevin`, `andersen`, and `bussi_thermostat`
- `berendsen_thermostat` and `nose_hoover_chain` do not read `seed`
- `middle_langevin` converts `tau` internally to a friction coefficient
  `gamma = 1 / tau`
- `andersen` converts `tau` to an internal collision update interval

## Algorithm-specific Parameters

### `middle_langevin`

```toml
[thermostat]
mode = "middle_langevin"
tau = 0.1
seed = 2026
```

Uses `tau` and `seed` in the `thermostat` scope, plus the top-level
`velocity_max`.

### `andersen`

```toml
[thermostat]
mode = "andersen"
tau = 0.1
seed = 2026
```

Uses `tau` and `seed` in the `thermostat` scope, plus the top-level
`velocity_max`.

### `berendsen_thermostat`

```toml
[thermostat]
mode = "berendsen_thermostat"
tau = 0.1
```

Uses only `tau` in the `thermostat` scope.

### `bussi_thermostat`

```toml
[thermostat]
mode = "bussi_thermostat"
tau = 0.1
seed = 2026
```

Uses `tau` and `seed` in the `thermostat` scope.

### `nose_hoover_chain`

```toml
[thermostat]
mode = "nose_hoover_chain"
tau = 0.2

[nose_hoover_chain]
length = 3
restart_input = "nhc_restart.in"
restart_output = "nhc_restart.out"
crd = "nhc_crd.dat"
vel = "nhc_vel.dat"
```

Additional parameters:

| Parameter | Scope | Type | Default | Description |
|-----------|-------|------|---------|-------------|
| `length` | `nose_hoover_chain` | int | `1` | Chain length |
| `restart_input` | `nose_hoover_chain` | string | - | Read initial chain state from file |
| `restart_output` | `nose_hoover_chain` | string | - | Write chain state to file |
| `crd` | `nose_hoover_chain` | string | - | Write chain coordinate trajectory |
| `vel` | `nose_hoover_chain` | string | - | Write chain velocity trajectory |

`nose_hoover_chain` also reads the top-level `velocity_max`.

## Temperature Schedule

Temperature schedules are controlled by top-level keys, not by `[thermostat]`:

| Parameter | Type | Description |
|-----------|------|-------------|
| `target_temperature_schedule_mode` | string | `"step"` or `"linear"` |
| `target_temperature_schedule_steps` | array | Inline schedule: `[{ step = N, value = T }, ...]` |
| `target_temperature_schedule_file` | string | Path to a TOML schedule file |

Inline `target_temperature_schedule_steps` only works when the main mdin file is
TOML.

```toml
target_temperature = 300.0
target_temperature_schedule_mode = "linear"
target_temperature_schedule_steps = [
    { step = 0, value = 100.0 },
    { step = 10000, value = 300.0 },
    { step = 50000, value = 300.0 },
]
```

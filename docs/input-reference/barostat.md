# Barostat Parameters

SPONGE requires a barostat in `npt` mode.

For TOML input, prefer the grouped form:

```toml
mode = "npt"
target_pressure = 1.0

[barostat]
mode = "andersen_barostat"
tau = 0.1
update_interval = 10
```

The parser also accepts the flat compatibility keys `barostat`,
`barostat_tau`, `barostat_update_interval`, and similar `barostat_*` keys, but
new documentation should use grouped TOML form.

## Barostat Selection

| Parameter | Scope | Type | Default | Description |
|-----------|-------|------|---------|-------------|
| `mode` | `barostat` | string | required for NPT | Barostat algorithm |

Supported values:

| Value | Description |
|-------|-------------|
| `"andersen_barostat"` | Pressure-based Andersen barostat |
| `"berendsen_barostat"` | Pressure-based Berendsen barostat |
| `"bussi_barostat"` | Pressure-based Bussi barostat |
| `"monte_carlo_barostat"` | Monte Carlo barostat |

## Top-level Parameters

| Parameter | Scope | Type | Default | Description |
|-----------|-------|------|---------|-------------|
| `target_pressure` | `top-level` | float | `1.0` | Target pressure in bar |
| `target_surface_tensor` | `top-level` | float | `0.0` | Target surface tension term used by pressure-based barostats |

`target_pressure` stays at the top level even when barostat options are written
under `[barostat]`.

## Pressure-based Barostats

The algorithms `andersen_barostat`, `berendsen_barostat`, and `bussi_barostat`
share the same parameter set.

```toml
[barostat]
mode = "andersen_barostat"
tau = 0.1
compressibility = 4.5e-5
update_interval = 10
isotropy = "isotropic"
```

### Common Parameters

| Parameter | Scope | Type | Default | Description |
|-----------|-------|------|---------|-------------|
| `tau` | `barostat` | float | `1.0` | Barostat time constant (ps) |
| `compressibility` | `barostat` | float | `4.5e-5` | Isothermal compressibility in `bar^-1` |
| `update_interval` | `barostat` | int | `10` | Pressure-coupling update interval |
| `isotropy` | `barostat` | string | `"isotropic"` | Pressure-coupling mode |

Supported `isotropy` values:

| Value | Description |
|-------|-------------|
| `"isotropic"` | Uniform scaling on all axes |
| `"semiisotropic"` | Semi-isotropic coupling |
| `"semianisotropic"` | Semi-anisotropic coupling |
| `"anisotropic"` | Fully anisotropic coupling |

### Box-deformation Parameters

| Parameter | Scope | Type | Description |
|-----------|-------|------|-------------|
| `g11`, `g21`, `g22`, `g31`, `g32`, `g33` | `barostat` | float | Initial box-velocity tensor elements |
| `x_constant`, `y_constant`, `z_constant` | `barostat` | bool | Freeze box scaling on selected axes |

The `x_constant`, `y_constant`, and `z_constant` flags are booleans in the
source code, not floats.

### Surface-tension Parameters

Pressure-based barostats read the target surface term from the top-level key
`target_surface_tensor`. The compatibility key `barostat_target_surface_tensor`
is also accepted by the parser, but should not be the primary form shown in the
documentation.

## Monte Carlo Barostat

`monte_carlo_barostat` uses a different parameter set and currently supports
orthogonal boxes only.

```toml
[barostat]
mode = "monte_carlo_barostat"

[monte_carlo_barostat]
initial_ratio = 0.001
update_interval = 100
check_interval = 10
accept_rate_low = 30
accept_rate_high = 40
couple_dimension = "XYZ"
```

### Parameters

| Parameter | Scope | Type | Default | Description |
|-----------|-------|------|---------|-------------|
| `initial_ratio` | `monte_carlo_barostat` | float | `0.001` | Initial maximum fractional box-length move |
| `update_interval` | `monte_carlo_barostat` | int | `100` | MC attempt interval |
| `check_interval` | `monte_carlo_barostat` | int | `10` | Acceptance-rate adjustment interval |
| `accept_rate_low` | `monte_carlo_barostat` | float | `30` | Lower target acceptance rate in percent |
| `accept_rate_high` | `monte_carlo_barostat` | float | `40` | Upper target acceptance rate in percent |
| `couple_dimension` | `monte_carlo_barostat` | string | `"XYZ"` | Coupled box dimensions |
| `only_direction` | `monte_carlo_barostat` | string | - | Restrict moves within the selected coupling mode |
| `surface_number` | `monte_carlo_barostat` | int | `0` | Number of surfaces, only used for non-`NO`/non-`XYZ` coupling |
| `surface_tension` | `monte_carlo_barostat` | float | `0.0` | Surface tension, only used for non-`NO`/non-`XYZ` coupling |

Supported `couple_dimension` values are `"XYZ"`, `"NO"`, `"XY"`, `"XZ"`, and
`"YZ"`.

## Pressure Schedule

Pressure schedules are controlled by top-level keys, not by `[barostat]`:

| Parameter | Type | Description |
|-----------|------|-------------|
| `target_pressure_schedule_mode` | string | `"step"` or `"linear"` |
| `target_pressure_schedule_steps` | array | Inline schedule: `[{ step = N, value = P }, ...]` |
| `target_pressure_schedule_file` | string | Path to a TOML schedule file |

Inline `target_pressure_schedule_steps` only works when the main mdin file is
TOML.

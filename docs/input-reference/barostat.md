# Barostat Parameters

## Barostat Selection

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `barostat` | string | **required for NPT** | Barostat algorithm |

`barostat` options:

| Value | Description |
|-------|-------------|
| `"andersen_barostat"` | Andersen barostat |
| `"berendsen_barostat"` | Berendsen barostat |
| `"bussi_barostat"` | Bussi barostat |
| `"monte_carlo_barostat"` | Monte Carlo barostat |

## Common Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_pressure` | float | `1.0` | Target pressure (bar) |
| `barostat_tau` | float | - | Coupling time constant (ps) |
| `barostat_update_interval` | int | - | Barostat update interval (steps) |
| `barostat_compressibility` | float | - | Isothermal compressibility (bar^-1) |

## Anisotropy Control

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `barostat_isotropy` / `isotropy` | string | `"isotropic"` | Pressure coupling type |

`barostat_isotropy` options:

| Value | Description |
|-------|-------------|
| `"isotropic"` | Isotropic, uniform scaling on all axes |
| `"semiisotropic"` | Semi-isotropic |
| `"semianisotropic"` | Semi-anisotropic |
| `"anisotropic"` | Anisotropic, independent scaling per axis |

## Box Deformation Control

| Parameter | Type | Description |
|-----------|------|-------------|
| `barostat_g11` / `g21` / `g22` / `g31` / `g32` / `g33` | float | Box deformation matrix elements |
| `barostat_x_constant` / `y_constant` / `z_constant` | float | Fix box size along a specific axis |

## Surface Tension

| Parameter | Type | Description |
|-----------|------|-------------|
| `surface_tensor` | string | Surface tension control mode |
| `surface_number` | int | Number of surfaces |
| `surface_tension` | float | Surface tension value |

## Pressure Schedule

Similar to temperature schedule, supports changing target pressure during simulation:

| Parameter | Type | Description |
|-----------|------|-------------|
| `target_pressure_schedule_mode` | string | `"step"` or `"linear"` |
| `target_pressure_schedule_steps` | array | Inline schedule |
| `target_pressure_schedule_file` | string | External schedule file path |

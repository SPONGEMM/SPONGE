# Enhanced Sampling Parameters

## Collective Variables (CV)

| Parameter | Type | Description |
|-----------|------|-------------|
| `cv_in_file` | string | Collective variable definition file path |

CV definitions are loaded from the file pointed to by `cv_in_file`. New CV
files should use TOML; the legacy block format remains supported for
compatibility. See
[Collective Variables Guide](collective-variables.md) for the supported CV
types, virtual atoms, printing, and complete examples.

## Metadynamics

Configured via `[META]` or `[meta]` section:

| Parameter | Type | Description |
|-----------|------|-------------|
| `sink` | string | Sink metadynamics mode |

Detailed SinkMeta parameter list: [sinkmeta.md](sinkmeta.md)

## SITS

SITS parameters live under the `[SITS]` scope:

```toml
[SITS]
mode = "iteration"
atom_numbers = "ALL"
k_numbers = 40
T_low = 300.0
T_high = 600.0
record_interval = 1
update_interval = 100
```

### Common Parameters

| Parameter | Scope | Type | Default | Description |
|-----------|-------|------|---------|-------------|
| `mode` | `SITS` | string | required | SITS run mode |
| `atom_in_file` | `SITS` | string | - | File-based atom partition definition for selective enhancement |
| `atom_numbers` | `SITS` | int or string | required if `atom_in_file` is absent | Number of leading atoms in the selectively enhanced set, or `"ALL"` / `"ITS"` |
| `cross_enhance_factor` | `SITS` | float | `0.5` | Enhancement factor for the cross term |
| `fb_interval` | `SITS` | int | `1` | Feedback-bias update interval |

Exactly one atom-selection method is required: `atom_in_file` or
`atom_numbers`.

### Iteration / Production Parameters

| Parameter | Scope | Type | Default | Description |
|-----------|-------|------|---------|-------------|
| `k_numbers` | `SITS` | int | `40` | Number of temperature points |
| `T_low` | `SITS` | float | - | Lower temperature bound for generated ladder |
| `T_high` | `SITS` | float | - | Upper temperature bound for generated ladder |
| `T` | `SITS` | slash-separated string | - | Explicit temperature ladder used instead of `T_low` / `T_high` |
| `record_interval` | `SITS` | int | `1` | Energy record interval |
| `update_interval` | `SITS` | int | `100` | Parameter update interval |
| `pe_a` | `SITS` | float | `1.0` | Multiplicative energy scaling factor |
| `pe_b` | `SITS` | float | `0.0` | Additive energy offset |
| `fb_bias` | `SITS` | float | `0.0` | Additional bias applied to `fb` |
| `nk_rest` | `SITS` | bool | `false` in `iteration`, `true` otherwise | Whether to read restart `Nk` values |
| `nk_in_file` | `SITS` | string | required when `nk_rest = true` or in `production` mode | Input file for restarting `Nk` |
| `nk_fix` | `SITS` | bool | `false` in `iteration`, `true` otherwise | Freeze `Nk` during the run |
| `nk_rest_file` | `SITS` | string | auto-generated | Output file for restart `Nk` values when `nk_fix = false` |
| `nk_traj_file` | `SITS` | string | auto-generated | Output file for `Nk` history when `nk_fix = false` |

`mode` options:

| Value | Description |
|-------|-------------|
| `"observation"` | Observation phase |
| `"iteration"` | Iteration phase |
| `"production"` | Production phase |
| `"empirical"` | Empirical mode |
| `"amd"` | AMD mode |
| `"gamd"` | GaMD mode |

### Empirical / AMD / GaMD Parameters

| Parameter | Scope | Type | Default | Description |
|-----------|-------|------|---------|-------------|
| `pe_a` | `SITS` | float | required in `amd` / `gamd`, `1.0` in `empirical` | Bias strength parameter |
| `pe_b` | `SITS` | float | required in `amd` / `gamd`, `0.0` in `empirical` | Bias threshold parameter |
| `T_low` | `SITS` | float | required in `empirical` | Lower temperature bound |
| `T_high` | `SITS` | float | required in `empirical` | Upper temperature bound |

Notes:

- `amd`, `gamd`, and `empirical` do not use the full `k_numbers` / `Nk` update
  workflow.
- The current source reads `T` as a slash-separated string rather than a TOML
  numeric array.

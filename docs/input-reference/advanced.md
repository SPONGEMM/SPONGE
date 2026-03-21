# Advanced Parameters

## Device and Parallelism

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `device` | string | auto-detected | GPU device ID |
| `device_optimized_block` | int | - | GPU block size optimization |

## Wall Constraints

### Hard Walls

| Parameter | Type | Description |
|-----------|------|-------------|
| `hard_wall_x_low` / `hard_wall_x_high` | float | Hard wall position in X direction (A) |
| `hard_wall_y_low` / `hard_wall_y_high` | float | Hard wall position in Y direction (A) |
| `hard_wall_z_low` / `hard_wall_z_high` | float | Hard wall position in Z direction (A) |

### Soft Walls

| Parameter | Type | Description |
|-----------|------|-------------|
| `soft_walls_in_file` | string | Soft wall potential parameter file path |

## ReaxFF Reactive Force Field

Configured via `[REAXFF]` section:

```toml
[REAXFF]
in_file = "ffield.reax.cho"
type_in_file = "atom_types.txt"
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `in_file` | string | ReaxFF parameter file |
| `type_in_file` | string | Atom type mapping file |

## Custom Forces

| Parameter | Type | Description |
|-----------|------|-------------|
| `custom_force` | section | Custom force configuration |
| `pairwise_force` | section | Custom pairwise force configuration |

## Plugins

External plugins (e.g., PRIPS) are configured via the `[plugin]` section.

## Control Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `command_only` | bool | `false` | Use command-line arguments only, skip mdin file |
| `dont_check_input` | bool | `false` | Skip unused parameter checking |
| `end_pause` | bool | `false` | Pause after simulation ends |

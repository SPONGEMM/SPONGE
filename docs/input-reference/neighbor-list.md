# Neighbor List Parameters

Neighbor list parameters are set via the `[neighbor_list]` TOML section:

```toml
[neighbor_list]
max_neighbor_numbers = 1200
skin_permit = 0.5
```

## Parameter List

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `refresh_interval` | int | auto | Neighbor list rebuild interval (steps) |
| `skin_permit` | float | `0.5` | Skin distance growth allowance, used to trigger rebuilds |
| `max_neighbor_numbers` | int | `1200` | Maximum neighbors per atom |
| `max_atom_in_grid_numbers` | int | `150` | Maximum atoms per grid cell |
| `max_ghost_in_grid_numbers` | int | `150` | Maximum ghost atoms per grid cell |
| `max_padding` | int | - | Maximum array padding |
| `min_padding` | int | - | Minimum array padding |
| `check_overflow_interval` | int | - | Memory overflow check interval |
| `throw_error_when_overflow` | bool | `false` | Whether to throw an error on overflow (otherwise auto-expands) |

The rebuild strategy defaults to automatic detection based on `skin` distance. `refresh_interval = 0` enables automatic mode.

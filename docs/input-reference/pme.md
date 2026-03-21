# PME Electrostatics Parameters

PME (Particle Mesh Ewald) parameters are set via the `[PME]` TOML section:

```toml
[PME]
grid_spacing = 1.0
```

## Parameter List

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fftx` / `ffty` / `fftz` | int | auto | FFT grid dimensions |
| `grid_spacing` | float | `1.0` | Grid spacing (A), used to auto-determine FFT dimensions |
| `update_interval` | int | `1` | Reciprocal space summation update interval (steps) |
| `Direct_Tolerance` | float | - | Direct space cutoff tolerance |
| `calculate_excluded_part` | bool | - | Whether to compute excluded pair interactions |
| `calculate_reciprocal_part` | bool | - | Whether to compute the reciprocal space part |

If `fftx/ffty/fftz` are not specified, SPONGE automatically calculates them from `grid_spacing` and the box dimensions.

# SinkMeta/meta Input Parameters

This list is derived from `META::Initial()` in `SPONGE/bias/sinkmeta/config/config.cpp` and default values in `SPONGE/bias/sinkmeta/meta.h`.

## Parameters (module: `meta`)

| Parameter | Type / Shape | Required | Default (if any) | Meaning / Notes |
| --- | --- | --- | --- | --- |
| `CV` | string list | Yes | None | CV module names used by Meta; determines dimensionality. If missing, META is not initialized. |
| `dip` | float | No | `0.0` | Extra dip term for submarine/sink behavior (adds to bias shift via `kB*T`). |
| `welltemp_factor` | float | No | `1e9` | Well-tempered bias factor; if > 1 enables well-tempered mode. |
| `Ndim` | int | No | `CV` count | Explicit dimensionality; must match `CV` list length. |
| `subhill` | flag | No | `false` | Enable sub-hill (Gaussian) behavior. Presence-only (no value read). |
| `kde` | int | No | `0` | Nonzero enables KDE mode and also sets `subhill=true`. Also switches sigma scaling to `1.414/sigma`. |
| `mask` | int | No | `0` | Enable mask mode (n-dim area exit label). |
| `maxforce` | float | No | `0.1` | Edge force criterion for exit label; only read when `mask` is set. |
| `sink` | int | No | `0` | Nonzero enables negative-hill (sink/submarine) behavior. |
| `sumhill_freq` | int | No | `0` | History frequency for `sumhill` accumulation (affects Rbias/RCT). |
| `convmeta` | int | No | `0` | ConvolutionMeta flag; also sets `do_negative=true`. |
| `grw` | int | No | `0` | GRW flag; also sets `do_negative=true`. |
| `CV_period` | float array (ndim) | Yes | None in META | Periodic box length per CV. Always requested in `META::Initial()`. |
| `CV_sigma` | float array (ndim) | Yes | None in META | Gaussian width per CV (must be > 0). Stored internally as inverse sigma. |
| `cutoff` | float array (ndim) | No | `3 * CV_sigma` | Neighbor cutoff for lookup and border wall; if present enables `do_cutoff`. |
| `potential_in_file` | string | No | None | Read potential from file; if set, `Read_Potential()` is called (grid/scatter settings below are bypassed). |
| `scatter_in_file` | string | No | None | Read scatter potential from file; sets `use_scatter=true`, `usegrid=false`, and calls `Read_Potential()`. |
| `edge_in_file` | string | No | `sumhill.log` | Edge-effect cache file used by `EdgeEffect()`. If the file exists and matches the expected grid size, SinkMeta reads it directly; otherwise it regenerates and writes to the same path. |
| `scatter` | int | No | `0` | Number of scatter points; if > 0 uses scatter points instead of grid. |
| `CV_minimal` | float array (ndim) | Conditionally | None in META | Grid minimum per CV; required when not using `potential_in_file` or `scatter_in_file`. |
| `CV_maximum` | float array (ndim) | Conditionally | None in META | Grid maximum per CV; must be > `CV_minimal`. Required when not using `potential_in_file` or `scatter_in_file`. |
| `CV_grid` | int array (ndim) | Conditionally | None in META | Grid points per CV; must be > 1. Required when not using `potential_in_file` or `scatter_in_file`. |
| `height` | float | No | `1.0` | Initial hill height (`height_0`). |
| `wall_height` | float | No | None | Enables border wall and sets `border_potential_height`. |
| `potential_out_file` | string | No | `Meta_Potential.txt` | Output file name for writing potential. |
| `potential_update_interval` | int | No | `write_information_interval` or `1000` | Step interval for writing potential; if <= 0 it is forced to 1000. |

## Parameters used from other modules / global

| Parameter | Scope | Type / Shape | Required | Default (if any) | Meaning / Notes |
| --- | --- | --- | --- | --- | --- |
| `CV_point` | Each CV module | float array (scatter size) | Conditionally | None in META | Scatter point coordinates per CV; read only when `scatter > 0`. |
| `write_information_interval` | Controller | int | No | `1000` | Global controller interval; used as default for `potential_update_interval`. |

## Notes

- If both `potential_in_file` and `scatter_in_file` are present, `potential_in_file` takes precedence (the code uses `if/else if`).
- `CV_sigma` is inverted internally; `kde` mode uses `1.414 / sigma`, otherwise `1.0 / sigma`.
- Default file names set in code: `read_potential_file_name` and `write_potential_file_name` both start as `Meta_Potential.txt`.

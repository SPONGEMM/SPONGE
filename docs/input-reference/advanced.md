# Advanced Parameters

## Device and Parallelism

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `device` | string | auto-detected | GPU device ID |
| `device_optimized_block` | int | - | GPU block size optimization |

## Domain Decomposition

MPI domain decomposition parameters are read from the `[DOM_DEC]` scope:

```toml
[DOM_DEC]
update_interval = 100
split_nx = 2
split_ny = 2
split_nz = 1
```

| Parameter | Scope | Type | Default | Description |
|-----------|-------|------|---------|-------------|
| `update_interval` | `DOM_DEC` | int | `100` | Domain-decomposition refresh interval |
| `split_nx` | `DOM_DEC` | int | auto | Number of splits in X |
| `split_ny` | `DOM_DEC` | int | auto | Number of splits in Y |
| `split_nz` | `DOM_DEC` | int | auto | Number of splits in Z |

When the split counts are omitted, SPONGE chooses them automatically from the
PP MPI size and box geometry. Current MPI domain decomposition only supports
orthogonal boxes.

## Minimization

Minimization-specific parameters use the `[minimization]` scope:

```toml
mode = "minimization"

[minimization]
dynamic_dt = 1
max_move = 0.1
beta1 = 0.9
epsilon = 1e-4
```

| Parameter | Scope | Type | Default | Description |
|-----------|-------|------|---------|-------------|
| `dynamic_dt` | `minimization` | int | `1` | Use the ADAM-like dynamic-step minimizer |
| `max_move` | `minimization` | float | `0.1` | Maximum coordinate displacement per step in angstrom |
| `momentum_keep` | `minimization` | float | `0.0` when `dynamic_dt = 0` | Momentum retention factor in gradient-descent mode |
| `beta1` | `minimization` | float | `0.9` | First ADAM parameter used when `dynamic_dt != 0` |
| `beta2` | `minimization` | float | `0.9` in the current source | Second ADAM parameter used when `dynamic_dt != 0` |
| `epsilon` | `minimization` | float | `1e-4` | Numerical stabilizer used when `dynamic_dt != 0` |

When `dynamic_dt != 0`, SPONGE internally resets `dt` to `3e-4 ps`. When
`dynamic_dt = 0`, it uses a tiny fixed `dt` and the gradient-descent path.

## Rerun Mode

`rerun` uses the normal top-level trajectory file keys plus several
rerun-specific flat keys:

```toml
mode = "rerun"
crd = "traj.dat"
box = "traj.box"
vel = "traj.vel"
frame_limit = 1000
rerun_start = 0
rerun_strip = 0
rerun_need_box_update = 0
```

| Parameter | Scope | Type | Default | Description |
|-----------|-------|------|---------|-------------|
| `crd` | `top-level` | string | required | Input coordinate trajectory for rerun |
| `box` | `top-level` | string | required | Input box trajectory for rerun |
| `vel` | `top-level` | string | optional | Input velocity trajectory for rerun |
| `frame_limit` | `top-level` | int | - | Maximum number of rerun frames |
| `rerun_frame_limit` | `top-level` | int | alias of `frame_limit` | Compatibility alias |
| `rerun_start` | `top-level` | int | `0` | Starting frame offset |
| `rerun_strip` | `top-level` | int | `0` | Number of frames skipped between processed frames |
| `rerun_need_box_update` | `top-level` | int | `0` | Whether to update the box deformation tensor during rerun |

The current source uses flat `rerun_*` keys rather than a `[rerun]` TOML
section.

## Wall Constraints

### Hard Walls

Hard-wall limits are configured through the `[hard_wall]` prefix:

```toml
[hard_wall]
x_low = 0.0
x_high = 40.0
```

| Parameter | Scope | Type | Description |
|-----------|-------|------|-------------|
| `x_low`, `x_high` | `hard_wall` | float | Hard-wall position in the X direction |
| `y_low`, `y_high` | `hard_wall` | float | Hard-wall position in the Y direction |
| `z_low`, `z_high` | `hard_wall` | float | Hard-wall position in the Z direction |

Hard walls cannot be used in `npt` mode.

### Soft Walls

Soft walls are enabled by providing a configuration file:

```toml
[soft_walls]
in_file = "soft_walls.txt"
```

| Parameter | Scope | Type | Description |
|-----------|-------|------|-------------|
| `in_file` | `soft_walls` | string | Soft-wall configuration file path |

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

## Generalized Born

Generalized Born parameters are configured through the `[gb]` scope:

```toml
[gb]
in_file = "gb.txt"
epsilon = 78.5
radii_offset = 0.09
radii_cutoff = 25.0
```

| Parameter | Scope | Type | Default | Description |
|-----------|-------|------|---------|-------------|
| `in_file` | `gb` | string | - | Native GB parameter file |
| `epsilon` | `gb` | float | `78.5` | Relative dielectric constant |
| `radii_offset` | `gb` | float | `0.09` | Offset subtracted from self radii |
| `radii_cutoff` | `gb` | float | `cutoff` | Cutoff used when building effective Born radii |

## LJ Soft Core

If a native soft-core LJ parameter file is provided through
`LJ_soft_core_in_file`, SPONGE also reads several top-level alchemical
parameters:

| Parameter | Scope | Type | Default | Description |
|-----------|-------|------|---------|-------------|
| `lambda_lj` | `top-level` | float | required | Soft-core LJ coupling parameter |
| `soft_core_alpha` | `top-level` | float | `0.5` | Soft-core alpha parameter |
| `soft_core_powfer` | `top-level` | float | `1.0` | Soft-core power parameter |
| `soft_core_sigma` | `top-level` | float | `3.0` | Soft-core sigma parameter |
| `soft_core_sigma_min` | `top-level` | float | `0.0` | Lower bound for the effective sigma term |

These keys are top-level in the current source, not nested under
`[LJ_soft_core]`.

## Custom Forces

| Parameter | Scope | Type | Description |
|-----------|-------|------|-------------|
| `in_file` | `pairwise_force` | string | Pairwise-force configuration file |
| `in_file` | `listed_forces` | string | Listed-force configuration file |

## Output Mapping For Periodic Molecules

| Parameter | Scope | Type | Default | Description |
|-----------|-------|------|---------|-------------|
| `force_whole_output` | `top-level` | bool | `false` | Force whole-molecule output mapping for periodic systems |
| `make_output_whole` | `top-level` | string | - | Extra connectivity edges written as `atom_i-atom_j` pairs to help rebuild whole molecules |

These keys are only meaningful when `pbc = true`.

## Solvent LJ Shortcut

| Parameter | Scope | Type | Default | Description |
|-----------|-------|------|---------|-------------|
| `solvent_LJ` | `top-level` | bool | auto | Enable the water-specific solvent LJ shortcut |

On GPU builds, if `solvent_LJ` is omitted, SPONGE may enable it automatically
for suitable water-only tail residues. Setting the key explicitly overrides that
heuristic.

## Plugins

Plugins are configured by the top-level key `plugin`, which is a whitespace-
separated list of shared-library paths. This is not currently a TOML table
interface.

## Control Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `command_only` | bool | `false` | Use command-line arguments only, skip mdin file |
| `dont_check_input` | bool | `false` | Skip unused parameter checking |
| `end_pause` | bool | `false` | Pause after simulation ends |

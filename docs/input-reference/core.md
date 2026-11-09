# Core Simulation Parameters

## Simulation Identity

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `md_name` | string | `"Default SPONGE MD Task Name"` | Simulation task name |

## Simulation Mode

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | string | **required** | Simulation mode |

`mode` options:

| Value | Description |
|-------|-------------|
| `"nve"` | Microcanonical ensemble (constant energy) |
| `"nvt"` | Canonical ensemble (constant temperature), requires thermostat |
| `"npt"` | Isothermal-isobaric ensemble, requires thermostat + barostat |
| `"minimization"` / `"min"` | Energy minimization |
| `"rerun"` | Trajectory reanalysis |

## Periodic Boundary Conditions

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pbc` | bool | `true` | Enable periodic boundary conditions |

If `pbc = false`, SPONGE switches to the no-PBC path. This cannot be used with
`npt`, is not supported in multi-process mode, and expects a very large box.

## Time and Steps

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `step_limit` | int | `1000` | Total simulation steps |
| `dt` | float | `0.001` | Time step (ps) |
| `frame_limit` | int | - | Frame limit for rerun mode |

## Force Cutoff

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cutoff` | float | `8.0` | Non-bonded interaction cutoff distance (A) |
| `skin` | float | `2.0` | Neighbor list skin distance (A) |

## Temperature and Pressure Targets

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_temperature` | float | `300.0` | Top-level target temperature (K), used in NVT/NPT |
| `target_pressure` | float | `1.0` | Top-level target pressure (bar), used in NPT |

These two keys stay at the top level even when thermostat and barostat options are
written in `[thermostat]` and `[barostat]`.

Temperature and pressure also support schedules through the top-level keys
`target_temperature_schedule_*` and `target_pressure_schedule_*`. See
[Thermostat](thermostat.md) and [Barostat](barostat.md).

## Hydrogen Mass Repartitioning

See [HMR implementation and usage](../hmr.md) for the complete algorithm,
initialization order, mass-view table and focused test commands.

HMR is disabled unless explicitly enabled. It runs once during mass loading,
before inverse masses, thermostats, constraints and initial velocities are built.
Input topology files and the in-memory original topology masses are preserved.

```toml
[hmr]
enabled = true
ratio_xh1 = 2.0       # default: one H bonded to a heavy atom
ratio_other = 2.5     # required: multiple H; AMBER example (CHARMM: 3.0)
```

These are multiplicative factors applied to each original H mass, following the
heterogeneous mass rule of [Jung et al. (2021)](https://doi.org/10.1021/acs.jctc.1c00185).
For a donor X with n attached hydrogens, each H receives
`m_H * (ratio - 1)` from X; transfers are accumulated before updating X. All
resulting real-atom masses must remain positive. This implements the mass rule;
it does not implement the paper's integrator or temperature/pressure changes.

The supported input is an ordinary, unrepartitioned, fixed-bond, all-atom
protium topology for NVE/NVT/NPT. H is identified by original mass in
`[0.9, 1.1]` u and must have exactly one bonded heavy neighbor with mass above
4 u. Explicit bond connectivity is required, including for constrained X-H
bonds; arbitrary distance constraints are not used to invent chemical bonds.
GROMACS-imported constrained bonds and SETTLE connections are already retained
by its loader. Do not use this mass heuristic for isotopic, polarizable,
coarse-grained or already-repartitioned inputs. REAXFF is rejected.

Water is always excluded in this initial interface: an isolated real-atom OHH
component, with O mass in `[15.5, 16.5]` u. Atom ordering and a SETTLE H-H edge
do not affect detection; registered virtual sites are excluded from chemical
connectivity, allowing an OHH core in four/five-site water. Unknown zero-mass
atoms or ambiguous H connectivity cause an error. Water model identity cannot
be proved from masses and connectivity alone; chemically unusual OHH components
are outside this heuristic's scope.

Constraint membership uses original masses, and constraint solvers use effective
masses. Virtual COM geometry and restraint reference COM definitions retain
original mass weights. HMR does not enable constraints, change the time step or
establish that 4 fs is accurate for a particular system. Existing constraints
should be retained and a chosen time step validated separately.

H5 dynamic/full restart records and checks an effective-mass checksum. Continue
with matching effective masses. Legacy dynamic restarts without this checksum
are accepted with HMR disabled and rejected with HMR enabled. To switch between
HMR and ordinary dynamics, load structural state and explicitly regenerate
velocities with `[initial_velocity]`; this starts a new dynamical segment.
Legacy coordinate/velocity restarts carry no such mass check. HMR never rescales
input velocities automatically. Configuring ratios while HMR is disabled, unknown
`hmr` keys and invalid ratios are errors.

## Initial Velocities

SPONGE preserves the velocities supplied by the selected input loader unless
initialization is requested explicitly:

```toml
[initial_velocity]
mode = "maxwell"
seed = 202607220001
```

`mode = "maxwell"` samples velocities at the step-zero target temperature,
removes center-of-mass motion, projects constrained systems onto the
SHAKE/SETTLE velocity manifold without moving their coordinates, and rescales
using the final system degrees of freedom. `seed` is required and must be an
integer in `0..INT64_MAX`; it is independent of thermostat seeds. This mode is
available for dynamical simulations, not minimization or rerun.

## Working Directory

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `workspace` | string | directory of mdin file | Working directory path, can be absolute or relative |

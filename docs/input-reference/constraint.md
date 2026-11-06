# Constraint Algorithm Parameters

## Constraint Mode

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `constrain_mode` | string | none (constraints disabled) | Constraint algorithm |
| `constrain_precision` | string | `"double"` | Internal constraint arithmetic: `"single"` or `"double"` |

`constrain_mode` options:

| Value | Description |
|-------|-------------|
| `"SETTLE"` | SETTLE algorithm, specialized for rigid water molecules (triangle constraints) |
| `"SHAKE"` | SHAKE algorithm, general bond length constraints |
| `"LINCS"` | Experimental LINCS bond constraints |
| `"CCMA"` | Experimental CCMA bond constraints |

The base constraint list is configured through the `[constrain]` prefix:

```toml
constrain_mode = "SHAKE"
constrain_precision = "double"

[constrain]
mass = 3.3
```

## Internal Precision

The top-level `constrain_precision` applies to SETTLE, SHAKE, LINCS and CCMA,
including SETTLE pairs/triangles extracted before another selected solver runs.
The selected type covers geometry coefficients, saved old bond vectors,
iteration work arrays, position corrections, velocity projection and constraint
virial accumulation. Invalid values fail during constraint initialization.

`double` is the default precision reference. `single` retains a lower-precision
solver for comparison and hardware-specific performance evaluation. The same
algorithm implementation is instantiated for each precision; this setting does
not change SHAKE/CCMA tolerance, LINCS order or iteration limits.

The MD state (coordinates, velocities, masses, forces and output stress) remains
float. Each constraint operation loads that state into selected-precision
working buffers and writes it back once after the complete operation. Double
constraint arithmetic therefore does not recover input precision already lost
in float state, nor make the rest of MD double precision. Explicit distances
from `constrain_in_file` and initialized shared pair coefficients are retained
as double before the selected solver loads them; distances imported through
existing float topology data retain that topology precision.

Constraint velocity projection uses a roundoff bound for its internal scalar
type. Writing final velocities back to float introduces a separate storage
error. A tolerance smaller than the attainable coordinate precision is not a
promise of that accuracy; in particular, excessively tight single-precision
CCMA iteration can accumulate velocity corrections after coordinates stop
changing. Check residuals, momentum, constraint stress and NVE energy drift for
the intended model. GPU throughput and long-time transport accuracy require
separate validation.

## SETTLE Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `settle_disable` | bool | `false` | Top-level flag that disables SETTLE entirely |

## SHAKE Parameters

| Parameter | Scope | Type | Default | Description |
|-----------|-------|------|---------|-------------|
| `iteration_numbers` | `SHAKE` | int | `25` | Maximum count for small groups; fixed count for the general path |
| `tolerance` | `SHAKE` | float | `1e-4` | Relative bond-length threshold, between 0 and 1 |
| `step_length` | `SHAKE` | float | `1.0` | SHAKE step length / damping factor |

## `[constrain]` Parameters

| Parameter | Scope | Type | Default | Description |
|-----------|-------|------|---------|-------------|
| `in_file` | `constrain` | string | - | Extra bond-constraint list file |
| `mass` | `constrain` | float | `3.3`, or `0.0` when `in_file` is set | Auto-constrain bonds involving atoms lighter than this threshold |
| `angle` | `constrain` | bool | `false` | Reserved, currently not implemented |

With [HMR](../hmr.md) enabled, `constrain.mass` selects atoms using their
original topology masses. Solver mass coefficients and velocity projection use
the effective HMR masses. HMR itself neither enables constraints nor changes
the time step. Explicit X-H bond connectivity is still required for HMR donor
identification even when those bonds are constrained.

## LINCS Parameters (experimental)

LINCS controls accuracy through expansion order and rotation corrections,
not a tolerance parameter.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `order` | int | `8` | Expansion order, 1–64 |
| `corrections` | int | `1` | Additional rotation corrections, 0–16 |

Suggested starting points; specify these explicitly to override the defaults:

| Use case | `order` | `corrections` |
|----------|---------|---------------|
| Ordinary 2 fs MD with H-containing bond constraints | 4 | 1 |
| The tested 4 fs models | 4 | 2 |
| Other large-step models, including virtual sites | Consider 6 | Compare 1 and 2 |
| NVE with emphasis on energy conservation | 4 | Compare 1 and 2 |

Check constraint errors and, for NVE, energy drift. These are starting points,
not accuracy guarantees; larger settings can amplify single-precision rounding
errors. See [GROMACS LINCS guidance](https://manual.gromacs.org/documentation/2026.1/user-guide/mdp-options.html#lincs-order).

```toml
constrain_mode = "LINCS"
[LINCS]
order = 4
corrections = 1
```

## CCMA Parameters (experimental)

CCMA uses a fixed sparse approximate inverse built from the initial geometry.
Usually only `tolerance` needs adjustment.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tolerance` | float | `1e-4` | Relative bond-length tolerance, between 0 and 1 |
| `iteration_numbers` | int | `8` | Small-group iteration cap; fixed count for larger groups, 1–1000 |
| `inverse_cutoff` | float | `0.01` | Drop inverse entries below this magnitude, 0 ≤ value < 1 |
| `inverse_radius` | int | `3` | Graph radius for local inverse construction, 1–8 |
| `inverse_max_size` | int | `128` | Maximum local inverse dimension, 2–512 |

```toml
constrain_mode = "CCMA"
[CCMA]
tolerance = 1e-5
```

## Convergence and Limitations

SHAKE/CCMA automatically stop small groups when all relative bond-length errors
meet `tolerance`; no `early_stop` setting is needed. Small groups have at most
three constraints and four atoms. If any local component exceeds these limits,
the solver uses its fixed-iteration general path. Reaching the iteration cap
or writing single-precision coordinates can leave errors above the target;
normal MD does not perform a separate final error check.

SETTLE handles eligible independent pairs and triangles first. LINCS/CCMA
support CPU and GPU execution, but coupled constraints must remain on one MPI
rank. General coupled angle constraints and redundant rigid networks are not
validated. Large-step and NVE configurations require validation for the chosen
model; short tests do not establish long-term stability.

# Atom Restrain Parameters

The coordinate restrain module applies a harmonic positional potential to a
list of atoms. It is independent of [CV restraints](collective-variables.md)
and of [bond-length constraints](constraint.md) such as SHAKE and SETTLE.
Enabling one does not enable or disable the others.

## Quick Start

For a text reference file and an isotropic force constant:

```toml
[restrain]
atom_id = "restrain_atom_id.txt"
coordinate_in_file = "restrain_ref.txt"
single_weight = 20.0
refcoord_scaling = "no"
calc_virial = true
```

Grouped TOML keys are stored as `prefix_key`. For example,
`[restrain] coordinate_in_file` and the historical flat key
`restrain_coordinate_in_file` select the same setting. Use the grouped form
for new TOML inputs.

For text input, `atom_id` enables the module. If neither `coordinate_in_file`
nor `amber_rst7` is specified, the module copies the loaded main coordinates
as its initial reference. An active positional-restraint object in a native
H5 protocol can also enable the module; see [H5 Input and Restart](#h5-input-and-restart).

## Parameters

| Parameter | Type | Behavior |
|-----------|------|----------|
| `atom_id` | path | List of zero-based atom indices; enables the text-input module |
| `coordinate_in_file` | path | Full-system TXT reference coordinates; takes precedence over `amber_rst7` during text initialization |
| `amber_rst7` | path | Full-system reference coordinates from an ASCII Amber restart |
| `single_weight` | float | Explicit isotropic force constant; when present, `weight_in_file` is not read |
| `weight_in_file` | path | One `wx wy wz` triplet per selected atom; required for text input when `single_weight` is absent |
| `refcoord_scaling` | string | Reference response to NPT box scaling: `no` (default), `all`, `com_ug`, `com_res`, or `com_mol` |
| `calc_virial` | bool | Include the restraint virial when pressure is requested; defaults to `true` |

## Atom Selection

`restrain_atom_id.txt` contains one zero-based atom index per line:

```text
0
2
```

This selects the first and third atoms in the topology/coordinate order.
There is no atom-count header. Do not use PDB serial numbers or add comments;
all indices must be valid for the loaded system. List each selected index once.

The reference coordinate file contains **all system atoms**, not just the
selected atoms. The selected indices determine which reference rows are used.
The text reader cannot verify chemical identities: keep the reference and
current topology in exactly the same atom order.

## Reference Coordinate Files

### SPONGE TXT Coordinates and Restarts

Set `coordinate_in_file` to a coordinate file whose first line is one of:

```text
N
N time
N time step
```

`N` is the total system atom count. The reader consumes the entire header,
then reads one XYZ triplet per atom. For a four-atom system, this is a complete
reference file, including a minimization/restart-style header:

```text
4 0.0000000000 4999
1.0 2.0 3.0
4.0 5.0 6.0
7.0 8.0 9.0
10.0 11.0 12.0
```

Coordinates are in angstroms. Optional box data after the N coordinate rows
is ignored by the restraint reader. The header time and step do not change
the simulation time or step, and are never interpreted as coordinates.

The atom count must match the system. Missing XYZ values, truncated or
overlong input lines, and NaN/Inf coordinates are rejected before the reference
is copied to the device. This validation does not establish that the reference
has the intended geometry or atom order.

### Amber ASCII Restart Coordinates

To use the existing Amber reference reader:

```toml
[restrain]
atom_id = "restrain_atom_id.txt"
amber_rst7 = "reference.rst7"
single_weight = 20.0
```

The reader skips the title and reads `N [time]` from the second line, then
reads the N Cartesian coordinates from the ASCII coordinate block. Use a
matching full-system restart in the same atom order. This is separate from
the top-level `amber_rst7` setting that loads the main simulation state.

The restraint reference uses the XYZ coordinates, not the restart's velocity
or box as the running system state. This change leaves the legacy Amber
reader in place; the stricter TXT diagnostics above should not be assumed to
apply to malformed Amber files. If both reference keys are present,
`coordinate_in_file` is selected during text initialization.

## Force Constants and Weight Files

For each restrained atom, the potential is

```text
E = 0.5 * (wx * dx^2 + wy * dy^2 + wz * dz^2)
```

The displacement uses the engine's periodic-boundary displacement routine.
Force constants are in kcal mol^-1 angstrom^-2, with coordinates in angstroms
and energy in kcal/mol.

### Isotropic Weights

An explicit `single_weight` sets `wx = wy = wz` for every selected atom:

```toml
[restrain]
atom_id = "restrain_atom_id.txt"
single_weight = 20.0
```

This example copies the loaded main coordinates as the reference. Setting
`single_weight = 0.0` removes the force contribution, but still initializes
the module and reads any configured reference file.

The internal scalar starts at 20.0, but **omitting `single_weight` does not
select it as an automatic default for text input**. Supply either an explicit
`single_weight` or a `weight_in_file`.

### Per-atom Anisotropic Weights

```toml
[restrain]
atom_id = "restrain_atom_id.txt"
coordinate_in_file = "restrain_ref.txt"
weight_in_file = "restrain_weight.txt"
```

For the two-atom selection above, `restrain_weight.txt` could contain:

```text
20.0 20.0 0.0
10.0 10.0 10.0
```

Each row corresponds to the same row in `atom_id`, not to a full-system atom
index. The first selected atom is restrained along x and y; its z weight is
zero. Use finite, non-negative weights, with exactly one triplet per selected
atom. These are caller requirements: the legacy weight-file reader does not
validate the row count or finite/non-negative values. If `single_weight` is
also set, it takes precedence and the weight file is not read.

## Reference Scaling in NPT

`refcoord_scaling` controls the reference when the NPT workflow scales the
box and positions. It does not change the initial reference file or the
force constant, and the reference-update routine returns without scaling
outside NPT.

| Value | Reference update |
|-------|------------------|
| `no` | Keep the reference fixed in Cartesian space |
| `all` | Apply the box-scaling transformation to every reference position |
| `com_ug` | Translate each unit group's reference positions with its scaled center of mass, preserving internal reference offsets |
| `com_res` | Apply that center-of-mass translation by residue |
| `com_mol` | Apply that center-of-mass translation by molecule |

For molecule-based scaling, for example:

```toml
[restrain]
atom_id = "restrain_atom_id.txt"
coordinate_in_file = "restrain_ref.txt"
single_weight = 20.0
refcoord_scaling = "com_mol"
calc_virial = true
```

COM modes require the corresponding grouping metadata and atomic masses.
`com_mol` requires molecule ranges; `com_res` requires residue ranges;
`com_ug` requires unit groups. If they are unavailable, the reference update
prints a `not ready` message and skips that update. Group-dependent virial
handling can also fall back to the `no` form when its mapping is unavailable.
Check that the selected grouping is actually loaded.

These modes are also reached through NPT Monte Carlo box changes. The engine
does not enforce a universal requirement to choose a COM mode for that
barostat; select the scaling behavior as part of the intended simulation
protocol.

## Virial Contribution

`calc_virial = true` includes the restraint contribution when the workflow
requests pressure/virial. Its force arm depends on `refcoord_scaling`.
Setting it to `false` omits that contribution; restraint energy and forces
remain active. It does not disable the restraint or bond-length constraints.

## H5 Input and Restart

H5 files use structured input bindings, not `coordinate_in_file`. See the
[H5 bundle bindings and restart policies](io.md#h5-bundle-input-files).

A native H5 protocol positional-restraint object supplies atom indices,
weights/defaults and optional reference coordinates. Its reference array is
stored in the restart at
`/parameters/restart/references/restraint/<name>/coordinate` and must have
shape `[system_atom_count,3]` with finite values. If no native reference is
present, initialization copies the loaded main coordinates.

For native initialization, explicit `atom_id`, `weight_in_file`,
`coordinate_in_file`, or `amber_rst7` selects the text-input path instead.
Explicit `single_weight`, `refcoord_scaling`, and `calc_virial` can override
native defaults. Without an explicit scalar, native input uses its
`single_weight_default` when present, otherwise its per-atom weight array.

Continuation has an additional stage: `input.h5.restart.load = "protocol"`
or `"full"` restores the checkpoint's compatible, named restraint-reference
state **after module initialization**. That restored reference replaces the
initial reference, including one initialized from an explicit text file.
`structural` and `dynamic` do not perform that protocol-state restore.

Legacy sidecars embedded in an H5 restart can also inject text bindings.
For `protocol`/`full`, these checkpoint bindings replace same-named explicit
commands before initialization. An embedded `restrain_coordinate_in_file`
sidecar is also materialized for `structural`/`dynamic`. Thus, an explicit
text reference should not be assumed to override all checkpoint contents;
check the checkpoint bindings and load policy when changing protocols.

H5 input identity and atom-order compatibility checks remain in effect.
Passing a format check or a short runtime test does not establish equilibration
or production stability.

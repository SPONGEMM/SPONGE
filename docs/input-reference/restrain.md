# Atom Restrain Parameters

SPONGE has a coordinate restrain module that is separate from the CV-side
`restrain` described in [collective-variables.md](collective-variables.md).

This module is initialized directly in the main MD workflow and restrains a list
of atoms to reference coordinates.

## Input Style

This module currently uses historical flat prefix keys such as
`restrain_atom_id` and `restrain_single_weight`.

The parser stores grouped TOML keys as `prefix_key`, so the following TOML form
is equivalent and is preferred for new documentation:

```toml
[restrain]
atom_id = "restrain_atom_id.txt"
coordinate_in_file = "restrain_ref.txt"
single_weight = 20.0
refcoord_scaling = "com_mol"
calc_virial = true
```

## Minimal Example

```toml
[restrain]
atom_id = "restrain_atom_id.txt"
coordinate_in_file = "restrain_ref.txt"
single_weight = 20.0
```

If neither `coordinate_in_file` nor `amber_rst7` is provided, SPONGE copies the
current input coordinates as the restrain reference.

## Parameters

| Parameter | Scope | Type | Default | Description |
|-----------|-------|------|---------|-------------|
| `atom_id` | `restrain` | string | - | Required atom-list file; enables the module |
| `coordinate_in_file` | `restrain` | string | current input coordinates | Reference coordinate file |
| `amber_rst7` | `restrain` | string | - | Reference coordinates loaded from an AMBER rst7 file |
| `single_weight` | `restrain` | float | `20.0` | Isotropic restrain weight applied to every restrained atom |
| `weight_in_file` | `restrain` | string | - | Per-atom anisotropic restrain weights; used when `single_weight` is not set |
| `refcoord_scaling` | `restrain` | string | `"no"` | How reference coordinates are scaled during box updates |
| `calc_virial` | `restrain` | bool | `true` | Whether to accumulate restrain virial |

## `refcoord_scaling`

Supported values are:

| Value | Meaning |
|-------|---------|
| `"no"` | Do not scale the reference |
| `"all"` | Scale all reference coordinates with the box |
| `"com_ug"` | Scale by unit-group center of mass |
| `"com_res"` | Scale by residue center of mass |
| `"com_mol"` | Scale by molecule center of mass |

## Weight Input Modes

### Isotropic Weight

```toml
[restrain]
atom_id = "restrain_atom_id.txt"
single_weight = 20.0
```

This applies one scalar force constant to every restrained atom.

### Per-atom Anisotropic Weights

```toml
[restrain]
atom_id = "restrain_atom_id.txt"
weight_in_file = "restrain_weight.txt"
```

`weight_in_file` must contain one `wx wy wz` triplet per restrained atom.

## Notes

- The module is enabled only when `restrain_atom_id` or `atom_id` in the
  `restrain` scope is present.
- `coordinate_in_file` takes precedence over `amber_rst7`.
- If `single_weight` is not provided, `weight_in_file` is expected.
- This module is independent of CV `restrain`; both can be used in the same run
  if needed.

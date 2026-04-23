# Input/Output Parameters

SPONGE has three mutually exclusive structure/topology input families:

- native text inputs loaded by Xponge
- AMBER inputs loaded from `amber_parm7` / `amber_rst7`
- GROMACS inputs loaded from `gromacs_top` / `gromacs_gro`

If either GROMACS key exists, SPONGE uses the GROMACS loader. Otherwise, if
either AMBER key exists, SPONGE uses the AMBER loader. If neither family is
selected, SPONGE falls back to native inputs.

## Input Files

### Common Prefix

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `default_in_file_prefix` | string | - | Input filename prefix, auto-matches `<prefix>_coordinate.txt` etc. |
| `default_out_file_prefix` | string | - | Output filename prefix |

Setting `default_in_file_prefix = "WAT"` causes SPONGE to look for:
- `WAT_coordinate.txt` — coordinates
- `WAT_mass.txt` — masses
- `WAT_charge.txt` — charges
- `WAT_LJ.txt` — LJ parameters
- `WAT_bond.txt` — bonds
- `WAT_exclude.txt` — exclusion list
- etc.

### Native Input Files

The native loader reads a family of `<module>_in_file` keys. The most common
ones are:

| Parameter | Type | Description |
|-----------|------|-------------|
| `coordinate_in_file` | string | Coordinate file path |
| `velocity_in_file` | string | Velocity file path |
| `mass_in_file` | string | Mass file path |
| `charge_in_file` | string | Charge file path |
| `residue_in_file` | string | Residue membership file |
| `exclude_in_file` | string | Exclusion list file |
| `bond_in_file` | string | Bond parameter file |
| `angle_in_file` | string | Angle parameter file |
| `dihedral_in_file` | string | Dihedral parameter file |
| `improper_dihedral_in_file` | string | Improper dihedral file |
| `cmap_in_file` | string | CMAP file |
| `lj_in_file` | string | Lennard-Jones parameter file |
| `LJ_soft_core_in_file` | string | Soft-core Lennard-Jones parameter file |
| `nb14_in_file` | string | 1-4 interaction file |
| `nb14_extra_in_file` | string | Extra 1-4 interaction file |
| `urey_bradley_in_file` | string | Urey-Bradley file |
| `virtual_atom_in_file` | string | Native virtual-atom definition file |

Some modules add their own native files, for example `lj_soft_in_file`,
`gb_in_file`, and module-specific `in_file` keys documented on their
corresponding pages.

### External Format Import

| Parameter | Type | Description |
|-----------|------|-------------|
| `amber_parm7` | string | AMBER parm7 topology/parameter file |
| `amber_rst7` | string | AMBER rst7 coordinate/velocity file |
| `gromacs_gro` | string | GROMACS .gro coordinate file |
| `gromacs_top` | string | GROMACS .top topology file |
| `gromacs_include_dir` | string list | Extra include directories used when reading `.top` |
| `gromacs_define` | string list | Extra preprocessor defines used when reading `.top` |

## Output Files

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mdout` | string | controller default or `default_out_file_prefix + ".out"` | Standard output file (energy, temperature, etc.) |
| `mdinfo` | string | controller default or `default_out_file_prefix + ".info"` | Simulation info/log file |
| `crd` | string | - | Coordinate trajectory file (binary) |
| `vel` | string | - | Velocity trajectory file (binary) |
| `frc` | string | - | Force trajectory file (binary) |
| `box` | string | - | Box information trajectory file |
| `rst` | string | `SPONGE` or `default_out_file_prefix` | Restart filename prefix |

## Output Frequency Control

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `write_information_interval` | int | `1000` | mdinfo/mdout write interval (steps) |
| `write_mdout_interval` | int | `1000` | mdout write interval |
| `write_trajectory_interval` | int | same as `write_information_interval` | Trajectory write interval |
| `write_restart_file_interval` | int | `step_limit` | Restart file write interval |
| `max_restart_export_count` | int | `1` | Maximum number of restart files to keep in rotation |
| `buffer_frame` | int | `10` | File buffer frame count (affects I/O performance) |

## Output Content Control

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `print_zeroth_frame` | bool | `false` | Whether to output step 0 frame |
| `print_pressure` | bool | `false` | Whether to append pressure and virial terms to `mdout` |

`mdout` and `mdinfo` are controller-managed output files. Trajectory-related
files are created only when the corresponding key exists or when the default
coordinate/box trajectories are enabled by `write_trajectory_interval`.

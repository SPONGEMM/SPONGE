# Input/Output Parameters

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

### Individual Input Files

| Parameter | Type | Description |
|-----------|------|-------------|
| `coordinate_in_file` | string | Coordinate file path |
| `velocity_in_file` | string | Velocity file path |
| `mass_in_file` | string | Mass file path |
| `charge_in_file` | string | Charge file path |
| `atom_in_file` | string | Atom information file |
| `atom_type_in_file` | string | Atom type file |
| `edge_in_file` | string | Bond/edge information file |
| `angle_in_file` | string | Angle information file |
| `dihedral_in_file` | string | Dihedral information file |
| `atom_numbers` | int | Total atom count (can be auto-detected from files) |

### External Format Import

| Parameter | Type | Description |
|-----------|------|-------------|
| `amber_parm7` | string | AMBER parm7 topology/parameter file |
| `amber_rst7` | string | AMBER rst7 coordinate/velocity file |
| `gromacs_gro` | string | GROMACS .gro coordinate file |
| `gromacs_top` | string | GROMACS .top topology file |

## Output Files

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mdout` | string | `"mdout.txt"` | Standard output file (energy, temperature, etc.) |
| `mdinfo` | string | `"mdinfo.txt"` | Simulation info/log file |
| `crd` | string | - | Coordinate trajectory file (binary) |
| `vel` | string | - | Velocity trajectory file (binary) |
| `frc` | string | - | Force trajectory file (binary) |
| `box` | string | - | Box information trajectory file |
| `rst` | string | - | Restart file |

## Output Frequency Control

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `write_information_interval` | int | `1000` | mdinfo/mdout write interval (steps) |
| `write_mdout_interval` | int | `1000` | mdout write interval |
| `write_trajectory_interval` | int | same as `write_information_interval` | Trajectory write interval |
| `write_restart_file_interval` | int | `step_limit` | Restart file write interval |
| `max_restart_export_count` | int | - | Maximum number of restart files to keep |
| `buffer_frame` | int | `10` | File buffer frame count (affects I/O performance) |

## Output Content Control

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `print_zeroth_frame` | int | `0` | Whether to output step 0 frame (1 = yes) |
| `print_pressure` | int | `0` | Whether to output pressure |
| `print_detail` | int | `0` | Detailed output level |

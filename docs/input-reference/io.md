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
| `amber_rst7` | string | AMBER rst7 coordinates, optional velocities, and six box values (lengths and angles); the optional header time does not indicate velocity presence |
| `gromacs_gro` | string | GROMACS .gro coordinate file |
| `gromacs_top` | string | GROMACS .top topology file |
| `gromacs_include_dir` | string list | Extra include directories used when reading `.top` |
| `gromacs_define` | string list | Extra preprocessor defines used when reading `.top` |

## H5 Bundle Input Files

SPONGE keeps large input data and restart state in HDF5 containers while
retaining `mdin.spg.toml` as the editable binding and launch-policy file.

Use the following TOML tables to bind topology, protocol, and restart bundles:

```toml
[input.h5.topology]
path = "topologies/protein.topology.spgt.h5"

[input.h5.protocol]
path = "protocols/metadyn.protocol.spgp.h5"

[input.h5.restart]
path = "runs/prod_0007.restart.spgr.h5"
load = "structural"
```

For rerun input:

```toml
mode = "rerun"
rerun_frame_limit = 1000
rerun_start = 0
rerun_strip = 0

[input.h5.topology]
path = "topologies/protein.topology.spgt.h5"

[input.h5.protocol]
path = "protocols/analysis.protocol.spgp.h5"

[input.h5.trajectory]
path = "runs/prod.spg.h5md"
particle_stream = "all"
```

### Legacy Input Mapping

| Legacy key | H5 bundle replacement |
|------------|-----------------------|
| `coordinate_in_file` | `input.h5.restart.path`, structural position state |
| `velocity_in_file` | `input.h5.restart.path`, structural velocity state |
| `amber_rst7` / `rst7` | `input.h5.restart.path` |
| `crd` | `input.h5.trajectory.path`, position frames |
| `box` | `input.h5.trajectory.path`, box edge frames |
| `vel` | `input.h5.trajectory.path`, optional velocity frames |
| `frame_limit` | Keep as a top-level flat key, or use `rerun_frame_limit` |
| `rerun_start` | Unchanged top-level flat key |
| `rerun_strip` | Unchanged top-level flat key |
| `rerun_need_box_update` | Unchanged top-level flat key |

H5 and legacy inputs for the same role are mutually exclusive. For example,
`input_h5_restart_path` cannot be combined with `coordinate_in_file`,
`velocity_in_file`, or `rst7`; `input_h5_trajectory_path` cannot be combined
with rerun `crd`, `box`, or `vel`.

For structural restart input, `/particles/all/velocity/value` is optional. If
it is absent, SPONGE initializes every atomic velocity component to zero before
copying the launch state to the device. Thermostats may introduce stochastic
motion during later integration steps, but input assembly does not sample a
Maxwell-Boltzmann distribution.

### Restart Load Policy

`input.h5.restart.load = "structural"` restores coordinates, velocity, box,
step, and time. `dynamic` additionally restores compatible integrator,
thermostat, barostat, and RNG state; `protocol` restores compatible SITS,
MetaD, restraint-reference, and CV-reference continuation state; `full`
combines both. Each enabled module validates ownership and schema compatibility
before applying state. `custom` is reserved until explicit component-list keys
are defined.

## H5 Bundle Output Files

New SPONGE bundle output should use structured H5 output settings:

```toml
[output.h5.trajectory]
path = "prod.spg.h5md"
vds = true
chunk_size = 20
repair_policy = "strict"

[output.h5.restart]
path = "prod.spgr.h5"
# Required only for raw-input launches that request H5 restart output:
topology_hash = "sha256:..."
atom_order_hash = "sha256:..."
# Optional when no canonical protocol bundle exists:
protocol_hash = "sha256:..."

[output.h5.observable]
path = "prod.obs.spg.h5md"
```

With the current TOML flattening parser, these keys are visible as:

| Flattened key | Type | Default | Description |
|---------------|------|---------|-------------|
| `output_h5_trajectory_path` | string | - | Canonical trajectory H5MD output path; recommended suffix `*.spg.h5md` |
| `output_h5_trajectory_vds` | bool | `false` | Use chunked H5MD shards plus HDF5 VDS wrapper |
| `output_h5_trajectory_chunk_size` | int | `20` | VDS file-level shard size in trajectory frames |
| `output_h5_trajectory_repair_policy` | string | `strict` | VDS finalize policy: `strict` or `complete_prefix` |
| `output_h5_restart_path` | string | - | Canonical restart H5 output path; recommended suffix `*.spgr.h5` |
| `output_h5_restart_topology_hash` | string | derived from H5 topology | Canonical topology lineage required for raw-input H5 restart output |
| `output_h5_restart_atom_order_hash` | string | derived from H5 topology | Canonical atom-order lineage required for raw-input H5 restart output |
| `output_h5_restart_protocol_hash` | string | derived/optional | Canonical producer protocol lineage |
| `output_h5_observable_path` | string | - | Optional observable-only H5MD output path; recommended suffix `*.obs.spg.h5md` |

`output_h5_trajectory_chunk_size` is only meaningful when
`output_h5_trajectory_vds = true`. It is not an HDF5 dataset internal chunk
shape, and it does not change `write_trajectory_interval`.

The observable-only H5MD file contains `/h5md`, `/observables`, and
`/parameters`, but no `/particles` trajectory fields.

An H5 restart is always a strict `sponge.input.v2` launch artifact. When
`input_h5_topology_path` is bound, SPONGE derives and verifies its lineage from
the topology and optional protocol bundles. A raw-input launch must provide
`topology_hash` and `atom_order_hash` under `[output.h5.restart]`; SPONGE fails
at initialization instead of writing a restart with missing lineage.

Shard directories are writer-internal and are derived from
`output_h5_trajectory_path`; they are not configurable mdin fields.

`output_h5_trajectory_repair_policy = "complete_prefix"` is only valid with
`output_h5_trajectory_vds = true`. It allows explicit finalization from the
complete contiguous shard prefix and does not delete orphan shard files.

## Legacy Output Files

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mdout` | string | controller default or `default_out_file_prefix + ".out"` | Legacy scalar output file |
| `mdinfo` | string | controller default or `default_out_file_prefix + ".info"` | Legacy simulation info/log file |
| `crd` | string | - | Legacy coordinate trajectory file (binary), or rerun trajectory input |
| `vel` | string | - | Legacy velocity trajectory file (binary), or rerun velocity input |
| `frc` | string | - | Legacy force trajectory file (binary) |
| `box` | string | - | Legacy box information trajectory file, or rerun box input |
| `rst` | string | `SPONGE` or `default_out_file_prefix` | Legacy restart filename prefix |
| `trajectory_format` | string | `dat` | Coordinate trajectory format: `dat` or `xtc` |

If any canonical H5 output is enabled, legacy output files are disabled by
default. Legacy files are written only when their legacy path keys are
explicitly set. `trajectory_format` is not a legacy path key: asking for `xtc`
is an explicit format request, so it is not suppressed by the H5 sidecar policy
and the XTC file is opened even when canonical H5 output is active.

### GROMACS XTC Trajectories

Setting `trajectory_format = "xtc"` replaces the binary `mdcrd.dat` coordinate
trajectory with a GROMACS XTC file. Values are case-insensitive, so `"XTC"` is
accepted. The two formats are mutually exclusive: only one coordinate
trajectory is written per run. The legacy box text stream (`mdbox.txt`) and the
`vel` / `frc` streams are unaffected, because XTC carries positions and the box
only.

```toml
trajectory_format = "xtc"

[xtc]
path = "prod.xtc"
precision = 1000.0
flush = true
buffer_frames = 10
frame_index = true
append = false
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | string | `mdcrd.xtc` or `<default_out_file_prefix>.xtc` | XTC output path |
| `precision` | float | `1000.0` | Lossy precision; `1000.0` means 0.001 nm (0.01 Å) |
| `flush` | bool | `true` | Flush completed frames to the OS as they are written |
| `buffer_frames` | int | `10` | Frame-aligned stdio buffer size in whole frames; `0` or less falls back to a single-frame buffer |
| `frame_index` | bool | `false` | Also write `<path>.idx` with per-frame end offsets |
| `append` | bool | `false` | Append to an existing trajectory instead of truncating |

The `[xtc]` table is pure sugar: it flattens to `xtc_path`, `xtc_precision`,
`xtc_flush`, `xtc_buffer_frames`, `xtc_frame_index` and `xtc_append`, and the
flat spellings are equivalent. Those are the keys the loader reads.

Notes:

- **Schedule.** XTC frames follow `write_trajectory_interval` and are emitted in
  the same output slot as the legacy coordinate stream, so switching formats
  does not change which steps are recorded. The frame step is `sys.steps` and
  the frame time is `sys.Get_Current_Time()` in ps.
- **Units.** SPONGE works in Ångström; XTC is specified in nanometre. Every
  coordinate and box edge is scaled by 0.1 on write. This is not configurable:
  the container carries no unit marker, so a different factor would silently
  produce a trajectory that every other tool misreads.
- **Box.** Each frame carries the current cell as a lower-triangular matrix in
  nm, so the deformed triclinic cell of an NPT run is recorded per frame.
- **Precision.** XTC is lossy. `precision = 1000.0` quantizes coordinates to
  0.001 nm (0.01 Å), which is the GROMACS convention. xdrfile stores
  `coordinate_nm * precision` in a 32-bit integer, so the scaled coordinate has
  to stay inside that grid: a precision that cannot represent the first frame is
  rejected while that frame is being written, before any frame reaches the file,
  instead of silently corrupting coordinates.
- **Interruption safety.** An XTC frame is self-describing and the container has
  no trailer, so a file ending on a frame boundary is always readable. With
  `flush = true` and a frame-aligned `buffer_frames`, a frame never straddles a
  buffer boundary: an interrupted run leaves every completed frame on disk and
  drops at most the frame being written. `frame_index = true` additionally
  stores one 8-byte native-endian end offset per completed frame in
  `<path>.idx`, flushed as each frame completes, which allows exact truncation
  after an abrupt kill: truncate the trajectory to the last recorded offset.
- **Resumed runs.** With `append = true` the writer continues an existing file
  instead of truncating it, after verifying that its atom count matches the
  current system. The `.idx` sidecar is appended as well.
- **Excluded cases.** `trajectory_format = "xtc"` is rejected in `rerun` mode,
  and setting `crd` alongside it is ignored with a warning.
- **Build requirement.** The XTC writer is compiled only when the
  `SPONGE_WITH_XTC` CMake option is on (the default). A build without it
  rejects `trajectory_format = "xtc"` at initialization instead of silently
  falling back to the binary format.
- **Multi-replica runs.** XTC is a single-stream container and cannot represent
  replicas; use one file per replica.
- **Read-back.** The output is plain GROMACS XTC, so VMD, PyMOL, MDAnalysis,
  `gmx trjconv` and `gmx check` read it without conversion.

## Output Frequency Control

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `write_information_interval` | int | `1000` | mdinfo/mdout write interval (steps) |
| `write_mdout_interval` | int | `1000` | mdout write interval |
| `write_trajectory_interval` | int | same as `write_information_interval` | Trajectory write interval |
| `write_restart_file_interval` | int | `step_limit` | Restart file write interval |
| `max_restart_export_count` | int | `1` | Maximum number of restart files to keep in rotation |
| `buffer_frame` | int | `10` | File buffer frame count (affects I/O performance) |

For TOML mdin files, `[write.interval] information`, `trajectory`, `mdout`,
and `restart`/`restart_file` are accepted aliases for the corresponding
`write_*_interval` keys.

`buffer_frame` sizes the stdio buffer of the legacy binary streams only. The XTC
writer has its own `[xtc] buffer_frames`, which is expressed in whole frames of
the compressed format and has no effect on `mdcrd.dat`.

## Output Content Control

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `print_zeroth_frame` | bool | `false` | Whether to output step 0 frame |
| `print_pressure` | bool | `false` | Whether to append pressure and virial terms to `mdout` |

`mdout` and `mdinfo` are controller-managed output files. Trajectory-related
files are created only when the corresponding key exists or when the default
coordinate/box trajectories are enabled by `write_trajectory_interval`. In H5
bundle mode, their canonical data owners are the H5 output files rather than
these legacy sidecars.

# Running a short-range model with PRIPS

SPONGE owns MD, domain decomposition, neighbors and communication. Each particle
(PP) rank runs one model replica; an external adapter supplies the model-specific
calculation. Obtain a deployment object and its matching external model package
from the model maintainer before configuring a run. See the
[developer contract](LOCAL_MP.md) to implement an adapter and the
[maintainer design guide](../../docs/local-mp-design.md) for runtime internals.

## Requirements and current limits

The native `[mlip]` entry currently requires Linux, a CUDA+MPI SPONGE build,
NCCL, Python with development headers, and a compatible CUDA PyTorch
installation. CMake must find the repository's normal SPONGE build
dependencies as well. PRIPS uses SPONGE's `D_MPI` device transport (NCCL for this
CUDA backend); it does not create a `torch.distributed` process group. MPI must
match the build and launch environment on every node.

The current graph supports fully periodic orthogonal cells and at most 16 PP
ranks. Omit all three `DOM_DEC` splits to let SPONGE choose a domain partition.
To override it, provide `split_nx`, `split_ny` and `split_nz` together as positive
integers whose product equals the PP rank count. Native geometry/domain-size
admission checks still apply; increasing rank count does not guarantee the system fits.
The current hidden-feature exchange operates on owned atoms; models requiring
directed-edge hidden-state exchange need an additional protocol extension.
The native edge-force assembly service does not supply that exchange.
Long-range models and iterative services are refused. Local-MP with the MC
barostat is refused at startup. Enabling the SITS module together with local-MP
is also refused before force evaluation: its enhanced-energy and virial paths
are not integrated with the model outputs. This covers full-system ITS,
selective SITS and all modes of that module, including observation, empirical,
AMD and GaMD. Production-weight long MD, NPT, mixed classical/
ML force qualification and performance/scaling are not established by the
included tests. Qualify each deployment against an independent model reference
and check the intended force composition before production use; this guide
describes the input contract, not a production validation result.

Model adapters are maintained outside SPONGE. Obtain the adapter source and
matching dependencies and deployment exporter from its maintainer. Installing
the model library alone does not supply a PRIPS adapter or deployment exporter.
SPONGE does not choose models by name or install their packages.

## Build and install

Run from the repository root with the intended Python environment active. Build
SPONGE with tests **off**: enabling tests also enables HDF5 fault-injection code
in the executable. Enable the bundled native entry explicitly:

```bash
cmake -S . -B build-local-mp -G Ninja \
  -DPARALLEL=cuda -DMPI=ON -DSPONGE_BUILD_TESTS=OFF \
  -DSPONGE_ENABLE_MLIP=ON \
  -DPython3_EXECUTABLE="$(command -v python)" \
  -DCMAKE_INSTALL_PREFIX="$HOME/.local"
cmake --build build-local-mp --target SPONGE --parallel 4
cmake --install build-local-mp
```

This builds the existing PRIPS bridge and installs its private Python package
under `<libdir>/SPONGE/prips`. The executable finds that bridge relative to its
own installed location; the build-tree executable also finds its matching
bridge. Keep the executable and private package together when relocating an
installation. The bridge links the Python selected at CMake configure time, and native
initialization uses that interpreter path to locate its standard library and
packages. No `PYTHONHOME` setting is required. Activating a different Python
later does not switch that dependency. Python,
CUDA PyTorch and external model packages must remain available in the selected
environment on every rank. No separate
PRIPS wheel or source-tree `PYTHONPATH` is needed for this native entry.
`SPONGE_ENABLE_MLIP` defaults to OFF; ordinary builds do not gain a Python
dependency. The advanced standalone PRIPS build remains available below.

The default build uses `-march=native` for host C++ code, including code in a
CUDA executable. A binary built on a newer CPU can therefore fail with an
illegal instruction on an older compute node before model initialization.
Build on a CPU whose instruction set is supported by every intended node and
validate the executable on each node class. Selecting a CUDA architecture alone
does not make the host code portable.

## Configure a native deployment

Start from a SPONGE input with coordinates, masses, velocities, units and
integration parameters appropriate to the model. Add:

```toml
[mlip]
model = "models/deployment.pt"
types = "system.types.json"
```

This is an input fragment, not a complete physical system. Both paths are
required. Relative paths resolve against the directory containing `mdin`, even
when `workspace` points elsewhere. Absolute paths are also accepted. An ordinary
pure run needs no adapter name, Python entry file, plugin path, `device`, or
`PM.MPI_size` setting. Native `[mlip]` cannot be combined with `plugin`, `py`,
or `SPONGE_PRIPS_PY`.

`model` contains an external `LocalMPDeployment` object exported for this API,
not a raw training checkpoint or a LAMMPS deployment pickle. Its Python class
and model dependencies must be importable. Load only trusted deployment files:
Torch deserializes Python objects. The external exporter owns model/head,
precision, acceleration and supported-species choices; follow its instructions.

`types` is a JSON sidecar in the exact global SPONGE coordinate/topology order:

```json
{"type_names": ["Si", "O"], "atom_types": [0, 1, 0]}
```

Here `type_names` contains the model's explicit labels and `atom_types` indexes
that list; these indices are not atomic numbers. Names must be unique and
nonempty, and indices must be integers in range, with one entry per atom.
Export labels from explicit assignments **after** any topology exporter
reordering. Do not infer elements from masses, LJ types or atom names. If source
labels differ from model labels, explicitly map them when exporting the sidecar;
native `[mlip]` has no `type_map` table. The advanced Python helper supports a
mapping as described in the [deployment contract](LOCAL_MP.md).

`mode` defaults to `pure`: the model supplies the interatomic potential.
Real classical force-field parameters or nonzero charges are rejected rather
than silently ignored. Prepare an input without those terms. Explicitly
configured constraints, restraints, walls and supported CV biases remain active.
Pure mode defaults to `PM.MPI_size = 0`, so all ranks are PP ranks. Multi-rank
CV calculations retain the existing requirement for one dedicated PM/CV rank:
set `MPI_size = 1` in `[PM]` explicitly when configuring those CVs. This reservation
is never automatic, and pure mode rejects `PM.MPI_size = 1` without a CV
consumer. Single-rank CV does not require a PM rank.

To deliberately add the model to classical interactions, use:

```toml
[mlip]
model = "models/deployment.pt"
types = "system.types.json"
mode = "hybrid"
```

Hybrid mode keeps the existing classical force and PM configuration. The user
must establish that adding the two potentials is physically appropriate.
Neither mode changes the adapter's model graph cutoff/skin or unit factors.

Launch the prepared input with the installed executable:

```bash
mpiexec -n 1 "$HOME/.local/bin/SPONGE" -mdin mdin.spg.toml
```

Check process exit status and compare energy, force and virial with an
independent model reference before production MD. Output file presence alone
is not numerical validation. The [validation guide](LOCAL_MP_VALIDATION.md)
describes generic regressions and external-adapter qualification.

## More than one rank or node

Each MPI rank needs its own GPU, including particle-mesh (PM) ranks: the device
communicator spans all ranks although only PP ranks evaluate the model.
For native `[mlip]`, omitting `device` selects the MPI shared-memory local rank
when several GPUs are visible, or visible device 0 when exactly one is visible.
A local rank outside the visible-device range is an error; there is no modulo
wrapping. This rule does not prove that processes see different physical GPUs:
verify scheduler placement and visibility in the allocated environment.

Explicit `device` always wins. It accepts an integer applied to every rank or
an integer array with one visible CUDA device index per **global MPI rank**.
For two ranks on one node with both GPUs visible, `device = [0, 1]` is an explicit
mapping. With a distinct scheduler-assigned GPU exposed to each process,
`device = 0` selects each process's visible GPU. The advanced `plugin` + `py`
entry retains its legacy default of global rank, so specify devices explicitly
for multi-node use. Leave domain splits unset for automatic partitioning.
In an existing two-node allocation, an Open MPI example is:

```bash
mpiexec -n 2 --map-by ppr:1:node \
  "$HOME/.local/bin/SPONGE" -mdin mdin.spg.toml
```

Use your MPI implementation's equivalent placement flags. All ranks need access
to the same input/plugin/model files and Python environment; use a shared working
directory. Verify hostname, rank placement and GPU visibility in the allocated
environment before choosing the device list. For four processes per node on
two nodes, with all four local GPUs visible, the mapping is
`device = [0, 1, 2, 3, 0, 1, 2, 3]` **only if ranks 0–3 and 4–7 occupy the
respective nodes**. Automatic partitioning still requires a system large enough
for the chosen PP rank count and model halo.

## Advanced Python entry

Existing `plugin` + `py` usage remains additive and does not use the native pure
policy. To build a separate bridge from the same checkout as SPONGE:

```bash
cmake -S plugins/prips -B build-prips -G Ninja \
  -DPython3_EXECUTABLE="$(command -v python)"
cmake --build build-prips --target _prips --parallel 4
export PYTHONPATH="$PWD/plugins/prips${PYTHONPATH:+:$PYTHONPATH}"
```

Add this fragment to a prepared input, replacing both absolute paths:

```toml
plugin = "/absolute/path/to/SPONGE/build-prips/_prips.so"
py = "/absolute/path/to/external/adapter_entry.py"
device = 0

[PM]
MPI_size = 0
```

Keep the Python package available on every rank. The separate wheel install
remains supported. `py` is a Python entry file, not a checkpoint or model name;
`SPONGE_PRIPS_PY` is its fallback when `py` is absent. Use absolute plugin paths
or explicit relative paths such as `./build-prips/_prips.so`. Legacy plugin/py
paths resolve from the process working directory, not the mdin directory.
The entry exports `local_mp_adapter` and `atom_types` in global atom order, as
defined in the [adapter contract](LOCAL_MP.md). No `After_Initial` callback is
required. The Python entry owns loading and any explicit type mapping.

For a model-only advanced run, remove unwanted classical terms and explicitly
set `PM.MPI_size = 0`; this path does not apply native pure-mode admission.
Input `cutoff`/`skin` still control the classical table; the adapter separately
declares the model graph parameters. An explicit whole-run
`uses_classic_neighbor_list = False` declaration can permit skipping unused
classical neighbors; it does not disable classical forces. Unknown consumers
remain conservative. Native deployments can supply the same explicit
declaration on the adapter, as detailed in the contract.

## Common setup errors

- `[mlip]` unsupported or private PRIPS package missing: rebuild and install
  with `SPONGE_ENABLE_MLIP=ON`, preserving the installed executable/package layout.
- Pure mode rejects classical parameters or charges: prepare a model-only
  input, or deliberately choose hybrid mode; do not hide the conflicting terms.
- CUDA/MPI or NCCL unavailable: correct the build/runtime dependencies; CPU-only
  adapter tests do not establish a supported native CPU local-MP deployment.
- Model preparation failure or unsupported atom type: check the adapter error
  and the model-supported types. Use the same model, configuration and global
  type ordering on every PP rank; SPONGE checks communication declarations and
  preparation success, not model-content equality.
- Missing native edge-output service: build host and PRIPS from the matching
  checkout; the production edge-gradient path has no Python scatter fallback.
- Invalid geometry, domain split or MC barostat: choose a supported setup; do not
  suppress the admission check.
- MPI startup or device assignment failure: check allocation, MPI compatibility,
  shared paths and per-process device visibility before interpreting model results.

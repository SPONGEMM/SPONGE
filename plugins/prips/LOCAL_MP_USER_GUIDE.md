# Running a short-range model with PRIPS

SPONGE owns MD, domain decomposition, neighbors and communication. Each particle
(PP) rank runs one model replica; an external adapter supplies the model-specific
calculation. Obtain an adapter and model from its maintainer before configuring
a run. See the [developer contract](LOCAL_MP.md) to implement an adapter and
the [maintainer design guide](../../docs/local-mp-design.md) for runtime internals.

## Requirements and current limits

Use a CUDA+MPI SPONGE build, NCCL, Python with development headers, and a compatible
CUDA PyTorch installation. CMake must find the repository's normal SPONGE build
dependencies as well. PRIPS uses SPONGE's `D_MPI` device transport (NCCL for this
CUDA backend); it does not create a `torch.distributed` process group. MPI must
match the build and launch environment on every node.

The current graph supports fully periodic orthogonal cells and at most 16 PP
ranks. For multiple PP ranks, all three domain splits must be positive integers
whose product equals the PP rank count. Native geometry/domain-size admission
checks still apply; increasing rank count does not guarantee the system fits.
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
included tests.

Model adapters are maintained outside SPONGE. Obtain the adapter source and
matching dependencies from its maintainer, then follow that adapter's
supported-model and checkpoint instructions. Installing the model
library alone does not supply a PRIPS adapter.

## Production build

Run from the repository root with the intended Python environment active. Build
SPONGE with tests **off**: enabling tests also enables HDF5 fault-injection code
in the SPONGE executable. PRIPS has its own standalone CMake target; enabling
SPONGE tests is not necessary to build it.

```bash
cmake -S . -B build-local-mp -G Ninja \
  -DPARALLEL=cuda -DMPI=ON -DSPONGE_BUILD_TESTS=OFF
cmake --build build-local-mp --target SPONGE --parallel 4
cmake -S plugins/prips -B build-prips -G Ninja \
  -DPython3_EXECUTABLE="$(command -v python)"
cmake --build build-prips --target _prips --parallel 4
export PYTHONPATH="$PWD/plugins/prips${PYTHONPATH:+:$PYTHONPATH}"
export SPONGE_LOCAL_MP_ROOT="$PWD"
```

The plugin file is `build-prips/_prips.so`. Keep the source Python package on
`PYTHONPATH` for the commands below, on every rank. This recipe uses the build
artifact directly; it does not install a Python wheel. Use the same checkout
for native SPONGE, PRIPS and the Python package.

The default build uses `-march=native` for host C++ code, including code in a
CUDA executable. A binary built on a newer CPU can therefore fail with an
illegal instruction on an older compute node before model initialization.
Build on a CPU whose instruction set is supported by every intended node and
validate the executable on each node class. Selecting a CUDA architecture alone
does not make the host code portable.

## Configure an external adapter

Start from a SPONGE input for the intended system, including coordinates,
masses, velocities, units and integration parameters appropriate to the model.
Use the adapter maintainer's instructions to configure model loading and its
`atom_types` array; the array must match global SPONGE atom order.

Add the native PRIPS library and the external Python entry file to the input
(replace both absolute paths):

```toml
plugin = "/absolute/path/to/SPONGE/build-prips/_prips.so"
py = "/absolute/path/to/external/adapter_entry.py"
device = "0"

[DOM_DEC]
split_nx = 1
split_ny = 1
split_nz = 1

[PM]
MPI_size = 0
```

This fragment is for one PP rank and no PM rank; it is not a complete physical
input. The external entry exports `local_mp_adapter` and `atom_types` as defined
in the [adapter contract](LOCAL_MP.md). Keep classical force terms enabled only
when the intended physical model includes them: model output is added to existing
SPONGE forces and energies. Input `cutoff`/`skin` control the classical table;
the adapter separately declares the model graph cutoff/skin and unit factors.
A pure local-MP plugin may explicitly declare that it never uses the classical
neighbor table, as described in the contract.

From the run directory, launch the prepared input:

```bash
mpiexec -n 1 "$SPONGE_LOCAL_MP_ROOT/build-local-mp/SPONGE" -mdin mdin.spg.toml
```

Check process exit status and compare energy, force and virial with the adapter's
independent reference before running production MD. Output file presence alone
is not numerical validation. The [validation guide](LOCAL_MP_VALIDATION.md)
describes generic runtime regressions and external-adapter qualification.

## More than one rank or node

Each MPI rank needs its own GPU, including particle-mesh (PM) ranks: the device
communicator spans all ranks although only PP ranks evaluate the model.
`device` is a space-separated list with one visible CUDA device index per
**global MPI rank**. It is not automatically a local-rank assignment.

For two ranks on one node where each process sees both GPUs, change `device`
to `"0 1"`, change `split_nx` to 2, and launch with `mpiexec -n 2`. If the
scheduler exposes only the assigned GPU to each process, use `device = "0 0"`.
For one process on each of two nodes with one visible GPU per process, use
`device = "0 0"` and the same `2 1 1` splits. In an existing two-node allocation,
an Open MPI example is:

```bash
mpiexec -n 2 --map-by ppr:1:node \
  "$SPONGE_LOCAL_MP_ROOT/build-local-mp/SPONGE" -mdin mdin.spg.toml
```

Use your MPI implementation's equivalent placement flags. All ranks need access
to the same input/plugin/model files and Python environment; use a shared working
directory. Verify hostname, rank placement and GPU visibility in the allocated
environment before choosing the device list. For four processes per node on
two nodes, with all four local GPUs visible, the mapping is
`"0 1 2 3 0 1 2 3"` **only if ranks 0–3 and 4–7 occupy the respective nodes**;
an eight-PP run then needs splits such as `2 2 2` and a suitable system size.

## Common setup errors

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

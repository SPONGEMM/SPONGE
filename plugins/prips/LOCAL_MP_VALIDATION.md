# Local-MP validation and support boundary

A successful build establishes build compatibility. A parity test establishes
agreement for its fixture and tolerance. A short fixture trajectory tests
runtime integration; it does not validate a material model, long-time stability,
energy conservation or scaling. Keep those conclusions separate.

## Test coverage and support limits

The maintained regressions use four test source files: initial-coordinate
preservation, one/two-rank feature and MD parity, native edge-output parity
including an empty rank, and live restart coordinates/velocities with trajectory
output disabled and enabled. The MPI tests exercise the actual executable and
PRIPS plugin. Test registration does not establish a passing result in a
particular build or deployment.

The generic package has no model-library dependency. Validating an external
model requires its adapter and fixture revisions, which are not distributed
here, and independent full-system references. Fixture agreement
does not qualify arbitrary architectures, released checkpoints or production
physics. Models requiring directed-edge hidden-state exchange are not supported
by the current owned-atom feature exchange; native edge-gradient assembly does
not remove that limitation.

Cross-node runtime execution, production-weight long trajectories, NPT,
mixed classical/ML force qualification, general MPI portability and performance/
scaling are not established by these recipes. Two-node fixture success does not
establish arbitrary node-count scaling. MC barostat, an enabled ITS/SITS module,
long-range/iterative models
and nonorthogonal or nonperiodic cells are outside this interface's admitted
scope. Record the commands, environment, results and limitations of each
qualification run separately from this guide.

## Separate native test build

Allocate the GPUs and MPI slots required by the selected tests. The default
CUDA runtime selection requires two GPUs. Set `LOCAL_MP_TEST_GPU_COUNT` to the
number of available GPUs; only cases fitting that count are registered. Use a
separate build tree from the production recipe:

```bash
cmake -S . -B build-local-mp-tests -G Ninja \
  -DPARALLEL=cuda -DMPI=ON -DSPONGE_BUILD_TESTS=ON
cmake --build build-local-mp-tests --parallel 4
ctest --test-dir build-local-mp-tests -N -L local-mp
ctest --test-dir build-local-mp-tests --output-on-failure -L local-mp
ctest --test-dir build-local-mp-tests --output-on-failure -R '^test_initial_coordinate$'
```

Test-enabled SPONGE includes HDF5 fault-injection instrumentation; use the
[user-guide production build](LOCAL_MP_USER_GUIDE.md) for application runs.
CTest's GPU lock serializes these tests but does not reserve scheduler devices.
Configure `-DLOCAL_MP_TEST_GPU_COUNT=1` for single-GPU cases. Larger values
retain the same one/two-GPU selection. This setting does not allocate resources
or prove any case was run.

See [CTest registration](../../tests/prips/local_mp/CMakeLists.txt) for exact
names. For example, in a two-GPU allocation:

```bash
ctest --test-dir build-local-mp-tests --output-on-failure \
  -R '^(test_local_mp_two_pp_rank_parity|test_edge_output_two_pp)$'
```

The [edge-output regression](../../tests/prips/local_mp/test_edge_output_mpi.py)
compares native assembly with an analytic full-system reference. The
[message-passing regression](../../tests/prips/local_mp/test_local_mp_mpi.py)
compares distributed results with independently enumerated full-system geometry;
its model arithmetic is shared with the fixture. Both check physical outputs,
not hashes or implementation-internal storage layout. An MPI launcher failure
before calculation is an unexecuted check, not a passed numerical test.

These regressions are not a deployment qualification matrix. Grouped ownership,
nonadjacent owners, cross-node transport, PP/PM placements, larger GPU counts,
performance and production trajectories require separate validation against the
intended environment and external adapter.

## Qualifying an external adapter

Check the complete admitted model against an independent full-system reference
for energy, forces and physical 3x3 virial. Include unit conversion, shear signs,
periodic self-images, cutoff filtering while candidate graphs are reused,
zero-owned/zero-edge participation, migration and neighbor rebuilds. Exercise the actual forward and backward
communication schedule, not only a mocked exchange or an import test.

Record source/model versions, device/rank placement, backend, precision, atom
and edge counts, cutoff/skin, requested outputs, tolerances, actual errors and
failed or skipped cases. Preserve input and reference provenance. Scientific
validation additionally needs an appropriate trained potential and physical
observables; speed or capacity claims require separate timed and memory-measured
experiments with their success/OOM boundaries.

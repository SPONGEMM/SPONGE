# H5 Bundle Unit Tests

This directory contains opt-in unit and lightweight file-level tests for the new
SPONGE H5 bundle output paths.

The tests are not part of the default SPONGE build. Enable them explicitly:

```bash
cmake -S . -B build-h5-tests -DSPONGE_BUILD_TESTS=ON
cmake --build build-h5-tests
ctest --test-dir build-h5-tests --output-on-failure
```

When using pixi, run the same commands through the intended pixi environment,
for example:

```bash
pixi run -e dev-cpu cmake -S . -B build-h5-tests -DSPONGE_BUILD_TESTS=ON
pixi run -e dev-cpu cmake --build build-h5-tests
pixi run -e dev-cpu ctest --test-dir build-h5-tests --output-on-failure
```

## Test targets

| Target | Scope |
|---|---|
| `test_h5_output_plan` | Parser-visible H5 output keys, defaults, suffix helpers, helper null/empty-key behavior, empty H5 path handling, full legacy sidecar resolution matrix, explicit legacy sidecar provenance collection, VDS chunk size, repair policy validation. |
| `test_output_route_helpers` | H5 output routing helper contracts: mdout name sanitization/global uniqueness, numeric parsing, ReaxFF key recognition, output key lookup, and optional text sidecar reads. |
| `test_h5md_writers_with_mock_backend` | Public H5MD/restart/VDS path constants including SPONGE parameter roots/leaves, `H5MDWriter` common roots/schema/output metadata, trajectory writer dataset type/shape contracts, observable-only writer dataset type/shape contracts, observable-only module proxy paths, restart writer dataset type/shape contracts, optional particle-field disabled branches, failure status, restart module state, restart metad text components, legacy sidecar provenance. |
| `test_completion_tracker` | Output completion state machine and manifest validator behavior. |
| `test_module_h5_mappings_with_mock_backend` | NHC, SITS, metadynamics, ReaxFF path constants, H5 mapping paths, module dataset type/shape/chunk contracts, and module-level error behavior. |
| `test_vds_trajectory_writer_with_mock_backend` | VDS wrapper/shard rotation, manifest counters, particle/observable/module VDS source mapping, optional particle-field disabled branches, repair metadata, repair finalize behavior, wrapper diagnostics/provenance. |
| `test_highfive_backend_io` | Real HighFive/HDF5 backend behavior: backend factory, nested output paths, groups, datasets, appends, repeated dataset definitions, string-array metadata overwrite, hard links and repeated hard-link calls, element-level step/time paths, optional particle-field disabled branches, observable-only layout, invalid operations, restart single-frame state, zero-frame VDS finalize, complete-prefix VDS repair, and HDF5 VDS readback. |

## Coverage matrix

| Contract area | Covered by |
|---|---|
| `output_h5_trajectory_path` enables canonical trajectory H5 output | `test_h5_output_plan` |
| empty `output_h5_*_path` values do not enable H5 bundles or disable default legacy sidecars | `test_h5_output_plan` |
| `output_h5_trajectory_vds` does not enable output without path | `test_h5_output_plan` |
| `output_h5_trajectory_chunk_size` default and invalid values | `test_h5_output_plan` |
| `output_h5_trajectory_repair_policy` strict and complete-prefix behavior | `test_h5_output_plan`, `test_vds_trajectory_writer_with_mock_backend`, `test_highfive_backend_io` |
| `output_h5_restart_path` and `*.spgr.h5` suffix | `test_h5_output_plan` |
| `output_h5_observable_path` and `*.obs.spg.h5md` suffix | `test_h5_output_plan` |
| mdout key to H5MD-safe observable name conversion | `test_output_route_helpers` |
| duplicate sanitized observable name collision handling | `test_output_route_helpers` |
| sanitized observable names remain unique when user names already contain suffixes | `test_output_route_helpers` |
| mdout text value to double parsing | `test_output_route_helpers` |
| ReaxFF output key recognition for H5 routing | `test_output_route_helpers` |
| optional text sidecar read semantics | `test_output_route_helpers` |
| H5 output disables implicit legacy sidecars | `test_h5_output_plan` |
| Full 8-key legacy sidecar resolution matrix | `test_h5_output_plan` |
| Legacy sidecar helper null/unknown key behavior | `test_h5_output_plan` |
| Explicit legacy sidecar provenance collection excludes default sidecars | `test_h5_output_plan` |
| Explicit legacy sidecar provenance | `test_h5md_writers_with_mock_backend`, `test_vds_trajectory_writer_with_mock_backend` |
| Public H5MD/restart/VDS path constants | `test_h5md_writers_with_mock_backend` |
| Schema leaf paths and particle component group paths | `test_h5md_writers_with_mock_backend`, `test_highfive_backend_io` |
| SPONGE parameter roots and mdout column leaf paths | `test_h5md_writers_with_mock_backend`, `test_vds_trajectory_writer_with_mock_backend`, `test_highfive_backend_io` |
| Writer open preconditions reject unbound or wrong bundle plans | `test_h5md_writers_with_mock_backend`, `test_vds_trajectory_writer_with_mock_backend` |
| Bare H5MD writer without backend rejects all write paths safely | `test_h5md_writers_with_mock_backend` |
| H5MD common layout roots, schema metadata, and completion metadata | `test_h5md_writers_with_mock_backend`, `test_completion_tracker`, `test_highfive_backend_io` |
| H5MD failure error metadata path | `test_h5md_writers_with_mock_backend`, `test_completion_tracker`, `test_highfive_backend_io` |
| VDS output metadata paths: chunk size, status, repair policy/status/count | `test_h5md_writers_with_mock_backend`, `test_vds_trajectory_writer_with_mock_backend`, `test_highfive_backend_io` |
| Observable-only layout without `/particles` | `test_h5md_writers_with_mock_backend`, `test_highfive_backend_io` |
| Observable-only mdout/mdinfo/legacy/provenance paths | `test_h5md_writers_with_mock_backend`, `test_highfive_backend_io` |
| SPONGE provenance dynamic path builder | `test_h5md_writers_with_mock_backend`, `test_highfive_backend_io` |
| Observable-only module proxy paths | `test_h5md_writers_with_mock_backend`, `test_highfive_backend_io` |
| Observable-only element-level `step/time` paths | `test_highfive_backend_io` |
| Observable-only dataset type/shape/chunk contracts | `test_h5md_writers_with_mock_backend` |
| Ordinary observable dynamic value/step/time path builders | `test_h5md_writers_with_mock_backend`, `test_vds_trajectory_writer_with_mock_backend`, `test_highfive_backend_io` |
| Single-file trajectory particle/observable datasets | `test_h5md_writers_with_mock_backend` |
| Single-file trajectory dataset type/shape/chunk contracts | `test_h5md_writers_with_mock_backend` |
| Single-file trajectory optional velocity/force disabled branches | `test_h5md_writers_with_mock_backend`, `test_highfive_backend_io` |
| Single-file trajectory element-level `step/time` paths | `test_h5md_writers_with_mock_backend`, `test_highfive_backend_io` |
| Restart is single-state, not trajectory history | `test_h5md_writers_with_mock_backend` |
| Restart dataset type/shape/chunk contracts | `test_h5md_writers_with_mock_backend` |
| Restart optional velocity disabled branch | `test_h5md_writers_with_mock_backend`, `test_highfive_backend_io` |
| Restart element-level `step/time` paths | `test_h5md_writers_with_mock_backend`, `test_highfive_backend_io` |
| Restart NHC/SITS/metad state extensions | `test_h5md_writers_with_mock_backend`, `test_highfive_backend_io` |
| Restart SITS dynamic state components | `test_h5md_writers_with_mock_backend`, `test_highfive_backend_io` |
| Restart SITS/metad dynamic state path builders | `test_h5md_writers_with_mock_backend`, `test_highfive_backend_io` |
| Restart metad hills/history/edge/potential/direct text components | `test_h5md_writers_with_mock_backend`, `test_highfive_backend_io` |
| NHC/SITS/metad/ReaxFF path constants and observable mapping | `test_module_h5_mappings_with_mock_backend`, `test_vds_trajectory_writer_with_mock_backend`, `test_highfive_backend_io` |
| NHC coordinate/velocity and SITS module dynamic root path builders | `test_module_h5_mappings_with_mock_backend`, `test_vds_trajectory_writer_with_mock_backend` |
| Generic module scalar observable value/step/time leaf path builders | `test_module_h5_mappings_with_mock_backend`, `test_vds_trajectory_writer_with_mock_backend` |
| Metadynamics/ReaxFF dynamic leaf path builders | `test_module_h5_mappings_with_mock_backend`, `test_vds_trajectory_writer_with_mock_backend`, `test_highfive_backend_io` |
| NHC/SITS/metad/ReaxFF module dataset type/shape/chunk contracts | `test_module_h5_mappings_with_mock_backend` |
| Writer failure marks output failed | `test_h5md_writers_with_mock_backend`, `test_completion_tracker` |
| VDS shard rotation and manifest ranges | `test_vds_trajectory_writer_with_mock_backend` |
| VDS zero-frame finalize behavior | `test_vds_trajectory_writer_with_mock_backend`, `test_highfive_backend_io` |
| VDS optional velocity/force disabled branches | `test_vds_trajectory_writer_with_mock_backend`, `test_highfive_backend_io` |
| VDS shard manifest `path/status` string arrays | `test_vds_trajectory_writer_with_mock_backend`, `test_highfive_backend_io` |
| VDS repair metadata: policy/status/repaired shard count | `test_vds_trajectory_writer_with_mock_backend`, `test_highfive_backend_io` |
| VDS complete-prefix repair preserves only complete real shards | `test_vds_trajectory_writer_with_mock_backend`, `test_highfive_backend_io` |
| VDS source dims and virtual starts | `test_vds_trajectory_writer_with_mock_backend` |
| VDS source dataset paths match wrapper dataset paths | `test_vds_trajectory_writer_with_mock_backend` |
| VDS source path is relative to wrapper directory | `test_vds_trajectory_writer_with_mock_backend` |
| VDS particle virtual datasets use public particle path constants | `test_vds_trajectory_writer_with_mock_backend` |
| HighFive backend creates real datasets and hard links | `test_highfive_backend_io` |
| HighFive backend creates nested groups idempotently | `test_highfive_backend_io` |
| HighFive backend repeated hard-link calls are idempotent | `test_highfive_backend_io` |
| HighFive backend factory creates usable backends | `test_highfive_backend_io` |
| HighFive backend creates nested output file paths | `test_highfive_backend_io` |
| HighFive backend repeated dataset definition preserves appended data | `test_highfive_backend_io` |
| HighFive backend appendable/fixed dataset max-dims and chunk layout | `test_highfive_backend_io` |
| HighFive backend string-array metadata overwrite and empty-string elements | `test_highfive_backend_io` |
| HighFive backend creates readable HDF5 VDS | `test_highfive_backend_io` |
| HighFive backend rejects invalid operations | `test_highfive_backend_io` |

## Boundaries

These tests are intentionally not full MD runtime tests. They do not prove:

- a complete SPONGE simulation can run with H5 output enabled;
- CUDA/MPI rank behavior is correct;
- restart H5 can be read back into SPONGE;
- kill/resume across processes works;
- orphan shard cleanup or HDF5 dataset truncation works;
- metadynamics structured binary restart state is complete.

Those require runtime smoke tests or integration tests after the unit tests build
and pass.

## Maintenance rules

- Add parser-visible H5 keys to `test_h5_output_plan` first.
- Add pure output routing helper behavior to `test_output_route_helpers`.
- Add writer facade paths to mock-backend tests before adding runtime tests.
- Add any new module-specific H5 path to `test_module_h5_mappings_with_mock_backend`.
- Add any new VDS dataset mapping to `test_vds_trajectory_writer_with_mock_backend`, including source dims and virtual starts.
- Add any new HighFive backend primitive to `test_highfive_backend_io` with real file readback when possible.

## Selective CTest runs

After configuring with `SPONGE_BUILD_TESTS=ON`, tests can be run by label:

```bash
ctest --test-dir build-h5-tests -L h5_bundle --output-on-failure
ctest --test-dir build-h5-tests -L contract --output-on-failure
ctest --test-dir build-h5-tests -L mock-backend --output-on-failure
ctest --test-dir build-h5-tests -L module --output-on-failure
ctest --test-dir build-h5-tests -L vds --output-on-failure
ctest --test-dir build-h5-tests -L backend-io --output-on-failure
ctest --test-dir build-h5-tests -L failure --output-on-failure
```

Recommended first-pass order:

1. `contract`: parser-visible key and resolver logic.
2. `mock-backend`: writer facade behavior without real HDF5 IO.
3. `module`: module-specific mapping paths.
4. `vds`: wrapper/shard/VDS assembly logic.
5. `backend-io`: real HighFive/HDF5 file-level checks.

This ordering separates pure logic failures from HighFive/HDF5 API or runtime file
layout failures.

## Backend-IO VDS writer coverage

`test_highfive_backend_io` also covers the full `VdsTrajectoryH5Writer +
HighFiveBackendFactory` path. It writes a wrapper plus two one-frame shards,
then reads the wrapper back through HighFive and checks:

- particle VDS dataset existence and shape;
- absence of disabled velocity/force particle VDS datasets;
- observable VDS dataset existence and values;
- shard manifest dataset existence and frame counts;
- shard manifest path/status string arrays;
- output completion status and last complete step;
- trajectory chunk size metadata;
- VDS status text;
- repair policy/status metadata;
- repaired shard count metadata;
- VDS readback across shard files.

It also covers a zero-frame VDS finalize path: a wrapper can be finalized
without creating shard files or virtual datasets while still recording chunk
size, repair metadata, VDS status, completion metadata, and finalized output
status.

This remains a unit/integration boundary test for HDF5 IO. It does not run a full
SPONGE simulation.

## Backend-IO optional particle-field coverage

`test_highfive_backend_io` includes real HighFive readback for optional
particle-field disabled branches:

- trajectory `.spg.h5md` with velocity and force disabled;
- restart `.spgr.h5` with velocity disabled;
- VDS wrapper and shard `.spg.h5md` files with velocity and force disabled;
- absence of disabled `value/step/time` paths;
- continued presence and readability of position and box `value/step/time`
  paths;
- completion metadata and finalized output status.

These checks complement the mock-backend path-contract tests by proving the
same disabled-branch contract at the real HDF5 file layer.

## Backend-IO observable-only facade coverage

`test_highfive_backend_io` covers the full `ObservableH5Writer + HighFiveBackend`
path. It writes an observable-only H5MD file and reads it back through HighFive,
checking:

- `/particles` is absent;
- ordinary observable datasets exist and have the expected frame count;
- NHC, SITS, metadynamics, and ReaxFF observable `step/time` paths are readable;
- mdinfo, legacy sidecar provenance, and launch provenance are stored under `/parameters/sponge`.

## Path-constant contract coverage

`test_h5md_writers_with_mock_backend` and
`test_module_h5_mappings_with_mock_backend` include direct string-contract tests
for public H5 bundle paths. These tests intentionally duplicate the expected
literal paths to catch accidental layout drift during refactors.

`test_output_route_helpers` covers the pure helper functions used by
`MD_core/output.hpp` before data reaches a writer:

- conversion from SPONGE mdout keys to H5MD-safe observable names;
- collision suffixing after sanitization;
- strict numeric parsing for mdout scalar values;
- ReaxFF key classification for module routing;
- output key lookup semantics;
- optional text sidecar read semantics for parameter/provenance payloads.

Covered groups include:

- common H5MD roots: `/h5md`, `/particles`, `/observables`, `/parameters`;
- SPONGE parameter roots: `/parameters/sponge/schema`, `/parameters/sponge/output`;
- output completion paths: `frame_count`, `last_complete_step`, `last_complete_time`, `status`;
- VDS shard manifest paths;
- particle element `value/step/time` paths;
- restart state and restart extension paths;
- module paths for NHC, SITS, metadynamics, and ReaxFF;
- SITS dynamic `nk` path builders.

## Build-only aggregate target

The H5 bundle tests also provide an aggregate build target:

```bash
cmake --build build-h5-tests --target sponge_h5_bundle_tests
```

Use this when you want to check that every H5 bundle test executable compiles
before running any test. It is especially useful before the first CTest pass,
because it separates compile/link failures from runtime assertion failures.

## Backend-IO module VDS coverage

The real `VdsTrajectoryH5Writer + HighFiveBackendFactory` test also writes and
reads module-specific virtual datasets from the wrapper:

- NHC coordinate stream;
- SITS `nk` stream;
- metadynamics `meta` stream;
- ReaxFF energy term stream.

This complements the mock-backend VDS tests by checking that module VDS datasets
are readable through a real HDF5 wrapper and shard files.

## Mock-backend VDS schema coverage

`test_vds_trajectory_writer_with_mock_backend` checks the wrapper-side virtual
dataset specs for trajectory, observable, and module VDS outputs. The assertions
lock `DataType`, fixed total-frame shape, max shape, chunk shape, and
`appendable=false` for the VDS wrapper datasets created from completed shards.

## Test target manifest

See `TEST_TARGETS.md` for the authoritative list of H5 bundle build-only targets,
CTest targets, source files, labels, and the expected staged validation sequence.

## Cross-repository writer probe

`h5_bundle_writer_probe` uses the production `TrajectoryH5Writer` and
`VdsTrajectoryH5Writer` to provide deterministic output for consumer tests. It
supports finalized single-file output, a two-phase live SWMR publication, and a
two-shard VDS output:

```bash
cmake --build build-h5-tests --target h5_bundle_writer_probe

PYTHONPATH="$PWD/../Mokda/backend/src" \
SPONGE_BUNDLED_WRITER_PROBE=\
  "$PWD/build-h5-tests/tests/h5_bundle/h5_bundle_writer_probe" \
  python \
  ../Mokda/backend/tests/test_bundled_io_real_writer_e2e_unittest.py
```

The file-level output mode remains `single` for the local SWMR case;
`local_swmr` is the consumer transport mode rather than an H5 layout value.

## Staged runner script

A helper script is provided for staged validation:

```bash
tests/h5_bundle/run_h5_bundle_tests.sh staged
```

The script does not introduce new test logic; it wraps the documented CMake and
CTest commands. It supports overriding the build directory and tools:

```bash
SPONGE_H5_BUNDLE_TEST_BUILD_DIR=build-h5-tests \
  CMAKE=cmake CTEST=ctest \
  tests/h5_bundle/run_h5_bundle_tests.sh staged
```

With pixi:

```bash
pixi run -e dev-cpu tests/h5_bundle/run_h5_bundle_tests.sh staged
```

## Backend-IO string-array coverage

`test_highfive_backend_io` checks real HDF5 string-array readback for paths used
by SPONGE metadata, including generic string arrays, nested metadata paths,
overwrite semantics, empty-string elements, and VDS shard manifest `path/status`
arrays. This protects legacy sidecar provenance and manifest metadata paths from
backend string handling regressions.

### Current real-backend path emphasis

The real HighFive test target intentionally exercises more than basic HDF5 creation. It now checks the concrete file layout for trajectory, restart, observable-only, VDS wrapper, module observables, module diagnostics, legacy provenance, output status/error metadata, and restart single-frame constraints.

It also covers low-level backend behavior that the writer facades rely on:
factory-created backend instances, nested output path creation, append-driven
dataset shape growth, repeated `Create_Dataset` calls on an existing dataset
without truncating prior records, and string-array metadata overwrite semantics.

Mock-writer coverage additionally locks wrapper-level failure metadata for
trajectory append failure, observable-only missing values, restart second-state
rejection, VDS shard-finalize failure, and VDS virtual-dataset materialization
failure. It also covers VDS manifest write failure after virtual dataset
materialization, VDS repair metadata write failure, and final VDS status write
failure. It also distinguishes shard finalize failure from final wrapper
finalize failure. These tests assert that failures are written through the
standard `/parameters/sponge/output/status` and
`/parameters/sponge/output/error` paths instead of remaining only in in-memory
writer state.

This file documents intended coverage only; run the staged commands above to prove configure/build/CTest status.

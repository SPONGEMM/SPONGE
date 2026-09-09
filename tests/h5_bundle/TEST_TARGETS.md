# H5 Bundle Test Targets

This file is a lightweight manifest for the H5 bundle test suite. Keep it in
sync with `tests/h5_bundle/CMakeLists.txt`.

## Build-only targets

| Target | Depends on | Purpose |
|---|---|---|
| `sponge_h5_bundle_tests` | all H5 bundle tests | Compile/link the full H5 bundle test suite. |
| `sponge_h5_bundle_contract_tests` | `test_h5_output_plan`, `test_output_route_helpers`, `test_completion_tracker`, `test_topology_native_h5_reader`, `test_h5_input_validation`, `test_h5_input_matrix_contract`, `test_h5_io_contract_coverage`, `test_h5_io_matrix_spec` | Compile/link parser-visible key, H5 output routing helper, empty H5 path handling, legacy sidecar matrix, repair-policy, native topology reader, input validation/matrix contract, H5 sidecar coverage, static matrix specification, and completion logic tests. |
| `sponge_h5_bundle_mock_tests` | `test_h5md_writers_with_mock_backend`, `test_module_h5_mappings_with_mock_backend`, `test_vds_trajectory_writer_with_mock_backend` | Compile/link public path-constant, mock-backend writer, module mapping schema, VDS manifest, VDS optional particle-field, VDS wrapper schema, and VDS repair metadata tests. |
| `sponge_h5_bundle_backend_io_tests` | `test_highfive_backend_io`, `test_restart_h5_reader`, `test_topology_native_h5_reader`, `test_trajectory_h5_reader`, `test_h5_input_validation`, `test_h5_legacy_sidecar`, `test_h5_input_matrix_contract` | Compile/link real HighFive/HDF5 file-level tests, H5 input readers, validation, sidecar, and input matrix contract tests. |
| `sponge_h5_bundle_smoke_tests` | `test_h5_input_output_smoke_matrix`, `test_h5_reaxff_edip_runtime_parity`, `test_h5_restart_load_runtime_closure`, `test_h5_vds_terminal_resume_smoke` | Compile/link the opt-in runtime smoke matrix, REAXFF/EDIP sidecar parity closure, restart-load runtime closure, and VDS terminal/resume smoke. These tests skip by default unless `SPONGE_H5_ENABLE_RUNTIME_SMOKE=1`. |

## CTest targets

| CTest target | Source file | Labels |
|---|---|---|
| `test_h5_output_plan` | `test_h5_output_plan.cpp` | `h5_bundle;contract` |
| `test_output_route_helpers` | `test_output_route_helpers.cpp` | `h5_bundle;contract` |
| `test_h5md_writers_with_mock_backend` | `test_h5md_writers_with_mock_backend.cpp` | `h5_bundle;mock-backend;failure` |
| `test_completion_tracker` | `test_completion_tracker.cpp` | `h5_bundle;failure` |
| `test_module_h5_mappings_with_mock_backend` | `test_module_h5_mappings_with_mock_backend.cpp` | `h5_bundle;module;mock-backend` |
| `test_vds_trajectory_writer_with_mock_backend` | `test_vds_trajectory_writer_with_mock_backend.cpp` | `h5_bundle;vds;mock-backend;failure` |
| `test_highfive_backend_io` | `test_highfive_backend_io.cpp` | `h5_bundle;backend-io;vds` |
| `test_restart_h5_reader` | `test_restart_h5_reader.cpp` | `h5_bundle;input;backend-io` |
| `test_topology_native_h5_reader` | `test_topology_native_h5_reader.cpp` | `h5_bundle;input;backend-io;contract` |
| `test_trajectory_h5_reader` | `test_trajectory_h5_reader.cpp` | `h5_bundle;input;rerun;backend-io` |
| `test_h5_input_validation` | `test_h5_input_validation.cpp` | `h5_bundle;input;backend-io;contract` |
| `test_h5_legacy_sidecar` | `test_h5_legacy_sidecar.cpp` | `h5_bundle;input;backend-io` |
| `test_h5_input_matrix_contract` | `test_h5_input_matrix_contract.cpp` | `h5_bundle;input;contract` |
| `test_h5_io_contract_coverage` | `test_h5_io_contract_coverage.cpp` | `h5_bundle;contract;coverage` |
| `test_h5_io_contract_manifest` | `test_h5_io_contract_manifest.py` | `h5_bundle;contract;coverage` |
| `test_h5_input_fixture_equivalence` | `test_h5_input_fixture_equivalence.py` | `h5_bundle;input;contract;matrix` |
| `test_h5_test_targets_manifest` | `test_h5_test_targets_manifest.py` | `h5_bundle;contract` |
| `test_h5_matrix_plan_manifest` | `test_h5_matrix_plan_manifest.py` | `h5_bundle;contract;matrix` |
| `test_h5_ci_plan_manifest` | `test_h5_ci_plan_manifest.py` | `h5_bundle;contract;matrix` |
| `test_h5_io_matrix_spec` | `test_h5_io_matrix_spec.cpp` | `h5_bundle;contract;matrix;rerun;vds` |
| `test_h5_input_output_smoke_matrix` | `test_h5_input_output_smoke_matrix.cpp` | `h5_bundle;input;smoke;matrix;rerun;vds` |
| `test_h5_reaxff_edip_runtime_parity` | `test_h5_reaxff_edip_runtime_parity.cpp` | `h5_bundle;input;smoke;matrix;rerun;vds;manybody` |
| `test_h5_restart_load_runtime_closure` | `test_h5_restart_load_runtime_closure.cpp` | `h5_bundle;input;smoke;rerun;restart` |
| `test_h5_vds_terminal_resume_smoke` | `test_h5_vds_terminal_resume_smoke.cpp` | `h5_bundle;input;smoke;rerun;vds;failure` |

## Expected validation sequence

```bash
cmake -S . -B build-h5-tests -DSPONGE_BUILD_TESTS=ON
cmake --build build-h5-tests --target sponge_h5_bundle_contract_tests
ctest --test-dir build-h5-tests -L h5_bundle -L contract --output-on-failure
ctest --test-dir build-h5-tests -L h5_bundle -L coverage --output-on-failure
cmake --build build-h5-tests --target sponge_h5_bundle_smoke_tests
ctest --test-dir build-h5-tests -L h5_bundle -L matrix --output-on-failure
cmake --build build-h5-tests --target sponge_h5_bundle_mock_tests
ctest --test-dir build-h5-tests -L h5_bundle -L mock-backend --output-on-failure
ctest --test-dir build-h5-tests -L h5_bundle -L module --output-on-failure
cmake --build build-h5-tests --target sponge_h5_bundle_backend_io_tests
ctest --test-dir build-h5-tests -L h5_bundle -L vds --output-on-failure
ctest --test-dir build-h5-tests -L h5_bundle -L backend-io --output-on-failure
ctest --test-dir build-h5-tests -L h5_bundle -L rerun --output-on-failure
ctest --test-dir build-h5-tests -L h5_bundle -L smoke --output-on-failure
```

The `smoke` label builds and registers the runtime matrix but skips by default
unless `SPONGE_H5_ENABLE_RUNTIME_SMOKE=1` is present. Use the explicit runtime
gate on machines where SPONGE can initialize its selected backend:

```bash
SPONGE_H5_ENABLE_RUNTIME_SMOKE=1 \
  ctest --test-dir build-h5-tests -R 'test_h5_input_output_smoke_matrix|test_h5_reaxff_edip_runtime_parity|test_h5_restart_load_runtime_closure|test_h5_vds_terminal_resume_smoke' \
  --output-on-failure

tests/h5_bundle/run_h5_bundle_tests.sh test-smoke-runtime
```

The runtime smoke matrix covers normal-mode legacy/bundled/sidecar input
families, legacy and bundled output families, rerun bundled output with VDS
off/on, and an additional rerun frame-selection group with `rerun_start = 1`,
`rerun_strip = 0`, and `rerun_frame_limit = 1`. The selection group runs a
legacy second-frame baseline and a bundled-with-sidecar second-frame case; the
preparation inventory also checks the pure bundled second-frame mdin transform.
Bundled-output smoke cases also enable `output_h5_observable_path` and read
back trajectory-H5 and observable-only H5 observable streams, including frame
metadata, mdout column metadata, and per-column values matched against the same
case's `mdout.txt`. `test_h5_reaxff_edip_runtime_parity` separately runs
REAXFF/EDIP bundled-with-sidecar and pure bundled native rerun cases without
many-body scrub. Its temporary inputs replace the excluded, out-of-range atom
pair with the same interacting two-atom frames in legacy binary and H5
trajectory form, so both EDIP and REAXFF must be finite and non-zero. The test
compares legacy output and bundled output with VDS off/on and verifies the
many-body observable columns in H5 output.
`test_h5_restart_load_runtime_closure` separately runs `input_h5_restart_load`
policies that the broad matrix scrubs for stability: supported protocol
sidecar, dynamic NHC, and full NHC+SITS paths must run. The dynamic NHC case
uses an isolated core topology and minimal protocol binding, executes one
finite `dt=0` step, and requires exact finite, non-zero NHC-state roundtrip
into the output restart. Unsupported rerun NHC, metadynamics-without-module, and
pure-bundled custom-pair protocol gaps must fail with their precise runtime
diagnostics.
`test_h5_vds_terminal_resume_smoke` separately exercises VDS terminal
tail-shard failure with complete-prefix repair and the no-op resume-policy path
where all terminal shards finalize cleanly.

## Maintenance checklist

- Add every new H5 bundle CTest executable to `CMakeLists.txt` through
  `add_sponge_h5_bundle_test`. Standalone cross-repository probes are ordinary
  executable targets and are not part of this aggregate-target table.
- Assign at least one `h5_bundle` CTest label and one more specific label.
- Add compiled test executables to one of the build-only aggregate targets;
  CTest-only Python scripts do not need an aggregate build target.
- Add a row to this manifest.
- `test_h5_test_targets_manifest` fails if the CTest target table drifts from
  `CMakeLists.txt`.

## Expanded real-backend module path coverage

`test_highfive_backend_io` now covers module output paths through all three H5 output entry styles:

- single-file trajectory `.spg.h5md` via `TrajectoryH5Writer`
- observable-only `.obs.spg.h5md` via `ObservableH5Writer`
- VDS wrapper `.spg.h5md` via `VdsTrajectoryH5Writer`

The real-backend coverage includes backend factory behavior, nested output path creation, nested group idempotence, repeated dataset definition semantics, string-array metadata overwrite semantics, repeated hard-link calls, ordinary observable paths through dynamic helpers, SPONGE provenance paths through dynamic helpers, NHC, SITS nk, restart SITS dynamic state components through dynamic helpers, metad scalar/diagnostic paths through dynamic helpers, restart metad text components through dynamic helpers, ReaxFF multi-term observables through dynamic helpers, optional particle-field disabled branches, zero-frame VDS finalize behavior, complete-prefix VDS repair behavior, legacy sidecar provenance, output status/error metadata, and restart single-state invariants.

These additions are source-level only until the h5 bundle CTest targets are configured, compiled, and executed.

## Expanded path-contract coverage

The test suite now includes direct path-contract tests in addition to writer
behavior tests:

- `test_h5_output_plan` locks the parser-visible `output_h5_*` keys, suffix
  helpers, directory-preserving VDS shard-root derivation, null/empty-key helper
  behavior, empty H5 path handling, default VDS chunk size, repair-policy
  validation and error messages, strict boolean parsing for trajectory VDS mode,
  strict integer parsing for trajectory chunk size, the complete 8-key legacy
  sidecar resolution matrix, and explicit legacy sidecar provenance collection.
- `test_output_route_helpers` locks pure H5 output routing helper behavior for
  observable-name sanitization and global uniqueness, strict mdout scalar
  parsing, ReaxFF key classification, output key lookup, and optional text
  sidecar reads.
- `test_h5md_writers_with_mock_backend` locks public H5MD root paths, schema
  leaf paths, particle component group paths, SPONGE
  parameter roots/leaves, SPONGE provenance dynamic path builders, ordinary observable dynamic path builders, writer open preconditions, bare-writer no-backend behavior, common layout schema/output metadata, output completion paths, VDS output metadata paths, VDS shard manifest
  paths, particle element `value/step/time`
  paths, trajectory/observable/restart dataset type and shape contracts,
  observable axis paths, observable-only module proxy paths, restart `/run`
  paths, wrapper-level failure status/error paths, optional particle-field
  disabled branches, restart extension roots, restart SITS dynamic state
  components, and restart metad text components.
- `test_module_h5_mappings_with_mock_backend` locks module extension paths for
  NHC, SITS, metadynamics, ReaxFF, generic scalar observable leaf path
  builders, the NHC coordinate/velocity dynamic root path builders, the SITS
  module/`nk` path builders, and
  metadynamics/ReaxFF dynamic leaf path builders, and module dataset
  type/shape/chunk contracts.
- `test_vds_trajectory_writer_with_mock_backend` locks VDS wrapper metadata paths
  for chunk size, manifest path/status arrays, optional particle-field disabled
  branches, repair policy/status, `repaired_shard_count`, six-digit shard
  filename sequencing, public particle path constants, ordinary observable
  dynamic path builders, module dynamic leaf path builders, wrapper virtual
  dataset type/shape/chunk contracts, VDS source file/dataset path mappings,
  shard-finalize failure propagation into wrapper output status/error paths, and virtual-dataset materialization failure
  propagation into wrapper output status/error paths, plus manifest write
  failure, repair metadata write failure, and final VDS status write failure
  propagation into wrapper output status/error paths, and final wrapper
  finalize failure propagation into wrapper output status/error paths.
- `test_highfive_backend_io` reads back the same high-risk paths from real HDF5
  files where practical.

Keep these path-contract tests in sync with any future schema or naming change.
Changing a path intentionally should update the test expectation and the design
document in the same patch.

# H5 Input/Output Matrix Test Plan

This plan defines how to prove that the new bundled H5 I/O path stays aligned
with the legacy SPONGE I/O path. The strategy is intentionally staged: build
fast contract tests first, then a small end-to-end smoke matrix, then expand to
rerun, VDS, and full contract coverage.

## Goals

- Verify legacy input, bundled input, and bundled input with legacy sidecars.
- Verify legacy output and bundled output produce aligned results.
- Verify legacy sidecar materialization and legacy override/conflict behavior.
- Cover normal and rerun execution paths.
- Cover bundled trajectory output with VDS off and on.
- Keep the default test set small enough for PR runs while preserving a path to
  the full matrix for nightly or explicit CI jobs.

## Strategy

Use a staged test implementation, not a full runtime Cartesian matrix as the
first step.

The full matrix is the final target, but building it first gives poor failure
locality: a failure could come from fixture generation, H5 binding resolution,
sidecar injection, controller override semantics, SPONGE initialization, legacy
output routing, bundled output writing, VDS finalization, or comparison logic.
The staged plan keeps those responsibilities separate:

1. Static fixture and converter checks prove that the input file groups encode
   the intended contracts.
2. Internal input contract tests prove parser, resolver, validation, sidecar,
   and override behavior with precise errors.
3. Minimal runtime smoke proves that the resolved state can initialize and run
   SPONGE end to end.
4. Rerun and VDS smoke proves the high-risk execution modes.
5. The explicit legal matrix proves there are no accidental gaps in the
   supported combinations.

Input behavior must not be accepted solely by inspecting internal state.
Internal checks are required for diagnostics, but every input family must also
have at least one runtime smoke because controller injection order, path
resolution relative to the working directory, restart/rerun initialization, and
output compatibility are only proven by a real run.

## Implementation Order

Build the suite in layers and keep each layer independently useful:

1. Fixture invariants and manifest checks.
2. Input resolver, sidecar injection, and override/conflict contract tests.
3. Minimal normal-mode runtime smoke.
4. Sidecar-specific runtime smoke.
5. Rerun runtime smoke.
6. Bundled-output VDS off/on runtime smoke.
7. Full legal matrix coverage and CI gating.
8. Full-contract module parity expansion as native readers replace legacy
   sidecar bridges.

Do not block early phases on the full matrix. The matrix should be introduced
only after the input file groups and the focused smoke tests already give good
failure locality.

Each phase must have:

- A small fixture or fixture subset.
- A named test target.
- A clear pass/fail assertion boundary.
- A CI gate, either default or explicitly enabled.
- A manifest or static test that prevents accidental coverage drift.

## Implementation Work Breakdown

Treat this as the actionable build order. Each phase should land as a small,
reviewable change that leaves the default test suite green. Runtime-heavy checks
must be registered in CTest, but gated unless explicitly enabled.

| Phase | Deliverable | Default CI | Gated runtime | Blockers before next phase |
|---|---|---:|---:|---|
| 0 | Fixture helper and fixture inventory tests | Yes | No | Missing fixture path, unstable temp workspace, non-relocatable manifest path. |
| 1 | Input resolver, sidecar injection, and override contract tests | Yes | No | Missing sidecar relative-path assertion, missing absent/same/conflict override behavior, missing restart mutual-exclusion validation. |
| 2 | Minimal normal-mode smoke matrix preparation and comparisons | Preparation yes | Yes | Any input family lacks an end-to-end row, bundled output only checks process success, default legacy output sidecars leak into bundled-output rows. |
| 3 | Sidecar runtime smoke and pure-bundled sidecar-independence checks | Preparation yes | Yes | Bundled-with-sidecar does not run without explicit legacy keys, conflict is not caught before simulation, pure bundled still depends on `legacy_sidecars`. |
| 4 | Rerun and VDS smoke | Preparation yes | Yes | Rerun frame selection untested, VDS wrapper/shard metadata untested, rerun output state not compared to baseline. |
| 5 | Explicit legal matrix and drift guards | Yes | Optional full matrix | Case names differ between plan, static spec, runtime smoke, and CI manifests. |
| 6 | Full-contract fixture manifest and semantic coverage | Yes | No | Manifest bucket missing, converted typed payload only checked for presence, sidecar byte-match not checked. |
| 7 | Native reader parity expansion | Yes for each landed reader | Targeted smoke | Pure bundled parity claims exceed native reader coverage or leave sidecar-backed exceptions undocumented. |

For every new test row, record three facts in code rather than only in this
document:

- The input family: `legacy`, `bundled`, or `bundled_with_sidecar`.
- The output family: `legacy` or `bundled`, with VDS state if bundled trajectory
  output is active.
- The evidence type: static contract, prepared-mdin guard, runtime smoke, or
  semantic fixture comparison.

This keeps the suite from silently drifting into either of the two weak states:
a broad-looking matrix with no runtime proof, or a few runtime tests with no
manifest-level evidence that all I/O contracts are represented.

## Implementation File Map

Use this file map when turning the staged plan into tests. Each entry names the
file that owns the behavior, the phase it belongs to, and the minimum
verification expected before advancing to the next phase.

| Phase | Primary files | Verification boundary |
|---|---|---|
| 0 fixture plumbing | `h5_input_matrix_fixture.hpp`, `test_h5_io_matrix_spec.cpp`, `test_h5_io_contract_manifest.py` | Fixture roots copy into isolated workspaces; missing paths fail with path-bearing errors; mdin/H5/sidecar paths are relocatable. |
| 1 input contracts | `test_h5_input_matrix_contract.cpp`, `test_h5_legacy_sidecar.cpp`, `test_h5_input_validation.cpp` | Legacy fallback, H5 binding requirements, sidecar injection, allowed sidecar key sets, legacy override absent/same/conflict behavior, and restart/rerun mutual exclusion are checked before simulation. |
| 2 normal runtime smoke | `test_h5_input_output_smoke_matrix.cpp`, `test_h5_matrix_plan_manifest.py` | Each normal input family runs at least once; legacy and bundled outputs are compared; bundled-output rows reject default legacy trajectory/restart leakage. |
| 3 sidecar runtime smoke | `test_h5_input_output_smoke_matrix.cpp`, `test_h5_io_matrix_spec.cpp` | Bundled-with-sidecar input runs without explicit legacy keys, same-path override is idempotent, different-path override fails before simulation, and pure bundled input remains sidecar-independent. |
| 4 rerun and VDS smoke | `test_h5_input_output_smoke_matrix.cpp`, `test_h5_reaxff_edip_runtime_parity.cpp`, `test_h5_restart_load_runtime_closure.cpp`, `test_h5_vds_terminal_resume_smoke.cpp`, `test_vds_trajectory_writer_with_mock_backend.cpp`, `test_highfive_backend_io.cpp` | Rerun frame selection is checked; bundled rerun output is readable with VDS off/on; REAXFF/EDIP bundled-with-sidecar and pure bundled native runtime parity are closed without many-body scrub; restart-load dynamic/protocol/full runtime closure is tested outside the broad matrix; VDS terminal tail-shard repair and complete-prefix no-op resume-policy metadata are checked; wrapper metadata, shard metadata, shard payloads, and wrapper-relative source paths are validated. |
| 5 legal matrix guard | `test_h5_io_matrix_spec.cpp`, `test_h5_matrix_plan_manifest.py`, `test_h5_ci_plan_manifest.py` | The 15 legal case names, dimensions, runtime inventory, CTest labels, and runtime gate remain synchronized with this plan. |
| 6 full-contract coverage | `test_h5_io_contract_coverage.cpp`, `test_h5_io_contract_manifest.py`, `test_h5_input_fixture_equivalence.py` | Manifest buckets, H5 typed payloads, sidecar key/path tables, materialized sidecar byte provenance, and semantic equivalence checks cover the converter contract. |
| 7 native parity expansion | Reader-specific tests plus `test_h5_input_fixture_equivalence.py`, `test_h5_reaxff_edip_runtime_parity.cpp`, `test_h5_restart_load_runtime_closure.cpp`, and runtime smoke comparisons | Each sidecar-backed payload class is replaced by native H5 reader coverage before pure-bundled runtime parity claims are broadened; restart-load policy claims require both runtime success cases and expected-failure diagnostics. |

The file map is intentionally redundant with the phase descriptions below. The
phase descriptions explain why each layer exists; this map gives reviewers and
CI maintainers the concrete files that must move together.

## Phased Execution Checklist

Use this checklist as the implementation order. Do not skip directly to the
full Cartesian-looking matrix before the lower layers are green; otherwise
failures lose useful locality.

1. Fixture contract first:
   - Generate `legacy_input`, `bundled_input`, and
     `bundled_input_with_legacy_sidecar` from the same legacy source.
   - Verify directory shape, manifest schema, sidecar provenance, relocation
     safety, and pure-vs-sidecar H5 equivalence.
   - Run:

```bash
ctest --test-dir build-h5-tests \
  -R 'test_h5_io_contract_manifest|test_h5_input_fixture_equivalence' \
  --output-on-failure
```

2. Input behavior second:
   - Test legacy input fallback.
   - Test pure bundled H5 binding resolution.
   - Test bundled-with-sidecar injection.
   - Test legacy override behavior for absent key, same-path idempotence, and
     different-path conflict.
   - Run:

```bash
ctest --test-dir build-h5-tests \
  -R 'test_h5_input_matrix_contract|test_h5_legacy_sidecar|test_h5_input_validation' \
  --output-on-failure
```

3. Minimal normal smoke third:
   - Run the six normal-mode rows covering every input family and both output
     families.
   - Compare legacy and bundled output content; process success alone is not
     enough.
   - For bundled output, keep explicit `mdout` and `mdinfo` files for
     comparison but do not retain default legacy trajectory or restart sidecar
     outputs such as `crd`, `box`, `vel`, `frc`, or `rst`.
   - Run:

```bash
SPONGE_H5_ENABLE_RUNTIME_SMOKE=1 \
  ctest --test-dir build-h5-tests \
  -R 'test_h5_input_output_smoke_matrix|test_h5_reaxff_edip_runtime_parity|test_h5_restart_load_runtime_closure|test_h5_vds_terminal_resume_smoke' \
  --output-on-failure
```

4. Sidecar and override smoke fourth:
   - Keep sidecar runtime checks small and focused on injection and override
     behavior.
   - Prove pure bundled input still runs when compatibility sidecar files are
     unavailable.

5. Rerun and VDS fifth:
   - Add rerun trajectory input only after normal input/output smoke is stable.
   - Cover bundled trajectory output with VDS off and VDS on.
   - Validate wrapper metadata, shard metadata, and completed-frame sequence.

6. Legal matrix last:
   - Keep the case table explicit.
   - Represent every legal row either by runtime execution or by a static
     legality check.
   - Keep `test_h5_matrix_plan_manifest.py` as the drift guard between this
     document, `test_h5_io_matrix_spec.cpp`, and runtime smoke case names.

The result is staged implementation with a full legal matrix target, not a
single blind Cartesian test. Internal state checks are necessary for precise
input diagnostics, but they cannot be the only input proof; every input family
must have an end-to-end smoke row.

## Input Fixture Generation

The fixture source of truth is legacy SPONGE input. Use XPONGE to provide a
full-featured legacy-to-bundle converter, then commit small deterministic output
fixtures under `tests/h5_bundle/fixtures/input_matrix`.

The converter must be capable enough to generate fixtures for all I/O contracts
and SPONGE features that the tests claim to cover. It is not only a convenience
tool for tests; it is the canonical way to keep legacy and bundled fixtures
aligned.

Required converter outputs for each generated case:

- `legacy_input/`
  - Original legacy mdin and all referenced legacy payload files.
- `bundled_input/`
  - Bundled H5 topology/protocol/restart/trajectory files as required by the
    case.
  - Bundled mdin using only `input_h5_*` bindings for bundled input.
  - No `/parameters/sponge/files/legacy_sidecars` tables.
- `bundled_input_with_legacy_sidecar/`
  - The same H5 payloads as `bundled_input`.
  - Legacy sidecar key/path tables under
    `/parameters/sponge/files/legacy_sidecars`.
  - Sidecar files materialized below `legacy_sidecars/<key>/<basename>`.
  - Bundled mdin that can run without spelling those sidecar keys explicitly.
- `manifest.json`
  - Stable contract id.
  - Source legacy path for every converted payload.
  - Bundle file and bundle path for every H5 payload.
  - Sidecar key, sidecar path, component, direction, payload kind, and override
    policy for every sidecar payload.
  - Top-level relocation-safe paths for `case_root` and `bundled_mdin`.

Required fixture groups:

- `core_structural`
  - Minimal normal-mode case for fast contract tests and first smoke.
  - Must include all three input families: legacy, bundled, and bundled with
    sidecar.
  - Must include override fixtures for absent-key injection, same-path
    idempotence, and different-path conflict.
- `full_contract_rerun`
  - Broad rerun-mode case generated from a legacy sample that exercises the
    complete converter contract.
  - Must include topology, restart, rerun trajectory, protocol state, protocol
    sidecars, SITS, metadynamics, custom pairwise/listed force payloads, and
    ReaxFF coverage where supported by the converter.
  - Must include both sidecar-preserving and sidecar-stripped bundled input
    forms.

Required cross-family equivalence checks:

- Pure bundled H5 files and bundled-with-sidecar H5 files must be byte-equivalent
  after excluding `/parameters/sponge/files/legacy_sidecars`.
- Every materialized sidecar file must byte-match the legacy source file with
  the same basename.
- The manifest must list every sidecar key/path table entry with the original
  source key, source path, bundle file, bundle path, payload kind, direction,
  and override policy.
- Manifest top-level paths must be relocation-safe. Tests should validate by
  basename and expected fixture layout, not by requiring the original absolute
  generation path to still exist.
- Typed native datasets that replace legacy text files must be checked against
  legacy semantics, not only against H5 presence. For example, `mass.txt`
  and `/atoms/mass` must encode the same atom count and masses.

The input file groups should be constructed as follows:

| Family | Source | Required Contents | Purpose |
|---|---|---|---|
| `legacy_input` | Curated legacy SPONGE sample | `mdin.spg.toml` plus all referenced text/binary sidecars | Baseline behavior and source of truth. |
| `bundled_input` | XPONGE conversion with sidecar bridge stripped | H5 topology/protocol/restart/trajectory as needed, bundled mdin, no `legacy_sidecars` directory | Proves native H5 input can stand alone. |
| `bundled_input_with_legacy_sidecar` | XPONGE conversion preserving sidecar bridge | Same H5 payloads, `/parameters/sponge/files/legacy_sidecars`, materialized `legacy_sidecars/<key>/<basename>` files | Proves compatibility bridge, sidecar path resolution, and override behavior. |

The full converter remains necessary even though the first runtime smoke is
small. The tests claim coverage over I/O contracts and SPONGE features, so the
converter must be able to generate every payload class represented by the
manifest, including transitional sidecar-backed features.

Fixture regeneration should be explicit, deterministic, and reviewable. The
recommended workflow is:

```bash
# In XPONGE, regenerate bundle fixtures from legacy samples.
python -m xponge.tools.legacy_to_bundle \
  --input <legacy_case_root> \
  --output <tmp_fixture_root> \
  --mode normal-or-rerun \
  --with-sidecars \
  --without-sidecars \
  --manifest

# In SPONGE, copy the reviewed fixture output into tests/h5_bundle/fixtures.
ctest --test-dir build-h5-tests -R 'test_h5_input_fixture_equivalence|test_h5_io_contract_manifest' \
  --output-on-failure
```

The exact XPONGE command name may change with the converter implementation, but
the generated directory and manifest contracts above should remain stable.

## Fixture Roots

Use the static fixtures under:

```text
tests/h5_bundle/fixtures/input_matrix
```

Primary fixture groups:

- `core_structural`
  - Minimal normal-mode case.
  - Contains `legacy_input`, `bundled_input`, and
    `bundled_input_with_legacy_sidecar`.
  - Use this for the first contract tests and the first smoke matrix.
- `full_contract_rerun`
  - Broad rerun-mode case generated from XPONGE's full converter contract
    sample.
  - Use this for rerun smoke and later contract coverage.

## Phase 0: Fixture Helper

Add a test helper for consistent fixture access and temporary workspaces:

```text
tests/h5_bundle/h5_input_matrix_fixture.hpp
```

Responsibilities:

- Resolve the repository root and `tests/h5_bundle/fixtures/input_matrix`.
- Copy a selected fixture case into a per-test temporary working directory.
- Return paths to mdin files, bundle directories, and expected output
  locations.
- Normalize path handling so tests can run from CTest, direct binaries, or
  local scripts.

This phase should not run SPONGE. It only establishes stable test plumbing.

Exit criteria:

- Fixture helper can copy `core_structural` and `full_contract_rerun` into
  isolated temporary workspaces.
- Tests can locate legacy mdin, bundled mdin, topology/protocol/restart H5, and
  rerun trajectory H5 without depending on the process working directory.
- Missing fixture files fail with explicit messages.

## Phase 1: Input Contract Tests

Add:

```text
tests/h5_bundle/test_h5_input_matrix_contract.cpp
```

These tests should inspect resolved plans, validation behavior, and sidecar
injection without launching a full SPONGE run.

Required cases:

1. Legacy input:
   - No H5 input binding exists.
   - `legacy_input_allowed == true`.

2. Pure bundled input:
   - `input_h5_topology_path` is present.
   - `input_h5_protocol_path` is present.
   - Normal mode requires `input_h5_restart_path`.
   - H5 files do not contain
     `/parameters/sponge/files/legacy_sidecars`.
   - No legacy sidecar key is injected.

3. Bundled input with legacy sidecars:
   - H5 files contain
     `/parameters/sponge/files/legacy_sidecars/key` and
     `/parameters/sponge/files/legacy_sidecars/path`.
   - Sidecar paths resolve relative to the H5 container directory.
   - Allowed topology/protocol/restart sidecar keys are injected into the
     controller.

4. Legacy override/conflict behavior:
   - H5 sidecar injection of a key absent from mdin succeeds.
   - Existing mdin key with the same path is idempotent.
   - Existing mdin key with a different path fails with a conflict error.
   - Use:

```text
core_structural/bundled_input_with_legacy_sidecar/bundle/mdin.override_conflict.spg.toml
```

5. H5 and legacy restart input mutual exclusion:
   - `input_h5_restart_path` cannot be mixed with legacy
     `coordinate_in_file`, `velocity_in_file`, or `rst7`.

These tests are the first diagnostic layer. Failures here should be specific
and cheap to debug.

This phase intentionally checks internal state and validation results. It is
not enough to claim legacy alignment, but it is the fastest place to test:

- H5 input binding discovery.
- Legacy fallback and mutual exclusion.
- Sidecar key allowlists.
- Sidecar path resolution relative to the containing H5 file.
- Legacy override behavior:
  - absent key is injected;
  - same-key same-path is accepted;
  - same-key different-path is rejected before simulation.

Exit criteria:

- `test_h5_input_matrix_contract` passes.
- `test_h5_legacy_sidecar` passes.
- `test_h5_input_fixture_equivalence` passes.
- `test_h5_io_contract_manifest` passes.
- Pure bundled fixtures are proven not to carry legacy sidecar tables.
- Bundled-with-sidecar fixtures are proven to carry sidecar tables and
  materialized files.

## Phase 2: Minimal Normal-Mode Smoke Matrix

Add:

```text
tests/h5_bundle/test_h5_input_output_smoke_matrix.cpp
```

Use `core_structural` and run a small end-to-end matrix first:

```text
legacy in                 -> legacy out
legacy in                 -> bundled out
bundled in                -> legacy out
bundled with sidecar in   -> legacy out
bundled in                -> bundled out
bundled with sidecar in   -> bundled out
```

Do not include rerun or VDS in this phase.

Required assertions:

- SPONGE exits successfully for success cases.
- Expected output files exist.
- `mdout` or observable core fields align with the legacy baseline within
  tolerance.
- Bundled-output preparation keeps explicit `mdout` and `mdinfo` comparison
  files but does not keep default legacy trajectory/restart sidecar outputs.
- Bundled restart output matches expected:
  - position
  - velocity
  - box
  - step/time
- Bundled trajectory output matches expected:
  - frame count
  - step/time
  - position/box

This phase proves that input behavior is not only internally valid, but also
compatible with actual SPONGE initialization and output paths.

Run this phase only after Phase 1 passes. Keep it small and deterministic:

- Use `core_structural`.
- Use one or two MD steps.
- Compare the legacy baseline against each bundled-input or bundled-output row.
- Treat process success without output comparison as insufficient.

Exit criteria:

- Every input family has at least one successful SPONGE run:
  `legacy`, `bundled`, and `bundled with sidecar`.
- Both output families have at least one successful SPONGE run:
  `legacy` and `bundled`.
- Normal-mode bundled output is readable by the H5 readers used by tests.

## Phase 3: Legacy Sidecar Smoke Tests

Keep this phase focused on sidecar and override semantics.

Use:

```text
core_structural/bundled_input_with_legacy_sidecar
```

Required cases:

1. Bundled mdin without explicit legacy sidecar keys:
   - Sidecars are injected from H5.
   - SPONGE run succeeds.

2. Bundled mdin with an explicit same-key same-path legacy value:
   - Behavior is idempotent and succeeds.

3. Bundled mdin with an explicit same-key different-path legacy value:
   - Run fails before simulation.
   - Error message contains the conflicting key and indicates a conflict.

4. Pure bundled input:
   - Deleting or hiding legacy sidecar files must not affect the run.
   - This confirms pure bundled input does not accidentally depend on
     compatibility sidecars.

Exit criteria:

- A bundled-with-sidecar mdin without explicit legacy topology/protocol sidecar
  keys runs successfully.
- A same-key same-path explicit override remains idempotent at runtime.
- A same-key different-path explicit override fails before simulation and
  reports the conflicting key.
- Pure bundled input still runs after the `legacy_sidecars` directory is absent
  from the prepared workspace.

## Phase 4: Rerun Smoke

Use:

```text
full_contract_rerun
```

Start with:

```text
legacy rerun in               -> legacy out
pure bundled rerun in         -> legacy out
bundled with sidecar rerun in -> legacy out
bundled with sidecar rerun in -> legacy out, second-frame selection
legacy rerun in               -> bundled out, VDS off
legacy rerun in               -> bundled out, VDS on
pure bundled rerun in         -> bundled out, VDS off
pure bundled rerun in         -> bundled out, VDS on
bundled with sidecar rerun in -> bundled out, VDS off
bundled with sidecar rerun in -> bundled out, VDS on
```

Required assertions:

- Rerun frame selection obeys `rerun_start`, `rerun_strip`, and
  `rerun_frame_limit`.
- Trajectory input step/time and frame count are read correctly.
- Position and box values align with the legacy rerun baseline.
- Rerun output observables align with the legacy baseline.
- Bundled trajectory output is readable and has expected frame metadata.
- VDS output creates both the wrapper and shard files and exposes the expected
  completed-frame sequence.
- VDS terminal tail-shard failure is repaired to the complete prefix, and the
  complete-prefix resume policy is a no-op when all terminal shards finalize
  cleanly. This targeted smoke lives in
  `test_h5_vds_terminal_resume_smoke.cpp`.

Pure bundled rerun currently compares the native core mdout terms below.
Full-contract module/output parity for pure bundled rerun should be expanded as
the native module readers cover the remaining sidecar-only payloads.

Pure bundled rerun native core mdout columns:

```text
temperature
LJ_short
LJ_long
LJ
LJ_soft
LJ_soft_short
LJ_soft_long
PM
bond
angle
urey_bradley
dihedral
```

Current runtime native-parity exceptions:

| Exception | Runtime scrub behavior | Reason |
|---|---|---|
| `input_h5_restart_load` | Force H5 rerun restart loading to `structural` in prepared broad runtime smoke mdin. | Restart-load policy runtime closure is covered by `test_h5_restart_load_runtime_closure.cpp`; the broad matrix isolates structural restart state to keep baseline comparisons stable. |

Exit criteria:

- Legacy rerun input establishes the baseline.
- Pure bundled rerun input can read H5 trajectory frames and produce aligned
  native core observables.
- Bundled-with-sidecar rerun input can exercise the broader full-contract
  fixture and compare against the stable legacy baseline.
- Bundled-with-sidecar and pure bundled native REAXFF/EDIP rerun input have
  no-many-body-scrub closure tests for legacy output, bundled output VDS off,
  and bundled output VDS on.
- `input_h5_restart_load` dynamic, protocol, and full policies have targeted
  runtime closure: supported NHC/SITS and initialized metadynamics restart
  state must run; pure-bundled native custom pairwise/listed-force payloads
  must materialize and initialize; unsupported NHC and uninitialized
  metadynamics must fail with precise diagnostics.
- Rerun frame selection is tested separately from the default first-frame run.
- Bundled rerun output writes trajectory and observable H5 artifacts; normal
  bundled-output smoke covers restart H5 artifacts.
- VDS-on rerun output writes both the wrapper and shard directory and exposes
  the expected complete-frame sequence through the wrapper.

## Phase 5: Parameterized Matrix

After Phases 1-4 are stable, expand into a parameterized matrix.

Dimensions:

```text
mode:
  normal
  rerun

input:
  legacy
  bundled
  bundled_with_sidecar

output:
  legacy
  bundled

vds:
  off
  on
```

Use an explicit table of legal cases instead of blindly generating a Cartesian
product.

Rules:

- `vds` only applies to bundled trajectory output.
- Legacy output uses `vds = N/A`.
- Normal-mode cases that do not write trajectory output should not test VDS.
- Rerun plus bundled trajectory output is the main VDS on/off target.
- PR CI should run only the smoke subset.
- Nightly or explicit CI should run the full matrix.

Minimum legal case set:

| Case | Mode | Input | Output | VDS |
|---|---|---|---|---|
| `normal_legacy_in_legacy_out` | normal | legacy | legacy | N/A |
| `normal_legacy_in_bundled_out` | normal | legacy | bundled | off |
| `normal_bundled_in_legacy_out` | normal | bundled | legacy | N/A |
| `normal_sidecar_in_legacy_out` | normal | bundled with sidecar | legacy | N/A |
| `normal_bundled_in_bundled_out` | normal | bundled | bundled | off |
| `normal_sidecar_in_bundled_out` | normal | bundled with sidecar | bundled | off |
| `rerun_legacy_in_legacy_out` | rerun | legacy | legacy | N/A |
| `rerun_legacy_in_bundled_out_vds_off` | rerun | legacy | bundled | off |
| `rerun_legacy_in_bundled_out_vds_on` | rerun | legacy | bundled | on |
| `rerun_bundled_in_legacy_out` | rerun | bundled | legacy | N/A |
| `rerun_sidecar_in_legacy_out` | rerun | bundled with sidecar | legacy | N/A |
| `rerun_bundled_in_bundled_out_vds_off` | rerun | bundled | bundled | off |
| `rerun_bundled_in_bundled_out_vds_on` | rerun | bundled | bundled | on |
| `rerun_sidecar_in_bundled_out_vds_off` | rerun | bundled with sidecar | bundled | off |
| `rerun_sidecar_in_bundled_out_vds_on` | rerun | bundled with sidecar | bundled | on |

This is the complete support matrix for behavior, not necessarily the default
runtime matrix. The default suite may validate some rows statically and reserve
full runtime execution for explicit CI.

Implementation notes:

- Keep the legal case list explicit in both the plan and
  `test_h5_io_matrix_spec.cpp`.
- Keep `test_h5_matrix_plan_manifest.py` as the drift check between this plan,
  the static matrix spec, and the runtime smoke case names.
- The runtime smoke test may execute a staged subset by default when explicitly
  enabled, but every case name in the legal matrix must be represented either by
  runtime execution or by a static legality check.
- Avoid adding illegal Cartesian rows. In particular, `vds=on` is meaningful
  only for bundled trajectory output.

## Phase 6: Contract Coverage

Add:

```text
tests/h5_bundle/test_h5_io_contract_coverage.cpp
```

Use `full_contract_rerun/bundled_input_with_legacy_sidecar/manifest.json` to
verify that the broad fixture covers important contract classes.

Required coverage buckets:

- Topology typed datasets.
- Restart structural state.
- Restart dynamic state.
- Protocol sidecars.
- SITS state and sidecars.
- Metadynamics state and sidecars.
- Custom pairwise/listed force payloads.
- ReaxFF sidecars.
- Rerun trajectory input.
- Legacy sidecar key/path tables.
- Bundled output trajectory/restart/observable paths.

The goal is contract coverage, not multiplying every feature across the full
runtime matrix.

Exit criteria:

- `test_h5_io_contract_coverage` proves the broad fixture contains every
  required contract bucket.
- `test_h5_io_contract_manifest` proves the converter manifest has the expected
  statuses and override policies.
- `test_h5_input_fixture_equivalence` proves pure and sidecar-preserving H5
  payloads differ only by the legacy sidecar manifest tables.
- Sidecar file byte-match checks prove the sidecar bridge did not mutate the
  legacy payloads.

## Phase 7: Full-Contract Native Parity Expansion

This phase closes the gap between "pure bundled input runs" and "pure bundled
input is fully equivalent to the legacy path for every SPONGE feature in the
full-contract fixture."

Start with the sidecar-backed full-contract fixture because it can exercise the
complete legacy feature set through existing loaders. Then replace one sidecar
class at a time with native H5 readers and extend pure-bundled parity
assertions as each reader lands.

Recommended order:

1. Topology core terms: mass, charge, LJ, bonds, angles, dihedrals, NB14,
   exclusions.
2. Custom pairwise/listed force payloads.
3. Enhanced sampling and protocol sidecars: CV, restraints, SITS,
   metadynamics, steering, soft walls.
4. Many-body and specialized force fields: EAM, SW, EDIP, TERSOFF, ReaxFF.
5. Dynamic/protocol restart state such as Nose-Hoover chain, hills, and
   restart-embedded sidecar text.
   The current runtime closure target is
   `test_h5_restart_load_runtime_closure.cpp`, which covers supported dynamic
   NHC, protocol SITS sidecar, initialized metadynamics restart loading, full
   NHC+SITS loading, pure-bundled native custom pairwise/listed-force
   materialization, and the expected-failure diagnostics for inactive
   NHC/metadynamics runtime modules.

For each payload class:

- Add or extend a reader-level unit test.
- Add fixture semantic equivalence checks against the legacy source.
- Add a pure-bundled smoke assertion for the observable terms affected by that
  payload.
- Remove the sidecar dependency from the pure-bundled fixture for that payload.
- Keep the bundled-with-sidecar case as compatibility coverage.

Exit criteria:

- Pure bundled rerun compares all stable full-contract mdout/observable columns
  that the legacy baseline emits.
- Remaining sidecar-backed payloads, if any, are listed explicitly with the
  reason they cannot yet be native.
- The plan, manifest tests, static matrix, and runtime smoke case names agree.

## CI Labels

Use labels that allow staged execution:

```text
h5_bundle;contract
h5_bundle;smoke
h5_bundle;matrix
h5_bundle;rerun
h5_bundle;vds
```

Recommended CI split:

- PR:
  - Phase 1 contract tests.
  - Phase 2 minimal normal-mode smoke preparation.
  - Static matrix and manifest coverage.
- Nightly or explicit:
  - Phase 2 minimal normal-mode runtime smoke.
  - Phase 3 sidecar runtime smoke.
  - Phase 4 rerun runtime smoke.
  - Phase 5 full legal runtime matrix.

The runtime smoke test should be registered in CTest by default but skip unless
`SPONGE_H5_ENABLE_RUNTIME_SMOKE=1` is set. Before skipping, it should still
validate fixture inventory and generated mdin preparation so default CI keeps
the matrix wiring covered.

## Acceptance Criteria

The new I/O path is considered aligned with legacy only when:

- Internal plan and validation tests pass.
- Sidecar injection and conflict handling are explicitly covered.
- At least one normal-mode end-to-end smoke passes for each input family:
  legacy, bundled, and bundled with sidecar.
- At least one output smoke compares legacy output against bundled output.
- Rerun input is covered by an end-to-end smoke.
- Bundled trajectory output is covered with VDS off and on.
- The broad full-contract fixture has manifest-level coverage checks.
- The converter-generated sidecar files byte-match their legacy sources, and
  sidecar paths are relocation-safe.
- Legacy override behavior is covered for absent key, same-key same-path, and
  same-key different-path conflict.

Input testing must not rely only on internal state. Internal checks are required
for diagnostics, but at least one end-to-end smoke is required for each input
family because path resolution, controller injection, initialization order, and
actual output behavior are only proven by running SPONGE.

## Current Runtime Boundary

The default suite can prove parser, contract, fixture, reader, writer, static
matrix, sidecar provenance, and manifest behavior without launching SPONGE. The
final legacy-alignment claim still requires the gated runtime smoke to pass on a
machine where SPONGE can initialize its selected backend.

Use:

```bash
SPONGE_H5_ENABLE_RUNTIME_SMOKE=1 \
  ctest --test-dir build-h5-tests \
  -R 'test_h5_input_output_smoke_matrix|test_h5_reaxff_edip_runtime_parity|test_h5_restart_load_runtime_closure|test_h5_vds_terminal_resume_smoke' \
  --output-on-failure
```

or:

```bash
tests/h5_bundle/run_h5_bundle_tests.sh test-smoke-runtime
```

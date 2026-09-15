# Synchronous runtime I/O

`SPONGE/io` is the runtime boundary between simulation orchestration and file
I/O. `main.cpp` supplies explicit references through `SpongeIO::RuntimeState`
and invokes output events. `InitialStateBuilder` invokes the input phases as
part of [initial state construction](initial-state.md). Sessions borrow those objects;
the runtime objects must outlive the sessions.

## Input phases

`InputSession` preserves the launch order:

1. `Load_System()` resolves and validates H5 bindings, applies topology atom
   data, prepares compatibility sidecars, invokes the existing Xponge loader,
   then applies the topology force field.
2. `Restore_Dynamic_State()` runs after thermostats and barostats initialize.
3. `Prepare_Metadynamics()` prepares legacy restart text before `meta.Initial`.
4. `Restore_Protocol_State()` applies compatible protocol continuation state
   after the protocol modules initialize.
5. `Finish_Initialization()` releases cached launch payloads and file handles.

`SpongeH5MD::InputContext` keeps the resolved plan and lazily caches native
topology, restart protocol and restart dynamic payloads used by these phases.
Topology atom and force-field assembly consume the same payload. The restart
protocol preparation and restoration phases also share a payload. Calling
`Initial` again clears the previous context, including after a failed launch.

The plan describes launch bindings. Sidecar preparation may still populate
legacy controller commands consumed by existing loaders. Module-local H5
readers, structural restart loading in MD core and rerun's streaming reader
retain their existing implementations; the context does not cache trajectory
frames or claim to eliminate every H5 open in the program.

## Output events

`OutputSession` routes synchronous H5 and legacy output. It exposes schedule
queries, initialization, output events, publication and closing. Existing
`MD_INFORMATION::trajectory_output` writers retain the format-specific
implementation, output plans and per-family failure behavior.

Simulation orchestration invokes events where their source data is ready:

- MetaD hills follow metadynamics updates; SITS records follow SITS updates.
- ReaxFF and MetaD scalar events retain their positions among module print
  operations. Ordinary observables are written after those operations.
- Trajectory output retains coordinate mapping, device/host transfers,
  legacy writes, the MPI barrier and module trajectory writes in order.
- Force and restart events follow their existing schedules.
- `Publish()` follows the step's output events; `Close()` runs before the
  controller's final timing summary and teardown.

Streams retain their original `steps`/`steps + 1` and time conventions. Moving
all module output to one end-of-step callback would change those semantics.

`RestartOutputState` owns NHC, dynamic, SITS, MetaD, restraint and CV module
payloads and replaces the positional restart export argument list. Capturing
these payloads is gated by H5 restart output and MPI rank 0. Structural
coordinates and velocities are still captured synchronously by the exporter.
The exporter finalizes and closes its temporary file before atomic replacement;
trajectory and observable streams keep their existing publication protocol.

All operations remain synchronous. There is no output queue or background
writer, and `RestartOutputState` is not a complete asynchronous checkpoint.

## Validation

With the existing CUDA development environment:

```bash
pixi run --frozen -e dev-cuda13 cmake -S . -B build-dev-cuda13 -DSPONGE_BUILD_TESTS=ON
pixi run --frozen -e dev-cuda13 cmake --build build-dev-cuda13 --parallel 4 --target SPONGE sponge_h5_bundle_tests test_h5_input_plan test_h5_input_metadata test_h5_input_assembler test_h5_dynamic_state_modules test_h5_rerun_frame_selection
pixi run --frozen -e dev-cuda13 ctest --test-dir build-dev-cuda13 -L 'h5_bundle|h5_input' --output-on-failure
pixi run --frozen -e dev-cuda13 env SPONGE_H5_ENABLE_RUNTIME_SMOKE=1 ctest --test-dir build-dev-cuda13 -R 'test_h5_input_output_smoke_matrix|test_h5_reaxff_edip_runtime_parity|test_h5_restart_load_runtime_closure|test_h5_vds_terminal_resume_smoke' --output-on-failure
```

`test_h5_input_validation` additionally checks cache reuse, refresh on a new
launch, handle release, failed initialization and legacy-only bindings.

# Initial state construction

`SpongeRuntime::InitialStateBuilder` in `SPONGE/MD_core/initial_state.h` owns
the startup phase order and optional Maxwell velocity initialization. It uses
explicit references in `InitialStateBindings`; it does not access `main.cpp`
globals. Runtime objects and the input session must outlive the builder.

The phases are called once, in this order:

| Phase | State established | Prerequisites outside the builder |
| --- | --- | --- |
| `Load_Base_State` | Input system, CV atom layout, MD arrays, coordinates/velocity/box, time settings and step-zero temperature/pressure targets | Controller configuration |
| `Initialize_Dynamics` | Thermostat/barostat initialization followed by dynamic restart restoration | Plugin initialization |
| `Build_Constraints_And_Velocities` | Constraint topology, virtual atoms, final freedom, virtual coordinates and optional Maxwell velocities | Force-field connectivity and restraint setup |
| `Restore_Protocol_State` | Steering/restraint CV setup, MetaD initialization and protocol continuation state; release of cached input payloads | Neighbor-list resource initialization |
| `Build_Coordinate_Derivatives` | Update-group connectivity, residue/molecule grouping, periodic coordinate mapping and device coordinates | Protocol checks and plugin `After_Initial` hooks |
| `Distribute_State` | Process layout, initial local/ghost/module state, velocity projection/rescaling, plugin domain binding and PM synchronization | Solvent-LJ setup when periodic |
| `Finalize_Run_Range` | Checked absolute end step for the main loop | Output initialization |

Each phase checks its predecessor and advances only after successful completion.
An out-of-order or repeated call is rejected. The builder is not a restartable
resource manager; a failed partial startup is still handled by the existing
controller error path.

## State and synchronization rules

- Base construction delegates format-dependent input to `InputSession` and MD
  core. Existing missing-velocity defaults and restart load policies remain.
- Temperature/pressure schedules are sampled at step zero before thermostat,
  barostat and Maxwell initialization. Dynamic restart restoration retains the
  existing step/time policy. Per-step target updates stay in the simulation loop.
- Maxwell generation runs after constraints and virtual atoms determine the
  final degrees of freedom. Its existing MPI broadcast and device upload remain
  in `INITIAL_VELOCITY_INFORMATION::Initial`.
- Molecule initialization can modify global coordinates. Update-group and
  molecule construction remain before the initial distribution to local domains.
- `Distribute_State` invokes the supplied process-management and local-refresh
  callbacks in their original order. Only PP ranks perform the first local
  refresh and Maxwell constraint projection/rescaling. PM synchronization follows
  that finalization, on the same ranks as before.
- Restart restoration and Maxwell initialization do not gain new precedence
  rules in this refactor. No additional position projection, velocity generation
  or coordinate remapping is introduced.
- Ordinary force-field initialization, neighbor-list/PM resources and output
  writers stay with their modules. Later repartitioning and box changes reuse the
  existing local-refresh path without rerunning the startup builder.

## Regression coverage

The existing runtime suites exercise the physical results of these phases:

- `benchmarks/validation/misc/tests/test_initial_velocity.py`: native input
  preservation, Maxwell seeds/momentum/temperature, zero-mass atoms, step-zero
  schedules, SETTLE/SHAKE projection, invalid modes, AMBER and GROMACS input.
- `benchmarks/validation/barostat/tests/test_schedule_inputs.py`: scheduled
  temperature/pressure configuration.
- `benchmarks/validation/nopbc/tests/test_virtual_atoms.py`: virtual-atom
  coordinates, force redistribution and state synchronization.
- H5 runtime CTests: input/output matrix, dynamic/protocol restart closure,
  rerun, ReaxFF/EDIP parity and VDS terminal/resume behavior.

Point all runtime tests at the newly built executable. The Maxwell suite uses
`SPONGE_BIN`; the virtual-atom suite uses `SPONGE_EXECUTABLE`. The schedule suite
uses `SPONGE` from `PATH`, so prepend the build directory inside the activated
pixi environment.

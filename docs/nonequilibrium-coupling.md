> 本分支输入与代码职责已更新，当前接口见 [动力学组件](dynamics-components.md)。下文保留旧版方法推导/验证记录，其中 `[viscosity]`、`[[coupling]]` 需先由 Python 工作流展开。

# Nonequilibrium force coupling and mechanical work

The `[[coupling]]` array adds explicit external forces to the existing force
evaluation. Drivers can be superposed and use stable, zero-based global atom
indices. The current runtime supports **single-rank NVE**, fixed charges and
positive masses, without virtual atoms, SITS or velocity clipping. Group loads
additionally require `pbc=false`. Unsupported combinations fail explicitly.
Force, acceleration and electric-field drivers can also use periodic
[RATTLE constrained mechanics](constrained-nve-core.md), with continuous double
coordinates for path work. This driven use of `mode="nve"` selects dynamics
without a thermostat; external work can change the internal energy.

This slice implements forces, protocols and restartable mechanical path work.
It does **not** yet establish
a scientifically validated transport workflow. A separate
[cosine preset](nonequilibrium-cosine.md) now provides periodic forcing and
flow-aware Middle Langevin for unconstrained atoms. The
[Green–Kubo path](nonequilibrium-green-kubo.md) provides centered stress sampling
and explicit-window diagnostics for unconstrained or SETTLE/SHAKE/RATTLE NVE. The
[RNEMD path](nonequilibrium-rnemd.md) adds conservative momentum exchange,
spatial profiles and two-branch viscosity diagnostics.

## Input

Append one or more tables to an otherwise complete TOML input:

```toml
[[coupling]]
type = "force"
atoms = [0, 2]
force = [10.0, 0.0, 0.0]  # pN PER SELECTED ATOM
```

Omit `atoms` to select all atoms. An explicit empty selection, repeated IDs,
out-of-range IDs, nonfinite values and unknown keys are errors.

```toml
[[coupling]]
type = "acceleration"
acceleration = [0.01, 0.0, 0.0]  # nm/ps^2; force includes each atom's mass

[[coupling]]
type = "electric_field"
field = [0.0, 0.0, 0.05]        # V/nm; force is qE

[[coupling]]
type = "group_load"
atoms = [0, 1, 2]
force = [10.0, 0.0, 0.0]       # pN TOTAL for the group
torque = [0.0, 0.0, 1.0]       # pN nm about the current center of mass
```

Group loads require an explicit selection and at least one of `force` and
`torque`. The translational part is distributed in proportion to mass. With
COM-relative positions r, the rotational part is m (alpha cross r), where
I alpha = torque is solved using a symmetric pseudoinverse. Components outside
the range of the inertia tensor are rejected: e.g. two collinear atoms cannot
support a torque about their own line. Eigenvalues below 1e-12 of the largest
are treated as singular; the torque residual tolerance is 1e-8 relative.
No force component is silently dropped.

The initial group implementation gathers coordinates on the host for its
small matrix solve. It is a correctness implementation, not a large-group or
many-group performance claim. Periodic groups are rejected until their
continuous coordinates have a defined, tested interpretation.

## Time protocols

Omitting `schedule` gives a constant multiplier of one. A schedule multiplies
the entire driver, including both total force and torque for a group load.
Times are absolute simulation times in ps; phase is in radians.

```toml
schedule = {type="constant"}
# Or: zero until start, then ramp from zero to one, then remain at one.
schedule = {type="ramp", start=2.0, duration=5.0}
# Or: sin(2*pi*(time-start)/period + phase). start is a phase origin,
# not an activation gate; the sine also exists before start.
schedule = {type="sine", start=0.0, period=10.0, phase=0.0}
# Or: linear interpolation, clamped to the first/last value outside the table.
schedule = {type="table", points=[[0.0,0.0],[2.0,1.0],[5.0,-1.0]]}
```

Use only one `schedule` assignment in each coupling table. Ramp duration and
sine period must be positive; table times must strictly increase.

Force evaluation does not advance a protocol or accumulate work. The engine
supplies the evaluation time. Native restart inputs do not themselves preserve
absolute time; do not assume a scheduled run can be resumed by simply loading
coordinates. Use a dynamic H5 restart to preserve both absolute time and the work ledger,
with the same effective coupling configuration and time step. A structural
load intentionally starts a new experiment and resets accumulated work.

## Units, PME and observables

The engine uses Angstrom, kcal/mol and scaled charges internally. Conversion
occurs inside the coupling module: 1 pN is approximately 0.01439326 kcal/mol/A;
1 V/nm on a unit elementary charge is approximately 2.3060548 kcal/mol/A.
Native `charge.txt` already uses SPONGE-scaled charges, not elementary charges.

Uniform external qE is added independently of the internal PME force. Existing
PME settings continue to describe internal electrostatics. This does not
implement polarizable charges, metallic electrodes, constant displacement
fields or new slab boundary conditions.

Reported internal potential and internal virial do not acquire an r dot F
external-field term. Reported total force does include the external force.
NVT/NPT remain rejected in this slice; no thermostat is silently selected or
altered. Heat accounting and transport validation remain future work.

## Mechanical work and restart

`coupling_work` is cumulative external mechanical work in kcal/mol:

```text
W(n+1) - W(n) = sum_i 0.5 [F_ext(i,n) + F_ext(i,n+1)] dot [r(i,n+1) - r(i,n)]
```

Both endpoint forces use their own time and geometry, including group torque.
The displacement includes the integration and constraint updates and is
measured before output coordinate wrapping. It must not be replaced by a
minimum-image displacement. The currently supported single-rank iteration
preserves atom order during this operation.
For RATTLE, native coordinates are already wrapped during publication, so work
instead uses its canonical continuous double coordinates in force-owner order.
This work-coordinate convention is bound in the coupling restart configuration.

This is trapezoidal mechanical path work. It is not an exact finite-step
identity for leapfrog half-step kinetic energies, nor the parametric change
of an external potential. Smooth-drive tests check its second-order
continuous-time convergence. Discontinuous protocols and nonsmooth boundary
interactions require their own time-step checks.

Ordinary observable output retains the existing force-evaluation timestamp
`t_n` and reports `W_n`. A checkpoint represents the completed state at
`t_(n+1)` and stores `W_(n+1)`. Thus equal-time observable and checkpoint
values agree; comparing different timestamp conventions can look like a
one-step offset.

The H5 restart stores method version, effective coupling configuration, dt,
time, cumulative work, compensated-summation correction and completed-step
count under `/parameters/restart/coupling`. Every field is included in the
restart state hash. Dynamic loading rejects missing state, a changed/removed
driver, mismatched dt/time, unsupported fields and corrupted payloads. The
configuration comparison is exact on the decoded controller representation,
not a general proof of semantic equivalence between different input files.
Use structural loading when deliberately changing the experiment. Native
coordinate/velocity restart files do not carry this work ledger.

## Verification

`test_coupling_config` checks decoding, strict rejection, time protocols and
controller TOML round-trip. `test_group_load` checks net force/torque,
translation invariance, unequal masses, singular cases and nonfinite input.
`test_coupling_runtime` invokes the real executable with native inputs and
checks force units, charge signs, selections, superposition, zero-driver
compatibility, protocol timing, group torque, second-order smooth-drive
convergence of motion and work, periodic output wrapping, and qE superposition
with PME reciprocal calculations enabled. `test_work_ledger` checks precise
time serialization, compensated summation and malformed state rejection.
`test_coupling_restart` uses the real executable and HDF5 to check exact
continuous/split-run agreement for force and torque protocols, equal-time
observable/checkpoint values, intentional structural reset and hash tampering.

Run after building the SPONGE and test targets:

```sh
ctest --test-dir build-dev-cpu -R 'test_(coupling|work_ledger|group_load|mdin_schema|toml_command)' --output-on-failure
```

These checks validate the implemented force path. They do not establish
viscosity convergence, cross-method equivalence or agreement with experiment.

> 当前模块分工和输入见 [动力学组件](dynamics-components.md)；本文保留原数值推导。

# Mechanical constrained NVE core

`Advance_Constrained_NVE` in `nonequilibrium/constrained_nve.{h,cpp}` is a
host double-precision mechanical building block. The separate
`ConstrainedNVEState` runtime owner selects it with `constrain_mode="RATTLE"`.
Ordinary SHAKE/SETTLE and constrained cosine trajectories retain their existing paths.
Long-time liquid NVE convergence, molecular Green–Kubo transport and experimental
agreement remain unverified.

The function owns no force calculation, flow mode, thermostat, random numbers,
restart state or clock. It shares `ConstraintStressPlan` and its independent
distance topology, fixed orthogonal periodic geometry, component-size and
positive COM-free DOF requirements. A one-particle stress plan is therefore
outside this shared plan's supported domain, even though unconstrained
mechanical motion is mathematically defined for one particle.

At a force time, let `s` be the tangent staggered velocity and
`a = Pc(q) M^-1 F`. The physical velocity is `u = s + h*a/2`. The step applies
the projected full kick, followed by **one full-h RATTLE drift**:

```text
v = Pc(q) [s + h M^-1 F(q)]
(q_next, s_next) = RATTLE_Drift(q, v, h)
u_next = s_next + (h/2) Pc(q_next) M^-1 F(q_next)
```

The caller must evaluate the new force before constructing `u_next`. This
staggered B-A interval represents physical B/2-A-B/2 intervals with adjacent
half kicks merged at their common force time. It is different from the flow
core's B-A/2-O-A/2 mapping, even if O is exactly the identity: two finite
RATTLE half drifts need not equal one full drift.

The force-free drift follows the existing RATTLE primitive described in
[the constrained-flow core](constrained-flow-core.md). Incoming staggered
velocity must be tangent within its mass-norm numerical tolerance. Nonfinite
state/force, nonpositive timestep, invalid geometry, incompatible dimensions,
nonlocal constraint branch and failed solve are rejected.

The returned mechanical constraint impulse virial includes the projected
full kick and the drift. For a physical BAB interval, subtract the initial
half-B constraint impulse and add the final half-B one. This is not the
continuous force-time constraint stress used by Green–Kubo; that observer
remains `ConstraintStressPlan::Evaluate`. The drift's kinetic-energy change
is a numerical diagnostic, not thermostat heat.

`test_constrained_nve` checks an unequal-mass analytic rotating dimer and its
six impulse components, continuous rotor constraint stress, a coupled rigid
triangle, periodic images plus COM boost, and the unconstrained constant-force
velocity-Verlet limit. It checks 30 explicit physical BAB steps against the
staggered sequence and then reverses physical velocities. The staging
comparison shares projection/drift primitives and is not an independent
validation of those primitives. An independent RK4 angular equation for an
anisotropic harmonic rotor checks position-dependent forces: angle and maximum
physical-energy errors must decrease by factors in [3.8, 4.2] under h/2.

## Runtime selection and state

Use `mode="nve"`, `pbc=true`, `constrain_mode="RATTLE"` and the existing
`[constrain]` distance selection. RATTLE uses the complete list directly;
SETTLE does not consume any pairs first. Optional `[viscosity]`
`method="green_kubo"` uses canonical double q/s and the same native force/virial.
The first implementation requires a fixed orthogonal box, one rank, positive
masses and independent distance constraints. It supports explicit per-atom force,
acceleration and fixed-charge electric-field coupling. It rejects flow,
RNEMD, barostats, bias, walls, virtual sites, variable charges and velocity
clipping. Fixed-charge neutral PME remains supported with direct and reciprocal
forces every step. Host solves/copies are used also in a CUDA build; GPU runtime
execution and cost remain unverified.

With explicit coupling this is driven constrained mechanics: selecting `mode="nve"`
means no thermostat, not conservation of the internal energy under external work.
The existing force evaluation adds qE after internal PME; the total force enters
the tangent kick and instantaneous constraint reaction. Internal potential/virial
retain their existing definition and do not include an external r dot F term.
Equilibrium Green–Kubo remains incompatible with any explicit driver.
Group loads still require open boundaries and therefore cannot use this periodic
RATTLE path; no ambiguous periodic group geometry is inferred.

External work uses canonical unwrapped double coordinates, remapped to the same
owner order as the external-force arrays before and after the step. It therefore
retains displacements larger than a whole periodic box. The coupling ledger binds
`rattle_path_work_unwrapped_owner_double_v1` in its configuration; the mechanical
v3 identity and undriven restart contract are unchanged. Both ledgers share the
completed-step checkpoint time. No second clock, integrator or work stream is added.

Coordinates are unwrapped double values; publication wraps in double before
casting to the native float force engine. Stored s is a tangent **incoming
staggered velocity**, including on fresh text/structural loads. It is not an
automatic physical-velocity input: a desired u0 needs
`s0 = u0 - h Pc(q0) M^-1 F(q0)/2`. Fresh positions are projected first, so F(q0)
must correspond to the actual projected state. Fresh velocities are tangent
projected. Explicit `[initial_velocity] mode="maxwell"` retains the existing
fixed-energy initialization convention, with COM removal and normalization
after this double projection to `(3N-Nc-3) kB T/2` in s. This is not a canonical
kinetic-energy distribution or a guarantee that force-centered u has exactly
that temperature. Equilibrate separately before collecting transport data.

Fresh RATTLE coordinates enter this projection directly from float input,
without the legacy molecule connectivity reimaging roundtrip. Trajectory output
also preserves the atom-wrapped native publication; `force_whole_output=true`
is rejected. This avoids geometry changes caused by float cell arithmetic and
output cadence. The restart identity is `rattle_nve_double_v3`; earlier v1/v2
dynamic states require explicit structural loading for a new experiment.

At every force time, owner permutation, masses and native/canonical consistency
are checked. Native energy and virial are requested every step, independently
of GK and ordinary output cadence. Native energy/virial flags select different
force specializations whose rounding can differ; fixing both prevents output
frequency from changing the canonical trajectory. This has a computation cost
even on steps without output. The policy is bound in dynamic restart identity.
Temperature and pressure use the instantaneous constraint observer;
the previous discrete constraint impulse is not added. No float integration or
legacy constraint correction runs after this mechanical step.

The existing hashed HDF5 dynamic restart adds `constrained_nve_configuration`
and three float64 arrays under `integrator_state/double`:
`constrained_nve_position`, `constrained_nve_velocity`, and
`constrained_nve_drift_energy`. Identity binds algorithm, time layer, full drift,
solver tolerance, timestep, box, masses, distances and native publication.
Dynamic continuation preserves double values without reprojecting them;
missing state, old SHAKE state, changed settings or native/double mismatch are
rejected. Structural loading starts a new experiment. No thermostat RNG or
cosine modes are introduced.
Explicit `[initial_velocity]` regeneration conflicts with RATTLE dynamic
continuation and is rejected; remove it when continuing. Use structural load
when new random velocities are intended.

`test_constrained_nve_runtime` covers a full-engine analytic unequal-mass rotor
with second-order angle error and six stresses, multiple-boundary COM motion,
two-charge PME exact continuous/split restart, cadence invariance, malformed
restart rejection, and a multi-bond chain with a nonzero unconstrained spring.
Nonzero LJ/angle/torsion forces additionally check energy-output, trajectory,
GK and periodic-checkpoint cadence changes against exact double state and
common stress samples. The chain's physical velocity, instantaneous constraint virial and six pressures
are checked by independent NumPy Jacobian solves. Generated velocities are
checked after projection and for seed reproducibility. These bounded tests do
not establish liquid NVE convergence, liquid viscosity or experimental agreement,
and do not repair the existing float SHAKE path.

Additional full-engine tests combine native PME with constant force, acceleration
and electric fields, verify exact applied-force superposition, whole-box-crossing
path work and exact continuous/split state plus ledger. A sinusoidal electric
field checks cadence and restart identity; changed/removed fields are rejected.
Sinusoidal acceleration checks the analytic COM displacement under timestep
halving and bounds the physical COM energy/work discrepancy from native force
rounding. This does not assert an exact finite-step work/energy identity.

## Prepared hexane numerical pilot

`benchmarks/viscosity/hexane_rattle.py prepare` uses the test-only
`test_constrained_nve --prepare` probe to expose the projected q0 and tangent
velocity. The probe shares the engine core; independent checks use NumPy
Jacobians and LAMMPS potential forces/virials. It is not a second independent
position integrator. `hexane_model.py static --rattle` takes the actual RATTLE
initial force-time frame and subtracts independently computed kinetic and
continuous constraint stress for the potential-pressure comparison. Zero
incoming staggered velocity does not imply zero force-centered kinetic energy.

The `run` subcommand requires that bound static result and constructs each
dt-specific staggered input from a common target physical velocity. It checks
exact initial native force equality, canonical constraint residuals, native
and canonical physical kinetic energy, and initial six-component stress against
the independent LAMMPS/NumPy result. It preserves failed runs. Each output
records identities, physical handoff quantization error, raw data and an audit
CSV; energy is evaluated at one time layer with the LJ cutoff constant removed.
For longer sampling, `--duration` permits up to 200 ps with at most two million
steps. `--native-ps`, `--sample-interval` (steps), and `--checkpoint-ps` expose
aligned native, GK, and checkpoint cadences; the default short-pilot settings
are retained. Native frames are bounded to 20,001 including the initial frame.
The final checkpoint and all common sampled frames retain the same numerical
audit. These controls do not assert equilibrium or select a GK integration window.

A frozen 108-molecule pilot (648 UA sites, 540 distances), 2 ps at each of
0.5/0.25/0.125 fs, passed these gates. Maximum shifted physical-energy
excursions were 0.0072174/0.0019671/0.0005118 kcal/mol, adjacent ratios
3.669/3.844. This is bounded evidence of timestep improvement, without a
post-hoc energy threshold. It does not establish long-time stability, liquid
equilibration, transport convergence or experimental agreement. Old SHAKE
results used a different initial handoff and are not an identical-state
algorithm comparison.

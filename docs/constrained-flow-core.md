> 当前模块分工和输入见 [动力学组件](dynamics-components.md)；本文保留原数值推导。

# Constrained flow numerical core

The constrained cosine runtime uses these host double-precision building
blocks with the native force engine. It is available for single-rank, periodic
orthogonal fixed-box NVT with `middle_langevin`, a constant bath, and independent
distance constraints selected through `constrain_mode = "SHAKE"` or `"SETTLE"`.
These selections provide the distance topology; this flow path integrates it
with the RATTLE core below, without applying the legacy constraint step again.
There are no new input keys. Atomic RNEMD retains its constraint restriction.

**Validation is bounded to numerical kernels and short full-engine trajectories.**
Long-time molecular-liquid ensemble, timestep, driving-strength, system-size
and viscosity convergence remain unverified. This is not a claim of validated
water viscosity or agreement with experiment.

`ConstraintStressPlan::Prepare(q, box, tolerance)` builds a
`ConstraintTangentFrame` for one geometry. It validates distance lengths and
periodic images and factors the mass-weighted constraint Gram matrix per
connected component (at most 256 rows). Duplicate, dependent or ill-conditioned
constraints are rejected rather than regularized. `Project(v)` applies the
mass-orthogonal tangent projection; the constrained GK stress observer now uses
this same operation. Every frame owns its topology/factors and must be rebuilt
after a coordinate change. This is a host implementation for an orthogonal box.

`ConstrainedFlowFrame` receives that tangent frame, the actual per-atom cosine
values, and the flow axis. In whitened coordinates `w=sqrt(M)*v`, let `Pc` be the
constraint tangent projector. The three global translations and one raw flow
mode are first projected with `Pc`, then orthonormalized into `Q`. Two-pass
modified Gram-Schmidt uses the mass inner product without assembling a global
`3N x 3N` matrix. A mode residual norm squared below `1e-12` of its raw norm
squared is rejected. The thermal projector is

```
Ph = Pc - Q Q^T
thermal_dof = 3N - number_of_independent_constraints - 4
```

Only the global COM is removed. Relative translations of disconnected molecules
remain available to the thermal subspace except for their component along the
single retained flow mode. A mode which loses rank after constraint projection
is an error even if its raw cosine values have nonzero variance.

At a fixed geometry, `Apply_OU` implements

```
w_after = Q Q^T w_before + c Ph w_before
          + sqrt(kBT * (1-c) * (1+c)) Ph xi
```

Here `xi` is a vector of independent standard normal Cartesian draws supplied
by the caller, `c=exp(-gamma*dt)` and input velocity must already be tangent.
The core owns no RNG, restart clock, coordinates or forces. It rejects a
mass-norm tangent residual squared larger than
`1e-20 * max(1, ||v||_M^2)` in internal units. Small roundoff residuals are
projected; the exact `c=1` branch returns the validated input unchanged.
Heat is the actual returned kinetic-energy change, positive into the system.

At fixed q, the operation preserves total linear momentum and the raw
mass-weighted Fourier momentum of a tangent input. The normalized coefficient
in the tangent thermostat basis has a geometry-dependent denominator; it must
not replace the physical raw-velocity Fourier amplitude in the viscosity
estimator. The conditional noise covariance is `kBT*(1-c^2)*Ph`; the tangent
Maxwell covariance `kBT*Pc` is invariant when the retained modes are drawn from
that ensemble. These fixed-q facts alone do not establish the correct ensemble
or transport response after constrained drifts.

Validation is registered as `test_constrained_flow` and
`test_constrained_flow_covariance`. The latter independently constructs dense
NumPy SVD projectors, checks full matrices and OU updates, and uses 60,000
independent draws each for constrained and unconstrained geometries. It checks
all covariance entries, not only mean kinetic energy. Cases include unequal
masses, interleaved disconnected molecules, PBC representations, permutations,
oblique geometry and all flow axes. Existing constrained stress/rotor tests
continue to cover the observer using the extracted projection.

## Force-free constrained drift

`ConstraintStressPlan::Drift(q, v, h, box, tolerance)` performs one RATTLE
drift in host double precision. Position corrections follow the old constraint
normals; final velocities are projected at the new positions. This follows the
A substep in the appendix of [Leimkuhler and Matthews (2016), Efficient molecular
dynamics using geodesic integration and solvent–solute
splitting](https://pmc.ncbi.nlm.nih.gov/articles/PMC4893190/). Their constrained
BAOAB construction supplies the mechanical splitting; retaining a cosine mode
in the OU thermostat is our additional construction, not a result established
by that paper.

The position solver fixes each pair's initial periodic image and uses
Gauss–Seidel Newton corrections along the old normals, with relative length
tolerance `1e-12` and at most 256 sweeps. It rejects a nonlocal branch, singular
correction or unconverged solve. Signed nonzero `h` supports reversibility
checks. Input velocity must already be tangent at the input geometry.

The returned `impulse_virial` is the sum of pair displacement outer constraint
impulse at the old and new geometries. `Project(v, &impulse_virial)` can also
return its velocity-correction impulse virial. These have units of virial
times time; neither is the continuous force-time stress from `Evaluate`.
`kinetic_change` records the actual drift kinetic-energy change, including
numerical error; it must not be booked as heat.

## Complete staggered step

`Advance_Constrained_Flow` combines a projected full B kick, one RATTLE drift
of `h/2`, midpoint joint OU, and another RATTLE drift of `h/2`. It returns
positions, staggered velocity, OU heat, drift kinetic change, and mechanical
constraint impulse virial. It owns no force evaluation, RNG, restart or clock.
All arrays share stable atom indices; the caller supplies the current force.

Let `u_n` be the physical force-time velocity and `s_n` the stored velocity.
With `a_n = Pc(q_n) M^-1 F(q_n)`, the convention is

```
s_n = u_n - (h/2) a_n
u_n = s_n + (h/2) a_n
```

Starting from `s_n`, the full B kick produces `u_n + (h/2) a_n`.
The final A substep returns `s_(n+1)`, before the final physical half B kick.
Reconstructing `u_(n+1)` requires the force evaluated at `q_(n+1)`. This merges
the previous step's final half B and the next step's initial half B at the
same geometry and force; no kick is moved across a drift or OU substep.

The reported mechanical impulse sum covers this staggered B-A/2-O-A/2
interval. To obtain the physical B/2-A/2-O-A/2-B/2 interval, subtract the initial
half-B constraint impulse virial and add the final half-B one. OU impulses are
thermostat contributions and are excluded from that mechanical sum. Dividing
the staggered sum by `h` does not make it a force-time pressure estimator.

`test_constrained_flow_step` checks unequal-mass free dimers and rigid triangles
against analytic RATTLE rotations and impulse virials, reverse integration,
periodic images, and the unconstrained limit. A force-driven rotor is compared
to an independent angular ODE solved by RK4: halving the step reduces angle
and physical energy errors by approximately four. A separate 30-step test
compares an explicit physical BAOAB sequence to the staggered sequence under
position-dependent forces and nonidentity OU, including the endpoint impulse
correction. That sequence comparison shares the A/O primitives and validates
staging rather than independently validating those primitives.

## Runtime ownership, observables and restart

`FlowState` owns canonical global-atom `q` and staggered `s` in float64 across
steps. Native force evaluation and structural trajectory files use float32
images of that state. Owner IDs gather forces and scatter the next state;
Philox normal arrays depend only on the seed, invocation and array index.
For this single-rank path the noise array index is the canonical global atom
index, independently of the owner permutation. The same authoritative MD-core
internal timestep is used for kicks, drifts and OU damping.

Fresh input geometry receives one position projection after native velocity
initialization. The incoming velocity follows SPONGE's stored/staggered
convention and is projected onto the new tangent space. A dynamically resumed
trajectory restores double state without repeating this initialization.
Legacy molecule reimaging is skipped before this projection: the exact parsed
float coordinates are passed to the canonical integrator. Trajectory output
also preserves its atom-wrapped native publication and does not map coordinates
back into the force state. `force_whole_output=true` is explicitly unsupported.

At each force evaluation, before the step, `Evaluate` reconstructs tangent
physical velocity and continuous distance-reaction stress. Pressure is COM-free
and includes streaming kinetic motion; it excludes the external-body-force
virial. It is not a discrete constraint impulse divided by the timestep.
The existing `temperature` reports that COM-free physical kinetic temperature,
including flow. `thermal_temperature` uses the joint tangent thermal projector
and `3N - constraints - 4` degrees of freedom. Raw Fourier velocity remains the
viscosity observable. The engine requests both energy and virial every step on
this path. These flags select native force specializations, so keeping them
fixed prevents output cadence from changing force rounding and the trajectory.

The work ledger uses trapezoidal native external-force endpoints dotted with
the persistent double displacement, without minimum-image wrapping. Heat is
the actual double kinetic change in midpoint OU only. RATTLE drift kinetic
change is recorded separately as `constraint_drift_energy`; it is not heat.
Position-dependent conservative work still has force discretization error.
Native float force coordinates limit the accuracy of this mixed-precision
path. Force publication first reduces each coordinate modulo the box in double,
then casts to float, so its rounding scale depends on the box rather than the
number of periodic crossings. Canonical double positions remain unwrapped for
integration and work. Extremely large accumulated coordinates can still degrade
double constraint geometry or trigonometric argument precision.

The viscosity file declares `sponge.cosine.constrained_rattle.v1`, with the
sampling, stress, heat, precision and full constraint configuration recorded in
provenance. It adds six pressure components, constraint residuals, physical
kinetic temperature and cumulative drift kinetic change. `analyze_cosine.py`
accepts both contracts, validates the constrained heat/diagnostic columns and
reports them separately from viscosity and supplied energy. Reading a file is
not a dynamic ensemble or liquid transport acceptance test. Older archived
campaigns retain their frozen analyzer and source hashes.

Dynamic restart stores float64 `constrained_flow_position`,
`constrained_flow_velocity`, and `constrained_flow_drift_energy` under
`/parameters/restart/integrator_state/double`, included in the canonical hash.
The configuration binds the mode, timestep, box, masses, pairs and solver
contract; the work/heat ledger, clock and Philox state resume with it. Missing
state, changed configuration, dtype mismatch and hash tampering are rejected.
Use a structural load, explicitly starting a new experiment, when changing
these conditions. Measurement cadence and profile bin count do not change
integration state.

The configuration is `rattle_flow_double_v2`, including `raw_float_input_v1`,
`output_preserves_native_v1`, `energy_virial_every_step_v1` and
`native_periodic_double_v1`. Dynamic checkpoints from earlier policies are
rejected instead of silently changing initial geometry, force specialization
or force-coordinate rounding during a run. Archived campaigns retain their
original binary; structural loading starts a new experiment.
The runtime test covers both signs of a large COM displacement across repeated
periodic boundaries, box-scale native accuracy, and exact continuous versus
split-restart double states and work/heat ledgers.

Charged systems require a neutral cell, active direct and reciprocal PME paths,
and reciprocal force evaluation every step. Virtual sites, variable charges,
velocity clipping, walls, force plugins, additional bias, scheduled bath,
barostats and MPI decomposition are not supported on this path. Rank-deficient
constraints or retained flow modes fail explicitly.

`test_constrained_cosine_runtime` runs the actual engine. An unequal-mass free
rotor checks analytic discrete rotation, second-order continuous angle error,
all six force-time pressure components, geometry and tangency; a noisy rotor
checks independent kinetic/heat/drift accounting. Charged PME/cosine runs check
40 continuous steps versus 17+23 resumed steps, exact double/native/RNG/ledger
and sampled-output equality, unchanged state under different measurement
cadence, and rejection of altered checkpoints/configuration. These are short,
small-system tests, not stationary-ensemble or liquid transport validation.

The splitting follows the cited constrained BAOAB construction. The joint
flow-preserving OU extension still requires dynamic ensemble and transport
checks in real liquids; its fixed-geometry covariance proof does not replace
those checks.

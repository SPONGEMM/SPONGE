> 本分支输入与代码职责已更新，当前接口见 [动力学组件](dynamics-components.md)。下文保留旧版方法推导/验证记录，其中 `[viscosity]`、`[[coupling]]` 需先由 Python 工作流展开。

# Cosine forcing and flow-aware Middle Langevin

This development path implements periodic cosine acceleration, modal velocity
measurement and a projected Middle Langevin OU step. It is a building block
for viscosity experiments. No converged liquid viscosity, cross-engine
agreement or experimental accuracy is established by its ideal-particle tests.

## Input and supported systems

```toml
mode = "nvt"
target_temperature = 300.0
[thermostat]
mode = "middle_langevin"
tau = 1.0
seed = 1423
[viscosity]
method = "cosine"
acceleration = 0.001  # nm/ps^2; illustrative, not a validated production default
flow = "x"          # default x
gradient = "z"      # default z; must differ from flow
# Optional measurement settings, independent of ordinary output cadence:
bins = 20           # 4..10000; spatial diagnostics, not thermostat DOF
sample_interval = 1 # integration steps, global cadence across restart
output = "viscosity.h5" # fresh file for every segment
```

Append this to a complete input with masses, force field, coordinates, time
step and output settings. The gradient box length fixes the fundamental wave
number. Acceleration can be signed; zero permits equilibrium/mechanical tests.
There is no second manual coupling to add. Explicit `[[coupling]]` entries
cannot be combined with this viscosity preset.

Current support: single rank, fixed orthogonal periodic box, positive masses,
no virtual sites, variable charges, SITS or velocity clipping. Unconstrained
NVE is available for mechanics and transient checks. NVT requires explicitly
selected Middle Langevin. NPT and other thermostats are rejected; the friction
time must be finite and positive.

Distance-constrained NVT also supports `constrain_mode = "SHAKE"` or `"SETTLE"`
with a constant bath, no walls/plugins/additional bias, and a neutral charged
cell with active direct and reciprocal PME evaluated every step. These choices
supply distance topology to the host double RATTLE flow integrator; the legacy
constraint step is not applied again. See the [constrained runtime contract](
constrained-flow-core.md) for precision, initialization, stress, heat and strict
restart rules. Short numerical/runtime tests do not establish molecular-liquid
ensemble or transport convergence.

Separate [GK](nonequilibrium-green-kubo.md) and
[RNEMD](nonequilibrium-rnemd.md) paths have their own documented restrictions;
atomic RNEMD still excludes constraints.

The force is exactly

```text
F_i,flow = m_i A cos(k r_i,gradient),    k = 2 pi / L_gradient.
```

No instantaneous mean-force correction is silently added. At finite particle
number, sum(m_i cos_i) need not vanish; external forcing can therefore produce
COM drift. The output includes that drift and the mode mean. Production studies
must quantify this finite-system effect and use matched conventions in any
cross-engine reference.

## Unconstrained midpoint OU and temperature

The flow-enabled Middle Langevin path performs:

1. Full force kick.
2. Half coordinate drift.
3. At the midpoint, fit and subtract COM translation plus one cosine mode.
4. Apply OU damping and independent Maxwell noise to the residual velocities.
5. Project out the noise's COM/cosine components and restore the original fit.
6. Finish the second half coordinate drift.

Let M be the diagonal mass matrix and B contain the three translations and
the selected cosine velocity basis. The projection and OU step are

```text
P = B (B^T M B)^-1 B^T M,       Q = I - P,
v_after = P v_before + a Q v_before
          + sqrt((1-a^2) k_B T) Q M^-1/2 xi,
a = exp(-dt / tau).
```

The fit jointly handles the constant flow-direction velocity and cosine; it
does not assume the instantaneous mass-weighted cosine mean is zero or its
variance is 1/2. The projected noise has covariance
`(1-a^2) k_B T Q M^-1 Q^T`. No rescaling to an exactly fixed kinetic energy is
performed. A rank-deficient cosine mode (weighted variance <= 1e-12) is
rejected. With positive masses and full rank, thermal DOF is `3N-4`.

Only the **OU substep at fixed midpoint positions** preserves the fitted
collective velocities. Particle motion changes the projection between steps.
This global thermostat is not a model of a local hydrodynamic heat bath;
transport requires friction/time-step/size sensitivity checks. Its finite-N
projection and DOF conventions also differ from a default LAMMPS thermostat.
This is not a claim of identical finite-step dynamics between the engines.

The established path without cosine flow retains its original kernel and RNG
operation order. The new path uses the existing Philox seed/counter and
checkpoint mechanism. Projection coefficients are recomputed from state and
do not need a second persistent clock or RNG.

## Observables and viscosity interpretation

The original scalar diagnostics use the existing output cadence and ordinary
H5 output route. They refer to the stored pre-integration state. Velocity units
below are nm/ps. The independent `viscosity.h5` stream described next uses an
explicit integer-time velocity instead.

For the unconstrained path, these legacy cosine scalars use the incoming stored velocity at the
force-evaluation position; they do **not** apply the half-force time-centering
used in the separate GK/RNEMD streams. In particular, `cosine_v_centered` means
COM-subtracted Fourier amplitude, not force-time-centered velocity. Their names
and numerical convention remain unchanged for compatibility.

| Output | Definition |
|---|---|
| `cosine_v` | Raw literature Fourier estimator `2 sum(m_i v_flow cos_i) / M_total` |
| `cosine_v_centered` | Same estimator after subtracting COM flow velocity |
| `cosine_fit` | Finite-sample mass-weighted least-squares cosine coefficient |
| `cosine_mean` | `sum(m_i cos_i) / M_total` |
| `cosine_variance` | Mass-weighted variance of cosine |
| `thermal_temperature` | `2 K_residual / ((3N-4) k_B)` |
| `flow_com_x/y/z` | COM velocity components |
| `cosine_density` | Total mass / box volume, g/cm^3 |

For the unconstrained path, the existing `temperature` observable retains its
original full-velocity convention. Use `thermal_temperature` for the
flow-subtracted diagnostic. The constrained path instead uses cached tangent
physical force-time velocities for both the ordinary cosine scalars and its
independent stream. Its `temperature` is COM-free but includes flow;
`thermal_temperature` uses the joint tangent thermal subspace with
`3N - constraints - 4` degrees of freedom.
Raw Fourier amplitude, centered Fourier amplitude and least-squares amplitude
are deliberately distinct. The centered Fourier value is
`2 * cosine_variance * cosine_fit`; their equality is only a uniform-sampling
limit. A nonzero COM velocity and mode mean can contaminate the raw estimator.

### Independent integer-time stream

The `sponge.cosine.centered.v1` HDF5 contract samples after force evaluation
and before integration. At `q_n`, let `s_n` denote the stored incoming velocity
and `h` the actual internal time step. The stream uses

```text
u_n = s_n + h F_total(q_n) / (2m).
```

For NVE this changes the stored kick-drift map into velocity Verlet. For the
flow-aware Middle path, the same variable change yields
`B/2 -> A/2 -> O_projected -> A/2 -> B/2`: the closing half kick is represented
in `u_(n+1)` at the next force evaluation. The existing midpoint projection
and OU operation remain inside the map. Sampling copies arrays to the host;
it performs no extra OU draw, device velocity change, or force update.
This statement applies to the supported unconstrained fixed-box paths only.

The independent stream contains:

- `velocity_fourier`, `velocity_com_fourier`, `velocity_fit`: the three
  estimators above, now all using `u_n`.
- `mean_cosine`, `variance_cosine`, `thermal_temperature`, `com_vx/y/z`,
  `density`, `acceleration`, `wave_number`, `length`, with physical units.
- For each bin: `count`, `mass` (Da), `momentum` (Da nm/ps), `cosine_mass`
  and `sine_mass` (Da). The last two store sums of the actual particle basis,
  so finite-width spatial fits need not approximate it at bin centers.
- `coupling_work` (kcal/mol): cumulative trapezoidal external path work through
  the same `q_n`, before the next integration interval.
- In projected Middle NVT, `thermostat_heat` (kcal/mol): cumulative actual
  kinetic-energy change across the complete midpoint OU operation, positive
  into the particles, through the same completed intervals. It is absent in
  NVE; metadata explicitly says `thermostat_heat_method=none`.

Metadata records axes, bins, atom count, global sampling interval, time step,
dynamical configuration and velocity convention. The ordinary H5 writer owns
publication/finalization and its completion journal. Existing files and
input/output path collisions are rejected. Bin count, sampling interval and
output path can change between restart segments; they do not change the
force identity or thermostat state. The analyzer currently consumes one
finalized segment at a time; it does not silently concatenate/deduplicate files.

The constrained stream declares `sponge.cosine.constrained_rattle.v1` and
uses `u_n = Pc(q_n)[s_n + h F_total(q_n)/(2m)]`. It additionally records six
continuous pressure components, length/tangency residuals, physical kinetic
temperature and cumulative RATTLE drift kinetic change. Heat uses
`joint_tangent_ou_double_v1` and excludes drift kinetic change. The analyzer
validates and reports these fields separately; it retains the same raw Fourier
viscosity estimators. The constrained thermostat's mode coefficient is not
substituted for the physical Fourier velocity.

### Offline estimation

For a homogeneous Newtonian liquid in stationary weak forcing and a validated
long-wavelength limit, the periodic perturbation relation is
`eta = rho A / (k^2 <V>)`. Use the steady-state mean amplitude, not the average
of instantaneous inverse amplitudes. With rho in g/cm^3, A in nm/ps^2, k in
nm^-1, and V in nm/ps:

```text
eta [mPa s] = rho * A / (k^2 * <V>).
```

This relation and the Fourier estimator follow the periodic perturbation
method described by Hess (JCP 116, 209–217, 2002), as documented in the
[LAMMPS cosine acceleration](https://docs.lammps.org/fix_accelerate_cos.html)
and [cosine viscosity compute](https://docs.lammps.org/compute_viscosity_cos.html)
references. Applying the relation requires steady-state, profile, drive,
wave-length, time-step, thermostat and statistical checks. The runtime does
not print an instantaneous number labeled as a converged viscosity.

Run with a deliberately selected stationary window, block length and estimator:

```sh
python scripts/analyze_cosine.py viscosity.h5 \
  --start-ps 100 --block-ps 20 --estimator fourier --output analysis
```

These times are illustrative, not an equilibration criterion. The analyzer
validates units, completion history, cadence, populations and estimator
identities; writes `summary.json` and `profile.csv`; and retains all three
mean amplitudes. Select `fourier`, `com_fourier` or `fit` explicitly. Fit choice
is part of the protocol, not a parameter to tune toward experiment.

Statistics use disjoint equal-length blocks of uniformly sampled amplitudes,
drop incomplete tail blocks, and take the reciprocal of the overall mean.
Whole blocks are resampled, preserving within-block temporal dependence. The
95% interval of the sign-adjusted mean amplitude is inverted; when it touches
zero or has the wrong sign the viscosity interval is `null` and status is
`unresolved_response`. No bootstrap draws are removed. Raw and sign-adjusted
mean amplitudes have distinct fields. Block independence and interval coverage
still require block-length/replica checks; two blocks are only a technical
minimum, not sufficient scientific evidence.

Spatial diagnostics fit COM + cosine + sine to ratios of time-averaged bin
momentum and mass, using the stored mass-averaged basis and mass weights.
They retain phase and residuals; at least four occupied bins and a full-rank
fit are required. They do not substitute the profile coefficient for the
explicitly selected global amplitude estimator.

Current reductions gather arrays to the host and perform the fit in double
precision. This is an initial correctness path; GPU compilation is checked,
not GPU runtime throughput or transport accuracy. The constrained pressure
includes streaming motion, while its thermal-temperature diagnostic excludes
the retained flow mode. Neither observable alone validates the complete
transport workflow. Unconstrained projected OU heat accounting is described
below; the constrained path has the separate contract linked above.

## Matched LAMMPS reference

The benchmark `benchmarks/viscosity/lj_cosine_compare.py` uses native LAMMPS
`lj/cut` and `accelerate/cos` forces, with a **custom Python projected Middle
integrator** through `fix python/move`. This distinction matters: the pinned
29 Aug 2024 native `compute viscosity/cos` subtracts `V*cos(kz)` using raw
Fourier V. It neither jointly fits COM+cosine nor projects its thermostat noise
onto that joint complement. Native `nvt` or `langevin` with that compute is
therefore not an identical finite-N, finite-dt reference for this SPONGE path.

The independent reference uses mass-weighted QR for the four collective modes,
BAOAB integer-time splitting, and NumPy PCG64 noise. SPONGE uses its analytic
weighted fit and Philox noise. Matching drift/friction and Maxwell covariance
does not imply identical finite-temperature trajectories. Reference random
assignment and old/new external-force work pairing use global atom IDs, so
LAMMPS atom sorting does not change those identities.

Zero-noise, nonzero-force short runs test the splitting and work phase without
reporting a stationary viscosity. Finite-temperature pilots retain all three
amplitude estimators, profile fits, temperatures, COM and work. A native
`compute viscosity/cos` scalar/vector sample independently checks the raw
Fourier observer. The scalar must be evaluated first in this pinned source:
it refreshes V used by the vector. The runner checks agreement across every
sample. External work in both streams includes motion through the current
coordinate, with no one-step shift.

The reference is single-rank, atomic, fixed cubic periodic geometry with x/z
axes and one new run; it has no dynamic restart or molecular-constraint support.
Its Python integration and measurement overhead is included in timing and
must not be advertised as native LAMMPS throughput. It provides an independent
force implementation and separately implemented bath for correctness pilots;
formal transport agreement still requires independent repetitions and bias
checks. See the benchmark README for the pinned constants and startup protocol.

## Work, restart and tests

Cosine force enters the existing separate external-force buffer and
trapezoidal mechanical path-work ledger. Dynamic H5 restart identity includes
A, axes, box, integrator kind, temperature and friction; RNG counter is restored
through the existing thermostat mechanism. Changing those settings requires
starting a new experiment with a structural load. See the
[work and restart contract](nonequilibrium-coupling.md#mechanical-work-and-restart).

### Unconstrained projected OU heat ledger

No extra input switch is required. For cosine Middle NVT, `Measurement`
records the velocities after the force kick and first half drift, before bias
removal, then compares them with the actual velocities after bias restoration
and before the second half drift:

```text
dQ = sum_i 0.5 m_i (|v_i,after|^2 - |v_i,before|^2).
```

Positions stay fixed during this measurement interval. The definition includes
finite-precision changes from removing/restoring the bias; it does not pretend
that the floating-point projection exactly conserves the collective modes.
Each completed integration interval must contain exactly one completed heat
event. The measurement copies velocities and reduces energy changes without
additional noise draws or modifications to the state. CPU uses double sums;
GPU uses double block reductions. Cross-backend bitwise equality is not promised.

Work and heat have separate compensated double sums and share the existing
completed-step clock. The optional restart fields `heat_method`, `heat`,
`heat_correction` use method `projected_middle_actual_kinetic_delta_v1` and are
covered by the ordinary restart hash. Missing historical heat in an old cosine
NVT checkpoint is an explicit dynamic-load error; a structural load starts a
new experiment. Existing seven-field NVE work checkpoints remain supported.
No checkpoint can serialize an unfinished interval.

Ordinary output caches the completed heat at the force time `t_n`; the separate
viscosity stream reads the ledger before the next integration. Both therefore
report the same cumulative heat at the same time. Their output cadences can
still differ. The optional H5 extension retains the centered-v1 velocity
contract. The reader requires a declared method and kcal/mol units for heat;
old files with no heat metadata remain readable with heat reported as unknown,
not zero.

The offline report includes heat and work-plus-heat over the same explicitly
selected endpoints as its existing work diagnostic. **Work plus heat is supplied
energy, not an energy-conservation residual.** Compare it with the change of
same-time total internal energy, including thermal, streaming and COM kinetic
energy, and a potential consistent with the force/cutoff convention. Finite-step
splitting and path-work quadrature errors require their own timestep checks.
A thermostat setpoint alone does not establish the actual temperature under drive.

`test_flow_math` covers weighted projection on irregular masses/phases,
orthogonality, periodicity, rank rejection and projected Maxwell DOF/energy
fluctuations. `test_cosine_runtime` runs the real executable to check signed
force profiles/axes/mass scaling, pure-stream preservation at zero bath
temperature, analytic residual decay, finite-temperature mean/fluctuations,
COM preservation by projected noise and unsupported input rejection.
`test_coupling_restart` checks exact continuous versus split NVT cosine runs,
including positions, velocities, external-work/heat ledgers and equal-time flow
observables, and rejection of changed friction. New tests also verify finite-
force centering in both NVE and noisy Middle runs, global sample cadence,
unchanged trajectories when measurement settings change, and restart stream
values. Clock labels can differ by a double-precision ULP because the existing
clock regroups segment-origin and local-step additions. `test_cosine_analysis`
checks synthetic known responses, finite-bin basis, ratio-of-means statistics,
signed drive, unresolved intervals, and corrupt HDF5 rejection. These tests
are numerical/engineering checks; they do not establish a liquid viscosity.
Heat tests additionally check free-particle kinetic closure at zero/finite bath
temperature with unequal masses, cumulative sign and sampling time, hash coverage,
compensated restart, old-history rejection and strict optional H5 heat metadata.

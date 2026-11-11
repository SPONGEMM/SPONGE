# LJ transport benchmark: model and evidence gates

This directory prepares a frozen LJ model, verifies pair forces/energies/virials
against an independent double pair sum and LAMMPS, and generates equilibration
pilots. It does **not** yet establish viscosity convergence, method equivalence,
experimental agreement, or cost at a specified viscosity precision.

## Reference environment

`reference-env/pixi.toml` and its lock select LAMMPS **29 Aug 2024**, conda build
`cpu_py313_h2cff2b0_mpi_openmpi_6`, with Python/NumPy/h5py. The environment is
separate from SPONGE build environments. From the repository root:

```sh
pixi install --frozen --manifest-path benchmarks/viscosity/reference-env/pixi.toml
benchmarks/viscosity/reference-env/.pixi/envs/default/bin/lmp -h
```

The local evidence run used Pixi 0.68.1. Where the default cache is read-only,
set `PIXI_CACHE_DIR` to a writable project cache, as documented by
[Pixi](https://pixi.prefix.dev/latest/reference/environment_variables/).
The OpenMPI build can emit network-interface warnings in a restricted sandbox;
check the actual process exit and output files rather than interpreting those
warnings as completion or failure.

## Frozen model

Default model: 108 particles (`3³` FCC cells), requested reduced density 0.8,
reduced initial kinetic temperature 1, cutoff `2.5 sigma`, and `dt*=0.002`.
The physical mapping is epsilon=0.238 kcal/mol, sigma=3.4 Å, mass=39.948 Da.
It is a generic LJ validation model, not an experimental argon prediction.
Coordinates, box and masses are rounded to SPONGE storage precision and then
shared with LAMMPS. The actual density is reported after this rounding.

Both engines use minimum-image unshifted truncated forces. Native SPONGE LJ
input receives C12=`4 epsilon sigma^12` and C6=`4 epsilon sigma^6`; the loader
internally multiplies these by 12 and 6 for its force representation.

SPONGE always adds a C6-only long-range energy and diagonal virial correction:

```text
E_tail = -(2 pi / 3) N^2 C6 / (V rc^3)
P_tail = 2 E_tail / V
```

LAMMPS uses `pair_modify shift no tail no` here. The checker compares its direct
energy/virial with the pair sum, and separately adds SPONGE's C6 term when
checking SPONGE. LAMMPS full LJ tail correction is not interchangeable with this
C6-only convention. The tail supplies no force or shear virial and does not
restore the transport of an untruncated potential.

Reported unshifted energy jumps when a pair crosses the cutoff. For future NVE
energy-drift checks, reconstruct the continuous potential shifted at the cutoff;
do not mistake those reporting jumps for an integrator drift. The abrupt force
cutoff still needs timestep sensitivity checks in transport runs.

## Commands and static gate

Use new output directories. Substitute actual paths to the built executables:

```sh
python benchmarks/viscosity/lj_model.py prepare --output /path/to/evidence/model
python benchmarks/viscosity/lj_model.py static \
  --model /path/to/evidence/model --output /path/to/evidence/static \
  --sponge /path/to/SPONGE --lammps /path/to/lmp
```

`model.json` preserves parameters, seed, actual geometry, unit mapping and input
SHA256. Gaussian COM-free velocities and a jittered FCC configuration are
initial conditions, not an equilibrated liquid. Input hashes are checked before
any run directory is created. Static checks use zero velocities and zero SPONGE
timestep so that kinetic terms do not obscure the force/virial comparison.

The independent reference sums pairs directly in double precision. Tests verify
its forces and virial against finite differences and periodic translation.
The static executable gate checks both force arrays, both energies and all six
pressure components. LAMMPS pressure conversion uses its `real`-unit atm output;
small differences from SPONGE's SI constants have an explicit tolerance.
The summary preserves observed errors, binary hashes, script hash and elapsed
time, with raw logs/dumps/H5 alongside it.

## Short NVE propagation gate

```sh
python benchmarks/viscosity/lj_model.py dynamics \
  --model /path/to/evidence/model --output /path/to/evidence/dynamics \
  --sponge /path/to/SPONGE --lammps /path/to/lmp --steps 100
```

SPONGE input velocity is incoming leapfrog half-step velocity. Initialize LAMMPS
velocity-Verlet with `u_0=v_half+dt F_0/(2m)`. Compare subsequent SPONGE velocities
with the same force-time centering. The timestep and velocity conversions use
the frozen [LAMMPS real-unit constants](https://github.com/lammps/lammps/blob/stable_29Aug2024/src/update.cpp)
and SPONGE's actual float internal timestep. Thus the integration increment is
aligned explicitly; the small nominal physical-time difference is reported,
not hidden. This check does not imply identical stochastic thermostats.

Position errors use periodic minimum-image differences. Force errors include
cutoff membership differences; each frame whose pairs straddle the cutoff in
the two engines is recorded. A small position difference can produce a finite
force jump with this cutoff. The finite-horizon tolerances are distinct from
the tighter static gate, and are not viscosity accuracy criteria. Long chaotic
trajectories must be compared statistically, not by coordinate identity.

## Equilibration pilot

```sh
python benchmarks/viscosity/lj_pilot.py \
  --model /path/to/evidence/model --output /path/to/evidence/equilibration \
  --sponge /path/to/SPONGE --steps 20000 --seed 18273
```

Uses the existing Middle Langevin thermostat, tau=`1 LJ time`, at the specified
fixed density. It writes ordinary HDF5 observables and native position/velocity
frames every ten steps. Final paired frames supply `structural_seed.npz` with
half-step and reconstructed centered velocities. This is a new-experiment seed,
not a dynamic restart of the thermostat or its RNG. A later method start must
deliberately preserve or reset its time, protocol and ledger.

Diagnostics include latter-half potential/temperature blocks and COM-subtracted
single-origin MSD with periodic unwrapping. The maximum resolved frame jump is
checked; this cannot rule out an undetected multiple crossing between samples.
Neither MSD nor stable-looking temperature proves equilibrium. Removing the
heat bath requires an NVE temperature/energy-relaxation check before GK/RNEMD
production. Different heat-bath seeds from the same starting configuration are
pilot seeds, not automatically independent stationary replicas.

Execution wall time is saved before analysis, so an analysis error does not
erase cost evidence. A recovered early pilot whose full wall time was not saved
is explicitly marked `null` and excluded from cost comparisons. Short static
and propagation wall times include startup; they are not method-efficiency
rankings or estimates of cost at target viscosity precision.

## Paired Green–Kubo pilot

```sh
python benchmarks/viscosity/lj_gk_compare.py \
  --model /path/to/evidence/model \
  --seed-file /path/to/evidence/equilibration/structural_seed.npz \
  --output /path/to/evidence/gk-pair \
  --sponge /path/to/SPONGE --lammps /path/to/lmp \
  --steps 50000 --sample-interval 2 --audit-interval 1000 \
  --start-star 20 --block-star 10 --windows-star 0.5 1 2
```

This consumes a hashed structural seed from `lj_pilot.py`, removes COM velocity,
and constructs SPONGE's incoming half-step velocity and LAMMPS's integer-time
velocity from one quantized state. It does not restart the NVT bath or rescale
NVE temperature to the previous thermostat target. The two simulations run
sequentially with one thread each; each engine's wall time is saved before
analysis. Output directories must be new. Completed engines remain available
if postprocessing fails; `analyze_run(path)` can reanalyze without rerunning MD.
Failed engine jobs are retained but do not yet have an automatic resume command.

With Matplotlib installed, plot all declared windows without re-running MD:

```sh
python benchmarks/viscosity/plot_gk_pairs.py /path/to/gk-pair-1 /path/to/gk-pair-2 \
  --output /path/to/evidence/gk-diagnostics
```

This writes PNG and SVG figures. A full correlation curve, every declared
window with its conditional interval, and the energy audit are shown for each
seed separately. It does not pool seeds with different measured NVE temperatures.

LAMMPS explicitly uses `compute temp/com` and `compute pressure` with that
compute, so the kinetic pressure tensor also has COM removed. The six pressure
components are sampled without time averaging by `fix ave/time stride 1 stride`.
No per-atom trajectory is dumped by LAMMPS. See the primary documentation for
[COM temperature](https://docs.lammps.org/compute_temp_com.html),
[pressure tensor](https://docs.lammps.org/compute_pressure.html) and
[unaveraged global sampling](https://docs.lammps.org/fix_ave_time.html).
The runner checks the actual 29 Aug 2024 version banner. It converts atm to Pa,
Å³ to nm³ and fs to ps; the nominal SPONGE/LAMMPS time difference and the engines'
different numerical Boltzmann constants are retained in `plan.json`.

LAMMPS reports cutoff-shifted potential energy with tail disabled. This has
exactly the same forces as the unshifted frozen model. The sparse SPONGE audit
reconstructs the same energy by removing its C6 tail and subtracting `U(rc)`
for each active pair, then adds centered COM-free kinetic energy from the GK
stream. Sparse post-update native coordinates at completed audit steps align
with pre-integration force observables at those same steps. Coordinates within
2e-5 Å of the cutoff flag uncertainty in the reconstructed pair count; the
possible energy contribution is reported, not mistaken for certain drift.
This sparse O(N²) audit is bounded separately from production stress sampling.
It is not a time-step convergence test.

The initial-state gate uses the existing static float stress/energy tolerances
and a separate centered kinetic-energy tolerance. All requested GK integration
windows then use the **same existing analysis function** for both engines.
`canonical-series.npz` retains converted series; each window has a correlation
and running-integral CSV. `summary.json` retains every estimate and interval,
temperature, energy diagnostic and observed difference/ratio. Every pilot
comparison remains `inconclusive`: a point ratio inside 0.95–1.05, overlapping
intervals, or a small energy drift is not an equivalence proof. Independent
replicas, stationary-state and window/bias/precision checks remain necessary.

## Paired RNEMD pilot

```sh
python benchmarks/viscosity/lj_rnemd_compare.py \
  --model /path/to/evidence/model \
  --seed-file /path/to/evidence/equilibration/structural_seed.npz \
  --output /path/to/evidence/rnemd-pair \
  --sponge /path/to/SPONGE --lammps /path/to/lmp \
  --steps 50000 --exchange 100 --bins 20 \
  --start-star 20 --block-star 20 --exclude-neighbors 0 1
```

This uses the same frozen seed and centered/half-step conversion as the GK runner.
One signed pair exchanges every declared interval in a fixed periodic NVE box.
Profile and ledger cadence equal the exchange interval. Both LAMMPS sampling
fixes are defined **before** `fix viscosity`, so samples contain integer-time
velocities and impulses from strictly earlier exchanges, matching SPONGE.
`fix ave/chunk` uses `norm none` and one sample: mass, momentum and kinetic
sums are collected directly. The native LAMMPS scalar has the opposite impulse
sign; multiply by **−100** to convert Da Å/fs to the SPONGE Da nm/ps convention.
Do not repair signs with an absolute value. See the primary documentation for
[fix viscosity](https://docs.lammps.org/fix_viscosity.html) and
[fix ave/chunk](https://docs.lammps.org/fix_ave_chunk.html).

The reference uses `swap 1 vtarget INF`; source inspection of the pinned 29 Aug
2024 release places exchange at `END_OF_STEP`. Its finite numerical target and
local-order tie breaking differ from SPONGE's global-ID ties. Identical chaotic
trajectories are not an acceptance criterion. Version banners establish API
compatibility only; binary hashes, environment lock and separately inspected
source must be retained as distinct provenance evidence.

The adapter verifies cadence, population, geometry, mass normalization and
signed cumulative impulse. It reconstructs flow-subtracted thermal temperature
using the same occupied-bin DOF convention. The initial gate checks phase,
configuration, geometry, populations, momentum and centered temperature (after
accounting for the two numerical Boltzmann constants). Every declared fitting
exclusion uses the same aggregate-profile analyzer in both engines. Undefined
responses remain in the report. LAMMPS does not expose matching skip/residual
counters here; missing counters are not interpreted as zero.

Each engine's wall time is saved before analysis. New runs record plan and raw
output hashes at execution completion and verify them during reanalysis;
historical runs lacking those hashes are flagged explicitly. Reanalysis records
its own current script hashes, allowing intentional analysis upgrades without
rerunning MD. Preserve the previous summary and original execution timing when
upgrading an analysis. `analyze_run(path)` does not rerun or automatically resume
failed engine jobs. All intervals remain conditional pilot diagnostics.

With Matplotlib, plot aggregate profiles, every declared exclusion and thermal
history:

```sh
python benchmarks/viscosity/plot_rnemd_pairs.py /path/to/rnemd-pair-1 /path/to/rnemd-pair-2 \
  --output /path/to/evidence/rnemd-diagnostics
```

## Paired cosine pilot with a matched projected bath

```sh
python benchmarks/viscosity/lj_cosine_compare.py \
  --model /path/to/evidence/model \
  --seed-file /path/to/evidence/equilibration/structural_seed.npz \
  --output /path/to/evidence/cosine-pair \
  --sponge /path/to/SPONGE --lammps /path/to/lmp \
  --acceleration 0.1 --tau-star 1 --steps 50000 --stride 10 \
  --start-star 20 --block-star 10
```

This uses **native LAMMPS forces plus an explicit custom reference integrator**,
not its native biased thermostat. `lammps_projected_middle.py` implements the
integer-time B/2 A/2 O_projected A/2 B/2 map through the official
[python/move interface](https://docs.lammps.org/fix_python_move.html). It uses a
mass-weighted QR projection on three translations and one cosine mode, whereas
SPONGE uses the analytic weighted fit. Independent PCG64/Philox noise must be
compared statistically, never by stochastic trajectory identity. The reference
is not installed into LAMMPS or added to SPONGE's production input interface.

The runner adds the external force when constructing the incoming half-step
velocity. LAMMPS starts at the integer-time velocity represented by that
quantized state. Numerical acceleration is matched using
`A_LM = (10*A_SP/SP_TIME^2)/LM_TIME^2` (Å/fs²); this differs slightly from a
bare `A_SP*1e-5` because of the engines' historical time constants. Each analyzer
uses its engine's actual physical A and time labels. Bath temperature is
converted by `T_LM=T_SP*SP_KB/LM_KB` to match numerical kBT; damping matches
SPONGE's float friction/time calculation. These corrections are retained in
the run plan, not silently assumed to be exact decimal-unit identities.

The pinned reference environment provides its own Python 3.13 and NumPy.
The executable is launched with `PYTHONHOME` set to that prefix and user site
packages disabled, so embedded Python resolves the locked dependencies.
Source module/config copies are included in the immutable run-input hashes.
Reference snapshots use integer-time velocities after the closing half kick;
its trapezoidal work is paired by atom ID across sorting. The reference also
records its OU heat. SPONGE now records the matching actual projected-OU kinetic
change with compensated accumulation and dynamic restart, as documented in
[the cosine contract](../../docs/nonequilibrium-cosine.md#projected-ou-heat-ledger).

`compute viscosity/cos` is sampled separately as an observer, with scalar before
vector to refresh the cached amplitude in the pinned implementation. Its raw
Fourier amplitude must agree with the independent observer at every sampled
step. Its temperature has different bias/DOF conventions and is not substituted
for the QR-projected thermal temperature. See
[native cosine compute](https://docs.lammps.org/compute_viscosity_cos.html).

Use `--temperature 0` only for deterministic splitting gates. Such runs report
amplitude/temperature/work differences and no stationary viscosity. Actual
finite-temperature pilots retain every declared Fourier, COM-Fourier and fitted
estimate, including unresolved intervals. Plan/output hashes and per-engine
costs are saved; completed MD can be reanalyzed with `analyze_run(path)`.
Failed engine runs retain logs, available-file hashes and observed cost but do
not yet support automatic resume. Python reference overhead is part of the
observed runtime; this is not a native LAMMPS performance comparison.

`state_diagnostics` reports the actual thermal temperature and its departure
from the bath setpoint. A driven bath does not automatically keep the measured
kinetic temperature at its target. Compare methods only after matching the
actual state; a close pair of viscosity values at an overheated state is not
agreement with a lower-temperature GK run.

```sh
python benchmarks/viscosity/plot_cosine_pairs.py /path/to/cosine-A1 /path/to/cosine-A2 \
  --output /path/to/evidence/cosine-diagnostics
```

The figure retains all estimators, mean profile with fitted offset removed,
thermal history with bath setpoint and flow-direction COM history. It does not
pool the two driving strengths or turn conditional intervals into equivalence.

## Predeclared cosine sensitivity scan

`lj_cosine_scan.py` runs a fixed list of conditions without pooling them or
stopping when an interval happens to look favorable. A JSON specification is
explicit about reduced duration, sampling interval, startup exclusion, block
length and all cases, for example:

```json
{
  "contract": "sponge.lj.cosine-sensitivity.v1",
  "duration_star": 100, "sample_star": 0.02,
  "start_star": 20, "block_star": 10, "bins": 20,
  "cases": [
    {"id": "base", "acceleration": 0.05, "tau_star": 1, "dt_star": 0.002, "seed": 19575},
    {"id": "half-dt", "acceleration": 0.05, "tau_star": 1, "dt_star": 0.001, "seed": 19575}
  ]
}
```

These numbers are illustrative diagnostics, not validated production defaults.
Steps and stride must be integral: halving dt doubles both, preserving the
nominal reduced run length and sample spacing. Engine-specific physical time
labels retain the previously documented small unit-conversion difference.

```sh
python benchmarks/viscosity/lj_cosine_scan.py --spec /path/to/scan.json \
  --model /path/to/model --seed-file /path/to/structural_seed.npz \
  --output /path/to/new-scan --sponge /path/to/SPONGE --lammps /path/to/lmp
python benchmarks/viscosity/lj_cosine_scan.py --summarize /path/to/new-scan
python benchmarks/viscosity/plot_cosine_scan.py /path/to/new-scan \
  --output /path/to/scan-figure
```

The paired runner also accepts `--dt-star` directly. It verifies the original
model and seed hashes before deriving a new integration step. Source files are
never rewritten. The effective model metadata in the run plan has the new dt,
while `model_sha256` identifies the unchanged source file; `integration` records
source/effective dt and actual internal h. The centered initial state remains
canonicalized from the source seed and source dt, then a new driven incoming
half-step is built using effective h. It is a structural start, not a thermostat
restart across different dt.

Each case preserves both engines, all three estimators, raw profiles and
conditional intervals. Reports additionally contain block temperatures,
response divided by acceleration, the cosine-mode peak shear rate
`k*abs(mean fitted amplitude)`, and temperature block SE. Sample-weight duration
`number_of_samples*spacing` and work/heat endpoint duration are separate fields.
Neither their numerical difference nor the raw Fourier/fit difference is hidden.

The campaign archives its source modules, analyzer and upstream seed summary,
checks their hashes plus per-case inputs/results/executable identities, and
preserves failed status and elapsed cost. A failure stops later cases; it does
not silently rerun an engine or overwrite an old run. Summary requires every
declared case to complete. Scan-level costs sum each engine once per case,
separately from analysis and known upstream equilibration. Missing historical
setup cost stays unknown. The plots show observed conditional precision versus
observed runtime, not cost to scientifically accepted precision.

A common structural origin and reused random seeds do not supply independent
stationary replicas. Different A/tau/dt/sign are distinct conditions. A small
thermal-temperature departure alone does not establish thermostat-independent
transport. Drive, state, wavelength, statistical and thermostat effects must
still be separated before cross-method acceptance.

## Independent-initialization cosine replicas

`lj_cosine_replicas.py` adds independent randomized initial structures and
velocities, separate NVT equilibration, and a paired SPONGE/reference experiment
for every declared replica. Initialization, equilibration, SPONGE production and
reference production seed values must be globally unique. Their roles are
recorded separately from the bootstrap seed. All replicas share the explicit
physical parameters; `model.json` hashes intentionally differ because they also
identify the initial state. A separate physical-condition fingerprint excludes
initialization information and includes Hamiltonian, integration and target
state. Neither a different seed nor successful equilibration proves stationary
liquid sampling.

The strict JSON specification has the following shape (four replicas are the
minimum accepted by this diagnostic tool, not an adequate statistical guarantee):

```json
{
  "contract": "sponge.lj.cosine-replicas.v1",
  "model": {"cells": 3, "density": 0.8, "temperature": 1.0,
    "epsilon": 0.238, "sigma": 3.4, "mass": 39.948, "cutoff": 2.5, "dt": 0.002},
  "equilibration_steps": 50000,
  "duration_star": 200, "sample_star": 0.02,
  "start_star": 40, "block_star": 20, "bins": 20,
  "acceleration": 0.025, "tau_star": 1,
  "bootstrap_draws": 20000, "bootstrap_seed": 91827,
  "replicas": [
    {"id": "r01", "initialization_seed": 31001, "equilibration_seed": 41001, "production_seed": 51001},
    {"id": "r02", "initialization_seed": 31011, "equilibration_seed": 41011, "production_seed": 51011},
    {"id": "r03", "initialization_seed": 31021, "equilibration_seed": 41021, "production_seed": 51021},
    {"id": "r04", "initialization_seed": 31031, "equilibration_seed": 41031, "production_seed": 51031}
  ]
}
```

A is in nm/ps²; the model quantities retain the units of `lj_model.prepare`.
The equal-duration first/second-half diagnostic requires an even complete block
count. No adaptive early stopping, favorable-estimator selection, or pooling of
different conditions is performed.

```sh
python benchmarks/viscosity/lj_cosine_replicas.py --spec /path/to/replicas.json \
  --output /path/to/new-campaign --sponge /path/to/SPONGE --lammps /path/to/lmp
python benchmarks/viscosity/lj_cosine_replicas.py --summarize /path/to/new-campaign
python benchmarks/viscosity/plot_cosine_replicas.py /path/to/new-campaign \
  --output /path/to/replica-figure
```

For each engine, the sufficient statistic is signed mean response divided by
its known density/acceleration/wavenumber prefactor, i.e. fluidity. Average this
statistic over equal-duration replicas, then invert once. The uncertainty
calculation resamples whole replica IDs, using the same index matrix for both
engines and all estimators. It does not resample blocks as additional independent
replicas. Temperature and first/second-half response changes use the same
replica bootstrap. Ratios with a wrong-sign bootstrap response remain unresolved;
no draws are removed. Percentile intervals with few replicas have limited
coverage and do not establish equivalence.

The archive includes all randomized inputs, equilibration trajectories, seeds,
production outputs, source and manifests. Failed stages retain their elapsed
wall time and available artifacts. Completed campaigns distinguish disjoint
initialization/equilibration/production stage costs from nested engine/analysis
timings; the latter must not be added twice. Recording overhead and aggregation
are explicit. The reference includes Python integration overhead.

## Paired cross-method replicas

`lj_transport_replicas.py` consumes a **verified, completed** cosine replica
campaign and reuses every model/structural seed for native NVE GK and RNEMD
SPONGE/LAMMPS pairs. It requires matching executable hashes and nominal
production/startup/block durations. It copies frozen models/seeds; archived
upstream summaries retain the cosine observations and shared equilibration
cost. The original upstream archive remains necessary for its raw evidence.

```sh
python benchmarks/viscosity/lj_transport_replicas.py --spec /path/to/transport.json \
  --upstream /path/to/cosine-campaign --output /path/to/new-transport \
  --sponge /path/to/SPONGE --lammps /path/to/lmp
python benchmarks/viscosity/lj_transport_replicas.py --summarize /path/to/new-transport
python benchmarks/viscosity/plot_transport_replicas.py /path/to/new-transport --output /path/to/figure
```

The strict specification uses contract `sponge.lj.transport-replicas.v1`,
`duration_star`, `start_star`, `block_star`, `gk_sample_star`, `gk_audit_star`,
`gk_windows_star`, `reference_gk_window_star`, `rnemd_exchange`, `rnemd_bins`,
`rnemd_exclusions`, `bootstrap_draws`, and `bootstrap_seed`. The reference GK
window must belong to the retained grid; it is a comparison coordinate, not a
convergence claim. In this version of the campaign contract, RNEMD sampling
matches the exchange interval; this fixes the observation phase relative to
each exchange and must not be assumed to approximate a time average. Analysis
boundaries must align with both cadences and comprise an even number of blocks.

GK pools correlation-integral numerators and measured temperature separately,
then divides. This avoids silently averaging differently normalized finite-N
NVE trajectories; the mean of their separately normalized viscosities remains
an explicit diagnostic. RNEMD pools mass/momentum profiles and flux before
fitting and inverting the two signed branches. Whole-replica bootstrap indices
are shared across methods, engines, windows and exclusions. Undefined or
wrong-sign response draws are never removed to manufacture a bounded interval.
All intervals remain conditional; none certifies state matching or equivalence.

New RNEMD pair plans also request sparse coordinate/energy samples. The audit
subtracts SPONGE's C6 tail and active-pair cutoff energy and adds **full COM-free
centered kinetic energy**, preserving flow kinetic energy. LAMMPS records its
shifted energy before each exchange. This diagnoses integration/total-energy
drift separately from the exchange-only conservation ledger. Old plans without
this audit explicitly have no energy-audit result. New GK and RNEMD execution
records hash plans and raw energy/profile outputs; campaign manifests also
cover derived statistics. Failed stages retain elapsed time and artifacts.

For sampling-phase controls, `lj_rnemd_compare.py --stride 10 --exchange 200
--audit-stride 200` separates profile/temperature sampling, physical exchanges and
sparse total-energy audits. Both cadences must divide the run length, and the
audit cadence must be a multiple of the profile cadence. Omitted cadences
retain the historical exchange-period default. A dense run and its decimated
phases are observations of one trajectory, not independent replicas. Compare
the original trajectory and exchange ledger before attributing a difference
to sampling alone; increasing measurement frequency can perturb floating-point
execution paths. No existing archive or declared campaign is silently replaced.
LAMMPS permits reading its exchange-fix scalar only on exchange steps. Its
original impulse ledger stays at that cadence; pre-exchange values at the
next-or-equal exchange checkpoint reconstruct the piecewise constant cumulative
impulse at profile steps. There is no linear interpolation across impulses.
Temperature has a separate dense ledger, checked against shared exchange steps.

For analysis of **one already sampled trajectory**, the Python helper
`extract_rn(directory, sample_interval=10)` selects original frames in memory.
The requested interval must be a positive integer multiple of the original
interval, divide the run length, and align with analysis boundaries. Mass,
momentum, temperatures and cumulative impulse share the selected frames;
profile averages are recomputed with trapezoid endpoint weights. No source plan
or raw output changes. Omit the argument to preserve the original analysis.
Sampling variants share whole-replica bootstrap indices and are not additional
replicas. A sampling-sensitivity interval is not a physical-equivalence test.

## Molecular experimental references

`experimental_reference.py` evaluates the published IAPWS R12-08 water and
Michailidou et al. (2013) n-hexane viscosity correlations at **temperature and
mass density**. It supplies no equation of state, pressure or phase validation.
The output labels these as experiment-based correlations, not original
measurements or MD predictions. Water critical enhancement is omitted and its
specified critical rectangle is rejected. The n-hexane implementation follows
the original correlation's omission of critical enhancement.

```sh
python benchmarks/viscosity/experimental_reference.py \
  --fluid water --temperature 298.15 --density 998
python benchmarks/viscosity/experimental_reference.py \
  --fluid n_hexane --temperature 293.15 --density 659.29
python benchmarks/viscosity/compare_reference_data.py --output /path/to/new-report
```

Published verification tables supply 11 water and six n-hexane checkpoints in
`test_experimental_reference.py`, including dilute and dense states. Their
agreement verifies formula transcription and units; it is not a force-field
validation. Primary sources are [IAPWS R12-08](https://iapws.org/technical-guidance/release/viscosity.download)
and [the 2013 n-hexane correlation](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=913970).

`data/n_hexane_klein2019.csv` preserves all 11 n-hexane points from Table 2 of
[Klein et al. (2019)](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=928174),
including each point's expanded uncertainty (coverage factor 2). The companion
metadata records the source PDF hash and the authors' saturation approximation
with helium. The reported density is an input from literature to the surface
light scattering analysis. These are not constant-total-pressure 0.1 MPa data.
The report compares the two reference sources at their tabulated T/density;
it contains no SPONGE viscosity. Uncertainty bands remain separate because
their systematic components need not be independent.

Future MD comparisons must keep two routes separate: use experimentally
supported density to assess transport at a specified T/density, and use the
force field's equilibrated density at specified T/pressure to assess the
combined equation-of-state and transport prediction. Neither route is currently
validated by this reference preparation.

## Remaining transport gates

### Frozen SPC/E same-state checks

`water_model.py` creates a traceable three-site water fixture and compares its
LJ, Coulomb and full forces, potential energies and six pressure components
with LAMMPS. SETTLE/SHAKE initial constrained stress is checked against an
independent NumPy instantaneous constraint solve using the LAMMPS force. The
LAMMPS reference pressure includes only pair/kspace terms; its native finite-step
SHAKE virial is not used as an instantaneous reference.

```sh
python benchmarks/viscosity/water_model.py prepare --output /path/to/new-model
python benchmarks/viscosity/water_model.py static --model /path/to/new-model \
  --output /path/to/new-static --sponge /path/to/SPONGE --lammps /path/to/lmp
python benchmarks/viscosity/water_ewald_check.py --model /path/to/new-model \
  --output /path/to/new-diagnostic --sponge /path/to/SPONGE --lammps /path/to/lmp
```

Defaults are 216 molecules, 998 kg/m³, initial 298.15 K, 8 Å cutoff, PME
128³/256³ with `PM.Direct_Tolerance=1e-10`, and explicit Ewald `1e-10` with
`pair_modify table 0` on the LAMMPS side. These are diagnostic precision settings,
not a recommended production cost/accuracy choice. Preparation has `--cells`,
`--density`, `--temperature`, `--cutoff` and `--seed`; the static runner exposes
`--grids` and `--direct-tolerance`. The engine's global defaults are unchanged.

`model.json` records published nominal parameters, effective float32 masses,
charges and LJ coefficients, intramolecular exclusions, box rounding and every
input hash. Cross-engine charges absorb each engine's Coulomb unit constant.
The LJ table contains input C12/C6; the reference includes the loader's subsequent
float32 multiplication by 12/6. LAMMPS tails are off; its expected energy/pressure
receive SPONGE's C6-only tail once. Ordinary energy HDF5 fields retain output
formatting (`eff_pot` has `%.7e`; PM/LJ diagnostics only `%.2f`).

An initial 64³/128³, direct-tolerance `1e-7` trial failed the pure Coulomb pressure
gate. Independent real/reciprocal sums identified about 0.10 MPa of omitted
real-space virial, which increasing the FFT grid alone could not remove. The
diagnostic separates this term, compares 128³/256³ and both direct tolerances,
and preserves LAMMPS table 0/12 controls. The tighter follow-up retains the same
gates and is labeled exploratory. `Direct_Tolerance` bounds erfc(beta*rc)/rc,
not total pressure error; the LAMMPS kspace tolerance has different semantics.

The generated state is a randomly oriented molecular lattice, **not an
equilibrated liquid**. Only the initial constrained frame is cross-checked;
additional short output frames are not a dynamics-equivalence claim. This
supports frozen-state molecular mechanics, not a water viscosity or an
experimental agreement claim. Parameters follow the
[LAMMPS SPC/E model page](https://docs.lammps.org/Howto_spc.html) and
[Berendsen et al. (1987)](https://doi.org/10.1021/j100308a038).
The Coulomb table control follows [pair_modify](https://docs.lammps.org/pair_modify.html).

### Liquid transport validation still required

`water_pilot.py` prepares a bounded fixed-volume SPC/E trajectory with existing
SETTLE and middle Langevin, then starts a distinct NVE experiment. For example:

```sh
python benchmarks/viscosity/water_pilot.py nvt --model /path/to/model \
  --output /path/to/new-preparation --sponge /path/to/SPONGE \
  --steps 20000 --dt 0.0005 --stride 50
python benchmarks/viscosity/water_pilot.py nve --model /path/to/model \
  --seed-file /path/to/new-preparation/structural_seed.npz \
  --output /path/to/new-nve --sponge /path/to/SPONGE \
  --steps 2000 --dt 0.0005 --stride 20
```

The preparation records all samples, latter-half temperature/potential blocks,
molecular COM mobility and O–O RDF diagnostics. None automatically certifies
equilibration. Its final **trajectory** frame is the requested completed step;
the engine's extra terminal iteration does not supply a new seed. Native
coordinate/velocity files omit step zero even with `print_zeroth_frame=true`;
ordinary observables and GK stress include it. NVE analysis prepends the known
initial coordinates and verifies each cadence and completion journal.

The structural seed retains the native NVT velocity but does not reinterpret
it as integer-time NVE velocity. It draws fresh Maxwell velocities with the
explicit `--velocity-seed`, projects the complete molecular constraints and
removes mass-weighted COM velocity. There is no exact-temperature rescaling.
This defines a new experiment, not an NVT RNG restart or mode-switch continuity.
NVE recomputes forces at the actual requested grid, converts tangent velocity
to incoming half-step velocity, and verifies the initial projected temperature
after float32 storage. `--thermostat-seed` controls preparation separately.
The Python `snapshot_model(...)` helper exports a derived frozen fixture from
this state for the existing LAMMPS same-state gates.

NVE samples constrained GK stress every step and audits energy at `--stride`.
Its kinetic energy uses the constrained observer's COM-free DOF and actual
float32 temperature denominator. Analytic LJ/Coulomb cutoff shifts make the
reported audit potential continuous without changing force or trajectories;
the constant C6 tail stays included. Endpoint, range and slope are diagnostics,
not a time-step-convergence pass. Actual NVE temperature is measured and can
differ from the preparation target. Scripts preserve failed outputs and write
source/model/binary identities before execution. These pilot results do not
replace long trajectories, independent starts, sampling and size/precision scans.

The next measurements must preserve the method-specific diagnostics described
in the repository's GK, cosine and RNEMD documentation. Required work includes
independent liquid replicas, removal of
startup, drive/size/timestep/thermostat sensitivity, uncertainty propagation,
and predeclared equivalence limits. A wide interval covering a reference is
inconclusive, not a pass. Formal cost includes equilibration, startup, all scans,
production and analysis. Water and n-hexane experimental comparisons remain a
separate molecular-model validation stage.


### Fixed-bond n-hexane preparation pilot

```sh
python benchmarks/viscosity/hexane_model.py prepare --output /path/to/hexane-model
python benchmarks/viscosity/hexane_model.py static --model /path/to/hexane-model \
  --output /path/to/hexane-static --sponge /path/to/SPONGE --lammps /path/to/lmp
python benchmarks/viscosity/hexane_pilot.py --model /path/to/hexane-model \
  --static-force /path/to/hexane-static/sponge/force.dat \
  --output /path/to/hexane-preparation --sponge /path/to/SPONGE --duration 100
```

The default is 108 six-site molecules at 293.15 K and 659.29 kg/m³, with
five fixed C–C bonds per molecule, a 14 Å LJ cutoff, graph-distance ≤3
exclusions, and explicit quantized TraPPE-UA candidate parameters. The ordered
all-trans initial lattice is not an equilibrated liquid. The model metadata
records parameter provenance and the 355.03 K torsion coefficient, distinguished
from a rounded 355.0 K transcription. The directly retrieved author preprint
supports LJ/mixing/exclusions/fixed-bond details; it is not the original 1998
parameter-paper PDF. Native C6-only tail conventions are matched explicitly in
the static LAMMPS comparison.

The pilot uses the constrained cosine integrator with zero acceleration,
0.5 fs steps and a 0.1 ps middle-Langevin bath. Inputs remain ordinary engine
keys plus the existing viscosity table. Independent per-chain five-row mass
projections audit full and thermal kinetic energies at force time, including
COM and tangent-projected cosine removal. A separate dense-SVD test checks
this reference. Complete native trajectories, profile data, checkpoint and
source/input/binary hashes are retained. The first static force belongs to q0;
the static file also contains a later dt=0 frame that is not substituted for it.

Sampling is 0.01 ps for flow profiles and 0.1 ps for native q/s/F and ordinary
observables. The final double checkpoint is checked against the final native
periodic image with a 3e-6 Å gate. The default 100 ps run has a 14400 s timeout.
Final snapshots retain incoming half velocity and reconstructed physical
velocity separately; they are not interchangeable restart files.

Structural diagnostics use consecutive-bond periodic reconstruction, molecular
COM RDF and mobility, chain-axis nematic order, and a declared |torsion|<120°
gauche bin (all-trans is 180°). They report latter-half temperature, potential,
pressure, order and conformer blocks. Unshifted energy/heat balance includes
cutoff crossings and is diagnostic only. This bulk pilot does not independently
validate dynamic pressure or certify liquid equilibration, diffusion, viscosity,
experimental pressure, or experimental agreement.


`hexane_nve.py` exports a preparation snapshot and runs a separate NVE/GK
numerical pilot:

```sh
python benchmarks/viscosity/hexane_nve.py snapshot --model /path/to/hexane-model \
  --preparation /path/to/hexane-preparation --output /path/to/hexane-snapshot
python benchmarks/viscosity/hexane_model.py static --model /path/to/hexane-snapshot \
  --output /path/to/snapshot-static --sponge /path/to/SPONGE --lammps /path/to/lmp
python benchmarks/viscosity/hexane_nve.py run --model /path/to/hexane-snapshot \
  --static /path/to/snapshot-static --output /path/to/new-nve \
  --sponge /path/to/SPONGE --duration 2 --dt 0.0005
```

The snapshot checks the completed preparation's raw-file identities, wraps double
positions before quantization, and projects the physical velocity at the stored
geometry. It writes zero static velocity and retains physical velocity separately.
The new force probe and each requested timestep define new incoming half velocity;
the preparation's old staggered velocity is not reused across this experiment.
GK uses `3N-Nconstraints-3`, without cosine-mode subtraction. Ordinary SHAKE NVE
and the double constrained-flow integrator remain distinct numerical paths.

Every-step GK output is audited against native q/s/F at 0.01 ps intervals.
Independent per-chain constraint reactions augment the LAMMPS mechanical stress
for the initial six-pressure gate. A global squared-distance constraint reference
checks the local normalized-distance implementation. The LJ cutoff-offset
bookkeeping includes retained intramolecular 1-5 and more distant pairs, and
excludes graph-distance ≤3. Full kinetic energy is used for energy drift;
COM-free kinetic energy is checked against the reported GK temperature.
Time-step comparisons report energy and trajectory differences, with no automatic
energy-convergence or short-trajectory transport acceptance.


Explicit timestep conversion is centralized in `water_model.internal_timestep`:
MD core multiplies the parsed double input by its float time constant, then
stores the result as float. Precasting the input dt introduces a different
rounding and is not used by the water/hexane handoff or audit helpers. Frozen
historical scripts retain their original version; reanalysis must identify it.

A 108-molecule, 2 ps hexane pilot at 0.5/0.25/0.125 fs passed the output,
kinetic and initial pressure checks but did **not** establish ordinary SHAKE
NVE timestep convergence: energy drift increased at the smallest step. The
ordinary float-coordinate update and constraint correction divided by dt remain
under investigation. Zero-OU double constrained-flow controls are a separate
integrator path, with an initial coordinate projection; they are not ordinary
NVE/GK support or an exactly identical initial-state comparison. Passing the
pilot's listed checks must not be interpreted as resolving this limitation.

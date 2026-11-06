# HMR stability and sampling experiments

These optional local GPU experiments measure finite-duration stability and
observable-specific sampling efficiency. Successful exit is not scientific
acceptance. Each campaign freezes its binary and original inputs.

## Protein–ligand–water campaign

The current fixture is the physical jmc_28 endpoint of the TYK2
`jmc_28__jmc_27` edge: 31,247 atoms, 288 protein C-alpha atoms, 39 ligand atoms,
and 8,829 waters. It uses the original native topology, charges and coordinates,
not the dual-topology intermediate. Private fixture files and trajectories stay
under ignored `outputs/`; provenance records their SHA-256 hashes. This suite
does not download or distribute them.

`tests/run_campaign.py` reuses the shared benchmark runner and output preparation.
It generates fresh constrained Maxwell velocities for each comparison branch;
equilibration stages can continue velocities. HMR uses `ratio_xh1=2.0` and
`ratio_other=2.5`, excludes water, and keeps the same constraint set as the
ordinary-mass controls. Timestep values are in fs; run duration is in ps.
The thermostat is Middle Langevin, with tau 1 ps. NPT uses the Monte Carlo
barostat at 1 bar. NVT pressure is available; MC NPT pressure is **not** evaluated
by this engine revision, so its zero output must not be interpreted as pressure.

An initial diagnostic of engine commit `31aa55d` found water geometry errors with
the default double-precision GPU SETTLE path even at ordinary masses and 1 fs. The follow-up campaign
therefore explicitly uses `settle_disable=true`: all original solute and water
constraints are handled by double-precision SHAKE, with 100 iterations and
relative tolerance 1e-6. This is an alternative solver for the same constraints;
results must be identified as **all SHAKE**, and do not validate default SETTLE.
The diagnostic outputs are retained separately. This campaign's engine binary
was kept frozen. A separate corrected build caps double triangle kernel blocks
at 256 threads: the original 1024-thread blocks exceeded their register budgets.
Single precision passed a separate 20 ps diagnostic. The existing independent
constraint regression fails on the frozen GPU binary and passes on the corrected
CPU/CUDA builds, alongside both focused HMR tests. Corrected protein runs named
`fixed_settle_*` use default double SETTLE and are kept out of production groups.
In the corrected build, ordinary 1 fs and HMR 1/4/5 fs each passed a separate
50 ps NVT run: mean temperatures 300.23–300.61 K and maximum sampled relative
constraint errors below 1.8e-5.

For a new campaign, `--stage prepare --disable-settle` uses SHAKE throughout
minimization, restrained relaxation, and 200 ps of unrestrained NPT, and writes
`shake_equilibrated`. Inspect its output before proceeding. The saved campaign
also retains its earlier SETTLE diagnostics and its exact prepared starting
coordinates; reusing those coordinates reproduces the reported initial state.

Example after preparing and verifying the `shake_equilibrated` starting state:

```bash
python benchmarks/performance/hmr/tests/run_campaign.py \
  --root PATH_TO_CAMPAIGN --binary build-dev-cuda13/SPONGE \
  --stage screen --source shake_equilibrated --disable-settle \
  --tag allshake_ --duration-ps 50
python benchmarks/performance/hmr/tests/analyze_campaign.py \
  --root PATH_TO_CAMPAIGN --reference shake_equilibrated --pattern 'screen_*'
python benchmarks/performance/hmr/tests/run_campaign.py \
  --root PATH_TO_CAMPAIGN --binary build-dev-cuda13/SPONGE \
  --stage production --source shake_equilibrated --disable-settle \
  --tag allshake_ --dts normal:1,1,4,5 --seeds 712,713,714 \
  --modes nvt,npt --duration-ps 200
python benchmarks/performance/hmr/tests/analyze_campaign.py \
  --root PATH_TO_CAMPAIGN --reference shake_equilibrated \
  --pattern 'production_*' --rdf
python benchmarks/performance/hmr/tests/report_campaign.py \
  --root PATH_TO_CAMPAIGN
```

Validate a separately built SETTLE correction without mixing its trajectories
into the frozen production comparisons:

```bash
python benchmarks/performance/hmr/tests/run_campaign.py \
  --root PATH_TO_CAMPAIGN --binary build-hmr-cuda-fix/SPONGE \
  --stage validate-settle --source shake_equilibrated --duration-ps 50
python benchmarks/performance/hmr/tests/analyze_campaign.py \
  --root PATH_TO_CAMPAIGN --reference shake_equilibrated \
  --pattern 'fixed_settle_*'
```

The analysis reuses `Extractor`, native coordinate/box readers, and the shared
water RDF implementation. It checks complete output records, finite values,
sampled constraint residuals, protein C-alpha RMSD/Rg, ligand heavy-atom RMSD
after protein fitting, protein–ligand native contacts, backbone angle
distributions, density, and optional water O–O RDF. Selections retain **original**
topology masses even for HMR. Duplicate bond targets follow the engine's final
record per atom pair, including this endpoint's zero-force ligand placeholders.

Statistics exclude step zero and the first 20% of each comparison run. Five block
means give a rough within-run uncertainty estimate; independent random seeds are
needed for between-run uncertainty. NVE `U + DOF*kB*T/2` drift uses the printed
leapfrog kinetic temperature and is an approximate diagnostic. Constraint errors
are sampled at approximately 1 ps intervals from float32 coordinates; they are
not a bound on every integration step or the solver's internal double precision.
Short replicas cannot establish converged protein ensembles or universal HMR
timestep limits. No equivalence conclusion follows from an insignificant test.

The completed local campaign has 24 production trajectories (three seeds,
200 ps per condition, NVT and NPT; 4.8 ns total). All pass the catastrophic
screen. HMR 5 fs is the largest stable timestep tested; 6 fs and higher tested
values fail the 50 ps screen with nonfinite output. HMR 4/5 fs nevertheless show
finite-timestep shifts in potential energy and NPT density. See
[HMR validation limits](../../../docs/hmr.md#validation-and-its-limits).
Each `analysis_summary.*` describes the last analysis command's pattern;
`report.md` and `replicate_*.json` aggregate all available production analyses.
The ignored output directory retains per-case options, binary/input hashes,
failed diagnostics, trajectories, analyses and figures; private fixture files
are not committed.

## Alanine dipeptide sampling efficiency

`tests/run_sampling.py` is a separate, optional sampling experiment using the
public `performance/sits/statics/ala2_sits` native fixture (ACE–ALA–NME,
606 waters, 1840 atoms). It does not enable SITS. The fixture's topology and
parameters are used as supplied; no force-field family is inferred from its name.
This experiment does not scan timestep stability limits.

The comparison has three conditions: ordinary masses at 2 fs, HMR at 2 fs,
and HMR at 4 fs. The first comparison measures the mass effect; the second
measures the timestep effect at fixed masses. HMR factors are 2.0/2.5, with
water excluded. The original constraint selection is retained: double SETTLE
handles the rigid triangles/pairs, and SHAKE handles the remaining pairs.
NVT uses 300 K, Middle Langevin tau 1 ps and a common prepared volume.
Three Maxwell velocity seeds start at the same prepared coordinates. Runs
execute serially in a balanced order to reduce timing drift between conditions.

```bash
python benchmarks/performance/hmr/tests/run_sampling.py \
  --root PATH_TO_SAMPLING --binary build-hmr-cuda-fix/SPONGE --stage prepare
python benchmarks/performance/hmr/tests/run_sampling.py \
  --root PATH_TO_SAMPLING --binary build-hmr-cuda-fix/SPONGE --stage pilot
# Inspect equilibration and pilot diagnostics before production.
python benchmarks/performance/hmr/tests/run_sampling.py \
  --root PATH_TO_SAMPLING --binary build-hmr-cuda-fix/SPONGE \
  --stage production --duration-ps 2000
python benchmarks/performance/hmr/tests/analyze_sampling.py \
  --root PATH_TO_SAMPLING
python -m pytest benchmarks/performance/hmr/tests/test_sampling_statistics.py -q
```

Preparation comprises minimization, 50 ps NVT warmup, 400 ps MC-NPT density
equilibration at 1 fs and 100 ps NVT at 2 fs. Production regenerates velocities
using each condition's effective masses. The binary and input hashes are frozen
under ignored `outputs/`; existing incompatible or unfinished cases are never
silently overwritten. Production duration/seeds are recorded before the first
run and cannot change on resumption. Analysis requires all nine planned runs.

The primary observable is `I[cos(psi)>0]`; secondary observables are sine/cosine
of phi/psi and `I[phi>0]`. CV samples have the same 0.1 ps physical interval
in all conditions. First 20% of each production run is discarded. Per-trajectory
FFT autocorrelations and Geyer initial-positive/monotone paired sums estimate
statistical inefficiency `g` and `ESS=N/g`. Constant/unvisited indicators have
undefined ESS. Known AR(1) correlation, independent samples, periodic observables
and separated chains provide independent estimator tests.

Report both ESS per physical ns and ESS per actual wall hour. The latter includes
SPONGE initialization, production burn-in, output and completion checks, but
excludes preparation, case creation and subsequent analysis. It estimates a
rate from equal-physical-duration runs; these are not fixed-wall-time runs.
Nominal three-seed log-ratio intervals describe repeatability, not a formal
confidence interval for ESS. A throughput increase or an interval containing 2
does not establish twofold sampling efficiency.

Analysis also checks complete finite output, intended runtime HMR/constraint
configuration, constant common NVT box, sampled X-H constraint errors, and
phi/psi against an independent coordinate-based atan2 reference. Populations,
ordinary split Rhat, core-state alternations, rare-state warnings and finite-sample
phi/psi occupancy maps accompany the efficiency plots. These diagnostics do not
prove a converged ensemble, exact timestep equivalence, preserved kinetics or
sampling of a slow state absent from all trajectories.

The [2026-11-07 public-fixture result](../../../docs/hmr-sampling/README.md)
records nine 2 ns trajectories, figures, per-run statistics and hashes. It shows
observable-dependent rate gains and a potential-energy shift, rather than
establishing sampling-distribution equivalence.

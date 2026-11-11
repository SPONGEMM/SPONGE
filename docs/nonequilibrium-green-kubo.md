> 本分支输入与代码职责已更新，当前接口见 [动力学组件](dynamics-components.md)。下文保留旧版方法推导/验证记录，其中 `[viscosity]`、`[[coupling]]` 需先由 Python 工作流展开。

# Green–Kubo stress sampling and analysis

Implemented on `feature/nonequilibrium-coupling`. This is a sampling and numerical
analysis path, not a claim of converged liquid viscosity or agreement with
LAMMPS/experiment. See [the general framework](nonequilibrium-coupling.md) and
[cosine flow](nonequilibrium-cosine.md).

## Input

Keep the existing topology, force-field, integration and output settings. Add:

```toml
[viscosity]
method = "green_kubo"
sample_interval = 1       # integration steps; default 1
output = "viscosity.h5"  # default; must be a new file
```

This method adds no drive. Its cadence is independent of mdout and trajectory
output. Only its three keys are accepted; cosine parameters and explicit
`[[coupling]]` drivers are rejected. The file may not overwrite another file or
collide with an explicit input/output. Run a continuation in a new directory or
give the segment a new output name.

Current scope: single MPI rank, neutral periodic fixed box, NVE, positive
particle masses, no virtual sites, variable charges, SITS, velocity clipping,
walls, external bias or force plugins. Use a separately equilibrated liquid;
NVE production avoids unvalidated thermostat effects on stress correlations.
The input velocity is SPONGE's incoming leapfrog half-step velocity. Existing
initialization/restart conventions remain in effect.

Unconstrained atoms and independent SETTLE/SHAKE/RATTLE distance constraints are
supported. Constraints require an orthogonal box and positive dt, and use the
[instantaneous tangent observer](constraint-stress-observer.md) without new
input keys. LINCS/CCMA and virtual sites remain rejected. A constrained stress
stream is not by itself a validated water viscosity prediction. The new
`constrain_mode="RATTLE"` mechanical path retains canonical double coordinates
and staggered velocities while using native float forces; see its
[initialization, restart and scope](constrained-nve-core.md). Its geometry
tolerance is `2e-12`; its solver metadata records `rattle_pairs`,
`precision=canonical_double_native_force_v1` and `step=BA_full_drift`.
PME remains the internal electrostatic solver for fixed charges. Its single-rank virial is collected with the other forces.
Reciprocal forces must be evaluated every step. Net charge above `1e-4` in
native scaled-charge units is rejected until background-charge stress is
validated. Charged inputs must enable both direct and reciprocal paths; a
Coulomb-only model supplies a zero-coefficient LJ table because the current
direct Coulomb implementation shares the LJ kernel.

## Stress definition and timing

Immediately after the force evaluation at position `r_n`, before the integration
step or coordinate output wrapping, the unconstrained path reconstructs

\[
v_{i,n}=v_{i,n-1/2}+\frac{\Delta t}{2m_i}F_{i,n},\qquad
c_{i,n}=v_{i,n}-\frac{\sum_jm_jv_{j,n}}{\sum_jm_j}.
\]

The symmetric physical pressure tensor is

\[
P_{\alpha\beta}(t_n)=V^{-1}\left[
\sum_i m_i c_{i\alpha,n}c_{i\beta,n}+W_{\alpha\beta}(r_n)\right].
\]

SPONGE's internal `LTMatrix3` stores a symmetric sum in each off-diagonal
virial entry: physical `xy`, `xz`, `yz` therefore use half those stored values.
The diagonal entries require no factor. The existing LJ dispersion tail is
included in the diagonal pressure; it does not contribute to the shear entries.
All potential and kinetic terms in this output refer to the same time layer.
Existing ordinary pressure/temperature output is not redefined by this sampler.
The RATTLE mechanical runtime itself uses the same instantaneous observer for
its pressure and temperature.

With distance constraints, mass-metric projection puts the force-centered
velocity on the tangent space. A second solve enforces the instantaneous
acceleration constraint and contributes its physical pair virial. Use
`3N−Nc−3` thermal DOF for Nc independent distances. This is a continuous-time
constraint-force observer with measured finite-step error, not the previous
step's SHAKE correction virial. Full equations and numerical limits are in
[the observer design](constraint-stress-observer.md).

The sampler uses the existing `Measurement` and `OutputSession` objects plus an
instance of the existing `ObservableH5Writer`. It does not add a second restart
format, clock or integrator. It currently gathers arrays to the host and reduces
in double precision: performance optimization is deferred pending profiling.

## File contract

The HDF5 file uses the existing output schema and completion journal. Under
`/observables/all`, common `step` and `time` index these scalar `*/value` arrays:

| Fields | Units and meaning |
|---|---|
| `pxx`, `pyy`, `pzz`, `pxy`, `pxz`, `pyz` | Pa; physical pressure components |
| `temperature` | K; centered COM-subtracted kinetic temperature, `3N−Nc−3` DOF (`Nc=0` without constraints) |
| `volume` | nm³ |
| `com_vx`, `com_vy`, `com_vz` | nm/ps; removed center-of-mass velocity |
| `time` | ps; force-evaluation time, including the initial/final sample if due |

Unconstrained provenance records contract `sponge.green_kubo.centered_nve.v1`,
the stress convention, resolved commands, atom count, timestep and interval.
Constrained sampling uses `sponge.green_kubo.constrained_nve.v1` and adds
`constraint_observer`, `constraint_count`, `thermal_dof`, `constraint_solver`
and `constraint_length_tolerance`. Per-frame `constraint_wxx/wyy/wzz/wxy/wxz/wyz`
are physical virials in kcal/mol; `constraint_length_residual` is dimensionless,
`constraint_tangent_residual` is nm/ps and `constraint_acceleration_residual`
is nm/ps². These residuals describe the observer solve, not the native
SHAKE stopping criterion. A dynamic
restart keeps global step/time, so cadence does not restart at zero. No online
correlation state is stored. Constrained dynamic restarts additionally require
the same hashed observer/constraint/integration identity; old checkpoints or
changed constraints require structural load. Segment endpoints can overlap; the present analyzer
accepts one file only and does not concatenate segments or independent replicas.

Pressure conversion uses `4184 / N_A / 10^-30` from kcal/mol/Å³, with exact SI
Avogadro constant. The kinetic temperature uses SPONGE's existing internal `kB`;
the analysis uses exact SI `kB = 1.380649e-23 J/K`. Freeze engine constants along
with the force field in a cross-engine benchmark.

## Explicit-window analysis

The following command illustrates syntax; its times are not recommended liquid
sampling lengths:

```sh
python scripts/analyze_green_kubo.py viscosity.h5 \
  --start-ps 100 --window-ps 10 --block-ps 100 --output analysis
```

Requires NumPy and h5py. The output directory must be new. The reader rejects
unfinalized/failed files, length/completion mismatches, wrong units, nonfinite
values, nonuniform sample spacing, nonpositive temperature or changing volume.

For an isotropic equilibrium liquid, each shear component obeys

\[
\eta_{\alpha\beta}(t)=\frac{V}{k_B\langle T\rangle}
\int_0^t\langle P_{\alpha\beta}(s)P_{\alpha\beta}(0)\rangle\,ds.
\]

The implementation uses zero-padded FFT correlation, `N−lag` origin counts and
trapezoidal integration in SI units. Time windows round down to sample bins;
requested and actual times are both reported. After the explicit startup cut,
data are split into equal contiguous blocks; an incomplete trailing block is
dropped and counted. Each block must contain at least `2*max_lag+1` samples.
The default assumes theoretical zero mean shear stress; mean and RMS stress are
reported. Optional `--demean` subtracts the sample mean within each block and
warns about the resulting finite-block bias.

`summary.json` contains component integrals, per-block scalar estimates, their
mean/standard error, conditional percentile-bootstrap 95% interval, exact
analysis settings, input/script SHA256 and dependency versions. `correlation.csv`
contains block-averaged component correlations and cumulative integrals. The
three tensor components are averaged **within** each block, preserving their
covariance in resampling. The interval assumes approximately independent blocks;
it does not cover correlation-tail truncation, temperature estimation, force-field
error or finite-size/time-step bias. Two blocks are the minimum to compute the
diagnostic; even ten blocks do not establish independence or convergence.

No plateau detection or scientific pass/fail is automated. Repeat analysis with
predeclared windows/block lengths and use independent trajectories, timestep,
size and sampling-interval checks before a viscosity claim. The file hash and
commands are not a full force-field provenance manifest; formal benchmarks must
also preserve all inputs and software/hardware identities.

## Verification and remaining acceptance

Real executable tests check six ideal-particle pressure components, COM removal,
temperature, units, independent cadence, unchanged native trajectories, analytic
LJ virial including the diagonal tail, harmonic bond virial, and force-centered
kinetic stress at finite timestep. A neutral two-charge PME cell is checked in
all six components against independent double-precision direct Ewald energy
strain derivatives, using even/odd FFT grids and two finite-difference steps.
Full versus split dynamic restart compares equal-time samples exactly.
Numerical tests compare FFT with direct multiple-origin sums, check trapezoid SI
scaling and verify that correlated tensor components are not counted as separate
replicates. An end-to-end CLI test analyzes the ideal stress stream; its result
is intentionally not interpreted as liquid viscosity.

This check exposed and corrected an existing reciprocal virial defect: the
accumulation sign was reversed, the energy half-factor was applied twice, and
off-diagonal coefficients did not follow the symmetric-sum storage convention.
Both initial and updated-box coefficient construction are corrected; odd FFT
lengths no longer treat their last plane as a Nyquist boundary. This correction
also changes ordinary PME pressure and pressure-based barostat behavior; it is
not confined to GK. Forces/energy kernels are unchanged. Molecular NPT and GPU
runtime validation remain necessary before release.

Constrained runtime tests additionally compare integrated/offline rotor stress,
interleaved-ID mixed solver topology, analytic central-force cancellation,
PME with/without exclusions, DOF metadata and exact restart/cadence behavior.
Still required: broader molecular bonded/PME and cross-engine stress coverage,
statistical convergence and acceptance of the existing same-model
GK/cosine/RNEMD benchmarks, plus experimental predictions. CUDA compilation alone
does not establish GPU runtime correctness.

Reference for the equilibrium stress-correlation formula and the distinction
from driven methods: [LAMMPS viscosity guide](https://docs.lammps.org/Howto_viscosity.html).
See also Hess, *J. Chem. Phys.* **116**, 209–217 (2002),
[DOI:10.1063/1.1421362](https://doi.org/10.1063/1.1421362).

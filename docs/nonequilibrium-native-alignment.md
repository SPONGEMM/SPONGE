> 本分支输入与代码职责已更新，当前接口见 [动力学组件](dynamics-components.md)。下文保留旧版方法推导/验证记录，其中 `[viscosity]`、`[[coupling]]` 需先由 Python 工作流展开。

# Native LAMMPS alignment

Validation follows model/force/stress, finite-step dynamics, and finally
transport statistics. The experimental viscosity is not an implementation
target. A static or short-trajectory pass does not certify a viscosity.

## Unconstrained LJ

`benchmarks/viscosity/lj_model.py static` checks energy, force and all six
virial components against both native LAMMPS and a direct double pair sum.
`dynamics` checks the explicit staggered-to-centered velocity handoff and
short NVE evolution. Existing `lj_gk_compare.py` and `lj_rnemd_compare.py`
use native LAMMPS integration and measurements.

`lj_cosine_compare.py` uses a custom Python projected Middle thermostat on
the reference side. Its purpose is splitting validation; it must not be
presented as an unmodified LAMMPS thermostat.

The separate `lj_native_cosine.py` runs **native** `fix nvt`,
`compute viscosity/cos`, and `fix accelerate/cos`. It accepts a completed
paired-cosine directory, verifies its plan/input/raw hashes, and copies its
LAMMPS data including integer-time velocities. It preserves all three
estimators and the original half-open block windows. Example:

```sh
python benchmarks/viscosity/lj_native_cosine.py \
  --prior /path/to/completed/paired \
  --output /path/to/new/native-reference --lammps /path/to/lmp
```

This is a comparison of two different thermostats: native Nose-Hoover with
the raw cosine bias, versus SPONGE projected Middle Langevin preserving COM
and the joint cosine mode. Native and joint-projection temperatures are
recorded separately; the latter provides a common observable. Equal target
temperatures or similarly named damping parameters do not establish equal
thermal states or equal finite-step dynamics.

The native benchmark is restricted to the existing single-mass LJ fixture.
Raw position/velocity dumps independently verify the native Fourier response,
COM-corrected response, least-squares response, native temperature (the frozen
default 3N-3 DOF), and joint temperature (3N-4 DOF). No Python integration
callback is loaded into LAMMPS. Statistical equivalence still requires
independent replicas and drive, size, time and thermostat sensitivity checks.

## Constrained SPC/E water

`water_model.py static` separates LJ, Coulomb and full six-component stress.
SPONGE PME and native LAMMPS explicit Ewald use matched effective charges,
exclusions, cutoffs and documented tail corrections. Their tolerance inputs
have different meanings. `water_ewald_check.py` supplies a separate direct
Ewald diagnostic; increasing mesh size alone does not remove real-space
truncation error.

`water_constrained_pair.py` compares SPONGE RATTLE with unmodified LAMMPS
`rigid/nve/small molecule`. Three fixed distances in a three-site water
molecule define the same rigid geometry. The integrators differ, so compare
bounded errors and timestep sensitivity rather than demand identical long
trajectories. Continuous constraint reaction stress is reconstructed
independently; the native discrete fix virial is preserved separately.

## Fixed-bond hexane

The current six-site hexane model has five connected constrained backbone
bonds with flexible angles and torsions. Native LAMMPS 29Aug2024 RATTLE
rejects this topology with `Shake clusters are connected`. Making the whole
chain rigid or replacing its constraints with harmonic bonds changes the
model. This support limitation must not be reported as a viscosity result.

## Current evidence boundary

The project-level `notes/evidence/native-alignment-20261028` retains fresh LJ
static/dynamics checks, water static checks and two additional water starts
at three timesteps, a repeated native hexane rejection, and six native LJ
cosine references. The latter give a SPONGE/native LAMMPS Fourier viscosity
ratio about 1.099, with conditional whole-replica 95% interval [1.022, 1.239].
This does **not** satisfy the ±5% equivalence criterion. Numerical checks
pass in their declared scope; full cross-method liquid transport validation
remains open. No force-field parameters, analysis windows or acceptance
thresholds were adjusted to match experiment.

> 当前模块分工和输入见 [动力学组件](dynamics-components.md)；本文保留原数值推导。

# Constrained force-time stress observer

`nonequilibrium/constraint_stress.{h,cpp}` is a host numerical building block
used by Measurement for SETTLE/SHAKE/RATTLE Green–Kubo sampling. It adds no user
input keys and does not alter coordinates, stored velocities, forces,
thermostats or barostats. This is a bounded numerical implementation;
converged molecular liquid viscosity and experimental agreement remain open.

## Equations and conventions

Let each independent distance constraint be `g_a(q)=|q_i-q_j|-l_a=0`.
Use the minimum-image displacement `r_a` and unit vector `e_a=r_a/|r_a|`.
The Jacobian row has `+e_a` on atom i and `-e_a` on atom j. With diagonal mass
matrix M and the unconstrained force F at q, form

```text
w = s_incoming + (h/2) M^-1 F
A = J M^-1 J^T
A beta = -J w
u = w + M^-1 J^T beta

c_a = (|u_i-u_j|^2 - (e_a dot (u_i-u_j))^2) / |r_a|
A lambda = -J M^-1 F - c
C = J^T lambda
W_constraint = sum_a lambda_a |r_a| e_a outer e_a
```

The first solve is the mass-metric tangent projection. The second solves the
continuous acceleration constraint at that same state: `J M^-1(F+C)+c=0`.
This is an **instantaneous** constraint force, not the finite-step SHAKE
position-correction impulse divided by h. No equality with the engine's native
finite-step constraint virial is assumed. The remaining O(h²), solver and
coordinate-storage errors must be measured for each supported integration path.

Remove only the global mass COM from u for kinetic stress. Add physical
unconstrained virial and W_constraint, then divide by volume. Components are
`xx,yy,zz,xy,xz,yz`, without the engine LTMatrix3 off-diagonal doubling.
The temperature DOF is `3N - number_of_independent_constraints - 3`; the caller
uses `2K/(DOF*kB)`. Projection preserves molecule-relative translation and
thermal fluctuations; there is no kinetic-energy rescaling.

Inputs use one consistent internal unit system (SPONGE: Å, Da, internal time,
kcal/mol); pressure output is kcal/mol/Å³. Conversion to Pa belongs at the
Measurement boundary. Pairwise central constraint virial is independent of
origin and per-atom periodic wrapping when the minimum image is unique.

The kinetic/constraint-DOF and pressure conventions agree with the definitions
in the [GROMACS MD manual](https://manual.gromacs.org/documentation/2026.0/reference-manual/algorithms/molecular-dynamics.html).
SHAKE position constraints and RATTLE velocity constraints are distinct, as
described in the [LAMMPS SHAKE/RATTLE documentation](https://docs.lammps.org/fix_shake.html).
The equations above are the present observer design, not a claim that either
engine implements this identical observer.

## Numerical scope

- Fixed orthogonal periodic box, positive finite masses, stable common atom
  indexing, independent distance constraints. No virtual sites or MPI mapping
  is performed here.
- Cholesky solves each connected constraint component separately. This avoids
  a dense matrix spanning all water molecules. The current host path rejects
  components over 256 constraints; it is not a polymer-scale solver.
- Duplicate constraints, inconsistent dimensions, nonfinite input, ambiguous
  images, excessive length residual and dependent/ill-conditioned Jacobians
  are rejected. No regularization silently changes the constraint rank or DOF.
- The caller must supply a justified length tolerance; the observer reports
  observed length, tangent and acceleration residuals. It cannot repair an
  invalid input trajectory.

## Tests and executable probe

`test_constraint_stress` checks an unequal-mass rotating dimer, radial and
tangential force contributions, all six stress components, a rigid three-site
rotor, separate molecules, atom permutation, periodic wrapping, rank rejection,
and projected Maxwell energy mean/variance. Its `--stream` mode is test-only.

```sh
python benchmarks/viscosity/constraint_observer_probe.py \
  --sponge build-dev-cpu/SPONGE \
  --observer build-dev-cpu/tests/control/test_constraint_stress \
  --output /path/to/new-evidence
```

The probe enables the integrated HDF5 observer in real SETTLE and SHAKE runs,
compares it with this C++ observer applied offline to native coordinate,
velocity and force files, and checks the continuous analytic solution. It uses free planar dimer and triangle
rotors, each with three predeclared steps and analytic continuous rotation.
The incoming velocity is the correct chord velocity with
`theta=asin(h*omega)`, so its tangent projection is the specified initial
angular velocity. Positions/velocities are quantized exactly as the executable
input; this small initial error is included, not fitted away. A triangle here
is a mechanical fixture, not a water force field or experimental prediction.

The expected error reduction under h→h/2 is checked with a predeclared interval
[2.8,5.2], together with energy/angular-momentum and constraint residual gates.
This is a bounded second-order test, not a liquid viscosity result. The source
snapshot and binary identities are recorded separately. The output directory
must be new; failures retain completed runs and their logs.

## Runtime integration and remaining acceptance

`InitialStateBuilder` captures every distance before SETTLE consumes its
pairs/triangles. After solver setup, Measurement builds the plan from global
masses and the complete list. At every observation it gathers position,
incoming velocity and unconstrained force using `dd.atom_local[local]=global`,
checks the owner permutation/masses, and reduces only the current per-atom
unconstrained virial. It never adds the previous step's `sys.d_stress`.

The existing `[viscosity] method="green_kubo"` automatically selects this path
when `constrain_mode` is SETTLE, SHAKE or RATTLE. It requires positive dt,
a fixed orthogonal neutral periodic NVE system and independent active
constraints; the other GK restrictions still apply. Unhandled SETTLE pairs,
LINCS/CCMA, virtual sites, constrained RNEMD and constrained flow thermostats
remain rejected by this GK path. Geometry tolerance is fixed at relative `1e-4`
for native SETTLE/SHAKE and `2e-12` for canonical double RATTLE, and is recorded.
RATTLE supplies global canonical q/s; native forces and virials still come from
the actual force engine. Its separate mechanical owner is documented in
[constrained NVE](constrained-nve-core.md).

The HDF5 contract is `sponge.green_kubo.constrained_nve.v1`. It records
observer version, solver partition/settings, constraint count and thermal DOF,
plus six constraint virials and length/tangent/acceleration residuals per frame.
The analyzer validates this metadata and the diagnostic shapes/units; legacy
unconstrained v1 is unchanged. See [the GK contract](nonequilibrium-green-kubo.md).

The observer has no history. Dynamic checkpoints nevertheless preserve its
configuration identity in the existing hashed coupling-state map: dt in ps
and internal units, box, masses, complete ordered pairs/lengths, observer
version/tolerance, solver partition and effective SHAKE parameters. It also
binds the selected legacy constraint arithmetic precision; changing between
single and double requires structural loading. Checkpoints from before this
precision identity was recorded likewise require a new structural experiment.
Restore is deferred until constraint setup completes; changing/removing constraints
or loading an older checkpoint without this identity requires structural load.
Sampling cadence/output names may change without altering identity or motion.

Real tests cover interleaved global atom IDs with a mixed SETTLE triangle and
SHAKE chain, all six stresses against independent NumPy solves, central LJ
force/constraint cancellation, and neutral two-charge PME with/without the
pair exclusion against independent double-precision Ewald forces and energy
strain derivatives. PME is a bounded two-charge test (128³ grid for the
constrained cancellation case), not a liquid-water electrostatics acceptance.
Continuous/split constrained restarts and observation-cadence changes preserve
states and common stress samples exactly on the tested CPU fixtures.

Still required: broader interacting molecular and cross-engine stress checks,
liquid finite-step/size/window/uncertainty convergence, water/hexane experimental
predictions, and GPU execution. The host observer's allocation/copy cost also
needs measurement before any cost-to-precision claim.

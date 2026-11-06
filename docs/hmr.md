# Hydrogen mass repartitioning

SPONGE can apply hydrogen mass repartitioning (HMR) once at startup while
reading an ordinary, unrepartitioned topology. The input files and original
topology masses are preserved. The implementation uses the heterogeneous mass
rule of [Jung et al. (2021)](https://doi.org/10.1021/acs.jctc.1c00185).
The existing integrators and temperature/pressure evaluations are retained;
this feature implements the mass rule rather than the complete numerical
scheme studied in that paper.

## Configuration

HMR is disabled by default. Add the following to an NVE/NVT/NPT input:

```toml
[hmr]
enabled = true
ratio_xh1 = 2.0
ratio_other = 2.5
```

| Parameter | Default | Meaning |
| --- | --- | --- |
| `enabled` | `false` when the section is absent | Enable the startup transform; required when any HMR key is present |
| `ratio_xh1` | `2.0` | H mass multiplier when its heavy-atom donor has exactly one H |
| `ratio_other` | Required when enabled | H mass multiplier when its donor has two or more H; examples are `2.5` for AMBER and `3.0` for CHARMM |

Both ratios must be finite and at least one. The force-field family is not
inferred from the file format. Unknown HMR keys and ratios supplied alongside
`enabled=false` are errors. There is currently no water-selection option:
recognized water is always excluded.

For example, with an ordinary native topology prefix:

```toml
md_name = "HMR example"
mode = "nvt"
default_in_file_prefix = "system"
step_limit = 1000
dt = 0.002
constrain_mode = "SHAKE"
target_temperature = 300.0

[hmr]
enabled = true
ratio_other = 2.5

[thermostat]
mode = "middle_langevin"
tau = 0.1
seed = 2026
```

The time step and constraints are explicit user choices. HMR does not enable
constraints or change `dt`. Retain the intended model constraints and validate
the chosen time step, including for rigid water, which remains at its original
masses. The example is configuration syntax, not a timestep-accuracy benchmark.

## Identification and mass transfer

The current interface supports ordinary fixed-bond, all-atom protium systems.
It uses original masses and bond connectivity, rather than residue names,
atom ordering or coordinate distances:

1. Mark registered force-field virtual sites and exclude them from chemical
   connectivity. Build and deduplicate the bond adjacency lists.
2. Identify ordinary H by original mass in `[0.9, 1.1]` u.
3. Exclude isolated OHH real-atom components, with O mass in `[15.5, 16.5]` u.
   The two H may also have an H-H edge from SETTLE. Neither H may connect to
   an external real atom. Removing registered virtual sites from this check
   allows the OHH core of four/five-site water to be recognized.
4. Require each remaining H to have exactly one bond neighbor, a heavy-atom
   donor with original mass above 4 u. Count the H attached to each donor.
5. Select `ratio_xh1` for one attached H and `ratio_other` otherwise, then
   transfer the additional H mass from that donor.

For donor X and its bonded hydrogens, the rule is

$$
m'_H = r_X m_H, \qquad
m'_X = m_X - \sum_{H\text{ bonded to }X}(m'_H-m_H).
$$

All decisions use original masses. Each donor's transfers are accumulated in
double precision before its effective mass is written to float. Mass is
conserved within each donor/H group up to final float rounding. With integer
masses for illustration, CH3 with ratio `2.5` changes from C=12 and three H=1
to C=7.5 and three H=2.5; the group mass remains 15. Any nonpositive resulting
donor mass or nonfinite mass causes an error.

There is no explicit five/six-membered ring detection. A ring atom with one H
uses the XH1 ratio through its local connectivity. This rule alone does not
establish ring stability at a particular time step.

Explicit chemical bond connectivity is required even for constrained X-H
bonds. The GROMACS loader already retains constrained bonds and SETTLE
connections. Arbitrary standalone distance constraints are not used to invent
chemical bonds. Missing or ambiguous H donors and unregistered zero-mass atoms
are rejected. Isotopic, polarizable, coarse-grained and already-repartitioned
inputs are outside the mass heuristic's supported scope; REAXFF is rejected.
Mass/connectivity matching does not prove water-model identity, so chemically
unusual isolated OHH components also lie outside that scope.

## Initialization and the two mass views

`MD_INFORMATION::Read_Mass()` copies the original topology masses to `h_mass`,
calls `Apply_HMR()`, and only then calculates inverse masses and total mass and
uploads the effective masses. Thermostats, barostats, constraints and Maxwell
velocities are initialized afterward. Domain decomposition receives effective
masses through the existing mass-distribution path.

| Consumer | Mass view |
| --- | --- |
| H/water identification and donor counts | Original topology masses |
| Integration, kinetic quantities, thermostat/barostat mass inputs | Effective HMR masses |
| Maxwell velocity generation and constraint projection | Effective HMR masses |
| Automatic constraint membership via `constrain.mass` | Original topology masses |
| Constraint solver mass coefficients | Effective HMR masses |
| Virtual center-of-mass geometry and restraint reference/group COM definitions | Original topology masses |

Separating selection and model geometry from dynamics prevents an H mass above
the constraint-selection cutoff from silently removing its constraint, and
prevents HMR from redefining a mass-weighted virtual point. Bonded and nonbonded
force-field parameters are not transformed. Because water is unchanged, this
feature requires no HMR-specific change to the current solvent-LJ mass matching.

The pure CPU transform and checksum are in `SPONGE/MD_core/hmr.h`; configuration
and the MD adapter are in `SPONGE/MD_core/hmr.cpp`. See
[initial state construction](initial-state.md) and
[constraint parameters](input-reference/constraint.md).

## Restart and velocity semantics

New H5 dynamic restart state stores `mass_signature`, an atom-ordered checksum
of the effective float masses including atom count. Dynamic/full loading checks
it before restoring dynamic module state. Matching masses are required; the
checksum is a compatibility check, not a cryptographic identity guarantee.

| Restart case | Behavior |
| --- | --- |
| Signature present and effective masses match | Dynamic/full continuation allowed, subject to the other restart checks |
| Signature present and effective masses differ | Dynamic/full loading rejected |
| Legacy signature absent, HMR enabled | Dynamic/full loading rejected |
| Legacy signature absent, HMR disabled | Existing dynamic/full loading allowed |
| Structural-only loading | Dynamic mass-signature check is not applied |

To switch between HMR and ordinary masses, use
`input.h5.restart.load="structural"` and explicitly generate velocities for a
new dynamical segment with `[initial_velocity] mode="maxwell"` and a seed.
Structural loading otherwise preserves the stored velocities; HMR never
rescales them automatically. Legacy coordinate/velocity restart formats do not
carry the H5 mass check. See [restart inputs](input-reference/io.md) and
[initial velocities](input-reference/core.md#initial-velocities).

## Validation and its limits

`tests/hmr/test_hmr.cpp` checks XH1/XH2/XH3/XH4 factors, duplicate/reversed
bonds, cumulative transfers and mass conservation, preservation of original
masses, five/six-membered ring XH1 selection, water ordering/H-H edges/virtual
sites, alcohol exclusion from water matching, invalid inputs and checksum changes.

`tests/hmr/test_hmr_runtime.py` compares automatic HMR against a manually
repartitioned topology for short NVE/NVT/NPT trajectories, including Maxwell
velocities and SHAKE selection above the usual H mass cutoff. It also checks
virtual COM bias-force invariance, dynamic restart continuation, rejection of
changed masses, structural loading and invalid configuration. Reproduce these
focused tests with the CPU development build:

```bash
pixi run --frozen -e dev-cpu cmake --build build-dev-cpu --target SPONGE test_hmr --parallel 4
pixi run --frozen -e dev-cpu ctest --test-dir build-dev-cpu -R '^test_hmr' --output-on-failure
```

These checks establish implementation consistency for the covered cases.
They do not establish long-time ensemble accuracy, universal 4 fs stability,
GPU/MPI runtime behavior, transport properties or performance. Time-step
convergence and scientific validation remain system-specific.

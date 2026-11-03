# Pressure, SETTLE and TOML numerical corrections

These corrections apply to ordinary SPONGE inputs and do not require a driving
force, a viscosity observer or a new integrator.

## PME reciprocal virial

The reciprocal virial is the negative strain derivative of reciprocal energy.
Its R2C boundary-plane weights already contain the energy half factor. The
correction removes a second half factor, fixes the accumulation sign and
matches the doubled off-diagonal convention of `LTMatrix3`. For odd FFT lengths,
the final R2C plane is an ordinary conjugate pair, not a Nyquist plane. Both
initialization and box-update coefficient construction use these conventions.

The PME force and energy formulas are unchanged. Pressure and stress change;
pressure feedback can therefore change barostat trajectories.

## Dihedral virial

Using atom j as local image origin, let a=ri-rj, b=rk-rj, c=rk-rl and
g=dE_drj_part. The force kernel produces fi, fj=g-fi, fk=-fl-g and fl. Thus

```text
W = fi ⊗ a + fk ⊗ b + fl ⊗ (b-c)
  = fi ⊗ a - g ⊗ b - fl ⊗ c.
```

The corrected two signs retain the existing bonded image convention. Torsion
energy and Cartesian forces are unchanged; the pressure contribution changes.

## Constraint precision and local SETTLE corrections

The old triangle path formed an absolute center of mass, subtracted it from
the absolute A coordinate, and reconstructed the absolute constrained positions.
This can round local molecular geometry according to the molecule's position
in the box. Velocity corrections then divide the resulting error by dt.

The corrected path forms local offsets directly from minimum-image B-A and C-A
distances. It solves the existing triangle geometry and returns the local
correction to each trial position. The same correction is added to each stored
coordinate and used for the velocity and constraint-virial update. Absolute
center-of-mass coordinates do not enter the solve or velocity difference.
The existing gamma and half-exp-gamma factors are retained. Coordinates can
remain in their individual periodic images; the correction does not rebuild a
different molecular image at each call.

The top-level `constrain_precision = "single" | "double"` selects internal
arithmetic for SETTLE, SHAKE, LINCS and CCMA, including position updates, saved
old bonds, coefficients, iteration buffers, velocity projection and constraint
virial. The default is `double`. Both paths instantiate the same solver source.
There is no runtime geometry lookup by rounded float coefficients. Owned
precision-specific buffers are released with the solver and reused across
operations; local constraint remapping keeps the selected precision.

Global MD coordinates, velocities, masses, nonbonded forces and output stress
remain float. Conversion happens at the operation boundary, after all internal
iterations and constraint-stress accumulation; this is mixed precision, not
full-double dynamics. Explicit constraint-file distances are parsed as double,
while topology-derived distances retain their source precision. Solver
tolerance/order and iteration counts remain independent settings. Even in
single mode, initialization and exceptional pair-discriminant guards retain
selected existing double safety calculations.

The local-distance formulation retains the existing triangle formulas and
integrator factors. Internal double precision reduces local roundoff but does
not remove float write-back error, change the integrator, or establish GPU
performance or long-term viscosity accuracy. See
[input precision parameters](input-reference/constraint.md#internal-precision).

An independent old-bond Newton solve tests deformed equal-B/C-mass triangles.
Unequal-B/C-mass stationary triangles are included in the translation tests;
the legacy asymmetric dynamic rotation has a separate discrepancy with that
Newton solve and is not validated or extended by this precision correction.

## TOML floating values

Default stream precision can truncate binary64 input when decoded TOML values
are flattened into command strings. General-format `std::to_chars` preserves
the decoded value on round-trip; finite integral-looking floating values receive
`.0` so structured TOML keeps their floating type. This applies to scalar,
array and structured values. It preserves the parsed value rather than
arbitrary decimal digits beyond binary64 precision.

## Regression coverage

With `SPONGE_BUILD_TESTS=ON`, CTest includes:

- `test_toml_command_aliases`: high-precision scalar and array values, and a
  structured serialize/decode round-trip with binary64 value and type checks.
- `test_pressure_virial_regressions`: ordinary `print_pressure` mdout and force
  output, independent dihedral energy gradients and nine strain derivatives,
  reflected/permuted/periodic configurations, and a neutral two-charge direct
  Ewald strain reference with odd/even PME grids. The absolute tolerance includes
  mdout's 0.01 bar decimal rounding. These are static force/virial regressions,
  not an ensemble or transport benchmark.

Standalone CPU probes use the existing miscellaneous-validation harness:

```sh
python -m pytest -q \
  benchmarks/validation/misc/tests/test_pme_virial.py \
  benchmarks/validation/misc/tests/test_settle_triangle_precision.py \
  benchmarks/validation/misc/tests/test_settle_pair_guard.py \
  benchmarks/validation/misc/tests/test_constraint_velocity_projection.py \
  benchmarks/validation/misc/tests/test_matrix_constraint_precision.py
```

The PME probe directly excites boundary and interior Fourier planes, including
the odd-grid last plane; unlike the smooth two-charge fixture, its edge-mode
signal cannot vanish under Ewald damping. It also checks rebuilt coefficients
against the independent strain derivative of a conjugate reciprocal-mode pair
at two box sizes. The SETTLE probe compares local corrections with a double
Newton position solve across translations, axes, time steps and gamma factors,
and checks mass-weighted correction and translation-independent virial. It
also checks the three final bond lengths and the constraint virial against
the independent Newton corrections. PME coefficient rebuilding is tested
with both orthogonal and skew reciprocal cells.

These probes compile production source with the established optimized CPU
flags. The triangle probe also runs with strict floating-point evaluation,
because compiler reassociation can hide cancellation in the old inlined solve.
They require CMake, a C++ compiler, OpenMP and FFTW-compatible headers in
the development environment. GPU execution, performance profiling and molecular
NPT validation are separate checks.

The constraint-precision runtime regression runs all four algorithms in both
precisions. It compares a force-free position update to an independent mass-
weighted old-bond Newton solve and checks the corresponding velocity impulse,
momentum and six mdout pressure components. SHAKE/LINCS/CCMA tests disable
SETTLE extraction and cover both small and larger connected components. It also
checks default/invalid precision and generated-velocity initialization. The
matrix probe independently checks LINCS/CCMA velocity projection against a dense
mass-metric projection; the SETTLE/SHAKE velocity probe runs both scalar types.
These finite small-system regressions do not establish long-time ensemble or
transport accuracy. The known unequal-B/C-mass dynamic SETTLE limitation remains.

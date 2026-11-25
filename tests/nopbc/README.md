# Nonperiodic atom-order regression

Build with `SPONGE_BUILD_TESTS=ON`, then run:

```sh
ctest --test-dir build-dev-cuda13 -R test_nopbc_atom_order --output-on-failure
```

The same runtime test works with CPU and CUDA executables. Python and NumPy
are required; missing NumPy reports a skipped test (77).

An eight-atom synthetic fixture has noncontiguous constraint groups, distinct
LJ types, charges, GB radii and scale factors. Fourteen executable runs compare
unconstrained, SHAKE, SHAKE/SETTLE, input permutations, reversed exclusion rows,
and force-only versus force-and-energy steps, both with and without GB.
The near-zero integration timestep isolates force evaluation from dynamics.
Energies and global-order forces must remain invariant; the vacuum results
are also checked against an independent all-pairs LJ/Coulomb reference.

This guards against passing packed DD coordinates to global-order parameters
and losing one-sided exclusions when the local pair order reverses. NOPBC
already rejects multiple MPI processes; this test does not add MPI support.
These short synthetic checks do not establish production MD stability.

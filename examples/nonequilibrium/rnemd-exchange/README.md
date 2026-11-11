# Unequal-mass RNEMD exchange check

Copy these files to a new run directory and execute `SPONGE -mdin mdin.spg.toml`.
Native coordinates and velocities use engine units (Å and native velocity),
while the viscosity HDF5 stream has explicit physical units.

Two free particles have masses 2 and 4 and flow velocities 2 and −1.
Their COM velocity is zero; the first exchange reflects them to −2 and 1.
Momentum and kinetic energy are conserved. At profile step 2, the cumulative
middle-slab impulse is `8 * float(20.455f) / 10` Da nm/ps, approximately 16.364.
Later attempts have no signed candidate. The recorded exchange energy and
momentum residuals are zero for this exactly representable case.

This is an integration/output diagnostic, not a liquid. Most bins are empty;
the viscosity analyzer correctly rejects it rather than reporting a viscosity.
See [RNEMD documentation](../../../docs/nonequilibrium-rnemd.md).

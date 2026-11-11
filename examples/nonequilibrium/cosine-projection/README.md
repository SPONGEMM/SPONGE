# Cosine projection mechanics check

Copy this directory to a scratch location and run `SPONGE -mdin mdin.spg.toml`.
The eight noninteracting particles start in a pure cosine x flow plus constant
COM motion. Zero applied acceleration and a zero-temperature Langevin bath
isolate the projection: the bath must preserve that stream, while reported
`thermal_temperature` remains near zero. `cosine_fit` is approximately
1.6364 nm/ps. Native input velocity units are internal SPONGE units.

This is an integration check, not a liquid viscosity example or production
thermostat recommendation. See `docs/nonequilibrium-cosine.md` for the input,
time-layer, observable and validation contracts.

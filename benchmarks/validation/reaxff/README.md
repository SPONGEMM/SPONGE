# ReaxFF Validation Benchmarks

This directory contains validation-style tests for ReaxFF in SPONGE2.0.

## CHO NVE case

`statics/cho_nve` keeps both:
- input files (`coordinate.txt`, `mass.txt`, `type.txt`, `ffield.reax.cho`, `frc.dat`, `heat.spg.toml`, `nve.spg.toml`)
- reference outputs from a validated run (`reference/`)

Validation workflow:
1. Run `heat.spg.toml` (NVT heating stage).
2. Generate and run `nve.long.spg.toml` (NVE stage, 20000 steps).
3. Reconstruct kinetic energy from temperature with `K = 0.5 * dof * kB * T`.
4. Check `E = U + K` drift stability.

## Run

From `SPONGE2.0` root:

```bash
pytest -q benchmarks/validation/reaxff/tests/test_cho_nve.py
```

Optional binary override:

```bash
SPONGE_BIN=/path/to/SPONGE pytest -q benchmarks/validation/reaxff/tests/test_cho_nve.py
```

Generated runtime outputs are placed in:
- `benchmarks/validation/reaxff/outputs/cho_nve_heat_nve`

# Point Energy Validation

This suite validates one-frame point energies against SPONGE2.0 single-process
baseline references for two systems:

- `covid_tip4p`
- `softcore`

For each case, the test performs 5 runs:

1. `baseline_np1` (single process, reference binary)
2. `gpu_np1` (single process, GPU binary)
3. `cpu_np1` (single process, CPU binary)
4. `cpu_np2` (2 MPI processes, CPU binary)
5. `cpu_np4` (4 MPI processes, CPU binary)

Each run compares all energy terms in `mdout` (except `step/time/temperature`) to
`statics/<case>/reference/energy_reference_np1.json`.

## Environment variables

- `SPONGE_POINT_REF_BIN`: reference SPONGE binary for `baseline_np1`
- `SPONGE_POINT_GPU_BIN`: SPONGE binary for `gpu_np1`
- `SPONGE_POINT_CPU_BIN`: SPONGE binary for `cpu_np1/cpu_np2/cpu_np4`
- `SPONGE_POINT_MPIRUN`: mpirun command (default: `mpirun`)

If unset:
- reference defaults to `build/SPONGE` (or `SPONGE` in PATH)
- CPU/GPU binaries fall back to reference binary

## Run

```bash
cd SPONGE2.0
pytest -q benchmarks/validation/point_energy/tests/test_point_energy.py -s
```

Outputs are written to:
- `benchmarks/validation/point_energy/outputs/`

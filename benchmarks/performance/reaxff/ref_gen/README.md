# ReaxFF PETN Reference Generation

Use this directory to regenerate the static LAMMPS reference files for the
PETN single-frame performance validation test.

Command:

```bash
python benchmarks/performance/reaxff/ref_gen/generate_petn_lammps_reference.py
```

This will refresh:

- `benchmarks/performance/reaxff/statics/petn_16240/reference/in.lammps`
- `benchmarks/performance/reaxff/statics/petn_16240/reference/log.lammps`
- `benchmarks/performance/reaxff/statics/petn_16240/reference/forces.dump`
- `benchmarks/performance/reaxff/statics/petn_16240/reference/charges.dump`

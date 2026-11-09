# Rerun cell history and PME periodic-boundary regressions

These tests generate a neutral two-atom system and run the real SPONGE executable.
They require only Python's standard library and a working runtime for the selected
SPONGE backend. No benchmark trajectories, network access, or external molecular
preparation software are needed.

After configuring with `-DSPONGE_BUILD_TESTS=ON` and building `SPONGE`, run:

```sh
cmake --build build --target SPONGE test_pme_grid_coordinate
ctest --test-dir build -R '^(test_rerun_|test_pme_grid_coordinate)' --output-on-failure
```

Or run either case directly (use an absolute executable path):

```sh
python tests/rerun/test_regressions.py --sponge /path/to/SPONGE \
  --case cell_history --output /tmp/rerun-cell-history
python tests/rerun/test_regressions.py --sponge /path/to/SPONGE \
  --case pme_boundary --output /tmp/rerun-pme-boundary
```

- **cell_history:** replay a 40 A cubic cell followed by three identical 48 A
  cells. Every 48 A frame must agree with an independently initialized 48 A
  replay. Every frame uses its trajectory cell without an input switch. This
  guards against history-dependent cell updates when successive boxes are equal.
  The former cell-increment implementation is superseded by direct per-frame
  cell updates; deprecated disabling values are rejected by `test_rerun_cells`.
- **pme_boundary:** for each axis of a 64 A cell with a 64-point PME grid,
  alternate an atom between zero, -5e-7 A, zero, +5e-7 A, +64 A, and -5e-7 A.
  Energies must be continuous across the periodic boundary. The tiny negative
  fraction can wrap to exactly 1 in float32; choosing grid anchor 63 with offset
  zero instead of the periodic anchor 0 changes the reciprocal energy. Alternating
  positions is an end-to-end continuity smoke test. Every frame has the same
  box, so the test isolates periodic coordinate handling from cell changes. Coordinate
  preprocessing can hide the original PME defect in this small system, so this
  smoke test alone is not a negative control.
- **test_pme_grid_coordinate:** directly calls the same grid-coordinate helper
  used by `PME_Atom_Near`, on the host and, in CUDA builds, on the GPU. Checks
  anchors and offsets for tiny negative values, zero, exact periodic images,
  ordinary interior points, and negative/non-unit images, with grid sizes 32,
  64, and 96. Removing the upper-edge wrap makes the first three cases select
  the last grid anchor instead of zero. This is the PME regression oracle.

Both Python cases check finite output, successful termination, and the exact number of
frames. They intentionally omit `rerun_frame_limit` to exercise normal EOF.
Energy comparisons use the higher-precision `eff_pot` column with an absolute
5e-4 kcal/mol tolerance, rather than the two-decimal `potential`/`PM` columns.
Each run preserves its generated input, `mdout.txt`, and `execution.log` in the
output directory for diagnosis.

These are focused cubic-cell regressions. They do not establish arbitrary
triclinic-cell accuracy, force accuracy, MPI equivalence, sampling convergence,
or reproducibility of an entire free-energy calculation. No additional
near-orthogonal-cell projection is required or added by these fixes.

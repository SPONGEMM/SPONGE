# CV regression tests

These tests generate small, noninteracting systems and run the **built SPONGE
executable** in rerun mode. NumPy supplies independent weighted-sum, Kabsch, and
cubic-spline references. Both CV values and harmonic bias forces are checked.
Python with NumPy is required (included in the repository's Pixi development
environments).

Configure with `SPONGE_BUILD_TESTS=ON`, build `SPONGE`, then run:

```sh
ctest --test-dir <build-directory> -L cv --output-on-failure
```

The same tests run on CPU and CUDA builds. They cover:

- `center` and `center_of_mass` with 1, 32, 33, 64, 65, and 132 source atoms,
  nonuniform masses, unnormalized custom weights, and force redistribution;
- exact-zero RMSD with a squared RMSD restraint (finite, zero force);
- 33/1025-atom RMSD with and without alignment, including a full-rank reflected
  reference that requires a proper rotation;
- tabulated CV values and derivatives at both endpoints, just below the upper
  endpoint, and inside the range, with two or four knots, explicit padding,
  noninteger spacing, and invalid table/range rejection.

Each test retains its input files and `execution.log` under
`<build-directory>/tests/collective_variable/`. The scalar tolerance accounts for
the four decimal places in `mdout`; force checks use the binary float output.

## Check endpoint memory safety

Ordinary numerical checks alone cannot reliably detect the former upper-endpoint
read: its spline coefficient can be zero. Run the production CPU executable with
AddressSanitizer as well. Add these options to a **separate** CPU configuration:

```sh
-DSPONGE_BUILD_TESTS=ON \
-DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer" \
-DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address"
```

Build `SPONGE` and run:

```sh
ASAN_OPTIONS=detect_leaks=0 ctest --test-dir <asan-build-directory> \
  -R test_cv_tabulated --output-on-failure
```

Leak detection is disabled to focus this run on invalid memory accesses, since
these changes do not address process-exit allocation cleanup. For a CUDA memory
check, generated cases can also be run directly under `compute-sanitizer` after
regenerating their rerun inputs with `test_regressions.py`.

To run a single group without CTest:

```sh
python tests/collective_variable/test_regressions.py \
  --sponge /absolute/path/to/SPONGE --case rmsd_rotation --output /tmp/cv-rmsd
```

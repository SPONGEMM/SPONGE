# Floating-point classification regression tests

`SpongeFloat::{Is_Finite,Is_Nan,Is_Inf}` in
`SPONGE/utils/float_classification.hpp` supports IEEE-754 binary32 `float` and
binary64 `double` (checked at compile time). Use these helpers for validation
and runtime guards compiled with fast math. `Bits` also supports guards that
need exact sign/zero representation, such as SETTLE. A classifier cannot undo
changes fast math makes to the arithmetic that produced its input.

Host code copies representations with `memcpy` (no aliasing violation), then
uses a GCC/Clang integer assembly barrier or a volatile integer round trip on
other compilers, including MSVC. This prevents classification from depending
on finite-math assumptions, without disabling fast math for simulation kernels.
CUDA/HIP device passes use bit-reinterpretation intrinsics. GPU host passes use
the host implementation; NVCC's `--use_fast_math` is tested separately from
host `-ffast-math`.

SVD still checks only NaN. Hard walls still allow infinite bounds. SETTLE keeps
its exact sign and zero checks. Initial-velocity changes from issue/PR #57 are
not present in the base revision of this branch; that work should use this
header when integrated.

Run without SPONGE's external math/HDF5 dependencies:

```sh
cmake -S tests/floating_point -B build-float -DCMAKE_BUILD_TYPE=Release
cmake --build build-float --config Release --parallel 3
ctest --test-dir build-float -C Release --output-on-failure
```

Select another compiler with `-DCMAKE_CXX_COMPILER=clang++` (or `cl` with an
appropriate Windows generator/environment). Tests cover strict, fast-math,
and fast-math with IPO/LTO when CMake reports support. Explicit failures remain
active under `NDEBUG`. Command-line integer representations supply both signs
of zero, finite values, subnormals, infinity, and quiet/signaling NaNs at runtime.
The restart-assembler regression additionally verifies rejection of parsed
`inf`, `-inf`, and `nan` values through a production validation path.

For CUDA or HIP, add `-DPARALLEL_BACKEND=cuda` or `-DPARALLEL_BACKEND=hip` and
any architecture/toolchain settings required by your installation. This builds
an additional test that runs the same inputs on both host and device under
backend fast-math flags. Without an available GPU it runs host checks and
returns CTest skip code 77. These targets are also included when the full
project is configured with `SPONGE_BUILD_TESTS=ON`.

Full CPU validation using the locked Pixi environment:

```sh
pixi install -e dev-cpu --locked
pixi run --locked -e dev-cpu configure
pixi run --locked -e dev-cpu cmake -S . -B build-dev-cpu -DSPONGE_BUILD_TESTS=ON
pixi run --locked -e dev-cpu cmake --build build-dev-cpu --parallel 4
pixi run --locked -e dev-cpu env SPONGE_H5_ENABLE_RUNTIME_SMOKE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 ctest --test-dir build-dev-cpu --output-on-failure --parallel 2 --timeout 180
```

For the local RTX 2080 SUPER (compute capability 7.5), the equivalent Pixi GPU
build and test commands are:

```sh
pixi install -e dev-cuda12 --locked
pixi run --locked -e dev-cuda12 configure
pixi run --locked -e dev-cuda12 cmake -S . -B build-dev-cuda12 -DSPONGE_BUILD_TESTS=ON -DCUDA_ARCH=75
pixi run --locked -e dev-cuda12 cmake --build build-dev-cuda12 --parallel 4
pixi run --locked -e dev-cuda12 env SPONGE_H5_ENABLE_RUNTIME_SMOKE=1 OMP_NUM_THREADS=2 ctest --test-dir build-dev-cuda12 --output-on-failure --timeout 180
```

Run GPU commands outside a sandbox that hides the NVIDIA device/driver. Use the
architecture appropriate for your GPU instead of `75`.

Local validation for issue #58:

- GCC 13.4 and Pixi GCC 11.4: strict/fast/LTO classification and restart-assembler
  regressions passed.
- Clang 22.1 from Pixi: strict/fast regressions passed. CMake skipped LTO because
  this environment lacks the LLVMgold linker plugin.
- NVCC 12.4: host and GPU checks passed under `--use_fast_math` outside the
  sandbox.
- Pixi `dev-cpu` supplied MKL 2025.3 and built the complete SPONGE CPU executable
  and C++ test suite. The existing SETTLE/SHAKE velocity-projection probe also
  passed with MKL headers and fast-math.
- All 38 CPU CTests passed with runtime smoke tests enabled. The VDS smoke
  matrix checks preserved step 0 observable values against `mdout` independently
  of particle step 1. Its runtime EAM fixture uses valid type 0 for both atoms
  in native, legacy, and sidecar representations; the source contract fixture
  is unchanged. The standalone CMAP test supplies its controller rank definition
  for CUDA linking.
- Pixi CUDA 12.8/GCC 11.4 built the full GPU executable and test suite for the
  RTX 2080 SUPER. All 39 GPU-configured CTests passed with runtime smoke enabled,
  including device classification, VDS input/output, and restart tests. No tests
  were skipped.
- HIP and MSVC were unavailable locally. The standalone CI workflow covers GCC,
  Clang, and MSVC.

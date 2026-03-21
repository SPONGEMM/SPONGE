# Build Guide

## Build workflow

```bash
pixi install -e <env>             # install dependencies
pixi run -e <env> configure       # CMake configure
pixi run -e <env> compile         # build and install into the environment
```

The binary is installed to `$CONDA_PREFIX/bin/SPONGE`. Build artifacts go into `build-<env>`.

## Environments

| Environment | Purpose | Parallel backend |
|-------------|---------|------------------|
| `dev-cuda13` | CUDA 13 development (recommended) | CUDA |
| `dev-cuda12` | CUDA 12 development | CUDA |
| `dev-hip` | AMD GPU / Hygon DCU development | HIP |
| `dev-cpu` | CPU development | defaults to `none`, SIMD can be specified manually |
| `dev-cpu-mpi` | CPU + MPI development | defaults to `none` + MPI, SIMD can be specified manually |
| `cuda13` / `cuda12` / `hip` / `cpu` / `cpu-mpi` | build-only (no dev tools) | corresponding backend |

`dev-*` environments include development tools (python, pytest, clang-format, ruff, etc.). Benchmark and format tasks are only available in `dev-*` environments.

## CUDA build

```bash
pixi run -e dev-cuda13 configure
pixi run -e dev-cuda13 compile
```

pixi automatically installs `cuda-nvcc`, `libcufft-dev`, `libcublas-dev`, and other CUDA dependencies.

Compilation parallelism defaults to 4 threads. Pass a positional argument to adjust:

```bash
pixi run -e dev-cuda13 compile 8     # 8 threads
```

## HIP build

```bash
pixi run -e dev-hip configure
pixi run -e dev-hip compile
```

> **Note:** pixi does not install the HIP/ROCm SDK. It must be pre-installed on the system, e.g. via `apt install`, from the official ROCm repository, or via `module load`. CMake detects the HIP environment by looking for `hipcc` / `hipconfig` in the system PATH.

## CPU build

```bash
pixi run -e dev-cpu configure          # defaults to no SIMD
pixi run -e dev-cpu configure avx2     # specify AVX2
pixi run -e dev-cpu compile
```

Available SIMD backends:

| x86_64 | ARM |
|--------|-----|
| `avx512`, `avx2`, `avx`, `sse42` | `sve2`, `sve`, `neon` |

## CPU + MPI build

```bash
pixi run -e dev-cpu-mpi configure
pixi run -e dev-cpu-mpi compile
```

Automatically adds `-DMPI=ON` and the MPI compiler configuration.

## Platform differences

| Platform | Compiler | Math library | Notes |
|----------|----------|-------------|-------|
| Linux x86_64 | GCC 11 (conda) | MKL | CUDA + MPI requires NCCL |
| Linux aarch64 | GCC 11 (conda) | OpenBLAS + FFTW | supports SVE/NEON |
| Windows x64 | MSVC (vs2022) | MKL | may need `pixi run install-msvc` first |
| macOS ARM | Clang 22.1 | OpenBLAS + FFTW | CPU backend only |

## Global compile settings

The following are hardcoded in `cmake/utils/common.cmake`:

- C++ standard: C++17
- Build type: Release
- Fast math: `-ffast-math` / `/fp:fast` / `--use_fast_math`
- OpenMP: always enabled

## Formatting

```bash
pixi run -e dev-cuda13 format        # auto-fix
pixi run -e dev-cuda13 format-check  # check only
```

See [Contributing](contributing.md) for details.

## Packaging

```bash
pixi run -e dev-cuda13 package
```

Produces a `.conda` package in `conda/outputs/`.

## Troubleshooting

### configure cannot find CUDA

Make sure you are using a CUDA environment (`dev-cuda12` or `dev-cuda13`). Do not specify `-DPARALLEL=cuda` in a CPU environment.

### nvcc warning "incompatible redefinition for option compiler-bindir"

Known warning under the conda environment. Does not affect the build. Safe to ignore.

### Wrong system compiler or libraries picked up

When pixi provides all required dependencies, use `pixi install -e <env> --clean-envs` to rebuild a clean environment. However, this does not apply when the build depends on external toolchains (MSVC on Windows, HIP/ROCm); in those cases, set up the environment variables yourself before invoking pixi.

### format reports missing BOM

C++ files require UTF-8 with BOM. Run `pixi run format` to fix automatically.

### Cleaning build directories

```bash
rm -rf build-dev-cuda13   # clean a specific environment
rm -rf build-*            # clean all
```

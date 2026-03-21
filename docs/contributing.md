# Contributing

## Formatting Tools

| Language | Tool | Version | Config |
|----------|------|---------|--------|
| C++ | clang-format | 22.1.0 | `.clang-format` |
| Python | ruff format | 0.15.1 | `ruff.toml` |
| CMake | cmake-format | 0.6.13 | default |

```bash
pixi run format        # auto-fix all files
pixi run format-check  # check only
```

## File Encoding

C++ source files (`.cpp`, `.h`, `.hpp`) require UTF-8 with BOM. `pixi run format` adds the BOM automatically.

## C++ Style

Based on Google style with the following modifications:

- 4-space indentation
- Allman brace style (opening brace on its own line)
- 80-character line width

### Naming Conventions

| Element | Style | Example |
|---------|-------|---------|
| Struct | UPPER_CASE | `PAIRWISE_FORCE` |
| Method | PascalCase | `Compute_Force()` |
| Variable | lower_case | `atom_numbers` |
| Constant / Macro | UPPER_CASE | `CHAR_LENGTH_MAX` |

File and directory names follow existing conventions. New modules should reference sibling directories for naming style.

### Core Abstractions

SPONGE uses two abstraction layers for cross-backend code:

**device_api** (`SPONGE/third_party/device_backend/`) — unified GPU/CPU runtime interface:

```cpp
// Use unified interface; do not call cudaMalloc / hipMalloc directly
deviceMalloc(&ptr, size);
deviceMemcpy(dst, src, size, deviceMemcpyDeviceToHost);
Launch_Device_Kernel(kernel, grid, block, args...);
```

**LaneGroup** (`SPONGE/third_party/lane_group/`) — unified warp/SIMD lane-level operations:

```cpp
// Use LaneGroup interface; do not write __shfl_sync or SIMD intrinsics directly
int width = LaneGroup::Width();            // CUDA: 32, AVX: 8, SSE: 4
float sum = LaneGroup::Reduce_Sum(value);  // warp/vector reduction
LaneMask mask = LaneGroup::Ballot(pred);   // predicate ballot
```

Guidelines:

- Use `LaneGroup` for parallel code; do not write backend intrinsics directly
- Use `device_api` macros for device memory and streams
- Isolate backend-specific logic with `#ifdef USE_GPU` / `USE_CUDA` / `USE_HIP`

## Python Style

- 80-character line width, double quotes, space indentation
- Lint checks import sorting (isort) only
- Functions: `snake_case`, classes: `PascalCase`, constants: `UPPER_CASE`

## Commit Workflow

1. Write code
2. Run `pixi run format` to fix formatting
3. Run `pixi run format-check` to verify
4. Commit

## Benchmarks

After adding or modifying functionality, run relevant benchmarks:

```bash
pixi run -e dev-cuda13 vali-thermostat   # thermostat validation
pixi run -e dev-cuda13 vali-barostat     # barostat validation
pixi run -e dev-cuda13 vali-cv           # collective variable validation
pixi run -e dev-cuda13 vali-misc         # miscellaneous validation
pixi run -e dev-cuda13 perf-amber        # AMBER performance
pixi run -e dev-cuda13 perf-nonortho     # non-orthogonal box
```

For new benchmarks, follow the existing structure in the `benchmarks/` directory.

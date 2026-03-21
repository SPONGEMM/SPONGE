# Getting Started

SPONGE (Simulation Package Toward Next GEneration of molecular modelling) is a GPU-accelerated molecular dynamics simulation engine supporting CUDA, HIP (AMD GPU / Hygon DCU), and various CPU SIMD backends.

## Install pixi

SPONGE uses [pixi](https://pixi.sh) to manage dependencies and build workflows.

### Linux / macOS

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

China mirror:

```bash
curl -fsSL https://conda.spongemm.cn/pixi/install.sh | bash
```

### Windows

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://pixi.sh/install.ps1 | iex"
```

China mirror:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://conda.spongemm.cn/pixi/install.ps1 | iex"
```

## Build SPONGE

### Choose an environment

Pick the environment that matches your hardware:

| Environment | Hardware |
|-------------|----------|
| `dev-cuda13` | NVIDIA GPU (recommended) |
| `dev-cuda12` | NVIDIA GPU (older drivers) |
| `dev-hip` | AMD GPU / Hygon DCU (requires system-installed HIP/ROCm) |
| `dev-cpu` | CPU |
| `dev-cpu-mpi` | CPU + MPI |

### Compile

```bash
pixi install -e dev-cuda13          # install dependencies
pixi run -e dev-cuda13 configure    # CMake configure
pixi run -e dev-cuda13 compile      # build
```

The `SPONGE` binary is installed into the pixi environment upon completion.

### Verify

```bash
pixi run -e dev-cuda13 which SPONGE
pixi run -e dev-cuda13 SPONGE --help
```

## Run a simulation

Prepare a TOML input file `mdin.spg.toml`:

```toml
md_name = "NVT water"
mode = "nvt"
step_limit = 50000
dt = 0.002
cutoff = 8.0
default_in_file_prefix = "WAT"
constrain_mode = "SHAKE"
thermostat = "middle_langevin"
thermostat_tau = 0.1
thermostat_seed = 2026
target_temperature = 300.0
write_information_interval = 1000
```

Run:

```bash
pixi run -e dev-cuda13 SPONGE -mdin mdin.spg.toml
```

Or enter a shell first:

```bash
pixi shell -e dev-cuda13
SPONGE -mdin mdin.spg.toml
```

## Run benchmarks

```bash
pixi run -e dev-cuda13 perf-amber       # AMBER force field performance
pixi run -e dev-cuda13 perf-nonortho    # non-orthogonal box
pixi run -e dev-cuda13 vali-thermostat  # thermostat validation
pixi run -e dev-cuda13 vali-barostat    # barostat validation
```

## Next steps

- [Build Guide](build-guide.md) — detailed multi-platform build instructions
- [Input Reference](input-reference/README.md) — full TOML input parameter reference
- [Contributing](contributing.md) — code style and contribution guidelines

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

## Install from binary distributions

If you want to use a prebuilt SPONGE binary, install one of the published
packages instead of building from source.

### Choose a package

| Package | Hardware | Platforms |
|---------|----------|-----------|
| `sponge-cuda13` | NVIDIA GPU (driver >= 570) | Linux x86_64, Windows x64 |
| `sponge-cuda12` | NVIDIA GPU (driver >= 525) | Linux x86_64, Windows x64 |
| `sponge-cpu` | CPU only | Linux x86_64 / aarch64, Windows x64, macOS ARM64 |
| `sponge-cpu-mpi` | CPU + MPI | Linux x86_64 / aarch64 |

The commands below use `sponge-cpu` as an example. Replace it with another
package from the table when appropriate.

### Install in a Pixi project

Create a Pixi project, add the SPONGE channel, and install the package into the
project environment:

```bash
pixi init sponge-project
cd sponge-project
pixi project channel add https://conda.spongemm.cn
pixi add sponge-cpu
pixi run SPONGE -v
```

If you are already in a Pixi project that uses `conda-forge`, only the last
three commands are needed.

### Install globally with Pixi

Install SPONGE into a Pixi-managed global environment when you want to run it
outside a project:

```bash
pixi global install \
  --channel https://conda.spongemm.cn \
  --channel conda-forge \
  sponge-cpu
SPONGE -v
```

## Optional: Build SPONGE from source

Most users should install a binary distribution above. To compile SPONGE from
source or set up a development environment, follow the
[Build Guide](build-guide.md).

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

Use the command that matches how SPONGE was installed.

For a global installation:

```bash
SPONGE -mdin mdin.spg.toml
```

For an installation in a Pixi project:

```bash
pixi run SPONGE -mdin mdin.spg.toml
```

For an optional source build in a development environment:

```bash
pixi run -e dev-cuda13 SPONGE -mdin mdin.spg.toml
```

You can also enter that development environment first:

```bash
pixi shell -e dev-cuda13
SPONGE -mdin mdin.spg.toml
```

## Run benchmarks

The benchmark tasks are available from a source checkout with a development
environment installed:

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

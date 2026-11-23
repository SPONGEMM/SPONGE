# SPONGE_MANAGER

The Linux manager starts a persistent SPONGE process in each window directory.
Each window keeps its own input parameters. Accepted exchanges transfer
coordinates and velocities, and NPT exchanges transfer the periodic box too.

```sh
cmake -S manager -B manager-build -DCMAKE_BUILD_TYPE=Release
cmake --build manager-build
manager-build/SPONGE_MANAGER /path/to/SPONGE 100 180000 42 exchange \
  0 /path/window-0 1 /path/window-1
```

Set `SPONGE_BUILD_REPLICA_MANAGER=ON` on the top-level project to build and
install both executables. The manager also builds independently as above.
All windows reuse the same SPONGE binary.

Provide a complete `mdin.spg.toml` in each directory. The positional
arguments set the exchange interval in steps, timeout in milliseconds,
random seed, and either `exchange` or `sync`. The `sync` control performs
the same probe and rejects all exchanges. GPU identifiers come from the
scheduler allocation. Reuse an identifier for multiple processes on the
same GPU or use `-` to inherit device visibility. Pairing alternates along
the given directory order. With two windows, every round tries that pair.

## Input compatibility

Use compatible atom order, masses, constraints, timesteps and step counts.
The same window interface supports different temperature, charge, bonded,
soft-core LJ, and fixed SITS parameters. Set them in each window's own
input. For fixed SITS, use production weights and `SITS_fb_interval = 1`.
Use `SITS_bias_strength` from 0 to 1 to scale the configured bias and its
force. At strength 0 the bias contribution is zero. The exchange workflow
supports prepared REST2 input windows with a common coordinate space.

Supported runs use single-rank NVT or NPT and middle Langevin with static
targets. Match the NVT boxes exactly. For NPT, use orthogonal boxes with the
same aspect ratio and isotropic Bussi or XYZ Monte Carlo pressure coupling.
Different volumes and pressure targets are supported. Keep restraint
reference coordinates fixed. Use isotropic Bussi for NPT with SITS.
The handshake compares masses, constraints and compatible box information,
along with the H5 atom-order identity when available. Verify atom order
explicitly for legacy input.

## Exchange steps

The manager uses sockets for decisions and shared anonymous memory for
configurations. Each window first evaluates its own configuration. It then
loads the partner's coordinates and velocities with the appropriate
temperature scaling, updates the box for NPT, and evaluates the incoming
configuration with its own force-field parameters. The NPT comparison
includes the pressure of the evaluating window times the volume of the
configuration being evaluated. A rejected proposal restores the original
configuration and force. Each barostat continues in its own window after
the decision, with its own target pressure and random stream.

`pair` logs record paired windows, log acceptance, random draw and decision.
`label` records the walker in each fixed window. `timing` measures the
report-arrival span after publishing coordinates and the manager's decision
and reply time. Use total wall time to measure configuration staging and
force evaluations as well. Each window has its own `sponge.log`.

## Group restart

Configure H5 topology input and H5 restart output in each window.

```toml
[input.h5.topology]
path = "system.spgt.h5"

[output.h5.restart]
path = "regular.spgr.h5"
```

Use `--checkpoint /path/run.state 10` before the engine argument to save
every ten exchange rounds. The manager publishes the manifest atomically
after all windows have published H5 snapshots of the same completed step.
The manifest stores the walker mapping, exchange RNG, window targets and
snapshot locations.

Use `--resume /path/run.state 10` to continue, with the same directory
ordering, interval and exchange mode. Each replica input supplies the
continuation length. Keep fixed window inputs and use fresh trajectory
output paths if the previous outputs have frames beyond the checkpoint.
Use the new manifest format for this version.

## Validation

```sh
export SPONGE_MANAGER="$PWD/manager-build/SPONGE_MANAGER"
export SPONGE_ENGINE=/path/to/SPONGE
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 tests/replica_exchange/test_manager.py
python3 tests/replica_exchange/test_engine.py

python3 benchmarks/performance/replica_exchange/alanine.py \
  "$SPONGE_ENGINE" "$SPONGE_MANAGER" ./remd-benchmark \
  --devices 0 1 --temperatures 300 302 --steps 10001 --interval 100
```

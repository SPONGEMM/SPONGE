# Ideal-particle stress diagnostic

Copy this directory to a fresh run directory and run `SPONGE -mdin mdin.spg.toml`.
Then, from the run directory, invoke the repository analyzer with NumPy/h5py:

```sh
python /path/to/SPONGE/scripts/analyze_green_kubo.py viscosity.h5 \
  --start-ps 0 --window-ps 0.0002 --block-ps 0.0007 --output analysis
```

These three noninteracting particles have constant COM-subtracted kinetic stress.
The correlation integral grows with the window: there is no finite relaxed liquid
viscosity here. This example only exercises sampling, physical units and analysis
output. A narrow or zero block interval is not evidence of a converged viscosity.
Use an equilibrated interacting liquid and justified production/analysis windows
for transport measurements. See `docs/nonequilibrium-green-kubo.md`.

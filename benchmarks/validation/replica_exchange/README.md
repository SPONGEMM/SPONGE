# Temperature replica-exchange validation

The analytic harmonic-dimer test uses a three-dimensional bond. Its reference
distribution integrates the radial measure r² dr. It checks the mean potential
and cumulative probabilities at four reference energy quantiles.

The peptide study uses ACE–ALA–NME with ff14SB and explicit TIP3P. The window
runner reads the Amber topology, resolves the central alanine dihedrals by
residue and atom name, and checks that the solvent has three sites per water.
All windows use the same minimized coordinates, 8 Å cutoff, SHAKE and 2 fs
middle-Langevin dynamics. Keep the same SPONGE binary and Amber input for the
independent controls. Use separate seeds and output directories for repeats.

Each sampled frame is recovered from the SPONGE trajectory and rescored via
SPONGE rerun mode. The script checks both coordinates and box frame counts
against the run and checks the rerun frame sequence. It extracts the two
dihedrals from those same coordinate frames with periodic minimum images.

```sh
python3 alanine_study.py /path/to/SPONGE /path/to/SPONGE_MANAGER model remd-0 \
  --mode remd --devices 0 1 2 3 --temperatures 300 308 316 324 \
  --steps 2000001 --sample 500 --interval 250 --seed 202700
python3 alanine_study.py /path/to/SPONGE /path/to/SPONGE_MANAGER model nvt-0 \
  --mode nvt --devices 0 1 --temperatures 300 312 \
  --steps 2000001 --sample 500 --seed 202800
```

The production analysis uses NumPy, SciPy, PyMBAR and Matplotlib. It compares
the direct 300 K window with independent 300 K NVT and predicts 312 K from
the four sampled temperatures using MBAR. Supply multiple independent repeats.
In block resampling, the windows of one exchange run share each selected
time block. The default analysis discards 300 ps and uses 100 ps blocks.
Also check sensitivity to 200 ps blocks. Each estimate and reference needs at least 20 samples in and outside a
region, using the MBAR weight effective count for the prediction. The declared comparison margins are
five percentage points for each region and 0.2 control standard deviations
for the mean potential.

```sh
python analyze_alanine.py --remd remd-* --nvt nvt-* --output analysis
```

The angle regions are defined explicitly in `features()`. They are reporting
regions for comparison between these runs. They do not prescribe basin
boundaries for other force fields. Efficiency uses estimated autocorrelation
of the 300 K alpha-R indicator and MD wall time charged across the allocated
GPUs. The MBAR weight effective count is reported separately, before time
correlation. Raw coordinates remain in each window directory.

## Implicit-solvent alanine dipeptide

Use the Amber GB1 fixture with 22 atoms, ff14SB, and mbondi radii. Generate
the native GB parameter file from the Amber topology RADII and SCREEN fields
with `write_gb_in_file_from_parm7` in
[`benchmarks/comparison/reference/amber/generate.py`](../../comparison/reference/amber/generate.py).
Copy the fixture's `system.parm7` and `system.rst7` alongside `gb.txt`, then
run `python gb_starts.py model` to create PPII, alpha-R and positive-phi starts.

Each of the REMD and independent NVT ladder arms uses 290, 320, 350, and
380 K. The independent control arm uses 305 and 365 K, which are excluded
from the MBAR input. Run six repeats per arm with distinct seeds, using
each starting conformer twice. For example, from this directory:

```sh
python3 alanine_study.py /path/to/SPONGE /path/to/SPONGE_MANAGER model data/remd-0 \
  --gb-start model/ppii.rst7 --mode remd --devices 0 1 2 3 \
  --temperatures 290 320 350 380 --dt 0.001 --steps 20000001 \
  --sample 1000 --interval 500 --seed 61100
```

Name the independent ladder repeats `nvt-0` through `nvt-5` and the
held-out NVT repeats `control-0` through `control-5`. Run both with
`--mode nvt` and one device argument per temperature.

```sh
python3 analyze_gb.py data analysis-500ps --block-ps 500
python3 analyze_gb.py data analysis-1000ps --block-ps 1000
```

The results include recomputed potentials, direct window comparisons, MBAR
overlap, held-out comparisons, and conformational round trips tracked along
each walker. The aggregate analysis consumes the per-run `samples.csv`
files and retains the independent NVT trajectories for comparison.

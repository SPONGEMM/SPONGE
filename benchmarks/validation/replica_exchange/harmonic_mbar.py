"""Reweight the harmonic-dimer T-REMD outputs to 450 K and test the analytic CDF."""

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_alanine import MBAR, bootstrap_fit, mbar_input, weights
from harmonic_dimer import (
    BURN_IN,
    BLOCK_SIZE,
    INTERVAL,
    TEMPERATURES,
    reference,
    reference_quantiles,
)


def load(directory):
    energies = []
    for temperature in TEMPERATURES:
        with (directory / str(temperature) / "mdout.txt").open() as stream:
            energy_index = next(stream).split().index("eff_pot")
            rows = [
                (int(row[0]), float(row[energy_index]))
                for line in stream
                if (row := line.split())
            ]
        energies.append(
            np.asarray(
                [
                    (step, energy)
                    for step, energy in rows
                    if step > BURN_IN and step % INTERVAL
                ]
            )
        )
    if not np.array_equal(energies[0][:, 0], energies[1][:, 0]):
        raise ValueError("harmonic windows must have the same sampled steps")
    return energies[0][:, 0], [data[:, 1:] for data in energies]


def run(directory, output):
    target = 450
    steps, samples = load(directory)
    temps = np.array(TEMPERATURES)
    thresholds, probabilities = map(np.array, zip(*reference_quantiles(target)))
    counts, frames, centered, scaled = mbar_input(samples, temps)
    model = MBAR(scaled, counts, solver_protocol="robust")
    obs = np.column_stack((frames[:, 0], frames <= thresholds))
    estimate = weights(model.f_k, scaled, counts, centered, target) @ obs
    ground_truth = np.r_[reference(target), probabilities]
    time_blocks = (steps - 1) // BLOCK_SIZE
    labels = np.unique(time_blocks)
    indices = [np.flatnonzero(time_blocks == label) for label in labels]
    rng = np.random.default_rng(450)
    trials = []
    for _ in range(100):
        selected = np.concatenate(
            [indices[i] for i in rng.integers(len(indices), size=len(indices))]
        )
        counts, frames, centered, scaled = mbar_input(
            [window[selected] for window in samples], temps
        )
        w = weights(
            bootstrap_fit(scaled, counts), scaled, counts, centered, target
        )
        trials.append(w @ np.column_stack((frames[:, 0], frames <= thresholds)))
    limits = np.quantile(trials, [0.025, 0.975], axis=0)
    matches = (limits[0] <= ground_truth) & (ground_truth <= limits[1])
    result = {
        "target_temperature": target,
        "cdf_cutoffs": thresholds.tolist(),
        "mean_and_cdf": estimate.tolist(),
        "analytic_mean_and_cdf": ground_truth.tolist(),
        "confidence_intervals": limits.T.tolist(),
        "reference_in_ci": matches.tolist(),
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dimer_results", type=Path)
    parser.add_argument("output", type=Path)
    opts = parser.parse_args()
    run(opts.dimer_results, opts.output)

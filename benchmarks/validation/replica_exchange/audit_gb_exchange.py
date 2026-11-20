"""Compare the logged T-REMD acceptance with rerun energy of pre-exchange frames."""

import argparse
import json
import math
from pathlib import Path
import statistics
import struct


def potentials(path):
    with path.open() as stream:
        header = next(stream).split()
        frame, energy = header.index("frame"), header.index("eff_pot")
        return {
            int(row[frame]): float(row[energy])
            for line in stream
            if (row := line.split())
        }


def verify(directory, temperatures):
    # SPONGE's CONSTANT_kB is a single-precision constant in kcal/mol/K.
    (boltzmann,) = struct.unpack("f", struct.pack("f", 0.00198716))
    beta_difference = (1 / temperatures[0] - 1 / temperatures[1]) / boltzmann
    left, right = (
        potentials(directory / f"window-{index}" / "rerun.txt")
        for index in (0, 1)
    )
    errors = []
    accepts = 0
    with (directory / "manager.log").open() as stream:
        for line in stream:
            if not line.startswith("pair "):
                continue
            _, step, i, j, log_acceptance, draw, accepted = line.split()
            if (i, j) != ("0", "1"):
                raise ValueError("expected a two-window audit")
            # Trajectories omit step zero and save before the next exchange.
            # Rerun frame s is the saved coordinate at the start of MD step s.
            log_acceptance, draw, accepted = (
                float(log_acceptance),
                float(draw),
                int(accepted),
            )
            expected = beta_difference * (left[int(step)] - right[int(step)])
            if not all(map(math.isfinite, (expected, log_acceptance, draw))):
                raise ValueError(f"non-finite result at step {step}")
            if accepted != int(math.log(draw) < log_acceptance):
                raise ValueError(f"inconsistent logged decision at step {step}")
            if accepted != int(math.log(draw) < expected):
                raise ValueError(
                    f"rerun predicts a different decision at step {step}"
                )
            errors.append(abs(expected - log_acceptance))
            accepts += accepted
    if not errors or max(errors) > 1e-4:
        raise ValueError(
            "rerun energies do not reproduce the exchange exponent within 1e-4"
        )
    return {
        "attempts": len(errors),
        "accepted": accepts,
        "maximum_absolute_log_acceptance_error": max(errors),
        "median_absolute_log_acceptance_error": statistics.median(errors),
        "maximum_pair_energy_difference_error_kcal_mol": max(errors)
        / abs(beta_difference),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--temperatures", nargs=2, type=float, default=(300, 330)
    )
    args = parser.parse_args()
    print(json.dumps(verify(args.directory, args.temperatures), indent=2))

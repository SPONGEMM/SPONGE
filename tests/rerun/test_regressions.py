"""Synthetic energy regressions through the actual SPONGE rerun executable."""

import argparse
import json
import math
import os
import struct
import subprocess
from pathlib import Path


def run(binary, directory, frames, boxes, **overrides):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "mass.txt").write_text("2\n12\n12\n")
    (directory / "charge.txt").write_text("2\n18.2223\n-18.2223\n")
    (directory / "coordinate.txt").write_text(
        "2\n"
        + "\n".join(" ".join(map(str, row)) for row in frames[0])
        + "\n"
        + " ".join(map(str, boxes[0]))
        + "\n"
    )
    (directory / "frames.dat").write_bytes(
        b"".join(
            struct.pack("=6f", *(x for atom in frame for x in atom))
            for frame in frames
        )
    )
    (directory / "frames.box").write_text(
        "".join(" ".join(map(str, box)) + "\n" for box in boxes)
    )
    settings = dict(
        mode="rerun",
        step_limit=len(frames) + 2,
        rerun_start=0,
        rerun_strip=0,
        cutoff=8.0,
        skin=2.0,
        mass_in_file="mass.txt",
        charge_in_file="charge.txt",
        coordinate_in_file="coordinate.txt",
        crd="frames.dat",
        box="frames.box",
        rerun_output_crd="output.dat",
        rerun_output_box="output.box",
        print_zeroth_frame=1,
        write_mdout_interval=1,
        write_information_interval=1,
        write_trajectory_interval=1,
        dont_check_input=1,
        PM_fftx=64,
        PM_ffty=64,
        PM_fftz=64,
    )
    settings.update(overrides)
    (directory / "mdin.spg.toml").write_text(
        "\n".join(
            f"{key} = {json.dumps(value)}" for key, value in settings.items()
        )
        + "\n"
    )
    result = subprocess.run(
        [str(binary), "-mdin", "mdin.spg.toml"],
        cwd=directory,
        env=dict(os.environ, OMP_NUM_THREADS="1"),
        capture_output=True,
        text=True,
        timeout=90,
    )
    (directory / "execution.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, f"SPONGE failed: {directory}/execution.log"
    lines = (directory / "mdout.txt").read_text().splitlines()
    rows = [
        dict(zip(lines[0].split(), map(float, line.split())))
        for line in lines[1:]
        if line.strip()
    ]
    assert len(rows) == len(frames), (directory, len(rows), len(frames))
    assert all(math.isfinite(x) for row in rows for x in row.values()), (
        directory
    )
    return rows


# eff_pot retains more digits than the two-decimal potential/PM columns.
# This tolerance allows ordinary float FFT/reduction noise, but is much smaller
# than a one-grid-cell anchor shift or a repeated cell increment in this fixture.
ENERGY_TOLERANCE = 5e-4


def check_energy(rows, expected, label):
    for index, row in enumerate(rows):
        error = abs(row["eff_pot"] - expected)
        assert error <= ENERGY_TOLERANCE, (
            f"{label}, frame {index}: {row['eff_pot']} vs {expected}, "
            f"error {error} > {ENERGY_TOLERANCE} kcal/mol"
        )


def pme_boundary(binary, output):
    box = [64, 64, 64, 90, 90, 90]
    for axis in range(3):
        origin = [0.0, 0.0, 0.0]
        negative = origin.copy()
        negative[axis] = -5e-7
        positive = origin.copy()
        positive[axis] = 5e-7
        translated = origin.copy()
        translated[axis] = 64.0
        # Alternation exercises both initial and cached PME anchors. At float32
        # precision the negative fraction can wrap to exactly 1.0. It must be
        # energetically continuous with zero and the periodic image at +L.
        frames = [
            [p, [7.25, 9.5, 11.75]]
            for p in [origin, negative, origin, positive, translated, negative]
        ]
        rows = run(
            binary,
            output / f"axis{axis}",
            frames,
            [box] * len(frames),
        )
        check_energy(rows, rows[0]["eff_pot"], f"PME axis {axis}")


def cell_history(binary, output):
    coords = [[2.25, 3.5, 4.75], [9.5, 11.25, 13.75]]
    for name, initial, changed in [
        ("orthogonal", [40, 40, 40, 90, 90, 90], [48, 48, 48, 90, 90, 90]),
    ]:
        reference = run(
            binary,
            output / name / "reference",
            [coords] * 2,
            [changed] * 2,
        )
        boxes = [initial, changed, changed, changed]
        rows = run(binary, output / name / "sequence", [coords] * 4, boxes)
        check_energy(rows[1:], reference[-1]["eff_pot"], name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sponge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--case", choices=["pme_boundary", "cell_history"], required=True
    )
    args = parser.parse_args()
    globals()[args.case](args.sponge.resolve(), args.output.resolve())
    print(f"{args.case}: passed")

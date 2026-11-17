"""Repeated multi-pair kernel/force smoke; dt=0, no sampling or ns/day claim."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sponge", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if (
        not 1 <= args.pairs <= 256
        or not 1 <= args.steps <= 10000
        or not 2 <= args.repeats <= 10
    ):
        parser.error(
            "bounded smoke requires pairs<=256, steps<=10000 and repeats 2..10"
        )
    engine = Path(args.sponge).resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    count = args.pairs * 2
    xyz = []
    for i in range(args.pairs):
        origin = np.array(
            [2 + 6 * (i % 8), 2 + 6 * ((i // 8) % 8), 2 + 6 * (i // 64)],
            dtype=float,
        )
        xyz.extend([origin, origin + [3.2, 0, 0]])
    xyz = np.array(xyz)
    sigma, epsilon, weight = 3.2, 0.15, 0.5
    c6 = weight * 4 * epsilon * sigma**6
    c12 = c6 * sigma**6
    records = []
    for repeat in range(args.repeats):
        # Rotate order to avoid always measuring one mode first.
        modes = ["legacy", "zero", "soft"]
        modes = modes[repeat % 3 :] + modes[: repeat % 3]
        for mode in modes:
            work = args.output / f"{repeat}-{mode}"
            work.mkdir()
            delta = (
                0.5 * (1 - weight) ** 2 * sigma**6 if mode == "soft" else 0.0
            )
            files = {
                "mass": str(count) + "\n" + "12\n" * count,
                "charge": str(count) + "\n" + "0\n" * count,
                "velocity": str(count) + "\n" + "0 0 0\n" * count,
                "coordinate": str(count)
                + "\n"
                + "\n".join(" ".join(map(str, r)) for r in xyz)
                + "\n64 64 64\n90 90 90\n",
                "nb14_extra": str(args.pairs)
                + "\n"
                + "".join(
                    f"{2 * i} {2 * i + 1} {c12:.17g} {c6:.17g} 0.5\n"
                    for i in range(args.pairs)
                ),
            }
            if mode != "legacy":
                files["nb14_softcore"] = (
                    str(args.pairs)
                    + "\n"
                    + "".join(
                        f"{2 * i} {2 * i + 1} {delta:.17g}\n"
                        for i in range(args.pairs)
                    )
                )
            config = dict(
                md_name="nb14-repeated",
                mode="nve",
                dt=0.0,
                step_limit=args.steps,
                cutoff=8.0,
                print_zeroth_frame=True,
                print_pressure=True,
                write_mdout_interval=args.steps,
                write_information_interval=args.steps,
                write_trajectory_interval=args.steps,
                write_restart_file_interval=0,
                frc="force.dat",
            )
            config["PM.MPI_size"] = 1
            for name, text in files.items():
                (work / (name + ".txt")).write_text(text)
                config[name + "_in_file"] = name + ".txt"
            (work / "mdin.spg.toml").write_text(
                "\n".join(f"{k}={json.dumps(v)}" for k, v in config.items())
                + "\n"
            )
            start = time.monotonic()
            result = subprocess.run(
                [str(engine), "-mdin", "mdin.spg.toml"],
                cwd=work,
                capture_output=True,
                text=True,
                timeout=60,
            )
            wall = time.monotonic() - start
            (work / "engine.log").write_text(result.stdout + result.stderr)
            assert result.returncode == 0, str(work)
            force = np.fromfile(work / "force.dat", dtype=np.float32).reshape(
                -1, count, 3
            )
            assert len(force) >= 2 and np.isfinite(force).all(), str(work)
            s = 3.2**6 + delta
            expected = 3.2**5 * (6 * c6 / s**2 - 12 * c12 / s**3)
            oracle = np.zeros((count, 3))
            oracle[::2, 0] = expected
            oracle[1::2, 0] = -expected
            for frame in force:
                np.testing.assert_allclose(frame, oracle, rtol=3e-4, atol=0.003)
            np.testing.assert_allclose(
                force[0], force[-1], rtol=1e-6, atol=1e-6
            )
            elapsed = float(
                re.search(
                    r"Calculate_Force\s*\|\s*([\d.]+) seconds", result.stdout
                )[1]
            )
            records.append(
                dict(
                    mode=mode,
                    repeat=repeat,
                    wallSeconds=wall,
                    calculateForceSeconds=elapsed,
                    forceFrames=len(force),
                    maxForceError=float(np.max(abs(force - oracle))),
                )
            )
    summary = {
        mode: {
            "median": statistics.median(
                r["calculateForceSeconds"] for r in records if r["mode"] == mode
            ),
            "min": min(
                r["calculateForceSeconds"] for r in records if r["mode"] == mode
            ),
            "max": max(
                r["calculateForceSeconds"] for r in records if r["mode"] == mode
            ),
        }
        for mode in ("legacy", "zero", "soft")
    }
    report = dict(
        engine=str(engine),
        binarySha256=hashlib.sha256(engine.read_bytes()).hexdigest(),
        pairs=args.pairs,
        steps=args.steps,
        dt=0,
        records=records,
        calculateForceSummary=summary,
        timingScope="Whole Calculate_Force region; not isolated nb14 latency or MD throughput",
    )
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()

"""Nonperiodic energies/forces must survive update-group and input permutations."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError:
    raise SystemExit(77)

SPONGE = None


class AtomOrder(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="sponge-nopbc-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def run_case(self, name, order, constraints, gb, reverse=False, interval=1):
        directory = self.root / name
        directory.mkdir()
        n = 8
        order = np.asarray(order)
        inverse = np.argsort(order)
        # Noncontiguous constraint groups and heterogeneous immutable parameters.
        x = np.array(
            [
                [3, 3, 3],
                [7, 4, 4],
                [4, 8, 6],
                [4, 3, 3],
                [8, 4, 4],
                [4, 9, 6],
                [3, 4, 3],
                [7, 5, 4],
            ],
            float,
        )
        masses = np.array([12, 16, 14, 1, 1, 1, 1, 1])
        charges = np.array([1.3, -2.1, 0.8, -0.4, 0.5, -0.7, 0.2, 0.4])
        types = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        for key, values in (
            ("mass", masses[order]),
            ("charge", charges[order]),
        ):
            (directory / f"{key}.txt").write_text(
                str(n) + "\n" + "\n".join(map(str, values)) + "\n"
            )
        for key, values in (
            ("coordinate", x[order]),
            ("velocity", np.zeros((n, 3))),
        ):
            (directory / f"{key}.txt").write_text(
                str(n)
                + "\n"
                + "\n".join(" ".join(map(str, row)) for row in values)
                + ("\n999 999 999\n90 90 90\n" if key == "coordinate" else "\n")
            )
        bonds = [(0, 3), (0, 6), (1, 4), (1, 7), (2, 5), (4, 7)]
        (directory / "bond.txt").write_text(
            str(len(bonds))
            + "\n"
            + "\n".join(
                f"{inverse[a]} {inverse[b]} 0 {np.linalg.norm(x[a] - x[b]):.12g}"
                for a, b in bonds
            )
            + "\n"
        )
        pairs = bonds + [(3, 6), (1, 3)]
        rows = [[] for _ in range(n)]
        for a, b in pairs:
            a, b = sorted((int(inverse[a]), int(inverse[b])))
            if reverse:
                a, b = b, a
            rows[a].append(b)
        (directory / "exclude.txt").write_text(
            f"{n} {len(pairs)}\n"
            + "\n".join(" ".join(map(str, [len(row), *row])) for row in rows)
            + "\n"
        )
        a = np.arange(1, 37) * 100.0
        b = np.arange(1, 37) * 3.0
        (directory / "lj.txt").write_text(
            f"{n} 8\n"
            + "\n".join(map(str, [*a, *b, *map(int, types[order])]))
            + "\n"
        )
        radii = np.linspace(1.1, 1.9, n)
        scales = np.linspace(0.65, 0.9, n)
        (directory / "gb.txt").write_text(
            f"{n}\n"
            + "\n".join(f"{radii[i]} {scales[i]}" for i in order)
            + "\n"
        )
        options = dict(
            mode="nve",
            pbc=False,
            dt=1e-8,
            step_limit=2,
            cutoff=999,
            print_zeroth_frame=True,
            write_information_interval=interval,
            write_mdout_interval=interval,
            write_trajectory_interval=1,
            write_restart_file_interval=0,
            crd="out.dat",
            vel="vel.dat",
            frc="frc.dat",
        )
        for key in (
            "mass",
            "charge",
            "coordinate",
            "velocity",
            "bond",
            "exclude",
        ):
            options[f"{key}_in_file"] = f"{key}.txt"
        options["LJ_in_file"] = "lj.txt"
        options["PM.MPI_size"] = 0
        if gb:
            options["gb_in_file"] = "gb.txt"
        if constraints:
            options["constrain_mode"] = "SHAKE"
            options["settle_disable"] = constraints == "shake"
        (directory / "mdin.toml").write_text(
            "\n".join(f"{k}={json.dumps(v)}" for k, v in options.items()) + "\n"
        )
        result = subprocess.run(
            [SPONGE, "-mdin", "mdin.toml"],
            cwd=directory,
            env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
            capture_output=True,
            text=True,
            timeout=40,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        lines = (directory / "mdout.txt").read_text().splitlines()
        header = lines[0].split()
        energies = [
            dict(zip(header, map(float, line.split())))
            for line in lines[1:]
            if line.strip()
        ]
        self.assertTrue(energies, result.stdout)
        self.assertTrue(
            all(np.isfinite(list(row.values())).all() for row in energies)
        )
        force = np.fromfile(directory / "frc.dat", dtype=np.float32).reshape(
            -1, n, 3
        )[:, inverse]
        self.assertGreaterEqual(len(force), 2)
        self.assertTrue(np.isfinite(force).all())
        if not gb:
            # Independent all-pairs oracle: also detects accidentally excluding
            # legitimate interactions, not just permutation inconsistencies.
            reference = np.zeros((n, 3))
            lj_energy = coulomb_energy = 0.0
            excluded = {tuple(sorted(pair)) for pair in pairs}
            for i in range(n):
                for j in range(i + 1, n):
                    if (i, j) in excluded:
                        continue
                    dr = x[j] - x[i]
                    r2 = float(dr @ dr)
                    hi, lo = max(types[i], types[j]), min(types[i], types[j])
                    pair = hi * (hi + 1) // 2 + lo
                    lj_energy += a[pair] / r2**6 - b[pair] / r2**3
                    coulomb_energy += charges[i] * charges[j] / r2**0.5
                    fij = (
                        -12 * a[pair] / r2**7
                        + 6 * b[pair] / r2**4
                        - charges[i] * charges[j] / r2**1.5
                    ) * dr
                    reference[i] += fij
                    reference[j] -= fij
            self.assertAlmostEqual(energies[0]["LJ"], lj_energy, delta=0.006)
            self.assertAlmostEqual(
                energies[0]["Coulomb"], coulomb_energy, delta=0.006
            )
            np.testing.assert_allclose(
                force[0], reference, atol=3e-4, rtol=3e-4
            )
        return energies, force

    def test_permutation_and_constraints(self):
        for gb in (False, True):
            with self.subTest(gb=gb):
                expected, forces = self.run_case(
                    f"baseline-{gb}", range(8), False, gb
                )
                for i, (order, constraints, reverse, interval) in enumerate(
                    [
                        (range(8), "shake", False, 1),
                        (range(8), "settle", False, 1),
                        ([4, 2, 6, 0, 7, 3, 1, 5], "shake", False, 1),
                        ([7, 6, 5, 4, 3, 2, 1, 0], "settle", True, 1),
                        (range(8), "shake", False, 2),
                        (range(8), False, True, 1),
                    ]
                ):
                    actual, frc = self.run_case(
                        f"case-{gb}-{i}",
                        order,
                        constraints,
                        gb,
                        reverse,
                        interval,
                    )
                    for key in (
                        "LJ",
                        "Coulomb",
                        "potential",
                        *(["gb"] if gb else []),
                    ):
                        by_step = {row["step"]: row for row in expected}
                        for row in actual:
                            self.assertAlmostEqual(
                                row[key],
                                by_step[row["step"]][key],
                                delta=0.02,
                                msg=f"{gb=} {i=} {key=}",
                            )
                    np.testing.assert_allclose(
                        frc,
                        forces,
                        rtol=3e-4,
                        atol=3e-4,
                        err_msg=f"{gb=} {i=}",
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sponge", required=True)
    args, remaining = parser.parse_known_args()
    SPONGE = str(Path(args.sponge).resolve())
    unittest.main(argv=[sys.argv[0], *remaining])

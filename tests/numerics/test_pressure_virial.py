"""Ordinary mdout stress against independent energy derivatives (no GK module)."""

import argparse
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

try:
    import numpy as np
except ImportError:
    raise SystemExit(77)

SPONGE = None
AXES = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
NAMES = ("Pxx", "Pyy", "Pzz", "Pxy", "Pxz", "Pyz")
# mdout reports bar, using the engine's historical conversion constant.
BAR = float(np.float32(6.946827162543585e4))


class PressureVirial(unittest.TestCase):
    def run_case(
        self, positions, masses, *, charges=None, options=None, files=None
    ):
        with tempfile.TemporaryDirectory(prefix="sponge-virial-") as tmp:
            path = Path(tmp)
            n = len(masses)
            for name, values in (
                ("mass", masses),
                ("charge", [0] * n if charges is None else charges),
            ):
                (path / f"{name}.txt").write_text(
                    str(n) + "\n" + "\n".join(map(str, values)) + "\n"
                )
            for name, values in (
                ("coordinate", positions),
                ("velocity", np.zeros((n, 3))),
            ):
                tail = (
                    "\n32 32 32\n90 90 90\n" if name == "coordinate" else "\n"
                )
                (path / f"{name}.txt").write_text(
                    str(n)
                    + "\n"
                    + "\n".join(" ".join(map(str, row)) for row in values)
                    + tail
                )
            for name, data in (files or {}).items():
                (path / name).write_text(data)
            config = dict(
                md_name="virial",
                mode="nve",
                pbc=True,
                dt=0.0,
                step_limit=1,
                cutoff=8.0,
                print_zeroth_frame=True,
                print_pressure=True,
                write_mdout_interval=1,
                write_information_interval=1,
                write_trajectory_interval=1,
                write_restart_file_interval=0,
                frc="frc.dat",
            )
            config["PM.MPI_size"] = 1
            config.update(
                {
                    f"{name}_in_file": f"{name}.txt"
                    for name in ("mass", "charge", "coordinate", "velocity")
                }
            )
            config.update(options or {})
            (path / "mdin.spg.toml").write_text(
                "\n".join(f"{k}={json.dumps(v)}" for k, v in config.items())
                + "\n"
            )
            run = subprocess.run(
                [SPONGE, "-mdin", "mdin.spg.toml"],
                cwd=path,
                env={
                    **os.environ,
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                },
                capture_output=True,
                text=True,
                timeout=45,
            )
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            lines = (path / "mdout.txt").read_text().splitlines()
            header = next(
                i
                for i, line in enumerate(lines)
                if all(key in line.split() for key in NAMES)
            )
            keys = lines[header].split()
            rows = [
                dict(zip(keys, map(float, line.split())))
                for line in lines[header + 1 :]
                if len(line.split()) == len(keys)
            ]
            self.assertTrue(rows, "missing pressure rows")
            stress = np.array([rows[0][key] for key in NAMES])
            self.assertTrue(np.isfinite(stress).all())
            force = (
                np.frombuffer((path / "frc.dat").read_bytes(), np.float32)
                .reshape(-1, n, 3)[0]
                .copy()
            )
            return stress, force

    def test_dihedral_energy_strain(self):
        original = np.array(
            [
                [1.0, 2.0, 3.0],
                [2.5, 2.25, 3.5],
                [3.0, 3.5, 3.25],
                [4.25, 3.0, 4.5],
            ]
        )
        coefficients = np.array([0.7, -0.14, 1.57], dtype=np.float32).astype(
            float
        )

        def energy(q):
            axis = q[2] - q[1]
            axis /= np.linalg.norm(axis)
            v, w = q[0] - q[1], q[3] - q[2]
            v -= axis * (v @ axis)
            w -= axis * (w @ axis)
            c = (v @ w) / np.linalg.norm(v) / np.linalg.norm(w)
            return coefficients @ [1 + c, 2 - 2 * c * c, 1 + 4 * c**3 - 3 * c]

        for transform, shift in [
            (np.eye(3), 0),
            (np.eye(3), 29),
            (np.diag([-1, 1, 1]), 0),
            (np.eye(3)[:, [2, 0, 1]], 0),
        ]:
            with self.subTest(shift=shift, transform=transform.tolist()):
                q = original @ transform
                force = np.zeros_like(q)
                tensor = np.zeros((3, 3))
                h = 1e-5
                for index in np.ndindex(q.shape):
                    plus, minus = q.copy(), q.copy()
                    plus[index] += h
                    minus[index] -= h
                    force[index] = -(energy(plus) - energy(minus)) / (2 * h)
                for a, b in itertools.product(range(3), repeat=2):
                    plus, minus = q.copy(), q.copy()
                    plus[:, a] += h * q[:, b]
                    minus[:, a] -= h * q[:, b]
                    tensor[a, b] = -(energy(plus) - energy(minus)) / (2 * h)
                np.testing.assert_allclose(tensor, tensor.T, atol=1e-8)
                np.testing.assert_allclose(tensor, force.T @ q, atol=1e-8)
                terms = [
                    f"0 1 2 3 {i + 1} {c:.17g} {np.pi if i == 1 else 0:.17g}"
                    for i, c in enumerate(coefficients)
                ]
                actual, actual_force = self.run_case(
                    (q + shift) % 32,
                    [15, 14, 14, 15],
                    options={"dihedral_in_file": "dihedral.txt"},
                    files={"dihedral.txt": "3\n" + "\n".join(terms) + "\n"},
                )
                expected = (
                    np.array([tensor[a, b] for a, b in AXES]) * BAR / 32**3
                )
                # mdout has fixed decimal output; allow its rounding as well as float arithmetic.
                np.testing.assert_allclose(
                    actual, expected, rtol=4e-5, atol=5.1e-3
                )
                np.testing.assert_allclose(
                    actual_force, force, rtol=3e-5, atol=3e-5
                )

    def test_pme_energy_strain_even_and_odd_fft(self):
        q = np.array([1.0, -1.0]) * float(np.float32(18.2223))
        positions = np.array([[1.0, 2.0, 3.0], [4.0, 6.0, 5.0]])
        images = np.array(list(itertools.product(range(-1, 2), repeat=3)))
        modes = np.array(
            [
                n
                for n in itertools.product(range(-16, 17), repeat=3)
                if n != (0, 0, 0)
            ]
        )

        def energy(deformation):
            alpha = 0.25
            cell = deformation * 32
            xyz = positions @ deformation.T
            value = -alpha / math.sqrt(math.pi) * np.sum(q * q)
            for i, j in itertools.product(range(2), repeat=2):
                distances = np.linalg.norm(
                    xyz[i] - xyz[j] + images @ cell.T, axis=1
                )
                for distance in distances[distances > 0]:
                    value += (
                        0.5
                        * q[i]
                        * q[j]
                        * math.erfc(alpha * distance)
                        / distance
                    )
            wave = 2 * np.pi * modes @ np.linalg.inv(cell)
            k2 = (wave * wave).sum(axis=1)
            rho = np.exp(1j * wave @ xyz.T) @ q
            return value + 2 * np.pi / np.linalg.det(cell) * np.sum(
                np.exp(-k2 / (4 * alpha**2)) / k2 * abs(rho) ** 2
            )

        references = []
        for h in (1e-4, 5e-5):
            result = []
            for a, b in AXES:
                plus, minus = np.eye(3), np.eye(3)
                plus[a, b] += h
                minus[a, b] -= h
                result.append(
                    -(energy(plus) - energy(minus)) / (2 * h) * BAR / 32**3
                )
            references.append(result)
        np.testing.assert_allclose(references[0], references[1], rtol=2e-7)
        for grid in (64, 63):
            with self.subTest(grid=grid):
                actual, _ = self.run_case(
                    positions,
                    [1, 1],
                    charges=q,
                    options={
                        "PM.fftx": grid,
                        "PM.ffty": grid,
                        "PM.fftz": grid,
                        "LJ_in_file": "lj.txt",
                    },
                    files={"lj.txt": "2 1\n0\n0\n0\n0\n"},
                )
                np.testing.assert_allclose(
                    actual, references[1], rtol=2e-4, atol=5.1e-3
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sponge", required=True)
    args, extra = parser.parse_known_args()
    SPONGE = str(Path(args.sponge).resolve())
    unittest.main(argv=[__file__, *extra])

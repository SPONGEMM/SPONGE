"""Constraint storage boundary against an independent old-bond Newton solve."""

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError:
    raise SystemExit(77)

SPONGE = None
BAR = float(np.float32(6.946827162543585e4))
H = float(np.float32(0.001 * float(np.float32(20.455))))
AXES = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
NAMES = ("Pxx", "Pyy", "Pzz", "Pxy", "Pxz", "Pyz")


def reference(old, trial, mass, pairs, lengths):
    """Solve position constraints along old bonds, without production formulas."""
    directions = np.zeros((len(pairs), len(old), 3))
    for i, (a, b) in enumerate(pairs):
        d = old[a] - old[b]
        directions[i, a] = d / mass[a]
        directions[i, b] = -d / mass[b]
    q = trial.copy()
    for _ in range(50):
        residual = np.zeros(len(pairs))
        jacobian = np.zeros((len(pairs), len(pairs)))
        for i, (a, b) in enumerate(pairs):
            d = q[a] - q[b]
            residual[i] = d @ d - lengths[i] ** 2
            jacobian[i] = 2 * ((directions[:, a] - directions[:, b]) @ d)
        if np.max(np.abs(residual)) < 1e-14:
            return q
        q -= np.einsum(
            "i,iad->ad", np.linalg.solve(jacobian, residual), directions
        )
    raise AssertionError("independent Newton reference did not converge")


class ConstraintPrecision(unittest.TestCase):
    def run_case(
        self, mode, precision, size=4, generated=False, disable_settle=True
    ):
        if mode == "SETTLE":
            if size == 2:
                q = np.array([[0, 0, 0], [1, 0, 0]])
                pairs = [(0, 1)]
            else:
                q = np.array([[0, 0, 0], [0.375, 0.5, 0], [-0.375, 0.5, 0]])
                pairs = [(0, 1), (0, 2), (1, 2)]
        else:
            q = np.array(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [1, 0.75, 1],
                    [1.5, 1.5, 1.5],
                    [2.5, 1.5, 1.5],
                ]
            )[:size]
            pairs = [(i, i + 1) for i in range(len(q) - 1)]
        q = (q + 2).astype(np.float32).astype(float)
        mass = np.ones(len(q))
        mass[0] = 16
        v = np.array(
            [
                [0.03, -0.02, 0.01],
                [-0.04, 0.05, -0.01],
                [0.02, -0.03, 0.04],
                [0.01, 0.02, -0.03],
                [-0.02, 0.01, 0.03],
            ]
        )[: len(q)]
        lengths = np.array([np.linalg.norm(q[a] - q[b]) for a, b in pairs])
        with tempfile.TemporaryDirectory(prefix="sponge-constraint-") as tmp:
            path = Path(tmp)
            for name, values in (("mass", mass), ("charge", np.zeros(len(q)))):
                (path / f"{name}.txt").write_text(
                    str(len(q)) + "\n" + "\n".join(map(str, values)) + "\n"
                )
            for name, values in (("coordinate", q), ("velocity", v)):
                tail = (
                    "\n32 32 32\n90 90 90\n" if name == "coordinate" else "\n"
                )
                (path / f"{name}.txt").write_text(
                    str(len(q))
                    + "\n"
                    + "\n".join(" ".join(map(str, row)) for row in values)
                    + tail
                )
            (path / "constraints.txt").write_text(
                str(len(pairs))
                + "\n"
                + "\n".join(
                    f"{a} {b} {d:.17g}" for (a, b), d in zip(pairs, lengths)
                )
                + "\n"
            )
            config = dict(
                md_name="constraint_precision",
                mode="nve",
                pbc=True,
                dt=0.001,
                step_limit=1,
                cutoff=8.0,
                print_zeroth_frame=True,
                print_pressure=True,
                write_mdout_interval=1,
                write_information_interval=1,
                write_trajectory_interval=1,
                write_restart_file_interval=0,
                crd="crd.dat",
                vel="vel.dat",
                constrain_mode=mode,
                settle_disable=disable_settle and mode != "SETTLE",
            )
            config.update(
                {
                    f"{name}_in_file": f"{name}.txt"
                    for name in ("mass", "charge", "coordinate", "velocity")
                }
            )
            config["PM.MPI_size"] = 1
            config["constrain.in_file"] = "constraints.txt"
            if precision is not None:
                config["constrain_precision"] = precision
            if mode == "SHAKE":
                config.update(
                    {"SHAKE.iteration_numbers": 200, "SHAKE.tolerance": 1e-10}
                )
            if mode == "LINCS":
                config.update({"LINCS.order": 24, "LINCS.corrections": 4})
            if mode == "CCMA":
                config.update(
                    {
                        "CCMA.iteration_numbers": 8
                        if precision == "single"
                        else 30,
                        "CCMA.tolerance": 1e-6
                        if precision == "single"
                        else 1e-10,
                        "CCMA.inverse_cutoff": 0.0,
                    }
                )
            if generated:
                del config["velocity_in_file"]
                config.update(
                    {
                        "initial_velocity.mode": "maxwell",
                        "initial_velocity.seed": 17,
                        "target_temperature": 300.0,
                    }
                )
            (path / "mdin.spg.toml").write_text(
                "\n".join(
                    f"{k}={json.dumps(value)}" for k, value in config.items()
                )
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
                timeout=30,
            )
            if os.environ.get("SPONGE_CONSTRAINT_EVIDENCE"):
                destination = (
                    Path(os.environ["SPONGE_CONSTRAINT_EVIDENCE"])
                    / f"{mode}-{precision}-{size}-{generated}"
                )
                shutil.copytree(path, destination, dirs_exist_ok=True)
                (destination / "stdout.log").write_text(run.stdout + run.stderr)
            if precision == "invalid":
                self.assertNotEqual(run.returncode, 0)
                self.assertIn(
                    "constrain_precision must be single or double",
                    run.stdout + run.stderr,
                )
                return None
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn(
                f"constraint internal precision: {precision or 'double'}",
                run.stdout,
            )
            positions = (
                np.frombuffer((path / "crd.dat").read_bytes(), np.float32)
                .reshape(-1, len(q), 3)
                .astype(float)
            )
            velocities = (
                np.frombuffer((path / "vel.dat").read_bytes(), np.float32)
                .reshape(-1, len(q), 3)
                .astype(float)
            )
            self.assertEqual(len(positions), 1)
            self.assertEqual(len(velocities), 1)
            self.assertTrue(
                np.isfinite(positions).all() and np.isfinite(velocities).all()
            )
            lines = (path / "mdout.txt").read_text().splitlines()
            header = next(
                i
                for i, line in enumerate(lines)
                if all(k in line.split() for k in NAMES)
            )
            keys = lines[header].split()
            rows = [
                dict(zip(keys, map(float, line.split())))
                for line in lines[header + 1 :]
                if len(line.split()) == len(keys)
            ]
            # The step-0 iteration produces the sole q_1/v_(1/2) frame.
            # Its constraint stress is printed in row 0; row 1 is the final
            # force-only observation and must not be paired with that update.
            pressure = np.array([rows[0][k] for k in NAMES])
            if generated:
                self.assertIn("INITIAL VELOCITY", run.stdout)
                for (a, b), length in zip(pairs, lengths):
                    self.assertLess(
                        abs(
                            np.linalg.norm(positions[-1, a] - positions[-1, b])
                            - length
                        ),
                        2e-6,
                    )
                return positions, velocities, pressure
            incoming = v.astype(np.float32).astype(float)
            # CPU/GPU FMA choices can round the float trial by one ULP. Include
            # this storage boundary in the comparison; do not call it FP64 MD.
            trial = (q + H * incoming).astype(np.float32).astype(float)
            expected = reference(q, trial, mass, pairs, lengths)
            correction = expected - trial
            expected_v = incoming + correction / H
            np.testing.assert_allclose(
                positions[-1],
                expected,
                rtol=0,
                atol=6e-7
                + (
                    1e-6 * max(lengths)
                    if precision == "single" and mode == "CCMA"
                    else 0
                ),
            )
            np.testing.assert_allclose(
                velocities[-1],
                expected_v,
                rtol=0,
                atol=1e-4 if precision == "single" else 2e-5,
            )
            for (a, b), d in zip(pairs, lengths):
                self.assertLess(
                    abs(
                        np.linalg.norm(positions[-1, a] - positions[-1, b]) - d
                    ),
                    1e-6,
                )
            np.testing.assert_allclose(
                np.sum(mass[:, None] * (velocities[-1] - incoming), axis=0),
                0,
                atol=2e-6,
            )
            force = mass[:, None] * correction / H**2
            tensor = (
                force.T @ q
                + (mass[:, None] * velocities[-1]).T @ velocities[-1]
            )
            # mdout converts the internally doubled off-diagonal components back.
            expected_p = (
                np.array(
                    [
                        tensor[a, b]
                        if a == b
                        else 0.5 * (tensor[a, b] + tensor[b, a])
                        for a, b in AXES
                    ]
                )
                * BAR
                / 32**3
            )
            np.testing.assert_allclose(pressure, expected_p, rtol=0, atol=0.035)
            return positions, velocities, pressure

    def test_algorithms_small_and_general(self):
        for precision in ("single", "double"):
            for mode in ("SETTLE", "SHAKE", "LINCS", "CCMA"):
                for size in (2, 3) if mode == "SETTLE" else (4, 5):
                    with self.subTest(
                        precision=precision, mode=mode, size=size
                    ):
                        self.run_case(mode, precision, size)

    def test_initial_velocity_projection(self):
        for precision in ("single", "double"):
            for mode in ("SETTLE", "SHAKE", "LINCS", "CCMA"):
                with self.subTest(precision=precision, mode=mode):
                    self.run_case(mode, precision, generated=True)

    def test_default_and_invalid_precision(self):
        default = self.run_case("SETTLE", None)
        explicit = self.run_case("SETTLE", "double")
        for actual, expected in zip(default, explicit):
            np.testing.assert_array_equal(actual, expected)
        self.run_case("SHAKE", "invalid")

    def test_runtime_breakdown_is_fatal(self):
        for reason in (
            "nonfinite_energy",
            "nonfinite_kinetic",
            "impossible_settle",
            "finite",
        ):
            for interval, zeroth, steps in (
                (1, True, 2),
                (100, False, 2),
                (100, False, 0),
            ):
                with self.subTest(
                    reason=reason, interval=interval, steps=steps
                ):
                    self.check_runtime_exit(reason, interval, zeroth, steps)

    def check_runtime_exit(self, reason, interval, zeroth, steps):
        with tempfile.TemporaryDirectory(
            prefix="sponge-runtime-breakdown-"
        ) as temporary:
            path = Path(temporary)
            overlap = reason == "nonfinite_energy"
            (path / "mass.txt").write_text("2\n12\n1\n")
            (path / "charge.txt").write_text(
                "2\n1\n1\n" if overlap else "2\n0\n0\n"
            )
            (path / "coordinate.txt").write_text(
                "2\n2 2 2\n"
                + ("2 2 2\n" if overlap else "3 2 2\n")
                + "1000 1000 1000\n90 90 90\n"
            )
            (path / "velocity.txt").write_text(
                "2\n0 0 0\n0 1e30 0\n"
                if reason == "nonfinite_kinetic"
                else "2\n0 0 0\n0 100 0\n"
            )
            config = dict(
                md_name=reason,
                mode="nve",
                pbc=False,
                dt=0.001,
                step_limit=steps,
                cutoff=8,
                print_zeroth_frame=zeroth,
                write_mdout_interval=interval,
                write_information_interval=1,
                write_trajectory_interval=0,
                write_restart_file_interval=0,
            )
            config.update(
                {
                    f"{name}_in_file": f"{name}.txt"
                    for name in ("mass", "charge", "coordinate", "velocity")
                }
            )
            config["PM.MPI_size"] = 1
            if reason == "impossible_settle":
                # Perpendicular trial displacement exceeds the target bond:
                # no correction parallel to the old bond can solve it.
                (path / "constraints.txt").write_text("1\n0 1 1\n")
                config.update(
                    constrain_mode="SETTLE", constrain_precision="double"
                )
                config["constrain.in_file"] = "constraints.txt"
            (path / "mdin.spg.toml").write_text(
                "\n".join(
                    f"{key}={json.dumps(value)}"
                    for key, value in config.items()
                )
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
                timeout=30,
            )
            output = run.stdout + run.stderr
            if reason == "finite":
                self.assertEqual(run.returncode, 0, output)
                self.assertIn("Core Run Speed:", output)
                return
            self.assertNotEqual(run.returncode, 0, output)
            self.assertIn(
                "SETTLE"
                if reason == "impossible_settle"
                else "CONTROLLER::Check_Error",
                output,
            )
            self.assertNotIn("Core Run Speed:", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sponge", required=True)
    args, remaining = parser.parse_known_args()
    SPONGE = str(Path(args.sponge).resolve())
    unittest.main(argv=[__file__, *remaining])

"""Independent constraints/thermal-state oracles for the water benchmark."""

from pathlib import Path
import sys
import tempfile
import unittest

try:
    import h5py  # noqa: F401 -- required by the benchmark's profile reader
    import numpy as np
except ImportError:
    raise SystemExit(77)

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "benchmarks/viscosity")
)
import water_model as water
import water_cosine_validation as check


class WaterCosineReference(unittest.TestCase):
    def test_batch_projection_matches_global_mass_metric_nullspace(self):
        with tempfile.TemporaryDirectory() as directory:
            model = water.prepare(
                Path(directory) / "model", cells=2, cutoff=2.0
            )
            _, state = water.load_model(Path(directory) / "model")
            q, m, pairs = state["positions"], state["masses"], state["pairs"]
            jac = np.zeros((len(pairs), q.size))
            for row, (a, b, length) in enumerate(pairs):
                a, b = int(a), int(b)
                d = q[a] - q[b]
                d -= model["box_angstrom"] * np.rint(d / model["box_angstrom"])
                jac[row, 3 * a : 3 * a + 3] = d / np.linalg.norm(d)
                jac[row, 3 * b : 3 * b + 3] = -d / np.linalg.norm(d)
            inv = 1 / np.repeat(m, 3)
            projector = (
                np.eye(q.size)
                - (inv[:, None] * jac.T)
                @ np.linalg.pinv((jac * inv) @ jac.T)
                @ jac
            )
            v = np.random.default_rng(19).normal(size=q.shape)
            actual = check.tangent(
                v, check.tangent_geometry(q, m, model["box_angstrom"])
            )
            expected = (projector @ v.ravel()).reshape(q.shape)
            np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=1e-13)
            np.testing.assert_allclose(jac @ actual.ravel(), 0, atol=1e-13)
            np.testing.assert_allclose(
                (m[:, None] * actual).sum(axis=0),
                (m[:, None] * v).sum(axis=0),
                atol=1e-12,
            )

            # Global weighted QR supplies a separate oracle for the joint
            # translation/mode thermal subspace and its actual rank.
            basis = np.zeros((q.size, 4))
            for axis in range(3):
                basis[axis::3, axis] = 1
            basis[0::3, 3] = np.cos(2 * np.pi * q[:, 2] / model["box_angstrom"])
            basis = projector @ basis
            weighted = np.sqrt(np.repeat(m, 3))[:, None] * basis
            orth, _ = np.linalg.qr(weighted)
            flat = actual.ravel() * np.sqrt(np.repeat(m, 3))
            residual = flat - orth @ (orth.T @ flat)
            temperature = (residual @ residual) / (
                (q.size - len(pairs) - 4) * water.KB
            )
            measured = check.measure(q, actual, m, model["box_angstrom"])
            self.assertAlmostEqual(
                measured["thermal_temperature"], temperature, places=10
            )
            boosted = check.measure(
                q, actual + np.array([0.3, -0.4, 0.2]), m, model["box_angstrom"]
            )
            self.assertAlmostEqual(
                boosted["thermal_temperature"], temperature, places=10
            )

    def test_converted_acceleration_generates_same_native_force(self):
        mass = np.array([15.9994, 1.008, 1.008])
        q = np.array([[0, 0, 0], [0, 0, 2], [0, 0, 5.0]])
        amplitude = 0.05
        box = 10.0
        force = check.force_drive(q, mass, dict(box_angstrom=box), amplitude)
        lammps_acceleration = (
            amplitude * 10 / (water.SP_TIME * check.LM_TIME) ** 2
        )
        expected = (
            mass
            * lammps_acceleration
            * check.LM_TIME**2
            * np.cos(2 * np.pi * q[:, 2] / box)
        )
        np.testing.assert_allclose(
            force[:, 0], expected, atol=1e-17, rtol=1e-14
        )


if __name__ == "__main__":
    unittest.main()

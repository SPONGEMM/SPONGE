"""Independent derivatives and immutable-input checks for the LJ benchmark."""
from pathlib import Path
import sys
import tempfile
import unittest

try:
    import numpy as np
    import h5py
except ImportError:
    raise SystemExit(77)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"benchmarks/viscosity"))
import lj_model as lj


class Model(unittest.TestCase):
    def test_force_and_volume_derivatives(self):
        crd = np.array([[1., 2., 3.], [2.1, 2.7, 3.2], [5.1, 3.7, 2.8]])
        box, c12, c6, cutoff = 12., 4., 4., 5.
        energy, force, virial, distance = lj.pair_reference(crd, box, c12, c6, cutoff)
        h = 1e-5
        for i in range(len(crd)):
            for axis in range(3):
                plus, minus = crd.copy(), crd.copy()
                plus[i, axis] += h; minus[i, axis] -= h
                derivative = (lj.pair_reference(plus, box, c12, c6, cutoff)[0]-lj.pair_reference(minus, box, c12, c6, cutoff)[0])/(2*h)
                self.assertAlmostEqual(force[i, axis], -derivative, places=7)
        derivative = (lj.pair_reference(crd*(1+h), box*(1+h), c12, c6, cutoff)[0]-lj.pair_reference(crd*(1-h), box*(1-h), c12, c6, cutoff)[0])/(2*h)
        self.assertAlmostEqual(np.trace(virial), -derivative, places=7)
        np.testing.assert_allclose(force.sum(axis=0), 0, atol=1e-14)
        np.testing.assert_allclose(virial, virial.T, atol=1e-14)
        shifted = crd.copy(); shifted[0] += [box, -box, 2*box]
        translated = lj.pair_reference(shifted, box, c12, c6, cutoff)
        np.testing.assert_allclose(translated[1], force, rtol=1e-12, atol=1e-13)

    def test_model_hash_and_geometry_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = lj.prepare(root/"model")
            self.assertLess(2*model["cutoff_angstrom"], model["box_angstrom"])
            for name, digest in model["input_sha256"].items():
                self.assertEqual(lj.digest(root/"model"/name), digest)
            with self.assertRaises(FileExistsError): lj.prepare(root/"model")
            with self.assertRaises(ValueError): lj.prepare(root/"small", cells=2)
            (root/"model/lj.txt").write_text("changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                lj.static_check(root/"model", root/"run", None, None)
            self.assertFalse((root/"run").exists())


if __name__ == "__main__":
    unittest.main()

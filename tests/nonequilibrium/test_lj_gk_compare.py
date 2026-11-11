"""Paired benchmark guards and cutoff-energy continuity; no MD convergence claim."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
try:
    import numpy as np
    import h5py
except ImportError:
    raise SystemExit(77)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'benchmarks/viscosity'))
import lj_model as lj
import lj_gk_compare as pair


class PairedGK(unittest.TestCase):
    def test_shifted_energy_is_continuous_across_cutoff(self):
        box, cutoff, c12, c6 = 10., 2.5, 4., 4.
        values = []
        for displacement in (-1e-6, 1e-6):
            q = np.array([[0., 0., 0.], [cutoff+displacement, 0., 0.]])
            raw = lj.pair_reference(q, box, c12, c6, cutoff)[0]
            count, nearest, ambiguous = pair.cutoff_audit(q, box, cutoff)
            self.assertEqual(ambiguous, 1)
            self.assertLess(nearest, 2e-6)
            values.append(raw-count*(c12/cutoff**12-c6/cutoff**6))
            q[1, 0] -= box
            self.assertEqual(pair.cutoff_audit(q, box, cutoff)[0], count)
        self.assertLess(abs(values[0]-values[1]), 4e-8)

    def test_lammps_reader_units_cadence_and_com_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'stress.dat'
            rows = np.array([[i, 300, 8000, 1, 2, 3, 4, 5, 6, -10, 5] for i in range(0, 11, 2)], dtype=float)
            np.savetxt(path, rows)
            result = pair.read_lammps(path, 10, 2, .002)
            np.testing.assert_allclose(result['time'], np.arange(0, 11, 2)*.002)
            np.testing.assert_allclose(result['pressure'][0, 3:], np.array([4, 5, 6])*101325)
            self.assertEqual(result['volume'][0], 8)
            for bad in (rows[1:], rows[:-1], rows[:, :-1]):
                np.savetxt(path, bad)
                with self.assertRaises(ValueError): pair.read_lammps(path, 10, 2, .002)
            bad = rows.copy(); bad[2, 0] = 5; np.savetxt(path, bad)
            with self.assertRaises(ValueError): pair.read_lammps(path, 10, 2, .002)

    def test_seed_identity_and_centering_before_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = lj.prepare(root/'model')
            seed = root/'seed'; seed.mkdir()
            with np.load(root/'model/state.npz') as state:
                np.savez(seed/'structural_seed.npz', positions=state['positions'],
                         centered_velocities=state['velocities']+[.2, .3, .4], box=model['box_angstrom'])
            report = dict(seed_sha256=lj.digest(seed/'structural_seed.npz'), model_sha256=lj.digest(root/'model/model.json'))
            pair.write_json(seed/'summary.json', report)
            actual, q, half, u, h = pair.frozen_inputs(root/'model', seed/'structural_seed.npz')
            _, force, _, _ = lj.pair_reference(q, model['box_angstrom'], model['c12'], model['c6'], model['cutoff_angstrom'])
            np.testing.assert_array_equal(u, half+.5*h*force/model['mass_da'])
            self.assertLess(np.abs(u.mean(axis=0)).max(), 1e-8)
            with self.assertRaises(ValueError):
                pair.prepare(root/'model', seed/'structural_seed.npz', root/'invalid', 1000, 2, 100, 10, 10, [.5])
            self.assertFalse((root/'invalid').exists())
            plan = pair.prepare(root/'model', seed/'structural_seed.npz', root/'valid', 2000, 2, 100, 0, 1, [.1])
            self.assertFalse(plan['formal_equivalence_eligible'])
            script = (root/'valid/lammps/in.lammps').read_text()
            self.assertIn('compute pressure all pressure kinetic', script)
            self.assertIn('format " %.17g"', script)
            report['seed_sha256'] = 'changed'; pair.write_json(seed/'summary.json', report)
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                pair.frozen_inputs(root/'model', seed/'structural_seed.npz')

    def test_energy_diagnostic_uses_centered_kinetic(self):
        result = pair.energy_diagnostics(np.arange(3)*100, [1, 3, 5], [5, 3, 1], .5, 10)
        self.assertEqual(result['energy_range_per_atom_epsilon'], 0)
        self.assertEqual(result['energy_change_per_atom_epsilon'], 0)


if __name__ == '__main__':
    unittest.main()

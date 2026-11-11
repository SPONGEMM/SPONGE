"""Physical tangent seed, explicit half-step conversion and real NVT/NVE handoff."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest

try:
    import h5py
    import numpy as np
except ImportError:
    sys.exit(77)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'benchmarks/viscosity'))
import water_model as water
import water_pilot as pilot
import water_nve_audit as audit

SPONGE = None


class WaterPilot(unittest.TestCase):
    def test_explicit_timestep_matches_md_core_single_rounding(self):
        # These are the float results of MD_core's atof(dt)*float(20.455).
        # The pre-rounded-dt expression differs by one ULP in each case.
        for dt, expected in ((.0005, .010227499529719353),
                             (.00025, .005113749764859676),
                             (.000125, .002556874882429838)):
            self.assertEqual(water.internal_timestep(dt), expected)
            self.assertNotEqual(expected, float(np.float32(dt)*np.float32(water.SP_TIME)))

    def test_kinetic_audit_retains_com_and_centers_force(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = water.prepare(Path(tmp)/'model', cells=2, cutoff=2.)
            _, state = water.load_model(Path(tmp)/'model')
            q = state['positions']; mass = state['masses']
            local = q.reshape(-1, 3, 3)-q.reshape(-1, 3, 3)[:, :1]
            local -= model['box_angstrom']*np.floor(local/model['box_angstrom']+.5)
            local -= np.average(local, axis=1, weights=mass[:3])[:, None]
            # Independent rigid rotation plus known uniform translation is tangent.
            u = np.cross(np.array([.2, -.3, .4]), local).reshape(-1, 3)
            boost = np.array([.15, -.2, .3]); u += boost
            force = np.random.default_rng(42).normal(size=q.shape)
            h = .01
            result = audit.frame_audit(q, u-.5*h*force/mass[:, None], force, state, model, h)
            np.testing.assert_allclose(result['momentum'], mass.sum()*boost, rtol=0, atol=1e-13)
            self.assertAlmostEqual(result['kinetic_com'], .5*mass.sum()*(boost@boost), places=12)
            self.assertAlmostEqual(result['kinetic_full'], .5*np.sum(mass[:, None]*u*u), places=12)
            self.assertLess(result['projection_momentum_error'], 1e-13)

    def test_tangent_maxwell_and_half_step_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = water.prepare(Path(tmp)/'model', cells=2, cutoff=2.)
            _, state = water.load_model(Path(tmp)/'model')
            q = state['positions']; mass = state['masses']; pairs = state['pairs']
            u, temp = pilot.centered_seed(q, state, model, 7)
            np.testing.assert_array_equal(u, pilot.centered_seed(q, state, model, 7)[0])
            np.testing.assert_allclose(np.sum(mass[:, None]*u, axis=0), 0., atol=1e-14)
            self.assertGreater(abs(temp-298.15), 1.)  # draw, not exact-temperature scaling
            for a, b, _ in pairs:
                a, b = int(a), int(b); d = q[a]-q[b]
                d -= model['box_angstrom']*np.floor(d/model['box_angstrom']+.5)
                self.assertAlmostEqual(float(d@(u[a]-u[b])), 0., delta=1e-14)
            force = np.random.default_rng(8).normal(size=q.shape)
            for dt in (.0005, .00025):
                half, observed, h = pilot.half_step(u, force, state, q, model, dt)
                self.assertEqual(h, water.internal_timestep(dt))
                np.testing.assert_allclose(observed['velocity'], u, atol=1e-7, rtol=0)
                self.assertAlmostEqual(observed['temperature'], temp, delta=2e-5)

    def test_cutoff_energy_shift_counts_only_intermolecular_sites(self):
        model = dict(atoms=6, box_angstrom=20., cutoff_angstrom=8., effective_c12=10., effective_c6=2.)
        # Only one intermolecular OO pair; zero charges isolate exact LJ offset.
        x = np.array([[1, 1, 1], [1, 2, 1], [2, 1, 1], [4, 1, 1], [4, 2, 1], [5, 1, 1]], float)
        state = {'native_charges': np.zeros(6)}
        self.assertAlmostEqual(pilot.energy_shift(x, state, model), 10/8**12-2/8**6, places=15)
        x[3:] += np.array([8, 0, 0])
        self.assertEqual(pilot.energy_shift(x, state, model), 0.)

    def test_real_structural_handoff_and_mutation_rejection(self):
        if SPONGE is None:
            self.skipTest('engine path not provided')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); model = root/'model'
            water.prepare(model, cells=2, cutoff=2.)
            prep = pilot.equilibrate(model, root/'nvt', SPONGE, steps=100, dt=.00025, stride=10, grid=64)
            seed = root/'nvt/structural_seed.npz'
            result = pilot.nve(model, seed, root/'nve', SPONGE, steps=100, dt=.000125, stride=10, grid=64)
            self.assertLess(prep['maximum_length_residual'], 1e-4)
            self.assertLess(result['maximum_length_residual'], 1e-4)
            self.assertFalse(result['formal_viscosity_acceptance'])
            handoff = json.loads((root/'nve/handoff.json').read_text())
            self.assertAlmostEqual(result['initial_temperature_k'], handoff['initial_temperature_k'], delta=1e-3)
            energy_table = np.loadtxt(root/'nve/energy.csv', delimiter=',', skiprows=1)
            np.testing.assert_array_equal(energy_table[:, 0], np.arange(0, 101, 10))
            self.assertEqual(energy_table[-1, 1], .0125)
            self.assertTrue(np.all(np.isfinite(energy_table)))
            checked = audit.audit(model, root/'nve', root/'kinetic-audit')
            self.assertEqual(checked['status'], 'passed_sampled_kinetic_agreement')
            self.assertLess(checked['maximum_projection_momentum_error_native'], 1e-12)
            self.assertEqual(checked['samples'], 11)
            # Finalized HDF5 alone is insufficient: preserve the pilot/engine
            # completion and the force actually used for the half-step handoff.
            for filename, status, message in (
                    ('summary.json', 'incomplete', 'pilot did not complete'),
                    ('execution.json', 'engine_failed', 'engine did not complete')):
                path = root/'nve'/filename; original = path.read_bytes()
                changed = json.loads(original); changed['status'] = status
                water.save(path, changed)
                with self.assertRaisesRegex(ValueError, message):
                    audit.audit(model, root/'nve', root/'invalid-audit')
                self.assertFalse((root/'invalid-audit').exists())
                path.write_bytes(original)
            failure = root/'nve/failure.json'
            water.save(failure, {'error': 'postprocessing failed'})
            with self.assertRaisesRegex(ValueError, 'pilot did not complete'):
                audit.audit(model, root/'nve', root/'invalid-audit')
            failure.unlink()
            force_probe = root/'nve/handoff-force/force.dat'; original = force_probe.read_bytes()
            force_probe.write_bytes(original[:-12])
            with self.assertRaisesRegex(ValueError, 'handoff force identity mismatch'):
                audit.audit(model, root/'nve', root/'invalid-audit')
            force_probe.write_bytes(original)
            # A newly hashed but phase-inconsistent force stream must also fail.
            force_path = root/'nve/force.dat'; original = force_path.read_bytes()
            summary_path = root/'nve/summary.json'; summary_bytes = summary_path.read_bytes()
            force = np.frombuffer(original, dtype=np.float32).copy(); force[0] += .5
            force.tofile(force_path)
            changed = json.loads(summary_bytes); changed['raw_sha256']['force.dat'] = water.digest(force_path)
            water.save(summary_path, changed)
            with self.assertRaisesRegex(ValueError, 'initial force does not match handoff phase'):
                audit.audit(model, root/'nve', root/'invalid-audit')
            self.assertFalse((root/'invalid-audit').exists())
            force_path.write_bytes(original); summary_path.write_bytes(summary_bytes)
            derived = pilot.snapshot_model(model, seed, root/'snapshot')
            loaded, snapshot = water.load_model(root/'snapshot')
            self.assertEqual(derived, loaded)
            with np.load(seed) as saved:
                np.testing.assert_array_equal(snapshot['positions'], saved['positions'])
            self.assertIn('no equilibrium acceptance', loaded['initialization'])
            atom_text = (root/'snapshot/lammps.data').read_text().split('Atoms # full\n\n')[1].split('\nBonds\n')[0]
            xyz = np.array([line.split()[4:] for line in atom_text.splitlines() if line.strip()], float)
            np.testing.assert_array_equal(xyz, snapshot['positions'])
            seed.write_bytes(seed.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                pilot.nve(model, seed, root/'invalid', SPONGE, steps=100, stride=10, grid=64)
            self.assertFalse((root/'invalid').exists())
            force_path = root/'nve/force.dat'
            force_path.write_bytes(force_path.read_bytes()[:-12])
            with self.assertRaisesRegex(ValueError, 'raw identity mismatch'):
                audit.audit(model, root/'nve', root/'invalid-audit')
            self.assertFalse((root/'invalid-audit').exists())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sponge', type=Path)
    args, extra = parser.parse_known_args()
    SPONGE = args.sponge.resolve() if args.sponge else None
    unittest.main(argv=[sys.argv[0], *extra])

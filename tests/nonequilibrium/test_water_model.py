"""Frozen molecular input identity and independent mechanical references."""
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
from water_ewald_check import explicit_ewald
import water_constrained_pair as pair


class WaterModel(unittest.TestCase):
    def test_periodic_rigid_images_preserve_rotation_and_continuous_stress(self):
        box = 10.
        x = np.array([[9.8, 9.7, 5.], [10.8, 9.7, 5.], [9.5, 10.6, 5.]])
        mass = np.array([16., 1., 1.])
        center = np.average(x, weights=mass, axis=0)
        velocity = np.cross([.2, -.3, .4], x-center)+[.1, .2, .3]
        force = np.array([[.1, .3, -.4], [-.2, .5, .7], [.8, -.1, .2]])
        pairs = np.array([[a,b,np.linalg.norm(x[a]-x[b])] for a,b in ((0,1),(0,2),(1,2))])
        wrapped = x % box; saved = wrapped.copy()
        whole = pair.whole_water(wrapped, box)
        np.testing.assert_array_equal(wrapped, saved)
        np.testing.assert_allclose(whole, x, atol=1e-14)
        def observe(q):
            return water.constrained_reference(q, velocity, force, mass, pairs, box, 0, np.zeros((3,3)))
        a,b = observe(whole), observe(wrapped)
        np.testing.assert_allclose(a['velocity'], velocity, atol=1e-14)
        np.testing.assert_allclose(a['pressure'], b['pressure'], atol=1e-7)
        np.testing.assert_allclose(a['constraint'], b['constraint'], atol=1e-13)

    def test_paired_dump_rejects_wrong_phase_order_and_nonfinite_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'state.dump'
            frame = ('ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n3\n'
                     'ITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\n'
                     'ITEM: ATOMS id x y z vx vy vz f_raw[1] f_raw[2] f_raw[3]\n'
                     '1 1 2 3 .1 .2 .3 4 5 6\n2 2 2 3 .4 .5 .6 7 8 9\n3 1 3 3 .7 .8 .9 1 2 3\n')
            path.write_text(frame)
            q,u,f = pair.read_dump(path,3,0,10.)
            np.testing.assert_array_equal(q[0,0],[1,2,3])
            np.testing.assert_allclose(u[0,0],np.array([.1,.2,.3])*pair.LM_TIME)
            np.testing.assert_array_equal(f[0,0],[4,5,6])
            for invalid in (frame.replace('TIMESTEP\n0','TIMESTEP\n1'),
                            frame.replace('2 2 2 3','1 2 2 3'),
                            frame.replace('.4 .5 .6','nan .5 .6'),frame+frame):
                path.write_text(invalid)
                with self.assertRaises(ValueError): pair.read_dump(path,3,0,10.)

    def test_frozen_inputs_geometry_units_and_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'model'
            model = water.prepare(path, cells=2, cutoff=2.)
            loaded, state = water.load_model(path)
            self.assertEqual(model, loaded)
            self.assertEqual(model['thermal_dof'], 6*8-3)
            x, pairs = state['positions'], state['pairs']
            for a, b, length in pairs:
                d = x[int(a)]-x[int(b)]
                d -= model['box_angstrom']*np.floor(d/model['box_angstrom']+.5)
                self.assertAlmostEqual(np.linalg.norm(d), length, delta=1e-6)
            q = state['native_charges']
            self.assertEqual(q.sum(), 0.)
            np.testing.assert_allclose(np.array(model['mapped_lammps_charge_e'])**2*water.LM_COULOMB,
                                       q[:3]**2, rtol=1e-15)
            # Read the file as the native loader does: input float32, then 12/6.
            lines = (path/'lj.txt').read_text().splitlines()
            actual_a = float(np.float32(lines[1])*np.float32(12))/12
            actual_b = float(np.float32(lines[4])*np.float32(6))/6
            self.assertEqual(actual_a, model['effective_c12'])
            self.assertEqual(actual_b, model['effective_c6'])
            eps, sig = model['lammps_epsilon'], model['lammps_sigma']
            self.assertAlmostEqual(4*eps*sig**12/actual_a, 1., places=14)
            self.assertAlmostEqual(4*eps*sig**6/actual_b, 1., places=14)
            exclusion = (path/'exclude.txt').read_text().splitlines()
            self.assertEqual(exclusion[0], '24 24')
            self.assertEqual(exclusion[1:4], ['2 1 2', '1 2', '0'])
            observed = water.constrained_reference(x, state['half_velocity'], np.zeros_like(x),
                        state['masses'], pairs, model['box_angstrom'], 0, np.zeros((3, 3)))
            self.assertAlmostEqual(observed['temperature'], 298.15, delta=2e-5)
            (path/'charge.txt').write_text('changed\n')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                water.load_model(path)
            model['input_sha256'].pop('charge.txt')
            water.save(path/'model.json', model)
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                water.load_model(path)

    def test_analytic_rigid_triangle_force_and_rotation_cancel(self):
        x = np.array([[1., 0, 0], [-.5, np.sqrt(3)/2, 0], [-.5, -np.sqrt(3)/2, 0]])
        mass = np.ones(3)
        velocity = np.cross([0, 0, 2.], x)+np.array([.3, -.2, .1])
        force = -7*x
        pairs = np.array([[0, 1, np.sqrt(3)], [0, 2, np.sqrt(3)], [1, 2, np.sqrt(3)]])
        reference = water.constrained_reference(x, velocity, force, mass, pairs, 10., 0, x.T@force)
        np.testing.assert_allclose(reference['pressure'], 0., atol=1e-7)
        np.testing.assert_allclose(reference['velocity'], velocity, atol=1e-14)
        np.testing.assert_allclose(reference['constraint'], np.diag([4.5, 4.5, 0]), atol=1e-14)
        bad = pairs.copy(); bad[0, 0] = .5
        with self.assertRaises(ValueError):
            water.constrained_reference(x, velocity, force, mass, bad, 10., 0, x.T@force)

    def test_ewald_beta_invariance_and_energy_strain_derivative(self):
        x = np.array([[1., 1., 1.], [2.1, 1.6, 1.3]])
        q, box = np.array([1., -1.]), np.array([7., 8., 9.])
        def evaluate(positions=x, lattice=box, beta=.65, kmax=12, images=1):
            return explicit_ewald(positions, q, [], lattice, 3., beta, kmax, images)
        base, other = evaluate(), evaluate(beta=.8, kmax=16, images=2)
        self.assertAlmostEqual(sum(base['energy'].values()), sum(other['energy'].values()), delta=1e-12)
        np.testing.assert_allclose(sum(base['virial'].values()), sum(other['virial'].values()), atol=1e-11)
        for axis in range(3):
            h = 1e-5
            scale = np.ones(3); scale[axis] += h
            plus = sum(evaluate(x*scale, box*scale)['energy'].values())
            scale[axis] -= 2*h
            minus = sum(evaluate(x*scale, box*scale)['energy'].values())
            self.assertAlmostEqual(-(plus-minus)/(2*h), sum(base['virial'].values())[axis, axis], delta=1e-9)

    def test_inputs_expose_precision_and_exclude_reference_fix_virial(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); model = water.prepare(root/'model', cells=2, cutoff=2.)
            sp = root/'sp'; sp.mkdir(); lm = root/'lm'; lm.mkdir()
            water.sponge_input(sp, root/'model', model, algorithm='SHAKE')
            water.lammps_input(lm, root/'model', model, 'full')
            self.assertIn('PM.Direct_Tolerance=1e-10', (sp/'mdin.spg.toml').read_text())
            text = (lm/'in.lammps').read_text()
            self.assertIn('pair_modify table 0', text)
            self.assertIn('pressure NULL pair kspace', text)
            self.assertIn('pair_modify shift no tail no', text)
            self.assertNotIn('fix ', text)


if __name__ == '__main__':
    unittest.main()

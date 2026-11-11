"""Independent numerical checks for FFT normalization, SI integration and blocks."""
from pathlib import Path
import sys
import tempfile
import unittest

try:
    import numpy as np
    import h5py
except ImportError:
    raise SystemExit(77)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"scripts"))
import analyze_green_kubo as gk


class Analysis(unittest.TestCase):
    def write_contract_fixture(self, path, constrained=True):
        with h5py.File(path, "w") as file:
            provenance = file.require_group("parameters/sponge/provenance")
            metadata = dict(viscosity_contract=gk.CONSTRAINED_CONTRACT if constrained else gk.CONTRACT,
                            sample_interval="1", dt_ps="0.1", atom_count="3")
            if constrained:
                metadata.update(constraint_observer="instantaneous_tangent_v1", constraint_count="3",
                                thermal_dof="3", constraint_solver="SETTLE|settle_pairs=0|settle_triangles=1|shake_pairs=0",
                                constraint_length_tolerance="0.0001")
            for key, value in metadata.items():
                provenance.create_dataset(key, data=value)
            group = file.require_group("observables/all")
            group.create_dataset("step", data=np.arange(4))
            group.create_dataset("time", data=np.arange(4)*.1).attrs["unit"] = "ps"
            units = {**{x: "Pa" for x in gk.NAMES}, "temperature": "K", "volume": "nm^3"}
            if constrained:
                units.update(gk.CONSTRAINT_UNITS)
            for name, unit in units.items():
                value = 0. if name.endswith("residual") else 1.
                group.create_dataset(f"{name}/value", data=np.full(4, value)).attrs["unit"] = unit
            journal = file.require_group("parameters/sponge/output")
            for key, value in dict(status="finalized", frame_count=[4], last_complete_step=[3], last_complete_time=[.3+.0]).items():
                journal.create_dataset(key, data=value)
            journal["last_complete_time"][0] = group["time"][-1]

    def test_contract_compatibility_and_constraint_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"stress.h5"
            for constrained in (False, True):
                self.write_contract_fixture(path, constrained)
                loaded = gk.load_stress(path)
                self.assertEqual(loaded[1].shape, (4, 3))
            bad_metadata = [("constraint_observer", "unknown", "observer"),
                            ("thermal_dof", "6", "thermal DOF"),
                            ("constraint_count", "0", "thermal DOF"),
                            ("constraint_solver", "LINCS", "solver"),
                            ("constraint_solver", "SETTLE|garbage", "solver"),
                            ("constraint_solver", "SETTLE|settle_pairs=0|settle_triangles=0|shake_pairs=0", "solver"),
                            ("constraint_length_tolerance", "nan", "tolerance")]
            for key, value, message in bad_metadata:
                self.write_contract_fixture(path)
                with h5py.File(path, "r+") as file:
                    file[f"parameters/sponge/provenance/{key}"][()] = value
                with self.assertRaisesRegex(ValueError, message):
                    gk.load_stress(path)
            for key in ("constraint_count", "constraint_solver", "constraint_observer"):
                self.write_contract_fixture(path)
                with h5py.File(path, "r+") as file:
                    del file[f"parameters/sponge/provenance/{key}"]
                with self.assertRaisesRegex(ValueError, "missing constrained"):
                    gk.load_stress(path)
            for name, value, message in (("constraint_length_residual", .001, "exceeds"),
                                          ("constraint_tangent_residual", -1, "negative"),
                                          ("constraint_wxy", float("nan"), "diagnostic")):
                self.write_contract_fixture(path)
                with h5py.File(path, "r+") as file:
                    file[f"observables/all/{name}/value"][0] = value
                with self.assertRaisesRegex(ValueError, message):
                    gk.load_stress(path)
            self.write_contract_fixture(path)
            with h5py.File(path, "r+") as file:
                file["observables/all/constraint_wxy/value"].attrs["unit"] = "Pa"
            with self.assertRaisesRegex(ValueError, "unit"):
                gk.load_stress(path)
            self.write_contract_fixture(path)
            with h5py.File(path, "r+") as file:
                del file["observables/all/constraint_wxy"]
            with self.assertRaisesRegex(ValueError, "missing constraint diagnostic"):
                gk.load_stress(path)

    def test_legacy_solver_precision_metadata(self):
        solver = "SETTLE|settle_pairs=0|settle_triangles=1|shake_pairs=0"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"stress.h5"
            for suffix, invalid in (("", False), ("|precision=double", False),
                                    ("|precision=single", False), ("|precision=float", True),
                                    ("|precision=double|precision=single", True),
                                    ("|precision=double|unknown=1", True)):
                with self.subTest(suffix=suffix):
                    self.write_contract_fixture(path)
                    with h5py.File(path, "r+") as file:
                        file["parameters/sponge/provenance/constraint_solver"][()] = solver+suffix
                    if invalid:
                        with self.assertRaisesRegex(ValueError, "solver metadata"):
                            gk.load_stress(path)
                    else:
                        gk.load_stress(path)

    def test_rattle_state_metadata(self):
        solver = 'RATTLE|rattle_pairs=3|precision=canonical_double_native_force_v1|step=BA_full_drift'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'stress.h5'
            cases = [(solver, '2e-12', None),
                     (solver.replace('pairs=3', 'pairs=2'), '2e-12', 'solver'),
                     (solver.replace('canonical_double_native_force_v1', 'float'), '2e-12', 'solver'),
                     (solver.replace('BA_full_drift', 'BAOAB'), '2e-12', 'solver'),
                     (solver+'|extra=1', '2e-12', 'solver'),
                     (solver+'|rattle_pairs=3', '2e-12', 'solver'),
                     (solver.split('|step=')[0], '2e-12', 'solver'),
                     (solver, '0.0001', 'tolerance'), (solver, 'nan', 'tolerance')]
            for candidate, tolerance, error in cases:
                with self.subTest(solver=candidate, tolerance=tolerance):
                    self.write_contract_fixture(path)
                    with h5py.File(path, 'r+') as file:
                        metadata = file['parameters/sponge/provenance']
                        metadata['constraint_solver'][()] = candidate
                        metadata['constraint_length_tolerance'][()] = tolerance
                    if error:
                        with self.assertRaisesRegex(ValueError, error):
                            gk.load_stress(path)
                    else:
                        gk.load_stress(path)

    def test_fft_matches_direct_origins_with_and_without_demeaning(self):
        data = np.random.default_rng(9).normal(size=(37, 3)) + np.array([1, 0, -2])
        for demean in (False, True):
            actual = gk.autocorrelation(data, 15, demean)
            x = data-data.mean(axis=0) if demean else data
            expected = np.array([(x[:len(x)-lag]*x[lag:]).mean(axis=0) for lag in range(16)])
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_isotropic_tensor_normalization_and_rotation(self):
        # Orthonormal basis of symmetric traceless tensors gives an exact
        # isotropic second moment, without a random-sampling tolerance.
        basis = np.zeros((5, 3, 3))
        basis[0] = np.diag([1, -1, 0])/np.sqrt(2)
        basis[1] = np.diag([1, 1, -2])/np.sqrt(6)
        for index, (a, b) in enumerate(((0, 1), (0, 2), (1, 2)), 2):
            basis[index, a, b] = basis[index, b, a] = 1/np.sqrt(2)
        rotation, _ = np.linalg.qr(np.random.default_rng(19).normal(size=(3, 3)))
        for q in (np.eye(3), rotation):
            tensors = q @ np.concatenate((basis, -basis)) @ q.T
            # Hydrostatic pressure must cancel even if it is large.
            tensors += 17*np.eye(3)
            values = gk.stress_components(tensors[:, (0, 1, 2, 0, 0, 1), (0, 1, 2, 1, 2, 2)])
            np.testing.assert_allclose((values**2).mean(axis=0), np.full(5, .1), atol=1e-14)
            self.assertAlmostEqual(np.mean(values[:, 3]*values[:, 4]), -.05)

    def test_five_component_loading_and_units(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'stress.h5'
            self.write_contract_fixture(path)
            with self.assertRaises(KeyError):
                gk.load_stress(path, 'isotropic5')
            with h5py.File(path, 'r+') as file:
                for name, value in [('pxx', 9.), ('pyy', 5.), ('pzz', -1.)]:
                    file.create_dataset(f'observables/all/{name}/value', data=np.full(4, value)).attrs['unit'] = 'Pa'
            loaded = gk.load_stress(path, 'isotropic5')
            np.testing.assert_array_equal(loaded[1], [[1, 1, 1, 2, 3]]*4)
            self.assertEqual(gk.load_stress(path)[1].shape, (4, 3))
            with h5py.File(path, 'r+') as file:
                file['observables/all/pxx/value'].attrs['unit'] = 'bar'
            with self.assertRaisesRegex(ValueError, 'unit'):
                gk.load_stress(path, 'isotropic5')

    def test_five_component_blocks_match_direct_integrals(self):
        data = np.random.default_rng(42).normal(size=(120, 5))
        data[:, 4] = 2*data[:, 3]  # Deliberately correlated, unequal variances.
        report, _, _, curves = gk.analyze(np.arange(120)*.1, data, np.full(120, 300.),
            np.full(120, 100.), 0, .3, 3., bootstrap=200, components='isotropic5')
        direct = []
        for block in data.reshape(4, 30, 5):
            corr = np.array([(block[:30-lag]*block[lag:]).mean(axis=0) for lag in range(4)])
            direct.append((.5*corr[0]+corr[1]+corr[2]+.5*corr[3])*.1e-12*100e-27/(gk.KB*300))
        direct = np.array(direct)
        np.testing.assert_allclose(curves[-1], direct.mean(axis=0), atol=1e-30)
        np.testing.assert_allclose(report['block_viscosity_pa_s'], direct.mean(axis=1), atol=1e-30)
        self.assertAlmostEqual(report['block_standard_error_pa_s']/direct.mean(axis=1).std(ddof=1), .5)

    def test_trapezoid_and_si_units(self):
        correlation = np.array([[4., 8., 12.], [2., 4., 6.], [0., 0., 0.]])*1e12
        actual = gk.integrate(correlation, .5, 100, 300)
        expected = np.array([[0., 0., 0.], [1.5, 3., 4.5], [2., 4., 6.]])*1e-25/(1.380649e-23*300)
        np.testing.assert_allclose(actual, expected, rtol=1e-14)

    def test_component_covariance_and_explicit_cutoff(self):
        # All components identical: treating them as three replicates would
        # spuriously reduce uncertainty. Three blocks have known integrals.
        data = np.repeat(np.repeat([1., 2., 3.], 5)[:, None], 3, axis=1)
        times = np.arange(16)*.5
        data = np.vstack((data, [999, 999, 999]))
        report, lags, corr, curves = gk.analyze(times, data, np.full(16, 300.), np.full(16, 100.),
                                             0, .5, 2.5, bootstrap=200, seed=5)
        factor = .5e-12*100e-27/(gk.KB*300)
        np.testing.assert_allclose(report["block_viscosity_pa_s"], np.array([1, 4, 9])*factor)
        self.assertAlmostEqual(report["block_standard_error_pa_s"]/factor, np.std([1, 4, 9], ddof=1)/np.sqrt(3))
        self.assertEqual(report["dropped_tail_frames"], 1)
        self.assertEqual(report["used_frames"], 15)
        np.testing.assert_allclose(corr, np.full((2, 3), 14/3))
        self.assertEqual(report["status"], "diagnostic_only")

    def test_insufficient_data_and_bad_windows_rejected(self):
        args = (np.arange(10.), np.ones((10, 3)), np.full(10, 300.), np.full(10, 10.))
        for window, block in ((0, 5), (3, 5), (1, 20), (float("nan"), 5)):
            with self.assertRaises(ValueError):
                gk.analyze(*args, 0, window, block)

    def test_direct_api_rejects_invalid_samples(self):
        data = np.ones((10, 3))
        temperature, volume = np.full(10, 300.), np.full(10, 10.)
        for times in (np.array([0.]), np.arange(10.)[::-1], np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 10])):
            with self.assertRaises(ValueError):
                gk.analyze(times, data, temperature, volume, 0, 1, 5)
        with self.assertRaises(ValueError):
            gk.text(["valid", "ignored"])


if __name__ == "__main__":
    unittest.main()

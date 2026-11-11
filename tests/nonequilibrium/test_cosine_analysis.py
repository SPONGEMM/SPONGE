"""Synthetic response/statistics fixtures, not measured liquid viscosity."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest
try:
    import numpy as np
    import h5py
except ImportError:
    raise SystemExit(77)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"scripts"))
import analyze_cosine as cosine


def fixture(amplitudes=None):
    amplitudes = np.asarray(amplitudes if amplitudes is not None else [.25]*40)
    n, bins, length = len(amplitudes), 8, 4.
    # Finite bin width attenuates bin means. The actual basis must be used.
    basis = np.exp(2j*np.pi*(np.arange(bins)+.5)/bins)*np.sinc(1/bins)
    velocity = .17+amplitudes[:, None]*basis.real+.03*basis.imag
    data = {name: np.zeros(n) for name in cosine.UNITS}
    data.update(time=np.arange(n, dtype=float), step=np.arange(n),
                velocity_fourier=amplitudes.copy(), velocity_com_fourier=amplitudes.copy(),
                velocity_fit=amplitudes.copy(), variance_cosine=np.full(n, .5),
                density=np.full(n, .8), acceleration=np.full(n, .25*(2*np.pi/length)**2/.8),
                wave_number=np.full(n, 2*np.pi/length), length=np.full(n, length),
                thermal_temperature=np.full(n, 300.), com_vx=np.full(n, .17),
                bin_mass=np.full((n, bins), 10.), bin_count=np.ones((n, bins)),
                bin_momentum=10*velocity, bin_cosine_mass=np.tile(10*basis.real, (n, 1)),
                bin_sine_mass=np.tile(10*basis.imag, (n, 1)))
    return data


def write_fixture(path, data):
    n, bins = data["bin_mass"].shape
    with h5py.File(path, "w") as file:
        provenance = file.create_group("parameters/sponge/provenance")
        for key, value in dict(viscosity_contract=cosine.CONTRACT, dt_ps="1", sample_interval="1",
                               bins=str(bins), atom_count=str(bins), flow="0", gradient="2",
                               fixture="synthetic response; not MD").items():
            provenance[key] = value
        output = file.create_group("parameters/sponge/output")
        output["status"] = "finalized"
        output["frame_count"] = [0, n]
        output["last_complete_step"] = [-1, n-1]
        output["last_complete_time"] = [0., float(n-1)]
        group = file.create_group("observables/all")
        group["time"] = data["time"]; group["time"].attrs["unit"] = "ps"
        group["step"] = data["step"]
        for name, unit in cosine.UNITS.items():
            group[f"{name}/value"] = data[name]
            group[f"{name}/value"].attrs["unit"] = unit
        for name, unit in dict(mass="Da", momentum="Da nm/ps", count="1", cosine_mass="Da", sine_mass="Da").items():
            for i in range(bins):
                group[f"bin_{i}_{name}/value"] = data[f"bin_{name}"][:, i]
                group[f"bin_{i}_{name}/value"].attrs["unit"] = unit


class Analysis(unittest.TestCase):
    def test_units_and_finite_bin_basis(self):
        report, profile = cosine.analyze(fixture(), 0, 4, "fourier", bootstrap=200)
        self.assertAlmostEqual(report["viscosity_mpa_s"], 1.)
        np.testing.assert_allclose(report["conditional_bootstrap_95_interval_mpa_s"], [1., 1.])
        fit = report["spatial_fit"]
        self.assertAlmostEqual(fit["cosine_nm_per_ps"], .25)
        self.assertAlmostEqual(fit["sine_nm_per_ps"], .03)
        self.assertAlmostEqual(fit["offset_nm_per_ps"], .17)
        self.assertLess(fit["residual_rms_nm_per_ps"], 1e-15)
        self.assertEqual(len(profile), 8)
        self.assertEqual(report["blocks"], 10)
        self.assertAlmostEqual(report["mean_response_per_acceleration_ps"], .8/(2*np.pi/4)**2)
        self.assertAlmostEqual(report["cosine_mode_peak_shear_rate_ps_inv"], .25*2*np.pi/4)
        self.assertEqual(report["temperature_block_standard_error_k"],0)
        self.assertEqual(report["sample_weight_duration_ps"],40)
        self.assertEqual(report["ledger_endpoint_duration_ps"],39)
        self.assertEqual(report["block_records"][0]["thermal_temperature_k"],300)

    def test_ratio_of_means_not_mean_reciprocal(self):
        amplitudes = np.tile([.1, .4], 20)
        data = fixture(amplitudes)
        report, _ = cosine.analyze(data, 0, 4, "fit", bootstrap=100)
        self.assertAlmostEqual(report["viscosity_mpa_s"], 1.)
        self.assertNotAlmostEqual(report["viscosity_mpa_s"], np.mean(.25/amplitudes))
        for name in ("velocity_fourier", "velocity_com_fourier", "velocity_fit", "acceleration"):
            data[name] *= -1
        reverse, _ = cosine.analyze(data, 0, 4, "com_fourier", bootstrap=100)
        self.assertAlmostEqual(reverse["viscosity_mpa_s"], 1.)
        self.assertAlmostEqual(reverse["raw_mean_amplitude_nm_per_ps"], -.25)
        self.assertAlmostEqual(reverse["signed_mean_amplitude_nm_per_ps"], .25)

    def test_zero_crossings_are_not_discarded(self):
        data = fixture(np.repeat([-.2, .3], 20))
        report, _ = cosine.analyze(data, 0, 20, "fit", bootstrap=500)
        self.assertEqual(report["status"], "unresolved_response")
        self.assertIsNone(report["conditional_bootstrap_95_interval_mpa_s"])
        self.assertGreater(report["nonpositive_bootstrap_draws"], 0)
        self.assertAlmostEqual(report["viscosity_mpa_s"], 5.)
        report, _ = cosine.analyze(fixture([-.2]*40), 0, 4, "fit", bootstrap=100)
        self.assertIsNone(report["viscosity_mpa_s"])

    def test_window_tail_and_joint_blocks(self):
        data = fixture(np.repeat([.1, .2, .3, .4], 10))
        report, _ = cosine.analyze(data, 10, 10, "fit", bootstrap=200)
        self.assertAlmostEqual(report["viscosity_mpa_s"], .25/.3)
        self.assertEqual(report["blocks"], 3)
        self.assertEqual(report["last_time_ps"], 39.)
        np.testing.assert_allclose([r["amplitude_nm_per_ps"] for r in report["block_records"]], [.2, .3, .4])
        self.assertAlmostEqual(report["block_standard_error_nm_per_ps"], .1/np.sqrt(3))
        report, _ = cosine.analyze(data, 11, 10, "fit", bootstrap=100)
        self.assertEqual(report["dropped_tail_frames"], 9)

    def test_invalid_direct_inputs(self):
        for key, value in (("density", np.zeros(40)), ("acceleration", np.zeros(40)),
                           ("length", np.arange(40)), ("thermal_temperature", np.full(40, -1)),
                           ("velocity_fit", np.full(40, np.nan)), ("time", np.zeros(40))):
            broken = fixture(); broken[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                cosine.analyze(broken, 0, 4, "fit")
        with self.assertRaises(ValueError): cosine.analyze(fixture(), 0, 4, "guess")
        with self.assertRaises(ValueError): cosine.analyze(fixture(), 0, 100, "fit")

    def test_optional_heat_history_is_never_assumed_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"heat.h5"
            write_fixture(path, fixture())
            data, _ = cosine.load_profile(path)
            report, _ = cosine.analyze(data, 0, 4, "fit", bootstrap=100)
            self.assertFalse(report["heat_available"])
            self.assertIsNone(report["heat_in_window_kcal_per_mol"])
            with h5py.File(path,"r+") as file:
                file["parameters/sponge/provenance/thermostat_heat_method"] = cosine.HEAT_METHOD
            with self.assertRaisesRegex(ValueError,"missing"): cosine.load_profile(path)
            with h5py.File(path,"r+") as file:
                node = file.create_dataset("observables/all/thermostat_heat/value", data=-np.arange(40.))
                node.attrs["unit"] = "kcal/mol"
            data, _ = cosine.load_profile(path)
            report, _ = cosine.analyze(data, 0, 4, "fit", bootstrap=100)
            self.assertEqual(report["heat_in_window_kcal_per_mol"], -39.)
            self.assertEqual(report["work_plus_heat_in_window_kcal_per_mol"], -39.)
            with h5py.File(path,"r+") as file:
                file["observables/all/thermostat_heat/value"].attrs["unit"] = "wrong"
            with self.assertRaisesRegex(ValueError,"units"): cosine.load_profile(path)
            with h5py.File(path,"r+") as file:
                file["observables/all/thermostat_heat/value"].attrs["unit"] = "kcal/mol"
                file["parameters/sponge/provenance/thermostat_heat_method"][()] = "none"
            with self.assertRaisesRegex(ValueError,"undeclared"): cosine.load_profile(path)
            data["thermostat_heat"][0] = np.nan
            with self.assertRaisesRegex(ValueError,"heat data"): cosine.analyze(data,0,4,"fit")

    def test_constrained_contract_and_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"constrained.h5"
            write_fixture(path, fixture())
            with h5py.File(path, "r+") as file:
                meta = file["parameters/sponge/provenance"]
                meta["viscosity_contract"][()] = cosine.CONSTRAINED_CONTRACT
            with self.assertRaisesRegex(ValueError, "provenance missing"):
                cosine.load_profile(path)
            with h5py.File(path, "r+") as file:
                meta = file["parameters/sponge/provenance"]
                for key in ("constraint_configuration", "sampling_convention", "stress_convention",
                            "work_convention", "state_precision", "thermostat_heat_convention"):
                    meta[key] = "synthetic schema fixture, not trajectory evidence"
                meta["thermostat_heat_method"] = cosine.CONSTRAINED_HEAT_METHOD
                for name, unit in {**cosine.CONSTRAINED_UNITS, "thermostat_heat": "kcal/mol"}.items():
                    node = file.create_dataset(f"observables/all/{name}/value", data=np.zeros(40))
                    node.attrs["unit"] = unit
                file["observables/all/constraint_drift_energy/value"][...] = np.arange(40.)*.01
                file["observables/all/thermostat_heat/value"][...] = -np.arange(40.)
            data, meta = cosine.load_profile(path)
            report, _ = cosine.analyze(data, 0, 4, "fit", bootstrap=100)
            self.assertEqual(meta["viscosity_contract"], cosine.CONSTRAINED_CONTRACT)
            self.assertAlmostEqual(report["viscosity_mpa_s"], 1.)
            self.assertAlmostEqual(report["constrained_diagnostics"]["drift_kinetic_change_in_window_kcal_per_mol"], .39)
            self.assertEqual(report["heat_in_window_kcal_per_mol"], -39.)
            with h5py.File(path, "r+") as file:
                file["parameters/sponge/provenance/thermostat_heat_method"][()] = cosine.HEAT_METHOD
            with self.assertRaisesRegex(ValueError, "joint tangent OU"):
                cosine.load_profile(path)
            with h5py.File(path, "r+") as file:
                file["parameters/sponge/provenance/thermostat_heat_method"][()] = cosine.CONSTRAINED_HEAT_METHOD
                file["observables/all/pxy/value"].attrs["unit"] = "wrong"
            with self.assertRaisesRegex(ValueError, "units"):
                cosine.load_profile(path)
            with h5py.File(path, "r+") as file:
                file["parameters/sponge/provenance/viscosity_contract"][()] = "unknown"
            with self.assertRaisesRegex(ValueError, "unsupported cosine contract"):
                cosine.load_profile(path)

    def test_reader_cli_and_contract_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = root/"synthetic.h5"
            write_fixture(path, fixture())
            data, meta = cosine.load_profile(path)
            self.assertIn("not MD", meta["fixture"])
            run = subprocess.run([sys.executable, cosine.__file__, str(path), "--start-ps", "0",
                                  "--block-ps", "4", "--estimator", "fit", "--output", str(root/"analysis")],
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            report = json.loads((root/"analysis/summary.json").read_text())
            self.assertAlmostEqual(report["viscosity_mpa_s"], 1.)
            for key, value in (("parameters/sponge/output/frame_count", [0, 39]),
                               ("parameters/sponge/output/status", "failed"),
                               ("observables/all/velocity_fit/value", np.ones(40)),
                               ("observables/all/bin_0_momentum/value", np.ones(40)),
                               ("observables/all/bin_0_count/value", np.full(40, .5))):
                write_fixture(path, fixture())
                with h5py.File(path, "r+") as file: file[key][...] = value
                with self.subTest(key=key), self.assertRaises(ValueError): cosine.load_profile(path)
            broken = fixture()
            broken["bin_cosine_mass"][:, 0] = 10
            broken["bin_sine_mass"][:, 0] = 10
            write_fixture(path, broken)
            with self.assertRaisesRegex(ValueError, "unit disk"): cosine.load_profile(path)
            write_fixture(path, fixture())
            with h5py.File(path, "r+") as file:
                file["observables/all/density/value"].attrs["unit"] = "wrong"
            with self.assertRaises(ValueError): cosine.load_profile(path)


if __name__ == "__main__":
    unittest.main()

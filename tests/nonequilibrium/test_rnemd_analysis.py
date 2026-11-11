"""Synthetic triangular-flow tests; these fixtures are not MD trajectories."""
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
import analyze_rnemd as rn


def fixture():
    n, bins = 13, 12
    times = np.arange(n, dtype=float)
    z = (np.arange(bins)+.5)*4/bins
    velocity = .1*np.where(z < 2, z-1, 3-z) + .07
    data = dict(time=times, bin_mass=np.full((n, bins), 10.), bin_momentum=np.tile(velocity*10, (n, 1)), bin_count=np.ones((n, bins)), area=np.full(n, 16.), length=np.full(n, 4.))
    # eta=1 mPa s, du/dz=1e11 s^-1 => momentum flux=1e8 Pa.
    # 1 Da nm/ps = 1.66053906660e-24 kg m/s; 1 nm^2 ps = 1e-30 m^2 s.
    data["momentum"] = times*1e8*(2*16*1e-30)/1.66053906660e-24
    data.update({key: np.zeros(n) for key in ("momentum_error", "energy_error", "empty", "no_candidate")})
    data.update(attempts=times.copy(), swaps=times.copy(), thermal_temperature=np.full(n, 300.), kinetic_temperature=np.full(n, 301.))
    return data


def write_fixture(path, data):
    with h5py.File(path, "w") as file:
        provenance = file.create_group("parameters/sponge/provenance")
        for key, value in dict(viscosity_contract=rn.CONTRACT, dt_ps="1", sample_interval="1", bins="12", atom_count="12", fixture="synthetic triangular profile; not MD").items():
            provenance[key] = value
        output = file.create_group("parameters/sponge/output")
        output["status"] = "finalized"
        output["frame_count"] = [0, 13]
        output["last_complete_step"] = [-1, 12]
        output["last_complete_time"] = [0., 12.]
        group = file.create_group("observables/all")
        group["time"] = data["time"]; group["time"].attrs["unit"] = "ps"
        group["step"] = np.arange(13)
        units = dict(momentum="Da nm/ps", momentum_error="Da nm/ps", energy_error="kcal/mol", area="nm^2", length="nm", thermal_temperature="K", kinetic_temperature="K", attempts="1", swaps="1", empty="1", no_candidate="1")
        for name, unit in units.items():
            group[f"{name}/value"] = data[name]
            group[f"{name}/value"].attrs["unit"] = unit
        for name, unit in dict(mass="Da", momentum="Da nm/ps", count="1").items():
            for i in range(12):
                group[f"bin_{i}_{name}/value"] = data[f"bin_{name}"][:, i]
                group[f"bin_{i}_{name}/value"].attrs["unit"] = unit


class Analysis(unittest.TestCase):
    def test_si_two_paths_and_branches(self):
        report, profiles = rn.analyze(fixture(), 0, 4, bootstrap=200)
        self.assertEqual(report["blocks"], 3)
        np.testing.assert_allclose(report["branch_viscosity_pa_s"], [.001, .001], rtol=1e-12)
        self.assertAlmostEqual(report["viscosity_pa_s"], .001)
        self.assertEqual(len(profiles), 36)
        for block in report["block_records"]:
            np.testing.assert_allclose([fit["slope_per_ps"] for fit in block["fits"]], [.1, -.1], rtol=1e-12)
            self.assertAlmostEqual(block["flux_pa"], 1e8)

    def test_covariance_and_matching_endpoints(self):
        data = fixture()
        increments = np.repeat([1., 2., 3.], 4)*data["momentum"][1]
        data["momentum"] = np.r_[0, np.cumsum(increments)]
        report, _ = rn.analyze(data, 0, 4, bootstrap=200)
        estimates = np.array([.001, .002, .003])
        self.assertAlmostEqual(report["viscosity_pa_s"], .002)
        self.assertNotIn("block_standard_error_pa_s", report)
        np.testing.assert_allclose([row["flux_pa"] for row in report["block_records"]], [1e8, 2e8, 3e8])
        np.testing.assert_allclose([row["branch_viscosity_pa_s"] for row in report["block_records"]], np.repeat(estimates[:, None], 2, axis=1))
        cut, _ = rn.analyze(data, 4, 4, bootstrap=200)
        self.assertAlmostEqual(cut["viscosity_pa_s"], .0025)
        self.assertEqual(cut["first_time_ps"], 4)

    def test_exclusion_and_wrong_sign_or_missing_bins(self):
        data = fixture()
        data["bin_momentum"][:, [0, 6]] = 1e6  # Exchange slab jumps must not enter fit.
        report, _ = rn.analyze(data, 0, 4, exclude_neighbors=1, bootstrap=100)
        self.assertAlmostEqual(report["viscosity_pa_s"], .001)
        for mode in ("empty", "zero_flux", "box"):
            broken = fixture()
            if mode == "empty": broken["bin_mass"][:, 2] = 0
            if mode == "zero_flux": broken["momentum"][:] = 0
            if mode == "box": broken["length"][-1] *= 2
            with self.assertRaises(ValueError): rn.analyze(broken, 0, 4)
        with self.assertRaises(ValueError): rn.analyze(data, 0, 4, exclude_neighbors=2)
        with self.assertRaises(ValueError): rn.analyze(data, 0, 20)

    def test_aggregate_ratio_not_mean_of_block_inverses(self):
        data = fixture()
        factor = .5+data["time"]/4  # exact block means 1,2,3
        data["bin_momentum"] *= factor[:, None]
        report, _ = rn.analyze(data, 0, 4, bootstrap=200)
        self.assertAlmostEqual(report["viscosity_pa_s"], .0005)
        old_mean = np.mean([row["branch_viscosity_pa_s"] for row in report["block_records"]])
        self.assertGreater(abs(old_mean-report["viscosity_pa_s"]), .0001)
        # Vary mass as well: pooling extensive quantities must precede division.
        data["bin_mass"] *= factor[:, None]
        data["bin_momentum"] *= factor[:, None]
        report, _ = rn.analyze(data, 0, 4, bootstrap=200)
        weights = np.ones(13); weights[[0, -1]] = .5
        expected = (weights@data["bin_momentum"])/(weights@data["bin_mass"])
        np.testing.assert_allclose(report["aggregate_profile"]["velocity_nm_per_ps"], expected)
        expected_factor = np.dot(weights, factor**2)/np.dot(weights, factor)
        self.assertAlmostEqual(report["viscosity_pa_s"], .001/expected_factor)

    def test_joint_resampling_retains_flux_profile_covariance(self):
        data = fixture(); factor = .5+data["time"]/4
        increment = data["momentum"][1]
        data["momentum"] = np.r_[0, np.cumsum((factor[:-1]+factor[1:])*.5*increment)]
        data["bin_momentum"] *= factor[:, None]
        report, _ = rn.analyze(data, 0, 4, bootstrap=200)
        np.testing.assert_allclose(report["conditional_bootstrap_95_interval_pa_s"], [.001, .001], rtol=1e-12)

    def test_signed_and_undefined_draws_are_not_filtered(self):
        data = fixture(); data["bin_momentum"] *= -1
        report, _ = rn.analyze(data, 0, 4, bootstrap=200)
        self.assertEqual(report["status"], "unresolved_response")
        self.assertIsNone(report["viscosity_pa_s"])
        self.assertEqual(report["bootstrap_nonpositive_branch_draws"], [200, 200])
        data = fixture(); data["bin_momentum"] *= (-1+data["time"]/4)[:, None]
        report, _ = rn.analyze(data, 0, 4, bootstrap=1000)
        self.assertGreater(report["viscosity_pa_s"], 0)
        self.assertIsNone(report["conditional_bootstrap_95_interval_pa_s"])
        self.assertGreater(report["bootstrap_nonpositive_branch_draws"][0], 0)
        self.assertLess(report["conditional_bootstrap_95_fluidity_interval_per_pa_s"][0], 0)
        # One no-transfer block: its all-same-block bootstrap draw is undefined.
        data = fixture(); increment = data["momentum"][1]
        data["momentum"] = np.r_[0, np.cumsum(np.repeat([0., 1., 1.], 4)*increment)]
        report, _ = rn.analyze(data, 0, 4, bootstrap=1000)
        self.assertGreater(report["bootstrap_invalid_draws"], 0)
        self.assertIsNone(report["conditional_bootstrap_95_interval_pa_s"])
        self.assertIsNone(report["conditional_bootstrap_95_fluidity_interval_per_pa_s"])

    def test_direct_api_optional_column_validation(self):
        for key, value in (("attempts", np.arange(2)), ("energy_error", np.full(13, np.nan)), ("swaps", np.full(13, -1)), ("attempts", np.arange(13)+1)):
            data = fixture(); data[key] = value
            with self.assertRaises(ValueError): rn.analyze(data, 0, 4)

    def test_reader_cli_and_failed_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = root/"synthetic.h5"
            write_fixture(path, fixture())
            data, meta = rn.load_profile(path)
            self.assertIn("not MD", meta["fixture"])
            np.testing.assert_array_equal(data["bin_mass"], fixture()["bin_mass"])
            run = subprocess.run([sys.executable, rn.__file__, str(path), "--start-ps", "0", "--block-ps", "4", "--output", str(root/"analysis")], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            report = json.loads((root/"analysis/summary.json").read_text())
            self.assertAlmostEqual(report["viscosity_pa_s"], .001)
            self.assertEqual(report["status"], "diagnostic_only")
            for key, value in (("parameters/sponge/output/frame_count", [0, 12]), ("parameters/sponge/output/status", "failed")):
                write_fixture(path, fixture())
                with h5py.File(path, "r+") as file: file[key][...] = value
                with self.assertRaises(ValueError): rn.load_profile(path)
            write_fixture(path, fixture())
            with h5py.File(path, "r+") as file: file["observables/all/momentum/value"].attrs["unit"] = "wrong"
            with self.assertRaises(ValueError): rn.load_profile(path)


if __name__ == "__main__":
    unittest.main()

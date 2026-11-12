"""New generic streams and frozen legacy streams share numerical analysis."""

from pathlib import Path
import tempfile
import unittest
import h5py
import numpy as np
import test_cosine_analysis as cos_fixture
import test_green_kubo_analysis as gk_fixture
import test_rnemd_analysis as rn_fixture
from tools.viscosity.analysis import cosine, green_kubo, rnemd


class GenericAnalysis(unittest.TestCase):
    def test_rnemd_requires_centered_pre_exchange_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.h5"
            rn_fixture.write_fixture(path, rn_fixture.fixture())
            with h5py.File(path, "r+") as file:
                p = file["parameters/sponge/provenance"]
                del p["viscosity_contract"]
                p["observable_contract"] = "sponge.observables.physical_state.v1"
                p["profile_kind"] = "slab"
                p["exchange_interval"] = "1"
                p["sampling_phase"] = "post_force_pre_exchange_pre_integrate"
                p["velocity_time_layer"] = "force_centered"
            data, _ = rnemd.load_profile(path)
            report, _ = rnemd.analyze(data, 0, 4, bootstrap=200)
            self.assertAlmostEqual(report["viscosity_pa_s"], .001)
            for key, invalid in [("sampling_phase", "post_exchange"),
                                 ("velocity_time_layer", "stored")]:
                with h5py.File(path, "r+") as file:
                    p = file["parameters/sponge/provenance"]
                    original = p[key][()]
                    p[key][()] = invalid
                with self.assertRaisesRegex(ValueError, "pre-exchange"):
                    rnemd.load_profile(path)
                with h5py.File(path, "r+") as file:
                    file["parameters/sponge/provenance"][key][()] = original

    def test_cosine_uses_declared_protocol_not_a_measured_acceleration_column(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.h5"
            data = cos_fixture.fixture()
            cos_fixture.write_fixture(path, data)
            acceleration = float(data["acceleration"][0])
            with h5py.File(path, "r+") as file:
                p = file["parameters/sponge/provenance"]
                del p["viscosity_contract"]
                p["observable_contract"] = (
                    "sponge.observables.physical_state.v1"
                )
                p["velocity_time_layer"] = "force_centered"
                p["profile_kind"] = "cosine"
                p["effective_commands"] = (
                    'external_force=[{type="acceleration",acceleration=['
                    + str(acceleration)
                    + ',0,0],profile={type="cosine",axis="z",mode=1}}]\n'
                )
                del file["observables/all/acceleration"]
            loaded, meta = cosine.load_profile(path)
            np.testing.assert_array_equal(
                loaded["acceleration"], data["acceleration"]
            )
            self.assertNotIn("viscosity_contract", meta)
            with h5py.File(path, "r+") as file:
                file["parameters/sponge/provenance/effective_commands"][()] = (
                    "external_force=[]"
                )
            with self.assertRaisesRegex(ValueError, "declared cosine"):
                cosine.load_profile(path)

    def test_gk_rejects_driven_or_non_equilibrium_segments(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.h5"
            gk_fixture.Analysis().write_contract_fixture(path, False)
            with h5py.File(path, "r+") as file:
                p = file["parameters/sponge/provenance"]
                del p["viscosity_contract"]
                p["observable_contract"] = (
                    "sponge.observables.physical_state.v1"
                )
                p["velocity_time_layer"] = "force_centered"
                p["effective_commands"] = "mode=nve\n"
            self.assertEqual(green_kubo.load_stress(path)[1].shape, (4, 3))
            for commands in ("mode=nvt\n", "mode=nve\nexternal_force=[]\n"):
                with h5py.File(path, "r+") as file:
                    file["parameters/sponge/provenance/effective_commands"][
                        ()
                    ] = commands
                with self.assertRaisesRegex(ValueError, "NVE|driven"):
                    green_kubo.load_stress(path)


if __name__ == "__main__":
    unittest.main()

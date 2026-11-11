"""Analytic exchange checks against the real executable, not liquid viscosity."""
import argparse
from pathlib import Path
import unittest
import test_green_kubo_runtime as base
np = base.np


class RnemdRuntime(unittest.TestCase):
    run_case = base.GreenKuboRuntime.run_case

    def test_unequal_mass_conservation_profile_and_pre_exchange_time(self):
        for masses, velocities in (((2, 4), (2, -1)), ((2, 3), (3, -1))):
            out = self.run_case(masses=masses, velocities=[(v, 0, 0) for v in velocities],
                                positions=[(1, 1, 1), (2, 2, 17)],
                                method='method="rnemd"\nexchange_interval=1\nbins=8',
                                options={"step_limit": 3})
            stored = np.frombuffer(out["vel"], dtype=np.float32).reshape(-1, 2, 3)
            mass, original = np.array(masses), np.array(velocities)
            com = np.average(original, weights=mass)
            after = 2*com-original
            np.testing.assert_array_equal(stored[0, :, 0], original)
            np.testing.assert_allclose(stored[1:, :, 0], np.tile(after, (2, 1)), rtol=2e-7)
            np.testing.assert_allclose(stored[:, :, 0]@mass, original@mass, atol=5e-7)
            np.testing.assert_allclose((stored[:, :, 0]**2)@mass, (original**2)@mass, rtol=2e-7)
            actual = mass[1]*(float(np.float32(after[1]))-original[1])*base.TIME/10
            values = out["values"]
            np.testing.assert_allclose(values["momentum"], [0, 0, actual, actual], rtol=1e-12)
            np.testing.assert_array_equal(values["attempts"], [0, 0, 1, 2])
            np.testing.assert_array_equal(values["swaps"], [0, 0, 1, 1])
            np.testing.assert_array_equal(values["no_candidate"], [0, 0, 0, 1])
            np.testing.assert_array_equal(values["bin_0_count"], [1]*4)
            np.testing.assert_array_equal(values["bin_4_mass"], [mass[1]]*4)
            np.testing.assert_allclose(values["bin_0_momentum"][0], mass[0]*original[0]*base.TIME/10)
            np.testing.assert_allclose(values["thermal_temperature"], 0, atol=1e-9)

    def test_ties_use_global_id_and_periodic_bins(self):
        out = self.run_case(masses=(1, 1, 1, 1), velocities=[(2, 0, 0), (2, 0, 0), (-3, 0, 0), (-3, 0, 0)],
                            positions=[(1, 1, 33), (2, 1, 2), (3, 1, -15), (4, 1, 18)],
                            method='method="rnemd"\nexchange_interval=1\nbins=8', options={"step_limit": 2})
        vel = np.frombuffer(out["vel"], dtype=np.float32).reshape(-1, 4, 3)
        np.testing.assert_array_equal(vel[1, :, 0], [-3, 2, 2, -3])
        np.testing.assert_array_equal(out["values"]["bin_0_count"], [2]*3)

    def test_no_exchange_for_empty_slab_and_sparse_cadence(self):
        out = self.run_case(masses=(1, 1), velocities=[(2, 0, 0), (-2, 0, 0)],
                            positions=[(1, 1, 1), (2, 1, 5)],
                            method='method="rnemd"\nexchange_interval=2\nbins=8\nsample_interval=3',
                            options={"step_limit": 6})
        np.testing.assert_array_equal(out["step"], [0, 3, 6])
        np.testing.assert_array_equal(out["values"]["empty"], [0, 1, 2])
        np.testing.assert_array_equal(out["values"]["momentum"], [0, 0, 0])

    def test_invalid_configuration(self):
        for method, error in [('method="rnemd"', 'exchange_interval is required'),
                              ('method="rnemd"\nexchange_interval=1\nbins=5', 'bins must be even'),
                              ('method="rnemd"\nexchange_interval=1\nflow="z"', 'flow and gradient must differ'),
                              ('method="rnemd"\nexchange_interval=1\nacceleration=1', 'unknown RNEMD')]:
            self.run_case(method=method, error=error)
        self.run_case(method='method="rnemd"\nexchange_interval=1', options={"dt": 0}, error="positive dt")

    def test_interacting_exchange_uses_integer_time_velocity(self):
        masses = np.array([2., 3.])
        dt = float(np.float32(.001)*np.float32(base.TIME))
        out = self.run_case(masses=masses, velocities=[(2, 0, 0), (-1, 0, 0)],
                            positions=[(1, 1, 1), (4, 1, 17)],
                            method='method="rnemd"\nexchange_interval=1\nbins=8',
                            options={"dt": .001, "step_limit": 3, "bond_in_file": "bond.txt"},
                            files={"bond.txt": "1\n0 1 0.2 13\n"})
        velocities = np.frombuffer(out["vel"], dtype=np.float32).reshape(-1, 2, 3)
        forces = np.frombuffer(out["frc"], dtype=np.float32).reshape(-1, 2, 3)
        before = velocities[0, :, 0]+.5*dt*forces[1, :, 0]/masses
        after = 2*np.average(before, weights=masses)-before
        predicted = after+.5*dt*forces[1, :, 0]/masses
        np.testing.assert_allclose(velocities[1, :, 0], predicted, rtol=3e-7, atol=1e-7)
        # The unexchanged transverse components retain the ordinary kick.
        np.testing.assert_allclose(velocities[1, :, 1:], velocities[0, :, 1:]+dt*forces[1, :, 1:]/masses[:, None], rtol=3e-7, atol=1e-7)
        self.assertLess(abs(out["values"]["energy_error"][-1]), 2e-6)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sponge", required=True)
    args, extra = parser.parse_known_args()
    base.SPONGE = str(Path(args.sponge).resolve())
    unittest.main(argv=[__file__, *extra])

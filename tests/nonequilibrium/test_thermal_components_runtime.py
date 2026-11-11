"""Short substep invariants for all thermostats; no liquid viscosity claim."""

import argparse
from pathlib import Path
import unittest
import numpy as np
import test_coupling_runtime as base

THERMOSTATS = (
    "middle_langevin",
    "andersen",
    "berendsen_thermostat",
    "bussi_thermostat",
    "nose_hoover_chain",
)


class ThermalComponents(unittest.TestCase):
    run_case = base.CouplingRuntime.run_case

    def project(self, v, m, z):
        c = np.cos(2 * np.pi * np.array(z) / 32)
        basis = np.column_stack([np.ones(len(m)), c])
        coeff = np.linalg.solve((basis.T * m) @ basis, (basis.T * m) @ v[:, 0])
        com = np.average(v, axis=0, weights=m)
        return np.r_[coeff, com[1:]]

    def test_all_thermostats_retain_stream_at_their_native_substep(self):
        n = 8
        z = np.arange(n) * 4.0
        m = np.array([1, 2, 3, 5, 7, 11, 13, 17.0])
        q = np.column_stack([np.ones(n), np.full(n, 2), z])
        v = (
            np.column_stack(
                [
                    0.3
                    + 0.8 * np.cos(2 * np.pi * z / 32)
                    + 0.4 * np.sin(2 * np.pi * z / 32),
                    0.2 + 0.1 * np.sin(2 * np.pi * z / 32),
                    np.zeros(n),
                ]
            )
            .astype(np.float32)
            .astype(float)
        )
        dt = 0.00001
        h = float(np.float32(np.float32(dt) * np.float32(base.TIME)))
        for thermostat in THERMOSTATS:
            for driven in (False, True):
                with self.subTest(thermostat=thermostat, driven=driven):
                    # A nonzero laboratory force must keep accelerating the flow.
                    drive = (
                        '[[external_force]]\ntype="force"\nforce=[100,0,0]\n'
                        if driven
                        else ""
                    )
                    config = (
                        drive
                        + '[velocity_bias]\nmode="cosine"\nflow="x"\ngradient="z"\n'
                    )
                    out = self.run_case(
                        config,
                        masses=m,
                        coordinates=q,
                        velocities=v,
                        settings=dict(
                            pbc=True,
                            mode="nvt",
                            thermostat=thermostat,
                            target_temperature=50,
                            thermostat_tau=dt
                            if thermostat == "andersen"
                            else 0.001,
                            thermostat_seed=17,
                            dt=dt,
                            step_limit=1,
                        ),
                    )
                    final = np.array(out["vel"][-1]).reshape(n, 3)
                    kicked = v.copy()
                    if driven:
                        kicked[:, 0] += h * (100 * base.PN / m)
                    np.testing.assert_allclose(
                        self.project(final, m, z),
                        self.project(kicked, m, z),
                        atol=3e-7,
                        rtol=1e-6,
                    )
                    self.assertGreater(np.max(np.abs(final - kicked)), 1e-8)
                    heat = np.array(
                        __import__("re").findall(
                            r"thermostat_heat\s*=\s*([-+.0-9eE]+)", out["log"]
                        ),
                        dtype=float,
                    )
                    delta = 0.5 * np.sum(
                        m[:, None] * (final - kicked) * (final + kicked)
                    )
                    # Heat is the kinetic change at the thermal/friction stage,
                    # excluding force work. Floating kick order may differ by ulps.
                    np.testing.assert_allclose(
                        heat[-1], delta, atol=1e-6, rtol=1e-5
                    )

    def test_com_only_bias_is_independent_of_spatial_geometry(self):
        v = np.array([[0.4, 0.1, 0.2], [-0.2, 0.5, -0.3], [0.7, -0.2, 0.1]])
        for thermostat in THERMOSTATS:
            out = self.run_case(
                '[velocity_bias]\nmode="com"\n',
                masses=[1, 2, 3],
                velocities=v,
                settings=dict(
                    mode="nvt",
                    thermostat=thermostat,
                    target_temperature=100,
                    thermostat_tau=0.00001
                    if thermostat == "andersen"
                    else 0.001,
                    thermostat_seed=31,
                    dt=0.00001,
                    step_limit=1,
                ),
            )
            final = np.array(out["vel"][-1]).reshape(3, 3)
            np.testing.assert_allclose(
                np.average(final, axis=0, weights=[1, 2, 3]),
                np.average(v, axis=0, weights=[1, 2, 3]),
                atol=2e-7,
            )

    def test_exchange_and_profile_compose_without_method_selector(self):
        # No method selector is required to compose the two engine components.
        config = '[momentum_exchange]\ninterval=1\nbins=4\nflow="x"\ngradient="z"\n[sampling]\nprofile="slab"\nbins=4\nflow="x"\ngradient="z"\noutput="sample.h5"\n'
        out = self.run_case(
            config,
            masses=[2, 2],
            coordinates=[(1, 2, 1), (1, 2, 17)],
            velocities=[(1, 0, 0), (-1, 0, 0)],
            settings=dict(pbc=True, dt=0.00001, step_limit=2),
            viscosity_path="sample.h5",
        )
        np.testing.assert_allclose(
            out["vel"][-1], [-1, 0, 0, 1, 0, 0], atol=1e-7
        )
        self.assertEqual(out["viscosity"]["swaps"][-1], 1)
        self.assertGreater(out["viscosity"]["momentum"][-1], 0)


import test_coupling_restart as restart


class ThermalRestart(unittest.TestCase):
    setUp = restart.CouplingRestart.setUp
    run_case = restart.CouplingRestart.run_case
    read_state = restart.CouplingRestart.read_state

    def test_all_thermostat_bias_restarts_match_continuous_dynamics(self):
        for thermostat in THERMOSTATS:
            with self.subTest(thermostat=thermostat):
                args = dict(
                    drive='[velocity_bias]\nmode="com"\n',
                    settings=dict(
                        mode="nvt",
                        thermostat=thermostat,
                        target_temperature=100,
                        thermostat_tau=2 * restart.DT
                        if thermostat == "andersen"
                        else 0.1,
                        thermostat_seed=871,
                    ),
                )
                full = self.run_case(thermostat + "_full", 11, **args)
                first = self.run_case(thermostat + "_first", 4, **args)
                split = self.run_case(
                    thermostat + "_split",
                    7,
                    restart=first / "checkpoint.spgr.h5",
                    **args,
                )
                a, b = self.read_state(full), self.read_state(split)
                self.assertEqual(a[0], b[0])
                self.assertEqual(a[3], b[3])
                np.testing.assert_array_equal(a[1], b[1])
                np.testing.assert_array_equal(a[2], b[2])
                self.assertIn("velocity_bias_configuration", a[0])
                self.run_case(
                    thermostat + "_changed",
                    1,
                    restart=first / "checkpoint.spgr.h5",
                    **{**args, "drive": ""},
                    error="bias restart configuration mismatch",
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sponge", required=True)
    args, extra = parser.parse_known_args()
    base.SPONGE = str(Path(args.sponge).resolve())
    restart.SPONGE = base.SPONGE
    unittest.main(argv=[__file__, *extra])

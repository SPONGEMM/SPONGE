"""Native NHC+SHAKE thermal subspace and restart, without viscosity claims."""

import argparse
import re
import sys
import unittest
from pathlib import Path

import test_coupling_restart as base

np, h5py = base.np, base.h5py
sys.path.insert(0, str(base.ROOT))
from tools.viscosity.analysis import cosine


class NativeNHCConstraints(unittest.TestCase):
    setUp = base.CouplingRestart.setUp
    run_case = base.CouplingRestart.run_case
    read_state = base.CouplingRestart.read_state

    def options(self):
        return {
            "pbc": True,
            "mode": "nvt",
            "thermostat": "nose_hoover_chain",
            "target_temperature": 25,
            "thermostat_tau": 0.1,
            "constrain_mode": "SHAKE",
            "settle_disable": True,
            "constrain.in_file": "constraint.txt",
            "constrain.precision": "double",
            "SHAKE.tolerance": 1e-10,
            "SHAKE.iteration_numbers": 100,
            "SHAKE.step_length": 1.0,
            "PM.MPI_size": 1,
            "nose_hoover_chain.length": 1,
            "LJ_in_file": "lj.txt",
        }

    def arguments(self, stride=1):
        return dict(
            settings=self.options(),
            coordinates="2\n1 2 1\n3.4 2 4.2\n32 32 32\n90 90 90\n",
            velocities="2\n0.5 1 0.2\n0.4 -0.0625 0.2\n",
            files={
                "constraint.txt": "1\n0 1 4\n",
                "lj.txt": "2 1\n0\n0\n0\n0\n",
            },
            drive='[[external_force]]\ntype="acceleration"\nacceleration=[0,0,0]\n'
            'profile={type="cosine",axis="z",mode=1}\n'
            '[velocity_bias]\nmode="cosine"\nflow="x"\ngradient="z"\n'
            '[sampling]\nprofile="cosine"\nflow="x"\ngradient="z"\n'
            f'interval={stride}\nvelocity_time_layer="stored"\noutput="profile.h5"\n',
        )

    def test_constraint_dof_and_native_friction_feedback(self):
        args = self.arguments()
        args["files"]["charge.txt"] = "2\n0\n0\n"
        args["settings"].update(
            dt=1e-5,
            **{
                "nose_hoover_chain.restart_input": "chain.txt",
                "nose_hoover_chain.vel": "chain-velocity.txt",
            },
        )
        args["files"]["chain.txt"] = "0 0.7\n"
        path = self.run_case("one-step", 3, **args)
        # Independent global mass-weighted nullspace and QR oracle, including
        # a stored normal component. Two sites/one distance leave one thermal DOF.
        q = np.array([[1, 2, 1], [3.4, 2, 4.2]], np.float32).astype(float)
        v = np.array([[0.5, 1, 0.2], [0.4, -0.0625, 0.2]], np.float32).astype(
            float
        )
        m = np.array([1.0, 16.0])
        inverse = 1 / np.repeat(m, 3)
        d = q[0] - q[1]
        jac = np.r_[d, -d][None, :]
        pc = (
            np.eye(6)
            - (inverse[:, None] * jac.T)
            @ np.linalg.inv((jac * inverse) @ jac.T)
            @ jac
        )
        modes = np.zeros((6, 4))
        for axis in range(3):
            modes[axis::3, axis] = 1
        modes[0::3, 3] = np.cos(2 * np.pi * q[:, 2] / 32)
        root_mass = np.sqrt(np.repeat(m, 3))
        orth, _ = np.linalg.qr(root_mass[:, None] * (pc @ modes))
        x = root_mass * (pc @ v.ravel())
        thermal = (x - orth @ (orth.T @ x)) / root_mass
        energy = 0.5 * np.sum(np.repeat(m, 3) * thermal**2)
        kb = float(np.float32(0.00198716))
        data, meta = cosine.load_profile(path / "profile.h5")
        self.assertEqual(meta["velocity_time_layer"], "stored")
        self.assertEqual(int(meta["thermal_dof"]), 1)
        np.testing.assert_allclose(
            data["thermal_temperature"][0], 2 * energy / kb, rtol=1e-12
        )
        np.testing.assert_allclose(
            data["velocity_fourier"][0],
            2
            * np.average(v[:, 0] * np.cos(2 * np.pi * q[:, 2] / 32), weights=m)
            * float(np.float32(20.455))
            / 10,
            atol=1e-12,
        )
        mass = float(
            re.search(
                r"chain mass is\s+([-+\d.eE]+)", (path / "run.log").read_text()
            )[1]
        )
        h = float(np.float32(1e-5 * float(np.float32(20.455))))
        kbt = float(np.float32(np.float32(kb) * np.float32(25)))
        expected = 0.7 + (2 * energy - kbt) / mass * h
        chain = np.loadtxt(path / "chain-velocity.txt", ndmin=1)
        self.assertAlmostEqual(float(chain[0]), expected, delta=1.2e-6)
        native = np.fromfile(path / "vel.dat", np.float32).reshape(-1, 2, 3)[0]
        # SHAKE correction is along the original xz bond: it cannot change y
        # during this first drift, so its y update isolates NHC friction.
        expected_v = v.ravel() - h * expected * thermal
        np.testing.assert_allclose(
            native[:, 1], expected_v.reshape(2, 3)[:, 1], atol=8e-8, rtol=0
        )

    def test_restart_cadence_and_constraint_identity(self):
        args = self.arguments()
        full = self.run_case("full", 40, **args)
        first = self.run_case("first", 17, **args)
        split = self.run_case(
            "split", 23, restart=first / "checkpoint.spgr.h5", **args
        )
        a, b = self.read_state(full), self.read_state(split)
        self.assertEqual(a[0], b[0])
        self.assertEqual(a[3], b[3])
        np.testing.assert_array_equal(a[1], b[1])
        np.testing.assert_array_equal(a[2], b[2])
        with (
            h5py.File(full / "profile.h5") as f,
            h5py.File(split / "profile.h5") as g,
        ):
            for name, node in g["observables/all"].items():
                if isinstance(node, h5py.Group):
                    np.testing.assert_array_equal(
                        f[f"observables/all/{name}/value"][()][17:],
                        node["value"][()],
                    )
        sparse = self.run_case("sparse", 40, **self.arguments(stride=5))
        c = self.read_state(sparse)
        np.testing.assert_array_equal(a[1], c[1])
        np.testing.assert_array_equal(a[2], c[2])
        changed = self.arguments()
        changed["settings"]["SHAKE.tolerance"] = 1e-8
        self.run_case(
            "changed",
            2,
            restart=first / "checkpoint.spgr.h5",
            error="restart configuration mismatch",
            **changed,
        )
        removed = self.arguments()
        removed["settings"].pop("constrain_mode")
        removed["settings"].pop("constrain.in_file")
        self.run_case(
            "removed",
            2,
            restart=first / "checkpoint.spgr.h5",
            error="constraint bias restart configuration mismatch",
            **removed,
        )
        centered = self.arguments()
        centered["drive"] = centered["drive"].replace(
            'velocity_time_layer="stored"',
            'velocity_time_layer="force_centered"',
        )
        self.run_case(
            "centered", 2, error="velocity_time_layer=stored", **centered
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sponge", required=True)
    args, left = parser.parse_known_args()
    base.SPONGE = str(Path(args.sponge).resolve())
    unittest.main(argv=[sys.argv[0]] + left)

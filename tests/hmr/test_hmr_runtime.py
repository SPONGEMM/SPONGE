"""HMR startup, effective dynamics, constraints and restart compatibility."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path

try:
    import h5py
    import numpy as np
except ImportError:
    print("h5py/numpy unavailable: HMR runtime verification skipped")
    raise SystemExit(77)

SPONGE = None


class HMRRuntime(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="sponge-hmr-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def restart_case(
        self,
        name,
        steps,
        restart=None,
        load="dynamic",
        error=None,
        settings=None,
        files=None,
    ):
        directory = self.root / name
        directory.mkdir()
        fixture = (
            Path(__file__).resolve().parents[1]
            / "h5_bundle/fixtures/input_matrix/core_structural/bundled_input/bundle/topology.spgt.h5"
        )
        shutil.copy2(fixture, directory / "topology.spgt.h5")

        def text(dataset):
            value = dataset[()]
            return value.decode() if isinstance(value, bytes) else str(value)

        with h5py.File(fixture) as file:
            topology_hash = text(file["topology/topology_hash"])
            order_hash = text(file["topology/atom_order_hash"])
            for key in ("mass", "charge"):
                (directory / f"{key}.txt").write_text(
                    "2\n" + "\n".join(map(str, file[f"atoms/{key}"][()])) + "\n"
                )
        # Synthetic empty protocol tied to the immutable two-atom fixture.
        payload = json.dumps(
            {"topology": topology_hash, "cv": [], "restraint": []},
            sort_keys=True,
            separators=(",", ":"),
        )
        protocol_hash = "sha256:" + hashlib.sha256(payload.encode()).hexdigest()
        with h5py.File(directory / "protocol.spgp.h5", "w") as file:
            for key, value in {
                "schema/name": "sponge.protocol.h5",
                "schema/version": "sponge.input.v2",
                "identity/uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, payload)),
                "identity/content_hash": protocol_hash,
                "protocol/topology_compatibility/topology_hash": topology_hash,
            }.items():
                file.create_dataset(key, data=value)
        # Keep the pair outside the no-PBC cutoff, with nonzero velocities.
        (directory / "coordinate.txt").write_text(
            "2\n1 2 3\n51 2 3\n1000 1000 1000\n90 90 90\n"
        )
        (directory / "velocity.txt").write_text(
            "2\n0.03 0.01 0\n-0.01 0.02 0\n"
        )
        for filename, content in (files or {}).items():
            (directory / filename).write_text(content)
        options = dict(
            md_name=name,
            mode="nve",
            pbc=False,
            dt=0.000123456789,
            step_limit=steps,
            cutoff=8,
            print_zeroth_frame=True,
            write_mdout_interval=1,
            write_information_interval=1,
            write_trajectory_interval=1,
            write_restart_file_interval=1,
            output_h5_restart_path="checkpoint.spgr.h5",
            output_h5_restart_topology_hash=topology_hash,
            output_h5_restart_atom_order_hash=order_hash,
            output_h5_restart_protocol_hash=protocol_hash,
            crd="crd.dat",
            vel="vel.dat",
            frc="frc.dat",
        )
        options["PM.MPI_size"] = 0
        options.update(settings or {})
        if restart:
            options.update(
                input_h5_topology_path="topology.spgt.h5",
                input_h5_protocol_path="protocol.spgp.h5",
                input_h5_restart_path=str(restart),
                input_h5_restart_load=load,
            )
        else:
            for key in ("mass", "charge", "coordinate", "velocity"):
                options[f"{key}_in_file"] = f"{key}.txt"
        (directory / "mdin.spg.toml").write_text(
            "\n".join(f"{k}={json.dumps(v)}" for k, v in options.items()) + "\n"
        )
        run = subprocess.run(
            [SPONGE, "-mdin", "mdin.spg.toml"],
            cwd=directory,
            env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
            capture_output=True,
            text=True,
            timeout=40,
        )
        output = run.stdout + run.stderr
        (directory / "run.log").write_text(output)
        if error:
            self.assertNotEqual(run.returncode, 0, output)
            self.assertIn(error, output)
        else:
            self.assertEqual(run.returncode, 0, output)
        return directory

    def native(
        self,
        name,
        mass,
        hmr=None,
        constrained=False,
        generated=False,
        extra=None,
        cv=None,
    ):
        directory = self.root / name
        directory.mkdir()
        # CH3 plus a separate OHH water; all original bonds are slightly strained.
        q = np.array(
            [
                [3, 3, 3],
                [4.1, 3, 3],
                [3, 4.1, 3],
                [3, 3, 4.1],
                [12, 12, 12],
                [13, 12, 12],
                [12, 13, 12],
            ]
        )
        for key, values in (("mass", mass), ("charge", np.zeros(7))):
            (directory / f"{key}.txt").write_text(
                "7\n" + "\n".join(map(str, values)) + "\n"
            )
        for key, values in (("coordinate", q), ("velocity", np.zeros((7, 3)))):
            (directory / f"{key}.txt").write_text(
                "7\n"
                + "\n".join(" ".join(map(str, row)) for row in values)
                + ("\n32 32 32\n90 90 90\n" if key == "coordinate" else "\n")
            )
        (directory / "bond.txt").write_text(
            "5\n0 1 100 1\n0 2 100 1\n0 3 100 1\n4 5 100 1\n4 6 100 1\n"
        )
        options = dict(
            md_name=name,
            mode="nve",
            pbc=True,
            dt=0.0002,
            step_limit=3,
            cutoff=8,
            print_zeroth_frame=True,
            write_mdout_interval=1,
            write_information_interval=1,
            write_trajectory_interval=1,
            write_restart_file_interval=0,
            crd="crd.dat",
            vel="vel.dat",
            frc="frc.dat",
        )
        for key in ("mass", "charge", "coordinate", "velocity", "bond"):
            options[f"{key}_in_file"] = f"{key}.txt"
        options["PM.MPI_size"] = 1
        if hmr is not None:
            options.update({"hmr.enabled": True, "hmr.ratio_other": hmr})
        if constrained:
            options.update(constrain_mode="SHAKE", settle_disable=True)
            options.update(
                {"SHAKE.iteration_numbers": 200, "SHAKE.tolerance": 1e-8}
            )
            if hmr is None:
                options["constrain.mass"] = 4.0
        if generated:
            options.update(
                {
                    "initial_velocity.mode": "maxwell",
                    "initial_velocity.seed": 712,
                    "target_temperature": 300,
                }
            )
        if cv is not None:
            (directory / "cv.txt").write_text(cv)
            options["cv_in_file"] = "cv.txt"
        options.update(extra or {})
        (directory / "mdin.spg.toml").write_text(
            "\n".join(f"{k}={json.dumps(v)}" for k, v in options.items()) + "\n"
        )
        run = subprocess.run(
            [SPONGE, "-mdin", "mdin.spg.toml"],
            cwd=directory,
            env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
            capture_output=True,
            text=True,
            timeout=40,
        )
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        return directory, run.stdout

    def test_effective_mass_matches_manual_topology(self):
        original = np.array([12, 1, 1, 1, 16, 1, 1], dtype=float)
        for constrained, generated in (
            (False, False),
            (False, True),
            (True, True),
        ):
            suffix = f"{constrained}-{generated}"
            # Above the default constraint mass cutoff: tests membership based
            # on original masses, while solver and Maxwell use effective ones.
            hmr, log = self.native(
                "hmr-" + suffix, original, 3.5, constrained, generated
            )
            effective = original.copy()
            effective[:4] = [4.5, 3.5, 3.5, 3.5]
            manual, _ = self.native(
                "manual-" + suffix, effective, None, constrained, generated
            )
            self.assertIn("3 hydrogens repartitioned, 1 waters excluded", log)
            for filename in ("crd.dat", "vel.dat", "frc.dat"):
                a = np.fromfile(hmr / filename, dtype=np.float32)
                b = np.fromfile(manual / filename, dtype=np.float32)
                self.assertGreater(a.size, 0)
                np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(
                np.loadtxt(hmr / "mass.txt", skiprows=1), original
            )

    def test_thermostat_and_barostat_use_effective_masses(self):
        original = [12, 1, 1, 1, 16, 1, 1]
        effective = [7.5, 2.5, 2.5, 2.5, 16, 1, 1]
        for mode in ("nvt", "npt"):
            settings = dict(
                mode=mode,
                thermostat="middle_langevin",
                thermostat_tau=0.1,
                thermostat_seed=17,
                target_temperature=300,
            )
            if mode == "npt":
                settings.update(
                    barostat="berendsen_barostat",
                    barostat_tau=1,
                    barostat_update_interval=1,
                )
            automatic, _ = self.native(
                mode + "-hmr", original, 2.5, extra=settings
            )
            manual, _ = self.native(mode + "-manual", effective, extra=settings)
            for filename in ("crd.dat", "vel.dat", "frc.dat"):
                np.testing.assert_array_equal(
                    np.fromfile(automatic / filename, np.float32),
                    np.fromfile(manual / filename, np.float32),
                )

    def test_virtual_com_keeps_original_potential(self):
        cv = """print
{
    CV = com_x
}
com
{
    vatom_type = center_of_mass
    atom = 0 1
}
com_x
{
    CV_type = position_x
    atom = com
}
restrain
{
    CV = com_x
    weight = 2
    reference = 0
}
"""
        original = [12, 1, 1, 1, 16, 1, 1]
        normal, _ = self.native("com-normal", original, extra={"dt": 0}, cv=cv)
        hmr, _ = self.native("com-hmr", original, 2.5, extra={"dt": 0}, cv=cv)
        np.testing.assert_array_equal(
            np.fromfile(normal / "frc.dat", np.float32),
            np.fromfile(hmr / "frc.dat", np.float32),
        )

    def test_dynamic_restart_mass_contract(self):
        settings = {
            "bond_in_file": "bond.txt",
            "hmr.enabled": True,
            "hmr.ratio_other": 2.5,
        }
        files = {"bond.txt": "1\n0 1 0 50\n"}
        common = dict(settings=settings, files=files)
        full = self.restart_case("full", 5, **common)
        first = self.restart_case("first", 2, **common)
        split = self.restart_case(
            "split", 3, restart=first / "checkpoint.spgr.h5", **common
        )
        with (
            h5py.File(full / "checkpoint.spgr.h5") as a,
            h5py.File(split / "checkpoint.spgr.h5") as b,
        ):
            for key in ("position", "velocity"):
                np.testing.assert_array_equal(
                    a[f"particles/all/{key}/value"][()],
                    b[f"particles/all/{key}/value"][()],
                )
            self.assertIn(
                "mass_signature", a["parameters/restart/integrator_state"]
            )
        no_hmr = {"bond_in_file": "bond.txt"}
        self.restart_case(
            "disable-rejected",
            1,
            settings=no_hmr,
            files=files,
            restart=first / "checkpoint.spgr.h5",
            error="effective masses differ",
        )
        changed = {**settings, "hmr.ratio_xh1": 2.2}
        self.restart_case(
            "ratio-rejected",
            1,
            **{**common, "settings": changed},
            restart=first / "checkpoint.spgr.h5",
            error="effective masses differ",
        )
        # Structural load intentionally permits a fresh segment with new masses.
        self.restart_case(
            "structural",
            1,
            settings={
                **no_hmr,
                "initial_velocity.mode": "maxwell",
                "initial_velocity.seed": 7,
                "target_temperature": 300,
            },
            files=files,
            restart=first / "checkpoint.spgr.h5",
            load="structural",
        )
        normal = self.restart_case("normal", 1, settings=no_hmr, files=files)
        self.restart_case(
            "enable-rejected",
            1,
            **common,
            restart=normal / "checkpoint.spgr.h5",
            error="effective masses differ",
        )

    def test_invalid_configuration(self):
        for name, settings, error in (
            ("missing-ratio", {"hmr.enabled": True}, "ratio_other is required"),
            ("unknown", {"hmr.enabled": True, "hmr.typo": 2}, "unknown HMR"),
            ("bad-ratio", {"hmr.enabled": True, "hmr.ratio_other": 0}, ">= 1"),
            (
                "no-bonds",
                {"hmr.enabled": True, "hmr.ratio_other": 2.5},
                "one heavy-atom bond",
            ),
        ):
            self.restart_case(name, 1, settings=settings, error=error)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sponge", required=True)
    args, remaining = parser.parse_known_args()
    SPONGE = str(Path(args.sponge).resolve())
    unittest.main(argv=[sys.argv[0], *remaining])

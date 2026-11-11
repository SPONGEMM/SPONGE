"""Real executable/HDF5 restart closure for external mechanical work."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.viscosity.workflows.config import write_engine_input
import shutil
import subprocess
import sys
import tempfile
import unittest
import uuid

try:
    import h5py
    import numpy as np
except ImportError:
    print("h5py/numpy unavailable: coupling H5 runtime verification skipped")
    raise SystemExit(77)

ROOT = Path(__file__).resolve().parents[2]
SPONGE = None
DT = 0.000123456789
DRIVE = """
[[coupling]]
type="force"
force=[100,0,0]
schedule={type="sine", period=0.004321987654}
"""


def text(dataset):
    value=dataset[()]
    return value.decode() if isinstance(value,bytes) else str(value)


class CouplingRestart(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(prefix="sponge-coupling-restart-")
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)

    def run_case(self,name,steps,drive=DRIVE,restart=None,load="dynamic",error=None,settings=None,coordinates=None,velocities=None,files=None):
        directory=self.root/name
        directory.mkdir()
        fixture=ROOT/"tests/h5_bundle/fixtures/input_matrix/core_structural/bundled_input/bundle/topology.spgt.h5"
        shutil.copy2(fixture,directory/"topology.spgt.h5")
        with h5py.File(fixture) as file:
            topology_hash=text(file["topology/topology_hash"])
            atom_order_hash=text(file["topology/atom_order_hash"])
            for key in ("mass","charge"):
                values=file[f"atoms/{key}"][()]
                (directory/f"{key}.txt").write_text("2\n"+"\n".join(map(str,values))+"\n")
        # Synthetic empty protocol, with a digest of this explicit canonical
        # fixture payload; no force field or experimental data is generated.
        payload=json.dumps({"topology":topology_hash,"cv":[],"restraint":[]},sort_keys=True,separators=(",",":"))
        protocol_hash="sha256:"+hashlib.sha256(payload.encode()).hexdigest()
        with h5py.File(directory/"protocol.spgp.h5","w") as file:
            for key,value in {"schema/name":"sponge.protocol.h5","schema/version":"sponge.input.v2",
                              "identity/uuid":str(uuid.uuid5(uuid.NAMESPACE_URL,payload)),
                              "identity/content_hash":protocol_hash,
                              "protocol/topology_compatibility/topology_hash":topology_hash}.items():
                file.create_dataset(key,data=value)
        # The immutable fixture has masses 1/16 and charges +/-0.1 in native
        # units. Keep the two atoms outside the no-PBC cutoff for free motion.
        (directory/"coordinate.txt").write_text(coordinates or "2\n1 2 3\n51 2 3\n1000 1000 1000\n90 90 90\n")
        (directory/"velocity.txt").write_text(velocities or "2\n0 0 0\n0 0 0\n")
        for filename, content in (files or {}).items():
            (directory/filename).write_text(content)
        options=dict(md_name=name,mode="nve",pbc=False,dt=DT,step_limit=steps,cutoff=8,
                     print_zeroth_frame=True,
                     write_mdout_interval=1,write_information_interval=1,
                     write_trajectory_interval=1,write_restart_file_interval=1,
                     output_h5_restart_path="checkpoint.spgr.h5",
                     output_h5_restart_topology_hash=topology_hash,
                     output_h5_restart_atom_order_hash=atom_order_hash,
                     output_h5_restart_protocol_hash=protocol_hash,
                     output_h5_observable_path="observables.h5",
                     crd="crd.dat",vel="vel.dat",frc="frc.dat")
        options["PM.MPI_size"]=0
        options.update(settings or {})
        if restart:
            options.update(input_h5_topology_path="topology.spgt.h5",input_h5_protocol_path="protocol.spgp.h5",
                           input_h5_restart_path=str(restart),input_h5_restart_load=load)
        else:
            options.update(mass_in_file="mass.txt",charge_in_file="charge.txt",
                           coordinate_in_file="coordinate.txt",velocity_in_file="velocity.txt")
        write_engine_input(directory/"mdin.spg.toml", "\n".join(f"{k}={json.dumps(v)}" for k,v in options.items())+"\n"+drive)
        result=subprocess.run([SPONGE,"-mdin","mdin.spg.toml"],cwd=directory,
                              env={**os.environ,"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1"},
                              stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=40)
        (directory/"run.log").write_text(result.stdout)
        if error:
            self.assertNotEqual(result.returncode,0,result.stdout)
            self.assertIn(error,result.stdout)
            return directory
        self.assertEqual(result.returncode,0,result.stdout)
        return directory

    def read_state(self,directory):
        with h5py.File(directory/"checkpoint.spgr.h5") as file:
            ledger={(name.removeprefix("energy_").removeprefix("exchange_")):text(value) for name,value in file["parameters/restart/coupling"].items()}
            coords=file["particles/all/position/value"][()]
            vel=file["particles/all/velocity/value"][()]
            time=text(file["parameters/restart/integrator_state/time"])
            return ledger,coords,vel,time

    def test_continuous_and_split_restart_are_identical(self):
        full=self.run_case("full",8)
        first=self.run_case("first",3)
        second=self.run_case("second",5,restart=first/"checkpoint.spgr.h5")
        full_state=self.read_state(full)
        split_state=self.read_state(second)
        self.assertEqual(full_state[0],split_state[0])
        np.testing.assert_array_equal(full_state[1],split_state[1])
        np.testing.assert_array_equal(full_state[2],split_state[2])
        self.assertEqual(full_state[3],split_state[3])
        self.assertEqual(int(full_state[0]["steps"]),8)
        self.assertEqual(float(full_state[0]["time"]),8*DT)
        self.assertEqual(float(full_state[3]),8*DT)
        with h5py.File(full/"observables.h5") as file:
            work=file["observables/all/coupling_work/value"][()]
            times=file["observables/all/time"][()]
            self.assertEqual(float(np.ravel(work)[0]),0)
            self.assertEqual(float(np.ravel(work)[-1]),float(full_state[0]["work"]))
            self.assertEqual(float(np.ravel(times)[-1]),8*DT)

    def test_group_torque_work_and_geometry_survive_restart(self):
        drive = """
[[coupling]]
type="group_load"
atoms=[0,1]
force=[10,0,0]
torque=[0,0,100]
schedule={type="sine", period=0.004321987654}
"""
        full = self.run_case("group_full", 8, drive=drive)
        first = self.run_case("group_first", 3, drive=drive)
        second = self.run_case("group_second", 5, drive=drive,
                               restart=first/"checkpoint.spgr.h5")
        expected, actual = self.read_state(full), self.read_state(second)
        self.assertEqual(expected[0], actual[0])
        self.assertGreater(float(expected[0]["work"]), 0)
        np.testing.assert_array_equal(expected[1], actual[1])
        np.testing.assert_array_equal(expected[2], actual[2])
        self.assertEqual(expected[3], actual[3])

    def test_cosine_projected_langevin_restart_is_identical(self):
        drive='[viscosity]\nmethod="cosine"\nacceleration=0.25\n'
        settings={"pbc":True,"mode":"nvt","thermostat":"middle_langevin",
                  "target_temperature":300,"thermostat_tau":0.013,"thermostat_seed":1423,
                  "PM.MPI_size":1,"LJ_in_file":str(ROOT/"tests/h5_bundle/fixtures/input_matrix/full_contract_rerun/bundled_input_with_legacy_sidecar/bundle/legacy_sidecars/LJ_in_file/lj.txt")}
        coords="2\n1 2 1\n17 2 9\n32 32 32\n90 90 90\n"
        full=self.run_case("cos_full",8,drive=drive,settings=settings,coordinates=coords)
        first=self.run_case("cos_first",3,drive=drive,settings=settings,coordinates=coords)
        second=self.run_case("cos_second",5,drive=drive,settings=settings,restart=first/"checkpoint.spgr.h5")
        expected,actual=self.read_state(full),self.read_state(second)
        self.assertEqual(expected[0],actual[0])
        np.testing.assert_array_equal(expected[1],actual[1])
        np.testing.assert_array_equal(expected[2],actual[2])
        self.assertEqual(expected[3],actual[3])
        with h5py.File(full/"observables.h5") as a,h5py.File(second/"observables.h5") as b:
            for name in ("bias_amplitude","thermal_temperature","bias_com_x","thermostat_heat"):
                self.assertEqual(float(np.ravel(a[f"observables/all/{name}/value"])[-1]),
                                 float(np.ravel(b[f"observables/all/{name}/value"])[-1]))
        with h5py.File(full/"viscosity.h5") as a, h5py.File(second/"viscosity.h5") as b:
            np.testing.assert_array_equal(a["observables/all/step"][()][3:], b["observables/all/step"][()])
            # Existing restart clock adds segment origin plus local dt;
            # regrouping the double sum may differ by one ULP.
            np.testing.assert_allclose(a["observables/all/time"][()][3:], b["observables/all/time"][()], rtol=3e-16, atol=0)
            for name, node in b["observables/all"].items():
                if isinstance(node, h5py.Group) and "value" in node:
                    np.testing.assert_array_equal(a[f"observables/all/{name}/value"][()][3:], node["value"][()])
        self.assertIn("heat_correction", expected[0])
        self.assertEqual(expected[0]["heat_method"], "thermal_substep_v1")
        with h5py.File(full/"viscosity.h5") as a, h5py.File(full/"observables.h5") as b:
            heat = np.ravel(a["observables/all/thermostat_heat/value"][()])
            np.testing.assert_array_equal(heat, np.ravel(b["observables/all/thermostat_heat/value"][()]))
            self.assertEqual(heat[-1], float(expected[0]["heat"]))
        # Every extra ledger field is covered by the existing restart hash.
        for key in ("heat", "heat_correction", "heat_method"):
            tampered = self.root/f"tampered-{key}.spgr.h5"
            shutil.copy2(first/"checkpoint.spgr.h5", tampered)
            with h5py.File(tampered,"r+") as file:
                file[f"parameters/restart/coupling/energy_{key}"][()] = b"broken"
            self.run_case(f"tampered_{key}",1,drive=drive,settings=settings,restart=tampered,error="hash")
        # Measurement settings can change between segments without changing
        # the dynamical identity, work ledger or RNG continuation.
        changed = self.run_case("cos_new_stream", 5,
                                drive=drive+"bins=7\nsample_interval=2\noutput='new.h5'\n",
                                settings=settings, restart=first/"checkpoint.spgr.h5")
        updated = self.read_state(changed)
        self.assertEqual(expected[0], updated[0])
        np.testing.assert_array_equal(expected[1], updated[1])
        np.testing.assert_array_equal(expected[2], updated[2])
        with h5py.File(full/"viscosity.h5") as a, h5py.File(changed/"new.h5") as b:
            np.testing.assert_array_equal(b["observables/all/step"][()], [4, 6, 8])
            for name in ("velocity_fit", "coupling_work", "thermal_temperature", "thermostat_heat"):
                np.testing.assert_array_equal(a[f"observables/all/{name}/value"][()][4::2], b[f"observables/all/{name}/value"][()])
        self.run_case("cos_changed_tau",1,drive=drive,settings={**settings,"thermostat_tau":.014},
                      restart=first/"checkpoint.spgr.h5",error="configuration")

    def test_changed_or_removed_coupling_is_rejected(self):
        first=self.run_case("first",2)
        checkpoint=first/"checkpoint.spgr.h5"
        self.run_case("changed",1,drive=DRIVE.replace("100","101"),restart=checkpoint,error="configuration")
        self.run_case("removed",1,drive="",restart=checkpoint,error="configuration missing or removed")
        new=self.run_case("new_experiment",1,drive=DRIVE.replace("100","101"),restart=checkpoint,load="structural")
        self.assertEqual(int(self.read_state(new)[0]["steps"]),1)

    def test_green_kubo_restart_samples_keep_global_cadence(self):
        drive = '[viscosity]\nmethod="green_kubo"\nsample_interval=2\n'
        lj_file = ROOT/"tests/h5_bundle/fixtures/input_matrix/full_contract_rerun/bundled_input_with_legacy_sidecar/bundle/legacy_sidecars/LJ_in_file/lj.txt"
        settings = {"pbc": True, "PM.MPI_size": 1, "LJ_in_file": str(lj_file)}
        coords = "2\n1 2 1\n4 5 2\n32 32 32\n90 90 90\n"
        full = self.run_case("gk_full", 9, drive=drive, settings=settings, coordinates=coords)
        first = self.run_case("gk_first", 4, drive=drive, settings=settings, coordinates=coords)
        second = self.run_case("gk_second", 5, drive=drive, settings=settings, restart=first/"checkpoint.spgr.h5")
        with h5py.File(full/"viscosity.h5") as a, h5py.File(second/"viscosity.h5") as b:
            steps = a["observables/all/step"][()]
            mask = steps >= 4
            np.testing.assert_array_equal(steps[mask], b["observables/all/step"][()])
            np.testing.assert_array_equal(a["observables/all/time"][()][mask], b["observables/all/time"][()])
            for name in ("pxx", "pyy", "pzz", "pxy", "pxz", "pyz", "temperature", "volume"):
                np.testing.assert_array_equal(a[f"observables/all/{name}/value"][()][mask], b[f"observables/all/{name}/value"][()])

    def test_constrained_green_kubo_restart_identity_and_cadence(self):
        drive = '[viscosity]\nmethod="green_kubo"\nsample_interval=2\n'
        coords = "2\n1 2 1\n5 2 1\n32 32 32\n90 90 90\n"
        velocity = "2\n0 .1 .2\n0 -.00625 -.0125\n"
        files = {"constraint.txt": "1\n0 1 4\n", "lj.txt": "2 1\n0\n0\n0\n0\n"}
        for algorithm in ("SETTLE", "SHAKE"):
            settings = {"pbc": True, "PM.MPI_size": 1, "LJ_in_file": "lj.txt",
                        "constrain_mode": algorithm, "settle_disable": algorithm == "SHAKE",
                        "constrain.in_file": "constraint.txt", "SHAKE.tolerance": 1e-6}
            args = dict(drive=drive, settings=settings, files=files)
            full = self.run_case(algorithm+"_gk_full", 9, coordinates=coords, velocities=velocity, **args)
            first = self.run_case(algorithm+"_gk_first", 4, coordinates=coords, velocities=velocity, **args)
            checkpoint = first/"checkpoint.spgr.h5"
            second = self.run_case(algorithm+"_gk_second", 5, restart=checkpoint, **args)
            expected, actual = self.read_state(full), self.read_state(second)
            self.assertEqual(expected[0], actual[0])
            self.assertEqual(set(actual[0]), {"constraint_configuration"})
            np.testing.assert_array_equal(expected[1], actual[1])
            np.testing.assert_array_equal(expected[2], actual[2])
            self.assertEqual(expected[3], actual[3])
            with h5py.File(full/"viscosity.h5") as a, h5py.File(second/"viscosity.h5") as b:
                mask = a["observables/all/step"][()] >= 4
                for name, node in b["observables/all"].items():
                    if isinstance(node, h5py.Group) and "value" in node:
                        np.testing.assert_array_equal(a[f"observables/all/{name}/value"][()][mask], node["value"][()])
            changed = self.run_case(algorithm+"_gk_cadence", 5, restart=checkpoint,
                                    **{**args, "drive": drive.replace("interval=2", "interval=3")})
            updated = self.read_state(changed)
            self.assertEqual(expected[0], updated[0])
            np.testing.assert_array_equal(expected[1], updated[1])
            np.testing.assert_array_equal(expected[2], updated[2])
            with h5py.File(changed/"viscosity.h5") as h5:
                np.testing.assert_array_equal(h5["observables/all/step"][()], [6, 9])
            for label, update in (("dt", {"dt": DT*1.01}),
                                  ("precision", {"constrain_precision": "single"}),
                                  ("algorithm", {"constrain_mode": "SHAKE" if algorithm == "SETTLE" else "SETTLE", "settle_disable": algorithm == "SETTLE"})):
                self.run_case(algorithm+"_gk_change_"+label, 1, restart=checkpoint, error="configuration mismatch",
                              **{**args, "settings": {**settings, **update}})
            single = self.run_case(algorithm+"_gk_single", 2, coordinates=coords, velocities=velocity,
                                   **{**args, "settings": {**settings, "constrain_precision": "single"}})
            self.assertIn("precision=single", self.read_state(single)[0]["constraint_configuration"])
            self.run_case(algorithm+"_gk_change_distance", 1, restart=checkpoint, error="configuration mismatch",
                          **{**args, "files": {**files, "constraint.txt": "1\n0 1 4.00001\n"}})
            old = self.run_case(algorithm+"_gk_old", 2, coordinates=coords, velocities=velocity, **{**args, "drive": ""})
            self.run_case(algorithm+"_gk_missing", 1, restart=old/"checkpoint.spgr.h5", error="missing identity", **args)
            self.run_case(algorithm+"_gk_structural", 1, restart=old/"checkpoint.spgr.h5", load="structural", **args)
            tampered = self.root/f"{algorithm}-tampered.spgr.h5"
            shutil.copy2(checkpoint, tampered)
            with h5py.File(tampered, "r+") as h5:
                h5["parameters/restart/coupling/constraint_configuration"][()] = b"broken"
            self.run_case(algorithm+"_gk_hash", 1, restart=tampered, error="hash", **args)
        # Changes to actual SHAKE solver settings also change dynamical identity.
        self.run_case("SHAKE_gk_change_tolerance", 1, restart=checkpoint, error="configuration mismatch",
                      **{**args, "settings": {**settings, "SHAKE.tolerance": 1e-5}})
        self.run_case("SHAKE_gk_remove_constraints", 1, restart=checkpoint, error="configuration mismatch",
                      **{**args, "settings": {k: v for k, v in settings.items() if k not in ("constrain_mode", "constrain.in_file", "settle_disable", "SHAKE.tolerance")}})

    def test_rnemd_restart_preserves_exchange_ledger_and_samples(self):
        drive = '[viscosity]\nmethod="rnemd"\nexchange_interval=1\nbins=8\n'
        lj_file = ROOT/"tests/h5_bundle/fixtures/input_matrix/full_contract_rerun/bundled_input_with_legacy_sidecar/bundle/legacy_sidecars/LJ_in_file/lj.txt"
        settings = {"pbc": True, "PM.MPI_size": 1, "LJ_in_file": str(lj_file)}
        coords = "2\n1 2 1\n4 5 17\n32 32 32\n90 90 90\n"
        velocity = "2\n2 0 0\n-0.125 0 0\n"
        full = self.run_case("rn_full", 8, drive=drive, settings=settings, coordinates=coords, velocities=velocity)
        first = self.run_case("rn_first", 3, drive=drive, settings=settings, coordinates=coords, velocities=velocity)
        second = self.run_case("rn_second", 5, drive=drive, settings=settings, restart=first/"checkpoint.spgr.h5")
        expected, actual = self.read_state(full), self.read_state(second)
        self.assertEqual(expected[0], actual[0])
        self.assertGreater(float(actual[0]["momentum"]), 0)
        np.testing.assert_array_equal(expected[1], actual[1])
        np.testing.assert_array_equal(expected[2], actual[2])
        with h5py.File(full/"viscosity.h5") as a, h5py.File(second/"viscosity.h5") as b:
            for name, node in b["observables/all"].items():
                if isinstance(node, h5py.Group) and "value" in node:
                    np.testing.assert_array_equal(a[f"observables/all/{name}/value"][()][3:], node["value"][()])
        self.run_case("rn_changed", 2, drive=drive.replace("interval=1", "interval=2"), settings=settings,
                      restart=first/"checkpoint.spgr.h5", error="configuration")
        self.run_case("rn_removed", 2, drive="", settings=settings, restart=first/"checkpoint.spgr.h5", error="configuration missing or removed")
        old = self.run_case("rn_equilibrium", 2, drive="", settings=settings, coordinates=coords)
        self.run_case("rn_missing", 2, drive=drive, settings=settings, restart=old/"checkpoint.spgr.h5", error="configuration missing or removed")

    def test_missing_ledger_cannot_silently_reset_work(self):
        old=self.run_case("equilibrium",2,drive="")
        self.run_case("new_drive",1,restart=old/"checkpoint.spgr.h5",error="configuration missing or removed")

    def test_work_payload_is_covered_by_restart_hash(self):
        first=self.run_case("first",2)
        tampered=self.root/"tampered.spgr.h5"
        shutil.copy2(first/"checkpoint.spgr.h5",tampered)
        with h5py.File(tampered,"r+") as file:
            file["parameters/restart/coupling/energy_work"][()]=b"100.0"
        self.run_case("tampered",1,restart=tampered,error="hash")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--sponge",required=True)
    args,extra=parser.parse_known_args()
    SPONGE=str(Path(args.sponge).resolve())
    unittest.main(argv=[__file__,*extra])

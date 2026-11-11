"""Analytic checks against the real executable; no third-party Python modules.

Legacy trajectories store post-update frames for loop indices 0..limit-1;
force output also includes the final force evaluation at index limit.
The leapfrog input velocity is the incoming half-step velocity.
"""
import argparse
import json
import math
import os
import re
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.viscosity.workflows.config import write_engine_input
import struct
import subprocess
import tempfile
import unittest

PN = 6.02214076e23 * 1e-22 / 4184
TIME = 20.455
EV = 23.060548
SPONGE = None


class CouplingRuntime(unittest.TestCase):
    def run_case(self, coupling="", masses=(12.0,), charges=None,
                 coordinates=None, velocities=None, settings=None, error=None, viscosity_path=None):
        with tempfile.TemporaryDirectory(prefix="sponge-coupling-") as directory:
            path = Path(directory)
            n = len(masses)
            charges = charges if charges is not None else [0.0]*n
            coordinates = coordinates if coordinates is not None else [(1+3*i, 2, 3) for i in range(n)]
            velocities = velocities if velocities is not None else [(0,0,0)]*n
            # Native charge.txt stores the SPONGE-scaled charge, unlike the
            # physical elementary-charge values accepted by this fixture.
            for name, values in (("mass", masses), ("charge", [q*18.2223 for q in charges])):
                (path/f"{name}.txt").write_text(str(n)+"\n"+"\n".join(map(str, values))+"\n")
            for name, values in (("coordinate", coordinates), ("velocity", velocities)):
                box = "32 32 32" if (settings or {}).get("pbc") else "1000 1000 1000"
                tail = box+"\n90 90 90\n" if name == "coordinate" else ""
                (path/f"{name}.txt").write_text(str(n)+"\n"+"\n".join(" ".join(map(str,v)) for v in values)+"\n"+tail)
            options = dict(md_name="coupling_test", mode="nve", pbc=False,
                           step_limit=3, dt=0.001, cutoff=8, print_zeroth_frame=True,
                           write_mdout_interval=1, write_information_interval=1,
                           write_trajectory_interval=1, write_restart_file_interval=0,
                           crd="crd.dat", vel="vel.dat", frc="frc.dat")
            options["PM.MPI_size"] = 0
            for name in ("mass", "charge", "coordinate", "velocity"):
                options[f"{name}_in_file"] = f"{name}.txt"
            options.update(settings or {})
            try:
                write_engine_input(path/"mdin.spg.toml", "\n".join(f"{k}={json.dumps(v)}" for k,v in options.items())+"\n"+coupling)
            except ValueError as failure:
                if not error: raise
                self.assertIn(error,str(failure))
                return
            result = subprocess.run([SPONGE, "-mdin", "mdin.spg.toml"], cwd=path,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, timeout=45,
                                    env={**os.environ, "OMP_NUM_THREADS":"1", "MKL_NUM_THREADS":"1"})
            if error:
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn(error, result.stdout)
                return
            self.assertEqual(result.returncode, 0, result.stdout)
            output = {"log": result.stdout, "work": [float(x) for x in re.findall(r"coupling_work\s*=\s*([-+.0-9eE]+)",result.stdout)]}
            for name in ("frc", "vel", "crd"):
                raw = (path/f"{name}.dat").read_bytes()
                self.assertEqual(len(raw) % (12*n), 0)
                output[name+"_raw"] = raw
                values = struct.unpack(f"={len(raw)//4}f", raw)
                output[name] = [values[i:i+3*n] for i in range(0,len(values),3*n)]
            if viscosity_path:
                import h5py
                with h5py.File(path/viscosity_path) as file:
                    output["viscosity"] = {
                        name: node["value"][()] for name, node in file["observables/all"].items()
                        if isinstance(node, h5py.Group) and "value" in node}
                    output["viscosity_step"] = file["observables/all/step"][()]
                    output["viscosity_time"] = file["observables/all/time"][()]
                    output["viscosity_units"] = {
                        name: node["value"].attrs["unit"] for name, node in file["observables/all"].items()
                        if isinstance(node, h5py.Group) and "value" in node}
            return output

    def close(self, actual, expected, atol=1e-7, rtol=2e-6):
        self.assertLessEqual(abs(actual-expected), atol+rtol*abs(expected), (actual, expected))

    def test_force_units_and_leapfrog(self):
        out = self.run_case('[[coupling]]\ntype="force"\nforce=[100,-25,0]\n')
        self.assertEqual(len(out["vel"]), 3)
        self.assertEqual(len(out["frc"]), 4)
        for row in out["frc"]:
            self.close(row[0], 100*PN)
            self.close(row[1], -25*PN)
        for step, row in enumerate(out["vel"], 1):
            self.close(row[0], step*0.001*TIME*100*PN/12)
        for step, row in enumerate(out["crd"], 1):
            self.close(row[0], 1+step*(step+1)/2*(0.001*TIME)**2*100*PN/12, atol=1e-6)
            self.close(out["work"][step],100*PN*(row[0]-1)-25*PN*(row[1]-2),atol=1e-10)
        self.assertEqual(out["work"][0],0)

    def test_acceleration_mass_independence_and_selection(self):
        out = self.run_case('[[coupling]]\ntype="acceleration"\natoms=[2,0]\nacceleration=[1,0,0]\n', masses=(1,2,4))
        for step, row in enumerate(out["vel"], 1):
            self.close(row[0], step*0.001*10/TIME)
            self.close(row[6], row[0])
            self.assertEqual(row[3], 0)
        self.close(out["frc"][0][6], 4*out["frc"][0][0])

    def test_electric_charge_sign_and_units(self):
        for q in (1,-1,0):
            with self.subTest(charge=q):
                out = self.run_case('[[coupling]]\ntype="electric_field"\nfield=[0,0,0.5]\n', charges=(q,))
                self.close(out["frc"][0][2], q*0.5*EV/10)

    def test_superposition_and_disabled_identity(self):
        base = self.run_case(velocities=((0.2,-0.1,0),))
        zero = self.run_case('[[coupling]]\ntype="force"\nforce=[0,0,0]\n', velocities=((0.2,-0.1,0),))
        for name in ("frc", "vel", "crd"):
            self.assertEqual(base[name+"_raw"], zero[name+"_raw"])
        out = self.run_case('[[coupling]]\ntype="force"\nforce=[100,0,0]\n[[coupling]]\ntype="force"\nforce=[-40,20,0]\n')
        self.close(out["frc"][0][0], 60*PN)
        self.close(out["frc"][0][1], 20*PN)

    def test_absolute_time_protocols(self):
        cases = [("{type='ramp',start=0,duration=0.002}", [0,.5,1,1]),
                 ("{type='sine',period=0.004}", [0,1,0,-1]),
                 ("{type='table',points=[[0,0],[0.002,1],[0.003,-1]]}", [0,.5,1,-1])]
        for schedule, scales in cases:
            with self.subTest(schedule=schedule):
                out = self.run_case('[[coupling]]\ntype="force"\nforce=[100,0,0]\nschedule='+schedule+'\n')
                for row, scale in zip(out["frc"], scales): self.close(row[0], 100*PN*scale, atol=2e-6)
                for step, row in enumerate(out["vel"], 1):
                    self.close(row[0], sum(scales[:step])*100*PN*.001*TIME/12)

    def test_group_force_and_torque(self):
        out = self.run_case('[[coupling]]\ntype="group_load"\natoms=[1,0]\nforce=[10,0,0]\ntorque=[0,0,2]\n',
                            masses=(2,2), coordinates=((-1,0,0),(1,0,0)), settings={"dt":0})
        f = out["frc"][0]
        self.close(f[0]+f[3], 10*PN)
        self.close(f[1]+f[4], 0)
        self.close(-f[1]+f[4], 2*10*PN)
        self.close(f[0], 5*PN)
        self.assertEqual(f[2], 0)

    def test_smooth_drive_second_order_convergence(self):
        # Continuous a(t)=A sin(wt), v(0)=x(0)=0. Leapfrog requires
        # v(-dt/2) on input and writes v(t_end-dt/2), x(t_end).
        amplitude, period, end = 20.0, 0.04, 0.03
        omega = 2*math.pi/period
        errors = []
        for dt in (0.002, 0.001, 0.0005):
            n = round(end/dt)
            v0 = amplitude/omega*(1-math.cos(omega*dt/2))*10/TIME
            out = self.run_case('[[coupling]]\ntype="acceleration"\nacceleration=[20,0,0]\nschedule={type="sine",period=0.04}\n',
                                coordinates=((0,0,0),), velocities=((v0,0,0),), settings={"dt":dt,"step_limit":n})
            x_exact = amplitude/omega*(end-math.sin(omega*end)/omega)*10
            v_exact = amplitude/omega*(1-math.cos(omega*(end-dt/2)))*10/TIME
            v_end = amplitude/omega*(1-math.cos(omega*end))*10/TIME
            work_exact = 0.5*12*v_end*v_end
            errors.append((abs(out["crd"][-1][0]-x_exact), abs(out["vel"][-1][0]-v_exact), abs(out["work"][-1]-work_exact)))
        for coarse, fine in zip(errors, errors[1:]):
            for a,b in zip(coarse, fine):
                self.assertGreater(a/b, 3.5)
                self.assertLess(a/b, 4.5)

    def test_work_across_periodic_output_wrapping(self):
        out=self.run_case('[[coupling]]\ntype="force"\nforce=[100,0,0]\n',
                          coordinates=((31.8,0,0),),velocities=((1,0,0),),
                          settings={"pbc":True,"PM.MPI_size":1,"dt":0.1})
        for n in range(4):
            displacement=n*.1*TIME+n*(n+1)/2*(.1*TIME)**2*100*PN/12
            self.close(out["work"][n],100*PN*displacement,atol=2e-5)

    def test_invalid_inputs_and_unsupported_paths(self):
        for text, error in [
            ('[[coupling]]\ntype="force"\nforce=[nan,0,0]', "finite"),
            ('[[coupling]]\ntype="force"\nforce=[1,0,0]\natoms=[2]', "outside system"),
            ('[[coupling]]\ntype="force"\nforce=[1,0,0]\natoms=[0.0]', "integer"),
            ('[[coupling]]\ntype="group_load"\natoms=[0]\ntorque=[0,0,1]', "not representable"),
            ('[viscosity]\nmethod="sllod"', "not implemented"),
        ]:
            with self.subTest(error=error): self.run_case(text, error=error)
        self.run_case('[[coupling]]\ntype="force"\nforce=[1,0,0]', settings={"velocity_max":1}, error="velocity clipping")

    def test_field_adds_to_periodic_electrostatics(self):
        kwargs = dict(masses=(12,12), charges=(1,-1), settings={"pbc":True,"dt":0,"step_limit":1,"PM.MPI_size":1})
        baseline = self.run_case(**kwargs)
        driven = self.run_case('[[coupling]]\ntype="electric_field"\nfield=[0.1,0,0]\n', **kwargs)
        self.assertNotIn("PM RECI NOT INITIALIZED", baseline["log"])
        self.assertIn("START INITIALIZING PME", baseline["log"])
        for atom, q in enumerate((1,-1)):
            for axis in range(3):
                self.close(driven["frc"][0][3*atom+axis]-baseline["frc"][0][3*atom+axis], q*.1*EV/10 if axis == 0 else 0, atol=1e-5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sponge", required=True)
    args, extra = parser.parse_known_args()
    SPONGE = str(Path(args.sponge).resolve())
    unittest.main(argv=[__file__, *extra])

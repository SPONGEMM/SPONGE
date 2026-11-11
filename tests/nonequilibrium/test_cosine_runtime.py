"""Analytic and statistical checks for the actual cosine/Middle OU path.

These ideal-particle tests validate mechanics, not liquid transport accuracy.
"""
import argparse
import math
from pathlib import Path
import re
import statistics
import sys
import unittest
import test_coupling_runtime as base

TIME = base.TIME

def series(out, name):
    return [float(v) for v in re.findall(rf"\b{name}\s*=\s*([-+.0-9eE]+)",out["log"])]

def cosine(amplitude=0,flow="x",gradient="z"):
    return f'[viscosity]\nmethod="cosine"\nacceleration={amplitude}\nflow="{flow}"\ngradient="{gradient}"\n'

class CosineRuntime(unittest.TestCase):
    run_case=base.CouplingRuntime.run_case
    close=base.CouplingRuntime.close

    def test_force_profile_and_mass_scaling(self):
        r=[(1,2,z) for z in (0,8,16,24)]
        masses=(1,2,4,8)
        out=self.run_case(cosine(5),masses=masses,coordinates=r,settings={"pbc":True,"dt":0})
        for i in range(4):
            expected=masses[i]*5*10/TIME**2*math.cos(2*math.pi*r[i][2]/32)
            self.close(out["frc"][0][3*i],expected)
            self.assertEqual(out["frc"][0][3*i+1],0)
        # Axis permutation must use the declared box coordinate and force axis.
        out=self.run_case(cosine(-3,"y","x"),masses=masses,
                          coordinates=[(x,2,1) for x in (0,8,16,24)],settings={"pbc":True,"dt":0})
        for i,x in enumerate((0,8,16,24)):
            self.close(out["frc"][0][3*i+1],masses[i]*-3*10/TIME**2*math.cos(2*math.pi*x/32))

    def test_pure_stream_is_preserved_by_zero_temperature_ou(self):
        r=[(1,2,4*i) for i in range(8)]
        v=[(.3+.8*math.cos(2*math.pi*x[2]/32),-.2,0) for x in r]
        out=self.run_case(cosine(),masses=[1,2,3,5,7,11,13,17],coordinates=r,velocities=v,
                          settings={"pbc":True,"mode":"nvt","thermostat":"middle_langevin",
                                    "target_temperature":0,"thermostat_tau":.002,"dt":.001,"thermostat_seed":13})
        for row in out["vel"]:
            for i in range(8):
                for j in range(3):self.close(row[3*i+j],v[i][j],atol=3e-7)
        self.assertLess(max(series(out,"thermal_temperature")),1e-8)
        for a in series(out,"bias_amplitude"):self.close(a,.8*TIME/10,atol=1e-6)

    def test_thermal_residual_has_ou_decay(self):
        r=[(1,2,4*i) for i in range(8)]
        v=[(.8*math.cos(2*math.pi*x[2]/32)+.4*math.sin(2*math.pi*x[2]/32),0,0) for x in r]
        out=self.run_case(cosine(),masses=[12]*8,coordinates=r,velocities=v,
                          settings={"pbc":True,"mode":"nvt","thermostat":"middle_langevin",
                                    "target_temperature":0,"thermostat_tau":.002,"dt":.001,"thermostat_seed":13})
        for step,row in enumerate(out["vel"],1):
            for i in range(8):
                c=2*math.pi*r[i][2]/32
                self.close(row[3*i],.8*math.cos(c)+.4*math.sin(c)*math.exp(-.5*step),atol=4e-7)
        temps=series(out,"thermal_temperature")
        for i,temp in enumerate(temps):self.close(temp,temps[0]*math.exp(-i),atol=2e-4)

    def test_actual_ou_heat_matches_kinetic_change(self):
        import numpy as np
        r = [(1, 2, 4*i) for i in range(8)]
        masses = np.array([1, 2, 3, 5, 7, 11, 13, 17], dtype=float)
        # Nonzero COM, flow and residual, unequal masses, moving midpoint basis.
        initial = np.array([(.3+.8*math.cos(2*math.pi*x[2]/32),
                             .2*math.sin(i), .1*math.cos(i)) for i,x in enumerate(r)], dtype=np.float32)
        for target in (0, 300):
            out = self.run_case(cosine(), masses=masses, coordinates=r, velocities=initial,
                                viscosity_path="viscosity.h5",
                                settings={"pbc":True, "mode":"nvt", "thermostat":"middle_langevin",
                                          "target_temperature":target, "thermostat_tau":.002,
                                          "dt":.001, "thermostat_seed":13, "step_limit":12})
            velocities = np.concatenate([initial.astype(float)[None],
                                         np.array(out["vel"]).reshape(-1,8,3)])
            kinetic = .5*np.sum(masses[None,:,None]*velocities**2, axis=(1,2))
            heat = np.ravel(out["viscosity"]["thermostat_heat"])
            # With zero forces, only the measured OU operation changes K.
            np.testing.assert_allclose(heat, kinetic-kinetic[0], rtol=2e-13, atol=2e-13)
            np.testing.assert_array_equal(heat, series(out,"thermostat_heat"))
            self.assertEqual(heat[0], 0)
            np.testing.assert_array_equal(out["viscosity_step"], np.arange(13))
            np.testing.assert_allclose(out["viscosity_time"], np.arange(13)*.001, atol=1e-17)
            unit = out["viscosity_units"]["thermostat_heat"]
            if isinstance(unit, bytes): unit = unit.decode()
            self.assertEqual(unit, "kcal/mol")
            if target == 0:
                self.assertLess(heat[-1], 0)
                self.assertTrue(np.all(np.diff(heat) < 0))

    def test_projected_noise_temperature_and_fluctuations(self):
        n=64
        r=[(1+4*(i%8),2,32*(i+.5)/n) for i in range(n)]
        out=self.run_case(cosine(),masses=[12]*n,coordinates=r,
                          settings={"pbc":True,"mode":"nvt","thermostat":"middle_langevin",
                                    "target_temperature":300,"thermostat_tau":1e-5,"dt":1e-5,
                                    "thermostat_seed":9851,"step_limit":1200})
        temps=series(out,"thermal_temperature")[100:]
        self.assertGreater(len(temps),1000)
        self.assertLess(abs(statistics.mean(temps)-300),5)
        variance=statistics.variance(temps)
        expected=2*300**2/(3*n-4)
        self.assertLess(abs(variance/expected-1),.18)
        # Projecting the noise must not inject net translational momentum.
        for axis in "xyz":self.assertLess(max(map(abs,series(out,"bias_com_"+axis))),2e-6)

    def test_integer_time_measurement_does_not_perturb_dynamics(self):
        try:
            import h5py
            import numpy as np
        except ImportError:
            self.skipTest("h5py/numpy needed for H5 verification")
        r = [(1, 2, 4*i+.7) for i in range(8)]
        masses = np.array([1, 2, 3, 5, 7, 11, 13, 17])
        initial = np.array([(.2+.8*math.cos(2*math.pi*x[2]/32), -.1, .05) for x in r])
        for mode in ("nve", "nvt"):
            settings = {"pbc": True, "mode": mode, "dt": .001, "step_limit": 6}
            if mode == "nvt":
                settings.update(thermostat="middle_langevin", target_temperature=300,
                                thermostat_tau=.013, thermostat_seed=1423)
            kwargs = dict(masses=masses, coordinates=r, velocities=initial, settings=settings)
            full = self.run_case(cosine(7)+"bins=8\n", viscosity_path="viscosity.h5", **kwargs)
            sparse = self.run_case(cosine(7)+"bins=5\nsample_interval=2\noutput='custom.h5'\n",
                                   viscosity_path="custom.h5", **kwargs)
            for name in ("frc", "crd", "vel"):
                self.assertEqual(full[name+"_raw"], sparse[name+"_raw"])
            np.testing.assert_array_equal(sparse["viscosity_step"], np.arange(0, 7, 2))
            np.testing.assert_array_equal(sparse["viscosity_time"], full["viscosity_time"][::2])
            if mode == "nvt":
                np.testing.assert_array_equal(sparse["viscosity"]["thermostat_heat"],
                                              full["viscosity"]["thermostat_heat"][::2])
                np.testing.assert_array_equal(full["viscosity"]["thermostat_heat"],
                                              series(full,"thermostat_heat"))
            else:
                self.assertNotIn("thermostat_heat", full["viscosity"])
            for name in ("velocity_fit", "velocity_fourier", "thermal_temperature", "coupling_work"):
                np.testing.assert_array_equal(sparse["viscosity"][name], full["viscosity"][name][::2])
            # Read-only finite-force half kick at q_n, for both deterministic
            # and noisy projected AOA updates. Legacy stored velocities differ.
            h = float(np.float32(np.float32(.001)*np.float32(TIME)))
            for n in range(7):
                q = np.array(r if n == 0 else full["crd"][n-1]).reshape(-1, 3)
                stored = np.array(initial if n == 0 else full["vel"][n-1], dtype=np.float32).reshape(-1, 3).astype(float)
                force = np.array(full["frc"][n]).reshape(-1, 3)
                u = stored+.5*h*force/masses[:, None]
                c = np.cos(2*np.pi*q[:, 2]/32)
                mean_c = np.average(c, weights=masses)
                com = np.average(u, weights=masses, axis=0)
                variance = np.average((c-mean_c)**2, weights=masses)
                fit = np.average((c-mean_c)*(u[:, 0]-com[0]), weights=masses)/variance
                values = full["viscosity"]
                # Positions in native binary trajectories are wrapped/rounded;
                # profile samples see the current internal coordinate instead.
                self.close(values["velocity_fit"][n], fit*TIME/10, atol=3e-6)
                self.close(values["velocity_fourier"][n], 2*np.average(c*u[:, 0], weights=masses)*TIME/10, atol=3e-6)
                self.close(sum(values[f"bin_{b}_mass"][n] for b in range(8)), masses.sum())
                self.close(sum(values[f"bin_{b}_momentum"][n] for b in range(8)), np.sum(masses*u[:, 0])*TIME/10, atol=3e-6)
                self.close(values["coupling_work"][n], full["work"][n], atol=1e-9)
            self.close(full["viscosity"]["velocity_fit"][0]-series(full, "bias_amplitude")[0], .5*.001*7, atol=5e-7)

    def test_invalid_and_incompatible_inputs(self):
        kwargs=dict(masses=[12]*4,coordinates=[(1,2,z) for z in (0,8,16,24)],settings={"pbc":True})
        for text,error in [(cosine()+"extra=1\n","unknown workflow"),
                           (cosine()+"bins=3\n","integer range"),
                           (cosine()+"bins=4.5\n","integer"),
                           (cosine()+"sample_interval=0\n","integer range"),
                           (cosine().replace('gradient="z"','gradient="x"'),"must differ"),
                           (cosine().replace('acceleration=0','acceleration=nan'),"finite"),
                           ('[viscosity]\nmethod="cosine"',"required"),
]:
            with self.subTest(error=error):self.run_case(text,error=error,**kwargs)
        self.run_case(cosine(),masses=[12]*4,error="periodic orthogonal")

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--sponge",required=True)
    args,extra=parser.parse_known_args();base.SPONGE=str(Path(args.sponge).resolve())
    unittest.main(argv=[__file__,*extra])

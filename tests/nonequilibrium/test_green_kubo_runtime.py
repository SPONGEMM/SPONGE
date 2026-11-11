"""Real executable stress checks, not a converged liquid viscosity benchmark."""
import argparse
import json
import itertools
import math
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.viscosity.workflows.config import write_engine_input
import subprocess
import sys
import tempfile
import unittest

try:
    import h5py
    import numpy as np
except ImportError:
    raise SystemExit(77)

SPONGE = None
PA = 4184 / 6.02214076e23 / 1e-30
TIME = float(np.float32(20.455))
AXES = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
NAMES = ("pxx", "pyy", "pzz", "pxy", "pxz", "pyz")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"scripts"))
import analyze_green_kubo as gk
import analyze_rnemd as rn


class GreenKuboRuntime(unittest.TestCase):
    def run_case(self, masses=(1, 2, 3), velocities=None, positions=None, charges=None,
                 options=None, method='method="green_kubo"', files=None,
                 error=None, enabled=True, analyze=False):
        with tempfile.TemporaryDirectory(prefix="sponge-gk-") as tmp:
            path = Path(tmp)
            n = len(masses)
            velocities = velocities if velocities is not None else [(1, 2, 3), (3, -2, 1), (-1, 1, -2)][:n]
            positions = positions if positions is not None else [(1 + 5*i, 2, 3) for i in range(n)]
            for name, values in (("mass", masses), ("charge", charges if charges is not None else [0]*n)):
                (path/f"{name}.txt").write_text(str(n)+"\n"+"\n".join(map(str, values)))
            for name, values in (("coordinate", positions), ("velocity", velocities)):
                tail = "\n32 32 32\n90 90 90\n" if name == "coordinate" else "\n"
                (path/f"{name}.txt").write_text(str(n)+"\n"+"\n".join(" ".join(map(str, row)) for row in values)+tail)
            for name, value in (files or {}).items():
                (path/name).write_text(value)
            config = dict(md_name="gk_test", mode="nve", pbc=True, dt=.0001,
                          step_limit=8, cutoff=8, print_zeroth_frame=True,
                          write_mdout_interval=7, write_information_interval=7,
                          write_trajectory_interval=1, write_restart_file_interval=0,
                          crd="crd.dat", vel="vel.dat", frc="frc.dat")
            config["PM.MPI_size"] = 1
            config.update({f"{name}_in_file": f"{name}.txt" for name in ("mass", "charge", "coordinate", "velocity")})
            config.update(options or {})
            try:
                write_engine_input(path/"mdin.spg.toml", "\n".join(f"{k}={json.dumps(v)}" for k, v in config.items()) +
                                               ("\n[viscosity]\n"+method if enabled else ""))
            except ValueError as failure:
                if not error: raise
                self.assertIn(error,str(failure))
                return
            run = subprocess.run([SPONGE, "-mdin", "mdin.spg.toml"], cwd=path,
                                 env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=45)
            if error:
                self.assertNotEqual(run.returncode, 0, run.stdout)
                self.assertIn(error, run.stdout)
                if files and "viscosity.h5" in files:
                    self.assertEqual((path/"viscosity.h5").read_text(), files["viscosity.h5"])
                return
            self.assertEqual(run.returncode, 0, run.stdout)
            result = {name: (path/f"{name}.dat").read_bytes() for name in ("crd", "vel", "frc")}
            if enabled:
                with h5py.File(path/"viscosity.h5") as file:
                    result["metadata"] = {k: gk.text(v[()]) for k, v in file["parameters/sponge/provenance"].items()}
                    result["time"] = file["observables/all/time"][()]
                    result["step"] = file["observables/all/step"][()]
                    result["values"] = {name: node["value"][()] for name, node in file["observables/all"].items()
                                        if isinstance(node, h5py.Group) and "value" in node}
                    if "pxy" in result["values"]:
                        self.assertEqual(file["observables/all/pxy/value"].attrs["unit"], "Pa")
                    else:
                        self.assertEqual(file["observables/all/momentum/value"].attrs["unit"], "Da nm/ps")
                    self.assertEqual(file["observables/all/time"].attrs["unit"], "ps")
                if "constraint_observer" in result["metadata"] and len(result["step"]) >= 4:
                    gk.load_stress(path/"viscosity.h5")
                if "pxy" not in result["values"]:
                    profile, _ = rn.load_profile(path/"viscosity.h5")
                    np.testing.assert_array_equal(profile["momentum"], result["values"]["momentum"])
                if analyze:
                    args = gk.load_stress(path/"viscosity.h5")
                    self.assertEqual(len(args[0]), 9)
                    tool = Path(__file__).resolve().parents[2]/"scripts/analyze_green_kubo.py"
                    analysis = subprocess.run([sys.executable, str(tool), str(path/"viscosity.h5"),
                                               "--start-ps", "0", "--window-ps", ".0001", "--block-ps", ".0003",
                                               "--output", str(path/"analysis")], capture_output=True, text=True)
                    self.assertEqual(analysis.returncode, 0, analysis.stderr)
                    report = json.loads((path/"analysis/summary.json").read_text())
                    expected = np.mean([result["values"][name][0]**2 for name in ("pxy", "pxz", "pyz")])
                    expected *= 32**3*1e-30/(gk.KB*result["values"]["temperature"][0])*.0001e-12
                    np.testing.assert_allclose(report["viscosity_pa_s"], expected, rtol=1e-12)
                    self.assertEqual(report["status"], "diagnostic_only")
                    with h5py.File(path/"viscosity.h5", "r+") as file:
                        file["parameters/sponge/output/frame_count"][0] = 999
                    with self.assertRaisesRegex(ValueError, "completion"):
                        gk.load_stress(path/"viscosity.h5")
                    with h5py.File(path/"viscosity.h5", "r+") as file:
                        file["parameters/sponge/output/frame_count"][0] = 0
                        file["parameters/sponge/output/status"][()] = "failed"
                    with self.assertRaisesRegex(ValueError, "not finalized"):
                        gk.load_stress(path/"viscosity.h5")
                    with h5py.File(path/"viscosity.h5", "r+") as file:
                        file["parameters/sponge/output/status"][()] = "finalized"
                        file["observables/all/pxy/value"].attrs["unit"] = "bar"
                    with self.assertRaisesRegex(ValueError, "unit"):
                        gk.load_stress(path/"viscosity.h5")
            return result

    def test_analysis_cli_and_contract(self):
        self.run_case(analyze=True)

    def test_ideal_stress_units_com_and_independent_cadence(self):
        out = self.run_case(method='method="green_kubo"\nsample_interval=2')
        np.testing.assert_array_equal(out["step"], [0, 2, 4, 6, 8])
        np.testing.assert_allclose(out["time"], np.arange(0, 9, 2)*.0001, rtol=1e-14)
        mass = np.array([1, 2, 3])
        vel = np.array([[1, 2, 3], [3, -2, 1], [-1, 1, -2]])
        com = np.average(vel, axis=0, weights=mass)
        c = vel-com
        tensor = (c.T*mass)@c
        for name, (a, b) in zip(NAMES, AXES):
            np.testing.assert_allclose(out["values"][name], tensor[a, b]/32**3*PA, rtol=3e-7)
        np.testing.assert_allclose(out["values"]["temperature"], np.trace(tensor)/(6*float(np.float32(.00198716))), rtol=3e-7)
        for j in range(3):
            np.testing.assert_allclose(out["values"]["com_v"+"xyz"[j]], com[j]*TIME/10, rtol=3e-7)
        np.testing.assert_allclose(out["values"]["volume"], 32**3/1000)
        baseline = self.run_case(enabled=False)
        for key in ("crd", "vel", "frc"):
            self.assertEqual(out[key], baseline[key])

    def test_lj_virial_and_centered_velocity(self):
        # U(r)=A/r^12-B/r^6; analytic virial F_ij tensor r_ij.
        d = np.array([1., 2., 1.])
        r2 = d@d
        factor = 12*1000/r2**7-6*10/r2**4
        force = np.array([-factor*d, factor*d])
        mass = np.array([2., 4.])
        vel = np.array([[.1, -.2, .3], [-.3, .1, .2]], dtype=np.float32).astype(float)
        for dt in (0, .001):
            out = self.run_case(masses=mass, velocities=vel, positions=[(1, 1, 1), (2, 3, 2)],
                                options={"LJ_in_file": "lj.txt", "dt": dt, "step_limit": 1},
                                files={"lj.txt": "2 1\n1000\n10\n0\n0\n"})
            actual_force = np.frombuffer(out["frc"], dtype=np.float32).reshape(-1, 2, 3)[0]
            np.testing.assert_allclose(actual_force, force, rtol=2e-5, atol=1e-7)
            centered = vel + .5*float(np.float32(dt)*np.float32(TIME))*force/mass[:, None]
            centered -= np.average(centered, axis=0, weights=mass)
            expected = ((centered.T*mass)@centered + factor*np.outer(d, d))/32**3*PA
            # SPONGE's existing dispersion tail adds diagonal pressure only.
            expected -= np.eye(3)*(4*np.pi/3*len(mass)**2*10/8**3/32**6*PA)
            for name, (a, b) in zip(NAMES, AXES):
                np.testing.assert_allclose(out["values"][name][0], expected[a, b], rtol=2e-5, atol=2)

    def test_constrained_mixed_topology_cadence_and_trajectory(self):
        # SETTLE consumes the first triangle; SHAKE handles a separate chain.
        # Independent dense NumPy solve verifies the complete captured topology,
        # all six physical stress entries and the temperature DOF at force time.
        mass = np.array([16., 1., 1., 2., 3., 4., 5.])
        initial = np.array([[5, 5, 5], [5.6, 5.8, 5], [4.4, 5.8, 5],
                            [12, 12, 12], [13, 12, 12], [13, 13, 12], [13, 13, 13]], dtype=np.float32).astype(float)
        velocity = np.random.default_rng(172).normal(size=(7, 3)).astype(np.float32).astype(float)*.1
        velocity = velocity.astype(np.float32).astype(float)
        pairs = [(0, 1, 1), (0, 2, 1), (1, 2, 1.2), (3, 4, 1), (4, 5, 1), (5, 6, 1)]
        # Interleave the molecules in global input order. Single-domain DD
        # groups connected atoms, so the observer must gather local->global.
        order = np.array([0, 3, 1, 4, 2, 5, 6])
        inverse = np.argsort(order)
        mass, initial, velocity = mass[order], initial[order], velocity[order]
        pairs = [(inverse[a], inverse[b], r) for a, b, r in pairs]
        files = {"constraint.txt": "6\n"+"\n".join(f"{a} {b} {r}" for a, b, r in pairs)+"\n"}
        options = {"constrain_mode": "SHAKE", "constrain.in_file": "constraint.txt",
                   "SHAKE.iteration_numbers": 100, "SHAKE.tolerance": 1e-6}
        args = dict(masses=mass, positions=initial, velocities=velocity, files=files, options=options)
        dense = self.run_case(**args)
        sparse = self.run_case(**args, method='method="green_kubo"\nsample_interval=2')
        off = self.run_case(**args, enabled=False)
        for name in ("crd", "vel", "frc"):
            self.assertEqual(dense[name], sparse[name])
            self.assertEqual(dense[name], off[name])
        for name in dense["values"]:
            np.testing.assert_array_equal(dense["values"][name][::2], sparse["values"][name])
        meta = dense["metadata"]
        self.assertEqual(meta["constraint_count"], "6")
        self.assertEqual(meta["thermal_dof"], "12")
        self.assertIn("settle_triangles=1|shake_pairs=3", meta["constraint_solver"])
        q = np.concatenate(([initial], np.frombuffer(dense["crd"], np.float32).reshape(-1, 7, 3)))
        vel = np.concatenate(([velocity], np.frombuffer(dense["vel"], np.float32).reshape(-1, 7, 3)))
        for k, (position, half) in enumerate(zip(q, vel)):
            jac = np.zeros((6, 21))
            displacements = []
            for row, (a, b, _) in enumerate(pairs):
                d = position[a]-position[b]
                d -= 32*np.floor(d/32+.5)
                e = d/np.linalg.norm(d)
                displacements.append(d)
                jac[row, 3*a:3*a+3] = e
                jac[row, 3*b:3*b+3] = -e
            inv = 1/np.repeat(mass, 3)
            metric = (jac*inv)@jac.T
            u = half.ravel()+inv*(jac.T@np.linalg.solve(metric, -jac@half.ravel()))
            u = u.reshape(7, 3)
            curvature = [((u[a]-u[b])@(u[a]-u[b]) - ((u[a]-u[b])@(d/np.linalg.norm(d)))**2)/np.linalg.norm(d)
                         for (a, b, _), d in zip(pairs, displacements)]
            multipliers = np.linalg.solve(metric, -np.array(curvature))
            virial = sum(lam*np.outer(d, d)/np.linalg.norm(d) for lam, d in zip(multipliers, displacements))
            c = u-np.average(u, axis=0, weights=mass)
            kinetic = (c.T*mass)@c
            pressure = (kinetic+virial)/32**3*PA
            for name, (a, b) in zip(NAMES, AXES):
                np.testing.assert_allclose(dense["values"][name][k], pressure[a, b], rtol=2e-6, atol=.05)
                np.testing.assert_allclose(dense["values"]["constraint_w"+name[1:]][k], virial[a, b], rtol=2e-6, atol=1e-7)
            np.testing.assert_allclose(dense["values"]["temperature"][k], np.trace(kinetic)/(12*float(np.float32(.00198716))), rtol=2e-6)

    def test_constraint_force_cancels_internal_central_lj_stress(self):
        # A distance constraint cancels an internal central pair force/virial.
        # This also tests the projected half kick at a nonzero time step.
        args = dict(masses=(2, 4), velocities=[(.1, -.2, .3), (-.3, .1, .2)],
                    positions=[(1, 1, 1), (2, 3, 2)])
        files = {"constraint.txt": f"1\n0 1 {np.sqrt(6):.17g}\n"}
        for algorithm in ("SETTLE", "SHAKE"):
            options = {"constrain_mode": algorithm, "settle_disable": algorithm == "SHAKE",
                       "constrain.in_file": "constraint.txt", "SHAKE.tolerance": 1e-6}
            free = self.run_case(**args, options=options, files=files)
            interacting = self.run_case(**args, options={**options, "LJ_in_file": "lj.txt"},
                                        files={**files, "lj.txt": "2 1\n1000\n10\n0\n0\n"})
            self.assertEqual(interacting["metadata"]["thermal_dof"], "2")
            tail = 4*np.pi/3*4*10/8**3/32**6*PA
            for name in NAMES:
                expected = free["values"][name][0] - (tail if name in NAMES[:3] else 0)
                np.testing.assert_allclose(interacting["values"][name][0], expected, rtol=3e-5, atol=1)

    def test_constrained_scope_rejected(self):
        base = {"constrain_mode": "SHAKE", "settle_disable": True,
                "constrain.in_file": "constraint.txt"}
        files = {"constraint.txt": "1\n0 1 5\n"}
        self.run_case(options={**base, "dt": 0}, files=files, error="positive dt")
        self.run_case(options={**base, "constrain_mode": "LINCS"}, files=files, error="SETTLE/SHAKE/RATTLE only")
        self.run_case(options={**base, "constrain_mode": "SETTLE"}, files=files, error="not handled by SETTLE")
        self.run_case(options=base, files={"constraint.txt": "1\n0 1 4\n"}, error="length residual")

    def test_rejections_and_existing_output_preserved(self):
        self.run_case(files={"viscosity.h5": "protected data"}, error="already exists")
        self.run_case(method='method="green_kubo"\nsample_interval=0', error="integer range")
        self.run_case(method='method="green_kubo"\nacceleration=1', error="unknown workflow")
        self.run_case(options={"mode": "nvt", "thermostat": "middle_langevin"})
        self.run_case(charges=(1, 0, 0), error="neutral fixed cells")
        self.run_case(charges=(1, -1, 0), error="active direct and reciprocal")

    def test_harmonic_bond_virial(self):
        d = np.array([1., 2., 1.])
        factor = -2*2*(np.linalg.norm(d)-2)/np.linalg.norm(d)
        out = self.run_case(masses=(2, 4), velocities=[(0, 0, 0)]*2,
                            positions=[(1, 1, 1), (2, 3, 2)],
                            options={"bond_in_file": "bond.txt", "dt": 0, "step_limit": 1},
                            files={"bond.txt": "1\n0 1 2 2\n"})
        expected = factor*np.outer(d, d)/32**3*PA
        for name, (a, b) in zip(NAMES, AXES):
            np.testing.assert_allclose(out["values"][name][0], expected[a, b], rtol=2e-5)

    def test_pme_stress_against_direct_ewald_energy_strain(self):
        # Independent double-precision Ewald sums, tin-foil boundary, neutral
        # two-charge cell. No SPONGE force or stress enters the reference.
        q = np.array([1., -1.])*float(np.float32(18.2223))
        initial = np.array([[1., 2., 3.], [4., 6., 5.]])
        images = np.array(list(itertools.product(range(-1, 2), repeat=3)))
        modes = np.array([n for n in itertools.product(range(-16, 17), repeat=3) if n != (0, 0, 0)])

        def energy(deformation, source=initial):
            alpha = .25
            # Columns of cell are lattice vectors: both positions and image
            # vectors undergo F; reciprocal row vectors use n @ cell^-1.
            cell = deformation*32
            positions = source@deformation.T
            value = -alpha/math.sqrt(math.pi)*np.sum(q*q)
            for i in range(2):
                for j in range(2):
                    distances = np.linalg.norm(positions[i]-positions[j]+images@cell.T, axis=1)
                    for distance in distances[distances > 0]:
                        value += .5*q[i]*q[j]*math.erfc(alpha*distance)/distance
            wave = 2*np.pi*modes@np.linalg.inv(cell)
            k2 = (wave*wave).sum(axis=1)
            rho = np.exp(1j*wave@positions.T)@q
            value += 2*np.pi/np.linalg.det(cell)*np.sum(np.exp(-k2/(4*alpha**2))/k2*abs(rho)**2)
            return value

        references = []
        for h in (1e-4, 5e-5):
            pressure = []
            for a, b in AXES:
                plus, minus = np.eye(3), np.eye(3)
                plus[a, b] += h
                minus[a, b] -= h
                pressure.append(-(energy(plus)-energy(minus))/(2*h)/32**3*PA)
            references.append(pressure)
        np.testing.assert_allclose(references[0], references[1], rtol=2e-7)
        for grid in (64, 63):
            out = self.run_case(masses=(1, 1), charges=q, positions=initial, velocities=[(0, 0, 0)]*2,
                                options={"dt": 0, "step_limit": 1, "PM.fftx": grid, "PM.ffty": grid, "PM.fftz": grid,
                                         "LJ_in_file": "lj.txt"}, files={"lj.txt": "2 1\n0\n0\n0\n0\n"})
            measured = [out["values"][name][0] for name in NAMES]
            np.testing.assert_allclose(measured, references[1], rtol=2e-4, atol=1)

        # Add a rigid distance to the same neutral PME system. Independent
        # finite-displacement Ewald forces and strain derivatives supply the
        # reference; no SPONGE force enters the expected constrained stress.
        force = np.zeros((2, 3))
        for axis in range(3):
            plus, minus = initial.copy(), initial.copy()
            plus[0, axis] += 1e-5
            minus[0, axis] -= 1e-5
            force[0, axis] = -(energy(np.eye(3), plus)-energy(np.eye(3), minus))/(2e-5)
        force[1] = -force[0]
        virial = np.zeros((3, 3))
        for value, (a, b) in zip(references[1], AXES):
            virial[a, b] = virial[b, a] = value*32**3/PA
        mass = np.array([1., 2.])
        velocity = np.array([[.2, .1, -.1], [-.05, .3, .2]], dtype=np.float32).astype(float)
        d = initial[0]-initial[1]
        r = np.linalg.norm(d)
        unit = d/r
        h = float(np.float32(.0001)*np.float32(TIME))
        for excluded in (False, True):
            expected_force = force.copy()
            expected_virial = virial.copy()
            if excluded:
                central = q[0]*q[1]*d/r**3
                expected_force -= np.array([central, -central])
                expected_virial -= q[0]*q[1]*np.outer(d, d)/r**3
            u = velocity + .5*h*expected_force/mass[:, None]
            beta = -unit@(u[0]-u[1])/(1/mass).sum()
            u += np.array([beta*unit/mass[0], -beta*unit/mass[1]])
            relative = u[0]-u[1]
            curvature = (relative@relative-(unit@relative)**2)/r
            lam = -(unit@(expected_force[0]/mass[0]-expected_force[1]/mass[1])+curvature)/(1/mass).sum()
            constraint = lam*np.outer(d, d)/r
            c = u-np.average(u, axis=0, weights=mass)
            expected = ((c.T*mass)@c+expected_virial+constraint)/32**3*PA
            for algorithm in ("SETTLE", "SHAKE"):
                options = {"dt": .0001, "step_limit": 4, "PM.fftx": 128, "PM.ffty": 128, "PM.fftz": 128,
                           "constrain_mode": algorithm, "settle_disable": algorithm == "SHAKE",
                           "constrain.in_file": "constraint.txt", "LJ_in_file": "lj.txt"}
                files = {"constraint.txt": f"1\n0 1 {r:.17g}\n", "lj.txt": "2 1\n0\n0\n0\n0\n"}
                if excluded:
                    files["exclude.txt"] = "2 1\n1 1\n0\n"
                    options["exclude_in_file"] = "exclude.txt"
                out = self.run_case(masses=mass, charges=q, positions=initial, velocities=velocity,
                                    options=options, files=files)
                measured_force = np.frombuffer(out["frc"], np.float32).reshape(-1, 2, 3)[0]
                np.testing.assert_allclose(measured_force, expected_force, rtol=3e-4, atol=2e-5)
                # Bound the six-component tensor error, using the same 2e-4
                # relative scale as the unconstrained PME check above. A small
                # individual component can result from cancellation of much
                # larger kinetic/electrostatic/constraint terms in float32.
                measured = np.array([out["values"][name][0] for name in NAMES])
                reference = np.array([expected[a, b] for a, b in AXES])
                self.assertLessEqual(np.max(abs(measured-reference)), 2+2e-4*np.max(abs(reference)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sponge", required=True)
    args, extra = parser.parse_known_args()
    SPONGE = str(Path(args.sponge).resolve())
    unittest.main(argv=[__file__, *extra])

#!/usr/bin/env python3
"""Frozen truncated LJ inputs and independent static cross-engine verification.

This gate checks the Hamiltonian/units, not liquid equilibration or viscosity.
All runs and evidence go to a new explicit directory outside the source tree.
"""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.viscosity.workflows.config import write_engine_input
import subprocess
import time

import h5py
import numpy as np

PA = 4184/6.02214076e23/1e-30
ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            result.update(chunk)
    return result.hexdigest()


def pair_reference(positions, box, c12, c6, cutoff):
    """Direct double pair sum; unshifted energy and unshifted truncated forces."""
    force = np.zeros_like(positions, dtype=float)
    virial = np.zeros((3, 3))
    energy = 0.
    nearest_cutoff = float("inf")
    for i in range(len(positions)-1):
        delta = positions[i]-positions[i+1:]
        delta -= box*np.rint(delta/box)
        r2 = np.sum(delta*delta, axis=1)
        if np.any(r2 == 0):
            raise ValueError("overlapping particles")
        nearest_cutoff = min(nearest_cutoff, float(np.min(np.abs(np.sqrt(r2)-cutoff))))
        active = r2 < cutoff**2
        r6 = r2[active]**-3
        coefficient = (12*c12*r6**2-6*c6*r6)/r2[active]
        pairs = coefficient[:, None]*delta[active]
        force[i] += pairs.sum(axis=0)
        force[np.arange(i+1, len(positions))[active]] -= pairs
        virial += pairs.T@delta[active]
        energy += np.sum(c12*r6**2-c6*r6)
    return energy, force, virial, nearest_cutoff


def prepare(directory, cells=3, seed=20261027, density=.8, temperature=1.,
            epsilon=.238, sigma=3.4, mass=39.948, cutoff=2.5, dt=.002):
    if cells < 3 or not all(np.isfinite(x) and x > 0 for x in (density, temperature, epsilon, sigma, mass, cutoff, dt)):
        raise ValueError("cells >= 3 and finite positive model parameters required")
    n = 4*cells**3
    box = float(np.float32(sigma*(n/density)**(1/3)))
    rc = float(np.float32(cutoff*sigma))
    if 2*rc >= box:
        raise ValueError("cutoff must be smaller than half box length")
    basis = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
    positions = np.concatenate([(basis+cell)*box/cells for cell in itertools.product(range(cells), repeat=3)])
    rng = np.random.default_rng(seed)
    # Break crystal symmetry for a nontrivial six-component virial check.
    positions = ((positions+rng.normal(0, .025*sigma, positions.shape)) % box).astype(np.float32).astype(float)
    masses = np.full(n, np.float32(mass), dtype=float)
    velocities = rng.normal(size=(n, 3))*np.sqrt(epsilon*temperature/masses[:, None])
    velocities -= np.average(velocities, axis=0, weights=masses)
    velocities *= np.sqrt((3*n-3)*epsilon*temperature/np.sum(masses[:, None]*velocities**2))
    velocities = velocities.astype(np.float32).astype(float)
    c12, c6 = 4*epsilon*sigma**12, 4*epsilon*sigma**6
    tau_ps = sigma*np.sqrt(float(np.float32(mass))/epsilon)/float(np.float32(20.455))
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    model = dict(contract="sponge.lj.transport-model.v1", atoms=n, cells=cells, seed=seed,
                 epsilon_kcal_mol=epsilon, sigma_angstrom=sigma, mass_da=float(masses[0]),
                 box_angstrom=box, cutoff_angstrom=rc, requested_density_star=density,
                 actual_density_star=n/(box/sigma)**3, temperature_star=temperature,
                 c12=c12, c6=c6, dt_star=dt, dt_ps=dt*tau_ps, tau_ps=tau_ps,
                 target_temperature_k=epsilon*temperature/float(np.float32(.00198716)),
                 force="unshifted truncated LJ; minimum image; no Coulomb",
                 sponge_tail="C6 only, N^2 convention, energy and diagonal pressure; no force",
                 lammps_tail="off; compare direct terms then account separately for SPONGE C6 tail",
                 initialization="jittered FCC and Gaussian COM-free velocities; not equilibrated liquid")
    np.savez(directory/"state.npz", positions=positions, velocities=velocities, masses=masses)
    for name, values in (("mass", masses), ("charge", np.zeros(n))):
        (directory/f"{name}.txt").write_text(f"{n}\n"+"\n".join(format(x, ".17g") for x in values)+"\n")
    for name, values in (("coordinate", positions), ("velocity", velocities), ("zero_velocity", np.zeros_like(velocities))):
        text = f"{n}\n"+"\n".join(" ".join(format(x, ".17g") for x in row) for row in values)+"\n"
        if name == "coordinate": text += f"{box:.17g} {box:.17g} {box:.17g}\n90 90 90\n"
        (directory/f"{name}.txt").write_text(text)
    (directory/"lj.txt").write_text(f"{n} 1\n{c12:.17g}\n{c6:.17g}\n"+"0\n"*n)
    lines = ["Frozen LJ cross-engine configuration", "", f"{n} atoms", "1 atom types", "",
             f"0 {box:.17g} xlo xhi", f"0 {box:.17g} ylo yhi", f"0 {box:.17g} zlo zhi", "", "Masses", "", f"1 {masses[0]:.17g}", "", "Atoms # atomic", ""]
    lines.extend(f"{i+1} 1 "+" ".join(format(x, ".17g") for x in row) for i, row in enumerate(positions))
    (directory/"lammps.data").write_text("\n".join(lines)+"\n")
    model["input_sha256"] = {p.name: digest(p) for p in sorted(directory.iterdir()) if p.is_file()}
    (directory/"model.json").write_text(json.dumps(model, indent=2)+"\n")
    return model


def execute(command, directory, timeout=120):
    start = time.perf_counter()
    with (directory/"run.log").open("w") as log:
        result = subprocess.run(list(map(str, command)), cwd=directory,
                                env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                                stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    elapsed = time.perf_counter()-start
    if result.returncode:
        raise RuntimeError(f"run failed ({result.returncode}); inspect {directory/'run.log'}")
    return elapsed


def static_check(model_dir, output, sponge, lammps):
    model_dir, output = Path(model_dir).resolve(), Path(output).resolve()
    model = json.loads((model_dir/"model.json").read_text())
    for name, expected in model["input_sha256"].items():
        if digest(model_dir/name) != expected: raise ValueError(f"model input hash mismatch: {name}")
    output.mkdir(parents=True, exist_ok=False)
    sp, lm = output/"sponge", output/"lammps"
    sp.mkdir(); lm.mkdir()
    config = dict(md_name="lj_static_gate", mode="nve", pbc=True, dt=0, step_limit=1,
                  cutoff=model["cutoff_angstrom"], skin=.3*model["sigma_angstrom"],
                  print_zeroth_frame=True, write_mdout_interval=1, write_information_interval=1,
                  write_trajectory_interval=1, write_restart_file_interval=0, frc="force.dat")
    config["PM.MPI_size"] = 1
    for name in ("mass", "charge", "coordinate"):
        config[f"{name}_in_file"] = str(model_dir/f"{name}.txt")
    config["velocity_in_file"] = str(model_dir/"zero_velocity.txt")
    config["LJ_in_file"] = str(model_dir/"lj.txt")
    write_engine_input(sp/"mdin.spg.toml", "\n".join(f"{k}={json.dumps(v)}" for k, v in config.items())+'\n[viscosity]\nmethod="green_kubo"\n')
    (lm/"in.lammps").write_text(f'''units real
atom_style atomic
boundary p p p
read_data {model_dir/'lammps.data'}
pair_style lj/cut {model['cutoff_angstrom']:.17g}
pair_coeff 1 1 {model['epsilon_kcal_mol']:.17g} {model['sigma_angstrom']:.17g}
pair_modify shift no tail no
neighbor {model['sigma_angstrom']*.3:.17g} bin
neigh_modify every 1 delay 0 check yes
thermo_style custom step pe pxx pyy pzz pxy pxz pyz
thermo_modify norm no format float %.17g
run 0
write_dump all custom forces.dump id fx fy fz modify sort id format float %.17g
print "$(pe:%.17g) $(pxx:%.17g) $(pyy:%.17g) $(pzz:%.17g) $(pxy:%.17g) $(pxz:%.17g) $(pyz:%.17g)" file static.dat
''')
    elapsed = {"sponge": execute([sponge, "-mdin", "mdin.spg.toml"], sp),
               "lammps": execute([lammps, "-in", "in.lammps"], lm)}
    state = np.load(model_dir/"state.npz")
    energy, force, virial, distance = pair_reference(state["positions"], model["box_angstrom"], model["c12"], model["c6"], model["cutoff_angstrom"])
    volume = model["box_angstrom"]**3
    tail_energy = -2*np.pi/3*model["atoms"]**2*model["c6"]/model["cutoff_angstrom"]**3/volume
    tail_pressure = 2*tail_energy/volume*PA
    axes = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
    pressure = np.array([virial[a, b]/volume*PA for a, b in axes])
    sp_force = np.fromfile(sp/"force.dat", dtype=np.float32).reshape(-1, model["atoms"], 3)[0]
    with h5py.File(sp/"viscosity.h5") as file:
        sp_pressure = np.array([file[f"observables/all/{name}/value"][0] for name in ("pxx", "pyy", "pzz", "pxy", "pxz", "pyz")])
    lm_values = np.loadtxt(lm/"static.dat")
    lm_force = np.loadtxt(lm/"forces.dump", skiprows=9)[:, 1:]
    mdout = np.genfromtxt(sp/"mdout.txt", names=True)
    sp_energy = float(np.atleast_1d(mdout["eff_pot"])[0])
    sp_expected = pressure.copy(); sp_expected[:3] += tail_pressure
    # Tight double-engine check and a separate bound for SPONGE float kernels.
    checks = {
        "lammps_force_vs_pair": np.allclose(lm_force, force, rtol=1e-10, atol=1e-10),
        "lammps_energy_vs_pair": np.isclose(lm_values[0], energy, rtol=1e-11, atol=1e-9),
        "lammps_pressure_vs_pair": np.allclose(lm_values[1:]*101325, pressure, rtol=5e-7, atol=.1),
        "sponge_force_vs_pair": np.allclose(sp_force, force, rtol=2e-4, atol=2e-5),
        "sponge_energy_with_c6_tail": np.isclose(sp_energy, energy+tail_energy, rtol=2e-6, atol=2e-5),
        "sponge_pressure_with_c6_tail": np.allclose(sp_pressure, sp_expected, rtol=2e-4, atol=20),
    }
    report = dict(status="passed" if all(checks.values()) else "failed", scope="static Hamiltonian/force/virial gate only; not transport validation",
                  checks={k: bool(v) for k, v in checks.items()}, model_sha256=digest(model_dir/"model.json"),
                  benchmark_script_sha256=digest(__file__),
                  source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  sponge_executable_sha256=digest(sponge), lammps_executable_sha256=digest(lammps),
                  lammps_version_line=next((line for line in (lm/"run.log").read_text().splitlines() if line.startswith("LAMMPS (")), "unknown"),
                  wall_seconds=elapsed, reference_energy_kcal_mol=float(energy),
                  sponge_energy_kcal_mol=sp_energy,
                  c6_tail_energy_kcal_mol=float(tail_energy), c6_tail_diagonal_pressure_pa=float(tail_pressure),
                  reference_pressure_pa=pressure.tolist(), sponge_pressure_pa=sp_pressure.tolist(), lammps_pressure_pa=(lm_values[1:]*101325).tolist(),
                  sponge_force_max_error=float(np.max(np.abs(sp_force-force))), lammps_force_max_error=float(np.max(np.abs(lm_force-force))),
                  nearest_pair_to_cutoff_angstrom=distance)
    (output/"summary.json").write_text(json.dumps(report, indent=2)+"\n")
    if not all(checks.values()): raise RuntimeError(f"static gate failed; inspect {output/'summary.json'}")
    return report


def dynamics_check(model_dir, output, sponge, lammps, steps=100):
    """Match leapfrog half-step and velocity-Verlet integer-step states explicitly."""
    model_dir, output = Path(model_dir).resolve(), Path(output).resolve()
    model = json.loads((model_dir/"model.json").read_text())
    for name, expected in model["input_sha256"].items():
        if digest(model_dir/name) != expected: raise ValueError(f"model input hash mismatch: {name}")
    if not 1 <= steps <= 1000: raise ValueError("short propagation gate requires 1..1000 steps")
    output.mkdir(parents=True, exist_ok=False)
    sp, lm = output/"sponge", output/"lammps"
    sp.mkdir(); lm.mkdir()
    state = np.load(model_dir/"state.npz")
    _, force, _, _ = pair_reference(state["positions"], model["box_angstrom"], model["c12"], model["c6"], model["cutoff_angstrom"])
    dt_internal = float(np.float32(np.float32(model["dt_ps"])*np.float32(20.455)))
    # Frozen LAMMPS 29Aug2024 real-unit mvv2e=48.88821291^2.
    # Align the actual integration increment, not rounded nominal time units.
    lammps_time = 48.88821291
    centered = state["velocities"]+.5*dt_internal*force/state["masses"][:, None]
    data = (model_dir/"lammps.data").read_text()+"\nVelocities\n\n"
    data += "\n".join(f"{i+1} "+" ".join(format(x/lammps_time, ".17g") for x in row) for i, row in enumerate(centered))+"\n"
    (lm/"model.data").write_text(data)
    config = dict(md_name="lj_nve_gate", mode="nve", pbc=True, dt=model["dt_ps"], step_limit=steps,
                  cutoff=model["cutoff_angstrom"], skin=.3*model["sigma_angstrom"], print_zeroth_frame=True,
                  write_mdout_interval=10, write_information_interval=10, write_trajectory_interval=1,
                  write_restart_file_interval=0, crd="crd.dat", vel="vel.dat", frc="force.dat")
    config["PM.MPI_size"] = 1
    for name in ("mass", "charge", "coordinate", "velocity"):
        config[f"{name}_in_file"] = str(model_dir/f"{name}.txt")
    config["LJ_in_file"] = str(model_dir/"lj.txt")
    write_engine_input(sp/"mdin.spg.toml", "\n".join(f"{k}={json.dumps(v)}" for k, v in config.items())+'\n[viscosity]\nmethod="green_kubo"\n')
    (lm/"in.lammps").write_text(f'''units real
atom_style atomic
boundary p p p
read_data model.data
pair_style lj/cut {model['cutoff_angstrom']:.17g}
pair_coeff 1 1 {model['epsilon_kcal_mol']:.17g} {model['sigma_angstrom']:.17g}
pair_modify shift no tail no
neighbor {model['sigma_angstrom']*.3:.17g} bin
neigh_modify every 1 delay 0 check yes
timestep {dt_internal*lammps_time:.17g}
fix integrate all nve
dump trajectory all custom 1 trajectory.dump id x y z vx vy vz fx fy fz
dump_modify trajectory sort id format float %.17g
thermo 10
thermo_style custom step pe ke etotal temp
thermo_modify norm no format float %.17g
run {steps}
''')
    elapsed = {"sponge": execute([sponge, "-mdin", "mdin.spg.toml"], sp),
               "lammps": execute([lammps, "-in", "in.lammps"], lm)}
    n = model["atoms"]
    lines = (lm/"trajectory.dump").read_text().splitlines()
    frames = []
    for offset in range(0, len(lines), n+9):
        if int(lines[offset+1]) != len(frames): raise ValueError("unexpected LAMMPS frame step")
        frame = np.array([[float(x) for x in row.split()] for row in lines[offset+9:offset+9+n]])
        if not np.array_equal(frame[:, 0], np.arange(1, n+1)): raise ValueError("LAMMPS atom order mismatch")
        frames.append(frame[:, 1:])
    frames = np.array(frames)
    if len(frames) != steps+1: raise ValueError("LAMMPS trajectory length mismatch")
    sp_position = np.fromfile(sp/"crd.dat", dtype=np.float32).reshape(-1, n, 3)
    sp_half = np.fromfile(sp/"vel.dat", dtype=np.float32).reshape(-1, n, 3)
    sp_force = np.fromfile(sp/"force.dat", dtype=np.float32).reshape(-1, n, 3)
    if len(sp_position) != steps or len(sp_force) != steps+1: raise ValueError("SPONGE trajectory length mismatch")
    sp_centered = sp_half+.5*dt_internal*sp_force[1:]/state["masses"][None, :, None]
    displacement = sp_position-frames[1:, :, :3]
    displacement -= model["box_angstrom"]*np.rint(displacement/model["box_angstrom"])
    velocity_difference = sp_centered-frames[1:, :, 3:6]*lammps_time
    force_difference = sp_force-frames[:, :, 6:9]
    position_error = np.max(np.abs(displacement), axis=(1, 2))
    velocity_error = np.max(np.abs(velocity_difference), axis=(1, 2))
    force_error = np.max(np.abs(force_difference), axis=(1, 2))
    cutoff_crossings = []
    a, b = np.triu_indices(n, 1)
    positions = np.concatenate((state["positions"][None], sp_position))
    for step in range(steps+1):
        distances = []
        for crd in (positions[step], frames[step, :, :3]):
            delta = crd[a]-crd[b]
            delta -= model["box_angstrom"]*np.rint(delta/model["box_angstrom"])
            distances.append(np.linalg.norm(delta, axis=1))
        crossing = np.where((distances[0] < model["cutoff_angstrom"]) != (distances[1] < model["cutoff_angstrom"]))[0]
        if len(crossing):
            cutoff_crossings.append(dict(step=step, pairs=[[int(a[j]), int(b[j])] for j in crossing],
                                         sponge_r=distances[0][crossing].tolist(), lammps_r=distances[1][crossing].tolist(),
                                         max_force_difference=float(force_error[step])))
    # Finite-horizon float-vs-double checks, not a long-time trajectory identity.
    checks = dict(position=bool(position_error.max() < 2e-3),
                  centered_velocity=bool(velocity_error.max() < 2e-4), force=bool(force_error.max() < 5e-3))
    report = dict(status="passed" if all(checks.values()) else "failed", checks=checks,
                  scope="short deterministic NVE propagation; not ensemble/transport equivalence",
                  steps=steps, sponge_dt_ps=model["dt_ps"], actual_internal_dt=dt_internal,
                  aligned_lammps_dt_fs=dt_internal*lammps_time,
                  nominal_time_relative_difference=dt_internal*lammps_time/(model["dt_ps"]*1000)-1,
                  velocity_phase="SPONGE half-step + dt*force/(2m) versus LAMMPS integer-step",
                  model_sha256=digest(model_dir/"model.json"), wall_seconds=elapsed,
                  benchmark_script_sha256=digest(__file__), cutoff_crossings=cutoff_crossings,
                  max_position_error_angstrom=float(position_error.max()),
                  max_centered_velocity_error_native=float(velocity_error.max()),
                  max_force_error_kcal_mol_angstrom=float(force_error.max()),
                  source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  sponge_executable_sha256=digest(sponge), lammps_executable_sha256=digest(lammps))
    (output/"summary.json").write_text(json.dumps(report, indent=2)+"\n")
    np.savetxt(output/"errors.csv", np.column_stack((np.arange(1, steps+1), position_error, velocity_error, force_error[1:])), delimiter=",", header="step,position_A,centered_velocity_native,force_kcal_mol_A", comments="")
    if not all(checks.values()): raise RuntimeError(f"NVE gate failed; inspect {output/'summary.json'}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("prepare")
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--cells", type=int, default=3)
    generate.add_argument("--seed", type=int, default=20261027)
    check = sub.add_parser("static")
    check.add_argument("--model", type=Path, required=True)
    check.add_argument("--output", type=Path, required=True)
    check.add_argument("--sponge", type=Path, required=True)
    check.add_argument("--lammps", type=Path, required=True)
    dynamics = sub.add_parser("dynamics")
    dynamics.add_argument("--model", type=Path, required=True)
    dynamics.add_argument("--output", type=Path, required=True)
    dynamics.add_argument("--sponge", type=Path, required=True)
    dynamics.add_argument("--lammps", type=Path, required=True)
    dynamics.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    if args.command == "prepare": result = prepare(args.output, args.cells, args.seed)
    elif args.command == "static": result = static_check(args.model, args.output, args.sponge.resolve(), args.lammps.resolve())
    else: result = dynamics_check(args.model, args.output, args.sponge.resolve(), args.lammps.resolve(), args.steps)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prepare a bounded LJ equilibration pilot, with trajectory/state diagnostics.

This is pilot data for planning transport runs. It does not assert convergence.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.viscosity.workflows.config import write_engine_input
import subprocess

import h5py
import numpy as np

from lj_model import digest, execute, pair_reference, ROOT


def equilibrate(model_dir, output, sponge, steps=20000, seed=18273):
    model_dir, output = Path(model_dir).resolve(), Path(output).resolve()
    model = json.loads((model_dir/"model.json").read_text())
    for name, expected in model["input_sha256"].items():
        if digest(model_dir/name) != expected: raise ValueError(f"changed model input: {name}")
    if steps < 1000 or steps % 100:
        raise ValueError("pilot steps must be >=1000 and divisible by 100")
    output.mkdir(parents=True, exist_ok=False)
    config = dict(md_name="lj_equilibration_pilot", mode="nvt", pbc=True,
                  dt=model["dt_ps"], step_limit=steps, cutoff=model["cutoff_angstrom"],
                  skin=.3*model["sigma_angstrom"], target_temperature=model["target_temperature_k"],
                  thermostat="middle_langevin", thermostat_tau=model["tau_ps"], thermostat_seed=seed,
                  print_zeroth_frame=True, write_mdout_interval=100, write_information_interval=100,
                  write_trajectory_interval=10, write_restart_file_interval=steps,
                  crd="crd.dat", vel="vel.dat", output_h5_observable_path="observables.h5")
    config["PM.MPI_size"] = 1
    for name in ("mass", "charge", "coordinate", "velocity"):
        config[f"{name}_in_file"] = str(model_dir/f"{name}.txt")
    config["LJ_in_file"] = str(model_dir/"lj.txt")
    write_engine_input(output/"mdin.spg.toml", "\n".join(f"{k}={json.dumps(v)}" for k, v in config.items())+"\n")
    elapsed = execute([sponge, "-mdin", "mdin.spg.toml"], output, timeout=300)
    (output/"execution.json").write_text(json.dumps(dict(wall_seconds=elapsed, steps=steps, seed=seed))+"\n")
    return analyze(model_dir, output, sponge, steps, seed, elapsed)


def analyze(model_dir, output, sponge, steps, seed, elapsed=None):
    model_dir, output = Path(model_dir).resolve(), Path(output).resolve()
    model = json.loads((model_dir/"model.json").read_text())
    n, box = model["atoms"], model["box_angstrom"]
    position = np.fromfile(output/"crd.dat", dtype=np.float32).reshape(-1, n, 3).astype(float)
    if len(position) != steps//10: raise ValueError("trajectory frame count mismatch")
    if not np.all(np.isfinite(position)): raise ValueError("nonfinite trajectory")
    increments = np.diff(position, axis=0)
    increments -= box*np.rint(increments/box)
    # This check cannot prove absence of multiple crossings between samples.
    # It detects resolved jumps approaching the unwrap ambiguity scale.
    max_jump = float(np.max(np.abs(increments)))
    if max_jump >= box/4: raise ValueError("trajectory too sparse to unwrap reliably")
    continuous = np.concatenate((position[:1], position[:1]+np.cumsum(increments, axis=0)))
    begin = len(position)//2
    displacement = continuous[begin:]-continuous[begin]
    displacement -= displacement.mean(axis=1, keepdims=True)
    msd = np.mean(np.sum(displacement**2, axis=2), axis=1)
    with h5py.File(output/"observables.h5") as file:
        if file["parameters/sponge/output/status"][()] != b"finalized":
            raise ValueError("pilot observable stream not finalized")
        group = file["observables/all"]
        recorded_steps = group["step"][()]
        temperature, potential = group["temperature/value"][()], group["eff_pot/value"][()]
        if not np.array_equal(recorded_steps, np.arange(0, steps+1, 100)):
            raise ValueError("unexpected pilot observable cadence")
        journal = file["parameters/sponge/output"]
        if int(journal["frame_count"][-1]) != len(recorded_steps) or int(journal["last_complete_step"][-1]) != steps:
            raise ValueError("pilot completion journal mismatch")
        if not np.all(np.isfinite(temperature)) or not np.all(np.isfinite(potential)):
            raise ValueError("nonfinite pilot observables")
    cut = len(recorded_steps)//2
    summaries = []
    for indices in np.array_split(np.arange(cut, len(recorded_steps)), 5):
        summaries.append(dict(first_step=int(recorded_steps[indices[0]]), last_step=int(recorded_steps[indices[-1]]),
                              mean_temperature_k=float(np.mean(temperature[indices])),
                              mean_potential_kcal_mol=float(np.mean(potential[indices]))))
    # Final native trajectory frame supplies a structural seed, not a dynamic
    # thermostat restart. It has the same completed step in both arrays.
    velocities = np.fromfile(output/"vel.dat", dtype=np.float32).reshape(-1, n, 3).astype(float)
    if velocities.shape != position.shape or not np.all(np.isfinite(velocities)):
        raise ValueError("velocity trajectory mismatch/nonfinite values")
    coordinates, velocities = position[-1], velocities[-1]
    _, force, _, _ = pair_reference(coordinates, box, model["c12"], model["c6"], model["cutoff_angstrom"])
    dt_internal = float(np.float32(np.float32(model["dt_ps"])*np.float32(20.455)))
    centered = velocities+.5*dt_internal*force/model["mass_da"]
    np.savez(output/"structural_seed.npz", positions=coordinates, half_velocities=velocities, centered_velocities=centered, box=box)
    report = dict(status="pilot_only", model_sha256=digest(model_dir/"model.json"),
                  source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  executable_sha256=digest(sponge), script_sha256=digest(__file__),
                  steps=steps, thermostat_seed=seed, duration_ps=steps*model["dt_ps"],
                  duration_star=steps*model["dt_star"], wall_seconds=elapsed, cpu_threads=1,
                  mean_latter_half_temperature_k=float(np.mean(temperature[cut:])),
                  target_temperature_k=model["target_temperature_k"], latter_half_blocks=summaries,
                  latter_half_single_origin_msd_sigma2=float(msd[-1]/model["sigma_angstrom"]**2),
                  maximum_resolved_frame_jump_angstrom=max_jump,
                  seed_sha256=digest(output/"structural_seed.npz"),
                  warnings=["Single pilot, not independent replicas or a viscosity estimate.",
                            "Native temperature uses the existing velocity convention; do not treat it as centered flow thermal temperature.",
                            "MSD is a single-origin mobility diagnostic, not a converged diffusion coefficient or proof of equilibrium.",
                            "Final trajectory state seeds a new experiment; no thermostat/RNG dynamic continuity is claimed.",
                            "Production NVE temperature/energy relaxation must be checked after removing the heat bath."])
    if elapsed is None:
        report["warnings"].append("End-to-end wall time was not retained for this recovered pilot; exclude it from quantitative cost comparisons.")
    (output/"summary.json").write_text(json.dumps(report, indent=2)+"\n")
    np.savetxt(output/"msd.csv", np.column_stack((np.arange(len(msd))*10*model["dt_ps"], msd/model["sigma_angstrom"]**2)), delimiter=",", header="elapsed_ps,msd_sigma2", comments="")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sponge", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=18273)
    args = parser.parse_args()
    print(json.dumps(equilibrate(args.model, args.output, args.sponge.resolve(), args.steps, args.seed), indent=2))

"""Bounded physical-endpoint HMR timestep campaign; never edits source inputs."""

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from benchmarks.utils import Outputer, Runner


def run_case(
    root,
    binary,
    name,
    *,
    mode,
    dt_fs,
    duration_ps=0,
    steps=None,
    hmr=False,
    seed=712,
    source=None,
    regenerate=True,
    restraint=False,
    thermostat="middle_langevin",
    extra_options=None,
):
    destination = root / "runs" / name
    status_file = destination / "status.json"
    dt = dt_fs / 1000
    steps = steps if steps is not None else int(math.ceil(duration_ps / dt))
    binary_hash = hashlib.sha256(binary.read_bytes()).hexdigest()
    if status_file.exists():
        cached = json.loads(status_file.read_text())
        expected = dict(
            mode=mode,
            dt_fs=dt_fs,
            steps=steps,
            hmr=hmr,
            seed=seed,
            source=source,
            binary_sha256=binary_hash,
            regenerated_velocities=regenerate,
        )
        if any(cached.get(k) != v for k, v in expected.items()):
            raise RuntimeError(
                f"Cached result has different inputs: {destination}"
            )
        if cached.get("extra_options", {}) != (extra_options or {}):
            raise RuntimeError(
                f"Cached result has different options: {destination}"
            )
        if (
            mode in ("nvt", "npt")
            and cached.get("thermostat", "middle_langevin") != thermostat
        ):
            raise RuntimeError(
                f"Cached result has different thermostat: {destination}"
            )
        return cached
    if destination.exists():
        raise RuntimeError(f"Unfinished case exists: {destination}")
    case = Outputer.prepare_output_case(
        root / "statics", root / "runs", "tyk2_jmc28", run_tag=name
    )
    output_interval = max(1, round(0.1 / dt))
    options = dict(
        md_name=name,
        mode=mode,
        default_in_file_prefix="TMP",
        pbc=True,
        cutoff=8.0,
        skin=2.0,
        step_limit=steps,
        dt=dt,
        constrain_mode="SHAKE",
        constrain_precision="double",
        print_zeroth_frame=True,
        print_pressure=True,
        write_mdout_interval=output_interval,
        write_information_interval=max(1000, output_interval),
        write_trajectory_interval=max(1, round(1.0 / dt)),
        write_restart_file_interval=steps,
        crd="mdcrd.dat",
        box="mdbox.txt",
        mdout="mdout.txt",
        mdinfo="mdinfo.txt",
        rst="restart",
    )
    options.update(
        {
            "PM.MPI_size": 1,
            "SHAKE.iteration_numbers": 100,
            "SHAKE.tolerance": 1e-6,
        }
    )
    if hmr:
        options.update(
            {"hmr.enabled": True, "hmr.ratio_xh1": 2.0, "hmr.ratio_other": 2.5}
        )
    if source:
        options["coordinate_in_file"] = str(
            (root / "runs" / source / "restart_coordinate.txt").resolve()
        )
        if not regenerate:
            options["velocity_in_file"] = str(
                (root / "runs" / source / "restart_velocity.txt").resolve()
            )
    if mode != "minimization" and regenerate:
        options.update(
            {
                "initial_velocity.mode": "maxwell",
                "initial_velocity.seed": seed,
                "target_temperature": 300.0,
            }
        )
    if mode in ("nvt", "npt"):
        options.update(
            thermostat=thermostat,
            thermostat_tau=1.0,
            thermostat_seed=seed + 10000,
            target_temperature=300.0,
        )
    if mode == "npt":
        options.update(
            barostat="monte_carlo_barostat",
            target_pressure=1.0,
            barostat_seed=seed + 20000,
            barostat_update_interval=max(1, round(0.05 / dt)),
        )
    if restraint:
        options.update(
            restrain_atom_id="protein_heavy.txt", restrain_single_weight=2.0
        )
    options.update(extra_options or {})
    (case / "mdin.spg.toml").write_text(
        "\n".join(f"{k}={json.dumps(v)}" for k, v in options.items()) + "\n"
    )
    status = dict(
        name=name,
        mode=mode,
        dt_fs=dt_fs,
        steps=steps,
        duration_ps=steps * dt,
        hmr=hmr,
        seed=seed,
        source=source,
        regenerated_velocities=regenerate,
        thermostat=thermostat if mode in ("nvt", "npt") else None,
        extra_options=extra_options or {},
        binary_sha256=binary_hash,
    )
    start = time.monotonic()
    try:
        with (
            (case / "run.log").open("w") as log,
            contextlib.redirect_stdout(log),
        ):
            output = Runner.run_sponge(
                case,
                sponge_cmd=binary,
                timeout=1800,
                env={
                    **os.environ,
                    "OMP_NUM_THREADS": "2",
                    "MKL_NUM_THREADS": "1",
                },
            )
            log.write(output)
        status["completed"] = True
    except Exception as error:
        status.update(completed=False, error=repr(error))
    status["wall_seconds"] = time.monotonic() - start
    status["ns_per_day"] = (
        steps * dt / 1000 / max(status["wall_seconds"], 1e-9) * 86400
    )
    log = (case / "run.log").read_text(errors="replace")
    for key, pattern in [
        ("dof", r"freedom (\d+)"),
        ("constraint_count", r"(\d+) constrain pairs"),
    ]:
        matches = re.findall(pattern, log)
        if matches:
            status[key] = int(matches[-1])
    status_file.write_text(json.dumps(status, indent=2))
    print(json.dumps(status), flush=True)
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=["pilot", "prepare", "screen", "production", "validate-settle"],
        required=True,
    )
    parser.add_argument("--dts", default="1,2,3,4,5,6,8,10,12")
    parser.add_argument("--duration-ps", type=float, default=50)
    parser.add_argument("--seeds", default="712,713,714")
    parser.add_argument("--modes", default="nvt")
    parser.add_argument("--source", default="equilibrated")
    parser.add_argument("--disable-settle", action="store_true")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    root, binary = args.root.resolve(), args.binary.resolve()
    if args.stage in ("pilot", "prepare"):
        prefix = "shake_" if args.disable_settle else ""
        solver_options = (
            {"settle_disable": True} if args.disable_settle else None
        )
        result = run_case(
            root,
            binary,
            prefix + "minimize",
            mode="minimization",
            dt_fs=1,
            steps=5000,
            regenerate=False,
            extra_options=solver_options,
        )
        if not result["completed"]:
            raise RuntimeError("Initial minimization failed")
        result = run_case(
            root,
            binary,
            prefix + "pilot",
            mode="nvt",
            dt_fs=1,
            steps=2000,
            source=prefix + "minimize",
            restraint=True,
            extra_options=solver_options,
        )
        if not result["completed"]:
            raise RuntimeError("GPU pilot failed")
        if args.stage == "pilot":
            return
        for name, mode, duration, source, restraint in [
            ("warmup", "nvt", 20, "pilot", True),
            ("density_equil", "npt", 100, "warmup", True),
            (
                "equilibrated",
                "npt",
                200 if args.disable_settle else 100,
                "density_equil",
                False,
            ),
        ]:
            result = run_case(
                root,
                binary,
                prefix + name,
                mode=mode,
                dt_fs=1,
                duration_ps=duration,
                source=prefix + source,
                regenerate=False,
                restraint=restraint,
                extra_options=solver_options,
            )
            if not result["completed"]:
                raise RuntimeError(f"Preparation failed: {name}")
        return
    if not (root / "runs" / args.source / "restart_coordinate.txt").is_file():
        raise RuntimeError("Complete preparation first")
    if args.stage == "validate-settle":
        if args.disable_settle:
            raise ValueError("validate-settle requires default SETTLE enabled")
        for hmr, dt in [(False, 1), (True, 1), (True, 4), (True, 5)]:
            run_case(
                root,
                binary,
                f"fixed_settle_{'hmr' if hmr else 'normal'}_{dt}fs_nvt",
                mode="nvt",
                dt_fs=dt,
                duration_ps=args.duration_ps,
                hmr=hmr,
                seed=712,
                source=args.source,
            )
        return
    if args.stage == "screen":
        for hmr, dt in [(False, 1), (False, 2)] + [
            (True, float(x)) for x in args.dts.split(",")
        ]:
            for mode in ("nve", "nvt"):
                name = f"screen_{args.tag}{'hmr' if hmr else 'normal'}_{dt:g}fs_{mode}"
                run_case(
                    root,
                    binary,
                    name,
                    mode=mode,
                    dt_fs=dt,
                    duration_ps=args.duration_ps,
                    hmr=hmr,
                    source=args.source,
                    extra_options={"settle_disable": True}
                    if args.disable_settle
                    else None,
                )
    else:
        for mode in args.modes.split(","):
            for seed in map(int, args.seeds.split(",")):
                for label in args.dts.split(","):
                    hmr = not label.startswith("normal:")
                    dt = float(label.split(":")[-1])
                    name = f"production_{args.tag}{'hmr' if hmr else 'normal'}_{dt:g}fs_{mode}_{seed}"
                    run_case(
                        root,
                        binary,
                        name,
                        mode=mode,
                        dt_fs=dt,
                        duration_ps=args.duration_ps,
                        hmr=hmr,
                        seed=seed,
                        source=args.source,
                        extra_options={"settle_disable": True}
                        if args.disable_settle
                        else None,
                    )


if __name__ == "__main__":
    main()

"""Conventional alanine-dipeptide HMR sampling experiment with existing inputs."""

import argparse
import contextlib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from benchmarks.utils import Extractor, Outputer, Runner
from benchmarks.validation.cv.tests.utils import write_simple_cv_file

REPO = Path(__file__).resolve().parents[4]
FIXTURE = REPO / "benchmarks/performance/sits/statics/ala2_sits"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def initialize(root, binary):
    if (root / "provenance.json").exists():
        provenance = json.loads((root / "provenance.json").read_text())
        if sha(root / "SPONGE") != provenance["binary_sha256"]:
            raise RuntimeError("Frozen binary changed")
        for name, digest in provenance["fixture_sha256"].items():
            if sha(root / "statics/ala2_sits" / name) != digest:
                raise RuntimeError(f"Frozen fixture changed: {name}")
        return
    root.mkdir(parents=True, exist_ok=False)
    (root / "statics").mkdir()
    shutil.copytree(FIXTURE, root / "statics/ala2_sits")
    shutil.copy2(binary, root / "SPONGE")
    provenance = dict(
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        binary_sha256=sha(root / "SPONGE"),
        fixture=str(FIXTURE),
        fixture_sha256={p.name: sha(p) for p in FIXTURE.glob("ALA_*")},
        atom_count=1840,
        solute_atoms=22,
        waters=606,
        phi=[4, 6, 8, 14],
        psi=[6, 8, 14, 16],
        conditions=["normal_2fs", "hmr_2fs", "hmr_4fs"],
        hmr=dict(ratio_xh1=2, ratio_other=2.5, water="unchanged"),
        ensemble="NVT, Middle Langevin tau=1 ps, 300 K",
        sampling_interval_ps=0.1,
        burn_fraction=0.2,
        scope="ESS is observable-specific; throughput alone is not sampling efficiency",
    )
    (root / "provenance.json").write_text(json.dumps(provenance, indent=2))


def run_case(
    root,
    name,
    *,
    mode,
    dt_fs,
    steps,
    hmr=False,
    seed=812,
    source=None,
    regenerate=True,
):
    case = root / "runs" / name
    options = dict(
        md_name=name,
        mode=mode,
        default_in_file_prefix="ALA",
        pbc=True,
        cutoff=8,
        skin=2,
        dt=dt_fs / 1000,
        step_limit=steps,
        constrain_mode="SHAKE",
        constrain_precision="double",
        print_zeroth_frame=True,
        print_pressure=True,
        write_mdout_interval=max(1, round(0.1 / (dt_fs / 1000))),
        write_information_interval=max(1, round(50 / (dt_fs / 1000))),
        write_trajectory_interval=max(1, round(10 / (dt_fs / 1000))),
        write_restart_file_interval=steps,
        crd="mdcrd.dat",
        box="mdbox.txt",
        mdout="mdout.txt",
        rst="restart",
        cv_in_file="cv.txt",
        target_temperature=300,
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
            {"hmr.enabled": True, "hmr.ratio_xh1": 2, "hmr.ratio_other": 2.5}
        )
    source_hash = None
    if source:
        q = root / "runs" / source / "restart_coordinate.txt"
        options["coordinate_in_file"] = str(q)
        source_hash = sha(q)
        if not regenerate:
            options["velocity_in_file"] = str(
                root / "runs" / source / "restart_velocity.txt"
            )
    if mode != "minimization" and regenerate:
        options.update(
            {"initial_velocity.mode": "maxwell", "initial_velocity.seed": seed}
        )
    if mode in ("nvt", "npt"):
        options.update(
            thermostat="middle_langevin",
            thermostat_tau=1,
            thermostat_seed=seed + 10000,
        )
    if mode == "npt":
        options.update(
            barostat="monte_carlo_barostat",
            target_pressure=1,
            barostat_seed=seed + 20000,
            barostat_update_interval=max(1, round(0.05 / (dt_fs / 1000))),
        )
    specification = dict(
        name=name,
        options=options,
        source_sha256=source_hash,
        binary_sha256=sha(root / "SPONGE"),
    )
    if (case / "status.json").exists():
        status = json.loads((case / "status.json").read_text())
        if status["specification"] != specification or not status["completed"]:
            raise RuntimeError(f"Existing incompatible or failed run: {case}")
        return status
    if case.exists():
        raise RuntimeError(f"Existing unfinished run: {case}")
    Outputer.prepare_output_case(
        root / "statics", root / "runs", "ala2_sits", run_tag=name
    )
    write_simple_cv_file(case)
    cv_path = case / "cv.txt"
    cv_path.write_text(
        cv_path.read_text().replace(
            "CV = distance angle phi psi", "CV = phi psi"
        )
    )
    (case / "mdin.spg.toml").write_text(
        "\n".join(f"{k}={json.dumps(v)}" for k, v in options.items()) + "\n"
    )
    status = dict(
        specification=specification,
        completed=False,
        mode=mode,
        hmr=hmr,
        dt_fs=dt_fs,
        seed=seed,
        duration_ps=steps * dt_fs / 1000,
    )
    start = time.monotonic()
    try:
        with (
            (case / "run.log").open("w") as log,
            contextlib.redirect_stdout(log),
        ):
            output = Runner.run_sponge(
                case,
                sponge_cmd=root / "SPONGE",
                timeout=7200,
                env={
                    **os.environ,
                    "OMP_NUM_THREADS": "2",
                    "MKL_NUM_THREADS": "1",
                },
            )
            log.write(output)
        rows = Extractor.parse_mdout_rows(
            case / "mdout.txt",
            ["step", "temperature", "potential", "phi", "psi"],
        )
        interval = options["write_mdout_interval"]
        if [r["step"] for r in rows] != list(range(0, steps + 1, interval)):
            raise RuntimeError("Incomplete mdout")
        if not np.isfinite(
            [
                [r[k] for k in ("temperature", "potential", "phi", "psi")]
                for r in rows
            ]
        ).all():
            raise RuntimeError("Nonfinite output")
        output = (case / "run.log").read_text()
        if re.search(
            r"Fatal SPONGE|(?:cuda|hip)Error[A-Z]\w*|SITS mode =", output
        ):
            raise RuntimeError("Runtime error or unintended SITS bias")
        status["completed"] = True
        for key, pattern in [
            ("core_seconds", r"Core Run Wall Time: ([0-9.eE+-]+) seconds"),
            (
                "initialization_seconds",
                r"Initialization\s*\|\s*([0-9.eE+-]+) seconds",
            ),
        ]:
            match = re.search(pattern, output)
            status[key] = float(match[1]) if match else None
    except Exception as error:
        status["error"] = repr(error)
    status["wall_seconds"] = time.monotonic() - start
    (case / "status.json").write_text(json.dumps(status, indent=2))
    print(
        name,
        "PASS" if status["completed"] else "FAIL",
        f"wall={status['wall_seconds']:.2f}s",
        flush=True,
    )
    if not status["completed"]:
        raise RuntimeError(f"Simulation failed: {case}")
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=["prepare", "pilot", "production"], required=True
    )
    parser.add_argument("--duration-ps", type=float, default=2000)
    args = parser.parse_args()
    root = args.root.resolve()
    initialize(root, args.binary.resolve())
    if args.stage == "prepare":
        run_case(root, "minimize", mode="minimization", dt_fs=1, steps=2000)
        run_case(
            root, "warmup", mode="nvt", dt_fs=1, steps=50000, source="minimize"
        )
        run_case(
            root,
            "density_equil",
            mode="npt",
            dt_fs=1,
            steps=400000,
            source="warmup",
            regenerate=False,
        )
        run_case(
            root,
            "equilibrated",
            mode="nvt",
            dt_fs=2,
            steps=50000,
            source="density_equil",
            regenerate=False,
        )
        return
    if args.stage == "pilot":
        run_case(
            root,
            "pilot_hmr_4fs",
            mode="nvt",
            dt_fs=4,
            steps=25000,
            hmr=True,
            source="equilibrated",
        )
        return
    conditions = [("normal", 2, False), ("hmr", 2, True), ("hmr", 4, True)]
    plan = dict(
        duration_ps=args.duration_ps,
        seeds=[812, 813, 814],
        order="balanced Latin rotation",
        burn_fraction=0.2,
        sample_ps=0.1,
    )
    plan_path = root / "production_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise RuntimeError("Production plan changed")
    plan_path.write_text(json.dumps(plan, indent=2))
    for i, seed in enumerate(plan["seeds"]):
        for label, dt, hmr in conditions[i:] + conditions[:i]:
            run_case(
                root,
                f"production_{label}_{dt}fs_{seed}",
                mode="nvt",
                dt_fs=dt,
                steps=round(args.duration_ps * 1000 / dt),
                hmr=hmr,
                seed=seed,
                source="equilibrated",
            )


if __name__ == "__main__":
    main()

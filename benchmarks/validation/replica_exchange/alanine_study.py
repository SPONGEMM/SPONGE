"""Run and rescore fixed-window T-REMD or independent NVT on ACE-ALA-NME."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import time


def parm_fields(path):
    fields = {}
    for section in path.read_text().split("%FLAG ")[1:]:
        name, fmt, *lines = section.splitlines()
        match = re.search(r"\d+[aAiI](\d+)", fmt)
        if match:
            width = int(match.group(1))
            fields[name.strip()] = [
                data.strip()
                for line in lines
                if not line.startswith("%")
                for offset in range(0, len(line), width)
                if (data := line[offset : offset + width]).strip()
            ]
    return fields


def structure(source, implicit=False):
    fields = parm_fields(source / "system.parm7")
    names = fields["ATOM_NAME"]
    labels = fields["RESIDUE_LABEL"]
    offsets = [int(item) - 1 for item in fields["RESIDUE_POINTER"]] + [
        len(names)
    ]
    if labels[:3] != ["ACE", "ALA", "NME"]:
        raise ValueError("expected an ACE-ALA-NME solute")

    def index(residue, atom):
        return offsets[residue] + names[
            offsets[residue] : offsets[residue + 1]
        ].index(atom)

    waters = labels.count("WAT")
    if implicit and len(labels) != 3:
        raise ValueError("expected a dry ACE-ALA-NME topology")
    if not implicit and (not waters or len(names) - offsets[3] != 3 * waters):
        raise ValueError("expected explicit three-site water")
    return {
        "atom_count": len(names),
        "phi": [index(0, "C"), index(1, "N"), index(1, "CA"), index(1, "C")],
        "psi": [index(1, "N"), index(1, "CA"), index(1, "C"), index(2, "N")],
        "waters": waters,
    }


def write_config(path, settings):
    path.write_text(
        "".join(
            f"{key} = {json.dumps(value)}\n" for key, value in settings.items()
        )
    )


def execute(command, directory, gpu=None):
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    start = time.monotonic()
    log = directory / "execution.log"
    with log.open("w") as stream:
        result = subprocess.run(
            command,
            cwd=directory,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    if result.returncode:
        raise RuntimeError(f"{command[0]} failed in {directory}; see {log}")
    return time.monotonic() - start


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def periodic_dihedral(points, lengths):
    bonds = []
    for a, b in zip(points, points[1:]):
        bonds.append(
            tuple(
                (dy := y - x) - length * round(dy / length)
                for x, y, length in zip(a, b, lengths)
            )
        )
    b0, b1, b2 = bonds
    n0, n1 = cross(b0, b1), cross(b1, b2)
    return math.degrees(
        math.atan2(dot(cross(n0, n1), b1) / math.sqrt(dot(b1, b1)), dot(n0, n1))
    )


def rescore(args, directory, geometry, common, device):
    trajectory = directory / "mdcrd.dat"
    boxes = [
        tuple(map(float, line.split()))
        for line in (directory / "mdbox.txt").read_text().splitlines()
        if line.strip()
    ]
    atom_count = geometry["atom_count"]
    size = atom_count * 12
    frames, remainder = divmod(trajectory.stat().st_size, size)
    if remainder or frames != len(boxes) or frames != args.steps // args.sample:
        raise ValueError(
            f"trajectory frame mismatch: {directory}: {frames} {remainder} {len(boxes)}"
        )
    rerun = directory / "rerun"
    rerun.mkdir()
    settings = dict(
        common,
        mode="rerun",
        rerun_start=0,
        rerun_strip=0,
        rerun_frame_limit=frames,
        crd=str(trajectory),
        box=str(directory / "mdbox.txt"),
        print_zeroth_frame=1,
        write_mdout_interval=1,
        write_trajectory_interval=0,
        write_information_interval=0,
    )
    write_config(rerun / "mdin.spg.toml", settings)
    execute([args.engine, "-mdin", "mdin.spg.toml"], rerun, device)
    with (rerun / "mdout.txt").open() as stream:
        header = next(stream).split()
        energy_column = header.index("eff_pot")
        index_column = header.index("frame")
        energies = [
            (int(row[index_column]), float(row[energy_column]))
            for line in stream
            if (row := line.split())
        ]
    if [index for index, _ in energies] != list(range(1, frames + 1)):
        raise ValueError(f"rerun frame indices do not match the input: {rerun}")
    with (
        trajectory.open("rb") as coordinates,
        (directory / "samples.csv").open("w") as output,
    ):
        writer = csv.writer(output)
        writer.writerow(
            ["frame", "time_ps", "potential_kcal_mol", "phi_deg", "psi_deg"]
        )
        for frame_number, potential in energies:
            index = frame_number - 1
            frame = coordinates.read(size)
            lengths = boxes[index][:3]
            if not all(math.isfinite(value) for value in (*lengths, potential)):
                raise ValueError("non-finite sample")

            def angle(name):
                return periodic_dihedral(
                    [
                        struct.unpack_from("=3f", frame, atom * 12)
                        for atom in geometry[name]
                    ],
                    lengths,
                )

            writer.writerow(
                [
                    index,
                    (index + 1) * args.sample * args.dt,
                    potential,
                    angle("phi"),
                    angle("psi"),
                ]
            )


def run(args):
    root = args.output
    root.mkdir()
    geometry = structure(args.source, args.gb_start is not None)
    common = dict(
        md_name="ACE-ALA-NME TIP3P",
        amber_parm7=str(args.source / "system.parm7"),
        amber_rst7=str(args.source / "system_minimized.rst7"),
        cutoff=8.0,
        solvent_LJ=1,
        constrain_mode="SHAKE",
    )
    if args.gb_start is not None:
        common = dict(
            md_name="ACE-ALA-NME GB1",
            amber_parm7=common["amber_parm7"],
            amber_rst7=str(args.gb_start.resolve()),
            gb_in_file=str(args.source / "gb.txt"),
            cutoff=100.0,
            pbc=0,
        )
    settings = dict(
        common,
        mode="nvt",
        step_limit=args.steps,
        dt=args.dt,
        thermostat="middle_langevin",
        thermostat_tau=0.1,
        write_mdout_interval=args.sample,
        write_trajectory_interval=args.sample,
        write_information_interval=args.steps,
    )
    directories = []
    for index, temperature in enumerate(args.temperatures):
        directory = root / f"window-{index}"
        directory.mkdir()
        write_config(
            directory / "mdin.spg.toml",
            dict(
                settings,
                target_temperature=temperature,
                thermostat_seed=args.seed * 10 + index,
            ),
        )
        directories.append(directory)
    inputs = {
        key: hashlib.sha256(Path(common[key]).read_bytes()).hexdigest()
        for key in ("amber_parm7", "amber_rst7", "gb_in_file")
        if key in common
    }
    metadata = {
        "mode": args.mode,
        "seed": args.seed,
        "temperatures": args.temperatures,
        "steps": args.steps,
        "dt_ps": args.dt,
        "sample_steps": args.sample,
        "exchange_steps": args.interval,
        "dedicated_gpu_per_window": True,
        "engine_sha256": hashlib.sha256(
            Path(args.engine).read_bytes()
        ).hexdigest(),
        "manager_sha256": hashlib.sha256(
            Path(args.manager).read_bytes()
        ).hexdigest(),
        "geometry": geometry,
        "inputs": inputs,
    }
    if args.mode == "remd":
        placements = [
            value
            for gpu, directory in zip(args.devices, directories)
            for value in (gpu, str(directory))
        ]
        elapsed = execute(
            [
                args.manager,
                args.engine,
                str(args.interval),
                "300000",
                str(args.seed),
                "exchange",
                *placements,
            ],
            root,
        )
        metadata["wall_seconds_by_run"] = [elapsed]
    else:
        with ThreadPoolExecutor(max_workers=len(directories)) as pool:
            metadata["wall_seconds_by_run"] = list(
                pool.map(
                    lambda pair: execute(
                        [args.engine, "-mdin", "mdin.spg.toml"], *pair
                    ),
                    zip(directories, args.devices),
                )
            )
    with ThreadPoolExecutor(max_workers=len(directories)) as pool:
        list(
            pool.map(
                lambda pair: rescore(args, pair[0], geometry, common, pair[1]),
                zip(directories, args.devices),
            )
        )
    (root / "study.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine")
    parser.add_argument("manager")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--gb-start", type=Path, help="dry GB1 starting Amber restart"
    )
    parser.add_argument("--mode", choices=("remd", "nvt"), required=True)
    parser.add_argument("--devices", nargs="+", required=True)
    parser.add_argument("--temperatures", nargs="+", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=50_001)
    parser.add_argument("--interval", type=int, default=250)
    parser.add_argument("--sample", type=int, default=250)
    parser.add_argument("--dt", type=float, default=0.002)
    args = parser.parse_args()
    if len(args.devices) != len(args.temperatures):
        parser.error("each window needs a GPU identifier and temperature")
    if (
        min(args.steps, args.sample, args.interval, args.dt, args.seed) <= 0
        or (args.steps - 1) % args.sample != 0
    ):
        parser.error(
            "use positive parameters and steps = sample * frame count + 1"
        )
    args.engine = str(Path(args.engine).resolve())
    args.manager = str(Path(args.manager).resolve())
    args.source = args.source.resolve()
    args.output = args.output.resolve()
    run(args)

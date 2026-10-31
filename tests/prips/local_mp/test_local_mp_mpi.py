"""SPONGE executable parity checks against the independent full-system toy model.

CTest runs one named case per process. Failed case directories and per-rank
JSON logs remain under --work-root for MPI diagnosis.
"""

import argparse
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import torch

PRIPS_SOURCE = Path(__file__).resolve().parents[3] / "plugins" / "prips"
sys.path.insert(0, str(PRIPS_SOURCE))

from fixture_model import (
    ATOM_TYPES,
    CELL,
    POSITIONS,
    SEED_FORCE,
    full_system_reference,
)


RTOL = 1e-5
ATOL = 1e-5
PRESSURE_CONVERSION = 6.946827162543585e4


def _close(actual, expected, *, rtol=RTOL, atol=ATOL):
    torch.testing.assert_close(
        torch.as_tensor(actual, dtype=torch.float64),
        torch.as_tensor(expected, dtype=torch.float64),
        rtol=rtol,
        atol=atol,
    )


def _write_input(
    directory,
    args,
    *,
    ranks,
    plugin=True,
    positions=POSITIONS,
    cell=CELL,
    atom_types=ATOM_TYPES,
    steps=0,
    dt=0.0,
    update_interval=100,
    velocities=None,
):
    directory.mkdir(parents=True, exist_ok=True)
    atom_types = tuple(atom_types)
    masses = {6: 12.0, 8: 16.0}
    (directory / "coordinate.txt").write_text(
        f"{len(positions)} 0\n"
        + "\n".join(" ".join(map(str, p)) for p in positions)
        + f"\n{cell[0][0]} {cell[1][1]} {cell[2][2]}\n90 90 90\n",
        encoding="utf-8",
    )
    (directory / "mass.txt").write_text(
        f"{len(positions)}\n"
        + "\n".join(str(masses[atom_type]) for atom_type in atom_types)
        + "\n",
        encoding="utf-8",
    )
    (directory / "charge.txt").write_text(
        f"{len(positions)}\n" + "0\n" * len(positions), encoding="utf-8"
    )
    if velocities is None:
        velocities = [(0.0, 0.0, 0.0)] * len(positions)
    (directory / "velocity.txt").write_text(
        f"{len(positions)}\n"
        + "\n".join(" ".join(map(str, row)) for row in velocities)
        + "\n",
        encoding="utf-8",
    )
    lines = [
        'md_name = "local MP parity"',
        'mode = "nve"',
        f"step_limit = {steps}",
        f"dt = {dt}",
        "cutoff = 0.5",
        "skin = 0.2",
        f'device = "{" ".join(map(str, range(ranks)))}"',
        "print_zeroth_frame = 1",
        "write_mdout_interval = 1",
        "write_trajectory_interval = 1",
        "print_pressure = true",
        'coordinate_in_file = "coordinate.txt"',
        'velocity_in_file = "velocity.txt"',
        'mass_in_file = "mass.txt"',
        'charge_in_file = "charge.txt"',
        'frc = "frc.dat"',
        'mdout = "mdout.txt"',
    ]
    if plugin:
        fixture_path = getattr(args, "py", None)
        if fixture_path is not None:
            fixture_path = str(fixture_path)
        if not fixture_path:
            fixture_path = str(Path(__file__).with_name("fixture_model.py"))
        lines.extend(
            (
                f'plugin = "{args.plugin}"',
                f'py = "{fixture_path}"',
            )
        )
    lines.extend(
        (
            "[DOM_DEC]",
            f"update_interval = {update_interval}",
            f"split_nx = {ranks}",
            "split_ny = 1",
            "split_nz = 1",
            "[PM]",
            "MPI_size = 0",
        )
    )
    (directory / "mdin.spg.toml").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _events(directory):
    events = []
    for path in sorted((directory / "rank_logs").glob("rank*.jsonl")):
        events.extend(
            json.loads(line) for line in path.read_text().splitlines()
        )
    return events


def _run(
    args,
    label,
    *,
    ranks=1,
    plugin=True,
    positions=POSITIONS,
    steps=0,
    dt=0.0,
    update_interval=100,
    cell=CELL,
    atom_types=ATOM_TYPES,
    velocities=None,
):
    directory = args.work_root / f"{args.case}-{label}-{os.getpid()}"
    _write_input(
        directory,
        args,
        ranks=ranks,
        plugin=plugin,
        positions=positions,
        cell=cell,
        atom_types=atom_types,
        steps=steps,
        dt=dt,
        update_interval=update_interval,
        velocities=velocities,
    )
    env = os.environ.copy()
    env["SPONGE_LOCAL_MP_LOG_DIR"] = str(directory / "rank_logs")
    env["SPONGE_LOCAL_MP_ATOM_TYPES"] = json.dumps(tuple(atom_types))
    source_package = str(PRIPS_SOURCE)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (source_package, env.get("PYTHONPATH")))
    )
    command = [args.mpiexec, "--oversubscribe"]
    command.extend([args.numproc_flag, str(ranks)])
    command.extend([args.sponge, "-mdin", "mdin.spg.toml"])
    try:
        result = subprocess.run(
            command,
            cwd=directory,
            env=env,
            capture_output=True,
            text=True,
            timeout=args.timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        for name, output in (
            ("stdout", error.stdout),
            ("stderr", error.stderr),
        ):
            if isinstance(output, bytes):
                output = output.decode("utf-8", errors="replace")
            (directory / f"{name}.log").write_text(
                output or "", encoding="utf-8"
            )
        raise
    (directory / "stdout.log").write_text(result.stdout, encoding="utf-8")
    (directory / "stderr.log").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise AssertionError(
            f"Unexpected SPONGE exit {result.returncode}; logs: {directory}"
        )
    return (
        directory,
        _events(directory),
    )


def _assemble(events, index, num_atoms=None):
    evaluations = [
        event
        for event in events
        if event["event"] == "evaluate" and event["count"] == index
    ]
    assert evaluations, f"missing model evaluation {index}"
    if num_atoms is None:
        atom_ids = [
            int(atom_id) for event in evaluations for atom_id in event["owned"]
        ]
        if not atom_ids:
            raise AssertionError("no owned atom outputs were recorded")
        num_atoms = max(atom_ids) + 1
    force = [None] * num_atoms
    positions = [None] * num_atoms
    energy = 0.0
    virial = torch.zeros(3, 3, dtype=torch.float64)
    for event in evaluations:
        energy += event["energy"]
        virial += torch.tensor(event["virial"], dtype=torch.float64)
        for atom_id, atom_force, position in zip(
            event["owned"], event["forces"], event["positions"]
        ):
            assert force[atom_id] is None, (
                f"atom {atom_id} read out more than once"
            )
            force[atom_id] = atom_force
            positions[atom_id] = position
    assert all(item is not None for item in force), (
        "some global atoms lack owner output"
    )
    return energy, force, virial, positions


def _compare_model(
    events,
    index=1,
    *,
    cell=CELL,
    atom_types=ATOM_TYPES,
    rtol=RTOL,
    atol=ATOL,
):
    energy, force, virial, evaluated_positions = _assemble(
        events, index, num_atoms=len(atom_types)
    )
    direct = full_system_reference(evaluated_positions, cell, atom_types)
    _close(energy, direct.energy.item(), rtol=rtol, atol=atol)
    _close(force, direct.forces.tolist(), rtol=rtol, atol=atol)
    _close(virial, direct.virial.tolist(), rtol=rtol, atol=atol)
    return direct


def _global_edges(build):
    local_ids = build["owned"] + build["ghost"]
    return {
        (local_ids[target], local_ids[source])
        for target, source in build["edges"]
    }


def _read_force(directory, num_atoms=len(ATOM_TYPES)):
    raw = (directory / "frc.dat").read_bytes()
    width = num_atoms * 3
    assert len(raw) >= width * 4 and len(raw) % (width * 4) == 0
    return [
        list(row)
        for row in zip(
            *[iter(struct.unpack("<" + "f" * width, raw[-width * 4 :]))] * 3
        )
    ]


def _read_mdout(directory):
    lines = (directory / "mdout.txt").read_text(encoding="utf-8").splitlines()
    header = lines[0].split()
    values = [
        line.split() for line in lines[1:] if len(line.split()) == len(header)
    ]
    assert values, f"missing complete mdout row: {directory}"
    return dict(zip(header, map(float, values[-1])))


def _check_sponge_force_energy(plugin_dir, baseline_dir, reference):
    num_atoms = int(reference.forces.shape[0])
    plugin_force = _read_force(plugin_dir, num_atoms)
    baseline_force = _read_force(baseline_dir, num_atoms)
    difference = [
        [a - b for a, b in zip(row_a, row_b)]
        for row_a, row_b in zip(plugin_force, baseline_force)
    ]
    expected = [
        [row[0] + SEED_FORCE, row[1], row[2]]
        for row in reference.forces.tolist()
    ]
    _close(difference, expected)

    observed, baseline = _read_mdout(plugin_dir), _read_mdout(baseline_dir)
    _close(observed["eff_pot"] - baseline["eff_pot"], reference.energy.item())


def _check_sponge_outputs(plugin_dir, baseline_dir, reference, cell=CELL):
    _check_sponge_force_energy(plugin_dir, baseline_dir, reference)
    # mdout potential uses scientific precision; printed stress is rounded to
    # 0.01 bar, while strict virial parity is checked from rank-local FP32 logs.
    observed, baseline = _read_mdout(plugin_dir), _read_mdout(baseline_dir)
    volume = (
        torch.linalg.det(torch.as_tensor(cell, dtype=torch.float64))
        .abs()
        .item()
    )
    expected_pxx = reference.virial[0, 0].item() / volume * PRESSURE_CONVERSION
    assert abs((observed["Pxx"] - baseline["Pxx"]) - expected_pxx) <= 0.03


def _require_gpus(count):
    available = torch.cuda.device_count()
    if available < count:
        raise RuntimeError(
            f"requires {count} visible CUDA devices, found {available}; no parity result"
        )


def run_case(args):
    case = args.case
    if case in (
        "test_local_mp_single_gpu_parity",
        "test_local_mp_skin_rebuild",
    ):
        required = 1
    else:
        required = 2
    _require_gpus(required)
    args.work_root.mkdir(parents=True, exist_ok=True)
    if case in (
        "test_local_mp_single_gpu_parity",
        "test_local_mp_two_pp_rank_parity",
    ):
        pp = 1 if case == "test_local_mp_single_gpu_parity" else 2
        plugin_dir, events = _run(args, "plugin", ranks=pp)
        direct = _compare_model(events)
        baseline_dir, _ = _run(args, "baseline", ranks=pp, plugin=False)
        _check_sponge_outputs(plugin_dir, baseline_dir, direct)
        if case == "test_local_mp_two_pp_rank_parity":
            one_dir, one_events = _run(args, "one_pp_reference", ranks=1)
            _compare_model(one_events)
            one_energy, one_force, one_virial, _ = _assemble(one_events, 1)
            two_energy, two_force, two_virial, _ = _assemble(events, 1)
            _close(two_energy, one_energy)
            _close(two_force, one_force)
            _close(two_virial, one_virial)
            _close(_read_force(plugin_dir), _read_force(one_dir))
            _close(
                _read_mdout(plugin_dir)["eff_pot"],
                _read_mdout(one_dir)["eff_pot"],
            )
    elif case == "test_local_mp_md_halo_refresh":
        _, events = _run(
            args,
            "ordinary_md",
            ranks=2,
            steps=1,
            dt=0.001,
            positions=[
                (3.3, 1.0, 1.0),
                (1.2, 4.0, 1.0),
                (5.5, 1.0, 1.0),
                (8.0, 4.0, 1.0),
            ],
            # Atom 0 starts outside the right halo, then moves into its
            # physical cutoff while remaining owned by the left rank.
            velocities=[
                (70.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
            ],
            update_interval=100,
        )
        for index in (1, 2):
            _compare_model(events, index, rtol=2e-6, atol=2e-6)
        initial = _assemble(events, 1)[3]
        moved = _assemble(events, 2)[3]
        displacement = max(abs(b[0] - a[0]) for a, b in zip(initial, moved))
        assert displacement > 0.2
        assert moved[0][0] < 5.0
        right = [
            event
            for event in events
            if event["rank"] == 1 and event["event"] == "evaluate"
        ]
        assert 0 not in right[0]["ghost"]
        assert 0 in right[1]["ghost"]
        for rank in (0, 1):
            builds = [
                event
                for event in events
                if event["rank"] == rank and event["event"] == "build"
            ]
            assert len(builds) >= 2, f"rank {rank} did not refresh its halo"
            assert builds[1]["neighbor_epoch"] > builds[0]["neighbor_epoch"]
            evaluations = [
                event
                for event in events
                if event["rank"] == rank and event["event"] == "evaluate"
            ]
            assert [event["neighbor_epoch"] for event in evaluations[:2]] == [
                build["neighbor_epoch"] for build in builds[:2]
            ]
        right_builds = [
            event
            for event in events
            if event["rank"] == 1 and event["event"] == "build"
        ]
        assert (2, 0) not in _global_edges(right_builds[0])
        assert (2, 0) in _global_edges(right_builds[1])
    elif case == "test_local_mp_skin_rebuild":
        # Native NVE motion crosses half the model skin before the second
        # force. Pair (0, 1) starts beyond cutoff + skin, then enters cutoff.
        positions = (
            (0.4, 1.0, 1.0),
            (2.1, 1.0, 1.0),
            (4.8, 1.0, 1.0),
            (5.2, 1.0, 1.0),
        )
        _, events = _run(
            args,
            "native_motion",
            ranks=1,
            steps=1,
            dt=0.001,
            positions=positions,
            velocities=((35.0, 0.0, 0.0),) + ((0.0, 0.0, 0.0),) * 3,
            update_interval=100,
        )
        for index in (1, 2):
            _compare_model(events, index, rtol=2e-6, atol=2e-6)
        initial = _assemble(events, 1)[3]
        moved = _assemble(events, 2)[3]
        assert moved[0][0] - initial[0][0] > 0.2
        assert abs(initial[1][0] - initial[0][0]) > 1.6
        assert abs(moved[1][0] - moved[0][0]) < 1.2
        builds = [event for event in events if event["event"] == "build"]
        assert len(builds) >= 2, f"native skin motion did not rebuild: {builds}"
        assert builds[1]["epoch"] == builds[0]["epoch"]
        assert builds[1]["neighbor_epoch"] > builds[0]["neighbor_epoch"]
        assert builds[1]["count"] > builds[0]["count"]
        evaluations = [
            event for event in events if event["event"] == "evaluate"
        ]
        assert [event["neighbor_epoch"] for event in evaluations[:2]] == [
            build["neighbor_epoch"] for build in builds[:2]
        ]
        assert (0, 1) not in _global_edges(builds[0])
        assert (0, 1) in _global_edges(builds[1])
    else:
        raise ValueError(f"unknown CTest case: {case}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--sponge", required=True)
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--py", type=Path)
    parser.add_argument("--mpiexec", required=True)
    parser.add_argument("--numproc-flag", default="-n")
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args()
    if args.py is not None:
        args.py = args.py.resolve()
    try:
        run_case(args)
    except Exception:
        print(
            f"Local-MP case {args.case} failed; rank logs under {args.work_root}",
            file=sys.stderr,
        )
        raise


if __name__ == "__main__":
    main()

"""Real-runtime restart-only/trajectory regression with a generic arithmetic model.

Check restart output with and without trajectory output in the same build.
Restart at label1 is x2,v3/2; force evaluation3 independently observes x2.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import signal

import numpy as np
from test_local_mp_mpi import _write_input
from fixture_model import SEED_FORCE


def run_case(args, trajectory):
    out = args.work_root / f"r{args.ranks}-trajectory{trajectory}-{os.getpid()}"
    # Intentionally interleave domain ownership in global atom order.
    positions = [(5.2, 1, 1), (0.4, 1, 1), (9.6, 1, 1), (4.8, 1, 1)]
    types = [6, 8, 6, 8]
    initial_v = np.array([[0.02, -0.01, 0.005]] * 4, dtype=np.float32)
    _write_input(
        out,
        args,
        ranks=args.ranks,
        positions=positions,
        atom_types=types,
        velocities=initial_v.tolist(),
        steps=3,
        dt=0.001,
    )
    mdin = out / "mdin.spg.toml"
    text = mdin.read_text().replace(
        "write_trajectory_interval = 1",
        f"write_trajectory_interval = {trajectory}",
    )
    text = text.replace(
        "[DOM_DEC]", "write_restart_file_interval = 2\n[DOM_DEC]"
    )
    mdin.write_text(text)
    env = os.environ.copy()
    env["SPONGE_LOCAL_MP_LOG_DIR"] = str(out / "rank_logs")
    env["SPONGE_LOCAL_MP_ATOM_TYPES"] = json.dumps(types)
    package = Path(__file__).resolve().parents[3] / "plugins/prips"
    env["PYTHONPATH"] = str(package) + os.pathsep + env.get("PYTHONPATH", "")
    command = [
        args.mpiexec,
        "--bind-to",
        "none",
        "--mca",
        "coll_hcoll_enable",
        "0",
        "--oversubscribe",
        "-n",
        str(args.ranks),
        args.sponge,
        "-mdin",
        "mdin.spg.toml",
    ]
    with (
        (out / "stdout.log").open("w") as stdout,
        (out / "stderr.log").open("w") as stderr,
    ):
        process = subprocess.Popen(
            command,
            cwd=out,
            env=env,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=120)
        finally:
            # Only this fixture's MPI process group; never node-wide PID cleanup.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    assert returncode == 0, (out, returncode)
    frames = {}
    for path in (out / "rank_logs").glob("rank*.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["event"] != "evaluate":
                continue
            frame = frames.setdefault(row["count"], {})
            for gid, xyz, force in zip(
                row["owned"], row["positions"], row["forces"]
            ):
                assert gid not in frame
                frame[gid] = (xyz, force)
    for frame in frames.values():
        assert sorted(frame) == list(range(4))
    coords = np.loadtxt(out / "restart_coordinate.txt", skiprows=1, max_rows=4)
    velocity = np.loadtxt(out / "restart_velocity.txt", skiprows=1, max_rows=4)
    for name in ["coordinate", "velocity"]:
        header = (
            (out / f"restart_{name}.txt").read_text().splitlines()[0].split()
        )
        assert (
            int(header[2]) == 1 and abs(float(header[1]) - 2 * 0.001) <= 5.1e-11
        )
    expected_x = np.array([frames[3][i][0] for i in range(4)], dtype=np.float32)
    # Float32 acceleration and a single-rounded CUDA fused multiply-add for
    # velocity; the float64 intermediate reproduces this fixture's FMA result.
    # Force includes the fixture's native x seed.
    expected_v = initial_v.copy()
    dt = np.float32(0.001 * float(np.float32(20.455)))
    invmass = np.array([1 / 12, 1 / 16, 1 / 12, 1 / 16], dtype=np.float32)
    for call in [1, 2]:
        force = np.array(
            [frames[call][i][1] for i in range(4)], dtype=np.float32
        )
        force[:, 0] += np.float32(SEED_FORCE)
        acc = invmass[:, None] * force
        expected_v = (
            expected_v.astype(np.float64) + float(dt) * acc.astype(np.float64)
        ).astype(np.float32)
    # Legacy text has seven decimal places; no exact float round-trip claim.
    np.testing.assert_allclose(coords, expected_x, atol=5.1e-8, rtol=0)
    np.testing.assert_allclose(velocity, expected_v, atol=5.1e-8, rtol=0)
    assert np.max(abs(expected_x - np.array(positions))) > 1e-5
    report = dict(
        ranks=args.ranks,
        trajectory_interval=trajectory,
        restart_label=1,
        completed_updates=2,
        force_frame=3,
        coordinate_max_error=float(np.max(abs(coords - expected_x))),
        velocity_max_error=float(np.max(abs(velocity - expected_v))),
    )
    (out / "validated.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sponge", required=True)
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--mpiexec", required=True)
    parser.add_argument("--ranks", type=int, choices=[1, 2], required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--trajectory-interval", type=int, choices=[0, 2])
    args = parser.parse_args()
    args.work_root = args.work_root.resolve()
    for interval in (
        [0, 2]
        if args.trajectory_interval is None
        else [args.trajectory_interval]
    ):
        print(json.dumps(run_case(args, interval)))

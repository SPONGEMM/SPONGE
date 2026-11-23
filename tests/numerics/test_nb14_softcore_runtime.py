"""Native nb14 soft-core: two-particle kernel checks, no FEP sampling."""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import hashlib
import uuid
import numpy as np
import h5py


def run(
    engine, root, weight, distance, source, delta_override=None, bad_raw=None
):
    root.mkdir()
    c6 = 4 * 0.15 * 3.2**6
    c12 = c6 * 3.2**6
    delta = (
        0.5 * (1 - weight) ** 2 * 3.2**6
        if delta_override is None
        else delta_override
    )
    config = dict(
        md_name="nb14-test",
        mode="nve",
        pbc=True,
        dt=0.0,
        step_limit=1,
        cutoff=8.0,
        print_zeroth_frame=True,
        print_pressure=True,
        write_mdout_interval=1,
        write_information_interval=1,
        write_trajectory_interval=1,
        write_restart_file_interval=0,
        frc="frc.dat",
    )
    config["PM.MPI_size"] = 1
    data = {
        "mass": "2\n12\n12\n",
        "charge": "2\n0\n0\n",
        "coordinate": f"2\n1 1 1\n{1 + distance:.12g} 1 1\n32 32 32\n90 90 90\n",
        "velocity": "2\n0 0 0\n0 0 0\n",
    }
    for name, value in data.items():
        (root / f"{name}.txt").write_text(value)
        config[name + "_in_file"] = name + ".txt"
    if source in ("raw", "legacy"):
        (root / "nb14.txt").write_text(
            f"1\n0 1 {weight * c12:.12g} {weight * c6:.12g} 0.5\n"
        )
        config["nb14_extra_in_file"] = "nb14.txt"
        if source == "raw":
            (root / "soft.txt").write_text(bad_raw or f"1\n0 1 {delta:.12g}\n")
            config["nb14_softcore_in_file"] = "soft.txt"
    else:
        with h5py.File(root / "topology.h5", "w") as h5:
            for key, value in {
                "schema/name": "sponge.topology.h5",
                "schema/version": "sponge.input.v2",
                "identity/uuid": str(uuid.uuid4()),
                "topology/atom_order_hash": "two-atoms",
                "topology/topology_hash": hashlib.sha256(
                    f"{weight}:{delta}:{source}".encode()
                ).hexdigest(),
                "topology/forcefield_hash": "nb14-fixture",
            }.items():
                h5.create_dataset(key, data=value)
            topology_hash = h5["topology/topology_hash"][()].decode()
            h5.create_dataset("topology/atom_count", data=2)
            h5.create_dataset("atoms/mass", data=[12.0, 12.0])
            h5.create_dataset("atoms/charge", data=[0.0, 0.0])
            group = h5.create_group("forcefield/" + source)
            group.create_dataset(
                "atoms", data=np.array([[0, 1]], dtype=np.int32)
            )
            # Canonical materialized params are derivative coefficients;
            # nb14_extra uses unmultiplied C12/C6, matching legacy text.
            factors = (12, 6) if source == "nb14" else (1, 1)
            group.create_dataset(
                "params",
                data=[
                    [factors[0] * weight * c12, factors[1] * weight * c6, 0.5]
                ],
            )
            group.create_dataset("softcore_delta", data=[delta])
        config["input.h5.topology.path"] = "topology.h5"
        with h5py.File(root / "protocol.h5", "w") as h5:
            for key, value in {
                "schema/name": "sponge.protocol.h5",
                "schema/version": "sponge.input.v2",
                "identity/uuid": str(uuid.uuid4()),
                "identity/content_hash": "nb14-protocol",
                "protocol/topology_compatibility/topology_hash": topology_hash,
            }.items():
                h5.create_dataset(key, data=value)
        with h5py.File(root / "trajectory.h5", "w") as h5:
            for key, value in {
                "parameters/sponge/schema/name": "sponge.output.h5md",
                "parameters/sponge/schema/version": "sponge.output.v2",
                "identity/uuid": str(uuid.uuid4()),
                "run/atom_order_hash": "two-atoms",
            }.items():
                h5.create_dataset(key, data=value)
            h5.create_dataset("particles/all/step", data=[0])
            h5.create_dataset("particles/all/time", data=[0.0])
            for name, value in {
                "position": np.array(
                    [[[1, 1, 1], [1 + distance, 1, 1]]], dtype=np.float32
                ),
                "velocity": np.zeros((1, 2, 3), dtype=np.float32),
                "box/edges": np.array([np.eye(3) * 32], dtype=np.float32),
            }.items():
                h5.create_dataset(f"particles/all/{name}/value", data=value)
                h5[f"particles/all/{name}/step"] = h5["particles/all/step"]
                h5[f"particles/all/{name}/time"] = h5["particles/all/time"]
        config.update(
            {
                "mode": "rerun",
                "rerun_frame_limit": 1,
                "input.h5.protocol.path": "protocol.h5",
                "input.h5.trajectory.path": "trajectory.h5",
            }
        )
        for name in data:
            config.pop(name + "_in_file")
    (root / "mdin.spg.toml").write_text(
        "\n".join(f"{k}={json.dumps(v)}" for k, v in config.items()) + "\n"
    )
    result = subprocess.run(
        [engine, "-mdin", "mdin.spg.toml"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if bad_raw:
        assert result.returncode != 0, "bad raw softcore table accepted"
        return None
    assert result.returncode == 0, result.stdout + result.stderr
    lines = (root / "mdout.txt").read_text().splitlines()
    headers = [
        i
        for i, line in enumerate(lines)
        if "nb14_LJ" in line.split() and "Pxx" in line.split()
    ]
    assert headers, f"{source}: {lines}\n{result.stdout}\n{result.stderr}"
    header = headers[0]
    keys = lines[header].split()
    values = next(
        dict(zip(keys, map(float, line.split())))
        for line in lines[header + 1 :]
        if len(line.split()) == len(keys)
    )
    force = np.fromfile(root / "frc.dat", dtype=np.float32).reshape(-1, 2, 3)[0]
    s = distance**6 + delta
    expected_energy = weight * (c12 / s**2 - c6 / s)
    expected_force = weight * distance**5 * (6 * c6 / s**2 - 12 * c12 / s**3)
    assert np.isfinite(force).all()
    np.testing.assert_allclose(
        force,
        [[expected_force, 0, 0], [-expected_force, 0, 0]],
        rtol=3.0e-4,
        atol=0.003,
    )
    # mdout scalar energies/pressure are rounded to two decimal places.
    assert abs(values["nb14_LJ"] - expected_energy) <= 0.011 + 3.0e-4 * abs(
        expected_energy
    ), values
    expected_pressure = (
        -distance
        * expected_force
        / 32**3
        * float(np.float32(6.946827162543585e4))
    )
    # Rerun evaluates energies/forces but the MD pressure reduction is not run.
    # Check the actual virial via the RAW dt=0 MD cases; H5 rerun checks E/F parity.
    if source in ("raw", "legacy"):
        assert abs(values["Pxx"] - expected_pressure) <= 0.011 + 3.0e-4 * abs(
            expected_pressure
        ), values
    return force


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sponge", required=True)
    args = parser.parse_args()
    engine = str(Path(args.sponge).resolve())
    with tempfile.TemporaryDirectory(prefix="nb14-kernel-") as tmp:
        root = Path(tmp)
        count = 0
        for w, r in [
            (0.0, 0.0),
            (0.2, 0.0),
            (0.0, 0.001),
            (1.0e-6, 0.001),
            (0.2, 0.01),
            (0.5, 1.0),
            (0.9, 2.0),
            (1.0, 3.2),
        ]:
            results = []
            for source in ("raw", "nb14", "nb14_extra"):
                results.append(run(engine, root / str(count), w, r, source))
                count += 1
            for other in results[1:]:
                np.testing.assert_allclose(
                    results[0], other, rtol=1.0e-6, atol=1.0e-6
                )
        old = run(
            engine, root / str(count), 1.0, 3.2, "legacy", delta_override=0.0
        )
        count += 1
        new = run(
            engine, root / str(count), 1.0, 3.2, "raw", delta_override=0.0
        )
        count += 1
        np.testing.assert_allclose(old, new, rtol=1.0e-5, atol=1.0e-5)
        for bad in [
            "1\n1 0 3\n",
            "1\n0 1 -1\n",
            "1\n0 1 nan\n",
            "1\n0 1 inf\n",
            "0\n",
            "1\n0 1 3\njunk\n",
        ]:
            run(engine, root / str(count), 0.5, 1.0, "raw", bad_raw=bad)
            count += 1
        print(f"{count} native nb14 kernel/input cases passed")


if __name__ == "__main__":
    main()

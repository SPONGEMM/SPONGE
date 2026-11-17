"""Frozen independent ordinary LJ checks; no FEP sampling."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import uuid
import numpy as np
import h5py


def run(engine, root, branches, weights, distance, invalid=None, source="raw"):
    root.mkdir()
    c6a = 0.6 * 3.2**6
    c12a = c6a * 3.2**6
    c6b = 0.8 * 2.8**6
    c12b = c6b * 2.8**6
    config = dict(md_name="independent-sterics-test", mode="nve", pbc=True,
                  dt=0., step_limit=1, cutoff=8., print_zeroth_frame=True,
                  print_pressure=True, write_mdout_interval=1,
                  write_information_interval=1, write_trajectory_interval=1,
                  write_restart_file_interval=0, frc="frc.dat", lambda_lj=0.37,
                  soft_core_alpha=0.5, soft_core_powfer=2., solvent_LJ=False)
    config["PM.MPI_size"] = 1
    data = {"mass": "2\n12\n12\n", "charge": "2\n0\n0\n",
            "coordinate": f"2\n1 1 1\n{1+distance:.12g} 1 1\n32 32 32\n90 90 90\n",
            "velocity": "2\n0 0 0\n0 0 0\n",
            "LJ_soft_core": f"2 1 1\n{c12a:.12g}\n{c6a:.12g}\n{c12b:.12g}\n{c6b:.12g}\n0 0\n0 0\n",
            "LJ_soft_core_independent_sterics": invalid or
                f"2 {weights[0]:.12g} {weights[1]:.12g}\n0 {branches[0]}\n1 {branches[1]}\n"}
    if set(branches) == {1, 2}:
        data["exclude"] = "2 1\n1 1\n0\n"
    for name, value in data.items():
        (root / f"{name}.txt").write_text(value)
        config[name + "_in_file"] = name + ".txt"
    if source == "h5":
        topology_hash = str(uuid.uuid4())
        with h5py.File(root / "topology.h5", "w") as h5:
            for key, value in {"schema/name": "sponge.topology.h5", "schema/version": "sponge.input.v2",
                               "identity/uuid": str(uuid.uuid4()), "topology/atom_order_hash": "two-atoms",
                               "topology/topology_hash": topology_hash,
                               "topology/forcefield_hash": "independent-sterics-fixture"}.items():
                h5.create_dataset(key, data=value)
            h5.create_dataset("topology/atom_count", data=2)
            h5.create_dataset("atoms/mass", data=[12., 12.])
            h5.create_dataset("atoms/charge", data=[0., 0.])
            if set(branches) == {1, 2}:
                h5.create_dataset("topology/exclusions/offset", data=[0, 1, 1])
                h5.create_dataset("topology/exclusions/list", data=[1])
            group = h5.create_group("forcefield/lj_soft_core")
            for suffix, value in {"pair_AA": [c12a], "pair_AB": [c6a],
                                  "pair_BA": [c12b], "pair_BB": [c6b],
                                  "atom_type_A": np.zeros(2, dtype=np.int32),
                                  "atom_type_B": np.zeros(2, dtype=np.int32),
                                  "independent_sterics/atom_branch": np.array(branches, dtype=np.int32),
                                  "independent_sterics/weights": weights}.items():
                group.create_dataset(suffix, data=value)
        with h5py.File(root / "protocol.h5", "w") as h5:
            for key, value in {"schema/name": "sponge.protocol.h5", "schema/version": "sponge.input.v2",
                               "identity/uuid": str(uuid.uuid4()), "identity/content_hash": "sterics-protocol",
                               "protocol/topology_compatibility/topology_hash": topology_hash}.items():
                h5.create_dataset(key, data=value)
        with h5py.File(root / "trajectory.h5", "w") as h5:
            for key, value in {"parameters/sponge/schema/name": "sponge.output.h5md",
                               "parameters/sponge/schema/version": "sponge.output.v2",
                               "identity/uuid": str(uuid.uuid4()), "run/atom_order_hash": "two-atoms"}.items():
                h5.create_dataset(key, data=value)
            h5.create_dataset("particles/all/step", data=[0])
            h5.create_dataset("particles/all/time", data=[0.])
            for name, value in {"position": np.array([[[1, 1, 1], [1+distance, 1, 1]]], dtype=np.float32),
                                "velocity": np.zeros((1, 2, 3), dtype=np.float32),
                                "box/edges": np.array([np.eye(3)*32], dtype=np.float32)}.items():
                h5.create_dataset(f"particles/all/{name}/value", data=value)
                for axis in ("step", "time"):
                    h5[f"particles/all/{name}/{axis}"] = h5[f"particles/all/{axis}"]
        for name in data:
            config.pop(name + "_in_file")
        config.update({"mode": "rerun", "rerun_frame_limit": 1,
                       "input.h5.topology.path": "topology.h5",
                       "input.h5.protocol.path": "protocol.h5",
                       "input.h5.trajectory.path": "trajectory.h5"})
    (root / "mdin.spg.toml").write_text("\n".join(f"{k}={json.dumps(v)}" for k, v in config.items()) + "\n")
    result = subprocess.run([engine, "-mdin", "mdin.spg.toml"], cwd=root,
                            capture_output=True, text=True, timeout=30)
    (root / "stdout.txt").write_text(result.stdout + result.stderr)
    if invalid:
        assert result.returncode != 0, "malformed branch table accepted"
        return
    assert result.returncode == 0, result.stdout + result.stderr
    lines = (root / "mdout.txt").read_text().splitlines()
    header = next(i for i, line in enumerate(lines) if "LJ_soft_short" in line.split() and "Pxx" in line.split())
    keys = lines[header].split()
    values = next(dict(zip(keys, map(float, line.split()))) for line in lines[header + 1:] if len(line.split()) == len(keys))
    force = np.fromfile(root / "frc.dat", dtype=np.float32).reshape(-1, 2, 3)[0]
    def lj(c12, c6, w, sigma):
        if w == 0:
            return 0., 0.
        denom = distance**6 + 0.5 * (1-w)**2 * sigma**6
        return (w * (c12 / denom**2 - c6 / denom),
                w * distance**5 * (6*c6 / denom**2 - 12*c12 / denom**3))
    if set(branches) == {1, 2}:
        energy, radial = 0., 0.
    elif 1 in branches:
        energy, radial = lj(c12a, c6a, weights[0], 3.2)
    elif 2 in branches:
        energy, radial = lj(c12b, c6b, weights[1], 2.8)
    else:
        a, b = lj(c12a, c6a, 1, 3.2), lj(c12b, c6b, 1, 2.8)
        energy, radial = [(1-0.37)*x + 0.37*y for x, y in zip(a, b)]
    assert np.isfinite(force).all()
    np.testing.assert_allclose(force, [[radial, 0, 0], [-radial, 0, 0]], rtol=3.e-4, atol=0.003)
    assert abs(values["LJ_soft_short"] - energy) <= 0.011 + 3.e-4 * abs(energy), values
    # Isotropic tail virial is present in each diagonal. Remove it to compare
    # the pair's xx virial; yy has no pair contribution in this fixture.
    pair_pressure = -distance * radial / 32**3 * 6.946827162543585e4
    if source == "raw":
        assert abs(values["Pxx"] - values["Pyy"] - pair_pressure) <= 0.021 + 3.e-4 * abs(pair_pressure), values
    return force


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sponge", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    temporary = tempfile.TemporaryDirectory(prefix="independent-sterics-") if not args.output else None
    root = Path(args.output).resolve() if args.output else Path(temporary.name) / "cases"
    root.mkdir()
    engine = str(Path(args.sponge).resolve())
    count = 0
    for branches in [(0, 0), (0, 1), (1, 1), (0, 2), (2, 2), (1, 2)]:
        for weights in [(1., 0.), (0.8, 0.9), (0., 1.)]:
            forces = []
            for source in ("raw", "h5"):
                forces.append(run(engine, root / str(count), branches, weights, 2.7, source=source))
                count += 1
            np.testing.assert_allclose(forces[0], forces[1], rtol=1.e-5, atol=1.e-5)
    for branches, weights in [((0, 1), (0., 1.)), ((0, 2), (1., 0.2)), ((1, 2), (0.8, 0.9))]:
        for source in ("raw", "h5"):
            run(engine, root / str(count), branches, weights, 0., source=source)
            count += 1
    for invalid in ["2 .5 .6\n1 0\n0 1\n", "2 nan .6\n0 0\n1 1\n",
                    "2 .5 .6\n0 3\n1 1\n", "2 .5 .6\n0 0\n1 1\njunk\n",
                    "1 .5 .6\n0 0\n", "2 .5 inf\n0 0\n1 1\n"]:
        run(engine, root / str(count), (0, 1), (0.5, 0.6), 2.7, invalid)
        count += 1
    # Actual current-charge Coulomb must be unchanged when wA+wB=2.
    # Zero LJ isolates electrostatics from endpoint parameter interpolation.
    for distance, branches in [(2.7, (0, 1)), (0., (1, 2))]:
        case = root / str(count)
        run(engine, case, branches, (1., 1.), distance)
        (case / "charge.txt").write_text("2\n1\n-1\n")
        (case / "LJ_soft_core.txt").write_text("2 1 1\n0\n0\n0\n0\n0 0\n0 0\n")
        readings = []
        energies = []
        for independent in (True, False):
            if not independent:
                config = (case / "mdin.spg.toml").read_text().splitlines()
                (case / "mdin.spg.toml").write_text("\n".join(
                    line for line in config if not line.startswith("LJ_soft_core_independent_sterics_in_file=")) + "\n")
            result = subprocess.run([engine, "-mdin", "mdin.spg.toml"], cwd=case,
                                    capture_output=True, text=True, timeout=30)
            (case / f"charged-{independent}.log").write_text(result.stdout + result.stderr)
            assert result.returncode == 0, result.stdout + result.stderr
            force = np.fromfile(case / "frc.dat", dtype=np.float32).reshape(-1, 2, 3)[0]
            assert np.isfinite(force).all()
            readings.append(force)
            lines = (case / "mdout.txt").read_text().splitlines()
            header = next(i for i, line in enumerate(lines) if "eff_pot" in line.split())
            keys = lines[header].split()
            values = next(dict(zip(keys, map(float, line.split()))) for line in lines[header + 1:]
                          if len(line.split()) == len(keys))
            energies.append(values["eff_pot"])
            assert np.isfinite(energies[-1])
        np.testing.assert_allclose(readings[0], readings[1], rtol=3.e-5, atol=1.e-4)
        np.testing.assert_allclose(energies[0], energies[1], rtol=3.e-5, atol=1.e-4)
        count += 1
    print(f"{count} independent ordinary sterics kernel/input cases passed")
    if temporary is not None:
        temporary.cleanup()


if __name__ == "__main__":
    main()

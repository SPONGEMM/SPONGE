"""Compare native H5 restoration with the independently checked RAW path."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid

import h5py
import numpy as np


def main():
    engine = str(Path(sys.argv[1]).resolve())
    temporary = tempfile.TemporaryDirectory(prefix="ordinary-h5-runtime-") if len(sys.argv) == 2 else None
    root = Path(temporary.name) if temporary else Path(sys.argv[2]).resolve()
    root.mkdir(exist_ok=True)
    raw_root = root / "raw"
    subprocess.run([sys.executable, str(Path(__file__).with_name("test_ordinary_restoration_runtime.py")),
                    engine, str(raw_root)], check=True, timeout=120)
    reference = json.loads((raw_root / "restoration-report.json").read_text())
    count = 0
    for geometry, distance in [("near", 4.), ("beyond_cutoff", 10.), ("periodic_image", 20.)]:
        for name, ordinary, native in [("restored", 1., 0.), ("partial_excluded", .4, 0.),
                                        ("partial_native14", .4, .6)]:
            work = root / geometry / name
            work.mkdir(parents=True)
            identity = str(uuid.uuid4())
            with h5py.File(work / "topology.h5", "w") as h5:
                for key, value in {"schema/name": "sponge.topology.h5", "schema/version": "sponge.input.v2",
                                   "identity/uuid": str(uuid.uuid4()), "topology/atom_order_hash": "ordinary-pair",
                                   "topology/topology_hash": identity, "topology/forcefield_hash": "ordinary-fixture"}.items():
                    h5[key] = value
                h5["topology/atom_count"] = 2
                h5["atoms/mass"] = [12.011, 12.011]
                h5["atoms/charge"] = [4.2, -4.2]
                h5["topology/exclusions/offset"] = [0, 1, 1]
                h5["topology/exclusions/list"] = [1]
                for suffix, value in {"pair_AA": [59500.], "pair_AB": [120.],
                                      "pair_BA": [59500.], "pair_BB": [120.],
                                      "atom_type_A": np.zeros(2, dtype=np.int32),
                                      "atom_type_B": np.zeros(2, dtype=np.int32)}.items():
                    h5["forcefield/lj_soft_core/" + suffix] = value
                h5["forcefield/fep_core_ordinary/atoms"] = [[0, 1]]
                h5["forcefield/fep_core_ordinary/params"] = [[ordinary*59500., ordinary*120., ordinary]]
                if native:
                    h5["forcefield/nb14/atoms"] = [[0, 1]]
                    # Canonical nb14 H5 uses derivative coefficients, while
                    # ordinary restoration H5 explicitly uses physical C12/C6.
                    h5["forcefield/nb14/params"] = [[12*native*.5*59500., 6*native*.5*120., native*.8333333]]
            with h5py.File(work / "protocol.h5", "w") as h5:
                for key, value in {"schema/name": "sponge.protocol.h5", "schema/version": "sponge.input.v2",
                                   "identity/uuid": str(uuid.uuid4()), "identity/content_hash": "ordinary-protocol",
                                   "protocol/topology_compatibility/topology_hash": identity}.items():
                    h5[key] = value
            with h5py.File(work / "trajectory.h5", "w") as h5:
                for key, value in {"parameters/sponge/schema/name": "sponge.output.h5md",
                                   "parameters/sponge/schema/version": "sponge.output.v2",
                                   "identity/uuid": str(uuid.uuid4()), "run/atom_order_hash": "ordinary-pair"}.items():
                    h5[key] = value
                h5["particles/all/step"] = [0]
                h5["particles/all/time"] = [0.]
                for key, value in {"position": np.array([[[1, 2, 3], [1+distance, 2, 3]]], dtype=np.float32),
                                   "velocity": np.zeros((1, 2, 3), dtype=np.float32),
                                   "box/edges": np.array([np.eye(3)*24], dtype=np.float32)}.items():
                    h5[f"particles/all/{key}/value"] = value
                    for axis in ("step", "time"):
                        h5[f"particles/all/{key}/{axis}"] = h5[f"particles/all/{axis}"]
            config = {"md_name": "ordinary-h5", "mode": "rerun", "rerun_frame_limit": 1,
                      "input.h5.topology.path": "topology.h5", "input.h5.protocol.path": "protocol.h5",
                      "input.h5.trajectory.path": "trajectory.h5", "cutoff": 8., "lambda_lj": .5,
                      "print_pressure": True, "write_trajectory_interval": 1, "frc": "force.dat",
                      "mdout": "result.mdout", "PM.MPI_size": 1}
            (work / "mdin.spg.toml").write_text("\n".join(f"{k}={json.dumps(v)}" for k, v in config.items())+"\n")
            result = subprocess.run([engine, "-mdin", "mdin.spg.toml"], cwd=work,
                                    capture_output=True, text=True, timeout=30)
            (work / "engine.log").write_text(result.stdout+result.stderr)
            assert result.returncode == 0, result.stdout+result.stderr
            lines = (work / "result.mdout").read_text().splitlines()
            header = next(i for i, line in enumerate(lines) if "eff_pot" in line.split())
            keys = lines[header].split()
            values = next(dict(zip(keys, map(float, line.split()))) for line in lines[header+1:]
                          if len(line.split()) == len(keys))
            force = np.fromfile(work / "force.dat", dtype=np.float32).reshape(-1, 2, 3)[0]
            expected = reference[geometry]["states"][name]
            np.testing.assert_allclose(values["eff_pot"], expected["energy"], rtol=0, atol=2.e-5)
            raw_force = np.fromfile(raw_root / geometry / name / "one.frc", dtype=np.float32).reshape(-1, 2, 3)[0]
            np.testing.assert_allclose(force, raw_force, rtol=0, atol=2.e-5)
            count += 1
            if geometry == "near" and name == "partial_native14":
                (work / "duplicate.txt").write_text("1\n0 1 1 1 .4\n")
                result = subprocess.run([engine, "-mdin", "mdin.spg.toml", "-fep_core_ordinary_extra_in_file", "duplicate.txt"],
                                        cwd=work, capture_output=True, text=True, timeout=30)
                (work / "conflict.log").write_text(result.stdout+result.stderr)
                assert result.returncode != 0 and "both own fep_core_ordinary" in result.stdout+result.stderr
                count += 1
    print(f"{count} ordinary restoration H5 rerun/parity/conflict cases passed")
    if temporary:
        temporary.cleanup()


if __name__ == "__main__":
    main()

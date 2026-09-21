"""Real SPONGE execution gates for both RMSD H5 producers and conversion.

Set SPONGE_EXECUTABLE, XPONGE_PYTHON, and XPONGE_CPP_PYTHON to run both
producers. Optional XPONGE_SOURCE / XPONGE_CPP_SOURCE select source roots
(the latter points to XpongeCPP/src). All subprocess output stays in tmp_path.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tomllib

import h5py
import numpy as np
import pytest


@pytest.fixture(params=[("Xponge", "XPONGE"), ("XpongeCPP", "XPONGE_CPP")])
def producer(request):
    package, prefix = request.param
    python = os.environ.get(prefix + "_PYTHON")
    executable = os.environ.get("SPONGE_EXECUTABLE")
    if not python or not executable:
        pytest.skip(f"set {prefix}_PYTHON and SPONGE_EXECUTABLE for real RMSD execution")
    for path in (python, executable):
        assert Path(path).is_file() and os.access(path, os.X_OK), path
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONDONTWRITEBYTECODE="1")
    source = os.environ.get(prefix + "_SOURCE")
    if source:
        env["PYTHONPATH"] = source
    return package, python, str(Path(executable).resolve()), env


def _checked(command, cwd, env):
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True,
                            text=True, timeout=120, check=False)
    (cwd / "execution.log").write_text(result.stdout + "\n" + result.stderr)
    assert result.returncode == 0, f"{command}\n{result.stdout}\n{result.stderr}"


def _generate(producer, root, rotate):
    package, python, _, env = producer
    script = Path(__file__).with_name("rmsd_cv_e2e_producer.py")
    _checked([python, str(script), package, str(root), str(rotate).lower()], root, env)
    with np.load(root / "oracle.npz") as data:
        return {key: data[key] for key in data.files}


def _run(producer, case, *, rerun=True, steps=1, restart=None):
    _, _, executable, env = producer
    mdin = "mdin.legacy.spg.toml" if case.name == "legacy" else "mdin.bundled.spg.toml"
    command = [executable, "-mdin", mdin, "-mode", "rerun" if rerun else "nve",
               "-step_limit", str(steps + 1 if rerun else steps), "-dt", "0.000001",
               "-print_zeroth_frame", "1", "-write_information_interval", "1",
               "-write_mdout_interval", "1", "-write_trajectory_interval", "1",
               "-write_restart_file_interval", str(steps),
               "-mdout", "mdout.txt", "-mdinfo", "mdinfo.txt", "-frc", "forces.dat",
               "-output_h5_trajectory_path", "trajectory.h5",
               "-output_h5_trajectory_vds", "0"]
    if rerun:
        command += ["-crd", "frames.dat", "-box", "frames.box",
                    "-rerun_frame_limit", "2", "-rerun_start", "0", "-rerun_strip", "0"]
    else:
        command += ["-output_h5_restart_path", "final.spgr.h5", "-vel", "velocity.dat"]
    if restart is not None:
        command += ["-input_h5_restart_path", str(restart)]
    # SPONGE rejects duplicate mdin/CLI keys; write a single resolved launch
    # file instead of relying on command-line override semantics.
    settings = tomllib.loads((case / mdin).read_text())
    settings.update({key.removeprefix("-"): value
                     for key, value in zip(command[3::2], command[4::2])})
    (case / "run.spg.toml").write_text("\n".join(
        f"{key} = {json.dumps(value)}" for key, value in settings.items()
    ) + "\n")
    _checked([executable, "-mdin", "run.spg.toml"], case, env)
    with h5py.File(case / "trajectory.h5") as handle:
        return {
            "rmsd": handle["/observables/all/rmsd_cv/value"][...],
            "bias": handle["/observables/all/restrain_cv/value"][...],
            "force": handle["/particles/all/force/value"][...],
            "position": handle["/particles/all/position/value"][...],
        }


def _rmsd(positions, reference, rotate):
    points = np.asarray(positions, dtype=np.float64)
    target = np.asarray(reference, dtype=np.float64)
    points = points - points.mean(axis=0)
    target = target - target.mean(axis=0)
    if rotate:
        left, _, right = np.linalg.svd(target.T @ points)
        correction = np.diag([1.0, 1.0, np.linalg.det(left @ right)])
        target = target @ (left @ correction @ right)
    return np.sqrt(np.sum((points - target) ** 2) / len(points))


def _bias_force(positions, reference, rotate):
    # Finite differences of an independent NumPy/Kabsch energy oracle.
    points = np.asarray(positions, dtype=np.float64).copy()
    force = np.zeros_like(points)
    delta = 1e-5
    for index in np.ndindex(points.shape):
        original = points[index]
        points[index] = original + delta
        plus = 2.0 * (_rmsd(points, reference, rotate) - 0.2) ** 2
        points[index] = original - delta
        minus = 2.0 * (_rmsd(points, reference, rotate) - 0.2) ** 2
        points[index] = original
        force[index] = -(plus - minus) / (2 * delta)
    return force


@pytest.mark.parametrize("rotate", [False, True])
def test_rmsd_native_and_converted_legacy_value_and_force(producer, tmp_path, rotate):
    data = _generate(producer, tmp_path, rotate)
    results = {}
    for name in ("native", "baseline", "legacy"):
        case = tmp_path / name
        # SPONGE excludes the final rerun frame from trajectory output. Two
        # identical frames give one force snapshot at the exact oracle input.
        np.stack([data["positions"]] * 2).astype(np.float32).tofile(case / "frames.dat")
        np.savetxt(case / "frames.box", [[*np.diag(data["box"]), 90, 90, 90]] * 2)
        results[name] = _run(producer, case)
    selected = data["positions"][data["selection"]]
    expected = _rmsd(selected, data["reference"], rotate)
    for name in ("native", "baseline", "legacy"):
        np.testing.assert_allclose(results[name]["rmsd"], [expected] * 2, atol=6e-5, rtol=0)
        assert results[name]["force"].shape == (1, len(data["positions"]), 3)
    # Observables use the runtime's printed precision (4 decimals for CV,
    # 2 for bias energy); force trajectories retain float32 precision.
    np.testing.assert_allclose(results["native"]["bias"], [2 * (expected - 0.2) ** 2] * 2, atol=0.006)
    expected_force = np.zeros_like(data["positions"])
    expected_force[data["selection"]] = _bias_force(selected, data["reference"], rotate)
    difference = results["native"]["force"][0] - results["baseline"]["force"][0]
    np.testing.assert_allclose(difference, expected_force, atol=2e-3, rtol=2e-3)
    np.testing.assert_allclose(results["legacy"]["force"], results["native"]["force"],
                               atol=2e-2, rtol=2e-5)


@pytest.mark.parametrize("rotate", [False, True])
def test_rmsd_restart_matches_uninterrupted_nve(producer, tmp_path, rotate):
    data = _generate(producer, tmp_path, rotate)
    for name in ("continuous", "first", "resumed"):
        shutil.copytree(tmp_path / "native", tmp_path / name)
    continuous = _run(producer, tmp_path / "continuous", rerun=False, steps=4)
    _run(producer, tmp_path / "first", rerun=False, steps=2)
    restart = tmp_path / "first/final.spgr.h5"
    with h5py.File(restart) as handle:
        np.testing.assert_array_equal(
            handle["/parameters/restart/references/cv/rmsd_cv/coordinate"][...],
            data["reference"],
        )
        assert "/particles/all/velocity/value" in handle
    resumed = _run(producer, tmp_path / "resumed", rerun=False, steps=2, restart=restart)
    np.testing.assert_allclose(resumed["rmsd"][-1], continuous["rmsd"][-1], atol=1e-4, rtol=0)
    for field in ("position", "force"):
        assert len(resumed[field]) > 0 and len(continuous[field]) > 0
        np.testing.assert_allclose(resumed[field][-1], continuous[field][-1], atol=2e-3, rtol=2e-5)
    with h5py.File(tmp_path / "continuous/final.spgr.h5") as full, \
            h5py.File(tmp_path / "resumed/final.spgr.h5") as split:
        np.testing.assert_allclose(split["/particles/all/position/time"][...],
                                   full["/particles/all/position/time"][...],
                                   atol=1e-12, rtol=0)
        for field in ("position", "velocity"):
            np.testing.assert_allclose(split[f"/particles/all/{field}/value"][...],
                                       full[f"/particles/all/{field}/value"][...],
                                       atol=2e-5, rtol=2e-5)

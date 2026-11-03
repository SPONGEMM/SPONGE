"""Short real-executable regressions; set SPONGE_EXECUTABLE to the binary."""

import os
import struct
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def executable():
    path = os.environ.get("SPONGE_EXECUTABLE")
    if not path:
        pytest.skip("set SPONGE_EXECUTABLE to run SITS integration regressions")
    return str(Path(path).resolve())


def run_case(
    executable,
    case,
    *,
    selection=None,
    interaction="lj",
    weight=0.5,
    mode="observation",
):
    case.mkdir()
    files = {
        "coordinate.txt": "3\n1 2 3\n4 3 5\n6 5 4\n24 24 24 90 90 90\n",
        "mass.txt": "3\n12\n12\n12\n",
        "charge.txt": "3\n1\n-1\n0\n",
        "velocity.txt": "3\n0.1 0.2 0.3\n-0.1 0.1 0\n0 0 -0.1\n",
    }
    settings = {
        "md_name": "SITS regression",
        "mode": "nve",
        "dt": 0.0001,
        "step_limit": 3,
        "cutoff": 8,
        "skin": 1,
        "coordinate_in_file": "coordinate.txt",
        "mass_in_file": "mass.txt",
        "charge_in_file": "charge.txt",
        "velocity_in_file": "velocity.txt",
        "print_zeroth_frame": 1,
        "print_pressure": 1,
        "write_mdout_interval": 1,
        "write_information_interval": 1,
        "write_trajectory_interval": 1,
        "mdout": "mdout.txt",
        "frc": "force.dat",
    }
    if interaction == "lj":
        files["lj.txt"] = "3 1\n10000000\n1000\n0\n0\n0\n"
        settings["LJ_in_file"] = "lj.txt"
    else:
        # Both nonzero endpoints exercise the ordinary branch of the soft
        # kernel; a zero B endpoint exercises its softened-distance branch.
        endpoint = "5000000\n500" if interaction == "soft_regular" else "0\n0"
        files["soft.txt"] = (
            "3 1 1\n10000000\n1000\n" + endpoint + "\n0 0\n0 0\n0 0\n"
        )
        settings.update(LJ_soft_core_in_file="soft.txt", lambda_lj=0.4)
    if selection is not None:
        settings.update(
            SITS_mode=mode,
            SITS_atom_numbers=selection,
            SITS_cross_enhance_factor=weight,
        )
        if mode == "iteration":
            settings.update(
                SITS_k_numbers=4,
                SITS_T_low=280,
                SITS_T_high=420,
                SITS_record_interval=1,
                SITS_update_interval=2,
                target_temperature=300,
            )
    for name, body in files.items():
        (case / name).write_text(body)
    (case / "mdin.spg.toml").write_text(
        "\n".join(f"{key} = {value!r}" for key, value in settings.items())
        + "\n"
    )
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = subprocess.run(
        [executable, "-mdin", "mdin.spg.toml"],
        cwd=case,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    (case / "execution.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    lines = (case / "mdout.txt").read_text().splitlines()
    headers = lines[0].split()
    rows = [
        dict(zip(headers, map(float, line.split())))
        for line in lines[1:]
        if line.strip()
    ]
    assert len(rows) == 4
    raw = (case / "force.dat").read_bytes()
    forces = struct.unpack(f"{len(raw) // 4}f", raw)
    assert len(forces) == 36
    return rows, forces


@pytest.mark.parametrize("selection", ["ALL", "ITS"])
@pytest.mark.parametrize("mode", ["observation", "iteration"])
def test_full_system_energy(executable, tmp_path, selection, mode):
    rows, forces = run_case(
        executable, tmp_path / "sits", selection=selection, mode=mode
    )
    if mode == "observation":
        baseline, base_forces = run_case(executable, tmp_path / "base")
        assert forces == pytest.approx(base_forces, rel=1e-5, abs=1e-5)
        for before, after in zip(baseline, rows):
            assert after["SITS_bias"] == 0
            for column in (
                "eff_pot",
                "pressure",
                "Pxx",
                "Pyy",
                "Pzz",
                "Pxy",
                "Pxz",
                "Pyz",
            ):
                assert after[column] == pytest.approx(
                    before[column], rel=1e-5, abs=0.011
                )
    for row in rows:
        assert abs(row["potential"]) > 0.1
        # SITS prints two decimals; eff_pot retains more precision.
        assert row["SITS_AA_kAB"] == pytest.approx(row["eff_pot"], abs=0.011)


@pytest.mark.parametrize("interaction", ["lj", "soft_regular", "soft_core"])
@pytest.mark.parametrize("weight", [0.0, 0.5, 1.0])
def test_observation_preserves_dynamics(
    executable, tmp_path, interaction, weight
):
    baseline, base_forces = run_case(
        executable, tmp_path / "base", interaction=interaction
    )
    observed, forces = run_case(
        executable,
        tmp_path / "observed",
        selection=1,
        interaction=interaction,
        weight=weight,
    )
    assert forces == pytest.approx(base_forces, rel=1e-5, abs=1e-5)
    for before, after in zip(baseline, observed):
        assert after["SITS_fb"] == 1
        assert after["SITS_bias"] == 0
        # Charged systems must include direct Coulomb energy exactly once,
        # without contaminating the separate LJ or PME energy outputs.
        assert after["eff_pot"] == pytest.approx(
            before["eff_pot"], rel=1e-6, abs=1e-6
        )
        lj_column = "LJ" if interaction == "lj" else "LJ_soft"
        for column in (lj_column, "PM"):
            assert after[column] == pytest.approx(before[column], abs=0.011)
        for column in ("pressure", "Pxx", "Pyy", "Pzz", "Pxy", "Pxz", "Pyz"):
            assert after[column] == pytest.approx(
                before[column], rel=1e-5, abs=0.011
            ), column

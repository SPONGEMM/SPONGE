import json
import math
import os
import re
import shutil
import struct
import subprocess
import textwrap
from pathlib import Path

import pytest

KB_KCAL_PER_MOL_K = 0.00198716
REPO_ROOT = Path(__file__).resolve().parents[4]


def _write_native_case(
    case_dir,
    *,
    masses=(12.0, 16.0, 1.0, 0.0),
    velocities=None,
    mode="nve",
    seed=202607220001,
    enable_initial_velocity=True,
    target_temperature=300.0,
    settings_overrides=None,
    extra_root_lines=(),
):
    case_dir.mkdir(parents=True)
    atom_count = len(masses)
    coordinates = [
        (1.0 + 2.0 * atom, 2.0 + 0.5 * atom, 3.0 + 0.25 * atom)
        for atom in range(atom_count)
    ]
    (case_dir / "mass.txt").write_text(
        f"{atom_count}\n" + "\n".join(str(mass) for mass in masses) + "\n",
        encoding="utf-8",
    )
    (case_dir / "charge.txt").write_text(
        f"{atom_count}\n" + "0\n" * atom_count,
        encoding="utf-8",
    )
    coordinate_lines = [str(atom_count)]
    coordinate_lines.extend(
        " ".join(str(component) for component in coordinate)
        for coordinate in coordinates
    )
    coordinate_lines.extend(("1000 1000 1000", "90 90 90"))
    (case_dir / "coordinate.txt").write_text(
        "\n".join(coordinate_lines) + "\n", encoding="utf-8"
    )

    settings = {
        "md_name": case_dir.name,
        "mode": mode,
        "pbc": False,
        "step_limit": 1,
        "dt": 0,
        "cutoff": 8.0,
        "PM.MPI_size": 0,
        "target_temperature": target_temperature,
        "mass_in_file": "mass.txt",
        "charge_in_file": "charge.txt",
        "coordinate_in_file": "coordinate.txt",
        "print_zeroth_frame": True,
        "write_mdout_interval": 1,
        "write_information_interval": 1,
        "write_trajectory_interval": 1,
        "write_restart_file_interval": 0,
    }
    if mode != "rerun":
        settings["vel"] = "vel.dat"
    if velocities is not None:
        assert len(velocities) == atom_count
        velocity_lines = [str(atom_count)]
        velocity_lines.extend(
            " ".join(str(component) for component in velocity)
            for velocity in velocities
        )
        (case_dir / "velocity.txt").write_text(
            "\n".join(velocity_lines) + "\n", encoding="utf-8"
        )
        settings["velocity_in_file"] = "velocity.txt"
    if settings_overrides:
        settings.update(settings_overrides)

    lines = [f"{key} = {json.dumps(value)}" for key, value in settings.items()]
    lines.extend(extra_root_lines)
    if enable_initial_velocity:
        lines.extend(
            (
                "",
                "[initial_velocity]",
                'mode = "maxwell"',
                f"seed = {seed}",
            )
        )
    (case_dir / "mdin.spg.toml").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _run_case(case_dir):
    sponge_bin = os.environ.get("SPONGE_BIN", "SPONGE")
    if os.sep in sponge_bin and not os.path.isabs(sponge_bin):
        sponge_bin = os.path.abspath(sponge_bin)
    result = subprocess.run(
        [sponge_bin, "-mdin", "mdin.spg.toml"],
        cwd=case_dir,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    return result, result.stdout + "\n" + result.stderr


def _read_velocity_frame(case_dir, atom_count, name="vel.dat"):
    raw = (case_dir / name).read_bytes()
    assert len(raw) == 3 * atom_count * 4
    values = struct.unpack(f"={3 * atom_count}f", raw)
    return raw, [
        values[index : index + 3] for index in range(0, len(values), 3)
    ]


def _assert_succeeded(result, output):
    assert result.returncode == 0, output


def _final_freedom(output):
    match = re.search(r"INITIAL VELOCITY:.*final freedom (\d+)", output)
    assert match is not None, output
    return int(match.group(1))


def _temperature(velocities, masses, freedom):
    kinetic_energy = 0.5 * sum(
        mass * sum(component * component for component in velocity)
        for mass, velocity in zip(masses, velocities)
    )
    return 2.0 * kinetic_energy / (freedom * KB_KCAL_PER_MOL_K)


def _write_constrained_water_case(case_dir, constrain_mode):
    case_dir.mkdir()
    masses = (15.999, 1.008, 1.008)
    coordinates = (
        (10.0, 10.0, 10.0),
        (10.9572, 10.0, 10.0),
        (9.760013, 10.926627, 10.0),
    )
    hydrogen_distance = (
        sum(
            (coordinates[1][axis] - coordinates[2][axis]) ** 2
            for axis in range(3)
        )
        ** 0.5
    )
    (case_dir / "mass.txt").write_text(
        "3\n" + "\n".join(str(mass) for mass in masses) + "\n",
        encoding="utf-8",
    )
    (case_dir / "charge.txt").write_text("3\n0\n0\n0\n", encoding="utf-8")
    (case_dir / "coordinate.txt").write_text(
        "3\n"
        + "\n".join(
            " ".join(str(component) for component in coordinate)
            for coordinate in coordinates
        )
        + "\n1000 1000 1000\n90 90 90\n",
        encoding="utf-8",
    )
    (case_dir / "bond.txt").write_text(
        f"3\n0 1 0 0.9572\n0 2 0 0.9572\n1 2 0 {hydrogen_distance:.12g}\n",
        encoding="utf-8",
    )
    settings = {
        "md_name": case_dir.name,
        "mode": "nve",
        "pbc": False,
        "step_limit": 1,
        "dt": 0.001,
        "cutoff": 8.0,
        "PM.MPI_size": 0,
        "target_temperature": 315.0,
        "mass_in_file": "mass.txt",
        "charge_in_file": "charge.txt",
        "coordinate_in_file": "coordinate.txt",
        "bond_in_file": "bond.txt",
        "constrain_mode": constrain_mode,
        "settle_disable": constrain_mode == "SHAKE",
        "mdout": "mdout.txt",
        "print_zeroth_frame": True,
        "write_mdout_interval": 1,
        "write_information_interval": 1,
        "write_trajectory_interval": 0,
        "write_restart_file_interval": 0,
    }
    lines = [f"{key} = {json.dumps(value)}" for key, value in settings.items()]
    lines.extend(("", "[initial_velocity]", 'mode = "maxwell"', "seed = 29"))
    (case_dir / "mdin.spg.toml").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def test_missing_section_preserves_native_velocity_bits(tmp_path):
    velocities = (
        (0.125, -0.25, 0.5),
        (-0.75, 1.0, -1.25),
        (1.5, -1.75, 2.0),
        (-2.25, 2.5, -2.75),
    )
    case_dir = tmp_path / "native_velocity_passthrough"
    _write_native_case(
        case_dir,
        velocities=velocities,
        enable_initial_velocity=False,
    )

    result, output = _run_case(case_dir)
    _assert_succeeded(result, output)
    raw, _ = _read_velocity_frame(case_dir, len(velocities))
    expected = struct.pack(
        f"={3 * len(velocities)}f",
        *(component for velocity in velocities for component in velocity),
    )
    assert raw == expected
    assert "INITIAL VELOCITY:" not in output


def test_seed_reproducibility_uses_all_64_bits(tmp_path):
    first = tmp_path / "seed_first"
    repeated = tmp_path / "seed_repeated"
    high_bits_differ = tmp_path / "seed_high_bits_differ"
    _write_native_case(first, seed=1)
    _write_native_case(repeated, seed=1)
    _write_native_case(high_bits_differ, seed=1 + (1 << 32))

    outputs = []
    frames = []
    for case_dir in (first, repeated, high_bits_differ):
        result, output = _run_case(case_dir)
        _assert_succeeded(result, output)
        outputs.append(output)
        frames.append(_read_velocity_frame(case_dir, 4)[0])

    assert frames[0] == frames[1]
    assert frames[0] != frames[2]
    assert "seed 1," in outputs[0]
    assert f"seed {1 + (1 << 32)}," in outputs[2]


def test_maxwell_velocity_has_zero_com_zero_mass_and_target_energy(tmp_path):
    masses = (12.0, 16.0, 1.0, 0.0)
    target = 337.0
    case_dir = tmp_path / "maxwell_invariants"
    _write_native_case(
        case_dir,
        masses=masses,
        seed=9223372036854775807,
        target_temperature=target,
    )

    result, output = _run_case(case_dir)
    _assert_succeeded(result, output)
    _, velocities = _read_velocity_frame(case_dir, len(masses))
    freedom = _final_freedom(output)

    assert velocities[-1] == (0.0, 0.0, 0.0)
    momentum = [
        sum(mass * velocity[axis] for mass, velocity in zip(masses, velocities))
        for axis in range(3)
    ]
    momentum_scale = max(
        1.0,
        sum(
            mass * abs(component)
            for mass, velocity in zip(masses, velocities)
            for component in velocity
        ),
    )
    assert (
        max(abs(component) for component in momentum) <= 2.0e-7 * momentum_scale
    )
    assert _temperature(velocities, masses, freedom) == pytest.approx(
        target, rel=1.0e-5
    )
    assert "projected" in output
    assert f"final {target:g} K" in output


def test_step_zero_temperature_schedule_controls_initial_energy(tmp_path):
    target = 425.0
    case_dir = tmp_path / "step_zero_schedule"
    _write_native_case(
        case_dir,
        target_temperature=300.0,
        extra_root_lines=(
            'target_temperature_schedule_mode = "step"',
            "target_temperature_schedule_steps = "
            "[{ step = 0, value = 425.0 }, { step = 1, value = 300.0 }]",
        ),
    )

    result, output = _run_case(case_dir)
    _assert_succeeded(result, output)
    _, velocities = _read_velocity_frame(case_dir, 4)
    assert _temperature(
        velocities, (12.0, 16.0, 1.0, 0.0), _final_freedom(output)
    ) == pytest.approx(target, rel=1.0e-5)
    assert "target 425 K" in output


@pytest.mark.parametrize("constrain_mode", ["SETTLE", "SHAKE"])
def test_maxwell_finalize_with_water_constraints(tmp_path, constrain_mode):
    case_dir = tmp_path / constrain_mode.lower()
    _write_constrained_water_case(case_dir, constrain_mode)

    result, output = _run_case(case_dir)
    _assert_succeeded(result, output)
    expected_backend_log = (
        "rigid triangle numbers is 1"
        if constrain_mode == "SETTLE"
        else "END INITIALIZING SHAKE"
    )
    assert expected_backend_log in output
    assert _final_freedom(output) == 6
    final_match = re.search(
        r"INITIAL VELOCITY: projected [^ ]+ K, final ([^ ]+) K", output
    )
    assert final_match is not None, output
    assert float(final_match.group(1)) == pytest.approx(315.0, rel=1.0e-6)

    mdout_lines = (
        (case_dir / "mdout.txt").read_text(encoding="utf-8").splitlines()
    )
    assert len(mdout_lines) >= 2, output
    assert all(math.isfinite(float(value)) for value in mdout_lines[1].split())


@pytest.mark.parametrize("mode", ["minimization", "rerun"])
def test_maxwell_is_rejected_for_non_dynamical_modes(tmp_path, mode):
    case_dir = tmp_path / mode
    settings = {}
    if mode == "minimization":
        settings.update(
            {
                "minimization_dynamic_dt": 0,
                "minimization_max_move": 0.1,
                "dt": 0.001,
            }
        )
    else:
        settings.update(
            {
                "crd": "trajectory.dat",
                "box": "box.txt",
                "rerun_need_box_update": 1,
            }
        )
    _write_native_case(
        case_dir,
        mode=mode,
        settings_overrides=settings,
    )
    if mode == "rerun":
        (case_dir / "trajectory.dat").write_bytes(
            struct.pack("=12f", *([0.0] * 12))
        )
        (case_dir / "box.txt").write_text(
            "1000 1000 1000 90 90 90\n", encoding="utf-8"
        )

    result, output = _run_case(case_dir)
    assert result.returncode != 0, output
    assert (
        "mode=maxwell is not defined for minimization or rerun mode" in output
    )


def test_maxwell_runs_through_amber_loader(tmp_path):
    source = (
        REPO_ROOT
        / "benchmarks/comparison/reference/amber/statics/alanine_dipeptide_gb1"
    )
    case_dir = tmp_path / "amber_loader"
    case_dir.mkdir()
    for name in ("system.parm7", "system.rst7"):
        shutil.copy2(source / name, case_dir / name)
    lines = [
        'md_name = "initial velocity amber loader"',
        'mode = "nve"',
        "pbc = false",
        "step_limit = 1",
        "dt = 0",
        "cutoff = 999.0",
        'amber_parm7 = "system.parm7"',
        'amber_rst7 = "system.rst7"',
        'vel = "vel.dat"',
        "write_mdout_interval = 1",
        "write_information_interval = 1",
        "write_trajectory_interval = 1",
        "write_restart_file_interval = 0",
        "",
        "[initial_velocity]",
        'mode = "maxwell"',
        "seed = 17",
    ]
    (case_dir / "mdin.spg.toml").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    result, output = _run_case(case_dir)
    _assert_succeeded(result, output)
    raw, _ = _read_velocity_frame(case_dir, 22)
    assert any(raw)
    assert "INITIAL VELOCITY: mode=maxwell, seed 17" in output


def _gro_atom(resid, resname, atomname, atomnr, xyz):
    x, y, z = xyz
    return (
        f"{resid:5d}{resname:<5}{atomname:>5}{atomnr:5d}"
        f"{x:8.3f}{y:8.3f}{z:8.3f}"
    )


def test_maxwell_runs_through_direct_gromacs_loader(tmp_path):
    case_dir = tmp_path / "gromacs_loader"
    case_dir.mkdir()
    (case_dir / "topol.top").write_text(
        textwrap.dedent(
            """
            [ defaults ]
            1 2 yes 1.0 1.0

            [ atomtypes ]
            A A 12.0 0.0 A 0.3 0.4184
            B B 16.0 0.0 A 0.3 0.4184

            [ moleculetype ]
            PAIR 0

            [ atoms ]
            1 A 1 PAIR A 1 0.0 12.0
            2 B 1 PAIR B 2 0.0 16.0

            [ system ]
            initial velocity direct loader test

            [ molecules ]
            PAIR 1
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    (case_dir / "conf.gro").write_text(
        "\n".join(
            (
                "initial velocity direct loader test",
                "2",
                _gro_atom(1, "PAIR", "A", 1, (0.0, 0.0, 0.0)),
                _gro_atom(1, "PAIR", "B", 2, (0.5, 0.0, 0.0)),
                " 100.00000 100.00000 100.00000",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (case_dir / "mdin.spg.toml").write_text(
        textwrap.dedent(
            """
            md_name = "initial velocity direct GROMACS loader"
            mode = "nve"
            pbc = false
            step_limit = 1
            dt = 0
            cutoff = 8.0
            gromacs_top = "topol.top"
            gromacs_gro = "conf.gro"
            vel = "vel.dat"
            write_mdout_interval = 1
            write_information_interval = 1
            write_trajectory_interval = 1
            write_restart_file_interval = 0

            [initial_velocity]
            mode = "maxwell"
            seed = 23
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    result, output = _run_case(case_dir)
    _assert_succeeded(result, output)
    _, velocities = _read_velocity_frame(case_dir, 2)
    freedom = _final_freedom(output)
    assert _temperature(velocities, (12.0, 16.0), freedom) == pytest.approx(
        300.0, rel=1.0e-5
    )
    assert "INITIAL VELOCITY: mode=maxwell, seed 23" in output

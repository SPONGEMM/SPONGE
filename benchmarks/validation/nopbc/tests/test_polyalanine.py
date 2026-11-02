import math
import shutil
from pathlib import Path

import numpy as np
import pytest

from benchmarks.utils import Extractor, Runner
from benchmarks.validation.thermostat.tests.utils import (
    read_mass_values,
    write_velocity_file_for_temperature,
)

POLYALANINE_RESIDUES = 12
PREFIX = "polyala12"
OPEN_BOX_LENGTH = 999.0
GLOBAL_X_SHIFT = 1100.0


def _require_serial_open_boundary(mpi_np):
    if mpi_np not in (None, 1):
        pytest.skip("SPONGE NOPBC currently supports one MPI rank only")


def _build_polyalanine12(output_dir):
    import Xponge
    import Xponge.forcefield.amber.ff14sb  # noqa: F401
    from Xponge.forcefield.special import gb

    molecule = Xponge.ResidueType.get_type("ACE")
    alanine = Xponge.ResidueType.get_type("ALA")
    for _ in range(POLYALANINE_RESIDUES):
        molecule += alanine
    molecule += Xponge.ResidueType.get_type("NME")
    gb.set_gb_radius(molecule)

    built = Xponge.save_sponge_input(molecule, PREFIX, dirname=output_dir)
    ca_indices = tuple(
        built.atoms.index(
            next(atom for atom in residue.atoms if atom.name == "CA")
        )
        for residue in built.residues[1:-1]
    )
    assert len(built.residues) == POLYALANINE_RESIDUES + 2
    assert len(ca_indices) == POLYALANINE_RESIDUES
    return {
        "path": Path(output_dir),
        "atom_count": len(built.atoms),
        "ca_indices": ca_indices,
    }


@pytest.fixture(scope="module")
def polyalanine12_template(outputs_path):
    template_dir = outputs_path / "polyalanine12" / "_xponge_template"
    if template_dir.exists():
        shutil.rmtree(template_dir)
    template_dir.mkdir(parents=True)
    return _build_polyalanine12(template_dir)


def _prepare_case(outputs_path, template, run_name):
    case_dir = outputs_path / "polyalanine12" / run_name
    if case_dir.exists():
        shutil.rmtree(case_dir)
    shutil.copytree(template["path"], case_dir)
    return case_dir


def _load_coordinates(path):
    tokens = Path(path).read_text(encoding="utf-8").split()
    atom_count = int(tokens[0])
    coordinates = np.asarray(
        [float(value) for value in tokens[1 : 1 + 3 * atom_count]],
        dtype=np.float64,
    ).reshape(atom_count, 3)
    return tokens, coordinates


def _translate_coordinates(path, shift):
    tokens, coordinates = _load_coordinates(path)
    coordinates += np.asarray(shift, dtype=np.float64)
    flat = coordinates.reshape(-1)
    tokens[1 : 1 + flat.size] = [f"{value:.8f}" for value in flat]
    Path(path).write_text("\n".join(tokens) + "\n", encoding="utf-8")
    return coordinates


def _write_virtual_center_cv(
    case_dir,
    ca_indices,
    *,
    steer_weight=None,
    restrain_weight=None,
    restrain_reference=None,
    metadynamics=False,
):
    midpoint = len(ca_indices) // 2
    head_atoms = " ".join(str(index) for index in ca_indices[:midpoint])
    tail_atoms = " ".join(str(index) for index in ca_indices[midpoint:])
    lines = [
        "print",
        "{",
        "    CV = head_x chain_span",
        "}",
        "head_center",
        "{",
        "    vatom_type = center_of_mass",
        f"    atom = {head_atoms}",
        "}",
        "tail_center",
        "{",
        "    vatom_type = center_of_mass",
        f"    atom = {tail_atoms}",
        "}",
        "head_x",
        "{",
        "    CV_type = position_x",
        "    atom = head_center",
        "}",
        "chain_span",
        "{",
        "    CV_type = distance",
        "    atom = head_center tail_center",
        "}",
    ]
    if steer_weight is not None:
        lines.extend(
            [
                "steer",
                "{",
                "    CV = chain_span",
                f"    weight = {steer_weight}",
                "}",
            ]
        )
    if restrain_weight is not None:
        lines.extend(
            [
                "restrain",
                "{",
                "    CV = chain_span",
                f"    weight = {restrain_weight}",
                f"    reference = {restrain_reference}",
                "}",
            ]
        )
    if metadynamics:
        lines.extend(
            [
                "meta",
                "{",
                "    Ndim = 1",
                "    CV = chain_span",
                "    CV_minimal = 0.0",
                "    CV_maximum = 100.0",
                "    CV_period = 0.0",
                "    CV_grid = 200",
                "    CV_sigma = 0.5",
                "    height = 0.2",
                "    potential_update_interval = 1",
                "}",
            ]
        )
    (case_dir / "cv.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_open_mdin(
    case_dir,
    *,
    mode="nve",
    step_limit=0,
    dt=0.0,
    cv_file=None,
    default_prefix=True,
    extra_lines=(),
):
    lines = [
        'md_name = "NOPBC polyalanine12 validation"',
        f'mode = "{mode}"',
        f"step_limit = {step_limit}",
        f"dt = {dt}",
        "cutoff = 999.0",
        "pbc = false",
        'frc = "frc.dat"',
        'crd = "mdcrd.dat"',
        "dont_check_input = 1",
        "print_zeroth_frame = 1",
        "write_mdout_interval = 1",
        "write_information_interval = 1",
        "write_trajectory_interval = 1",
    ]
    if default_prefix:
        lines.insert(6, f'default_in_file_prefix = "{PREFIX}"')
    if cv_file is not None:
        lines.append(f'cv_in_file = "{cv_file}"')
    lines.extend(extra_lines)
    (case_dir / "mdin.spg.toml").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _read_coordinate_trajectory(path, atom_count):
    raw = np.fromfile(path, dtype=np.float32)
    frame_width = atom_count * 3
    assert raw.size > 0
    assert raw.size % frame_width == 0
    return raw.reshape((-1, atom_count, 3)).astype(np.float64)


def _read_hydrogen_constraints(case_dir):
    masses = np.asarray(
        read_mass_values(case_dir / f"{PREFIX}_mass.txt"), dtype=np.float64
    )
    lines = (
        (case_dir / f"{PREFIX}_bond.txt")
        .read_text(encoding="utf-8")
        .splitlines()[1:]
    )
    constraints = []
    for line in lines:
        atom_i, atom_j, _, target = line.split()[:4]
        atom_i = int(atom_i)
        atom_j = int(atom_j)
        if 0.0 < masses[atom_i] < 3.3 or 0.0 < masses[atom_j] < 3.3:
            constraints.append((atom_i, atom_j, float(target)))
    assert constraints
    return masses, constraints


def test_xponge_polyalanine12_nopbc_nvt_shake_gb_keeps_open_coordinates(
    outputs_path, polyalanine12_template, mpi_np
):
    _require_serial_open_boundary(mpi_np)
    case_dir = _prepare_case(
        outputs_path, polyalanine12_template, "nvt_shake_gb"
    )
    coordinate_path = case_dir / f"{PREFIX}_coordinate.txt"
    translated = _translate_coordinates(
        coordinate_path, (GLOBAL_X_SHIFT, 0.0, 0.0)
    )
    _write_virtual_center_cv(case_dir, polyalanine12_template["ca_indices"])
    masses, constraints = _read_hydrogen_constraints(case_dir)
    write_velocity_file_for_temperature(
        case_dir / "initial_velocity.txt",
        masses,
        temperature=300.0,
        seed=2026,
        degrees_of_freedom=3 * len(masses) - len(constraints),
    )
    _write_open_mdin(
        case_dir,
        mode="nvt",
        step_limit=20,
        dt=0.0002,
        cv_file="cv.txt",
        extra_lines=(
            'thermostat = "middle_langevin"',
            "thermostat_tau = 0.1",
            "thermostat_seed = 2026",
            "target_temperature = 300.0",
            'velocity_in_file = "initial_velocity.txt"',
            'constrain_mode = "SHAKE"',
        ),
    )

    Runner.run_sponge(case_dir, timeout=1200, mpi_np=mpi_np)

    rows = Extractor.parse_mdout_rows(
        case_dir / "mdout.txt",
        ("potential", "temperature", "gb", "head_x", "chain_span"),
        int_columns=(),
    )
    assert len(rows) == 21
    assert all(
        math.isfinite(float(value)) for row in rows for value in row.values()
    )

    ca_indices = polyalanine12_template["ca_indices"]
    midpoint = len(ca_indices) // 2
    head_indices = np.asarray(ca_indices[:midpoint], dtype=int)
    tail_indices = np.asarray(ca_indices[midpoint:], dtype=int)
    expected_head_x = float(np.mean(translated[head_indices, 0]))
    expected_span = float(
        np.linalg.norm(
            np.mean(translated[head_indices], axis=0)
            - np.mean(translated[tail_indices], axis=0)
        )
    )
    assert rows[0]["head_x"] == pytest.approx(expected_head_x, abs=5.0e-3)
    assert rows[0]["chain_span"] == pytest.approx(expected_span, abs=5.0e-3)

    trajectory = _read_coordinate_trajectory(
        case_dir / "mdcrd.dat", polyalanine12_template["atom_count"]
    )
    assert np.isfinite(trajectory).all()
    assert float(np.min(trajectory[:, :, 0])) > OPEN_BOX_LENGTH

    final_coordinates = trajectory[-1]
    constraint_errors = [
        abs(
            np.linalg.norm(
                final_coordinates[atom_i] - final_coordinates[atom_j]
            )
            - target
        )
        for atom_i, atom_j, target in constraints
    ]
    assert max(constraint_errors) < 2.0e-3


def test_nopbc_virtual_center_bias_and_meta_redistribute_force(
    outputs_path, polyalanine12_template, mpi_np
):
    _require_serial_open_boundary(mpi_np)
    baseline_dir = _prepare_case(
        outputs_path, polyalanine12_template, "virtual_center_baseline"
    )
    _write_virtual_center_cv(baseline_dir, polyalanine12_template["ca_indices"])
    _write_open_mdin(baseline_dir, cv_file="cv.txt")
    Runner.run_sponge(baseline_dir, timeout=1200, mpi_np=mpi_np)

    coordinates = _load_coordinates(baseline_dir / f"{PREFIX}_coordinate.txt")[
        1
    ]
    ca_indices = polyalanine12_template["ca_indices"]
    midpoint = len(ca_indices) // 2
    head_indices = np.asarray(ca_indices[:midpoint], dtype=int)
    tail_indices = np.asarray(ca_indices[midpoint:], dtype=int)
    span = float(
        np.linalg.norm(
            np.mean(coordinates[head_indices], axis=0)
            - np.mean(coordinates[tail_indices], axis=0)
        )
    )

    biased_dir = _prepare_case(
        outputs_path, polyalanine12_template, "virtual_center_bias_meta"
    )
    _write_virtual_center_cv(
        biased_dir,
        ca_indices,
        steer_weight=1.25,
        restrain_weight=2.0,
        restrain_reference=span + 0.5,
        metadynamics=True,
    )
    _write_open_mdin(biased_dir, step_limit=1, dt=0.0, cv_file="cv.txt")
    Runner.run_sponge(biased_dir, timeout=1200, mpi_np=mpi_np)

    baseline_force = Extractor.extract_sponge_forces(
        baseline_dir, polyalanine12_template["atom_count"]
    )
    biased_force = Extractor.extract_sponge_forces(
        biased_dir, polyalanine12_template["atom_count"]
    )
    force_delta = biased_force - baseline_force
    selected = np.asarray(ca_indices, dtype=int)
    unselected = np.ones(polyalanine12_template["atom_count"], dtype=bool)
    unselected[selected] = False
    assert float(np.linalg.norm(force_delta[selected])) > 1.0e-3
    np.testing.assert_allclose(force_delta[unselected], 0.0, atol=3.0e-3)
    np.testing.assert_allclose(np.sum(force_delta, axis=0), 0.0, atol=5.0e-3)
    np.testing.assert_allclose(
        np.sum(force_delta[selected[:midpoint]], axis=0),
        -np.sum(force_delta[selected[midpoint:]], axis=0),
        atol=5.0e-3,
    )

    rows = Extractor.parse_mdout_rows(
        biased_dir / "mdout.txt",
        ("chain_span", "steer_cv", "restrain_cv", "meta"),
        int_columns=(),
    )
    assert len(rows) == 2
    assert all(
        math.isfinite(float(value)) for row in rows for value in row.values()
    )
    assert rows[-1]["meta"] != pytest.approx(0.0, abs=1.0e-6)


def _sits_lines(atom_selection):
    return (
        'SITS_mode = "empirical"',
        f"SITS_atom_numbers = {atom_selection}",
        "SITS_T_low = 280.0",
        "SITS_T_high = 420.0",
    )


def test_nopbc_full_system_sits_runs_on_polyalanine12(
    outputs_path, polyalanine12_template, mpi_np
):
    _require_serial_open_boundary(mpi_np)
    case_dir = _prepare_case(
        outputs_path, polyalanine12_template, "full_system_sits_all"
    )
    _write_open_mdin(
        case_dir,
        step_limit=2,
        dt=0.0,
        extra_lines=_sits_lines('"ALL"'),
    )

    Runner.run_sponge(case_dir, timeout=1200, mpi_np=mpi_np)

    rows = Extractor.parse_mdout_rows(
        case_dir / "mdout.txt",
        ("SITS_AA_kAB", "SITS_bias", "SITS_fb"),
        int_columns=(),
    )
    assert len(rows) == 3
    assert all(
        math.isfinite(float(value)) for row in rows for value in row.values()
    )


def test_nopbc_selective_sits_is_rejected_on_polyalanine12(
    outputs_path, polyalanine12_template, mpi_np, capsys
):
    _require_serial_open_boundary(mpi_np)
    case_dir = _prepare_case(
        outputs_path, polyalanine12_template, "selective_sits_rejected"
    )
    _write_open_mdin(
        case_dir,
        extra_lines=_sits_lines("3"),
    )

    with pytest.raises(RuntimeError):
        Runner.run_sponge(case_dir, timeout=1200, mpi_np=mpi_np)
    captured = capsys.readouterr()
    assert "selective SITS requires periodic boundary conditions" in (
        captured.out + captured.err
    )


def test_nopbc_rejects_multiple_mpi_ranks(
    outputs_path, polyalanine12_template, mpi_np, capsys
):
    if mpi_np is None or mpi_np < 2:
        pytest.skip("run with --mpi=2 or greater to validate NOPBC rejection")
    case_dir = _prepare_case(
        outputs_path, polyalanine12_template, "multiple_mpi_ranks_rejected"
    )
    _write_open_mdin(case_dir)

    with pytest.raises(RuntimeError):
        Runner.run_sponge(case_dir, timeout=1200, mpi_np=mpi_np)
    captured = capsys.readouterr()
    assert "NOPBC can not be used in Multi-Process mode" in (
        captured.out + captured.err
    )

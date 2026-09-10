import math

import numpy as np
import pytest

from benchmarks.utils import Extractor, Outputer, Runner
from benchmarks.validation.cv.tests.utils import (
    ANGLE_ATOMS,
    DISTANCE_ATOMS,
    PHI_ATOMS,
    PSI_ATOMS,
    RMSD_ATOMS,
    compute_angle,
    compute_dihedral,
    compute_distance,
    compute_kabsch_rmsd,
    compute_tabulated_distance,
    load_coordinates,
    perturb_coordinates,
    rewrite_coordinate_for_open_boundary,
    write_displacement_cv_file,
    write_extended_cv_file,
    write_rmsd_reference_file,
    write_validation_mdin,
)
from benchmarks.validation.utils import parse_mdout_column

OPEN_BOX_LENGTH = 1000.0
OPEN_SHIFT_X = 600.0
ATOM_COUNT = 2129


def _require_serial_open_boundary(mpi_np):
    if mpi_np not in (None, 1):
        pytest.skip("SPONGE NOPBC currently supports one MPI rank only")


def _prepare_open_case(
    statics_path,
    outputs_path,
    mpi_np,
    run_name,
    *,
    shifted_atom=DISTANCE_ATOMS[1],
):
    case_dir = Outputer.prepare_output_case(
        statics_path=statics_path,
        outputs_path=outputs_path,
        case_name="alanine_dipeptide_phi_psi",
        mpi_np=mpi_np,
        run_name=run_name,
    )
    rewrite_coordinate_for_open_boundary(
        case_dir / "sys_flexible_coordinate.txt",
        shifted_atom=shifted_atom,
        shift_x=OPEN_SHIFT_X,
    )
    return case_dir


def test_nopbc_cv_uses_cartesian_displacement(
    statics_path, outputs_path, mpi_np
):
    _require_serial_open_boundary(mpi_np)
    case_dir = _prepare_open_case(
        statics_path, outputs_path, mpi_np, "nopbc_cartesian_displacement"
    )
    write_displacement_cv_file(case_dir)
    write_validation_mdin(case_dir, cutoff=100.0)
    with (case_dir / "mdin.spg.toml").open("a", encoding="utf-8") as mdin:
        mdin.write("pbc = false\n")

    Runner.run_sponge(case_dir, timeout=1200, mpi_np=mpi_np)

    coordinates = load_coordinates(case_dir / "sys_flexible_coordinate.txt")
    p0, p1 = (coordinates[index] for index in DISTANCE_ATOMS)
    expected = {
        "distance": compute_distance(coordinates, DISTANCE_ATOMS),
        "dx": p1[0] - p0[0],
        "dy": p1[1] - p0[1],
        "dz": p1[2] - p0[2],
    }
    assert expected["dx"] > OPEN_BOX_LENGTH / 2.0
    periodic_dx = expected["dx"] - OPEN_BOX_LENGTH
    periodic_distance = math.sqrt(
        periodic_dx**2 + expected["dy"] ** 2 + expected["dz"] ** 2
    )
    for column, expected_value in expected.items():
        actual = float(parse_mdout_column(case_dir / "mdout.txt", column)[0])
        assert math.isfinite(actual)
        assert actual == pytest.approx(expected_value, abs=5.0e-3)
        periodic_value = {
            "distance": periodic_distance,
            "dx": periodic_dx,
            "dy": expected["dy"],
            "dz": expected["dz"],
        }[column]
        if column in ("distance", "dx"):
            assert actual != pytest.approx(periodic_value, abs=5.0e-3)


def test_nopbc_supported_cv_types_match_cartesian_geometry(
    statics_path, outputs_path, mpi_np
):
    _require_serial_open_boundary(mpi_np)
    case_dir = _prepare_open_case(
        statics_path,
        outputs_path,
        mpi_np,
        "nopbc_supported_cv_types",
        shifted_atom=None,
    )
    coordinates = load_coordinates(case_dir / "sys_flexible_coordinate.txt")
    reference_coordinates = perturb_coordinates(coordinates, RMSD_ATOMS)
    write_rmsd_reference_file(
        reference_coordinates, case_dir / "rmsd_ref.txt", RMSD_ATOMS
    )
    write_extended_cv_file(case_dir)
    cv_path = case_dir / "cv.txt"
    cv_text = cv_path.read_text(encoding="utf-8")
    cv_text = cv_text.replace(
        "CV = distance angle phi psi combo tab_distance_linear rmsd_ala",
        "CV = px py pz distance angle phi psi combo "
        "tab_distance_linear rmsd_ala",
    )
    cv_text += (
        "px\n{\n    CV_type = position_x\n    atom = 6\n}\n"
        "py\n{\n    CV_type = position_y\n    atom = 6\n}\n"
        "pz\n{\n    CV_type = position_z\n    atom = 6\n}\n"
    )
    cv_path.write_text(cv_text, encoding="utf-8")
    write_validation_mdin(case_dir, cutoff=100.0)
    with (case_dir / "mdin.spg.toml").open("a", encoding="utf-8") as mdin:
        mdin.write("pbc = false\n")

    Runner.run_sponge(case_dir, timeout=1200, mpi_np=mpi_np)

    point = coordinates[6]
    distance = compute_distance(coordinates, DISTANCE_ATOMS)
    angle = compute_angle(coordinates, ANGLE_ATOMS)
    expected = {
        "px": point[0],
        "py": point[1],
        "pz": point[2],
        "distance": distance,
        "angle": angle,
        "phi": compute_dihedral(coordinates, PHI_ATOMS),
        "psi": compute_dihedral(coordinates, PSI_ATOMS),
        "combo": distance + 0.5 * angle,
        "tab_distance_linear": compute_tabulated_distance(distance),
        "rmsd_ala": compute_kabsch_rmsd(
            [reference_coordinates[index] for index in RMSD_ATOMS],
            [coordinates[index] for index in RMSD_ATOMS],
        ),
    }
    for column, expected_value in expected.items():
        actual = float(parse_mdout_column(case_dir / "mdout.txt", column)[0])
        assert math.isfinite(actual)
        assert actual == pytest.approx(expected_value, abs=5.0e-4)


def _write_position_bias_cv_file(
    case_dir, *, steer_weight=None, restrain_weight=None, reference=None
):
    lines = [
        "print",
        "{",
        "    CV = px",
        "}",
        "px",
        "{",
        "    CV_type = position_x",
        "    atom = 6",
        "}",
    ]
    if steer_weight is not None:
        lines.extend(
            ["steer", "{", "    CV = px", f"    weight = {steer_weight}", "}"]
        )
    if restrain_weight is not None:
        lines.extend(
            [
                "restrain",
                "{",
                "    CV = px",
                f"    weight = {restrain_weight}",
                f"    reference = {reference}",
                "}",
            ]
        )
    (case_dir / "cv.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_nopbc_steer_and_restrain_apply_cartesian_force(
    statics_path, outputs_path, mpi_np
):
    _require_serial_open_boundary(mpi_np)
    baseline_dir = _prepare_open_case(
        statics_path,
        outputs_path,
        mpi_np,
        "nopbc_position_bias_baseline",
        shifted_atom=None,
    )
    _write_position_bias_cv_file(baseline_dir)
    write_validation_mdin(baseline_dir, cutoff=100.0)
    with (baseline_dir / "mdin.spg.toml").open("a", encoding="utf-8") as mdin:
        mdin.write("pbc = false\n")
    Runner.run_sponge(baseline_dir, timeout=1200, mpi_np=mpi_np)

    coordinates = load_coordinates(baseline_dir / "sys_flexible_coordinate.txt")
    px = coordinates[6][0]
    steer_weight = 2.0
    restrain_weight = 1.5
    reference = px + 0.5
    biased_dir = _prepare_open_case(
        statics_path,
        outputs_path,
        mpi_np,
        "nopbc_position_bias_combined",
        shifted_atom=None,
    )
    _write_position_bias_cv_file(
        biased_dir,
        steer_weight=steer_weight,
        restrain_weight=restrain_weight,
        reference=reference,
    )
    write_validation_mdin(biased_dir, cutoff=100.0)
    with (biased_dir / "mdin.spg.toml").open("a", encoding="utf-8") as mdin:
        mdin.write("pbc = false\n")
    Runner.run_sponge(biased_dir, timeout=1200, mpi_np=mpi_np)

    baseline_force = Extractor.extract_sponge_forces(baseline_dir, ATOM_COUNT)
    biased_force = Extractor.extract_sponge_forces(biased_dir, ATOM_COUNT)
    expected_delta = np.zeros_like(baseline_force)
    expected_delta[6, 0] = -steer_weight - 2.0 * restrain_weight * (
        px - reference
    )
    np.testing.assert_allclose(
        biased_force - baseline_force,
        expected_delta,
        rtol=1.0e-5,
        atol=3.0e-3,
    )

    steer_energy = float(
        parse_mdout_column(biased_dir / "mdout.txt", "steer_cv")[0]
    )
    restrain_energy = float(
        parse_mdout_column(biased_dir / "mdout.txt", "restrain_cv")[0]
    )
    # Legacy mdout formats both bias energies with two decimal places.
    assert steer_energy == pytest.approx(steer_weight * px, abs=1.0e-2)
    assert restrain_energy == pytest.approx(
        restrain_weight * (px - reference) ** 2, abs=1.0e-2
    )


def test_nopbc_metadynamics_runs_on_cartesian_position_cv(
    statics_path, outputs_path, mpi_np
):
    _require_serial_open_boundary(mpi_np)
    case_dir = _prepare_open_case(
        statics_path,
        outputs_path,
        mpi_np,
        "nopbc_position_metadynamics",
        shifted_atom=None,
    )
    (case_dir / "cv.txt").write_text(
        "print\n"
        "{\n"
        "    CV = px\n"
        "}\n"
        "px\n"
        "{\n"
        "    CV_type = position_x\n"
        "    atom = 6\n"
        "}\n"
        "meta\n"
        "{\n"
        "    Ndim = 1\n"
        "    CV = px\n"
        "    CV_minimal = 0.0\n"
        "    CV_maximum = 100.0\n"
        "    CV_period = 0.0\n"
        "    CV_grid = 100\n"
        "    CV_sigma = 1.0\n"
        "    height = 0.2\n"
        "    potential_update_interval = 1\n"
        "}\n",
        encoding="utf-8",
    )
    write_validation_mdin(case_dir, step_limit=1, dt=0.0, cutoff=100.0)
    with (case_dir / "mdin.spg.toml").open("a", encoding="utf-8") as mdin:
        mdin.write("pbc = false\n")

    Runner.run_sponge(case_dir, timeout=1200, mpi_np=mpi_np)

    px_values = parse_mdout_column(case_dir / "mdout.txt", "px")
    meta_values = parse_mdout_column(case_dir / "mdout.txt", "meta")
    assert len(px_values) == 2
    assert len(meta_values) == 2
    assert all(math.isfinite(float(value)) for value in px_values)
    assert all(math.isfinite(float(value)) for value in meta_values)
    assert float(meta_values[-1]) != pytest.approx(0.0, abs=1.0e-6)


@pytest.mark.parametrize("cv_type", ["scaled_position_x", "box_length_x"])
def test_nopbc_rejects_box_dependent_cv(
    statics_path, outputs_path, mpi_np, capsys, cv_type
):
    _require_serial_open_boundary(mpi_np)
    case_dir = _prepare_open_case(
        statics_path,
        outputs_path,
        mpi_np,
        f"nopbc_{cv_type}_rejected",
    )
    atom_line = "    atom = 4\n" if cv_type.startswith("scaled_") else ""
    (case_dir / "cv.txt").write_text(
        "print\n{\n    CV = rejected\n}\n"
        f"rejected\n{{\n    CV_type = {cv_type}\n{atom_line}}}\n",
        encoding="utf-8",
    )
    write_validation_mdin(case_dir, cutoff=100.0)
    with (case_dir / "mdin.spg.toml").open("a", encoding="utf-8") as mdin:
        mdin.write("pbc = false\n")

    with pytest.raises(RuntimeError):
        Runner.run_sponge(case_dir, timeout=1200, mpi_np=mpi_np)
    captured = capsys.readouterr()
    expected_error = (
        f"CV type '{cv_type}' requires periodic boundary conditions"
    )
    assert expected_error in captured.out + captured.err


def _write_nopbc_sits_mdin(case_dir, atom_selection):
    write_validation_mdin(case_dir, cutoff=100.0, cv_file="unused-cv.txt")
    mdin_path = case_dir / "mdin.spg.toml"
    text = mdin_path.read_text(encoding="utf-8")
    text = text.replace('cv_in_file = "unused-cv.txt"\n', "")
    text += (
        "pbc = false\n"
        'SITS_mode = "empirical"\n'
        f"SITS_atom_numbers = {atom_selection}\n"
        "SITS_T_low = 280.0\n"
        "SITS_T_high = 420.0\n"
    )
    mdin_path.write_text(text, encoding="utf-8")


@pytest.mark.parametrize("atom_selection", ['"ALL"', '"ITS"'])
def test_nopbc_supports_full_system_sits(
    statics_path, outputs_path, mpi_np, atom_selection
):
    _require_serial_open_boundary(mpi_np)
    case_dir = _prepare_open_case(
        statics_path,
        outputs_path,
        mpi_np,
        f"nopbc_full_system_sits_{atom_selection.strip(chr(34)).lower()}",
    )
    _write_nopbc_sits_mdin(case_dir, atom_selection)

    Runner.run_sponge(case_dir, timeout=1200, mpi_np=mpi_np)

    for column in ("SITS_AA_kAB", "SITS_bias", "SITS_fb"):
        actual = float(parse_mdout_column(case_dir / "mdout.txt", column)[0])
        assert math.isfinite(actual)


def test_nopbc_rejects_selective_sits(
    statics_path, outputs_path, mpi_np, capsys
):
    _require_serial_open_boundary(mpi_np)
    case_dir = _prepare_open_case(
        statics_path, outputs_path, mpi_np, "nopbc_selective_sits_rejected"
    )
    _write_nopbc_sits_mdin(case_dir, "10")

    with pytest.raises(RuntimeError):
        Runner.run_sponge(case_dir, timeout=1200, mpi_np=mpi_np)
    captured = capsys.readouterr()
    assert "selective SITS requires periodic boundary conditions" in (
        captured.out + captured.err
    )

import math
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from benchmarks.utils import Extractor, Runner


def _case_directory(outputs_path, name, mpi_np):
    suffix = f"mpi{mpi_np}" if mpi_np is not None else "serial"
    case_dir = outputs_path / "virtual_atoms" / f"{name}_{suffix}"
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True)
    return case_dir


def _write_vector_file(path, count, rows):
    Path(path).write_text(
        f"{count}\n"
        + "".join(" ".join(str(value) for value in row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_case(case_dir, coordinates, charges, records, *, pbc):
    atom_count = len(coordinates)
    assert len(charges) == atom_count
    box_length = 10.0 if pbc else 999.0
    cutoff = 4.0 if pbc else 100.0
    _write_vector_file(
        case_dir / "mass.txt", atom_count, ((1.0,) for _ in charges)
    )
    _write_vector_file(
        case_dir / "charge.txt", atom_count, ((q,) for q in charges)
    )
    _write_vector_file(
        case_dir / "velocity.txt",
        atom_count,
        ((0.0, 0.0, 0.0) for _ in charges),
    )
    coordinate_lines = [f"{atom_count} 0.0"]
    coordinate_lines.extend(
        " ".join(str(value) for value in xyz) for xyz in coordinates
    )
    coordinate_lines.extend(
        [f"{box_length} {box_length} {box_length}", "90.0 90.0 90.0"]
    )
    (case_dir / "coordinate.txt").write_text(
        "\n".join(coordinate_lines) + "\n", encoding="utf-8"
    )
    (case_dir / "virtual_atom.txt").write_text(
        "\n".join(records) + "\n", encoding="utf-8"
    )
    (case_dir / "mdin.spg.toml").write_text(
        "\n".join(
            [
                'md_name = "virtual atom boundary validation"',
                'mode = "nve"',
                f"pbc = {'true' if pbc else 'false'}",
                "step_limit = 1",
                "dt = 0.0",
                f"cutoff = {cutoff}",
                "skin = 0.4",
                'mass_in_file = "mass.txt"',
                'charge_in_file = "charge.txt"',
                'coordinate_in_file = "coordinate.txt"',
                'velocity_in_file = "velocity.txt"',
                'virtual_atom_in_file = "virtual_atom.txt"',
                'crd = "mdcrd.dat"',
                'frc = "frc.dat"',
                "dont_check_input = 1",
                "print_zeroth_frame = 0",
                "write_mdout_interval = 1",
                "write_information_interval = 1",
                "write_trajectory_interval = 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _run(case_dir, atom_count, mpi_np):
    Runner.run_sponge(
        case_dir,
        timeout=1200,
        mpi_np=mpi_np,
        sponge_cmd=os.environ.get("SPONGE_EXECUTABLE"),
    )
    coordinates = np.fromfile(case_dir / "mdcrd.dat", dtype=np.float32)
    assert coordinates.size >= atom_count * 3
    coordinates = coordinates[-atom_count * 3 :].reshape(atom_count, 3)
    forces = Extractor.extract_sponge_forces(case_dir, atom_count)
    assert np.isfinite(coordinates).all()
    assert np.isfinite(forces).all()
    return coordinates.astype(np.float64), forces


def _periodic_delta(left, right, box_length):
    delta = np.asarray(left, dtype=np.float64) - np.asarray(
        right, dtype=np.float64
    )
    return delta - np.floor(delta / box_length + 0.5) * box_length


def _expected_position(kind, coordinates, pbc):
    boundary_delta = (
        (lambda left, right: _periodic_delta(left, right, 10.0))
        if pbc
        else (lambda left, right: np.asarray(left) - np.asarray(right))
    )
    r1, r2 = (
        np.asarray(coordinates[index], dtype=np.float64) for index in (0, 1)
    )
    if kind == 1:
        return r1 + 0.25 * boundary_delta(r2, r1)
    r3 = np.asarray(coordinates[2], dtype=np.float64)
    if kind == 2:
        return r1 + 0.2 * boundary_delta(r2, r1) + 0.3 * boundary_delta(r3, r1)
    direction = boundary_delta(r2, r1) + 0.5 * boundary_delta(r3, r2)
    return r1 + 0.5 * direction / np.linalg.norm(direction)


def test_type0_virtual_atom_reflects_force_and_clears_target(
    outputs_path, mpi_np
):
    if mpi_np not in (None, 1):
        pytest.skip("SPONGE NOPBC currently supports one MPI rank only")
    coordinates = [
        (100.0, 100.0, 3.0),
        (0.0, 0.0, 0.0),
        (104.0, 102.0, 8.0),
    ]
    charges = [0.0, 1.0, -1.0]
    case_dir = _case_directory(outputs_path, "type0_nopbc", mpi_np)
    _write_case(case_dir, coordinates, charges, ["0 1 0 2.0"], pbc=False)

    output_coordinates, forces = _run(case_dir, len(coordinates), mpi_np)

    np.testing.assert_allclose(
        output_coordinates[1], (100.0, 100.0, 1.0), atol=2.0e-5
    )
    np.testing.assert_allclose(forces[1], 0.0, atol=1.0e-7)
    virtual_force = -forces[2]
    np.testing.assert_allclose(
        forces[0],
        (virtual_force[0], virtual_force[1], -virtual_force[2]),
        rtol=2.0e-5,
        atol=2.0e-5,
    )


def test_type2_shared_sources_accumulate_atomically(outputs_path, mpi_np):
    if mpi_np not in (None, 1):
        pytest.skip("SPONGE NOPBC currently supports one MPI rank only")
    coordinates = [
        (0.0, 0.0, 0.0),
        (4.0, 0.0, 0.0),
        (0.0, 4.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    ]
    charges = [0.0, 0.0, 0.0, 1.0, -1.0]
    records = [
        "2 3 0 1 2 0.2 0.3",
        "2 4 0 1 2 0.4 0.1",
    ]
    case_dir = _case_directory(outputs_path, "type2_shared_sources", mpi_np)
    _write_case(case_dir, coordinates, charges, records, pbc=False)

    _, forces = _run(case_dir, len(coordinates), mpi_np)

    np.testing.assert_allclose(forces[3:5], 0.0, atol=1.0e-7)
    np.testing.assert_allclose(forces[0], 0.0, atol=2.0e-5)
    np.testing.assert_allclose(forces[1] + forces[2], 0.0, atol=2.0e-5)
    assert float(np.linalg.norm(forces[1])) > 1.0e-6


@pytest.mark.parametrize("kind", (1, 2, 3))
@pytest.mark.parametrize("pbc", (False, True), ids=("nopbc", "pbc"))
def test_virtual_atom_coordinates_and_redistributed_forces(
    outputs_path, mpi_np, kind, pbc
):
    if not pbc and mpi_np not in (None, 1):
        pytest.skip("SPONGE NOPBC currently supports one MPI rank only")
    if kind == 1:
        coordinates = (
            [(9.0, 0.0, 0.0), (0.5, 0.0, 0.0), (4.0, 0.0, 0.0), (7.0, 0.0, 0.0)]
            if pbc
            else [
                (900.0, 0.0, 0.0),
                (100.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (650.0, 0.0, 0.0),
            ]
        )
        target, external = 2, 3
        records = ["1 2 0 1 0.25"]
        weights = (0.75, 0.25)
    else:
        coordinates = (
            [
                (9.0, 0.0, 0.0),
                (0.5, 0.0, 0.0),
                (8.0, 1.0, 0.0),
                (4.0, 0.0, 0.0),
                (7.0, 0.0, 0.0),
            ]
            if pbc
            else [
                (900.0, 0.0, 0.0),
                (100.0, 0.0, 0.0),
                (700.0, 100.0, 0.0),
                (0.0, 0.0, 0.0),
                ((850.0 if kind == 3 else 650.0), 0.0, 0.0),
            ]
        )
        target, external = 3, 4
        records = ["2 3 0 1 2 0.2 0.3" if kind == 2 else "3 3 0 1 2 0.5 0.5"]
        weights = (0.5, 0.2, 0.3) if kind == 2 else None
    charges = [0.0] * len(coordinates)
    charges[target] = 1.0
    charges[external] = -1.0
    case_dir = _case_directory(
        outputs_path, f"type{kind}_{'pbc' if pbc else 'nopbc'}", mpi_np
    )
    _write_case(case_dir, coordinates, charges, records, pbc=pbc)

    output_coordinates, forces = _run(case_dir, len(coordinates), mpi_np)

    np.testing.assert_allclose(
        output_coordinates[target],
        _expected_position(kind, coordinates, pbc),
        atol=2.0e-5,
    )
    np.testing.assert_allclose(forces[target], 0.0, atol=1.0e-7)
    source_count = 2 if kind == 1 else 3
    redistributed_force = np.sum(forces[:source_count], axis=0)
    assert float(np.linalg.norm(redistributed_force)) > 1.0e-6
    if not pbc:
        np.testing.assert_allclose(
            redistributed_force + forces[external], 0.0, atol=2.0e-5
        )
    if weights is not None:
        for source, weight in enumerate(weights):
            np.testing.assert_allclose(
                forces[source],
                weight * redistributed_force,
                rtol=2.0e-5,
                atol=2.0e-5,
            )


def test_virtual_atom_layers_are_input_order_independent(outputs_path, mpi_np):
    coordinates = [
        (0.0, 0.0, 0.0),
        (4.0, 0.0, 0.0),
        (5.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (4.0, 0.0, 0.0),
    ]
    charges = [0.0, 0.0, 0.0, 0.0, 1.0, -1.0]
    records = [
        "1 4 3 2 0.5",  # Child deliberately precedes its virtual source.
        "1 3 0 1 0.25",
    ]
    case_dir = _case_directory(outputs_path, "unordered_layers_pbc", mpi_np)
    _write_case(case_dir, coordinates, charges, records, pbc=True)

    output_coordinates, forces = _run(case_dir, len(coordinates), mpi_np)

    np.testing.assert_allclose(
        _periodic_delta(output_coordinates[3], (1.0, 0.0, 0.0), 10.0),
        0.0,
        atol=2.0e-5,
    )
    np.testing.assert_allclose(
        _periodic_delta(output_coordinates[4], (3.0, 0.0, 0.0), 10.0),
        0.0,
        atol=2.0e-5,
    )
    np.testing.assert_allclose(forces[3:5], 0.0, atol=1.0e-7)
    child_force = np.sum(forces[:3], axis=0)
    assert float(np.linalg.norm(child_force)) > 1.0e-6
    for source, weight in enumerate((0.375, 0.125, 0.5)):
        np.testing.assert_allclose(
            forces[source], weight * child_force, rtol=2.0e-5, atol=2.0e-5
        )


def test_degenerate_type3_geometry_is_rejected(outputs_path, mpi_np, capsys):
    if mpi_np not in (None, 1):
        pytest.skip("failure diagnostic is validated on one rank")
    coordinates = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (3.0, 0.0, 0.0),
    ]
    charges = [0.0, 0.0, 0.0, 1.0, -1.0]
    case_dir = _case_directory(outputs_path, "degenerate_type3", mpi_np)
    _write_case(
        case_dir, coordinates, charges, ["3 3 0 1 2 1.0 -1.0"], pbc=True
    )

    with pytest.raises(RuntimeError):
        _run(case_dir, len(coordinates), mpi_np)
    captured = capsys.readouterr()
    assert "degenerate type-3 geometry" in captured.out + captured.err

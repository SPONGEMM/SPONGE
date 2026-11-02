"""ReaxFF EEQ must not disable the PP/CV force exchange."""

import os
import shutil
import signal
import subprocess
from pathlib import Path

import numpy as np
import pytest

from benchmarks.validation.utils import parse_mdout_rows

PARAMETERS = (
    Path(__file__).resolve().parents[4]
    / "tests/h5_bundle/fixtures/input_matrix/full_contract_rerun/legacy_input"
)


def _run_case(root, name, *, ranks, with_cv):
    case = root / name
    case.mkdir()
    for filename in ("reaxff.txt", "reaxff_type.txt"):
        shutil.copy2(PARAMETERS / filename, case / filename)
    (case / "coordinate.txt").write_text("2\n5 5 5\n6 5 5\n20 20 20 90 90 90\n")
    (case / "mass.txt").write_text("2\n15.999\n1.008\n")
    (case / "charge.txt").write_text("2\n0\n0\n")
    mdin = [
        'mode = "nve"',
        "step_limit = 3",
        "dt = 0.0",
        "cutoff = 4.0",
        "skin = 1.0",
        'coordinate_in_file = "coordinate.txt"',
        'mass_in_file = "mass.txt"',
        'charge_in_file = "charge.txt"',
        'mdout = "mdout.txt"',
        'mdinfo = "mdinfo.txt"',
        'frc = "frc.dat"',
        "print_zeroth_frame = 1",
        "write_mdout_interval = 1",
        "write_trajectory_interval = 1",
    ]
    if with_cv:
        mdin.append('cv_in_file = "cv.txt"')
        (case / "cv.txt").write_text(
            "print\n{\n    CV = distance\n}\n"
            "distance\n{\n    CV_type = distance\n    atom = 0 1\n}\n"
            "steer\n{\n    CV = distance\n    weight = 2.5\n}\n"
            "restrain\n{\n    CV = distance\n    weight = 1.75\n"
            "    reference = 2.0\n}\n"
        )
    mdin.extend(
        [
            "[PME]",
            "MPI_size = 1",
            "[REAXFF]",
            'in_file = "reaxff.txt"',
            'type_in_file = "reaxff_type.txt"',
        ]
    )
    (case / "mdin.spg.toml").write_text("\n".join(mdin) + "\n")
    command = [os.environ.get("SPONGE_BIN", "SPONGE"), "-mdin", "mdin.spg.toml"]
    if ranks is not None:
        command = ["mpirun", "--oversubscribe", "-np", str(ranks)] + command
    # Kill the whole MPI process group on timeout, including blocked ranks.
    with (case / "run.log").open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=case,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            pytest.fail(f"ReaxFF force exchange timed out: {case}")
    output = (case / "run.log").read_text()
    assert process.returncode == 0, output
    if ranks is not None:
        assert "PP_MPI_size=1, PM_MPI_size=1" in output
    assert "REAXFF_EEQ" in output
    assert "missing input files" not in output
    forces = np.fromfile(case / "frc.dat", dtype=np.float32).reshape(-1, 2, 3)
    assert forces.shape[0] == 4
    assert np.isfinite(forces).all()
    charges = np.loadtxt(case / "eeq_charges.txt")
    assert np.isfinite(charges).all()
    assert np.any(np.abs(charges) > 1e-6)
    rows = parse_mdout_rows(
        case / "mdout.txt", columns=("potential",), int_columns=()
    )
    energies = np.array([float(row["potential"]) for row in rows])
    assert energies.size == 4
    assert np.isfinite(energies).all()
    return forces, energies


def test_mpi_reaxff_eeq_receives_cv_force(tmp_path, mpi_np):
    if mpi_np != 2 or os.name != "posix":
        pytest.skip("requires POSIX MPI with --mpi=2 (one PP and one PM rank)")
    serial = {}
    parallel = {}
    for with_cv in (False, True):
        serial[with_cv] = _run_case(
            tmp_path, f"serial_{with_cv}", ranks=None, with_cv=with_cv
        )
        parallel[with_cv] = _run_case(
            tmp_path, f"mpi_{with_cv}", ranks=mpi_np, with_cv=with_cv
        )
        np.testing.assert_allclose(
            parallel[with_cv][0], serial[with_cv][0], rtol=1e-5, atol=3e-3
        )
        np.testing.assert_allclose(
            parallel[with_cv][1], serial[with_cv][1], rtol=1e-5, atol=2e-3
        )
    serial_delta = serial[True][0] - serial[False][0]
    mpi_delta = parallel[True][0] - parallel[False][0]
    assert np.max(np.abs(serial_delta)) > 0.1
    np.testing.assert_allclose(mpi_delta, serial_delta, rtol=1e-5, atol=3e-3)

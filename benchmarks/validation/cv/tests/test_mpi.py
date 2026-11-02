from pathlib import Path

import numpy as np
import pytest

from benchmarks.utils import Outputer, Runner
from benchmarks.validation.cv.tests.utils import (
    write_bias_cv_file,
    write_validation_mdin,
)
from benchmarks.validation.utils import parse_mdout_rows

CASE_NAME = "alanine_dipeptide_phi_psi"
ATOM_COUNT = 2129
MPI_SCALAR_ABS_TOL = 2.0e-3
MPI_FORCE_ABS_TOL = 3.0e-3


def _run_distance_bias_case(
    statics_path, outputs_path, *, run_name, mpi_np, bias_enabled=True
):
    case_dir = Outputer.prepare_output_case(
        statics_path=statics_path,
        outputs_path=outputs_path,
        case_name=CASE_NAME,
        mpi_np=mpi_np,
        run_name=run_name,
    )
    write_bias_cv_file(
        case_dir,
        target_cv="distance",
        steer_weight=2.5 if bias_enabled else 0.0,
        restrain_weight=1.75 if bias_enabled else 0.0,
        restrain_reference=2.0,
    )
    with (case_dir / "cv.txt").open("a", encoding="utf-8") as cv_file:
        cv_file.write(
            "meta\n"
            "{\n"
            "    Ndim = 1\n"
            "    CV = distance\n"
            "    CV_minimal = 0.0\n"
            "    CV_maximum = 4.0\n"
            "    CV_period = 0.0\n"
            "    CV_grid = 80\n"
            "    CV_sigma = 0.2\n"
            f"    height = {0.1 if bias_enabled else 0.0}\n"
            "    potential_update_interval = 1\n"
            "}\n"
        )
    write_validation_mdin(case_dir, step_limit=1, dt=0.0)
    Runner.run_sponge(case_dir, timeout=1200, mpi_np=mpi_np)
    row = parse_mdout_rows(
        case_dir / "mdout.txt",
        columns=(
            "potential",
            "distance",
            "steer_cv",
            "restrain_cv",
            "meta",
            "rbias",
            "rct",
        ),
        int_columns=(),
    )[-1]
    forces = np.fromfile(case_dir / "frc.dat", dtype=np.float32)
    assert forces.size == 2 * ATOM_COUNT * 3
    return case_dir, row, forces.reshape(2, ATOM_COUNT, 3).astype(np.float64)


def test_mpi_cv_owner_matches_single_process_bias_result(
    statics_path, outputs_path, mpi_np
):
    if mpi_np is None or mpi_np < 2:
        pytest.skip("run with --mpi=2 or greater to validate MPI CV ownership")

    serial_dir, serial_row, serial_force = _run_distance_bias_case(
        statics_path,
        outputs_path,
        run_name="cv_owner_serial_reference",
        mpi_np=None,
    )
    mpi_dir, mpi_row, mpi_force = _run_distance_bias_case(
        statics_path,
        outputs_path,
        run_name=f"cv_owner_mpi_{mpi_np}",
        mpi_np=mpi_np,
    )

    for key, serial_value in serial_row.items():
        assert float(mpi_row[key]) == pytest.approx(
            float(serial_value), abs=MPI_SCALAR_ABS_TOL
        )
    assert float(serial_row["meta"]) != pytest.approx(0.0, abs=1.0e-6)
    _, _, serial_unbiased_force = _run_distance_bias_case(
        statics_path,
        outputs_path,
        run_name="cv_owner_serial_unbiased",
        mpi_np=None,
        bias_enabled=False,
    )
    _, _, mpi_unbiased_force = _run_distance_bias_case(
        statics_path,
        outputs_path,
        run_name=f"cv_owner_mpi_{mpi_np}_unbiased",
        mpi_np=mpi_np,
        bias_enabled=False,
    )
    # Frame zero evaluates identical input coordinates. The integration step
    # can round mapped coordinates differently between MPI runs even at dt=0.
    np.testing.assert_allclose(
        mpi_force[0], serial_force[0], rtol=1.0e-5, atol=MPI_FORCE_ABS_TOL
    )
    serial_bias_force = serial_force - serial_unbiased_force
    mpi_bias_force = mpi_force - mpi_unbiased_force
    assert np.linalg.norm(serial_bias_force[0]) > 1.0e-3
    np.testing.assert_allclose(
        mpi_bias_force[0],
        serial_bias_force[0],
        rtol=1.0e-5,
        atol=MPI_FORCE_ABS_TOL,
    )
    unbiased_atoms = np.ones(ATOM_COUNT, dtype=bool)
    unbiased_atoms[[4, 6]] = False
    np.testing.assert_allclose(
        mpi_bias_force[0, unbiased_atoms], 0.0, atol=MPI_FORCE_ABS_TOL
    )
    # After depositing a hill, also verify force delivery to the CV atoms.
    assert np.linalg.norm(serial_bias_force[1, [4, 6]]) > 1.0e-3
    np.testing.assert_allclose(
        mpi_bias_force[1, [4, 6]],
        serial_bias_force[1, [4, 6]],
        rtol=1.0e-5,
        atol=MPI_FORCE_ABS_TOL,
    )

    serial_info = Path(serial_dir, "mdinfo.txt").read_text(encoding="utf-8")
    mpi_info = Path(mpi_dir, "mdinfo.txt").read_text(encoding="utf-8")
    assert "CV_MPI_rank=0" in serial_info
    assert f"CV_MPI_rank={mpi_np - 1}" in mpi_info

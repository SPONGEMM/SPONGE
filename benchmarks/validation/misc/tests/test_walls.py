import pytest

from benchmarks.validation.misc.tests.utils import (
    is_cuda_init_failure,
    parse_restart_coordinate_zmax,
    prepare_output_case,
    print_validation_vertical,
    run_sponge,
    write_mdin,
)


def _run_tip3p_case(
    statics_path,
    outputs_path,
    run_tag,
    *,
    hard_wall_z_high,
    soft_walls_in_file=None,
    step_limit=1000,
    mpi_np=None,
):
    case_dir = prepare_output_case(
        statics_path=statics_path,
        outputs_path=outputs_path,
        case_name="tip3p",
        run_tag=run_tag,
    )
    write_mdin(
        case_dir,
        hard_wall_z_high=hard_wall_z_high,
        step_limit=step_limit,
        soft_walls_in_file=soft_walls_in_file,
    )
    try:
        run_sponge(case_dir, timeout=1200, mpi_np=mpi_np)
    except RuntimeError as e:
        if is_cuda_init_failure(str(e)):
            pytest.skip(
                "SPONGE CUDA initialization failed. "
                "Use CPU binary or set SPONGE_BIN to a working executable."
            )
        raise
    return case_dir


def test_tip3p_hard_wall_zmax_not_out_of_bound(
    statics_path, outputs_path, mpi_np, mpi_run_tag
):
    hard_wall_z_high = 30.0
    case_dir = _run_tip3p_case(
        statics_path,
        outputs_path,
        f"tip3p_hard_wall_{mpi_run_tag}",
        hard_wall_z_high=hard_wall_z_high,
        step_limit=1000,
        mpi_np=mpi_np,
    )

    zmax = parse_restart_coordinate_zmax(case_dir / "restart_coordinate.txt")
    headers = ["Case", "ZLow", "ZHigh", "RestartZMax", "Status"]
    rows = [
        [
            "tip3p_hard_wall",
            "5.0",
            f"{hard_wall_z_high:.1f}",
            f"{zmax:.4f}",
            "PASS",
        ]
    ]
    print_validation_vertical(
        headers, rows[0], title="Misc Validation: TIP3P Hard Wall"
    )
    assert zmax <= hard_wall_z_high + 0.5


def test_tip3p_soft_wall_zmax_not_out_of_bound(
    statics_path, outputs_path, mpi_np, mpi_run_tag
):
    hard_wall_z_high = 30.0
    case_dir = _run_tip3p_case(
        statics_path,
        outputs_path,
        f"tip3p_soft_wall_{mpi_run_tag}",
        hard_wall_z_high=hard_wall_z_high,
        soft_walls_in_file="soft_walls.txt",
        step_limit=1000,
        mpi_np=mpi_np,
    )

    zmax = parse_restart_coordinate_zmax(case_dir / "restart_coordinate.txt")
    headers = ["Case", "ZLow", "ZHigh", "RestartZMax", "Status"]
    rows = [
        [
            "tip3p_soft_wall",
            "5.0",
            f"{hard_wall_z_high:.1f}",
            f"{zmax:.4f}",
            "PASS",
        ]
    ]
    print_validation_vertical(
        headers, rows[0], title="Misc Validation: TIP3P Soft Wall"
    )
    assert zmax <= hard_wall_z_high + 0.5

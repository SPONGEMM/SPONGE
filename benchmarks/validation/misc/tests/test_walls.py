import pytest

from benchmarks.utils import Outputer, Runner

from benchmarks.validation.misc.tests.utils import (
    parse_restart_coordinate_zmax,
    run_sponge,
    write_mdin,
)


def _run_tip3p_case(
    statics_path,
    outputs_path,
    run_name,
    *,
    hard_wall_z_high,
    soft_walls_in_file=None,
    step_limit=1000,
    mpi_np=None,
):
    case_dir = Outputer.prepare_output_case(
        statics_path=statics_path,
        outputs_path=outputs_path,
        case_name="tip3p",
        mpi_np=mpi_np,
        run_name=run_name,
    )
    write_mdin(
        case_dir,
        hard_wall_z_high=hard_wall_z_high,
        step_limit=step_limit,
        soft_walls_in_file=soft_walls_in_file,
    )
    run_sponge(case_dir, timeout=1200, mpi_np=mpi_np)
    return case_dir


def test_tip3p_hard_wall_zmax_not_out_of_bound(
    statics_path, outputs_path, mpi_np
):
    hard_wall_z_high = 30.0
    case_dir = _run_tip3p_case(
        statics_path,
        outputs_path,
        "tip3p_hard_wall",
        hard_wall_z_high=hard_wall_z_high,
        step_limit=1000,
        mpi_np=mpi_np,
    )

    zmax = parse_restart_coordinate_zmax(case_dir / "restart_coordinate.txt")
    Outputer.print_table(
        ["Metric", "Value"],
        [
            ["Case", "tip3p_hard_wall"],
            ["ZLow", "5.0"],
            ["ZHigh", f"{hard_wall_z_high:.1f}"],
            ["RestartZMax", f"{zmax:.4f}"],
            ["Status", "PASS"],
        ],
        title="Misc Validation: TIP3P Hard Wall",
    )
    assert zmax <= hard_wall_z_high + 0.5


def test_tip3p_soft_wall_zmax_not_out_of_bound(
    statics_path, outputs_path, mpi_np
):
    hard_wall_z_high = 30.0
    case_dir = _run_tip3p_case(
        statics_path,
        outputs_path,
        "tip3p_soft_wall",
        hard_wall_z_high=hard_wall_z_high,
        soft_walls_in_file="soft_walls.txt",
        step_limit=1000,
        mpi_np=mpi_np,
    )

    zmax = parse_restart_coordinate_zmax(case_dir / "restart_coordinate.txt")
    Outputer.print_table(
        ["Metric", "Value"],
        [
            ["Case", "tip3p_soft_wall"],
            ["ZLow", "5.0"],
            ["ZHigh", f"{hard_wall_z_high:.1f}"],
            ["RestartZMax", f"{zmax:.4f}"],
            ["Status", "PASS"],
        ],
        title="Misc Validation: TIP3P Soft Wall",
    )
    assert zmax <= hard_wall_z_high + 0.5

from pathlib import Path

import pytest

from benchmarks.utils import Outputer, Runner

from benchmarks.validation.misc.tests.utils import run_sponge


def write_minimization_mdin(case_dir, *, step_limit=100000):
    mdin = (
        'md_name = "validation tip3p minimization bad coordinate"\n'
        'mode = "minimization"\n'
        f"step_limit = {step_limit}\n"
        "cutoff = 8.0\n"
        'default_in_file_prefix = "tip3p"\n'
        'coordinate_in_file = "bad_coordinate.txt"\n'
        "minimization_dynamic_dt = 1\n"
        "minimization_max_move = 0.05\n"
        "print_zeroth_frame = 1\n"
        "write_mdout_interval = 1000\n"
        "write_information_interval = 1000\n"
    )
    Path(case_dir, "mdin.spg.toml").write_text(mdin)


def parse_potential_by_step(mdout_path):
    lines = Path(mdout_path).read_text().splitlines()
    if len(lines) < 2:
        raise ValueError(f"Invalid mdout file: {mdout_path}")

    headers = lines[0].split()
    if "step" not in headers or "potential" not in headers:
        raise ValueError(f"Missing step/potential columns in {mdout_path}")
    step_idx = headers.index("step")
    pot_idx = headers.index("potential")

    values = {}
    for line in lines[1:]:
        fields = line.split()
        if len(fields) <= max(step_idx, pot_idx):
            continue
        try:
            step = int(fields[step_idx])
            potential = float(fields[pot_idx])
        except ValueError:
            continue
        values[step] = potential
    return values


def test_tip3p_bad_coordinate_minimization_runs(
    statics_path, outputs_path, mpi_np
):
    step_limit = 100000
    case_dir = Outputer.prepare_output_case(
        statics_path=statics_path,
        outputs_path=outputs_path,
        case_name="tip3p",
        mpi_np=mpi_np,
        run_name="tip3p_min_bad_coordinate",
    )
    write_minimization_mdin(case_dir, step_limit=step_limit)
    run_sponge(case_dir, timeout=1200, mpi_np=mpi_np)

    potential_by_step = parse_potential_by_step(case_dir / "mdout.txt")
    milestones = [
        ("Initial", 0),
        ("1/4", step_limit // 4),
        ("1/2", step_limit // 2),
        ("3/4", (step_limit * 3) // 4),
        ("Final", step_limit),
    ]
    milestone_rows = []
    for stage, step in milestones:
        if step not in potential_by_step:
            raise AssertionError(
                f"Missing step {step} in mdout.txt; "
                "check write_mdout_interval setting."
            )
        milestone_rows.append(
            [stage, str(step), f"{potential_by_step[step]:.6e}"]
        )
    Outputer.print_table(
        ["Stage", "Step", "Potential"],
        milestone_rows,
        title="Misc Validation: TIP3P Minimization Energy Milestones",
    )

    Outputer.print_table(
        ["Metric", "Value"],
        [
            ["Case", "tip3p"],
            ["InputCoordinate", "bad_coordinate.txt"],
            ["Mode", "minimization"],
            ["StepLimit", str(step_limit)],
            ["Status", "PASS"],
        ],
        title="Misc Validation: TIP3P Minimization",
    )

    final_potential = potential_by_step[step_limit]
    assert final_potential < -4000.0

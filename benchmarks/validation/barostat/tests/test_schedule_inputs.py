import pytest

from benchmarks.utils import Outputer, Runner
from benchmarks.validation.utils import parse_mdout_rows


def _write_mdin(case_dir, extra_lines):
    base = (
        'md_name = "validation barostat schedule inputs"\n'
        'mode = "npt"\n'
        "step_limit = 20\n"
        "dt = 0.002\n"
        "cutoff = 8.0\n"
        'thermostat = "middle_langevin"\n'
        "thermostat_tau = 0.1\n"
        "thermostat_seed = 2026\n"
        "target_temperature = 300.0\n"
        "target_pressure = 1.0\n"
        'barostat = "berendsen_barostat"\n'
        "barostat_tau = 0.1\n"
        "barostat_update_interval = 10\n"
        'default_in_file_prefix = "tip3p"\n'
        'constrain_mode = "SHAKE"\n'
        "print_zeroth_frame = 1\n"
        "write_mdout_interval = 10\n"
        "write_information_interval = 10\n"
    )
    (case_dir / "mdin.spg.toml").write_text(
        base + "\n".join(extra_lines) + "\n"
    )


def _run(case_dir, mpi_np):
    Runner.run_sponge(case_dir, timeout=600, mpi_np=mpi_np)


def test_schedule_inline_steps_object_array_is_supported(
    statics_path, outputs_path, mpi_np
):
    case_dir = Outputer.prepare_output_case(
        statics_path=statics_path,
        outputs_path=outputs_path,
        case_name="tip3p_water",
        mpi_np=mpi_np,
        run_name="schedule_inline_object_array",
    )

    _write_mdin(
        case_dir,
        [
            'target_temperature_schedule_mode = "linear"',
            "target_temperature_schedule_steps = [{step = 0, value = 300.0}, {step = 20, value = 350.0}]",
            'target_pressure_schedule_mode = "step"',
            "target_pressure_schedule_steps = [{step = 0, value = 1.0}, {step = 10, value = 50.0}]",
        ],
    )

    _run(case_dir, mpi_np)

    rows = parse_mdout_rows(
        case_dir / "mdout.txt", columns=("step",), int_columns=("step",)
    )
    assert rows


def test_default_prefix_detects_temp_pres_toml_schedule_files(
    statics_path, outputs_path, mpi_np
):
    case_dir = Outputer.prepare_output_case(
        statics_path=statics_path,
        outputs_path=outputs_path,
        case_name="tip3p_water",
        mpi_np=mpi_np,
        run_name="schedule_default_prefix_toml",
    )

    (case_dir / "tip3p.temp.spg.toml").write_text(
        'mode = "linear"\nsteps = [{step = 0, value = 330.0}, {step = 20, value = 300.0}]\n'
    )
    (case_dir / "tip3p.pres.spg.toml").write_text(
        'mode = "step"\nsteps = [{step = 0, value = 1.0}, {step = 10, value = 200.0}]\n'
    )

    _write_mdin(case_dir, [])
    _run(case_dir, mpi_np)

    rows = parse_mdout_rows(
        case_dir / "mdout.txt", columns=("step",), int_columns=("step",)
    )
    assert rows


def test_explicit_txt_schedule_file_is_rejected(
    statics_path, outputs_path, mpi_np
):
    case_dir = Outputer.prepare_output_case(
        statics_path=statics_path,
        outputs_path=outputs_path,
        case_name="tip3p_water",
        mpi_np=mpi_np,
        run_name="schedule_txt_file_rejected",
    )

    (case_dir / "legacy_temp.txt").write_text("0 300.0\n20 350.0\n")

    _write_mdin(
        case_dir,
        [
            'target_temperature_schedule_file = "legacy_temp.txt"',
            'target_temperature_schedule_mode = "linear"',
        ],
    )

    with pytest.raises(RuntimeError):
        _run(case_dir, mpi_np)

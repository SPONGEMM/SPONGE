import json

import pytest

from benchmarks.utils import Outputer, Runner
from benchmarks.validation.utils import parse_mdout_rows


def test_softcore(statics_path, outputs_path, mpi_np):
    case_dir = Outputer.prepare_output_case(
        statics_path=statics_path,
        outputs_path=outputs_path,
        case_name="softcore",
        mpi_np=mpi_np,
        run_name="softcore",
    )

    Runner.run_sponge(
        case_dir,
        timeout=1800,
        mpi_np=mpi_np,
        mdin_name="mdin.spg.in",
    )

    reference = json.loads(
        (case_dir / "reference" / "energy_reference_np1.json").read_text()
    )["energies"]
    columns = list(reference.keys())
    energies = parse_mdout_rows(
        case_dir / "mdout.txt",
        columns=columns,
        int_columns=("step",),
    )[0]
    comparable = {
        key: abs(float(energies[key]) - float(ref_val))
        for key, ref_val in reference.items()
        if key not in {"step", "time", "temperature"}
        and key in energies
        and isinstance(ref_val, (int, float))
        and isinstance(energies[key], (int, float))
    }
    max_err_key = max(comparable, key=comparable.get) if comparable else "N/A"
    max_abs_error = comparable[max_err_key] if comparable else 0.0
    abs_tol = 2.0

    Outputer.print_table(
        [
            "Case",
            "MPI_NP",
            "MaxErrTerm",
            "MaxAbsErr",
            "AbsTol",
            "Status",
        ],
        [
            [
                "softcore",
                str(mpi_np) if mpi_np is not None else "direct",
                max_err_key,
                f"{max_abs_error:.6f}",
                f"{abs_tol:.3f}",
                "PASS" if max_abs_error <= abs_tol else "FAIL",
            ]
        ],
        title="Misc Validation: Softcore Point Energy",
    )

    assert max_abs_error <= abs_tol


def _prepare_sits_softcore_case(statics_path, outputs_path, mpi_np, run_name):
    case_dir = Outputer.prepare_output_case(
        statics_path=statics_path,
        outputs_path=outputs_path,
        case_name="softcore",
        mpi_np=mpi_np,
        run_name=run_name,
    )
    # This fixture has softcore parameters and explicit 1-4 coefficients,
    # but no regular LJ parameters from which to derive 1-4 coefficients.
    assert not (case_dir / "softcore_LJ.txt").exists()
    assert not (case_dir / "softcore_nb14.txt").exists()
    atom_numbers = int(
        (case_dir / "softcore_coordinate.txt").read_text().split()[0]
    )
    mdin_path = case_dir / "mdin.spg.in"
    mdin_path.write_text(
        mdin_path.read_text()
        + "\nSITS_mode = observation\n"
        + f"SITS_atom_numbers = {atom_numbers}\n"
    )
    return case_dir


@pytest.mark.parametrize(
    "input_case", ["extra_only", "zero_and_extra", "neither"]
)
def test_sits_nb14_extra_without_regular_lj(
    statics_path, outputs_path, mpi_np, input_case
):
    case_dir = _prepare_sits_softcore_case(
        statics_path, outputs_path, mpi_np, f"sits_nb14_{input_case}"
    )
    mdin_path = case_dir / "mdin.spg.in"
    mdin = mdin_path.read_text()
    if input_case != "neither":
        mdin += "sits_nb14_extra_in_file = softcore_nb14_extra.txt\n"
    if input_case == "zero_and_extra":
        (case_dir / "sits_nb14.txt").write_text("0\n")
        mdin += "sits_nb14_in_file = sits_nb14.txt\n"
    mdin_path.write_text(mdin)
    Runner.run_sponge(
        case_dir,
        timeout=1800,
        mpi_np=mpi_np,
        mdin_name="mdin.spg.in",
    )
    if input_case == "neither":
        assert "sits_nb14_LJ" not in (case_dir / "mdout.txt").read_text()
        assert "sits_nb14_EE" not in (case_dir / "mdout.txt").read_text()
        return

    # Reuse the full-system 1-4 list so the selected energies have an
    # independent reference and a silently skipped initialization fails.
    reference = json.loads(
        (case_dir / "reference" / "energy_reference_np1.json").read_text()
    )["energies"]
    energies = parse_mdout_rows(
        case_dir / "mdout.txt",
        columns=["sits_nb14_LJ", "sits_nb14_EE"],
        int_columns=(),
    )[0]
    for term in ("LJ", "EE"):
        assert energies[f"sits_nb14_{term}"] == pytest.approx(
            reference[f"nb14_{term}"], abs=2.0
        )


@pytest.mark.parametrize(
    ("contents", "expected_error"),
    [
        (
            "1\n0 4 1.0 1.0\n",
            "Use 'sits_nb14_extra_in_file' with explicit A, B",
        ),
        ("", "the format of nb14_in_file is not right"),
        ("-1\n", "the format of nb14_in_file is not right"),
    ],
    ids=["nonempty_without_lj", "missing_count", "negative_count"],
)
def test_sits_nb14_invalid_without_regular_lj(
    statics_path, outputs_path, mpi_np, capsys, contents, expected_error
):
    case_dir = _prepare_sits_softcore_case(
        statics_path, outputs_path, mpi_np, "sits_nb14_invalid"
    )
    (case_dir / "sits_nb14.txt").write_text(contents)
    mdin_path = case_dir / "mdin.spg.in"
    mdin_path.write_text(
        mdin_path.read_text()
        + "sits_nb14_in_file = sits_nb14.txt\n"
        + "sits_nb14_extra_in_file = softcore_nb14_extra.txt\n"
    )
    with pytest.raises(RuntimeError, match="Command failed"):
        Runner.run_sponge(
            case_dir,
            timeout=1800,
            mpi_np=mpi_np,
            mdin_name="mdin.spg.in",
        )
    assert expected_error in capsys.readouterr().out

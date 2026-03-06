import pytest

from utils import (
    compare_energies,
    dump_json,
    load_reference,
    parse_mdout_one_frame,
    parse_pme_backend,
    prepare_output_case,
    print_validation_table,
    resolve_binary_triplet,
    run_point_energy,
    summarize_errors,
)


CASES = [
    pytest.param(
        {
            "id": "covid_tip4p",
            "mdin": "mdin.spg.in",
            "reference_json": "reference/energy_reference_np1.json",
            "abs_tol": 1.0,
        },
        id="covid_tip4p",
    ),
    pytest.param(
        {
            "id": "softcore",
            "mdin": "mdin.spg.in",
            "reference_json": "reference/energy_reference_np1.json",
            "abs_tol": 2.0,
        },
        id="softcore",
    ),
]


@pytest.mark.parametrize("cfg", CASES)
def test_point_energy_multi_backend_and_mpi(statics_path, outputs_path, cfg):
    case_name = cfg["id"]
    case_dir = prepare_output_case(
        statics_path=statics_path,
        outputs_path=outputs_path,
        case_name=case_name,
        run_tag=f"{case_name}_point_energy",
    )

    bins = resolve_binary_triplet()

    runs = [
        {
            "id": "baseline_np1",
            "binary": bins["reference"],
            "np": 1,
            "role": "baseline",
        },
        {
            "id": "gpu_np1",
            "binary": bins["gpu"],
            "np": 1,
            "role": "compare",
        },
        {
            "id": "cpu_np1",
            "binary": bins["cpu"],
            "np": 1,
            "role": "compare",
        },
        {
            "id": "cpu_np2",
            "binary": bins["cpu"],
            "np": 2,
            "role": "compare",
        },
        {
            "id": "cpu_np4",
            "binary": bins["cpu"],
            "np": 4,
            "role": "compare",
        },
    ]

    reference = load_reference(case_dir / cfg["reference_json"])

    summary_payload = {
        "case": case_name,
        "reference_json": cfg["reference_json"],
        "binaries": bins,
        "runs": {},
    }

    rows = []
    hard_fail = False

    for run in runs:
        mdout_path, run_log = run_point_energy(
            case_dir,
            sponge_bin=run["binary"],
            mdin_name=cfg["mdin"],
            run_tag=run["id"],
            mpi_np=run["np"],
            timeout=1800,
        )
        energies = parse_mdout_one_frame(mdout_path)
        term_errors = compare_energies(reference, energies)
        summary = summarize_errors(term_errors)
        backend = parse_pme_backend(run_log)

        run_ok = summary["max_abs_error"] <= cfg["abs_tol"]
        hard_fail = hard_fail or (not run_ok)

        summary_payload["runs"][run["id"]] = {
            "role": run["role"],
            "binary": run["binary"],
            "np": run["np"],
            "pme_backend": backend,
            "max_abs_error": summary["max_abs_error"],
            "max_abs_error_key": summary["max_abs_error_key"],
            "term_errors": term_errors,
        }

        rows.append(
            [
                run["id"],
                str(run["np"]),
                backend,
                summary["max_abs_error_key"],
                f"{summary['max_abs_error']:.6f}",
                f"{cfg['abs_tol']:.3f}",
                "PASS" if run_ok else "FAIL",
            ]
        )

    dump_json(
        summary_payload, case_dir / "point_energy_comparison_summary.json"
    )

    print_validation_table(
        [
            "Run",
            "MPI_NP",
            "PME_Backend",
            "MaxErrTerm",
            "MaxAbsErr",
            "AbsTol",
            "Status",
        ],
        rows,
        title=f"Point Energy Validation: {case_name}",
    )

    assert not hard_fail

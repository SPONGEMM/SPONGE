import pytest

from benchmarks.utils import Outputer

from benchmarks.validation.thermostat.tests.utils import (
    evaluate_temperature_distribution,
    parse_temperature_series,
    read_atom_count_from_mass,
    read_constrain_pair_count_from_runlog,
    run_sponge_thermostat,
    write_thermostat_mdin,
)


THERMOSTAT_CASES = [
    pytest.param(
        {
            "id": "middle_langevin",
            "thermostat": "middle_langevin",
            "thermostat_tau": 0.1,
            "step_limit": 50000,
            "burn_in": 10000,
            "mean_tol_k": 1.5,
            "std_ratio_min": 0.90,
            "std_ratio_max": 1.10,
        },
        id="middle_langevin",
    ),
    pytest.param(
        {
            "id": "andersen",
            "thermostat": "andersen",
            "thermostat_tau": 0.1,
            "step_limit": 50000,
            "burn_in": 10000,
            "mean_tol_k": 1.5,
            "std_ratio_min": 0.90,
            "std_ratio_max": 1.10,
        },
        id="andersen",
    ),
    pytest.param(
        {
            "id": "berendsen_thermostat",
            "thermostat": "berendsen_thermostat",
            "thermostat_tau": 0.1,
            "step_limit": 50000,
            "burn_in": 10000,
            "mean_tol_k": 1.5,
            "std_ratio_min": 0.90,
            "std_ratio_max": 1.10,
            "check_std": False,
        },
        id="berendsen_thermostat",
    ),
    pytest.param(
        {
            "id": "nose_hoover_chain",
            "thermostat": "nose_hoover_chain",
            "thermostat_tau": 0.1,
            "step_limit": 50000,
            "burn_in": 10000,
            "mean_tol_k": 1.5,
            "std_ratio_min": 0.90,
            "std_ratio_max": 1.10,
        },
        id="nose_hoover_chain",
    ),
    pytest.param(
        {
            "id": "bussi_thermostat",
            "thermostat": "bussi_thermostat",
            "thermostat_mode": "bussi_thermostat",
            "thermostat_tau": 0.1,
            "step_limit": 50000,
            "burn_in": 10000,
            "mean_tol_k": 1.5,
            "std_ratio_min": 0.90,
            "std_ratio_max": 1.10,
        },
        id="bussi_thermostat",
    ),
]


@pytest.mark.parametrize("cfg", THERMOSTAT_CASES)
def test_thermostat_tip3p_water(statics_path, outputs_path, cfg, mpi_np):
    case_name = "tip3p_water"
    file_prefix = "tip3p"
    case_dir = Outputer.prepare_output_case(
        statics_path=statics_path,
        outputs_path=outputs_path,
        case_name=case_name,
        mpi_np=mpi_np,
        run_name=cfg["id"],
    )

    write_thermostat_mdin(
        case_dir,
        step_limit=cfg["step_limit"],
        dt=0.002,
        cutoff=8.0,
        thermostat=cfg["thermostat"],
        thermostat_mode=cfg.get("thermostat_mode"),
        constrain_mode="SETTLE",
        target_temperature=300.0,
        thermostat_tau=cfg["thermostat_tau"],
        thermostat_seed=2026,
        default_in_file_prefix=file_prefix,
        md_name="benchmark tip3p_water thermostat",
    )

    run_output = run_sponge_thermostat(case_dir, timeout=600, mpi_np=mpi_np)

    temperatures = parse_temperature_series(case_dir / "mdout.txt")
    n_atoms = read_atom_count_from_mass(case_dir / f"{file_prefix}_mass.txt")
    constrain_pair_count = read_constrain_pair_count_from_runlog(run_output)
    stats = evaluate_temperature_distribution(
        temperatures,
        target_temperature=300.0,
        n_atoms=n_atoms,
        burn_in=cfg["burn_in"],
        constrain_pair_count=constrain_pair_count,
    )

    mean_tol_k = cfg["mean_tol_k"]
    std_ratio_min = cfg["std_ratio_min"]
    std_ratio_max = cfg["std_ratio_max"]
    check_std = cfg.get("check_std", True)

    mean_ok = abs(stats["mean_temp"] - 300.0) <= mean_tol_k
    std_ok = (
        std_ratio_min <= stats["std_ratio"] <= std_ratio_max
        if check_std
        else True
    )
    std_ratio_range = (
        f"[{std_ratio_min:.2f}, {std_ratio_max:.2f}]"
        if check_std
        else "N/A (mean only)"
    )

    headers = [
        "Case",
        "Thermostat",
        "Samples",
        "Mean(K)",
        "Std(K)",
        "ExpectedStd(K)",
        "StdRatio",
        "MeanTol(K)",
        "StdRatioRange",
        "Status",
    ]
    rows = [
        [
            case_name,
            cfg["id"],
            str(stats["sample_count"]),
            f"{stats['mean_temp']:.3f}",
            f"{stats['std_temp']:.3f}",
            f"{stats['expected_std']:.3f}",
            f"{stats['std_ratio']:.3f}",
            f"{mean_tol_k:.1f}",
            std_ratio_range,
            "PASS" if (mean_ok and std_ok) else "FAIL",
        ]
    ]
    Outputer.print_table(
        headers,
        rows,
        title="Thermostat Validation: TIP3P Water",
    )

    assert mean_ok
    if check_std:
        assert std_ok

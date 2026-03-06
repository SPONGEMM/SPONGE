import pytest

from benchmarks.utils import Outputer

from benchmarks.validation.barostat.tests.utils import (
    boltzmann_reweight_mean,
    parse_density_series_from_mdbox,
    read_total_mass_amu,
    run_sponge_barostat,
    write_barostat_mdin,
)


BAROSTAT_CASES = [
    pytest.param(
        {
            "id": "andersen_barostat",
            "barostat": "andersen_barostat",
            "step_limit": 100000,
            "burn_in": 300,
            "write_information_interval": 10,
            "reweight_abs_tol": 0.03,
            "target_temperature": 300.0,
            "timeout": 1500,
        },
        id="andersen_barostat",
    ),
    pytest.param(
        {
            "id": "bussi_barostat",
            "barostat": "bussi_barostat",
            "step_limit": 100000,
            "burn_in": 300,
            "write_information_interval": 10,
            "reweight_abs_tol": 0.03,
            "target_temperature": 300.0,
            "timeout": 1500,
        },
        id="bussi_barostat",
    ),
    pytest.param(
        {
            "id": "berendsen_barostat",
            "barostat": "berendsen_barostat",
            "step_limit": 100000,
            "burn_in": 300,
            "write_information_interval": 10,
            "reweight_abs_tol": 0.03,
            "target_temperature": 300.0,
            "timeout": 1500,
            "check_reweighting": False,
        },
        id="berendsen_barostat",
    ),
]


def _run_tip3p_barostat_case(
    statics_path, outputs_path, run_name, *, cfg, target_pressure, mpi_np
):
    case_name = "tip3p_water"
    case_dir = Outputer.prepare_output_case(
        statics_path=statics_path,
        outputs_path=outputs_path,
        case_name=case_name,
        mpi_np=mpi_np,
        run_name=run_name,
    )
    write_barostat_mdin(
        case_dir,
        step_limit=cfg["step_limit"],
        dt=0.002,
        cutoff=8.0,
        thermostat="middle_langevin",
        thermostat_tau=0.1,
        thermostat_seed=2026,
        target_temperature=cfg["target_temperature"],
        target_pressure=target_pressure,
        barostat=cfg["barostat"],
        barostat_tau=1.0,
        barostat_update_interval=10,
        write_information_interval=cfg["write_information_interval"],
        default_in_file_prefix="tip3p",
        constrain_mode="SETTLE",
    )

    run_sponge_barostat(case_dir, timeout=cfg["timeout"], mpi_np=mpi_np)

    total_mass_amu = read_total_mass_amu(case_dir / "tip3p_mass.txt")
    densities, volumes = parse_density_series_from_mdbox(
        case_dir / "mdbox.txt", total_mass_amu
    )
    return {
        "case_dir": case_dir,
        "densities": densities,
        "volumes": volumes,
    }


@pytest.mark.parametrize("cfg", BAROSTAT_CASES)
def test_tip3p_density_boltzmann_reweighting_1bar_500bar(
    statics_path, outputs_path, cfg, mpi_np
):
    pressure_low = 1.0
    pressure_high = 500.0
    run_low = _run_tip3p_barostat_case(
        statics_path=statics_path,
        outputs_path=outputs_path,
        run_name=f"{cfg['id']}_1bar",
        cfg=cfg,
        target_pressure=pressure_low,
        mpi_np=mpi_np,
    )
    run_high = _run_tip3p_barostat_case(
        statics_path=statics_path,
        outputs_path=outputs_path,
        run_name=f"{cfg['id']}_500bar",
        cfg=cfg,
        target_pressure=pressure_high,
        mpi_np=mpi_np,
    )

    low_density_stats = Outputer.summarize_series(
        run_low["densities"], cfg["burn_in"]
    )
    high_density_stats = Outputer.summarize_series(
        run_high["densities"], cfg["burn_in"]
    )
    low_density_sample = run_low["densities"][cfg["burn_in"] :]
    high_density_sample = run_high["densities"][cfg["burn_in"] :]
    low_volume_sample = run_low["volumes"][cfg["burn_in"] :]
    high_volume_sample = run_high["volumes"][cfg["burn_in"] :]

    low_to_high = boltzmann_reweight_mean(
        low_density_sample,
        low_volume_sample,
        from_pressure_bar=pressure_low,
        to_pressure_bar=pressure_high,
        temperature_k=cfg["target_temperature"],
    )
    high_to_low = boltzmann_reweight_mean(
        high_density_sample,
        high_volume_sample,
        from_pressure_bar=pressure_high,
        to_pressure_bar=pressure_low,
        temperature_k=cfg["target_temperature"],
    )

    low_to_high_err = abs(low_to_high["mean"] - high_density_stats["mean"])
    high_to_low_err = abs(high_to_low["mean"] - low_density_stats["mean"])

    check_reweighting = cfg.get("check_reweighting", True)
    low_to_high_ok = low_to_high_err <= cfg["reweight_abs_tol"]
    high_to_low_ok = high_to_low_err <= cfg["reweight_abs_tol"]
    if check_reweighting:
        status = "PASS" if (low_to_high_ok and high_to_low_ok) else "FAIL"
    else:
        status = "SKIP"

    rows = [
        ["Case", "tip3p_water"],
        ["Barostat", cfg["id"]],
        ["Direct(1bar)", f"{low_density_stats['mean']:.4f}"],
        ["Direct(500bar)", f"{high_density_stats['mean']:.4f}"],
        ["RW(1->500)", f"{low_to_high['mean']:.4f}"],
        ["RW(500->1)", f"{high_to_low['mean']:.4f}"],
        ["AbsErr(1->500)", f"{low_to_high_err:.4f}"],
        ["AbsErr(500->1)", f"{high_to_low_err:.4f}"],
        ["Status", status],
    ]
    Outputer.print_table(
        ["Metric", "Value"],
        rows,
        title="Barostat Validation: TIP3P Water Density Reweighting (1bar <-> 500bar)",
    )

    if check_reweighting:
        assert low_to_high_ok
        assert high_to_low_ok

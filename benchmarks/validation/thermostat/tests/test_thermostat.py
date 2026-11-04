import pytest

from benchmarks.utils import Outputer

from benchmarks.validation.thermostat.tests.utils import (
    evaluate_temperature_block_average,
    parse_temperature_series,
    run_sponge_thermostat,
    write_thermostat_mdin,
)

COMMON_CFG = {
    "step_limit": 1000,
    "burn_in": 20,
    "block_size": 100,
    "z_score_max": 3.0,
    "thermostat_tau": 0.1,
    "target_temperature": 300.0,
    "seed": 2026,
}


THERMOSTAT_CASES = [
    pytest.param(
        {
            **COMMON_CFG,
            "id": "middle_langevin",
            "thermostat": "middle_langevin",
        },
        id="middle_langevin",
    ),
    pytest.param(
        {
            **COMMON_CFG,
            "id": "andersen",
            "thermostat": "andersen",
        },
        id="andersen",
    ),
    pytest.param(
        {
            **COMMON_CFG,
            "id": "berendsen_thermostat",
            "thermostat": "berendsen_thermostat",
        },
        id="berendsen_thermostat",
    ),
    pytest.param(
        {
            **COMMON_CFG,
            "id": "nose_hoover_chain",
            "thermostat": "nose_hoover_chain",
        },
        id="nose_hoover_chain",
    ),
    pytest.param(
        {
            **COMMON_CFG,
            "id": "bussi_thermostat",
            "thermostat": "bussi_thermostat",
            "thermostat_mode": "bussi_thermostat",
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
    # Project onto the rigid-water velocity manifold before normalizing the
    # temperature. Unconstrained velocities lose kinetic energy on the first
    # SETTLE step and contaminate this short stationary-temperature check.
    write_thermostat_mdin(
        case_dir,
        step_limit=cfg["step_limit"],
        dt=0.002,
        cutoff=8.0,
        thermostat=cfg["thermostat"],
        thermostat_mode=cfg.get("thermostat_mode"),
        constrain_mode="SETTLE",
        target_temperature=cfg["target_temperature"],
        thermostat_tau=cfg["thermostat_tau"],
        thermostat_seed=cfg["seed"],
        default_in_file_prefix=file_prefix,
        md_name="benchmark tip3p_water thermostat",
        initial_velocity_seed=cfg["seed"],
    )

    run_sponge_thermostat(case_dir, timeout=600, mpi_np=mpi_np)

    temperatures = parse_temperature_series(case_dir / "mdout.txt")
    stats = evaluate_temperature_block_average(
        temperatures,
        target_temperature=cfg["target_temperature"],
        burn_in=cfg["burn_in"],
        block_size=cfg["block_size"],
    )
    z_score_max = cfg["z_score_max"]
    mean_ok = stats["z_score"] <= z_score_max

    headers = [
        "Thermostat",
        "Samples",
        "Blocks",
        "Mean(K)",
        "BlockStd(K)",
        "SEM(K)",
        "ZScore",
        "ZScoreMax",
        "Status",
    ]
    rows = [
        [
            cfg["id"],
            str(stats["sample_count"]),
            str(stats["block_count"]),
            f"{stats['mean_temp']:.3f}",
            f"{stats['block_std']:.3f}",
            f"{stats['sem']:.3f}",
            f"{stats['z_score']:.3f}",
            f"{z_score_max:.1f}",
            "PASS" if mean_ok else "FAIL",
        ]
    ]
    Outputer.print_table(
        headers,
        rows,
        title=f"Thermostat Validation: {case_name}",
    )

    assert mean_ok

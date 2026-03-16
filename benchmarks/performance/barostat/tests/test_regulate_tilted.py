import math
import shutil

import pytest

from benchmarks.utils import Outputer

from benchmarks.performance.barostat.tests.utils import run_sponge_barostat
from benchmarks.validation.barostat.tests.utils import (
    AMU_PER_A3_TO_G_PER_CM3,
    parse_density_series_from_mdbox,
    read_total_mass_amu,
    rescale_coordinate_box,
    triclinic_volume_a3,
    write_barostat_mdin,
)


REGULATE_CASES = [
    pytest.param(
        {
            "id": "andersen_barostat",
            "barostat": "andersen_barostat",
        },
        id="andersen_barostat",
    ),
    pytest.param(
        {
            "id": "bussi_barostat",
            "barostat": "bussi_barostat",
        },
        id="bussi_barostat",
    ),
    pytest.param(
        {
            "id": "berendsen_barostat",
            "barostat": "berendsen_barostat",
        },
        id="berendsen_barostat",
    ),
]

COMPRESS_STEP_LIMIT = 10000
RELAX_STEP_LIMIT = 5000
WRITE_INFORMATION_INTERVAL = 1000
WRITE_MDOUT_INTERVAL = 1000
DENSITY_TAIL_SAMPLES = 3


def _write_and_run_stage(
    case_dir,
    *,
    stage_tag,
    step_limit,
    target_pressure,
    barostat,
    barostat_tau,
    barostat_update_interval,
    write_information_interval,
    default_in_file_prefix,
    constrain_mode,
    timeout,
    mpi_np,
):
    write_barostat_mdin(
        case_dir,
        step_limit=step_limit,
        dt=0.002,
        cutoff=8.0,
        thermostat="middle_langevin",
        thermostat_tau=0.1,
        thermostat_seed=2026,
        target_temperature=300.0,
        target_pressure=target_pressure,
        barostat=barostat,
        barostat_tau=barostat_tau,
        barostat_update_interval=barostat_update_interval,
        write_information_interval=write_information_interval,
        write_mdout_interval=WRITE_MDOUT_INTERVAL,
        default_in_file_prefix=default_in_file_prefix,
        constrain_mode=constrain_mode,
    )
    run_sponge_barostat(case_dir, timeout=timeout, mpi_np=mpi_np)
    shutil.copyfile(case_dir / "mdout.txt", case_dir / f"mdout_{stage_tag}.txt")
    shutil.copyfile(case_dir / "mdbox.txt", case_dir / f"mdbox_{stage_tag}.txt")


def _parse_mdbox_box6(mdbox_path):
    records = []
    for line in mdbox_path.read_text().splitlines():
        fields = line.split()
        if len(fields) < 6:
            continue
        records.append(tuple(float(x) for x in fields[:6]))
    if not records:
        raise ValueError(f"No box records parsed from {mdbox_path}")
    return records


@pytest.mark.parametrize("cfg", REGULATE_CASES)
def test_wat_nonortho_regulate_from_expanded_nonorthogonal_box(
    statics_path, outputs_path, cfg, mpi_np
):
    case_dir = Outputer.prepare_output_case(
        statics_path=statics_path,
        outputs_path=outputs_path,
        case_name="WAT_nonortho",
        mpi_np=mpi_np,
        run_name=f"{cfg['id']}_regulate_WAT_nonortho_expanded",
    )

    rescale_coordinate_box(
        case_dir / "WAT_coordinate.txt",
        new_lx=52.0,
        new_ly=52.0,
        new_lz=52.0,
        scale_coordinates=False,
    )

    total_mass_amu = read_total_mass_amu(case_dir / "WAT_mass.txt")
    box_fields = (
        (case_dir / "WAT_coordinate.txt").read_text().splitlines()[-1].split()
    )
    init_lx, init_ly, init_lz = map(float, box_fields[:3])
    init_alpha, init_beta, init_gamma = map(float, box_fields[3:6])
    initial_density = (
        total_mass_amu
        * AMU_PER_A3_TO_G_PER_CM3
        / triclinic_volume_a3(
            init_lx,
            init_ly,
            init_lz,
            alpha_deg=init_alpha,
            beta_deg=init_beta,
            gamma_deg=init_gamma,
        )
    )

    _write_and_run_stage(
        case_dir,
        stage_tag="compress_1000bar",
        step_limit=COMPRESS_STEP_LIMIT,
        target_pressure=1000.0,
        barostat=cfg["barostat"],
        barostat_tau=0.1,
        barostat_update_interval=10,
        write_information_interval=WRITE_INFORMATION_INTERVAL,
        default_in_file_prefix="WAT",
        constrain_mode="SETTLE",
        timeout=1200,
        mpi_np=mpi_np,
    )
    shutil.copyfile(
        case_dir / "restart_coordinate.txt",
        case_dir / "WAT_coordinate.txt",
    )
    _write_and_run_stage(
        case_dir,
        stage_tag="relax_1bar",
        step_limit=RELAX_STEP_LIMIT,
        target_pressure=1.0,
        barostat=cfg["barostat"],
        barostat_tau=0.1,
        barostat_update_interval=10,
        write_information_interval=WRITE_INFORMATION_INTERVAL,
        default_in_file_prefix="WAT",
        constrain_mode="SETTLE",
        timeout=1200,
        mpi_np=mpi_np,
    )

    densities, _ = parse_density_series_from_mdbox(
        case_dir / "mdbox_relax_1bar.txt", total_mass_amu
    )
    box_records = _parse_mdbox_box6(case_dir / "mdbox_relax_1bar.txt")

    assert len(densities) == RELAX_STEP_LIMIT // WRITE_INFORMATION_INTERVAL
    assert len(box_records) == RELAX_STEP_LIMIT // WRITE_INFORMATION_INTERVAL

    density_tail_n = DENSITY_TAIL_SAMPLES
    density_stats = Outputer.summarize_series(
        densities, burn_in=len(densities) - density_tail_n
    )

    alphas = [r[3] for r in box_records]
    betas = [r[4] for r in box_records]
    gammas = [r[5] for r in box_records]
    alpha_stats = Outputer.summarize_series(
        alphas, burn_in=len(alphas) - density_tail_n
    )
    beta_stats = Outputer.summarize_series(
        betas, burn_in=len(betas) - density_tail_n
    )
    gamma_stats = Outputer.summarize_series(
        gammas, burn_in=len(gammas) - density_tail_n
    )

    target_density = 0.992
    density_abs_tol = 0.010
    density_error = abs(density_stats["mean"] - target_density)
    density_ok = density_error <= density_abs_tol
    angles_finite = all(
        math.isfinite(v)
        for v in (
            alpha_stats["mean"],
            beta_stats["mean"],
            gamma_stats["mean"],
            alpha_stats["std"],
            beta_stats["std"],
            gamma_stats["std"],
        )
    )

    rows = [
        ["Case", "WAT_nonortho"],
        ["Barostat", cfg["id"]],
        ["InitialBox(A)", f"{init_lx:.1f} x {init_ly:.1f} x {init_lz:.1f}"],
        [
            "InitialAngles(deg)",
            f"{init_alpha:.1f}, {init_beta:.1f}, {init_gamma:.1f}",
        ],
        ["InitialDensity(g/cm3)", f"{initial_density:.4f}"],
        ["DensityTailSamples", str(density_tail_n)],
        ["FinalDensityMean(g/cm3)", f"{density_stats['mean']:.4f}"],
        ["FinalDensityStd", f"{density_stats['std']:.4f}"],
        ["DensityTarget(g/cm3)", f"{target_density:.3f}"],
        ["DensityAbsTol", f"{density_abs_tol:.3f}"],
        ["DensityAbsError", f"{density_error:.4f}"],
        ["FinalAlphaMean(deg)", f"{alpha_stats['mean']:.3f}"],
        ["FinalBetaMean(deg)", f"{beta_stats['mean']:.3f}"],
        ["FinalGammaMean(deg)", f"{gamma_stats['mean']:.3f}"],
        ["FinalAlphaStd", f"{alpha_stats['std']:.3f}"],
        ["FinalBetaStd", f"{beta_stats['std']:.3f}"],
        ["FinalGammaStd", f"{gamma_stats['std']:.3f}"],
        ["Status", "PASS" if (density_ok and angles_finite) else "FAIL"],
    ]
    Outputer.print_table(
        ["Metric", "Value"],
        rows,
        title=(
            "Barostat Validation: WAT_nonortho Regulate from Expanded "
            f"Non-orthogonal Box ({cfg['id']})"
        ),
    )

    assert angles_finite
    assert density_ok

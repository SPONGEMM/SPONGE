import shutil

import pytest

from benchmarks.utils import Outputer
from benchmarks.validation.barostat.tests.utils import (
    AMU_PER_A3_TO_G_PER_CM3,
    parse_density_series_from_mdbox,
    parse_mdbox_lengths,
    read_total_mass_amu,
    rescale_coordinate_box,
    run_sponge_barostat,
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
        default_in_file_prefix="tip3p",
        constrain_mode="SHAKE",
    )
    run_sponge_barostat(case_dir, timeout=timeout, mpi_np=mpi_np)
    shutil.copyfile(case_dir / "mdout.txt", case_dir / f"mdout_{stage_tag}.txt")
    shutil.copyfile(case_dir / "mdbox.txt", case_dir / f"mdbox_{stage_tag}.txt")


@pytest.mark.parametrize("cfg", REGULATE_CASES)
def test_tip3p_regulate_from_30a_box(statics_path, outputs_path, cfg, mpi_np):
    case_dir = Outputer.prepare_output_case(
        statics_path=statics_path,
        outputs_path=outputs_path,
        case_name="tip3p_water",
        mpi_np=mpi_np,
        run_name=f"{cfg['id']}_regulate_30A",
    )

    # Expand only the periodic box to start from a low-density state.
    rescale_coordinate_box(
        case_dir / "tip3p_coordinate.txt",
        new_lx=30.0,
        new_ly=30.0,
        new_lz=30.0,
        scale_coordinates=False,
    )
    total_mass_amu = read_total_mass_amu(case_dir / "tip3p_mass.txt")
    expanded_box = (
        (case_dir / "tip3p_coordinate.txt").read_text().splitlines()[-1].split()
    )
    box_lx, box_ly, box_lz = map(float, expanded_box[:3])
    initial_density = (
        total_mass_amu * AMU_PER_A3_TO_G_PER_CM3 / (box_lx * box_ly * box_lz)
    )

    # Stage 1: fast high-pressure compression from very low density.
    _write_and_run_stage(
        case_dir,
        stage_tag="compress_1000bar",
        step_limit=100000,
        target_pressure=1000.0,
        barostat=cfg["barostat"],
        barostat_tau=0.1,
        barostat_update_interval=10,
        write_information_interval=10,
        timeout=1800,
        mpi_np=mpi_np,
    )
    shutil.copyfile(
        case_dir / "restart_coordinate.txt",
        case_dir / "tip3p_coordinate.txt",
    )

    # Stage 2: switch to target barostat at 1 bar and verify density.
    _write_and_run_stage(
        case_dir,
        stage_tag="relax_1bar",
        step_limit=50000,
        target_pressure=1.0,
        barostat=cfg["barostat"],
        barostat_tau=0.1,
        barostat_update_interval=10,
        write_information_interval=10,
        timeout=1800,
        mpi_np=mpi_np,
    )

    densities, _ = parse_density_series_from_mdbox(
        case_dir / "mdbox_relax_1bar.txt", total_mass_amu
    )
    box_lengths = parse_mdbox_lengths(case_dir / "mdbox_relax_1bar.txt")

    expected_samples = 50000 // 10
    assert len(densities) == expected_samples
    assert len(box_lengths) == expected_samples

    density_tail_n = 500
    final_density_stats = Outputer.summarize_series(
        densities, burn_in=len(densities) - density_tail_n
    )
    final_lx_stats = Outputer.summarize_series(
        [xyz[0] for xyz in box_lengths], burn_in=2500
    )
    final_ly_stats = Outputer.summarize_series(
        [xyz[1] for xyz in box_lengths], burn_in=2500
    )
    final_lz_stats = Outputer.summarize_series(
        [xyz[2] for xyz in box_lengths], burn_in=2500
    )

    target_density = 0.982
    density_abs_tol = 0.010
    density_error = abs(final_density_stats["mean"] - target_density)
    final_density_ok = density_error <= density_abs_tol

    rows = [
        ["Case", "tip3p_water"],
        ["Barostat", cfg["id"]],
        ["InitialDensity(g/cm3)", f"{initial_density:.4f}"],
        ["DensityTailSamples", str(density_tail_n)],
        ["FinalDensityMean(g/cm3)", f"{final_density_stats['mean']:.4f}"],
        ["FinalDensityStd", f"{final_density_stats['std']:.4f}"],
        ["DensityTarget(g/cm3)", f"{target_density:.3f}"],
        ["DensityAbsTol", f"{density_abs_tol:.3f}"],
        ["DensityAbsError", f"{density_error:.4f}"],
        ["FinalLxMean(A)", f"{final_lx_stats['mean']:.3f}"],
        ["FinalLyMean(A)", f"{final_ly_stats['mean']:.3f}"],
        ["FinalLzMean(A)", f"{final_lz_stats['mean']:.3f}"],
        ["Status", "PASS" if final_density_ok else "FAIL"],
    ]
    Outputer.print_table(
        ["Metric", "Value"],
        rows,
        title="Barostat Validation: TIP3P Regulate from 30A Box to Target Density",
    )

    assert final_density_ok

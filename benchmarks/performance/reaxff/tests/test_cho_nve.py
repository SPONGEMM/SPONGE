import shutil

import pytest

from benchmarks.utils import Outputer, Runner

from benchmarks.performance.reaxff.tests.utils import (
    dump_summary_json,
    parse_mdout_series,
    read_atom_count_from_coordinate,
    save_energy_plots,
    summarize_energy_stability,
    write_nve_long_mdin,
)


def test_reaxff_cho_heat_then_nve_energy_stability(
    statics_path, outputs_path, mpi_np
):
    case_dir = Outputer.prepare_output_case(
        statics_path=statics_path,
        outputs_path=outputs_path,
        case_name="cho_nve",
        mpi_np=mpi_np,
        run_name="cho_nve_heat_nve",
    )

    Runner.run_sponge(
        case_dir,
        mdin_name="heat.spg.toml",
        timeout=2400,
        mpi_np=mpi_np,
    )

    shutil.copyfile(case_dir / "mdout.txt", case_dir / "mdout.heat.txt")

    write_nve_long_mdin(
        case_dir,
        source_nve_mdin="nve.spg.toml",
        output_mdin="nve.long.spg.toml",
        step_limit=20000,
        write_information_interval=10,
        rst="nve_long",
    )

    Runner.run_sponge(
        case_dir,
        mdin_name="nve.long.spg.toml",
        timeout=2400,
        mpi_np=mpi_np,
    )

    shutil.copyfile(case_dir / "mdout.txt", case_dir / "mdout.nve_long.txt")

    nve_rows = parse_mdout_series(case_dir / "mdout.nve_long.txt")
    atom_count = read_atom_count_from_coordinate(case_dir / "coordinate.txt")
    dof = 3 * atom_count

    stats = summarize_energy_stability(nve_rows, dof=dof)
    dump_summary_json(stats, case_dir / "energy_stability_summary.json")
    save_energy_plots(nve_rows, dof=dof, output_dir=case_dir)

    rows = [
        ["Case", "cho_nve"],
        ["Mode", "heat (nvt) -> nve"],
        ["N_atoms", str(atom_count)],
        ["DOF", str(dof)],
        ["NVE_samples", str(stats["samples"])],
        ["E0(kcal/mol)", f"{stats['e0']:.6f}"],
        ["MeanE(kcal/mol)", f"{stats['mean_e']:.6f}"],
        ["StdE(kcal/mol)", f"{stats['std_e']:.6f}"],
        ["FinalDrift(kcal/mol)", f"{stats['final_drift']:.6f}"],
        ["MaxAbsDrift(kcal/mol)", f"{stats['max_abs_drift']:.6f}"],
        ["FinalRelDrift", f"{stats['final_rel_drift']:.6e}"],
        ["MaxRelDrift", f"{stats['max_rel_drift']:.6e}"],
        ["Slope(kcal/mol/ps)", f"{stats['slope_kcal_per_mol_ps']:.6f}"],
    ]
    Outputer.print_table(
        ["Metric", "Value"],
        rows,
        title="ReaxFF Validation: CHO Heat->NVE Energy Stability",
    )

    assert stats["final_rel_drift"] <= 2.0e-3
    assert stats["max_rel_drift"] <= 5.0e-3

from pathlib import Path

from benchmarks.utils import Outputer, Runner

from benchmarks.performance.sinkmeta.tests.utils import (
    dump_summary_json,
    parse_box_lengths,
    parse_cv_rows,
    parse_meta_potential,
    parse_scatter_axis_file,
    project_xyz_to_path_sr,
    reimage_xyz_to_axis,
    save_sinkmeta_plots,
    summarize_sinkmeta,
    write_dna_cou_cv_file,
    write_sinkmeta_mdin,
)

SINKMETA_HEIGHT = 0.2
SINKMETA_SIGMA = 0.6
SINKMETA_DIP = 1.0


def test_dna_cou_3d_positional_sinkmeta_long_nvt(
    statics_path,
    outputs_path,
    sinkmeta_steps,
    mpi_np,
):
    case_name = "dna_cou_sinkmeta"
    run_name = "dna_cou_sinkmeta_long_nvt"
    edge_cache_path = Path(outputs_path) / case_name / "edge_sumhill.log"
    edge_cache_path.parent.mkdir(parents=True, exist_ok=True)
    case_dir = Outputer.prepare_output_case(
        statics_path=statics_path,
        outputs_path=outputs_path,
        case_name=case_name,
        mpi_np=mpi_np,
        run_name=run_name,
    )

    write_dna_cou_cv_file(
        case_dir,
        height=SINKMETA_HEIGHT,
        sigma=SINKMETA_SIGMA,
        dip=SINKMETA_DIP,
        edge_file=edge_cache_path,
    )
    write_sinkmeta_mdin(case_dir, step_limit=sinkmeta_steps)

    Runner.run_sponge(case_dir, timeout=172800, mpi_np=mpi_np)

    mdout_path = Path(case_dir) / "mdout.txt"
    potential_path = Path(case_dir) / "Meta_Potential.txt"
    axis_path = Path(case_dir) / "14d5.txt"
    box_path = Path(case_dir) / "mdbox.txt"
    rows = parse_cv_rows(mdout_path)
    potential_rows = parse_meta_potential(potential_path)
    axis_rows = parse_scatter_axis_file(axis_path)
    box_lengths = parse_box_lengths(box_path)
    wrapped_xyz, axis_distance = reimage_xyz_to_axis(
        rows, axis_rows, box_lengths
    )
    sr_projection = project_xyz_to_path_sr(rows, axis_rows, box_lengths)
    summary = summarize_sinkmeta(
        rows,
        wrapped_xyz=sr_projection["wrapped_xyz"],
        axis_distance=axis_distance,
        s_values=sr_projection["s"],
        r_values=sr_projection["r"],
    )
    summary["sinkmeta_steps"] = int(sinkmeta_steps)
    summary["sinkmeta_height"] = float(SINKMETA_HEIGHT)
    summary["sinkmeta_sigma"] = float(SINKMETA_SIGMA)
    summary["sinkmeta_dip"] = float(SINKMETA_DIP)
    summary["meta_potential_rows"] = int(len(potential_rows))
    summary["axis_points"] = int(len(axis_rows))

    dump_summary_json(summary, Path(case_dir) / "sinkmeta_summary.json")
    plot_paths = save_sinkmeta_plots(
        rows,
        potential_rows,
        axis_rows,
        case_dir,
        wrapped_xyz=sr_projection["wrapped_xyz"],
        axis_distance=axis_distance,
        s_values=sr_projection["s"],
        r_values=sr_projection["r"],
        axis_s=sr_projection["axis_s"],
    )

    table_rows = [
        ["RunName", run_name],
        ["StepLimit", str(sinkmeta_steps)],
        ["Samples", str(summary["samples"])],
        ["AxisPoints", str(summary["axis_points"])],
        ["FiniteFraction", f"{summary['finite_fraction']:.4f}"],
        ["UniqueXYZ@1e-3", str(summary["unique_xyz_count_1e3"])],
        ["MetaPotentialRows", str(summary["meta_potential_rows"])],
        [
            "cx_range",
            f"{summary['cx']['min']:.4f} -> {summary['cx']['max']:.4f}",
        ],
        [
            "cy_range",
            f"{summary['cy']['min']:.4f} -> {summary['cy']['max']:.4f}",
        ],
        [
            "cz_range",
            f"{summary['cz']['min']:.4f} -> {summary['cz']['max']:.4f}",
        ],
        [
            "meta_range",
            f"{summary['meta']['min']:.4f} -> {summary['meta']['max']:.4f}",
        ],
        [
            "rbias_range",
            f"{summary['rbias']['min']:.4f} -> {summary['rbias']['max']:.4f}",
        ],
        [
            "axis_dist_mean",
            f"{summary['axis_distance']['mean']:.4f}",
        ],
        [
            "s_range",
            f"{summary['s']['min']:.4f} -> {summary['s']['max']:.4f}",
        ],
        [
            "r_mean",
            f"{summary['r']['mean']:.4f}",
        ],
        [
            "timeseries_plot",
            plot_paths.get("timeseries", Path("not-generated")).name,
        ],
        [
            "occupancy_plot",
            plot_paths.get("occupancy", Path("not-generated")).name,
        ],
        ["Result", "PLOTTED"],
    ]
    Outputer.print_table(
        ["Metric", "Value"],
        table_rows,
        title="SinkMeta Performance: DNA+Coumarin 3D Positional Long NVT",
    )

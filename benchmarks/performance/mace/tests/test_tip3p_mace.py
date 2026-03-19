import os
import math

import numpy as np
import pytest

from benchmarks.utils import Extractor, Outputer, Runner
from benchmarks.performance.mace.tests.utils import (
    build_atoms,
    compute_reference_forces,
    compute_oo_rdf,
    dump_json,
    load_metrics,
    load_plugin_forces,
    load_plugin_positions,
    load_box_trajectory,
    load_coordinate_trajectory,
    oxygen_indices_from_atom_names,
    resolve_prips_plugin_path,
    save_rdf_plot,
    summarize_force_errors,
    write_mace_plugin_script,
    write_mdin,
)

MACE_FAMILY = "off"
MACE_MODEL = "small"


def resolve_mace_device():
    env_name = os.environ.get("PIXI_ENVIRONMENT_NAME", "").strip().lower()
    if "cpu" in env_name:
        return "cpu"
    if "cuda" in env_name:
        return "cuda"
    if "hip" in env_name or "rocm" in env_name:
        # PyTorch ROCm typically uses CUDA-compatible device strings.
        return "cuda"

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


MACE_DEVICE = resolve_mace_device()


def test_tip3p_mace_force_inference_perf(
    statics_path,
    outputs_path,
    mpi_np,
):
    pytest.importorskip("mace")

    plugin_path = resolve_prips_plugin_path()
    case_dir = Outputer.prepare_output_case(
        statics_path=statics_path,
        outputs_path=outputs_path,
        case_name="tip3p_mace",
        mpi_np=mpi_np,
        run_name="tip3p_mace_force",
    )
    write_mace_plugin_script(
        case_dir,
        family=MACE_FAMILY,
        model=MACE_MODEL,
        device=MACE_DEVICE,
    )
    write_mdin(
        case_dir,
        plugin_path,
        step_limit=1,
        write_trajectory_interval=1,
    )

    Runner.run_sponge(case_dir, timeout=3600, mpi_np=mpi_np)

    metrics = load_metrics(case_dir)
    atom_count = Extractor.read_first_field_int(
        case_dir / "tip3p_coordinate.txt"
    )
    plugin_forces = load_plugin_forces(case_dir)
    plugin_positions = load_plugin_positions(case_dir)
    sponge_forces = Extractor.extract_sponge_forces(case_dir, atom_count)
    reference_forces = compute_reference_forces(
        case_dir,
        family=MACE_FAMILY,
        model=MACE_MODEL,
        device=MACE_DEVICE,
        positions=plugin_positions,
    )
    plugin_error = summarize_force_errors(plugin_forces, reference_forces)
    output_error = summarize_force_errors(sponge_forces, reference_forces)

    rows = [
        ["Case", "tip3p_mace_force"],
        ["Family", metrics["family"]],
        ["Model", metrics["model"]],
        ["Device", metrics["device"]],
        ["Backend", metrics["backend"]],
        ["Atoms", str(metrics["atom_count"])],
        ["ForceCalls", str(metrics["force_calls"])],
        ["FirstCall(ms)", f"{metrics['first_call_ms']:.3f}"],
        ["MeanCall(ms)", f"{metrics['mean_call_ms']:.3f}"],
        ["SteadyMean(ms)", f"{metrics['steady_mean_call_ms']:.3f}"],
        ["MaxAbsForceErr", f"{plugin_error['max_abs']:.6e}"],
        ["MeanAbsForceErr", f"{plugin_error['mean_abs']:.6e}"],
        ["MaxAbsOutputErr", f"{output_error['max_abs']:.6e}"],
    ]
    Outputer.print_table(
        ["Metric", "Value"],
        rows,
        title="Performance Benchmark: TIP3P MACE Force Inference",
    )

    assert metrics["atom_count"] == atom_count
    assert metrics["force_calls"] >= 1
    assert metrics["first_call_ms"] > 0.0
    assert metrics["mean_call_ms"] > 0.0
    assert metrics["steady_mean_call_ms"] > 0.0
    assert all(
        math.isfinite(record["elapsed_s"]) for record in metrics["records"]
    )
    assert all(record["elapsed_s"] > 0.0 for record in metrics["records"])
    assert all(
        math.isfinite(record["force_l2"]) for record in metrics["records"]
    )
    assert all(record["force_l2"] > 0.0 for record in metrics["records"])
    assert plugin_error["max_abs"] <= 5.0e-4
    assert output_error["max_abs"] <= 5.0e-4


def test_tip3p_mace_long_nvt_perf(
    statics_path,
    outputs_path,
    mpi_np,
    mace_steps,
):
    pytest.importorskip("mace")
    report_interval = min(1000, mace_steps)

    plugin_path = resolve_prips_plugin_path()
    case_dir = Outputer.prepare_output_case(
        statics_path=statics_path,
        outputs_path=outputs_path,
        case_name="tip3p_mace",
        mpi_np=mpi_np,
        run_name="tip3p_mace_long_nvt",
    )
    write_mace_plugin_script(
        case_dir,
        family=MACE_FAMILY,
        model=MACE_MODEL,
        device=MACE_DEVICE,
    )
    write_mdin(
        case_dir,
        plugin_path,
        step_limit=mace_steps,
        mode="nvt",
        dt=0.0002,
        write_mdout_interval=mace_steps,
        write_information_interval=mace_steps,
        write_trajectory_interval=report_interval,
        thermostat="middle_langevin",
        thermostat_tau=0.2,
        thermostat_seed=2026,
        target_temperature=300.0,
    )

    Runner.run_sponge(case_dir, timeout=14400, mpi_np=mpi_np)

    metrics = load_metrics(case_dir)
    atom_count = Extractor.read_first_field_int(
        case_dir / "tip3p_coordinate.txt"
    )
    initial_positions = build_atoms(case_dir).positions.copy()
    plugin_positions = load_plugin_positions(case_dir)
    plugin_forces = load_plugin_forces(case_dir)
    sponge_forces = Extractor.extract_sponge_forces(case_dir, atom_count)
    reference_forces = compute_reference_forces(
        case_dir,
        family=MACE_FAMILY,
        model=MACE_MODEL,
        device=MACE_DEVICE,
        positions=plugin_positions,
    )
    plugin_error = summarize_force_errors(plugin_forces, reference_forces)
    output_error = summarize_force_errors(sponge_forces, reference_forces)
    displacement = plugin_positions - initial_positions
    max_abs_displacement = float(np.max(np.abs(displacement)))
    trajectory = load_coordinate_trajectory(case_dir, atom_count)
    box_lengths = load_box_trajectory(case_dir)
    oxygen_indices = oxygen_indices_from_atom_names(case_dir)
    rdf = compute_oo_rdf(
        trajectory,
        box_lengths,
        oxygen_indices,
        r_max=8.0,
        bin_width=0.05,
    )
    dump_json(
        {
            "frame_count": rdf["frame_count"],
            "oxygen_count": rdf["oxygen_count"],
            "peak_r": rdf["peak_r"],
            "peak_g_r": rdf["peak_g_r"],
            "r_max": rdf["r_max"],
            "bin_width": rdf["bin_width"],
        },
        case_dir / "oo_rdf_summary.json",
    )
    save_rdf_plot(
        rdf["r"],
        rdf["g_r"],
        case_dir / "oo_rdf.png",
        title="TIP3P MACE O-O RDF",
    )

    rows = [
        ["Case", "tip3p_mace_long_nvt"],
        ["FirstCall(ms)", f"{metrics['first_call_ms']:.3f}"],
        ["MeanCall(ms)", f"{metrics['mean_call_ms']:.3f}"],
        ["SteadyMean(ms)", f"{metrics['steady_mean_call_ms']:.3f}"],
        ["Frames", str(rdf["frame_count"])],
        ["OxygenCount", str(rdf["oxygen_count"])],
        ["OO_RDF_Peak_r(A)", f"{rdf['peak_r']:.3f}"],
        ["OO_RDF_Peak_g(r)", f"{rdf['peak_g_r']:.3f}"],
    ]
    Outputer.print_table(
        ["Metric", "Value"],
        rows,
        title="Performance Benchmark: TIP3P MACE Long NVT",
    )

    assert metrics["atom_count"] == atom_count
    assert metrics["force_calls"] >= mace_steps
    assert metrics["first_call_ms"] > 0.0
    assert metrics["mean_call_ms"] > 0.0
    assert metrics["steady_mean_call_ms"] > 0.0
    assert max_abs_displacement > 1.0e-4
    assert all(
        math.isfinite(record["elapsed_s"]) for record in metrics["records"]
    )
    assert all(record["elapsed_s"] > 0.0 for record in metrics["records"])
    assert all(
        math.isfinite(record["force_l2"]) for record in metrics["records"]
    )
    assert all(record["force_l2"] > 0.0 for record in metrics["records"])
    assert plugin_error["max_abs"] <= 5.0e-4
    assert output_error["max_abs"] <= 5.0e-4

import time

from benchmarks.utils import Outputer, Runner

from benchmarks.performance.reaxff.tests.utils import (
    dump_summary_json,
    parse_mdout_series,
    read_atom_count_from_coordinate,
    summarize_runtime,
    write_reaxff_perf_mdin,
)


def test_reaxff_petn_nve_throughput(
    statics_path, outputs_path, mpi_np, reaxff_steps
):
    case_dir = Outputer.prepare_output_case(
        statics_path=statics_path,
        outputs_path=outputs_path,
        case_name="petn_16240",
        mpi_np=mpi_np,
        run_name="petn_nve_perf",
    )

    write_reaxff_perf_mdin(
        case_dir,
        source_mdin="mdin.spg.toml",
        output_mdin="petn.perf.spg.toml",
        step_limit=reaxff_steps,
        write_information_interval=reaxff_steps,
        rst="petn_perf",
    )

    start = time.perf_counter()
    Runner.run_sponge(
        case_dir,
        mdin_name="petn.perf.spg.toml",
        timeout=7200,
        mpi_np=mpi_np,
    )
    elapsed_s = time.perf_counter() - start

    rows = parse_mdout_series(case_dir / "mdout.txt")
    runtime = summarize_runtime(rows, elapsed_s=elapsed_s)
    atom_count = read_atom_count_from_coordinate(case_dir / "coordinate.txt")

    summary = {
        "atom_count": atom_count,
        "elapsed_s": runtime["elapsed_s"],
        "ns_per_day": runtime["ns_per_day"],
        "samples": runtime["samples"],
        "simulated_ps": runtime["simulated_ps"],
        "steps_per_s": runtime["steps_per_s"],
        "total_steps": runtime["total_steps"],
    }
    dump_summary_json(summary, case_dir / "throughput_summary.json")

    Outputer.print_table(
        ["Metric", "Value"],
        [
            ["Case", "petn_16240_nve_perf"],
            ["Atoms", str(atom_count)],
            ["Steps", str(runtime["total_steps"])],
            ["Samples", str(runtime["samples"])],
            ["Elapsed(s)", f"{runtime['elapsed_s']:.3f}"],
            ["Simulated(ps)", f"{runtime['simulated_ps']:.6f}"],
            ["Steps/s", f"{runtime['steps_per_s']:.3f}"],
            ["ns/day", f"{runtime['ns_per_day']:.6f}"],
        ],
        title="Performance Benchmark: ReaxFF PETN 16240 NVE Throughput",
    )

    assert runtime["total_steps"] == reaxff_steps
    assert runtime["samples"] >= 2
    assert runtime["elapsed_s"] > 0.0
    assert runtime["steps_per_s"] > 0.0
    assert runtime["ns_per_day"] > 0.0

import json
import math
import statistics
from pathlib import Path

from benchmarks.utils import (
    Outputer,
    Runner,
    Extractor,
)

CONSTANT_KB_KCAL_PER_MOL_K = 0.00198716
SECONDS_PER_DAY = 86400.0


def run_sponge(case_dir, mdin_name, timeout=2400, mpi_np=None):
    return Runner.run_sponge(
        case_dir,
        mpi_np=mpi_np,
        mdin_name=mdin_name,
        timeout=timeout,
    )


def _rewrite_top_level_fields(
    case_dir,
    *,
    source_mdin,
    output_mdin,
    updates,
):
    lines = Path(case_dir, source_mdin).read_text().splitlines()
    updates = dict(updates)

    updated_lines = []
    in_reaxff_block = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("["):
            in_reaxff_block = stripped.lower() == "[reaxff]"
        if not in_reaxff_block:
            for key, value in updates.items():
                if stripped.startswith(key):
                    updated_lines.append(f"{key} = {value}")
                    break
            else:
                updated_lines.append(line)
            continue
        updated_lines.append(line)

    for key, value in updates.items():
        if any(l.strip().startswith(key) for l in updated_lines):
            continue
        updated_lines.append(f"{key} = {value}")

    Path(case_dir, output_mdin).write_text("\n".join(updated_lines) + "\n")


def write_reaxff_perf_mdin(
    case_dir,
    *,
    source_mdin="mdin.spg.toml",
    output_mdin="perf.spg.toml",
    step_limit=100,
    write_information_interval=None,
    rst="perf",
):
    if write_information_interval is None:
        write_information_interval = max(1, step_limit)

    _rewrite_top_level_fields(
        case_dir,
        source_mdin=source_mdin,
        output_mdin=output_mdin,
        updates={
            "step_limit": step_limit,
            "write_information_interval": write_information_interval,
            "rst": json.dumps(rst),
        },
    )


def read_atom_count_from_coordinate(coordinate_path):
    return Extractor.read_first_field_int(coordinate_path)


def parse_mdout_series(mdout_path):
    return Extractor.parse_mdout_rows(
        mdout_path,
        ["step", "time", "temperature", "potential"],
    )


def summarize_energy_stability(nve_rows, *, dof):
    if dof <= 0:
        raise ValueError(f"Invalid dof: {dof}")

    energies = []
    times = []
    for r in nve_rows:
        kinetic = 0.5 * dof * CONSTANT_KB_KCAL_PER_MOL_K * r["temperature"]
        total = r["potential"] + kinetic
        energies.append(total)
        times.append(r["time"])

    e0 = energies[0]
    drifts = [e - e0 for e in energies]
    final_drift = drifts[-1]
    max_abs_drift = max(abs(v) for v in drifts)

    mean_e = statistics.fmean(energies)
    std_e = statistics.pstdev(energies)

    x_mean = statistics.fmean(times)
    y_mean = statistics.fmean(energies)
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(times, energies))
    den = sum((x - x_mean) ** 2 for x in times)
    slope = num / den if den > 0 else 0.0

    return {
        "samples": len(energies),
        "dof": dof,
        "e0": e0,
        "mean_e": mean_e,
        "std_e": std_e,
        "final_drift": final_drift,
        "max_abs_drift": max_abs_drift,
        "final_rel_drift": abs(final_drift) / max(abs(e0), 1e-12),
        "max_rel_drift": max_abs_drift / max(abs(e0), 1e-12),
        "slope_kcal_per_mol_ps": slope,
    }


def dump_summary_json(summary, out_path):
    Path(out_path).write_text(json.dumps(summary, indent=2, sort_keys=True))


def summarize_runtime(mdout_rows, *, elapsed_s):
    if elapsed_s <= 0.0:
        raise ValueError(f"Invalid elapsed time: {elapsed_s}")

    last = mdout_rows[-1]
    total_steps = int(last["step"])
    simulated_ps = float(last["time"])
    return {
        "samples": len(mdout_rows),
        "total_steps": total_steps,
        "simulated_ps": simulated_ps,
        "elapsed_s": float(elapsed_s),
        "steps_per_s": total_steps / elapsed_s,
        "ns_per_day": simulated_ps * SECONDS_PER_DAY / elapsed_s / 1000.0,
    }


def save_energy_plots(nve_rows, *, dof, output_dir):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    times = [r["time"] for r in nve_rows]
    temperatures = [r["temperature"] for r in nve_rows]
    potentials = [r["potential"] for r in nve_rows]
    kinetics = [
        0.5 * dof * CONSTANT_KB_KCAL_PER_MOL_K * t for t in temperatures
    ]
    totals = [u + k for u, k in zip(potentials, kinetics)]
    drifts = [e - totals[0] for e in totals]

    fig, axs = plt.subplots(2, 1, figsize=(9.5, 6.5), dpi=160, sharex=True)
    axs[0].plot(times, potentials, lw=1.0, label="Potential U")
    axs[0].plot(times, kinetics, lw=1.0, label="Kinetic K")
    axs[0].plot(times, totals, lw=1.3, label="Total E=U+K")
    axs[0].set_ylabel("Energy (kcal/mol)")
    axs[0].set_title("CHO NVE Energy Components")
    axs[0].grid(alpha=0.25)
    axs[0].legend(frameon=False, ncol=3)

    axs[1].plot(times, drifts, lw=1.2, color="tab:red")
    axs[1].axhline(0.0, ls="--", lw=1.0, color="gray")
    axs[1].set_xlabel("Time (ps)")
    axs[1].set_ylabel("E(t)-E(0) (kcal/mol)")
    axs[1].set_title("CHO NVE Total-Energy Drift")
    axs[1].grid(alpha=0.25)

    fig.tight_layout()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(output_dir) / "cho_nve_energy_statistics.png")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(7.0, 4.6), dpi=160)
    ax2.hist(drifts, bins=50, color="tab:blue", alpha=0.8)
    ax2.set_xlabel("E(t)-E(0) (kcal/mol)")
    ax2.set_ylabel("Count")
    ax2.set_title("CHO NVE Drift Distribution")
    ax2.grid(alpha=0.25)
    fig2.tight_layout()
    fig2.savefig(Path(output_dir) / "cho_nve_drift_hist.png")
    plt.close(fig2)

    return True

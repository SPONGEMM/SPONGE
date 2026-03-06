import json
import math
import os
import shlex
import shutil
import statistics
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
CONSTANT_KB_KCAL_PER_MOL_K = 0.00198716


def print_validation_table(headers, rows, title=None):
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            text = str(val)
            col_widths[i] = max(col_widths[i], len(text))
    col_widths = [w + 2 for w in col_widths]
    row_fmt = " | ".join([f"{{:<{w}}}" for w in col_widths])
    divider = "-" * (sum(col_widths) + 3 * (len(headers) - 1))

    if title:
        print(f"\n{title}")
    else:
        print()
    print(divider)
    print(row_fmt.format(*headers))
    print(divider)
    for row in rows:
        print(row_fmt.format(*[str(v) for v in row]))
    print(divider)


def prepare_output_case(statics_path, outputs_path, case_name, run_tag=None):
    static_case = statics_path / case_name
    if not static_case.exists():
        raise FileNotFoundError(f"Static case not found: {static_case}")

    case_dir = outputs_path / (run_tag or case_name)
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(static_case, case_dir)
    return case_dir


def resolve_sponge_command():
    env_bin = os.environ.get("SPONGE_BIN")
    if env_bin:
        return shlex.split(env_bin)

    local_cpu_bin = REPO_ROOT / "build-cpu" / "SPONGE"
    if local_cpu_bin.exists():
        return [str(local_cpu_bin)]

    local_build_bin = REPO_ROOT / "build" / "SPONGE"
    if local_build_bin.exists():
        return [str(local_build_bin)]

    sponge_in_path = shutil.which("SPONGE")
    if sponge_in_path:
        return [sponge_in_path]

    raise FileNotFoundError(
        "Cannot find SPONGE binary. Build it first or set SPONGE_BIN."
    )


def is_cuda_init_failure(error_text):
    lowered = error_text.lower()
    return (
        "fail to initialize cuda" in lowered
        or "spongeerrormallocfailed raised by controller::init_device"
        in lowered
    )


def run_sponge(case_dir, mdin_name, log_name, timeout=2400):
    cmd = resolve_sponge_command() + ["-mdin", mdin_name]
    result = subprocess.run(
        cmd,
        cwd=case_dir,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    output = result.stdout + "\n" + result.stderr
    Path(case_dir, log_name).write_text(output)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed in {case_dir} with code {result.returncode}\n"
            f"Command: {' '.join(cmd)}\n"
            f"Output tail:\n{output[-4000:]}"
        )
    return output


def write_nve_long_mdin(
    case_dir,
    *,
    source_nve_mdin="nve.spg.toml",
    output_mdin="nve.long.spg.toml",
    step_limit=20000,
    write_information_interval=10,
    rst="nve_long",
):
    lines = Path(case_dir, source_nve_mdin).read_text().splitlines()

    updated_lines = []
    in_reaxff_block = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("["):
            in_reaxff_block = stripped.lower() == "[reaxff]"
        if not in_reaxff_block and stripped.startswith("step_limit"):
            updated_lines.append(f"step_limit = {step_limit}")
            continue
        if not in_reaxff_block and stripped.startswith(
            "write_information_interval"
        ):
            updated_lines.append(
                f"write_information_interval = {write_information_interval}"
            )
            continue
        if not in_reaxff_block and stripped.startswith("rst"):
            updated_lines.append(f'rst = "{rst}"')
            continue
        updated_lines.append(line)

    if not any(l.strip().startswith("rst") for l in updated_lines):
        updated_lines.append(f'rst = "{rst}"')

    Path(case_dir, output_mdin).write_text("\n".join(updated_lines) + "\n")


def read_atom_count_from_coordinate(coordinate_path):
    first_line = Path(coordinate_path).read_text().splitlines()[0]
    fields = first_line.split()
    if not fields:
        raise ValueError(f"Invalid coordinate header: {coordinate_path}")
    return int(fields[0])


def parse_mdout_series(mdout_path):
    lines = Path(mdout_path).read_text().splitlines()
    if len(lines) < 2:
        raise ValueError(f"Invalid mdout file: {mdout_path}")

    headers = lines[0].split()
    required = ["step", "time", "temperature", "potential"]
    for key in required:
        if key not in headers:
            raise ValueError(f"Missing '{key}' column in {mdout_path}")

    step_idx = headers.index("step")
    time_idx = headers.index("time")
    temp_idx = headers.index("temperature")
    pot_idx = headers.index("potential")

    rows = []
    for line in lines[1:]:
        fields = line.split()
        if len(fields) <= max(step_idx, time_idx, temp_idx, pot_idx):
            continue
        try:
            row = {
                "step": int(fields[step_idx]),
                "time": float(fields[time_idx]),
                "temperature": float(fields[temp_idx]),
                "potential": float(fields[pot_idx]),
            }
        except ValueError:
            continue
        rows.append(row)

    if not rows:
        raise ValueError(f"No frame data parsed from {mdout_path}")
    return rows


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
    kinetics = [0.5 * dof * CONSTANT_KB_KCAL_PER_MOL_K * t for t in temperatures]
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

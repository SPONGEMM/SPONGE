import re
import shutil
import statistics
import subprocess
from pathlib import Path


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


def _run_command(cmd, cwd, timeout=600):
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    output = result.stdout + "\n" + result.stderr
    Path(cwd, "run.log").write_text(output)
    if result.returncode != 0:
        cmd0 = Path(str(cmd[0])).name.lower() if cmd else ""
        if cmd0 in {"sponge", "sponge.exe"}:
            print("\n[SPONGE stdout]\n")
            print(result.stdout)
            print("\n[SPONGE stderr]\n")
            print(result.stderr)
        raise RuntimeError(
            f"Command failed in {cwd} with code {result.returncode}\n"
            f"Command: {' '.join(cmd)}\n"
            f"Output tail:\n{output[-3000:]}"
        )
    return output


def is_cuda_init_failure(error_text):
    lowered = error_text.lower()
    return (
        "fail to initialize cuda" in lowered
        or "spongeerrormallocfailed raised by controller::init_device"
        in lowered
    )


def write_thermostat_mdin(
    case_dir,
    *,
    step_limit=500,
    dt=0.001,
    cutoff=12.0,
    thermostat="middle_langevin",
    thermostat_mode=None,
    constrain_mode=None,
    target_temperature=300.0,
    thermostat_tau=0.01,
    thermostat_seed=2026,
    default_in_file_prefix="sys_flexible",
    md_name="benchmark thermostat validation",
):
    mdin = (
        f'md_name = "{md_name}"\n'
        'mode = "nvt"\n'
        f"step_limit = {step_limit}\n"
        f"dt = {dt}\n"
        f"cutoff = {cutoff}\n"
        f'thermostat = "{thermostat}"\n'
        f"thermostat_tau = {thermostat_tau}\n"
        f"thermostat_seed = {thermostat_seed}\n"
        f"target_temperature = {target_temperature}\n"
        f'default_in_file_prefix = "{default_in_file_prefix}"\n'
        "print_zeroth_frame = 1\n"
        "write_mdout_interval = 1\n"
        "write_information_interval = 1\n"
    )
    if thermostat_mode is not None:
        mdin += f'thermostat_mode = "{thermostat_mode}"\n'
    if constrain_mode is not None:
        mdin += f'constrain_mode = "{constrain_mode}"\n'
    Path(case_dir, "mdin.spg.toml").write_text(mdin)


def run_sponge_thermostat(case_dir, timeout=600):
    cmd = ["SPONGE", "-mdin", "mdin.spg.toml"]
    return _run_command(cmd, cwd=case_dir, timeout=timeout)


def parse_temperature_series(mdout_path):
    lines = Path(mdout_path).read_text().splitlines()
    if len(lines) < 2:
        raise ValueError(f"Invalid mdout file: {mdout_path}")

    headers = lines[0].split()
    if "temperature" not in headers:
        raise ValueError(f"No temperature column in {mdout_path}")
    temp_idx = headers.index("temperature")

    temperatures = []
    for line in lines[1:]:
        fields = line.split()
        if len(fields) <= temp_idx:
            continue
        try:
            int(fields[0])
            temp = float(fields[temp_idx])
        except ValueError:
            continue
        temperatures.append(temp)

    if not temperatures:
        raise ValueError(f"No temperature samples parsed from {mdout_path}")
    return temperatures


def read_atom_count_from_mass(mass_path):
    first_line = Path(mass_path).read_text().splitlines()[0]
    parts = first_line.split()
    if not parts:
        raise ValueError(f"Invalid mass file header: {mass_path}")
    return int(parts[0])


def read_constrain_pair_count_from_runlog(runlog_path):
    text = Path(runlog_path).read_text()
    match = re.search(r"constrain pair number is\s+(\d+)", text)
    if not match:
        return 0
    return int(match.group(1))


def read_ug_count_from_runlog(runlog_path):
    text = Path(runlog_path).read_text()
    match = re.search(r"md_info->ug\.ug_numbers:\s+(\d+)", text)
    if not match:
        match = re.search(
            r"max_atom_numbers=\d+,\s*max_res_numbers=(\d+)", text
        )
    if not match:
        raise ValueError(f"Cannot parse update_group count from {runlog_path}")
    return int(match.group(1))


def evaluate_temperature_distribution(
    temperatures,
    *,
    target_temperature=300.0,
    n_atoms=None,
    burn_in=100,
    constrain_pair_count=0,
    ug_count=None,
):
    if n_atoms is None or n_atoms <= 0:
        raise ValueError(f"Invalid n_atoms: {n_atoms}")
    if burn_in >= len(temperatures) - 1:
        raise ValueError(
            f"burn_in={burn_in} is too large for sample count {len(temperatures)}"
        )

    sample = temperatures[burn_in:]
    if ug_count is not None:
        if ug_count <= 0:
            raise ValueError(f"Invalid ug_count: {ug_count}")
        dof = 3 * ug_count
    else:
        dof = 3 * n_atoms - constrain_pair_count
    if dof <= 1:
        raise ValueError(
            f"Invalid degree of freedom after constraints: dof={dof}, "
            f"n_atoms={n_atoms}, constrain_pair_count={constrain_pair_count}"
        )
    mean_temp = statistics.fmean(sample)
    std_temp = statistics.stdev(sample)
    expected_std = target_temperature * (2.0 / dof) ** 0.5
    std_ratio = std_temp / expected_std

    return {
        "total_samples": len(temperatures),
        "sample_count": len(sample),
        "dof": dof,
        "mean_temp": mean_temp,
        "std_temp": std_temp,
        "expected_std": expected_std,
        "std_ratio": std_ratio,
    }

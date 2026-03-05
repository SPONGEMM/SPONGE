import shutil
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


def print_validation_vertical(headers, row, title=None):
    metric_col = "Metric"
    value_col = "Value"
    metric_width = max(len(metric_col), *(len(str(h)) for h in headers)) + 2
    value_width = max(len(value_col), *(len(str(v)) for v in row)) + 2
    row_fmt = f"{{:<{metric_width}}} | {{:<{value_width}}}"
    divider = "-" * (metric_width + value_width + 3)

    if title:
        print(f"\n{title}")
    else:
        print()
    print(divider)
    print(row_fmt.format(metric_col, value_col))
    print(divider)
    for metric, value in zip(headers, row):
        print(row_fmt.format(str(metric), str(value)))
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


def _run_command(cmd, cwd, timeout=900):
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


def run_sponge(case_dir, timeout=900):
    cmd = ["SPONGE", "-mdin", "mdin.spg.toml"]
    return _run_command(cmd, cwd=case_dir, timeout=timeout)


def is_cuda_init_failure(error_text):
    lowered = error_text.lower()
    return (
        "fail to initialize cuda" in lowered
        or "spongeerrormallocfailed raised by controller::init_device"
        in lowered
    )


def write_mdin(
    case_dir,
    *,
    hard_wall_z_low=5.0,
    hard_wall_z_high,
    step_limit=200,
    soft_walls_in_file=None,
):
    mdin = (
        'md_name = "validation tip3p walls"\n'
        'mode = "nvt"\n'
        f"step_limit = {step_limit}\n"
        "dt = 0.001\n"
        "cutoff = 8.0\n"
        'default_in_file_prefix = "tip3p"\n'
        'constrain_mode = "SETTLE"\n'
        "print_zeroth_frame = 1\n"
        "write_mdout_interval = 1\n"
        "write_information_interval = 20\n"
        'thermostat = "middle_langevin"\n'
        "thermostat_tau = 0.01\n"
        "thermostat_seed = 2026\n"
        "target_temperature = 300.0\n"
        f"hard_wall_z_low = {hard_wall_z_low}\n"
        f"hard_wall_z_high = {hard_wall_z_high}\n"
    )
    if soft_walls_in_file is not None:
        mdin += f'soft_walls_in_file = "{soft_walls_in_file}"\n'
    Path(case_dir, "mdin.spg.toml").write_text(mdin)


def parse_restart_coordinate_zmax(restart_coordinate_path):
    lines = Path(restart_coordinate_path).read_text().splitlines()
    if len(lines) < 2:
        raise ValueError(
            f"Invalid restart coordinate file: {restart_coordinate_path}"
        )
    header = lines[0].split()
    if not header:
        raise ValueError(
            f"Invalid restart coordinate header: {restart_coordinate_path}"
        )
    atom_count = int(header[0])
    coordinate_lines = lines[1 : 1 + atom_count]

    z_values = []
    for line in coordinate_lines:
        fields = line.split()
        if len(fields) < 3:
            continue
        z_values.append(float(fields[2]))
    if not z_values:
        raise ValueError(
            f"No coordinates parsed from {restart_coordinate_path}"
        )
    return max(z_values)

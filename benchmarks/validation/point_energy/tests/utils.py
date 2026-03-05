import json
import math
import os
import shlex
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
IGNORE_COLUMNS = {"step", "time", "temperature"}


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


def _resolve_binary(path_or_name):
    if not path_or_name:
        return None
    p = Path(path_or_name)
    if p.is_file():
        return str(p)
    found = shutil.which(path_or_name)
    return found


def resolve_binary_triplet():
    ref_env = os.environ.get("SPONGE_POINT_REF_BIN")
    cpu_env = os.environ.get("SPONGE_POINT_CPU_BIN")
    gpu_env = os.environ.get("SPONGE_POINT_GPU_BIN")

    default_local = REPO_ROOT / "build" / "SPONGE"
    default_ref = str(default_local) if default_local.exists() else "SPONGE"

    ref_bin = _resolve_binary(ref_env or os.environ.get("SPONGE_BIN") or default_ref)
    if not ref_bin:
        raise FileNotFoundError(
            "Cannot resolve reference SPONGE binary. "
            "Set SPONGE_POINT_REF_BIN or SPONGE_BIN."
        )

    cpu_bin = _resolve_binary(cpu_env or ref_bin)
    if not cpu_bin:
        raise FileNotFoundError(
            "Cannot resolve CPU SPONGE binary. Set SPONGE_POINT_CPU_BIN."
        )

    gpu_bin = _resolve_binary(gpu_env or ref_bin)
    if not gpu_bin:
        raise FileNotFoundError(
            "Cannot resolve GPU SPONGE binary. Set SPONGE_POINT_GPU_BIN."
        )

    return {
        "reference": ref_bin,
        "cpu": cpu_bin,
        "gpu": gpu_bin,
        "gpu_is_fallback": gpu_env is None,
        "cpu_is_fallback": cpu_env is None,
    }


def _run_command(cmd, cwd, log_name, timeout=1200):
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    output = result.stdout + "\n" + result.stderr
    Path(cwd, log_name).write_text(output)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed in {cwd} with code {result.returncode}\n"
            f"Command: {' '.join(cmd)}\n"
            f"Output tail:\n{output[-3000:]}"
        )
    return output


def run_point_energy(case_dir, *, sponge_bin, mdin_name, run_tag, mpi_np=1, timeout=1200):
    if mpi_np <= 1:
        cmd = [sponge_bin, "-mdin", mdin_name]
    else:
        mpirun_cmd = shlex.split(os.environ.get("SPONGE_POINT_MPIRUN", "mpirun"))
        cmd = mpirun_cmd + ["-np", str(mpi_np), sponge_bin, "-mdin", mdin_name]

    log_name = f"run_{run_tag}.log"
    _run_command(cmd, cwd=case_dir, log_name=log_name, timeout=timeout)

    mdout_path = Path(case_dir, "mdout.txt")
    tagged_mdout = Path(case_dir, f"mdout_{run_tag}.txt")
    shutil.copyfile(mdout_path, tagged_mdout)
    return tagged_mdout, Path(case_dir, log_name)


def parse_mdout_one_frame(mdout_path):
    lines = Path(mdout_path).read_text().splitlines()
    if len(lines) < 2:
        raise ValueError(f"Invalid mdout file: {mdout_path}")

    headers = lines[0].split()
    values = lines[1].split()
    if len(values) < len(headers):
        raise ValueError(f"Unexpected mdout row format: {mdout_path}")

    data = {}
    for key, value in zip(headers, values):
        if key == "step":
            data[key] = int(value)
            continue
        try:
            data[key] = float(value)
        except ValueError:
            data[key] = value
    return data


def load_reference(reference_json_path):
    obj = json.loads(Path(reference_json_path).read_text())
    return obj["energies"]


def dump_json(data, out_path):
    Path(out_path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def parse_pme_backend(run_log_path):
    text = Path(run_log_path).read_text()
    marker = "PME backend library:"
    for line in text.splitlines():
        if marker in line:
            return line.split(marker, 1)[1].strip()
    return "unknown"


def compare_energies(reference, current):
    errors = {}
    for key, ref_val in reference.items():
        if key in IGNORE_COLUMNS:
            continue
        cur_val = current.get(key)
        if cur_val is None:
            continue
        if not isinstance(ref_val, (int, float)) or not isinstance(cur_val, (int, float)):
            continue
        if not math.isfinite(float(ref_val)) or not math.isfinite(float(cur_val)):
            continue
        errors[key] = {
            "reference": float(ref_val),
            "current": float(cur_val),
            "abs_error": abs(float(cur_val) - float(ref_val)),
        }
    return errors


def summarize_errors(term_errors):
    if not term_errors:
        return {"max_abs_error": 0.0, "max_abs_error_key": "N/A"}

    max_key = max(term_errors.keys(), key=lambda k: term_errors[k]["abs_error"])
    return {
        "max_abs_error": term_errors[max_key]["abs_error"],
        "max_abs_error_key": max_key,
    }

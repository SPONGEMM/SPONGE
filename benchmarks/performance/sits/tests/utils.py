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


def is_cuda_init_failure(error_text):
    lowered = error_text.lower()
    return (
        "fail to initialize cuda" in lowered
        or "spongeerrormallocfailed raised by controller::init_device"
        in lowered
    )


def write_meta_mdin(
    case_dir,
    *,
    step_limit=400,
    dt=0.001,
    cutoff=12.0,
    thermostat="middle_langevin",
    thermostat_tau=0.01,
    thermostat_seed=2026,
    target_temperature=300.0,
    write_information_interval=20,
):
    mdin = (
        'md_name = "benchmark alanine_dipeptide_tip3p_water enhanced_sampling"\n'
        'mode = "nvt"\n'
        f"step_limit = {step_limit}\n"
        f"dt = {dt}\n"
        f"cutoff = {cutoff}\n"
        f'thermostat = "{thermostat}"\n'
        f"thermostat_tau = {thermostat_tau}\n"
        f"thermostat_seed = {thermostat_seed}\n"
        f"target_temperature = {target_temperature}\n"
        'default_in_file_prefix = "sys_flexible"\n'
        'cv_in_file = "cv.txt"\n'
        "print_zeroth_frame = 1\n"
        "write_mdout_interval = 1\n"
        f"write_information_interval = {write_information_interval}\n"
    )
    Path(case_dir, "mdin.spg.toml").write_text(mdin)


def write_sits_mdin(
    case_dir,
    *,
    step_limit=200,
    dt=0.001,
    cutoff=12.0,
    thermostat="middle_langevin",
    thermostat_tau=0.01,
    thermostat_seed=2026,
    target_temperature=300.0,
    write_information_interval=20,
    write_mdout_interval=1,
    default_in_file_prefix="sys_flexible",
    sits_mode="iteration",
    sits_atom_numbers=23,
    sits_k_numbers=4,
    sits_t_low=280.0,
    sits_t_high=420.0,
    sits_record_interval=1,
    sits_update_interval=20,
    sits_nk_fix=False,
    sits_nk_in_file=None,
    sits_pe_a=None,
    sits_pe_b=None,
    constrain_mode=None,
    coordinate_in_file=None,
    velocity_in_file=None,
    write_restart_file_interval=None,
):
    mdin = (
        'md_name = "benchmark alanine_dipeptide_tip3p_water SITS"\n'
        'mode = "nvt"\n'
        f"step_limit = {step_limit}\n"
        f"dt = {dt}\n"
        f"cutoff = {cutoff}\n"
        f'thermostat = "{thermostat}"\n'
        f"thermostat_tau = {thermostat_tau}\n"
        f"thermostat_seed = {thermostat_seed}\n"
        f"target_temperature = {target_temperature}\n"
        f'default_in_file_prefix = "{default_in_file_prefix}"\n'
    )
    if coordinate_in_file is not None:
        mdin += f'coordinate_in_file = "{coordinate_in_file}"\n'
    if velocity_in_file is not None:
        mdin += f'velocity_in_file = "{velocity_in_file}"\n'
    mdin += (
        "print_zeroth_frame = 1\n"
        f"write_mdout_interval = {write_mdout_interval}\n"
        f"write_information_interval = {write_information_interval}\n"
    )
    if write_restart_file_interval is not None:
        mdin += f"write_restart_file_interval = {write_restart_file_interval}\n"
    mdin += (
        f'SITS_mode = "{sits_mode}"\nSITS_atom_numbers = {sits_atom_numbers}\n'
    )
    if sits_mode == "iteration":
        mdin += (
            f"SITS_k_numbers = {sits_k_numbers}\n"
            f"SITS_T_low = {sits_t_low}\n"
            f"SITS_T_high = {sits_t_high}\n"
            f"SITS_record_interval = {sits_record_interval}\n"
            f"SITS_update_interval = {sits_update_interval}\n"
            f"SITS_nk_fix = {1 if sits_nk_fix else 0}\n"
        )
    if sits_mode == "production":
        mdin += (
            f"SITS_k_numbers = {sits_k_numbers}\n"
            f"SITS_T_low = {sits_t_low}\n"
            f"SITS_T_high = {sits_t_high}\n"
            f"SITS_record_interval = {sits_record_interval}\n"
            f"SITS_update_interval = {sits_update_interval}\n"
            f"SITS_nk_fix = {1 if sits_nk_fix else 0}\n"
        )
        if sits_nk_in_file is not None:
            mdin += f'SITS_nk_in_file = "{sits_nk_in_file}"\n'
    if sits_mode == "empirical":
        mdin += f"SITS_T_low = {sits_t_low}\nSITS_T_high = {sits_t_high}\n"
    if sits_pe_a is not None:
        mdin += f"SITS_pe_a = {sits_pe_a}\n"
    if sits_pe_b is not None:
        mdin += f"SITS_pe_b = {sits_pe_b}\n"
    if constrain_mode is not None:
        mdin += f'constrain_mode = "{constrain_mode}"\n'
    Path(case_dir, "mdin.spg.toml").write_text(mdin)


def write_cv_bias_mdin(
    case_dir,
    *,
    step_limit=200,
    dt=0.001,
    cutoff=12.0,
    thermostat="middle_langevin",
    thermostat_tau=0.01,
    thermostat_seed=2026,
    target_temperature=300.0,
    write_information_interval=20,
):
    mdin = (
        'md_name = "benchmark alanine_dipeptide_tip3p_water CV bias"\n'
        'mode = "nvt"\n'
        f"step_limit = {step_limit}\n"
        f"dt = {dt}\n"
        f"cutoff = {cutoff}\n"
        f'thermostat = "{thermostat}"\n'
        f"thermostat_tau = {thermostat_tau}\n"
        f"thermostat_seed = {thermostat_seed}\n"
        f"target_temperature = {target_temperature}\n"
        'default_in_file_prefix = "sys_flexible"\n'
        'cv_in_file = "cv.txt"\n'
        "print_zeroth_frame = 1\n"
        "write_mdout_interval = 1\n"
        f"write_information_interval = {write_information_interval}\n"
    )
    Path(case_dir, "mdin.spg.toml").write_text(mdin)


def write_cv_mdin(
    case_dir,
    *,
    step_limit=200,
    dt=0.001,
    cutoff=12.0,
    thermostat="middle_langevin",
    thermostat_tau=0.01,
    thermostat_seed=2026,
    target_temperature=300.0,
    write_information_interval=20,
):
    mdin = (
        'md_name = "benchmark alanine_dipeptide_tip3p_water collective_variables"\n'
        'mode = "nvt"\n'
        f"step_limit = {step_limit}\n"
        f"dt = {dt}\n"
        f"cutoff = {cutoff}\n"
        f'thermostat = "{thermostat}"\n'
        f"thermostat_tau = {thermostat_tau}\n"
        f"thermostat_seed = {thermostat_seed}\n"
        f"target_temperature = {target_temperature}\n"
        'default_in_file_prefix = "sys_flexible"\n'
        'cv_in_file = "cv.txt"\n'
        "print_zeroth_frame = 1\n"
        "write_mdout_interval = 1\n"
        f"write_information_interval = {write_information_interval}\n"
    )
    Path(case_dir, "mdin.spg.toml").write_text(mdin)


def read_coordinate_xyzs(coordinate_path):
    lines = Path(coordinate_path).read_text().splitlines()
    if len(lines) < 2:
        raise ValueError(f"Invalid coordinate file: {coordinate_path}")
    xyzs = []
    for line in lines[1:]:
        fields = line.split()
        if len(fields) < 3:
            continue
        xyzs.append((float(fields[0]), float(fields[1]), float(fields[2])))
    if not xyzs:
        raise ValueError(f"No coordinates parsed from {coordinate_path}")
    return xyzs


def write_cv_types_file(case_dir):
    xyzs = read_coordinate_xyzs(case_dir / "sys_flexible_coordinate.txt")
    rmsd_atoms = [0, 4, 10, 12]
    rmsd_ref = " ".join(
        f"{xyzs[i][0]} {xyzs[i][1]} {xyzs[i][2]}" for i in rmsd_atoms
    )

    cv_text = (
        "print\n"
        "{\n"
        "    CV = da ang phi rms tda\n"
        "}\n"
        "da\n"
        "{\n"
        "    CV_type = distance\n"
        "    atom = 0 10\n"
        "}\n"
        "ang\n"
        "{\n"
        "    CV_type = angle\n"
        "    atom = 0 4 10\n"
        "}\n"
        "phi\n"
        "{\n"
        "    CV_type = dihedral\n"
        "    atom = 10 12 14 20\n"
        "}\n"
        "rms\n"
        "{\n"
        "    CV_type = rmsd\n"
        "    atom = 0 4 10 12\n"
        f"    coordinate = {rmsd_ref}\n"
        "}\n"
        "tda\n"
        "{\n"
        "    CV_type = tabulated\n"
        "    CV = da\n"
        "    min = 0.0\n"
        "    max = 15.0\n"
        "    parameter = 0.0 0.5 1.0 1.5\n"
        "    min_padding = 0.0\n"
        "    max_padding = 2.0\n"
        "}\n"
    )
    Path(case_dir, "cv.txt").write_text(cv_text)


def write_meta_cv(
    case_dir,
    *,
    phi_atoms=(10, 12, 14, 20),
    psi_atoms=(0, 4, 10, 12),
    cv_period=6.283185307179586,
    cv_min=-3.141592653589793,
    cv_max=3.141592653589793,
    cv_grid=72,
    cv_sigma=0.20,
    height=0.2,
    welltemp_factor=None,
    potential_update_interval=20,
):
    phi_atom_text = " ".join(str(i) for i in phi_atoms)
    psi_atom_text = " ".join(str(i) for i in psi_atoms)
    cv_text = (
        "print\n"
        "{\n"
        "    CV = phi psi\n"
        "}\n"
        "phi\n"
        "{\n"
        "    CV_type = dihedral\n"
        f"    atom = {phi_atom_text}\n"
        "}\n"
        "psi\n"
        "{\n"
        "    CV_type = dihedral\n"
        f"    atom = {psi_atom_text}\n"
        "}\n"
        "meta\n"
        "{\n"
        "    CV = phi psi\n"
        f"    CV_period = {cv_period} {cv_period}\n"
        f"    CV_minimal = {cv_min} {cv_min}\n"
        f"    CV_maximum = {cv_max} {cv_max}\n"
        f"    CV_grid = {cv_grid} {cv_grid}\n"
        f"    CV_sigma = {cv_sigma} {cv_sigma}\n"
        f"    height = {height}\n"
    )
    if welltemp_factor is not None:
        cv_text += f"    welltemp_factor = {welltemp_factor}\n"
    cv_text += (
        f"    potential_update_interval = {potential_update_interval}\n}}\n"
    )
    Path(case_dir, "cv.txt").write_text(cv_text)


def write_restrain_cv(
    case_dir,
    *,
    phi_atoms=(10, 12, 14, 20),
    weight=0.5,
    reference=0.0,
    period=6.283185307179586,
):
    phi_atom_text = " ".join(str(i) for i in phi_atoms)
    cv_text = (
        "print\n"
        "{\n"
        "    CV = phi\n"
        "}\n"
        "phi\n"
        "{\n"
        "    CV_type = dihedral\n"
        f"    atom = {phi_atom_text}\n"
        "}\n"
        "restrain\n"
        "{\n"
        "    CV = phi\n"
        f"    weight = {weight}\n"
        f"    reference = {reference}\n"
        f"    period = {period}\n"
        "}\n"
    )
    Path(case_dir, "cv.txt").write_text(cv_text)


def write_steer_cv(
    case_dir,
    *,
    phi_atoms=(10, 12, 14, 20),
    weight=0.5,
):
    phi_atom_text = " ".join(str(i) for i in phi_atoms)
    cv_text = (
        "print\n"
        "{\n"
        "    CV = phi\n"
        "}\n"
        "phi\n"
        "{\n"
        "    CV_type = dihedral\n"
        f"    atom = {phi_atom_text}\n"
        "}\n"
        "steer\n"
        "{\n"
        "    CV = phi\n"
        f"    weight = {weight}\n"
        "}\n"
    )
    Path(case_dir, "cv.txt").write_text(cv_text)


def run_sponge_enhanced_sampling(case_dir, timeout=900):
    cmd = ["SPONGE", "-mdin", "mdin.spg.toml"]
    return _run_command(cmd, cwd=case_dir, timeout=timeout)


def run_sponge_cv(case_dir, timeout=900):
    return run_sponge_enhanced_sampling(case_dir, timeout=timeout)


def parse_column_series(mdout_path, column_name):
    lines = Path(mdout_path).read_text().splitlines()
    if len(lines) < 2:
        raise ValueError(f"Invalid mdout file: {mdout_path}")

    headers = lines[0].split()
    if column_name not in headers:
        raise ValueError(f"No {column_name} column in {mdout_path}")
    col_idx = headers.index(column_name)

    values = []
    for line in lines[1:]:
        fields = line.split()
        if len(fields) <= col_idx:
            continue
        try:
            int(fields[0])
            val = float(fields[col_idx])
        except ValueError:
            continue
        values.append(val)

    if not values:
        raise ValueError(f"No {column_name} samples parsed from {mdout_path}")
    return values


def parse_meta_potential_raw(meta_potential_path):
    raw_idx = None
    values = []
    for line in Path(meta_potential_path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            header_tokens = line[1:].split()
            if "potential_raw" in header_tokens:
                raw_idx = header_tokens.index("potential_raw")
            continue
        fields = line.split()
        if raw_idx is None or len(fields) <= raw_idx:
            continue
        try:
            values.append(float(fields[raw_idx]))
        except ValueError:
            continue
    if not values:
        raise ValueError(
            f"No numeric potential samples parsed from {meta_potential_path}"
        )
    return values


def summarize_series(series, burn_in=0):
    if burn_in >= len(series) - 1:
        raise ValueError(
            f"burn_in={burn_in} is too large for sample count {len(series)}"
        )
    sample = series[burn_in:]
    return {
        "sample_count": len(sample),
        "mean": statistics.fmean(sample),
        "std": statistics.stdev(sample),
        "min": min(sample),
        "max": max(sample),
    }


def parse_numeric_values(path):
    values = []
    for token in Path(path).read_text().split():
        try:
            values.append(float(token))
        except ValueError:
            continue
    if not values:
        raise ValueError(f"No numeric values parsed from {path}")
    return values


def write_sits_nk_in_file(case_dir, values, file_name="sits_nk_in.txt"):
    value_text = " ".join(str(v) for v in values)
    path = Path(case_dir, file_name)
    path.write_text(value_text + "\n")
    return path

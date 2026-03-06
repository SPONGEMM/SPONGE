from pathlib import Path

from benchmarks.utils import Outputer, Runner


def run_sponge(case_dir, timeout=900, mpi_np=None):
    return Runner.run_sponge(
        case_dir,
        timeout=timeout,
        mpi_np=mpi_np,
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

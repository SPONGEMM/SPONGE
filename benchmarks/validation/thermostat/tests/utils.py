import re
import statistics
from pathlib import Path

from benchmarks.utils import (
    Outputer,
    Runner,
    Extractor,
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


def run_sponge_thermostat(case_dir, timeout=600, mpi_np=None):
    return Runner.run_sponge(
        case_dir,
        timeout=timeout,
        mpi_np=mpi_np,
    )


def parse_temperature_series(mdout_path):
    rows = Extractor.parse_mdout_rows(
        mdout_path, ["temperature"], int_columns=()
    )
    return [row["temperature"] for row in rows]


def read_atom_count_from_mass(mass_path):
    return Extractor.read_first_field_int(mass_path)


def read_constrain_pair_count_from_runlog(runlog_text):
    match = re.search(r"constrain pair number is\s+(\d+)", runlog_text)
    if not match:
        return 0
    return int(match.group(1))


def read_ug_count_from_runlog(runlog_text):
    match = re.search(r"md_info->ug\.ug_numbers:\s+(\d+)", runlog_text)
    if not match:
        match = re.search(
            r"max_atom_numbers=\d+,\s*max_res_numbers=(\d+)", runlog_text
        )
    if not match:
        raise ValueError("Cannot parse update_group count from run output")
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

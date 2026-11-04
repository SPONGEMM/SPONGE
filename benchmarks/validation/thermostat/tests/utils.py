import math
import random
import statistics
from pathlib import Path

from benchmarks.utils import (
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
    velocity_in_file=None,
    initial_velocity_seed=None,
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
    if velocity_in_file is not None:
        mdin += f'velocity_in_file = "{velocity_in_file}"\n'
    if initial_velocity_seed is not None:
        mdin += (
            '\n[initial_velocity]\nmode = "maxwell"\n'
            f"seed = {initial_velocity_seed}\n"
        )
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


def read_mass_values(mass_path):
    tokens = Path(mass_path).read_text(encoding="utf-8").split()
    atom_numbers = int(tokens[0])
    masses = [float(token) for token in tokens[1 : 1 + atom_numbers]]
    if len(masses) != atom_numbers:
        raise ValueError(
            f"Mass value count mismatch in {mass_path}: "
            f"expected {atom_numbers}, got {len(masses)}"
        )
    return masses


def write_velocity_file_for_temperature(
    output_path,
    masses,
    *,
    temperature,
    seed,
    degrees_of_freedom=None,
    k_b=0.00198716,
):
    rng = random.Random(seed)
    velocities = []
    for mass in masses:
        sigma = math.sqrt(k_b * temperature / mass)
        velocities.append(
            [
                rng.gauss(0.0, sigma),
                rng.gauss(0.0, sigma),
                rng.gauss(0.0, sigma),
            ]
        )

    total_mass = sum(masses)
    for axis in range(3):
        center_velocity = (
            sum(
                mass * velocity[axis]
                for mass, velocity in zip(masses, velocities)
            )
            / total_mass
        )
        for velocity in velocities:
            velocity[axis] -= center_velocity

    kinetic = 0.5 * sum(
        mass * sum(component * component for component in velocity)
        for mass, velocity in zip(masses, velocities)
    )
    dof = degrees_of_freedom
    if dof is None:
        dof = 3 * len(masses) - 3
    scale = math.sqrt(temperature * dof * k_b / (2.0 * kinetic))
    for velocity in velocities:
        for axis in range(3):
            velocity[axis] *= scale

    lines = [str(len(masses))]
    lines.extend(
        f"{velocity[0]:.7f} {velocity[1]:.7f} {velocity[2]:.7f}"
        for velocity in velocities
    )
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_temperature_block_average(
    temperatures,
    *,
    target_temperature=300.0,
    burn_in=20,
    block_size=100,
):
    if block_size < 2:
        raise ValueError(f"Invalid block_size: {block_size}")
    if burn_in >= len(temperatures) - 1:
        raise ValueError(
            f"burn_in={burn_in} is too large for sample count {len(temperatures)}"
        )

    sample = temperatures[burn_in:]
    block_count = len(sample) // block_size
    if block_count < 2:
        raise ValueError(
            f"Not enough samples for block averaging: "
            f"sample_count={len(sample)}, block_size={block_size}"
        )

    trimmed = sample[: block_count * block_size]
    block_means = [
        statistics.fmean(trimmed[i * block_size : (i + 1) * block_size])
        for i in range(block_count)
    ]
    mean_temp = statistics.fmean(block_means)
    block_std = statistics.stdev(block_means)
    sem = block_std / (block_count**0.5)
    z_score = (
        abs(mean_temp - target_temperature) / sem
        if sem > 0.0
        else (0.0 if mean_temp == target_temperature else float("inf"))
    )

    return {
        "total_samples": len(temperatures),
        "sample_count": len(trimmed),
        "block_size": block_size,
        "block_count": block_count,
        "mean_temp": mean_temp,
        "block_std": block_std,
        "sem": sem,
        "z_score": z_score,
    }

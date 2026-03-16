import math
from pathlib import Path

from benchmarks.utils import (
    Outputer,
    Runner,
)

AMU_PER_A3_TO_G_PER_CM3 = 1.66053906660
BAR_A3_TO_KJ_PER_MOL = 6.02214076e-5
BOLTZMANN_KJ_PER_MOL_K = 0.00831446261815324


def write_barostat_mdin(
    case_dir,
    *,
    step_limit=20000,
    dt=0.002,
    cutoff=8.0,
    thermostat="middle_langevin",
    thermostat_tau=0.1,
    thermostat_seed=2026,
    target_temperature=300.0,
    target_pressure=1.0,
    barostat="andersen_barostat",
    barostat_mode=None,
    barostat_isotropy="isotropic",
    barostat_tau=1.0,
    barostat_update_interval=10,
    write_information_interval=1000,
    write_mdout_interval=1000,
    default_in_file_prefix="tip3p",
    constrain_mode="SETTLE",
    monte_carlo_initial_ratio=0.001,
    monte_carlo_update_interval=50,
    monte_carlo_check_interval=10,
):
    mdin = (
        'md_name = "benchmark tip3p_water barostat reweighting"\n'
        'mode = "npt"\n'
        f"step_limit = {step_limit}\n"
        f"dt = {dt}\n"
        f"cutoff = {cutoff}\n"
        f'thermostat = "{thermostat}"\n'
        f"thermostat_tau = {thermostat_tau}\n"
        f"thermostat_seed = {thermostat_seed}\n"
        f"target_temperature = {target_temperature}\n"
        f"target_pressure = {target_pressure}\n"
        f'barostat = "{barostat}"\n'
        f'default_in_file_prefix = "{default_in_file_prefix}"\n'
        f'constrain_mode = "{constrain_mode}"\n'
        "print_zeroth_frame = 1\n"
        f"write_mdout_interval = {write_mdout_interval}\n"
        f"write_information_interval = {write_information_interval}\n"
    )
    if barostat_mode is not None:
        mdin += f'barostat_mode = "{barostat_mode}"\n'
    if barostat_isotropy is not None:
        mdin += f'barostat_isotropy = "{barostat_isotropy}"\n'
    if barostat == "monte_carlo_barostat":
        mdin += (
            f"monte_carlo_barostat_initial_ratio = {monte_carlo_initial_ratio}\n"
            f"monte_carlo_barostat_update_interval = {monte_carlo_update_interval}\n"
            f"monte_carlo_barostat_check_interval = {monte_carlo_check_interval}\n"
        )
    else:
        mdin += (
            f"barostat_tau = {barostat_tau}\n"
            f"barostat_update_interval = {barostat_update_interval}\n"
        )
    Path(case_dir, "mdin.spg.toml").write_text(mdin)


def run_sponge_barostat(case_dir, timeout=1200, mpi_np=None):
    return Runner.run_sponge(
        case_dir,
        mpi_np=mpi_np,
        timeout=timeout,
    )


def rescale_coordinate_box(
    coordinate_path,
    *,
    new_lx,
    new_ly,
    new_lz,
    new_alpha=None,
    new_beta=None,
    new_gamma=None,
    scale_coordinates=False,
):
    lines = Path(coordinate_path).read_text().splitlines()
    if len(lines) < 3:
        raise ValueError(f"Invalid coordinate file: {coordinate_path}")

    header = lines[0]
    coord_lines = lines[1:-1]
    old_box_fields = lines[-1].split()
    if len(old_box_fields) < 6:
        raise ValueError(
            f"Missing box info in coordinate file: {coordinate_path}"
        )

    try:
        atom_count = int(header.split()[0])
    except (ValueError, IndexError) as exc:
        raise ValueError(
            f"Invalid coordinate header in {coordinate_path}: {header}"
        ) from exc
    if atom_count != len(coord_lines):
        raise ValueError(
            f"Coordinate count mismatch in {coordinate_path}: "
            f"header={atom_count}, body={len(coord_lines)}"
        )

    old_lx, old_ly, old_lz = map(float, old_box_fields[:3])
    if old_lx <= 0.0 or old_ly <= 0.0 or old_lz <= 0.0:
        raise ValueError(f"Non-positive old box in {coordinate_path}")
    if new_lx <= 0.0 or new_ly <= 0.0 or new_lz <= 0.0:
        raise ValueError(
            f"New box lengths must be positive: {new_lx}, {new_ly}, {new_lz}"
        )

    old_alpha, old_beta, old_gamma = map(float, old_box_fields[3:6])
    alpha = old_alpha if new_alpha is None else float(new_alpha)
    beta = old_beta if new_beta is None else float(new_beta)
    gamma = old_gamma if new_gamma is None else float(new_gamma)

    if scale_coordinates:
        scale_x = new_lx / old_lx
        scale_y = new_ly / old_ly
        scale_z = new_lz / old_lz
        updated_coord_lines = []
        for i, line in enumerate(coord_lines, start=1):
            fields = line.split()
            if len(fields) < 3:
                raise ValueError(
                    f"Invalid coordinate line {i} in {coordinate_path}"
                )
            updated_coord_lines.append(
                f"{float(fields[0]) * scale_x:12.7f} "
                f"{float(fields[1]) * scale_y:12.7f} "
                f"{float(fields[2]) * scale_z:12.7f}"
            )
    else:
        updated_coord_lines = coord_lines

    box_line = (
        f"{new_lx:12.7f} {new_ly:12.7f} {new_lz:12.7f} "
        f"{alpha:12.7f} {beta:12.7f} {gamma:12.7f}"
    )
    Path(coordinate_path).write_text(
        "\n".join([header, *updated_coord_lines, box_line]) + "\n"
    )


def read_total_mass_amu(mass_path):
    lines = Path(mass_path).read_text().splitlines()
    if len(lines) < 2:
        raise ValueError(f"Invalid mass file: {mass_path}")
    return sum(float(x) for x in lines[1:] if x.strip())


def parse_mdbox_lengths(mdbox_path):
    lines = Path(mdbox_path).read_text().splitlines()
    if not lines:
        raise ValueError(f"Empty mdbox file: {mdbox_path}")

    lengths = []
    for line in lines:
        fields = line.split()
        if len(fields) < 3:
            continue
        lengths.append((float(fields[0]), float(fields[1]), float(fields[2])))

    if not lengths:
        raise ValueError(f"No box samples parsed from {mdbox_path}")
    return lengths


def triclinic_volume_a3(
    lx, ly, lz, alpha_deg=90.0, beta_deg=90.0, gamma_deg=90.0
):
    if lx <= 0.0 or ly <= 0.0 or lz <= 0.0:
        raise ValueError(f"Non-positive box lengths: lx={lx}, ly={ly}, lz={lz}")
    alpha = math.radians(alpha_deg)
    beta = math.radians(beta_deg)
    gamma = math.radians(gamma_deg)
    cos_a = math.cos(alpha)
    cos_b = math.cos(beta)
    cos_g = math.cos(gamma)
    metric_det = (
        1.0
        + 2.0 * cos_a * cos_b * cos_g
        - cos_a * cos_a
        - cos_b * cos_b
        - cos_g * cos_g
    )
    # Clamp tiny negative round-off for nearly orthogonal boxes.
    if metric_det < 0.0 and metric_det > -1e-12:
        metric_det = 0.0
    if metric_det <= 0.0:
        raise ValueError(
            "Invalid triclinic box angles: "
            f"alpha={alpha_deg}, beta={beta_deg}, gamma={gamma_deg}"
        )
    return lx * ly * lz * math.sqrt(metric_det)


def parse_volume_series(mdbox_path):
    lines = Path(mdbox_path).read_text().splitlines()
    if not lines:
        raise ValueError(f"Empty mdbox file: {mdbox_path}")

    volumes = []
    for line in lines:
        fields = line.split()
        if len(fields) < 3:
            continue
        lx, ly, lz = map(float, fields[:3])
        if len(fields) >= 6:
            alpha, beta, gamma = map(float, fields[3:6])
            volume = triclinic_volume_a3(
                lx, ly, lz, alpha_deg=alpha, beta_deg=beta, gamma_deg=gamma
            )
        else:
            volume = lx * ly * lz
        if volume <= 0.0:
            raise ValueError(f"Non-positive volume detected in {mdbox_path}")
        volumes.append(volume)

    if not volumes:
        raise ValueError(f"No box samples parsed from {mdbox_path}")
    return volumes


def parse_density_series_from_mdbox(mdbox_path, total_mass_amu):
    volumes = parse_volume_series(mdbox_path)
    densities = [
        total_mass_amu * AMU_PER_A3_TO_G_PER_CM3 / volume for volume in volumes
    ]
    return densities, volumes


def boltzmann_reweight_mean(
    observable_series,
    volume_series,
    *,
    from_pressure_bar,
    to_pressure_bar,
    temperature_k,
):
    if len(observable_series) != len(volume_series):
        raise ValueError(
            "Observable and volume series length mismatch: "
            f"{len(observable_series)} != {len(volume_series)}"
        )
    if not observable_series:
        raise ValueError("Cannot reweight an empty series")
    if temperature_k <= 0:
        raise ValueError(f"temperature_k must be positive, got {temperature_k}")

    beta = 1.0 / (BOLTZMANN_KJ_PER_MOL_K * temperature_k)
    delta_p = to_pressure_bar - from_pressure_bar
    log_weight_scale = -beta * delta_p * BAR_A3_TO_KJ_PER_MOL
    log_weights = [log_weight_scale * volume for volume in volume_series]
    log_weight_ref = max(log_weights)

    weights = [math.exp(lw - log_weight_ref) for lw in log_weights]
    weight_sum = sum(weights)
    if weight_sum <= 0.0:
        raise ValueError("Non-positive weight sum in reweighting")

    mean = (
        sum(w * obs for w, obs in zip(weights, observable_series)) / weight_sum
    )
    weight_sq_sum = sum(w * w for w in weights)
    ess = (weight_sum * weight_sum) / weight_sq_sum
    return {
        "mean": mean,
        "ess": ess,
        "ess_ratio": ess / len(weights),
        "delta_pressure_bar": delta_p,
    }

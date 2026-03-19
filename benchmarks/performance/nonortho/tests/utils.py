import math
import random
from pathlib import Path

import numpy as np

from benchmarks.utils import Outputer, Runner


def run_sponge_barostat(case_dir, timeout=2400, mpi_np=None):
    return Runner.run_sponge(
        case_dir,
        mpi_np=mpi_np,
        timeout=timeout,
    )


def write_nonortho_long_run_mdin(
    case_dir,
    *,
    mode,
    step_limit,
    trajectory_interval,
    velocity_in_file,
    dt=0.002,
    cutoff=8.0,
    target_temperature=300.0,
    target_pressure=1.0,
):
    mode_upper = str(mode).upper()
    mdin = (
        f'md_name = "nonortho {mode_upper} long-run rdf"\n'
        f'mode = "{mode_upper.lower()}"\n'
        f"step_limit = {int(step_limit)}\n"
        f"dt = {float(dt)}\n"
        f"cutoff = {float(cutoff)}\n"
        'default_in_file_prefix = "WAT"\n'
        'constrain_mode = "SETTLE"\n'
        'crd = "mdcrd.dat"\n'
        'box = "mdbox.txt"\n'
        'mdout = "mdout.txt"\n'
        'mdinfo = "mdinfo.txt"\n'
        "print_zeroth_frame = 1\n"
        "write_information_interval = 1000\n"
        "write_mdout_interval = 1000\n"
        f"write_trajectory_interval = {int(trajectory_interval)}\n"
        f'velocity_in_file = "{velocity_in_file}"\n'
    )

    if mode_upper in {"NVT", "NPT"}:
        mdin += (
            'thermostat = "middle_langevin"\n'
            "thermostat_tau = 0.1\n"
            "thermostat_seed = 2026\n"
            f"target_temperature = {float(target_temperature)}\n"
        )
    if mode_upper == "NPT":
        mdin += (
            'barostat = "andersen_barostat"\n'
            "barostat_tau = 0.1\n"
            "barostat_update_interval = 10\n"
            f"target_pressure = {float(target_pressure)}\n"
        )

    Path(case_dir, "mdin.spg.toml").write_text(mdin, encoding="utf-8")


def write_velocity_file_with_zero_mass_support(
    output_path,
    masses,
    *,
    temperature,
    seed,
    k_b=0.00198716,
):
    rng = random.Random(seed)
    velocities = []
    mobile_indices = []
    for idx, mass in enumerate(masses):
        if mass > 0.0:
            sigma = math.sqrt(k_b * temperature / mass)
            velocities.append(
                [
                    rng.gauss(0.0, sigma),
                    rng.gauss(0.0, sigma),
                    rng.gauss(0.0, sigma),
                ]
            )
            mobile_indices.append(idx)
        else:
            velocities.append([0.0, 0.0, 0.0])

    total_mass = sum(masses[idx] for idx in mobile_indices)
    if total_mass <= 0.0:
        raise ValueError(
            "No positive-mass atoms found for velocity initialization"
        )

    for axis in range(3):
        center_velocity = (
            sum(masses[idx] * velocities[idx][axis] for idx in mobile_indices)
            / total_mass
        )
        for idx in mobile_indices:
            velocities[idx][axis] -= center_velocity

    kinetic = 0.5 * sum(
        masses[idx]
        * sum(component * component for component in velocities[idx])
        for idx in mobile_indices
    )
    dof = 3 * len(mobile_indices) - 3
    scale = math.sqrt(temperature * dof * k_b / (2.0 * kinetic))
    for idx in mobile_indices:
        for axis in range(3):
            velocities[idx][axis] *= scale

    lines = [str(len(masses))]
    lines.extend(
        f"{velocity[0]:.7f} {velocity[1]:.7f} {velocity[2]:.7f}"
        for velocity in velocities
    )
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_atom_names(atom_name_path):
    lines = Path(atom_name_path).read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"Empty atom name file: {atom_name_path}")
    atom_numbers = int(lines[0].split()[0])
    atom_names = [line.strip() for line in lines[1:] if line.strip()]
    if len(atom_names) != atom_numbers:
        raise ValueError(
            f"Atom name count mismatch in {atom_name_path}: "
            f"expected {atom_numbers}, got {len(atom_names)}"
        )
    return atom_names


def read_coordinate_atom_count(coordinate_path):
    header = Path(coordinate_path).read_text(encoding="utf-8").splitlines()[0]
    return int(header.split()[0])


def read_box_trajectory(mdbox_path):
    rows = []
    for line in Path(mdbox_path).read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 6:
            continue
        rows.append([float(x) for x in fields[:6]])
    if not rows:
        raise ValueError(f"No box frames parsed from {mdbox_path}")
    return np.asarray(rows, dtype=np.float64)


def read_coordinate_trajectory(mdcrd_path, atom_count):
    raw = np.fromfile(mdcrd_path, dtype=np.float32)
    frame_width = int(atom_count) * 3
    if frame_width <= 0 or raw.size % frame_width != 0:
        raise ValueError(
            f"Invalid trajectory shape in {mdcrd_path}: "
            f"size={raw.size}, atom_count={atom_count}"
        )
    return raw.reshape((-1, int(atom_count), 3)).astype(np.float64)


def triclinic_cell_matrix(lx, ly, lz, alpha_deg, beta_deg, gamma_deg):
    alpha = math.radians(alpha_deg)
    beta = math.radians(beta_deg)
    gamma = math.radians(gamma_deg)
    cos_a = math.cos(alpha)
    cos_b = math.cos(beta)
    cos_g = math.cos(gamma)
    sin_g = math.sin(gamma)
    if abs(sin_g) < 1e-12:
        raise ValueError("Invalid triclinic box: sin(gamma) is zero")
    c_y = lz * (cos_a - cos_b * cos_g) / sin_g
    c_z_sq = lz * lz - (lz * cos_b) ** 2 - c_y**2
    if c_z_sq < 0.0 and c_z_sq > -1e-10:
        c_z_sq = 0.0
    if c_z_sq < 0.0:
        raise ValueError(
            "Invalid triclinic box encountered while building cell matrix"
        )
    return np.asarray(
        [
            [lx, 0.0, 0.0],
            [ly * cos_g, ly * sin_g, 0.0],
            [lz * cos_b, c_y, math.sqrt(c_z_sq)],
        ],
        dtype=np.float64,
    )


def compute_oo_rdf(
    trajectory,
    box_trajectory,
    oxygen_indices,
    *,
    r_max=None,
    bin_width=0.05,
):
    if trajectory.shape[0] == 0:
        raise ValueError("Empty coordinate trajectory for RDF")
    if len(oxygen_indices) < 2:
        raise ValueError("At least two oxygen atoms are required for O-O RDF")

    frame_count = min(trajectory.shape[0], box_trajectory.shape[0])
    coords = trajectory[:frame_count, oxygen_indices, :]
    boxes = box_trajectory[:frame_count]
    oxygen_count = coords.shape[1]

    if r_max is None:
        r_max = 0.45 * float(np.min(boxes[:, :3]))
    if r_max <= 0.0:
        raise ValueError(f"Invalid r_max for RDF: {r_max}")
    bin_edges = np.arange(0.0, r_max + bin_width, bin_width, dtype=np.float64)
    if bin_edges.size < 2:
        raise ValueError("Insufficient RDF bins")
    hist = np.zeros(bin_edges.size - 1, dtype=np.float64)

    volumes = []
    for frame_idx in range(frame_count):
        cell = triclinic_cell_matrix(*boxes[frame_idx])
        rcell = np.linalg.inv(cell)
        volumes.append(abs(np.linalg.det(cell)))
        points = coords[frame_idx]
        for atom_i in range(oxygen_count - 1):
            delta = points[atom_i + 1 :] - points[atom_i]
            fractional = delta @ rcell
            fractional -= np.round(fractional)
            wrapped = fractional @ cell
            distances = np.linalg.norm(wrapped, axis=1)
            valid = distances[(distances > 1e-8) & (distances < r_max)]
            if valid.size:
                hist += 2.0 * np.histogram(valid, bins=bin_edges)[0]

    radii = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    shell_volumes = (
        (4.0 / 3.0)
        * math.pi
        * (np.power(bin_edges[1:], 3) - np.power(bin_edges[:-1], 3))
    )
    number_density = oxygen_count / float(np.mean(volumes))
    normalization = frame_count * oxygen_count * number_density * shell_volumes
    rdf = np.divide(
        hist,
        normalization,
        out=np.zeros_like(hist),
        where=normalization > 0.0,
    )
    return radii, rdf


def save_rdf_plot(radii, rdf, output_path, *, title):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.6), dpi=180)
    ax.plot(radii, rdf, color="tab:blue", lw=1.6)
    ax.set_xlabel("r (Angstrom)")
    ax.set_ylabel("g_OO(r)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    output_path = Path(output_path)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def summarize_rdf(radii, rdf):
    finite_mask = np.isfinite(rdf)
    if not np.any(finite_mask):
        raise ValueError("RDF contains no finite samples")
    finite_r = radii[finite_mask]
    finite_g = rdf[finite_mask]
    peak_mask = (finite_r >= 2.0) & (finite_r <= 4.0)
    if np.any(peak_mask):
        peak_index = int(np.argmax(finite_g[peak_mask]))
        peak_r = float(finite_r[peak_mask][peak_index])
        peak_g = float(finite_g[peak_mask][peak_index])
    else:
        peak_index = int(np.argmax(finite_g))
        peak_r = float(finite_r[peak_index])
        peak_g = float(finite_g[peak_index])
    return {
        "rdf_peak_r": peak_r,
        "rdf_peak_g": peak_g,
        "rdf_max": float(np.max(finite_g)),
        "rdf_tail_mean": float(np.mean(finite_g[-10:])),
    }

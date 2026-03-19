import math
from pathlib import Path

import numpy as np

from benchmarks.utils import Runner


def run_sponge_amber(case_dir, timeout=2400, mpi_np=None):
    return Runner.run_sponge(
        case_dir,
        mpi_np=mpi_np,
        timeout=timeout,
    )


def write_amber_long_run_mdin(
    case_dir,
    *,
    mode,
    step_limit,
    trajectory_interval,
    amber_rst7,
    dt=0.002,
    cutoff=8.0,
    target_temperature=300.0,
    target_pressure=1.0,
):
    mode_upper = str(mode).upper()
    mdin = (
        f'md_name = "amber {mode_upper} long-run rdf"\n'
        f'mode = "{mode_upper.lower()}"\n'
        'amber_parm7 = "model-protein-RNA-complex.prmtop"\n'
        f'amber_rst7 = "{amber_rst7}"\n'
        "amber_irest = 1\n"
        f"step_limit = {int(step_limit)}\n"
        f"dt = {float(dt)}\n"
        f"cutoff = {float(cutoff)}\n"
        'constrain_mode = "SHAKE"\n'
        'crd = "mdcrd.dat"\n'
        'box = "mdbox.txt"\n'
        'mdout = "mdout.txt"\n'
        'mdinfo = "mdinfo.txt"\n'
        "print_zeroth_frame = 1\n"
        "write_information_interval = 1000\n"
        "write_mdout_interval = 1000\n"
        f"write_trajectory_interval = {int(trajectory_interval)}\n"
    )
    if mode_upper in {"NVT", "NPT"}:
        mdin += (
            'thermostat = "middle_langevin"\n'
            "langevin_gamma = 10.0\n"
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
    max_oxygen_samples=2048,
    sample_seed=2026,
    chunk_size=256,
):
    if trajectory.shape[0] == 0:
        raise ValueError("Empty coordinate trajectory for RDF")
    if len(oxygen_indices) < 2:
        raise ValueError("At least two oxygen atoms are required for O-O RDF")

    frame_count = min(trajectory.shape[0], box_trajectory.shape[0])
    oxygen_indices = np.asarray(oxygen_indices, dtype=int)
    if oxygen_indices.size > max_oxygen_samples:
        rng = np.random.default_rng(sample_seed)
        oxygen_indices = np.sort(
            rng.choice(
                oxygen_indices,
                size=int(max_oxygen_samples),
                replace=False,
            )
        )

    coords = trajectory[:frame_count, oxygen_indices, :]
    boxes = box_trajectory[:frame_count]
    oxygen_count = coords.shape[1]

    if r_max is None:
        r_max = 0.45 * float(np.min(boxes[:, :3]))
    bin_edges = np.arange(0.0, r_max + bin_width, bin_width, dtype=np.float64)
    hist = np.zeros(bin_edges.size - 1, dtype=np.float64)
    volumes = []

    for frame_idx in range(frame_count):
        cell = triclinic_cell_matrix(*boxes[frame_idx])
        rcell = np.linalg.inv(cell)
        volumes.append(abs(np.linalg.det(cell)))
        points = coords[frame_idx]
        for start in range(0, oxygen_count - 1, chunk_size):
            end = min(start + chunk_size, oxygen_count - 1)
            block = points[start:end]
            rest = points[start + 1 :]
            if rest.size == 0:
                continue
            delta = block[:, None, :] - rest[None, :, :]
            fractional = delta @ rcell
            fractional -= np.round(fractional)
            wrapped = fractional @ cell
            distances = np.linalg.norm(wrapped, axis=2)

            row_indices = np.arange(start, end)[:, None]
            col_indices = np.arange(start + 1, oxygen_count)[None, :]
            valid_mask = (
                (col_indices > row_indices)
                & (distances < r_max)
                & (distances > 1e-8)
            )
            valid = distances[valid_mask]
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
    return radii, rdf, oxygen_indices


def summarize_rdf(radii, rdf):
    finite_mask = np.isfinite(rdf)
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

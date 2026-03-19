import json
from pathlib import Path

import numpy as np

from benchmarks.validation.utils import parse_mdout_rows


def write_dna_cou_cv_file(
    case_dir,
    cv_file="cv.txt",
    *,
    height=0.2,
    sigma=0.6,
    dip=1.0,
    edge_file=None,
):
    lines = [
        "ML",
        "{",
        "    vatom_type = center_of_mass",
        "    atom_in_file = ligand.list",
        "}",
        "cx",
        "{",
        "    CV_type = position_x",
        "    atom = ML",
        "}",
        "cy",
        "{",
        "    CV_type = position_y",
        "    atom = ML",
        "}",
        "cz",
        "{",
        "    CV_type = position_z",
        "    atom = ML",
        "}",
        "meta",
        "{",
        "    Ndim = 3",
        "    CV = cx cy cz",
        "    CV_minimal = 20.0 20.0 25.0",
        "    CV_maximum = 45.0 45.0 70.0",
        "    CV_period = 0 0 0",
        "    CV_grid = 50 50 90",
        f"    CV_sigma = {float(sigma):.6f} {float(sigma):.6f} {float(sigma):.6f}",
        f"    height = {float(height):.6f}",
        "    potential_update_interval = 500",
        "    welltemp_factor = 20",
        "    sink = 1",
        "    scatter_in_file = 14d5.txt",
        "    kde = 1",
        f"    dip = {float(dip):.6f}",
        "    convmeta = 1",
    ]
    if edge_file is not None:
        lines.append(f"    edge_in_file = {Path(edge_file).as_posix()}")
    lines.extend(
        [
            "}",
            "print",
            "{",
            "    CV = cx cy cz",
            "}",
        ]
    )
    Path(case_dir, cv_file).write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_sinkmeta_mdin(
    case_dir,
    *,
    step_limit,
    dt=0.002,
    target_temperature=300.0,
    write_information_interval=500,
    write_trajectory_interval=None,
    write_restart_file_interval=10000,
    default_in_file_prefix="2m2c",
    coordinate_in_file="Pmin_coordinate.txt",
    velocity_in_file=None,
    cv_file="cv.txt",
    restrain_atom_id="restrain_dnaH.txt",
    restrain_single_weight=10.0,
):
    if write_trajectory_interval is None:
        write_trajectory_interval = int(step_limit)
    mdin = (
        'md_name = "performance DNA_COU sinkmeta"\n'
        'mode = "nvt"\n'
        f"dt = {dt}\n"
        f"step_limit = {step_limit}\n"
        f"write_information_interval = {write_information_interval}\n"
        f"write_mdout_interval = {write_information_interval}\n"
        f"write_trajectory_interval = {write_trajectory_interval}\n"
        f"write_restart_file_interval = {write_restart_file_interval}\n"
        'thermostat = "middle_langevin"\n'
        f"target_temperature = {target_temperature}\n"
        f'default_in_file_prefix = "{default_in_file_prefix}"\n'
        f'coordinate_in_file = "{coordinate_in_file}"\n'
        f'cv_in_file = "{cv_file}"\n'
        'mdout = "mdout.txt"\n'
        'mdinfo = "mdinfo.txt"\n'
        'box = "mdbox.txt"\n'
        'crd = "mdcrd.dat"\n'
        'rst = "restart"\n'
        "print_zeroth_frame = 1\n"
        'constrain_mode = "SHAKE"\n'
        f'restrain_atom_id = "{restrain_atom_id}"\n'
        'restrain_refcoord_scaling = "all"\n'
        f"restrain_single_weight = {restrain_single_weight}\n"
        "dont_check_input = 1\n"
    )
    if velocity_in_file is not None:
        mdin += f'velocity_in_file = "{velocity_in_file}"\n'
    Path(case_dir, "mdin.spg.toml").write_text(mdin, encoding="utf-8")


def parse_cv_rows(mdout_path):
    return parse_mdout_rows(
        mdout_path,
        ["step", "time", "cx", "cy", "cz", "meta", "rbias"],
        int_columns=("step",),
    )


def parse_meta_potential(path):
    rows = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 5:
            continue
        try:
            rows.append(
                {
                    "cx": float(fields[0]),
                    "cy": float(fields[1]),
                    "cz": float(fields[2]),
                    "potential_raw": float(fields[3]),
                    "potential_shifted": float(fields[4]),
                }
            )
        except ValueError:
            continue
    if not rows:
        raise ValueError(f"No scatter potential rows parsed from {path}")
    return rows


def parse_scatter_axis_file(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if len(lines) < 5:
        raise ValueError(f"Invalid scatter axis file: {path}")
    scatter_size = int(lines[4].split()[-1])
    rows = []
    for raw_line in lines[5 : 5 + scatter_size]:
        fields = raw_line.split()
        if len(fields) < 3:
            continue
        rows.append(
            {
                "cx": float(fields[0]),
                "cy": float(fields[1]),
                "cz": float(fields[2]),
            }
        )
    if not rows:
        raise ValueError(f"No axis points parsed from {path}")
    return rows


def parse_box_lengths(path):
    fields = Path(path).read_text(encoding="utf-8").split()
    if len(fields) < 3:
        raise ValueError(f"Invalid box file: {path}")
    return np.asarray(
        [float(fields[0]), float(fields[1]), float(fields[2])], dtype=float
    )


def reimage_xyz_to_axis(rows, axis_rows, box_lengths):
    xyz = np.column_stack(
        (
            np.asarray([row["cx"] for row in rows], dtype=float),
            np.asarray([row["cy"] for row in rows], dtype=float),
            np.asarray([row["cz"] for row in rows], dtype=float),
        )
    )
    axis_xyz = np.column_stack(
        (
            np.asarray([row["cx"] for row in axis_rows], dtype=float),
            np.asarray([row["cy"] for row in axis_rows], dtype=float),
            np.asarray([row["cz"] for row in axis_rows], dtype=float),
        )
    )
    deltas = xyz[:, None, :] - axis_xyz[None, :, :]
    deltas -= np.round(deltas / box_lengths) * box_lengths
    distances = np.linalg.norm(deltas, axis=2)
    nearest_index = np.argmin(distances, axis=1)
    wrapped = (
        axis_xyz[nearest_index] + deltas[np.arange(len(rows)), nearest_index]
    )
    nearest_distance = distances[np.arange(len(rows)), nearest_index]
    return wrapped, nearest_distance


def project_xyz_to_path_sr(rows, axis_rows, box_lengths):
    xyz = np.column_stack(
        (
            np.asarray([row["cx"] for row in rows], dtype=float),
            np.asarray([row["cy"] for row in rows], dtype=float),
            np.asarray([row["cz"] for row in rows], dtype=float),
        )
    )
    axis_xyz = np.column_stack(
        (
            np.asarray([row["cx"] for row in axis_rows], dtype=float),
            np.asarray([row["cy"] for row in axis_rows], dtype=float),
            np.asarray([row["cz"] for row in axis_rows], dtype=float),
        )
    )
    if axis_xyz.shape[0] < 2:
        raise ValueError(
            "At least two axis points are required for s-r projection"
        )

    segment_vectors = axis_xyz[1:] - axis_xyz[:-1]
    segment_vectors -= np.round(segment_vectors / box_lengths) * box_lengths
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    cumulative_s = np.concatenate(([0.0], np.cumsum(segment_lengths)))

    wrapped_xyz = np.zeros_like(xyz)
    s_values = np.zeros(xyz.shape[0], dtype=float)
    r_values = np.zeros(xyz.shape[0], dtype=float)

    for i, point in enumerate(xyz):
        best_r2 = None
        best_wrapped = None
        best_s = None
        for seg_idx, vec in enumerate(segment_vectors):
            seg_len2 = float(np.dot(vec, vec))
            if seg_len2 <= 1e-12:
                continue
            delta = point - axis_xyz[seg_idx]
            delta -= np.round(delta / box_lengths) * box_lengths
            t = float(np.dot(delta, vec) / seg_len2)
            t = min(1.0, max(0.0, t))
            proj = axis_xyz[seg_idx] + t * vec
            radial = delta - t * vec
            r2 = float(np.dot(radial, radial))
            if best_r2 is None or r2 < best_r2:
                best_r2 = r2
                best_wrapped = proj + radial
                best_s = float(
                    cumulative_s[seg_idx] + t * segment_lengths[seg_idx]
                )
        wrapped_xyz[i] = best_wrapped
        s_values[i] = best_s
        r_values[i] = np.sqrt(best_r2)

    axis_s = cumulative_s
    return {
        "wrapped_xyz": wrapped_xyz,
        "s": s_values,
        "r": r_values,
        "axis_xyz": axis_xyz,
        "axis_s": axis_s,
    }


def summarize_sinkmeta(
    rows, *, wrapped_xyz=None, axis_distance=None, s_values=None, r_values=None
):
    arrays = {
        "meta": np.asarray([row["meta"] for row in rows], dtype=float),
        "rbias": np.asarray([row["rbias"] for row in rows], dtype=float),
    }
    if wrapped_xyz is None:
        arrays["cx"] = np.asarray([row["cx"] for row in rows], dtype=float)
        arrays["cy"] = np.asarray([row["cy"] for row in rows], dtype=float)
        arrays["cz"] = np.asarray([row["cz"] for row in rows], dtype=float)
    else:
        arrays["cx"] = np.asarray(wrapped_xyz[:, 0], dtype=float)
        arrays["cy"] = np.asarray(wrapped_xyz[:, 1], dtype=float)
        arrays["cz"] = np.asarray(wrapped_xyz[:, 2], dtype=float)

    summary = {"samples": int(len(rows))}
    for key, values in arrays.items():
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            summary[key] = {
                "min": float("nan"),
                "max": float("nan"),
                "span": float("nan"),
                "mean": float("nan"),
            }
            continue
        summary[key] = {
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
            "span": float(np.max(finite) - np.min(finite)),
            "mean": float(np.mean(finite)),
        }

    xyz = np.column_stack((arrays["cx"], arrays["cy"], arrays["cz"]))
    _, unique_indices = np.unique(
        np.round(xyz, decimals=3), axis=0, return_index=True
    )
    summary["unique_xyz_count_1e3"] = int(len(unique_indices))
    summary["finite_fraction"] = float(
        np.mean(np.all(np.isfinite(xyz), axis=1) & np.isfinite(arrays["meta"]))
    )
    if axis_distance is not None:
        finite_distance = axis_distance[np.isfinite(axis_distance)]
        summary["axis_distance"] = {
            "min": float(np.min(finite_distance)),
            "max": float(np.max(finite_distance)),
            "mean": float(np.mean(finite_distance)),
        }
    if s_values is not None:
        finite_s = s_values[np.isfinite(s_values)]
        summary["s"] = {
            "min": float(np.min(finite_s)),
            "max": float(np.max(finite_s)),
            "span": float(np.max(finite_s) - np.min(finite_s)),
            "mean": float(np.mean(finite_s)),
        }
    if r_values is not None:
        finite_r = r_values[np.isfinite(r_values)]
        summary["r"] = {
            "min": float(np.min(finite_r)),
            "max": float(np.max(finite_r)),
            "span": float(np.max(finite_r) - np.min(finite_r)),
            "mean": float(np.mean(finite_r)),
        }
    return summary


def dump_summary_json(summary, out_path):
    Path(out_path).write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )


def save_sinkmeta_plots(
    rows,
    potential_rows,
    axis_rows,
    output_dir,
    *,
    wrapped_xyz=None,
    axis_distance=None,
    s_values=None,
    r_values=None,
    axis_s=None,
):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    steps = np.asarray([row["step"] for row in rows], dtype=float)
    times = np.asarray([row["time"] for row in rows], dtype=float)
    if wrapped_xyz is None:
        cx = np.asarray([row["cx"] for row in rows], dtype=float)
        cy = np.asarray([row["cy"] for row in rows], dtype=float)
        cz = np.asarray([row["cz"] for row in rows], dtype=float)
    else:
        cx = np.asarray(wrapped_xyz[:, 0], dtype=float)
        cy = np.asarray(wrapped_xyz[:, 1], dtype=float)
        cz = np.asarray(wrapped_xyz[:, 2], dtype=float)
    meta = np.asarray([row["meta"] for row in rows], dtype=float)
    rbias = np.asarray([row["rbias"] for row in rows], dtype=float)
    ax_cx = np.asarray([row["cx"] for row in axis_rows], dtype=float)
    ax_cy = np.asarray([row["cy"] for row in axis_rows], dtype=float)
    ax_cz = np.asarray([row["cz"] for row in axis_rows], dtype=float)

    paths = {}

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), dpi=180, sharex=True)
    axes[0].plot(steps, cx, lw=1.1, label="cx")
    axes[0].plot(steps, cy, lw=1.1, label="cy")
    axes[0].plot(steps, cz, lw=1.1, label="cz")
    axes[0].set_ylabel("wrapped CV")
    axes[0].set_title("DNA_COU SinkMeta CV Time Series")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, ncol=3)

    axes[1].plot(steps, meta, lw=1.2, color="tab:red", label="meta")
    axes[1].plot(steps, rbias, lw=1.2, color="tab:purple", label="rbias")
    axes[1].set_ylabel("Bias (kcal/mol)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, ncol=2)

    if axis_distance is not None:
        axes[2].plot(steps, axis_distance, lw=1.2, color="tab:green")
        axes[2].set_ylabel("dist(axis)")
        axes[2].grid(alpha=0.25)
    else:
        axes[2].axis("off")

    displacement = np.sqrt(
        (cx - cx[0]) ** 2 + (cy - cy[0]) ** 2 + (cz - cz[0]) ** 2
    )
    axes[3].plot(steps, displacement, lw=1.2, color="tab:olive")
    axes[3].set_xlabel("Step")
    axes[3].set_ylabel("Wrapped drift")
    axes[3].grid(alpha=0.25)

    fig.tight_layout()
    time_series_path = output_dir / "dna_cou_sinkmeta_timeseries.png"
    fig.savefig(time_series_path, bbox_inches="tight")
    plt.close(fig)
    paths["timeseries"] = time_series_path

    fig2, axes2 = plt.subplots(1, 3, figsize=(13.5, 4.2), dpi=180)
    projections = [
        ("cx", "cy", cx, cy),
        ("cx", "cz", cx, cz),
        ("cy", "cz", cy, cz),
    ]
    for ax, (xlabel, ylabel, xvals, yvals) in zip(axes2, projections):
        hb = ax.hexbin(xvals, yvals, gridsize=35, cmap="viridis", mincnt=1)
        if xlabel == "cx" and ylabel == "cy":
            ax.plot(ax_cx, ax_cy, color="black", lw=1.1, alpha=0.9, zorder=3)
        elif xlabel == "cx" and ylabel == "cz":
            ax.plot(ax_cx, ax_cz, color="black", lw=1.1, alpha=0.9, zorder=3)
        else:
            ax.plot(ax_cy, ax_cz, color="black", lw=1.1, alpha=0.9, zorder=3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{xlabel}-{ylabel} occupancy")
        fig2.colorbar(hb, ax=ax, fraction=0.048, pad=0.04)
    fig2.tight_layout()
    occupancy_path = output_dir / "dna_cou_sinkmeta_occupancy.png"
    fig2.savefig(occupancy_path, bbox_inches="tight")
    plt.close(fig2)
    paths["occupancy"] = occupancy_path

    return paths

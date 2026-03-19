import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from ase import Atoms


def resolve_prips_plugin_path():
    try:
        import prips

        plugin_path = Path(prips.__file__).resolve().parent / "_prips.so"
    except Exception:
        result = subprocess.run(
            [sys.executable, "-m", "prips"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "failed to resolve prips plugin path from `python -m prips`\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

        output = result.stdout + "\n" + result.stderr
        match = re.search(r"Plugin Path:\s*(.+)", output)
        if match is None:
            raise RuntimeError(
                "failed to resolve Plugin Path from `python -m prips`"
            )
        plugin_path = Path(match.group(1).strip())

    if not plugin_path.exists():
        raise FileNotFoundError(f"prips plugin does not exist: {plugin_path}")
    return plugin_path


def read_atom_names(case_dir, prefix="tip3p"):
    lines = (
        (Path(case_dir) / f"{prefix}_atom_name.txt").read_text().splitlines()
    )
    return [line.strip() for line in lines[1:] if line.strip()]


def atom_names_to_symbols(atom_names):
    symbols = []
    for atom_name in atom_names:
        match = re.match(r"[A-Za-z]+", atom_name.strip())
        if match is None:
            raise ValueError(
                f"Failed to infer element from atom name: {atom_name}"
            )
        token = match.group(0)
        if len(token) == 1:
            symbols.append(token.upper())
        else:
            symbols.append(token[0].upper() + token[1:].lower())
    return symbols


def read_positions(case_dir, prefix="tip3p"):
    lines = (
        (Path(case_dir) / f"{prefix}_coordinate.txt").read_text().splitlines()
    )
    atom_count = int(lines[0].split()[0])
    coords = np.array(
        [
            [float(value) for value in line.split()[:3]]
            for line in lines[1 : atom_count + 1]
        ],
        dtype=np.float64,
    )
    if coords.shape != (atom_count, 3):
        raise ValueError(
            f"Invalid coordinate shape {coords.shape} in {prefix}_coordinate.txt"
        )
    return coords


def build_atoms(case_dir, prefix="tip3p"):
    atom_names = read_atom_names(case_dir, prefix=prefix)
    return Atoms(
        symbols=atom_names_to_symbols(atom_names),
        positions=read_positions(case_dir, prefix=prefix),
        pbc=False,
    )


def summarize_force_errors(lhs, rhs):
    diff = np.asarray(lhs, dtype=np.float64) - np.asarray(rhs, dtype=np.float64)
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
    }


def oxygen_indices_from_atom_names(case_dir, prefix="tip3p"):
    atom_names = read_atom_names(case_dir, prefix=prefix)
    return np.array(
        [
            i
            for i, name in enumerate(atom_names)
            if name.strip().upper().startswith("O")
        ],
        dtype=np.int64,
    )


def load_coordinate_trajectory(case_dir, atom_count, traj_name="mdcrd.dat"):
    raw = np.fromfile(Path(case_dir) / traj_name, dtype=np.float32)
    frame_width = atom_count * 3
    if frame_width <= 0 or raw.size % frame_width != 0:
        raise ValueError(
            f"Invalid trajectory shape: atom_count={atom_count}, raw_size={raw.size}"
        )
    return raw.reshape(-1, atom_count, 3).astype(np.float64)


def load_box_trajectory(case_dir, box_name="mdbox.txt"):
    rows = []
    for line in (Path(case_dir) / box_name).read_text().splitlines():
        fields = line.split()
        if len(fields) < 3:
            continue
        rows.append([float(fields[0]), float(fields[1]), float(fields[2])])
    if not rows:
        raise ValueError(f"No box records found in {box_name}")
    return np.asarray(rows, dtype=np.float64)


def compute_oo_rdf(
    trajectory,
    box_lengths,
    oxygen_indices,
    *,
    r_max=8.0,
    bin_width=0.05,
):
    oxy = np.asarray(oxygen_indices, dtype=np.int64)
    if oxy.size < 2:
        raise ValueError("Need at least two oxygen atoms for O-O RDF")

    box_lengths = np.asarray(box_lengths, dtype=np.float64)
    if box_lengths.ndim == 1:
        box_lengths = np.repeat(
            box_lengths[None, :], trajectory.shape[0], axis=0
        )
    if box_lengths.shape[0] != trajectory.shape[0]:
        if box_lengths.shape[0] == 1:
            box_lengths = np.repeat(box_lengths, trajectory.shape[0], axis=0)
        else:
            raise ValueError(
                "Box frame count does not match trajectory frame count"
            )

    edges = np.arange(0.0, r_max + bin_width, bin_width, dtype=np.float64)
    hist = np.zeros(edges.size - 1, dtype=np.float64)
    shell_volumes = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    n_oxy = oxy.size

    for frame_idx in range(trajectory.shape[0]):
        positions = trajectory[frame_idx, oxy, :]
        box = box_lengths[frame_idx]
        deltas = positions[:, None, :] - positions[None, :, :]
        deltas -= box * np.round(deltas / box)
        distances = np.linalg.norm(deltas, axis=-1)
        pair_distances = distances[np.triu_indices(n_oxy, k=1)]
        hist += np.histogram(pair_distances, bins=edges)[0]

    mean_volume = float(np.mean(np.prod(box_lengths, axis=1)))
    density = n_oxy / mean_volume
    frame_count = trajectory.shape[0]
    pair_norm = 0.5 * n_oxy * frame_count
    rdf = hist / np.maximum(pair_norm * shell_volumes * density, 1e-12)
    centers = 0.5 * (edges[:-1] + edges[1:])
    peak_idx = int(np.argmax(rdf))
    return {
        "r": centers,
        "g_r": rdf,
        "peak_r": float(centers[peak_idx]),
        "peak_g_r": float(rdf[peak_idx]),
        "frame_count": int(frame_count),
        "oxygen_count": int(n_oxy),
        "r_max": float(r_max),
        "bin_width": float(bin_width),
    }


def save_rdf_plot(r, g_r, output_path, *, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=160)
    ax.plot(r, g_r, lw=1.4, color="tab:blue")
    ax.set_xlabel("r (Angstrom)")
    ax.set_ylabel("g_OO(r)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def dump_json(data, out_path):
    Path(out_path).write_text(json.dumps(data, indent=2, sort_keys=True))


def create_mace_calculator(*, family, model, device):
    if family == "off":
        from mace.calculators import mace_off

        return mace_off(model=model, device=device)
    if family == "mp":
        from mace.calculators import mace_mp

        return mace_mp(
            model=model,
            dispersion=False,
            default_dtype="float32",
            device=device,
        )
    raise ValueError(f"Unsupported MACE family: {family}")


def compute_reference_forces(
    case_dir,
    *,
    family,
    model,
    device,
    prefix="tip3p",
    positions=None,
):
    atoms = build_atoms(case_dir, prefix=prefix)
    if positions is not None:
        atoms.positions = np.asarray(positions, dtype=np.float64)
    atoms.calc = create_mace_calculator(
        family=family,
        model=model,
        device=device,
    )
    return np.asarray(atoms.get_forces(), dtype=np.float64)


def load_metrics(case_dir):
    return json.loads((Path(case_dir) / "mace_metrics.json").read_text())


def load_plugin_forces(case_dir):
    return np.load(Path(case_dir) / "mace_last_forces.npy")


def load_plugin_positions(case_dir):
    return np.load(Path(case_dir) / "mace_last_positions.npy")


def write_mdin(
    case_dir,
    plugin_path,
    *,
    step_limit,
    prefix="tip3p",
    mode="nve",
    dt=0.0,
    write_mdout_interval=1,
    write_information_interval=1,
    write_trajectory_interval=0,
    print_zeroth_frame=1,
    thermostat=None,
    thermostat_tau=None,
    thermostat_seed=None,
    target_temperature=None,
):
    lines = [
        'md_name = "benchmark tip3p perf mace"',
        f'mode = "{mode}"',
        f"step_limit = {step_limit}",
        f"dt = {dt}",
        f'coordinate_in_file = "{prefix}_coordinate.txt"',
        f'mass_in_file = "{prefix}_mass.txt"',
        f"print_zeroth_frame = {print_zeroth_frame}",
        f"write_mdout_interval = {write_mdout_interval}",
        f"write_information_interval = {write_information_interval}",
        f"write_trajectory_interval = {write_trajectory_interval}",
        'frc = "frc.dat"',
        f'plugin = "{Path(plugin_path).as_posix()}"',
        'py = "mace_plugin.py"',
    ]
    if thermostat is not None:
        lines.append(f'thermostat = "{thermostat}"')
    if thermostat_tau is not None:
        lines.append(f"thermostat_tau = {thermostat_tau}")
    if thermostat_seed is not None:
        lines.append(f"thermostat_seed = {thermostat_seed}")
    if target_temperature is not None:
        lines.append(f"target_temperature = {target_temperature}")
    mdin = "\n".join(lines) + "\n"
    Path(case_dir, "mdin.spg.toml").write_text(mdin, encoding="utf-8")


def write_mace_plugin_script(
    case_dir, *, family, model, device, prefix="tip3p"
):
    script = f"""import json
import time
from pathlib import Path

import numpy as np
from ase import Atoms

from prips import Sponge

CASE_DIR = Path(__file__).resolve().parent
PREFIX = {prefix!r}
MACE_FAMILY = {family!r}
MACE_MODEL = {model!r}
MACE_DEVICE = {device!r}
ATOM_COUNT = 0
ATOMS = None
CALC = None
RECORDS = []
LAST_FORCES = None
LAST_POSITIONS = None

backend_name = "numpy" if Sponge._backend == 1 else "pytorch"
Sponge.set_backend(backend_name)


def _read_atom_names():
    lines = (CASE_DIR / f"{{PREFIX}}_atom_name.txt").read_text().splitlines()
    return [line.strip() for line in lines[1:] if line.strip()]


def _atom_names_to_symbols(atom_names):
    symbols = []
    for atom_name in atom_names:
        token = ""
        for char in atom_name.strip():
            if char.isalpha():
                token += char
            else:
                break
        if not token:
            raise ValueError(f"Failed to infer element from atom name: {{atom_name}}")
        if len(token) == 1:
            symbols.append(token.upper())
        else:
            symbols.append(token[0].upper() + token[1:].lower())
    return symbols


def _read_positions():
    lines = (CASE_DIR / f"{{PREFIX}}_coordinate.txt").read_text().splitlines()
    atom_count = int(lines[0].split()[0])
    positions = np.array(
        [[float(value) for value in line.split()[:3]] for line in lines[1 : atom_count + 1]],
        dtype=np.float64,
    )
    return atom_count, positions


def _create_calculator():
    if MACE_FAMILY == "off":
        from mace.calculators import mace_off

        return mace_off(model=MACE_MODEL, device=MACE_DEVICE)
    if MACE_FAMILY == "mp":
        from mace.calculators import mace_mp

        return mace_mp(
            model=MACE_MODEL,
            dispersion=False,
            default_dtype="float32",
            device=MACE_DEVICE,
        )
    raise ValueError(f"Unsupported MACE family: {{MACE_FAMILY}}")


def _tensor_to_numpy(tensor):
    if backend_name == "pytorch":
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def _as_atom_xyz(array):
    arr = np.asarray(array)
    if arr.shape != (ATOM_COUNT, 3):
        raise ValueError(
            f"Expected coordinate/force tensor shape ({{ATOM_COUNT}}, 3), "
            f"got {{arr.shape}}"
        )
    return arr


def _write_forces(target, forces):
    target_arr = _tensor_to_numpy(target)
    target_shape = tuple(target_arr.shape)
    if backend_name == "pytorch":
        import torch

        target.zero_()
        if target_shape != (ATOM_COUNT, 3):
            raise ValueError(
                f"Expected force tensor shape ({{ATOM_COUNT}}, 3), "
                f"got {{target_shape}}"
            )
        target[:ATOM_COUNT, :] = torch.as_tensor(
            forces,
            device=target.device,
            dtype=target.dtype,
        )
        if target.shape[0] > ATOM_COUNT:
            target[ATOM_COUNT:, :] = 0
        return

    if target_shape != (ATOM_COUNT, 3):
        raise ValueError(
            f"Expected force tensor shape ({{ATOM_COUNT}}, 3), "
            f"got {{target_shape}}"
        )
    target[...] = 0
    target[:ATOM_COUNT, :] = forces.astype(target.dtype, copy=False)


def After_Initial():
    global ATOM_COUNT, ATOMS, CALC

    atom_names = _read_atom_names()
    symbols = _atom_names_to_symbols(atom_names)
    ATOM_COUNT, positions = _read_positions()
    ATOMS = Atoms(symbols=symbols, positions=positions, pbc=False)
    CALC = _create_calculator()
    ATOMS.calc = CALC


def Calculate_Force():
    global LAST_FORCES, LAST_POSITIONS

    source_crd = Sponge.dd.crd if Sponge.dd is not None and Sponge.dd.crd is not None else Sponge.md_info.crd
    target_frc = Sponge.dd.frc if Sponge.dd is not None and Sponge.dd.frc is not None else Sponge.md_info.frc
    positions = _as_atom_xyz(_tensor_to_numpy(source_crd)).astype(np.float64, copy=True)
    ATOMS.positions = positions
    LAST_POSITIONS = positions.copy()

    start = time.perf_counter()
    forces = np.asarray(ATOMS.get_forces(), dtype=np.float32)
    elapsed_s = time.perf_counter() - start

    _write_forces(target_frc, forces)
    LAST_FORCES = forces.astype(np.float64, copy=True)
    RECORDS.append(
        {{
            "step": int(Sponge.md_info.sys.steps),
            "elapsed_s": float(elapsed_s),
            "force_l2": float(np.linalg.norm(forces)),
            "force_max_abs": float(np.max(np.abs(forces))),
        }}
    )


def Mdout_Print():
    if not RECORDS:
        return

    elapsed = [item["elapsed_s"] for item in RECORDS]
    steady = elapsed[1:] if len(elapsed) > 1 else elapsed
    summary = {{
        "family": MACE_FAMILY,
        "model": MACE_MODEL,
        "device": MACE_DEVICE,
        "backend": backend_name,
        "atom_count": ATOM_COUNT,
        "force_calls": len(RECORDS),
        "first_call_ms": elapsed[0] * 1000.0,
        "mean_call_ms": float(sum(elapsed) / len(elapsed) * 1000.0),
        "steady_mean_call_ms": float(sum(steady) / len(steady) * 1000.0),
        "records": RECORDS,
    }}
    (CASE_DIR / "mace_metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if LAST_FORCES is not None:
        np.save(CASE_DIR / "mace_last_forces.npy", LAST_FORCES)
    if LAST_POSITIONS is not None:
        np.save(CASE_DIR / "mace_last_positions.npy", LAST_POSITIONS)
"""
    Path(case_dir, "mace_plugin.py").write_text(script, encoding="utf-8")

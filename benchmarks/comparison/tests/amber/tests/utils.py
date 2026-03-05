import json
import os
import re
import shlex
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np

AMBER_REFERENCE_JSON_REL_PATH = "reference/amber/reference.json"
AMBER_REFERENCE_ROOT_REL_DIR = "reference/amber"
AMBER_REFERENCE_SYSTEMS_REL_DIR = "reference/amber/statics"


def print_validation_table(headers, rows, title=None):
    if not headers:
        return

    if title:
        print(f"\n{title}")
    else:
        print()

    if len(rows) == 1:
        row = [str(v) for v in rows[0]]
        if len(row) != len(headers):
            raise ValueError(
                "Header/row length mismatch in validation table: "
                f"headers={len(headers)}, row={len(row)}"
            )
        key_width = max(len(h) for h in headers)
        value_width = max(len(v) for v in row) if row else 0
        divider = "-" * (key_width + 3 + value_width)
        print(divider)
        for key, value in zip(headers, row):
            print(f"{key:<{key_width}} : {value}")
        print(divider)
        return

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            text = str(val)
            col_widths[i] = max(col_widths[i], len(text))
    col_widths = [w + 2 for w in col_widths]
    row_fmt = " | ".join([f"{{:<{w}}}" for w in col_widths])
    divider = "-" * (sum(col_widths) + 3 * (len(headers) - 1))

    print(divider)
    print(row_fmt.format(*headers))
    print(divider)
    for row in rows:
        print(row_fmt.format(*[str(v) for v in row]))
    print(divider)


def _get_comparison_root_from_statics(statics_path):
    return Path(statics_path).resolve().parents[2]


def _get_amber_reference_root(statics_path):
    return (
        _get_comparison_root_from_statics(statics_path) / "reference" / "amber"
    )


def _get_amber_reference_json_path(statics_path):
    return (
        _get_comparison_root_from_statics(statics_path)
        / AMBER_REFERENCE_JSON_REL_PATH
    )


def _get_amber_reference_system_case_dir(statics_path, case_name):
    return (
        _get_comparison_root_from_statics(statics_path)
        / AMBER_REFERENCE_SYSTEMS_REL_DIR
        / case_name
    )


@lru_cache(maxsize=4)
def _load_amber_reference_entries(reference_json_path_str):
    reference_json_path = Path(reference_json_path_str)
    if not reference_json_path.exists():
        raise FileNotFoundError(
            f"Missing AMBER reference file: {reference_json_path}"
        )
    payload = json.loads(reference_json_path.read_text())
    entries = payload.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError(
            f"Invalid AMBER reference format in {reference_json_path}: "
            "'entries' must be a list."
        )

    index = {}
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(
                f"Invalid AMBER reference entry #{idx}: expected object"
            )
        try:
            case_name = str(entry["case_name"])
            iteration = int(entry["iteration"])
        except KeyError as exc:
            raise ValueError(
                f"Invalid AMBER reference entry #{idx}: missing {exc}"
            ) from exc
        key = (case_name, iteration)
        if key in index:
            raise ValueError(
                f"Duplicate AMBER reference entry for case={case_name}, "
                f"iteration={iteration}"
            )
        index[key] = entry
    return payload, index


def load_amber_reference_entry(statics_path, case_name, iteration):
    json_path = _get_amber_reference_json_path(statics_path).resolve()
    _payload, index = _load_amber_reference_entries(str(json_path))
    key = (str(case_name), int(iteration))
    if key not in index:
        raise KeyError(
            "AMBER reference entry not found for "
            f"case={case_name}, iteration={iteration} in {json_path}"
        )
    return index[key]


def load_amber_reference_energy(statics_path, case_name, iteration):
    entry = load_amber_reference_entry(statics_path, case_name, iteration)
    return float(entry["energy_epot"])


def load_amber_reference_forces(statics_path, case_name, iteration):
    entry = load_amber_reference_entry(statics_path, case_name, iteration)
    forces_rel_path = entry.get("forces_file")
    if not isinstance(forces_rel_path, str) or not forces_rel_path:
        raise ValueError(
            "AMBER reference entry missing 'forces_file': "
            f"case={case_name}, iteration={iteration}"
        )
    force_path = (
        _get_amber_reference_root(statics_path) / forces_rel_path
    ).resolve()
    if not force_path.exists():
        raise FileNotFoundError(
            f"Missing AMBER reference forces file: {force_path}"
        )
    arr = np.load(force_path)
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(
            f"Invalid AMBER reference forces shape {arr.shape} in {force_path}"
        )
    return arr


def copy_amber_reference_system_files(statics_path, case_dir, case_name):
    src_dir = _get_amber_reference_system_case_dir(
        statics_path, case_name
    ).resolve()
    dst_dir = Path(case_dir).resolve()
    if not src_dir.exists():
        raise FileNotFoundError(
            f"Missing AMBER reference system directory: {src_dir}"
        )

    copied = []
    for src_file in sorted(src_dir.glob("*")):
        if not src_file.is_file():
            continue
        dst_file = dst_dir / src_file.name
        shutil.copy2(src_file, dst_file)
        copied.append(dst_file.name)

    if not copied:
        raise ValueError(
            f"No reference system files found in {src_dir} for case {case_name}"
        )
    return copied


def require_ambertools():
    for exe in ("tleap", "sander"):
        if shutil.which(exe) is None:
            raise RuntimeError(
                f"Required executable '{exe}' is not available in PATH."
            )


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


def _run_command(cmd, cwd):
    result = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, check=False
    )
    output = result.stdout + "\n" + result.stderr
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


def run_tleap(case_dir):
    return _run_command(["tleap", "-f", "tleap.in"], cwd=case_dir)


def run_sander_run0(case_dir):
    cmd = [
        "sander",
        "-O",
        "-i",
        "sander.in",
        "-o",
        "sander.out",
        "-p",
        "system.parm7",
        "-c",
        "system.rst7",
        "-r",
        "system_out.rst7",
        "-x",
        "system.nc",
        "-frc",
        "mdfrc",
        "-inf",
        "system.mdinfo",
    ]
    return _run_command(cmd, cwd=case_dir)


def _resolve_sponge_command():
    # Default to SPONGE from current environment (e.g., pixi dev-cuda).
    # Allow overriding for local debugging (e.g., SPONGE_BIN=./build-cpu/SPONGE).
    sponge_bin = os.environ.get("SPONGE_BIN", "SPONGE")
    return shlex.split(sponge_bin)


def run_sponge_run0(case_dir):
    cmd = _resolve_sponge_command() + ["-mdin", "sponge.mdin"]
    return _run_command(cmd, cwd=case_dir)


def is_cuda_init_failure(error_text):
    lowered = error_text.lower()
    return (
        "fail to initialize cuda" in lowered
        or "spongeerrormallocfailed raised by controller::init_device"
        in lowered
    )


def sponge_solvent_lj_not_initialized(output_text):
    return "SOLVENT LJ IS NOT INITIALIZED" in output_text


def extract_sander_epot(sander_out_path):
    text = Path(sander_out_path).read_text()
    matches = re.findall(
        r"EPtot\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)", text
    )
    if not matches:
        raise ValueError(f"Failed to parse EPtot from {sander_out_path}")
    return float(matches[-1])


def extract_sander_pressure(sander_out_path):
    text = Path(sander_out_path).read_text()
    matches = re.findall(
        r"PRESS\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)", text
    )
    if not matches:
        raise ValueError(f"Failed to parse PRESS from {sander_out_path}")
    return float(matches[-1])


def extract_sponge_potential(case_dir):
    from Xponge.analysis import MdoutReader

    mdout_path = Path(case_dir) / "mdout.txt"
    if not mdout_path.exists():
        raise FileNotFoundError(f"SPONGE mdout not found: {mdout_path}")

    mdout = MdoutReader(str(mdout_path))
    if not hasattr(mdout, "potential"):
        raise ValueError("No 'potential' field found in SPONGE mdout.")
    return float(mdout.potential[-1])


def extract_sponge_pressure(case_dir):
    mdout_path = Path(case_dir) / "mdout.txt"
    lines = mdout_path.read_text().splitlines()
    if len(lines) < 2:
        raise ValueError(f"Invalid SPONGE mdout: {mdout_path}")

    headers = lines[0].split()
    data_line = None
    for line in reversed(lines[1:]):
        if not line.strip():
            continue
        if line.strip()[0].isdigit():
            data_line = line
            break

    if data_line is None:
        raise ValueError(f"Failed to find numeric data line in {mdout_path}")

    values = data_line.split()
    if len(values) != len(headers):
        raise ValueError(
            f"mdout header/value mismatch in {mdout_path}: "
            f"{len(headers)} vs {len(values)}"
        )
    kv = dict(zip(headers, values))
    try:
        pxx = float(kv["Pxx"])
        pyy = float(kv["Pyy"])
        pzz = float(kv["Pzz"])
    except KeyError as e:
        raise ValueError(
            f"Missing pressure component {e} in {mdout_path}"
        ) from e
    return (pxx + pyy + pzz) / 3.0


def _read_parm7_flag_values(parm7_path, flag_name):
    values = []
    in_target_flag = False
    in_data = False
    target = f"%FLAG {flag_name}"

    with open(parm7_path, "r") as f:
        for line in f:
            if line.startswith("%FLAG"):
                if in_target_flag:
                    break
                in_target_flag = line.strip() == target
                in_data = False
                continue

            if not in_target_flag:
                continue

            if line.startswith("%FORMAT"):
                in_data = True
                continue

            if in_data:
                values.extend(line.split())

    if not values:
        raise ValueError(f"Failed to read %FLAG {flag_name} from {parm7_path}")
    return [float(v.replace("D", "E")) for v in values]


def _read_parm7_flag_strings(parm7_path, flag_name):
    values = []
    in_target_flag = False
    in_data = False
    target = f"%FLAG {flag_name}"

    with open(parm7_path, "r") as f:
        for line in f:
            if line.startswith("%FLAG"):
                if in_target_flag:
                    break
                in_target_flag = line.strip() == target
                in_data = False
                continue

            if not in_target_flag:
                continue

            if line.startswith("%FORMAT"):
                in_data = True
                continue

            if in_data:
                values.extend(line.split())

    if not values:
        raise ValueError(f"Failed to read %FLAG {flag_name} from {parm7_path}")
    return values


def write_gb_in_file_from_parm7(parm7_path, gb_out_path):
    radii = _read_parm7_flag_values(parm7_path, "RADII")
    screen = _read_parm7_flag_values(parm7_path, "SCREEN")
    if len(radii) != len(screen):
        raise ValueError(
            f"RADII/SCREEN length mismatch in {parm7_path}: "
            f"{len(radii)} vs {len(screen)}"
        )

    with open(gb_out_path, "w") as f:
        f.write(f"{len(radii)}\n")
        for r, s in zip(radii, screen):
            f.write(f"{r:.6f} {s:.6f}\n")


def write_tip4p_virtual_atom_from_parm7(
    parm7_path, virtual_atom_out_path, a=0.12797, b=0.12797
):
    atom_names = _read_parm7_flag_strings(parm7_path, "ATOM_NAME")
    atom_types = _read_parm7_flag_strings(parm7_path, "AMBER_ATOM_TYPE")
    residue_labels = _read_parm7_flag_strings(parm7_path, "RESIDUE_LABEL")
    residue_pointer = [
        int(v) for v in _read_parm7_flag_values(parm7_path, "RESIDUE_POINTER")
    ]

    if len(atom_names) != len(atom_types):
        raise ValueError(
            f"ATOM_NAME/AMBER_ATOM_TYPE length mismatch in {parm7_path}: "
            f"{len(atom_names)} vs {len(atom_types)}"
        )
    if len(residue_labels) != len(residue_pointer):
        raise ValueError(
            f"RESIDUE_LABEL/RESIDUE_POINTER length mismatch in {parm7_path}: "
            f"{len(residue_labels)} vs {len(residue_pointer)}"
        )

    residue_pointer.append(len(atom_names) + 1)
    lines = []
    for i, resname in enumerate(residue_labels):
        if resname not in {"WAT", "HOH"}:
            continue

        start = residue_pointer[i] - 1
        end = residue_pointer[i + 1] - 1  # exclusive
        if end <= start:
            continue

        local_names = atom_names[start:end]
        local_types = atom_types[start:end]

        o_candidates = [
            start + j
            for j, (name, atype) in enumerate(zip(local_names, local_types))
            if atype == "OW" or name == "O"
        ]
        h_candidates = [
            start + j
            for j, (name, atype) in enumerate(zip(local_names, local_types))
            if atype == "HW" or name in {"H1", "H2"} or name.startswith("H")
        ]
        ep_candidates = [
            start + j
            for j, (name, atype) in enumerate(zip(local_names, local_types))
            if atype == "EP" or name.startswith("EP") or name == "M"
        ]

        if (
            len(o_candidates) != 1
            or len(ep_candidates) != 1
            or len(h_candidates) < 2
        ):
            raise ValueError(
                f"Unexpected TIP4P water residue layout at residue {i + 1} "
                f"({resname}) in {parm7_path}: names={local_names}, types={local_types}"
            )

        h1, h2 = sorted(h_candidates)[:2]
        o = o_candidates[0]
        ep = ep_candidates[0]
        lines.append(f"2 {ep} {o} {h1} {h2} {a:.6f} {b:.6f}")

    if not lines:
        raise ValueError(
            f"No TIP4P water residues found in {parm7_path}. "
            "Cannot generate virtual_atom_in_file."
        )

    Path(virtual_atom_out_path).write_text("\n".join(lines) + "\n")


def read_rst7_coords(rst7_path):
    lines = Path(rst7_path).read_text().splitlines()
    if len(lines) < 2:
        raise ValueError(f"Invalid rst7 (too short): {rst7_path}")

    title = lines[0]
    count_line = lines[1]
    head = count_line.split()
    if not head:
        raise ValueError(f"Invalid rst7 count line: {rst7_path}")
    natom = int(head[0])

    target = natom * 3
    coords_vals = []
    idx = 2
    while idx < len(lines) and len(coords_vals) < target:
        line = lines[idx]
        for j in range(0, len(line), 12):
            chunk = line[j : j + 12]
            if chunk.strip():
                coords_vals.append(float(chunk))
                if len(coords_vals) == target:
                    break
        idx += 1

    if len(coords_vals) != target:
        raise ValueError(
            f"Failed to parse rst7 coordinates: {rst7_path}, "
            f"expected {target}, got {len(coords_vals)}"
        )

    coords = np.array(coords_vals, dtype=np.float64).reshape(natom, 3)
    tail_lines = lines[idx:]
    return title, count_line, coords, tail_lines


def write_rst7_coords(rst7_path, title, count_line, coords, tail_lines):
    coords = np.asarray(coords, dtype=np.float64)
    natom = coords.shape[0]
    flat = coords.reshape(-1)

    out_lines = [title, count_line]
    for i in range(0, len(flat), 6):
        block = flat[i : i + 6]
        out_lines.append("".join(f"{v:12.7f}" for v in block))
    out_lines.extend(tail_lines)
    Path(rst7_path).write_text("\n".join(out_lines) + "\n")

    # Sanity: preserve atom count in count line.
    if int(count_line.split()[0]) != natom:
        raise ValueError("rst7 atom count mismatch after writing.")


def perturb_rst7_inplace(rst7_path, perturbation, seed):
    title, count_line, coords, tail_lines = read_rst7_coords(rst7_path)
    if perturbation > 0:
        rng = np.random.RandomState(seed)
        noise = (rng.rand(*coords.shape) - 0.5) * 2.0 * perturbation
        coords = coords + noise
    write_rst7_coords(rst7_path, title, count_line, coords, tail_lines)
    return coords


def perturb_rst7_with_rigid_water_inplace(
    rst7_path, parm7_path, perturbation, seed, water_resnames=("WAT", "HOH")
):
    title, count_line, coords, tail_lines = read_rst7_coords(rst7_path)
    if perturbation <= 0:
        return coords

    residue_labels = _read_parm7_flag_strings(parm7_path, "RESIDUE_LABEL")
    residue_pointer = [
        int(v) for v in _read_parm7_flag_values(parm7_path, "RESIDUE_POINTER")
    ]
    residue_pointer.append(coords.shape[0] + 1)

    if len(residue_labels) + 1 != len(residue_pointer):
        raise ValueError(
            f"RESIDUE_LABEL/RESIDUE_POINTER mismatch in {parm7_path}: "
            f"{len(residue_labels)} labels vs {len(residue_pointer) - 1} ranges"
        )

    rng = np.random.RandomState(seed)
    water_resname_set = set(water_resnames)
    for i, resname in enumerate(residue_labels):
        start = residue_pointer[i] - 1
        end = residue_pointer[i + 1] - 1
        if end <= start:
            continue

        if resname in water_resname_set:
            # Keep water internal geometry unchanged: one random translation per water residue.
            delta = (rng.rand(3) - 0.5) * 2.0 * perturbation
            coords[start:end] = coords[start:end] + delta
        else:
            delta = (rng.rand(end - start, 3) - 0.5) * 2.0 * perturbation
            coords[start:end] = coords[start:end] + delta

    write_rst7_coords(rst7_path, title, count_line, coords, tail_lines)
    return coords


def extract_sander_forces_mdfrc(mdfrc_path):
    # AMBER ASCII mdfrc uses fixed-width records (typically 10F8.3).
    lines = Path(mdfrc_path).read_text().splitlines()
    if len(lines) < 2:
        raise ValueError(f"Invalid mdfrc (too short): {mdfrc_path}")

    values = []
    for i, line in enumerate(lines):
        if i == 0:
            continue
        for j in range(0, len(line), 8):
            chunk = line[j : j + 8]
            if chunk.strip():
                values.append(float(chunk))
    if len(values) % 3 != 0:
        raise ValueError(
            f"Invalid mdfrc force length: {len(values)} in {mdfrc_path}"
        )
    return np.array(values, dtype=np.float64).reshape(-1, 3)


def extract_sponge_forces_frc_dat(frc_path, natom):
    raw = np.fromfile(frc_path, dtype=np.float32)
    frame_width = natom * 3
    if frame_width == 0 or raw.size % frame_width != 0:
        raise ValueError(
            f"Invalid frc.dat length: size={raw.size}, natom={natom}, path={frc_path}"
        )
    return raw[-frame_width:].reshape(natom, 3).astype(np.float64)


def force_stats(reference, predicted):
    reference = np.asarray(reference, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if reference.shape != predicted.shape:
        raise ValueError(
            f"Force shape mismatch: ref={reference.shape}, pred={predicted.shape}"
        )
    diff = predicted - reference
    abs_diff = np.abs(diff)
    max_abs = float(abs_diff.max())
    rms = float(np.sqrt(np.mean(diff * diff)))

    ref_norm = np.linalg.norm(reference, axis=1)
    pred_norm = np.linalg.norm(predicted, axis=1)
    denom = ref_norm * pred_norm
    valid = denom > 1.0e-12
    if np.any(valid):
        cos = float(
            np.mean(
                np.sum(reference[valid] * predicted[valid], axis=1)
                / denom[valid]
            )
        )
    else:
        cos = 1.0
    return {
        "max_abs_diff": max_abs,
        "rms_diff": rms,
        "cosine_similarity": cos,
    }


def force_stats_with_rigid_water_entities(
    parm7_path,
    reference_forces,
    predicted_forces,
    water_resnames=("WAT", "HOH"),
):
    reference_forces = np.asarray(reference_forces, dtype=np.float64)
    predicted_forces = np.asarray(predicted_forces, dtype=np.float64)
    if reference_forces.shape != predicted_forces.shape:
        raise ValueError(
            f"Force shape mismatch: ref={reference_forces.shape}, "
            f"pred={predicted_forces.shape}"
        )

    residue_labels = _read_parm7_flag_strings(parm7_path, "RESIDUE_LABEL")
    residue_pointer = [
        int(v) for v in _read_parm7_flag_values(parm7_path, "RESIDUE_POINTER")
    ]
    residue_pointer.append(reference_forces.shape[0] + 1)
    if len(residue_labels) + 1 != len(residue_pointer):
        raise ValueError(
            f"RESIDUE_LABEL/RESIDUE_POINTER mismatch in {parm7_path}: "
            f"{len(residue_labels)} labels vs {len(residue_pointer) - 1} ranges"
        )

    water_set = set(water_resnames)
    entity_ref = []
    entity_pred = []
    for i, resname in enumerate(residue_labels):
        start = residue_pointer[i] - 1
        end = residue_pointer[i + 1] - 1
        if end <= start:
            continue
        if resname in water_set:
            # Compare rigid-water translational force by residue-net force.
            entity_ref.append(reference_forces[start:end].sum(axis=0))
            entity_pred.append(predicted_forces[start:end].sum(axis=0))
        else:
            entity_ref.extend(reference_forces[start:end])
            entity_pred.extend(predicted_forces[start:end])

    return force_stats(np.asarray(entity_ref), np.asarray(entity_pred))

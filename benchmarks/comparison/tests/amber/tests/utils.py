import shutil
from pathlib import Path

import numpy as np

from benchmarks.utils import Extractor
from benchmarks.comparison.utils import (
    force_stats,
    get_reference_json_path,
    get_reference_root,
    get_reference_statics_case_dir,
    load_reference_npy,
)
from benchmarks.comparison.utils import (
    load_reference_entry as load_common_reference_entry,
)


def load_amber_reference_entry(statics_path, case_name, iteration):
    return load_common_reference_entry(
        get_reference_json_path(statics_path, "amber"),
        "AMBER",
        case_name,
        iteration,
    )


def load_amber_reference_energy(statics_path, case_name, iteration):
    entry = load_amber_reference_entry(statics_path, case_name, iteration)
    return float(entry["energy_epot"])


def load_amber_reference_forces(statics_path, case_name, iteration):
    entry = load_amber_reference_entry(statics_path, case_name, iteration)
    return load_reference_npy(
        get_reference_root(statics_path, "amber"),
        entry,
        key="forces_file",
        suite_label="AMBER",
        expected_ndim=2,
        expected_last_dim=3,
    )


def copy_amber_reference_system_files(statics_path, case_dir, case_name):
    src_dir = get_reference_statics_case_dir(
        statics_path, "amber", case_name
    ).resolve()
    dst_dir = Path(case_dir).resolve()
    if not src_dir.exists():
        raise FileNotFoundError(
            f"Missing AMBER reference system directory: {src_dir}"
        )

    copied = False
    for src_file in sorted(src_dir.glob("*")):
        if not src_file.is_file():
            continue
        dst_file = dst_dir / src_file.name
        shutil.copy2(src_file, dst_file)
        copied = True

    if not copied:
        raise ValueError(
            f"No reference system files found in {src_dir} for case {case_name}"
        )


def _read_parm7_flag_items(parm7_path, flag_name):
    target = f"%FLAG {flag_name}"
    items = []
    in_target_flag = False
    in_data = False

    for line in Path(parm7_path).read_text().splitlines():
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
            items.extend(line.split())

    if not items:
        raise ValueError(f"Failed to read %FLAG {flag_name} from {parm7_path}")
    return items


def extract_sponge_potential(case_dir):
    return Extractor.extract_sponge_potential(case_dir)


def _read_parm7_flag_values(parm7_path, flag_name):
    return [
        float(v.replace("D", "E"))
        for v in _read_parm7_flag_items(parm7_path, flag_name)
    ]


def _read_parm7_flag_strings(parm7_path, flag_name):
    return _read_parm7_flag_items(parm7_path, flag_name)


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


def extract_sponge_forces_frc_dat(frc_path, natom):
    frc_path = Path(frc_path)
    return Extractor.extract_sponge_forces(
        frc_path.parent, natom, frc_name=frc_path.name
    )


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
            entity_ref.append(reference_forces[start:end].sum(axis=0))
            entity_pred.append(predicted_forces[start:end].sum(axis=0))
        else:
            entity_ref.extend(reference_forces[start:end])
            entity_pred.extend(predicted_forces[start:end])

    return force_stats(np.asarray(entity_ref), np.asarray(entity_pred))

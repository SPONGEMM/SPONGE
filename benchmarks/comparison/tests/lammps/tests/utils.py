import json
from functools import lru_cache
import numpy as np
import pathlib
import subprocess
from Xponge.analysis import MdoutReader

EV_TO_KCAL_MOL = 23.060548
ATM_PER_KCAL_MOL_A3 = 68568.415
BAR_TO_ATM = 1.0 / 1.01325
LAMMPS_REFERENCE_JSON_REL_PATH = "reference/lammps/reference.json"
LAMMPS_REFERENCE_ROOT_REL_DIR = "reference/lammps"


def run_sponge_command(work_dir, mdin_file=None):
    cmd = ["SPONGE"]
    if mdin_file is not None:
        cmd.extend(["-mdin", mdin_file])
    result = subprocess.run(
        cmd,
        cwd=work_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print("\n[SPONGE stdout]\n")
        print(result.stdout)
        print("\n[SPONGE stderr]\n")
        print(result.stderr)
        raise subprocess.CalledProcessError(
            result.returncode,
            cmd,
            output=result.stdout,
            stderr=result.stderr,
        )
    return result


def _detect_lammps_units_from_case(work_dir):
    lammps_input = work_dir.parent / "lammps" / "in.lammps"
    if not lammps_input.exists():
        return "metal"
    with open(lammps_input, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("units"):
                parts = line.split()
                if len(parts) >= 2:
                    return parts[1].lower()
    return "metal"


def _detect_lammps_units_from_log(log_path):
    input_path = pathlib.Path(log_path).parent / "in.lammps"
    if not input_path.exists():
        return "metal"
    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("units"):
                parts = line.split()
                if len(parts) >= 2:
                    return parts[1].lower()
    return "metal"


def _sponge_pressure_scale_to_lammps(work_dir, lammps_units=None):
    units = (lammps_units or _detect_lammps_units_from_case(work_dir)).lower()
    # SPONGE mdout pressure/stress are in bar.
    # LAMMPS real uses atm; metal uses bar.
    if units == "real":
        return BAR_TO_ATM
    return 1.0


def extract_lammps_potential(log_path):
    with open(log_path, "r") as f:
        lines = f.readlines()
    data_index = -1
    headers = []
    for i, line in enumerate(lines):
        if "Step" in line and "PotEng" in line:
            headers = line.strip().split()
            data_index = i + 1
            break
    if data_index == -1 or data_index >= len(lines):
        raise ValueError(f"在 {log_path} 中未找到热力学输出。")
    data_line = lines[data_index].strip().split()
    try:
        pot_eng_idx = headers.index("PotEng")
        pot_eng = float(data_line[pot_eng_idx])
    except (ValueError, IndexError) as e:
        raise ValueError(f"无法从该行解析 PotEng: {lines[data_index]}") from e
    units = _detect_lammps_units_from_log(log_path)
    if units == "real":
        return pot_eng
    if units == "metal":
        return pot_eng * EV_TO_KCAL_MOL
    raise ValueError(f"不支持的LAMMPS单位体系: {units}")


def extract_lammps_pressure(log_path):
    with open(log_path, "r") as f:
        lines = f.readlines()
    data_index = -1
    headers = []
    for i, line in enumerate(lines):
        if "Step" in line and "Press" in line:
            headers = line.strip().split()
            data_index = i + 1
            break
    if data_index == -1 or data_index >= len(lines):
        raise ValueError(f"在 {log_path} 中未找到压强输出。")
    data_line = lines[data_index].strip().split()
    try:
        press_idx = headers.index("Press")
        pressure = float(data_line[press_idx])
    except (ValueError, IndexError) as e:
        raise ValueError(f"无法从该行解析 Press: {lines[data_index]}") from e
    return pressure


def extract_lammps_stress(log_path):
    with open(log_path, "r") as f:
        lines = f.readlines()
    data_index = -1
    headers = []
    for i, line in enumerate(lines):
        if "Step" in line and "Pxx" in line:
            headers = line.strip().split()
            data_index = i + 1
            break
    if data_index == -1 or data_index >= len(lines):
        raise ValueError(f"在 {log_path} 中未找到应力信息。")
    data_line = lines[data_index].strip().split()
    stress = {}
    for key in ["Pxx", "Pyy", "Pzz", "Pxy", "Pxz", "Pyz"]:
        try:
            idx = headers.index(key)
            stress[key] = float(data_line[idx])
        except (ValueError, IndexError):
            stress[key] = float("nan")
    return stress


def extract_sponge_pressure(work_dir, lammps_units=None):
    mdout_path = work_dir / "mdout.txt"
    if not mdout_path.exists():
        raise FileNotFoundError(f"未找到 SPONGE 输出文件: {mdout_path}")
    mdout = MdoutReader(str(mdout_path))
    if hasattr(mdout, "pressure"):
        scale = _sponge_pressure_scale_to_lammps(work_dir, lammps_units)
        return mdout.pressure[-1] * scale
    raise ValueError("在 SPONGE 输出中未找到压力信息。")


def extract_sponge_stress(work_dir, lammps_units=None):
    mdout_path = work_dir / "mdout.txt"
    if not mdout_path.exists():
        raise FileNotFoundError(f"未找到 SPONGE 输出文件: {mdout_path}")
    mdout = MdoutReader(str(mdout_path))
    scale = _sponge_pressure_scale_to_lammps(work_dir, lammps_units)
    stress = {}
    for key in ["Pxx", "Pyy", "Pzz", "Pxy", "Pxz", "Pyz"]:
        if hasattr(mdout, key):
            stress[key] = getattr(mdout, key)[-1] * scale
        else:
            stress[key] = float("nan")
    return stress


def extract_sponge_potential(work_dir):
    mdout_path = work_dir / "mdout.txt"
    if not mdout_path.exists():
        raise FileNotFoundError(f"未找到 SPONGE 输出文件: {mdout_path}")
    mdout = MdoutReader(str(mdout_path))
    if hasattr(mdout, "potential"):
        return mdout.potential[-1]
    raise ValueError("在 SPONGE 输出中未找到势能信息。")


def extract_lammps_forces(work_dir):
    dump_file = work_dir / "forces.dump"
    if not dump_file.exists():
        raise FileNotFoundError(f"LAMMPS力输出文件未找到: {dump_file}")
    forces = {}
    with open(dump_file, "r") as f:
        lines = f.readlines()
    start_idx = -1
    for i, line in enumerate(lines):
        if line.startswith("ITEM: ATOMS"):
            start_idx = i + 1
            break
    if start_idx == -1:
        raise ValueError("LAMMPS力输出文件格式错误，未找到原子力数据")
    for line in lines[start_idx:]:
        parts = line.split()
        if len(parts) >= 4:
            try:
                atom_id = int(parts[0])
                fx, fy, fz = float(parts[1]), float(parts[2]), float(parts[3])
                forces[atom_id] = np.array([fx, fy, fz])
            except ValueError:
                continue
    sorted_ids = sorted(forces.keys())
    return np.array([forces[i] for i in sorted_ids])


def extract_sponge_forces(work_dir, num_atoms):
    force_bin = work_dir / "frc.dat"
    if force_bin.exists():
        raw = np.fromfile(force_bin, dtype=np.float32)
        frame_width = num_atoms * 3
        if frame_width == 0 or raw.size % frame_width != 0:
            raise ValueError(
                f"SPONGE力轨迹文件尺寸异常: {force_bin}, size={raw.size}, num_atoms={num_atoms}"
            )
        return raw[-frame_width:].reshape(num_atoms, 3)

    raise FileNotFoundError(f"SPONGE力轨迹文件未找到: {force_bin}")


def compute_pressure_stress_from_coords_forces(coords, forces, box):
    coords = np.asarray(coords, dtype=float)
    forces = np.asarray(forces, dtype=float)
    box = np.asarray(box, dtype=float)
    if coords.shape != forces.shape:
        raise ValueError("coords 与 forces 形状不一致。")
    if box.shape != (3,):
        raise ValueError("box 必须是长度为3的向量。")
    volume = float(box[0] * box[1] * box[2])
    if volume <= 0:
        raise ValueError("box 体积必须大于0。")

    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    fx = forces[:, 0]
    fy = forces[:, 1]
    fz = forces[:, 2]

    scale = ATM_PER_KCAL_MOL_A3 / volume
    stress = {
        "Pxx": float(np.sum(x * fx) * scale),
        "Pyy": float(np.sum(y * fy) * scale),
        "Pzz": float(np.sum(z * fz) * scale),
        "Pxy": float(0.5 * np.sum(x * fy + y * fx) * scale),
        "Pxz": float(0.5 * np.sum(x * fz + z * fx) * scale),
        "Pyz": float(0.5 * np.sum(y * fz + z * fy) * scale),
    }
    pressure = (stress["Pxx"] + stress["Pyy"] + stress["Pzz"]) / 3.0
    return pressure, stress


def generate_diamond_structure(
    nx=2,
    ny=2,
    nz=2,
    perturbation=0.1,
    num_types=1,
    a=5.43,
    type_pattern="sequential",
):
    basis = (
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
                [0.25, 0.25, 0.25],
                [0.25, 0.75, 0.75],
                [0.75, 0.25, 0.75],
                [0.75, 0.75, 0.25],
            ]
        )
        * a
    )

    atoms = []
    atom_types = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                offset = np.array([i, j, k]) * a
                for b_idx, b in enumerate(basis):
                    atoms.append(b + offset)
                    if num_types == 2 and type_pattern == "sublattice":
                        atom_types.append(1 if b_idx < (len(basis) // 2) else 2)
                    else:
                        atom_types.append((len(atom_types) % num_types) + 1)

    coords = np.array(atoms)
    box = np.array([nx * a, ny * a, nz * a])

    noise = (np.random.random(coords.shape) - 0.5) * 2 * perturbation
    coords += noise
    coords = coords % box

    return coords, box, np.array(atom_types)


def rewrite_edip_atom_types(edip_file, atom_types):
    edip_file = pathlib.Path(edip_file)
    lines = edip_file.read_text().splitlines()

    marker_idx = None
    for idx, line in enumerate(lines):
        if line.strip().startswith("# Atom types"):
            marker_idx = idx
            break
    if marker_idx is None:
        raise ValueError(f"未找到 '# Atom types' 标记: {edip_file}")

    # SPONGE EDIP 文件约定原子类型从 0 开始。
    atom_type_line = " ".join(str(int(t) - 1) for t in atom_types)
    new_lines = lines[: marker_idx + 1] + [atom_type_line]
    edip_file.write_text("\n".join(new_lines) + "\n")


def write_sponge_coords(file_path, coords, box):
    num_atoms = len(coords)
    with open(file_path, "w") as f:
        f.write(f"{num_atoms} 0.0\n")
        for x, y, z in coords:
            f.write(f"{x:.12f} {y:.12f} {z:.12f}\n")
        f.write(f"{box[0]:.12f} {box[1]:.12f} {box[2]:.12f}\n")
        f.write("90.0 90.0 90.0\n")


def write_sponge_mass(file_path, masses):
    with open(file_path, "w") as f:
        f.write(f"{len(masses)}\n")
        for mass in masses:
            f.write(f"{mass}\n")


def write_sponge_types(file_path, atom_types):
    with open(file_path, "w") as f:
        f.write(f"{len(atom_types)}\n")
        for atom_type in atom_types:
            f.write(f"{atom_type}\n")


def write_lammps_data(file_path, coords, box, masses, atom_types=None):
    num_atoms = len(coords)
    if atom_types is None:
        atom_types = [1] * num_atoms

    unique_types = sorted(list(set(atom_types)))
    num_atom_types = len(masses)

    if len(masses) < len(unique_types):
        raise ValueError("提供的 masses 列表长度小于唯一原子类型数量。")

    with open(file_path, "w") as f:
        f.write("Generated by SPONGE Test\n\n")
        f.write(f"{num_atoms} atoms\n")
        f.write(f"{num_atom_types} atom types\n\n")
        f.write(f"0.0 {box[0]:.12f} xlo xhi\n")
        f.write(f"0.0 {box[1]:.12f} ylo yhi\n")
        f.write(f"0.0 {box[2]:.12f} zlo zhi\n\n")
        f.write("Masses\n\n")
        for i, m in enumerate(masses):
            f.write(f"{i + 1} {m}\n")
        f.write("\n")
        f.write("Atoms\n\n")
        for i, ((x, y, z), t) in enumerate(zip(coords, atom_types)):
            f.write(f"{i + 1} {t} {x:.12f} {y:.12f} {z:.12f}\n")


def write_lammps_charge_data(
    file_path,
    coords,
    box,
    masses,
    atom_types,
    type_id_map,
    title="Generated by SPONGE Test",
):
    num_atoms = len(coords)
    if len(atom_types) != num_atoms:
        raise ValueError("atom_types 的长度与坐标数量不一致。")

    ordered_types = sorted(type_id_map.items(), key=lambda item: item[1])
    if not ordered_types:
        raise ValueError("type_id_map 不能为空。")

    with open(file_path, "w") as f:
        f.write(f"{title}\n\n")
        f.write(f"{num_atoms} atoms\n")
        f.write(f"{len(ordered_types)} atom types\n\n")
        f.write(f"0.0 {box[0]:.12f} xlo xhi\n")
        f.write(f"0.0 {box[1]:.12f} ylo yhi\n")
        f.write(f"0.0 {box[2]:.12f} zlo zhi\n\n")
        f.write("Masses\n\n")
        for atom_name, atom_type in ordered_types:
            if atom_name not in masses:
                raise ValueError(f"masses 中缺少类型 {atom_name} 的质量。")
            f.write(f"{atom_type} {masses[atom_name]}\n")

        f.write("\n")
        f.write("Atoms\n\n")
        for i, ((x, y, z), atom_name) in enumerate(
            zip(coords, atom_types), start=1
        ):
            if atom_name not in type_id_map:
                raise ValueError(f"type_id_map 中缺少类型 {atom_name}。")
            f.write(
                f"{i} {type_id_map[atom_name]} 0.0 {x:.12f} {y:.12f} {z:.12f}\n"
            )


def generate_perturbed_water_system(nx, ny, nz, spacing, perturbation, seed):
    water_coords = np.array(
        [
            [0.000, 0.000, 0.000],
            [0.757, 0.586, 0.000],
            [-0.757, 0.586, 0.000],
        ]
    )
    water_types = ["O", "H", "H"]

    rng = np.random.RandomState(seed)
    coords = []
    atom_types = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                offset = np.array([i * spacing, j * spacing, k * spacing])
                molecule_perturb = (rng.rand(3) - 0.5) * perturbation
                for wc, atom_type in zip(water_coords, water_types):
                    atom_perturb = (rng.rand(3) - 0.5) * perturbation * 0.1
                    coords.append(wc + offset + molecule_perturb + atom_perturb)
                    atom_types.append(atom_type)

    box_size = [nx * spacing, ny * spacing, nz * spacing]
    return np.array(coords), box_size, atom_types


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
            str_val = str(val)
            col_widths[i] = max(col_widths[i], len(str_val))

    col_widths = [w + 2 for w in col_widths]

    header_fmt = " | ".join([f"{{:<{w}}}" for w in col_widths])

    divider = "-" * (sum(col_widths) + 3 * (len(headers) - 1))

    print(divider)
    print(header_fmt.format(*headers))
    print(divider)

    for row in rows:
        formatted_row = [str(val) for val in row]
        print(header_fmt.format(*formatted_row))
    print(divider)


def _get_comparison_root_from_statics(statics_path):
    return pathlib.Path(statics_path).resolve().parents[2]


def _get_lammps_reference_root(statics_path):
    return (
        _get_comparison_root_from_statics(statics_path)
        / LAMMPS_REFERENCE_ROOT_REL_DIR
    )


def _get_lammps_reference_json_path(statics_path):
    return (
        _get_comparison_root_from_statics(statics_path)
        / LAMMPS_REFERENCE_JSON_REL_PATH
    )


@lru_cache(maxsize=4)
def _load_lammps_reference_entries(reference_json_path_str):
    reference_json_path = pathlib.Path(reference_json_path_str)
    if not reference_json_path.exists():
        raise FileNotFoundError(
            f"Missing LAMMPS reference file: {reference_json_path}"
        )
    payload = json.loads(reference_json_path.read_text())
    entries = payload.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError(
            f"Invalid LAMMPS reference format in {reference_json_path}: "
            "'entries' must be a list."
        )

    index = {}
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(
                f"Invalid LAMMPS reference entry #{idx}: expected object"
            )
        try:
            case_name = str(entry["case_name"])
            iteration = int(entry["iteration"])
        except KeyError as exc:
            raise ValueError(
                f"Invalid LAMMPS reference entry #{idx}: missing {exc}"
            ) from exc
        key = (case_name, iteration)
        if key in index:
            raise ValueError(
                f"Duplicate LAMMPS reference entry for case={case_name}, "
                f"iteration={iteration}"
            )
        index[key] = entry
    return payload, index


def load_lammps_reference_entry(statics_path, case_name, iteration):
    json_path = _get_lammps_reference_json_path(statics_path).resolve()
    _payload, index = _load_lammps_reference_entries(str(json_path))
    key = (str(case_name), int(iteration))
    if key not in index:
        raise KeyError(
            "LAMMPS reference entry not found for "
            f"case={case_name}, iteration={iteration} in {json_path}"
        )
    return index[key]


def load_lammps_reference_forces(statics_path, case_name, iteration):
    entry = load_lammps_reference_entry(statics_path, case_name, iteration)
    forces_rel_path = entry.get("forces_file")
    if not isinstance(forces_rel_path, str) or not forces_rel_path:
        raise ValueError(
            "LAMMPS reference entry missing 'forces_file': "
            f"case={case_name}, iteration={iteration}"
        )
    force_path = (
        _get_lammps_reference_root(statics_path) / forces_rel_path
    ).resolve()
    if not force_path.exists():
        raise FileNotFoundError(
            f"Missing LAMMPS reference forces file: {force_path}"
        )
    arr = np.asarray(np.load(force_path), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(
            f"Invalid LAMMPS reference forces shape {arr.shape} in {force_path}"
        )
    return arr


def load_lammps_reference_charges(statics_path, case_name, iteration):
    entry = load_lammps_reference_entry(statics_path, case_name, iteration)
    charges_rel_path = entry.get("charges_file")
    if not isinstance(charges_rel_path, str) or not charges_rel_path:
        raise ValueError(
            "LAMMPS reference entry missing 'charges_file': "
            f"case={case_name}, iteration={iteration}"
        )
    charges_path = (
        _get_lammps_reference_root(statics_path) / charges_rel_path
    ).resolve()
    if not charges_path.exists():
        raise FileNotFoundError(
            f"Missing LAMMPS reference charges file: {charges_path}"
        )
    arr = np.asarray(np.load(charges_path), dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(
            f"Invalid LAMMPS reference charges shape {arr.shape} in {charges_path}"
        )
    return arr


def load_lammps_reference_stress(statics_path, case_name, iteration):
    entry = load_lammps_reference_entry(statics_path, case_name, iteration)
    stress = entry.get("stress")
    if not isinstance(stress, dict):
        raise ValueError(
            "LAMMPS reference entry missing 'stress' object: "
            f"case={case_name}, iteration={iteration}"
        )
    result = {}
    for key in ["Pxx", "Pyy", "Pzz", "Pxy", "Pxz", "Pyz"]:
        if key not in stress:
            raise ValueError(
                f"Missing stress component '{key}' in LAMMPS reference entry: "
                f"case={case_name}, iteration={iteration}"
            )
        result[key] = float(stress[key])
    return result


def load_lammps_reference_thermo(statics_path, case_name, iteration):
    entry = load_lammps_reference_entry(statics_path, case_name, iteration)
    thermo = entry.get("thermo")
    if not isinstance(thermo, dict):
        raise ValueError(
            "LAMMPS reference entry missing 'thermo' object: "
            f"case={case_name}, iteration={iteration}"
        )
    result = {}
    for key, value in thermo.items():
        result[str(key)] = float(value)
    return result

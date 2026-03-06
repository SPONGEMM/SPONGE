import pathlib

import numpy as np

from benchmarks.comparison.utils import (
    get_reference_json_path,
    get_reference_root,
    load_reference_npy,
)
from benchmarks.comparison.utils import (
    load_reference_entry as load_common_reference_entry,
)
from benchmarks.utils import Extractor

EV_TO_KCAL_MOL = 23.060548
ATM_PER_KCAL_MOL_A3 = 68568.415
BAR_TO_ATM = 1.0 / 1.01325


def prepare_case_dir(outputs_path, case_name, iteration, mpi_np=None):
    case_dir = pathlib.Path(outputs_path) / case_name / str(iteration)
    case_dir.mkdir(parents=True, exist_ok=True)
    return case_dir


def _detect_lammps_units(in_lammps_path):
    if not in_lammps_path.exists():
        return "metal"
    with open(in_lammps_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("units"):
                parts = line.split()
                if len(parts) >= 2:
                    return parts[1].lower()
    return "metal"


def _detect_lammps_units_for_case(work_dir):
    return _detect_lammps_units(work_dir.parent / "lammps" / "in.lammps")


def _detect_lammps_units_from_log(log_path):
    return _detect_lammps_units(pathlib.Path(log_path).parent / "in.lammps")


def _parse_lammps_thermo_row(log_path, required_headers):
    if isinstance(required_headers, str):
        required_headers = (required_headers,)
    headers = tuple(required_headers)
    with open(log_path, "r") as f:
        lines = f.read().splitlines()
    for i, line in enumerate(lines):
        parts = line.split()
        if (
            not parts
            or "Step" not in parts
            or not all(header in parts for header in headers)
        ):
            continue
        data_index = i + 1
        if data_index >= len(lines):
            break
        data_line = lines[data_index].split()
        if len(data_line) < len(parts):
            continue
        try:
            values = [float(value) for value in data_line]
        except ValueError:
            continue
        header_index = {header: idx for idx, header in enumerate(parts)}
        row = {
            header: values[index]
            for header, index in header_index.items()
            if index < len(values)
        }
        missing = [header for header in headers if header not in row]
        if missing:
            continue
        return row
    raise ValueError(
        f"Thermodynamic output with required headers {headers} not found in {log_path}."
    )


def _sponge_pressure_scale_to_lammps(work_dir, lammps_units=None):
    units = (lammps_units or _detect_lammps_units_for_case(work_dir)).lower()
    # SPONGE mdout pressure/stress are in bar.
    # LAMMPS real uses atm; metal uses bar.
    if units == "real":
        return BAR_TO_ATM
    return 1.0


def extract_lammps_potential(log_path):
    row = _parse_lammps_thermo_row(log_path, "PotEng")
    pot_eng = row["PotEng"]
    units = _detect_lammps_units_from_log(log_path)
    if units == "real":
        return pot_eng
    if units == "metal":
        return pot_eng * EV_TO_KCAL_MOL
    raise ValueError(f"Unsupported LAMMPS unit system: {units}")


def extract_lammps_pressure(log_path):
    row = _parse_lammps_thermo_row(log_path, "Press")
    pressure = row["Press"]
    return pressure


def extract_lammps_stress(log_path):
    row = _parse_lammps_thermo_row(
        log_path, ("Pxx", "Pyy", "Pzz", "Pxy", "Pxz", "Pyz")
    )
    stress = {}
    for key in ["Pxx", "Pyy", "Pzz", "Pxy", "Pxz", "Pyz"]:
        stress[key] = row.get(key, float("nan"))
    return stress


def extract_sponge_pressure(work_dir, lammps_units=None):
    pressure = Extractor.extract_sponge_pressure(work_dir)
    return pressure * _sponge_pressure_scale_to_lammps(work_dir, lammps_units)


def extract_sponge_stress(work_dir, lammps_units=None):
    stress = Extractor.extract_sponge_stress(work_dir)
    scale = _sponge_pressure_scale_to_lammps(work_dir, lammps_units)
    return {key: value * scale for key, value in stress.items()}


def extract_sponge_potential(work_dir):
    return Extractor.extract_sponge_potential(work_dir)


def extract_lammps_forces(work_dir):
    dump_file = work_dir / "forces.dump"
    if not dump_file.exists():
        raise FileNotFoundError(
            f"LAMMPS force output file not found: {dump_file}"
        )
    forces = {}
    with open(dump_file, "r") as f:
        lines = f.readlines()
    start_idx = -1
    for i, line in enumerate(lines):
        if line.startswith("ITEM: ATOMS"):
            start_idx = i + 1
            break
    if start_idx == -1:
        raise ValueError(
            "Invalid LAMMPS force output format: missing atomic force section."
        )
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
    return Extractor.extract_sponge_forces(work_dir, num_atoms)


def compute_pressure_stress_from_coords_forces(coords, forces, box):
    coords = np.asarray(coords, dtype=float)
    forces = np.asarray(forces, dtype=float)
    box = np.asarray(box, dtype=float)
    if coords.shape != forces.shape:
        raise ValueError("Coordinate and force arrays shape mismatch.")
    if box.shape != (3,):
        raise ValueError("Box must be a vector of length 3.")
    volume = float(box[0] * box[1] * box[2])
    if volume <= 0:
        raise ValueError("Box volume must be positive.")

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
        raise ValueError(f"Marker '# Atom types' not found: {edip_file}")

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
        raise ValueError(
            "Mass list length is smaller than the number of unique atom types."
        )

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
        raise ValueError(
            "The length of atom_types does not match the number of atoms."
        )

    ordered_types = sorted(type_id_map.items(), key=lambda item: item[1])
    if not ordered_types:
        raise ValueError("type_id_map must not be empty.")

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
                raise ValueError(
                    f"Mass entry for atom type '{atom_name}' is missing in masses."
                )
            f.write(f"{atom_type} {masses[atom_name]}\n")

        f.write("\n")
        f.write("Atoms\n\n")
        for i, ((x, y, z), atom_name) in enumerate(
            zip(coords, atom_types), start=1
        ):
            if atom_name not in type_id_map:
                raise ValueError(
                    f"Atom type '{atom_name}' is missing in type_id_map."
                )
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


def load_lammps_reference_entry(statics_path, case_name, iteration):
    return load_common_reference_entry(
        get_reference_json_path(statics_path, "lammps"),
        "LAMMPS",
        case_name,
        iteration,
    )


def load_lammps_reference_forces(statics_path, case_name, iteration):
    entry = load_lammps_reference_entry(statics_path, case_name, iteration)
    return load_reference_npy(
        get_reference_root(statics_path, "lammps"),
        entry,
        key="forces_file",
        suite_label="LAMMPS",
        expected_ndim=2,
        expected_last_dim=3,
    )


def load_lammps_reference_charges(statics_path, case_name, iteration):
    entry = load_lammps_reference_entry(statics_path, case_name, iteration)
    return load_reference_npy(
        get_reference_root(statics_path, "lammps"),
        entry,
        key="charges_file",
        suite_label="LAMMPS",
        expected_ndim=1,
    )


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

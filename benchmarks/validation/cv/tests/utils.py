import math
from pathlib import Path


DISTANCE_ATOMS = (4, 6)
ANGLE_ATOMS = (4, 6, 8)
PHI_ATOMS = (4, 6, 8, 14)
PSI_ATOMS = (6, 8, 14, 16)
RMSD_ATOMS = tuple(range(22))


def write_simple_cv_file(case_dir, cv_file="cv.txt"):
    cv_text = (
        "print\n"
        "{\n"
        "    CV = distance angle phi psi\n"
        "}\n"
        "distance\n"
        "{\n"
        "    CV_type = distance\n"
        "    atom = 4 6\n"
        "}\n"
        "angle\n"
        "{\n"
        "    CV_type = angle\n"
        "    atom = 4 6 8\n"
        "}\n"
        "phi\n"
        "{\n"
        "    CV_type = dihedral\n"
        "    atom = 4 6 8 14\n"
        "}\n"
        "psi\n"
        "{\n"
        "    CV_type = dihedral\n"
        "    atom = 6 8 14 16\n"
        "}\n"
    )
    Path(case_dir, cv_file).write_text(cv_text, encoding="utf-8")


def write_extended_cv_file(
    case_dir, cv_file="cv.txt", rmsd_ref_file="rmsd_ref.txt"
):
    rmsd_atoms = " ".join(str(atom) for atom in RMSD_ATOMS)
    cv_text = (
        "print\n"
        "{\n"
        "    CV = distance angle phi psi combo tab_distance_linear rmsd_ala\n"
        "}\n"
        "distance\n"
        "{\n"
        "    CV_type = distance\n"
        "    atom = 4 6\n"
        "}\n"
        "angle\n"
        "{\n"
        "    CV_type = angle\n"
        "    atom = 4 6 8\n"
        "}\n"
        "phi\n"
        "{\n"
        "    CV_type = dihedral\n"
        "    atom = 4 6 8 14\n"
        "}\n"
        "psi\n"
        "{\n"
        "    CV_type = dihedral\n"
        "    atom = 6 8 14 16\n"
        "}\n"
        "combo\n"
        "{\n"
        "    CV_type = combination\n"
        "    CV = distance angle\n"
        "    function = distance + 0.5 * angle\n"
        "}\n"
        "tab_distance_linear\n"
        "{\n"
        "    CV_type = tabulated\n"
        "    CV = distance\n"
        "    min = 0.0\n"
        "    max = 3.0\n"
        "    parameter = 0.0 1.0 2.0 3.0\n"
        "}\n"
        "rmsd_ala\n"
        "{\n"
        "    CV_type = rmsd\n"
        f"    atom = {rmsd_atoms}\n"
        f"    coordinate_in_file = {rmsd_ref_file}\n"
        "    rotate = 1\n"
        "}\n"
    )
    Path(case_dir, cv_file).write_text(cv_text, encoding="utf-8")


def write_distance_bias_cv_file(
    case_dir,
    *,
    cv_file="cv.txt",
    print_distance=True,
    steer_weight=None,
    restrain_weight=None,
    restrain_reference=None,
):
    lines = []
    if print_distance:
        lines.extend(
            [
                "print",
                "{",
                "    CV = distance",
                "}",
            ]
        )

    lines.extend(
        [
            "distance",
            "{",
            "    CV_type = distance",
            "    atom = 4 6",
            "}",
        ]
    )

    if steer_weight is not None:
        lines.extend(
            [
                "steer",
                "{",
                "    CV = distance",
                f"    weight = {steer_weight}",
                "}",
            ]
        )

    if restrain_weight is not None:
        lines.extend(
            [
                "restrain",
                "{",
                "    CV = distance",
                f"    weight = {restrain_weight}",
                f"    reference = {restrain_reference}",
                "}",
            ]
        )

    Path(case_dir, cv_file).write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_displacement_cv_file(case_dir, cv_file="cv.txt"):
    lines = ["print", "{", "    CV = distance dx dy dz", "}"]
    for name, cv_type in (
        ("distance", "distance"),
        ("dx", "displacement_x"),
        ("dy", "displacement_y"),
        ("dz", "displacement_z"),
    ):
        lines.extend(
            [name, "{", f"    CV_type = {cv_type}", "    atom = 4 6", "}"]
        )
    Path(case_dir, cv_file).write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def rewrite_coordinate_for_open_boundary(
    coordinate_path, *, shifted_atom=None, shift_x=0.0, box_length=1000.0
):
    path = Path(coordinate_path)
    tokens = path.read_text(encoding="utf-8").split()
    atom_numbers = int(tokens[0])
    if shifted_atom is not None:
        x_index = 1 + 3 * shifted_atom
        tokens[x_index] = str(float(tokens[x_index]) + shift_x)
    box_start = 1 + 3 * atom_numbers
    tokens[box_start : box_start + 3] = [str(box_length)] * 3
    path.write_text("\n".join(tokens) + "\n", encoding="utf-8")


def write_bias_cv_file(
    case_dir,
    *,
    target_cv,
    cv_file="cv.txt",
    print_cv=True,
    steer_weight=None,
    restrain_weight=None,
    restrain_reference=None,
    rmsd_ref_file="rmsd_ref.txt",
):
    rmsd_atoms = " ".join(str(atom) for atom in RMSD_ATOMS)
    cv_blocks = {
        "distance": [
            "distance",
            "{",
            "    CV_type = distance",
            "    atom = 4 6",
            "}",
        ],
        "angle": [
            "angle",
            "{",
            "    CV_type = angle",
            "    atom = 4 6 8",
            "}",
        ],
        "phi": [
            "phi",
            "{",
            "    CV_type = dihedral",
            "    atom = 4 6 8 14",
            "}",
        ],
        "psi": [
            "psi",
            "{",
            "    CV_type = dihedral",
            "    atom = 6 8 14 16",
            "}",
        ],
        "combo": [
            "combo",
            "{",
            "    CV_type = combination",
            "    CV = distance angle",
            "    function = distance + 0.5 * angle",
            "}",
        ],
        "tab_distance_linear": [
            "tab_distance_linear",
            "{",
            "    CV_type = tabulated",
            "    CV = distance",
            "    min = 0.0",
            "    max = 3.0",
            "    parameter = 0.0 1.0 2.0 3.0",
            "}",
        ],
        "rmsd_ala": [
            "rmsd_ala",
            "{",
            "    CV_type = rmsd",
            f"    atom = {rmsd_atoms}",
            f"    coordinate_in_file = {rmsd_ref_file}",
            "    rotate = 1",
            "}",
        ],
    }
    cv_dependencies = {
        "distance": (),
        "angle": (),
        "phi": (),
        "psi": (),
        "combo": ("distance", "angle"),
        "tab_distance_linear": ("distance",),
        "rmsd_ala": (),
    }

    def collect_required_cvs(name, seen):
        if name in seen:
            return
        for dependency in cv_dependencies[name]:
            collect_required_cvs(dependency, seen)
        seen.append(name)

    required_cvs = []
    collect_required_cvs(target_cv, required_cvs)
    lines = []
    if print_cv:
        lines.extend(["print", "{", f"    CV = {target_cv}", "}"])

    for cv_name in required_cvs:
        lines.extend(cv_blocks[cv_name])

    if steer_weight is not None:
        lines.extend(
            [
                "steer",
                "{",
                f"    CV = {target_cv}",
                f"    weight = {steer_weight}",
                "}",
            ]
        )

    if restrain_weight is not None:
        lines.extend(
            [
                "restrain",
                "{",
                f"    CV = {target_cv}",
                f"    weight = {restrain_weight}",
                f"    reference = {restrain_reference}",
                "}",
            ]
        )

    Path(case_dir, cv_file).write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_dihedral_cv_file(case_dir, cv_file="cv.txt"):
    write_simple_cv_file(case_dir, cv_file=cv_file)


def write_validation_mdin(
    case_dir,
    *,
    step_limit=0,
    dt=0.0,
    cutoff=12.0,
    default_in_file_prefix="sys_flexible",
    cv_file="cv.txt",
    print_pressure=False,
):
    mdin = (
        'md_name = "validation alanine_dipeptide cv"\n'
        'mode = "nve"\n'
        f"step_limit = {step_limit}\n"
        f"dt = {dt}\n"
        f"cutoff = {cutoff}\n"
        f'default_in_file_prefix = "{default_in_file_prefix}"\n'
        'frc = "frc.dat"\n'
        "dont_check_input = 1\n"
        "print_zeroth_frame = 1\n"
        "write_mdout_interval = 1\n"
        "write_information_interval = 1\n"
    )
    if print_pressure:
        mdin += "print_pressure = 1\n"
    mdin += f'cv_in_file = "{cv_file}"\n'
    Path(case_dir, "mdin.spg.toml").write_text(mdin, encoding="utf-8")


def load_coordinates(coordinate_path):
    tokens = Path(coordinate_path).read_text(encoding="utf-8").split()
    atom_numbers = int(tokens[0])
    values = [float(token) for token in tokens[1 : 1 + atom_numbers * 3]]
    expected = atom_numbers * 3
    if len(values) != expected:
        raise ValueError(
            f"Coordinate value count mismatch in {coordinate_path}: "
            f"expected {expected}, got {len(values)}"
        )
    return [
        tuple(values[3 * index : 3 * index + 3])
        for index in range(atom_numbers)
    ]


def load_box_lengths(coordinate_path):
    tokens = Path(coordinate_path).read_text(encoding="utf-8").split()
    atom_numbers = int(tokens[0])
    start = 1 + atom_numbers * 3
    tail = [float(token) for token in tokens[start : start + 6]]
    if len(tail) < 3:
        raise ValueError(f"Missing box information in {coordinate_path}")
    return tuple(tail[:3])


def write_rmsd_reference_file(reference_coordinates, output_path, atom_indices):
    values = []
    for index in atom_indices:
        values.extend(reference_coordinates[index])
    text = "\n".join(
        " ".join(f"{value:.6f}" for value in values[offset : offset + 3])
        for offset in range(0, len(values), 3)
    )
    Path(output_path).write_text(text + "\n", encoding="utf-8")


def perturb_coordinates(coordinates, atom_indices):
    perturbed = [tuple(point) for point in coordinates]
    for order, atom_index in enumerate(atom_indices):
        x, y, z = perturbed[atom_index]
        if order % 4 == 0:
            perturbed[atom_index] = (x + 1.10, y - 0.36, z + 0.18)
        elif order % 4 == 1:
            perturbed[atom_index] = (x - 0.82, y + 0.74, z - 0.27)
        elif order % 4 == 2:
            perturbed[atom_index] = (x + 0.48, y + 0.34, z - 1.00)
        else:
            perturbed[atom_index] = (x - 0.35, y - 0.88, z + 0.61)
    return perturbed


def compute_distance(coordinates, atom_indices):
    p0, p1 = (coordinates[index] for index in atom_indices)
    return math.sqrt(
        sum((p0[axis] - p1[axis]) * (p0[axis] - p1[axis]) for axis in range(3))
    )


def compute_angle(coordinates, atom_indices):
    p0, p1, p2 = (coordinates[index] for index in atom_indices)
    vec_01 = tuple(p0[axis] - p1[axis] for axis in range(3))
    vec_21 = tuple(p2[axis] - p1[axis] for axis in range(3))
    cosine = dot(vec_01, vec_21) / math.sqrt(
        dot(vec_01, vec_01) * dot(vec_21, vec_21)
    )
    cosine = max(-0.999999, min(0.999999, cosine))
    return math.acos(cosine)


def compute_dihedral(coordinates, atom_indices):
    p0, p1, p2, p3 = (coordinates[index] for index in atom_indices)
    b0 = tuple(p0[i] - p1[i] for i in range(3))
    b1 = tuple(p2[i] - p1[i] for i in range(3))
    b2 = tuple(p3[i] - p2[i] for i in range(3))

    n0 = cross(b0, b1)
    n1 = cross(negate(b1), b2)
    b1_norm = normalize(b1)
    m1 = cross(n0, b1_norm)

    x = dot(n0, n1)
    y = dot(m1, n1)
    return -math.atan2(y, x)


def compute_kabsch_rmsd(reference_xyz, mobile_xyz):
    ref = [tuple(point) for point in reference_xyz]
    mob = [tuple(point) for point in mobile_xyz]
    if len(ref) != len(mob):
        raise ValueError(
            f"Coordinate shape mismatch for RMSD: {len(ref)} vs {len(mob)}"
        )
    if not ref:
        raise ValueError("Cannot compute RMSD for empty atom set")

    ref_centered = center_points(ref)
    mob_centered = center_points(mob)
    covariance = matmul3x3(transpose_points(mob_centered), ref_centered)
    rotation = kabsch_rotation(covariance)
    mob_fitted = [matvec(point, rotation) for point in mob_centered]

    squared = 0.0
    for mob_point, ref_point in zip(mob_fitted, ref_centered):
        squared += sum(
            (mob_point[axis] - ref_point[axis]) ** 2 for axis in range(3)
        )
    return math.sqrt(squared / len(ref))


def compute_tabulated_distance(distance_value):
    min_value = 0.0
    max_value = 3.0
    parameter = [0.0, 1.0, 2.0, 3.0]
    if distance_value < min_value:
        return parameter[0]
    if distance_value > max_value:
        return parameter[-1]

    padded = [parameter[0], *parameter, parameter[-1]]
    delta = (max_value - min_value) / (len(parameter) - 1)
    x0 = (distance_value - min_value) / delta
    spline_index = int(x0) + 1
    x = spline_index - x0
    return (
        _bspline_4_1(x) * padded[spline_index - 1]
        + _bspline_4_2(x) * padded[spline_index]
        + _bspline_4_3(x) * padded[spline_index + 1]
        + _bspline_4_4(x) * padded[spline_index + 2]
    )


def center_points(points):
    count = len(points)
    center = tuple(
        sum(point[axis] for point in points) / count for axis in range(3)
    )
    return [
        tuple(point[axis] - center[axis] for axis in range(3))
        for point in points
    ]


def transpose_points(points):
    return [[point[axis] for point in points] for axis in range(3)]


def matmul3x3(left_tall, right_points):
    matrix = [[0.0, 0.0, 0.0] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            matrix[i][j] = sum(
                left_tall[i][k] * right_points[k][j]
                for k in range(len(right_points))
            )
    return matrix


def kabsch_rotation(covariance):
    import numpy as np

    cov = np.asarray(covariance, dtype=np.float64)
    v, _, wt = np.linalg.svd(cov)
    det_sign = 1.0 if np.linalg.det(v @ wt) >= 0.0 else -1.0
    return (v @ np.diag([1.0, 1.0, det_sign]) @ wt).tolist()


def matvec(vec, matrix):
    return tuple(sum(vec[k] * matrix[k][j] for k in range(3)) for j in range(3))


def cross(vec_a, vec_b):
    return (
        vec_a[1] * vec_b[2] - vec_a[2] * vec_b[1],
        vec_a[2] * vec_b[0] - vec_a[0] * vec_b[2],
        vec_a[0] * vec_b[1] - vec_a[1] * vec_b[0],
    )


def dot(vec_a, vec_b):
    return sum(a * b for a, b in zip(vec_a, vec_b))


def negate(vec):
    return tuple(-value for value in vec)


def normalize(vec):
    norm = math.sqrt(dot(vec, vec))
    if norm == 0.0:
        raise ValueError("Cannot normalize a zero-length vector")
    return tuple(value / norm for value in vec)


def _bspline_4_1(x):
    return x * x * x / 6.0


def _bspline_4_2(x):
    return -0.5 * x * x * x + 0.5 * x * x + 0.5 * x + 1.0 / 6.0


def _bspline_4_3(x):
    return 0.5 * x * x * x - x * x + 2.0 / 3.0


def _bspline_4_4(x):
    return -x * x * x / 6.0 + 0.5 * x * x - 0.5 * x + 1.0 / 6.0

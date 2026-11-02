#!/usr/bin/env python3
import argparse
import json
import math
import re
import shutil
import struct
import subprocess
from pathlib import Path


def fail(message):
    raise AssertionError(message)


def compare_h5_files(h5diff, pure_path, sidecar_path):
    command = [
        h5diff,
        "--quiet",
        "--exclude-path",
        "/parameters/sponge/files/legacy_sidecars",
        str(pure_path),
        str(sidecar_path),
    ]
    result = subprocess.run(
        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    if result.returncode != 0:
        fail(
            f"H5 payload differs for {pure_path.name} after excluding "
            f"legacy_sidecars:\n{result.stdout}"
        )


def run_h5dump(h5dump, h5_path, dataset_path):
    result = subprocess.run(
        [h5dump, "-d", dataset_path, str(h5_path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        fail(f"h5dump failed for {h5_path}:{dataset_path}\n{result.stdout}")
    match = re.search(
        r"DATA\s*\{(?P<body>.*?)\n\s*\}", result.stdout, re.DOTALL
    )
    if not match:
        fail(f"failed to parse h5dump DATA block for {h5_path}:{dataset_path}")
    return match.group("body")


def h5dump_payload(body):
    return "\n".join(
        line.split(":", 1)[1] if ":" in line else line
        for line in body.splitlines()
    )


def read_h5_ints(h5dump, h5_path, dataset_path):
    payload = h5dump_payload(run_h5dump(h5dump, h5_path, dataset_path))
    return [int(value) for value in re.findall(r"(?<![A-Za-z_])-?\d+", payload)]


def read_h5_floats(h5dump, h5_path, dataset_path):
    payload = h5dump_payload(run_h5dump(h5dump, h5_path, dataset_path))
    return [
        float(value)
        for value in re.findall(
            r"(?<![A-Za-z_])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
            payload,
        )
    ]


def read_h5_strings(h5dump, h5_path, dataset_path):
    payload = h5dump_payload(run_h5dump(h5dump, h5_path, dataset_path))
    return re.findall(r'"([^"]*)"', payload)


def read_h5_enum_bools(h5dump, h5_path, dataset_path):
    payload = h5dump_payload(run_h5dump(h5dump, h5_path, dataset_path))
    tokens = re.findall(r"\b(?:TRUE|FALSE)\b", payload)
    if not tokens:
        fail(f"expected enum bool payload for {h5_path}:{dataset_path}")
    return [token == "TRUE" for token in tokens]


def read_h5_scalar_string(h5dump, h5_path, dataset_path):
    result = subprocess.run(
        [h5dump, "-d", dataset_path, str(h5_path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        fail(f"h5dump failed for {h5_path}:{dataset_path}\n{result.stdout}")
    match = re.search(
        r'DATA\s*\{\s*\(\d+\): "(?P<value>.*)"\s*\n\s*\}\s*\n\}',
        result.stdout,
        re.DOTALL,
    )
    if not match:
        fail(f"failed to parse scalar string for {h5_path}:{dataset_path}")
    text = match.group("value")
    lines = text.splitlines()
    if len(lines) <= 1:
        return text

    continuation_lines = lines[1:]
    non_empty_continuations = [
        line for line in continuation_lines if line.strip()
    ]
    if non_empty_continuations:
        common_indent = min(
            len(line) - len(line.lstrip(" "))
            for line in non_empty_continuations
        )
        continuation_lines = [
            line[common_indent:] if len(line) >= common_indent else line
            for line in continuation_lines
        ]
    continuation_lines = [
        "" if not line.strip() else line for line in continuation_lines
    ]
    return "\n".join([lines[0]] + continuation_lines)


def read_h5_scalar_int(h5dump, h5_path, dataset_path):
    values = read_h5_ints(h5dump, h5_path, dataset_path)
    if len(values) != 1:
        fail(
            f"expected scalar integer for {h5_path}:{dataset_path}, "
            f"got {values}"
        )
    return values[0]


def read_legacy_counted_float_vector(path):
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        fail(f"legacy counted float vector is empty: {path}")
    try:
        count = int(lines[0].split()[0])
    except ValueError as err:
        fail(
            f"legacy counted float vector header is not an integer in {path}: {err}"
        )
    values = []
    for line in lines[1:]:
        for field in line.split():
            try:
                values.append(float(field))
            except ValueError as err:
                fail(
                    f"legacy counted float vector value is invalid in {path}: {err}"
                )
    if len(values) != count:
        fail(
            f"legacy counted float vector length mismatch for {path}: "
            f"header={count} values={len(values)}"
        )
    return values


def read_legacy_counted_int_vector(path):
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        fail(f"legacy counted int vector is empty: {path}")
    count = int(lines[0].split()[0])
    values = [int(field) for line in lines[1:] for field in line.split()]
    if len(values) != count:
        fail(
            f"legacy counted int vector length mismatch for {path}: "
            f"header={count} values={len(values)}"
        )
    return values


def read_legacy_int_list(path):
    return [
        int(field)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
        for field in line.split()
    ]


def read_legacy_float_list(path):
    return [
        float(field)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
        for field in line.split()
    ]


def read_legacy_binary_float32(path):
    data = path.read_bytes()
    if len(data) % 4 != 0:
        fail(f"legacy binary float32 file size is not divisible by 4: {path}")
    return list(struct.unpack(f"<{len(data) // 4}f", data))


def read_legacy_text_float_rows(path, expected_columns):
    rows = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != expected_columns:
            fail(
                f"legacy text float row in {path} must have "
                f"{expected_columns} columns: {line!r}"
            )
        rows.append([float(field) for field in fields])
    if not rows:
        fail(f"legacy text float file is empty: {path}")
    return rows


def read_legacy_coordinate_file(path):
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        fail(f"legacy coordinate file is empty: {path}")
    atom_count = int(lines[0].split()[0])
    coordinate_rows = []
    for line in lines[1 : 1 + atom_count]:
        fields = line.split()
        if len(fields) != 3:
            fail(
                f"legacy coordinate row must have 3 columns in {path}: {line!r}"
            )
        coordinate_rows.append([float(field) for field in fields])
    if len(coordinate_rows) != atom_count:
        fail(f"legacy coordinate file lacks atom rows: {path}")
    box_lines = lines[1 + atom_count :]
    if len(box_lines) == 1:
        box_fields = box_lines[0].split()
        if len(box_fields) != 6:
            fail(f"legacy coordinate box row must have 6 columns in {path}")
        box_values = [float(field) for field in box_fields]
    elif len(box_lines) == 2:
        box_rows = []
        for line in box_lines:
            fields = line.split()
            if len(fields) != 3:
                fail(
                    f"legacy coordinate box split row must have 3 columns in {path}"
                )
            box_rows.append([float(field) for field in fields])
        box_values = box_rows[0] + box_rows[1]
    else:
        fail(
            f"legacy coordinate file must contain atom count, atom rows, "
            f"and one 6-column or two 3-column box rows: {path}"
        )
    return {
        "atom_count": atom_count,
        "position": [value for row in coordinate_rows for value in row],
        "box": box_values,
    }


def read_legacy_velocity_file(path):
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        fail(f"legacy velocity file is empty: {path}")
    atom_count = int(lines[0].split()[0])
    velocity_rows = []
    for line in lines[1:]:
        fields = line.split()
        if len(fields) != 3:
            fail(f"legacy velocity row must have 3 columns in {path}: {line!r}")
        velocity_rows.append([float(field) for field in fields])
    if len(velocity_rows) != atom_count:
        fail(f"legacy velocity file row count mismatch: {path}")
    return {
        "atom_count": atom_count,
        "velocity": [value for row in velocity_rows for value in row],
    }


def box_lengths_angles_to_edges(row):
    a, b, c, alpha_deg, beta_deg, gamma_deg = row
    alpha = math.radians(alpha_deg)
    beta = math.radians(beta_deg)
    gamma = math.radians(gamma_deg)
    sin_gamma = math.sin(gamma)
    if abs(sin_gamma) < 1.0e-12:
        fail(f"invalid box gamma angle for H5 conversion: {gamma_deg}")
    ax, ay, az = a, 0.0, 0.0
    bx, by, bz = b * math.cos(gamma), b * sin_gamma, 0.0
    cx = c * math.cos(beta)
    cy = c * (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / sin_gamma
    cz_square = c * c - cx * cx - cy * cy
    if cz_square < 0 and abs(cz_square) < 1.0e-8:
        cz_square = 0.0
    if cz_square < 0:
        fail(f"invalid box lengths/angles produce negative cz^2: {row}")
    cz = math.sqrt(cz_square)
    return [ax, ay, az, bx, by, bz, cx, cy, cz]


def read_legacy_bonds(path):
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        fail(f"legacy bond file is empty: {path}")
    count = int(lines[0].split()[0])
    atoms = []
    k = []
    r0 = []
    for line in lines[1:]:
        fields = line.split()
        if len(fields) != 4:
            fail(f"legacy bond row must have 4 fields in {path}: {line!r}")
        atoms.extend([int(fields[0]), int(fields[1])])
        k.append(float(fields[2]))
        r0.append(float(fields[3]))
    if len(k) != count:
        fail(
            f"legacy bond count mismatch for {path}: header={count} rows={len(k)}"
        )
    return {"atoms": atoms, "k": k, "r0": r0}


def read_legacy_angles(path):
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        fail(f"legacy angle file is empty: {path}")
    count = int(lines[0].split()[0])
    atoms = []
    k = []
    theta0 = []
    for line in lines[1:]:
        fields = line.split()
        if len(fields) != 5:
            fail(f"legacy angle row must have 5 fields in {path}: {line!r}")
        atoms.extend([int(fields[0]), int(fields[1]), int(fields[2])])
        k.append(float(fields[3]))
        theta0.append(float(fields[4]))
    if len(k) != count:
        fail(
            f"legacy angle count mismatch for {path}: header={count} rows={len(k)}"
        )
    return {"atoms": atoms, "k": k, "theta0": theta0}


def read_legacy_lj(path):
    tokens = [
        field
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
        for field in line.split()
    ]
    if len(tokens) < 2:
        fail(f"legacy LJ file header is missing: {path}")
    atom_count = int(tokens[0])
    atom_type_count = int(tokens[1])
    pair_count = atom_type_count * (atom_type_count + 1) // 2
    expected_count = 2 + pair_count + pair_count + atom_count
    if len(tokens) != expected_count:
        fail(
            f"legacy LJ token count mismatch for {path}: "
            f"actual={len(tokens)} expected={expected_count}"
        )
    pair_a_begin = 2
    pair_b_begin = pair_a_begin + pair_count
    type_begin = pair_b_begin + pair_count
    return {
        "atom_type_count": atom_type_count,
        "pair_A_12": [
            float(value) for value in tokens[pair_a_begin:pair_b_begin]
        ],
        "pair_B_6": [float(value) for value in tokens[pair_b_begin:type_begin]],
        "type": [int(value) for value in tokens[type_begin:]],
    }


def read_legacy_dihedrals(path):
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        fail(f"legacy dihedral file is empty: {path}")
    count = int(lines[0].split()[0])
    atoms = []
    periodicity = []
    k = []
    phi0 = []
    for line in lines[1:]:
        fields = line.split()
        if len(fields) != 7:
            fail(f"legacy dihedral row must have 7 fields in {path}: {line!r}")
        atoms.extend(
            [int(fields[0]), int(fields[1]), int(fields[2]), int(fields[3])]
        )
        periodicity.append(int(fields[4]))
        k.append(float(fields[5]))
        phi0.append(float(fields[6]))
    if len(k) != count:
        fail(
            f"legacy dihedral count mismatch for {path}: "
            f"header={count} rows={len(k)}"
        )
    return {"atoms": atoms, "periodicity": periodicity, "k": k, "phi0": phi0}


def read_legacy_gb(path):
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        fail(f"legacy GB file is empty: {path}")
    count = int(lines[0].split()[0])
    params = []
    for line in lines[1:]:
        fields = line.split()
        if len(fields) != 2:
            fail(f"legacy GB row must have 2 fields in {path}: {line!r}")
        params.extend([float(fields[0]), float(fields[1])])
    if len(params) != count * 2:
        fail(f"legacy GB count mismatch for {path}: header={count}")
    return params


def read_legacy_urey_bradley(path):
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        fail(f"legacy Urey-Bradley file is empty: {path}")
    count = int(lines[0].split()[0])
    atoms = []
    angle_k = []
    angle_theta0 = []
    bond_k = []
    bond_r0 = []
    for line in lines[1:]:
        fields = line.split()
        if len(fields) != 7:
            fail(
                f"legacy Urey-Bradley row must have 7 fields in {path}: {line!r}"
            )
        atoms.extend([int(fields[0]), int(fields[1]), int(fields[2])])
        angle_k.append(float(fields[3]))
        angle_theta0.append(float(fields[4]))
        bond_k.append(float(fields[5]))
        bond_r0.append(float(fields[6]))
    if len(angle_k) != count:
        fail(
            f"legacy Urey-Bradley count mismatch for {path}: "
            f"header={count} rows={len(angle_k)}"
        )
    return {
        "atoms": atoms,
        "angle_k": angle_k,
        "angle_theta0": angle_theta0,
        "bond_k": bond_k,
        "bond_r0": bond_r0,
    }


def read_legacy_nb14_extra(path):
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        fail(f"legacy NB14 extra file is empty: {path}")
    count = int(lines[0].split()[0])
    atoms = []
    params = []
    for line in lines[1:]:
        fields = line.split()
        if len(fields) != 5:
            fail(
                f"legacy NB14 extra row must have 5 fields in {path}: {line!r}"
            )
        atoms.extend([int(fields[0]), int(fields[1])])
        params.extend([float(fields[2]), float(fields[3]), float(fields[4])])
    if len(atoms) != count * 2:
        fail(
            f"legacy NB14 extra count mismatch for {path}: "
            f"header={count} rows={len(atoms) // 2}"
        )
    return {"atoms": atoms, "params": params}


def read_legacy_constraints(path):
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        fail(f"legacy constrain file is empty: {path}")
    count = int(lines[0].split()[0])
    atoms = []
    r0 = []
    for line in lines[1:]:
        fields = line.split()
        if len(fields) != 3:
            fail(f"legacy constrain row must have 3 fields in {path}: {line!r}")
        atoms.extend([int(fields[0]), int(fields[1])])
        r0.append(float(fields[2]))
    if len(r0) != count:
        fail(
            f"legacy constrain count mismatch for {path}: "
            f"header={count} rows={len(r0)}"
        )
    return {"atoms": atoms, "r0": r0}


def read_legacy_float_matrix(
    path, expected_columns=None, skip_first_line=False
):
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if skip_first_line:
        lines = lines[1:]
    values = []
    row_count = 0
    for line in lines:
        fields = line.split()
        if expected_columns is not None and len(fields) != expected_columns:
            fail(
                f"legacy float matrix row in {path} must have "
                f"{expected_columns} columns: {line!r}"
            )
        values.extend(float(field) for field in fields)
        row_count += 1
    if row_count == 0:
        fail(f"legacy float matrix is empty: {path}")
    return {"row_count": row_count, "values": values}


def read_legacy_counted_xyz(path):
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        fail(f"legacy counted xyz file is empty: {path}")
    count = int(lines[0].split()[0])
    rows = lines[1:]
    if len(rows) != count:
        fail(
            f"legacy counted xyz row count mismatch for {path}: "
            f"header={count} rows={len(rows)}"
        )
    values = []
    for line in rows:
        fields = line.split()
        if len(fields) != 3:
            fail(
                f"legacy counted xyz row must have 3 columns in {path}: {line!r}"
            )
        values.extend(float(field) for field in fields)
    return {"count": count, "values": values}


def read_legacy_config_sections(path):
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    sections = []
    index = 0
    while index < len(lines):
        section_name = lines[index]
        index += 1
        if index >= len(lines) or lines[index] != "{":
            fail(
                f"legacy config section lacks opening brace in {path}: {section_name}"
            )
        index += 1
        entries = []
        while index < len(lines) and lines[index] != "}":
            if "=" not in lines[index]:
                fail(
                    f"legacy config line lacks '=' in {path}: {lines[index]!r}"
                )
            key, value = lines[index].split("=", 1)
            entries.append((key.strip(), value.strip()))
            index += 1
        if index >= len(lines) or lines[index] != "}":
            fail(
                f"legacy config section lacks closing brace in {path}: {section_name}"
            )
        index += 1
        sections.append({"name": section_name, "entries": entries})
    if not sections:
        fail(f"legacy config file is empty: {path}")
    return sections


def read_legacy_meta_edge(path):
    rows = read_legacy_text_float_rows(path, 5)
    return {
        "count": len(rows),
        "ndim": 2,
        "coordinate": [value for row in rows for value in row[:2]],
        "normalization": [row[2] for row in rows],
        "force": [value for row in rows for value in row[3:]],
    }


def read_legacy_meta_scatter(path):
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(lines) < 4:
        fail(f"legacy meta scatter file is too short: {path}")
    axis_rows = []
    header_index = None
    for index, line in enumerate(lines[1:], start=1):
        fields = line.split()
        if len(fields) == 2 and all(
            field.lstrip("-").isdigit() for field in fields
        ):
            header_index = index
            break
        if len(fields) != 3:
            fail(f"legacy meta scatter axis row has unexpected width: {path}")
        axis_rows.append([float(field) for field in fields])
    if header_index is None:
        fail(f"legacy meta scatter header lacks scatter size: {path}")
    header = lines[header_index].split()
    scatter_size = int(header[0])
    ndim = len(axis_rows)
    rows = lines[header_index + 1 :]
    if len(rows) != scatter_size:
        fail(
            f"legacy meta scatter row count mismatch for {path}: "
            f"header={scatter_size} rows={len(rows)}"
        )
    coordinates = []
    force = []
    values = []
    for line in rows:
        fields = line.split()
        if len(fields) != 2 * ndim + 1:
            fail(
                f"legacy meta scatter row has unexpected width in {path}: {line!r}"
            )
        coordinates.extend(float(field) for field in fields[:ndim])
        force.extend(float(field) for field in fields[ndim : 2 * ndim])
        values.append(float(fields[2 * ndim]))
    return {
        "scatter_size": scatter_size,
        "ndim": ndim,
        "axis_min": [row[0] for row in axis_rows],
        "axis_max": [row[1] for row in axis_rows],
        "axis_delta": [row[2] for row in axis_rows],
        "coordinate": coordinates,
        "force": force,
        "value": values,
    }


def read_legacy_meta_potential(path):
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(lines) < 4:
        fail(f"legacy meta potential file is too short: {path}")
    axis_rows = []
    header_index = None
    for index, line in enumerate(lines[1:], start=1):
        fields = line.split()
        if len(fields) >= 2 and all(
            field.lstrip("-").isdigit() for field in fields
        ):
            header_index = index
            break
        if len(fields) != 3:
            fail(f"legacy meta potential axis row has unexpected width: {path}")
        axis_rows.append([float(field) for field in fields])
    if header_index is None:
        fail(f"legacy meta potential grid header is missing: {path}")
    ndim = len(axis_rows)
    header = lines[header_index].split()
    if len(header) != ndim + 1:
        fail(f"legacy meta potential grid header width mismatch: {path}")
    grid = [int(field) for field in header[:ndim]]
    value_count = int(header[ndim])
    rows = lines[header_index + 1 :]
    if len(rows) != value_count:
        fail(
            f"legacy meta potential row count mismatch for {path}: "
            f"header={value_count} rows={len(rows)}"
        )
    coordinates = []
    force = []
    values = []
    for line in rows:
        fields = line.split()
        if len(fields) != 2 * ndim + 2:
            fail(
                f"legacy meta potential row has unexpected width in {path}: {line!r}"
            )
        coordinates.extend(float(field) for field in fields[:ndim])
        force.extend(float(field) for field in fields[ndim + 1 : 2 * ndim + 1])
        values.append(float(fields[2 * ndim + 1]))
    return {
        "ndim": ndim,
        "grid": grid,
        "axis_min": [row[0] for row in axis_rows],
        "axis_max": [row[1] for row in axis_rows],
        "axis_delta": [row[2] for row in axis_rows],
        "coordinate": coordinates,
        "force": force,
        "value": values,
    }


def read_legacy_exclusions(path):
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        fail(f"legacy exclusion file is empty: {path}")
    header = lines[0].split()
    if len(header) < 1:
        fail(f"legacy exclusion header is missing atom count: {path}")
    atom_count = int(header[0])
    if len(lines[1:]) != atom_count:
        fail(
            f"legacy exclusion row count mismatch for {path}: "
            f"header={atom_count} rows={len(lines[1:])}"
        )
    offsets = [0]
    excluded = []
    for line in lines[1:]:
        fields = line.split()
        row_count = int(fields[0])
        row_values = [int(field) for field in fields[1:]]
        if len(row_values) != row_count:
            fail(
                f"legacy exclusion row count mismatch for {path}: row={line!r}"
            )
        excluded.extend(row_values)
        offsets.append(len(excluded))
    return {"atom_count": atom_count, "offset": offsets, "list": excluded}


def read_legacy_residue(path):
    tokens = [
        int(field)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
        for field in line.split()
    ]
    if len(tokens) < 2:
        fail(f"legacy residue file is too short: {path}")
    atom_count = tokens[0]
    residue_count = tokens[1]
    residue_sizes = tokens[2:]
    if len(residue_sizes) != residue_count:
        fail(
            f"legacy residue count mismatch for {path}: "
            f"header={residue_count} sizes={len(residue_sizes)}"
        )
    if sum(residue_sizes) != atom_count:
        fail(
            f"legacy residue atom count mismatch for {path}: "
            f"header={atom_count} size_sum={sum(residue_sizes)}"
        )

    residue_index = []
    atom_offset = [0]
    for residue_id, size in enumerate(residue_sizes):
        if size < 0:
            fail(f"legacy residue size must be non-negative in {path}: {size}")
        residue_index.extend([residue_id] * size)
        atom_offset.append(len(residue_index))
    return {"residue_index": residue_index, "atom_offset": atom_offset}


def read_legacy_virtual_atoms(path):
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    records = []
    for line in lines:
        fields = line.split()
        if len(fields) < 5:
            fail(f"legacy virtual atom row is too short in {path}: {line!r}")
        virtual_type = int(fields[0])
        atom = int(fields[1])
        from_atoms = [int(fields[2]), int(fields[3]), int(fields[4])]
        parameters = [float(field) for field in fields[5:]]
        records.append(
            {
                "atom": atom,
                "type": virtual_type,
                "from": from_atoms,
                "parameter": parameters,
            }
        )
    if not records:
        fail(f"legacy virtual atom file is empty: {path}")
    from_offset = [0]
    parameter_offset = [0]
    all_from = []
    all_parameters = []
    for record in records:
        all_from.extend(record["from"])
        all_parameters.extend(record["parameter"])
        from_offset.append(len(all_from))
        parameter_offset.append(len(all_parameters))
    return {
        "atom": [record["atom"] for record in records],
        "type": [record["type"] for record in records],
        "from": all_from,
        "from_offset": from_offset,
        "parameter": all_parameters,
        "parameter_offset": parameter_offset,
    }


def read_legacy_lj_soft_core(path, subsystem_path):
    tokens = [
        field
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
        for field in line.split()
    ]
    if len(tokens) < 3:
        fail(f"legacy LJ soft-core header is missing: {path}")
    atom_count = int(tokens[0])
    atom_type_count_a = int(tokens[1])
    atom_type_count_b = int(tokens[2])
    pair_count_a = atom_type_count_a * (atom_type_count_a + 1) // 2
    pair_count_b = atom_type_count_b * (atom_type_count_b + 1) // 2
    mixed_count = atom_type_count_a * atom_type_count_b
    cursor = 3
    pair_aa = [float(value) for value in tokens[cursor : cursor + pair_count_a]]
    cursor += pair_count_a
    pair_ab = [float(value) for value in tokens[cursor : cursor + mixed_count]]
    cursor += mixed_count
    pair_ba = [float(value) for value in tokens[cursor : cursor + mixed_count]]
    cursor += mixed_count
    pair_bb = [float(value) for value in tokens[cursor : cursor + pair_count_b]]
    cursor += pair_count_b
    atom_type_a = [int(value) for value in tokens[cursor : cursor + atom_count]]
    cursor += atom_count
    atom_type_b = [int(value) for value in tokens[cursor : cursor + atom_count]]
    cursor += atom_count
    if cursor != len(tokens):
        fail(
            f"legacy LJ soft-core token count mismatch for {path}: "
            f"consumed={cursor} actual={len(tokens)}"
        )
    return {
        "atom_type_count_A": atom_type_count_a,
        "atom_type_count_B": atom_type_count_b,
        "pair_AA": pair_aa,
        "pair_AB": pair_ab,
        "pair_BA": pair_ba,
        "pair_BB": pair_bb,
        "atom_type_A": atom_type_a,
        "atom_type_B": atom_type_b,
        "subsys_division": read_legacy_counted_int_vector(subsystem_path),
    }


def read_legacy_cmap(path):
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(lines) < 4:
        fail(f"legacy CMAP file is too short: {path}")
    header = lines[0].split()
    if len(header) != 2:
        fail(f"legacy CMAP header must contain type and entry counts: {path}")
    type_count = int(header[0])
    entry_count = int(header[1])
    resolution = [int(lines[1].split()[0])]
    grid_size = resolution[0] * resolution[0]
    grid_values = []
    cursor = 2
    while len(grid_values) < grid_size:
        grid_values.extend(float(field) for field in lines[cursor].split())
        cursor += 1
    if len(grid_values) != grid_size:
        fail(f"legacy CMAP grid size mismatch for {path}")
    entry_lines = lines[cursor:]
    if len(entry_lines) != entry_count:
        fail(
            f"legacy CMAP entry count mismatch for {path}: "
            f"header={entry_count} rows={len(entry_lines)}"
        )
    atoms = []
    cmap_type = []
    for line in entry_lines:
        fields = line.split()
        if len(fields) != 6:
            fail(f"legacy CMAP entry must have 6 fields in {path}: {line!r}")
        atoms.extend(int(field) for field in fields[:5])
        cmap_type.append(int(fields[5]))
    return {
        "type_count": type_count,
        "atoms": atoms,
        "type": cmap_type,
        "resolution": resolution,
        "grid_value": grid_values,
    }


def read_legacy_manybody_pair_triple(
    path, pair_parameter_count, triple_parameter_count
):
    tokens = [
        field
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
        for field in line.split()
    ]
    if len(tokens) < 2:
        fail(f"legacy manybody pair/triple file is too short: {path}")
    atom_count = int(tokens[0])
    atom_type_count = int(tokens[1])
    pair_count = atom_type_count * (atom_type_count + 1) // 2
    triple_count = atom_type_count * atom_type_count * atom_type_count
    cursor = 2

    pair_type = []
    pair_parameters = []
    for _ in range(pair_count):
        if cursor + 2 + pair_parameter_count > len(tokens):
            fail(f"legacy manybody pair section is truncated: {path}")
        pair_type.extend([int(tokens[cursor]), int(tokens[cursor + 1])])
        cursor += 2
        pair_parameters.extend(
            float(value)
            for value in tokens[cursor : cursor + pair_parameter_count]
        )
        cursor += pair_parameter_count

    triple_type = []
    triple_parameters = []
    for _ in range(triple_count):
        if cursor + 3 + triple_parameter_count > len(tokens):
            fail(f"legacy manybody triple section is truncated: {path}")
        triple_type.extend(
            [
                int(tokens[cursor]),
                int(tokens[cursor + 1]),
                int(tokens[cursor + 2]),
            ]
        )
        cursor += 3
        triple_parameters.extend(
            float(value)
            for value in tokens[cursor : cursor + triple_parameter_count]
        )
        cursor += triple_parameter_count

    atom_type = [int(value) for value in tokens[cursor:]]
    if len(atom_type) != atom_count:
        fail(
            f"legacy manybody atom type count mismatch for {path}: "
            f"header={atom_count} atom_types={len(atom_type)}"
        )
    return {
        "atom_type_count": atom_type_count,
        "atom_type": atom_type,
        "pair_type": pair_type,
        "pair_parameters": pair_parameters,
        "triple_type": triple_type,
        "triple_parameters": triple_parameters,
    }


def read_legacy_tersoff(path):
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(lines) < 4:
        fail(f"legacy TERSOFF file is too short: {path}")
    header = lines[0].split()
    if len(header) != 2:
        fail(f"legacy TERSOFF header must contain atom and type counts: {path}")
    atom_count = int(header[0])
    atom_type_count = int(header[1])
    type_names = []
    cursor = 1
    while len(type_names) < atom_type_count:
        type_names.extend(lines[cursor].split())
        cursor += 1
    if len(type_names) != atom_type_count:
        fail(f"legacy TERSOFF type name count mismatch: {path}")

    atom_type = [int(value) for value in lines[-1].split()]
    if len(atom_type) != atom_count:
        fail(
            f"legacy TERSOFF atom type count mismatch for {path}: "
            f"header={atom_count} atom_types={len(atom_type)}"
        )

    entry_lines = lines[cursor:-1]
    if not entry_lines:
        fail(f"legacy TERSOFF has no entries: {path}")
    entry_type_names = []
    entry_type = []
    parameters_raw = []
    type_index = {name: index for index, name in enumerate(type_names)}
    for line in entry_lines:
        fields = line.split()
        if len(fields) != 17:
            fail(
                f"legacy TERSOFF entry must have 17 fields in {path}: {line!r}"
            )
        names = fields[:3]
        entry_type_names.extend(names)
        try:
            entry_type.extend(type_index[name] for name in names)
        except KeyError as err:
            fail(
                f"legacy TERSOFF entry references unknown type in {path}: {err}"
            )
        parameters_raw.extend(float(value) for value in fields[3:])

    return {
        "atom_type_count": atom_type_count,
        "atom_type": atom_type,
        "type_name": type_names,
        "map": list(range(atom_type_count)),
        "entry_count": len(entry_lines),
        "entry_type": entry_type,
        "entry_type_name": entry_type_names,
        "parameters_raw": parameters_raw,
    }


def read_legacy_eam(eam_path, atom_type_path):
    lines = [
        line.strip()
        for line in eam_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(lines) < 6:
        fail(f"legacy EAM funcfl file is too short: {eam_path}")
    atomic_fields = lines[1].split()
    if len(atomic_fields) != 4:
        fail(f"legacy EAM atomic header must have 4 fields: {eam_path}")
    grid_fields = lines[2].split()
    if len(grid_fields) != 5:
        fail(f"legacy EAM grid header must have 5 fields: {eam_path}")
    nrho = int(grid_fields[0])
    nr = int(grid_fields[2])
    expected_rows = 3 + 3
    if len(lines) != expected_rows:
        fail(f"legacy EAM fixture must contain three data rows: {eam_path}")
    embed_raw_ev = [float(value) for value in lines[3].split()]
    funcfl_z = [float(value) for value in lines[4].split()]
    electron_density = [float(value) for value in lines[5].split()]
    if len(embed_raw_ev) != nrho:
        fail(f"legacy EAM embed length mismatch: {eam_path}")
    if len(funcfl_z) != nr or len(electron_density) != nr:
        fail(f"legacy EAM radial data length mismatch: {eam_path}")
    return {
        "format": "funcfl",
        "atom_type_count": 1,
        "atom_type": read_legacy_int_list(atom_type_path),
        "atomic_number": [int(atomic_fields[0])],
        "mass": [float(atomic_fields[1])],
        "lattice_constant": [float(atomic_fields[2])],
        "lattice_type": [atomic_fields[3]],
        "nrho": nrho,
        "drho": float(grid_fields[1]),
        "nr": nr,
        "dr": float(grid_fields[3]),
        "cut": float(grid_fields[4]),
        "embed_raw_ev": embed_raw_ev,
        "funcfl_z": funcfl_z,
        "electron_density": electron_density,
    }


def split_reaxff_data_label(line):
    if "!" not in line:
        return line.strip(), ""
    data, label = line.split("!", 1)
    return data.strip(), label.strip()


def parse_reaxff_count_line(line, path):
    data, label = split_reaxff_data_label(line)
    fields = data.split()
    if len(fields) != 1:
        fail(
            f"ReaxFF count line must start with one integer in {path}: {line!r}"
        )
    return int(fields[0]), label


def parse_reaxff_numeric_fields(line):
    data, label = split_reaxff_data_label(line)
    return data.split(), label


def read_legacy_reaxff_types(path):
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        fail(f"legacy ReaxFF type file is empty: {path}")
    count = int(lines[0].split()[0])
    names = [line.split()[0] for line in lines[1:]]
    if len(names) != count:
        fail(
            f"legacy ReaxFF type count mismatch for {path}: "
            f"header={count} names={len(names)}"
        )
    return {"count": count, "name": names}


def read_legacy_reaxff(path):
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(lines) < 3:
        fail(f"legacy ReaxFF parameter file is too short: {path}")
    cursor = 0
    header = lines[cursor]
    cursor += 1

    general_count, general_count_label = parse_reaxff_count_line(
        lines[cursor], path
    )
    cursor += 1
    general_value = []
    general_label = []
    for _ in range(general_count):
        fields, label = parse_reaxff_numeric_fields(lines[cursor])
        if len(fields) != 1:
            fail(
                f"legacy ReaxFF general row must have one value: {lines[cursor]!r}"
            )
        general_value.append(float(fields[0]))
        general_label.append(label)
        cursor += 1

    atom_count, atom_count_label = parse_reaxff_count_line(lines[cursor], path)
    cursor += 1
    atom_header = lines[cursor : cursor + 3]
    cursor += 3
    atom_type_name = []
    atom_value = []
    atom_line_label = []
    atom_value_offset = [0]
    atom_line_value_offset = [0]
    for _ in range(atom_count):
        atom_fields, atom_label = parse_reaxff_numeric_fields(lines[cursor])
        if len(atom_fields) != 9:
            fail(
                f"legacy ReaxFF atom first row must have type plus 8 values: {path}"
            )
        atom_type_name.append(atom_fields[0])
        atom_value.extend(float(value) for value in atom_fields[1:])
        atom_line_label.append(atom_label)
        atom_line_value_offset.append(len(atom_value))
        cursor += 1
        for _ in range(3):
            fields, label = parse_reaxff_numeric_fields(lines[cursor])
            if len(fields) != 8:
                fail(
                    f"legacy ReaxFF atom continuation row must have 8 values: {path}"
                )
            atom_value.extend(float(value) for value in fields)
            atom_line_label.append(label)
            atom_line_value_offset.append(len(atom_value))
            cursor += 1
        atom_value_offset.append(len(atom_value))

    bond_count, bond_count_label = parse_reaxff_count_line(lines[cursor], path)
    cursor += 1
    bond_header = [lines[cursor]]
    cursor += 1
    bond_type = []
    bond_value = []
    bond_value_offset = [0]
    bond_line_label = []
    bond_line_value_offset = [0]
    for _ in range(bond_count):
        fields, label = parse_reaxff_numeric_fields(lines[cursor])
        if len(fields) != 10:
            fail(
                f"legacy ReaxFF bond first row must have 2 types plus 8 values: {path}"
            )
        bond_type.extend(int(value) for value in fields[:2])
        bond_value.extend(float(value) for value in fields[2:])
        bond_line_label.append(label)
        bond_line_value_offset.append(len(bond_value))
        cursor += 1
        fields, label = parse_reaxff_numeric_fields(lines[cursor])
        if len(fields) != 8:
            fail(
                f"legacy ReaxFF bond continuation row must have 8 values: {path}"
            )
        bond_value.extend(float(value) for value in fields)
        bond_line_label.append(label)
        bond_line_value_offset.append(len(bond_value))
        cursor += 1
        bond_value_offset.append(len(bond_value))

    off_diag_count, off_diag_count_label = parse_reaxff_count_line(
        lines[cursor], path
    )
    cursor += 1
    off_diag_type = []
    off_diag_value = []
    for _ in range(off_diag_count):
        fields, _ = parse_reaxff_numeric_fields(lines[cursor])
        if len(fields) != 8:
            fail(f"legacy ReaxFF off-diagonal row must have 8 fields: {path}")
        off_diag_type.extend(int(value) for value in fields[:2])
        off_diag_value.extend(float(value) for value in fields[2:])
        cursor += 1

    angle_count, angle_count_label = parse_reaxff_count_line(
        lines[cursor], path
    )
    cursor += 1
    angle_type = []
    angle_value = []
    for _ in range(angle_count):
        fields, _ = parse_reaxff_numeric_fields(lines[cursor])
        if len(fields) != 10:
            fail(f"legacy ReaxFF angle row must have 10 fields: {path}")
        angle_type.extend(int(value) for value in fields[:3])
        angle_value.extend(float(value) for value in fields[3:])
        cursor += 1

    torsion_count, torsion_count_label = parse_reaxff_count_line(
        lines[cursor], path
    )
    cursor += 1
    torsion_type = []
    torsion_value = []
    for _ in range(torsion_count):
        fields, _ = parse_reaxff_numeric_fields(lines[cursor])
        if len(fields) != 11:
            fail(f"legacy ReaxFF torsion row must have 11 fields: {path}")
        torsion_type.extend(int(value) for value in fields[:4])
        torsion_value.extend(float(value) for value in fields[4:])
        cursor += 1

    hydrogen_bond_count, hydrogen_bond_count_label = parse_reaxff_count_line(
        lines[cursor], path
    )
    cursor += 1
    hydrogen_bond_type = []
    hydrogen_bond_value = []
    for _ in range(hydrogen_bond_count):
        fields, _ = parse_reaxff_numeric_fields(lines[cursor])
        if len(fields) != 7:
            fail(f"legacy ReaxFF hydrogen-bond row must have 7 fields: {path}")
        hydrogen_bond_type.extend(int(value) for value in fields[:3])
        hydrogen_bond_value.extend(float(value) for value in fields[3:])
        cursor += 1

    if cursor != len(lines):
        fail(
            f"legacy ReaxFF parser did not consume all lines in {path}: "
            f"cursor={cursor} lines={len(lines)}"
        )

    return {
        "header": header,
        "general": {
            "count_label": general_count_label,
            "label": general_label,
            "value": general_value,
        },
        "atom": {
            "count": atom_count,
            "count_label": atom_count_label,
            "header": atom_header,
            "type_name": atom_type_name,
            "value": atom_value,
            "value_offset": atom_value_offset,
            "line_label": atom_line_label,
            "line_value_offset": atom_line_value_offset,
        },
        "bond": {
            "count": bond_count,
            "count_label": bond_count_label,
            "header": bond_header,
            "type": bond_type,
            "value": bond_value,
            "value_offset": bond_value_offset,
            "line_label": bond_line_label,
            "line_value_offset": bond_line_value_offset,
        },
        "off_diagonal": {
            "count": off_diag_count,
            "count_label": off_diag_count_label,
            "type": off_diag_type,
            "value": off_diag_value,
        },
        "angle": {
            "count": angle_count,
            "count_label": angle_count_label,
            "type": angle_type,
            "value": angle_value,
        },
        "torsion": {
            "count": torsion_count,
            "count_label": torsion_count_label,
            "type": torsion_type,
            "value": torsion_value,
        },
        "hydrogen_bond": {
            "count": hydrogen_bond_count,
            "count_label": hydrogen_bond_count_label,
            "type": hydrogen_bond_type,
            "value": hydrogen_bond_value,
        },
    }


def parse_custom_force_descriptor(path):
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        fail(f"custom force descriptor is empty: {path}")
    header = re.fullmatch(
        r"\[\[\[\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\]\]\]", lines[0]
    )
    if not header:
        fail(f"custom force descriptor has invalid header: {path}")

    sections = {}
    current = None
    for line in lines[1:]:
        section = re.fullmatch(
            r"\[\[\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\]\]", line
        )
        if section:
            current = section.group("name")
            sections[current] = []
            continue
        if current is None:
            fail(
                f"custom force descriptor line is outside a section in {path}: {line!r}"
            )
        sections[current].append(line)

    parameter_lines = sections.get("parameters", [])
    if len(parameter_lines) != 1:
        fail(f"custom force descriptor must have one parameters line: {path}")
    parameter_types = []
    parameter_names = []
    for parameter in parameter_lines[0].split(","):
        fields = parameter.strip().split()
        if len(fields) != 2:
            fail(f"invalid custom force parameter in {path}: {parameter!r}")
        parameter_types.append(fields[0])
        parameter_names.append(fields[1])

    return {
        "name": header.group("name"),
        "potential": "\n".join(sections.get("potential", [])),
        "parameter_type": parameter_types,
        "parameter_name": parameter_names,
        "with_ele": "\n".join(sections.get("with_ele", [])).strip().lower(),
        "connected_atoms": "\n".join(
            sections.get("connected_atoms", [])
        ).strip(),
        "constrain_distance": "\n".join(
            sections.get("constrain_distance", [])
        ).strip(),
    }


def parse_named_section_descriptor(path):
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        fail(f"named descriptor is empty: {path}")
    header = re.fullmatch(
        r"\[\[\[\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\]\]\]", lines[0]
    )
    if not header:
        fail(f"named descriptor has invalid header: {path}")
    sections = {}
    current = None
    for line in lines[1:]:
        section = re.fullmatch(
            r"\[\[\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\]\]", line
        )
        if section:
            current = section.group("name")
            sections[current] = []
            continue
        if current is None:
            fail(
                f"named descriptor line is outside a section in {path}: {line!r}"
            )
        sections[current].append(line)
    return {"name": header.group("name"), "sections": sections}


def read_legacy_custom_pair(path):
    tokens = [
        field
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
        for field in line.split()
    ]
    if len(tokens) < 4:
        fail(f"legacy custom pair file is too short: {path}")
    atom_count = int(tokens[0])
    type_count = int(tokens[1])
    pair_count = type_count * (type_count + 1) // 2
    remaining = len(tokens) - 2 - atom_count
    if remaining <= 0 or remaining % pair_count != 0:
        fail(f"legacy custom pair token count mismatch: {path}")
    parameter_count = remaining // pair_count
    value_end = 2 + pair_count * parameter_count
    return {
        "atom_count": atom_count,
        "type_count": type_count,
        "pair_count": pair_count,
        "value": [float(value) for value in tokens[2:value_end]],
        "atom_type": [int(value) for value in tokens[value_end:]],
    }


def read_legacy_custom_bond(path):
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(lines) < 2:
        fail(f"legacy custom bond file is too short: {path}")
    item_count = int(lines[0].split()[0])
    rows = lines[1:]
    if len(rows) != item_count:
        fail(
            f"legacy custom bond row count mismatch for {path}: "
            f"header={item_count} rows={len(rows)}"
        )
    values = []
    int_values = []
    for line in rows:
        fields = line.split()
        if len(fields) != 4:
            fail(
                f"legacy custom bond row must have 4 fields in {path}: {line!r}"
            )
        int_values.extend([int(fields[0]), int(fields[1]), 0, 0])
        values.extend([float(field) for field in fields])
    return {"item_count": item_count, "value": values, "int_value": int_values}


def compare_float_vectors(label, actual, expected, tolerance=1.0e-6):
    if len(actual) != len(expected):
        fail(
            f"{label} length mismatch: actual={len(actual)} "
            f"expected={len(expected)}"
        )
    for index, (actual_value, expected_value) in enumerate(
        zip(actual, expected)
    ):
        if abs(actual_value - expected_value) > tolerance:
            fail(
                f"{label} value mismatch at index {index}: "
                f"actual={actual_value} expected={expected_value}"
            )


def compare_int_vectors(label, actual, expected):
    if actual != expected:
        fail(f"{label} mismatch: actual={actual} expected={expected}")


def compare_counted_float_file_to_h5(h5dump, legacy_path, h5_path, h5_dataset):
    expected = read_legacy_counted_float_vector(legacy_path)
    actual = read_h5_floats(h5dump, h5_path, h5_dataset)
    compare_float_vectors(
        f"{h5_path}:{h5_dataset} against {legacy_path}", actual, expected
    )


def compare_bonds_to_h5(h5dump, legacy_path, h5_path):
    expected = read_legacy_bonds(legacy_path)
    compare_int_vectors(
        f"{h5_path}:/forcefield/bond/atoms against {legacy_path}",
        read_h5_ints(h5dump, h5_path, "/forcefield/bond/atoms"),
        expected["atoms"],
    )
    compare_float_vectors(
        f"{h5_path}:/forcefield/bond/k against {legacy_path}",
        read_h5_floats(h5dump, h5_path, "/forcefield/bond/k"),
        expected["k"],
    )
    compare_float_vectors(
        f"{h5_path}:/forcefield/bond/r0 against {legacy_path}",
        read_h5_floats(h5dump, h5_path, "/forcefield/bond/r0"),
        expected["r0"],
    )


def compare_angles_to_h5(h5dump, legacy_path, h5_path):
    expected = read_legacy_angles(legacy_path)
    compare_int_vectors(
        f"{h5_path}:/forcefield/angle/atoms against {legacy_path}",
        read_h5_ints(h5dump, h5_path, "/forcefield/angle/atoms"),
        expected["atoms"],
    )
    compare_float_vectors(
        f"{h5_path}:/forcefield/angle/k against {legacy_path}",
        read_h5_floats(h5dump, h5_path, "/forcefield/angle/k"),
        expected["k"],
    )
    compare_float_vectors(
        f"{h5_path}:/forcefield/angle/theta0 against {legacy_path}",
        read_h5_floats(h5dump, h5_path, "/forcefield/angle/theta0"),
        expected["theta0"],
    )


def compare_lj_to_h5(h5dump, legacy_path, h5_path):
    expected = read_legacy_lj(legacy_path)
    actual_atom_type_count = read_h5_scalar_int(
        h5dump, h5_path, "/forcefield/lj/atom_type_count"
    )
    if actual_atom_type_count != expected["atom_type_count"]:
        fail(
            f"{h5_path}:/forcefield/lj/atom_type_count mismatch against "
            f"{legacy_path}: actual={actual_atom_type_count} "
            f"expected={expected['atom_type_count']}"
        )
    compare_float_vectors(
        f"{h5_path}:/forcefield/lj/pair_A_12 against {legacy_path}",
        read_h5_floats(h5dump, h5_path, "/forcefield/lj/pair_A_12"),
        expected["pair_A_12"],
    )
    compare_float_vectors(
        f"{h5_path}:/forcefield/lj/pair_B_6 against {legacy_path}",
        read_h5_floats(h5dump, h5_path, "/forcefield/lj/pair_B_6"),
        expected["pair_B_6"],
    )
    compare_int_vectors(
        f"{h5_path}:/forcefield/lj/type against {legacy_path}",
        read_h5_ints(h5dump, h5_path, "/forcefield/lj/type"),
        expected["type"],
    )


def compare_dihedrals_to_h5(h5dump, legacy_path, h5_path):
    expected = read_legacy_dihedrals(legacy_path)
    compare_int_vectors(
        f"{h5_path}:/forcefield/dihedral/atoms against {legacy_path}",
        read_h5_ints(h5dump, h5_path, "/forcefield/dihedral/atoms"),
        expected["atoms"],
    )
    compare_float_vectors(
        f"{h5_path}:/forcefield/dihedral/k against {legacy_path}",
        read_h5_floats(h5dump, h5_path, "/forcefield/dihedral/k"),
        expected["k"],
    )
    compare_int_vectors(
        f"{h5_path}:/forcefield/dihedral/periodicity against {legacy_path}",
        read_h5_ints(h5dump, h5_path, "/forcefield/dihedral/periodicity"),
        expected["periodicity"],
    )
    compare_float_vectors(
        f"{h5_path}:/forcefield/dihedral/phi0 against {legacy_path}",
        read_h5_floats(h5dump, h5_path, "/forcefield/dihedral/phi0"),
        expected["phi0"],
    )


def compare_gb_to_h5(h5dump, legacy_path, h5_path):
    compare_float_vectors(
        f"{h5_path}:/forcefield/gb/params against {legacy_path}",
        read_h5_floats(h5dump, h5_path, "/forcefield/gb/params"),
        read_legacy_gb(legacy_path),
    )


def compare_urey_bradley_to_h5(h5dump, legacy_path, h5_path):
    expected = read_legacy_urey_bradley(legacy_path)
    compare_int_vectors(
        f"{h5_path}:/forcefield/urey_bradley/atoms against {legacy_path}",
        read_h5_ints(h5dump, h5_path, "/forcefield/urey_bradley/atoms"),
        expected["atoms"],
    )
    for key in ("angle_k", "angle_theta0", "bond_k", "bond_r0"):
        compare_float_vectors(
            f"{h5_path}:/forcefield/urey_bradley/{key} against {legacy_path}",
            read_h5_floats(h5dump, h5_path, f"/forcefield/urey_bradley/{key}"),
            expected[key],
        )


def compare_nb14_extra_to_h5(h5dump, legacy_path, h5_path):
    expected = read_legacy_nb14_extra(legacy_path)
    compare_int_vectors(
        f"{h5_path}:/forcefield/nb14_extra/atoms against {legacy_path}",
        read_h5_ints(h5dump, h5_path, "/forcefield/nb14_extra/atoms"),
        expected["atoms"],
    )
    compare_float_vectors(
        f"{h5_path}:/forcefield/nb14_extra/params against {legacy_path}",
        read_h5_floats(h5dump, h5_path, "/forcefield/nb14_extra/params"),
        expected["params"],
    )


def compare_exclusions_to_h5(h5dump, legacy_path, h5_path):
    expected = read_legacy_exclusions(legacy_path)
    actual_atom_count = read_h5_scalar_int(
        h5dump, h5_path, "/topology/atom_count"
    )
    if actual_atom_count != expected["atom_count"]:
        fail(
            f"{h5_path}:/topology/atom_count mismatch against {legacy_path}: "
            f"actual={actual_atom_count} expected={expected['atom_count']}"
        )
    compare_int_vectors(
        f"{h5_path}:/topology/exclusions/offset against {legacy_path}",
        read_h5_ints(h5dump, h5_path, "/topology/exclusions/offset"),
        expected["offset"],
    )
    compare_int_vectors(
        f"{h5_path}:/topology/exclusions/list against {legacy_path}",
        read_h5_ints(h5dump, h5_path, "/topology/exclusions/list"),
        expected["list"],
    )


def compare_residue_to_h5(h5dump, legacy_path, h5_path):
    expected = read_legacy_residue(legacy_path)
    compare_int_vectors(
        f"{h5_path}:/atoms/residue_index against {legacy_path}",
        read_h5_ints(h5dump, h5_path, "/atoms/residue_index"),
        expected["residue_index"],
    )
    compare_int_vectors(
        f"{h5_path}:/residues/atom_offset against {legacy_path}",
        read_h5_ints(h5dump, h5_path, "/residues/atom_offset"),
        expected["atom_offset"],
    )


def compare_virtual_atoms_to_h5(h5dump, legacy_path, h5_path):
    expected = read_legacy_virtual_atoms(legacy_path)
    for key in ("atom", "type", "from_offset", "from", "parameter_offset"):
        compare_int_vectors(
            f"{h5_path}:/forcefield/virtual_atom/{key} against {legacy_path}",
            read_h5_ints(h5dump, h5_path, f"/forcefield/virtual_atom/{key}"),
            expected[key],
        )
    compare_float_vectors(
        f"{h5_path}:/forcefield/virtual_atom/parameter against {legacy_path}",
        read_h5_floats(h5dump, h5_path, "/forcefield/virtual_atom/parameter"),
        expected["parameter"],
    )


def compare_lj_soft_core_to_h5(h5dump, legacy_path, subsystem_path, h5_path):
    expected = read_legacy_lj_soft_core(legacy_path, subsystem_path)
    for key in ("atom_type_count_A", "atom_type_count_B"):
        actual = read_h5_scalar_int(
            h5dump, h5_path, f"/forcefield/lj_soft_core/{key}"
        )
        if actual != expected[key]:
            fail(
                f"{h5_path}:/forcefield/lj_soft_core/{key} mismatch against "
                f"{legacy_path}: actual={actual} expected={expected[key]}"
            )
    for key in ("pair_AA", "pair_AB", "pair_BA", "pair_BB"):
        compare_float_vectors(
            f"{h5_path}:/forcefield/lj_soft_core/{key} against {legacy_path}",
            read_h5_floats(h5dump, h5_path, f"/forcefield/lj_soft_core/{key}"),
            expected[key],
        )
    for key in ("atom_type_A", "atom_type_B"):
        compare_int_vectors(
            f"{h5_path}:/forcefield/lj_soft_core/{key} against {legacy_path}",
            read_h5_ints(h5dump, h5_path, f"/forcefield/lj_soft_core/{key}"),
            expected[key],
        )
    compare_int_vectors(
        f"{h5_path}:/forcefield/subsys_division against {subsystem_path}",
        read_h5_ints(h5dump, h5_path, "/forcefield/subsys_division"),
        expected["subsys_division"],
    )


def compare_cmap_to_h5(h5dump, legacy_path, h5_path):
    expected = read_legacy_cmap(legacy_path)
    compare_int_vectors(
        f"{h5_path}:/forcefield/cmap/atoms against {legacy_path}",
        read_h5_ints(h5dump, h5_path, "/forcefield/cmap/atoms"),
        expected["atoms"],
    )
    compare_int_vectors(
        f"{h5_path}:/forcefield/cmap/type against {legacy_path}",
        read_h5_ints(h5dump, h5_path, "/forcefield/cmap/type"),
        expected["type"],
    )
    compare_int_vectors(
        f"{h5_path}:/forcefield/cmap/resolution against {legacy_path}",
        read_h5_ints(h5dump, h5_path, "/forcefield/cmap/resolution"),
        expected["resolution"],
    )
    compare_float_vectors(
        f"{h5_path}:/forcefield/cmap/grid_value against {legacy_path}",
        read_h5_floats(h5dump, h5_path, "/forcefield/cmap/grid_value"),
        expected["grid_value"],
    )


def compare_custom_pair_to_h5(h5dump, descriptor_path, data_path, h5_path):
    descriptor = parse_custom_force_descriptor(descriptor_path)
    expected = read_legacy_custom_pair(data_path)
    root = "/forcefield/custom_force/pairwise"
    data_root = root + "/data/" + descriptor["name"]

    actual_name = read_h5_scalar_string(h5dump, h5_path, root + "/name")
    if actual_name != descriptor["name"]:
        fail(
            f"{h5_path}:{root}/name mismatch against {descriptor_path}: "
            f"actual={actual_name!r} expected={descriptor['name']!r}"
        )
    actual_potential = read_h5_scalar_string(
        h5dump, h5_path, root + "/potential"
    )
    if actual_potential != descriptor["potential"]:
        fail(
            f"{h5_path}:{root}/potential mismatch against {descriptor_path}: "
            f"actual={actual_potential!r} expected={descriptor['potential']!r}"
        )
    actual_with_ele = read_h5_enum_bools(h5dump, h5_path, root + "/with_ele")
    expected_with_ele = descriptor["with_ele"] == "true"
    if actual_with_ele != [expected_with_ele]:
        fail(
            f"{h5_path}:{root}/with_ele mismatch against {descriptor_path}: "
            f"actual={actual_with_ele} expected={[expected_with_ele]}"
        )
    compare_int_vectors(
        f"{h5_path}:{data_root}/atom_type against {data_path}",
        read_h5_ints(h5dump, h5_path, data_root + "/atom_type"),
        expected["atom_type"],
    )
    compare_float_vectors(
        f"{h5_path}:{data_root}/parameter/value against {data_path}",
        read_h5_floats(h5dump, h5_path, data_root + "/parameter/value"),
        expected["value"],
    )
    for key in ("atom_count", "type_count", "pair_count"):
        actual = read_h5_scalar_int(h5dump, h5_path, data_root + "/" + key)
        if actual != expected[key]:
            fail(
                f"{h5_path}:{data_root}/{key} mismatch against {data_path}: "
                f"actual={actual} expected={expected[key]}"
            )
    for dataset in (root + "/parameters", data_root + "/parameter"):
        compare_int_vectors(
            f"{h5_path}:{dataset}/name against {descriptor_path}",
            read_h5_strings(h5dump, h5_path, dataset + "/name"),
            descriptor["parameter_name"],
        )
        compare_int_vectors(
            f"{h5_path}:{dataset}/type against {descriptor_path}",
            read_h5_strings(h5dump, h5_path, dataset + "/type"),
            descriptor["parameter_type"],
        )


def compare_custom_bond_to_h5(h5dump, descriptor_path, data_path, h5_path):
    descriptor = parse_custom_force_descriptor(descriptor_path)
    expected = read_legacy_custom_bond(data_path)
    root = "/forcefield/custom_force/listed"
    data_root = root + "/data/" + descriptor["name"]

    for key, expected_value in (
        ("name", [descriptor["name"]]),
        ("potential", [descriptor["potential"]]),
        ("connected_atoms", [descriptor["connected_atoms"]]),
        ("constrain_distance", [descriptor["constrain_distance"]]),
    ):
        compare_int_vectors(
            f"{h5_path}:{root}/{key} against {descriptor_path}",
            read_h5_strings(h5dump, h5_path, root + "/" + key),
            expected_value,
        )
    actual_data_name = read_h5_scalar_string(
        h5dump, h5_path, data_root + "/name"
    )
    if actual_data_name != descriptor["name"]:
        fail(
            f"{h5_path}:{data_root}/name mismatch against {descriptor_path}: "
            f"actual={actual_data_name!r} expected={descriptor['name']!r}"
        )
    actual_count = read_h5_scalar_int(
        h5dump, h5_path, data_root + "/item_count"
    )
    if actual_count != expected["item_count"]:
        fail(
            f"{h5_path}:{data_root}/item_count mismatch against {data_path}: "
            f"actual={actual_count} expected={expected['item_count']}"
        )
    compare_float_vectors(
        f"{h5_path}:{data_root}/parameter/value against {data_path}",
        read_h5_floats(h5dump, h5_path, data_root + "/parameter/value"),
        expected["value"],
    )
    compare_int_vectors(
        f"{h5_path}:{data_root}/parameter/int_value against {data_path}",
        read_h5_ints(h5dump, h5_path, data_root + "/parameter/int_value"),
        expected["int_value"],
    )
    expected_is_int = [value == "int" for value in descriptor["parameter_type"]]
    compare_int_vectors(
        f"{h5_path}:{data_root}/parameter/is_int against {descriptor_path}",
        read_h5_enum_bools(h5dump, h5_path, data_root + "/parameter/is_int"),
        expected_is_int,
    )
    for dataset in (root + "/parameters", data_root + "/parameter"):
        compare_int_vectors(
            f"{h5_path}:{dataset}/name against {descriptor_path}",
            read_h5_strings(h5dump, h5_path, dataset + "/name"),
            descriptor["parameter_name"],
        )
        compare_int_vectors(
            f"{h5_path}:{dataset}/type against {descriptor_path}",
            read_h5_strings(h5dump, h5_path, dataset + "/type"),
            descriptor["parameter_type"],
        )


def compare_manybody_pair_triple_to_h5(
    h5dump,
    legacy_path,
    h5_path,
    h5_root,
    pair_parameter_count,
    triple_parameter_count,
):
    expected = read_legacy_manybody_pair_triple(
        legacy_path, pair_parameter_count, triple_parameter_count
    )
    actual_atom_type_count = read_h5_scalar_int(
        h5dump, h5_path, h5_root + "/atom_type_count"
    )
    if actual_atom_type_count != expected["atom_type_count"]:
        fail(
            f"{h5_path}:{h5_root}/atom_type_count mismatch against "
            f"{legacy_path}: actual={actual_atom_type_count} "
            f"expected={expected['atom_type_count']}"
        )
    compare_int_vectors(
        f"{h5_path}:{h5_root}/atom_type against {legacy_path}",
        read_h5_ints(h5dump, h5_path, h5_root + "/atom_type"),
        expected["atom_type"],
    )
    compare_int_vectors(
        f"{h5_path}:{h5_root}/pair/type against {legacy_path}",
        read_h5_ints(h5dump, h5_path, h5_root + "/pair/type"),
        expected["pair_type"],
    )
    compare_float_vectors(
        f"{h5_path}:{h5_root}/pair/parameters against {legacy_path}",
        read_h5_floats(h5dump, h5_path, h5_root + "/pair/parameters"),
        expected["pair_parameters"],
    )
    compare_int_vectors(
        f"{h5_path}:{h5_root}/triple/type against {legacy_path}",
        read_h5_ints(h5dump, h5_path, h5_root + "/triple/type"),
        expected["triple_type"],
    )
    compare_float_vectors(
        f"{h5_path}:{h5_root}/triple/parameters against {legacy_path}",
        read_h5_floats(h5dump, h5_path, h5_root + "/triple/parameters"),
        expected["triple_parameters"],
    )


def compare_tersoff_to_h5(h5dump, legacy_path, h5_path):
    expected = read_legacy_tersoff(legacy_path)
    root = "/manybody/tersoff"
    actual_atom_type_count = read_h5_scalar_int(
        h5dump, h5_path, root + "/atom_type_count"
    )
    if actual_atom_type_count != expected["atom_type_count"]:
        fail(
            f"{h5_path}:{root}/atom_type_count mismatch against {legacy_path}: "
            f"actual={actual_atom_type_count} "
            f"expected={expected['atom_type_count']}"
        )
    actual_entry_count = read_h5_scalar_int(
        h5dump, h5_path, root + "/entry/count"
    )
    if actual_entry_count != expected["entry_count"]:
        fail(
            f"{h5_path}:{root}/entry/count mismatch against {legacy_path}: "
            f"actual={actual_entry_count} expected={expected['entry_count']}"
        )
    compare_int_vectors(
        f"{h5_path}:{root}/atom_type against {legacy_path}",
        read_h5_ints(h5dump, h5_path, root + "/atom_type"),
        expected["atom_type"],
    )
    compare_int_vectors(
        f"{h5_path}:{root}/map against {legacy_path}",
        read_h5_ints(h5dump, h5_path, root + "/map"),
        expected["map"],
    )
    compare_int_vectors(
        f"{h5_path}:{root}/entry/type against {legacy_path}",
        read_h5_ints(h5dump, h5_path, root + "/entry/type"),
        expected["entry_type"],
    )
    compare_int_vectors(
        f"{h5_path}:{root}/type_name against {legacy_path}",
        read_h5_strings(h5dump, h5_path, root + "/type_name"),
        expected["type_name"],
    )
    compare_int_vectors(
        f"{h5_path}:{root}/entry/type_name against {legacy_path}",
        read_h5_strings(h5dump, h5_path, root + "/entry/type_name"),
        expected["entry_type_name"],
    )
    compare_float_vectors(
        f"{h5_path}:{root}/entry/parameters_raw against {legacy_path}",
        read_h5_floats(h5dump, h5_path, root + "/entry/parameters_raw"),
        expected["parameters_raw"],
    )


def compare_eam_to_h5(h5dump, legacy_eam_path, legacy_atom_type_path, h5_path):
    expected = read_legacy_eam(legacy_eam_path, legacy_atom_type_path)
    root = "/manybody/eam"
    actual_format = read_h5_scalar_string(h5dump, h5_path, root + "/format")
    if actual_format != expected["format"]:
        fail(
            f"{h5_path}:{root}/format mismatch against {legacy_eam_path}: "
            f"actual={actual_format!r} expected={expected['format']!r}"
        )
    actual_atom_type_count = read_h5_scalar_int(
        h5dump, h5_path, root + "/atom_type_count"
    )
    if actual_atom_type_count != expected["atom_type_count"]:
        fail(
            f"{h5_path}:{root}/atom_type_count mismatch against "
            f"{legacy_eam_path}: actual={actual_atom_type_count} "
            f"expected={expected['atom_type_count']}"
        )
    for key in ("nrho", "nr"):
        actual = read_h5_scalar_int(h5dump, h5_path, root + "/" + key)
        if actual != expected[key]:
            fail(
                f"{h5_path}:{root}/{key} mismatch against {legacy_eam_path}: "
                f"actual={actual} expected={expected[key]}"
            )
    for key in ("drho", "dr", "cut"):
        compare_float_vectors(
            f"{h5_path}:{root}/{key} against {legacy_eam_path}",
            read_h5_floats(h5dump, h5_path, root + "/" + key),
            [expected[key]],
        )
    compare_int_vectors(
        f"{h5_path}:{root}/atom_type against {legacy_atom_type_path}",
        read_h5_ints(h5dump, h5_path, root + "/atom_type"),
        expected["atom_type"],
    )
    compare_int_vectors(
        f"{h5_path}:{root}/atomic_number against {legacy_eam_path}",
        read_h5_ints(h5dump, h5_path, root + "/atomic_number"),
        expected["atomic_number"],
    )
    for key in ("mass", "lattice_constant"):
        compare_float_vectors(
            f"{h5_path}:{root}/{key} against {legacy_eam_path}",
            read_h5_floats(h5dump, h5_path, root + "/" + key),
            expected[key],
        )
    compare_int_vectors(
        f"{h5_path}:{root}/lattice_type against {legacy_eam_path}",
        read_h5_strings(h5dump, h5_path, root + "/lattice_type"),
        expected["lattice_type"],
    )
    compare_float_vectors(
        f"{h5_path}:{root}/embed/raw_ev against {legacy_eam_path}",
        read_h5_floats(h5dump, h5_path, root + "/embed/raw_ev"),
        expected["embed_raw_ev"],
    )
    compare_float_vectors(
        f"{h5_path}:{root}/funcfl/z against {legacy_eam_path}",
        read_h5_floats(h5dump, h5_path, root + "/funcfl/z"),
        expected["funcfl_z"],
    )
    compare_float_vectors(
        f"{h5_path}:{root}/electron_density/value against {legacy_eam_path}",
        read_h5_floats(h5dump, h5_path, root + "/electron_density/value"),
        expected["electron_density"],
    )


def compare_reaxff_type_to_h5(h5dump, legacy_path, h5_path):
    expected = read_legacy_reaxff_types(legacy_path)
    root = "/manybody/reaxff/type"
    actual_count = read_h5_scalar_int(h5dump, h5_path, root + "/count")
    if actual_count != expected["count"]:
        fail(
            f"{h5_path}:{root}/count mismatch against {legacy_path}: "
            f"actual={actual_count} expected={expected['count']}"
        )
    compare_int_vectors(
        f"{h5_path}:{root}/name against {legacy_path}",
        read_h5_strings(h5dump, h5_path, root + "/name"),
        expected["name"],
    )


def compare_reaxff_counted_section_to_h5(
    h5dump, h5_path, legacy_path, section_name, expected
):
    root = "/manybody/reaxff/parameters/" + section_name
    actual_count = read_h5_scalar_int(h5dump, h5_path, root + "/count")
    if actual_count != expected["count"]:
        fail(
            f"{h5_path}:{root}/count mismatch against {legacy_path}: "
            f"actual={actual_count} expected={expected['count']}"
        )
    actual_count_label = read_h5_scalar_string(
        h5dump, h5_path, root + "/count_label"
    )
    if actual_count_label != expected["count_label"]:
        fail(
            f"{h5_path}:{root}/count_label mismatch against {legacy_path}: "
            f"actual={actual_count_label!r} expected={expected['count_label']!r}"
        )
    compare_int_vectors(
        f"{h5_path}:{root}/type against {legacy_path}",
        read_h5_ints(h5dump, h5_path, root + "/type"),
        expected["type"],
    )
    compare_float_vectors(
        f"{h5_path}:{root}/value against {legacy_path}",
        read_h5_floats(h5dump, h5_path, root + "/value"),
        expected["value"],
    )


def compare_reaxff_to_h5(h5dump, legacy_path, legacy_type_path, h5_path):
    expected = read_legacy_reaxff(legacy_path)
    root = "/manybody/reaxff/parameters"
    actual_header = read_h5_scalar_string(h5dump, h5_path, root + "/header")
    if actual_header != expected["header"]:
        fail(
            f"{h5_path}:{root}/header mismatch against {legacy_path}: "
            f"actual={actual_header!r} expected={expected['header']!r}"
        )

    compare_reaxff_type_to_h5(h5dump, legacy_type_path, h5_path)

    general_root = root + "/general"
    actual_general_count = read_h5_scalar_int(
        h5dump, h5_path, general_root + "/count"
    )
    actual_general_count_label = read_h5_scalar_string(
        h5dump, h5_path, general_root + "/count_label"
    )
    if actual_general_count_label != expected["general"]["count_label"]:
        fail(
            f"{h5_path}:{general_root}/count_label mismatch against "
            f"{legacy_path}: actual={actual_general_count_label!r} "
            f"expected={expected['general']['count_label']!r}"
        )
    if actual_general_count != len(expected["general"]["value"]):
        fail(
            f"{h5_path}:{general_root}/count mismatch against legacy general "
            f"values: "
            f"actual={actual_general_count} "
            f"legacy={len(expected['general']['value'])}"
        )
    compare_int_vectors(
        f"{h5_path}:{general_root}/label against {legacy_path}",
        read_h5_strings(h5dump, h5_path, general_root + "/label"),
        expected["general"]["label"][:actual_general_count],
    )
    compare_float_vectors(
        f"{h5_path}:{general_root}/value against {legacy_path}",
        read_h5_floats(h5dump, h5_path, general_root + "/value"),
        expected["general"]["value"][:actual_general_count],
    )

    atom_root = root + "/atom"
    actual_atom_count = read_h5_scalar_int(
        h5dump, h5_path, atom_root + "/count"
    )
    if actual_atom_count != expected["atom"]["count"]:
        fail(
            f"{h5_path}:{atom_root}/count mismatch against {legacy_path}: "
            f"actual={actual_atom_count} expected={expected['atom']['count']}"
        )
    actual_atom_count_label = read_h5_scalar_string(
        h5dump, h5_path, atom_root + "/count_label"
    )
    if actual_atom_count_label != expected["atom"]["count_label"]:
        fail(
            f"{h5_path}:{atom_root}/count_label mismatch against {legacy_path}: "
            f"actual={actual_atom_count_label!r} "
            f"expected={expected['atom']['count_label']!r}"
        )
    for key in ("header", "type_name", "line_label"):
        compare_int_vectors(
            f"{h5_path}:{atom_root}/{key} against {legacy_path}",
            read_h5_strings(h5dump, h5_path, atom_root + "/" + key),
            expected["atom"][key],
        )
    for key in ("value_offset", "line_value_offset"):
        compare_int_vectors(
            f"{h5_path}:{atom_root}/{key} against {legacy_path}",
            read_h5_ints(h5dump, h5_path, atom_root + "/" + key),
            expected["atom"][key],
        )
    compare_float_vectors(
        f"{h5_path}:{atom_root}/value against {legacy_path}",
        read_h5_floats(h5dump, h5_path, atom_root + "/value"),
        expected["atom"]["value"],
    )

    bond_root = root + "/bond"
    actual_bond_count = read_h5_scalar_int(
        h5dump, h5_path, bond_root + "/count"
    )
    if actual_bond_count != expected["bond"]["count"]:
        fail(
            f"{h5_path}:{bond_root}/count mismatch against {legacy_path}: "
            f"actual={actual_bond_count} expected={expected['bond']['count']}"
        )
    actual_bond_count_label = read_h5_scalar_string(
        h5dump, h5_path, bond_root + "/count_label"
    )
    if actual_bond_count_label != expected["bond"]["count_label"]:
        fail(
            f"{h5_path}:{bond_root}/count_label mismatch against {legacy_path}: "
            f"actual={actual_bond_count_label!r} "
            f"expected={expected['bond']['count_label']!r}"
        )
    for key in ("header", "line_label"):
        compare_int_vectors(
            f"{h5_path}:{bond_root}/{key} against {legacy_path}",
            read_h5_strings(h5dump, h5_path, bond_root + "/" + key),
            expected["bond"][key],
        )
    for key in ("type", "value_offset", "line_value_offset"):
        compare_int_vectors(
            f"{h5_path}:{bond_root}/{key} against {legacy_path}",
            read_h5_ints(h5dump, h5_path, bond_root + "/" + key),
            expected["bond"][key],
        )
    compare_float_vectors(
        f"{h5_path}:{bond_root}/value against {legacy_path}",
        read_h5_floats(h5dump, h5_path, bond_root + "/value"),
        expected["bond"]["value"],
    )

    for section_name in ("off_diagonal", "angle", "torsion", "hydrogen_bond"):
        compare_reaxff_counted_section_to_h5(
            h5dump, h5_path, legacy_path, section_name, expected[section_name]
        )


def compare_trajectory_to_legacy(h5dump, legacy_root, bundle_root):
    trajectory = bundle_root / "trajectory.spg.h5md"
    atom_count = len(read_legacy_counted_float_vector(legacy_root / "mass.txt"))
    position = read_legacy_binary_float32(legacy_root / "traj.dat")
    velocity = read_legacy_binary_float32(legacy_root / "traj_vel.dat")
    values_per_frame = atom_count * 3
    if len(position) % values_per_frame != 0:
        fail(
            f"legacy trajectory position length is not a whole number of frames: "
            f"{legacy_root / 'traj.dat'}"
        )
    frame_count = len(position) // values_per_frame
    if len(velocity) != len(position):
        fail(
            f"legacy trajectory velocity length mismatch: "
            f"position={len(position)} velocity={len(velocity)}"
        )
    box_rows = read_legacy_text_float_rows(legacy_root / "traj_box.dat", 6)
    if len(box_rows) != frame_count:
        fail(
            f"legacy trajectory box frame count mismatch: "
            f"position={frame_count} box={len(box_rows)}"
        )
    expected_box_edges = [
        value for row in box_rows for value in box_lengths_angles_to_edges(row)
    ]

    compare_int_vectors(
        f"{trajectory}:/particles/all/step",
        read_h5_ints(h5dump, trajectory, "/particles/all/step"),
        list(range(frame_count)),
    )
    compare_float_vectors(
        f"{trajectory}:/particles/all/time",
        read_h5_floats(h5dump, trajectory, "/particles/all/time"),
        [float(frame) for frame in range(frame_count)],
    )
    compare_float_vectors(
        f"{trajectory}:/particles/all/position/value against legacy traj.dat",
        read_h5_floats(h5dump, trajectory, "/particles/all/position/value"),
        position,
    )
    compare_float_vectors(
        f"{trajectory}:/particles/all/velocity/value against legacy traj_vel.dat",
        read_h5_floats(h5dump, trajectory, "/particles/all/velocity/value"),
        velocity,
    )
    compare_float_vectors(
        f"{trajectory}:/particles/all/box/edges/value against legacy traj_box.dat",
        read_h5_floats(h5dump, trajectory, "/particles/all/box/edges/value"),
        expected_box_edges,
        tolerance=1.0e-5,
    )


def compare_restart_structural_to_legacy(h5dump, legacy_root, bundle_root):
    restart = bundle_root / "restart.spgr.h5"
    coordinate = read_legacy_coordinate_file(legacy_root / "coordinate.txt")
    velocity = read_legacy_velocity_file(legacy_root / "velocity.txt")
    if coordinate["atom_count"] != velocity["atom_count"]:
        fail(
            f"legacy restart atom count mismatch: "
            f"coordinate={coordinate['atom_count']} velocity={velocity['atom_count']}"
        )
    compare_int_vectors(
        f"{restart}:/particles/all/step",
        read_h5_ints(h5dump, restart, "/particles/all/step"),
        [0],
    )
    compare_float_vectors(
        f"{restart}:/particles/all/time",
        read_h5_floats(h5dump, restart, "/particles/all/time"),
        [0.0],
    )
    compare_float_vectors(
        f"{restart}:/particles/all/position/value against coordinate.txt",
        read_h5_floats(h5dump, restart, "/particles/all/position/value"),
        coordinate["position"],
    )
    compare_float_vectors(
        f"{restart}:/particles/all/velocity/value against velocity.txt",
        read_h5_floats(h5dump, restart, "/particles/all/velocity/value"),
        velocity["velocity"],
    )
    compare_float_vectors(
        f"{restart}:/particles/all/box/edges/value against coordinate.txt box",
        read_h5_floats(h5dump, restart, "/particles/all/box/edges/value"),
        box_lengths_angles_to_edges(coordinate["box"]),
        tolerance=1.0e-5,
    )


def compare_embedded_sidecar_text(
    h5dump, manifest_path, bundle_root, legacy_root
):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checked_count = 0
    for entry in manifest.get("entries", []):
        if entry.get("status") != "sidecar_embedded":
            continue
        checked_count += 1
        bundle_file = entry.get("bundle_file")
        bundle_path = entry.get("bundle_path")
        source_path = entry.get("source_path")
        contract_id = entry.get("contract_id", "<unknown>")
        if not bundle_file or not bundle_path or not source_path:
            fail(f"embedded sidecar manifest entry lacks paths: {contract_id}")
        h5_path = bundle_root / bundle_file
        if not h5_path.is_file():
            fail(
                f"embedded sidecar H5 file is missing for {contract_id}: {h5_path}"
            )
        source = legacy_root / Path(source_path).name
        if not source.is_file():
            fail(
                f"embedded sidecar source file is missing for {contract_id}: {source}"
            )
        actual = read_h5_scalar_string(h5dump, h5_path, bundle_path)
        expected = source.read_text(encoding="utf-8")
        if actual != expected:
            fail(
                f"embedded sidecar text mismatch for {contract_id}:\n"
                f"h5={h5_path}:{bundle_path}\nsource={source_path}\n"
                f"actual={actual!r}\nexpected={expected!r}"
            )
    if checked_count == 0:
        fail(f"no embedded sidecar manifest entries found in {manifest_path}")


def compare_protocol_sits_atom_to_h5(h5dump, legacy_path, h5_path):
    compare_int_vectors(
        f"{h5_path}:/sits/atom_indices against {legacy_path}",
        read_h5_ints(h5dump, h5_path, "/sits/atom_indices"),
        read_legacy_int_list(legacy_path),
    )


def compare_protocol_constraints_to_h5(h5dump, legacy_path, h5_path):
    expected = read_legacy_constraints(legacy_path)
    compare_int_vectors(
        f"{h5_path}:/constraint/default/pairs/atoms against {legacy_path}",
        read_h5_ints(h5dump, h5_path, "/constraint/default/pairs/atoms"),
        expected["atoms"],
    )
    compare_float_vectors(
        f"{h5_path}:/constraint/default/pairs/r0 against {legacy_path}",
        read_h5_floats(h5dump, h5_path, "/constraint/default/pairs/r0"),
        expected["r0"],
    )


def compare_protocol_restraint_default_to_h5(
    h5dump, atom_id_path, weight_path, h5_path
):
    compare_int_vectors(
        f"{h5_path}:/restraint/default/atom_indices against {atom_id_path}",
        read_h5_ints(h5dump, h5_path, "/restraint/default/atom_indices"),
        read_legacy_int_list(atom_id_path),
    )
    expected_weight = read_legacy_float_matrix(weight_path, expected_columns=3)
    compare_float_vectors(
        f"{h5_path}:/restraint/default/weight against {weight_path}",
        read_h5_floats(h5dump, h5_path, "/restraint/default/weight"),
        expected_weight["values"],
    )


def read_h5_config_sections(h5dump, h5_path, root):
    section_names = read_h5_strings(h5dump, h5_path, root + "/section/name")
    section_count = read_h5_scalar_int(h5dump, h5_path, root + "/section/count")
    if section_count != len(section_names):
        fail(
            f"{h5_path}:{root}/section/count mismatch: "
            f"actual={section_count} names={len(section_names)}"
        )
    key_offsets = read_h5_ints(h5dump, h5_path, root + "/section/key_offset")
    keys = read_h5_strings(h5dump, h5_path, root + "/key")
    values = read_h5_strings(h5dump, h5_path, root + "/value")
    if len(keys) != len(values):
        fail(
            f"{h5_path}:{root} key/value length mismatch: "
            f"keys={len(keys)} values={len(values)}"
        )
    if len(key_offsets) != section_count + 1:
        fail(
            f"{h5_path}:{root}/section/key_offset length mismatch: "
            f"actual={len(key_offsets)} expected={section_count + 1}"
        )
    if key_offsets[0] != 0 or key_offsets[-1] != len(keys):
        fail(
            f"{h5_path}:{root}/section/key_offset boundary mismatch: "
            f"offsets={key_offsets} key_count={len(keys)}"
        )
    sections = []
    for index, name in enumerate(section_names):
        begin = key_offsets[index]
        end = key_offsets[index + 1]
        sections.append(
            {
                "name": name,
                "entries": list(zip(keys[begin:end], values[begin:end])),
            }
        )
    return sections


def compare_config_sections_to_h5(h5dump, legacy_path, h5_path, h5_root):
    expected = read_legacy_config_sections(legacy_path)
    actual = read_h5_config_sections(h5dump, h5_path, h5_root)
    if actual != expected:
        fail(
            f"{h5_path}:{h5_root} config mismatch against {legacy_path}:\n"
            f"actual={actual}\nexpected={expected}"
        )


def compare_protocol_meta_edge_to_h5(h5dump, legacy_path, h5_path):
    expected = read_legacy_meta_edge(legacy_path)
    root = "/meta/default/grid"
    for key in ("count", "ndim"):
        actual = read_h5_scalar_int(h5dump, h5_path, root + "/" + key)
        if actual != expected[key]:
            fail(
                f"{h5_path}:{root}/{key} mismatch against {legacy_path}: "
                f"actual={actual} expected={expected[key]}"
            )
    for key in ("coordinate", "normalization", "force"):
        compare_float_vectors(
            f"{h5_path}:{root}/{key} against {legacy_path}",
            read_h5_floats(h5dump, h5_path, root + "/" + key),
            expected[key],
        )


def compare_soft_walls_to_h5(h5dump, legacy_path, h5_path):
    descriptor = parse_named_section_descriptor(legacy_path)
    root = "/wall/soft"
    actual_count = read_h5_scalar_int(h5dump, h5_path, root + "/count")
    if actual_count != 1:
        fail(
            f"{h5_path}:{root}/count mismatch: actual={actual_count} expected=1"
        )
    compare_int_vectors(
        f"{h5_path}:{root}/name against {legacy_path}",
        read_h5_strings(h5dump, h5_path, root + "/name"),
        [descriptor["name"]],
    )
    compare_int_vectors(
        f"{h5_path}:{root}/potential against {legacy_path}",
        read_h5_strings(h5dump, h5_path, root + "/potential"),
        ["\n".join(descriptor["sections"].get("potential", []))],
    )


def compare_restart_sits_nk_to_h5(h5dump, legacy_path, restart_path):
    compare_float_vectors(
        f"{restart_path}:/parameters/restart/bias/sits/SITS/nk against {legacy_path}",
        read_h5_floats(
            h5dump, restart_path, "/parameters/restart/bias/sits/SITS/nk"
        ),
        read_legacy_float_list(legacy_path),
    )


def compare_restart_restraint_reference_to_h5(
    h5dump, legacy_path, restart_path
):
    expected = read_legacy_counted_xyz(legacy_path)
    compare_float_vectors(
        f"{restart_path}:/parameters/restart/references/restraint/default/coordinate "
        f"against {legacy_path}",
        read_h5_floats(
            h5dump,
            restart_path,
            "/parameters/restart/references/restraint/default/coordinate",
        ),
        expected["values"],
    )


def compare_restart_nhc_to_h5(h5dump, legacy_path, restart_path):
    expected = read_legacy_float_matrix(legacy_path, expected_columns=2)
    compare_float_vectors(
        f"{restart_path}:/parameters/restart/thermostat/nose_hoover_chain "
        f"against {legacy_path}",
        read_h5_floats(
            h5dump,
            restart_path,
            "/parameters/restart/thermostat/nose_hoover_chain",
        ),
        expected["values"],
    )


def compare_restart_hills_to_h5(h5dump, legacy_path, restart_path):
    expected = read_legacy_float_matrix(legacy_path, expected_columns=3)
    count_path = "/parameters/restart/bias/meta/default/hills_typed/count"
    column_count_path = (
        "/parameters/restart/bias/meta/default/hills_typed/column_count"
    )
    value_path = "/parameters/restart/bias/meta/default/hills_typed/value"
    actual_count = read_h5_scalar_int(h5dump, restart_path, count_path)
    if actual_count != expected["row_count"]:
        fail(
            f"{restart_path}:{count_path} mismatch against {legacy_path}: "
            f"actual={actual_count} expected={expected['row_count']}"
        )
    actual_column_count = read_h5_scalar_int(
        h5dump, restart_path, column_count_path
    )
    if actual_column_count != 3:
        fail(
            f"{restart_path}:{column_count_path} mismatch: "
            f"actual={actual_column_count} expected=3"
        )
    compare_float_vectors(
        f"{restart_path}:{value_path} against {legacy_path}",
        read_h5_floats(h5dump, restart_path, value_path),
        expected["values"],
    )


def compare_restart_meta_scatter_to_h5(h5dump, legacy_path, restart_path):
    expected = read_legacy_meta_scatter(legacy_path)
    root = "/parameters/restart/bias/meta/default/scatter"
    actual_size = read_h5_scalar_int(
        h5dump, restart_path, root + "/scatter_size"
    )
    if actual_size != expected["scatter_size"]:
        fail(
            f"{restart_path}:{root}/scatter_size mismatch against {legacy_path}: "
            f"actual={actual_size} expected={expected['scatter_size']}"
        )
    actual_ndim = read_h5_scalar_int(h5dump, restart_path, root + "/ndim")
    if actual_ndim != expected["ndim"]:
        fail(
            f"{restart_path}:{root}/ndim mismatch against {legacy_path}: "
            f"actual={actual_ndim} expected={expected['ndim']}"
        )
    compare_float_vectors(
        f"{restart_path}:{root}/coordinate against {legacy_path}",
        read_h5_floats(h5dump, restart_path, root + "/coordinate"),
        expected["coordinate"],
    )
    for key in ("min", "max", "delta"):
        compare_float_vectors(
            f"{restart_path}:{root}/axis/{key} against {legacy_path}",
            read_h5_floats(h5dump, restart_path, root + f"/axis/{key}"),
            expected[f"axis_{key}"],
        )
    compare_float_vectors(
        f"{restart_path}:{root}/force against {legacy_path}",
        read_h5_floats(h5dump, restart_path, root + "/force"),
        expected["force"],
    )
    compare_float_vectors(
        f"{restart_path}:{root}/value against {legacy_path}",
        read_h5_floats(h5dump, restart_path, root + "/value"),
        expected["value"],
    )


def compare_restart_meta_potential_to_h5(h5dump, legacy_path, restart_path):
    expected = read_legacy_meta_potential(legacy_path)
    root = "/parameters/restart/bias/meta/default/potential"
    actual_ndim = read_h5_scalar_int(h5dump, restart_path, root + "/ndim")
    if actual_ndim != expected["ndim"]:
        fail(
            f"{restart_path}:{root}/ndim mismatch against {legacy_path}: "
            f"actual={actual_ndim} expected={expected['ndim']}"
        )
    compare_int_vectors(
        f"{restart_path}:{root}/grid against {legacy_path}",
        read_h5_ints(h5dump, restart_path, root + "/grid"),
        expected["grid"],
    )
    for key in ("min", "max", "delta"):
        compare_float_vectors(
            f"{restart_path}:{root}/axis/{key} against {legacy_path}",
            read_h5_floats(h5dump, restart_path, root + f"/axis/{key}"),
            expected[f"axis_{key}"],
        )
    compare_float_vectors(
        f"{restart_path}:{root}/coordinate against {legacy_path}",
        read_h5_floats(h5dump, restart_path, root + "/coordinate"),
        expected["coordinate"],
    )
    compare_float_vectors(
        f"{restart_path}:{root}/force against {legacy_path}",
        read_h5_floats(h5dump, restart_path, root + "/force"),
        expected["force"],
    )
    compare_float_vectors(
        f"{restart_path}:{root}/value against {legacy_path}",
        read_h5_floats(h5dump, restart_path, root + "/value"),
        expected["value"],
    )


def discover_h5_names(bundle_root):
    return {
        path.name
        for path in bundle_root.iterdir()
        if path.is_file() and path.suffix in {".h5", ".h5md"}
    }


def compare_group(h5diff, fixture_root, group, required_h5_names):
    pure = fixture_root / group / "bundled_input" / "bundle"
    sidecar = (
        fixture_root / group / "bundled_input_with_legacy_sidecar" / "bundle"
    )
    pure_names = discover_h5_names(pure)
    sidecar_names = discover_h5_names(sidecar)
    if pure_names != sidecar_names:
        fail(
            f"H5 file set mismatch for {group}: "
            f"pure={sorted(pure_names)} sidecar={sorted(sidecar_names)}"
        )
    missing_required = set(required_h5_names) - pure_names
    if missing_required:
        fail(f"H5 file set for {group} is missing {sorted(missing_required)}")

    h5_names = sorted(pure_names)
    for h5_name in h5_names:
        pure_path = pure / h5_name
        sidecar_path = sidecar / h5_name
        compare_h5_files(h5diff, pure_path, sidecar_path)


def compare_group_mass_charge(h5dump, fixture_root, group):
    legacy_root = fixture_root / group / "legacy_input"
    for family in ("bundled_input", "bundled_input_with_legacy_sidecar"):
        topology = fixture_root / group / family / "bundle" / "topology.spgt.h5"
        compare_counted_float_file_to_h5(
            h5dump, legacy_root / "mass.txt", topology, "/atoms/mass"
        )
        compare_counted_float_file_to_h5(
            h5dump, legacy_root / "charge.txt", topology, "/atoms/charge"
        )


def compare_full_contract_core_topology(h5dump, fixture_root):
    legacy_root = fixture_root / "full_contract_rerun" / "legacy_input"
    for family in ("bundled_input", "bundled_input_with_legacy_sidecar"):
        topology = (
            fixture_root
            / "full_contract_rerun"
            / family
            / "bundle"
            / "topology.spgt.h5"
        )
        compare_bonds_to_h5(h5dump, legacy_root / "bond.txt", topology)
        compare_angles_to_h5(h5dump, legacy_root / "angle.txt", topology)
        compare_lj_to_h5(h5dump, legacy_root / "lj.txt", topology)
        compare_dihedrals_to_h5(h5dump, legacy_root / "dihedral.txt", topology)
        compare_gb_to_h5(h5dump, legacy_root / "gb.txt", topology)
        compare_urey_bradley_to_h5(
            h5dump, legacy_root / "urey_bradley.txt", topology
        )
        compare_nb14_extra_to_h5(
            h5dump, legacy_root / "nb14_extra.txt", topology
        )
        compare_exclusions_to_h5(h5dump, legacy_root / "exclude.txt", topology)
        compare_residue_to_h5(h5dump, legacy_root / "residue.txt", topology)
        compare_virtual_atoms_to_h5(
            h5dump, legacy_root / "virtual_atom.txt", topology
        )
        compare_lj_soft_core_to_h5(
            h5dump,
            legacy_root / "lj_soft_core.txt",
            legacy_root / "subsys_division.txt",
            topology,
        )
        compare_cmap_to_h5(h5dump, legacy_root / "cmap.txt", topology)
        compare_custom_pair_to_h5(
            h5dump,
            legacy_root / "pairwise_force.txt",
            legacy_root / "custom_pair.txt",
            topology,
        )
        compare_custom_bond_to_h5(
            h5dump,
            legacy_root / "listed_forces.txt",
            legacy_root / "custom_bond.txt",
            topology,
        )
        compare_manybody_pair_triple_to_h5(
            h5dump,
            legacy_root / "sw.txt",
            topology,
            "/manybody/sw",
            pair_parameter_count=8,
            triple_parameter_count=3,
        )
        compare_manybody_pair_triple_to_h5(
            h5dump,
            legacy_root / "edip.txt",
            topology,
            "/manybody/edip",
            pair_parameter_count=8,
            triple_parameter_count=9,
        )
        compare_tersoff_to_h5(h5dump, legacy_root / "tersoff.txt", topology)
        compare_eam_to_h5(
            h5dump,
            legacy_root / "eam.txt",
            legacy_root / "eam_atom_type.txt",
            topology,
        )
        compare_reaxff_to_h5(
            h5dump,
            legacy_root / "reaxff.txt",
            legacy_root / "reaxff_type.txt",
            topology,
        )


def compare_full_contract_trajectory(h5dump, fixture_root):
    legacy_root = fixture_root / "full_contract_rerun" / "legacy_input"
    for family in ("bundled_input", "bundled_input_with_legacy_sidecar"):
        compare_trajectory_to_legacy(
            h5dump,
            legacy_root,
            fixture_root / "full_contract_rerun" / family / "bundle",
        )


def compare_group_restart_structural(h5dump, fixture_root, group):
    legacy_root = fixture_root / group / "legacy_input"
    for family in ("bundled_input", "bundled_input_with_legacy_sidecar"):
        compare_restart_structural_to_legacy(
            h5dump,
            legacy_root,
            fixture_root / group / family / "bundle",
        )


def compare_group_restart_protocol_sidecars(h5dump, fixture_root, group):
    case_root = fixture_root / group / "bundled_input_with_legacy_sidecar"
    compare_embedded_sidecar_text(
        h5dump,
        case_root / "manifest.json",
        case_root / "bundle",
        fixture_root / group / "legacy_input",
    )


def compare_group_restart_sits_nk(h5dump, fixture_root, group):
    legacy_path = fixture_root / group / "legacy_input" / "sits_nk.txt"
    for family in ("bundled_input", "bundled_input_with_legacy_sidecar"):
        compare_restart_sits_nk_to_h5(
            h5dump,
            legacy_path,
            fixture_root / group / family / "bundle" / "restart.spgr.h5",
        )


def compare_full_contract_protocol_typed(h5dump, fixture_root):
    legacy_root = fixture_root / "full_contract_rerun" / "legacy_input"
    for family in ("bundled_input", "bundled_input_with_legacy_sidecar"):
        protocol = (
            fixture_root
            / "full_contract_rerun"
            / family
            / "bundle"
            / "protocol.spgp.h5"
        )
        compare_protocol_sits_atom_to_h5(
            h5dump, legacy_root / "sits_atom.txt", protocol
        )
        compare_protocol_constraints_to_h5(
            h5dump, legacy_root / "constrain.txt", protocol
        )
        compare_protocol_restraint_default_to_h5(
            h5dump,
            legacy_root / "restrain_atom_id.txt",
            legacy_root / "restrain_weight.txt",
            protocol,
        )
        compare_config_sections_to_h5(
            h5dump, legacy_root / "cv.txt", protocol, "/cv/config"
        )
        compare_config_sections_to_h5(
            h5dump, legacy_root / "sits.txt", protocol, "/sits/config"
        )
        compare_config_sections_to_h5(
            h5dump, legacy_root / "restrain.txt", protocol, "/restraint/config"
        )
        compare_config_sections_to_h5(
            h5dump,
            legacy_root / "restrain_cv.txt",
            protocol,
            "/restraint/cv/config",
        )
        compare_config_sections_to_h5(
            h5dump, legacy_root / "steer_cv.txt", protocol, "/steer/config"
        )
        compare_protocol_meta_edge_to_h5(
            h5dump, legacy_root / "meta_edge.txt", protocol
        )
        compare_soft_walls_to_h5(
            h5dump, legacy_root / "soft_walls.txt", protocol
        )


def compare_full_contract_restart_dynamic_typed(h5dump, fixture_root):
    legacy_root = fixture_root / "full_contract_rerun" / "legacy_input"
    for family in ("bundled_input", "bundled_input_with_legacy_sidecar"):
        restart = (
            fixture_root
            / "full_contract_rerun"
            / family
            / "bundle"
            / "restart.spgr.h5"
        )
        compare_restart_nhc_to_h5(
            h5dump, legacy_root / "nhc_restart.txt", restart
        )
        compare_restart_hills_to_h5(h5dump, legacy_root / "hills.txt", restart)
        compare_restart_restraint_reference_to_h5(
            h5dump, legacy_root / "restrain_coordinate.txt", restart
        )
        compare_restart_meta_potential_to_h5(
            h5dump, legacy_root / "meta_potential.txt", restart
        )
        compare_restart_meta_scatter_to_h5(
            h5dump, legacy_root / "meta_scatter.txt", restart
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-root", required=True)
    parser.add_argument("--h5diff", default=None)
    parser.add_argument("--h5dump", default=None)
    args = parser.parse_args()
    fixture_root = Path(args.fixture_root)
    h5diff = args.h5diff or shutil.which("h5diff")
    if not h5diff:
        fail("h5diff executable was not found")
    h5dump = args.h5dump or shutil.which("h5dump")
    if not h5dump:
        fail("h5dump executable was not found")

    compare_group(
        h5diff,
        fixture_root,
        "core_structural",
        ["topology.spgt.h5", "protocol.spgp.h5", "restart.spgr.h5"],
    )
    compare_group(
        h5diff,
        fixture_root,
        "full_contract_rerun",
        [
            "topology.spgt.h5",
            "protocol.spgp.h5",
            "restart.spgr.h5",
            "trajectory.spg.h5md",
        ],
    )
    compare_group_mass_charge(h5dump, fixture_root, "core_structural")
    compare_group_mass_charge(h5dump, fixture_root, "full_contract_rerun")
    compare_group_restart_structural(h5dump, fixture_root, "core_structural")
    compare_group_restart_structural(
        h5dump, fixture_root, "full_contract_rerun"
    )
    compare_group_restart_protocol_sidecars(
        h5dump, fixture_root, "core_structural"
    )
    compare_group_restart_protocol_sidecars(
        h5dump, fixture_root, "full_contract_rerun"
    )
    compare_group_restart_sits_nk(h5dump, fixture_root, "core_structural")
    compare_group_restart_sits_nk(h5dump, fixture_root, "full_contract_rerun")
    compare_full_contract_protocol_typed(h5dump, fixture_root)
    compare_full_contract_restart_dynamic_typed(h5dump, fixture_root)
    compare_full_contract_core_topology(h5dump, fixture_root)
    compare_full_contract_trajectory(h5dump, fixture_root)


if __name__ == "__main__":
    main()

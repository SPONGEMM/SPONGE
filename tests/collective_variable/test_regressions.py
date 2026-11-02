"""Exercise CV values and bias forces through the actual SPONGE executable."""

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np


def block(name, **values):
    return (
        name
        + "\n{\n"
        + "".join(f"    {key} = {value}\n" for key, value in values.items())
        + "}\n"
    )


def run_case(
    binary, case, coordinates, cv, files=None, masses=None, expected_error=None
):
    case.mkdir(parents=True, exist_ok=True)
    coordinates = np.asarray(coordinates, dtype=np.float32)
    count = len(coordinates)
    if masses is None:
        masses = np.full(count, 12.0)
    for name, values in [("mass", masses), ("charge", np.zeros(count))]:
        (case / f"{name}.txt").write_text(
            str(count) + "\n" + "\n".join(map(str, values)) + "\n"
        )
    (case / "coordinate.txt").write_text(
        str(count)
        + "\n"
        + "\n".join(" ".join(map(str, row)) for row in coordinates)
        + "\n999 999 999 90 90 90\n"
    )
    for name, values in (files or {}).items():
        np.savetxt(case / name, values, fmt="%.9g")
    (case / "cv.txt").write_text(cv)
    # Two identical frames also exercise cached CV state on successive steps.
    np.stack([coordinates, coordinates]).tofile(case / "frames.dat")
    np.savetxt(case / "frames.box", [[999, 999, 999, 90, 90, 90]] * 2)
    settings = {
        "mode": "rerun",
        "step_limit": 3,
        "rerun_frame_limit": 2,
        "rerun_start": 0,
        "rerun_strip": 0,
        "pbc": False,
        "cutoff": 999.0,
        "skin": 2.0,
        "mass_in_file": "mass.txt",
        "charge_in_file": "charge.txt",
        "coordinate_in_file": "coordinate.txt",
        "cv_in_file": "cv.txt",
        "crd": "frames.dat",
        "box": "frames.box",
        "frc": "forces.dat",
        "print_zeroth_frame": 1,
        "write_mdout_interval": 1,
        "write_information_interval": 1,
        "write_trajectory_interval": 1,
        "dont_check_input": 1,
    }
    (case / "mdin.spg.toml").write_text(
        "\n".join(
            f"{key} = {json.dumps(value)}" for key, value in settings.items()
        )
        + "\n"
    )
    result = subprocess.run(
        [str(binary), "-mdin", "mdin.spg.toml"],
        cwd=case,
        env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"),
        capture_output=True,
        text=True,
        timeout=120,
    )
    (case / "execution.log").write_text(result.stdout + result.stderr)
    if expected_error is not None:
        assert result.returncode != 0, f"invalid input accepted in {case}"
        assert expected_error in result.stdout + result.stderr
        return
    if result.returncode:
        raise AssertionError(
            f"SPONGE failed in {case}:\n{result.stdout}{result.stderr}"
        )
    lines = (case / "mdout.txt").read_text().splitlines()
    rows = [
        dict(zip(lines[0].split(), map(float, line.split())))
        for line in lines[1:]
        if line.strip()
    ]
    forces = np.fromfile(case / "forces.dat", dtype=np.float32)
    assert len(rows) >= 2, f"missing CV frames in {case}"
    # Rerun writes the final force snapshot, while mdout contains both frames.
    assert forces.size == count * 3, f"missing force snapshot in {case}"
    forces = forces.reshape(1, count, 3)
    assert np.isfinite(forces).all(), f"non-finite bias force in {case}"
    return rows, forces


def check_scalar(rows, name, expected):
    # mdout prints four decimal places; force checks use binary float output.
    np.testing.assert_allclose(
        [row[name] for row in rows], expected, atol=6e-5, rtol=0, err_msg=name
    )


def centers(binary, output):
    count = 132
    coordinates = np.array(
        [
            [2 * (i % 11), 2 * ((i // 11) % 7), -20 + 2 * (i // 77)]
            for i in range(count)
        ],
        dtype=np.float32,
    )
    masses = np.array([1 + i % 17 for i in range(count)], dtype=np.float64)
    files, definitions, expected = {}, "", {}
    force = np.zeros_like(coordinates, dtype=np.float64)
    for size in [1, 32, 33, 64, 65, count]:
        files[f"atoms{size}.txt"] = np.arange(size)
        for kind in ["center", "center_of_mass"]:
            name = f"{'w' if kind == 'center' else 'm'}{size}"
            weights = masses[:size] / masses[:size].sum()
            args = dict(vatom_type=kind, atom_in_file=f"atoms{size}.txt")
            if kind == "center":
                # Custom weights need not sum to one.
                weights = weights * 1.75
                files[f"weights{size}.txt"] = weights
                args["weight_in_file"] = f"weights{size}.txt"
            definitions += block(name, **args)
            center = weights @ coordinates[:size]
            for axis, value in zip("xyz", center):
                cv_name = f"{name}{axis}"
                definitions += block(
                    cv_name, CV_type=f"position_{axis}", atom=name
                )
                expected[cv_name] = value
            if size == count:
                force[:, 2] += -4 * center[2] * weights
    cv = block("print", CV=" ".join(expected)) + definitions
    cv += block(
        "restrain", CV=f"w{count}z m{count}z", weight="2 2", reference="0 0"
    )
    rows, forces = run_case(binary, output, coordinates, cv, files, masses)
    for name, value in expected.items():
        check_scalar(rows, name, value)
    np.testing.assert_allclose(
        forces, np.broadcast_to(force, forces.shape), atol=2e-5, rtol=2e-6
    )


def rmsd_case(binary, case, coordinates, reference, rotate):
    count = len(coordinates)
    x = coordinates.astype(np.float64) - coordinates.mean(
        axis=0, dtype=np.float64
    )
    y = reference.astype(np.float64) - reference.mean(axis=0, dtype=np.float64)
    if rotate:
        u, _, vt = np.linalg.svd(y.T @ x)
        correction = np.diag([1.0, 1.0, np.linalg.det(u @ vt)])
        y = y @ (u @ correction @ vt)
    difference = x - y
    expected = np.sqrt(np.mean(np.sum(difference**2, axis=1)))
    # E = 2 * RMSD**2 is smooth even at RMSD=0; its force must be zero there.
    force = -4 * difference / count
    cv = (
        block("print", CV="rr")
        + block(
            "rr",
            CV_type="rmsd",
            atom_in_file="atoms.txt",
            coordinate_in_file="reference.txt",
            rotate=int(rotate),
        )
        + block("restrain", CV="rr", weight=2, reference=0)
    )
    rows, forces = run_case(
        binary,
        case,
        coordinates,
        cv,
        {"atoms.txt": np.arange(count), "reference.txt": reference},
    )
    check_scalar(rows, "rr", expected)
    np.testing.assert_allclose(
        forces, np.broadcast_to(force, forces.shape), atol=5e-6, rtol=2e-4
    )


def rmsd_zero(binary, output):
    coordinates = np.array(
        [[-1, -1, -1], [1, 1, -1], [1, -1, 1], [-1, 1, 1]], dtype=np.float32
    )
    rmsd_case(binary, output, coordinates, coordinates.copy(), False)


def rmsd_rotation(binary, output):
    for count, rotate, reflect in [
        (33, False, False),
        (1025, False, False),
        (1025, True, False),
        (33, True, True),
    ]:
        coordinates = np.array(
            [
                [
                    50 + 2 * (i % 11),
                    50 + 2 * ((i // 11) % 7),
                    50 + 2 * (i // 77),
                ]
                for i in range(count)
            ],
            dtype=np.float32,
        )
        if reflect:
            # A full-rank geometry requires the proper-rotation sign correction.
            coordinates[:, 2] += np.arange(count) % 5
        reference = coordinates.copy()
        reference[0] += [0.5, -0.25, 0.125]
        reference[-1] += [-0.125, 0.375, -0.5]
        if reflect:
            reference[:, 0] *= -1
        rmsd_case(
            binary,
            output / f"n{count}-rotate{rotate}-reflect{reflect}",
            coordinates,
            reference,
            rotate,
        )


def spline_reference(x, low, high, values, padding):
    delta = (high - low) / (len(values) - 1)
    t = (x - low) / delta
    interval = min(int(t), len(values) - 2)
    u = t - interval
    points = [padding[0], *values, padding[1]][interval : interval + 4]
    basis = [
        (1 - u) ** 3 / 6,
        (3 * u**3 - 6 * u * u + 4) / 6,
        (-3 * u**3 + 3 * u * u + 3 * u + 1) / 6,
        u**3 / 6,
    ]
    derivative = [
        -((1 - u) ** 2) / 2,
        (9 * u * u - 12 * u) / 6,
        (-9 * u * u + 6 * u + 3) / 6,
        u * u / 2,
    ]
    return np.dot(points, basis), np.dot(points, derivative) / delta


def tabulated(binary, output):
    for values, low, high in [
        ([-2, 1, 4, 2], 0, 3),
        ([1, 4], -1, 2),
        ([-2, 1, 4, 2], -0.25, 1.75),
    ]:
        for x in [low, (low + high) / 2, high - 1e-4, high]:
            x = float(np.float32(x))
            padding = (-5, 7)
            value, derivative = spline_reference(x, low, high, values, padding)
            cv = block("print", CV="tab") + block(
                "px", CV_type="position_x", atom=0
            )
            cv += block(
                "tab",
                CV_type="tabulated",
                CV="px",
                min=low,
                max=high,
                parameter=" ".join(map(str, values)),
                min_padding=padding[0],
                max_padding=padding[1],
            )
            cv += block("restrain", CV="tab", weight=2, reference=0)
            coordinates = [[x, 0, 0], [4, 1, 0], [8, 0, 1], [12, 1, 1]]
            rows, forces = run_case(
                binary, output / f"n{len(values)}-lo{low}-x{x}", coordinates, cv
            )
            check_scalar(rows, "tab", value)
            force = np.zeros((4, 3))
            force[0, 0] = -4 * value * derivative
            np.testing.assert_allclose(
                forces,
                np.broadcast_to(force, forces.shape),
                atol=3e-5,
                rtol=2e-5,
            )

    for index, (parameters, low, high) in enumerate(
        [("1", 0, 1), ("1 2", 1, 1), ("1 2", 2, 1)]
    ):
        cv = block("print", CV="tab") + block(
            "px", CV_type="position_x", atom=0
        )
        cv += block(
            "tab",
            CV_type="tabulated",
            CV="px",
            min=low,
            max=high,
            parameter=parameters,
        )
        run_case(
            binary,
            output / f"invalid{index}",
            [[0, 0, 0], [4, 1, 0], [8, 0, 1], [12, 1, 1]],
            cv,
            expected_error="tabulated CV requires at least two parameters",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sponge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--case",
        choices=["centers", "rmsd_zero", "rmsd_rotation", "tabulated"],
        required=True,
    )
    args = parser.parse_args()
    globals()[args.case](args.sponge.resolve(), args.output.resolve())
    print(f"PASS: {args.case}")

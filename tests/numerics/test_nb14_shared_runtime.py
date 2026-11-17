"""Shared-atom nb14 accumulation and input permutations; dt=0, no sampling.

Compare actual kernel output against a double finite-difference energy oracle.
Input permutations exercise atom/pair identity; they are not an MPI/DD test.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import numpy as np

PAIRS = [(0, 1), (0, 2), (0, 3), (2, 3)]
XYZ = np.array(
    [[3.0, 3.0, 3.0], [5.0, 3.6, 3.2], [3.5, 5.2, 4.0], [4.2, 3.8, 5.5]]
)
BAR = 6.946827162543585e4


def parameters(weight):
    weights = np.array([1.0, weight, weight, weight])
    sigma = np.array([1.5, 1.7, 1.9, 1.6])
    c6 = 4 * np.array([0.15, 0.2, 0.11, 0.18]) * sigma**6 * weights
    c12 = c6 * sigma**6
    delta = 0.5 * (1 - weights) ** 2 * sigma**6
    charge = np.array(
        [0.7, -0.4, 0.2 if weight == 1 else 0.0, -0.5 if weight == 1 else 0.0]
    )
    return c12, c6, delta, charge


def energy(xyz, weight, electrostatic):
    c12, c6, delta, charge = parameters(weight)
    lj = ee = 0.0
    for k, (i, j) in enumerate(PAIRS):
        r = np.linalg.norm(xyz[j] - xyz[i])
        s = r**6 + delta[k]
        lj += c12[k] / s**2 - c6[k] / s
        ee += electrostatic * 0.5 * charge[i] * charge[j] / r
    return np.array([lj, ee])


def oracle(weight, electrostatic):
    force = np.empty_like(XYZ)
    for i in range(4):
        for d in range(3):
            plus, minus = XYZ.copy(), XYZ.copy()
            plus[i, d] += 1.0e-5
            minus[i, d] -= 1.0e-5
            force[i, d] = (
                -(
                    energy(plus, weight, electrostatic).sum()
                    - energy(minus, weight, electrostatic).sum()
                )
                / 2.0e-5
            )
    return (
        energy(XYZ, weight, electrostatic),
        force,
        XYZ.T @ force / 32**3 * BAR,
    )


def run(engine, root, weight, electrostatic, permutation, baseline=None):
    root.mkdir()
    order = np.array([2, 0, 3, 1] if permutation else range(4))
    inverse = np.argsort(order)
    pair_order = [3, 1, 0, 2] if permutation else range(4)
    c12, c6, delta, charge = parameters(weight)
    if baseline is None:
        c12 = np.zeros_like(c12)
        c6 = np.zeros_like(c6)
        electrostatic = 0.0
    files = {
        "mass": "4\n" + "12\n" * 4,
        "charge": "4\n" + "\n".join(map(str, charge[order])) + "\n",
        "coordinate": "4\n"
        + "\n".join(" ".join(map(str, row)) for row in XYZ[order])
        + "\n32 32 32\n90 90 90\n",
        "velocity": "4\n" + "0 0 0\n" * 4,
        "nb14_extra": "4\n"
        + "".join(
            f"{inverse[PAIRS[k][0]]} {inverse[PAIRS[k][1]]} {c12[k]:.17g} {c6[k]:.17g} {electrostatic * 0.5}\n"
            for k in pair_order
        ),
        "nb14_softcore": "4\n"
        + "".join(
            f"{inverse[PAIRS[k][0]]} {inverse[PAIRS[k][1]]} {delta[k]:.17g}\n"
            for k in pair_order
        ),
    }
    config = dict(
        md_name="nb14-shared",
        mode="nve",
        dt=0.0,
        step_limit=10,
        cutoff=8.0,
        print_zeroth_frame=True,
        print_pressure=True,
        write_mdout_interval=10,
        write_information_interval=10,
        write_trajectory_interval=10,
        write_restart_file_interval=0,
        frc="force.dat",
    )
    config["PM.MPI_size"] = 1
    for name, text in files.items():
        (root / (name + ".txt")).write_text(text)
        config[name + "_in_file"] = name + ".txt"
    (root / "mdin.spg.toml").write_text(
        "\n".join(f"{k}={json.dumps(v)}" for k, v in config.items()) + "\n"
    )
    result = subprocess.run(
        [str(engine), "-mdin", "mdin.spg.toml"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    (root / "engine.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, str(root)
    force = np.fromfile(root / "force.dat", dtype=np.float32).reshape(-1, 4, 3)[
        :, inverse, :
    ]
    lines = (root / "mdout.txt").read_text().splitlines()
    index = next(
        i
        for i, l in enumerate(lines)
        if "nb14_LJ" in l.split() and "Pxx" in l.split()
    )
    keys = lines[index].split()
    values = next(
        dict(zip(keys, map(float, l.split())))
        for l in lines[index + 1 :]
        if len(l.split()) == len(keys)
    )
    if baseline is None:
        return force, values
    # Ordinary/PME terms are present even without an LJ table. Identical
    # charges, coordinates and PME settings cancel in the zero-nb14 control.
    baseline_force, baseline_values = baseline
    force = force - baseline_force
    expected, expected_force, pressure = oracle(weight, electrostatic)
    np.testing.assert_allclose(
        force,
        np.broadcast_to(expected_force, force.shape),
        rtol=3.0e-4,
        atol=3.0e-4,
    )
    assert len(force) >= 2
    np.testing.assert_allclose(force[0], force[-1], rtol=1.0e-6, atol=1.0e-6)
    assert abs(values["nb14_LJ"] - expected[0]) < 0.011 + 3.0e-4 * abs(
        expected[0]
    ), values
    assert abs(values["nb14_EE"] - expected[1]) < 0.011 + 3.0e-4 * abs(
        expected[1]
    ), values
    for name, i, j in [
        ("Pxx", 0, 0),
        ("Pyy", 1, 1),
        ("Pzz", 2, 2),
        ("Pxy", 0, 1),
        ("Pxz", 0, 2),
        ("Pyz", 1, 2),
    ]:
        assert abs(
            (values[name] - baseline_values[name]) - pressure[i, j]
        ) < 0.011 + 3.0e-4 * abs(pressure[i, j]), (
            name,
            values[name],
            pressure[i, j],
        )
    return dict(
        weight=weight,
        electrostatic=electrostatic,
        permuted=permutation,
        maxForceError=float(np.max(abs(force - expected_force))),
        pressure=pressure.tolist(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sponge", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    engine = args.sponge.resolve()
    temporary = (
        tempfile.TemporaryDirectory(prefix="nb14-shared-")
        if args.output is None
        else None
    )
    if temporary is not None:
        args.output = Path(temporary.name) / "cases"
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for weight in (0.0, 0.2, 1.0):
        for permute in (False, True):
            baseline = run(
                engine,
                args.output / f"baseline-{weight}-{permute}",
                weight,
                0.0,
                permute,
            )
            for ee in (0.0, 1.0):
                records.append(
                    run(
                        engine,
                        args.output / str(len(records)),
                        weight,
                        ee,
                        permute,
                        baseline,
                    )
                )
    report = dict(
        engine=str(engine),
        binarySha256=hashlib.sha256(engine.read_bytes()).hexdigest(),
        scope="shared atoms; charged native14; six RAW pressure components; input permutations; no MPI/DD or sampling",
        records=records,
    )
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(f"{len(records)} shared-atom nb14 cases passed")


if __name__ == "__main__":
    main()

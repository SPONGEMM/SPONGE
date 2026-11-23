"""Create distinct dry ACE-ALA-NME starting conformers by backbone rotations."""

import argparse
from pathlib import Path

import numpy as np

from alanine_study import parm_fields, periodic_dihedral, structure


def angle(coordinates, atoms):
    return periodic_dihedral(coordinates[atoms], (1e6,) * 3)


def rotate(coordinates, bonds, atoms, target):
    _, fixed, moving, _ = atoms
    affected = {moving}
    pending = [moving]
    while pending:
        for neighbor in bonds[pending.pop()]:
            if neighbor != fixed and neighbor not in affected:
                affected.add(neighbor)
                pending.append(neighbor)
    axis = coordinates[moving] - coordinates[fixed]
    axis /= np.linalg.norm(axis)
    offset = np.radians(target - angle(coordinates, atoms))
    indices = sorted(affected)
    points = coordinates[indices] - coordinates[fixed]
    coordinates[indices] = (
        coordinates[fixed]
        + np.cos(offset) * points
        + np.sin(offset) * np.cross(axis, points)
        + (1 - np.cos(offset)) * (points @ axis)[:, None] * axis
    )


def generate(source):
    geom = structure(source, implicit=True)
    count = geom["atom_count"]
    fields = parm_fields(source / "system.parm7")
    topology = fields["BONDS_INC_HYDROGEN"] + fields["BONDS_WITHOUT_HYDROGEN"]
    bonds = [set() for _ in range(count)]
    for i in range(0, len(topology), 3):
        a, b = (int(x) // 3 for x in topology[i : i + 2])
        bonds[a].add(b)
        bonds[b].add(a)

    lines = (source / "system.rst7").read_text().splitlines()
    values = np.fromstring(" ".join(lines[2:]), sep=" ")
    if values.size != 3 * count + 6:
        raise ValueError(
            "expected coordinates and an Amber box without velocities"
        )
    for name, phi, psi in (
        ("ppii", -75, 140),
        ("alpha", -70, -25),
        ("positive", 60, 35),
    ):
        crd = values[: 3 * count].reshape(-1, 3).copy()
        for atoms, target in ((geom["phi"], phi), (geom["psi"], psi)):
            rotate(crd, bonds, atoms, target)
        if not np.allclose(
            (angle(crd, geom["phi"]), angle(crd, geom["psi"])), (phi, psi)
        ):
            raise ValueError(f"incorrect generated dihedrals for {name}")
        data = np.r_[crd.ravel(), values[-6:]]
        rows = [
            "".join(f"{value:12.7f}" for value in data[i : i + 6])
            for i in range(0, data.size, 6)
        ]
        (source / f"{name}.rst7").write_text(
            f"{name}\n{count:6d}\n" + "\n".join(rows) + "\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    generate(parser.parse_args().source)

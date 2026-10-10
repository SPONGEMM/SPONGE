"""Optional atomtype columns and type-based distance constraints."""

import math
import os
import subprocess

import h5py
import numpy as np
import pytest
from comp_direct_topgro_section_integrity import (
    _extract_mdout_term,
    _minimal_topology,
    _run_topology,
)


@pytest.mark.parametrize(
    "row",
    [
        "A 12.0 0.0 A 0.30 0.4184",
        "A 6 12.0 0.0 A 0.30 0.4184",
        "A BA 12.0 0.0 A 0.30 0.4184",
        "A BA 6 12.0 0.0 A 0.30 0.4184",
    ],
)
def test_atomtypes_optional_columns(tmp_path, row):
    topology = _minimal_topology().replace("A A 12.0 0.0 A 0.30 0.4184", row)
    result, _ = _run_topology(tmp_path, topology)
    assert result.returncode == 0, result.stdout + result.stderr


def _constraint_topology(
    *,
    funct=1,
    type_funct=None,
    inline="",
    reverse=False,
    aliases=False,
    missing=False,
    duplicate=False,
):
    type_funct = funct if type_funct is None else type_funct
    ai, aj = ("BA", "BB") if aliases else ("A", "B")
    if reverse:
        ai, aj = aj, ai
    types = f"{ai} {aj} {type_funct} 0.15"
    if missing:
        types = f"{ai} UNKNOWN {type_funct} 0.15"
    if duplicate:
        types += f"\n{ai} {aj} {type_funct} 0.16"
    a = "A BA 6 12.0 0.0 A 0.30 4.184" if aliases else "A 12.0 0.0 A 0.30 4.184"
    b = "B BB 12.0 0.0 A 0.30 4.184" if aliases else "B 12.0 0.0 A 0.30 4.184"
    return f"""
[ defaults ]
1 2 yes 1.0 1.0
[ atomtypes ]
{a}
{b}
[ constrainttypes ]
{types}
[ moleculetype ]
PAIR 1
[ atoms ]
1 A 1 ONE A 1 0.0 12.0
2 B 1 ONE B 2 0.0 12.0
[ constraints ]
1 2 {funct} {inline}
[ system ]
constraint types
[ molecules ]
PAIR 1
"""


@pytest.mark.parametrize("funct", [1, 2])
@pytest.mark.parametrize(
    "reverse,aliases", [(False, False), (True, False), (True, True)]
)
def test_constrainttypes_resolve_pairs_and_preserve_exclusions(
    tmp_path, funct, reverse, aliases
):
    result, mdout = _run_topology(
        tmp_path,
        _constraint_topology(funct=funct, reverse=reverse, aliases=aliases),
        atoms=(("A", 0.0), ("B", 0.35)),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    x = (0.30 / 0.35) ** 6
    expected = 0.0 if funct == 1 else 4.0 * (x * x - x)
    assert math.isclose(
        _extract_mdout_term(mdout, "LJ_short"), expected, abs_tol=0.01
    )


@pytest.mark.parametrize(
    "options", [{"missing": True}, {"type_funct": 2}, {"funct": 3}]
)
def test_unresolved_or_unsupported_constraint_fails(tmp_path, options):
    result, _ = _run_topology(
        tmp_path,
        _constraint_topology(**options),
        atoms=(("A", 0.0), ("B", 0.15)),
    )
    assert result.returncode != 0
    assert "spongeErrorBadFileFormat" in result.stdout + result.stderr


@pytest.mark.parametrize(
    "options,expected",
    [
        ({"inline": "0.16", "missing": True}, 1.6),
        ({"duplicate": True}, 1.6),
        ({"reverse": True}, 1.5),
    ],
)
def test_constraint_distance_is_applied(tmp_path, options, expected):
    # First write a valid launch with the shared fixture, then advance one
    # constrained step and inspect its actual coordinates.
    result, _ = _run_topology(
        tmp_path,
        _constraint_topology(**options),
        atoms=(("A", 0.0), ("B", 0.15)),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    mdin = tmp_path / "mdin.spg.toml"
    text = mdin.read_text().replace("step_limit = 0", "step_limit = 1")
    text = text.replace("dt = 0", "dt = 0.00001")
    text += '\nconstrain_mode = "SHAKE"\nwrite_trajectory_interval = 1\n'
    text += (
        f'output_h5_trajectory_path = "{tmp_path / "trajectory.spg.h5md"}"\n'
    )
    mdin.write_text(text)
    result = subprocess.run(
        [os.environ.get("SPONGE_BIN", "SPONGE"), "-mdin", str(mdin)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    with h5py.File(tmp_path / "trajectory.spg.h5md", "r") as f:
        xyz = f["particles/all/position/value"][-1]
    assert np.linalg.norm(xyz[1] - xyz[0]) == pytest.approx(expected, abs=0.001)


@pytest.mark.parametrize("optional", ["BA", "BA 6"])
def test_bonded_aliases_apply_to_all_bonded_parameter_tables(
    tmp_path, optional
):
    topology = """
[ defaults ]
1 2 yes 1.0 1.0
[ atomtypes ]
A {optional} 12.0 0.0 A 0.0 0.0
[ bondtypes ]
BA BA 1 0.15 418.4
[ angletypes ]
BA BA BA 1 100.0 8.368
[ dihedraltypes ]
BA BA BA BA 1 30.0 4.184 1
[ cmaptypes ]
BA BA BA BA BA 1 2 2 4.184 4.184 4.184 4.184
[ moleculetype ]
FIVE 3
[ atoms ]
1 A 1 ONE A1 1 0.0 12.0
2 A 1 ONE A2 2 0.0 12.0
3 A 1 ONE A3 3 0.0 12.0
4 A 1 ONE A4 4 0.0 12.0
5 A 1 ONE A5 5 0.0 12.0
[ bonds ]
1 2 1
[ angles ]
1 2 3 1
[ dihedrals ]
1 2 3 4 1
[ cmap ]
1 2 3 4 5 1
[ system ]
bonded aliases
[ molecules ]
FIVE 1
""".format(optional=optional)
    atoms = (
        ("A1", (0.0, 0.1, 0.0)),
        ("A2", (0.0, 0.0, 0.0)),
        ("A3", (0.1, 0.0, 0.0)),
        ("A4", (0.1, 0.0, 0.1)),
        ("A5", (0.2, 0.1, 0.15)),
    )
    result, aliased_mdout = _run_topology(
        tmp_path / "aliased", topology, atoms=atoms
    )
    reference = topology.replace("A " + optional + " 12.0", "A 6 12.0").replace(
        "BA", "A"
    )
    reference_result, reference_mdout = _run_topology(
        tmp_path / "reference", reference, atoms=atoms
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert reference_result.returncode == 0, (
        reference_result.stdout + reference_result.stderr
    )
    for term in ("bond", "urey_bradley", "dihedral", "cmap", "potential"):
        assert _extract_mdout_term(aliased_mdout, term) == pytest.approx(
            _extract_mdout_term(reference_mdout, term), abs=0.01
        )

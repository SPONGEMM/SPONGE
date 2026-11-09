import math
import os
import subprocess
import textwrap
from pathlib import Path

import pytest

_PERMUTED_TWO_TYPE_NPI = [3, 1, 1, 2]


def _flag(name, fmt, values, items_per_line=10):
    lines = [f"%FLAG {name}", f"%FORMAT({fmt})"]
    rendered = [str(value) for value in values]
    for start in range(0, len(rendered), items_per_line):
        lines.append(" ".join(rendered[start : start + items_per_line]))
    return "\n".join(lines) + "\n"


def _write_prmtop(
    path,
    *,
    atom_types,
    normal_a,
    normal_b,
    nonbonded_parm_index=None,
    hbond_a=None,
    hbond_b=None,
    excluded_counts=None,
    excluded_list=(),
    dihedral_rows=(),
    lj14_a=None,
    lj14_b=None,
    charges=None,
    scee=1.0,
    scnb=2.0,
):
    atom_types = list(atom_types)
    atom_count = len(atom_types)
    atom_type_count = max(atom_types)
    if nonbonded_parm_index is None:
        nonbonded_parm_index = _PERMUTED_TWO_TYPE_NPI
    if excluded_counts is None:
        excluded_counts = [0] * atom_count
    hbond_type_count = max(
        len(hbond_a) if hbond_a is not None else 0,
        len(hbond_b) if hbond_b is not None else 0,
    )

    pointers = [0] * 31
    pointers[0] = atom_count  # NATOM
    pointers[1] = atom_type_count  # NTYPES
    pointers[7] = len(dihedral_rows)  # MPHIA
    pointers[10] = len(excluded_list)  # NNB
    pointers[11] = 1  # NRES
    pointers[14] = len(dihedral_rows)  # NPHIA
    pointers[17] = 1 if dihedral_rows else 0  # NPTRA
    pointers[19] = hbond_type_count  # NPHB
    pointers[27] = 1  # IFBOX

    sections = [
        "%VERSION VERSION_STAMP = V0001.000 DATE = 01/01/70 00:00:00\n",
        _flag("POINTERS", "10I8", pointers),
        _flag("CHARGE", "5E16.8", charges or [0.0] * atom_count, 5),
        _flag("MASS", "5E16.8", [12.0] * atom_count, 5),
        _flag("ATOM_TYPE_INDEX", "10I8", atom_types),
        _flag(
            "NONBONDED_PARM_INDEX",
            "10I8",
            nonbonded_parm_index,
        ),
        _flag("NUMBER_EXCLUDED_ATOMS", "10I8", excluded_counts),
        _flag("RESIDUE_POINTER", "10I8", [1]),
    ]
    if hbond_a is not None:
        sections.append(_flag("HBOND_ACOEF", "5E16.8", hbond_a, 5))
    if hbond_b is not None:
        sections.append(_flag("HBOND_BCOEF", "5E16.8", hbond_b, 5))
    if dihedral_rows:
        sections.extend(
            [
                _flag("DIHEDRAL_FORCE_CONSTANT", "5E16.8", [0.0], 5),
                _flag("DIHEDRAL_PERIODICITY", "5E16.8", [1.0], 5),
                _flag("DIHEDRAL_PHASE", "5E16.8", [0.0], 5),
            ]
        )
    if dihedral_rows and scee is not None:
        sections.append(_flag("SCEE_SCALE_FACTOR", "5E16.8", [scee], 5))
    if dihedral_rows and scnb is not None:
        sections.append(_flag("SCNB_SCALE_FACTOR", "5E16.8", [scnb], 5))
    sections.extend(
        [
            _flag("LENNARD_JONES_ACOEF", "3E24.16", normal_a, 3),
            _flag("LENNARD_JONES_BCOEF", "3E24.16", normal_b, 3),
        ]
    )
    if lj14_a is not None:
        sections.append(_flag("LENNARD_JONES_14_ACOEF", "3E24.16", lj14_a, 3))
    if lj14_b is not None:
        sections.append(_flag("LENNARD_JONES_14_BCOEF", "3E24.16", lj14_b, 3))
    if dihedral_rows:
        sections.append(
            _flag(
                "DIHEDRALS_WITHOUT_HYDROGEN",
                "10I8",
                [value for row in dihedral_rows for value in row],
            )
        )
    sections.append(_flag("EXCLUDED_ATOMS_LIST", "10I8", excluded_list))
    path.write_text("".join(sections))


def _write_rst7(path, coordinates):
    flat_coordinates = [
        value for coordinate in coordinates for value in coordinate
    ]
    coordinate_lines = []
    for start in range(0, len(flat_coordinates), 6):
        coordinate_lines.append(
            "".join(
                f"{value:12.7f}"
                for value in flat_coordinates[start : start + 6]
            )
        )
    path.write_text(
        "\n".join(
            [
                "synthetic AMBER NONBONDED_PARM_INDEX test",
                f"{len(coordinates):6d}",
                *coordinate_lines,
                "  50.0000000  50.0000000  50.0000000",
                "  90.0000000  90.0000000  90.0000000",
            ]
        )
        + "\n"
    )


def _write_mdin(path, prmtop_path, rst7_path, mdout_path):
    path.write_text(
        textwrap.dedent(
            f"""
            md_name = "synthetic_amber_nb14"
            mode = "nve"
            step_limit = 0
            dt = 0
            cutoff = 8.0
            amber_parm7 = {str(prmtop_path)!r}
            amber_rst7 = {str(rst7_path)!r}
            mdout = {str(mdout_path)!r}
            print_zeroth_frame = 1
            write_mdout_interval = 1
            """
        ).strip()
        + "\n"
    )


def _run_sponge(tmp_path, coordinates, **prmtop_options):
    prmtop_path = tmp_path / "synthetic.prmtop"
    rst7_path = tmp_path / "synthetic.rst7"
    mdin_path = tmp_path / "mdin.spg.toml"
    mdout_path = tmp_path / "mdout.txt"
    _write_prmtop(prmtop_path, **prmtop_options)
    _write_rst7(rst7_path, coordinates)
    _write_mdin(mdin_path, prmtop_path, rst7_path, mdout_path)
    result = subprocess.run(
        [os.environ.get("SPONGE_BIN", "SPONGE"), "-mdin", str(mdin_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    return result, mdout_path


def _extract_mdout_term(mdout_path, term_name):
    lines = Path(mdout_path).read_text().splitlines()
    headers = lines[0].split()
    values = lines[1].split()
    assert len(headers) == len(values)
    return float(dict(zip(headers, values))[term_name])


def test_nonbonded_parm_index_remaps_ordinary_lj_matrix(tmp_path):
    result, mdout_path = _run_sponge(
        tmp_path,
        [(0.0, 0.0, 0.0), (math.sqrt(3.0), 0.0, 0.0)],
        atom_types=[1, 2],
        # The cross term is source entry 1, not canonical array slot 2.
        normal_a=[729.0, 200.0, 300.0],
        normal_b=[54.0, 20.0, 30.0],
    )
    output = result.stdout + "\n" + result.stderr

    assert result.returncode == 0, output
    assert math.isclose(
        _extract_mdout_term(mdout_path, "LJ_short"), -1.0, abs_tol=0.01
    )


def test_zero_hbond_placeholder_maps_to_zero_lj(tmp_path):
    result, mdout_path = _run_sponge(
        tmp_path,
        [(0.0, 0.0, 0.0), (math.sqrt(3.0), 0.0, 0.0)],
        atom_types=[1, 2],
        normal_a=[729.0, 200.0, 300.0],
        normal_b=[54.0, 20.0, 30.0],
        nonbonded_parm_index=[3, -1, -1, 2],
        hbond_a=[0.0],
        hbond_b=[0.0],
    )
    output = result.stdout + "\n" + result.stderr

    assert result.returncode == 0, output
    assert math.isclose(
        _extract_mdout_term(mdout_path, "LJ_short"), 0.0, abs_tol=0.01
    )


def test_nonzero_hbond_term_fails_closed(tmp_path):
    result, _mdout_path = _run_sponge(
        tmp_path,
        [(0.0, 0.0, 0.0), (math.sqrt(3.0), 0.0, 0.0)],
        atom_types=[1, 2],
        normal_a=[729.0, 200.0, 300.0],
        normal_b=[54.0, 20.0, 30.0],
        nonbonded_parm_index=[3, -1, -1, 2],
        hbond_a=[1.0],
        hbond_b=[0.0],
    )
    output = result.stdout + "\n" + result.stderr

    assert result.returncode != 0
    assert "AMBER 10-12 nonbonded terms are not supported" in output


_FOUR_ATOM_COORDINATES = [
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (1.0, 0.0, 1.0),
]


def _four_atom_nb14_options(**overrides):
    options = {
        "atom_types": [1, 1, 2, 2],
        "normal_a": [729.0, 200.0, 300.0],
        "normal_b": [27.0, 20.0, 30.0],
        "lj14_a": [729.0, 400.0, 500.0],
        "lj14_b": [54.0, 40.0, 50.0],
        "dihedral_rows": [(0, 3, 6, 9, 1)],
        "excluded_counts": [3, 2, 1, 0],
        "excluded_list": [2, 3, 4, 3, 4, 4],
    }
    options.update(overrides)
    return options


def test_chamber_lj14_matrix_uses_nonbonded_parm_index(tmp_path):
    result, mdout_path = _run_sponge(
        tmp_path,
        _FOUR_ATOM_COORDINATES,
        **_four_atom_nb14_options(),
    )
    output = result.stdout + "\n" + result.stderr

    assert result.returncode == 0, output
    assert "non-bond 14 numbers is 1" in output
    assert math.isclose(
        _extract_mdout_term(mdout_path, "nb14_LJ"), -0.5, abs_tol=0.01
    )


def test_missing_chamber_lj14_matrix_falls_back_to_ordinary_lj(tmp_path):
    result, mdout_path = _run_sponge(
        tmp_path,
        _FOUR_ATOM_COORDINATES,
        **_four_atom_nb14_options(
            normal_b=[54.0, 20.0, 30.0],
            lj14_a=None,
            lj14_b=None,
        ),
    )
    output = result.stdout + "\n" + result.stderr

    assert result.returncode == 0, output
    assert math.isclose(
        _extract_mdout_term(mdout_path, "nb14_LJ"), -0.5, abs_tol=0.01
    )


def test_chamber_lj14_coefficients_must_be_complete(tmp_path):
    result, _mdout_path = _run_sponge(
        tmp_path,
        _FOUR_ATOM_COORDINATES,
        **_four_atom_nb14_options(lj14_b=None),
    )
    output = result.stdout + "\n" + result.stderr

    assert result.returncode != 0
    assert (
        "LENNARD_JONES_14_ACOEF and LENNARD_JONES_14_BCOEF must either both "
        "be present or both be absent"
    ) in output


@pytest.mark.parametrize(
    "scee,scnb",
    [(None, None), (None, 4.0), (2.4, None), (2.4, 4.0), (0.0, 0.0)],
)
def test_nb14_missing_scale_factors_use_amber_defaults(tmp_path, scee, scnb):
    result, mdout_path = _run_sponge(
        tmp_path,
        _FOUR_ATOM_COORDINATES,
        **_four_atom_nb14_options(
            charges=[1.0, 0.0, 0.0, 1.0],
            scee=scee,
            scnb=scnb,
        ),
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    lj_divisor = 2.0 if scnb is None else scnb
    cf_divisor = 1.2 if scee is None else scee
    expected_lj = -1.0 / lj_divisor if lj_divisor else 0.0
    expected_cf = 1.0 / (math.sqrt(3.0) * cf_divisor) if cf_divisor else 0.0
    assert math.isclose(
        _extract_mdout_term(mdout_path, "nb14_LJ"), expected_lj, abs_tol=0.01
    )
    assert math.isclose(
        _extract_mdout_term(mdout_path, "nb14_EE"), expected_cf, abs_tol=0.01
    )

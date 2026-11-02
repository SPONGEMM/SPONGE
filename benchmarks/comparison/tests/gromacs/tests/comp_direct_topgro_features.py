import json
import math
import os
import subprocess
import textwrap
from pathlib import Path

import pytest


def _toml_string(value):
    return json.dumps(os.fspath(value))


def _gro_atom(resid, resname, atomname, atomnr, xyz):
    x, y, z = xyz
    return (
        f"{resid:5d}{resname:<5}{atomname:>5}{atomnr:5d}"
        f"{x:8.3f}{y:8.3f}{z:8.3f}"
    )


def _extract_mdout_term(mdout_path, term):
    lines = mdout_path.read_text().splitlines()
    headers = lines[0].split()
    values = next(
        line.split()
        for line in reversed(lines[1:])
        if line.strip() and line.lstrip()[0].isdigit()
    )
    return float(dict(zip(headers, values))[term])


def test_direct_gromacs_topgro_accepts_settles_constraints_and_cmap(tmp_path):
    top_path = tmp_path / "topol.top"
    gro_path = tmp_path / "conf.gro"
    mdin_path = tmp_path / "mdin.spg.toml"

    top_path.write_text(
        textwrap.dedent(
            """
            [ defaults ]
            1 2 yes 0.5 0.833333

            [ atomtypes ]
            OW OW 15.9994 -0.834 A 0.315075 0.636386
            HW HW 1.008 0.417 A 0.0 0.0
            CT CT 12.011 0.0 A 0.34 0.276144

            [ cmaptypes ]
            CT CT CT CT CT 1 2 2 0.0 0.0 0.0 0.0

            [ moleculetype ]
            SOL 2

            [ atoms ]
            1 OW 1 SOL OW 1 -0.834 15.9994
            2 HW 1 SOL HW1 2 0.417 1.008
            3 HW 1 SOL HW2 3 0.417 1.008

            [ settles ]
            1 1 0.09572 0.15139

            [ moleculetype ]
            LIN 1

            [ atoms ]
            1 CT 1 LIN C1 1 0.0 12.011
            2 CT 1 LIN C2 2 0.0 12.011

            [ constraints ]
            1 2 1 0.150

            [ moleculetype ]
            CMAP 1

            [ atoms ]
            1 CT 1 CMP C1 1 0.0 12.011
            2 CT 1 CMP C2 2 0.0 12.011
            3 CT 1 CMP C3 3 0.0 12.011
            4 CT 1 CMP C4 4 0.0 12.011
            5 CT 1 CMP C5 5 0.0 12.011

            [ cmap ]
            1 2 3 4 5 1

            [ system ]
            direct feature smoke

            [ molecules ]
            SOL 1
            LIN 1
            CMAP 1
            """
        ).strip()
        + "\n"
    )

    gro_lines = [
        "direct feature smoke",
        "10",
        _gro_atom(1, "SOL", "OW", 1, (0.000, 0.000, 0.000)),
        _gro_atom(1, "SOL", "HW1", 2, (0.096, 0.000, 0.000)),
        _gro_atom(1, "SOL", "HW2", 3, (-0.024, 0.093, 0.000)),
        _gro_atom(2, "LIN", "C1", 4, (0.500, 0.000, 0.000)),
        _gro_atom(2, "LIN", "C2", 5, (0.650, 0.000, 0.000)),
        _gro_atom(3, "CMP", "C1", 6, (1.000, 0.000, 0.000)),
        _gro_atom(3, "CMP", "C2", 7, (1.150, 0.050, 0.020)),
        _gro_atom(3, "CMP", "C3", 8, (1.300, 0.000, 0.070)),
        _gro_atom(3, "CMP", "C4", 9, (1.450, -0.040, 0.030)),
        _gro_atom(3, "CMP", "C5", 10, (1.600, 0.020, -0.050)),
        "   5.00000   5.00000   5.00000",
    ]
    gro_path.write_text("\n".join(gro_lines) + "\n")

    mdin_path.write_text(
        textwrap.dedent(
            f"""
            md_name = "direct_gromacs_feature_smoke"
            mode = "nve"
            step_limit = 0
            dt = 0
            cutoff = 8.0
            constrain_mode = "SETTLE"
            gromacs_top = {_toml_string(top_path)}
            gromacs_gro = {_toml_string(gro_path)}
            default_out_file_prefix = {_toml_string(tmp_path / "direct_feature")}
            print_zeroth_frame = 1
            write_mdout_interval = 1
            """
        ).strip()
        + "\n"
    )

    result = subprocess.run(
        [os.environ.get("SPONGE_BIN", "SPONGE"), "-mdin", str(mdin_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )

    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    output = result.stdout + "\n" + result.stderr
    assert "constrain pair number is 4" in output
    assert "rigid triangle numbers is 1" in output
    assert "rigid pair numbers is 1" in output


def test_direct_gromacs_harmonic_improper_uses_half_force_constant(tmp_path):
    top_path = tmp_path / "topol.top"
    gro_path = tmp_path / "conf.gro"
    mdin_path = tmp_path / "mdin.spg.toml"
    mdout_path = tmp_path / "mdout.txt"

    top_path.write_text(
        textwrap.dedent(
            """
            [ defaults ]
            1 2 yes 1.0 1.0

            [ atomtypes ]
            A A 12.0 0.0 A 0.0 0.0
            B B 12.0 0.0 A 0.0 0.0
            C C 12.0 0.0 A 0.0 0.0
            D D 12.0 0.0 A 0.0 0.0

            [ dihedraltypes ]
            A B C D 2 0.0 8.368

            [ moleculetype ]
            IMPLICIT 0

            [ atoms ]
            1 A 1 IMP A 1 0.0 12.0
            2 B 1 IMP B 2 0.0 12.0
            3 C 1 IMP C 3 0.0 12.0
            4 D 1 IMP D 4 0.0 12.0

            [ dihedrals ]
            1 2 3 4 2

            [ moleculetype ]
            EXPLICIT 0

            [ atoms ]
            1 A 1 EXP A 1 0.0 12.0
            2 B 1 EXP B 2 0.0 12.0
            3 C 1 EXP C 3 0.0 12.0
            4 D 1 EXP D 4 0.0 12.0

            [ dihedrals ]
            1 2 3 4 2 0.0 8.368

            [ system ]
            harmonic improper convention

            [ molecules ]
            IMPLICIT 1
            EXPLICIT 1
            """
        ).strip()
        + "\n"
    )

    gro_lines = [
        "harmonic improper convention",
        "8",
        _gro_atom(1, "IMP", "A", 1, (0.000, 0.100, 0.000)),
        _gro_atom(1, "IMP", "B", 2, (0.000, 0.000, 0.000)),
        _gro_atom(1, "IMP", "C", 3, (0.100, 0.000, 0.000)),
        _gro_atom(1, "IMP", "D", 4, (0.100, 0.000, 0.100)),
        _gro_atom(2, "EXP", "A", 5, (1.000, 0.100, 0.000)),
        _gro_atom(2, "EXP", "B", 6, (1.000, 0.000, 0.000)),
        _gro_atom(2, "EXP", "C", 7, (1.100, 0.000, 0.000)),
        _gro_atom(2, "EXP", "D", 8, (1.100, 0.000, 0.100)),
        "   5.00000   5.00000   5.00000",
    ]
    gro_path.write_text("\n".join(gro_lines) + "\n")

    mdin_path.write_text(
        textwrap.dedent(
            f"""
            md_name = "direct_gromacs_harmonic_improper"
            mode = "nve"
            step_limit = 0
            dt = 0
            cutoff = 8.0
            gromacs_top = {_toml_string(top_path)}
            gromacs_gro = {_toml_string(gro_path)}
            mdout = {_toml_string(mdout_path)}
            print_zeroth_frame = 1
            write_mdout_interval = 1
            """
        ).strip()
        + "\n"
    )

    result = subprocess.run(
        [os.environ.get("SPONGE_BIN", "SPONGE"), "-mdin", str(mdin_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )

    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert "dihedral_numbers is 2" in result.stdout + result.stderr

    # k=8.368 kJ/mol/rad^2 is 2 kcal/mol/rad^2. Both impropers have
    # phi=pi/2 and each contributes 1/2*k*phi^2.
    expected = 2.0 * 0.5 * (8.368 / 4.184) * (math.pi / 2.0) ** 2
    assert _extract_mdout_term(
        mdout_path, "improper_dihedral"
    ) == pytest.approx(expected, abs=0.02)

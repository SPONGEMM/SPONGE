import json
import os
import subprocess
import textwrap
from pathlib import Path


def _toml_string(value):
    return json.dumps(os.fspath(value))


def _gro_atom(resid, resname, atomname, atomnr, xyz):
    x, y, z = xyz
    return (
        f"{resid:5d}{resname:<5}{atomname:>5}{atomnr:5d}"
        f"{x:8.3f}{y:8.3f}{z:8.3f}"
    )


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
